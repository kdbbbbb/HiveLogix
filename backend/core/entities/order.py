#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 订单实体 (Section 1)

Order 代表一次完整的物流配送请求，是整个调度系统中价值流转的原子单元。
从生成到送达，Order 经历一套严格的状态机流转，并记录完整的时间戳链路
以支持事后的履约分析、超时惩罚计算和 RL 训练数据提取。

设计要点：
  - 使用 update_status() 进行状态机校验，禁止外部直接修改 status
  - get_delay_penalty() 实现软时间窗惩罚（超出 deadline 后按比例计算罚金）
  - get_rl_state_vector() 为强化学习调度器提供归一化的状态向量接口
  - to_telemetry_dict() 序列化为 WGS84 格式供 WebSocket 推送前端
"""

from __future__ import annotations

import logging
from typing import Optional

from core.entities.primitives import Position3D, SourceType, TaskStatus

logger = logging.getLogger(__name__)


class Order:
    """
    物流订单实体。

    Attributes:
        order_id:              全局唯一 ID，格式由上层生成器保证
        create_time:           仿真时间轴上的生成时间（秒）
        deadline:              期望最晚送达时间（软时间窗，秒）
        delivery_loc:          客户收货静态三维坐标（UTM）
        pickup_source_id:      货物所在载具/节点 ID（如 'depot_1', 'truck_2'）
        source_type:           取货源类型（DEPOT / TRUCK）
        payload_weight:        货物重量（kg）
        status:                订单当前状态（通过 update_status() 修改）
        assigned_vehicle_id:   接单的载具 ID（Mode E 多跳中记录发起派单的实体）
        assigned_mode:         记录选用的履约模式（'A'~'E'，未分配时为 None）
        penalty_rate:          单位时间超时罚金权重（元/秒 或 无量纲）
        actual_deliver_time:   实际送达时间（None 表示未送达）
    """

    __slots__ = (
        "order_id",
        "create_time",
        "deadline",
        "delivery_loc",
        "pickup_source_id",
        "source_type",
        "payload_weight",
        "_status",
        "assigned_vehicle_id",
        "assigned_mode",
        "penalty_rate",
        "actual_deliver_time",
    )

    def __init__(
        self,
        *,
        order_id: str,
        create_time: float,
        deadline: float,
        delivery_loc: Position3D,
        pickup_source_id: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        payload_weight: float,
        penalty_rate: float = 1.0,
        assigned_vehicle_id: Optional[str] = None,
        assigned_mode: Optional[str] = None,
    ) -> None:
        # ── 输入校验 ──────────────────────────────────────────────────────────
        if not order_id:
            raise ValueError("order_id 不能为空")
        if deadline <= create_time:
            raise ValueError(
                f"[{order_id}] deadline ({deadline}) 必须晚于 create_time ({create_time})"
            )
        if payload_weight <= 0:
            raise ValueError(f"[{order_id}] payload_weight 必须为正数: {payload_weight}")
        if penalty_rate < 0:
            raise ValueError(f"[{order_id}] penalty_rate 不能为负数: {penalty_rate}")

        self.order_id: str = order_id
        self.create_time: float = create_time
        self.deadline: float = deadline
        self.delivery_loc: Position3D = delivery_loc
        self.pickup_source_id: Optional[str] = pickup_source_id
        self.source_type: Optional[SourceType] = source_type
        self.payload_weight: float = payload_weight
        self._status: TaskStatus = TaskStatus.PENDING
        self.assigned_vehicle_id: Optional[str] = assigned_vehicle_id
        self.assigned_mode: Optional[str] = assigned_mode
        self.penalty_rate: float = penalty_rate
        self.actual_deliver_time: Optional[float] = None

    # ══════════════════════════════════════════════════════════════════════════
    # 属性访问器
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def status(self) -> TaskStatus:
        """只读属性，使用 update_status() 进行修改。"""
        return self._status

    @property
    def is_terminal(self) -> bool:
        """是否已进入终态（COMPLETED / REJECTED）。"""
        return self._status in {TaskStatus.COMPLETED, TaskStatus.REJECTED}

    @property
    def time_window_seconds(self) -> float:
        """时间窗宽度（秒）。"""
        return self.deadline - self.create_time

    # ══════════════════════════════════════════════════════════════════════════
    # 状态机
    # ══════════════════════════════════════════════════════════════════════════

    def update_status(self, new_status: TaskStatus) -> None:
        """
        推进订单状态，并校验状态转移的合法性。

        非法转移（如 COMPLETED → ASSIGNED）会记录错误日志并抛出 ValueError，
        防止仿真引擎中的 bug 导致数据污染。

        Args:
            new_status: 目标状态

        Raises:
            ValueError: 若目标状态不在当前状态的合法转移集合中
        """
        valid_next = TaskStatus.valid_transitions().get(self._status, set())
        if new_status not in valid_next:
            msg = (
                f"[{self.order_id}] 非法状态转移: "
                f"{self._status.value} → {new_status.value}，"
                f"合法目标: {[s.value for s in valid_next]}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # ── COMPLETED 时记录实际送达时间（如外部未手动赋值）─────────────────
        if new_status == TaskStatus.COMPLETED and self.actual_deliver_time is None:
            logger.warning(
                "[%s] update_status → COMPLETED 时 actual_deliver_time 尚未设置，"
                "请在调用 update_status() 前赋值。",
                self.order_id,
            )

        logger.debug(
            "[%s] 状态转移: %s → %s",
            self.order_id, self._status.value, new_status.value,
        )
        self._status = new_status

    # ══════════════════════════════════════════════════════════════════════════
    # 业务计算
    # ══════════════════════════════════════════════════════════════════════════

    def get_delay_penalty(self, current_time: float) -> float:
        """
        计算当前时刻的超时惩罚成本。

        软时间窗模型：deadline 前无惩罚，超出后按 penalty_rate 线性累计。
        公式：max(0, current_time - deadline) × penalty_rate

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            超时惩罚总量（与 penalty_rate 单位一致）
        """
        return max(0.0, current_time - self.deadline) * self.penalty_rate

    def get_remaining_time(self, current_time: float) -> float:
        """
        距截止时间的剩余秒数（负值表示已超时）。

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            剩余秒数，负值代表超时
        """
        return self.deadline - current_time

    def get_rl_state_vector(
        self,
        ref_loc: Position3D,
        current_time: float,
        max_distance: float = 10000.0,
        max_time: float = 7200.0,
        max_weight: float = 20.0,
    ) -> list[float]:
        """
        [RL 接口] 返回归一化状态向量，供强化学习调度器使用。

        向量构成（4维）：
          [0] 相对 dx（归一化）= (delivery_x - ref_x) / max_distance
          [1] 相对 dy（归一化）= (delivery_y - ref_y) / max_distance
          [2] 剩余时间（归一化）= remaining_time / max_time，裁剪至 [-1, 1]
          [3] 载重（归一化）= payload_weight / max_weight，裁剪至 [0, 1]

        Args:
            ref_loc:      参考系坐标（通常为当前载具位置）
            current_time: 当前仿真时间（秒）
            max_distance: 归一化距离基准（米）
            max_time:     归一化时间基准（秒）
            max_weight:   归一化重量基准（kg）

        Returns:
            长度为 4 的浮点列表，各维度均裁剪至 [-1, 1] 或 [0, 1]
        """
        dx = (self.delivery_loc.x - ref_loc.x) / max_distance
        dy = (self.delivery_loc.y - ref_loc.y) / max_distance
        remaining = self.get_remaining_time(current_time) / max_time
        weight = self.payload_weight / max_weight

        # 裁剪防止 RL 网络输入爆炸
        return [
            max(-1.0, min(1.0, dx)),
            max(-1.0, min(1.0, dy)),
            max(-1.0, min(1.0, remaining)),
            max(0.0, min(1.0, weight)),
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # 序列化
    # ══════════════════════════════════════════════════════════════════════════

    def to_telemetry_dict(self) -> dict:
        """
        序列化为 WebSocket 推送字典，坐标转换为 WGS84。

        Returns:
            JSON 可序列化的订单状态字典
        """
        lon, lat = self.delivery_loc.to_wgs84()
        # Handle source_type: could be SourceType enum or str
        source_type_value = None
        if self.source_type:
            source_type_value = self.source_type.value if hasattr(self.source_type, 'value') else str(self.source_type)
        return {
            "entity_type":         "order",
            "order_id":            self.order_id,
            "status":              self._status.value,
            "source_type":         source_type_value,
            "pickup_source_id":    self.pickup_source_id,
            "payload_weight":      self.payload_weight,
            "create_time":         self.create_time,
            "deadline":            self.deadline,
            "actual_deliver_time": self.actual_deliver_time,
            "assigned_vehicle_id": self.assigned_vehicle_id,
            "assigned_mode":       self.assigned_mode,
            "delivery_lng":        lon,
            "delivery_lat":        lat,
            "delivery_z":          self.delivery_loc.z,
        }

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id!r}, "
            f"status={self._status.value}, "
            f"deadline={self.deadline:.0f}, "
            f"weight={self.payload_weight}kg)"
        )
