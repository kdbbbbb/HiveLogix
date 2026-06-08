#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 仓库实体 (Section 5)

Depot 是整个物流网络的资源起点与归宿：
  - 无人机飞队的归属母港（Mode C 返仓）
  - 卡车的调度发起点和装车地
  - 固定充换电宿主（继承 ChargingHost）
  - 待配送订单的临时池

与 SwapStation 不同，仓库在换电完成后需要将无人机重新加入 idle_drones，
因此其 tick_update() 提供了回调钩子供仿真引擎注册处理逻辑。

语义边界：
  - Depot 继承的队列逻辑只表示“返仓后的充换电服务队列”；
  - 不表示 mode C 中等待卡车回收的阶段。
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from core.entities.charging_host import ChargingHost
from core.entities.order import Order
from core.entities.primitives import Position3D, TaskStatus

logger = logging.getLogger(__name__)


class Depot(ChargingHost):
    """
    全局起点：仓库。

    parking_slots 对应原设计中的 recharge_slots，充换电并发逻辑由基类接管。
    换电完成后，仓库需将无人机重新标记为 idle_drones，通过注册回调
    on_drone_charged(drone_id) 通知仿真引擎，避免引入循环依赖。

    Args:
        depot_id:      全局唯一 ID，格式建议：'D-{序号}'
        location:      仓库固定 UTM 三维坐标
        swap_time:     仓库充换电耗时（秒）
        parking_slots: 充换电并行位数（即 recharge_slots）
        capacity:      仓库订单吞吐量上限（同一时刻最大待处理订单数）
    """

    def __init__(
        self,
        *,
        depot_id: str,
        location: Position3D,
        swap_time: float,
        parking_slots: int,
        capacity: int = 1000,
    ) -> None:
        super().__init__(
            swap_time=swap_time,
            parking_slots=parking_slots,
            host_id=depot_id,
        )
        self.depot_id: str = depot_id
        self.location: Position3D = location
        self.capacity: int = capacity

        # ── 资产注册表 ────────────────────────────────────────────────────────
        self.drone_fleet: list[str] = []       # 归属本仓的无人机全集（ID）
        self.truck_fleet: list[str] = []       # 归属本仓的卡车全集（ID）
        self.idle_drones: list[str] = []       # 当前在库空闲、可调度的无人机 ID

        # ── 订单池 ────────────────────────────────────────────────────────────
        self.pending_orders: list[Order] = []  # 待分配/装载的订单

        # ── 换电完成回调（由仿真引擎注册，仓库完成换电后调用）─────────────────
        # 签名：on_drone_charged(depot_id: str, drone_id: str) -> None
        self._on_drone_charged: Optional[Callable[[str, str], None]] = None

    # ══════════════════════════════════════════════════════════════════════════
    # 回调注册
    # ══════════════════════════════════════════════════════════════════════════

    def register_charge_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """
        注册换电完成回调函数。

        仿真引擎启动时调用此方法，使仓库在每次换电完成后通知引擎更新
        对应无人机的状态（CHARGING → IDLE）并将其加入 idle_drones。

        Args:
            callback: 签名为 (depot_id: str, drone_id: str) -> None 的可调用对象
        """
        self._on_drone_charged = callback

    # ══════════════════════════════════════════════════════════════════════════
    # 充换电宿主（扩展基类 tick_update）
    # ══════════════════════════════════════════════════════════════════════════

    def tick_update(self, current_time: float) -> list[str]:
        """
        重写 tick_update，换电完成后将无人机加入 idle_drones 并触发回调。

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            本轮完成换电的无人机 ID 列表
        """
        completed = super().tick_update(current_time)
        for drone_id in completed:
            self._on_charge_complete(drone_id)
        return completed

    def _on_charge_complete(self, drone_id: str) -> None:
        """
        换电完成后的仓库侧处理：将无人机加入 idle_drones。

        Args:
            drone_id: 完成换电的无人机 ID
        """
        if drone_id in self.drone_fleet and drone_id not in self.idle_drones:
            self.idle_drones.append(drone_id)
            logger.debug("[Depot %s] drone %s 换电完成，已加入 idle_drones。",
                         self.depot_id, drone_id)

        if self._on_drone_charged is not None:
            try:
                self._on_drone_charged(self.depot_id, drone_id)
            except Exception as exc:
                logger.exception(
                    "[Depot %s] on_drone_charged 回调异常（drone=%s）: %s",
                    self.depot_id, drone_id, exc,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # 无人机资产管理
    # ══════════════════════════════════════════════════════════════════════════

    def register_drone(self, drone_id: str, is_idle: bool = True) -> None:
        """
        将无人机登记为归属本仓的资产。

        Args:
            drone_id: 无人机 ID
            is_idle:  是否初始标记为空闲可调度（默认 True）
        """
        if drone_id not in self.drone_fleet:
            self.drone_fleet.append(drone_id)
        if is_idle and drone_id not in self.idle_drones:
            self.idle_drones.append(drone_id)

    def register_truck(self, truck_id: str) -> None:
        """
        将卡车登记为归属本仓的资产。

        Args:
            truck_id: 卡车 ID
        """
        if truck_id not in self.truck_fleet:
            self.truck_fleet.append(truck_id)

    def dispatch_drone(self, drone_id: str) -> None:
        """
        从 idle_drones 移除无人机，标记为已出发。

        调度器在派单后调用，仿真引擎负责更新无人机状态为 FLYING_TO_PICKUP 等。

        Args:
            drone_id: 出发的无人机 ID

        Raises:
            ValueError: 无人机不在 idle_drones 中
        """
        if drone_id not in self.idle_drones:
            raise ValueError(
                f"[Depot {self.depot_id}] drone {drone_id} 不在 idle_drones 中，"
                f"无法派出。当前空闲: {self.idle_drones}"
            )
        self.idle_drones.remove(drone_id)
        logger.debug("[Depot %s] drone %s 已出发。剩余空闲: %d 架。",
                     self.depot_id, drone_id, len(self.idle_drones))

    def receive_drone(self, drone_id: str, current_time: float) -> None:
        """
        接收返仓的无人机，进入充换电服务队列。

        换电完成后由 tick_update() → _on_charge_complete() 自动将无人机
        加回 idle_drones，无需调用方手动操作。

        Args:
            drone_id:    返仓的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        self.arrive(drone_id, current_time)
        logger.debug("[Depot %s] 接收 drone %s 返仓，进入充换电服务队列。",
                     self.depot_id, drone_id)

    # ══════════════════════════════════════════════════════════════════════════
    # 订单管理
    # ══════════════════════════════════════════════════════════════════════════

    def add_order(self, order: Order) -> None:
        """
        将新生成的订单加入待处理池。

        Args:
            order: 新订单对象（status 应为 PENDING）

        Raises:
            ValueError: 订单池已满（超过 capacity）
        """
        if len(self.pending_orders) >= self.capacity:
            raise ValueError(
                f"[Depot {self.depot_id}] 订单池已满 ({self.capacity} 件)，"
                f"无法接收订单 {order.order_id}"
            )
        self.pending_orders.append(order)

    def pop_order(self, order_id: str) -> Order:
        """
        从待处理池中取出指定订单（用于装载至卡车或分配给无人机）。

        Args:
            order_id: 要取出的订单 ID

        Returns:
            对应的 Order 对象

        Raises:
            KeyError: 订单不在待处理池中
        """
        for idx, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                return self.pending_orders.pop(idx)
        raise KeyError(
            f"[Depot {self.depot_id}] 订单 {order_id} 不在待处理池中。"
            f"当前待处理订单数: {len(self.pending_orders)}"
        )

    def load_truck(
        self,
        truck_id: str,
        order_ids: list[str],
        truck_capacity: int,
    ) -> list[Order]:
        """
        为指定卡车批量装车，转移 pending_orders 中对应的订单。

        受 truck_capacity 约束（不超过卡车 max_inventory）。
        超出部分订单留在 pending_orders，不报错。

        Args:
            truck_id:       卡车 ID（仅用于日志）
            order_ids:      要装载的订单 ID 列表
            truck_capacity: 卡车当前剩余可装载数量

        Returns:
            实际装载的 Order 对象列表（len ≤ min(len(order_ids), truck_capacity)）
        """
        loaded: list[Order] = []
        for order_id in order_ids:
            if len(loaded) >= truck_capacity:
                logger.warning(
                    "[Depot %s] 卡车 %s 装载空间已满，剩余 %d 件订单未装载。",
                    self.depot_id, truck_id, len(order_ids) - len(loaded),
                )
                break
            try:
                order = self.pop_order(order_id)
                loaded.append(order)
            except KeyError as exc:
                logger.warning("[Depot %s] load_truck 跳过: %s", self.depot_id, exc)

        logger.info("[Depot %s] 为卡车 %s 装载 %d 件订单。",
                    self.depot_id, truck_id, len(loaded))
        return loaded

    @property
    def pending_count(self) -> int:
        """当前待处理订单数量。"""
        return len(self.pending_orders)

    @property
    def idle_drone_count(self) -> int:
        """当前空闲可调度无人机数量。"""
        return len(self.idle_drones)

    # ══════════════════════════════════════════════════════════════════════════
    # 实现抽象方法
    # ══════════════════════════════════════════════════════════════════════════

    def get_location(self, current_time: float = 0.0) -> Position3D:
        """
        返回仓库固定位置（忽略时间参数）。

        Args:
            current_time: 仿真时间（忽略，保持接口一致）

        Returns:
            仓库 UTM 三维坐标
        """
        return self.location

    def to_telemetry_dict(self) -> dict:
        """
        序列化仓库状态为 WebSocket 推送字典（坐标转 WGS84）。

        Returns:
            JSON 可序列化的仓库状态字典
        """
        lon, lat = self.location.to_wgs84()
        with self._lock:
            serving_ids = list(self.serving_drones.keys())

        return {
            "entity_type":       "depot",
            "depot_id":          self.depot_id,
            "lng":               lon,
            "lat":               lat,
            "altitude":          self.location.z,
            "capacity":          self.capacity,
            "pending_count":     self.pending_count,
            "idle_drone_count":  self.idle_drone_count,
            "drone_fleet_count": len(self.drone_fleet),
            "truck_fleet_count": len(self.truck_fleet),
            "parking_slots":     self.parking_slots,
            "available_slots":   self.available_slots,
            "queue_length":      self.queue_length,
            "swap_time":         self.swap_time,
            "charging_drone_ids": serving_ids,
        }

    def to_dynamic_state(self) -> dict:
        """
        轻量序列化，仅含 TICK 帧所需字段。
        包含所有 DepotConfig 非 Optional 字段，防止 setRuntimeAll 全量替换后字段丢失。
        name 由 EntityManager.get_telemetry() 从 _metadata 合并。
        """
        lon, lat = self.location.to_wgs84()
        return {
            "depot_id":         self.depot_id,
            # ── 静态 DepotConfig 必填字段 ─────────────────────────────────────
            "lng":              lon,
            "lat":              lat,
            "altitude":         self.location.z,
            "capacity":         self.capacity,
            "swap_time":        self.swap_time,
            "parking_slots":    self.parking_slots,
            # ── 动态字段 ─────────────────────────────────────────────────────
            "pending_count":    self.pending_count,
            "idle_drone_count": self.idle_drone_count,
            "available_slots":  self.available_slots,
            "queue_length":     self.queue_length,
        }

    def __repr__(self) -> str:
        return (
            f"Depot(id={self.depot_id!r}, "
            f"idle_drones={len(self.idle_drones)}/{len(self.drone_fleet)}, "
            f"pending_orders={self.pending_count})"
        )
