#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 订单管理器 (Section 2.2)

OrderManager 按仿真时钟自驱生成订单，维护全局三池（pending / assigned / completed），
并向调度引擎暴露可用任务。前端只传入生成策略参数，不再主动推送订单。

关键设计：
  - 订单生成完全由后端时钟控制，消除双时钟问题
  - pickup_source_id / source_type 留 None，由调度算法后续填充
  - 配送目的地坐标通过 bbox 均匀随机采样生成，不依赖 scene_service 内存缓存
  - completed_orders 仅保留末尾 N 条，防止内存无限增长
  - 可选：通过 set_scheduled_dynamic_orders 在指定仿真时刻注入预定义动态订单（与 JSON 对齐、可复现）

导入规则（依赖 app.py 已将 BASE_DIR 注入 sys.path）：
  from utils.coord_utils import wgs84_to_utm
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

from core.entities.order import Order
from core.entities.primitives import Position3D, SourceType, TaskStatus
from utils.coord_utils import wgs84_to_utm

if TYPE_CHECKING:
    from entity_manager import EntityManager

logger = logging.getLogger(__name__)

# completed_orders 最大保留条数
_MAX_COMPLETED = 2000


class OrderManager:
    """
    订单生命周期管理器。

    存储结构：
      pending_orders:   {order_id: Order}  待分配
      assigned_orders:  {order_id: Order}  配送中
      completed_orders: [Order]            完成/超时（末尾 N 条）

      _gen_config:      dict               生成策略参数（来自 order_gen_config）
      _next_order_time: float              下一次生成订单的仿真时间点
      _bbox:            dict               地图边界（min/max lng/lat）
    """

    def __init__(self) -> None:
        self.pending_orders:   dict[str, Order] = {}
        self.assigned_orders:  dict[str, Order] = {}
        self.completed_orders: list[Order]      = []

        self._gen_config:       dict  = {}
        self._next_order_time:  float = 0.0
        self._bbox:             dict  = {}
        self._order_seq:        int   = 0      # 全局序号，构造 order_id
        # 预定义动态订单（spawn_sim_s 触发，参数固定，便于算法对比复现）
        self._scheduled_dynamic: list[dict] = []
        self._scheduled_dynamic_i: int = 0

    # ══════════════════════════════════════════════════════════════════════════
    # 配置
    # ══════════════════════════════════════════════════════════════════════════

    def configure(self, gen_config: dict, bbox: dict) -> None:
        """
        接收前端传入的订单生成策略参数与地图边界，重置内部状态。

        Args:
            gen_config: 对应前端 OrderGeneratorConfig 接口的完整字典
            bbox:       地图地理边界 {"min_lng", "min_lat", "max_lng", "max_lat"}
        """
        self._gen_config      = dict(gen_config)
        self._bbox            = dict(bbox)
        self._next_order_time = 0.0
        self._order_seq       = 0
        self.pending_orders.clear()
        self.assigned_orders.clear()
        self.completed_orders.clear()
        self._scheduled_dynamic.clear()
        self._scheduled_dynamic_i = 0
        logger.info(
            "[OrderManager] 配置完成：arrival_rate=%.1f/min，bbox=%s",
            gen_config.get("arrival_rate", 4),
            bbox,
        )

    def set_scheduled_dynamic_orders(self, entries: Optional[list]) -> None:
        """
        设置「调度过程中按仿真时刻注入」的订单列表（全量替换）。

        每条记录建议字段：
          order_id, spawn_sim_s, delivery_lng, delivery_lat,
          deadline_offset_s 或 deadline_sim_s, payload_weight,
          delivery_z?, pickup_source_id?, source_type?

        spawn_sim_s：相对仿真起点的秒；create_time 固定为该值；deadline 由偏移或绝对时刻给出。
        """
        self._scheduled_dynamic = sorted(
            (dict(e) for e in (entries or [])),
            key=lambda e: float(e["spawn_sim_s"]),
        )
        self._scheduled_dynamic_i = 0
        logger.info(
            "[OrderManager] 已加载 %d 条预定义动态订单",
            len(self._scheduled_dynamic),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 仿真步进
    # ══════════════════════════════════════════════════════════════════════════

    def tick(self, current_time: float, entity_mgr: "EntityManager") -> None:
        """
        每个物理步进周期调用一次，按策略生成订单。

        生成频率：基于 arrival_rate（件/分钟）计算下次生成时刻 _next_order_time；
        burst 模式下在 burst_duration_s 窗口内提升到 arrival_rate × burst_multiplier。

        Args:
            current_time: 仿真累计时间（秒）
            entity_mgr:   EntityManager 实例，用于获取当前仓库列表（预留扩展）
        """
        # {} 视为「无泊松配置」，但仍可配合预定义动态订单工作
        if not self._scheduled_dynamic and not self._gen_config:
            return
        cfg = self._gen_config or {}

        max_orders = cfg.get("max_orders")
        if max_orders is not None:
            max_orders = int(max_orders)
        total = len(self.pending_orders) + len(self.assigned_orders) + len(self.completed_orders)
        if max_orders is None or total < max_orders:
            self._spawn_scheduled_dynamic_orders(current_time, max_orders)

        # 泊松流订单依赖完整 gen_config（含 bbox 等），无配置则跳过
        if not self._gen_config:
            return

        cfg = self._gen_config
        total = len(self.pending_orders) + len(self.assigned_orders) + len(self.completed_orders)
        if max_orders is not None and total >= max_orders:
            return

        # 若还未到下次生成时刻，直接返回
        if current_time < self._next_order_time:
            return

        # 生成一批订单（处理积压：循环直到 _next_order_time > current_time）
        while self._next_order_time <= current_time:
            # 当前是否处于 burst 窗口（简化实现：不跟踪 burst 窗口起止，
            # 仅在 burst_enabled=True 且按 burst_multiplier 提升频率）
            arrival_rate = float(cfg.get("arrival_rate", 4))
            if cfg.get("burst_enabled"):
                arrival_rate *= float(cfg.get("burst_multiplier", 3))

            # arrival_rate<=0：关闭泊松流（仅依赖 initial_orders + scheduled_dynamic）
            if arrival_rate <= 0:
                self._next_order_time = float("inf")
                break

            # 生成一个订单
            order = self._create_order(current_time, cfg)
            if order is not None:
                self.pending_orders[order.order_id] = order
                logger.debug("[OrderManager] 新订单 %s t=%.1fs", order.order_id, current_time)

            # 推进 _next_order_time（指数分布间隔，泊松过程）
            inter_arrival = random.expovariate(arrival_rate / 60.0)
            self._next_order_time += inter_arrival

            # 保护：防止 arrival_rate=∞ 导致无限循环
            if self._next_order_time <= current_time:
                # 最多生成 1 批，避免卡死
                self._next_order_time = current_time + 0.01
                break

    def update_timeouts(self, current_time: float) -> None:
        """
        将超出 deadline 的 pending 订单标记为 TIMEOUT，并移入 completed_orders。

        注意：TIMEOUT 状态仍需后续履约，此处仅标记并计入统计池，
        不从 pending_orders 中移除（待调度引擎决定后续处理）。
        实际上将订单从 pending_orders 移到 completed_orders，
        后续调度引擎可从 completed_orders 中筛选 TIMEOUT 记录做补偿调度。

        Args:
            current_time: 仿真累计时间（秒）
        """
        timed_out = [
            oid for oid, o in self.pending_orders.items()
            if o.deadline <= current_time and o.status == TaskStatus.PENDING
        ]
        for oid in timed_out:
            order = self.pending_orders.pop(oid)
            order.update_status(TaskStatus.TIMEOUT)
            self.completed_orders.append(order)
            logger.debug("[OrderManager] 订单 %s 超时", oid)

        # 裁剪 completed_orders，防止内存无限增长
        if len(self.completed_orders) > _MAX_COMPLETED:
            self.completed_orders = self.completed_orders[-_MAX_COMPLETED:]

    # ══════════════════════════════════════════════════════════════════════════
    # 统计汇总
    # ══════════════════════════════════════════════════════════════════════════

    def get_status_summary(self) -> dict:
        """
        返回四项订单计数，用于 TICK 帧 stats 字段与大屏展示。

        v4.5 修正：键名使用 orders_ 前缀，与前端 SimStats 接口对齐。

        Returns:
            dict with keys: orders_pending, orders_assigned, orders_completed, orders_timeout
        """
        completed_count = sum(
            1 for o in self.completed_orders
            if o.status == TaskStatus.COMPLETED
        )
        timeout_count = sum(
            1 for o in self.completed_orders
            if o.status == TaskStatus.TIMEOUT
        )
        return {
            "orders_pending":   len(self.pending_orders),
            "orders_assigned":  len(self.assigned_orders),
            "orders_completed": completed_count,
            "orders_timeout":   timeout_count,
        }

    def get_recent_orders(self, limit: int = 100) -> list[dict]:
        """
        返回最近 N 条订单的遥测字典，供 GET /api/sim/orders 接口使用。

        包含 time_domain="sim_s" 标记，使前端展示层能正确解析仿真秒时间戳。

        Args:
            limit: 最多返回条数

        Returns:
            list of order telemetry dicts
        """
        all_orders: list[Order] = (
            list(self.pending_orders.values()) +
            list(self.assigned_orders.values()) +
            self.completed_orders
        )
        # 按 create_time 降序排列，取末尾 limit 条
        all_orders.sort(key=lambda o: o.create_time, reverse=True)
        result = []
        for order in all_orders[:limit]:
            d = order.to_telemetry_dict()
            d["time_domain"] = "sim_s"   # 标记时间单位为仿真秒
            result.append(d)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 内部工具
    # ══════════════════════════════════════════════════════════════════════════

    def _spawn_scheduled_dynamic_orders(
        self,
        current_time: float,
        max_orders: Optional[int],
    ) -> None:
        """在 current_time 到达 spawn_sim_s 时注入预定义订单（确定性、可复现）。"""
        while self._scheduled_dynamic_i < len(self._scheduled_dynamic):
            entry = self._scheduled_dynamic[self._scheduled_dynamic_i]
            spawn_t = float(entry["spawn_sim_s"])
            if current_time < spawn_t:
                break
            tot = (
                len(self.pending_orders)
                + len(self.assigned_orders)
                + len(self.completed_orders)
            )
            if max_orders is not None and tot >= max_orders:
                break
            self._scheduled_dynamic_i += 1
            order = self._order_from_scheduled_entry(entry, spawn_t)
            if order is not None:
                self.pending_orders[order.order_id] = order
                logger.debug(
                    "[OrderManager] 预定义动态订单 %s t=%.1fs",
                    order.order_id,
                    spawn_t,
                )

    def _order_from_scheduled_entry(
        self,
        entry: dict,
        spawn_t: float,
    ) -> Optional[Order]:
        try:
            oid = str(entry["order_id"])
            lon = float(entry["delivery_lng"])
            lat = float(entry["delivery_lat"])
            x, y = wgs84_to_utm(lon, lat)
            z = float(entry.get("delivery_z", 0.0))
            delivery_loc = Position3D(x=x, y=y, z=z)
            if "deadline_sim_s" in entry:
                deadline = float(entry["deadline_sim_s"])
            else:
                deadline = spawn_t + float(entry.get("deadline_offset_s", 900.0))
            if deadline <= spawn_t:
                logger.warning(
                    "[OrderManager] 动态订单 %s deadline 必须晚于 spawn_sim_s，已跳过",
                    oid,
                )
                return None
            weight = float(entry.get("payload_weight", 1.0))
            st_raw = entry.get("source_type")
            source_type = (
                None if st_raw is None else SourceType(str(st_raw))
            )
            return Order(
                order_id=oid,
                create_time=spawn_t,
                deadline=deadline,
                delivery_loc=delivery_loc,
                pickup_source_id=entry.get("pickup_source_id"),
                source_type=source_type,
                payload_weight=weight,
            )
        except Exception as exc:
            logger.warning(
                "[OrderManager] 解析预定义动态订单失败: %s entry=%s",
                exc,
                entry.get("order_id", "?"),
            )
            return None

    def _create_order(self, current_time: float, cfg: dict) -> Optional[Order]:
        """
        生成一个新订单实例。

        配送目的地在 _bbox 范围内均匀随机采样（或其他 geo_mode 的简化实现）；
        pickup_source_id / source_type 留 None，由调度算法后续填充。

        Args:
            current_time: 当前仿真时间（秒）
            cfg:          生成策略字典

        Returns:
            新建的 Order 实例，若 bbox 未配置则返回 None
        """
        if not self._bbox:
            logger.warning("[OrderManager._create_order] bbox 未配置，跳过生成")
            return None

        # ── 配送目的地（WGS84 → UTM）────────────────────────────────────────
        lon = random.uniform(self._bbox["min_lng"], self._bbox["max_lng"])
        lat = random.uniform(self._bbox["min_lat"], self._bbox["max_lat"])
        x, y = wgs84_to_utm(lon, lat)
        delivery_loc = Position3D(x=x, y=y, z=0.0)

        # ── 时间窗（前端传分钟，后端换算秒）────────────────────────────────
        window_min_s = float(cfg.get("window_min", 20)) * 60.0
        window_max_s = float(cfg.get("window_max", 60)) * 60.0
        window_s     = random.uniform(window_min_s, window_max_s)
        deadline     = current_time + window_s

        # ── 货物重量 ──────────────────────────────────────────────────────
        weight = self._sample_payload_weight(cfg)

        # ── 唯一 ID（seed 可复现）──────────────────────────────────────────
        # benchmark 静态单可能复用历史泊松生成的 ORD-* ID；训练 reset 时先注入
        # 静态 UAV 单，再生成新的泊松单，因此这里必须避免覆盖 pending 池中的同名订单。
        existing_ids = self._existing_order_ids()
        for _ in range(1000):
            self._order_seq += 1
            short_hex = f"{random.getrandbits(32):08X}"
            order_id = f"ORD-{short_hex}-{self._order_seq}"
            if order_id not in existing_ids:
                break
        else:
            raise RuntimeError("[OrderManager._create_order] 无法生成唯一订单 ID")

        return Order(
            order_id=order_id,
            create_time=current_time,
            deadline=deadline,
            delivery_loc=delivery_loc,
            pickup_source_id=None,
            source_type=None,
            payload_weight=weight,
        )

    @staticmethod
    def _sample_payload_weight(cfg: dict) -> float:
        """按可选分段配置采样载重；未配置时回退到旧的均匀分布。"""

        bands = cfg.get("weight_bands") or ()
        if bands:
            draw = random.random()
            cumulative = 0.0
            last_valid: dict | None = None
            for raw_band in bands:
                band = dict(raw_band)
                probability = float(band.get("probability", 0.0))
                if probability <= 0.0:
                    continue
                band_floor = cumulative
                cumulative += probability
                last_valid = band
                if draw <= cumulative:
                    band_min = float(
                        band.get("weight_min", band.get("weight_min_kg"))
                    )
                    band_max = float(
                        band.get("weight_max", band.get("weight_max_kg"))
                    )
                    local_u = min(1.0, max(0.0, (draw - band_floor) / probability))
                    return band_min + local_u * (band_max - band_min)
            if last_valid is not None:
                band_min = float(
                    last_valid.get("weight_min", last_valid.get("weight_min_kg"))
                )
                band_max = float(
                    last_valid.get("weight_max", last_valid.get("weight_max_kg"))
                )
                return band_max

        return random.uniform(
            float(cfg.get("weight_min", 0.5)),
            float(cfg.get("weight_max", 5.0)),
        )

    def _existing_order_ids(self) -> set[str]:
        return {
            *self.pending_orders.keys(),
            *self.assigned_orders.keys(),
            *(order.order_id for order in self.completed_orders),
        }
