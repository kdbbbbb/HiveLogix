#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 仿真引擎 (Section 2.3)

SimulationEngine 统筹仿真时钟、物理步进与 WebSocket 广播节奏。

核心设计：
  - 后台线程固定每 100ms（wall-clock）唤醒一次
  - 每次唤醒推进仿真时间 = 0.1 × speed_ratio 秒
  - 广播频率固定 10fps，不随 speed_ratio 变化（彻底解耦精度与带宽）
  - 通过 telemetry.broadcast_tick() 线程安全地向所有 WS 客户端推送 TICK 帧

外部依赖：
  - entity_manager.EntityManager  → tick_all / get_telemetry
  - order_manager.OrderManager     → tick / update_timeouts / get_status_summary
  - api.websockets.telemetry       → broadcast_tick
"""

from __future__ import annotations

import logging
import time
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from entity_manager import EntityManager
    from order_manager import OrderManager

logger = logging.getLogger(__name__)

# 后台线程真实睡眠间隔（100ms wall-clock）
_PHYSICS_INTERVAL_S: float = 0.1


class SimulationEngine:
    """
    仿真引擎单例。

    Attributes:
        current_time (float): 仿真累计时间（秒）
        is_running (bool):    引擎是否正在运行
        speed_ratio (float):  仿真加速倍率（默认 1.0）
        sim_start_wall_ms (int): 引擎启动时的 wall-clock 毫秒时间戳
    """

    def __init__(self) -> None:
        self.current_time:      float = 0.0
        self.is_running:        bool  = False
        self.speed_ratio:       float = 1.0
        self.sim_start_wall_ms: int   = 0

        self._entity_mgr: Optional["EntityManager"] = None
        self._order_mgr:  Optional["OrderManager"]  = None
        self._thread:     Optional[threading.Thread] = None
        self._stop_event: threading.Event            = threading.Event()

        # 动态订单自动调度支持
        self._dispatch_engine = None
        self._dispatch_bbox:     Optional[dict] = None
        self._dispatch_scene_id: Optional[str]  = None
        self._pending_snapshot: set[str] = set()
        self._pending_incremental_ids: set[str] = set()
        self._last_incremental_dispatch_sim_time: float = -1e9
        self._last_ga_urgent_rescue_check_sim_time: float = -1e9
        self._last_new_order_sim_time: float = -1e9
        self._first_pending_incremental_sim_time: float = -1e9
        # 动态调度节流参数：最短间隔 + 防抖 + 最长等待，减少高频重规划抖动。
        self._incremental_dispatch_min_interval_s: float = 2.0
        self._incremental_dispatch_debounce_s: float = 0.0
        self._incremental_dispatch_max_wait_s: float = 5.0

    # ══════════════════════════════════════════════════════════════════════════
    # 初始化注入
    # ══════════════════════════════════════════════════════════════════════════

    def attach(
        self,
        entity_mgr: "EntityManager",
        order_mgr: "OrderManager",
    ) -> None:
        """
        注入 EntityManager 与 OrderManager（在 /api/sim/init 后调用）。

        Args:
            entity_mgr: 已完成 load_from_config() 的 EntityManager
            order_mgr:  已完成 configure() 的 OrderManager
        """
        self._entity_mgr = entity_mgr
        self._order_mgr  = order_mgr
        logger.info("[SimEngine] 已挂载 EntityManager 和 OrderManager")

    def attach_dispatch_engine(
        self,
        dispatch_engine,
        bbox: dict,
        scene_id: Optional[str] = None,
    ) -> None:
        """注入调度引擎，使仿真循环能在动态订单出现时自动触发增量调度。"""
        self._dispatch_engine = dispatch_engine
        self._dispatch_bbox = bbox
        self._dispatch_scene_id = scene_id
        logger.info("[SimEngine] 已挂载 DispatchEngine，自动调度已启用")

    # ══════════════════════════════════════════════════════════════════════════
    # 生命周期控制
    # ══════════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """
        启动仿真后台线程。

        若已在运行，直接返回。
        """
        if self.is_running:
            logger.warning("[SimEngine] start() 被重复调用，忽略")
            return
        if self._entity_mgr is None or self._order_mgr is None:
            raise RuntimeError("调用 start() 前必须先调用 attach()")

        self._stop_event.clear()
        self.is_running        = True
        self.sim_start_wall_ms = int(time.time() * 1000)

        self._thread = threading.Thread(
            target=self._run_loop,
            name="SimEngine-MainLoop",
            daemon=True,       # 主进程退出时自动终止
        )
        self._thread.start()
        logger.info("[SimEngine] 仿真引擎启动，speed_ratio=%.2f", self.speed_ratio)

    def pause(self) -> None:
        """
        暂停仿真（保留 current_time 与所有实体状态）。
        """
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("[SimEngine] 仿真引擎已暂停，current_time=%.2f s", self.current_time)

    def reset(self) -> None:
        """
        重置引擎到初始状态（停止后台线程，清空时钟）。

        调用 reset() 后需重新调用 attach() + start() 才能再次运行。
        """
        self.pause()
        self.current_time      = 0.0
        self.sim_start_wall_ms = 0
        self._entity_mgr       = None
        self._order_mgr        = None
        self._dispatch_engine  = None
        self._dispatch_bbox    = None
        self._dispatch_scene_id = None
        self._pending_snapshot = set()
        self._pending_incremental_ids = set()
        self._last_incremental_dispatch_sim_time = -1e9
        self._last_new_order_sim_time = -1e9
        self._first_pending_incremental_sim_time = -1e9
        logger.info("[SimEngine] 仿真引擎已重置")

    def set_speed(self, speed_ratio: float) -> None:
        """
        调整仿真加速倍率（线程安全，立即生效）。

        Args:
            speed_ratio: 新倍率，必须 > 0
        """
        if speed_ratio <= 0:
            raise ValueError(f"speed_ratio 必须 > 0，收到: {speed_ratio}")
        self.speed_ratio = speed_ratio
        logger.info("[SimEngine] speed_ratio 调整为 %.2f", self.speed_ratio)

    # ══════════════════════════════════════════════════════════════════════════
    # 主循环（后台线程）
    # ══════════════════════════════════════════════════════════════════════════

    def _run_loop(self) -> None:
        """
        仿真主循环，每 100ms（wall-clock）唤醒一次。

        每次唤醒执行：
          1. 推进仿真时间
          2. 驱动所有实体物理步进
          3. 订单生成 + 超时检测
          4. 广播 TICK 帧
        """
        # 延迟导入避免循环依赖（simulation_bp → sim_engine → telemetry）
        from api.websockets.telemetry import broadcast_tick

        logger.info("[SimEngine._run_loop] 主循环启动")

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                dt = _PHYSICS_INTERVAL_S * self.speed_ratio

                # ── 1. 推进仿真时间 ─────────────────────────────────────────
                self.current_time += dt

                # ── 2. 物理步进 ─────────────────────────────────────────────
                if self._entity_mgr is not None:
                    self._entity_mgr.tick_all(
                        self.current_time,
                        dt,
                        self._order_mgr,
                    )

                # ── 3. 订单生成 & 超时检测 ──────────────────────────────────
                if self._order_mgr is not None and self._entity_mgr is not None:
                    self._order_mgr.tick(self.current_time, self._entity_mgr)
                    self._order_mgr.update_timeouts(self.current_time)

                # ── 3.5 动态订单自动增量调度 ──────────────────────────────────
                self._try_auto_dispatch()

                # ── 3.6 契约兑现检查 ─────────────────────────────────────────
                if self._dispatch_engine is not None:
                    try:
                        self._dispatch_engine.try_fulfill_contracts(self.current_time)
                    except Exception:
                        pass

                # ── 4. 广播 TICK ─────────────────────────────────────────────
                payload = self._build_tick_payload()
                broadcast_tick(payload)

            except Exception:
                logger.exception("[SimEngine._run_loop] 主循环异常，跳过本帧")

            # ── 精确睡眠补偿：保证 100ms wall-clock 间隔 ────────────────────
            elapsed   = time.monotonic() - loop_start
            sleep_sec = max(0.0, _PHYSICS_INTERVAL_S - elapsed)
            if sleep_sec > 0:
                self._stop_event.wait(timeout=sleep_sec)

        logger.info("[SimEngine._run_loop] 主循环退出")

    def _try_auto_dispatch(self) -> None:
        """检测新出现的 pending 订单，自动触发增量调度。

        通过对比 pending_orders 的 id 快照来识别本轮新增的订单。
        仅在调度引擎已挂载且已完成过首次调度后生效。
        """
        if (
            self._dispatch_engine is None
            or self._order_mgr is None
            or self._dispatch_bbox is None
        ):
            return

        current_ids = set(self._order_mgr.pending_orders.keys())
        new_ids = current_ids - self._pending_snapshot
        self._pending_snapshot = current_ids

        if new_ids:
            self._pending_incremental_ids.update(new_ids)
            self._last_new_order_sim_time = self.current_time
            if self._first_pending_incremental_sim_time < -1e8:
                self._first_pending_incremental_sim_time = self.current_time

        if not self._pending_incremental_ids:
            self._try_ga_urgent_rescue_dispatch(current_ids)
            return

        # 防抖：若新订单仍在连续到达，先短暂合批；但不超过最长等待上限。
        since_last_new = self.current_time - self._last_new_order_sim_time
        since_first_pending = self.current_time - self._first_pending_incremental_sim_time
        if (
            since_last_new < self._incremental_dispatch_debounce_s
            and since_first_pending < self._incremental_dispatch_max_wait_s
        ):
            return

        if (
            self.current_time - self._last_incremental_dispatch_sim_time
            < self._incremental_dispatch_min_interval_s
        ):
            return

        new_orders = {
            oid: self._order_mgr.pending_orders[oid]
            for oid in sorted(self._pending_incremental_ids)
            if oid in self._order_mgr.pending_orders
        }
        if not new_orders:
            self._pending_incremental_ids.clear()
            self._first_pending_incremental_sim_time = -1e9
            return

        try:
            logger.info(
                "[SimEngine] 批量触发动态调度：%d 单（首单等待 %.2fs）: %s",
                len(new_orders),
                max(0.0, since_first_pending),
                list(new_orders.keys()),
            )
            self._dispatch_engine.execute_incremental(
                new_orders,
                self.current_time,
                self._dispatch_bbox,
                scene_id=self._dispatch_scene_id,
            )
            self._last_incremental_dispatch_sim_time = self.current_time
            self._pending_incremental_ids.clear()
            self._first_pending_incremental_sim_time = -1e9
            # 调度完成后刷新快照（feasible 的单已移入 assigned）
            self._pending_snapshot = set(self._order_mgr.pending_orders.keys())
        except Exception:
            logger.exception("[SimEngine] 动态订单增量调度失败")
            # 失败后同样设置最小重试间隔，避免每帧抛错导致卡顿。
            self._last_incremental_dispatch_sim_time = self.current_time

    def _try_ga_urgent_rescue_dispatch(self, current_pending_ids: set[str]) -> None:
        """GA-MMCE 专用：无新 pending 单时周期性触发临期订单救援检查。"""
        if (
            self._dispatch_engine is None
            or self._order_mgr is None
            or self._dispatch_bbox is None
        ):
            return
        if str(getattr(self._dispatch_engine, "solver_name", "") or "").lower() != "ga_mmce":
            return
        if current_pending_ids or not self._order_mgr.assigned_orders:
            return

        solver = getattr(self._dispatch_engine, "solver", None)
        config = getattr(solver, "config", None)
        if not bool(getattr(config, "urgent_rescue_enabled", True)):
            return
        interval_s = max(1.0, float(getattr(config, "urgent_rescue_check_interval_s", 30.0) or 30.0))
        if self.current_time - self._last_ga_urgent_rescue_check_sim_time < interval_s:
            return
        if self.current_time - self._last_incremental_dispatch_sim_time < interval_s:
            return

        self._last_ga_urgent_rescue_check_sim_time = self.current_time
        try:
            logger.info(
                "[SimEngine] GA-MMCE 周期临期救援检查：assigned=%d t=%.1fs",
                len(self._order_mgr.assigned_orders),
                self.current_time,
            )
            self._dispatch_engine.execute_incremental(
                {},
                self.current_time,
                self._dispatch_bbox,
                scene_id=self._dispatch_scene_id,
            )
            self._last_incremental_dispatch_sim_time = self.current_time
            self._pending_snapshot = set(self._order_mgr.pending_orders.keys())
        except Exception:
            logger.exception("[SimEngine] GA-MMCE 临期救援调度失败")
            self._last_incremental_dispatch_sim_time = self.current_time

    def _build_tick_payload(self) -> dict:
        """
        构造 TICK 帧的 payload 字典。

        Returns:
            包含 sim_time / entities / orders / stats 的字典
        """
        entities = {}
        orders   = []
        stats    = {}

        if self._entity_mgr is not None:
            entities = self._entity_mgr.get_telemetry()

        if self._order_mgr is not None:
            # 推送所有订单的状态（pending + assigned + completed）
            orders = self._order_mgr.get_recent_orders(limit=500)
            stats = self._order_mgr.get_status_summary()

        if self._dispatch_engine is not None and hasattr(self._dispatch_engine, "get_runtime_metrics"):
            try:
                stats.update(self._dispatch_engine.get_runtime_metrics())
            except Exception:
                pass

        return {
            "sim_time": round(self.current_time, 3),
            "entities": entities,
            "orders": orders,
            "stats":    stats,
            "paths": self._build_runtime_path_snapshot(),
        }

    def build_full_snapshot(self) -> dict:
        """
        构造 FULL_SNAPSHOT 帧的完整 payload（建连时推送）。

        由 telemetry.set_snapshot_builder() 注入为快照构造函数。

        Returns:
            包含 type="FULL_SNAPSHOT" 的完整消息字典
        """
        entities = {}
        stats    = {}

        if self._entity_mgr is not None:
            entities = self._entity_mgr.get_static_snapshot()

        if self._order_mgr is not None:
            stats = self._order_mgr.get_status_summary()

        if self._dispatch_engine is not None and hasattr(self._dispatch_engine, "get_runtime_metrics"):
            try:
                stats.update(self._dispatch_engine.get_runtime_metrics())
            except Exception:
                pass

        return {
            "type": "FULL_SNAPSHOT",
            "payload": {
                "sim_time":          round(self.current_time, 3),
                "is_running":        self.is_running,
                "speed_ratio":       self.speed_ratio,
                "sim_start_wall_ms": self.sim_start_wall_ms,
                "entities":          entities,
                "stats":             stats,
                "paths":             self._build_runtime_path_snapshot(),
            },
        }

    def _build_runtime_path_snapshot(self) -> dict[str, list[dict]]:
        if self._entity_mgr is None:
            return {"trucks": [], "drones": []}

        return {
            "trucks": self._build_active_truck_paths(),
            "drones": self._build_active_drone_paths(),
        }

    @staticmethod
    def _frontend_path(points: list) -> list[list[float]]:
        coords: list[list[float]] = []
        for point in points:
            if point is None:
                continue
            lon, lat = point.to_wgs84()
            coords.append([round(lon, 6), round(lat, 6)])
        return coords

    def _build_active_truck_paths(self) -> list[dict]:
        paths: list[dict] = []
        for truck in self._entity_mgr.trucks.values():
            route_data = list(getattr(truck, "_route_data", []) or [])
            if not getattr(getattr(truck, "status", None), "is_mobile", False) or len(route_data) < 2:
                continue

            elapsed = max(0.0, self.current_time - float(getattr(truck, "_departure_time", 0.0) or 0.0))
            dist_traveled = elapsed * float(getattr(truck, "speed", 0.0) or 0.0)
            next_node = None
            for node in route_data:
                if float(getattr(node, "cumulative_dist", 0.0) or 0.0) > dist_traveled + 1e-6:
                    next_node = node
                    break
            if next_node is None:
                continue

            current = truck.get_location(self.current_time)
            target = next_node.position
            paths.append(
                {
                    "entity_id": truck.truck_id,
                    "status": "active",
                    "motion_mode": "truck_current_leg",
                    "distance_m": round(current.distance_2d(target), 2),
                    "path": self._frontend_path([current, target]),
                }
            )
        return paths

    def _build_active_drone_paths(self) -> list[dict]:
        paths: list[dict] = []
        for drone in self._entity_mgr.drones.values():
            if not getattr(getattr(drone, "status", None), "is_flying", False):
                continue
            route_plan = list(getattr(drone, "route_plan", []) or [])
            idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            if idx >= len(route_plan):
                continue

            start_idx, target_idx = self._active_drone_display_slice(drone, route_plan, idx)
            waypoint = route_plan[target_idx]
            action = getattr(waypoint, "action", None)
            action_name = getattr(action, "value", action)
            segment_points = [wp.loc for wp in route_plan[start_idx:target_idx + 1]]
            paths.append(
                {
                    "entity_id": drone.drone_id,
                    "status": "active",
                    "motion_mode": "drone_current_leg",
                    "target_node_id": getattr(waypoint, "target_entity_id", None),
                    "target_node_type": str(action_name or ""),
                    "order_id": getattr(drone, "carrying_order_id", None)
                    or getattr(drone, "pending_release_order_id", None)
                    or getattr(waypoint, "target_entity_id", None),
                    "distance_m": round(self._polyline_distance_2d(segment_points), 2),
                    "path": self._frontend_path(segment_points),
                }
            )
        return paths

    def _active_drone_display_slice(self, drone, route_plan: list, start_idx: int) -> tuple[int, int]:
        ga_slice = self._active_ga_drone_display_slice(drone, route_plan, start_idx)
        if ga_slice is not None:
            return ga_slice

        target_idx = self._active_drone_target_waypoint_index(drone, route_plan, start_idx)
        action = getattr(route_plan[target_idx], "action", None)
        action_name = str(getattr(action, "value", action) or "")

        if action_name == "DELIVER":
            segment_start = 0
            for idx in range(target_idx - 1, -1, -1):
                prev_action = getattr(route_plan[idx], "action", None)
                prev_name = str(getattr(prev_action, "value", prev_action) or "")
                if prev_name in {"DELIVER", "DOCK_DEPOT", "DOCK_TRUCK", "SWAP_BATTERY"}:
                    segment_start = idx + 1
                    break
            return max(0, segment_start), target_idx

        if action_name in {"DOCK_DEPOT", "DOCK_TRUCK", "SWAP_BATTERY"}:
            segment_start = start_idx
            for idx in range(min(start_idx, target_idx), -1, -1):
                prev_action = getattr(route_plan[idx], "action", None)
                prev_name = str(getattr(prev_action, "value", prev_action) or "")
                if prev_name == "DELIVER":
                    segment_start = idx
                    break
            return max(0, segment_start), target_idx

        return max(0, start_idx), target_idx

    def _active_ga_drone_display_slice(self, drone, route_plan: list, start_idx: int) -> tuple[int, int] | None:
        segments = list(getattr(drone, "_ga_runtime_segments", []) or [])
        if not segments:
            return None

        current_segment = None
        for segment in segments:
            seg_start = int(getattr(segment, "start_idx", 0) or 0)
            seg_end = int(getattr(segment, "dock_idx", seg_start) or seg_start)
            if seg_start <= start_idx <= seg_end:
                current_segment = segment
                break
        if current_segment is None:
            current_segment = segments[-1]

        seg_start = int(getattr(current_segment, "start_idx", 0) or 0)
        deliver_idx = int(getattr(current_segment, "deliver_idx", seg_start) or seg_start)
        dock_idx = int(getattr(current_segment, "dock_idx", deliver_idx) or deliver_idx)
        status = str(getattr(getattr(drone, "status", None), "value", getattr(drone, "status", "")) or "")
        if status in {"FLYING_TO_STATION", "FLYING_TO_TRUCK", "RETURNING_TO_DEPOT"} or start_idx > deliver_idx:
            return max(0, deliver_idx), min(len(route_plan) - 1, dock_idx)
        return max(0, seg_start), min(len(route_plan) - 1, deliver_idx)

    @staticmethod
    def _polyline_distance_2d(points: list) -> float:
        if len(points) < 2:
            return 0.0
        return sum(points[idx - 1].distance_2d(points[idx]) for idx in range(1, len(points)))

    @staticmethod
    def _active_drone_target_waypoint_index(drone, route_plan: list, start_idx: int) -> int:
        status = getattr(getattr(drone, "status", None), "value", getattr(drone, "status", ""))
        remaining = range(start_idx, len(route_plan))

        def action_name_at(index: int) -> str:
            action = getattr(route_plan[index], "action", None)
            return str(getattr(action, "value", action) or "")

        if status == "FLYING_TO_DELIVER" or getattr(drone, "carrying_order_id", None):
            for idx in remaining:
                if action_name_at(idx) == "DELIVER":
                    return idx

        if status in {"FLYING_TO_STATION", "FLYING_TO_TRUCK", "RETURNING_TO_DEPOT"}:
            for idx in remaining:
                if action_name_at(idx) in {"DOCK_DEPOT", "DOCK_TRUCK", "SWAP_BATTERY"}:
                    return idx

        for idx in remaining:
            if action_name_at(idx) in {"DELIVER", "DOCK_DEPOT", "DOCK_TRUCK", "SWAP_BATTERY"}:
                return idx
        return start_idx
