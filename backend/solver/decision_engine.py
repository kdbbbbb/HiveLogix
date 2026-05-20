#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 调度决策编排层

职责：
  1. 触发求解算法（贪心、ALNS、DRL）并获得分配方案
  2. 将分配方案应用到实体和订单状态
  3. 记录调度日志供前端展示

编排流程：
    [OrderManager.pending_orders] → [GreedyMMCE.dispatch()]
    → [DispatchDecisionEngine.execute_plan()]
    → [更新订单状态 + 触发实体行为]
"""

from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import TYPE_CHECKING

from config.loader import load_solver_energy_params
from solver.factory import create_solver, list_solvers
from solver.greedy_mmce import AllocationResult, DispatchPlan
from solver.market_based_solver import MarketBasedSolver
from solver.interfaces import DispatchSolver
from core.entities.primitives import RouteWaypoint, WaypointAction, SourceType, DroneStatus, Position3D

try:
    from environment.path_planning.planner import PathPlanner
except ImportError:
    PathPlanner = None

if TYPE_CHECKING:
    from entity_manager import EntityManager
    from order_manager import OrderManager

logger = logging.getLogger(__name__)


class DispatchDecisionEngine:
    """
    调度决策编排器。

    职责：
        1. 调用已配置求解器（Greedy/ALNS/DRL 等）进行分配
      2. 应用分配结果到实体和订单的状态机
      3. 发出后续行为触发（如驾驶、飞行、充电）
    """

    def __init__(
        self,
        entity_mgr: "EntityManager",
        order_mgr: "OrderManager",
        solver: DispatchSolver | None = None,
        solver_name: str = "greedy",
    ) -> None:
        """
        Args:
            entity_mgr: EntityManager 实例
            order_mgr:  OrderManager 实例
            solver:     可选，外部注入求解器实例
            solver_name:未注入 solver 时，按名称从工厂创建（默认 greedy）
        """
        self.entity_mgr = entity_mgr
        self.order_mgr = order_mgr
        if solver is not None:
            self.solver = solver
            self.solver_name = solver.__class__.__name__.lower()
        else:
            self.solver_name = solver_name.strip().lower()
            self.solver = create_solver(self.solver_name, self.entity_mgr)

        if isinstance(self.solver, MarketBasedSolver):
            self.solver.bind_order_manager(self.order_mgr)

        runtime_cfg = load_solver_energy_params()
        self.TRUCK_SERVICE_TIME_ORDER = runtime_cfg.truck_service_time_order_s
        self.TRUCK_DRONE_LAUNCH_TIME = runtime_cfg.truck_drone_launch_time_s
        self.TRUCK_DRONE_RECOVER_TIME = runtime_cfg.truck_drone_recover_time_s
        self.TRUCK_STATION_HOLD_TIME = max(
            self.TRUCK_DRONE_LAUNCH_TIME,
            self.TRUCK_DRONE_RECOVER_TIME,
        )
        # truck 事件先于 drone 事件处理，同拍到站可能漏回收，增加保护缓冲。
        self.RECOVERY_EVENT_GUARD_S = 1.0
        self._truck_energy_wh_per_meter = runtime_cfg.truck_energy_wh_per_meter
        self._cum_cost_total = 0.0
        self._dispatch_count = 0
        self._active_scene_id: str | None = None
        self._active_path_planner = None

    def set_solver(self, solver_name: str) -> None:
        """按名称切换求解器实例。"""
        target = solver_name.strip().lower()
        if not target:
            raise ValueError("solver_name 不能为空")
        if target == self.solver_name:
            return
        self.solver = create_solver(target, self.entity_mgr)
        self.solver_name = target
        if isinstance(self.solver, MarketBasedSolver):
            self.solver.bind_order_manager(self.order_mgr)
        logger.info("[DispatchDecisionEngine] 已切换求解器为 %s", target)

    def _load_buildings_geojson(self, scene_id: str | None) -> dict | None:
        backend_dir = Path(__file__).resolve().parent.parent
        candidate_paths: list[Path] = []
        if scene_id:
            candidate_paths.append(backend_dir / "test_data" / scene_id / "no_fly_zones.geojson")
            candidate_paths.append(backend_dir / "test_data" / scene_id / "buildings.geojson")
            if scene_id == "default_test_4x4km":
                candidate_paths.append(backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson")
                candidate_paths.append(backend_dir / "test_data" / "default_scene" / "buildings.geojson")
        candidate_paths.append(backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson")
        candidate_paths.append(backend_dir / "test_data" / "default_scene" / "buildings.geojson")

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, (dict, list)):
                    num_features = len(data) if isinstance(data, list) else len(data.get("features", []))
                    logger.info("[DispatchDecisionEngine] 已加载地图数据: %s (features=%d)", path.name, num_features)
                    return data
            except Exception:
                logger.warning("[DispatchDecisionEngine] 建筑缓存读取失败: %s", path)
        return None

    def _activate_path_planner(self, scene_id: str | None) -> None:
        self._active_scene_id = scene_id
        if PathPlanner is None:
            self._active_path_planner = None
            return

        buildings_geojson = self._load_buildings_geojson(scene_id)
        if buildings_geojson is None:
            self._active_path_planner = None
            if hasattr(self.solver, "set_path_planner"):
                try:
                    self.solver.set_path_planner(None)
                except Exception:
                    logger.exception("[DispatchDecisionEngine] 清空求解器 PathPlanner 失败")
            return

        self._active_path_planner = PathPlanner(buildings_geojson)
        if hasattr(self.solver, "set_path_planner"):
            try:
                self.solver.set_path_planner(self._active_path_planner)
            except Exception:
                logger.exception("[DispatchDecisionEngine] 向求解器注入 PathPlanner 失败")

    @staticmethod
    def get_available_solvers() -> list[str]:
        """返回当前可用求解器列表。"""
        return list_solvers()

    def execute(self, current_time: float, bbox: dict, scene_id: str | None = None) -> DispatchPlan:
        """
        执行一轮调度决策。

        流程：
          1. 从 order_mgr.pending_orders 取快照
          2. 调用贪心求解器
          3. 应用分配方案（更新订单状态）

        Args:
            current_time: 仿真时刻（秒）
            bbox: 地图边界 {"minx": float, "miny": float, "maxx": float, "maxy": float}
            scene_id: 预设场景 ID（可选，如 'default_test_4x4km'）

        Returns:
            分配方案（DispatchPlan）
        """
        pending = dict(self.order_mgr.pending_orders)  # 快照
        if not pending:
            logger.debug("[DispatchDecisionEngine] 无待分配订单")
            return DispatchPlan(
                allocations=[],
                cost_total=0.0,
                summary={"total_orders": 0, "feasible": 0, "modes": {}},
            )

        # 调用求解器，传递 scene_id 以支持使用缓存的 OSM 数据
        self._activate_path_planner(scene_id)
        plan = self.solver.dispatch(pending, current_time, bbox, scene_id=scene_id)
        plan.summary["solver"] = self.solver_name
        self._normalize_plan_for_runtime(plan)

        if self.solver_name != "market":
            bad = [a.order_id for a in plan.allocations if a.mode == "B_DYNAMIC"]
            if bad:
                raise RuntimeError(
                    f"当前求解器={self.solver_name}，但返回了 B_DYNAMIC 分配: {bad}"
                )
        self._accumulate_plan_metrics(plan)

        # 应用分配（更新订单状态）
        self._apply_plan(plan, current_time)

        # 构建无人机路由信息用于前端展示
        self._build_drone_routes(plan, current_time)

        # 输出卡车路径摘要
        for truck_id, route in plan.truck_routes.items():
            logger.info(
                "[DispatchDecisionEngine] 卡车 %s 路径 %d 节点，里程 %.0fm，"
                "经停充电站 %s",
                truck_id,
                len(route.nodes),
                route.total_distance,
                route.charging_stop_ids or "（无）",
            )
            # 详细路线打印
            self._log_detailed_route(route, plan.allocations)

        return plan

    def execute_incremental(
        self,
        new_orders: dict,
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
        replan_unfinished: bool | None = None,
    ) -> DispatchPlan:
        """增量调度：由 solver 策略决定走“纯增量”或“滚动重优化”。"""
        if not new_orders:
            logger.debug("[DispatchDecisionEngine] 无新增订单")
            return DispatchPlan(
                allocations=[],
                cost_total=0.0,
                summary={"total_orders": 0, "feasible": 0, "modes": {}},
            )

        if replan_unfinished is None:
            replan_unfinished = self.solver.should_replan_unfinished()

        if replan_unfinished:
            return self._execute_replan_unfinished(
                new_orders,
                current_time,
                bbox,
                scene_id=scene_id,
            )

        self._activate_path_planner(scene_id)
        plan = self.solver.dispatch_incremental(new_orders, current_time, bbox, scene_id=scene_id)
        plan.summary["solver"] = self.solver_name
        plan.summary["dispatch_type"] = "incremental"
        plan.summary["new_orders"] = len(new_orders)
        self._normalize_plan_for_runtime(plan)

        if self.solver_name != "market":
            bad = [a.order_id for a in plan.allocations if a.mode == "B_DYNAMIC"]
            if bad:
                raise RuntimeError(
                    f"当前求解器={self.solver_name}，但返回了 B_DYNAMIC 分配: {bad}"
                )

        self._accumulate_plan_metrics(plan)
        self._apply_plan(plan, current_time, incremental=True)
        self._build_drone_routes(plan, current_time)

        for truck_id, route in plan.truck_routes.items():
            logger.info(
                "[DispatchDecisionEngine] 增量调度卡车 %s 路径 %d 节点，里程 %.0fm",
                truck_id, len(route.nodes), route.total_distance,
            )
            self._log_detailed_route(route, plan.allocations)

        return plan

    def _execute_replan_unfinished(
        self,
        new_orders: dict,
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """滚动重优化：新单 + 未开工旧单一起重分配。"""
        from core.entities.primitives import TaskStatus

        # 自愈：历史异常可能留下“pending池里状态=ASSIGNED”的脏数据。
        for oid, order in list(self.order_mgr.pending_orders.items()):
            if order.status == TaskStatus.ASSIGNED:
                self.order_mgr.pending_orders.pop(oid, None)
                self.order_mgr.assigned_orders[oid] = order

        # 不在求解前修改订单池：只构造“重优化视图”，避免异常时污染全局状态。
        replannable_assigned: dict[str, object] = {
            oid: order
            for oid, order in self.order_mgr.assigned_orders.items()
            if order.status == TaskStatus.ASSIGNED
        }
        planning_pool = dict(self.order_mgr.pending_orders)
        planning_pool.update(replannable_assigned)

        if not planning_pool:
            return DispatchPlan(
                allocations=[],
                cost_total=0.0,
                summary={
                    "total_orders": 0,
                    "feasible": 0,
                    "modes": {},
                    "dispatch_type": "dynamic_replan",
                },
            )

        self._activate_path_planner(scene_id)
        if self.solver_name == "ga_mmce":
            setattr(self.solver, "_ga_dynamic_new_order_ids", set(new_orders))
        try:
            plan = self.solver.dispatch_replan_current_state(
                planning_pool,
                current_time,
                bbox,
                scene_id=scene_id,
            )
        finally:
            if self.solver_name == "ga_mmce":
                setattr(self.solver, "_ga_dynamic_new_order_ids", set())
        plan.summary["solver"] = self.solver_name
        plan.summary["dispatch_type"] = "dynamic_replan"
        plan.summary["replanned_assigned_orders"] = len(replannable_assigned)
        plan.summary["new_orders"] = len(new_orders)
        self._normalize_plan_for_runtime(plan)

        if self.solver_name != "market":
            bad = [a.order_id for a in plan.allocations if a.mode == "B_DYNAMIC"]
            if bad:
                raise RuntimeError(
                    f"当前求解器={self.solver_name}，但返回了 B_DYNAMIC 分配: {bad}"
                )

        self._accumulate_plan_metrics(plan)
        if self.solver_name == "ga_mmce":
            replace_order_ids = self._ga_dynamic_replace_future_order_ids(plan)
            logger.info(
                "[DispatchDecisionEngine] GA 动态重优化使用增量后缀切换，替换未来订单停靠: %s",
                sorted(replace_order_ids),
            )
            self._apply_plan(
                plan,
                current_time,
                incremental=True,
                replace_future_order_ids=replace_order_ids,
            )
        else:
            self._apply_plan(plan, current_time, incremental=False)
        self._build_drone_routes(plan, current_time)

        logger.info(
            "[DispatchDecisionEngine] 动态重优化完成: new=%d 回收旧单=%d 总待分配=%d 可行=%d",
            len(new_orders),
            len(replannable_assigned),
            plan.summary.get("total_orders", 0),
            plan.summary.get("feasible", 0),
        )
        return plan

    def _normalize_plan_for_runtime(self, plan: DispatchPlan) -> None:
        if self.solver_name != "ga_mmce":
            return
        from solver.ga_mmce.runtime_adapter import normalize_ga_allocation_modes

        normalize_ga_allocation_modes(plan)

    def _ga_dynamic_replace_future_order_ids(self, plan: DispatchPlan) -> set[str]:
        """Return GA dynamic orders whose future truck stops may be replaced.

        Locked and frozen-tail orders keep their existing runtime route entries;
        only the reoptimized window is allowed to rewrite the not-yet-executed
        truck suffix.
        """
        summary = plan.summary or {}
        patch_replace = summary.get("ga_route_patch_replace_order_ids", [])
        if patch_replace:
            patch_ids = {str(oid) for oid in patch_replace if str(oid)}
            locked = {str(oid) for oid in (summary.get("locked_ids", []) or [])}
            frozen = {str(oid) for oid in (summary.get("frozen_future_order_ids", []) or [])}
            return {
                oid for oid in patch_ids
                if oid and oid not in locked and oid not in frozen
            }
        locked = {str(oid) for oid in (summary.get("locked_ids", []) or [])}
        frozen = {str(oid) for oid in (summary.get("frozen_future_order_ids", []) or [])}
        reoptimized = {
            str(oid)
            for oid in (summary.get("reoptimized_order_ids", []) or [])
        }
        if not reoptimized:
            reoptimized = {
                str(getattr(alloc, "order_id", "") or "")
                for alloc in plan.allocations
                if getattr(alloc, "feasible", False)
            }
        return {
            oid for oid in reoptimized
            if oid and oid not in locked and oid not in frozen
        }

    def _accumulate_plan_metrics(self, plan: DispatchPlan) -> None:
        """累计调度成本，用于运行时 KPI 展示。"""
        total = float(plan.cost_total or 0.0)
        if total > 0:
            self._cum_cost_total += total
        self._dispatch_count += 1

    def _estimate_runtime_energy_wh(self) -> float:
        """按实体真实运动累计估算系统总能耗（Wh）。"""
        truck_wh = 0.0
        for truck in self.entity_mgr.trucks.values():
            dist_m = float(getattr(truck, "cumulative_distance_m", 0.0) or 0.0)
            truck_wh += max(0.0, dist_m) * self._truck_energy_wh_per_meter

        drone_wh = 0.0
        for drone in self.entity_mgr.drones.values():
            energy_j = float(getattr(drone, "cumulative_energy_j", 0.0) or 0.0)
            drone_wh += max(0.0, energy_j) / 3600.0

        return max(0.0, truck_wh + drone_wh)

    def get_runtime_metrics(self) -> dict:
        """返回调度运行时累计指标。"""
        return {
            "dispatch_count": self._dispatch_count,
            "total_energy_cost_wh": self._estimate_runtime_energy_wh(),
            "total_dispatch_cost": max(0.0, self._cum_cost_total),
            "active_solver": self.solver_name,
        }

    def try_fulfill_contracts(self, current_time: float) -> None:
        """尝试兑现已完成回收的契约。

        遍历求解器的活跃契约，若无人机已不再飞行（IDLE/CHARGING）
        且当前时刻已超过 uav_arrival_time，则标记契约为 fulfilled。
        供仿真引擎在每个 tick 中调用。
        """
        for contract in self.solver.get_active_contracts():
            if contract.status != "active":
                continue
            drone = self.entity_mgr.drones.get(contract.drone_id)
            if drone is None:
                continue
            if current_time >= contract.uav_arrival_time and not drone.status.is_flying():
                self.solver.fulfill_contract(contract.contract_id)

    def _apply_plan(
        self,
        plan: DispatchPlan,
        current_time: float,
        incremental: bool = False,
        replace_future_order_ids: set[str] | None = None,
    ) -> None:
        """
        应用分配方案，更新订单和实体状态。

        Args:
            incremental: 增量模式。为 True 时保护正在执行中的卡车路由，
                         仅合并未执行后缀，不覆盖已执行/正在执行路线。
            replace_future_order_ids: 增量模式下，先从未执行后缀移除这些订单的旧停靠，
                         再插入新计划中的对应停靠。
        """
        for alloc in plan.allocations:
            order = self.order_mgr.pending_orders.get(alloc.order_id)
            source_pool = "pending"
            if not order:
                order = self.order_mgr.assigned_orders.get(alloc.order_id)
                source_pool = "assigned"
            if not order:
                continue

            if alloc.feasible:
                # 将订单转入 assigned
                if source_pool == "pending":
                    self.order_mgr.pending_orders.pop(alloc.order_id, None)
                self.order_mgr.assigned_orders[alloc.order_id] = order

                # 更新订单的分配信息
                order.assigned_vehicle_id = alloc.vehicle_id
                order.assigned_mode = alloc.mode
                
                # 状态更新：新单执行 PENDING→ASSIGNED；重优化中的旧单保持 ASSIGNED。
                from core.entities.primitives import TaskStatus
                if order.status == TaskStatus.PENDING:
                    order.update_status(TaskStatus.ASSIGNED)
                elif order.status != TaskStatus.ASSIGNED:
                    logger.warning(
                        "[DispatchDecisionEngine] 订单 %s 当前状态=%s，不执行 ASSIGNED 覆盖",
                        alloc.order_id,
                        order.status.value,
                    )

                if alloc.mode in ("B_WAIT", "B_DYNAMIC"):
                    logger.info(
                        "[DispatchDecisionEngine] 分配 %s 至 %s（模式 %s, 出发站 %s, 距离 %.1fm, 回收点 %s）",
                        alloc.order_id,
                        alloc.vehicle_id,
                        alloc.mode,
                        alloc.launch_station_id or "-",
                        alloc.distance,
                        alloc.recovery_station_id or "-",
                    )
                else:
                    logger.info(
                        "[DispatchDecisionEngine] 分配 %s 至 %s（模式 %s, 距离 %.1fm, 回收点 %s）",
                        alloc.order_id,
                        alloc.vehicle_id,
                        alloc.mode,
                        alloc.distance,
                        alloc.recovery_station_id or "-",
                    )
            else:
                # 不可行，保持 pending
                logger.warning(
                    "[DispatchDecisionEngine] 分配失败 %s: %s",
                    alloc.order_id,
                    alloc.reason,
                )

        # ── 第二步：应用卡车路由 ────────────────────────────────────────────
        for truck_id, route in plan.truck_routes.items():
            truck = self.entity_mgr.trucks.get(truck_id)
            if truck is None:
                logger.warning("[DispatchDecisionEngine] 卡车 %s 不存在", truck_id)
                continue

            self._recalculate_truck_route_timing_for_b_wait(route, plan.allocations, current_time)

            if incremental:
                # 增量模式：保护正在执行的卡车路由，只合并未执行后缀
                self._merge_incremental_truck_stops(
                    truck,
                    route,
                    plan.allocations,
                    current_time,
                    replace_future_order_ids=replace_future_order_ids,
                    route_patch=(
                        plan.summary.get("ga_route_patch")
                        if self.solver_name == "ga_mmce" and bool(plan.summary.get("ga_route_patch_enabled", False))
                        else None
                    ),
                )
                continue

            route_nodes = [node.node_id for node in route.nodes]
            route_positions = [node.position for node in route.nodes]

            try:
                truck.set_route(route_nodes, route_positions, current_time, geometry=route.geometry)
                truck._planned_route_stops = [
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "position": node.position,
                        "arrival_time": node.arrival_time,
                        "departure_time": node.departure_time,
                        "order_id": node.order_id,
                        "action": str(getattr(node, "action", "") or ""),
                    }
                    for node in route.nodes
                ]
                truck._planned_route_solver = self.solver_name
                truck._planned_route_cursor = 0
                logger.info(
                    "[DispatchDecisionEngine] 已将路由应用至卡车 %s（%d 个关键节点，几何路径 %d 点）",
                    truck_id, len(route_nodes), len(route.geometry),
                )
            except Exception as e:
                logger.exception(
                    "[DispatchDecisionEngine] 为卡车 %s 应用路由失败: %s",
                    truck_id, str(e),
                )

        # ── 补齐 B_WAIT/B_DYNAMIC 起飞时刻：使用卡车实际到达出发站时刻 ─────────
        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode not in ("B_WAIT", "B_DYNAMIC") or not alloc.launch_station_id:
                continue
            truck_route = plan.truck_routes.get(alloc.vehicle_id)
            if truck_route is None:
                continue

            launch_candidates = [
                n for n in truck_route.nodes
                if n.node_id == alloc.launch_station_id
            ]
            if not launch_candidates:
                continue

            if alloc.launch_time > 0:
                target_arrival = alloc.launch_time - self.TRUCK_DRONE_LAUNCH_TIME
                future_candidates = [
                    n for n in launch_candidates
                    if n.arrival_time >= current_time - 1e-6
                ]
                search_candidates = future_candidates or launch_candidates
                not_earlier = [n for n in search_candidates if n.arrival_time >= target_arrival - 1e-6]
                if not_earlier:
                    launch_node = min(not_earlier, key=lambda n: n.arrival_time - target_arrival)
                else:
                    launch_node = min(
                        search_candidates,
                        key=lambda n: abs(n.arrival_time - target_arrival),
                    )
            else:
                future_candidates = [
                    n for n in launch_candidates
                    if n.arrival_time >= current_time - 1e-6
                ]
                launch_node = future_candidates[0] if future_candidates else launch_candidates[0]

            alloc.launch_time = launch_node.arrival_time + self.TRUCK_DRONE_LAUNCH_TIME

        # ── 第三步：应用无人机路由 ────────────────────────────────────────────
        self._setup_drone_routes(plan, current_time)

    def _merge_incremental_truck_stops(
        self,
        truck,
        new_route: "TruckRoute",
        allocations: list,
        current_time: float,
        replace_future_order_ids: set[str] | None = None,
        route_patch: dict | None = None,
    ) -> None:
        """增量模式下合并卡车未执行后缀，保护已经执行/正在执行的路线。

        原则：
          - 保留已执行和正在执行的停靠事件（cursor 之前）
          - 将新的 recovery / customer 停靠按时间顺序插入未执行区间
          - 后缀重建从卡车当前位置开始，避免把车辆拉回 GA snapshot 路线起点
        """
        existing_stops: list[dict] = getattr(truck, "_planned_route_stops", None) or []
        cursor = int(getattr(truck, "_planned_route_cursor", 0))
        truck._planned_route_solver = self.solver_name
        route_patch = route_patch or {}
        patch_enabled = self.solver_name == "ga_mmce" and bool(route_patch.get("enabled", False))
        patch_replace_ids = {
            str(oid)
            for oid in (route_patch.get("replace_order_ids", []) or [])
            if str(oid)
        }
        patch_anchor_by_order = {
            str(order_id): dict(anchor or {})
            for order_id, anchor in (route_patch.get("anchor_by_order", {}) or {}).items()
        }

        if not existing_stops:
            # 卡车尚未有路线（首次被增量调度命中），按全量模式设置
            route_nodes = [node.node_id for node in new_route.nodes]
            route_positions = [node.position for node in new_route.nodes]
            try:
                truck.set_route(route_nodes, route_positions, current_time, geometry=new_route.geometry)
                truck._planned_route_stops = [
                    {
                        "node_id": n.node_id, "node_type": n.node_type,
                        "position": n.position, "arrival_time": n.arrival_time,
                        "departure_time": n.departure_time, "order_id": n.order_id,
                        "action": str(getattr(n, "action", "") or ""),
                    }
                    for n in new_route.nodes
                ]
                truck._planned_route_solver = self.solver_name
                truck._planned_route_cursor = 0
                logger.info(
                    "[DispatchDecisionEngine] 增量模式首次为卡车 %s 设置路由 (%d 节点)",
                    truck.truck_id, len(route_nodes),
                )
                self._sync_waiting_drone_launch_times_for_truck(truck, current_time)
            except Exception as e:
                logger.exception("[DispatchDecisionEngine] 增量首次路由设置失败 %s: %s", truck.truck_id, e)
            return

        # 仅对未执行区间做去重：已执行过的同站点在新批次中应允许再次停靠。
        future_stops = existing_stops[cursor:]
        replace_ids = patch_replace_ids or {str(oid) for oid in (replace_future_order_ids or set())}
        if replace_ids:
            before = len(future_stops)
            future_stops = [
                stop for stop in future_stops
                if str(stop.get("order_id", "") or "") not in replace_ids
            ]
            removed = before - len(future_stops)
            if removed > 0:
                logger.info(
                    "[DispatchDecisionEngine] 增量后缀切换：卡车 %s 移除未来旧停靠 %d 个 orders=%s",
                    truck.truck_id,
                    removed,
                    sorted(replace_ids),
                )

        def _stop_key(stop_dict: dict) -> tuple:
            node_type = stop_dict.get("node_type", "")
            node_id = stop_dict.get("node_id", "")
            if self.solver_name == "ga_mmce":
                order_id = str(stop_dict.get("order_id", "") or "")
                action = str(stop_dict.get("action", "") or "")
                return (node_type, node_id, order_id, action)
            return (node_type, node_id)

        existing_future_keys = {_stop_key(s) for s in future_stops}

        # 提取新路由中需要追加的事件性节点（customer / recovery / station）
        new_stops = []
        for node in new_route.nodes:
            if node.node_type in ("customer", "recovery", "station"):
                order_id = str(node.order_id or "")
                if patch_enabled and order_id not in replace_ids:
                    continue
                key = _stop_key(
                    {
                        "node_type": node.node_type,
                        "node_id": node.node_id,
                        "order_id": order_id,
                        "action": str(getattr(node, "action", "") or ""),
                    }
                )
                if key in existing_future_keys:
                    continue
                new_stops.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "position": node.position,
                    "arrival_time": node.arrival_time,
                    "departure_time": node.departure_time,
                    "order_id": order_id,
                    "action": str(getattr(node, "action", "") or ""),
                })
                existing_future_keys.add(key)

        if not new_stops:
            if patch_enabled:
                truck._planned_route_stops = existing_stops
                self._sync_waiting_drone_launch_times_for_truck(truck, current_time)
                logger.info(
                    "[DispatchDecisionEngine] GA route patch 卡车 %s 无新增停靠，保留旧后缀",
                    truck.truck_id,
                )
                return
            # 即便无新增停靠，也需要应用本批次后缀更新（例如 recovery 等待时长更新）。
            # 只更新已有 future_stops 的时间，切勿截断其他未来停靠点。
            refreshed_future = [dict(stop) for stop in future_stops]
            for node in new_route.nodes:
                if node.node_type in ("customer", "recovery", "station"):
                    node_key = _stop_key({
                        "node_type": node.node_type,
                        "node_id": node.node_id,
                        "order_id": node.order_id,
                        "action": str(getattr(node, "action", "") or ""),
                    })
                    for stop in refreshed_future:
                        if _stop_key(stop) == node_key:
                            # 累加等待时间或取最大值，保持原有业务逻辑（至少确保 departure_time 合理）
                            stop["departure_time"] = max(
                                float(stop.get("departure_time", 0)),
                                float(node.departure_time)
                            )
            
            truck._planned_route_stops = existing_stops[:cursor] + refreshed_future
            # 重新构建物理路由，保证旧停靠点不丢失
            try:
                rebuilt_route = self.solver.build_incremental_route_from_stops(
                    truck,
                    refreshed_future,
                    current_time,
                )
                if rebuilt_route and len(rebuilt_route.nodes) >= 2:
                    truck.set_route(
                        [n.node_id for n in rebuilt_route.nodes],
                        [n.position for n in rebuilt_route.nodes],
                        current_time,
                        geometry=rebuilt_route.geometry,
                    )
            except Exception as e:
                logger.exception(
                    "[DispatchDecisionEngine] 增量无新增停靠时应用后缀失败 %s: %s",
                    truck.truck_id,
                    e,
                )
            self._sync_waiting_drone_launch_times_for_truck(truck, current_time)
            logger.info(
                "[DispatchDecisionEngine] 增量模式卡车 %s 无新增停靠，已刷新后缀时序",
                truck.truck_id,
            )
            return

        new_route_stops = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "position": node.position,
                "arrival_time": node.arrival_time,
                "departure_time": node.departure_time,
                "order_id": str(node.order_id or ""),
                "action": str(getattr(node, "action", "") or ""),
            }
            for node in new_route.nodes
            if node.node_type in ("customer", "recovery", "station")
            and (not patch_enabled or str(node.order_id or "") in replace_ids)
        ]

        # 稳定增量：保留旧 future 的相对顺序，只插入新停靠，避免前批动态单被后批重排。
        merged: list[dict] = [dict(stop) for stop in future_stops]

        def _last_idx_of_key(seq: list[dict], key: tuple) -> int:
            for i in range(len(seq) - 1, -1, -1):
                if _stop_key(seq[i]) == key:
                    return i
            return -1

        def _first_idx_of_key(seq: list[dict], key: tuple) -> int:
            for i, stop in enumerate(seq):
                if _stop_key(stop) == key:
                    return i
            return -1

        def _last_idx_of_order(seq: list[dict], order_id: str) -> int:
            for i in range(len(seq) - 1, -1, -1):
                if str(seq[i].get("order_id", "") or "") == order_id:
                    return i
            return -1

        def _first_idx_of_order(seq: list[dict], order_id: str) -> int:
            for i, stop in enumerate(seq):
                if str(stop.get("order_id", "") or "") == order_id:
                    return i
            return -1

        inserted_via_anchor = 0
        for idx, stop in enumerate(new_route_stops):
            key = _stop_key(stop)
            if _last_idx_of_key(merged, key) >= 0:
                continue

            insert_at = len(merged)

            order_id = str(stop.get("order_id", "") or "")
            if patch_enabled and order_id:
                same_order_prev_idx = _last_idx_of_order(merged, order_id)
                if same_order_prev_idx >= 0:
                    insert_at = same_order_prev_idx + 1
                else:
                    anchor = patch_anchor_by_order.get(order_id, {})
                    after_order_id = str(anchor.get("after_order_id", "") or "")
                    before_order_id = str(anchor.get("before_order_id", "") or "")
                    after_idx = _last_idx_of_order(merged, after_order_id) if after_order_id else -1
                    before_idx = _first_idx_of_order(merged, before_order_id) if before_order_id else -1
                    if after_idx >= 0:
                        insert_at = after_idx + 1
                    elif before_idx >= 0:
                        insert_at = before_idx
            else:
                # 先找前驱锚点：尽量插在新路由语义上的前驱之后。
                for prev in reversed(new_route_stops[:idx]):
                    prev_key = _stop_key(prev)
                    prev_idx = _last_idx_of_key(merged, prev_key)
                    if prev_idx >= 0:
                        insert_at = prev_idx + 1
                        break
                else:
                    # 若没有前驱锚点，再找后继锚点并插到其前。
                    for nxt in new_route_stops[idx + 1:]:
                        nxt_key = _stop_key(nxt)
                        nxt_idx = _first_idx_of_key(merged, nxt_key)
                        if nxt_idx >= 0:
                            insert_at = nxt_idx
                            break

            merged.insert(insert_at, dict(stop))
            inserted_via_anchor += 1

        logger.info(
            "[DispatchDecisionEngine] 增量模式卡车 %s 稳定插入新停靠 %d 个（旧future=%d）",
            truck.truck_id,
            inserted_via_anchor,
            len(future_stops),
        )

        # 重新按卡车当前位置进行后缀时序推演，避免“时刻表触发快于物理到达”。
        retimed_future: list[dict] = []
        cur_pos = truck.get_location(current_time)
        cur_time = current_time
        speed = max(1e-6, float(getattr(truck, "speed", 0.0)))
        for stop in merged:
            stop_pos = stop.get("position")
            if stop_pos is None:
                continue
            travel_time = cur_pos.distance_2d(stop_pos) / speed
            arrival = cur_time + travel_time
            prev_arrival = float(stop.get("arrival_time", arrival))
            prev_departure = float(stop.get("departure_time", prev_arrival))
            service_time = max(0.0, prev_departure - prev_arrival)
            departure = arrival + service_time

            updated = dict(stop)
            updated["arrival_time"] = arrival
            updated["departure_time"] = departure
            retimed_future.append(updated)

            cur_pos = stop_pos
            cur_time = departure

        rebuilt_route = None
        try:
            rebuilt_route = self.solver.build_incremental_route_from_stops(
                truck,
                retimed_future,
                current_time,
            )
        except Exception as e:
            logger.exception(
                "[DispatchDecisionEngine] OSM 后缀路线重建失败 %s: %s",
                truck.truck_id,
                e,
            )

        if rebuilt_route is not None and len(rebuilt_route.nodes) >= 2:
            rebuilt_stops = [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "position": n.position,
                    "arrival_time": n.arrival_time,
                    "departure_time": n.departure_time,
                    "order_id": n.order_id,
                    "action": str(getattr(n, "action", "") or ""),
                }
                for n in rebuilt_route.nodes
                if n.node_type in ("customer", "recovery", "station")
            ]
            truck._planned_route_stops = existing_stops[:cursor] + rebuilt_stops
            try:
                truck.set_route(
                    [n.node_id for n in rebuilt_route.nodes],
                    [n.position for n in rebuilt_route.nodes],
                    current_time,
                    geometry=rebuilt_route.geometry,
                )
            except Exception as e:
                logger.exception(
                    "[DispatchDecisionEngine] 应用 OSM 后缀路线失败 %s: %s",
                    truck.truck_id,
                    e,
                )
        else:
            # 兜底：若 OSM 后缀重建失败，保留旧计划避免不可达停靠阻塞后续事件游标。
            logger.warning(
                "[DispatchDecisionEngine] 增量模式卡车 %s 后缀重建失败，保留旧计划（不写入新停靠）",
                truck.truck_id,
            )
            truck._planned_route_stops = existing_stops

        self._sync_waiting_drone_launch_times_for_truck(truck, current_time)

        logger.info(
            "[DispatchDecisionEngine] 增量模式追加 %d 个停靠到卡车 %s "
            "(cursor=%d, 总计 %d 停靠)",
            len(new_stops), truck.truck_id, cursor, len(truck._planned_route_stops),
        )

    def _sync_waiting_drone_launch_times_for_truck(self, truck, current_time: float) -> None:
        """将车上等待起飞无人机的 launch_time 对齐到卡车当前时刻表。"""
        planned_stops: list[dict] = getattr(truck, "_planned_route_stops", None) or []
        if not planned_stops:
            return

        arrivals_by_node: dict[str, list[float]] = {}
        arrivals_by_order_node: dict[tuple[str, str], list[float]] = {}
        for stop in planned_stops:
            node_id = str(stop.get("node_id", "") or "")
            if not node_id:
                continue
            arr = float(stop.get("arrival_time", float("inf")))
            if not math.isfinite(arr):
                continue
            arrivals_by_node.setdefault(node_id, []).append(arr)
            order_id = str(stop.get("order_id", "") or "")
            if order_id:
                arrivals_by_order_node.setdefault((order_id, node_id), []).append(arr)

        for node_id in arrivals_by_node:
            arrivals_by_node[node_id].sort()
        for key in arrivals_by_order_node:
            arrivals_by_order_node[key].sort()

        def _active_ga_launch_order_id(drone, launch_station_id: str) -> str:
            segments = getattr(drone, "_ga_runtime_segments", []) or []
            if not segments:
                return ""
            current_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            matching = [
                segment for segment in segments
                if bool(getattr(segment, "truck_launch", False))
                and str(getattr(segment, "launch_node_id", "") or "") == launch_station_id
            ]
            if not matching:
                return ""
            for segment in matching:
                if int(getattr(segment, "start_idx", 0) or 0) >= current_idx:
                    return str(getattr(segment, "order_id", "") or "")
            return str(getattr(matching[-1], "order_id", "") or "")

        updated = 0
        for drone in self.entity_mgr.drones.values():
            if getattr(drone, "transport_truck_id", "") != truck.truck_id:
                continue

            launch_station_id = getattr(drone, "launch_station_id", "")
            if not launch_station_id:
                continue

            active_order_id = _active_ga_launch_order_id(drone, launch_station_id)
            station_arrivals = []
            if self.solver_name == "ga_mmce" and active_order_id:
                station_arrivals = arrivals_by_order_node.get((active_order_id, launch_station_id), [])
            if not station_arrivals:
                station_arrivals = arrivals_by_node.get(launch_station_id, [])
            future_arrivals = [
                t for t in station_arrivals
                if t >= current_time - 1e-6
            ]
            if not future_arrivals:
                last_log_t = float(getattr(drone, "_last_missing_launch_station_log_time", -1e9))
                if current_time - last_log_t >= 10.0:
                    future_node_ids = [
                        str(s.get("node_id", ""))
                        for s in planned_stops
                        if float(s.get("arrival_time", float("inf"))) >= current_time - 1e-6
                    ]
                    logger.warning(
                        "[DispatchDecisionEngine] 卡车 %s 未找到等待无人机 %s 的起飞站 %s order=%s；"
                        "未来停靠=%s",
                        truck.truck_id,
                        drone.drone_id,
                        launch_station_id,
                        active_order_id or "-",
                        future_node_ids[:12],
                    )
                    drone._last_missing_launch_station_log_time = current_time
                continue
            station_arrival = future_arrivals[0]

            target_launch = max(
                current_time,
                station_arrival + self.TRUCK_DRONE_LAUNCH_TIME,
            )
            prev_launch = float(getattr(drone, "scheduled_launch_time", target_launch))
            if abs(prev_launch - target_launch) <= 1e-6:
                continue

            drone.scheduled_launch_time = target_launch
            updated += 1
            logger.info(
                "[DispatchDecisionEngine] GA-MMCE launch sync truck=%s drone=%s order=%s station=%s "
                "station_arrival=%.1f scheduled_launch_time=%.1f current_time=%.1f",
                truck.truck_id,
                drone.drone_id,
                active_order_id or "-",
                launch_station_id,
                station_arrival,
                target_launch,
                current_time,
            )

        if updated > 0:
            logger.info(
                "[DispatchDecisionEngine] 卡车 %s 同步 %d 架等待无人机起飞时刻",
                truck.truck_id,
                updated,
            )

    def _plan_drone_leg(
        self,
        start: Position3D,
        goal: Position3D,
        altitude: float,
    ) -> list[Position3D]:
        if self._active_path_planner is None:
            return [start, goal]
        path = self._active_path_planner.plan(start, goal, altitude)
        if len(path) < 2:
            return [start, goal]
        return path

    @staticmethod
    def _merge_drone_legs(legs: list[list[Position3D]]) -> list[Position3D]:
        merged: list[Position3D] = []
        for leg in legs:
            if not leg:
                continue
            if not merged:
                merged.extend(leg)
                continue
            if merged[-1].distance_2d(leg[0]) <= 1e-6:
                merged.extend(leg[1:])
            else:
                merged.extend(leg)
        return merged

    def _build_uav_polyline(
        self,
        launch_loc: Position3D,
        delivery_loc: Position3D,
        recovery_loc: Position3D,
        altitude: float,
    ) -> tuple[list[Position3D], int]:
        leg_out = self._plan_drone_leg(launch_loc, delivery_loc, altitude)
        leg_back = self._plan_drone_leg(delivery_loc, recovery_loc, altitude)
        polyline = self._merge_drone_legs([leg_out, leg_back])
        delivery_idx = max(0, len(leg_out) - 1)
        if polyline and delivery_idx >= len(polyline) - 1:
            insert_idx = max(0, len(polyline) - 1)
            polyline.insert(insert_idx, delivery_loc)
            delivery_idx = insert_idx
        return polyline, delivery_idx

    @staticmethod
    def _polyline_distance(polyline: list[Position3D]) -> float:
        if len(polyline) < 2:
            return 0.0
        return sum(polyline[i - 1].distance_2d(polyline[i]) for i in range(1, len(polyline)))

    def _log_uav_path_if_obstacle_avoided(
        self,
        alloc: "AllocationResult",
        polyline: list[Position3D],
        launch_loc: Position3D,
        delivery_loc: Position3D,
        recovery_loc: Position3D,
        tag: str,
    ) -> None:
        if len(polyline) < 2:
            return
        direct_dist = launch_loc.distance_2d(delivery_loc) + delivery_loc.distance_2d(recovery_loc)
        path_dist = self._polyline_distance(polyline)
        has_detour = len(polyline) > 3 or path_dist > direct_dist + 1.0

        path_utm = [[round(p.x, 2), round(p.y, 2), round(p.z, 2)] for p in polyline]
        path_wgs84 = []
        for p in polyline:
            lon, lat = p.to_wgs84()
            path_wgs84.append([round(lon, 6), round(lat, 6)])

        log_prefix = "[DispatchDecisionEngine][UAV-AVOID]" if has_detour else "[DispatchDecisionEngine][UAV-DIRECT]"

        logger.info(
            "%s %s order=%s drone=%s mode=%s points=%d path_dist=%.1f direct=%.1f "
            "launch=(%.2f,%.2f) delivery=(%.2f,%.2f) recovery=(%.2f,%.2f) path_utm=%s path_wgs84=%s",
            log_prefix,
            tag,
            alloc.order_id,
            alloc.drone_id,
            alloc.mode,
            len(polyline),
            path_dist,
            direct_dist,
            launch_loc.x,
            launch_loc.y,
            delivery_loc.x,
            delivery_loc.y,
            recovery_loc.x,
            recovery_loc.y,
            path_utm,
            path_wgs84,
        )

    @staticmethod
    def _polyline_to_waypoints(
        polyline: list[Position3D],
        delivery_idx: int,
        order_id: str,
        recovery_id: str,
    ) -> list[RouteWaypoint]:
        if not polyline:
            return []
        last_idx = len(polyline) - 1
        waypoints: list[RouteWaypoint] = []
        for idx, pos in enumerate(polyline):
            if idx == 0:
                action = WaypointAction.PICKUP
                target_id = order_id
            elif idx == last_idx:
                action = WaypointAction.DOCK_DEPOT
                target_id = recovery_id
            elif idx == delivery_idx:
                action = WaypointAction.DELIVER
                target_id = order_id
            else:
                # RouteWaypoint 需要动作语义；中间点使用无副作用的 PICKUP 充当过渡点。
                action = WaypointAction.PICKUP
                target_id = order_id
            waypoints.append(RouteWaypoint(pos, action, target_id))
        return waypoints

    def _recalculate_truck_route_timing_for_b_wait(
        self,
        route: "TruckRoute",
        allocations: list["AllocationResult"],
        current_time: float,
    ) -> None:
        """
        按 B_WAIT 的起飞站和回收站关系重算卡车停靠时序。

        关键逻辑：
          - recovery 节点是否需要等待，取决于“无人机预计返航时刻”与“卡车到站时刻”的差
          - 无人机预计返航时刻 = launch_station_arrival + alloc.wait_duration
          - 若卡车到达 recovery 时无人机已返航，则 recovery 等待时间应为 0
        """
        if len(route.nodes) < 2:
            return

        related_allocs = [
            alloc for alloc in allocations
            if alloc.feasible
            and alloc.mode == "B_WAIT"
            and alloc.vehicle_id == route.truck_id
            and alloc.recovery_station_id
        ]

        allocs_by_recovery: dict[str, list["AllocationResult"]] = {}
        allocs_by_launch: dict[str, list["AllocationResult"]] = {}
        for alloc in related_allocs:
            allocs_by_recovery.setdefault(alloc.recovery_station_id, []).append(alloc)
            if alloc.launch_station_id:
                allocs_by_launch.setdefault(alloc.launch_station_id, []).append(alloc)

        # 并入运行时承诺：当前批次外、但实际在途/待回收的无人机也会约束 recovery 等待。
        runtime_recovery_eta: dict[str, float] = {}

        def _resolve_owner_truck_id(drone) -> str:
            carrier_id = getattr(drone, "transport_truck_id", "")
            if carrier_id in self.entity_mgr.trucks:
                return carrier_id
            for tid, t in self.entity_mgr.trucks.items():
                if drone.drone_id in getattr(t, "docked_drones", []):
                    return tid
            if getattr(drone, "home_type", None) == SourceType.TRUCK and getattr(drone, "home_id", "") in self.entity_mgr.trucks:
                return getattr(drone, "home_id", "")
            if len(self.entity_mgr.trucks) == 1:
                return next(iter(self.entity_mgr.trucks.keys()))
            return ""

        for drone in self.entity_mgr.drones.values():
            owner_truck_id = _resolve_owner_truck_id(drone)
            if owner_truck_id != route.truck_id:
                continue

            waiting_station_id = getattr(drone, "waiting_recovery_station_id", "")
            if waiting_station_id and waiting_station_id in self.entity_mgr.stations:
                eta = current_time
                runtime_recovery_eta[waiting_station_id] = max(
                    runtime_recovery_eta.get(waiting_station_id, 0.0),
                    eta,
                )
                continue

            # 车载待起飞：即便尚未进入 flying，也要提前占位回收等待窗口。
            transport_truck_id = getattr(drone, "transport_truck_id", "")
            launch_station_id = getattr(drone, "launch_station_id", "")
            scheduled_launch_time = float(getattr(drone, "scheduled_launch_time", 0.0) or 0.0)
            if (
                transport_truck_id == route.truck_id
                and launch_station_id
                and launch_station_id in self.entity_mgr.stations
                and math.isfinite(scheduled_launch_time)
                and scheduled_launch_time >= current_time - 1e-6
            ):
                route_plan = getattr(drone, "route_plan", None) or []
                start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
                if 0 <= start_idx < len(route_plan):
                    cur = route_plan[start_idx].loc
                else:
                    cur = self.entity_mgr.stations[launch_station_id].location

                remaining_dist = 0.0
                extra_service = 0.0
                dock_station_id = ""
                for wp in route_plan[start_idx:]:
                    remaining_dist += cur.distance_3d(wp.loc)
                    cur = wp.loc
                    if wp.action == WaypointAction.DELIVER:
                        extra_service += float(getattr(self.entity_mgr, "DRONE_SERVICE_TIME_ORDER", 0.0))
                    if wp.action in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                        target_id = wp.target_entity_id or ""
                        if target_id in self.entity_mgr.stations:
                            dock_station_id = target_id
                        break

                if dock_station_id:
                    cruise_speed = max(1e-6, float(getattr(drone, "cruise_speed", 0.0)))
                    eta = max(current_time, scheduled_launch_time) + remaining_dist / cruise_speed + extra_service
                    runtime_recovery_eta[dock_station_id] = max(
                        runtime_recovery_eta.get(dock_station_id, 0.0),
                        eta,
                    )
                    continue

            if not getattr(drone.status, "is_flying", False):
                continue

            route_plan = getattr(drone, "route_plan", None) or []
            start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            if start_idx >= len(route_plan):
                continue

            cur = drone.current_loc
            remaining_dist = 0.0
            extra_service = 0.0
            dock_station_id = ""
            for wp in route_plan[start_idx:]:
                remaining_dist += cur.distance_3d(wp.loc)
                cur = wp.loc
                if wp.action == WaypointAction.DELIVER:
                    extra_service += float(getattr(self.entity_mgr, "DRONE_SERVICE_TIME_ORDER", 0.0))
                if wp.action in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                    target_id = wp.target_entity_id or ""
                    if target_id in self.entity_mgr.stations:
                        dock_station_id = target_id
                    break

            if not dock_station_id:
                continue

            cruise_speed = max(1e-6, float(getattr(drone, "cruise_speed", 0.0)))
            eta = current_time + remaining_dist / cruise_speed + extra_service
            runtime_recovery_eta[dock_station_id] = max(
                runtime_recovery_eta.get(dock_station_id, 0.0),
                eta,
            )

        original_arrivals = [node.arrival_time for node in route.nodes]
        original_departures = [node.departure_time for node in route.nodes]
        base_services = [
            self._resolve_truck_route_base_service_time(
                node_type=node.node_type,
                original_service_time=max(0.0, dep - arr),
            )
            for node, arr, dep in zip(route.nodes, original_arrivals, original_departures)
        ]
        travel_deltas = [0.0]
        for i in range(1, len(route.nodes)):
            travel_deltas.append(max(0.0, original_arrivals[i] - original_departures[i - 1]))

        observed_arrivals: dict[str, list[float]] = {}

        def resolve_launch_arrival(alloc: "AllocationResult") -> float:
            """在已到访记录中匹配该任务对应的起飞站到达时刻。"""
            visits = observed_arrivals.get(alloc.launch_station_id, [])
            if not visits:
                if alloc.launch_time > 0:
                    return alloc.launch_time - self.TRUCK_DRONE_LAUNCH_TIME
                return current_time

            if alloc.launch_time > 0:
                target_arrival = alloc.launch_time - self.TRUCK_DRONE_LAUNCH_TIME
                prior_visits = [t for t in visits if t <= target_arrival + 1e-6]
                if prior_visits:
                    return max(prior_visits)
                return min(visits, key=lambda t: abs(t - target_arrival))

            return visits[-1]

        for i, node in enumerate(route.nodes):
            if i == 0:
                arrival = original_arrivals[0]
            else:
                arrival = route.nodes[i - 1].departure_time + travel_deltas[i]

            observed_arrivals.setdefault(node.node_id, []).append(arrival)

            if node.node_type == "recovery":
                launch_ops = allocs_by_launch.get(node.node_id, [])
                recovery_ops = allocs_by_recovery.get(node.node_id, [])
                runtime_eta = runtime_recovery_eta.get(node.node_id)
                service_time = base_services[i]

                needed_departure = arrival + service_time
                for alloc in recovery_ops:
                    launch_arrival = resolve_launch_arrival(alloc)
                    expected_recovery_time = launch_arrival + alloc.wait_duration + self.RECOVERY_EVENT_GUARD_S
                    needed_departure = max(needed_departure, expected_recovery_time)
                if runtime_eta is not None:
                    needed_departure = max(
                        needed_departure,
                        runtime_eta + self.TRUCK_DRONE_RECOVER_TIME + self.RECOVERY_EVENT_GUARD_S,
                    )
                service_time = max(service_time, needed_departure - arrival)
                op_hold = 0.0
                if launch_ops:
                    op_hold = max(op_hold, self.TRUCK_DRONE_LAUNCH_TIME)
                if recovery_ops:
                    op_hold = max(op_hold, self.TRUCK_DRONE_RECOVER_TIME)
                if runtime_eta is not None:
                    op_hold = max(op_hold, self.TRUCK_DRONE_RECOVER_TIME)
                if op_hold > 0.0:
                    # 若同站同时有放飞和回收，按可并行处理取 max，而非累加。
                    service_time = max(service_time, op_hold)
            else:
                service_time = base_services[i]

            departure = arrival + service_time
            node.arrival_time = arrival
            node.departure_time = departure

    def _resolve_truck_route_base_service_time(
        self,
        *,
        node_type: str,
        original_service_time: float,
    ) -> float:
        base_service_time = max(0.0, float(original_service_time))
        if node_type == "customer":
            return max(base_service_time, self.TRUCK_SERVICE_TIME_ORDER)
        if node_type in {"station", "recovery"}:
            return max(base_service_time, self.TRUCK_STATION_HOLD_TIME)
        return base_service_time

    def _log_detailed_route(self, route: "TruckRoute", allocations: list["AllocationResult"]) -> None:
        """
        打印卡车的详细路线，包括每个节点的信息和关联的订单/无人机操作。
        """
        logger.info("[DispatchDecisionEngine] 卡车 %s 详细路线：", route.truck_id)
        for i, node in enumerate(route.nodes):
            action_desc = self._get_node_action_description(node, allocations)
            logger.info(
                "  [%d] %s (%s) - 到达: %.1fs, 离开: %.1fs | %s",
                i + 1,
                node.node_id,
                node.node_type,
                node.arrival_time,
                node.departure_time,
                action_desc,
            )

    def _get_node_action_description(self, node: "TruckRouteNode", allocations: list["AllocationResult"]) -> str:
        """
        根据节点类型和关联信息，生成动作描述。
        """
        if node.node_type == "depot":
            if "_return" in node.node_id:
                return "返回仓库"
            else:
                return "从仓库出发"
        elif node.node_type == "customer":
            return f"配送订单 {node.order_id}"
        action = str(getattr(node, "action", "") or "")
        if action == "launch":
            return f"放飞无人机（订单: {node.order_id or '-'}）"
        if action == "recover":
            return f"回收无人机（订单: {node.order_id or '-'}）"
        if node.node_type == "recovery":
            launch_orders = [
                alloc.order_id for alloc in allocations
                if alloc.feasible and alloc.mode in ("B_WAIT", "B_DYNAMIC") and alloc.launch_station_id == node.node_id
            ]
            recovery_orders = [
                alloc.order_id for alloc in allocations
                if alloc.feasible and alloc.mode in ("B", "B_WAIT") and alloc.recovery_station_id == node.node_id
            ]

            if launch_orders and recovery_orders:
                return (
                    f"放飞+回收无人机（放飞订单: {', '.join(launch_orders)}；"
                    f"回收订单: {', '.join(recovery_orders)}）"
                )
            if launch_orders:
                return f"放飞无人机（订单: {', '.join(launch_orders)}）"
            if recovery_orders:
                return f"回收无人机（订单: {', '.join(recovery_orders)}）"
            return "回收锚点停靠（历史批次或当前批次无新增无人机任务）"
        elif node.node_type == "station":
            return "经停充电站（广播给无人机）"
        else:
            return "未知动作"

    def _setup_drone_routes(self, plan: DispatchPlan, current_time: float) -> None:
        """
        为分配的无人机设置路由计划。

        增量安全：跳过正在飞行中的无人机，避免覆盖其当前路径和位置。
        """
        if self.solver_name == "ga_mmce":
            from solver.ga_mmce.runtime_adapter import apply_ga_mmce_runtime_plan

            apply_ga_mmce_runtime_plan(
                entity_mgr=self.entity_mgr,
                order_mgr=self.order_mgr,
                plan=plan,
                current_time=current_time,
            )
            return

        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if drone is None:
                logger.warning("[DispatchDecisionEngine] 无人机 %s 不存在", alloc.drone_id)
                continue

            # 状态冻结：正在飞行的无人机使用路径拼接（串联任务）
            is_relay = getattr(alloc, "_is_relay", False)
            if drone.status.is_flying and not is_relay:
                logger.warning(
                    "[DispatchDecisionEngine] 无人机 %s 正在飞行(status=%s)，"
                    "跳过路由设置以保护轨迹连续性",
                    alloc.drone_id, drone.status.value,
                )
                continue

            order = self.order_mgr.assigned_orders.get(alloc.order_id)
            if order is None:
                logger.warning("[DispatchDecisionEngine] 订单 %s 不存在", alloc.order_id)
                continue

            try:
                if alloc.mode in ("B", "B_WAIT", "B_DYNAMIC"):
                    truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                    if truck is None:
                        logger.warning("[DispatchDecisionEngine] 卡车 %s 不存在", alloc.vehicle_id)
                        continue

                    if alloc.mode in ("B_WAIT", "B_DYNAMIC"):
                        launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
                        if launch_station is None:
                            logger.warning(
                                "[DispatchDecisionEngine] 充电站 %s 不存在（订单 %s）",
                                alloc.launch_station_id, alloc.order_id
                            )
                            continue
                        launch_loc = launch_station.location
                        wait_duration = alloc.wait_duration
                    else:
                        launch_loc = truck.get_location(current_time)
                        wait_duration = 0.0

                    delivery_loc = order.delivery_loc
                    recovery_id = alloc.recovery_station_id

                    if alloc.mode == "B_DYNAMIC":
                        recovery_entity = self.entity_mgr.depots.get(recovery_id)
                    else:
                        recovery_entity = (
                            self.entity_mgr.stations.get(recovery_id)
                            or self.entity_mgr.depots.get(recovery_id)
                        )
                    if recovery_entity is None:
                        logger.warning("[DispatchDecisionEngine] 回收点 %s 不存在", recovery_id)
                        continue

                    recovery_loc = recovery_entity.location
                    cruise_altitude = float(max(launch_loc.z, delivery_loc.z, recovery_loc.z, drone.current_loc.z))
                    polyline, delivery_idx = self._build_uav_polyline(
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                        altitude=cruise_altitude,
                    )
                    self._log_uav_path_if_obstacle_avoided(
                        alloc=alloc,
                        polyline=polyline,
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                        tag="setup",
                    )
                    waypoints = self._polyline_to_waypoints(
                        polyline=polyline,
                        delivery_idx=delivery_idx,
                        order_id=alloc.order_id,
                        recovery_id=recovery_id,
                    )
                    drone.set_route(waypoints)
                    try:
                        drone.assign_order(alloc.order_id, order.payload_weight)
                    except ValueError as e:
                        logger.warning("[DispatchDecisionEngine] 为无人机 %s 分配订单失败: %s", alloc.drone_id, e)
                    
                    from core.entities.primitives import TaskStatus
                    if order.status == TaskStatus.ASSIGNED:
                        order.update_status(TaskStatus.PICKED_UP)
                        order.update_status(TaskStatus.DELIVERING)
                    
                    if alloc.mode in ("B_WAIT", "B_DYNAMIC"):
                        drone.current_loc = truck.get_location(current_time)
                        drone.status = DroneStatus.IDLE
                        drone.transport_truck_id = truck.truck_id
                        drone.scheduled_launch_time = alloc.launch_time
                        drone.launch_station_id = alloc.launch_station_id
                        drone.waiting_recovery_station_id = ""
                        if alloc.drone_id not in truck.docked_drones:
                            truck.docked_drones.append(alloc.drone_id)
                        logger.info(
                            "[DispatchDecisionEngine] 为无人机 %s 设置路由（模式 %s）："
                            "随卡车 %s 运输至充电站 %s（t=%.1fs）后起飞，"
                            "等待+任务总时长 %.1fs → 回收点 %s，订单: %s",
                            alloc.drone_id, alloc.mode, alloc.vehicle_id,
                            alloc.launch_station_id,
                            alloc.launch_time, wait_duration,
                            recovery_id, alloc.order_id,
                        )
                    else:
                        drone.transport_truck_id = None
                        drone.scheduled_launch_time = 0.0
                        drone.launch_station_id = ""
                        drone.waiting_recovery_station_id = ""
                        drone.status = DroneStatus.FLYING_TO_PICKUP
                        logger.info(
                            "[DispatchDecisionEngine] 为无人机 %s 设置路由（模式 B）："
                            "卡车位置 → 配送点 → 充电站 %s，订单: %s",
                            alloc.drone_id, recovery_id, alloc.order_id,
                        )

                elif alloc.mode == "C":
                    depot = self.entity_mgr.depots.get(alloc.vehicle_id)
                    if depot is None:
                        logger.warning("[DispatchDecisionEngine] 仓库 %s 不存在", alloc.vehicle_id)
                        continue

                    depot_loc = depot.location
                    delivery_loc = order.delivery_loc

                    if is_relay and drone.status.is_flying:
                        # 串联任务：拼接新路径到现有路径末尾
                        relay_origin = getattr(alloc, "_relay_origin", None) or delivery_loc
                        cruise_altitude = float(max(relay_origin.z, delivery_loc.z, depot_loc.z, drone.current_loc.z))
                        relay_polyline, relay_delivery_idx = self._build_uav_polyline(
                            launch_loc=relay_origin,
                            delivery_loc=delivery_loc,
                            recovery_loc=depot_loc,
                            altitude=cruise_altitude,
                        )
                        self._log_uav_path_if_obstacle_avoided(
                            alloc=alloc,
                            polyline=relay_polyline,
                            launch_loc=relay_origin,
                            delivery_loc=delivery_loc,
                            recovery_loc=depot_loc,
                            tag="setup-relay",
                        )
                        relay_waypoints = self._polyline_to_waypoints(
                            polyline=relay_polyline,
                            delivery_idx=relay_delivery_idx,
                            order_id=alloc.order_id,
                            recovery_id=alloc.vehicle_id,
                        )
                        drone.append_route(relay_waypoints)
                        logger.info(
                            "[DispatchDecisionEngine] 无人机 %s 串联任务：当前任务完成后 → "
                            "取货 → 配送点 → 仓库，新订单: %s",
                            alloc.drone_id, alloc.order_id,
                        )
                    else:
                        cruise_altitude = float(max(depot_loc.z, delivery_loc.z, drone.current_loc.z))
                        polyline, delivery_idx = self._build_uav_polyline(
                            launch_loc=depot_loc,
                            delivery_loc=delivery_loc,
                            recovery_loc=depot_loc,
                            altitude=cruise_altitude,
                        )
                        self._log_uav_path_if_obstacle_avoided(
                            alloc=alloc,
                            polyline=polyline,
                            launch_loc=depot_loc,
                            delivery_loc=delivery_loc,
                            recovery_loc=depot_loc,
                            tag="setup",
                        )
                        waypoints = self._polyline_to_waypoints(
                            polyline=polyline,
                            delivery_idx=delivery_idx,
                            order_id=alloc.order_id,
                            recovery_id=alloc.vehicle_id,
                        )
                        drone.set_route(waypoints)
                        drone.transport_truck_id = None
                        drone.scheduled_launch_time = 0.0
                        drone.launch_station_id = ""
                        drone.waiting_recovery_station_id = ""
                        drone.status = DroneStatus.FLYING_TO_PICKUP

                    try:
                        drone.assign_order(alloc.order_id, order.payload_weight)
                    except ValueError as e:
                        logger.warning("[DispatchDecisionEngine] 为无人机 %s 分配订单失败: %s", alloc.drone_id, e)

                    from core.entities.primitives import TaskStatus
                    if order.status == TaskStatus.ASSIGNED:
                        order.update_status(TaskStatus.PICKED_UP)
                        order.update_status(TaskStatus.DELIVERING)

                    if not is_relay:
                        logger.info(
                            "[DispatchDecisionEngine] 为无人机 %s 设置路由（模式 C）：仓库 → 配送点 → 仓库，订单: %s",
                            alloc.drone_id, alloc.order_id,
                        )

            except Exception as e:
                logger.exception(
                    "[DispatchDecisionEngine] 为无人机 %s 设置路由失败: %s",
                    alloc.drone_id, str(e),
                )

    def _build_drone_routes(self, plan: DispatchPlan, current_time: float) -> None:
        """
        为前端构建无人机路由信息（用于可视化展示）。

        将分配中的无人机路由转换为DroneRoute对象，包含完整的飞行路径坐标序列。
        """
        if self.solver_name == "ga_mmce":
            from solver.ga_mmce.runtime_adapter import build_ga_mmce_drone_routes

            build_ga_mmce_drone_routes(
                entity_mgr=self.entity_mgr,
                order_mgr=self.order_mgr,
                plan=plan,
            )
            return

        from solver.greedy_mmce import DroneRoute

        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            order = self.order_mgr.assigned_orders.get(alloc.order_id)
            if order is None:
                continue

            drone = self.entity_mgr.drones.get(alloc.drone_id)
            drone_altitude = float(drone.current_loc.z) if drone is not None else 0.0

            try:
                if alloc.mode == "B":
                    # 模式 B：卡-空协同
                    truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                    if truck is None:
                        continue
                    
                    launch_loc = truck.get_location(current_time)
                    delivery_loc = order.delivery_loc
                    recovery_id = alloc.recovery_station_id
                    recovery_entity = (
                        self.entity_mgr.stations.get(recovery_id)
                        or self.entity_mgr.depots.get(recovery_id)
                    )
                    if recovery_entity is None:
                        continue
                    recovery_loc = recovery_entity.location

                    cruise_altitude = float(max(launch_loc.z, delivery_loc.z, recovery_loc.z, drone_altitude))
                    polyline, _ = self._build_uav_polyline(
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                        altitude=cruise_altitude,
                    )
                    
                    drone_route = DroneRoute(
                        drone_id=alloc.drone_id,
                        order_id=alloc.order_id,
                        path=polyline,
                        mode="B",
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                    )
                    plan.drone_routes[alloc.drone_id] = drone_route

                elif alloc.mode in ("B_WAIT", "B_DYNAMIC"):
                    launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
                    if launch_station is None:
                        continue
                    
                    launch_loc = launch_station.location
                    delivery_loc = order.delivery_loc
                    recovery_id = alloc.recovery_station_id

                    if alloc.mode == "B_DYNAMIC":
                        recovery_entity = self.entity_mgr.depots.get(recovery_id)
                    else:
                        recovery_entity = (
                            self.entity_mgr.stations.get(recovery_id)
                            or self.entity_mgr.depots.get(recovery_id)
                        )
                    if recovery_entity is None:
                        continue
                    recovery_loc = recovery_entity.location

                    cruise_altitude = float(max(launch_loc.z, delivery_loc.z, recovery_loc.z, drone_altitude))
                    polyline, _ = self._build_uav_polyline(
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                        altitude=cruise_altitude,
                    )
                    
                    drone_route = DroneRoute(
                        drone_id=alloc.drone_id,
                        order_id=alloc.order_id,
                        path=polyline,
                        mode=alloc.mode,
                        launch_loc=launch_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=recovery_loc,
                    )
                    plan.drone_routes[alloc.drone_id] = drone_route

                elif alloc.mode == "C":
                    # 模式 C：仓-空直递
                    depot = self.entity_mgr.depots.get(alloc.vehicle_id)
                    if depot is None:
                        continue
                    
                    depot_loc = depot.location
                    delivery_loc = order.delivery_loc

                    cruise_altitude = float(max(depot_loc.z, delivery_loc.z, drone_altitude))
                    polyline, _ = self._build_uav_polyline(
                        launch_loc=depot_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=depot_loc,
                        altitude=cruise_altitude,
                    )
                    
                    drone_route = DroneRoute(
                        drone_id=alloc.drone_id,
                        order_id=alloc.order_id,
                        path=polyline,
                        mode="C",
                        launch_loc=depot_loc,
                        delivery_loc=delivery_loc,
                        recovery_loc=depot_loc,
                    )
                    plan.drone_routes[alloc.drone_id] = drone_route

            except Exception as e:
                logger.exception(
                    "[DispatchDecisionEngine._build_drone_routes] 构建无人机 %s 路由失败: %s",
                    alloc.drone_id, str(e),
                )
