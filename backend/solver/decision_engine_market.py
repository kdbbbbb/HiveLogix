#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Market 专用调度决策编排层。

该类只承载 MarketBasedSolver 的执行语义，避免把 B_WAIT/B_DYNAMIC 的
站点起飞、卡车回收和契约兑现逻辑继续叠加到通用贪心编排器中。
"""

from __future__ import annotations

import logging
from typing import Any

from core.entities.primitives import DroneStatus, RouteWaypoint, TaskStatus, WaypointAction
from solver.decision_engine import DispatchDecisionEngine
from solver.greedy_mmce import AllocationResult, DispatchPlan
from solver.interfaces import DispatchSolver
from solver.market_based_solver import MarketBasedSolver

logger = logging.getLogger("solver.decision_engine_market")


class MarketDispatchDecisionEngine(DispatchDecisionEngine):
    """MarketBasedSolver 专用编排器。"""
    MODE_C_DEPOT_TOLERANCE_M = 25.0

    def __init__(
        self,
        entity_mgr: Any,
        order_mgr: Any,
        solver: DispatchSolver | None = None,
        solver_name: str = "market",
    ) -> None:
        super().__init__(
            entity_mgr=entity_mgr,
            order_mgr=order_mgr,
            solver=solver,
            solver_name=solver_name,
        )
        if not isinstance(self.solver, MarketBasedSolver):
            raise ValueError(
                "MarketDispatchDecisionEngine 只能绑定 MarketBasedSolver，"
                f"当前 solver={self.solver.__class__.__name__}"
            )
        self.solver_name = "market"
        self.solver.bind_order_manager(self.order_mgr)

    def set_solver(self, solver_name: str) -> None:
        """Market 专用编排器不允许切换到非 market 求解器。"""
        target = solver_name.strip().lower()
        if target != "market":
            raise ValueError("MarketDispatchDecisionEngine 仅支持 solver='market'")
        if not isinstance(self.solver, MarketBasedSolver):
            self.solver = MarketBasedSolver(self.entity_mgr)
            self.solver.bind_order_manager(self.order_mgr)
        self.solver_name = "market"

    def _apply_plan(
        self,
        plan: DispatchPlan,
        current_time: float,
        incremental: bool = False,
    ) -> None:
        """应用前先过滤无法真实车载起飞的 market 分配。"""
        self._sanitize_market_allocations(plan, current_time)
        super()._apply_plan(plan, current_time, incremental=incremental)

    def _sanitize_market_allocations(self, plan: DispatchPlan, current_time: float) -> None:
        """执行前校验分配的物理一致性，避免幽灵挂载与仓库幻觉起飞。"""
        for alloc in plan.allocations:
            if not alloc.feasible or not alloc.drone_id:
                continue

            if alloc.mode == "C":
                if getattr(alloc, "_is_relay", False):
                    continue
                if self._drone_can_start_mode_c_from_depot(alloc.drone_id, alloc.vehicle_id):
                    continue
                alloc.feasible = False
                alloc.reason = "market执行层拒绝：Mode C 无人机未满足仓库起飞条件"
                logger.warning(
                    "[MarketDispatchDecisionEngine] 拒绝订单 %s 的 Mode C 分配："
                    "drone=%s 未满足 depot=%s 的真实起飞条件",
                    alloc.order_id,
                    alloc.drone_id,
                    alloc.vehicle_id,
                )
                order = self.order_mgr.assigned_orders.get(alloc.order_id)
                if order is not None and order.status == TaskStatus.ASSIGNED:
                    self.order_mgr.assigned_orders.pop(alloc.order_id, None)
                    self.order_mgr.pending_orders[alloc.order_id] = order
                continue

            if alloc.mode not in ("B_WAIT", "B_DYNAMIC"):
                continue

            truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if truck is None or drone is None:
                alloc.feasible = False
                alloc.reason = "market执行层校验失败：卡车或无人机不存在"
                self._release_contract_for_rejected_allocation(alloc)
                continue

            if self._drone_is_physically_on_truck(alloc.drone_id, alloc.vehicle_id):
                continue

            # 第一阶段采取保守策略：market 只允许已在目标卡车上的无人机执行
            # 站点起飞任务。非车载接驳需要求解器侧把接驳时间纳入投标准入后再开放。
            alloc.feasible = False
            alloc.reason = (
                "market执行层拒绝：无人机未真实停靠在目标卡车上，"
                "不能在分配瞬间加入 truck.docked_drones"
            )
            logger.warning(
                "[MarketDispatchDecisionEngine] 拒绝订单 %s 的 %s 分配："
                "drone=%s 未在 truck=%s 负载上",
                alloc.order_id,
                alloc.mode,
                alloc.drone_id,
                alloc.vehicle_id,
            )
            self._release_contract_for_rejected_allocation(alloc)

            order = self.order_mgr.assigned_orders.get(alloc.order_id)
            if order is not None and order.status == TaskStatus.ASSIGNED:
                self.order_mgr.assigned_orders.pop(alloc.order_id, None)
                self.order_mgr.pending_orders[alloc.order_id] = order

        plan.summary["feasible"] = sum(1 for alloc in plan.allocations if alloc.feasible)
        modes: dict[str, int] = {}
        for alloc in plan.allocations:
            if alloc.feasible:
                modes[alloc.mode] = modes.get(alloc.mode, 0) + 1
        plan.summary["modes"] = modes

    def _drone_is_physically_on_truck(self, drone_id: str, truck_id: str) -> bool:
        truck = self.entity_mgr.trucks.get(truck_id)
        drone = self.entity_mgr.drones.get(drone_id)
        if truck is None or drone is None:
            return False
        if drone.status.is_flying:
            return False
        if getattr(drone, "waiting_recovery_station_id", ""):
            return False
        if drone_id in getattr(truck, "docked_drones", []):
            return True
        return getattr(drone, "transport_truck_id", "") == truck_id

    def _drone_can_start_mode_c_from_depot(self, drone_id: str, depot_id: str) -> bool:
        drone = self.entity_mgr.drones.get(drone_id)
        depot = self.entity_mgr.depots.get(depot_id)
        if drone is None or depot is None:
            return False
        if getattr(drone, "transport_truck_id", None):
            return False
        if getattr(drone, "waiting_recovery_station_id", ""):
            return False
        for truck in self.entity_mgr.trucks.values():
            if drone_id in getattr(truck, "docked_drones", []):
                return False
        if drone.current_loc.distance_2d(depot.location) > self.MODE_C_DEPOT_TOLERANCE_M:
            return False
        return True

    def _release_contract_for_rejected_allocation(self, alloc: AllocationResult) -> None:
        """释放已授标但执行层拒绝的 B_WAIT 契约，避免无人机被幽灵契约锁定。"""
        if alloc.mode != "B_WAIT":
            return
        for contract in self.solver.get_active_contracts():
            if (
                contract.status == "active"
                and contract.order_id == alloc.order_id
                and contract.drone_id == alloc.drone_id
                and contract.truck_id == alloc.vehicle_id
            ):
                contract.status = "released"
                logger.info(
                    "[MarketDispatchDecisionEngine] 释放未执行契约 %s：订单 %s 分配已被执行层拒绝",
                    contract.contract_id,
                    alloc.order_id,
                )

    def try_fulfill_contracts(self, current_time: float) -> None:
        """只有无人机真实进入目标卡车负载后才兑现契约。"""
        for contract in self.solver.get_active_contracts():
            if contract.status != "active":
                continue

            truck = self.entity_mgr.trucks.get(contract.truck_id)
            drone = self.entity_mgr.drones.get(contract.drone_id)
            if truck is None or drone is None:
                continue

            if current_time + 1e-6 < contract.t_sync:
                continue

            if getattr(drone, "waiting_recovery_station_id", ""):
                logger.debug(
                    "[MarketDispatchDecisionEngine] 契约 %s 尚未兑现："
                    "drone=%s 仍在站点 %s 等待卡车",
                    contract.contract_id,
                    contract.drone_id,
                    getattr(drone, "waiting_recovery_station_id", ""),
                )
                continue

            if (
                contract.drone_id in getattr(truck, "docked_drones", [])
                and getattr(drone, "transport_truck_id", "") == contract.truck_id
            ):
                self.solver.fulfill_contract(contract.contract_id)

    def _setup_drone_routes(self, plan: DispatchPlan, current_time: float) -> None:
        """为 market 分配设置无人机航路，B_WAIT/B_DYNAMIC 严格要求真实车载。"""
        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            if alloc.mode not in ("B_WAIT", "B_DYNAMIC"):
                self._setup_non_market_drone_route(alloc, current_time)
                continue

            self._setup_station_launch_route(alloc, current_time)

    def _setup_station_launch_route(self, alloc: AllocationResult, current_time: float) -> None:
        truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
        drone = self.entity_mgr.drones.get(alloc.drone_id)
        order = self.order_mgr.assigned_orders.get(alloc.order_id)
        if truck is None or drone is None or order is None:
            logger.warning(
                "[MarketDispatchDecisionEngine] 无法设置无人机 %s 路由：实体或订单不存在",
                alloc.drone_id,
            )
            return

        if not self._drone_is_physically_on_truck(alloc.drone_id, alloc.vehicle_id):
            logger.warning(
                "[MarketDispatchDecisionEngine] 跳过无人机 %s 的 %s 路由：未真实停靠在卡车 %s",
                alloc.drone_id,
                alloc.mode,
                alloc.vehicle_id,
            )
            return

        launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
        if launch_station is None:
            logger.warning(
                "[MarketDispatchDecisionEngine] 起飞充电站 %s 不存在（订单 %s）",
                alloc.launch_station_id,
                alloc.order_id,
            )
            return

        recovery_id = alloc.recovery_station_id
        if alloc.mode == "B_DYNAMIC":
            recovery_entity = self.entity_mgr.depots.get(recovery_id)
            dock_action = WaypointAction.DOCK_DEPOT
        else:
            recovery_entity = self.entity_mgr.stations.get(recovery_id)
            dock_action = WaypointAction.DOCK_TRUCK

        if recovery_entity is None:
            logger.warning("[MarketDispatchDecisionEngine] 回收点 %s 不存在", recovery_id)
            return

        current_carry = getattr(drone, "carrying_order_id", None)
        current_pending = getattr(drone, "pending_release_order_id", None)
        if current_carry and current_carry != alloc.order_id:
            logger.warning(
                "[MarketDispatchDecisionEngine] 跳过无人机 %s 的 %s 路由："
                "已携带订单 %s，拒绝改派到 %s",
                alloc.drone_id,
                alloc.mode,
                current_carry,
                alloc.order_id,
            )
            return
        if current_pending and current_pending != alloc.order_id:
            logger.warning(
                "[MarketDispatchDecisionEngine] 跳过无人机 %s 的 %s 路由："
                "仍在完成订单 %s，拒绝改派到 %s",
                alloc.drone_id,
                alloc.mode,
                current_pending,
                alloc.order_id,
            )
            return

        waypoints = [
            RouteWaypoint(launch_station.location, WaypointAction.PICKUP, alloc.order_id),
            RouteWaypoint(order.delivery_loc, WaypointAction.DELIVER, alloc.order_id),
            RouteWaypoint(recovery_entity.location, dock_action, recovery_id),
        ]
        if not current_carry:
            try:
                drone.assign_order(alloc.order_id, order.payload_weight)
            except ValueError as exc:
                logger.warning(
                    "[MarketDispatchDecisionEngine] 为无人机 %s 分配订单失败: %s",
                    alloc.drone_id,
                    exc,
                )
                return
        drone.set_route(waypoints)

        if order.status == TaskStatus.ASSIGNED:
            order.update_status(TaskStatus.PICKED_UP)
            order.update_status(TaskStatus.DELIVERING)

        drone.current_loc = truck.current_loc
        drone.status = DroneStatus.IDLE
        drone.transport_truck_id = truck.truck_id
        drone.scheduled_launch_time = alloc.launch_time
        drone.launch_station_id = alloc.launch_station_id
        drone.waiting_recovery_station_id = ""
        if alloc.drone_id not in truck.docked_drones:
            # 仅修复 transport_truck_id 已表明车载、但 docked_drones 缺失的历史不一致。
            truck.docked_drones.append(alloc.drone_id)

        logger.info(
            "[MarketDispatchDecisionEngine] 为无人机 %s 设置路由（模式 %s）："
            "已在卡车 %s 上，至充电站 %s 后起飞，回收点 %s，订单 %s",
            alloc.drone_id,
            alloc.mode,
            alloc.vehicle_id,
            alloc.launch_station_id,
            recovery_id,
            alloc.order_id,
        )

    def _setup_non_market_drone_route(self, alloc: AllocationResult, current_time: float) -> None:
        """复用通用编排器处理 B / C / relay 等非站点车载起飞模式。"""
        temp_plan = DispatchPlan(
            allocations=[alloc],
            cost_total=0.0,
            summary={"total_orders": 1, "feasible": 1, "modes": {alloc.mode: 1}},
        )
        super()._setup_drone_routes(temp_plan, current_time)
