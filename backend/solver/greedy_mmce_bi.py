#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — MMCE Backbone Insertion

在 GreedyMMCE 基础上实现“骨干路径 + 增量插入”策略：
1) 静态批次：先构建卡车骨干路径（重货单），再增量插入其余订单。
2) 动态批次：基于卡车当前未执行后缀路线做订单插入，不做全局重排。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.entities.primitives import Position3D, SourceType, WaypointAction
from solver.greedy_mmce import AllocationResult, DispatchPlan, GreedyMMCE, TruckRoute

if TYPE_CHECKING:
    from core.entities.drone import Drone
    from core.entities.order import Order
    from core.entities.truck import Truck


logger = logging.getLogger(__name__)


@dataclass
class _InsertionChoice:
    """模式 A 增量插入结果。"""

    truck_id: str
    insert_idx: int
    delta_score: float
    cost_dist: float
    cost_energy: float
    cost_penalty: float
    distance: float


@dataclass
class _ModeBCandidate:
    """模式 B 候选（含插站动作）。"""

    truck_id: str
    drone_id: str
    launch_station_id: str
    launch_insert_idx: int
    recovery_station_id: str
    recovery_insert_idx: int
    launch_time: float
    wait_duration: float
    delta_score: float
    cost_dist: float
    cost_energy: float
    cost_penalty: float
    distance: float
    detour_ratio: float


@dataclass
class _ModeBDiagnostics:
    """模式 B 枚举诊断信息，用于解释为何未入候选。"""

    has_drone: bool = True
    has_station: bool = True
    available_drone_candidates: int = 0
    capable_drone_candidates: int = 0
    total_station_trials: int = 0
    scenario_feasible_trials: int = 0
    rejected_detour: int = 0
    rejected_score: int = 0
    accepted_trials: int = 0
    best_raw_score: float = math.inf
    best_raw_detour_ratio: float = math.inf
    best_raw_cost_dist: float = 0.0
    best_raw_cost_energy: float = 0.0
    best_raw_cost_penalty: float = 0.0


@dataclass
class _StationLaunchOption:
    """B_WAIT 出发站候选（含在当前未来路线中的最佳插入信息）。"""

    station_id: str
    station_pos: Position3D
    launch_insert_idx: int
    launch_extra: float
    launch_path_distance: float
    launch_eta: float
    task_distance: float
    source: str


class GreedyMMCEBackboneInsertion(GreedyMMCE):
    """MMCE-BI：骨干构建 + 多模式增量插入。"""

    # 模式 B 合法性阈值：路径长度增幅上限（10%~20% 通常可接受）
    B_DETOUR_RATIO_LIMIT = 0.15
    # 骨干插站仅允许“近似顺路”而非明显绕路。
    BACKBONE_STATION_DETOUR_LIMIT = 0.02
    BACKBONE_STATION_MAX_EXTRA_M = 30.0
    B_WAIT_ROUTE_TOP_K = 3
    B_WAIT_TASK_TOP_K = 3
    B_WAIT_ETA_TOP_K = 3
    B_WAIT_MAX_CANDIDATES = 8
    WAIT_PENALTY_TIME_SCALE_S = 60.0
    FIXED_TRUCK_DRONE_INDEXES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 11})
    FIXED_DEPOT_DRONE_INDEXES = frozenset({9, 10, 12})

    def __init__(self, entity_mgr) -> None:
        super().__init__(entity_mgr)
        self._fixed_loadout_applied = False
        self._fixed_truck_drone_pool_by_truck: dict[str, set[str]] = {}
        self._fixed_depot_drone_pool: set[str] = set()

    @staticmethod
    def _extract_numeric_suffix(entity_id: str) -> int | None:
        match = re.search(r"(\d+)$", str(entity_id))
        if not match:
            return None
        return int(match.group(1))

    def _resolve_fixed_drone_pools(self) -> tuple[str, str, set[str], set[str]] | None:
        """解析固定装载规则对应的实体 ID 池。"""
        if len(self.entity_mgr.trucks) != 1 or not self.entity_mgr.depots:
            return None

        truck_id = next(iter(self.entity_mgr.trucks.keys()))
        depot_id = next(iter(self.entity_mgr.depots.keys()))

        if self._is_ga_mmce_dynamic_delegate():
            return self._resolve_ga_dynamic_drone_pools(truck_id, depot_id)

        truck_pool: set[str] = set()
        depot_pool: set[str] = set()
        for drone_id in self.entity_mgr.drones:
            idx = self._extract_numeric_suffix(drone_id)
            if idx is None:
                continue
            if idx in self.FIXED_TRUCK_DRONE_INDEXES:
                truck_pool.add(drone_id)
            elif idx in self.FIXED_DEPOT_DRONE_INDEXES:
                depot_pool.add(drone_id)

        if len(truck_pool) < len(self.FIXED_TRUCK_DRONE_INDEXES):
            return None
        if len(depot_pool) < len(self.FIXED_DEPOT_DRONE_INDEXES):
            return None
        return truck_id, depot_id, truck_pool, depot_pool

    def _resolve_ga_dynamic_drone_pools(
        self,
        truck_id: str,
        depot_id: str,
    ) -> tuple[str, str, set[str], set[str]] | None:
        """GA 动态委托时使用 GAConfig 的初始装载名单，不覆盖普通贪心基线。"""
        config = getattr(self, "_ga_mmce_config", None)
        if config is None:
            try:
                from solver.ga_mmce.config import GAConfig
            except Exception:
                try:
                    from .ga_mmce.config import GAConfig
                except Exception:
                    GAConfig = None
            config = GAConfig() if GAConfig is not None else None

        truck_pool = {
            str(drone_id).strip()
            for drone_id in (getattr(config, "initial_truck_drone_ids", ()) or ())
            if str(drone_id).strip() in self.entity_mgr.drones
        }
        depot_pool = {
            str(drone_id).strip()
            for drone_id in (getattr(config, "initial_depot_drone_ids", ()) or ())
            if str(drone_id).strip() in self.entity_mgr.drones
        }
        if not truck_pool and not depot_pool:
            return None
        return truck_id, depot_id, truck_pool, depot_pool

    def _apply_fixed_initial_drone_loadout(self, current_time: float) -> None:
        """按规则固化初始 9/3 归属，并避免重规划后车载无人机异常增长。"""
        if self._fixed_loadout_applied:
            return

        resolved = self._resolve_fixed_drone_pools()
        if resolved is None:
            self._fixed_loadout_applied = True
            logger.warning("[MMCE-BI] 固定 9/3 无人机装载规则未生效：实体 ID 不匹配或非单车场景")
            return

        truck_id, depot_id, truck_pool, depot_pool = resolved
        truck = self.entity_mgr.trucks.get(truck_id)
        depot = self.entity_mgr.depots.get(depot_id)
        if truck is None or depot is None:
            self._fixed_loadout_applied = True
            return

        self._fixed_truck_drone_pool_by_truck = {truck_id: set(truck_pool)}
        self._fixed_depot_drone_pool = set(depot_pool)

        # 起降平台并发槽位至少覆盖固定车载无人机数量，保证可回收可复用。
        if int(getattr(truck, "parking_slots", 0)) < len(truck_pool):
            truck.parking_slots = len(truck_pool)

        truck.docked_drones = sorted(truck_pool)
        for other_truck_id, other_truck in self.entity_mgr.trucks.items():
            if other_truck_id == truck_id:
                continue
            other_truck.docked_drones = [
                did for did in other_truck.docked_drones
                if did not in truck_pool
            ]

        for drone_id, drone in self.entity_mgr.drones.items():
            if drone_id in truck_pool:
                drone.home_type = SourceType.TRUCK
                drone.home_id = truck_id
                drone.transport_truck_id = truck_id
                if not drone.status.is_flying:
                    drone.current_loc = truck.get_location(current_time)
            elif drone_id in depot_pool:
                drone.home_type = SourceType.DEPOT
                drone.home_id = depot_id
                drone.transport_truck_id = None
                if not drone.status.is_flying:
                    drone.current_loc = depot.location

        self._fixed_loadout_applied = True
        logger.info(
            "[MMCE-BI] 固定无人机装载已生效: truck=%s (%d 架) depot=%s (%d 架)",
            truck_id,
            len(truck_pool),
            depot_id,
            len(depot_pool),
        )

    def _drone_in_truck_pool(self, drone_id: str, truck_id: str) -> bool:
        pool = self._fixed_truck_drone_pool_by_truck.get(truck_id)
        if pool is not None:
            return drone_id in pool

        owner = self._resolve_drone_owner_truck_id(self.entity_mgr.drones[drone_id])
        return owner == truck_id

    def _get_available_drones_for_truck(
        self,
        truck_id: str,
        allocated_drones: set[str],
    ) -> list["Drone"]:
        """返回满足“车载且空闲”的无人机。"""
        truck = self.entity_mgr.trucks.get(truck_id)
        if truck is None:
            return []

        docked = set(getattr(truck, "docked_drones", []))
        available: list["Drone"] = []
        for drone in self._get_available_drones():
            if drone.drone_id in allocated_drones:
                continue
            if self._drone_reserved_outside_ga_dynamic_scope(drone):
                continue
            if not self._drone_in_truck_pool(drone.drone_id, truck_id):
                continue

            transport_truck_id = getattr(drone, "transport_truck_id", "")
            is_carried = drone.drone_id in docked or transport_truck_id == truck_id
            if not is_carried:
                continue
            available.append(drone)
        return available

    def _try_mode_c(
        self,
        order: "Order",
        current_time: float,
        allocated_drones: set[str],
        truck_last_pos: dict[str, Position3D],
    ) -> AllocationResult:
        """MMCE-BI 保留仓库直发模式 C。"""
        if not self._is_ga_mmce_dynamic_delegate():
            return super()._try_mode_c(order, current_time, allocated_drones, truck_last_pos)

        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="C",
                distance=float("inf"),
                feasible=False,
                reason="无可用仓库",
            )

        depot = depots[0]
        available_drones = self._get_available_drones()
        available_drones = [
            d for d in available_drones
            if d.drone_id not in allocated_drones
            and not self._drone_reserved_outside_ga_dynamic_scope(d)
            and self._is_drone_ready_for_depot_launch(d, depot)
        ]
        drone = self._find_capable_drone(order.payload_weight, available_drones)
        if drone is None:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="C",
                distance=float("inf"),
                feasible=False,
                reason="无位于仓库且载重匹配的可用无人机",
            )

        energy_out = self._flight_energy(drone, depot.location, order.delivery_loc, order.payload_weight)
        energy_back = self._flight_energy(drone, order.delivery_loc, depot.location, 0.0)
        energy_needed = (energy_out + energy_back) * self.ENERGY_SAFETY_FACTOR

        if energy_needed > drone.battery_current:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="C",
                distance=float("inf"),
                feasible=False,
                reason=(
                    f"仓-空往返电量不足（需 {energy_needed:.0f} J，"
                    f"剩余 {drone.battery_current:.0f} J）"
                ),
            )

        return AllocationResult(
            order_id=order.order_id,
            vehicle_id=depot.depot_id,
            mode="C",
            distance=self._uav_path_distance(
                depot.location,
                order.delivery_loc,
                altitude=self.UAV_CRUISE_ALTITUDE_M,
            ),
            feasible=True,
            recovery_station_id=depot.depot_id,
            drone_id=drone.drone_id,
        )

    def _is_ga_mmce_dynamic_delegate(self) -> bool:
        """仅 GA-MMCE 动态委托贪心时启用兼容保护。"""
        return bool(getattr(self, "_ga_mmce_dynamic_delegate", False))

    def _active_ga_mmce_dynamic_order_ids(self) -> set[str]:
        return {
            str(order_id)
            for order_id in (getattr(self, "_ga_mmce_dynamic_order_ids", set()) or set())
            if str(order_id)
        }

    def _forced_ga_mmce_depot_direct_order_ids(self) -> set[str]:
        if not self._is_ga_mmce_dynamic_delegate():
            return set()
        return {
            str(order_id)
            for order_id in (
                getattr(self, "_ga_mmce_force_depot_direct_order_ids", set())
                or set()
            )
            if str(order_id)
        }

    def _drone_reserved_outside_ga_dynamic_scope(self, drone: "Drone") -> bool:
        """GA 动态委托时，排除仍被未重写未来任务预约的无人机。"""
        if not self._is_ga_mmce_dynamic_delegate():
            return False

        allowed_order_ids = self._active_ga_mmce_dynamic_order_ids()
        future_order_ids = self._collect_drone_future_order_ids(drone)
        blocked_order_ids = sorted(
            order_id for order_id in future_order_ids
            if order_id not in allowed_order_ids
        )
        if not blocked_order_ids:
            return False

        logger.debug(
            "[MMCE-BI] GA动态委托跳过已被未来任务预约的无人机 %s orders=%s",
            drone.drone_id,
            blocked_order_ids,
        )
        return True

    def _collect_drone_future_order_ids(self, drone: "Drone") -> set[str]:
        """从运行时航路和 GA segment 中读取无人机尚未完成的订单集合。"""
        result: set[str] = set()
        for attr_name in ("carrying_order_id", "pending_release_order_id"):
            order_id = str(getattr(drone, attr_name, "") or "")
            if order_id:
                result.add(order_id)

        route_plan = getattr(drone, "route_plan", None) or []
        start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
        start_idx = max(0, min(start_idx, len(route_plan)))
        for wp in route_plan[start_idx:]:
            if wp.action in (WaypointAction.PICKUP, WaypointAction.DELIVER):
                order_id = str(wp.target_entity_id or "")
                if order_id:
                    result.add(order_id)

        segments = getattr(drone, "_ga_runtime_segments", None) or []
        for segment in segments:
            order_id = str(getattr(segment, "order_id", "") or "")
            if not order_id:
                continue
            dock_idx = int(getattr(segment, "dock_idx", -1) or -1)
            if dock_idx >= start_idx:
                result.add(order_id)

        return result

    def _nearest_station_for_pos(self, pos: Position3D) -> tuple[str, Position3D] | None:
        """返回距离给定位置最近的充电站 (station_id, location)。"""
        stations = list(self.entity_mgr.stations.values())
        if not stations:
            return None
        best = min(stations, key=lambda s: self._dist(pos, s.location))
        return best.station_id, best.location

    def _terminal_anchor_pos(self, pos: Position3D) -> Position3D | None:
        """终点锚点：优先最近充电站，缺失时回退仓库。"""
        nearest_station = self._nearest_station_for_pos(pos)
        if nearest_station is not None:
            return nearest_station[1]
        depots = list(self.entity_mgr.depots.values())
        if depots:
            return depots[0].location
        return None

    def _estimate_distance_to_nearest_depot(self, pos: Position3D) -> float:
        """重载基类口径：MMCE-BI 末端锚点改为最近充电站（无站时回退仓库）。"""
        anchor = self._terminal_anchor_pos(pos)
        if anchor is None:
            return 0.0
        return self._road_dist(pos, anchor)

    def should_replan_unfinished(self) -> bool:
        """MMCE-BI 默认走纯增量（新单插入），不触发全局重排。"""
        return False

    def dispatch(
        self,
        pending_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """静态调度：重货骨干 + 其余订单增量插入。"""
        return self._dispatch_backbone_insertion(
            orders=pending_orders,
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            incremental=False,
            start_from_current_state=False,
            dispatch_type_override="full",
        )

    def dispatch_incremental(
        self,
        new_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """动态调度：在既有后缀路径上插入新订单。"""
        return self._dispatch_backbone_insertion(
            orders=new_orders,
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            incremental=True,
            start_from_current_state=True,
            dispatch_type_override="incremental",
        )

    def dispatch_replan_current_state(
        self,
        replan_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """兼容协议：保持插入式策略，不做全局重排。"""
        return self._dispatch_backbone_insertion(
            orders=replan_orders,
            current_time=current_time,
            bbox=bbox,
            scene_id=scene_id,
            incremental=True,
            start_from_current_state=True,
            dispatch_type_override="dynamic_replan",
        )

    def _dispatch_backbone_insertion(
        self,
        orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None,
        incremental: bool,
        start_from_current_state: bool,
        dispatch_type_override: str,
    ) -> DispatchPlan:
        if not orders:
            return DispatchPlan(
                allocations=[],
                cost_total=0.0,
                summary={
                    "total_orders": 0,
                    "feasible": 0,
                    "modes": {},
                    "dispatch_type": dispatch_type_override,
                    "cost_breakdown": {"dist": 0.0, "energy": 0.0, "penalty": 0.0},
                },
            )

        self._road_distance_memo.clear()
        self._uav_path_distance_memo.clear()
        self._activate_path_planner(scene_id)
        self._load_road_graph(bbox, scene_id)
        self._apply_fixed_initial_drone_loadout(current_time)

        sorted_orders = sorted(
            orders.values(),
            key=lambda o: (o.deadline - current_time, o.deadline),
        )

        stops_by_truck = self._init_truck_stops(current_time, incremental)
        allocations: list[AllocationResult] = []
        mode_counter: dict[str, int] = {}
        changed_trucks: set[str] = set()
        allocated_drones: set[str] = set()
        recovery_wait_times: dict[tuple[str, str], float] = {}
        force_depot_direct_ids = self._forced_ga_mmce_depot_direct_order_ids()

        if incremental:
            heavy_orders = [o for o in sorted_orders if self._is_truck_only_order(o)]
            flex_orders = [o for o in sorted_orders if not self._is_truck_only_order(o)]
        else:
            heavy_orders = [o for o in sorted_orders if self._is_truck_only_order(o)]
            flex_orders = [o for o in sorted_orders if not self._is_truck_only_order(o)]

        # 阶段 1：重货订单先形成骨干路径。
        for order in heavy_orders:
            choice = self._best_truck_only_insertion(
                order,
                current_time,
                stops_by_truck,
                start_from_current_state,
            )
            if choice is None:
                allocations.append(
                    AllocationResult(
                        order_id=order.order_id,
                        vehicle_id="",
                        mode="REJECT",
                        distance=float("inf"),
                        feasible=False,
                        reason="无可用卡车执行重货订单",
                    )
                )
                continue

            prev_id, next_id, pos_tag = self._describe_insert_position(
                stops_by_truck[choice.truck_id],
                choice.insert_idx,
            )
            self._insert_customer_stop(stops_by_truck[choice.truck_id], order, choice.insert_idx)
            changed_trucks.add(choice.truck_id)
            mode_counter["A"] = mode_counter.get("A", 0) + 1
            logger.info(
                "[MMCE-BI] 重货订单 %s 骨干插入: mode=A score=%.2f "
                "(dist=%.2f, energy=%.2f, penalty=%.2f)",
                order.order_id,
                choice.delta_score,
                choice.cost_dist,
                choice.cost_energy,
                choice.cost_penalty,
            )
            logger.info(
                "[MMCE-BI] 订单 %s 卡车插入位: truck=%s idx=%d (%s) prev=%s next=%s",
                order.order_id,
                choice.truck_id,
                choice.insert_idx,
                pos_tag,
                prev_id,
                next_id,
            )
            allocations.append(
                AllocationResult(
                    order_id=order.order_id,
                    vehicle_id=choice.truck_id,
                    mode="A",
                    distance=choice.distance,
                    feasible=True,
                    score_total=choice.delta_score,
                    cost_dist=choice.cost_dist,
                    cost_energy=choice.cost_energy,
                    cost_penalty=choice.cost_penalty,
                )
            )

        # 阶段 1.5：沿骨干路径补可达充电结构。
        if not incremental:
            for truck_id, stops in stops_by_truck.items():
                if not stops:
                    continue
                added = self._augment_backbone_stations(
                    truck_id,
                    stops,
                    current_time,
                    start_from_current_state=False,
                    return_to_depot=True,
                )
                if added:
                    changed_trucks.add(truck_id)

        # 阶段 2：其余订单按 A/B/C 候选增量插入。
        for order in flex_orders:
            if order.order_id in force_depot_direct_ids:
                c_alloc = self._try_mode_c(order, current_time, allocated_drones, {})
                if c_alloc.feasible:
                    c_score, c_dist, c_energy, c_penalty = self._score_allocation(
                        c_alloc,
                        order,
                        current_time,
                        {},
                    )
                    c_alloc.score_total = c_score
                    c_alloc.cost_dist = c_dist
                    c_alloc.cost_energy = c_energy
                    c_alloc.cost_penalty = c_penalty
                    allocations.append(c_alloc)
                    allocated_drones.add(c_alloc.drone_id)
                    mode_counter["C"] = mode_counter.get("C", 0) + 1
                    logger.info(
                        "[MMCE-BI] GA临期救援订单 %s 强制仓空直递: drone=%s score=%.2f "
                        "(dist=%.2f, energy=%.2f, penalty=%.2f)",
                        order.order_id,
                        c_alloc.drone_id,
                        c_score,
                        c_dist,
                        c_energy,
                        c_penalty,
                    )
                    continue
                logger.info(
                    "[MMCE-BI] GA临期救援订单 %s 仓空直递不可行: %s，回退常规 A/B/C 候选",
                    order.order_id,
                    c_alloc.reason,
                )

            a_choice = self._best_truck_only_insertion(
                order,
                current_time,
                stops_by_truck,
                start_from_current_state,
            )
            c_alloc = self._try_mode_c(order, current_time, allocated_drones, {})
            c_score = float("inf")
            if c_alloc.feasible:
                c_score, c_dist, c_energy, c_penalty = self._score_allocation(c_alloc, order, current_time, {})
                c_alloc.score_total = c_score
                c_alloc.cost_dist = c_dist
                c_alloc.cost_energy = c_energy
                c_alloc.cost_penalty = c_penalty

            b_choice, b_diag = self._best_mode_b_insertion(
                order=order,
                current_time=current_time,
                stops_by_truck=stops_by_truck,
                start_from_current_state=start_from_current_state,
                allocated_drones=allocated_drones,
                a_baseline=(a_choice.delta_score if a_choice is not None else float("inf")),
            )

            if b_choice is None:
                if not b_diag.has_drone:
                    if b_diag.available_drone_candidates > 0 and b_diag.capable_drone_candidates == 0:
                        logger.info(
                            "[MMCE-BI] 订单 %s B_WAIT 未入候选: 车载空闲无人机 %d 架，"
                            "但载重均不足（订单载重=%.2fkg）",
                            order.order_id,
                            b_diag.available_drone_candidates,
                            float(order.payload_weight),
                        )
                    else:
                        logger.info(
                            "[MMCE-BI] 订单 %s B_WAIT 未入候选: 无可用车载无人机（空闲=%d）",
                            order.order_id,
                            b_diag.available_drone_candidates,
                        )
                elif not b_diag.has_station:
                    logger.info(
                        "[MMCE-BI] 订单 %s B_WAIT 未入候选: 无可用充电站",
                        order.order_id,
                    )
                else:
                    raw_score_text = (
                        "inf"
                        if not math.isfinite(b_diag.best_raw_score)
                        else f"{b_diag.best_raw_score:.2f}"
                    )
                    raw_detour_text = (
                        "inf"
                        if not math.isfinite(b_diag.best_raw_detour_ratio)
                        else f"{b_diag.best_raw_detour_ratio:.3f}"
                    )
                    raw_dist_text = (
                        "inf"
                        if not math.isfinite(b_diag.best_raw_score)
                        else f"{b_diag.best_raw_cost_dist:.2f}"
                    )
                    raw_energy_text = (
                        "inf"
                        if not math.isfinite(b_diag.best_raw_score)
                        else f"{b_diag.best_raw_cost_energy:.2f}"
                    )
                    raw_penalty_text = (
                        "inf"
                        if not math.isfinite(b_diag.best_raw_score)
                        else f"{b_diag.best_raw_cost_penalty:.2f}"
                    )
                    a_text = (
                        "inf"
                        if a_choice is None or not math.isfinite(a_choice.delta_score)
                        else f"{a_choice.delta_score:.2f}"
                    )
                    logger.info(
                        "[MMCE-BI] 订单 %s B_WAIT 未入候选: 枚举=%d, 可行=%d, "
                        "detour拒绝=%d, 分数拒绝=%d, "
                        "最佳raw(score=%s, dist=%s, energy=%s, penalty=%s, detour=%s), "
                        "A基线=%s",
                        order.order_id,
                        b_diag.total_station_trials,
                        b_diag.scenario_feasible_trials,
                        b_diag.rejected_detour,
                        b_diag.rejected_score,
                        raw_score_text,
                        raw_dist_text,
                        raw_energy_text,
                        raw_penalty_text,
                        raw_detour_text,
                        a_text,
                    )

            candidate_items: list[dict[str, float | str]] = []
            candidate_metrics: dict[str, tuple[float, float, float, float]] = {}
            if a_choice is not None:
                candidate_items.append(
                    {
                        "mode": "A",
                        "score": round(a_choice.delta_score, 2),
                        "dist": round(a_choice.cost_dist, 2),
                        "energy": round(a_choice.cost_energy, 2),
                        "penalty": round(a_choice.cost_penalty, 2),
                    }
                )
                candidate_metrics["A"] = (
                    a_choice.delta_score,
                    a_choice.cost_dist,
                    a_choice.cost_energy,
                    a_choice.cost_penalty,
                )
            if b_choice is not None:
                candidate_items.append(
                    {
                        "mode": "B_WAIT",
                        "score": round(b_choice.delta_score, 2),
                        "dist": round(b_choice.cost_dist, 2),
                        "energy": round(b_choice.cost_energy, 2),
                        "penalty": round(b_choice.cost_penalty, 2),
                    }
                )
                candidate_metrics["B"] = (
                    b_choice.delta_score,
                    b_choice.cost_dist,
                    b_choice.cost_energy,
                    b_choice.cost_penalty,
                )
            if c_alloc.feasible:
                candidate_items.append(
                    {
                        "mode": "C",
                        "score": round(c_alloc.score_total, 2),
                        "dist": round(c_alloc.cost_dist, 2),
                        "energy": round(c_alloc.cost_energy, 2),
                        "penalty": round(c_alloc.cost_penalty, 2),
                    }
                )
                candidate_metrics["C"] = (
                    c_alloc.score_total,
                    c_alloc.cost_dist,
                    c_alloc.cost_energy,
                    c_alloc.cost_penalty,
                )

            candidate_scores = {
                "A": a_choice.delta_score if a_choice is not None else float("inf"),
                "B": b_choice.delta_score if b_choice is not None else float("inf"),
                "C": c_score,
            }
            best_mode = min(candidate_scores, key=candidate_scores.get)
            best_score = candidate_scores[best_mode]

            if not math.isfinite(best_score):
                allocations.append(
                    AllocationResult(
                        order_id=order.order_id,
                        vehicle_id="",
                        mode="REJECT",
                        distance=float("inf"),
                        feasible=False,
                        reason="A/B/C 均不可行",
                    )
                )
                logger.info(
                    "[MMCE-BI] 订单 %s 候选评分: %s | 选中=REJECT",
                    order.order_id,
                    candidate_items,
                )
                continue

            selected_score, selected_dist, selected_energy, selected_penalty = candidate_metrics[best_mode]
            selected_mode_name = "B_WAIT" if best_mode == "B" else best_mode
            logger.info(
                "[MMCE-BI] 订单 %s 候选评分: %s | 选中=%s score=%.2f "
                "(dist=%.2f, energy=%.2f, penalty=%.2f)",
                order.order_id,
                candidate_items,
                selected_mode_name,
                selected_score,
                selected_dist,
                selected_energy,
                selected_penalty,
            )

            if best_mode == "A" and a_choice is not None:
                prev_id, next_id, pos_tag = self._describe_insert_position(
                    stops_by_truck[a_choice.truck_id],
                    a_choice.insert_idx,
                )
                self._insert_customer_stop(stops_by_truck[a_choice.truck_id], order, a_choice.insert_idx)
                changed_trucks.add(a_choice.truck_id)
                mode_counter["A"] = mode_counter.get("A", 0) + 1
                logger.info(
                    "[MMCE-BI] 订单 %s 卡车插入位: truck=%s idx=%d (%s) prev=%s next=%s",
                    order.order_id,
                    a_choice.truck_id,
                    a_choice.insert_idx,
                    pos_tag,
                    prev_id,
                    next_id,
                )
                allocations.append(
                    AllocationResult(
                        order_id=order.order_id,
                        vehicle_id=a_choice.truck_id,
                        mode="A",
                        distance=a_choice.distance,
                        feasible=True,
                        score_total=a_choice.delta_score,
                        cost_dist=a_choice.cost_dist,
                        cost_energy=a_choice.cost_energy,
                        cost_penalty=a_choice.cost_penalty,
                    )
                )
                continue

            if best_mode == "B" and b_choice is not None:
                truck_stops = stops_by_truck[b_choice.truck_id]
                launch_pos = self.entity_mgr.stations[b_choice.launch_station_id].location
                recovery_entity = (
                    self.entity_mgr.stations.get(b_choice.recovery_station_id)
                    or self.entity_mgr.depots.get(b_choice.recovery_station_id)
                )
                recovery_pos = recovery_entity.location if recovery_entity is not None else launch_pos

                launch_prev, launch_next, launch_tag = self._describe_insert_position(
                    truck_stops,
                    b_choice.launch_insert_idx,
                )
                launch_idx = self._ensure_recovery_stop(
                    stops=truck_stops,
                    station_id=b_choice.launch_station_id,
                    station_pos=launch_pos,
                    suggested_idx=b_choice.launch_insert_idx,
                    min_service=self.TRUCK_DRONE_LAUNCH_TIME,
                    desired_node_type=(
                        "station" if self._is_ga_mmce_dynamic_delegate() else "recovery"
                    ),
                    order_id=order.order_id,
                    semantic_role="launch",
                )
                logger.info(
                    "[MMCE-BI] 订单 %s B_WAIT 放飞站插入: truck=%s station=%s idx=%d (%s) prev=%s next=%s",
                    order.order_id,
                    b_choice.truck_id,
                    b_choice.launch_station_id,
                    launch_idx,
                    launch_tag,
                    launch_prev,
                    launch_next,
                )
                recovery_idx = launch_idx
                if b_choice.recovery_station_id != b_choice.launch_station_id:
                    recovery_prev, recovery_next, recovery_tag = self._describe_insert_position(
                        truck_stops,
                        max(b_choice.recovery_insert_idx, launch_idx + 1),
                    )
                    recovery_idx = self._ensure_recovery_stop(
                        stops=truck_stops,
                        station_id=b_choice.recovery_station_id,
                        station_pos=recovery_pos,
                        suggested_idx=max(b_choice.recovery_insert_idx, launch_idx + 1),
                        min_service=b_choice.wait_duration,
                        desired_node_type="recovery",
                        order_id=order.order_id,
                        semantic_role="recovery",
                    )
                    logger.info(
                        "[MMCE-BI] 订单 %s B_WAIT 回收站插入: truck=%s station=%s idx=%d (%s) prev=%s next=%s",
                        order.order_id,
                        b_choice.truck_id,
                        b_choice.recovery_station_id,
                        recovery_idx,
                        recovery_tag,
                        recovery_prev,
                        recovery_next,
                    )
                else:
                    truck_stops[launch_idx]["service_time"] = max(
                        float(truck_stops[launch_idx].get("service_time", 0.0)),
                        b_choice.wait_duration,
                    )

                changed_trucks.add(b_choice.truck_id)
                allocated_drones.add(b_choice.drone_id)
                recovery_wait_times[(b_choice.truck_id, b_choice.recovery_station_id)] = max(
                    recovery_wait_times.get((b_choice.truck_id, b_choice.recovery_station_id), 0.0),
                    b_choice.wait_duration,
                )
                mode_counter["B_WAIT"] = mode_counter.get("B_WAIT", 0) + 1
                allocations.append(
                    AllocationResult(
                        order_id=order.order_id,
                        vehicle_id=b_choice.truck_id,
                        mode="B_WAIT",
                        distance=b_choice.distance,
                        feasible=True,
                        recovery_station_id=b_choice.recovery_station_id,
                        drone_id=b_choice.drone_id,
                        launch_station_id=b_choice.launch_station_id,
                        launch_time=b_choice.launch_time,
                        wait_duration=b_choice.wait_duration,
                        score_total=b_choice.delta_score,
                        cost_dist=b_choice.cost_dist,
                        cost_energy=b_choice.cost_energy,
                        cost_penalty=b_choice.cost_penalty,
                    )
                )
                continue

            # best_mode == "C"
            mode_counter["C"] = mode_counter.get("C", 0) + 1
            allocations.append(c_alloc)
            allocated_drones.add(c_alloc.drone_id)

        # 清理“非任务必需”的 station 节点，避免无意义停靠（如末尾先去站点再回仓）。
        self._prune_nonessential_station_stops(
            stops_by_truck,
            allocations,
            current_time=current_time,
            start_from_current_state=start_from_current_state,
        )

        truck_routes: dict[str, TruckRoute] = {}
        route_trucks = changed_trucks if incremental else {tid for tid, s in stops_by_truck.items() if s}
        if incremental and self._is_ga_mmce_dynamic_delegate():
            route_trucks = set(route_trucks) | {tid for tid, s in stops_by_truck.items() if s}

        for truck_id in route_trucks:
            truck = self.entity_mgr.trucks.get(truck_id)
            if truck is None:
                continue

            stops = stops_by_truck.get(truck_id, [])
            if not stops:
                continue

            # 终点锚定到“最后任务点最近充电站”，不再回仓。
            if self._ensure_terminal_station_stop(stops):
                changed_trucks.add(truck_id)

            route_stops = [
                {
                    "node_id": stop["node_id"],
                    "node_type": stop["node_type"],
                    "position": stop["position"],
                    "arrival_time": current_time,
                    "departure_time": current_time + max(0.0, float(stop.get("service_time", 0.0))),
                    "order_id": stop.get("order_id", ""),
                }
                for stop in stops
            ]

            rebuilt = self.build_incremental_route_from_stops(
                truck=truck,
                ordered_stops=route_stops,
                current_time=current_time,
            )
            if rebuilt is None:
                continue

            # recovery 节点服务时长按本轮分配刷新，避免等待被短路。
            for node in rebuilt.nodes:
                if node.node_type != "recovery":
                    continue
                wait_s = recovery_wait_times.get((truck_id, node.node_id))
                if wait_s is None:
                    continue
                node.departure_time = max(node.departure_time, node.arrival_time + wait_s)

            truck_routes[truck_id] = rebuilt

        cost_dist_total, cost_energy_total, cost_total = self._recalculate_plan_route_costs(
            allocations,
            truck_routes,
            current_time,
            orders,
        )
        cost_penalty_total = sum(a.cost_penalty for a in allocations if a.feasible)

        summary = {
            "total_orders": len(allocations),
            "feasible": sum(1 for a in allocations if a.feasible),
            "modes": mode_counter,
            "dispatch_type": dispatch_type_override,
            "cost_breakdown": {
                "dist": cost_dist_total,
                "energy": cost_energy_total,
                "penalty": cost_penalty_total,
            },
        }
        logger.info("[MMCE-BI] 分配完成：%s", summary)
        return DispatchPlan(
            allocations=allocations,
            cost_total=cost_total,
            summary=summary,
            truck_routes=truck_routes,
        )

    def _init_truck_stops(self, current_time: float, incremental: bool) -> dict[str, list[dict]]:
        """初始化每辆卡车的可插入停靠序列。"""
        result: dict[str, list[dict]] = {}
        for truck_id, truck in self.entity_mgr.trucks.items():
            seq: list[dict] = []
            if incremental:
                planned = getattr(truck, "_planned_route_stops", None) or []
                cursor = int(getattr(truck, "_planned_route_cursor", 0) or 0)
                planned_solver = str(getattr(truck, "_planned_route_solver", "") or "")
                protect_ga_static_stops = (
                    self._is_ga_mmce_dynamic_delegate()
                    and planned_solver == "ga_mmce"
                )
                force_depot_direct_ids = self._forced_ga_mmce_depot_direct_order_ids()
                for stop in planned[cursor:]:
                    node_type = stop.get("node_type", "")
                    if node_type not in ("customer", "recovery", "station"):
                        continue
                    order_id = str(stop.get("order_id", "") or "")
                    if force_depot_direct_ids and order_id in force_depot_direct_ids:
                        continue
                    position = stop.get("position")
                    if position is None:
                        continue
                    arr = float(stop.get("arrival_time", current_time))
                    dep = float(stop.get("departure_time", arr))

                    # 增量重调度时，旧 future 停靠可能已经部分等待；
                    # 若继续使用“原始整段服务时长”，会导致同一站点反复从头等待并卡住。
                    if current_time > arr:
                        remaining_service = max(0.0, dep - current_time)
                        arr = current_time
                        dep = current_time + remaining_service
                        stop["arrival_time"] = arr
                        stop["departure_time"] = dep

                    copied_stop = {
                        "node_id": stop.get("node_id", ""),
                        "node_type": node_type,
                        "position": position,
                        "order_id": order_id,
                        "service_time": max(0.0, dep - arr),
                    }
                    if protect_ga_static_stops and node_type in ("station", "recovery"):
                        copied_stop["_ga_static_stop"] = True
                        copied_stop["_semantic_role"] = (
                            "launch" if node_type == "station" else "recovery"
                        )
                    seq.append(copied_stop)
            result[truck_id] = seq
        return result

    def _is_truck_only_order(self, order: "Order") -> bool:
        """重货判定：超出系统内任意无人机载重上限。"""
        if not self.entity_mgr.drones:
            return True
        max_payload = max((float(d.payload_capacity) for d in self.entity_mgr.drones.values()), default=0.0)
        return float(order.payload_weight) > max_payload

    def _get_route_start_pos(self, truck: "Truck", current_time: float, start_from_current_state: bool) -> Position3D:
        """返回插入评估的路线起点。"""
        if start_from_current_state:
            return truck.get_location(current_time)
        depots = list(self.entity_mgr.depots.values())
        if depots:
            return depots[0].location
        return truck.get_location(current_time)

    def _sequence_distance(
        self,
        truck: "Truck",
        stops: list[dict],
        current_time: float,
        start_from_current_state: bool,
        return_to_depot: bool,
    ) -> float:
        """计算当前停靠序列总里程（用于 detour 比率）。"""
        total = 0.0
        cur = self._get_route_start_pos(truck, current_time, start_from_current_state)
        for stop in stops:
            total += self._road_dist(cur, stop["position"])
            cur = stop["position"]
        if return_to_depot:
            depots = list(self.entity_mgr.depots.values())
            if depots:
                total += self._road_dist(cur, depots[0].location)
        return total

    def _best_truck_only_insertion(
        self,
        order: "Order",
        current_time: float,
        stops_by_truck: dict[str, list[dict]],
        start_from_current_state: bool,
    ) -> Optional[_InsertionChoice]:
        """枚举所有 (truck, i, j) 插入点，返回模式 A 最小增量。"""
        best: Optional[_InsertionChoice] = None

        for truck_id, truck in self.entity_mgr.trucks.items():
            seq = stops_by_truck.get(truck_id, [])
            start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
            return_to_depot = True
            
            # NOTE: 我们需要计算按序到达各个站点的预期时间（含原有的停靠与等待时间），以便精准评估插入导致的迟到！
            arrival_times_at_idx = [current_time]
            cur_t = current_time
            cur_pos = start_pos
            speed = max(1.0, float(getattr(truck, "speed", 0.0)))
            for stop in seq:
                arr = cur_t + self._road_dist(cur_pos, stop.get("position", cur_pos)) / speed
                # 若是已有站点可能规定了最小出发时间或服务时间
                dep = max(arr, stop.get("departure_time", arr))
                arr_with_service = max(arr + stop.get("service_time", 0.0), dep)
                arrival_times_at_idx.append(arr)
                cur_t = arr_with_service
                cur_pos = stop.get("position", cur_pos)

            for idx in range(len(seq) + 1):
                prev_pos = start_pos if idx == 0 else seq[idx - 1].get("position", start_pos)
                
                if idx < len(seq):
                    next_pos = seq[idx].get("position")
                    replaced = self._road_dist(prev_pos, next_pos)
                else:
                    next_pos = None
                    base_anchor = self._terminal_anchor_pos(prev_pos)
                    replaced = 0.0 if base_anchor is None else self._road_dist(prev_pos, base_anchor)

                added = self._road_dist(prev_pos, order.delivery_loc)
                if idx < len(seq) and next_pos is not None:
                    added += self._road_dist(order.delivery_loc, next_pos)
                else:
                    new_anchor = self._terminal_anchor_pos(order.delivery_loc)
                    if new_anchor is not None:
                        added += self._road_dist(order.delivery_loc, new_anchor)
                
                delta_dist = max(0.0, added - replaced)

                # 精确配送时间 = 该索引处原有到达时间 + 插入点路程耗时
                dist_to_insert = self._road_dist(prev_pos, order.delivery_loc)
                delivery_time_est = arrival_times_at_idx[idx] + dist_to_insert / speed + self.SERVICE_TIME_CUSTOMER

                lateness = max(0.0, delivery_time_est - order.deadline)
                cost_dist = self.C_DIST_ET * delta_dist
                cost_energy = self.C_ENERGY_ET * self._truck_energy_wh(delta_dist)
                cost_penalty = self.LAMBDA_TIME * order.penalty_rate * lateness
                delta_score = cost_dist + cost_energy + cost_penalty


                cand = _InsertionChoice(
                    truck_id=truck_id,
                    insert_idx=idx,
                    delta_score=delta_score,
                    cost_dist=cost_dist,
                    cost_energy=cost_energy,
                    cost_penalty=cost_penalty,
                    distance=delta_dist,
                )
                if best is None or cand.delta_score < best.delta_score:
                    best = cand

        return best

    def _insert_customer_stop(self, seq: list[dict], order: "Order", idx: int) -> None:
        """在指定位置插入 customer 节点。"""
        stop = {
            "node_id": order.order_id,
            "node_type": "customer",
            "position": order.delivery_loc,
            "order_id": order.order_id,
            "service_time": self.SERVICE_TIME_CUSTOMER,
        }
        seq.insert(max(0, min(idx, len(seq))), stop)

    @staticmethod
    def _describe_insert_position(seq: list[dict], idx: int) -> tuple[str, str, str]:
        """描述插入位置：返回 (prev_id, next_id, 头/中/尾标签)。"""
        pos = max(0, min(idx, len(seq)))
        prev_id = "ORIGIN" if pos == 0 else str(seq[pos - 1].get("node_id", ""))
        next_id = "END" if pos >= len(seq) else str(seq[pos].get("node_id", ""))
        if pos == 0:
            tag = "head"
        elif pos == len(seq):
            tag = "tail"
        else:
            tag = "middle"
        return prev_id, next_id, tag

    def _best_station_insertion(
        self,
        truck: "Truck",
        seq: list[dict],
        station_pos: Position3D,
        current_time: float,
        start_from_current_state: bool,
        return_to_depot: bool,
    ) -> tuple[int, float]:
        """返回某站点最佳插入位置与附加里程。"""
        start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
        best_idx = 0
        best_extra = float("inf")

        for idx in range(len(seq) + 1):
            prev_pos = start_pos if idx == 0 else seq[idx - 1]["position"]
            if idx < len(seq):
                next_pos = seq[idx]["position"]
                replaced = self._road_dist(prev_pos, next_pos)
                extra = self._road_dist(prev_pos, station_pos) + self._road_dist(station_pos, next_pos) - replaced
            else:
                base_anchor = self._terminal_anchor_pos(prev_pos)
                new_anchor = self._terminal_anchor_pos(station_pos)
                if base_anchor is None or new_anchor is None:
                    extra = self._road_dist(prev_pos, station_pos)
                else:
                    replaced = self._road_dist(prev_pos, base_anchor)
                    extra = (
                        self._road_dist(prev_pos, station_pos)
                        + self._road_dist(station_pos, new_anchor)
                        - replaced
                    )

            if extra < best_extra:
                best_extra = extra
                best_idx = idx

        return best_idx, max(0.0, best_extra)

    def _best_mode_b_insertion(
        self,
        order: "Order",
        current_time: float,
        stops_by_truck: dict[str, list[dict]],
        start_from_current_state: bool,
        allocated_drones: set[str],
        a_baseline: float,
    ) -> tuple[Optional[_ModeBCandidate], _ModeBDiagnostics]:
        """枚举 i/(i,j)+s 方案，应用 B 约束后返回最优候选。"""
        diag = _ModeBDiagnostics(has_drone=False)

        stations = list(self.entity_mgr.stations.values())
        if not stations:
            diag.has_station = False
            return None, diag

        recovery_pool = self._get_recovery_pool()
        best: Optional[_ModeBCandidate] = None

        for truck_id, truck in self.entity_mgr.trucks.items():
            truck_drones = self._get_available_drones_for_truck(truck_id, allocated_drones)
            diag.available_drone_candidates += len(truck_drones)

            capable_drones = [
                d for d in truck_drones
                if float(d.payload_capacity) >= float(order.payload_weight)
            ]
            diag.capable_drone_candidates += len(capable_drones)

            drone = self._find_capable_drone(order.payload_weight, capable_drones)
            if drone is None:
                continue
            diag.has_drone = True

            seq = stops_by_truck.get(truck_id, [])
            start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
            return_to_depot = True
            base_len = max(
                1.0,
                self._sequence_distance(
                    truck,
                    seq,
                    current_time,
                    start_from_current_state,
                    return_to_depot,
                ),
            )

            # 候选站点：任务点最近2个 + 未来路线插入最友好2个（去重后最多4个）。
            launch_options = self._select_b_wait_station_candidates(
                order=order,
                truck=truck,
                seq=seq,
                current_time=current_time,
                start_from_current_state=start_from_current_state,
                return_to_depot=return_to_depot,
                stations=stations,
            )

            # 使用候选站点评估 detour；launch_delay 用“从当前时刻到该站估计到达时刻”。
            for option in launch_options:
                diag.total_station_trials += 1
                launch_idx = option.launch_insert_idx
                launch_extra = option.launch_extra
                launch_delay = option.launch_eta

                scenario = self._evaluate_charging_station_departure(
                    drone=drone,
                    truck_tail_loc=start_pos,
                    launch_loc=option.station_pos,
                    launch_station_id=option.station_id,
                    truck_distance_to_launch=option.launch_path_distance,
                    launch_delay=launch_delay,
                    delivery_loc=order.delivery_loc,
                    payload=order.payload_weight,
                    recovery_pool=recovery_pool,
                    current_time=current_time,
                    order=order,
                )
                if not scenario or not scenario.get("feasible", False):
                    continue
                diag.scenario_feasible_trials += 1

                recovery_id = str(scenario["recovery_station_id"])
                recovery_insert_idx = launch_idx
                recovery_extra = 0.0
                if recovery_id in self.entity_mgr.stations and recovery_id != option.station_id:
                    recovery_insert_idx, recovery_extra = self._best_station_insertion(
                        truck,
                        seq,
                        self.entity_mgr.stations[recovery_id].location,
                        current_time,
                        start_from_current_state,
                        return_to_depot,
                    )

                extra_total = launch_extra + recovery_extra
                detour_ratio = extra_total / base_len

                recovery_entity = (
                    self.entity_mgr.stations.get(recovery_id)
                    or self.entity_mgr.depots.get(recovery_id)
                )
                recovery_pos = recovery_entity.location if recovery_entity is not None else option.station_pos
                scenario_truck_increment = self._estimate_truck_increment_for_drone_support(
                    start_pos,
                    order.delivery_loc,
                    option.station_pos,
                    recovery_pos,
                )
                scenario_truck_increment = max(0.0, scenario_truck_increment)
                scenario_uav_cost_dist = max(
                    0.0,
                    float(scenario.get("cost_dist", 0.0))
                    - self.C_DIST_ET * scenario_truck_increment,
                )
                scenario_uav_cost_energy = max(
                    0.0,
                    float(scenario.get("cost_energy", 0.0))
                    - self.C_ENERGY_ET * self._truck_energy_wh(scenario_truck_increment),
                )

                waiting_penalty = self.WAIT_PENALTY_FACTOR * max(
                    0.0,
                    float(scenario.get("wait_duration", 0.0)),
                ) / max(1.0, self.WAIT_PENALTY_TIME_SCALE_S)
                delta_dist = scenario_uav_cost_dist + self.C_DIST_ET * extra_total
                delta_energy = scenario_uav_cost_energy + self.C_ENERGY_ET * self._truck_energy_wh(extra_total)
                delta_penalty = float(scenario.get("cost_penalty", 0.0)) + waiting_penalty
                delta_score = delta_dist + delta_energy + delta_penalty

                if delta_score < diag.best_raw_score:
                    diag.best_raw_score = delta_score
                    diag.best_raw_detour_ratio = detour_ratio
                    diag.best_raw_cost_dist = delta_dist
                    diag.best_raw_cost_energy = delta_energy
                    diag.best_raw_cost_penalty = delta_penalty

                # 关键约束 1：只有当 B 优于 A 才允许进入候选。
                if detour_ratio > self.B_DETOUR_RATIO_LIMIT:
                    diag.rejected_detour += 1
                    continue
                if not math.isfinite(a_baseline) or delta_score >= a_baseline:
                    diag.rejected_score += 1
                    continue

                cand = _ModeBCandidate(
                    truck_id=truck_id,
                    drone_id=drone.drone_id,
                    launch_station_id=option.station_id,
                    launch_insert_idx=launch_idx,
                    recovery_station_id=recovery_id,
                    recovery_insert_idx=recovery_insert_idx,
                    launch_time=float(scenario.get("launch_time", current_time + launch_delay)),
                    wait_duration=float(scenario.get("wait_duration", 0.0)),
                    delta_score=delta_score,
                    cost_dist=delta_dist,
                    cost_energy=delta_energy,
                    cost_penalty=delta_penalty,
                    distance=float(scenario.get("distance", 0.0)),
                    detour_ratio=detour_ratio,
                )
                if best is None or cand.delta_score < best.delta_score:
                    best = cand
                diag.accepted_trials += 1

        return best, diag

    def _select_b_wait_station_candidates(
        self,
        order: "Order",
        truck: "Truck",
        seq: list[dict],
        current_time: float,
        start_from_current_state: bool,
        return_to_depot: bool,
        stations: list,
    ) -> list[_StationLaunchOption]:
        """选择 B_WAIT 出发站候选：路线增量优先 + 最早可放飞 + 任务近邻并集。"""
        if not stations:
            return []

        speed = max(1e-6, float(getattr(truck, "speed", 0.0)))
        
        start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
        
        # 计算按序到达各个站点的预期时间（含原有的停靠与等待时间）
        arrival_times_at_idx = [current_time]
        cur_t = current_time
        cur_pos = start_pos
        for stop in seq:
            arr = cur_t + self._road_dist(cur_pos, stop.get("position", cur_pos)) / speed
            dep = max(arr, stop.get("departure_time", arr))
            cur_t = max(arr + stop.get("service_time", 0.0), dep)
            arrival_times_at_idx.append(arr)
            cur_pos = stop.get("position", cur_pos)

        route_ranked: list[tuple[float, float, float, _StationLaunchOption]] = []
        eta_ranked: list[tuple[float, float, float, _StationLaunchOption]] = []
        task_ranked: list[tuple[float, float, float, _StationLaunchOption]] = []

        for station in stations:
            launch_idx, launch_extra = self._best_station_insertion(
                truck,
                seq,
                station.location,
                current_time,
                start_from_current_state,
                return_to_depot,
            )
            launch_path_distance = self._distance_to_inserted_station(
                truck,
                seq,
                launch_idx,
                station.location,
                current_time,
                start_from_current_state,
            )
            # 使用包含停靠与等待历史的精确时间
            base_arr = arrival_times_at_idx[launch_idx]
            prev_pos = self._get_route_start_pos(truck, current_time, start_from_current_state) if launch_idx == 0 else seq[launch_idx - 1].get("position", start_pos)
            dist_to_insert = self._road_dist(prev_pos, station.location)
            true_arrival_time = base_arr + dist_to_insert / speed
            launch_eta = max(0.0, true_arrival_time - current_time)
            
            task_dist = self._dist(order.delivery_loc, station.location)
            option = _StationLaunchOption(
                station_id=station.station_id,
                station_pos=station.location,
                launch_insert_idx=launch_idx,
                launch_extra=max(0.0, launch_extra),
                launch_path_distance=max(0.0, launch_path_distance),
                launch_eta=max(0.0, launch_eta),
                task_distance=task_dist,
                source="",
            )
            route_ranked.append((option.launch_extra, option.launch_eta, option.task_distance, option))
            eta_ranked.append((option.launch_eta, option.launch_extra, option.task_distance, option))
            task_ranked.append((option.task_distance, option.launch_eta, option.launch_extra, option))

        # 未来路线友好：按插入增量优先，ETA 与任务距离作为次级排序。
        route_ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        route_top: list[_StationLaunchOption] = []
        for _, _, _, opt in route_ranked[: self.B_WAIT_ROUTE_TOP_K]:
            route_top.append(
                _StationLaunchOption(
                    station_id=opt.station_id,
                    station_pos=opt.station_pos,
                    launch_insert_idx=opt.launch_insert_idx,
                    launch_extra=opt.launch_extra,
                    launch_path_distance=opt.launch_path_distance,
                    launch_eta=opt.launch_eta,
                    task_distance=opt.task_distance,
                    source=f"route_top{self.B_WAIT_ROUTE_TOP_K}",
                )
            )

        # 最早可放飞：优先 ETA，补偿“路线上很顺但很晚才经过”的候选。
        eta_ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        eta_top: list[_StationLaunchOption] = []
        for _, _, _, opt in eta_ranked[: self.B_WAIT_ETA_TOP_K]:
            eta_top.append(
                _StationLaunchOption(
                    station_id=opt.station_id,
                    station_pos=opt.station_pos,
                    launch_insert_idx=opt.launch_insert_idx,
                    launch_extra=opt.launch_extra,
                    launch_path_distance=opt.launch_path_distance,
                    launch_eta=opt.launch_eta,
                    task_distance=opt.task_distance,
                    source=f"eta_top{self.B_WAIT_ETA_TOP_K}",
                )
            )

        # 任务点近邻：按任务距离排序。
        task_ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        task_top: list[_StationLaunchOption] = []
        for _, _, _, opt in task_ranked[: self.B_WAIT_TASK_TOP_K]:
            task_top.append(
                _StationLaunchOption(
                    station_id=opt.station_id,
                    station_pos=opt.station_pos,
                    launch_insert_idx=opt.launch_insert_idx,
                    launch_extra=opt.launch_extra,
                    launch_path_distance=opt.launch_path_distance,
                    launch_eta=opt.launch_eta,
                    task_distance=opt.task_distance,
                    source=f"task_top{self.B_WAIT_TASK_TOP_K}",
                )
            )

        # 去重合并：优先路线增量，再补最早放飞与任务近邻。
        merged: list[_StationLaunchOption] = []
        seen: set[str] = set()
        for opt in route_top + eta_top + task_top:
            if opt.station_id in seen:
                continue
            seen.add(opt.station_id)
            merged.append(opt)
            if len(merged) >= self.B_WAIT_MAX_CANDIDATES:
                break
        return merged

    def _distance_to_inserted_station(
        self,
        truck: "Truck",
        seq: list[dict],
        insert_idx: int,
        station_pos: Position3D,
        current_time: float,
        start_from_current_state: bool,
    ) -> float:
        """估计从当前起点到“在 insert_idx 处插入站点”时的到站里程。"""
        prefix_len = self._prefix_distance(
            truck,
            seq,
            insert_idx,
            current_time,
            start_from_current_state,
        )
        start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
        prev_pos = start_pos if insert_idx == 0 else seq[insert_idx - 1]["position"]
        return max(0.0, prefix_len + self._road_dist(prev_pos, station_pos))

    def _ensure_recovery_stop(
        self,
        stops: list[dict],
        station_id: str,
        station_pos: Position3D,
        suggested_idx: int,
        min_service: float,
        desired_node_type: str = "recovery",
        order_id: str = "",
        semantic_role: str = "",
    ) -> int:
        """确保序列中存在 recovery 站点，存在则提升服务时长，不存在则插入。"""
        if not self._is_ga_mmce_dynamic_delegate():
            for idx, stop in enumerate(stops):
                if stop.get("node_id") != station_id:
                    continue
                stop["node_type"] = "recovery"
                stop["position"] = station_pos
                stop["service_time"] = max(float(stop.get("service_time", 0.0)), float(min_service))
                return idx

            idx = max(0, min(suggested_idx, len(stops)))
            stops.insert(
                idx,
                {
                    "node_id": station_id,
                    "node_type": "recovery",
                    "position": station_pos,
                    "order_id": "",
                    "service_time": max(0.0, float(min_service)),
                },
            )
            return idx

        desired_node_type = str(desired_node_type or "recovery")
        order_id = str(order_id or "")
        semantic_role = semantic_role or ("launch" if desired_node_type == "station" else "recovery")
        for idx, stop in enumerate(stops):
            if stop.get("node_id") != station_id:
                continue

            if self._is_ga_mmce_dynamic_delegate():
                existing_order_id = str(stop.get("order_id", "") or "")
                existing_role = str(stop.get("_semantic_role", "") or "")
                existing_type = str(stop.get("node_type", "") or "")
                if existing_order_id and order_id and existing_order_id != order_id:
                    continue
                if existing_role and existing_role != semantic_role:
                    continue
                if existing_order_id and existing_type != desired_node_type:
                    continue
                if bool(stop.get("_ga_static_stop", False)) and existing_type != desired_node_type:
                    continue

            stop["node_type"] = desired_node_type
            stop["position"] = station_pos
            if order_id and not str(stop.get("order_id", "") or ""):
                stop["order_id"] = order_id
            if semantic_role:
                stop["_semantic_role"] = semantic_role
            stop["service_time"] = max(float(stop.get("service_time", 0.0)), float(min_service))
            return idx

        idx = max(0, min(suggested_idx, len(stops)))
        stops.insert(
            idx,
            {
                "node_id": station_id,
                "node_type": desired_node_type,
                "position": station_pos,
                "order_id": order_id,
                "service_time": max(0.0, float(min_service)),
                "_semantic_role": semantic_role,
            },
        )
        return idx

    def _prefix_distance(
        self,
        truck: "Truck",
        seq: list[dict],
        stop_count: int,
        current_time: float,
        start_from_current_state: bool,
    ) -> float:
        """从起点到前 stop_count 个停靠的累计里程。"""
        cur = self._get_route_start_pos(truck, current_time, start_from_current_state)
        total = 0.0
        for stop in seq[: max(0, min(stop_count, len(seq)))]:
            total += self._road_dist(cur, stop["position"])
            cur = stop["position"]
        return total

    def _augment_backbone_stations(
        self,
        truck_id: str,
        stops: list[dict],
        current_time: float,
        start_from_current_state: bool,
        return_to_depot: bool,
    ) -> int:
        """阶段 1：沿骨干路径补充可达充电站。"""
        truck = self.entity_mgr.trucks.get(truck_id)
        if truck is None:
            return 0

        stations = list(self.entity_mgr.stations.values())
        if not stations:
            return 0

        drone_pool = [d for d in self.entity_mgr.drones.values() if d.battery_current > self.min_reserve_energy]
        if not drone_pool:
            return 0

        inserted = 0
        idx = 0
        # 仅在“中间路段”插站，不在最后回仓段前强行加站。
        while idx < len(stops):
            start_pos = self._get_route_start_pos(truck, current_time, start_from_current_state)
            prev_pos = start_pos if idx == 0 else stops[idx - 1]["position"]

            if idx < len(stops):
                next_pos = stops[idx]["position"]
                direct = self._road_dist(prev_pos, next_pos)
            else:
                break

            if direct <= 1e-6:
                idx += 1
                continue

            best_station = None
            best_extra = float("inf")
            for station in stations:
                if any(s.get("node_id") == station.station_id for s in stops):
                    continue
                detour = self._road_dist(prev_pos, station.location) + self._road_dist(station.location, next_pos)
                extra = detour - direct
                ratio = extra / max(1.0, direct)
                if ratio > self.BACKBONE_STATION_DETOUR_LIMIT or extra > self.BACKBONE_STATION_MAX_EXTRA_M:
                    continue
                if not self._is_backbone_station_energy_feasible(drone_pool, prev_pos, next_pos, station.location):
                    continue
                if extra < best_extra:
                    best_extra = extra
                    best_station = station

            if best_station is None:
                idx += 1
                continue

            stops.insert(
                idx,
                {
                    "node_id": best_station.station_id,
                    "node_type": "station",
                    "position": best_station.location,
                    "order_id": "",
                    "service_time": 0.0,
                },
            )
            inserted += 1
            idx += 2

        return inserted

    def _prune_nonessential_station_stops(
        self,
        stops_by_truck: dict[str, list[dict]],
        allocations: list[AllocationResult],
        current_time: float | None = None,
        start_from_current_state: bool = False,
    ) -> None:
        """移除当前批次无任务需求的 station/recovery 停靠，减少无效绕行/停留。"""
        if self._is_ga_mmce_dynamic_delegate():
            self._prune_nonessential_station_stops_for_ga_dynamic(
                stops_by_truck,
                allocations,
                current_time=current_time,
                start_from_current_state=start_from_current_state,
            )
            return

        required_by_truck: dict[str, set[str]] = {
            tid: set() for tid in stops_by_truck
        }

        for alloc in allocations:
            if not alloc.feasible:
                continue
            if alloc.mode not in ("B", "B_WAIT"):
                continue
            truck_id = alloc.vehicle_id
            if truck_id not in required_by_truck:
                continue
            if alloc.launch_station_id:
                required_by_truck[truck_id].add(alloc.launch_station_id)
            if alloc.recovery_station_id and alloc.recovery_station_id in self.entity_mgr.stations:
                required_by_truck[truck_id].add(alloc.recovery_station_id)

        # 并入运行时承诺：避免删掉“无人机正在等回收/在途将回收”的站点。
        runtime_required_by_truck: dict[str, set[str]] = {
            tid: set() for tid in stops_by_truck
        }
        for drone in self.entity_mgr.drones.values():
            owner_truck_id = self._resolve_drone_owner_truck_id(drone)
            if owner_truck_id not in runtime_required_by_truck:
                continue

            waiting_station_id = getattr(drone, "waiting_recovery_station_id", "")
            if waiting_station_id and waiting_station_id in self.entity_mgr.stations:
                runtime_required_by_truck[owner_truck_id].add(waiting_station_id)

            route_plan = getattr(drone, "route_plan", None) or []
            start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            start_idx = max(0, min(start_idx, len(route_plan)))
            for wp in route_plan[start_idx:]:
                if wp.action not in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                    continue
                target_id = wp.target_entity_id or ""
                if target_id in self.entity_mgr.stations:
                    runtime_required_by_truck[owner_truck_id].add(target_id)
                break

        for truck_id, station_ids in runtime_required_by_truck.items():
            required_by_truck.setdefault(truck_id, set()).update(station_ids)

        for truck_id, seq in stops_by_truck.items():
            required = required_by_truck.get(truck_id, set())
            if not seq:
                continue

            kept: list[dict] = []
            removed = 0
            for stop in seq:
                if stop.get("node_type") not in ("station", "recovery"):
                    kept.append(stop)
                    continue

                sid = str(stop.get("node_id", ""))
                if sid and sid in required:
                    # 保留被任务/运行时承诺引用的站点，统一为 recovery 便于时序一致处理。
                    stop["node_type"] = "recovery"
                    kept.append(stop)
                else:
                    removed += 1

            if removed > 0:
                logger.info(
                    "[MMCE-BI] 卡车 %s 移除 %d 个非任务必需站点停靠",
                    truck_id,
                    removed,
                )
            stops_by_truck[truck_id] = kept

    def _prune_nonessential_station_stops_for_ga_dynamic(
        self,
        stops_by_truck: dict[str, list[dict]],
        allocations: list[AllocationResult],
        current_time: float | None = None,
        start_from_current_state: bool = False,
    ) -> None:
        """GA 动态委托专用：按订单和语义角色保护静态 station/recovery。"""
        required_keys_by_truck: dict[str, set[tuple[str, str, str]]] = {
            tid: set() for tid in stops_by_truck
        }
        waiting_recovery_by_truck: dict[str, list[tuple[str, str, Position3D]]] = {
            tid: [] for tid in stops_by_truck
        }

        def key(node_id: str, node_type: str, order_id: str) -> tuple[str, str, str]:
            return (str(node_id or ""), str(node_type or ""), str(order_id or ""))

        def matches_required_key(stop: dict, required_keys: set[tuple[str, str, str]]) -> bool:
            node_id = str(stop.get("node_id", "") or "")
            node_type = str(stop.get("node_type", "") or "")
            order_id = str(stop.get("order_id", "") or "")
            if key(node_id, node_type, order_id) in required_keys:
                return True
            if not order_id:
                return any(
                    node_id == required_node_id and node_type == required_type
                    for required_node_id, required_type, _ in required_keys
                )
            return any(
                node_id == required_node_id
                and node_type == required_type
                and not required_order_id
                for required_node_id, required_type, required_order_id in required_keys
            )

        def matching_required_order_ids(stop: dict, required_keys: set[tuple[str, str, str]]) -> set[str]:
            node_id = str(stop.get("node_id", "") or "")
            node_type = str(stop.get("node_type", "") or "")
            return {
                required_order_id
                for required_node_id, required_type, required_order_id in required_keys
                if node_id == required_node_id and node_type == required_type and required_order_id
            }

        for alloc in allocations:
            if not alloc.feasible or alloc.mode not in ("B", "B_WAIT"):
                continue
            truck_id = str(alloc.vehicle_id or "")
            if truck_id not in required_keys_by_truck:
                continue
            order_id = str(alloc.order_id or "")
            if alloc.launch_station_id:
                required_keys_by_truck[truck_id].add(
                    key(alloc.launch_station_id, "station", order_id)
                )
            if alloc.recovery_station_id and alloc.recovery_station_id in self.entity_mgr.stations:
                required_keys_by_truck[truck_id].add(
                    key(alloc.recovery_station_id, "recovery", order_id)
                )

        for drone in self.entity_mgr.drones.values():
            owner_truck_id = self._resolve_drone_owner_truck_id(drone)
            if owner_truck_id not in required_keys_by_truck:
                continue

            order_id = self._resolve_drone_current_future_order_id(drone)
            waiting_station_id = str(getattr(drone, "waiting_recovery_station_id", "") or "")
            if waiting_station_id and waiting_station_id in self.entity_mgr.stations:
                required_keys_by_truck[owner_truck_id].add(
                    key(waiting_station_id, "recovery", order_id)
                )
                station = self.entity_mgr.stations.get(waiting_station_id)
                if station is not None:
                    waiting_recovery_by_truck.setdefault(owner_truck_id, []).append(
                        (waiting_station_id, order_id, station.location)
                    )

            launch_station_id = str(getattr(drone, "launch_station_id", "") or "")
            transport_truck_id = str(getattr(drone, "transport_truck_id", "") or "")
            if (
                transport_truck_id == owner_truck_id
                and launch_station_id
                and launch_station_id in self.entity_mgr.stations
                and order_id
            ):
                required_keys_by_truck[owner_truck_id].add(
                    key(launch_station_id, "station", order_id)
                )

            route_plan = getattr(drone, "route_plan", None) or []
            start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            start_idx = max(0, min(start_idx, len(route_plan)))
            for wp in route_plan[start_idx:]:
                if wp.action not in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                    continue
                target_id = str(wp.target_entity_id or "")
                if target_id in self.entity_mgr.stations and order_id:
                    required_keys_by_truck[owner_truck_id].add(
                        key(target_id, "recovery", order_id)
                    )
                break

        for truck_id, seq in stops_by_truck.items():
            required_keys = required_keys_by_truck.get(truck_id, set())

            kept: list[dict] = []
            removed = 0
            for stop in seq:
                node_type = str(stop.get("node_type", "") or "")
                if node_type not in ("station", "recovery"):
                    kept.append(stop)
                    continue

                if bool(stop.get("_ga_static_stop", False)):
                    kept.append(stop)
                    continue

                stop_key = key(
                    str(stop.get("node_id", "") or ""),
                    node_type,
                    str(stop.get("order_id", "") or ""),
                )
                existing_order_id = str(stop.get("order_id", "") or "")
                if node_type == "recovery" and not existing_order_id:
                    matched_order_ids = matching_required_order_ids(stop, required_keys)
                    if len(matched_order_ids) == 1:
                        stop["order_id"] = next(iter(matched_order_ids))
                        stop["_semantic_role"] = "recovery"
                        kept.append(stop)
                        continue
                    if matched_order_ids:
                        removed += 1
                        continue
                if stop_key in required_keys or matches_required_key(stop, required_keys):
                    kept.append(stop)
                else:
                    removed += 1

            if removed > 0:
                logger.info(
                    "[MMCE-BI] GA动态委托卡车 %s 移除 %d 个非任务必需站点停靠",
                    truck_id,
                    removed,
                )
            seq = kept
            inserted = 0
            seen_waiting_requests: set[tuple[str, str]] = set()
            for station_id, order_id, station_pos in waiting_recovery_by_truck.get(truck_id, []):
                request_key = (station_id, order_id)
                if request_key in seen_waiting_requests:
                    continue
                seen_waiting_requests.add(request_key)
                if any(
                    str(stop.get("node_id", "") or "") == station_id
                    and str(stop.get("node_type", "") or "") == "recovery"
                    and (
                        not order_id
                        or str(stop.get("order_id", "") or "") in ("", order_id)
                    )
                    for stop in seq
                ):
                    continue

                truck = self.entity_mgr.trucks.get(truck_id)
                insert_idx = len(seq)
                if truck is not None and current_time is not None:
                    insert_idx, _ = self._best_station_insertion(
                        truck,
                        seq,
                        station_pos,
                        current_time,
                        start_from_current_state,
                        return_to_depot=True,
                    )
                self._ensure_recovery_stop(
                    stops=seq,
                    station_id=station_id,
                    station_pos=station_pos,
                    suggested_idx=insert_idx,
                    min_service=self.TRUCK_DRONE_RECOVER_TIME,
                    desired_node_type="recovery",
                    order_id=order_id,
                    semantic_role="recovery",
                )
                inserted += 1

            if inserted > 0:
                logger.info(
                    "[MMCE-BI] GA动态委托卡车 %s 补入 %d 个运行时等待无人机回收停靠",
                    truck_id,
                    inserted,
                )
            stops_by_truck[truck_id] = seq

    def _resolve_drone_current_future_order_id(self, drone: "Drone") -> str:
        for attr_name in ("carrying_order_id", "pending_release_order_id"):
            order_id = str(getattr(drone, attr_name, "") or "")
            if order_id:
                return order_id

        route_plan = getattr(drone, "route_plan", None) or []
        start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
        start_idx = max(0, min(start_idx, len(route_plan)))
        for wp in route_plan[start_idx:]:
            if wp.action in (WaypointAction.PICKUP, WaypointAction.DELIVER):
                order_id = str(wp.target_entity_id or "")
                if order_id:
                    return order_id
        for wp in reversed(route_plan[:start_idx]):
            if wp.action in (WaypointAction.PICKUP, WaypointAction.DELIVER):
                order_id = str(wp.target_entity_id or "")
                if order_id:
                    return order_id

        segments = getattr(drone, "_ga_runtime_segments", None) or []
        for segment in segments:
            order_id = str(getattr(segment, "order_id", "") or "")
            start = int(getattr(segment, "start_idx", 0) or 0)
            dock = int(getattr(segment, "dock_idx", -1) or -1)
            if order_id and start <= start_idx <= dock + 1:
                return order_id
        return ""

    def _ensure_terminal_station_stop(self, stops: list[dict]) -> bool:
        """确保路线最后停靠为“最后任务点最近充电站”。"""
        if not stops:
            return False

        last = stops[-1]
        anchor = self._nearest_station_for_pos(last["position"])
        if anchor is None:
            return False

        station_id, station_pos = anchor
        if (
            str(last.get("node_id", "")) == station_id
            and str(last.get("node_type", "")) in {"station", "recovery"}
        ):
            return False

        stops.append(
            {
                "node_id": station_id,
                "node_type": "station",
                "position": station_pos,
                "order_id": "",
                "service_time": 0.0,
            }
        )
        logger.info(
            "[MMCE-BI] 终点锚定充电站: station=%s（相对最后任务点）",
            station_id,
        )
        return True

    def _is_backbone_station_energy_feasible(
        self,
        drones: list["Drone"],
        pos_i: Position3D,
        pos_j: Position3D,
        station_pos: Position3D,
    ) -> bool:
        """校验 E >= i->s->i 或 i->s->j（至少一个可行）。"""
        for drone in drones:
            e_isi = (
                self._flight_energy(drone, pos_i, station_pos, 0.0)
                + self._flight_energy(drone, station_pos, pos_i, 0.0)
            ) * self.ENERGY_SAFETY_FACTOR
            e_isj = (
                self._flight_energy(drone, pos_i, station_pos, 0.0)
                + self._flight_energy(drone, station_pos, pos_j, 0.0)
            ) * self.ENERGY_SAFETY_FACTOR
            if min(e_isi, e_isj) <= float(drone.battery_current):
                return True
        return False
