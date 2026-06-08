#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 6 coarse planner bridge。

当前实现刻意保持保守：
  - 不复用 greedy / market 现有算法实现；
  - 只承担 coarse plan 刷新、触发判定与契约化输出；
  - 粗规划内容先使用稳定规则生成，后续可在不改接口的前提下替换为更强的 RH-ALNS。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    CoarsePlanView,
    PlannerMode,
    PlannerTriggerContext,
    PolicyMode,
    ReservationPlanOutcome,
    ReservationPlanStatus,
    RouteDriftRef,
    TruckPlanStopView,
    TruckReservationConstraint,
)
from .recovery_pool_selector import select_recovery_pool_for_order
from .scene_loader import DEFAULT_CONFIG_PATH


_TIME_EPS = 1e-6
_EXACT_ROUTE_SEARCH_MAX_NODES = 8
_RECOVERY_OPPORTUNITY_BONUS_SEC = 90.0
_RECOVERY_OPPORTUNITY_BONUS_CAP_SEC = 360.0


@dataclass(frozen=True)
class _PlannerConfig:
    coarse_replan_interval_sec: float
    coarse_new_order_trigger: int
    route_drift_trigger_ratio: float
    fallback_burst_trigger_count: int
    fallback_burst_window_sec: float
    hard_failure_trigger_count: int
    upper_horizon_sec: float
    support_radius_km: float
    min_orders_to_trigger: int
    patrol_stations_per_loop: int
    max_candidate_recovery_per_order: int
    recovery_pool_future_scan_limit: int
    rendezvous_execution_margin_sec: float
    rendezvous_max_wait_sec: float
    allow_empty_backbone_route: bool
    beam_width: int


@dataclass(frozen=True)
class _TruckPlanNode:
    node_id: str
    node_type: str
    order_id: str | None
    position: Any
    service_time_sec: float
    deadline: float | None = None
    reservation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TruckRouteCandidate:
    nodes: tuple[_TruckPlanNode, ...]
    arrival_times: tuple[float, ...]
    departure_times: tuple[float, ...]
    key: tuple[float, ...]


@dataclass(frozen=True)
class _ReservationWindow:
    constraint: TruckReservationConstraint
    earliest_eta: float
    latest_eta: float
    preferred_eta: float


@dataclass(frozen=True)
class _PlanVisit:
    node_id: str
    arrival_time: float
    departure_time: float


class PlannerBridge:
    """低频 coarse plan 桥接器。"""

    def __init__(
        self,
        *,
        future_backbone_provider: Callable[[float], Sequence[Any]],
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        heavy_payload_capacity: float | None = None,
        truck_speed_provider: Callable[[], float] | None = None,
        truck_travel_time_provider: Callable[[Any, Any], float] | None = None,
    ) -> None:
        self._cfg = _load_planner_config(Path(config_path))
        self._future_backbone_provider = future_backbone_provider
        self._heavy_payload_capacity = (
            float(heavy_payload_capacity)
            if heavy_payload_capacity is not None
            else _load_heavy_payload_capacity()
        )
        self._recovery_pool_drone_cruise_speed = (
            _load_recovery_pool_drone_cruise_speed()
        )
        self._truck_speed_provider = truck_speed_provider
        self._truck_travel_time_provider = truck_travel_time_provider
        self._truck_order_service_time_sec = _load_truck_order_service_time()
        self._drone_service_time_order_sec = _load_drone_order_service_time()
        self._fixed_node_service_time_sec = _load_fixed_node_service_time()
        self._runtime_allow_empty_backbone_route = bool(
            self._cfg.allow_empty_backbone_route
        )
        self._current_plan: CoarsePlanView | None = None

    @property
    def current_plan(self) -> CoarsePlanView | None:
        return self._current_plan

    def reset_episode(
        self,
        *,
        allow_empty_backbone_route: bool | None = None,
    ) -> None:
        """清空跨 episode coarse-plan 缓存，并同步运行时语义开关。"""
        self._current_plan = None
        if allow_empty_backbone_route is not None:
            self._runtime_allow_empty_backbone_route = bool(
                allow_empty_backbone_route
            )

    def maybe_replan(
        self,
        runtime_state: Any,
        trigger_ctx: PlannerTriggerContext,
        reservation_constraints: Sequence[TruckReservationConstraint] = (),
        allow_empty_backbone_route: bool | None = None,
    ) -> CoarsePlanView:
        if self._current_plan is None:
            self._current_plan = self._build_plan(
                runtime_state=runtime_state,
                t_now=float(trigger_ctx.t_now),
                plan_version=0,
                reservation_constraints=reservation_constraints,
                allow_empty_backbone_route=allow_empty_backbone_route,
            )
            return self._current_plan

        if not self._should_replan(trigger_ctx):
            return self._current_plan

        self._current_plan = self._build_plan(
            runtime_state=runtime_state,
            t_now=float(trigger_ctx.t_now),
            plan_version=self._current_plan.plan_version + 1,
            reservation_constraints=reservation_constraints,
            allow_empty_backbone_route=allow_empty_backbone_route,
        )
        return self._current_plan

    def _should_replan(self, trigger_ctx: PlannerTriggerContext) -> bool:
        assert self._current_plan is not None
        if trigger_ctx.t_now >= self._current_plan.valid_until - _TIME_EPS:
            return True
        if trigger_ctx.backlog_new_orders >= self._cfg.coarse_new_order_trigger:
            return True
        if trigger_ctx.route_drift_ratio >= self._cfg.route_drift_trigger_ratio:
            return True
        if (
            trigger_ctx.fallback_count_in_window
            >= self._cfg.fallback_burst_trigger_count
        ):
            return True
        if (
            trigger_ctx.hard_failure_count_in_window
            >= self._cfg.hard_failure_trigger_count
        ):
            return True
        return False

    def _build_plan(
        self,
        *,
        runtime_state: Any,
        t_now: float,
        plan_version: int,
        reservation_constraints: Sequence[TruckReservationConstraint],
        allow_empty_backbone_route: bool | None,
    ) -> CoarsePlanView:
        future_visits = self._dedupe_future_backbone(
            self._future_backbone_provider(t_now)
        )
        truck_plan_stops = self._build_dynamic_truck_plan_stops(
            runtime_state=runtime_state,
            t_now=t_now,
            baseline_visits=future_visits,
            reservation_constraints=reservation_constraints,
        )
        plan_fixed_visits = self._visits_from_truck_plan_stops(
            truck_plan_stops
        ) or future_visits
        truck_backbone_route = tuple(visit.node_id for visit in plan_fixed_visits)
        effective_allow_empty_backbone_route = (
            self._runtime_allow_empty_backbone_route
            if allow_empty_backbone_route is None
            else bool(allow_empty_backbone_route)
        )
        truck_eta_map = {
            visit.node_id: float(visit.arrival_time) for visit in plan_fixed_visits
        }
        route_drift_ref = {
            visit.node_id: RouteDriftRef(
                eta_ref=float(visit.arrival_time),
                route_index_ref=idx,
            )
            for idx, visit in enumerate(plan_fixed_visits)
        }
        reservation_outcomes = self._build_reservation_outcomes(
            reservation_constraints=reservation_constraints,
            truck_eta_map=self._truck_eta_map_for_reservations(
                truck_plan_stops=truck_plan_stops,
                fallback_eta_map=truck_eta_map,
            ),
        )

        authorized_orders: list[str] = []
        order_priority_band: dict[str, int] = {}
        order_pre_score: dict[str, float] = {}
        planner_mode_cap: dict[str, frozenset[PlannerMode]] = {}
        policy_mode_mask: dict[str, frozenset[PolicyMode]] = {}
        recovery_pool: dict[str, tuple[str, ...]] = {}

        pending_items = sorted(
            runtime_state.pending_orders.items(),
            key=lambda item: (float(item[1].deadline), item[0]),
        )
        for order_id, order in pending_items:
            if float(order.payload_weight) > self._heavy_payload_capacity:
                planner_mode_cap[order_id] = frozenset({PlannerMode.A})
                continue

            remaining = max(0.0, float(order.deadline) - t_now)
            window = max(_TIME_EPS, float(order.time_window_seconds))
            ratio = remaining / window
            if ratio <= (1.0 / 3.0):
                band = 0
            elif ratio <= (2.0 / 3.0):
                band = 1
            else:
                band = 2

            authorized_orders.append(order_id)
            order_priority_band[order_id] = band
            order_pre_score[order_id] = remaining
            planner_mode_cap[order_id] = frozenset({PlannerMode.B, PlannerMode.C})
            if truck_backbone_route:
                policy_mode_mask[order_id] = frozenset({PolicyMode.B, PolicyMode.C})
                recovery_pool[order_id] = select_recovery_pool_for_order(
                    order=order,
                    truck_backbone_route=truck_backbone_route,
                    truck_eta_map=truck_eta_map,
                    node_states=runtime_state.node_states,
                    max_candidates=self._cfg.max_candidate_recovery_per_order,
                    future_scan_limit=self._cfg.recovery_pool_future_scan_limit,
                    drone_cruise_speed=self._recovery_pool_drone_cruise_speed,
                    upper_horizon_sec=self._cfg.upper_horizon_sec,
                )
            else:
                policy_mode_mask[order_id] = frozenset({PolicyMode.B})
                recovery_pool[order_id] = ()

        node_charge_load_budget = {
            node_id: 0 for node_id in runtime_state.node_states
        }

        launch_candidate_stations = self._select_launch_candidate_stations(
            runtime_state=runtime_state,
            truck_backbone_route=truck_backbone_route,
        )

        return CoarsePlanView(
            plan_version=plan_version,
            issued_at=float(t_now),
            valid_until=max(
                float(t_now),
                min(
                    float(t_now + self._cfg.coarse_replan_interval_sec),
                    float(self._cfg.upper_horizon_sec),
                ),
            ),
            truck_backbone_route=truck_backbone_route,
            truck_eta_map=truck_eta_map,
            authorized_orders=tuple(authorized_orders),
            order_priority_band=order_priority_band,
            order_pre_score=order_pre_score,
            planner_mode_cap=planner_mode_cap,
            policy_mode_mask=policy_mode_mask,
            recovery_pool=recovery_pool,
            node_charge_load_budget=node_charge_load_budget,
            route_drift_ref=route_drift_ref,
            launch_candidate_stations=launch_candidate_stations,
            allow_empty_backbone_route=effective_allow_empty_backbone_route,
            reservation_outcomes=reservation_outcomes,
            truck_plan_stops=truck_plan_stops,
        )

    def _build_dynamic_truck_plan_stops(
        self,
        *,
        runtime_state: Any,
        t_now: float,
        baseline_visits: Sequence[Any],
        reservation_constraints: Sequence[TruckReservationConstraint],
    ) -> tuple[TruckPlanStopView, ...]:
        truck_only_orders = self._resolve_truck_mandatory_orders(runtime_state)
        require_station_backbone = bool(
            getattr(runtime_state, "require_station_backbone", False)
        )
        if (
            not truck_only_orders
            and not reservation_constraints
            and not require_station_backbone
        ):
            return ()

        reservation_nodes = self._build_reservation_plan_nodes(
            runtime_state=runtime_state,
            reservation_constraints=reservation_constraints,
        )
        truck_order_nodes = [
            _TruckPlanNode(
                node_id=str(order_id),
                node_type="customer",
                order_id=str(order_id),
                position=order.delivery_loc,
                service_time_sec=self._truck_order_service_time_sec,
                deadline=float(order.deadline),
            )
            for order_id, order in truck_only_orders
        ]
        mandatory_nodes = tuple(truck_order_nodes + list(reservation_nodes.values()))
        if not mandatory_nodes and not require_station_backbone:
            return ()

        start_pos = runtime_state.truck_current_loc
        truck_speed = self._resolve_truck_speed_mps()
        truck_route_ready_at = float(
            getattr(runtime_state, "truck_route_ready_at", None) or t_now
        )
        if mandatory_nodes:
            best = self._search_required_truck_route(
                start_pos=start_pos,
                start_time=truck_route_ready_at,
                mandatory_nodes=mandatory_nodes,
                reservation_constraints=reservation_constraints,
                truck_speed=truck_speed,
            )
            if best is None:
                return ()
            base_nodes = best.nodes
        else:
            base_nodes = ()

        depot_node = self._select_depot_node(runtime_state)
        if truck_order_nodes:
            coverage_nodes = self._select_coverage_nodes(
                runtime_state=runtime_state,
                baseline_visits=baseline_visits,
                selected_nodes={node.node_id for node in base_nodes},
            )
            nodes = self._append_station_coverage(
                runtime_state=runtime_state,
                base_nodes=base_nodes,
                coverage_nodes=coverage_nodes,
                depot_node=depot_node,
                start_pos=start_pos,
                start_time=truck_route_ready_at,
                reservation_constraints=reservation_constraints,
                truck_speed=truck_speed,
            )
        elif require_station_backbone:
            coverage_nodes = self._select_station_backbone_nodes(
                runtime_state=runtime_state,
                selected_nodes={node.node_id for node in base_nodes},
            )
            nodes = self._append_station_coverage(
                runtime_state=runtime_state,
                base_nodes=base_nodes,
                coverage_nodes=coverage_nodes,
                depot_node=depot_node,
                start_pos=start_pos,
                start_time=truck_route_ready_at,
                reservation_constraints=reservation_constraints,
                truck_speed=truck_speed,
            )
        else:
            # recovery-only route：没有卡车派送订单时，只兑现已锁定的
            # Mode C rendezvous 节点，然后回仓，不再强制补足巡站覆盖数量。
            nodes = base_nodes
        if depot_node is not None and (
            not nodes or nodes[-1].node_id != depot_node.node_id
        ):
            nodes = tuple(nodes) + (depot_node,)

        return self._simulate_truck_plan_stops(
            nodes=nodes,
            start_pos=start_pos,
            start_time=truck_route_ready_at,
            truck_speed=truck_speed,
            reservation_constraints=reservation_constraints,
        )

    def _resolve_truck_mandatory_orders(self, runtime_state: Any) -> list[tuple[str, Any]]:
        explicit_orders = getattr(runtime_state, "truck_mandatory_orders", None)
        if explicit_orders is not None:
            return sorted(
                ((str(order_id), order) for order_id, order in explicit_orders.items()),
                key=lambda item: (float(item[1].deadline), item[0]),
            )

        return [
            (order_id, order)
            for order_id, order in sorted(
                runtime_state.pending_orders.items(),
                key=lambda item: (float(item[1].deadline), item[0]),
            )
            if float(order.payload_weight) > self._heavy_payload_capacity
        ]

    def _build_reservation_outcomes(
        self,
        *,
        reservation_constraints: Sequence[TruckReservationConstraint],
        truck_eta_map: Mapping[str, float],
    ) -> dict[str, ReservationPlanOutcome]:
        outcomes: dict[str, ReservationPlanOutcome] = {}
        for constraint in reservation_constraints:
            new_eta = truck_eta_map.get(constraint.node_id)
            if new_eta is None:
                outcomes[constraint.reservation_id] = ReservationPlanOutcome(
                    reservation_id=constraint.reservation_id,
                    node_id=constraint.node_id,
                    old_eta=float(constraint.eta_ref),
                    new_eta=None,
                    eta_drift_sec=None,
                    status=ReservationPlanStatus.INVALIDATED,
                    invalidate_cause="node_not_in_truck_plan",
                )
                continue

            window = self._reservation_window(constraint)
            early_sec = max(0.0, window.earliest_eta - float(new_eta))
            late_sec = max(0.0, float(new_eta) - window.latest_eta)
            eta_drift_sec = max(0.0, float(new_eta) - float(constraint.eta_ref))
            if early_sec > _TIME_EPS:
                status = ReservationPlanStatus.INVALIDATED
                invalidate_cause = "arrived_before_reservation_window"
            elif late_sec > _TIME_EPS:
                status = ReservationPlanStatus.INVALIDATED
                invalidate_cause = "reservation_window_missed"
            elif eta_drift_sec <= _TIME_EPS:
                status = ReservationPlanStatus.KEPT
                invalidate_cause = None
            else:
                status = ReservationPlanStatus.DRIFTED
                invalidate_cause = None

            outcomes[constraint.reservation_id] = ReservationPlanOutcome(
                reservation_id=constraint.reservation_id,
                node_id=constraint.node_id,
                old_eta=float(constraint.eta_ref),
                new_eta=float(new_eta),
                eta_drift_sec=float(eta_drift_sec),
                status=status,
                invalidate_cause=invalidate_cause,
            )
        return outcomes

    def _build_reservation_plan_nodes(
        self,
        *,
        runtime_state: Any,
        reservation_constraints: Sequence[TruckReservationConstraint],
    ) -> dict[str, _TruckPlanNode]:
        reservation_ids_by_node: dict[str, list[str]] = {}
        for constraint in reservation_constraints:
            reservation_ids_by_node.setdefault(constraint.node_id, []).append(
                constraint.reservation_id
            )

        nodes: dict[str, _TruckPlanNode] = {}
        for node_id, reservation_ids in sorted(reservation_ids_by_node.items()):
            node_state = runtime_state.node_states.get(node_id)
            if node_state is None:
                continue
            nodes[node_id] = _TruckPlanNode(
                node_id=str(node_id),
                node_type=str(node_state.node_type),
                order_id=None,
                position=node_state.position,
                service_time_sec=self._fixed_node_service_time_sec,
                reservation_ids=tuple(sorted(reservation_ids)),
            )
        return nodes

    def _search_required_truck_route(
        self,
        *,
        start_pos: Any,
        start_time: float,
        mandatory_nodes: Sequence[_TruckPlanNode],
        reservation_constraints: Sequence[TruckReservationConstraint],
        truck_speed: float,
    ) -> _TruckRouteCandidate | None:
        if not mandatory_nodes:
            return None

        beam: list[tuple[_TruckPlanNode, ...]] = [()]
        ordered_nodes = tuple(
            sorted(
                mandatory_nodes,
                key=lambda node: (
                    float("inf") if node.deadline is None else float(node.deadline),
                    node.node_type,
                    node.node_id,
                ),
            )
        )
        if len(ordered_nodes) <= _EXACT_ROUTE_SEARCH_MAX_NODES:
            candidates = [
                self._evaluate_truck_route(
                    nodes=nodes,
                    start_pos=start_pos,
                    start_time=start_time,
                    reservation_constraints=reservation_constraints,
                    truck_speed=truck_speed,
                )
                for nodes in permutations(ordered_nodes)
            ]
            return min(
                candidates,
                key=lambda candidate: (
                    candidate.key,
                    tuple(node.node_id for node in candidate.nodes),
                ),
            )

        for _depth in range(len(ordered_nodes)):
            expanded: list[_TruckRouteCandidate] = []
            for prefix in beam:
                used = {node.node_id for node in prefix}
                for node in ordered_nodes:
                    if node.node_id in used:
                        continue
                    candidate_nodes = tuple(prefix) + (node,)
                    expanded.append(
                        self._evaluate_truck_route(
                            nodes=candidate_nodes,
                            start_pos=start_pos,
                            start_time=start_time,
                            reservation_constraints=reservation_constraints,
                            truck_speed=truck_speed,
                        )
                    )
            expanded.sort(
                key=lambda candidate: (
                    candidate.key,
                    tuple(node.node_id for node in candidate.nodes),
                )
            )
            beam = [item.nodes for item in expanded[: self._cfg.beam_width]]

        if not beam:
            return None
        finals = [
            self._evaluate_truck_route(
                nodes=nodes,
                start_pos=start_pos,
                start_time=start_time,
                reservation_constraints=reservation_constraints,
                truck_speed=truck_speed,
            )
            for nodes in beam
        ]
        return min(
            finals,
            key=lambda candidate: (
                candidate.key,
                tuple(node.node_id for node in candidate.nodes),
            ),
        )

    def _evaluate_truck_route(
        self,
        *,
        nodes: Sequence[_TruckPlanNode],
        start_pos: Any,
        start_time: float,
        reservation_constraints: Sequence[TruckReservationConstraint],
        truck_speed: float,
    ) -> _TruckRouteCandidate:
        arrival_times, departure_times = self._simulate_truck_node_times(
            nodes=nodes,
            start_pos=start_pos,
            start_time=start_time,
            truck_speed=truck_speed,
            reservation_constraints=reservation_constraints,
        )
        key = self._truck_route_key(
            nodes=nodes,
            arrival_times=arrival_times,
            reservation_constraints=reservation_constraints,
            start_time=start_time,
        )
        return _TruckRouteCandidate(
            nodes=tuple(nodes),
            arrival_times=arrival_times,
            departure_times=departure_times,
            key=key,
        )

    def _truck_route_key(
        self,
        *,
        nodes: Sequence[_TruckPlanNode],
        arrival_times: Sequence[float],
        reservation_constraints: Sequence[TruckReservationConstraint],
        start_time: float,
    ) -> tuple[float, ...]:
        truck_timeout_count = 0
        total_truck_lateness = 0.0
        arrival_by_node: dict[str, float] = {}
        for node, arrival in zip(nodes, arrival_times, strict=True):
            arrival_by_node.setdefault(node.node_id, float(arrival))
            if node.node_type != "customer" or node.deadline is None:
                continue
            lateness = max(0.0, float(arrival) - float(node.deadline))
            if lateness > _TIME_EPS:
                truck_timeout_count += 1
                total_truck_lateness += lateness

        strong_hard_invalid_count = 0
        hard_invalid_count = 0
        total_strong_hard_violation = 0.0
        total_hard_violation = 0.0
        total_reservation_wait = 0.0
        for constraint in reservation_constraints:
            arrival = arrival_by_node.get(constraint.node_id)
            window = self._reservation_window(constraint)
            if arrival is None:
                if constraint.state.value == "strong_hard":
                    strong_hard_invalid_count += 1
                else:
                    hard_invalid_count += 1
                continue
            early_sec = max(0.0, window.earliest_eta - float(arrival))
            late_sec = max(0.0, float(arrival) - window.latest_eta)
            violation_sec = early_sec + late_sec
            total_reservation_wait += max(0.0, float(arrival) - window.preferred_eta)
            if constraint.state.value == "strong_hard":
                total_strong_hard_violation += violation_sec
                if violation_sec > _TIME_EPS:
                    strong_hard_invalid_count += 1
            else:
                total_hard_violation += violation_sec
                if violation_sec > _TIME_EPS:
                    hard_invalid_count += 1

        last_departure = float(start_time)
        if nodes:
            _arrival_times, departure_times = self._simulate_truck_node_times(
                nodes=nodes,
                start_pos=None,
                start_time=start_time,
                truck_speed=1.0,
                precomputed_arrivals=tuple(arrival_times),
                reservation_constraints=reservation_constraints,
            )
            last_departure = departure_times[-1]
        total_route_time = max(0.0, last_departure - float(start_time))
        return (
            float(truck_timeout_count),
            float(total_truck_lateness),
            float(strong_hard_invalid_count),
            float(hard_invalid_count),
            float(total_strong_hard_violation),
            float(total_hard_violation),
            float(total_reservation_wait),
            float(total_route_time),
        )

    def _simulate_truck_node_times(
        self,
        *,
        nodes: Sequence[_TruckPlanNode],
        start_pos: Any | None,
        start_time: float,
        truck_speed: float,
        precomputed_arrivals: Sequence[float] | None = None,
        reservation_constraints: Sequence[TruckReservationConstraint] = (),
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        arrivals: list[float] = []
        departures: list[float] = []
        windows_by_node = self._reservation_windows_by_node(reservation_constraints)
        if precomputed_arrivals is not None:
            for node, arrival in zip(nodes, precomputed_arrivals, strict=True):
                effective_arrival = max(
                    float(arrival),
                    self._reservation_wait_until(node.node_id, windows_by_node),
                )
                arrivals.append(float(effective_arrival))
                departures.append(
                    float(effective_arrival) + float(node.service_time_sec)
                )
            return tuple(arrivals), tuple(departures)

        if start_pos is None:
            raise ValueError("未提供 start_pos 且没有 precomputed_arrivals")
        current_pos = start_pos
        t_cursor = float(start_time)
        for node in nodes:
            travel_time = self._truck_travel_time_between_positions(
                current_pos,
                node.position,
                truck_speed=truck_speed,
            )
            arrival = max(
                t_cursor + travel_time,
                self._reservation_wait_until(node.node_id, windows_by_node),
            )
            departure = arrival + float(node.service_time_sec)
            arrivals.append(float(arrival))
            departures.append(float(departure))
            current_pos = node.position
            t_cursor = departure
        return tuple(arrivals), tuple(departures)

    def _reservation_window(
        self,
        constraint: TruckReservationConstraint,
    ) -> _ReservationWindow:
        earliest = (
            float(constraint.earliest_eta)
            if constraint.earliest_eta is not None
            else float(constraint.eta_ref)
        )
        latest = (
            float(constraint.latest_eta)
            if constraint.latest_eta is not None
            else float(constraint.eta_ref) + float(constraint.max_eta_drift_sec)
        )
        preferred = (
            float(constraint.preferred_eta)
            if constraint.preferred_eta is not None
            else earliest
        )
        if latest < earliest:
            latest = earliest
        preferred = min(max(preferred, earliest), latest)
        return _ReservationWindow(
            constraint=constraint,
            earliest_eta=float(earliest),
            latest_eta=float(latest),
            preferred_eta=float(preferred),
        )

    def _reservation_windows_by_node(
        self,
        reservation_constraints: Sequence[TruckReservationConstraint],
    ) -> dict[str, tuple[_ReservationWindow, ...]]:
        windows: dict[str, list[_ReservationWindow]] = {}
        for constraint in reservation_constraints:
            windows.setdefault(str(constraint.node_id), []).append(
                self._reservation_window(constraint)
            )
        return {
            node_id: tuple(sorted(items, key=lambda item: item.earliest_eta))
            for node_id, items in windows.items()
        }

    @staticmethod
    def _reservation_wait_until(
        node_id: str,
        windows_by_node: Mapping[str, tuple[_ReservationWindow, ...]],
    ) -> float:
        windows = windows_by_node.get(str(node_id), ())
        if not windows:
            return 0.0
        return max(window.earliest_eta for window in windows)

    def _append_station_coverage(
        self,
        *,
        runtime_state: Any,
        base_nodes: Sequence[_TruckPlanNode],
        coverage_nodes: Sequence[_TruckPlanNode],
        depot_node: _TruckPlanNode | None,
        start_pos: Any,
        start_time: float,
        reservation_constraints: Sequence[TruckReservationConstraint],
        truck_speed: float,
    ) -> tuple[_TruckPlanNode, ...]:
        target_station_count = max(0, int(self._cfg.patrol_stations_per_loop))
        if target_station_count <= 0:
            return tuple(base_nodes)

        selected = tuple(base_nodes)
        if _station_count(selected) >= target_station_count:
            return selected

        selected_node_ids = {item.node_id for item in selected}
        remaining_coverage = tuple(
            node for node in coverage_nodes if node.node_id not in selected_node_ids
        )
        if not remaining_coverage:
            return selected

        base_eval_nodes = selected + ((depot_node,) if depot_node is not None else ())
        base_key = self._evaluate_truck_route(
            nodes=base_eval_nodes,
            start_pos=start_pos,
            start_time=start_time,
            reservation_constraints=reservation_constraints,
            truck_speed=truck_speed,
        ).key
        while _station_count(selected) < target_station_count:
            best_insert: (
                tuple[
                    float,
                    float,
                    float,
                    tuple[str, ...],
                    tuple[_TruckPlanNode, ...],
                    tuple[float, ...],
                ]
                | None
            ) = None
            selected_ids = {node.node_id for node in selected}
            for station in remaining_coverage:
                if station.node_id in selected_ids:
                    continue
                for insert_idx in range(len(selected) + 1):
                    candidate = (
                        selected[:insert_idx]
                        + (station,)
                        + selected[insert_idx:]
                    )
                    candidate_eval_nodes = candidate + (
                        (depot_node,) if depot_node is not None else ()
                    )
                    candidate_eval = self._evaluate_truck_route(
                        nodes=candidate_eval_nodes,
                        start_pos=start_pos,
                        start_time=start_time,
                        reservation_constraints=reservation_constraints,
                        truck_speed=truck_speed,
                    )
                    candidate_key = candidate_eval.key
                    if _worsens_primary_metrics(candidate_key, base_key, count=7):
                        continue
                    extra_route_time = candidate_key[-1] - base_key[-1]
                    station_eta = candidate_eval.arrival_times[insert_idx]
                    opportunity_bonus = self._station_recovery_opportunity_bonus(
                        runtime_state=runtime_state,
                        station=station,
                        station_eta=float(station_eta),
                    )
                    adjusted_extra_time = float(extra_route_time) - float(
                        opportunity_bonus
                    )
                    candidate_order = tuple(node.node_id for node in candidate)
                    current_best = best_insert
                    if current_best is None or (
                        adjusted_extra_time,
                        -opportunity_bonus,
                        extra_route_time,
                        candidate_order,
                    ) < (
                        current_best[0],
                        current_best[1],
                        current_best[2],
                        current_best[3],
                    ):
                        best_insert = (
                            float(adjusted_extra_time),
                            float(-opportunity_bonus),
                            float(extra_route_time),
                            candidate_order,
                            candidate,
                            candidate_key,
                        )
            if best_insert is None:
                break
            selected = best_insert[4]
            base_key = best_insert[5]
        return selected

    def _station_recovery_opportunity_bonus(
        self,
        *,
        runtime_state: Any,
        station: _TruckPlanNode,
        station_eta: float,
    ) -> float:
        """Estimate how useful a station visit is for pending UAV Mode C recovery.

        This is a coarse truck-planning signal only. CandidateBuilder remains the
        source of truth for per-drone energy and rendezvous legality.
        """
        if station.node_type != "station":
            return 0.0

        pending_orders = getattr(runtime_state, "pending_orders", {})
        if not pending_orders:
            return 0.0

        support_radius_m = max(100.0, float(self._cfg.support_radius_km) * 1000.0)
        bonus = 0.0
        for order in pending_orders.values():
            if float(order.payload_weight) > self._heavy_payload_capacity + _TIME_EPS:
                continue

            delivery_finish = self._estimate_earliest_uav_delivery_finish(
                runtime_state=runtime_state,
                order=order,
            )
            if delivery_finish is None:
                continue
            if float(delivery_finish) > float(order.deadline) + _TIME_EPS:
                continue

            distance_to_station = float(order.delivery_loc.distance_2d(station.position))
            spatial_score = max(0.0, 1.0 - distance_to_station / support_radius_m)
            if spatial_score <= _TIME_EPS:
                continue

            recovery_flight_time = distance_to_station / max(
                _TIME_EPS,
                float(self._recovery_pool_drone_cruise_speed),
            )
            uav_arrival = float(delivery_finish) + float(recovery_flight_time)
            planned_wait = float(station_eta) - float(uav_arrival)
            temporal_score = self._rendezvous_temporal_score(planned_wait)
            if temporal_score <= _TIME_EPS:
                continue

            remaining = max(0.0, float(order.deadline) - float(runtime_state.t_now))
            window = max(_TIME_EPS, float(order.time_window_seconds))
            urgency = 1.0 - min(1.0, remaining / window)
            bonus += (
                _RECOVERY_OPPORTUNITY_BONUS_SEC
                * spatial_score
                * temporal_score
                * (0.75 + 0.25 * urgency)
            )

        return min(_RECOVERY_OPPORTUNITY_BONUS_CAP_SEC, float(bonus))

    def _estimate_earliest_uav_delivery_finish(
        self,
        *,
        runtime_state: Any,
        order: Any,
    ) -> float | None:
        best_flight_time: float | None = None
        for drone in getattr(runtime_state, "drone_states", {}).values():
            if float(order.payload_weight) > float(drone.payload_capacity) + _TIME_EPS:
                continue
            speed = max(_TIME_EPS, float(drone.cruise_speed))
            flight_time = float(drone.current_loc.distance_2d(order.delivery_loc)) / speed
            if best_flight_time is None or flight_time < best_flight_time:
                best_flight_time = flight_time
        if best_flight_time is None:
            return None
        return (
            float(runtime_state.t_now)
            + float(best_flight_time)
            + float(self._drone_service_time_order_sec)
        )

    def _rendezvous_temporal_score(self, planned_wait: float) -> float:
        lower = float(self._cfg.rendezvous_execution_margin_sec)
        upper = float(self._cfg.rendezvous_max_wait_sec)
        if lower <= planned_wait <= upper:
            return 1.0
        span = max(_TIME_EPS, upper - lower)
        if planned_wait < lower:
            return max(0.0, 1.0 - (lower - float(planned_wait)) / span)
        return max(0.0, 1.0 - (float(planned_wait) - upper) / span)

    def _select_station_backbone_nodes(
        self,
        *,
        runtime_state: Any,
        selected_nodes: set[str],
    ) -> tuple[_TruckPlanNode, ...]:
        nodes: list[_TruckPlanNode] = []
        for node_id, node_state in sorted(runtime_state.node_states.items()):
            node_id = str(node_id)
            if node_id in selected_nodes:
                continue
            if node_state.node_type != "station":
                continue
            if (
                node_state.position.distance_2d(runtime_state.truck_current_loc)
                <= _TIME_EPS
            ):
                continue
            nodes.append(
                _TruckPlanNode(
                    node_id=node_id,
                    node_type="station",
                    order_id=None,
                    position=node_state.position,
                    service_time_sec=self._fixed_node_service_time_sec,
                )
            )
        return tuple(nodes)

    def _select_coverage_nodes(
        self,
        *,
        runtime_state: Any,
        baseline_visits: Sequence[Any],
        selected_nodes: set[str],
    ) -> tuple[_TruckPlanNode, ...]:
        nodes: list[_TruckPlanNode] = []
        seen = set(selected_nodes)
        for visit in baseline_visits:
            node_id = str(visit.node_id)
            if node_id in seen:
                continue
            node_state = runtime_state.node_states.get(node_id)
            if node_state is None or node_state.node_type != "station":
                continue
            seen.add(node_id)
            nodes.append(
                _TruckPlanNode(
                    node_id=node_id,
                    node_type="station",
                    order_id=None,
                    position=node_state.position,
                    service_time_sec=self._fixed_node_service_time_sec,
                )
            )
            if len(nodes) >= self._cfg.patrol_stations_per_loop:
                break
        return tuple(nodes)

    def _select_depot_node(self, runtime_state: Any) -> _TruckPlanNode | None:
        for node_id, node_state in sorted(runtime_state.node_states.items()):
            if node_state.node_type != "depot":
                continue
            return _TruckPlanNode(
                node_id=str(node_id),
                node_type="depot",
                order_id=None,
                position=node_state.position,
                service_time_sec=0.0,
            )
        return None

    def _simulate_truck_plan_stops(
        self,
        *,
        nodes: Sequence[_TruckPlanNode],
        start_pos: Any,
        start_time: float,
        truck_speed: float,
        reservation_constraints: Sequence[TruckReservationConstraint] = (),
    ) -> tuple[TruckPlanStopView, ...]:
        arrivals, departures = self._simulate_truck_node_times(
            nodes=nodes,
            start_pos=start_pos,
            start_time=start_time,
            truck_speed=truck_speed,
            reservation_constraints=reservation_constraints,
        )
        return tuple(
            TruckPlanStopView(
                seq=idx,
                node_type=node.node_type,
                node_id=node.node_id,
                order_id=node.order_id,
                arrival_time=float(arrivals[idx]),
                departure_time=float(departures[idx]),
            )
            for idx, node in enumerate(nodes)
        )

    def _visits_from_truck_plan_stops(
        self,
        truck_plan_stops: Sequence[TruckPlanStopView],
    ) -> tuple[_PlanVisit, ...]:
        visits: list[_PlanVisit] = []
        seen: set[str] = set()
        for stop in truck_plan_stops:
            if stop.node_type not in {"station", "depot"}:
                continue
            if stop.node_id in seen:
                continue
            seen.add(stop.node_id)
            visits.append(
                _PlanVisit(
                    node_id=stop.node_id,
                    arrival_time=float(stop.arrival_time),
                    departure_time=float(stop.departure_time),
                )
            )
        return tuple(visits)

    def _truck_eta_map_for_reservations(
        self,
        *,
        truck_plan_stops: Sequence[TruckPlanStopView],
        fallback_eta_map: Mapping[str, float],
    ) -> Mapping[str, float]:
        eta_map = dict(fallback_eta_map)
        for stop in truck_plan_stops:
            eta_map.setdefault(stop.node_id, float(stop.arrival_time))
        return eta_map

    def _resolve_truck_speed_mps(self) -> float:
        if self._truck_speed_provider is not None:
            return max(_TIME_EPS, float(self._truck_speed_provider()))
        return 8.0

    def _truck_travel_time_between_positions(
        self,
        from_pos: Any,
        to_pos: Any,
        *,
        truck_speed: float,
    ) -> float:
        if self._truck_travel_time_provider is not None:
            return max(
                0.0,
                float(self._truck_travel_time_provider(from_pos, to_pos)),
            )
        raise RuntimeError(
            "PlannerBridge 动态卡车重排必须提供 truck_travel_time_provider，"
            "不能回退到直线距离"
        )

    def _select_launch_candidate_stations(
        self,
        *,
        runtime_state: Any,
        truck_backbone_route: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not truck_backbone_route:
            return ()

        launch_nodes: list[str] = []
        for node_id in truck_backbone_route:
            node_state = runtime_state.node_states.get(node_id)
            if node_state is None or node_state.node_type != "station":
                continue
            launch_nodes.append(node_id)
        return tuple(launch_nodes)

    @staticmethod
    def _dedupe_future_backbone(visits: Sequence[Any]) -> tuple[Any, ...]:
        deduped: list[Any] = []
        seen_nodes: set[str] = set()
        sorted_visits = sorted(
            visits,
            key=lambda item: (float(item.arrival_time), str(item.node_id)),
        )
        for visit in sorted_visits:
            node_id = str(visit.node_id)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            deduped.append(visit)
        return tuple(deduped)


def _load_planner_config(config_path: Path) -> _PlannerConfig:
    raw = _load_yaml(config_path)
    planner = _require_mapping(raw, "planner")
    candidate = _require_mapping(raw, "candidate")
    max_candidate_recovery_per_order = int(
        candidate["max_candidate_recovery_per_order"]
    )
    recovery_pool_future_scan_limit = int(
        candidate.get(
            "recovery_pool_future_scan_limit",
            max_candidate_recovery_per_order,
        )
    )
    if max_candidate_recovery_per_order <= 0:
        raise ValueError("candidate.max_candidate_recovery_per_order 必须为正数")
    if recovery_pool_future_scan_limit < max_candidate_recovery_per_order:
        raise ValueError(
            "candidate.recovery_pool_future_scan_limit 不能小于 "
            "max_candidate_recovery_per_order"
        )
    return _PlannerConfig(
        coarse_replan_interval_sec=float(planner["coarse_replan_interval_sec"]),
        coarse_new_order_trigger=int(planner["coarse_new_order_trigger"]),
        route_drift_trigger_ratio=float(planner["route_drift_trigger_ratio"]),
        fallback_burst_trigger_count=int(planner["fallback_burst_trigger_count"]),
        fallback_burst_window_sec=float(planner["fallback_burst_window_sec"]),
        hard_failure_trigger_count=int(planner["hard_failure_trigger_count"]),
        upper_horizon_sec=float(planner["upper_horizon_sec"]),
        support_radius_km=float(planner["support_radius_km"]),
        min_orders_to_trigger=int(planner["min_orders_to_trigger"]),
        patrol_stations_per_loop=int(planner.get("patrol_stations_per_loop", 0)),
        max_candidate_recovery_per_order=max_candidate_recovery_per_order,
        recovery_pool_future_scan_limit=recovery_pool_future_scan_limit,
        rendezvous_execution_margin_sec=float(
            candidate.get(
                "rendezvous_execution_margin_sec",
                candidate.get("rendezvous_filter_margin_sec", 0.0),
            )
        ),
        rendezvous_max_wait_sec=float(candidate["rendezvous_max_wait_sec"]),
        allow_empty_backbone_route=bool(
            planner.get("allow_empty_backbone_route", False)
        ),
        beam_width=max(1, int(planner.get("beam_width", 8))),
    )


def _load_heavy_payload_capacity() -> float:
    from config.loader import load_drone_params

    return float(load_drone_params().heavy.payload_capacity)


def _load_recovery_pool_drone_cruise_speed() -> float:
    from config.loader import load_drone_params

    return float(load_drone_params().light.cruise_speed)


def _load_truck_order_service_time() -> float:
    from config.loader import load_solver_energy_params

    return float(load_solver_energy_params().truck_service_time_order_s)


def _load_drone_order_service_time() -> float:
    from config.loader import load_solver_energy_params

    return float(load_solver_energy_params().drone_service_time_order_s)


def _load_fixed_node_service_time() -> float:
    from config.loader import load_solver_energy_params

    params = load_solver_energy_params()
    return max(
        float(params.truck_drone_launch_time_s),
        float(params.truck_drone_recover_time_s),
    )


def _station_count(nodes: Sequence[_TruckPlanNode]) -> int:
    return sum(1 for node in nodes if node.node_type == "station")


def _worsens_primary_metrics(
    candidate_key: tuple[float, ...],
    base_key: tuple[float, ...],
    *,
    count: int,
) -> bool:
    for idx in range(min(count, len(candidate_key), len(base_key))):
        if candidate_key[idx] > base_key[idx] + _TIME_EPS:
            return True
    return False


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取 planner 配置") from exc

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML 顶层必须为 mapping: {config_path}")
    return raw


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置缺少 mapping 段: {key}")
    return value


__all__ = ["PlannerBridge"]
