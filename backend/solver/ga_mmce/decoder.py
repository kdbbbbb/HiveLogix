from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .adapters import GAAdapterContext, apply_initial_drone_layout_overlay, clone_state_for_decode
from .chromosome import Individual
from .config import GAConfig
from .physical_evaluator import GACandidate, PhysicalEvaluator

try:
    from utils.coord_utils import wgs84_to_utm
except Exception:  # pragma: no cover - isolated unit tests may not load app utils.
    wgs84_to_utm = None

try:
    from ..greedy_mmce import AllocationResult, DispatchPlan, DroneRoute, TruckRoute, TruckRouteNode
except Exception:  # pragma: no cover - supports isolated GA unit tests without app sys.path.
    try:
        from greedy_mmce import AllocationResult, DispatchPlan, DroneRoute, TruckRoute, TruckRouteNode
    except Exception:

        @dataclass
        class AllocationResult:
            order_id: str
            vehicle_id: str
            mode: str
            distance: float
            feasible: bool
            reason: str = ""
            recovery_station_id: str = ""
            drone_id: str = ""
            launch_station_id: str = ""
            launch_time: float = 0.0
            wait_duration: float = 0.0
            score_total: float = math.inf
            cost_dist: float = 0.0
            cost_energy: float = 0.0
            cost_penalty: float = 0.0
            mode_reward: float = 0.0

        @dataclass
        class TruckRouteNode:
            node_id: str
            node_type: str
            position: Any
            arrival_time: float = 0.0
            departure_time: float = 0.0
            order_id: str = ""

        @dataclass
        class TruckRoute:
            truck_id: str
            nodes: list[TruckRouteNode] = field(default_factory=list)
            total_distance: float = 0.0
            charging_stop_ids: list[str] = field(default_factory=list)
            geometry: list[Any] = field(default_factory=list)

        @dataclass
        class DroneRoute:
            drone_id: str
            order_id: str
            path: list[Any] = field(default_factory=list)
            mode: str = ""
            launch_loc: Any = None
            delivery_loc: Any = None
            recovery_loc: Any = None

        @dataclass
        class DispatchPlan:
            allocations: list[AllocationResult]
            cost_total: float
            summary: dict
            truck_routes: dict[str, TruckRoute] = field(default_factory=dict)
            drone_routes: dict[str, DroneRoute] = field(default_factory=dict)


@dataclass
class DecodeResult:
    plan: Any
    objective: float
    # Values are already weighted costs. Counts are stored in penalty_counts.
    penalties: dict[str, float]
    penalty_counts: dict[str, int]
    feasible: bool
    candidates: list[GACandidate]
    truck_ordered_stops: list[dict[str, Any]]
    drone_route_fragments: list[Any]
    unserved_order_ids: list[str]
    repaired: bool
    metrics: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class _ClosureInfo:
    penalties: dict[str, float] = field(default_factory=dict)
    penalty_counts: dict[str, int] = field(default_factory=dict)
    repaired: bool = False


@dataclass
class _RouteTruckProxy:
    truck_id: str
    speed: float
    start_position: Any

    def get_location(self, current_time: float) -> Any:
        return self.start_position


class GADecoder:
    """Decode a three-layer GA chromosome into a DispatchPlan-compatible plan.

    Decoder owns whole-individual aggregation and closure repair. Physical
    feasibility of each fixed action remains in PhysicalEvaluator.
    """

    def __init__(self, config: GAConfig, evaluator: PhysicalEvaluator):
        self.config = config
        self.evaluator = evaluator

    def decode(self, individual: Individual, state: Any, context: GAAdapterContext) -> DecodeResult:
        individual.validate_with_context(
            truck_drone_ids=context.truck_drone_ids,
            depot_drone_ids=context.depot_drone_ids,
            valid_drone_ids=context.all_drone_ids,
            support_node_ids=context.support_node_ids,
        )

        state_copy = clone_state_for_decode(state)
        apply_initial_drone_layout_overlay(state_copy, context, self.config)
        truck_id = self._select_default_truck_id(state_copy, context)
        initial_time = self._current_time(state_copy)
        initial_truck_position = self._snapshot_truck_position(state_copy, truck_id, initial_time)

        penalties: dict[str, float] = {}
        penalty_counts: dict[str, int] = {}
        allocations: list[AllocationResult] = []
        candidates: list[GACandidate] = []
        drone_route_fragments: list[Any] = []
        truck_ordered_stops: list[dict[str, Any]] = []
        unserved_order_ids: list[str] = []
        attempted_order_ids: set[str] = set()
        gene_attempts: list[dict[str, Any]] = []

        objective = 0.0
        feasible = True

        for order_id, gene, rv in zip(individual.sequence, individual.assignment, individual.rendezvous):
            attempted_order_ids.add(order_id)
            candidate = self._evaluate_gene(state_copy, order_id, gene, rv, truck_id)
            requested_mode = self._gene_mode(gene)
            gene_attempts.append(
                {
                    "order_id": order_id,
                    "gene": gene,
                    "requested_mode": requested_mode,
                    "accepted_mode": candidate.mode if candidate is not None and candidate.feasible else "",
                    "feasible": bool(candidate is not None and candidate.feasible),
                    "reason": candidate.reason if candidate is not None else "evaluation_none",
                    "score_total": candidate.score_total if candidate is not None else math.inf,
                }
            )

            if candidate is None or not candidate.feasible:
                feasible = False
                reason = candidate.reason if candidate is not None and candidate.reason else "infeasible"
                self._add_penalty_count(penalty_counts, reason, 1)
                self._add_penalty_cost(penalties, "infeasible_penalty", self.config.weight_infeasible)
                unserved_order_ids.append(order_id)
                continue

            candidates.append(candidate)
            allocations.append(self._candidate_to_allocation_fragment(candidate))
            truck_ordered_stops.extend(self._copy_stops(candidate.truck_stops))
            if candidate.drone_route_fragment is not None:
                drone_route_fragments.append(candidate.drone_route_fragment)

            objective += self._finite_or(candidate.score_total, self.config.big_m)
            self.evaluator.apply_candidate(state_copy, candidate)

        missing_order_ids = sorted(set(context.order_ids) - attempted_order_ids)
        if missing_order_ids:
            feasible = False
            unserved_order_ids.extend(missing_order_ids)
            self._add_penalty_count(penalty_counts, "unserved_order_penalty", len(missing_order_ids))
            self._add_penalty_cost(
                penalties,
                "unserved_order_penalty",
                len(missing_order_ids) * self.config.weight_infeasible,
            )

        closure_info = self._append_c_station_closure_stops(state_copy, truck_id, truck_ordered_stops)
        for name, value in closure_info.penalties.items():
            self._add_penalty_cost(penalties, name, value)
        for name, count in closure_info.penalty_counts.items():
            self._add_penalty_count(penalty_counts, name, count)
        if "final_return_penalty" in closure_info.penalties:
            feasible = False

        drone_reuse_penalty, drone_reuse_count = self._compute_static_drone_reuse_penalty(
            candidates,
            context,
        )
        if drone_reuse_penalty > 0.0:
            self._add_penalty_cost(penalties, "drone_reuse_penalty", drone_reuse_penalty)
            self._add_penalty_count(penalty_counts, "drone_reuse_penalty", drone_reuse_count)

        truck_routes = self._build_truck_routes_by_given_order(
            state_copy,
            truck_id,
            truck_ordered_stops,
            initial_time,
            initial_truck_position,
        )
        self._retime_closure_pickup_stops(state_copy, truck_routes, truck_ordered_stops)
        drone_routes = self._group_drone_route_fragments(drone_route_fragments)
        metrics = self._collect_plan_metrics(
            truck_routes=truck_routes,
            truck_ordered_stops=truck_ordered_stops,
            drone_route_fragments=drone_route_fragments,
            candidates=candidates,
            state=state_copy,
        )
        plan_completion_time = self._plan_completion_time(candidates, initial_time)
        objective += plan_completion_time * float(self.config.weight_completion)
        future_terms = self._collect_static_future_friendliness(
            original_state=state,
            decoded_state=state_copy,
            context=context,
            truck_routes=truck_routes,
        )
        objective += float(future_terms.get("future_objective", 0.0) or 0.0)
        metrics.update(
            {
                key: float(value)
                for key, value in future_terms.items()
                if key.startswith("future_") and isinstance(value, int | float)
            }
        )

        objective = self._finite_or(objective, self.config.big_m)
        diagnostics = self._collect_decode_diagnostics(gene_attempts, candidates, closure_info)
        if future_terms.get("future_preview_count", 0):
            diagnostics["future_friendliness"] = dict(future_terms)
        cost_breakdown = self._collect_cost_breakdown(
            candidates=candidates,
            metrics=metrics,
            penalties=penalties,
            fitness_total=objective + sum(float(value) for value in penalties.values()),
            plan_completion_time=plan_completion_time,
            future_terms=future_terms,
        )
        plan = self._build_dispatch_plan(
            allocations=allocations,
            truck_routes=truck_routes,
            drone_routes=drone_routes,
            objective=objective,
            penalties=penalties,
            penalty_counts=penalty_counts,
            feasible=feasible,
            metrics=metrics,
            cost_breakdown=cost_breakdown,
            diagnostics=diagnostics,
        )

        return DecodeResult(
            plan=plan,
            objective=objective,
            penalties=penalties,
            penalty_counts=penalty_counts,
            feasible=feasible,
            candidates=candidates,
            truck_ordered_stops=truck_ordered_stops,
            drone_route_fragments=drone_route_fragments,
            unserved_order_ids=list(dict.fromkeys(unserved_order_ids)),
            repaired=closure_info.repaired,
            metrics=metrics,
            diagnostics=diagnostics,
            cost_breakdown=cost_breakdown,
        )

    def _gene_mode(self, gene: str) -> str:
        if gene == "A":
            return "A"
        if gene.startswith("B_"):
            return "B"
        if gene.startswith("C_"):
            return "C"
        return "?"

    def _collect_decode_diagnostics(
        self,
        attempts: list[dict[str, Any]],
        candidates: list[GACandidate],
        closure_info: _ClosureInfo,
    ) -> dict[str, Any]:
        requested_counts = Counter(str(item.get("requested_mode", "?")) for item in attempts)
        accepted_counts = Counter(candidate.mode for candidate in candidates)
        failure_reasons: dict[str, dict[str, int]] = {"B": {}, "C": {}, "A": {}}
        for item in attempts:
            mode = str(item.get("requested_mode", "?"))
            if bool(item.get("feasible")):
                continue
            reason = str(item.get("reason") or "infeasible")
            failure_reasons.setdefault(mode, {})
            failure_reasons[mode][reason] = failure_reasons[mode].get(reason, 0) + 1

        b_scores = [
            float(candidate.score_total)
            for candidate in candidates
            if candidate.mode == "B" and math.isfinite(float(candidate.score_total))
        ]
        b_orders = sorted(candidate.order_id for candidate in candidates if candidate.mode == "B")
        return {
            "requested_mode_counts": dict(requested_counts),
            "accepted_mode_counts": dict(accepted_counts),
            "failure_reasons": failure_reasons,
            "b_decoded_success_count": int(accepted_counts.get("B", 0)),
            "b_candidate_accepted_count": int(accepted_counts.get("B", 0)),
            "b_infeasible_count": sum(failure_reasons.get("B", {}).values()),
            "b_repaired_count": 0,
            "b_failure_reasons": dict(failure_reasons.get("B", {})),
            "best_B_candidate_score": min(b_scores) if b_scores else math.inf,
            "avg_B_candidate_score": sum(b_scores) / len(b_scores) if b_scores else math.inf,
            "orders_where_B_feasible": b_orders,
            "c_decoded_success_count": int(accepted_counts.get("C", 0)),
            "c_candidate_accepted_count": int(accepted_counts.get("C", 0)),
            "c_infeasible_count": sum(failure_reasons.get("C", {}).values()),
            "c_repaired_count": int(closure_info.penalty_counts.get("repair_penalty", 0)),
            "c_failure_reasons": dict(failure_reasons.get("C", {})),
        }

    def _compute_static_drone_reuse_penalty(
        self,
        candidates: list[GACandidate],
        context: GAAdapterContext,
    ) -> tuple[float, int]:
        """Softly prefer spreading initial GA drone work across different UAVs."""
        if str(getattr(context, "mode", "") or "").lower() != "static":
            return 0.0, 0

        factor = float(getattr(self.config, "drone_reuse_penalty_factor", 0.0) or 0.0)
        if factor <= 0.0:
            return 0.0, 0

        drone_use_counts = Counter(
            str(candidate.drone_id or "")
            for candidate in candidates
            if candidate.mode in ("B", "C") and str(candidate.drone_id or "")
        )
        repeated_count = sum(max(0, count - 1) for count in drone_use_counts.values())
        if repeated_count <= 0:
            return 0.0, 0
        return repeated_count * factor, repeated_count

    def _collect_static_future_friendliness(
        self,
        *,
        original_state: Any,
        decoded_state: Any,
        context: GAAdapterContext,
        truck_routes: dict[str, TruckRoute],
    ) -> dict[str, float]:
        """Score how well the static backbone prepares for scheduled dynamics."""
        result = {
            "future_objective": 0.0,
            "future_preview_count": 0.0,
            "future_terminal_penalty": 0.0,
            "future_route_proximity_reward": 0.0,
            "future_station_coverage_reward": 0.0,
            "future_station_coverage_count": 0.0,
        }
        if str(getattr(context, "mode", "") or "").lower() != "static":
            return result
        if not bool(getattr(self.config, "future_preview_enabled", False)):
            return result

        future_positions = self._future_dynamic_positions(original_state)
        if not future_positions:
            return result
        max_orders = int(getattr(self.config, "future_preview_max_orders", 0) or 0)
        if max_orders > 0:
            future_positions = future_positions[:max_orders]

        route_positions = self._route_positions(truck_routes)
        if not route_positions:
            return result

        center = self._centroid(future_positions)
        terminal = self._terminal_route_position(truck_routes) or route_positions[-1]
        terminal_penalty = (
            self._distance_2d(terminal, center)
            * float(getattr(self.config, "future_terminal_position_weight", 0.0) or 0.0)
        )

        route_radius = max(1.0, float(getattr(self.config, "future_route_proximity_radius_m", 1.0) or 1.0))
        route_reward_unit = float(getattr(self.config, "future_route_proximity_reward", 0.0) or 0.0)
        route_reward = 0.0
        for pos in future_positions:
            nearest = min(self._distance_2d(pos, route_pos) for route_pos in route_positions)
            route_reward += route_reward_unit * max(0.0, 1.0 - nearest / route_radius)

        station_radius = max(1.0, float(getattr(self.config, "future_station_coverage_radius_m", 1.0) or 1.0))
        station_reward_unit = float(getattr(self.config, "future_station_coverage_reward", 0.0) or 0.0)
        covered_stations = self._covered_future_station_ids(
            decoded_state,
            future_positions,
            route_positions,
            station_radius,
        )
        station_reward = station_reward_unit * len(covered_stations)

        result.update(
            {
                "future_objective": terminal_penalty - route_reward - station_reward,
                "future_preview_count": float(len(future_positions)),
                "future_terminal_penalty": terminal_penalty,
                "future_route_proximity_reward": route_reward,
                "future_station_coverage_reward": station_reward,
                "future_station_coverage_count": float(len(covered_stations)),
            }
        )
        return result

    def _future_dynamic_positions(self, state: Any) -> list[Any]:
        order_mgr = self._read_field(state, "order_mgr")
        entries = self._read_field(order_mgr, "_scheduled_dynamic", []) if order_mgr is not None else []
        cursor = int(self._read_field(order_mgr, "_scheduled_dynamic_i", 0) or 0) if order_mgr is not None else 0
        now = self._current_time(state)
        positions: list[Any] = []
        for entry in list(entries)[max(0, cursor):]:
            if not isinstance(entry, dict):
                continue
            try:
                if float(entry.get("spawn_sim_s", 0.0)) < now - 1e-6:
                    continue
            except (TypeError, ValueError):
                pass
            pos = self._delivery_position_from_entry(entry)
            if pos is not None:
                positions.append(pos)
        return positions

    def _delivery_position_from_entry(self, entry: dict[str, Any]) -> Any | None:
        loc = entry.get("delivery_loc")
        if loc is not None and hasattr(loc, "x") and hasattr(loc, "y"):
            return loc
        if wgs84_to_utm is None:
            return None
        try:
            x, y = wgs84_to_utm(float(entry["delivery_lng"]), float(entry["delivery_lat"]))
            return SimpleNamespace(x=float(x), y=float(y), z=float(entry.get("delivery_z", 0.0) or 0.0))
        except Exception:
            return None

    def _route_positions(self, truck_routes: dict[str, TruckRoute]) -> list[Any]:
        positions: list[Any] = []
        for route in truck_routes.values():
            for node in route.nodes:
                pos = getattr(node, "position", None)
                if pos is not None and hasattr(pos, "x") and hasattr(pos, "y"):
                    positions.append(pos)
        return positions

    def _terminal_route_position(self, truck_routes: dict[str, TruckRoute]) -> Any | None:
        latest_time = -math.inf
        terminal = None
        for route in truck_routes.values():
            for node in route.nodes:
                pos = getattr(node, "position", None)
                if pos is None or not hasattr(pos, "x") or not hasattr(pos, "y"):
                    continue
                departure = float(getattr(node, "departure_time", getattr(node, "arrival_time", 0.0)) or 0.0)
                if departure >= latest_time:
                    latest_time = departure
                    terminal = pos
        return terminal

    def _covered_future_station_ids(
        self,
        state: Any,
        future_positions: list[Any],
        route_positions: list[Any],
        radius_m: float,
    ) -> set[str]:
        stations = self._mapping(state, "stations")
        if not isinstance(stations, dict):
            return set()
        covered: set[str] = set()
        for station_id, station in stations.items():
            station_pos = self._read_field(station, "location")
            if station_pos is None or not hasattr(station_pos, "x") or not hasattr(station_pos, "y"):
                continue
            near_future = any(self._distance_2d(station_pos, pos) <= radius_m for pos in future_positions)
            if not near_future:
                continue
            near_route = any(self._distance_2d(station_pos, pos) <= radius_m for pos in route_positions)
            if near_route:
                covered.add(str(station_id))
        return covered

    def _centroid(self, positions: list[Any]) -> Any:
        count = max(1, len(positions))
        return SimpleNamespace(
            x=sum(float(pos.x) for pos in positions) / count,
            y=sum(float(pos.y) for pos in positions) / count,
            z=sum(float(getattr(pos, "z", 0.0) or 0.0) for pos in positions) / count,
        )

    def _distance_2d(self, pos_a: Any, pos_b: Any) -> float:
        if hasattr(pos_a, "distance_2d"):
            try:
                return float(pos_a.distance_2d(pos_b))
            except Exception:
                pass
        dx = float(pos_a.x) - float(pos_b.x)
        dy = float(pos_a.y) - float(pos_b.y)
        return math.hypot(dx, dy)

    def _collect_cost_breakdown(
        self,
        candidates: list[GACandidate],
        metrics: dict[str, float],
        penalties: dict[str, float],
        fitness_total: float,
        plan_completion_time: float,
        future_terms: dict[str, float] | None = None,
    ) -> dict[str, float]:
        future_terms = future_terms or {}
        raw_truck_distance = sum(float(c.truck_distance or 0.0) for c in candidates)
        raw_uav_distance = sum(float(c.uav_distance or 0.0) for c in candidates)
        truck_distance_cost = 0.0
        uav_distance_cost = 0.0
        energy_cost = sum(float(c.cost_energy or 0.0) for c in candidates)
        time_cost = float(plan_completion_time) * float(self.config.weight_completion)
        waiting_cost = sum(float(c.waiting_time or 0.0) for c in candidates) * float(self.config.weight_waiting)
        delay_cost = sum(float(c.lateness or 0.0) for c in candidates) * float(self.config.weight_delay)
        mode_reward = sum(float(getattr(c, "mode_reward", 0.0) or 0.0) for c in candidates)
        closure_route_cost = 0.0
        repair_penalty = float(penalties.get("repair_penalty", 0.0) or 0.0)
        station_queue_penalty = float(penalties.get("station_queue_penalty", 0.0) or 0.0)
        unserved_penalty = float(penalties.get("unserved_order_penalty", 0.0) or 0.0)
        infeasible_penalty = float(penalties.get("infeasible_penalty", 0.0) or 0.0)
        final_return_penalty = float(penalties.get("final_return_penalty", 0.0) or 0.0)
        drone_reuse_penalty = float(penalties.get("drone_reuse_penalty", 0.0) or 0.0)
        future_terminal_penalty = float(future_terms.get("future_terminal_penalty", 0.0) or 0.0)
        future_route_proximity_reward = float(future_terms.get("future_route_proximity_reward", 0.0) or 0.0)
        future_station_coverage_reward = float(future_terms.get("future_station_coverage_reward", 0.0) or 0.0)
        total = (
            truck_distance_cost
            + uav_distance_cost
            + energy_cost
            + time_cost
            + waiting_cost
            + delay_cost
            + closure_route_cost
            + repair_penalty
            + station_queue_penalty
            + unserved_penalty
            + infeasible_penalty
            + final_return_penalty
            + drone_reuse_penalty
            + future_terminal_penalty
            - mode_reward
            - future_route_proximity_reward
            - future_station_coverage_reward
        )
        residual = float(fitness_total) - total
        return {
            "raw_truck_distance": raw_truck_distance,
            "raw_uav_distance": raw_uav_distance,
            "truck_distance_cost": truck_distance_cost,
            "uav_distance_cost": uav_distance_cost,
            "energy_cost": energy_cost,
            "time_cost": time_cost,
            "waiting_cost": waiting_cost,
            "delay_cost": delay_cost,
            "plan_completion_time": float(plan_completion_time),
            "air_ground_reward": mode_reward,
            "closure_route_cost": closure_route_cost,
            "repair_penalty": repair_penalty,
            "station_queue_penalty": station_queue_penalty,
            "unserved_penalty": unserved_penalty,
            "infeasible_penalty": infeasible_penalty,
            "final_return_penalty": final_return_penalty,
            "drone_reuse_penalty": drone_reuse_penalty,
            "future_terminal_penalty": future_terminal_penalty,
            "future_route_proximity_reward": future_route_proximity_reward,
            "future_station_coverage_reward": future_station_coverage_reward,
            "future_objective": float(future_terms.get("future_objective", 0.0) or 0.0),
            "residual_cost": residual,
            "total_fitness": float(fitness_total),
        }

    def _evaluate_gene(
        self,
        state: Any,
        order_id: str,
        gene: str,
        rv: dict[str, str] | None,
        truck_id: str,
    ) -> GACandidate | None:
        if gene == "A":
            return self.evaluator.evaluate_fixed_mode_a(state, order_id=order_id, truck_id=truck_id)

        if gene.startswith("B_"):
            if not isinstance(rv, dict):
                return GACandidate(order_id=order_id, mode="B", feasible=False, reason="missing_rendezvous")
            drone_id = gene.split("_", 1)[1]
            return self.evaluator.evaluate_fixed_mode_b(
                state,
                order_id=order_id,
                truck_id=truck_id,
                drone_id=drone_id,
                launch_node_id=rv["launch"],
                recover_node_id=rv["recover"],
            )

        if gene.startswith("C_"):
            if not isinstance(rv, dict):
                return GACandidate(order_id=order_id, mode="C", feasible=False, reason="missing_rendezvous")
            drone_id = gene.split("_", 1)[1]
            return self.evaluator.evaluate_fixed_mode_c(
                state,
                order_id=order_id,
                drone_id=drone_id,
                recover_node_id=rv["recover"],
            )

        return GACandidate(order_id=order_id, mode="", feasible=False, reason=f"unknown_gene:{gene}")

    def _append_c_station_closure_stops(
        self,
        state: Any,
        truck_id: str,
        truck_ordered_stops: list[dict[str, Any]],
    ) -> _ClosureInfo:
        waiting_by_station = self._collect_waiting_station_drones(state)
        if not waiting_by_station:
            return _ClosureInfo()

        if not self.config.enable_repair:
            return _ClosureInfo(
                penalties={"final_return_penalty": len(waiting_by_station) * self.config.weight_infeasible},
                penalty_counts={"final_return_penalty": sum(len(v) for v in waiting_by_station.values())},
                repaired=False,
            )

        penalties: dict[str, float] = {}
        penalty_counts: dict[str, int] = {}

        for station_id, drone_ids in self._sort_waiting_stations(state, waiting_by_station):
            station_pos = self.evaluator.get_node_position(station_id, state)
            truck_ordered_stops.append(
                {
                    "node_id": station_id,
                    "node_type": "station",
                    "position": station_pos,
                    "action": "closure_pickup_c_drone",
                    "source": "repair",
                    "drone_ids": drone_ids,
                }
            )
            self._add_penalty_cost(penalties, "repair_penalty", self.config.repair_penalty_factor)
            self._add_penalty_cost(
                penalties,
                "station_queue_penalty",
                len(drone_ids) * self.config.weight_waiting,
            )
            self._add_penalty_count(penalty_counts, "repair_penalty", 1)
            self._add_penalty_count(penalty_counts, "station_queue_penalty", len(drone_ids))
            self._mark_closure_pickup(state, truck_id, station_id, drone_ids)

        return _ClosureInfo(penalties=penalties, penalty_counts=penalty_counts, repaired=True)

    def _collect_waiting_station_drones(self, state: Any) -> dict[str, list[str]]:
        drones = self._mapping(state, "drones")
        stations = self._mapping(state, "stations")
        waiting_by_station: dict[str, list[str]] = {}

        for station_id, station in stations.items():
            for drone_id in self._as_list(self._read_field(station, "_ga_waiting_drones")):
                if drone_id:
                    waiting_by_station.setdefault(str(station_id), []).append(str(drone_id))

        for drone_id, drone in drones.items():
            if str(self._read_field(drone, "_ga_host_type", "")).upper() != "STATION":
                continue
            station_id = (
                self._read_field(drone, "_ga_waiting_station_id")
                or self._read_field(drone, "_ga_host_node_id")
            )
            if station_id:
                waiting_by_station.setdefault(str(station_id), []).append(str(drone_id))

        return {
            station_id: list(dict.fromkeys(drone_ids))
            for station_id, drone_ids in waiting_by_station.items()
            if drone_ids
        }

    def _sort_waiting_stations(
        self,
        state: Any,
        waiting_by_station: dict[str, list[str]],
    ) -> list[tuple[str, list[str]]]:
        drones = self._mapping(state, "drones")

        def first_wait_time(item: tuple[str, list[str]]) -> tuple[float, str]:
            station_id, drone_ids = item
            times = [
                float(self._read_field(drones.get(drone_id), "_ga_time", math.inf) or math.inf)
                for drone_id in drone_ids
                if drone_id in drones
            ]
            return (min(times) if times else math.inf, station_id)

        sorted_items = sorted(waiting_by_station.items(), key=first_wait_time)
        return [(station_id, sorted(drone_ids)) for station_id, drone_ids in sorted_items]

    def _mark_closure_pickup(self, state: Any, truck_id: str, station_id: str, drone_ids: list[str]) -> None:
        """Close C-mode station recoveries at Individual end.

        This terminal repair only marks lifecycle closure. It is not a hook for
        reusing these drones later in the same decode pass, so drone _ga_time is
        intentionally not advanced to the truck route arrival time here.
        """
        drones = self._mapping(state, "drones")
        for drone_id in drone_ids:
            drone = drones.get(drone_id)
            if drone is None:
                continue
            if hasattr(self.evaluator, "_set_drone_host"):
                self.evaluator._set_drone_host(  # noqa: SLF001 - GA internals share virtual state helpers.
                    state,
                    drone,
                    drone_id,
                    "TRUCK",
                    station_id,
                    truck_id=truck_id,
                    candidate=None,
                )
            else:
                self._write_field(drone, "_ga_host_type", "TRUCK")
                self._write_field(drone, "_ga_host_node_id", station_id)
                self._write_field(drone, "_ga_transport_truck_id", truck_id)
                self._write_field(drone, "_ga_waiting_station_id", None)

    def _build_truck_routes_by_given_order(
        self,
        state: Any,
        truck_id: str,
        truck_ordered_stops: list[dict[str, Any]],
        initial_time: float,
        initial_truck_position: Any,
    ) -> dict[str, TruckRoute]:
        trucks = self._mapping(state, "trucks")
        truck = trucks.get(truck_id)
        if truck is None:
            return {}

        route_stops = [self._normalize_stop_for_route(state, stop) for stop in truck_ordered_stops]
        route_truck = self._route_truck_proxy(truck, truck_id, initial_truck_position)
        try:
            route = self.evaluator.greedy.build_incremental_route_from_stops(
                truck=route_truck,
                ordered_stops=route_stops,
                current_time=initial_time,
            )
        except Exception:
            route = None
        if route is None:
            route = self._build_direct_truck_route(
                route_truck,
                route_stops,
                initial_time,
                initial_truck_position,
            )
        return {truck_id: route}

    def _build_direct_truck_route(
        self,
        truck: Any,
        ordered_stops: list[dict[str, Any]],
        current_time: float,
        initial_truck_position: Any,
    ) -> TruckRoute:
        route = TruckRoute(truck_id=self._read_field(truck, "truck_id", ""))
        cur_pos = initial_truck_position
        cur_time = current_time
        route.nodes.append(
            TruckRouteNode(
                node_id=f"{route.truck_id}_origin",
                node_type="origin",
                position=cur_pos,
                arrival_time=current_time,
                departure_time=current_time,
            )
        )
        route.geometry.append(cur_pos)

        speed = max(1e-6, float(self._read_field(truck, "speed", 0.0) or 0.0))
        total_dist = 0.0
        for stop in ordered_stops:
            stop_pos = stop.get("position")
            if stop_pos is None:
                continue
            dist = self._road_distance(cur_pos, stop_pos)
            arrival = cur_time + dist / speed
            service_time = max(
                0.0,
                float(stop.get("departure_time", arrival) or arrival)
                - float(stop.get("arrival_time", arrival) or arrival),
            )
            departure = arrival + service_time
            route.nodes.append(
                TruckRouteNode(
                    node_id=str(stop.get("node_id", "")),
                    node_type=str(stop.get("node_type", "")),
                    position=stop_pos,
                    arrival_time=arrival,
                    departure_time=departure,
                    order_id=str(stop.get("order_id", "")),
                )
            )
            if stop.get("node_type") == "station":
                route.charging_stop_ids.append(str(stop.get("node_id", "")))
            route.geometry.append(stop_pos)
            total_dist += dist
            cur_pos = stop_pos
            cur_time = departure

        route.total_distance = total_dist
        return route

    def _retime_closure_pickup_stops(
        self,
        state: Any,
        truck_routes: dict[str, TruckRoute],
        truck_ordered_stops: list[dict[str, Any]],
    ) -> None:
        """Make terminal recovery stops wait until already-committed drones arrive."""
        if not truck_routes or not truck_ordered_stops:
            return

        event_stops = [
            stop
            for stop in truck_ordered_stops
            if stop.get("position") is not None
        ]
        if not event_stops:
            return

        recover_time = self._recover_service_time()
        event_node_types = {"customer", "recovery", "station"}
        for route in truck_routes.values():
            delay = 0.0
            stop_index = 0
            for node in route.nodes:
                stop = None
                if node.node_type in event_node_types and stop_index < len(event_stops):
                    stop = event_stops[stop_index]
                    stop_index += 1

                if delay:
                    node.arrival_time += delay
                    node.departure_time += delay

                if not stop or stop.get("action") != "closure_pickup_c_drone":
                    continue

                ready_time = self._closure_ready_time(
                    state,
                    stop.get("drone_ids", []) or [],
                    node.arrival_time,
                )
                needed_departure = max(node.arrival_time, ready_time) + recover_time
                if node.departure_time + 1e-6 >= needed_departure:
                    continue
                extra_delay = needed_departure - node.departure_time
                node.departure_time = needed_departure
                delay += extra_delay

    def _closure_ready_time(self, state: Any, drone_ids: list[str], fallback: float) -> float:
        drones = self._mapping(state, "drones")
        ready_times: list[float] = []
        for drone_id in drone_ids:
            drone = drones.get(str(drone_id))
            if drone is None:
                continue
            ready_times.append(float(self._read_field(drone, "_ga_time", fallback) or fallback))
        return max(ready_times) if ready_times else float(fallback)

    def _recover_service_time(self) -> float:
        if hasattr(self.evaluator, "_recover_time"):
            return float(self.evaluator._recover_time())  # noqa: SLF001 - GA internals share service constants.
        return 10.0

    def _collect_plan_metrics(
        self,
        truck_routes: dict[str, TruckRoute],
        truck_ordered_stops: list[dict[str, Any]],
        drone_route_fragments: list[Any],
        candidates: list[GACandidate],
        state: Any,
    ) -> dict[str, float]:
        route_truck_distance = sum(float(route.total_distance or 0.0) for route in truck_routes.values())
        candidate_truck_distance = sum(float(candidate.truck_distance or 0.0) for candidate in candidates)
        closure_route_distance = max(0.0, route_truck_distance - candidate_truck_distance)
        closure_route_cost = 0.0

        closure_waiting_drone_count = 0.0
        closure_waiting_time = 0.0
        for stop in truck_ordered_stops:
            if stop.get("action") == "closure_pickup_c_drone":
                drone_ids = stop.get("drone_ids", []) or []
                closure_waiting_drone_count += len(drone_ids)
                closure_waiting_time += self._closure_stop_waiting_time(
                    state,
                    truck_routes,
                    stop,
                    drone_ids,
                )

        return {
            "candidate_count": float(len(candidates)),
            "truck_stop_count": float(len(truck_ordered_stops)),
            "drone_route_fragment_count": float(len(drone_route_fragments)),
            "route_truck_distance": route_truck_distance,
            "candidate_truck_distance": candidate_truck_distance,
            "closure_route_distance": closure_route_distance,
            "closure_route_cost": closure_route_cost,
            "closure_waiting_drone_count": closure_waiting_drone_count,
            "closure_waiting_time": closure_waiting_time,
        }

    def _plan_completion_time(self, candidates: list[GACandidate], initial_time: float = 0.0) -> float:
        if not candidates:
            return 0.0
        latest_completion = max(float(candidate.completion_time or 0.0) for candidate in candidates)
        return max(0.0, latest_completion - float(initial_time or 0.0))

    def _closure_stop_waiting_time(
        self,
        state: Any,
        truck_routes: dict[str, TruckRoute],
        stop: dict[str, Any],
        drone_ids: list[str],
    ) -> float:
        truck_arrival = self._route_arrival_time_for_stop(truck_routes, stop)
        if truck_arrival is None:
            return 0.0

        drones = self._mapping(state, "drones")
        total_wait = 0.0
        for drone_id in drone_ids:
            drone = drones.get(str(drone_id))
            if drone is None:
                continue
            drone_ready_time = float(self._read_field(drone, "_ga_time", truck_arrival) or truck_arrival)
            total_wait += max(0.0, truck_arrival - drone_ready_time)
        return total_wait

    def _route_arrival_time_for_stop(
        self,
        truck_routes: dict[str, TruckRoute],
        stop: dict[str, Any],
    ) -> float | None:
        node_id = str(stop.get("node_id", ""))
        action = stop.get("action")
        source = stop.get("source")
        for route in truck_routes.values():
            for node in route.nodes:
                if node.node_id != node_id:
                    continue
                if action == "closure_pickup_c_drone" and source == "repair":
                    return float(node.arrival_time)
                return float(node.arrival_time)
        return None

    def _build_dispatch_plan(
        self,
        allocations: list[AllocationResult],
        truck_routes: dict[str, TruckRoute],
        drone_routes: dict[str, DroneRoute],
        objective: float,
        penalties: dict[str, float],
        penalty_counts: dict[str, int],
        feasible: bool,
        metrics: dict[str, float],
        cost_breakdown: dict[str, float],
        diagnostics: dict[str, Any],
    ) -> DispatchPlan:
        total_penalty_cost = sum(float(value) for value in penalties.values())
        fitness_total = self._finite_or(objective + total_penalty_cost, self.config.big_m)
        summary = {
            "feasible": feasible,
            "objective": objective,
            "fitness_total": fitness_total,
            "penalties": dict(penalties),
            "penalty_counts": dict(penalty_counts),
            "metrics": dict(metrics),
            "cost_breakdown": dict(cost_breakdown),
            "diagnostics": dict(diagnostics),
            "total_penalty_cost": total_penalty_cost,
        }
        return DispatchPlan(
            allocations=allocations,
            cost_total=fitness_total,
            summary=summary,
            truck_routes=truck_routes,
            drone_routes=drone_routes,
        )

    def _candidate_to_allocation_fragment(self, candidate: GACandidate) -> AllocationResult:
        runtime_mode = candidate.mode
        if candidate.mode == "B" and candidate.launch_node_id:
            runtime_mode = "B_WAIT"
        return AllocationResult(
            order_id=candidate.order_id,
            vehicle_id=candidate.truck_id or candidate.launch_node_id or "DEPOT",
            mode=runtime_mode,
            distance=float(candidate.truck_distance or 0.0) + float(candidate.uav_distance or 0.0),
            feasible=candidate.feasible,
            reason=candidate.reason,
            recovery_station_id=candidate.recover_node_id,
            drone_id=candidate.drone_id,
            launch_station_id=candidate.launch_node_id,
            wait_duration=candidate.waiting_time,
            score_total=candidate.score_total,
            cost_dist=candidate.cost_dist,
            cost_energy=candidate.cost_energy,
            cost_penalty=candidate.cost_penalty,
        )

    def _group_drone_route_fragments(self, fragments: list[Any]) -> dict[str, DroneRoute]:
        routes: dict[str, DroneRoute] = {}
        for index, fragment in enumerate(fragments):
            route = self._fragment_to_drone_route(fragment)
            if route is None:
                continue
            # A drone may execute multiple tasks in one Individual; keep every route.
            key = route.drone_id
            if key in routes:
                key = f"{route.drone_id}#{route.order_id or index}#{index}"
            routes[key] = route
        return routes

    def _fragment_to_drone_route(self, fragment: Any) -> DroneRoute | None:
        if isinstance(fragment, DroneRoute):
            return fragment
        if fragment is None:
            return None

        drone_id = str(self._read_field(fragment, "drone_id", "") or "")
        order_id = str(self._read_field(fragment, "order_id", "") or "")
        mode = str(self._read_field(fragment, "mode", "") or "")
        path = list(self._read_field(fragment, "path", []) or [])
        if not drone_id:
            return None

        launch_loc = self._read_field(fragment, "launch_loc", path[0] if path else None)
        delivery_loc = self._read_field(fragment, "delivery_loc", path[1] if len(path) > 1 else launch_loc)
        recovery_loc = self._read_field(fragment, "recovery_loc", path[-1] if path else launch_loc)
        return DroneRoute(
            drone_id=drone_id,
            order_id=order_id,
            path=path,
            mode=mode,
            launch_loc=launch_loc,
            delivery_loc=delivery_loc,
            recovery_loc=recovery_loc,
        )

    def _normalize_stop_for_route(self, state: Any, stop: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(stop)
        node_id = str(normalized.get("node_id", ""))
        if node_id in self._mapping(state, "stations"):
            normalized["node_type"] = "station"
        elif self._is_depot_node_id(state, node_id):
            normalized["node_type"] = "depot"
        return normalized

    def _select_default_truck_id(self, state: Any, context: GAAdapterContext) -> str:
        if context.truck_ids:
            return context.truck_ids[0]
        trucks = self._mapping(state, "trucks")
        if trucks:
            return str(next(iter(trucks.keys())))
        raise ValueError("no truck available for GA decoding")

    def _snapshot_truck_position(self, state: Any, truck_id: str, current_time: float) -> Any:
        truck = self._mapping(state, "trucks").get(truck_id)
        if truck is None:
            return None
        return self._truck_start_position(truck, current_time)

    def _route_truck_proxy(self, truck: Any, truck_id: str, initial_truck_position: Any) -> _RouteTruckProxy:
        return _RouteTruckProxy(
            truck_id=truck_id,
            speed=float(self._read_field(truck, "speed", 0.0) or 0.0),
            start_position=initial_truck_position,
        )

    def _copy_stops(self, stops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(stop) for stop in stops]

    def _read_field(self, record: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(field_name, default)
        return getattr(record, field_name, default)

    def _write_field(self, record: Any, field_name: str, value: Any) -> None:
        if isinstance(record, dict):
            record[field_name] = value
        else:
            setattr(record, field_name, value)

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple | set):
            return list(value)
        return [value]

    def _mapping(self, state: Any, field_name: str) -> dict[str, Any]:
        if hasattr(self.evaluator, "_mapping"):
            return self.evaluator._mapping(state, field_name)  # noqa: SLF001 - shared GA adapter helper.
        value = self._read_field(state, field_name)
        if isinstance(value, dict):
            return value
        mgr = self._read_field(state, "entity_mgr") or self._read_field(state, "entity_manager") or state
        value = self._read_field(mgr, field_name, {})
        return value if isinstance(value, dict) else {}

    def _current_time(self, state: Any) -> float:
        if hasattr(self.evaluator, "_current_time"):
            return float(self.evaluator._current_time(state))  # noqa: SLF001
        return float(self._read_field(state, "current_time", 0.0) or 0.0)

    def _truck_start_position(self, truck: Any, current_time: float) -> Any:
        ga_pos = self._read_field(truck, "_ga_position")
        if ga_pos is not None:
            return ga_pos
        if hasattr(truck, "get_location"):
            return truck.get_location(current_time)
        return self._read_field(truck, "current_loc") or self._read_field(truck, "_ga_position")

    def _road_distance(self, pos_a: Any, pos_b: Any) -> float:
        try:
            return float(self.evaluator.greedy._road_dist(pos_a, pos_b))
        except Exception:
            return float(self.evaluator.greedy._dist(pos_a, pos_b))

    def _is_depot_node_id(self, state: Any, node_id: str) -> bool:
        normalized = str(node_id).strip().upper()
        return (
            normalized == "DEPOT"
            or normalized.startswith("DEPOT")
            or normalized.startswith("DEP-")
            or node_id in self._mapping(state, "depots")
        )

    def _add_penalty_cost(self, penalties: dict[str, float], name: str, value: float) -> None:
        penalties[name] = penalties.get(name, 0.0) + float(value)

    def _add_penalty_count(self, penalty_counts: dict[str, int], name: str, value: int) -> None:
        penalty_counts[name] = penalty_counts.get(name, 0) + int(value)

    def _finite_or(self, value: float, fallback: float) -> float:
        value = float(value)
        if math.isfinite(value):
            return value
        return float(fallback)
