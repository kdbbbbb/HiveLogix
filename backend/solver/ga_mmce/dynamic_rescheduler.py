from __future__ import annotations

import copy
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .adapters import build_ga_context, clone_state_for_decode
from .chromosome import Individual
from .config import DYNAMIC_GA_CONFIG, GAConfig, make_ga_config
from .decoder import AllocationResult, DispatchPlan
from .operators import make_random_rendezvous_for_gene, mutate
from .population import enforce_fixed_tail


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAST_DYNAMIC_REPLAN_STATS: dict[str, Any] = {}


@dataclass
class OrderBuckets:
    completed: dict[str, Any]
    locked: dict[str, Any]
    pending: dict[str, Any]
    new: dict[str, Any]


DYNAMIC_REPLAN_FIELDS = [
    "event_time",
    "new_order_ids",
    "completed_ids",
    "locked_ids",
    "pending_count",
    "reoptimized_order_count",
    "frozen_future_order_count",
    "warm_start_count",
    "population_size",
    "max_generations",
    "actual_generations",
    "elapsed_seconds",
    "time_budget_hit",
    "early_stop_triggered",
    "fallback_used",
    "fallback_level",
    "best_fitness",
    "final_A_count",
    "final_B_count",
    "final_C_count",
    "urgent_rescue_order_ids",
    "urgent_rescue_count",
    "unserved_order_ids",
]


def reschedule_on_event(state: Any, new_orders: Any, event_time: float) -> DispatchPlan:
    """Main dynamic entrypoint for GA-MMCE.

    Static GA dispatch remains handled by ``GAMMCESolver.solve``.  Dynamic
    events are delegated to the greedy backbone-insertion solver so new work is
    inserted into the current suffix without running the dynamic GA path.
    """
    started = time.time()
    advanced_state = _advance_or_snapshot_state(state, event_time)
    solver = _resolve_solver(advanced_state)

    incoming_new = _normalize_order_mapping(new_orders)
    buckets = _classify_orders(advanced_state, incoming_new, event_time)
    dynamic_config = make_ga_config(DYNAMIC_GA_CONFIG, base=getattr(solver, "config", None))

    urgent_rescue_orders = _select_urgent_rescue_orders(
        advanced_state,
        buckets,
        dynamic_config,
        event_time,
    )
    planning_orders = _select_greedy_dynamic_orders(advanced_state, buckets)
    if urgent_rescue_orders:
        planning_orders = _merge_order_mappings(planning_orders, urgent_rescue_orders)
    force_depot_direct_ids = {
        order_id for order_id in urgent_rescue_orders
        if order_id in planning_orders
    }
    reoptimized_ids = list(planning_orders)
    frozen_future_ids = [
        oid for oid in buckets.pending
        if oid not in set(reoptimized_ids) and oid in _future_planned_order_ids(advanced_state)
    ]
    if not planning_orders:
        plan = DispatchPlan(
            allocations=[],
            cost_total=0.0,
            summary={
                "total_orders": 0,
                "feasible": 0,
                "modes": {},
                "dispatch_type": "dynamic_replan",
                "solver": "ga_mmce",
                "dynamic_solver": "greedy_mmce_bi",
                "ga_dynamic_skipped": True,
                "ga_feasible": False,
            },
        )
        _annotate_dynamic_summary(
            plan,
            event_time=event_time,
            buckets=buckets,
            reoptimized_ids=reoptimized_ids,
            frozen_future_ids=frozen_future_ids,
            warm_start_count=0,
            config=dynamic_config,
            ga_plan=plan,
            ga_feasible=False,
            warm_start_feasible=False,
            greedy_insert_feasible=True,
            fallback_used=False,
            fallback_level="greedy_dynamic_primary",
            unserved_order_ids=[],
            elapsed_seconds=time.time() - started,
        )
        plan.summary["dynamic_solver"] = "greedy_mmce_bi"
        plan.summary["ga_dynamic_skipped"] = True
        plan.summary["urgent_rescue_order_ids"] = list(force_depot_direct_ids)
        plan.summary["urgent_rescue_count"] = len(force_depot_direct_ids)
        _write_dynamic_replan_csv(plan.summary)
        _store_last_stats(plan.summary)
        return plan

    bbox = _read_field(advanced_state, "bbox")
    if not bbox:
        final_plan = _unserved_dynamic_plan(planning_orders, reason="missing_bbox")
        greedy_insert_feasible = False
    else:
        helper = _resolve_dynamic_greedy_helper(solver)
        prev_delegate = getattr(helper, "_ga_mmce_dynamic_delegate", None)
        prev_order_ids = getattr(helper, "_ga_mmce_dynamic_order_ids", None)
        prev_config = getattr(helper, "_ga_mmce_config", None)
        prev_force_direct_ids = getattr(helper, "_ga_mmce_force_depot_direct_order_ids", None)
        setattr(helper, "_ga_mmce_dynamic_delegate", True)
        setattr(helper, "_ga_mmce_dynamic_order_ids", set(planning_orders))
        setattr(helper, "_ga_mmce_config", dynamic_config)
        setattr(helper, "_ga_mmce_force_depot_direct_order_ids", set(force_depot_direct_ids))
        try:
            final_plan = helper.dispatch_replan_current_state(
                planning_orders,
                float(event_time),
                bbox,
                scene_id=_read_field(advanced_state, "scene_id"),
            )
        except Exception:
            final_plan = _unserved_dynamic_plan(planning_orders, reason="greedy_dynamic_failed")
            greedy_insert_feasible = False
        else:
            greedy_insert_feasible = _plan_is_feasible_for_orders(final_plan, planning_orders)
        finally:
            if prev_delegate is None:
                try:
                    delattr(helper, "_ga_mmce_dynamic_delegate")
                except AttributeError:
                    pass
            else:
                setattr(helper, "_ga_mmce_dynamic_delegate", prev_delegate)
            if prev_order_ids is None:
                try:
                    delattr(helper, "_ga_mmce_dynamic_order_ids")
                except AttributeError:
                    pass
            else:
                setattr(helper, "_ga_mmce_dynamic_order_ids", prev_order_ids)
            if prev_config is None:
                try:
                    delattr(helper, "_ga_mmce_config")
                except AttributeError:
                    pass
            else:
                setattr(helper, "_ga_mmce_config", prev_config)
            if prev_force_direct_ids is None:
                try:
                    delattr(helper, "_ga_mmce_force_depot_direct_order_ids")
                except AttributeError:
                    pass
            else:
                setattr(helper, "_ga_mmce_force_depot_direct_order_ids", prev_force_direct_ids)

    unserved_order_ids = list(_unserved_order_ids(final_plan, planning_orders))
    _annotate_dynamic_summary(
        final_plan,
        event_time=event_time,
        buckets=buckets,
        reoptimized_ids=reoptimized_ids,
        frozen_future_ids=frozen_future_ids,
        warm_start_count=0,
        config=dynamic_config,
        ga_plan=final_plan,
        ga_feasible=False,
        warm_start_feasible=False,
        greedy_insert_feasible=greedy_insert_feasible,
        fallback_used=False,
        fallback_level="greedy_dynamic_primary",
        unserved_order_ids=unserved_order_ids,
        elapsed_seconds=time.time() - started,
    )
    final_plan.summary["dynamic_solver"] = "greedy_mmce_bi"
    final_plan.summary["ga_dynamic_skipped"] = True
    final_plan.summary["urgent_rescue_order_ids"] = list(force_depot_direct_ids)
    final_plan.summary["urgent_rescue_count"] = len(force_depot_direct_ids)
    _write_dynamic_replan_csv(final_plan.summary)
    _store_last_stats(final_plan.summary)
    return final_plan


def _resolve_dynamic_greedy_helper(solver: Any) -> Any:
    helper = getattr(solver, "dynamic_greedy_helper", None)
    if helper is not None:
        return helper

    helper = getattr(solver, "greedy_helper", None)
    if helper is not None:
        return helper

    try:
        from ..greedy_mmce_bi import GreedyMMCEBackboneInsertion
    except Exception:
        from solver.greedy_mmce_bi import GreedyMMCEBackboneInsertion
    return GreedyMMCEBackboneInsertion(getattr(solver, "entity_mgr", None))


def _merge_order_mappings(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for order_id, order in mapping.items():
            merged[str(order_id)] = order
    return merged


def _select_greedy_dynamic_orders(state: Any, buckets: OrderBuckets) -> dict[str, Any]:
    if buckets.new:
        return dict(buckets.new)

    order_mgr = _order_mgr(state)
    pending_pool = _read_field(order_mgr, "pending_orders", {}) if order_mgr is not None else {}
    if order_mgr is not None and isinstance(pending_pool, dict):
        return {
            oid: order
            for oid, order in buckets.pending.items()
            if oid in pending_pool
        }

    planned_future = _future_planned_order_ids(state)
    return {
        oid: order
        for oid, order in buckets.pending.items()
        if oid not in planned_future
    }


def _select_urgent_rescue_orders(
    state: Any,
    buckets: OrderBuckets,
    config: GAConfig,
    event_time: float,
) -> dict[str, Any]:
    """Pick GA-only rescue candidates that are near deadline but not yet running."""
    if not bool(getattr(config, "urgent_rescue_enabled", True)):
        return {}

    window_s = _safe_float(
        getattr(config, "urgent_rescue_deadline_window_s", 0.0),
        0.0,
    )
    if window_s <= 0.0:
        return {}

    max_orders = int(getattr(config, "urgent_rescue_max_orders_per_check", 0) or 0)
    order_mgr = _order_mgr(state)
    assigned_pool = _read_field(order_mgr, "assigned_orders", {}) if order_mgr is not None else {}
    if not isinstance(assigned_pool, dict):
        assigned_pool = {}

    future_planned = _future_planned_order_ids(state)
    already_running = _running_order_ids(state)
    candidates: list[tuple[float, str, Any]] = []
    rescue_pool = dict(buckets.pending)
    for order_id, order in buckets.locked.items():
        if _status_name(_read_field(order, "status")) == "ASSIGNED":
            rescue_pool.setdefault(order_id, order)

    for order_id, order in rescue_pool.items():
        oid = str(order_id)
        status = _status_name(_read_field(order, "status"))
        if oid in buckets.new:
            continue
        if oid in already_running and status != "ASSIGNED":
            continue
        if oid not in assigned_pool and oid not in future_planned:
            continue
        if _read_field(order, "actual_deliver_time") is not None:
            continue

        if status in {"PICKED_UP", "DELIVERING", "COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}:
            continue
        if str(_read_field(order, "assigned_mode", "") or "").upper() == "C":
            continue
        if _order_has_flying_drone_route(state, oid):
            continue

        deadline = _safe_float(_read_field(order, "deadline", math.inf), math.inf)
        remaining_s = deadline - float(event_time)
        if remaining_s > window_s:
            continue
        candidates.append((remaining_s, oid, order))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if max_orders > 0:
        candidates = candidates[:max_orders]
    return {oid: order for _, oid, order in candidates}


def _order_has_flying_drone_route(state: Any, order_id: str) -> bool:
    mgr = _entity_mgr(state)
    for drone in (_mapping(mgr, "drones") or {}).values():
        if not _drone_is_flying(drone):
            continue
        route_plan = _read_field(drone, "route_plan", []) or []
        start_idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
        for wp in route_plan[max(0, start_idx):]:
            if _waypoint_action_name(_read_field(wp, "action")) not in {"PICKUP", "DELIVER"}:
                continue
            if str(_read_field(wp, "target_entity_id", "") or "") == str(order_id):
                return True
    return False


def _future_planned_order_ids(state: Any) -> set[str]:
    mgr = _entity_mgr(state)
    result: set[str] = set()
    for truck in (_mapping(mgr, "trucks") or {}).values():
        planned_stops = list(_read_field(truck, "_planned_route_stops", []) or [])
        cursor = int(_read_field(truck, "_planned_route_cursor", 0) or 0)
        cursor = max(0, min(cursor, len(planned_stops)))
        for stop in planned_stops[cursor:]:
            if str(stop.get("node_type", "") or "") != "customer":
                continue
            order_id = str(stop.get("order_id", "") or "")
            if order_id:
                result.add(order_id)
    return result


def _unserved_dynamic_plan(planning_orders: dict[str, Any], reason: str) -> DispatchPlan:
    allocations = [
        AllocationResult(
            order_id=order_id,
            vehicle_id="",
            mode="UNSERVED",
            distance=0.0,
            feasible=False,
            reason=reason,
        )
        for order_id in planning_orders
    ]
    return DispatchPlan(
        allocations=allocations,
        cost_total=0.0,
        summary={
            "total_orders": len(planning_orders),
            "feasible": 0,
            "modes": {},
            "dispatch_type": "dynamic_replan",
            "solver": "ga_mmce",
            "dynamic_solver": "greedy_mmce_bi",
            "ga_dynamic_skipped": True,
            "ga_feasible": False,
        },
    )


def build_warm_start(
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual:
    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    support_node_ids = list(depot_ids) + list(station_ids)

    seq: list[str] = []
    assignment: list[str] = []
    rendezvous = []

    if previous_best is not None:
        for oid, gene, rv in zip(
            previous_best.sequence,
            previous_best.assignment,
            previous_best.rendezvous,
        ):
            if oid in excluded:
                continue
            seq.append(str(oid))
            assignment.append(str(gene))
            rendezvous.append(copy.deepcopy(rv))

    for oid in new_order_ids:
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        seq.append(str(oid))
        assignment.append(gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                gene,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def build_warm_start_population(
    previous_best: Individual | None,
    completed_ids: Iterable[str],
    locked_ids: Iterable[str],
    new_order_ids: Iterable[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    order_ids: list[str],
    reoptimized_order_ids: list[str],
    frozen_future_order_ids: list[str],
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] | None = None,
    orders: dict[str, Any] | None = None,
    allow_c_recover_station: bool = True,
    mutation_count: int = 0,
) -> list[Individual]:
    support_node_ids = list(depot_ids) + list(station_ids)
    excluded = {str(oid) for oid in completed_ids} | {str(oid) for oid in locked_ids}
    new_ids = [str(oid) for oid in new_order_ids if str(oid) in set(order_ids)]
    reopt_set = set(reoptimized_order_ids)
    fixed_tail_gene_by_order = fixed_tail_gene_by_order or {}
    population: list[Individual] = []

    base_order_ids = [oid for oid in order_ids if oid not in set(new_ids)]
    base = _previous_remaining_individual(
        previous_best,
        base_order_ids,
        excluded,
        gene_pool,
        support_node_ids,
        allow_c_recover_station,
    )
    if base is not None:
        population.append(_with_new_orders_appended(base, new_ids, gene_pool, depot_ids, station_ids, allow_c_recover_station))
        population.append(_with_new_orders_nearest(base, new_ids, gene_pool, depot_ids, station_ids, orders, allow_c_recover_station))

    for mode in ("A", "B", "C"):
        gene = _first_gene_for_mode(gene_pool, mode)
        if gene is not None:
            population.append(
                _new_orders_with_mode(
                    base,
                    order_ids,
                    new_ids,
                    gene,
                    depot_ids,
                    station_ids,
                    allow_c_recover_station,
                )
            )

    mutation_base = population[0] if population else _truck_only(order_ids)
    for _ in range(max(1, mutation_count)):
        mutated = copy.deepcopy(mutation_base)
        mutable_indices = [i for i, oid in enumerate(mutated.sequence) if oid in reopt_set]
        if len(mutable_indices) >= 2:
            i, j = random.sample(mutable_indices, 2)
            mutated.sequence[i], mutated.sequence[j] = mutated.sequence[j], mutated.sequence[i]
            mutated.assignment[i], mutated.assignment[j] = mutated.assignment[j], mutated.assignment[i]
            mutated.rendezvous[i], mutated.rendezvous[j] = mutated.rendezvous[j], mutated.rendezvous[i]
        try:
            mutate(
                mutated,
                gene_pool,
                support_node_ids,
                p_seq=0.0,
                p_assign=0.08,
                p_rendezvous=0.08,
                allow_c_recover_station=allow_c_recover_station,
            )
        except Exception:
            continue
        population.append(mutated)

    rv_perturbed = copy.deepcopy(mutation_base)
    for idx, gene in enumerate(rv_perturbed.assignment):
        if rv_perturbed.sequence[idx] not in reopt_set or gene == "A":
            continue
        rv_perturbed.rendezvous[idx] = make_random_rendezvous_for_gene(
            gene,
            depot_ids,
            station_ids,
            allow_c_recover_station,
        )
        break
    population.append(rv_perturbed)

    repaired: list[Individual] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    for ind in population:
        try:
            _repair_to_order_ids(
                ind,
                order_ids,
                gene_pool,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
            enforce_fixed_tail(ind, frozen_future_order_ids, fixed_tail_gene_by_order)
            key = (tuple(ind.sequence), tuple(ind.assignment), repr(ind.rendezvous))
            if key in seen:
                continue
            seen.add(key)
            ind.validate()
            repaired.append(ind)
        except Exception:
            continue
    return repaired


def _advance_or_snapshot_state(state: Any, event_time: float) -> Any:
    for target in (state, _entity_mgr(state)):
        if target is None:
            continue
        for name in ("snapshot_at", "advance_state_to", "apply_plan_until"):
            fn = getattr(target, name, None)
            if not callable(fn):
                continue
            try:
                advanced = fn(event_time)
            except TypeError:
                continue
            if advanced is not None:
                state = advanced
                break
    _write_field(state, "current_time", float(event_time))
    return state


def _build_dynamic_snapshot(
    state: Any,
    planning_orders: dict[str, Any],
    event_time: float,
    completed_ids: set[str],
    locked_ids: set[str],
) -> Any:
    base = SimpleNamespace(
        entity_mgr=_entity_mgr(state),
        orders=dict(planning_orders),
        current_time=float(event_time),
        bbox=_read_field(state, "bbox"),
        scene_id=_read_field(state, "scene_id"),
        _ga_context_mode="dynamic",
    )
    snapshot = clone_state_for_decode(base)
    _write_field(snapshot, "current_time", float(event_time))
    _write_field(snapshot, "orders", dict(planning_orders))
    _write_field(snapshot, "_ga_context_mode", "dynamic")
    _write_field(snapshot, "completed_order_ids", set(completed_ids))
    _write_field(snapshot, "locked_order_ids", set(locked_ids))
    _freeze_runtime_resources(snapshot, event_time, locked_ids)
    return snapshot


def _freeze_runtime_resources(snapshot: Any, event_time: float, locked_order_ids: set[str]) -> None:
    mgr = _entity_mgr(snapshot)
    locked_drone_ids: set[str] = set()
    busy_drone_ids: set[str] = set()

    for drone_id, drone in (_mapping(mgr, "drones") or {}).items():
        carrying = str(_read_field(drone, "carrying_order_id", "") or "")
        is_busy = bool(carrying) or _drone_is_flying(drone) or _has_pending_route(drone) or bool(_read_field(drone, "waiting_recovery_station_id"))
        if carrying in locked_order_ids or is_busy:
            busy_drone_ids.add(str(drone_id))
            locked_drone_ids.add(str(drone_id))
            available_time, available_pos = _estimate_drone_available(snapshot, drone, event_time)
            _write_field(drone, "_ga_available_time", available_time)
            _write_field(drone, "_ga_time", available_time)
            if available_pos is not None:
                _write_field(drone, "_ga_position", available_pos)
            host_type, host_node_id, truck_id = _future_drone_host(snapshot, drone, str(drone_id))
            if host_type and host_node_id:
                _write_field(drone, "_ga_force_available", True)
                _write_field(drone, "_ga_host_type", host_type)
                _write_field(drone, "_ga_host_node_id", host_node_id)
                _write_field(drone, "_ga_transport_truck_id", truck_id if host_type == "TRUCK" else None)
                _write_field(drone, "_ga_waiting_station_id", host_node_id if host_type == "STATION" else None)
                _add_drone_to_future_host(snapshot, str(drone_id), host_type, host_node_id, truck_id)

    for truck in (_mapping(mgr, "trucks") or {}).values():
        freeze_time, freeze_pos = _estimate_truck_available(truck, event_time, locked_order_ids)
        _write_field(truck, "_ga_time", freeze_time)
        if freeze_pos is not None:
            _write_field(truck, "_ga_position", freeze_pos)

    _write_field(snapshot, "locked_drone_ids", locked_drone_ids)
    _write_field(snapshot, "busy_drone_ids", busy_drone_ids)
    _write_field(snapshot, "running_drone_ids", locked_drone_ids)


def _future_drone_host(snapshot: Any, drone: Any, drone_id: str) -> tuple[str, str, str | None]:
    route_host = _pending_route_recovery_host(snapshot, drone)
    if route_host is not None:
        host_type, host_node_id = route_host
        return host_type, host_node_id, None

    truck_id = str(_read_field(drone, "_ga_transport_truck_id", _read_field(drone, "transport_truck_id", "")) or "")
    if truck_id:
        return "TRUCK", truck_id, truck_id

    mgr = _entity_mgr(snapshot)
    for tid, truck in (_mapping(mgr, "trucks") or {}).items():
        docked = list(_read_field(truck, "_ga_docked_drones", []) or [])
        docked.extend(_read_field(truck, "docked_drones", []) or [])
        if drone_id in docked:
            return "TRUCK", str(tid), str(tid)

    station_id = str(_read_field(drone, "_ga_waiting_station_id", _read_field(drone, "waiting_recovery_station_id", "")) or "")
    if station_id:
        return "STATION", station_id, None

    home_id = str(_read_field(drone, "home_id", "") or "")
    if home_id and home_id in (_mapping(mgr, "depots") or {}):
        return "DEPOT", home_id, None
    if home_id and home_id in (_mapping(mgr, "stations") or {}):
        return "STATION", home_id, None

    depot_ids = list((_mapping(mgr, "depots") or {}).keys())
    if depot_ids:
        return "DEPOT", str(depot_ids[0]), None
    return "", "", None


def _pending_route_recovery_host(snapshot: Any, drone: Any) -> tuple[str, str] | None:
    route_plan = list(_read_field(drone, "route_plan", []) or [])
    if not route_plan:
        return None

    start_idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    start_idx = max(0, min(start_idx, len(route_plan)))
    for wp in route_plan[start_idx:]:
        action_name = _waypoint_action_name(_read_field(wp, "action"))
        if action_name not in {"DOCK_TRUCK", "DOCK_DEPOT"}:
            continue
        node_id = str(_read_field(wp, "target_entity_id", "") or "")
        if not node_id:
            node_id = _nearest_support_node_id(snapshot, _read_field(wp, "loc"))
        host_type = _node_host_type(snapshot, node_id)
        if host_type in {"DEPOT", "STATION"}:
            return host_type, node_id
    return None


def _waypoint_action_name(action: Any) -> str:
    if action is None:
        return ""
    if hasattr(action, "value"):
        action = action.value
    return str(action).strip().upper()


def _node_host_type(snapshot: Any, node_id: str) -> str:
    if not node_id:
        return ""
    mgr = _entity_mgr(snapshot)
    if node_id in (_mapping(mgr, "stations") or {}):
        return "STATION"
    if node_id in (_mapping(mgr, "depots") or {}) or _is_depot_node_id(node_id):
        return "DEPOT"
    return ""


def _nearest_support_node_id(snapshot: Any, pos: Any) -> str:
    if pos is None:
        return ""
    mgr = _entity_mgr(snapshot)
    best_id = ""
    best_dist = math.inf
    for mapping_name in ("stations", "depots"):
        for node_id, node in (_mapping(mgr, mapping_name) or {}).items():
            dist = _distance(pos, _read_field(node, "location"))
            if dist < best_dist:
                best_dist = dist
                best_id = str(node_id)
    return best_id if best_dist <= 60.0 else ""


def _is_depot_node_id(node_id: str) -> bool:
    normalized = str(node_id).strip().upper()
    return normalized == "DEPOT" or normalized.startswith("DEPOT") or normalized.startswith("DEP-")


def _add_drone_to_future_host(snapshot: Any, drone_id: str, host_type: str, host_node_id: str, truck_id: str | None) -> None:
    mgr = _entity_mgr(snapshot)
    if host_type == "TRUCK" and truck_id:
        truck = (_mapping(mgr, "trucks") or {}).get(truck_id)
        if truck is not None:
            _append_unique_field(truck, "_ga_docked_drones", drone_id)
        return
    if host_type == "DEPOT":
        depot = (_mapping(mgr, "depots") or {}).get(host_node_id)
        if depot is not None:
            _append_unique_field(depot, "_ga_idle_drones", drone_id)
        return
    if host_type == "STATION":
        station = (_mapping(mgr, "stations") or {}).get(host_node_id)
        if station is not None:
            _append_unique_field(station, "_ga_waiting_drones", drone_id)


def _append_unique_field(record: Any, field_name: str, value: str) -> None:
    items = list(_read_field(record, field_name, []) or [])
    if value not in items:
        items.append(value)
    _write_field(record, field_name, items)


def _classify_orders(state: Any, new_orders: dict[str, Any], event_time: float) -> OrderBuckets:
    known = _all_known_orders(state)
    known.update(new_orders)
    new_ids = set(new_orders)
    completed: dict[str, Any] = {}
    locked: dict[str, Any] = {}
    pending: dict[str, Any] = {}
    new: dict[str, Any] = {}

    running_ids = _running_order_ids(state)
    for oid, order in known.items():
        status = _status_name(_read_field(order, "status"))
        if status in {"COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}:
            completed[oid] = order
            continue
        if oid in running_ids or status in {"PICKED_UP", "DELIVERING"}:
            locked[oid] = order
            continue
        if oid in new_ids:
            new[oid] = order
        else:
            pending[oid] = order

    return OrderBuckets(completed=completed, locked=locked, pending=pending, new=new)


def _select_reoptimization_window(
    previous_best: Individual | None,
    pending_orders: dict[str, Any],
    new_orders: dict[str, Any],
    config: GAConfig,
    event_time: float,
) -> tuple[list[str], list[str]]:
    new_ids = list(new_orders)
    pending_ids = list(pending_orders)
    previous_rank = {oid: i for i, oid in enumerate(previous_best.sequence)} if previous_best is not None else {}

    def sort_key(order_id: str) -> tuple[float, float, str]:
        order = pending_orders.get(order_id)
        rank = previous_rank.get(order_id, 10**9)
        deadline = _safe_float(_read_field(order, "deadline", math.inf), math.inf)
        return (rank, deadline - event_time, order_id)

    ordered_pending = sorted(pending_ids, key=sort_key)
    k = int(config.reopt_window_size or 0)
    if k <= 0:
        selected_pending = ordered_pending
    else:
        selected_pending = ordered_pending[:k]
    frozen_future = [oid for oid in ordered_pending if oid not in set(selected_pending)]
    reoptimized = list(dict.fromkeys(new_ids + selected_pending))
    return reoptimized, frozen_future


def _tail_gene_map(
    previous_best: Individual | None,
    frozen_future_order_ids: list[str],
    gene_pool: list[str],
) -> dict[str, tuple[str, dict[str, str] | None]]:
    allowed = set(gene_pool)
    result: dict[str, tuple[str, dict[str, str] | None]] = {}
    if previous_best is not None:
        for oid, gene, rv in zip(previous_best.sequence, previous_best.assignment, previous_best.rendezvous):
            if oid in frozen_future_order_ids and gene in allowed:
                result[oid] = (gene, copy.deepcopy(rv))
    for oid in frozen_future_order_ids:
        result.setdefault(oid, ("A", None))
    return result


def _best_feasible_warm_start_plan(
    solver: Any,
    snapshot: Any,
    warm_starts: list[Individual],
    config: GAConfig,
    planning_orders: dict[str, Any],
) -> DispatchPlan | None:
    if not warm_starts:
        return None
    previous_config = solver.config
    previous_evaluator_config = solver.evaluator.config
    previous_decoder_config = solver.decoder.config
    solver.config = config
    solver.evaluator.config = config
    solver.decoder.config = config
    try:
        context = build_ga_context(snapshot, config, mode="dynamic")
        best_plan = None
        best_fitness = math.inf
        for seed in warm_starts:
            candidate = copy.deepcopy(seed)
            try:
                enforce_fixed_tail(candidate, context.fixed_tail_order_ids, context.fixed_tail_gene_by_order)
                result = solver._evaluate_individual(candidate, snapshot, context)
            except Exception:
                continue
            plan = getattr(candidate, "decoded_plan", None)
            if result is None or plan is None or not _plan_is_feasible_for_orders(plan, planning_orders):
                continue
            fitness = float(getattr(candidate, "fitness", math.inf))
            if fitness < best_fitness:
                best_fitness = fitness
                best_plan = plan
        if best_plan is not None:
            best_plan.summary["fallback_used"] = True
            best_plan.summary["fallback_level"] = "warm_start_repaired"
        return best_plan
    finally:
        solver.config = previous_config
        solver.evaluator.config = previous_evaluator_config
        solver.decoder.config = previous_decoder_config


def _greedy_replan_fallback(
    solver: Any,
    snapshot: Any,
    planning_orders: dict[str, Any],
    event_time: float,
) -> DispatchPlan | None:
    bbox = _read_field(snapshot, "bbox")
    if not bbox:
        return None
    try:
        return solver.greedy_helper.dispatch_replan_current_state(
            planning_orders,
            float(event_time),
            bbox,
            scene_id=_read_field(snapshot, "scene_id"),
        )
    except Exception:
        return None


def _previous_plan_fallback(
    previous_plan: Any,
    planning_orders: dict[str, Any],
    new_order_ids: set[str],
    reason: str,
) -> DispatchPlan:
    allocations: list[AllocationResult] = []
    previous_allocs = {
        str(_read_field(alloc, "order_id", "")): copy.deepcopy(alloc)
        for alloc in getattr(previous_plan, "allocations", []) or []
    }
    for order_id in planning_orders:
        alloc = previous_allocs.get(order_id)
        if alloc is not None and order_id not in new_order_ids:
            allocations.append(alloc)
            continue
        allocations.append(
            AllocationResult(
                order_id=order_id,
                vehicle_id="",
                mode="UNSERVED",
                distance=0.0,
                feasible=False,
                reason=reason if order_id in new_order_ids else "previous_plan_missing",
            )
        )

    truck_routes = copy.deepcopy(getattr(previous_plan, "truck_routes", {}) or {})
    drone_routes = copy.deepcopy(getattr(previous_plan, "drone_routes", {}) or {})
    feasible_count = sum(1 for alloc in allocations if bool(_read_field(alloc, "feasible", False)))
    return DispatchPlan(
        allocations=allocations,
        cost_total=float(getattr(previous_plan, "cost_total", 0.0) or 0.0) if previous_plan is not None else 0.0,
        summary={
            "total_orders": len(planning_orders),
            "feasible": feasible_count,
            "modes": _mode_counts(allocations),
            "dispatch_type": "dynamic_replan",
            "solver": "ga_mmce",
            "ga_feasible": False,
            "new_orders_unserved": sorted(new_order_ids),
        },
        truck_routes=truck_routes,
        drone_routes=drone_routes,
    )


def _annotate_dynamic_summary(
    plan: DispatchPlan,
    event_time: float,
    buckets: OrderBuckets,
    reoptimized_ids: list[str],
    frozen_future_ids: list[str],
    warm_start_count: int,
    config: GAConfig,
    ga_plan: DispatchPlan,
    ga_feasible: bool,
    warm_start_feasible: bool,
    greedy_insert_feasible: bool,
    fallback_used: bool,
    fallback_level: str,
    unserved_order_ids: list[str],
    elapsed_seconds: float,
) -> None:
    modes = _mode_counts(plan.allocations)
    ga_summary = getattr(ga_plan, "summary", {}) or {}
    plan.summary.update(
        {
            "event_time": float(event_time),
            "new_order_ids": list(buckets.new),
            "completed_ids": list(buckets.completed),
            "locked_ids": list(buckets.locked),
            "pending_count": len(buckets.pending),
            "reoptimized_order_count": len(reoptimized_ids),
            "frozen_future_order_count": len(frozen_future_ids),
            "reoptimized_order_ids": list(reoptimized_ids),
            "frozen_future_order_ids": list(frozen_future_ids),
            "warm_start_count": int(warm_start_count),
            "population_size": int(config.population_size),
            "max_generations": int(config.generations),
            "actual_generations": int(ga_summary.get("actual_generations", 0) or 0),
            "elapsed_seconds": float(elapsed_seconds),
            "time_budget_hit": bool(ga_summary.get("time_budget_hit", False)),
            "early_stop_triggered": bool(ga_summary.get("early_stop_triggered", False)),
            "fallback_used": bool(fallback_used),
            "fallback_level": fallback_level,
            "ga_feasible": bool(ga_feasible),
            "warm_start_feasible": bool(warm_start_feasible),
            "greedy_insert_feasible": bool(greedy_insert_feasible),
            "unserved_order_ids": list(unserved_order_ids),
            "modes": modes,
            "final_A_count": int(modes.get("A", 0)),
            "final_B_count": int(modes.get("B", 0)),
            "final_C_count": int(modes.get("C", 0)),
            "best_fitness": float(plan.cost_total or ga_summary.get("best_fitness", 0.0) or 0.0),
            "dispatch_type": "dynamic_replan",
            "solver": "ga_mmce",
        }
    )


def _finalize_and_log(
    plan: DispatchPlan,
    started: float,
    event_time: float,
    buckets: OrderBuckets,
    reoptimized_ids: list[str],
    frozen_future_ids: list[str],
    warm_start_count: int,
    config: GAConfig,
    fallback_used: bool,
    fallback_level: str,
) -> None:
    _annotate_dynamic_summary(
        plan,
        event_time=event_time,
        buckets=buckets,
        reoptimized_ids=reoptimized_ids,
        frozen_future_ids=frozen_future_ids,
        warm_start_count=warm_start_count,
        config=config,
        ga_plan=plan,
        ga_feasible=True,
        warm_start_feasible=False,
        greedy_insert_feasible=False,
        fallback_used=fallback_used,
        fallback_level=fallback_level,
        unserved_order_ids=[],
        elapsed_seconds=time.time() - started,
    )
    _write_dynamic_replan_csv(plan.summary)
    _store_last_stats(plan.summary)


def _write_dynamic_replan_csv(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "logs" / "ga_dynamic_replan.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DYNAMIC_REPLAN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: _csv_value(summary.get(field, "")) for field in DYNAMIC_REPLAN_FIELDS})


def _store_last_stats(summary: dict[str, Any]) -> None:
    global LAST_DYNAMIC_REPLAN_STATS
    LAST_DYNAMIC_REPLAN_STATS = dict(summary)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return repr(value)
    return value


def _resolve_solver(state: Any) -> Any:
    solver = _read_field(state, "_ga_solver") or _read_field(state, "solver")
    if solver is not None:
        return solver
    from .solver import GAMMCESolver

    return GAMMCESolver(_entity_mgr(state), config=make_ga_config(DYNAMIC_GA_CONFIG))


def _all_known_orders(state: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    direct = _read_field(state, "orders")
    if isinstance(direct, dict):
        result.update({str(k): v for k, v in direct.items()})

    order_mgr = _order_mgr(state)
    if order_mgr is not None:
        for field_name in ("pending_orders", "assigned_orders"):
            value = _read_field(order_mgr, field_name, {}) or {}
            if isinstance(value, dict):
                result.update({str(k): v for k, v in value.items()})
        for order in _read_field(order_mgr, "completed_orders", []) or []:
            oid = _order_id(order)
            if oid:
                result[oid] = order
    return result


def _running_order_ids(state: Any) -> set[str]:
    mgr = _entity_mgr(state)
    running: set[str] = set()
    for drone in (_mapping(mgr, "drones") or {}).values():
        carrying = str(_read_field(drone, "carrying_order_id", "") or "")
        if carrying:
            running.add(carrying)
    for truck in (_mapping(mgr, "trucks") or {}).values():
        for stop in getattr(truck, "_planned_route_stops", []) or []:
            oid = str(stop.get("order_id", "") or "")
            if oid and float(stop.get("arrival_time", math.inf)) <= float(_read_field(state, "current_time", 0.0) or 0.0):
                running.add(oid)
    return running


def _normalize_order_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    result = {}
    for order in value:
        oid = _order_id(order)
        if oid:
            result[oid] = order
    return result


def _ordered_order_mapping(order_ids: list[str], source: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for order_id in order_ids:
        if order_id in source:
            result[order_id] = source[order_id]
    return result


def _previous_remaining_individual(
    previous_best: Individual | None,
    order_ids: list[str],
    excluded: set[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual | None:
    order_set = set(order_ids)
    gene_set = set(gene_pool)
    seq: list[str] = []
    assignment: list[str] = []
    rendezvous = []
    if previous_best is not None:
        for oid, gene, rv in zip(previous_best.sequence, previous_best.assignment, previous_best.rendezvous):
            if oid in excluded or oid not in order_set:
                continue
            seq.append(oid)
            if gene in gene_set:
                assignment.append(gene)
                rendezvous.append(copy.deepcopy(rv))
            else:
                assignment.append("A")
                rendezvous.append(None)

    for oid in order_ids:
        if oid in seq:
            continue
        gene = _preferred_gene(gene_pool, ("A",))
        seq.append(oid)
        assignment.append(gene)
        rendezvous.append(make_random_rendezvous_for_gene(gene, support_node_ids, allow_c_recover_station))
    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def _with_new_orders_appended(
    base: Individual,
    new_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base)
    existing = set(ind.sequence)
    for oid in new_ids:
        if oid in existing:
            continue
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        ind.sequence.append(oid)
        ind.assignment.append(gene)
        ind.rendezvous.append(make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station))
    return ind


def _with_new_orders_nearest(
    base: Individual,
    new_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    orders: dict[str, Any] | None,
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base)
    orders = orders or {}
    for oid in new_ids:
        if oid in ind.sequence:
            continue
        insert_at = len(ind.sequence)
        nearest_idx = _nearest_order_index(oid, ind.sequence, orders)
        if nearest_idx is not None:
            insert_at = nearest_idx + 1
        gene = _preferred_gene(gene_pool, ("C", "B", "A"))
        ind.sequence.insert(insert_at, oid)
        ind.assignment.insert(insert_at, gene)
        ind.rendezvous.insert(insert_at, make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station))
    return ind


def _new_orders_with_mode(
    base: Individual | None,
    order_ids: list[str],
    new_ids: list[str],
    gene: str,
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> Individual:
    ind = copy.deepcopy(base) if base is not None else _truck_only(order_ids)
    by_order = {
        oid: (g, copy.deepcopy(rv))
        for oid, g, rv in zip(ind.sequence, ind.assignment, ind.rendezvous)
    }
    for oid in new_ids:
        by_order[oid] = (
            gene,
            make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station),
        )
    sequence = [oid for oid in order_ids if oid in by_order]
    assignment = [by_order[oid][0] for oid in sequence]
    rendezvous = [copy.deepcopy(by_order[oid][1]) for oid in sequence]
    return Individual(sequence, assignment, rendezvous)


def _repair_to_order_ids(
    ind: Individual,
    order_ids: list[str],
    gene_pool: list[str],
    depot_ids: list[str],
    station_ids: list[str],
    allow_c_recover_station: bool,
) -> None:
    order_set = set(order_ids)
    by_order: dict[str, tuple[str, Any]] = {}
    for oid, gene, rv in zip(ind.sequence, ind.assignment, ind.rendezvous):
        if oid in order_set and oid not in by_order:
            if gene not in gene_pool:
                gene = "A"
                rv = None
            by_order[oid] = (gene, copy.deepcopy(rv))
    sequence: list[str] = []
    seen: set[str] = set()
    for oid in ind.sequence:
        if oid in order_set and oid not in seen:
            sequence.append(oid)
            seen.add(oid)
    for oid in order_ids:
        if oid in seen:
            continue
        sequence.append(oid)
        seen.add(oid)
        if oid in by_order:
            continue
        gene = _preferred_gene(gene_pool, ("A",))
        by_order[oid] = (
            gene,
            make_random_rendezvous_for_gene(gene, depot_ids, station_ids, allow_c_recover_station),
        )
    ind.sequence = sequence
    ind.assignment = [by_order[oid][0] for oid in ind.sequence]
    ind.rendezvous = [copy.deepcopy(by_order[oid][1]) for oid in ind.sequence]
    ind.validate()


def _truck_only(order_ids: list[str]) -> Individual:
    return Individual(list(order_ids), ["A"] * len(order_ids), [None] * len(order_ids))


def _preferred_gene(gene_pool: list[str], modes: Iterable[str]) -> str:
    for mode in modes:
        if mode == "A" and "A" in gene_pool:
            return "A"
        prefix = f"{mode}_"
        for gene in gene_pool:
            if gene.startswith(prefix):
                return gene
    return "A"


def _first_gene_for_mode(gene_pool: list[str], mode: str) -> str | None:
    if mode == "A":
        return "A" if "A" in gene_pool else None
    prefix = f"{mode}_"
    for gene in gene_pool:
        if gene.startswith(prefix):
            return gene
    return None


def _nearest_order_index(order_id: str, sequence: list[str], orders: dict[str, Any]) -> int | None:
    target = _read_field(orders.get(order_id), "delivery_loc")
    if target is None:
        return None
    best_idx = None
    best_dist = math.inf
    for idx, oid in enumerate(sequence):
        pos = _read_field(orders.get(oid), "delivery_loc")
        dist = _distance(target, pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _plan_is_feasible_for_orders(plan: Any, planning_orders: dict[str, Any]) -> bool:
    if plan is None:
        return False
    required = set(planning_orders)
    feasible_allocs = {
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if bool(_read_field(alloc, "feasible", False))
    }
    return required <= feasible_allocs


def _unserved_order_ids(plan: Any, planning_orders: dict[str, Any]) -> list[str]:
    required = set(planning_orders)
    feasible_allocs = {
        str(_read_field(alloc, "order_id", ""))
        for alloc in getattr(plan, "allocations", []) or []
        if bool(_read_field(alloc, "feasible", False))
    }
    return sorted(required - feasible_allocs)


def _mode_counts(allocations: Iterable[Any]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0}
    for alloc in allocations:
        mode = str(_read_field(alloc, "mode", "") or "")
        if mode.startswith("B"):
            counts["B"] += 1
        elif mode == "C":
            counts["C"] += 1
        elif mode == "A":
            counts["A"] += 1
    return counts


def _estimate_truck_available(truck: Any, event_time: float, locked_order_ids: set[str]) -> tuple[float, Any]:
    current_pos = truck.get_location(event_time) if hasattr(truck, "get_location") else _read_field(truck, "current_loc")
    freeze_time = float(event_time)
    freeze_pos = current_pos
    for stop in getattr(truck, "_planned_route_stops", []) or []:
        oid = str(stop.get("order_id", "") or "")
        arrival = _safe_float(stop.get("arrival_time"), math.inf)
        departure = _safe_float(stop.get("departure_time"), arrival)
        if arrival <= event_time < departure:
            freeze_time = max(freeze_time, departure)
            freeze_pos = stop.get("position", freeze_pos)
        if oid in locked_order_ids and departure >= event_time:
            freeze_time = max(freeze_time, departure)
            freeze_pos = stop.get("position", freeze_pos)
    return freeze_time, freeze_pos


def _estimate_drone_available(snapshot: Any, drone: Any, event_time: float) -> tuple[float, Any]:
    route_plan = _read_field(drone, "route_plan", []) or []
    idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    cur = _read_field(drone, "current_loc")
    if idx >= len(route_plan):
        return float(event_time), cur
    dist = 0.0
    service_time = _safe_float(_read_field(_entity_mgr(snapshot), "DRONE_SERVICE_TIME_ORDER", 0.0), 0.0)
    service_total = 0.0
    for wp in route_plan[idx:]:
        loc = _read_field(wp, "loc")
        dist += _distance(cur, loc)
        cur = loc
        if _waypoint_action_name(_read_field(wp, "action")) == "DELIVER":
            service_total += service_time
    speed = max(1e-6, _safe_float(_read_field(drone, "cruise_speed", 0.0), 0.0))
    return float(event_time) + dist / speed + service_total, cur


def _distance(pos_a: Any, pos_b: Any) -> float:
    if pos_a is None or pos_b is None:
        return math.inf
    if hasattr(pos_a, "distance_2d"):
        try:
            return float(pos_a.distance_2d(pos_b))
        except Exception:
            return math.inf
    if all(hasattr(pos, "x") and hasattr(pos, "y") for pos in (pos_a, pos_b)):
        dx = float(pos_a.x) - float(pos_b.x)
        dy = float(pos_a.y) - float(pos_b.y)
        return (dx * dx + dy * dy) ** 0.5
    return math.inf


def _drone_is_flying(drone: Any) -> bool:
    status = _read_field(drone, "status")
    is_flying = getattr(status, "is_flying", None)
    if is_flying is not None:
        return bool(is_flying)
    return _status_name(status) in {"FLYING_TO_PICKUP", "FLYING_TO_DELIVER", "FLYING_TO_STATION", "FLYING_TO_TRUCK", "RETURNING_TO_DEPOT"}


def _has_pending_route(drone: Any) -> bool:
    value = _read_field(drone, "has_pending_route")
    if value is not None:
        return bool(value)
    route_plan = _read_field(drone, "route_plan", []) or []
    idx = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    return idx < len(route_plan)


def _entity_mgr(state: Any) -> Any:
    return _read_field(state, "entity_mgr") or _read_field(state, "entity_manager") or state


def _order_mgr(state: Any) -> Any:
    return _read_field(state, "order_mgr") or _read_field(state, "order_manager") or _read_field(_entity_mgr(state), "order_mgr")


def _mapping(state_or_mgr: Any, field_name: str) -> dict[str, Any]:
    value = _read_field(state_or_mgr, field_name)
    return value if isinstance(value, dict) else {}


def _order_id(order: Any) -> str:
    return str(_read_field(order, "order_id", "") or "").strip()


def _status_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().upper()


def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _write_field(record: Any, field_name: str, value: Any) -> None:
    if isinstance(record, dict):
        record[field_name] = value
    else:
        setattr(record, field_name, value)


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default
