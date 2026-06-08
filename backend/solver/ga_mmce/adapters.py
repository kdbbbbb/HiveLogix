from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable

from .chromosome import Individual, make_gene_pool_by_location, make_node_pool
from .operators import find_depot_node, make_random_rendezvous_for_gene


_DEPOT_LAUNCH_TOLERANCE_M = 30.0


@dataclass
class GAAdapterContext:
    order_ids: list[str]
    truck_drone_ids: list[str]
    depot_drone_ids: list[str]
    all_drone_ids: list[str]
    truck_ids: list[str]
    depot_ids: list[str]
    station_ids: list[str]
    support_node_ids: list[str]
    gene_pool: list[str]
    mode: str = "static"
    fixed_tail_order_ids: list[str] = field(default_factory=list)
    fixed_tail_gene_by_order: dict[str, tuple[str, dict[str, str] | None]] = field(default_factory=dict)


def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _write_field(record: Any, field_name: str, value: Any) -> None:
    if isinstance(record, dict):
        record[field_name] = value
    else:
        setattr(record, field_name, value)


def _dedupe_preserve_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _status_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().upper()


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _as_keys_or_ids(value: Any, id_field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    return [str(_read_field(item, id_field, "")) for item in value if _read_field(item, id_field, "")]


def _current_time(state: Any) -> float:
    return float(_read_field(state, "current_time", 0.0) or 0.0)


def _entity_mgr(state: Any) -> Any:
    return (
        _read_field(state, "entity_mgr")
        or _read_field(state, "entity_manager")
        or state
    )


def _mapping(state: Any, field_name: str) -> Any:
    mgr = _entity_mgr(state)
    return _read_field(mgr, field_name)


def _order_sources(state: Any) -> list[Any]:
    sources: list[Any] = []

    for field_name in ("orders", "pending_orders"):
        value = _read_field(state, field_name)
        if value is not None:
            sources.extend(_as_values(value))

    for manager_name in ("order_mgr", "order_manager"):
        manager = _read_field(state, manager_name)
        if manager is None:
            continue
        for field_name in ("pending_orders", "assigned_orders", "orders"):
            value = _read_field(manager, field_name)
            if value is not None:
                sources.extend(_as_values(value))

    if sources:
        return sources

    # Some tests pass the EntityManager directly and store pending orders at depots.
    for depot in _as_values(_mapping(state, "depots")):
        sources.extend(_as_values(_read_field(depot, "pending_orders")))
    return sources


def _is_locked_order(order: Any) -> bool:
    for field_name in ("locked", "is_locked", "locked_action", "running_action"):
        value = _read_field(order, field_name)
        if bool(value):
            return True
    return False


def _is_active_order(order: Any) -> bool:
    if _is_locked_order(order):
        return False

    status = _status_name(_read_field(order, "status"))
    if status in {"COMPLETED", "REJECTED", "CANCELLED", "CANCELED"}:
        return False
    return bool(_read_field(order, "order_id"))


def extract_active_order_ids(state) -> list[str]:
    """Return order IDs that may still be optimized by GA."""
    seen: set[str] = set()
    order_ids: list[str] = []
    for order in _order_sources(state):
        order_id = str(_read_field(order, "order_id", "")).strip()
        if not order_id or order_id in seen or not _is_active_order(order):
            continue
        seen.add(order_id)
        order_ids.append(order_id)
    return order_ids


def _locked_drone_ids(state: Any) -> set[str]:
    locked: set[str] = set()
    for field_name in ("locked_drone_ids", "locked_drones", "running_drone_ids", "busy_drone_ids"):
        value = _read_field(state, field_name)
        if value is not None:
            locked.update(str(item) for item in value)

    for action in _as_values(_read_field(state, "locked_actions")):
        drone_id = _read_field(action, "drone_id")
        if drone_id:
            locked.add(str(drone_id))
    return locked


def _drone_is_idle(drone: Any) -> bool:
    status = _read_field(drone, "status")
    if hasattr(status, "is_dispatchable"):
        return bool(status.is_dispatchable)
    return _status_name(status) == "IDLE"


def _has_pending_route(drone: Any) -> bool:
    value = _read_field(drone, "has_pending_route")
    if value is not None:
        return bool(value)
    route_plan = _read_field(drone, "route_plan", [])
    waypoint_index = int(_read_field(drone, "current_waypoint_index", 0) or 0)
    return waypoint_index < len(route_plan)


def _has_enough_standby_energy(drone: Any) -> bool:
    battery_current = _read_field(drone, "battery_current")
    if battery_current is None:
        return True
    safe_margin = float(_read_field(drone, "safe_margin_j", 0.0) or 0.0)
    return float(battery_current) > safe_margin


def _drone_is_available(drone: Any, locked_ids: set[str], current_time: float | None = None) -> bool:
    drone_id = str(_read_field(drone, "drone_id", "")).strip()
    force_available = bool(_read_field(drone, "_ga_force_available", False))
    if not drone_id or (drone_id in locked_ids and not force_available):
        return False
    available_time = _read_field(drone, "_ga_available_time", _read_field(drone, "available_time"))
    if available_time is not None and current_time is not None and not force_available:
        try:
            if float(available_time) > float(current_time) + 1e-6:
                return False
        except (TypeError, ValueError):
            pass
    if force_available:
        return _has_enough_standby_energy(drone)
    if not _drone_is_idle(drone):
        return False
    if not _has_enough_standby_energy(drone):
        return False
    if _read_field(drone, "carrying_order_id"):
        return False
    if _read_field(drone, "waiting_recovery_station_id"):
        return False
    if _has_pending_route(drone):
        return False
    return True


def _truck_docked_drone_ids(state: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for truck in _as_values(_mapping(state, "trucks")):
        combined = list(_read_field(truck, "_ga_docked_drones", []) or [])
        combined.extend(_read_field(truck, "docked_drones", []) or [])
        for drone_id in combined:
            normalized = str(drone_id).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result


def extract_truck_drone_ids(state) -> list[str]:
    """Return currently truck-docked drones usable for B-mode genes."""
    drones = _mapping(state, "drones") or {}
    locked_ids = _locked_drone_ids(state)
    now = _current_time(state)
    result: list[str] = []
    for drone_id in _truck_docked_drone_ids(state):
        drone = drones.get(drone_id) if isinstance(drones, dict) else None
        if drone is None or not _drone_is_available(drone, locked_ids, now):
            continue
        result.append(drone_id)
    return result


def _distance_2d(pos_a: Any, pos_b: Any) -> float | None:
    if pos_a is None or pos_b is None:
        return None
    if hasattr(pos_a, "distance_2d"):
        return float(pos_a.distance_2d(pos_b))
    if all(hasattr(pos, "x") and hasattr(pos, "y") for pos in (pos_a, pos_b)):
        dx = float(pos_a.x) - float(pos_b.x)
        dy = float(pos_a.y) - float(pos_b.y)
        return (dx * dx + dy * dy) ** 0.5
    return None


def _is_at_depot(drone: Any, depot: Any) -> bool:
    drone_pos = _read_field(drone, "_ga_position") or _read_field(drone, "current_loc")
    distance = _distance_2d(drone_pos, _read_field(depot, "location"))
    if distance is None:
        return True
    return distance <= _DEPOT_LAUNCH_TOLERANCE_M


def extract_depot_drone_ids(state) -> list[str]:
    """Return currently depot-ready drones usable for C-mode genes."""
    drones = _mapping(state, "drones") or {}
    truck_docked = set(_truck_docked_drone_ids(state))
    locked_ids = _locked_drone_ids(state)
    now = _current_time(state)
    seen: set[str] = set()
    result: list[str] = []

    independent_nodes = list(_as_values(_mapping(state, "depots")))
    for node in independent_nodes:
        idle_drones = list(_read_field(node, "_ga_idle_drones", []) or [])
        idle_drones.extend(_read_field(node, "idle_drones", []) or [])
        for drone_id in idle_drones:
            normalized = str(drone_id).strip()
            if not normalized or normalized in seen or normalized in truck_docked:
                continue
            drone = drones.get(normalized) if isinstance(drones, dict) else None
            if drone is None or not _drone_is_available(drone, locked_ids, now):
                continue
            if _read_field(drone, "transport_truck_id"):
                continue
            if not _is_at_depot(drone, node):
                continue
            seen.add(normalized)
            result.append(normalized)

    return result


def extract_all_drone_ids(state) -> list[str]:
    """Return all drone IDs in the current entity manager."""
    drones = _mapping(state, "drones")
    if isinstance(drones, dict):
        return [str(key) for key in drones.keys()]
    return [str(_read_field(drone, "drone_id", "")) for drone in _as_values(drones) if _read_field(drone, "drone_id", "")]


def extract_truck_ids(state) -> list[str]:
    return _as_keys_or_ids(_mapping(state, "trucks"), "truck_id")


def extract_depot_ids(state) -> list[str]:
    return _as_keys_or_ids(_mapping(state, "depots"), "depot_id")


def extract_station_ids(state) -> list[str]:
    return _as_keys_or_ids(_mapping(state, "stations"), "station_id")


def extract_support_node_ids(state) -> list[str]:
    return make_node_pool(extract_depot_ids(state), extract_station_ids(state))


def _initial_layout_is_enabled(state: Any, config: Any | None) -> bool:
    if config is None:
        return False
    if not bool(_read_field(config, "initial_drone_layout_enabled", False)):
        return False
    max_time = float(_read_field(config, "initial_drone_layout_max_time_s", 0.0) or 0.0)
    return _current_time(state) <= max_time


def _configured_initial_layout_ids(
    state: Any,
    config: Any | None,
) -> tuple[list[str], list[str]] | None:
    if not _initial_layout_is_enabled(state, config):
        return None

    drones = _mapping(state, "drones") or {}
    if not isinstance(drones, dict):
        return None

    locked_ids = _locked_drone_ids(state)
    truck_ids = _dedupe_preserve_order(_read_field(config, "initial_truck_drone_ids", ()))
    depot_ids = _dedupe_preserve_order(_read_field(config, "initial_depot_drone_ids", ()))

    truck_result: list[str] = []
    depot_result: list[str] = []
    truck_set: set[str] = set()

    for drone_id in truck_ids:
        drone = drones.get(drone_id)
        if drone is None or not _drone_is_available(drone, locked_ids, _current_time(state)):
            continue
        truck_result.append(drone_id)
        truck_set.add(drone_id)

    for drone_id in depot_ids:
        if drone_id in truck_set:
            continue
        drone = drones.get(drone_id)
        if drone is None or not _drone_is_available(drone, locked_ids, _current_time(state)):
            continue
        depot_result.append(drone_id)

    return truck_result, depot_result


def _record_position(record: Any, state: Any) -> Any:
    ga_position = _read_field(record, "_ga_position")
    if ga_position is not None:
        return ga_position
    if hasattr(record, "get_location"):
        try:
            return record.get_location(_current_time(state))
        except Exception:
            pass
    return _read_field(record, "current_loc") or _read_field(record, "location")


def _write_ga_list(record: Any, field_name: str, values: Iterable[str]) -> None:
    _write_field(record, field_name, _dedupe_preserve_order(values))


def apply_initial_drone_layout_overlay(
    state: Any,
    context: GAAdapterContext,
    config: Any | None,
) -> bool:
    """Apply the configured initial drone layout to a GA-owned state copy."""
    if not _initial_layout_is_enabled(state, config):
        return False
    if not context.truck_ids or not context.depot_ids:
        return False

    trucks = _mapping(state, "trucks") or {}
    depots = _mapping(state, "depots") or {}
    drones = _mapping(state, "drones") or {}
    if not isinstance(trucks, dict) or not isinstance(depots, dict) or not isinstance(drones, dict):
        return False

    truck_id = context.truck_ids[0]
    depot_id = context.depot_ids[0]
    truck = trucks.get(truck_id)
    depot = depots.get(depot_id)
    if truck is None or depot is None:
        return False

    truck_position = _record_position(truck, state) or _record_position(depot, state)
    depot_position = _record_position(depot, state)
    now = _current_time(state)

    for item in trucks.values():
        _write_ga_list(item, "_ga_docked_drones", [])
    for item in depots.values():
        _write_ga_list(item, "_ga_idle_drones", [])
    for station in _as_values(_mapping(state, "stations")):
        _write_ga_list(station, "_ga_idle_drones", [])
        _write_ga_list(station, "_ga_waiting_drones", [])

    for drone_id in context.truck_drone_ids:
        drone = drones.get(drone_id)
        if drone is None:
            continue
        _write_field(drone, "_ga_host_type", "TRUCK")
        _write_field(drone, "_ga_host_node_id", depot_id)
        _write_field(drone, "_ga_transport_truck_id", truck_id)
        _write_field(drone, "_ga_waiting_station_id", None)
        _write_field(drone, "_ga_position", truck_position)
        _write_field(drone, "_ga_node_id", depot_id)
        _write_field(drone, "_ga_time", now)
        _write_field(drone, "_ga_energy", _read_field(drone, "battery_current", 0.0))

    for drone_id in context.depot_drone_ids:
        drone = drones.get(drone_id)
        if drone is None:
            continue
        _write_field(drone, "_ga_host_type", "DEPOT")
        _write_field(drone, "_ga_host_node_id", depot_id)
        _write_field(drone, "_ga_transport_truck_id", None)
        _write_field(drone, "_ga_waiting_station_id", None)
        _write_field(drone, "_ga_position", depot_position)
        _write_field(drone, "_ga_node_id", depot_id)
        _write_field(drone, "_ga_time", now)
        _write_field(drone, "_ga_energy", _read_field(drone, "battery_current", 0.0))

    _write_ga_list(truck, "_ga_docked_drones", context.truck_drone_ids)
    _write_ga_list(depot, "_ga_idle_drones", context.depot_drone_ids)
    return True


def _context_mode(state: Any, mode: str | None) -> str:
    raw = mode or _read_field(state, "_ga_context_mode", "static")
    normalized = str(raw or "static").strip().lower()
    return "dynamic" if normalized == "dynamic" else "static"


def _fixed_tail_from_state(state: Any) -> tuple[list[str], dict[str, tuple[str, dict[str, str] | None]]]:
    raw_tail = _read_field(state, "_ga_fixed_tail_order_ids", []) or []
    tail_ids = _dedupe_preserve_order(raw_tail)
    raw_gene_map = _read_field(state, "_ga_fixed_tail_gene_by_order", {}) or {}
    gene_map: dict[str, tuple[str, dict[str, str] | None]] = {}
    if isinstance(raw_gene_map, dict):
        for order_id, value in raw_gene_map.items():
            oid = str(order_id)
            if isinstance(value, tuple) and len(value) == 2:
                gene, rv = value
            elif isinstance(value, dict):
                gene = value.get("gene", "A")
                rv = value.get("rendezvous")
            else:
                continue
            gene_map[oid] = (str(gene), copy.deepcopy(rv))
    return tail_ids, gene_map


def make_gene_pool_for_state(
    state: Any,
    config: Any | None = None,
    mode: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    resolved_mode = _context_mode(state, mode)
    configured_layout = (
        _configured_initial_layout_ids(state, config)
        if resolved_mode == "static"
        else None
    )
    if configured_layout is None:
        truck_drone_ids = extract_truck_drone_ids(state)
        depot_drone_ids = extract_depot_drone_ids(state)
    else:
        truck_drone_ids, depot_drone_ids = configured_layout
    return truck_drone_ids, depot_drone_ids, make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)


def build_ga_context(
    state,
    config: Any | None = None,
    mode: str | None = None,
) -> GAAdapterContext:
    order_ids = extract_active_order_ids(state)
    resolved_mode = _context_mode(state, mode)
    truck_drone_ids, depot_drone_ids, gene_pool = make_gene_pool_for_state(
        state,
        config,
        resolved_mode,
    )
    all_drone_ids = extract_all_drone_ids(state)
    truck_ids = extract_truck_ids(state)
    depot_ids = extract_depot_ids(state)
    station_ids = extract_station_ids(state)
    support_node_ids = make_node_pool(depot_ids, station_ids)
    fixed_tail_order_ids, fixed_tail_gene_by_order = _fixed_tail_from_state(state)

    return GAAdapterContext(
        order_ids=order_ids,
        truck_drone_ids=truck_drone_ids,
        depot_drone_ids=depot_drone_ids,
        all_drone_ids=all_drone_ids,
        truck_ids=truck_ids,
        depot_ids=depot_ids,
        station_ids=station_ids,
        support_node_ids=support_node_ids,
        gene_pool=gene_pool,
        mode=resolved_mode,
        fixed_tail_order_ids=[
            order_id for order_id in fixed_tail_order_ids if order_id in set(order_ids)
        ],
        fixed_tail_gene_by_order=fixed_tail_gene_by_order,
    )


def clone_state_for_decode(state):
    """Clone only the mutable planning surface needed by GA decoding.

    Runtime entities contain locks (for example Truck._lock), so a full
    deepcopy can fail with "cannot pickle _thread.RLock". GA only writes _ga_*
    temporary fields and a small set of host lists/queues, so shallow-cloning
    entities plus copying mutable containers is enough and keeps the real
    runtime state clean.
    """
    state_copy = _clone_record_surface(state)

    mgr = _entity_mgr(state)
    mgr_copy = _clone_record_surface(mgr)

    for field_name in ("depots", "stations", "trucks", "drones"):
        mapping = _read_field(mgr, field_name, {}) or {}
        if isinstance(mapping, dict):
            setattr(mgr_copy, field_name, {
                key: _clone_entity_surface(value)
                for key, value in mapping.items()
            })

    direct_orders = _read_field(state, "orders")
    if isinstance(direct_orders, dict):
        cloned_orders = {
            key: _clone_entity_surface(value)
            for key, value in direct_orders.items()
        }
        setattr(state_copy, "orders", cloned_orders)
    else:
        orders = _read_field(mgr, "orders")
        if isinstance(orders, dict):
            setattr(mgr_copy, "orders", {
                key: _clone_entity_surface(value)
                for key, value in orders.items()
            })

    if _read_field(state, "entity_mgr") is not None:
        setattr(state_copy, "entity_mgr", mgr_copy)
    elif _read_field(state, "entity_manager") is not None:
        setattr(state_copy, "entity_manager", mgr_copy)

    return state_copy


def _clone_record_surface(record: Any) -> Any:
    if isinstance(record, dict):
        return dict(record)
    try:
        return copy.copy(record)
    except Exception:
        return record


def _clone_entity_surface(entity: Any) -> Any:
    if isinstance(entity, dict):
        return _clone_mapping_surface(entity)

    if hasattr(entity, "__slots__") and not hasattr(entity, "__dict__"):
        return _clone_slot_entity_surface(entity)

    try:
        cloned = copy.copy(entity)
    except Exception:
        return entity

    for name, value in getattr(entity, "__dict__", {}).items():
        if name.startswith("_ga_"):
            continue
        if isinstance(value, dict):
            setattr(cloned, name, _clone_mapping_surface(value))
        elif isinstance(value, list):
            setattr(cloned, name, list(value))
        elif isinstance(value, set):
            setattr(cloned, name, set(value))
        elif isinstance(value, tuple):
            setattr(cloned, name, tuple(value))

    return cloned


def _clone_slot_entity_surface(entity: Any) -> Any:
    data: dict[str, Any] = {}
    for cls in type(entity).mro():
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name.startswith("__"):
                continue
            try:
                value = getattr(entity, name)
            except AttributeError:
                continue
            if isinstance(value, dict):
                data[name] = _clone_mapping_surface(value)
            elif isinstance(value, list):
                data[name] = list(value)
            elif isinstance(value, set):
                data[name] = set(value)
            else:
                data[name] = value

    # Preserve common read-only properties that GA validators/evaluators read.
    for property_name in ("status",):
        if property_name not in data and hasattr(entity, property_name):
            data[property_name] = getattr(entity, property_name)

    return SimpleNamespace(**data)


def _clone_mapping_surface(mapping: dict) -> dict:
    cloned: dict = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, set):
            cloned[key] = set(value)
        else:
            cloned[key] = value
    return cloned


def _iter_allocations(greedy_plan) -> Iterable[Any]:
    if greedy_plan is None:
        return []
    if isinstance(greedy_plan, dict):
        return greedy_plan.values()
    if isinstance(greedy_plan, list):
        return greedy_plan
    for field_name in ("allocations", "results", "assignments"):
        value = _read_field(greedy_plan, field_name)
        if value is not None:
            return value.values() if isinstance(value, dict) else value
    return []


def _repair_rendezvous_if_needed(
    gene: str,
    rv: dict[str, str] | None,
    support_node_ids: list[str],
    allow_c_recover_station: bool,
) -> dict[str, str] | None:
    if gene == "A":
        return None
    if not isinstance(rv, dict):
        return make_random_rendezvous_for_gene(
            gene,
            support_node_ids,
            allow_c_recover_station,
        )

    support_set = {str(node) for node in support_node_ids}
    if rv.get("launch") not in support_set or rv.get("recover") not in support_set:
        return make_random_rendezvous_for_gene(
            gene,
            support_node_ids,
            allow_c_recover_station,
        )
    return rv


def _gene_context_from_pool(gene_pool: list[str]) -> tuple[list[str], list[str], list[str]]:
    truck_drone_ids: list[str] = []
    depot_drone_ids: list[str] = []
    for gene in gene_pool:
        if gene.startswith("B_"):
            truck_drone_ids.append(gene.split("_", 1)[1])
        elif gene.startswith("C_"):
            depot_drone_ids.append(gene.split("_", 1)[1])

    all_drone_ids = sorted(set(truck_drone_ids) | set(depot_drone_ids))
    return truck_drone_ids, depot_drone_ids, all_drone_ids


def greedy_plan_to_individual(
    greedy_plan,
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual | None:
    if not order_ids:
        return Individual(sequence=[], assignment=[], rendezvous=[])
    if not gene_pool or "A" not in gene_pool:
        raise ValueError('gene_pool must include "A"')
    if not support_node_ids:
        raise ValueError("support_node_ids must not be empty")

    allocations_by_order: dict[str, Any] = {}
    for alloc in _iter_allocations(greedy_plan):
        order_id = str(_read_field(alloc, "order_id", "")).strip()
        if order_id:
            allocations_by_order[order_id] = alloc

    gene_set = set(gene_pool)
    sequence = list(order_ids)
    assignment: list[str] = []
    rendezvous = []

    for order_id in sequence:
        alloc = allocations_by_order.get(order_id)
        mode = _status_name(_read_field(alloc, "mode")) if alloc is not None else ""
        drone_id = str(_read_field(alloc, "drone_id", "") or "").strip() if alloc is not None else ""

        if mode == "A" or not drone_id:
            gene = "A"
            rv = None
        elif mode in {"B", "B_WAIT", "B_DYNAMIC"}:
            gene = f"B_{drone_id}"
            if gene not in gene_set:
                gene = "A"
                rv = None
            else:
                launch = (
                    _read_field(alloc, "launch_station_id")
                    or _read_field(alloc, "launch_node_id")
                    or _read_field(alloc, "recovery_station_id")
                    or support_node_ids[0]
                )
                recover = (
                    _read_field(alloc, "recovery_station_id")
                    or _read_field(alloc, "recover_node_id")
                    or launch
                )
                rv = {"launch": str(launch), "recover": str(recover)}
        elif mode == "C":
            gene = f"C_{drone_id}"
            if gene not in gene_set:
                gene = "A"
                rv = None
            else:
                depot_node = find_depot_node(support_node_ids)
                recover = (
                    _read_field(alloc, "recovery_station_id")
                    or _read_field(alloc, "recover_node_id")
                    or depot_node
                )
                rv = {"launch": str(depot_node), "recover": str(recover)}
        else:
            gene = "A"
            rv = None

        rv = _repair_rendezvous_if_needed(
            gene,
            rv,
            support_node_ids,
            allow_c_recover_station,
        )
        assignment.append(gene)
        rendezvous.append(rv)

    individual = Individual(sequence=sequence, assignment=assignment, rendezvous=rendezvous)
    individual.validate()
    truck_drone_ids, depot_drone_ids, all_drone_ids = _gene_context_from_pool(gene_pool)
    individual.validate_with_context(
        truck_drone_ids,
        depot_drone_ids,
        valid_drone_ids=all_drone_ids,
        support_node_ids=support_node_ids,
    )
    return individual
