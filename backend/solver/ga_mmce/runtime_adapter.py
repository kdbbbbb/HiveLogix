# -*- coding: utf-8 -*-
"""Runtime bridge for GA-MMCE plans.

The public simulator already knows how to execute RouteWaypoint lists.  This
module keeps GA-specific semantics here: station launches are exposed as
B_WAIT-like allocations, and repeated use of the same drone is represented as a
single waypoint queue with lightweight runtime hooks.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from core.entities.primitives import DroneStatus, RouteWaypoint, TaskStatus, WaypointAction
from solver.greedy_mmce import DroneRoute

logger = logging.getLogger(__name__)


@dataclass
class GARuntimeSegment:
    order_id: str
    mode: str
    drone_id: str
    truck_id: str
    launch_node_id: str
    recovery_node_id: str
    launch_time: float
    payload_weight: float
    start_idx: int
    pickup_idx: int
    deliver_idx: int
    dock_idx: int
    launch_loc: Any
    delivery_loc: Any
    recovery_loc: Any
    truck_launch: bool
    truck_recovery: bool


class GADroneRouteLookup(dict):
    """dict-compatible route lookup that returns duplicate-drone routes in order.

    simulation_bp intentionally remains unchanged and calls
    plan.drone_routes.get(alloc.drone_id).  GA can assign several orders to the
    same drone, so this adapter returns the next route for that drone each time
    get() is called.
    """

    def __init__(self, routes_by_drone: dict[str, list[DroneRoute]]) -> None:
        super().__init__(
            (drone_id, routes[0])
            for drone_id, routes in routes_by_drone.items()
            if routes
        )
        self._routes_by_drone = routes_by_drone
        self._cursor: dict[str, int] = defaultdict(int)

    def get(self, key: Any, default: Any = None) -> Any:
        drone_id = str(key)
        routes = self._routes_by_drone.get(drone_id)
        if not routes:
            return default
        idx = self._cursor[drone_id]
        if idx >= len(routes):
            return routes[-1]
        self._cursor[drone_id] = idx + 1
        return routes[idx]


def _plan_drone_routes_by_drone(plan: Any) -> dict[str, list[DroneRoute]]:
    raw_routes = getattr(plan, "drone_routes", None)
    if not raw_routes:
        return {}
    if hasattr(raw_routes, "_routes_by_drone"):
        return {
            str(drone_id): list(routes)
            for drone_id, routes in getattr(raw_routes, "_routes_by_drone", {}).items()
        }

    result: dict[str, list[DroneRoute]] = defaultdict(list)
    values = raw_routes.values() if isinstance(raw_routes, dict) else raw_routes
    for route in values or []:
        drone_id = str(getattr(route, "drone_id", "") or "")
        if drone_id:
            result[drone_id].append(route)
    return result


def _route_for_alloc(
    routes_by_drone: dict[str, list[DroneRoute]],
    route_cursor: dict[str, int],
    alloc: Any,
) -> DroneRoute | None:
    drone_id = str(getattr(alloc, "drone_id", "") or "")
    routes = routes_by_drone.get(drone_id) or []
    if not routes:
        return None

    start = route_cursor.get(drone_id, 0)
    order_id = str(getattr(alloc, "order_id", "") or "")
    for idx in range(start, len(routes)):
        if str(getattr(routes[idx], "order_id", "") or "") == order_id:
            route_cursor[drone_id] = idx + 1
            return routes[idx]
    if start < len(routes):
        route_cursor[drone_id] = start + 1
        return routes[start]
    return routes[-1]


def _route_path_for_segment(route: DroneRoute | None, segment: "GARuntimeSegment") -> tuple[list[Any], int]:
    path = list(getattr(route, "path", []) or []) if route is not None else []
    if len(path) < 2:
        path = [segment.launch_loc, segment.delivery_loc, segment.recovery_loc]

    path[0] = segment.launch_loc
    path[-1] = segment.recovery_loc
    delivery_idx = min(
        range(len(path)),
        key=lambda idx: path[idx].distance_2d(segment.delivery_loc)
        if hasattr(path[idx], "distance_2d")
        else float("inf"),
    )
    if delivery_idx in (0, len(path) - 1):
        insert_idx = max(1, len(path) - 1)
        path.insert(insert_idx, segment.delivery_loc)
        delivery_idx = insert_idx
    else:
        path[delivery_idx] = segment.delivery_loc
    return path, delivery_idx


def _route_path_with_greedy_builder(
    route: DroneRoute | None,
    segment: "GARuntimeSegment",
    build_uav_polyline: Any = None,
) -> tuple[list[Any], int]:
    if callable(build_uav_polyline):
        altitude = float(
            max(
                getattr(segment.launch_loc, "z", 0.0),
                getattr(segment.delivery_loc, "z", 0.0),
                getattr(segment.recovery_loc, "z", 0.0),
                80.0,
            )
        )
        try:
            path, delivery_idx = build_uav_polyline(
                launch_loc=segment.launch_loc,
                delivery_loc=segment.delivery_loc,
                recovery_loc=segment.recovery_loc,
                altitude=altitude,
            )
            if path:
                return list(path), int(delivery_idx)
        except Exception:
            logger.exception(
                "[GA-MMCE runtime] failed to build greedy-style UAV polyline for order %s",
                segment.order_id,
            )
    return _route_path_for_segment(route, segment)


def _waypoints_for_segment(path: list[Any], delivery_idx: int, segment: "GARuntimeSegment") -> list[RouteWaypoint]:
    last_idx = len(path) - 1
    dock_action = WaypointAction.DOCK_TRUCK if segment.truck_recovery else WaypointAction.DOCK_DEPOT
    waypoints: list[RouteWaypoint] = []
    for idx, loc in enumerate(path):
        if idx == 0:
            action = WaypointAction.PICKUP
            target_id = segment.order_id
        elif idx == delivery_idx:
            action = WaypointAction.DELIVER
            target_id = segment.order_id
        elif idx == last_idx:
            action = dock_action
            target_id = segment.recovery_node_id
        else:
            action = WaypointAction.PICKUP
            target_id = segment.order_id
        waypoints.append(RouteWaypoint(loc, action, target_id))
    return waypoints


def normalize_ga_allocation_modes(plan: Any) -> None:
    """Expose GA station-launch B tasks through the simulator's B_WAIT contract."""
    for alloc in getattr(plan, "allocations", []) or []:
        if (
            getattr(alloc, "feasible", False)
            and getattr(alloc, "mode", "") == "B"
            and getattr(alloc, "launch_station_id", "")
        ):
            alloc.mode = "B_WAIT"
    summary = getattr(plan, "summary", None)
    if isinstance(summary, dict) and "modes" in summary:
        modes: dict[str, int] = {}
        for alloc in getattr(plan, "allocations", []) or []:
            mode = str(getattr(alloc, "mode", "") or "")
            if mode:
                modes[mode] = modes.get(mode, 0) + 1
        summary["modes"] = modes


def apply_ga_mmce_runtime_plan(
    *,
    entity_mgr: Any,
    order_mgr: Any,
    plan: Any,
    current_time: float,
    build_uav_polyline: Any = None,
) -> None:
    """Apply GA drone runtime state without changing shared solver semantics."""
    normalize_ga_allocation_modes(plan)

    grouped: dict[str, list[Any]] = defaultdict(list)
    for alloc in getattr(plan, "allocations", []) or []:
        if (
            getattr(alloc, "feasible", False)
            and getattr(alloc, "mode", "") != "A"
            and getattr(alloc, "drone_id", "")
        ):
            grouped[str(alloc.drone_id)].append(alloc)

    _clear_obsolete_ga_drone_routes(entity_mgr=entity_mgr, plan=plan)

    routes_by_drone = _plan_drone_routes_by_drone(plan)
    route_cursor: dict[str, int] = defaultdict(int)
    prepared = 0
    for drone_id, allocs in grouped.items():
        drone = entity_mgr.drones.get(drone_id)
        if drone is None:
            logger.warning("[GA-MMCE runtime] drone %s not found", drone_id)
            continue
        if getattr(getattr(drone, "status", None), "is_flying", False):
            logger.warning(
                "[GA-MMCE runtime] skip drone %s because it is already flying (%s)",
                drone_id,
                getattr(drone.status, "value", drone.status),
            )
            continue

        waypoints: list[RouteWaypoint] = []
        segments: list[GARuntimeSegment] = []
        for idx, alloc in enumerate(allocs):
            segment = _build_segment(
                entity_mgr=entity_mgr,
                order_mgr=order_mgr,
                alloc=alloc,
                current_time=current_time,
                start_idx=len(waypoints),
                is_last_for_drone=(idx == len(allocs) - 1),
            )
            if segment is None:
                continue
            route = _route_for_alloc(routes_by_drone, route_cursor, alloc)
            path, delivery_idx = _route_path_with_greedy_builder(route, segment, build_uav_polyline)
            segment.start_idx = len(waypoints)
            segment.pickup_idx = segment.start_idx
            segment.deliver_idx = segment.start_idx + delivery_idx
            segment.dock_idx = segment.start_idx + len(path) - 1
            waypoints.extend(_waypoints_for_segment(path, delivery_idx, segment))
            segments.append(segment)

        if not segments:
            continue

        drone.set_route(waypoints)
        _install_runtime_hooks(drone, segments, order_mgr)
        _prepare_segment_start(
            drone=drone,
            segment=segments[0],
            entity_mgr=entity_mgr,
            current_time=current_time,
        )
        prepared += 1

    logger.info("[GA-MMCE runtime] prepared %d drone route queues", prepared)


def build_ga_mmce_drone_routes(
    *,
    entity_mgr: Any,
    order_mgr: Any,
    plan: Any,
    build_uav_polyline: Any = None,
) -> None:
    """Build frontend route lookup while keeping simulation_bp unchanged."""
    normalize_ga_allocation_modes(plan)
    routes_by_drone: dict[str, list[DroneRoute]] = defaultdict(list)
    existing_routes_by_drone = _plan_drone_routes_by_drone(plan)
    route_cursor: dict[str, int] = defaultdict(int)

    for alloc in getattr(plan, "allocations", []) or []:
        if (
            not getattr(alloc, "feasible", False)
            or getattr(alloc, "mode", "") == "A"
            or not getattr(alloc, "drone_id", "")
        ):
            continue
        order = (
            order_mgr.assigned_orders.get(alloc.order_id)
            or order_mgr.pending_orders.get(alloc.order_id)
        )
        segment = _build_segment(
            entity_mgr=entity_mgr,
            order_mgr=order_mgr,
            alloc=alloc,
            current_time=0.0,
            start_idx=0,
            is_last_for_drone=True,
            order=order,
        )
        if segment is None:
            continue
        existing_route = _route_for_alloc(existing_routes_by_drone, route_cursor, alloc)
        path, _delivery_idx = _route_path_with_greedy_builder(existing_route, segment, build_uav_polyline)
        routes_by_drone[str(alloc.drone_id)].append(
            DroneRoute(
                drone_id=str(alloc.drone_id),
                order_id=str(alloc.order_id),
                path=path,
                mode=str(alloc.mode),
                launch_loc=segment.launch_loc,
                delivery_loc=segment.delivery_loc,
                recovery_loc=segment.recovery_loc,
            )
        )

    plan.drone_routes = GADroneRouteLookup(dict(routes_by_drone))


def handle_ga_drone_action(
    *,
    entity_mgr: Any,
    order_mgr: Any,
    drone: Any,
    action: WaypointAction,
    reached_wp: RouteWaypoint | None,
    reached_target: str | None,
    current_time: float,
) -> bool:
    segments: list[GARuntimeSegment] = getattr(drone, "_ga_runtime_segments", []) or []
    if not segments:
        return False

    reached_idx = int(getattr(drone, "current_waypoint_index", 0) or 0) - 1
    by_pickup = getattr(drone, "_ga_segment_by_pickup_idx", {}) or {}
    by_deliver = getattr(drone, "_ga_segment_by_deliver_idx", {}) or {}
    by_dock = getattr(drone, "_ga_segment_by_dock_idx", {}) or {}

    if action == WaypointAction.PICKUP:
        segment = by_pickup.get(reached_idx)
        if segment is None:
            return False
        _assign_segment_order(drone, segment, order_mgr)
        drone.status = DroneStatus.FLYING_TO_DELIVER
        logger.debug(
            "[GA-MMCE runtime] drone %s picked order %s at %s",
            drone.drone_id,
            segment.order_id,
            segment.launch_node_id,
        )
        return True

    if action == WaypointAction.DELIVER:
        segment = by_deliver.get(reached_idx)
        if segment is None:
            return False
        if drone.carrying_order_id:
            drone.pending_release_order_id = drone.carrying_order_id
            drone.delivery_service_end_time = (
                current_time + float(getattr(entity_mgr, "DRONE_SERVICE_TIME_ORDER", 0.0))
            )
            logger.info(
                "[GA-MMCE runtime] drone %s delivered order %s, service %.1fs",
                drone.drone_id,
                drone.carrying_order_id,
                float(getattr(entity_mgr, "DRONE_SERVICE_TIME_ORDER", 0.0)),
            )
        return True

    if action in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
        segment = by_dock.get(reached_idx)
        if segment is None:
            return False
        entity_mgr._recharge_drone_to_full(drone, f"GA recover at {segment.recovery_node_id}")
        next_segment = _next_segment_for_drone(drone)
        if segment.truck_recovery:
            drone.waiting_recovery_station_id = segment.recovery_node_id
            drone.status = DroneStatus.IDLE
            logger.info(
                "[GA-MMCE runtime] drone %s waits at %s for truck recovery",
                drone.drone_id,
                segment.recovery_node_id,
            )
        elif next_segment is not None:
            _prepare_segment_start(
                drone=drone,
                segment=next_segment,
                entity_mgr=entity_mgr,
                current_time=current_time,
            )
        else:
            drone.waiting_recovery_station_id = ""
            drone.transport_truck_id = None
            drone.scheduled_launch_time = 0.0
            drone.launch_station_id = ""
            drone.status = DroneStatus.IDLE
            logger.info(
                "[GA-MMCE runtime] drone %s completed GA route at %s",
                drone.drone_id,
                segment.recovery_node_id,
            )
        return True

    return False


def handle_ga_truck_recovered_drone(
    *,
    entity_mgr: Any,
    truck: Any,
    station_id: str,
    drone: Any,
    current_time: float,
) -> None:
    segments: list[GARuntimeSegment] = getattr(drone, "_ga_runtime_segments", []) or []
    if not segments:
        return

    next_segment = _next_segment_for_drone(drone)
    if next_segment is None:
        drone.scheduled_launch_time = float("inf")
        drone.launch_station_id = ""
        return

    _prepare_segment_start(
        drone=drone,
        segment=next_segment,
        entity_mgr=entity_mgr,
        current_time=current_time,
        recovered_by_truck=truck,
    )


def _build_segment(
    *,
    entity_mgr: Any,
    order_mgr: Any,
    alloc: Any,
    current_time: float,
    start_idx: int,
    is_last_for_drone: bool,
    order: Any | None = None,
) -> GARuntimeSegment | None:
    order = order or order_mgr.assigned_orders.get(alloc.order_id)
    if order is None:
        order = order_mgr.pending_orders.get(alloc.order_id)
    if order is None:
        logger.warning("[GA-MMCE runtime] order %s not found", alloc.order_id)
        return None

    mode = str(getattr(alloc, "mode", "") or "")
    drone_id = str(getattr(alloc, "drone_id", "") or "")
    truck_id = str(getattr(alloc, "vehicle_id", "") or "")
    launch_node_id = str(getattr(alloc, "launch_station_id", "") or "")
    recovery_node_id = str(getattr(alloc, "recovery_station_id", "") or "")

    if mode in {"B", "B_WAIT"}:
        truck = entity_mgr.trucks.get(truck_id)
        launch_entity = _resolve_support_entity(entity_mgr, launch_node_id)
        if launch_entity is not None:
            launch_loc = launch_entity.location
        elif truck is not None:
            launch_loc = truck.get_location(current_time)
        else:
            logger.warning("[GA-MMCE runtime] missing launch for order %s", alloc.order_id)
            return None
        recovery_entity = _resolve_support_entity(entity_mgr, recovery_node_id)
        if recovery_entity is None:
            logger.warning("[GA-MMCE runtime] recovery %s not found", recovery_node_id)
            return None
        return GARuntimeSegment(
            order_id=str(alloc.order_id),
            mode=mode,
            drone_id=drone_id,
            truck_id=truck_id,
            launch_node_id=launch_node_id,
            recovery_node_id=recovery_node_id,
            launch_time=float(getattr(alloc, "launch_time", 0.0) or current_time),
            payload_weight=float(getattr(order, "payload_weight", 0.0) or 0.0),
            start_idx=start_idx,
            pickup_idx=start_idx,
            deliver_idx=start_idx + 1,
            dock_idx=start_idx + 2,
            launch_loc=launch_loc,
            delivery_loc=order.delivery_loc,
            recovery_loc=recovery_entity.location,
            truck_launch=truck is not None and bool(launch_node_id),
            truck_recovery=truck is not None and bool(recovery_node_id),
        )

    if mode == "C":
        launch_node_id = launch_node_id or truck_id
        launch_entity = _resolve_support_entity(entity_mgr, launch_node_id)
        recovery_entity = _resolve_support_entity(entity_mgr, recovery_node_id or truck_id)
        if launch_entity is None or recovery_entity is None:
            logger.warning(
                "[GA-MMCE runtime] C segment nodes missing: launch=%s recovery=%s",
                launch_node_id,
                recovery_node_id or truck_id,
            )
            return None
        recover_is_station = recovery_node_id in entity_mgr.stations
        fallback_truck_id = next(iter(entity_mgr.trucks.keys()), "")
        return GARuntimeSegment(
            order_id=str(alloc.order_id),
            mode=mode,
            drone_id=drone_id,
            truck_id=fallback_truck_id,
            launch_node_id=launch_node_id,
            recovery_node_id=recovery_node_id or truck_id,
            launch_time=float(getattr(alloc, "launch_time", 0.0) or current_time),
            payload_weight=float(getattr(order, "payload_weight", 0.0) or 0.0),
            start_idx=start_idx,
            pickup_idx=start_idx,
            deliver_idx=start_idx + 1,
            dock_idx=start_idx + 2,
            launch_loc=launch_entity.location,
            delivery_loc=order.delivery_loc,
            recovery_loc=recovery_entity.location,
            truck_launch=False,
            truck_recovery=bool(recover_is_station and is_last_for_drone and fallback_truck_id),
        )

    return None


def _install_runtime_hooks(
    drone: Any,
    segments: list[GARuntimeSegment],
    order_mgr: Any,
) -> None:
    drone._ga_runtime_segments = segments
    drone._ga_segment_by_pickup_idx = {segment.pickup_idx: segment for segment in segments}
    drone._ga_segment_by_deliver_idx = {segment.deliver_idx: segment for segment in segments}
    drone._ga_segment_by_dock_idx = {segment.dock_idx: segment for segment in segments}
    drone._ga_segment_by_start_idx = {segment.start_idx: segment for segment in segments}
    drone._runtime_action_handler = handle_ga_drone_action
    drone._runtime_recovery_handler = handle_ga_truck_recovered_drone
    drone._runtime_solver = "ga_mmce"


def _prepare_segment_start(
    *,
    drone: Any,
    segment: GARuntimeSegment,
    entity_mgr: Any,
    current_time: float,
    recovered_by_truck: Any | None = None,
) -> None:
    if segment.truck_launch:
        truck = recovered_by_truck or entity_mgr.trucks.get(segment.truck_id)
        if truck is None:
            logger.warning(
                "[GA-MMCE runtime] truck %s not found for drone %s",
                segment.truck_id,
                drone.drone_id,
            )
            return
        drone.current_loc = truck.get_location(current_time)
        drone.status = DroneStatus.IDLE
        drone.transport_truck_id = truck.truck_id
        if segment.launch_node_id in getattr(entity_mgr, "depots", {}):
            drone.scheduled_launch_time = current_time
        else:
            drone.scheduled_launch_time = max(float(segment.launch_time or 0.0), current_time)
        drone.launch_station_id = segment.launch_node_id
        drone.waiting_recovery_station_id = ""
        if drone.drone_id not in truck.docked_drones:
            truck.docked_drones.append(drone.drone_id)
        return

    drone.current_loc = segment.launch_loc
    drone.status = DroneStatus.FLYING_TO_PICKUP
    drone.transport_truck_id = None
    drone.scheduled_launch_time = 0.0
    drone.launch_station_id = ""
    drone.waiting_recovery_station_id = ""
    _remove_from_all_trucks(entity_mgr, drone.drone_id)


def _next_segment_for_drone(drone: Any) -> GARuntimeSegment | None:
    idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
    by_start = getattr(drone, "_ga_segment_by_start_idx", {}) or {}
    return by_start.get(idx)


def _assign_segment_order(
    drone: Any,
    segment: GARuntimeSegment,
    order_mgr: Any,
) -> None:
    if drone.carrying_order_id == segment.order_id:
        return
    if drone.carrying_order_id:
        logger.warning(
            "[GA-MMCE runtime] drone %s already carries %s, cannot pick %s",
            drone.drone_id,
            drone.carrying_order_id,
            segment.order_id,
        )
        return
    try:
        drone.assign_order(segment.order_id, segment.payload_weight)
    except ValueError as exc:
        logger.warning(
            "[GA-MMCE runtime] drone %s assign order %s failed: %s",
            drone.drone_id,
            segment.order_id,
            exc,
        )
        return

    order = order_mgr.assigned_orders.get(segment.order_id)
    if order is None:
        return
    try:
        if order.status == TaskStatus.ASSIGNED:
            order.update_status(TaskStatus.PICKED_UP)
            order.update_status(TaskStatus.DELIVERING)
        elif order.status == TaskStatus.PICKED_UP:
            order.update_status(TaskStatus.DELIVERING)
    except ValueError as exc:
        logger.warning(
            "[GA-MMCE runtime] order %s pickup status update failed: %s",
            segment.order_id,
            exc,
        )


def _resolve_support_entity(entity_mgr: Any, node_id: str) -> Any | None:
    if not node_id:
        return None
    return entity_mgr.stations.get(node_id) or entity_mgr.depots.get(node_id)


def _clear_obsolete_ga_drone_routes(*, entity_mgr: Any, plan: Any) -> None:
    planned_by_drone: dict[str, set[str]] = defaultdict(set)
    for alloc in getattr(plan, "allocations", []) or []:
        if (
            getattr(alloc, "feasible", False)
            and getattr(alloc, "mode", "") != "A"
            and getattr(alloc, "drone_id", "")
        ):
            planned_by_drone[str(alloc.drone_id)].add(str(alloc.order_id))

    summary = getattr(plan, "summary", {}) or {}
    dynamic_scope: set[str] = set()
    if isinstance(summary, dict) and str(summary.get("dispatch_type", "") or "") in {
        "dynamic_replan",
        "incremental",
    }:
        dynamic_scope = {
            str(order_id)
            for order_id in (summary.get("reoptimized_order_ids", []) or [])
            if str(order_id)
        }

    for drone_id, drone in getattr(entity_mgr, "drones", {}).items():
        if str(getattr(drone, "_runtime_solver", "") or "").lower() != "ga_mmce":
            continue
        if getattr(getattr(drone, "status", None), "is_flying", False):
            continue
        if getattr(drone, "carrying_order_id", None):
            continue

        route_orders = _pending_route_order_ids(drone)
        expected_orders = planned_by_drone.get(str(drone_id), set())
        obsolete_orders = route_orders - expected_orders
        if dynamic_scope:
            obsolete_orders &= dynamic_scope
        if not obsolete_orders:
            continue

        drone.route_plan = []
        drone.current_waypoint_index = 0
        drone._ga_runtime_segments = []
        drone._ga_segment_by_pickup_idx = {}
        drone._ga_segment_by_deliver_idx = {}
        drone._ga_segment_by_dock_idx = {}
        drone._ga_segment_by_start_idx = {}
        drone.scheduled_launch_time = float("inf")
        drone.launch_station_id = ""
        drone.pending_release_order_id = None
        drone.delivery_service_end_time = 0.0
        logger.info(
            "[GA-MMCE runtime] cleared obsolete route for drone %s, orders=%s",
            drone_id,
            sorted(obsolete_orders),
        )


def _pending_route_order_ids(drone: Any) -> set[str]:
    route_plan = list(getattr(drone, "route_plan", []) or [])
    idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
    result: set[str] = set()
    for wp in route_plan[max(0, idx):]:
        action = getattr(wp, "action", None)
        action_name = getattr(action, "value", action)
        if str(action_name or "").upper() not in {"PICKUP", "DELIVER"}:
            continue
        order_id = str(getattr(wp, "target_entity_id", "") or "")
        if order_id:
            result.add(order_id)
    return result


def _remove_from_all_trucks(entity_mgr: Any, drone_id: str) -> None:
    for truck in entity_mgr.trucks.values():
        docked = getattr(truck, "docked_drones", None)
        if isinstance(docked, list) and drone_id in docked:
            docked.remove(drone_id)
