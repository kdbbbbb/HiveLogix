#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recovery pool deterministic selector shared by env_adapter and planner_bridge.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def select_recovery_pool_for_order(
    *,
    order: Any,
    truck_backbone_route: Sequence[str],
    truck_eta_map: Mapping[str, float],
    node_states: Mapping[str, Any],
    max_candidates: int,
    future_scan_limit: int,
    drone_cruise_speed: float,
    upper_horizon_sec: float,
) -> tuple[str, ...]:
    """Expose fixed nodes that the truck will still visit.

    The coarse layer keeps a small deterministic pool per order:
      1. scan the near-future truck backbone up to ``future_scan_limit`` valid nodes;
      2. sort by delivery-to-recovery distance, then ETA and route order;
      3. keep at most ``max_candidates`` nodes.

    Per-drone timing and energy feasibility still happen in CandidateBuilder.
    """
    if not truck_backbone_route:
        return ()
    if max_candidates <= 0 or future_scan_limit <= 0:
        return ()

    delivery_loc = getattr(order, "delivery_loc", None)
    if delivery_loc is None:
        return ()

    scanned: list[tuple[float, float, int, str]] = []
    seen: set[str] = set()
    for route_idx, node_id in enumerate(truck_backbone_route):
        node_id = str(node_id)
        if node_id in seen:
            continue
        t_arrive_truck = truck_eta_map.get(node_id)
        node_state = node_states.get(node_id)
        if t_arrive_truck is None or node_state is None:
            continue
        seen.add(node_id)
        if float(t_arrive_truck) > float(upper_horizon_sec):
            continue
        node_pos = getattr(node_state, "position", None)
        if node_pos is None:
            continue
        distance_m = float(delivery_loc.distance_2d(node_pos))
        eta = float(t_arrive_truck)
        # ``drone_cruise_speed`` is intentionally not part of the ordering while
        # all candidates share the same speed scalar here; keep it validated so
        # callers cannot accidentally pass a non-physical value unnoticed.
        if float(drone_cruise_speed) <= 0.0:
            continue
        scanned.append((distance_m, eta, int(route_idx), node_id))
        if len(scanned) >= int(future_scan_limit):
            break

    scanned.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return tuple(item[3] for item in scanned[: int(max_candidates)])
