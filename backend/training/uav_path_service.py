#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-side UAV path, time, and energy estimation.

This module centralizes the UAV movement semantics used by PPO training.  It
uses the same obstacle-avoidance PathPlanner family as the solver layer and
falls back to a direct segment only when no planner assets are available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.entities.primitives import Position3D
from .scene_loader import TrainingSceneContext

try:
    from environment.path_planning.planner import PathPlanner
except ImportError:  # pragma: no cover - optional runtime dependency
    PathPlanner = None


logger = logging.getLogger(__name__)

_TIME_EPS = 1e-6
UAV_CRUISE_ALTITUDE_M = 80.0


@dataclass(frozen=True)
class UavPathEstimate:
    """A single UAV leg estimate under the training path semantics."""

    path_points: tuple[Position3D, ...]
    cumulative_distances_m: tuple[float, ...]
    distance_m: float
    flight_time_sec: float
    energy_j: float
    motion_mode: str


class TrainingUavPathService:
    """Obstacle-aware UAV path service for training and online PPO runtime."""

    def __init__(
        self,
        *,
        scene_ctx: TrainingSceneContext,
        cruise_altitude_m: float = UAV_CRUISE_ALTITUDE_M,
    ) -> None:
        self._scene_ctx = scene_ctx
        self._cruise_altitude_m = float(cruise_altitude_m)
        self._planner = self._build_path_planner(scene_ctx)
        self._path_cache: dict[
            tuple[float, float, float, float, float],
            tuple[Position3D, ...],
        ] = {}

    @property
    def has_obstacle_planner(self) -> bool:
        return self._planner is not None

    def estimate(
        self,
        *,
        drone: Any,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
        altitude: float | None = None,
    ) -> UavPathEstimate:
        path_points = self.plan_path(
            from_pos=from_pos,
            to_pos=to_pos,
            altitude=altitude,
        )
        cumulative = _build_cumulative_distances(path_points)
        distance_m = cumulative[-1] if cumulative else 0.0
        speed = max(_TIME_EPS, float(getattr(drone, "cruise_speed")))
        flight_time = distance_m / speed
        energy_j = (
            _calculate_power(drone, payload, speed) * flight_time
            if distance_m > _TIME_EPS
            else 0.0
        )
        return UavPathEstimate(
            path_points=path_points,
            cumulative_distances_m=cumulative,
            distance_m=float(distance_m),
            flight_time_sec=float(flight_time),
            energy_j=float(energy_j),
            motion_mode=(
                "uav_avoidance_path"
                if self._planner is not None
                else "straight_line"
            ),
        )

    def can_reach(
        self,
        *,
        drone: Any,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
        safe_margin: float,
        battery_current: float,
        altitude: float | None = None,
    ) -> bool:
        estimate = self.estimate(
            drone=drone,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=payload,
            altitude=altitude,
        )
        return (
            float(battery_current) + _TIME_EPS
            >= estimate.energy_j + float(safe_margin)
        )

    def plan_path(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
        altitude: float | None = None,
    ) -> tuple[Position3D, ...]:
        effective_altitude = self._resolve_altitude(from_pos, to_pos, altitude)
        key = (
            round(float(from_pos.x), 2),
            round(float(from_pos.y), 2),
            round(float(to_pos.x), 2),
            round(float(to_pos.y), 2),
            round(float(effective_altitude), 1),
        )
        cached = self._path_cache.get(key)
        if cached is not None:
            return tuple(_clone_position(pos) for pos in cached)

        path: list[Position3D]
        if self._planner is None:
            path = [_clone_position(from_pos), _clone_position(to_pos)]
        else:
            try:
                planned = self._planner.plan(from_pos, to_pos, effective_altitude)
            except Exception:
                logger.exception(
                    "[TrainingUavPathService] UAV path planning failed; "
                    "falling back to direct segment"
                )
                planned = []
            path = (
                list(planned)
                if len(planned) >= 2
                else [_clone_position(from_pos), _clone_position(to_pos)]
            )
            path[0] = _clone_position(from_pos)
            path[-1] = _clone_position(to_pos)

        compacted = _dedupe_adjacent(path)
        self._path_cache[key] = tuple(_clone_position(pos) for pos in compacted)
        return tuple(_clone_position(pos) for pos in compacted)

    def path_distance(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
        altitude: float | None = None,
    ) -> float:
        path_points = self.plan_path(
            from_pos=from_pos,
            to_pos=to_pos,
            altitude=altitude,
        )
        cumulative = _build_cumulative_distances(path_points)
        return float(cumulative[-1]) if cumulative else 0.0

    def _resolve_altitude(
        self,
        from_pos: Position3D,
        to_pos: Position3D,
        altitude: float | None,
    ) -> float:
        if altitude is not None:
            return float(altitude)
        return max(self._cruise_altitude_m, float(from_pos.z), float(to_pos.z))

    @staticmethod
    def _build_path_planner(scene_ctx: TrainingSceneContext):
        if PathPlanner is None:
            return None
        buildings_geojson = _load_obstacle_geojson(scene_ctx)
        if buildings_geojson is None:
            return None
        return PathPlanner(buildings_geojson)


def _load_obstacle_geojson(
    scene_ctx: TrainingSceneContext,
) -> Mapping[str, Any] | Sequence[Mapping[str, Any]] | None:
    bundle_dir = Path(scene_ctx.scene_bundle_dir)
    backend_dir = Path(__file__).resolve().parents[1]
    candidate_paths = [
        bundle_dir / "no_fly_zones.geojson",
        bundle_dir / "buildings.geojson",
        backend_dir / "test_data" / str(scene_ctx.scene_id) / "no_fly_zones.geojson",
        backend_dir / "test_data" / str(scene_ctx.scene_id) / "buildings.geojson",
    ]
    if scene_ctx.scene_id == "default_test_4x4km":
        candidate_paths.extend(
            [
                backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson",
                backend_dir / "test_data" / "default_scene" / "buildings.geojson",
            ]
        )
    candidate_paths.extend(
        [
            backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson",
            backend_dir / "test_data" / "default_scene" / "buildings.geojson",
        ]
    )

    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            logger.warning(
                "[TrainingUavPathService] obstacle geojson load failed: %s",
                resolved,
            )
            continue
        if isinstance(data, (dict, list)):
            return data
    return None


def _calculate_power(drone: Any, payload: float, speed: float) -> float:
    calculate_power = getattr(drone, "calculate_power", None)
    if callable(calculate_power):
        return float(calculate_power(payload, speed))
    mass = float(getattr(drone, "empty_weight")) + max(0.0, float(payload))
    return float(getattr(drone, "k1")) * (mass**1.5) + float(getattr(drone, "k2")) * (
        speed**3
    )


def _build_cumulative_distances(points: Sequence[Position3D]) -> tuple[float, ...]:
    if not points:
        return ()
    cumulative = [0.0]
    for idx in range(1, len(points)):
        cumulative.append(
            cumulative[-1] + _segment_distance(points[idx - 1], points[idx])
        )
    return tuple(float(item) for item in cumulative)


def _segment_distance(a: Position3D, b: Position3D) -> float:
    # Solver-side UAV polyline costs are horizontal path distances.  Keep the
    # same semantics here so ETA/energy are aligned with Greedy path planning.
    return float(a.distance_2d(b))


def _dedupe_adjacent(points: Sequence[Position3D]) -> tuple[Position3D, ...]:
    compacted: list[Position3D] = []
    for point in points:
        pos = _clone_position(point)
        if (
            compacted
            and compacted[-1].distance_2d(pos) <= 1e-6
            and abs(compacted[-1].z - pos.z) <= 1e-6
        ):
            continue
        compacted.append(pos)
    if not compacted:
        return ()
    return tuple(compacted)


def _clone_position(pos: Position3D) -> Position3D:
    return Position3D(x=float(pos.x), y=float(pos.y), z=float(pos.z))
