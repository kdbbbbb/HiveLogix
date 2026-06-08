from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.entities.primitives import Position3D

from .grid_astar import Point2D, load_obstacle_polygons, plan_path


class PathPlanner:
    def __init__(self, buildings_geojson: dict):
        self._buildings_geojson: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = buildings_geojson
        self._obstacles_cache: dict[float, list] = {}
        self._path_cache: dict[tuple, list[Position3D]] = {}

    def plan(
        self,
        start: Position3D,
        goal: Position3D,
        altitude: float,
    ) -> list[Position3D]:
        start_xy: Point2D = (float(start.x), float(start.y))
        goal_xy: Point2D = (float(goal.x), float(goal.y))
        
        path_key = (
            round(start_xy[0], 2), round(start_xy[1], 2),
            round(goal_xy[0], 2), round(goal_xy[1], 2),
            round(altitude, 1)
        )
        if path_key in self._path_cache:
            return self._path_cache[path_key]

        # 缓存同一高度下加载并完成投影转换的障碍物多边形（极大减少重复转换开销）
        cache_key = round(altitude, 1)
        if cache_key not in self._obstacles_cache:
            self._obstacles_cache[cache_key] = load_obstacle_polygons(self._buildings_geojson, altitude)
        obstacles = self._obstacles_cache[cache_key]

        path_xy = plan_path(start_xy, goal_xy, obstacles)

        if not path_xy:
            path_xy = [start_xy, goal_xy]

        z = float(start.z)
        path_3d = [Position3D(x=x, y=y, z=z) for x, y in path_xy]
        if path_3d:
            path_3d[0] = start
            path_3d[-1] = goal
            
        self._path_cache[path_key] = path_3d
        return path_3d
