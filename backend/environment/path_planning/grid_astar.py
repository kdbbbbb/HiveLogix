from __future__ import annotations

import heapq
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape

Point2D = tuple[float, float]

GRID_RESOLUTION = 10.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_features(buildings_geojson: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if buildings_geojson is None:
        return []
    if isinstance(buildings_geojson, Mapping):
        if buildings_geojson.get("type") == "FeatureCollection":
            features = buildings_geojson.get("features", [])
            if isinstance(features, list):
                return [f for f in features if isinstance(f, Mapping)]
        return []
    if isinstance(buildings_geojson, Sequence):
        return [f for f in buildings_geojson if isinstance(f, Mapping)]
    return []


def _extract_polygons(feature: Mapping[str, Any]) -> list[Polygon]:
    geometry = feature.get("geometry")
    if not isinstance(geometry, Mapping):
        return []

    geom = shape(geometry)
    if geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        candidates = [geom]
    elif isinstance(geom, MultiPolygon):
        candidates = list(geom.geoms)
    else:
        return []

    from utils.coord_utils import wgs84_to_utm

    polygons: list[Polygon] = []
    for poly in candidates:
        if poly.is_empty:
            continue
            
        try:
            utm_exterior = [wgs84_to_utm(lon, lat) for lon, lat in poly.exterior.coords]
            utm_interiors = [[wgs84_to_utm(lon, lat) for lon, lat in ring.coords] for ring in poly.interiors]
            poly_utm = Polygon(utm_exterior, utm_interiors)
        except BaseException:
            poly_utm = poly

        normalized = poly_utm if poly_utm.is_valid else poly_utm.buffer(0)
        if isinstance(normalized, Polygon) and not normalized.is_empty:
            polygons.append(normalized)
    return polygons


def load_obstacle_polygons(
    buildings_geojson: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    drone_altitude: float,
) -> list[Polygon]:
    obstacles: list[Polygon] = []
    features = _iter_features(buildings_geojson)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[VisibilityGraph] load_obstacle_polygons: 总特征数=%d, 无人机高度=%.1f", len(features), drone_altitude)
    
    for feature in features:
        props = feature.get("properties", {})
        if not isinstance(props, Mapping):
            props = {}
        nf = _as_bool(props.get("nf", False))
        height = _as_float(props.get("h", 0.0), default=0.0)
        if not (nf or height > float(drone_altitude)):
            continue
        obstacles.extend(_extract_polygons(feature))
    
    logger.info("[VisibilityGraph] load_obstacle_polygons: 筛选后障碍物数=%d", len(obstacles))
    return obstacles


def _create_occupancy_grid(
    start: Point2D,
    goal: Point2D,
    obstacles: list[Polygon],
    resolution: float,
) -> tuple[dict[tuple[int, int], bool], float, float, int, int]:
    # 计算 bounding box with padding
    min_x = min(start[0], goal[0])
    max_x = max(start[0], goal[0])
    min_y = min(start[1], goal[1])
    max_y = max(start[1], goal[1])
    dist = math.dist(start, goal)
    padding = max(dist * 0.5, 200.0)
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    # 计算 grid 尺寸
    width = int((max_x - min_x) / resolution) + 1
    height = int((max_y - min_y) / resolution) + 1

    grid: dict[tuple[int, int], bool] = {}

    # 栅格化障碍物
    for poly in obstacles:
        poly_min_x, poly_min_y, poly_max_x, poly_max_y = poly.bounds
        start_i = max(0, int((poly_min_x - min_x) / resolution))
        end_i = min(width, int((poly_max_x - min_x) / resolution) + 1)
        start_j = max(0, int((poly_min_y - min_y) / resolution))
        end_j = min(height, int((poly_max_y - min_y) / resolution) + 1)

        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                cell_min_x = min_x + i * resolution
                cell_max_x = cell_min_x + resolution
                cell_min_y = min_y + j * resolution
                cell_max_y = cell_min_y + resolution
                cell_box = Polygon([
                    (cell_min_x, cell_min_y),
                    (cell_max_x, cell_min_y),
                    (cell_max_x, cell_max_y),
                    (cell_min_x, cell_max_y)
                ])
                if poly.intersects(cell_box):
                    grid[(i, j)] = True

    return grid, min_x, min_y, width, height


def _is_visible(p1: Point2D, p2: Point2D, obstacles: list[Polygon]) -> bool:
    line = LineString([p1, p2])
    for poly in obstacles:
        if line.crosses(poly) or line.within(poly):
            return False
    return True


def _smooth_path(
    path: list[Point2D],
    obstacles: list[Polygon],
) -> list[Point2D]:
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        farthest = i + 1
        for j in range(i + 2, len(path)):
            if _is_visible(path[i], path[j], obstacles):
                farthest = j
            else:
                break
        smoothed.append(path[farthest])
        i = farthest
        if i == len(path) - 1:
            break
    return smoothed


def _grid_astar(
    grid: dict[tuple[int, int], bool],
    start_idx: tuple[int, int],
    goal_idx: tuple[int, int],
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], int]:
    open_heap: list[tuple[float, tuple[int, int], float]] = [(0.0, start_idx, 0.0)]  # f, pos, g
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_cost: dict[tuple[int, int], float] = {start_idx: 0.0}
    closed: set[tuple[int, int]] = set()
    expanded = 0

    while open_heap:
        f, current, g = heapq.heappop(open_heap)
        if current in closed:
            continue
        expanded += 1
        if current == goal_idx:
            break
        closed.add(current)

        cx, cy = current
        neighbors = [
            (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1),
            (cx + 1, cy + 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx - 1, cy - 1)
        ]

        for nx, ny in neighbors:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in grid:
                dx = abs(nx - cx)
                dy = abs(ny - cy)
                cost = 1.0 if dx + dy == 1 else math.sqrt(2)
                tentative_g = g + cost
                if tentative_g < g_cost.get((nx, ny), float('inf')):
                    g_cost[(nx, ny)] = tentative_g
                    h = math.dist((nx, ny), goal_idx)
                    f_new = tentative_g + h
                    heapq.heappush(open_heap, (f_new, (nx, ny), tentative_g))
                    came_from[(nx, ny)] = current

    if goal_idx not in g_cost:
        return [], expanded

    path_indices = [goal_idx]
    current = goal_idx
    while current != start_idx:
        current = came_from[current]
        path_indices.append(current)
    path_indices.reverse()
    return path_indices, expanded


def plan_path(start: Point2D, goal: Point2D, obstacles: list[Polygon]) -> list[Point2D]:
    import logging
    logger = logging.getLogger(__name__)
    t0 = time.time()
    
    if start == goal:
        return [start]

    # 过滤局部障碍物
    dist = math.dist(start, goal)
    padding = max(dist * 0.5, 200.0)
    from shapely.geometry import box
    search_box = box(
        min(start[0], goal[0]) - padding,
        min(start[1], goal[1]) - padding,
        max(start[0], goal[0]) + padding,
        max(start[1], goal[1]) + padding
    )
    local_obstacles = [p for p in obstacles if search_box.intersects(p)]
    logger.info("[VisibilityGraph] plan_path: total_obstacles=%d local_obstacles=%d", len(obstacles), len(local_obstacles))

    # 创建 occupancy grid
    grid, min_x, min_y, width, height = _create_occupancy_grid(start, goal, local_obstacles, GRID_RESOLUTION)
    logger.info("[VisibilityGraph] plan_path: grid_size=(%d,%d) occupied_cells=%d", width, height, len(grid))

    # 转换 start 和 goal 到 grid indices
    start_i = int((start[0] - min_x) / GRID_RESOLUTION)
    start_j = int((start[1] - min_y) / GRID_RESOLUTION)
    goal_i = int((goal[0] - min_x) / GRID_RESOLUTION)
    goal_j = int((goal[1] - min_y) / GRID_RESOLUTION)

    # 确保 indices 在范围内
    start_i = max(0, min(width - 1, start_i))
    start_j = max(0, min(height - 1, start_j))
    goal_i = max(0, min(width - 1, goal_i))
    goal_j = max(0, min(height - 1, goal_j))

    start_idx = (start_i, start_j)
    goal_idx = (goal_i, goal_j)

    # 如果 start 或 goal 在 occupied cell，尝试调整（简单处理）
    if start_idx in grid:
        # 简单移到最近 free cell（这里简化，不实现复杂逻辑）
        pass  # 假设调用方确保 start/goal 不在障碍物内
    if goal_idx in grid:
        pass

    # 执行 A*
    path_indices, expanded = _grid_astar(grid, start_idx, goal_idx, width, height)
    logger.info("[VisibilityGraph] plan_path: astar_expanded_nodes=%d", expanded)

    if not path_indices:
        # 如果 A* 失败，返回直线路径
        path_indices = [start_idx, goal_idx]

    # 转换回世界坐标
    path = [(min_x + i * GRID_RESOLUTION, min_y + j * GRID_RESOLUTION) for i, j in path_indices]
    # 确保起点和终点精确
    if path:
        path[0] = start
        path[-1] = goal

    # 路径平滑
    smoothed_path = _smooth_path(path, local_obstacles)
    logger.info(
        "[GridAStar] path smoothing: raw=%d smoothed=%d",
        len(path),
        len(smoothed_path)
    )

    elapsed = time.time() - t0
    logger.info("[GridAStar] plan_path: computed path with %d waypoints, planning_time=%.3fs", len(smoothed_path), elapsed)
    return smoothed_path