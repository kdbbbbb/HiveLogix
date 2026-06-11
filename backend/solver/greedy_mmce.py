"""
HiveLogix — 基于多模式候选评估的贪心调度算法（Multi-Modal Candidate Evaluation-based Greedy Algorithm）

核心改进：
  1. 前瞻能量校验：分配无人机任务前，验证电量能支撑送达 + 安全降落到最近回收点
  2. 卡车路径规划：最近邻启发式（按时间窗紧迫度排序）+ 充电站顺路插入
  3. 无人机回收点选择：在充电站 + 仓库候选池中找到能量可行且最省电的降落点
  4. 模式优先级：B（卡-空协同）> C（仓-空直递）> A（卡车直递）

算法流程：
  Phase 1  按剩余时间窗紧迫度升序排列订单
  Phase 2  逐单贪心选模式（B → C → A），每次均执行前瞻能量校验
  Phase 3  汇总模式 A 订单，为每辆卡车构建最近邻路径（含经停充电站）
"""

from __future__ import annotations

import logging
import math
import json
import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from config.loader import load_solver_energy_params
from core.entities.primitives import DroneStatus, Position3D, SourceType, WaypointAction
from pyproj import Transformer
import osm_service as _osm_svc
import sys
import os

# 导入预设场景模块用于加载缓存的 OSM 数据
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environment", "geo"))
try:
    from preset_scenes import load_osm_from_cache
except ImportError:
    load_osm_from_cache = None

try:
    from environment.path_planning.planner import PathPlanner
except ImportError:
    PathPlanner = None

if TYPE_CHECKING:
    from core.entities.drone import Drone
    from core.entities.order import Order
    from core.entities.swap_station import SwapStation
    from core.entities.truck import Truck
    from entity_manager import EntityManager

logger = logging.getLogger(__name__)


def _bbox_cache_key(bbox: dict) -> tuple[float, float, float, float]:
    """将 bbox 归一化为稳定键，便于复用路网缓存。"""
    return (
        round(float(bbox["minx"]), 6),
        round(float(bbox["miny"]), 6),
        round(float(bbox["maxx"]), 6),
        round(float(bbox["maxy"]), 6),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationResult:
    """单条订单的分配结果。"""
    order_id: str
    vehicle_id: str              # 分配给哪个卡车 / 仓库
    mode: str                    # 'A' / 'B' / 'C' / 'B_WAIT' / 'REJECT'
    distance: float              # 关键路段距离（米）
    feasible: bool
    reason: str = ""
    recovery_station_id: str = ""      # 模式 B/C：无人机完成任务后的降落点 ID
    drone_id: str = ""                  # 实际分配的无人机 ID（模式 B/C 时填写）
    # 新增字段：支持无人机在充电站等待场景
    launch_station_id: str = ""         # 模式 B_WAIT：无人机出发的充电站 ID
    launch_time: float = 0.0            # 无人机实际起飞时间
    wait_duration: float = 0.0          # 在出发充电站等待的时间（秒）
    # 统一评分结果（用于模式间比较）
    score_total: float = math.inf
    cost_dist: float = 0.0
    cost_energy: float = 0.0
    cost_penalty: float = 0.0


@dataclass
class TruckRouteNode:
    """卡车配送路径中的单个节点。"""
    node_id: str
    node_type: str               # 'depot' / 'customer' / 'station'
    position: Position3D
    arrival_time: float = 0.0
    departure_time: float = 0.0
    order_id: str = ""           # 仅 customer 节点有效


@dataclass
class TruckRoute:
    """卡车完整行驶路径（depot → customers/stations → depot）。"""
    truck_id: str
    nodes: list[TruckRouteNode] = field(default_factory=list)
    total_distance: float = 0.0
    charging_stop_ids: list[str] = field(default_factory=list)  # 经停充电站 ID，广播给无人机
    geometry: list[Position3D] = field(default_factory=list)  # 完整路径几何（UTM 坐标序列）


@dataclass
class DroneRoute:
    """无人机飞行路径（仓库/卡车 → 配送点 → 回收点）。"""
    drone_id: str
    order_id: str
    path: list[Position3D] = field(default_factory=list)  # 完整飞行路径（WGS84 坐标序列，lon/lat）
    mode: str = ""  # 'B' / 'C'
    launch_loc: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))
    delivery_loc: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))
    recovery_loc: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))


@dataclass
class DispatchPlan:
    """全量分配方案（一个调度批次）。"""
    allocations: list[AllocationResult]
    cost_total: float
    summary: dict
    truck_routes: dict[str, TruckRoute] = field(default_factory=dict)  # truck_id → TruckRoute
    drone_routes: dict[str, DroneRoute] = field(default_factory=dict)  # drone_id → DroneRoute


# ══════════════════════════════════════════════════════════════════════════════
# 算法主体
# ══════════════════════════════════════════════════════════════════════════════

class GreedyMMCE:
    """
    贪心调度决策器 v2。

    调参常量：
      SERVICE_TIME_CUSTOMER  客户节点卸货停留时间 [s]
      MAX_DETOUR_RATIO       充电站插入可接受的绕路比（1.3 = 最多绕路 30%）
      ENERGY_SAFETY_FACTOR   能量安全系数（保留 20% 冗余）
    """

    SERVICE_TIME_CUSTOMER = 60.0   # [s]
    MAX_DETOUR_RATIO      = 1.3
    ENERGY_SAFETY_FACTOR  = 1.2
    DEPOT_LAUNCH_TOLERANCE_M = 30.0
    UAV_CRUISE_ALTITUDE_M = 80.0   # 无人机巡航高度，用于路径规划避障
    
    # 新增参数：控制无人机在充电站等待的成本权衡
    WAIT_PENALTY_FACTOR   = 0.5    # 每秒等待时间的成本系数（相对于距离）
    EARLY_ARRIVAL_PENALTY = 120.0  # 每分钟提早到达的惩罚（相对于距离）

    # 统一目标函数系数：f = f_dist + f_energy + f_penalty
    C_DIST_ET = 1.0
    C_DIST_UAV = 1.0
    C_ENERGY_ET = 1.0
    C_ENERGY_UAV = 1.0
    LAMBDA_TIME = 10000.0

    # 能耗模型参数（用于评分）
    TRUCK_ENERGY_KWH_PER_KM = 0.75  # ET: 每公里耗电 [kWh/km]
    TRUCK_ENERGY_WH_PER_METER = TRUCK_ENERGY_KWH_PER_KM  # = 0.75 Wh/m
    UAV_ALPHA_WH_PER_KG_KM = 0.24   # UAV: alpha_k [Wh/(kg·km)]

    # 业务时长参数（默认值，可由配置覆盖）
    TRUCK_DRONE_LAUNCH_TIME = 10.0
    TRUCK_DRONE_RECOVER_TIME = 10.0

    def __init__(self, entity_mgr: "EntityManager") -> None:
        self.entity_mgr = entity_mgr
        self.min_reserve_energy = 200.0   # 无人机绝对电量底线 [J]
        self.delivery_service_time = 30.0  # 配送点停留时间 [s]
        self._road_graph_cache: Optional[nx.DiGraph] = None
        self._road_nodes_cache: Optional[dict] = None
        self._road_cache_scene_id: str | None = None
        self._road_cache_bbox_key: tuple[float, float, float, float] | None = None
        self._road_distance_memo: dict[tuple[float, float, float, float], float] = {}
        self._path_planner = None
        self._path_planner_scene_id: str | None = None
        self._uav_path_distance_memo: dict[tuple[float, float, float, float, float], float] = {}

        # 评分与能耗参数统一从配置读取，便于所有算法共享。
        energy_cfg = load_solver_energy_params()
        self.C_DIST_ET = energy_cfg.c_dist_et
        self.C_DIST_UAV = energy_cfg.c_dist_uav
        self.C_ENERGY_ET = energy_cfg.c_energy_et
        self.C_ENERGY_UAV = energy_cfg.c_energy_uav
        self.LAMBDA_TIME = energy_cfg.lambda_time
        self.TRUCK_ENERGY_KWH_PER_KM = energy_cfg.truck_energy_kwh_per_km
        self.TRUCK_ENERGY_WH_PER_METER = energy_cfg.truck_energy_wh_per_meter
        self.UAV_ENERGY_MODEL = energy_cfg.uav_energy_model
        self.UAV_ALPHA_WH_PER_KG_KM = energy_cfg.uav_alpha_wh_per_kg_km
        self.ALLOW_MOVING_TRUCK_LAUNCH = energy_cfg.allow_moving_truck_launch
        self.SERVICE_TIME_CUSTOMER = energy_cfg.truck_service_time_order_s
        self.delivery_service_time = energy_cfg.drone_service_time_order_s
        self.TRUCK_DRONE_LAUNCH_TIME = energy_cfg.truck_drone_launch_time_s
        self.TRUCK_DRONE_RECOVER_TIME = energy_cfg.truck_drone_recover_time_s

    # ══════════════════════════════════════════════════════════════════════════
    # 公共接口
    # ══════════════════════════════════════════════════════════════════════════

    def dispatch(
        self,
        pending_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """
        执行一轮贪心调度。

        Phase 1  按剩余时间窗升序（最紧迫的先处理）
        Phase 2  逐单选模式（B → C → A）+ 前瞻能量校验
        Phase 3  为模式 A 订单构建卡车路径（最近邻 + 充电站插入）
        
        Args:
            pending_orders: 待分配的订单字典
            current_time: 当前仿真时刻
            bbox: 地图边界
            scene_id: 预设场景 ID，如 'default_test_4x4km'（可选）
        """
        return self._dispatch_impl(
            pending_orders,
            current_time,
            bbox,
            scene_id=scene_id,
            incremental=False,
        )

    def dispatch_incremental(
        self,
        new_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """增量调度：仅处理本轮新进入系统的订单，不全量重排既有订单。"""
        return self._dispatch_impl(
            new_orders,
            current_time,
            bbox,
            scene_id=scene_id,
            incremental=True,
        )

    def should_replan_unfinished(self) -> bool:
        """Greedy 默认启用滚动重优化。"""
        return True

    def dispatch_replan_current_state(
        self,
        replan_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """按当前实体状态重优化（全局重排 + 从当前位置起步 + 保留回收承诺）。"""
        return self._dispatch_impl(
            replan_orders,
            current_time,
            bbox,
            scene_id=scene_id,
            incremental=False,
            start_from_current_state=True,
            include_runtime_commitments=True,
            dispatch_type_override="dynamic_replan",
            log_structured_routes=True,
        )

    def get_active_contracts(self) -> list:
        """Greedy 无契约系统。"""
        return []

    def fulfill_contract(self, contract_id: str) -> None:
        """Greedy 无契约系统，no-op。"""
        return None

    def set_path_planner(self, planner) -> None:
        """允许编排层注入路径规划器，便于跨层共享场景数据。"""
        self._path_planner = planner
        self._path_planner_scene_id = None
        self._uav_path_distance_memo.clear()

    def _load_buildings_geojson(self, scene_id: str | None) -> dict | None:
        """从场景缓存加载建筑 GeoJSON。"""
        backend_dir = Path(__file__).resolve().parent.parent
        candidate_paths: list[Path] = []
        if scene_id:
            candidate_paths.append(backend_dir / "test_data" / scene_id / "no_fly_zones.geojson")
            candidate_paths.append(backend_dir / "test_data" / scene_id / "buildings.geojson")
            if scene_id == "default_test_4x4km":
                candidate_paths.append(backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson")
                candidate_paths.append(backend_dir / "test_data" / "default_scene" / "buildings.geojson")
        candidate_paths.append(backend_dir / "test_data" / "default_scene" / "no_fly_zones.geojson")
        candidate_paths.append(backend_dir / "test_data" / "default_scene" / "buildings.geojson")

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, (dict, list)):
                    num_features = len(data) if isinstance(data, list) else len(data.get("features", []))
                    logger.info("[GreedyMMCE] 已加载地图数据: %s (features=%d)", path.name, num_features)
                    return data
            except Exception:
                logger.warning("[GreedyMMCE] 建筑缓存读取失败: %s", path)
        return None

    def _activate_path_planner(self, scene_id: str | None) -> None:
        if PathPlanner is None:
            self._path_planner = None
            self._path_planner_scene_id = scene_id
            return

        if self._path_planner is not None and self._path_planner_scene_id is None:
            return

        if self._path_planner is not None and self._path_planner_scene_id == scene_id:
            return

        buildings_geojson = self._load_buildings_geojson(scene_id)
        if buildings_geojson is None:
            self._path_planner = None
            self._path_planner_scene_id = scene_id
            return

        self._path_planner = PathPlanner(buildings_geojson)
        self._path_planner_scene_id = scene_id

    def _uav_path_distance(
        self,
        from_pos: Position3D,
        to_pos: Position3D,
        altitude: float | None = None,
    ) -> float:
        if self._path_planner is None:
            return self._dist(from_pos, to_pos)

        z = float(max(from_pos.z, to_pos.z) if altitude is None else altitude)
        key = (
            round(from_pos.x, 2),
            round(from_pos.y, 2),
            round(to_pos.x, 2),
            round(to_pos.y, 2),
            round(z, 2),
        )
        cached = self._uav_path_distance_memo.get(key)
        if cached is not None:
            return cached

        path = self._path_planner.plan(from_pos, to_pos, z)
        if len(path) < 2:
            dist = self._dist(from_pos, to_pos)
        else:
            dist = sum(path[i - 1].distance_2d(path[i]) for i in range(1, len(path)))
        self._uav_path_distance_memo[key] = dist
        return dist

    def _dispatch_impl(
        self,
        orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
        incremental: bool = False,
        start_from_current_state: bool = False,
        include_runtime_commitments: bool = False,
        dispatch_type_override: str | None = None,
        log_structured_routes: bool = False,
    ) -> DispatchPlan:
        """统一调度实现，支持全量与增量两种入口。"""
        dispatch_type = dispatch_type_override or ("incremental" if incremental else "full")
        if not orders:
            return DispatchPlan(
                allocations=[],
                cost_total=0.0,
                summary={
                    "total_orders": 0,
                    "feasible": 0,
                    "modes": {},
                    "dispatch_type": dispatch_type,
                    "cost_breakdown": {"dist": 0.0, "energy": 0.0, "penalty": 0.0},
                },
            )

        # 每轮调度重置距离缓存，避免跨批次污染。
        self._road_distance_memo.clear()
        self._uav_path_distance_memo.clear()
        self._activate_path_planner(scene_id)
        road_graph, nodes = self._load_road_graph(bbox, scene_id)

        # ── Phase 1：排序 ──────────────────────────────────────────────────
        sorted_orders = sorted(
            orders.values(),
            key=lambda o: (o.deadline - current_time, o.deadline),
        )

        allocations: list[AllocationResult] = []
        mode_counter: dict[str, int] = {}
        truck_node_map: dict[str, dict] = {
            tid: {"orders": [], "recovery_stations": []}
            for tid in self.entity_mgr.trucks
        }

        # ── 追踪本次调度中已分配的无人机（防止同一个无人机被多次分配）────────
        allocated_drones: set[str] = set()

        # ── 追踪卡车最后一个接单/停留点，用于计算边际插入成本 ───────────
        truck_last_pos: dict[str, Position3D] = {
            tid: truck.get_location(current_time)
            for tid, truck in self.entity_mgr.trucks.items()
        }

        # ── Phase 2：逐单分配 ─────────────────────────────────────────────
        for order in sorted_orders:
            result = self._allocate_order(order, current_time, allocated_drones, truck_last_pos)
            allocations.append(result)
            if result.feasible:
                mode_counter[result.mode] = mode_counter.get(result.mode, 0) + 1
                if result.mode == "A" and result.vehicle_id in truck_node_map:
                    truck_node_map[result.vehicle_id]["orders"].append(order)
                    # 更新卡车虚拟停留点为订单点
                    truck_last_pos[result.vehicle_id] = order.delivery_loc
                elif result.mode in ("B", "B_WAIT", "C") and result.drone_id:
                    allocated_drones.add(result.drone_id)
                    if result.mode in ("B", "B_WAIT") and result.vehicle_id in truck_node_map:
                        truck_node_map[result.vehicle_id]["recovery_stations"].append(result.recovery_station_id)
                        if result.mode == "B_WAIT" and result.launch_station_id:
                            truck_node_map[result.vehicle_id]["recovery_stations"].append(result.launch_station_id)
                        # append-based 末端估计：将卡车尾点推进到“本单回收承诺点”。
                        recovery_station = (
                            self.entity_mgr.stations.get(result.recovery_station_id)
                            or self.entity_mgr.depots.get(result.recovery_station_id)
                        )
                        if recovery_station is not None:
                            truck_last_pos[result.vehicle_id] = recovery_station.location
                        elif result.mode == "B_WAIT" and result.launch_station_id:
                            launch_station = self.entity_mgr.stations.get(result.launch_station_id)
                            if launch_station is not None:
                                truck_last_pos[result.vehicle_id] = launch_station.location

        if include_runtime_commitments:
            self._merge_runtime_commitments(truck_node_map, orders, current_time, allocations)

        # ── Phase 3：构建卡车路径 ─────────────────────────────────────────
        # 构建 recovery station wait times 映射：station_id -> wait_duration
        recovery_station_wait_times: dict[str, float] = {}
        for alloc in allocations:
            if alloc.feasible and alloc.mode in ("B", "B_WAIT") and alloc.recovery_station_id:
                # 对于同一个站点被多个 allocation 使用的情况，取最大等待时间。
                # B_WAIT 的 wait_duration 已包含回收操作时长；B 仍需补上回收操作时长。
                if alloc.mode == "B_WAIT":
                    wait_s = max(0.0, alloc.wait_duration)
                else:
                    wait_s = max(0.0, alloc.wait_duration) + self.TRUCK_DRONE_RECOVER_TIME
                recovery_station_wait_times[alloc.recovery_station_id] = max(
                    recovery_station_wait_times.get(alloc.recovery_station_id, 0),
                    wait_s,
                )

        truck_routes: dict[str, TruckRoute] = {}
        for truck_id, node_data in truck_node_map.items():
            if not node_data["orders"] and not node_data["recovery_stations"]:
                continue
            truck = self.entity_mgr.trucks[truck_id]
            unique_recovery_stations = list(set(node_data["recovery_stations"]))
            unique_orders = list({o.order_id: o for o in node_data["orders"]}.values())
            if incremental:
                truck_routes[truck_id] = self._build_truck_route(
                    truck,
                    unique_orders,
                    unique_recovery_stations,
                    current_time,
                    road_graph,
                    nodes,
                    recovery_station_wait_times,
                    start_pos=truck.get_location(current_time),
                    return_to_depot=False,
                )
            elif start_from_current_state:
                truck_routes[truck_id] = self._build_truck_route(
                    truck,
                    unique_orders,
                    unique_recovery_stations,
                    current_time,
                    road_graph,
                    nodes,
                    recovery_station_wait_times,
                    start_pos=truck.get_location(current_time),
                    return_to_depot=True,
                )
            else:
                truck_routes[truck_id] = self._build_truck_route(
                    truck,
                    unique_orders,
                    unique_recovery_stations,
                    current_time,
                    road_graph,
                    nodes,
                    recovery_station_wait_times,
                )

        cost_dist_total, cost_energy_total, cost_total = self._recalculate_plan_route_costs(
            allocations,
            truck_routes,
            current_time,
            orders,
        )
        cost_penalty_total = sum(r.cost_penalty for r in allocations if r.feasible)

        plan = DispatchPlan(
            allocations=allocations,
            cost_total=cost_total,
            summary={
                "total_orders": len(allocations),
                "feasible": sum(1 for r in allocations if r.feasible),
                "modes": mode_counter,
                "dispatch_type": dispatch_type,
                "cost_breakdown": {
                    "dist": cost_dist_total,
                    "energy": cost_energy_total,
                    "penalty": cost_penalty_total,
                },
            },
            truck_routes=truck_routes,
        )
        logger.info("[GreedyMMCE] 分配完成：%s", plan.summary)
        if log_structured_routes and truck_routes:
            self._log_structured_truck_routes(truck_routes, allocations)
        return plan

    def _merge_runtime_commitments(
        self,
        truck_node_map: dict[str, dict],
        orders: dict[str, "Order"],
        current_time: float,
        allocations: list[AllocationResult],
    ) -> None:
        """将运行中的刚性承诺并入重优化路由（在途订单 + 无人机回收锚点）。"""
        customer_by_truck: dict[str, set[str]] = {tid: set() for tid in truck_node_map}
        recovery_by_truck: dict[str, set[str]] = {tid: set() for tid in truck_node_map}
        feasible_alloc_by_order: dict[str, AllocationResult] = {
            alloc.order_id: alloc
            for alloc in allocations
            if alloc.feasible
        }

        # 保留当前卡车未来停靠承诺，避免重优化丢失既有回收/配送节点。
        for truck_id, truck in self.entity_mgr.trucks.items():
            if truck_id not in truck_node_map:
                continue
            planned_stops = getattr(truck, "_planned_route_stops", None) or []
            cursor = int(getattr(truck, "_planned_route_cursor", 0) or 0)
            for stop in planned_stops[cursor:]:
                node_type = stop.get("node_type", "")
                if node_type == "customer":
                    oid = stop.get("order_id", "")
                    if oid:
                        customer_by_truck[truck_id].add(oid)
                elif node_type == "recovery":
                    sid = stop.get("node_id", "")
                    if sid:
                        recovery_by_truck[truck_id].add(sid)

        # 从无人机实时状态提取回收锚点（待起飞、在飞、已到站待回收）。
        for drone in self.entity_mgr.drones.values():
            owner_truck_id = self._resolve_drone_owner_truck_id(drone)
            if not owner_truck_id or owner_truck_id not in truck_node_map:
                continue

            waiting_station_id = getattr(drone, "waiting_recovery_station_id", "")
            if waiting_station_id and waiting_station_id in self.entity_mgr.stations:
                recovery_by_truck[owner_truck_id].add(waiting_station_id)

            launch_station_id = getattr(drone, "launch_station_id", "")
            scheduled_launch_time = float(getattr(drone, "scheduled_launch_time", 0.0) or 0.0)
            transport_truck_id = getattr(drone, "transport_truck_id", "")
            if (
                transport_truck_id == owner_truck_id
                and launch_station_id
                and launch_station_id in self.entity_mgr.stations
                and math.isfinite(scheduled_launch_time)
                and scheduled_launch_time >= current_time - 1e-6
            ):
                recovery_by_truck[owner_truck_id].add(launch_station_id)

            pending_station_id = self._get_pending_station_recovery_from_route(drone)
            if pending_station_id:
                recovery_by_truck[owner_truck_id].add(pending_station_id)

        for truck_id, order_ids in customer_by_truck.items():
            if not order_ids:
                continue
            existing_ids = {o.order_id for o in truck_node_map[truck_id]["orders"]}
            for oid in order_ids:
                alloc = feasible_alloc_by_order.get(oid)
                if alloc is not None:
                    # 若本轮已重分配为非 A，或 A 但已改派给其它卡车，
                    # 则不能继续保留旧 customer 承诺，避免同单被卡车+无人机重复履约。
                    if alloc.mode != "A" or alloc.vehicle_id != truck_id:
                        continue
                if oid in existing_ids:
                    continue
                order_obj = orders.get(oid)
                if order_obj is not None:
                    truck_node_map[truck_id]["orders"].append(order_obj)
                    existing_ids.add(oid)

        for truck_id, station_ids in recovery_by_truck.items():
            if station_ids:
                truck_node_map[truck_id]["recovery_stations"].extend(station_ids)

    def _resolve_drone_owner_truck_id(self, drone: "Drone") -> str:
        """解析无人机当前应归属的卡车，用于重优化时绑定回收承诺。"""
        transport_truck_id = getattr(drone, "transport_truck_id", "")
        if transport_truck_id in self.entity_mgr.trucks:
            return transport_truck_id

        # 反查停靠关系：即便 transport_truck_id 暂未写回，也可从车载列表恢复归属。
        for truck_id, truck in self.entity_mgr.trucks.items():
            if drone.drone_id in getattr(truck, "docked_drones", []):
                return truck_id

        home_type = getattr(drone, "home_type", None)
        home_id = getattr(drone, "home_id", "")
        if home_type == SourceType.TRUCK and home_id in self.entity_mgr.trucks:
            return home_id

        # 单车场景兜底：放飞后 transport_truck_id 会清空，
        # 若系统仅 1 辆卡车，则可安全绑定到唯一卡车，避免回收承诺丢失。
        if len(self.entity_mgr.trucks) == 1:
            return next(iter(self.entity_mgr.trucks.keys()))

        return ""

    def _get_pending_station_recovery_from_route(self, drone: "Drone") -> str:
        """从无人机剩余航路中提取待执行的站点回收锚点。"""
        route_plan = getattr(drone, "route_plan", None) or []
        if not route_plan:
            return ""

        start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
        start_idx = max(0, min(start_idx, len(route_plan)))
        for wp in reversed(route_plan[start_idx:]):
            if wp.action not in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                continue
            target_id = wp.target_entity_id or ""
            if target_id in self.entity_mgr.stations:
                return target_id
            return ""
        return ""

    def _log_structured_truck_routes(
        self,
        truck_routes: dict[str, TruckRoute],
        allocations: list[AllocationResult],
    ) -> None:
        """输出结构化卡车路线日志，便于动态重优化诊断与前端联动核验。"""
        for truck_id, route in truck_routes.items():
            logger.info(
                "[GreedyMMCE] 动态重优化卡车 %s 路径 %d 节点，里程 %.0fm，经停充电站 %s",
                truck_id,
                len(route.nodes),
                route.total_distance,
                route.charging_stop_ids or "（无）",
            )
            logger.info("[GreedyMMCE] 动态重优化卡车 %s 详细路线：", truck_id)
            for idx, node in enumerate(route.nodes, start=1):
                logger.info(
                    "[GreedyMMCE]   [%d] %s (%s) - 到达: %.1fs, 离开: %.1fs | %s",
                    idx,
                    node.node_id,
                    node.node_type,
                    node.arrival_time,
                    node.departure_time,
                    self._describe_route_node_action(node, allocations),
                )

    def _describe_route_node_action(
        self,
        node: TruckRouteNode,
        allocations: list[AllocationResult],
    ) -> str:
        """根据节点类型与分配结果生成路线动作描述。"""
        if node.node_type == "depot":
            if "_return" in node.node_id:
                return "返回仓库"
            return "从仓库出发"
        if node.node_type == "origin":
            return "从当前位置续行"
        if node.node_type == "customer":
            return f"配送订单 {node.order_id}"
        if node.node_type == "station":
            return "经停充电站（广播给无人机）"
        if node.node_type == "recovery":
            launch_orders = [
                alloc.order_id
                for alloc in allocations
                if alloc.feasible and alloc.mode in ("B_WAIT", "B_DYNAMIC") and alloc.launch_station_id == node.node_id
            ]
            recovery_orders = [
                alloc.order_id
                for alloc in allocations
                if alloc.feasible and alloc.mode in ("B", "B_WAIT", "B_DYNAMIC") and alloc.recovery_station_id == node.node_id
            ]
            if launch_orders and recovery_orders:
                return (
                    f"放飞+回收无人机（放飞订单: {', '.join(launch_orders)}；"
                    f"回收订单: {', '.join(recovery_orders)}）"
                )
            if launch_orders:
                return f"放飞无人机（订单: {', '.join(launch_orders)}）"
            if recovery_orders:
                return f"回收无人机（订单: {', '.join(recovery_orders)}）"
            return "回收锚点停靠"
        return "未知动作"

    def _load_road_graph(
        self,
        bbox: dict,
        scene_id: str | None,
    ) -> tuple[nx.DiGraph, dict]:
        """加载或复用 OSM 路网缓存，减少动态调度重复开销。"""
        bbox_key = _bbox_cache_key(bbox)
        if (
            self._road_graph_cache is not None
            and self._road_nodes_cache is not None
            and self._road_cache_bbox_key == bbox_key
            and self._road_cache_scene_id == scene_id
        ):
            return self._road_graph_cache, self._road_nodes_cache

        road_graph = None
        nodes = None

        if scene_id and load_osm_from_cache:
            logger.info("[GreedyMMCE] 尝试从预设场景 '%s' 缓存加载 OSM...", scene_id)
            osm_xml, _ = load_osm_from_cache(scene_id)
            if osm_xml:
                try:
                    road_graph, nodes = _osm_svc.build_road_graph(osm_xml)
                    logger.info(
                        "[GreedyMMCE] 从缓存加载成功: %d 个节点，%d 条边",
                        len(nodes),
                        len(road_graph.edges),
                    )
                except Exception as e:
                    logger.warning("[GreedyMMCE] 从缓存构建失败: %s，将重新下载", e)
                    road_graph = None

        if road_graph is None:
            logger.info("[GreedyMMCE] 下载 OSM 路网数据...")
            try:
                osm_xml = _osm_svc.download_osm(
                    bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
                )
                road_graph, nodes = _osm_svc.build_road_graph(osm_xml)
                logger.info(
                    "[GreedyMMCE] 构建路网图：%d 个节点，%d 条边",
                    len(nodes),
                    len(road_graph.edges),
                )
            except Exception as e:
                raise RuntimeError(f"OSM 路网下载/构建失败，已终止调度: {e}") from e

        self._road_graph_cache = road_graph
        self._road_nodes_cache = nodes
        self._road_cache_scene_id = scene_id
        self._road_cache_bbox_key = bbox_key
        return road_graph, nodes

    def _recalculate_plan_route_costs(
        self,
        allocations: list[AllocationResult],
        truck_routes: dict[str, TruckRoute],
        current_time: float,
        orders_by_id: dict[str, "Order"],
    ) -> tuple[float, float, float]:
        """按最终执行路径重算距离与能耗成本，避免局部评分与全局执行不一致。"""
        truck_distance_total = sum(route.total_distance for route in truck_routes.values())
        truck_energy_total = sum(
            self._truck_energy_wh(route.total_distance)
            for route in truck_routes.values()
        )

        uav_distance_total = 0.0
        uav_energy_total = 0.0

        for alloc in allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if drone is None:
                continue

            order_obj = orders_by_id.get(alloc.order_id)
            if order_obj is None:
                continue

            if alloc.mode == "B_WAIT":
                launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
                recovery = (
                    self.entity_mgr.stations.get(alloc.recovery_station_id)
                    or self.entity_mgr.depots.get(alloc.recovery_station_id)
                )
                if launch_station is None or recovery is None:
                    continue
                launch_loc = launch_station.location
                recovery_loc = recovery.location
            elif alloc.mode == "B":
                truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                recovery = (
                    self.entity_mgr.stations.get(alloc.recovery_station_id)
                    or self.entity_mgr.depots.get(alloc.recovery_station_id)
                )
                if truck is None or recovery is None:
                    continue
                launch_loc = truck.get_location(current_time)
                recovery_loc = recovery.location
            elif alloc.mode == "C":
                depot = self.entity_mgr.depots.get(alloc.vehicle_id)
                if depot is None:
                    continue
                launch_loc = depot.location
                recovery_loc = depot.location
            else:
                continue

            dist_out = self._uav_path_distance(launch_loc, order_obj.delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            dist_back = self._uav_path_distance(order_obj.delivery_loc, recovery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            uav_distance_total += dist_out + dist_back
            uav_energy_total += self._uav_energy_wh(
                drone, launch_loc, order_obj.delivery_loc, order_obj.payload_weight
            )
            uav_energy_total += self._uav_energy_wh(
                drone, order_obj.delivery_loc, recovery_loc, 0.0
            )

        cost_dist_total = self.C_DIST_ET * truck_distance_total + self.C_DIST_UAV * uav_distance_total
        cost_energy_total = self.C_ENERGY_ET * truck_energy_total + self.C_ENERGY_UAV * uav_energy_total
        return cost_dist_total, cost_energy_total, cost_dist_total + cost_energy_total

    def _road_dist(self, pos_a: Position3D, pos_b: Position3D) -> float:
        """优先使用 OSM 路径距离；不可用时回退二维欧氏距离。"""
        if self._road_graph_cache is None or self._road_nodes_cache is None:
            return self._dist(pos_a, pos_b)
        if len(self._road_graph_cache.nodes()) == 0:
            return self._dist(pos_a, pos_b)

        key = (
            round(pos_a.x, 1),
            round(pos_a.y, 1),
            round(pos_b.x, 1),
            round(pos_b.y, 1),
        )
        cached = self._road_distance_memo.get(key)
        if cached is not None:
            return cached

        try:
            graph = self._road_graph_cache
            nodes = self._road_nodes_cache
            start_node = _osm_svc.find_nearest_node(graph, nodes, pos_a.x, pos_a.y)
            end_node = _osm_svc.find_nearest_node(graph, nodes, pos_b.x, pos_b.y)
            if not start_node or not end_node:
                dist = self._dist(pos_a, pos_b)
            elif start_node == end_node:
                dist = self._dist(pos_a, pos_b)
            else:
                path = _osm_svc.shortest_path(graph, start_node, end_node)
                if not path:
                    dist = self._dist(pos_a, pos_b)
                else:
                    dist = 0.0
                    for i in range(len(path) - 1):
                        dist += graph[path[i]][path[i + 1]]["weight"]
        except Exception:
            dist = self._dist(pos_a, pos_b)

        self._road_distance_memo[key] = dist
        return dist

    def build_incremental_route_from_stops(
        self,
        truck: "Truck",
        ordered_stops: list[dict],
        current_time: float,
    ) -> Optional[TruckRoute]:
        """按给定停靠顺序重建增量后缀路线（严格走 OSM）。"""
        if self._road_graph_cache is None or self._road_nodes_cache is None:
            return None

        road_graph = self._road_graph_cache
        nodes = self._road_nodes_cache
        if len(road_graph.nodes()) == 0:
            return None

        route = TruckRoute(truck_id=truck.truck_id)
        start_pos = truck.get_location(current_time)
        route.nodes.append(
            TruckRouteNode(
                node_id=f"{truck.truck_id}_origin",
                node_type="origin",
                position=start_pos,
                arrival_time=current_time,
                departure_time=current_time,
            )
        )
        route.geometry.append(start_pos)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651")
        nearest_node_cache: dict[tuple[float, float], object] = {}
        segment_cache: dict[tuple[object, object], tuple[float, list[Position3D]]] = {}

        def get_nearest_node(pos: Position3D):
            key = (round(pos.x, 2), round(pos.y, 2))
            node = nearest_node_cache.get(key)
            if node is None:
                node = _osm_svc.find_nearest_node(road_graph, nodes, pos.x, pos.y)
                nearest_node_cache[key] = node
            return node

        def calc_dist_and_geometry(pos1: Position3D, pos2: Position3D) -> tuple[float, list[Position3D]]:
            start_node = get_nearest_node(pos1)
            end_node = get_nearest_node(pos2)
            if not start_node or not end_node:
                return pos1.distance_2d(pos2), [pos1, pos2]

            if start_node == end_node:
                return pos1.distance_2d(pos2), [pos1, pos2]

            cache_key = (start_node, end_node)
            cached = segment_cache.get(cache_key)
            if cached is not None:
                core_dist, core_geometry = cached
                seg_geometry: list[Position3D] = [pos1]
                for p in core_geometry:
                    if seg_geometry[-1].distance_2d(p) > 0.5:
                        seg_geometry.append(p)
                if seg_geometry[-1].distance_2d(pos2) > 0.5:
                    seg_geometry.append(pos2)
                seg_dist = sum(
                    seg_geometry[i - 1].distance_2d(seg_geometry[i])
                    for i in range(1, len(seg_geometry))
                )
                return seg_dist, seg_geometry

            path = _osm_svc.shortest_path(road_graph, start_node, end_node)
            if not path:
                return pos1.distance_2d(pos2), [pos1, pos2]

            dist = 0.0
            segment_geometry: list[Position3D] = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                dist += road_graph[u][v]["weight"]
                if not segment_geometry:
                    lon_u, lat_u = nodes[u]
                    x_u, y_u = transformer.transform(lat_u, lon_u)
                    segment_geometry.append(Position3D(x=x_u, y=y_u, z=0))
                lon_v, lat_v = nodes[v]
                x_v, y_v = transformer.transform(lat_v, lon_v)
                segment_geometry.append(Position3D(x=x_v, y=y_v, z=0))

            segment_cache[cache_key] = (dist, segment_geometry)

            seg_geometry: list[Position3D] = [pos1]
            for p in segment_geometry:
                if seg_geometry[-1].distance_2d(p) > 0.5:
                    seg_geometry.append(p)
            if seg_geometry[-1].distance_2d(pos2) > 0.5:
                seg_geometry.append(pos2)
            seg_dist = sum(
                seg_geometry[i - 1].distance_2d(seg_geometry[i])
                for i in range(1, len(seg_geometry))
            )
            return seg_dist, seg_geometry

        cur_pos = start_pos
        cur_time = current_time
        total_dist = 0.0
        speed = max(1e-6, float(getattr(truck, "speed", 0.0)))

        for stop in ordered_stops:
            stop_pos = stop.get("position")
            node_id = stop.get("node_id")
            node_type = stop.get("node_type")
            if stop_pos is None or not node_id or not node_type:
                continue

            seg_dist, seg_geom = calc_dist_and_geometry(cur_pos, stop_pos)
            if seg_geom:
                route.geometry.extend(seg_geom[1:] if route.geometry else seg_geom)

            arrival = cur_time + seg_dist / speed
            prev_arrival = float(stop.get("arrival_time", arrival))
            prev_departure = float(stop.get("departure_time", prev_arrival))
            service_time = max(0.0, prev_departure - prev_arrival)
            departure = arrival + service_time

            route.nodes.append(
                TruckRouteNode(
                    node_id=node_id,
                    node_type=node_type,
                    position=stop_pos,
                    arrival_time=arrival,
                    departure_time=departure,
                    order_id=stop.get("order_id", ""),
                )
            )

            if node_type == "station":
                route.charging_stop_ids.append(node_id)

            total_dist += seg_dist
            cur_pos = stop_pos
            cur_time = departure

        route.total_distance = total_dist
        return route

    # ══════════════════════════════════════════════════════════════════════════
    # 单订单分配
    # ══════════════════════════════════════════════════════════════════════════

    def _try_mode_b_with_waiting(self, order: "Order", current_time: float, allocated_drones: set[str], truck_last_pos: dict[str, Position3D]) -> AllocationResult:
        """
        改进的模式 B（无人机在充电站等待）：
        无人机在卡车将要经过的充电站上等待 → 起飞 → 配送 → 回到充电站降落。

        优点：
          - 无人机在车上等待期间可被卡车供电（省电）
          - 减少无人机的单独飞行时间
          - 更符合现实的卡-空协同场景

        策略：
          1. 获取可用无人机
          2. 对每辆卡车，遍历其预测经过的充电站
          3. 从各充电站出发评估方案，选择最佳的(距离+等待成本最低)
          4. 返回全局最优方案
        """
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="B_WAIT",
                distance=float("inf"), feasible=False, reason="无可用卡车",
            )

        # 获取未被分配过的可用无人机
        available_drones = self._get_available_drones()
        available_drones = [d for d in available_drones if d.drone_id not in allocated_drones]
        drone = self._find_capable_drone(order.payload_weight, available_drones)
        if drone is None:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="B_WAIT",
                distance=float("inf"), feasible=False, reason="无未分配的载重匹配无人机",
            )

        recovery_pool = self._get_recovery_pool()
        best_scenario = None
        best_truck_id = None

        # 尝试每辆卡车的充电站
        for truck in trucks:
            # 预测该卡车会经过的充电站
            predicted_stations = self._predict_truck_charging_stations(truck, current_time)

            for station_id, station_loc in predicted_stations:
                truck_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)
                truck_distance_to_launch = self._road_dist(truck_loc, station_loc)
                launch_delay = (
                    truck_distance_to_launch / truck.speed
                    if truck.speed > 0 else float("inf")
                )

                # 评估从该充电站出发的方案
                scenario = self._evaluate_charging_station_departure(
                    drone=drone,
                    truck_tail_loc=truck_loc,
                    launch_loc=station_loc,
                    launch_station_id=station_id,
                    truck_distance_to_launch=truck_distance_to_launch,
                    launch_delay=launch_delay,
                    delivery_loc=order.delivery_loc,
                    payload=order.payload_weight,
                    recovery_pool=recovery_pool,
                    current_time=current_time,
                    order=order,
                )

                if scenario and scenario["feasible"]:
                    # 该方案可行，比较得分
                    if best_scenario is None or scenario["score"] < best_scenario["score"]:
                        best_scenario = scenario
                        best_truck_id = truck.truck_id

        # 返回最佳方案
        if best_scenario:
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id=best_truck_id,
                mode="B_WAIT",
                distance=best_scenario["distance"],
                feasible=True,
                recovery_station_id=best_scenario["recovery_station_id"],
                drone_id=drone.drone_id,
                launch_station_id=best_scenario["launch_station_id"],
                launch_time=best_scenario["launch_time"],
                wait_duration=best_scenario["wait_duration"],
                score_total=best_scenario["score"],
                cost_dist=best_scenario["cost_dist"],
                cost_energy=best_scenario["cost_energy"],
                cost_penalty=best_scenario["cost_penalty"],
            )

        # 找不到可行方案
        return AllocationResult(
            order_id=order.order_id, vehicle_id="", mode="B_WAIT",
            distance=float("inf"), feasible=False,
            reason="无可行充电站方案（能量或距离不可行）",
        )

    # ══════════════════════════════════════════════════════════════════════════

    def _allocate_order(self, order: "Order", current_time: float, allocated_drones: set[str], truck_last_pos: dict[str, Position3D]) -> AllocationResult:
        """在可行候选中选择统一评分最小的方案。
        
        Args:
            order: 待分配订单
            current_time: 当前仿真时刻
            allocated_drones: 本次调度中已分配的无人机ID集合（用于防止重复分配）
            truck_last_pos: 卡车最后停留位置字典，用于通过边际增量计算路径评估成本
        """
        candidates: list[AllocationResult] = []

        try_fns = [self._try_mode_b_with_waiting, self._try_mode_c, self._try_mode_a]
        if self.ALLOW_MOVING_TRUCK_LAUNCH:
            try_fns.insert(1, self._try_mode_b)

        for try_fn in try_fns:
            result = try_fn(order, current_time, allocated_drones, truck_last_pos)
            if not result.feasible:
                continue

            score_total, cost_dist, cost_energy, cost_penalty = self._score_allocation(
                result, order, current_time, truck_last_pos
            )
            result.score_total = score_total
            result.cost_dist = cost_dist
            result.cost_energy = cost_energy
            result.cost_penalty = cost_penalty
            candidates.append(result)

        if candidates:
            best = min(candidates, key=lambda r: r.score_total)
            logger.info(
                "[GreedyMMCE] 订单 %s 候选评分: %s | 选中=%s score=%.2f (dist=%.2f, energy=%.2f, penalty=%.2f)",
                order.order_id,
                [
                    {
                        "mode": c.mode,
                        "score": round(c.score_total, 2),
                        "dist": round(c.cost_dist, 2),
                        "energy": round(c.cost_energy, 2),
                        "penalty": round(c.cost_penalty, 2),
                    }
                    for c in candidates
                ],
                best.mode,
                best.score_total,
                best.cost_dist,
                best.cost_energy,
                best.cost_penalty,
            )
            return best

        return AllocationResult(
            order_id=order.order_id,
            vehicle_id="",
            mode="REJECT",
            distance=float("inf"),
            feasible=False,
            reason="无可用资源（无卡车、无人机或电量不足）",
        )

    def _try_mode_b(self, order: "Order", current_time: float, allocated_drones: set[str], truck_last_pos: dict[str, Position3D]) -> AllocationResult:
        """
        模式 B（卡-空协同）：卡车当前位置起飞 → 送达 → 最近合法回收点降落。

        前瞻能量校验：drone.battery_current ≥
          (飞到配送点能耗 +飞到最近回收点能耗) × ENERGY_SAFETY_FACTOR
        """
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="B",
                distance=float("inf"), feasible=False, reason="无可用卡车",
            )

        # 获取未被分配过的可用无人机
        available_drones = self._get_available_drones()
        available_drones = [d for d in available_drones if d.drone_id not in allocated_drones]
        drone = self._find_capable_drone(
            order.payload_weight, available_drones
        )
        if drone is None:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="B",
                distance=float("inf"), feasible=False, reason="无未分配的载重匹配无人机",
            )

        recovery_pool = self._get_recovery_pool()

        # 按距离从近到远尝试卡车，找第一个能量可行的
        for truck in sorted(
            trucks,
            key=lambda t: self._dist(order.delivery_loc, truck_last_pos.get(t.truck_id) or t.get_location(current_time)),
        ):
            launch_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)
            recovery_id = self._check_energy_feasible(
                drone=drone,
                launch_loc=launch_loc,
                delivery_loc=order.delivery_loc,
                payload=order.payload_weight,
                recovery_pool=recovery_pool,
            )
            if recovery_id is None:
                continue

            return AllocationResult(
                order_id=order.order_id,
                vehicle_id=truck.truck_id,
                mode="B",
                distance=self._uav_path_distance(launch_loc, order.delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M),
                feasible=True,
                recovery_station_id=recovery_id,
                drone_id=drone.drone_id,
            )

        return AllocationResult(
            order_id=order.order_id, vehicle_id="", mode="B",
            distance=float("inf"), feasible=False,
            reason="无人机电量不足以完成送达并安全回收",
        )

    def _try_mode_c(self, order: "Order", current_time: float, allocated_drones: set[str], truck_last_pos: dict[str, Position3D]) -> AllocationResult:
        """
        模式 C（仓-空直递）：仓库 → 无人机 → 送达 → 仓库（往返）。

        前瞻能量校验：drone.battery_current ≥
          (仓库→配送点 + 配送点→仓库) × ENERGY_SAFETY_FACTOR
        """
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="C",
                distance=float("inf"), feasible=False, reason="无可用仓库",
            )

        depot = depots[0]
        # 获取未被分配过的可用无人机
        available_drones = self._get_available_drones()
        available_drones = [d for d in available_drones if d.drone_id not in allocated_drones]
        available_drones = [
            d for d in available_drones
            if self._is_drone_ready_for_depot_launch(d, depot)
        ]
        drone = self._find_capable_drone(
            order.payload_weight, available_drones
        )
        if drone is None:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="C",
                distance=float("inf"), feasible=False,
                reason="无位于仓库且载重匹配的可用无人机",
            )

        energy_out  = self._flight_energy(drone, depot.location, order.delivery_loc, order.payload_weight)
        energy_back = self._flight_energy(drone, order.delivery_loc, depot.location, 0.0)
        energy_needed = (energy_out + energy_back) * self.ENERGY_SAFETY_FACTOR

        if energy_needed > drone.battery_current:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="C",
                distance=float("inf"), feasible=False,
                reason=(
                    f"仓-空往返电量不足（需 {energy_needed:.0f} J，"
                    f"剩余 {drone.battery_current:.0f} J）"
                ),
            )

        return AllocationResult(
            order_id=order.order_id,
            vehicle_id=depot.depot_id,
            mode="C",
            distance=self._uav_path_distance(depot.location, order.delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M),
            feasible=True,
            recovery_station_id=depot.depot_id,
            drone_id=drone.drone_id,
        )

    def _try_mode_a(self, order: "Order", current_time: float, allocated_drones: set[str], truck_last_pos: dict[str, Position3D]) -> AllocationResult:
        """模式 A（卡车直递）：按 append-based 估计代价选择卡车。"""
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return AllocationResult(
                order_id=order.order_id, vehicle_id="", mode="A",
                distance=float("inf"), feasible=False, reason="无可用卡车",
            )

        nearest = min(
            trucks,
            key=lambda t: self._estimate_mode_a_append_distance(
                truck_last_pos.get(t.truck_id) or t.get_location(current_time),
                order.delivery_loc,
            ),
        )
        tail_pos = truck_last_pos.get(nearest.truck_id) or nearest.get_location(current_time)
        append_distance = self._estimate_mode_a_append_distance(tail_pos, order.delivery_loc)
        return AllocationResult(
            order_id=order.order_id,
            vehicle_id=nearest.truck_id,
            mode="A",
            distance=append_distance,
            feasible=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 卡车路径规划（最近邻 + 充电站顺路插入）
    # ══════════════════════════════════════════════════════════════════════════

    def _build_truck_route(
        self,
        truck: "Truck",
        orders: list,
        recovery_station_ids: list[str],
        current_time: float,
        road_graph: Optional[nx.DiGraph],
        nodes: dict,
        recovery_station_wait_times: Optional[dict[str, float]] = None,
        start_pos: Optional[Position3D] = None,
        return_to_depot: bool = True,
    ) -> TruckRoute:
        """
        为卡车构建配送路径（最近邻启发式）。

        流程：
          1. 从仓库出发
          2. 合并客户订单和回收点作为待访问节点
          3. 每步贪心选取最近的未访问节点
          4. 每段路径上若有充电站满足绕路比 ≤ MAX_DETOUR_RATIO，则顺路插入
             （插入的充电站 ID 写入 charging_stop_ids，广播给无人机作为回收候选）
          5. 返回仓库

        卡车到达充电站时不停留（短暂经停，无人机在此降落等待回收）。
        """
        route = TruckRoute(truck_id=truck.truck_id)
        depots   = list(self.entity_mgr.depots.values())
        depot    = depots[0] if depots else None
        stations = list(self.entity_mgr.stations.values())
        required_station_ids = {
            station_id
            for station_id in recovery_station_ids
            if station_id in self.entity_mgr.stations
        }
        station_candidates = [
            station
            for station in stations
            if station.station_id in required_station_ids
        ]

        route_start_pos = start_pos if start_pos is not None else (depot.location if depot else truck.get_location(current_time))
        if start_pos is None and depot:
            route.nodes.append(TruckRouteNode(
                node_id=depot.depot_id, node_type="depot",
                position=depot.location,
                arrival_time=current_time, departure_time=current_time,
            ))
            route.geometry.append(depot.location)
        else:
            route.nodes.append(TruckRouteNode(
                node_id=f"{truck.truck_id}_origin",
                node_type="origin",
                position=route_start_pos,
                arrival_time=current_time,
                departure_time=current_time,
            ))
            route.geometry.append(route_start_pos)

        # 合并待访问节点：客户订单 + 回收点
        unvisited_nodes = []
        for order in orders:
            unvisited_nodes.append({
                "type": "customer",
                "id": order.order_id,
                "pos": order.delivery_loc,
                "order_id": order.order_id,
            })
        for station_id in recovery_station_ids:
            station = self.entity_mgr.stations.get(station_id) or self.entity_mgr.depots.get(station_id)
            if station:
                unvisited_nodes.append({
                    "type": "recovery",
                    "id": station_id,
                    "pos": station.location,
                    "order_id": "",
                })

        if not unvisited_nodes:
            return route

        cur_pos    = route_start_pos
        cur_time   = current_time
        total_dist = 0.0
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651")
        nearest_node_cache: dict[tuple[float, float], object] = {}
        segment_cache: dict[tuple[object, object], tuple[float, list[Position3D]]] = {}

        def get_nearest_node(pos: Position3D):
            key = (round(pos.x, 2), round(pos.y, 2))
            node = nearest_node_cache.get(key)
            if node is None:
                node = _osm_svc.find_nearest_node(road_graph, nodes, pos.x, pos.y)
                nearest_node_cache[key] = node
            return node

        def calc_dist_and_geometry(pos1: Position3D, pos2: Position3D) -> tuple[float, list[Position3D]]:
            """计算距离并返回对应几何（UTM）；若 OSM 路段不可达，则回退至直线距离。"""
            if road_graph is None or len(road_graph.nodes()) == 0:
                raise RuntimeError("OSM 路网为空，无法进行卡车路径规划")

            start_node = get_nearest_node(pos1)
            end_node = get_nearest_node(pos2)
            if not start_node or not end_node:
                return pos1.distance_2d(pos2), [pos1, pos2]

            # 起终点吸附到同一 OSM 节点时，视为可达的局部短段（避免误判“不可达”）
            if start_node == end_node:
                return pos1.distance_2d(pos2), [pos1, pos2]

            cache_key = (start_node, end_node)
            cached = segment_cache.get(cache_key)
            if cached is not None:
                core_dist, core_geometry = cached
                seg_geometry: list[Position3D] = [pos1]
                for p in core_geometry:
                    if seg_geometry[-1].distance_2d(p) > 0.5:
                        seg_geometry.append(p)
                if seg_geometry[-1].distance_2d(pos2) > 0.5:
                    seg_geometry.append(pos2)
                seg_dist = sum(
                    seg_geometry[i - 1].distance_2d(seg_geometry[i])
                    for i in range(1, len(seg_geometry))
                )
                return seg_dist, seg_geometry

            path = _osm_svc.shortest_path(road_graph, start_node, end_node)
            if not path:
                return pos1.distance_2d(pos2), [pos1, pos2]

            dist = 0.0
            segment_geometry: list[Position3D] = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                dist += road_graph[u][v]['weight']
                if not segment_geometry:
                    lon_u, lat_u = nodes[u]
                    x_u, y_u = transformer.transform(lat_u, lon_u)
                    segment_geometry.append(Position3D(x=x_u, y=y_u, z=0))
                lon_v, lat_v = nodes[v]
                x_v, y_v = transformer.transform(lat_v, lon_v)
                segment_geometry.append(Position3D(x=x_v, y=y_v, z=0))

            segment_cache[cache_key] = (dist, segment_geometry)

            seg_geometry: list[Position3D] = [pos1]
            for p in segment_geometry:
                if seg_geometry[-1].distance_2d(p) > 0.5:
                    seg_geometry.append(p)
            if seg_geometry[-1].distance_2d(pos2) > 0.5:
                seg_geometry.append(pos2)
            seg_dist = sum(
                seg_geometry[i - 1].distance_2d(seg_geometry[i])
                for i in range(1, len(seg_geometry))
            )
            return seg_dist, seg_geometry

        def calc_dist(pos1: Position3D, pos2: Position3D) -> float:
            """仅返回距离，供贪心比较与站点插入判断使用。"""
            dist, _ = calc_dist_and_geometry(pos1, pos2)
            return dist

        while unvisited_nodes:
            # 贪心：选距当前位置最近的未访问节点
            next_node = min(unvisited_nodes, key=lambda n: calc_dist(cur_pos, n["pos"]))
            unvisited_nodes.remove(next_node)
            next_pos = next_node["pos"]

            # 仅在“存在明确无人机站点需求”时，尝试顺路插入相关充电站。
            station = None
            if station_candidates:
                station = self._find_insertable_station(
                    cur_pos, next_pos, station_candidates, route.charging_stop_ids, calc_dist
                )
            # 若待访问节点本身就是该充电站，不再额外插入同站 station 节点，避免重复停靠。
            if station is not None and station.station_id == next_node["id"]:
                station = None
            if station:
                s_pos    = station.location
                d_to_s, g_to_s   = calc_dist_and_geometry(cur_pos, s_pos)
                d_s_to_c, g_s_to_c = calc_dist_and_geometry(s_pos, next_pos)
                s_arrive = cur_time + d_to_s / truck.speed

                if g_to_s:
                    route.geometry.extend(g_to_s[1:] if route.geometry else g_to_s)
                if g_s_to_c:
                    route.geometry.extend(g_s_to_c[1:] if route.geometry else g_s_to_c)

                route.nodes.append(TruckRouteNode(
                    node_id=station.station_id, node_type="station",
                    position=s_pos,
                    arrival_time=s_arrive, departure_time=s_arrive,
                ))
                route.charging_stop_ids.append(station.station_id)
                total_dist += d_to_s
                cur_pos     = s_pos
                cur_time    = s_arrive
                total_dist += d_s_to_c
                arrive      = cur_time + d_s_to_c / truck.speed
            else:
                seg, g_seg = calc_dist_and_geometry(cur_pos, next_pos)
                if g_seg:
                    route.geometry.extend(g_seg[1:] if route.geometry else g_seg)
                total_dist += seg
                arrive = cur_time + seg / truck.speed

            # 添加节点
            node_type = "customer" if next_node["type"] == "customer" else "recovery"
            if node_type == "recovery" and recovery_station_wait_times:
                # 对于 recovery 节点，使用映射中的等待时间（无人机飞行往返时间）
                service_time = recovery_station_wait_times.get(next_node["id"], self.SERVICE_TIME_CUSTOMER)
            else:
                # 对于 customer 节点，使用标准配送时间
                service_time = self.SERVICE_TIME_CUSTOMER
            
            depart = arrive + service_time
            route.nodes.append(TruckRouteNode(
                node_id=next_node["id"], node_type=node_type,
                position=next_pos,
                arrival_time=arrive, departure_time=depart,
                order_id=next_node["order_id"],
            ))
            cur_pos  = next_pos
            cur_time = depart

        # 返回仓库
        if return_to_depot and depot:
            d, g_back = calc_dist_and_geometry(cur_pos, depot.location)
            if g_back:
                route.geometry.extend(g_back[1:] if route.geometry else g_back)
            total_dist += d
            route.nodes.append(TruckRouteNode(
                node_id=depot.depot_id + "_return", node_type="depot",
                position=depot.location,
                arrival_time=cur_time + d / truck.speed,
                departure_time=cur_time + d / truck.speed,
            ))

        route.total_distance = total_dist
        logger.info(
            "[GreedyMMCE] 卡车 %s 路径：%d 节点，总里程 %.0f m，"
            "经停充电站 %d 个（%s）",
            truck.truck_id, len(route.nodes), total_dist,
            len(route.charging_stop_ids), route.charging_stop_ids,
        )
        return route

    def _find_insertable_station(
        self,
        from_pos: Position3D,
        to_pos: Position3D,
        stations: list,
        already_inserted: list,
        dist_func,
    ) -> Optional["SwapStation"]:
        """
        在 from_pos → to_pos 路段上寻找可顺路插入的充电站。

        条件：绕路距离 ≤ 直线距离 × MAX_DETOUR_RATIO，且尚未插入路径。
        Returns 额外里程最小的充电站；无满足条件的站点则返回 None。
        """
        direct = dist_func(from_pos, to_pos)
        if direct < 1.0:
            return None

        best: Optional["SwapStation"] = None
        best_extra = float("inf")

        for s in stations:
            if s.station_id in already_inserted:
                continue
            detour = dist_func(from_pos, s.location) + dist_func(s.location, to_pos)
            if detour <= direct * self.MAX_DETOUR_RATIO:
                extra = detour - direct
                if extra < best_extra:
                    best_extra, best = extra, s
        return best

    # ══════════════════════════════════════════════════════════════════════════
    # 前瞻能量校验
    # ══════════════════════════════════════════════════════════════════════════

    def _check_energy_feasible(
        self,
        drone: "Drone",
        launch_loc: Position3D,
        delivery_loc: Position3D,
        payload: float,
        recovery_pool: list,
    ) -> Optional[str]:
        """
        前瞻能量校验：验证无人机能完成送达并安全降落到某个回收点。

        改进策略：
          优先选择充电站（距离配送点最近）>其次选择仓库（距离配送点最近）
          而不是全局能量消耗最小

        约束（含安全余量）：
          (E_launch→delivery + E_delivery→recovery) × ENERGY_SAFETY_FACTOR
            ≤ drone.battery_current
        """
        energy_to_deliver = self._flight_energy(drone, launch_loc, delivery_loc, payload)

        # 分离充电站和仓库
        stations_candidates = []
        depots_candidates = []

        for node_id, node_pos in recovery_pool:
            energy_to_recovery = self._flight_energy(drone, delivery_loc, node_pos, 0.0)
            total = (energy_to_deliver + energy_to_recovery) * self.ENERGY_SAFETY_FACTOR
            if total <= drone.battery_current:
                dist = self._uav_path_distance(delivery_loc, node_pos, altitude=self.UAV_CRUISE_ALTITUDE_M)
                if node_id in self.entity_mgr.stations:
                    stations_candidates.append((node_id, dist))
                elif node_id in self.entity_mgr.depots:
                    depots_candidates.append((node_id, dist))

        # 优先返回充电站中距离最近的
        if stations_candidates:
            best_id = min(stations_candidates, key=lambda x: x[1])[0]
            logger.info(
                "[GreedyMMCE] 回收点选择：充电站 %s（距离 %.0fm）",
                best_id, self._uav_path_distance(delivery_loc, self.entity_mgr.stations[best_id].location, altitude=self.UAV_CRUISE_ALTITUDE_M)
            )
            return best_id

        # 其次返回仓库中距离最近的
        if depots_candidates:
            best_id = min(depots_candidates, key=lambda x: x[1])[0]
            logger.info(
                "[GreedyMMCE] 回收点选择：仓库 %s（距离 %.0fm）",
                best_id, self._uav_path_distance(delivery_loc, self.entity_mgr.depots[best_id].location, altitude=self.UAV_CRUISE_ALTITUDE_M)
            )
            return best_id

        return None

    def _flight_energy(
        self,
        drone: "Drone",
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
    ) -> float:
        """
        计算无人机飞行一段路程的能量消耗（焦耳）。

        公式（多旋翼功耗模型）：
          E = P(payload, v_cruise) × (distance / v_cruise)
          P = k1 × (m_empty + payload)^1.5 + k2 × v^3
        """
        distance = self._uav_path_distance(from_pos, to_pos, altitude=self.UAV_CRUISE_ALTITUDE_M)
        if distance < 0.001:
            return 0.0
        flight_time = distance / drone.cruise_speed
        power = drone.calculate_power(payload, drone.cruise_speed)
        return power * flight_time

    # ══════════════════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════════════════

    def _get_available_drones(self) -> list:
        """返回当前空闲（IDLE）且电量高于底线的无人机列表。"""
        available = []
        for d in self.entity_mgr.drones.values():
            if d.status != DroneStatus.IDLE:
                continue
            if d.battery_current <= self.min_reserve_energy:
                continue
            if getattr(d, "carrying_order_id", None):
                continue
            if getattr(d, "waiting_recovery_station_id", ""):
                continue
            if getattr(d, "has_pending_route", False):
                continue
            available.append(d)
        return available

    @staticmethod
    def _find_capable_drone(payload_weight: float, drones: list) -> Optional[object]:
        """
        在可用无人机中找到载重能力 ≥ payload_weight 的无人机。
        优先选择电量最充足的，以降低能量校验失败概率。
        """
        capable = [d for d in drones if d.payload_capacity >= payload_weight]
        if not capable:
            return None
        return max(capable, key=lambda d: d.battery_current)

    def _is_drone_ready_for_depot_launch(self, drone: "Drone", depot: "object") -> bool:
        """判断无人机是否满足“仓库起飞”约束。"""
        if getattr(drone, "transport_truck_id", None):
            return False
        for truck in self.entity_mgr.trucks.values():
            if drone.drone_id in getattr(truck, "docked_drones", []):
                return False
        return drone.current_loc.distance_2d(depot.location) <= self.DEPOT_LAUNCH_TOLERANCE_M

    def _get_recovery_pool(self) -> list:
        """返回所有合法无人机降落点：充换电站 + 仓库。"""
        pool = []
        for s in self.entity_mgr.stations.values():
            pool.append((s.station_id, s.location))
        for dep in self.entity_mgr.depots.values():
            pool.append((dep.depot_id, dep.location))
        return pool

    @staticmethod
    def _dist(pos_a: Position3D, pos_b: Position3D) -> float:
        """
        二维欧氏距离（用于启发式排序/估计，不作为卡车路网主路径）。
        """
        dx = pos_a.x - pos_b.x
        dy = pos_a.y - pos_b.y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _distance_3d(pos_a: Position3D, pos_b: Position3D) -> float:
        """三维欧氏距离（保留旧名兼容外部调用）。"""
        dx = pos_a.x - pos_b.x
        dy = pos_a.y - pos_b.y
        dz = pos_a.z - pos_b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _truck_energy_wh(self, distance_m: float) -> float:
        """卡车电耗模型：0.75 kWh/km -> 0.75 Wh/m。"""
        return max(0.0, distance_m) * self.TRUCK_ENERGY_WH_PER_METER

    def _uav_energy_wh(
        self,
        drone: "Drone",
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
    ) -> float:
        """
        UAV 能耗模型（评分用）：
        - physics: 复用 drone.py 既有功率模型，先算焦耳再转 Wh
        - alpha:   线性经验模型 B_ij = alpha_k * (W_k + payload) * d_ij(km)
        """
        if self.UAV_ENERGY_MODEL == "physics":
            # _flight_energy 内部已调用 drone.calculate_power(...)，与实体耗能口径一致。
            return self._flight_energy(drone, from_pos, to_pos, payload) / 3600.0

        distance_km = self._uav_path_distance(from_pos, to_pos, altitude=self.UAV_CRUISE_ALTITUDE_M) / 1000.0
        total_mass = max(0.0, drone.empty_weight + max(0.0, payload))
        return self.UAV_ALPHA_WH_PER_KG_KM * total_mass * distance_km

    def _estimate_distance_to_nearest_depot(self, pos: Position3D) -> float:
        """估计某位置到最近仓库的路网距离（用于 append 后续锚点）。"""
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return 0.0
        return min(self._road_dist(pos, d.location) for d in depots)

    def _estimate_mode_a_append_distance(self, tail_pos: Position3D, delivery_pos: Position3D) -> float:
        """append-based 估计：尾点到客户 + 客户到最近仓库的后续锚点成本。"""
        direct = self._road_dist(tail_pos, delivery_pos)
        continuation = self._estimate_distance_to_nearest_depot(delivery_pos)
        return direct + continuation

    def _estimate_truck_increment_for_drone_support(
        self,
        truck_tail_pos: Position3D,
        delivery_pos: Position3D,
        launch_pos: Position3D,
        recovery_pos: Position3D,
    ) -> float:
        """估计为支持无人机任务导致的卡车净增里程（相对 A 模式 append 基线）。"""
        baseline_a = self._estimate_mode_a_append_distance(truck_tail_pos, delivery_pos)
        support_path = (
            self._road_dist(truck_tail_pos, launch_pos)
            + self._road_dist(launch_pos, recovery_pos)
            + self._estimate_distance_to_nearest_depot(recovery_pos)
        )
        return max(0.0, support_path - baseline_a)

    def _score_allocation(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
        truck_last_pos: dict[str, Position3D],
    ) -> tuple[float, float, float, float]:
        """
        统一评分：f = f_dist + f_energy + f_penalty。

        Returns:
            (score_total, cost_dist, cost_energy, cost_penalty)
        """
        if not alloc.feasible:
            return float("inf"), 0.0, 0.0, 0.0

        truck_distance = 0.0
        uav_distance = 0.0
        truck_energy = 0.0
        uav_energy = 0.0
        delivery_time_est = current_time

        if alloc.mode == "A":
            truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
            if truck is None or truck.speed <= 0:
                return float("inf"), 0.0, 0.0, 0.0
            
            truck_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)
            truck_distance = self._estimate_mode_a_append_distance(truck_loc, order.delivery_loc)
            truck_energy = self._truck_energy_wh(truck_distance)
            delivery_time_est = current_time + truck_distance / truck.speed + self.SERVICE_TIME_CUSTOMER

        elif alloc.mode in ("B", "B_WAIT"):
            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if drone is None or drone.cruise_speed <= 0:
                return float("inf"), 0.0, 0.0, 0.0

            if alloc.mode == "B_WAIT":
                truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
                if truck is None or launch_station is None or truck.speed <= 0:
                    return float("inf"), 0.0, 0.0, 0.0
                
                truck_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)
                launch_loc = launch_station.location
                truck_distance = self._road_dist(truck_loc, launch_loc)
                truck_energy = self._truck_energy_wh(truck_distance)
                launch_time_est = (
                    alloc.launch_time
                    if alloc.launch_time > current_time
                    else current_time + truck_distance / truck.speed + self.TRUCK_DRONE_LAUNCH_TIME
                )
            else:
                truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                if truck is None:
                    return float("inf"), 0.0, 0.0, 0.0
                
                truck_loc = truck_last_pos.get(truck.truck_id) or truck.get_location(current_time)
                launch_loc = truck_loc
                launch_time_est = current_time + self.TRUCK_DRONE_LAUNCH_TIME

            recovery = (
                self.entity_mgr.stations.get(alloc.recovery_station_id)
                or self.entity_mgr.depots.get(alloc.recovery_station_id)
            )
            if recovery is None:
                return float("inf"), 0.0, 0.0, 0.0

            dist_out = self._uav_path_distance(launch_loc, order.delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            dist_back = self._uav_path_distance(order.delivery_loc, recovery.location, altitude=self.UAV_CRUISE_ALTITUDE_M)
            uav_distance = dist_out + dist_back
            uav_energy = self._uav_energy_wh(drone, launch_loc, order.delivery_loc, order.payload_weight)
            uav_energy += self._uav_energy_wh(drone, order.delivery_loc, recovery.location, 0.0)

            # 车端代价使用“相对 A 基线”的净增里程，避免系统性高估 B/B_WAIT。
            truck_distance = self._estimate_truck_increment_for_drone_support(
                truck_loc,
                order.delivery_loc,
                launch_loc,
                recovery.location,
            )
            truck_energy = self._truck_energy_wh(truck_distance)

            delivery_time_est = launch_time_est + dist_out / drone.cruise_speed + self.delivery_service_time

        elif alloc.mode == "C":
            drone = self.entity_mgr.drones.get(alloc.drone_id)
            depot = self.entity_mgr.depots.get(alloc.vehicle_id)
            if drone is None or depot is None or drone.cruise_speed <= 0:
                return float("inf"), 0.0, 0.0, 0.0
            dist_out = self._uav_path_distance(depot.location, order.delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            dist_back = self._uav_path_distance(order.delivery_loc, depot.location, altitude=self.UAV_CRUISE_ALTITUDE_M)
            uav_distance = dist_out + dist_back
            uav_energy = self._uav_energy_wh(drone, depot.location, order.delivery_loc, order.payload_weight)
            uav_energy += self._uav_energy_wh(drone, order.delivery_loc, depot.location, 0.0)
            delivery_time_est = current_time + dist_out / drone.cruise_speed + self.delivery_service_time

        else:
            return float("inf"), 0.0, 0.0, 0.0

        cost_dist = self.C_DIST_ET * truck_distance + self.C_DIST_UAV * uav_distance
        cost_energy = self.C_ENERGY_ET * truck_energy + self.C_ENERGY_UAV * uav_energy
        lateness = max(0.0, delivery_time_est - order.deadline)
        cost_penalty = self.LAMBDA_TIME * order.penalty_rate * lateness
        score_total = cost_dist + cost_energy + cost_penalty
        return score_total, cost_dist, cost_energy, cost_penalty

    # ══════════════════════════════════════════════════════════════════════════
    # 改进的模式 B：支持无人机在充电站等待
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_charging_station_departure(
        self,
        drone: "Drone",
        truck_tail_loc: Position3D,
        launch_loc: Position3D,
        launch_station_id: str,
        truck_distance_to_launch: float,
        launch_delay: float,
        delivery_loc: Position3D,
        payload: float,
        recovery_pool: list,
        current_time: float,
        order: "Order",
    ) -> Optional[dict]:
        """
        评估从指定充电站出发的无人机方案。

        计算能量可行、时间可行、成本最低的回收点组合。
        返回 dict 包含计分信息；若无可行方案返回 None。

        Args:
            drone: 无人机对象
            launch_loc: 充电站位置（出发点）
            launch_station_id: 充电站 ID
            delivery_loc: 配送点位置
            payload: 载重 [kg]
            recovery_pool: 回收候选点列表 [(station_id, position), ...]
            current_time: 当前仿真时间 [s]
            order: 订单对象

        Returns:
            {
                'feasible': bool,
                'launch_station_id': str,
                'launch_time': float,
                'recovery_station_id': str,
                'wait_duration': float,
                'score': float,
                'distance': float,
            }
            或 None if 无可行方案
        """
        energy_to_deliver = self._flight_energy(drone, launch_loc, delivery_loc, payload)

        best_scenario = None
        best_score = float("inf")

        for recovery_id, recovery_loc in recovery_pool:
            # 约束：回收点必须是“起飞后”卡车可访问的节点。
            # 这里使用从当前车位到节点的路网距离作为时序近似：
            # 若 recovery 与 launch 不同，且 recovery 不比 launch 更“靠后”，则判为不可用。
            if recovery_id != launch_station_id:
                truck_distance_to_recovery = self._road_dist(truck_tail_loc, recovery_loc)
                if truck_distance_to_recovery <= truck_distance_to_launch + 1.0:
                    continue

            # 能量校验（可行性约束继续使用物理模型[J]）
            energy_to_recovery_j = self._flight_energy(drone, delivery_loc, recovery_loc, 0.0)
            total_energy_j = (energy_to_deliver + energy_to_recovery_j) * self.ENERGY_SAFETY_FACTOR

            if total_energy_j > drone.battery_current:
                continue

            # 时间计算（这里的 wait_duration 暂设为 0，实际由卡车路径决定）
            dist_out = self._uav_path_distance(launch_loc, delivery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            dist_back = self._uav_path_distance(delivery_loc, recovery_loc, altitude=self.UAV_CRUISE_ALTITUDE_M)
            total_distance = dist_out + dist_back

            # 估算飞行时间
            flight_time_out = dist_out / drone.cruise_speed if drone.cruise_speed > 0 else 1000
            flight_time_back = dist_back / drone.cruise_speed if drone.cruise_speed > 0 else 1000
            flight_time_total = flight_time_out + self.delivery_service_time + flight_time_back
            launch_operation_time = self.TRUCK_DRONE_LAUNCH_TIME
            recover_operation_time = self.TRUCK_DRONE_RECOVER_TIME

            # 统一评分：f = f_dist + f_energy + f_penalty
            delivery_time_est = (
                current_time
                + launch_delay
                + launch_operation_time
                + flight_time_out
                + self.delivery_service_time
            )
            lateness = max(0.0, delivery_time_est - order.deadline)

            cost_dist = (
                self.C_DIST_UAV * total_distance
                + self.C_DIST_ET * self._estimate_truck_increment_for_drone_support(
                    truck_tail_loc,
                    delivery_loc,
                    launch_loc,
                    recovery_loc,
                )
            )
            cost_energy = (
                self.C_ENERGY_UAV * (
                    self._uav_energy_wh(drone, launch_loc, delivery_loc, payload)
                    + self._uav_energy_wh(drone, delivery_loc, recovery_loc, 0.0)
                )
                + self.C_ENERGY_ET * self._truck_energy_wh(
                    self._estimate_truck_increment_for_drone_support(
                        truck_tail_loc,
                        delivery_loc,
                        launch_loc,
                        recovery_loc,
                    )
                )
            )
            cost_penalty = self.LAMBDA_TIME * order.penalty_rate * lateness
            score = cost_dist + cost_energy + cost_penalty

            if score < best_score:
                best_score = score
                best_scenario = {
                    'feasible': True,
                    'launch_station_id': launch_station_id,
                    'launch_time': current_time + launch_delay + launch_operation_time,
                    'recovery_station_id': recovery_id,
                    # 注意：wait_duration 从“卡车到达起飞站”起算，覆盖
                    # 放飞操作 + 飞行往返 + 回收操作，避免边界时刻错过回收触发。
                    'wait_duration': launch_operation_time + flight_time_total + recover_operation_time,
                    'score': best_score,
                    'distance': total_distance,
                    'energy_needed': total_energy_j,
                    'flight_time': flight_time_total,
                    'cost_dist': cost_dist,
                    'cost_energy': cost_energy,
                    'cost_penalty': cost_penalty,
                }

        return best_scenario

    def _predict_truck_charging_stations(
        self,
        truck: "Truck",
        current_time: float,
    ) -> list[tuple[str, Position3D]]:
        """
        快速预测卡车将经过的充电站（基于当前位置和已有的快速贪心预测）。

        此方法返回卡车路线上按顺序出现的充电站列表。

        Args:
            truck: 卡车对象
            current_time: 当前仿真时间

        Returns:
            [(station_id, station_location), ...] 按卡车轨迹顺序
        """
        stations = list(self.entity_mgr.stations.values())
        if not stations:
            return []

        # 简单启发式：返回离卡车当前位置最近的 3 个充电站
        # （在实际应用中应基于完整的卡车路由预测）
        truck_loc = truck.get_location(current_time)
        nearby_stations = sorted(
            stations,
            key=lambda s: self._dist(truck_loc, s.location),
        )[:3]

        return [(s.station_id, s.location) for s in nearby_stations]
