#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 仿真 API Blueprint (Section 3 + 4)

提供仿真控制 REST 接口：
  POST /api/sim/init      接收实体清单 + 订单生成参数，初始化所有 Manager
  POST /api/sim/control   start / pause / reset / set speed
  GET  /api/sim/state     返回完整快照（供 F5 刷新/重连恢复）
  GET  /api/sim/orders    返回最近 N 条订单（供 OrderTask 页面列表展示）

以及 WebSocket 端点：
  WS   /api/ws/telemetry  通过 telemetry.register_route(sock) 注册

flask-sock 实例 sock 在此模块创建后由 app.py 调用 sock.init_app(app) 绑定。

模块级单例：
  _entity_mgr : EntityManager
  _order_mgr  : OrderManager
  _sim_engine : SimulationEngine
"""

from __future__ import annotations

import logging
import math
import copy
import time
from typing import TYPE_CHECKING

from flask import Blueprint, jsonify, request
from flask_sock import Sock

from api.websockets.telemetry import broadcast_tick, register_route, set_snapshot_builder
from entity_manager import EntityManager
from order_manager import OrderManager
from sim_engine import SimulationEngine
from solver.decision_engine import DispatchDecisionEngine
from training.frontend_runtime_adapter import (
    OnlinePolicyRuntimePlayer,
    build_orders_raw_from_sim_init_payload,
    build_policy_activation_config,
    build_runtime_scene_context_from_payload,
    resolve_scene_context,
)
from training.frontend_training_runtime import (
    DEFAULT_TRAINING_SCENE_ID,
    FrontendPPOTrainingRuntime,
    build_frontend_training_activation,
)
from utils.coord_utils import utm_to_wgs84

# 导入预设场景模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "environment", "geo"))
from preset_scenes import get_preset_scene, save_preset_entities

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

sim_bp = Blueprint("sim", __name__)
sock   = Sock()  # app.py 调用 sock.init_app(app) 绑定 Flask 应用


def _optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# ── 模块级单例（由 /api/sim/init 初始化）─────────────────────────────────────
_entity_mgr: EntityManager           = EntityManager()
_order_mgr:  OrderManager            = OrderManager()
_sim_engine: SimulationEngine         = SimulationEngine()
_dispatch_engine: DispatchDecisionEngine | None = None
_policy_runtime: OnlinePolicyRuntimePlayer | None = None
_training_runtime: FrontendPPOTrainingRuntime | None = None
_last_init_payload: dict | None = None

# 注册 WebSocket 遥测端点，并将 sim_engine 的完整快照构造函数注入 telemetry 模块
register_route(sock)
set_snapshot_builder(_sim_engine.build_full_snapshot)


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_initial_orders_time_domain(initial_orders: list[dict]) -> list[dict]:
    """
    将 wall_ms 批次的 create_time / deadline 归一化到仿真秒（与 sim t=0 对齐）。

    规则：取本批最小 create_time 为 t0，每条订单
      create_sim  = (create - t0) / 1000
      deadline_sim = (deadline - t0) / 1000
    若 time_domain=='sim_s' 或 create_time 量级明显为秒（< 1e11），则不改写。
    """
    if not initial_orders:
        return initial_orders
    first = initial_orders[0]
    if first.get("time_domain") == "sim_s":
        return initial_orders
    ct0 = float(first["create_time"])
    if ct0 < 1e11:
        return initial_orders
    t0_ms = min(float(o["create_time"]) for o in initial_orders)
    out: list[dict] = []
    for o in initial_orders:
        d = dict(o)
        ct = float(d["create_time"])
        dl = float(d["deadline"])
        d["create_time"] = (ct - t0_ms) / 1000.0
        d["deadline"] = (dl - t0_ms) / 1000.0
        d["time_domain"] = "sim_s"
        out.append(d)
    return out


def _load_initial_orders(initial_orders: list[dict], order_mgr: OrderManager) -> None:
    """
    从前端发送的初始订单列表加载订单到 pending_orders。

    Args:
        initial_orders: 前端发送的订单列表（每项含 order_id, create_time, deadline, 等）
        order_mgr:      OrderManager 实例
    """
    from core.entities.order import Order
    from core.entities.primitives import Position3D
    from utils.coord_utils import wgs84_to_utm

    initial_orders = _normalize_initial_orders_time_domain(initial_orders)

    for order_data in initial_orders:
        try:
            # 将 WGS84 坐标转换为 UTM
            x, y = wgs84_to_utm(order_data["delivery_lng"], order_data["delivery_lat"])
            delivery_loc = Position3D(x=x, y=y, z=float(order_data.get("delivery_z", 0)))

            # 创建订单对象
            order = Order(
                order_id=order_data["order_id"],
                create_time=float(order_data["create_time"]),
                deadline=float(order_data["deadline"]),
                delivery_loc=delivery_loc,
                payload_weight=float(order_data.get("payload_weight", 1.0)),
                pickup_source_id=order_data.get("pickup_source_id"),
                source_type=order_data.get("source_type"),
            )

            # 添加到 pending_orders
            order_mgr.pending_orders[order.order_id] = order
            logger.debug("[_load_initial_orders] 加载初始订单 %s", order.order_id)

        except Exception as e:
            logger.warning("[_load_initial_orders] 加载订单 %s 失败: %s", 
                         order_data.get("order_id", "unknown"), str(e))

    logger.info("[_load_initial_orders] 加载了 %d 笔初始订单", len(initial_orders))


def _get_active_runtime():
    """返回当前对外提供 `/control` `/state` `/ws` 的活跃运行时。"""
    return _policy_runtime if _policy_runtime is not None else _sim_engine


def _restore_classic_runtime() -> None:
    """恢复 classic SimulationEngine 为当前活跃运行时。"""
    set_snapshot_builder(_sim_engine.build_full_snapshot)


def _shutdown_policy_runtime() -> None:
    """停止并移除当前 PPO 在线运行时。"""
    global _policy_runtime
    if _policy_runtime is not None:
        try:
            _policy_runtime.shutdown()
        except Exception:
            logger.exception("[policy_runtime] shutdown 失败")
        _policy_runtime = None
    _restore_classic_runtime()


def _shutdown_training_runtime() -> None:
    """移除已结束的 PPO 训练遥测运行时；运行中的训练不在此处强停。"""
    global _training_runtime
    if _training_runtime is None:
        return
    if _training_runtime.is_running:
        raise RuntimeError("PPO 训练正在运行，不能切换运行时")
    _training_runtime = None
    _restore_classic_runtime()


def _switch_to_policy_runtime(runtime: OnlinePolicyRuntimePlayer) -> None:
    """将遥测输出切换到 PPO 在线运行时。"""
    global _policy_runtime
    if _policy_runtime is not None and _policy_runtime is not runtime:
        _shutdown_policy_runtime()
    _policy_runtime = runtime
    set_snapshot_builder(runtime.build_full_snapshot)


def _switch_to_training_runtime(runtime: FrontendPPOTrainingRuntime) -> None:
    """将遥测输出切换到 PPO 训练过程。"""
    global _training_runtime
    if _training_runtime is not None and _training_runtime is not runtime:
        if _training_runtime.is_running:
            raise RuntimeError("PPO 训练已经在运行")
        _training_runtime = None
    _training_runtime = runtime
    set_snapshot_builder(runtime.build_full_snapshot)


def _build_policy_scene_context_from_last_init(base_scene_ctx, payload: dict):
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError("当前 init payload 缺少 entities，无法构造 policy scene context")
    orders_raw = build_orders_raw_from_sim_init_payload(
        initial_orders=list(payload.get("initial_orders") or []),
        scheduled_dynamic_orders=list(payload.get("scheduled_dynamic_orders") or []),
    )
    bounds = payload.get("bbox") or {}
    return build_runtime_scene_context_from_payload(
        scene_ctx=base_scene_ctx,
        entities_raw=entities,
        orders_raw=orders_raw,
        bounds_override=bounds if isinstance(bounds, dict) else None,
    )


def _serialize_truck_route(route: "TruckRoute") -> dict:
    """将 TruckRoute 转换为前端可显示的 WGS84 路线数据。"""
    nodes = []
    for node in route.nodes:
        lon, lat = utm_to_wgs84(node.position.x, node.position.y)
        nodes.append({
            "node_id": node.node_id,
            "node_type": node.node_type,
            "lng": lon,
            "lat": lat,
            "arrival_time": node.arrival_time,
            "departure_time": node.departure_time,
            "order_id": node.order_id,
        })
    geometry = []
    for pos in route.geometry:
        lon, lat = utm_to_wgs84(pos.x, pos.y)
        geometry.append({"lng": lon, "lat": lat})
    return {
        "truck_id": route.truck_id,
        "nodes": nodes,
        "total_distance": round(route.total_distance, 2),
        "charging_stop_ids": route.charging_stop_ids,
        "geometry": geometry,
    }


def _pick_mode_b_launch_loc(
    alloc: "AllocationResult",
    order: "Order",
    current_time: float,
    truck_routes: dict,
):
    """
    为模式 B 选择“可视化起飞点”。

    规则：优先从该卡车规划路径中的 station/recovery 节点选择，按与服务点距离最近；
    若无可用站点，则退回 depot；再退回卡车当前位置。
    """
    def _dist(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy)

    route = truck_routes.get(alloc.vehicle_id)
    if route is not None:
        candidates = []
        for idx, node in enumerate(route.nodes):
            node_type = getattr(node, "node_type", "")
            if node_type not in ("station", "recovery", "depot"):
                continue
            # 站点优先，其次仓库；同类按离配送点距离最近
            type_rank = 0 if node_type in ("station", "recovery") else 1
            candidates.append((type_rank, _dist(node.position, order.delivery_loc), idx, node))

        if candidates:
            _, _, _, best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
            return best.position, best.node_id, best.node_type

    truck = _entity_mgr.trucks.get(alloc.vehicle_id)
    if truck is not None:
        return truck.get_location(current_time), truck.truck_id, "truck"

    return None, "", ""


def _distance_2d(pos_a, pos_b) -> float:
    dx = pos_a.x - pos_b.x
    dy = pos_a.y - pos_b.y
    return math.sqrt(dx * dx + dy * dy)


def _flight_time(drone, from_pos, to_pos) -> float:
    dist = _distance_2d(from_pos, to_pos)
    if dist < 1e-6:
        return 0.0
    return dist / max(drone.cruise_speed, 1e-6)


def _flight_energy(drone, from_pos, to_pos, payload: float) -> float:
    t = _flight_time(drone, from_pos, to_pos)
    if t <= 0:
        return 0.0
    p = drone.calculate_power(payload, drone.cruise_speed)
    return p * t


def _normalize_route_node_id(node_id: str) -> str:
    if node_id in _entity_mgr.stations or node_id in _entity_mgr.depots:
        return node_id
    if node_id.endswith("_return"):
        base = node_id[:-7]
        if base in _entity_mgr.depots:
            return base
    return node_id


def _select_mode_b_launch_and_recovery(
    alloc: "AllocationResult",
    order: "Order",
    current_time: float,
    truck_routes: dict,
):
    """
    模式 B 航段重建（仅可视化）：
      - recovery 必须是卡车后续可到达节点（不允许“已经过站点”）
      - 比较两种方案并选总路程更短者：
          1) 前一站 -> 任务点 -> 最近站(锚点)
          2) 最近站(锚点) -> 任务点 -> 后一站
      - 同时满足：追车时间可行 + 电量可行
    """
    route = truck_routes.get(alloc.vehicle_id)
    drone = _entity_mgr.drones.get(alloc.drone_id)

    recovery_default = _entity_mgr.stations.get(alloc.recovery_station_id) or _entity_mgr.depots.get(alloc.recovery_station_id)

    def _fallback_result(tag: str):
        launch_loc, launch_node_id, launch_node_type = _pick_mode_b_launch_loc(alloc, order, current_time, truck_routes)
        if launch_loc is None:
            return None
        recovery_loc = recovery_default.location if recovery_default is not None else launch_loc
        recovery_id = alloc.recovery_station_id if alloc.recovery_station_id else launch_node_id
        return {
            "launch_loc": launch_loc,
            "launch_node_id": launch_node_id,
            "launch_node_type": launch_node_type,
            "recovery_loc": recovery_loc,
            "recovery_station_id": recovery_id,
            "strategy": tag,
        }

    if drone is None or route is None or recovery_default is None:
        return _fallback_result("fallback")

    # 抽取卡车路径可用站点（按时序）
    stops = []
    for seq, node in enumerate(route.nodes):
        node_type = getattr(node, "node_type", "")
        if node_type not in ("station", "recovery", "depot"):
            continue
        stops.append({
            "seq": seq,
            "node_id": node.node_id,
            "entity_id": _normalize_route_node_id(node.node_id),
            "node_type": node_type,
            "pos": node.position,
            "arrival": node.arrival_time,
            "departure": node.departure_time,
        })

    if len(stops) < 2:
        return _fallback_result("fallback-short-route")

    # 锚点：离任务点最近的站点/仓库（按路径节点，不是全局静态集合）
    anchor_idx = min(range(len(stops)), key=lambda i: _distance_2d(stops[i]["pos"], order.delivery_loc))

    def find_prev_distinct(i: int):
        cur_id = stops[i]["entity_id"]
        for j in range(i - 1, -1, -1):
            if stops[j]["entity_id"] != cur_id:
                return stops[j]
        return None

    def find_next_distinct(i: int):
        cur_id = stops[i]["entity_id"]
        for j in range(i + 1, len(stops)):
            if stops[j]["entity_id"] != cur_id:
                return stops[j]
        return None

    anchor = stops[anchor_idx]
    prev_stop = find_prev_distinct(anchor_idx)
    next_stop = find_next_distinct(anchor_idx)

    candidates = []
    if prev_stop is not None:
        candidates.append(("case1_prev_anchor", prev_stop, anchor))
    if next_stop is not None:
        candidates.append(("case2_anchor_next", anchor, next_stop))

    # 若锚点两侧都没有不同站点，退回默认
    if not candidates:
        return _fallback_result("fallback-no-candidates")

    safety_factor = 1.2
    evaluated = []
    for strategy, launch_stop, recovery_stop in candidates:
        # recovery 必须是“后续站点”，不允许回收点在起飞点之前
        if recovery_stop["seq"] <= launch_stop["seq"]:
            continue

        t_out = _flight_time(drone, launch_stop["pos"], order.delivery_loc)
        t_back = _flight_time(drone, order.delivery_loc, recovery_stop["pos"])
        total_t = t_out + t_back

        e_out = _flight_energy(drone, launch_stop["pos"], order.delivery_loc, order.payload_weight)
        e_back = _flight_energy(drone, order.delivery_loc, recovery_stop["pos"], 0.0)
        e_need = (e_out + e_back) * safety_factor
        energy_ok = e_need <= drone.battery_current

        drone_arrive_t = launch_stop["departure"] + total_t
        catch_ok = drone_arrive_t <= recovery_stop["departure"] + 1e-6

        total_dist = _distance_2d(launch_stop["pos"], order.delivery_loc) + _distance_2d(order.delivery_loc, recovery_stop["pos"])

        evaluated.append({
            "strategy": strategy,
            "launch": launch_stop,
            "recovery": recovery_stop,
            "total_dist": total_dist,
            "energy_ok": energy_ok,
            "catch_ok": catch_ok,
            "drone_arrive_t": drone_arrive_t,
            "truck_depart_t": recovery_stop["departure"],
            "energy_need": e_need,
        })

    # 优先：同时满足追车+电量，且总路程最短
    feasible = [x for x in evaluated if x["energy_ok"] and x["catch_ok"]]
    if feasible:
        best = min(feasible, key=lambda x: x["total_dist"])
    else:
        # 其次：只满足电量（追车失败），用于可视化兜底
        energy_only = [x for x in evaluated if x["energy_ok"]]
        if energy_only:
            best = min(energy_only, key=lambda x: x["total_dist"])
            logger.warning(
                "[_serialize_drone_route] 模式B订单 %s 无法满足追车约束，采用仅电量可行方案 %s",
                alloc.order_id,
                best["strategy"],
            )
        elif evaluated:
            # 都不可行：退回默认
            return _fallback_result("fallback-no-feasible")
        else:
            return _fallback_result("fallback-empty-eval")

    return {
        "launch_loc": best["launch"]["pos"],
        "launch_node_id": best["launch"]["entity_id"],
        "launch_node_type": best["launch"]["node_type"],
        "recovery_loc": best["recovery"]["pos"],
        "recovery_station_id": best["recovery"]["entity_id"],
        "strategy": best["strategy"],
        "distance": best["total_dist"],
    }


def _serialize_drone_route(
    alloc: "AllocationResult",
    order: "Order" | None,
    current_time: float,
    truck_routes: dict,
) -> dict | None:
    """将无人机任务路径转换为可视化航线。"""
    if order is None:
        return None
    
    if alloc.mode == "B":
        selected = _select_mode_b_launch_and_recovery(alloc, order, current_time, truck_routes)
        if selected is None:
            return None
        launch_loc = selected["launch_loc"]
        launch_node_id = selected["launch_node_id"]
        launch_node_type = selected["launch_node_type"]
        delivery_loc = order.delivery_loc
        recovery_loc = selected["recovery_loc"]
        recovery_station_id = selected["recovery_station_id"]

        # 调试详情打印：显示充电站是否有正确的坐标
        logger.info(
            f"[_serialize_drone_route] 模式 B: 订单 {alloc.order_id} → 无人机 {alloc.drone_id} "
            f"起飞点 {launch_node_id}({launch_node_type}) ({launch_loc.x:.1f}, {launch_loc.y:.1f}) → "
            f"配送点 ({delivery_loc.x:.1f}, {delivery_loc.y:.1f}) → "
            f"回收于 {recovery_station_id} ({recovery_loc.x:.1f}, {recovery_loc.y:.1f}) | 方案={selected.get('strategy', '-') }"
        )

        path = [
            list(utm_to_wgs84(launch_loc.x, launch_loc.y)),
            list(utm_to_wgs84(delivery_loc.x, delivery_loc.y)),
            list(utm_to_wgs84(recovery_loc.x, recovery_loc.y)),
        ]
        
        logger.debug(f"[_serialize_drone_route] 路径坐标: {path}")

        return {
            "drone_id": alloc.drone_id,
            "order_id": alloc.order_id,
            "mode": alloc.mode,
            "launch_node_id": launch_node_id,
            "launch_node_type": launch_node_type,
            "recovery_station_id": recovery_station_id,
            "path": path,
        }

    if alloc.mode == "C":
        depot = _entity_mgr.depots.get(alloc.vehicle_id)
        if depot is None:
            return None
        delivery_loc = order.delivery_loc
        depot_lonlat = utm_to_wgs84(depot.location.x, depot.location.y)

        logger.debug(f"[_serialize_drone_route] 模式 C: 订单 {alloc.order_id} → 无人机 {alloc.drone_id} 往返仓库 {alloc.vehicle_id}")

        return {
            "drone_id": alloc.drone_id,
            "order_id": alloc.order_id,
            "mode": alloc.mode,
            "recovery_station_id": alloc.vehicle_id,  # 仓库 ID
            "path": [
                list(depot_lonlat),
                list(utm_to_wgs84(delivery_loc.x, delivery_loc.y)),
                list(depot_lonlat),
            ],
        }

    return None


def _serialize_runtime_drone_route(
    alloc: "AllocationResult",
    runtime_route,
) -> dict | None:
    """
    优先使用调度执行层已下发的真实无人机航线进行序列化。

    这能保证前端紫色虚线与动态实体运动逻辑一致，避免“可视化推断路线”
    与执行层路线不一致。
    """
    if runtime_route is None:
        return None

    launch_loc = runtime_route.launch_loc
    delivery_loc = runtime_route.delivery_loc
    recovery_loc = runtime_route.recovery_loc

    runtime_path = getattr(runtime_route, "path", None) or []
    if runtime_path:
        path = [list(utm_to_wgs84(pos.x, pos.y)) for pos in runtime_path]
    else:
        path = [
            list(utm_to_wgs84(launch_loc.x, launch_loc.y)),
            list(utm_to_wgs84(delivery_loc.x, delivery_loc.y)),
            list(utm_to_wgs84(recovery_loc.x, recovery_loc.y)),
        ]

    if alloc.mode == "B_WAIT":
        launch_node_id = alloc.launch_station_id or alloc.vehicle_id
        launch_node_type = "station"
    elif alloc.mode == "C":
        launch_node_id = alloc.vehicle_id
        launch_node_type = "depot"
    else:
        launch_node_id = alloc.vehicle_id
        launch_node_type = "truck"

    return {
        "drone_id": alloc.drone_id,
        "order_id": alloc.order_id,
        "mode": alloc.mode,
        "launch_node_id": launch_node_id,
        "launch_node_type": launch_node_type,
        "launch_time": float(getattr(alloc, "launch_time", 0.0) or 0.0),
        "recovery_station_id": alloc.recovery_station_id,
        "path": path,
    }


# ══════════════════════════════════════════════════════════════════════════
# POST /init
# ══════════════════════════════════════════════════════════════════════════

@sim_bp.route("/init", methods=["POST"])
def sim_init():
    """
    接收实体清单（entities）+ 订单生成参数（order_gen_config）+ 地图边界（bbox）。

    重置所有 Manager，准备就绪后返回加载汇总。

    请求体 JSON 格式（详见 Section 3 文档示例）：
    {
      "scene_id": "...",
      "bbox": {"min_lng": ..., "min_lat": ..., "max_lng": ..., "max_lat": ...},
      "entities": { "depots": [...], "stations": [...], "trucks": [...], "drones": [...] },
      "order_gen_config": { "arrival_rate": 4, ... }
    }
    """
    global _entity_mgr, _order_mgr, _sim_engine, _dispatch_engine, _last_init_payload

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "请求体必须为 JSON"}), 400

    # ── 输入校验 ─────────────────────────────────────────────────────────────
    entities    = body.get("entities")
    bbox        = body.get("bbox")
    gen_config  = body.get("order_gen_config", {})

    if not entities:
        return jsonify({"error": "缺少 entities 字段"}), 400
    if not bbox or not all(k in bbox for k in ("min_lng", "min_lat", "max_lng", "max_lat")):
        return jsonify({"error": "缺少或格式错误的 bbox 字段"}), 400

    # ── 重置并重新初始化 ─────────────────────────────────────────────────────
    try:
        _shutdown_training_runtime()
        _shutdown_policy_runtime()
        _sim_engine.reset()

        _entity_mgr = EntityManager()
        _order_mgr  = OrderManager()
        _sim_engine = SimulationEngine()
        _dispatch_engine = DispatchDecisionEngine(_entity_mgr, _order_mgr)
        # 👇 新增：记录接收到的充电站坐标
        stations_received = [
            {"id": s["station_id"], "lng": s.get("lng"), "lat": s.get("lat")}
            for s in body.get("entities", {}).get("stations", [])
        ]
        logger.info("[sim_init] 收到充电站坐标: %s", stations_received)
        _entity_mgr.load_from_config(body)
        _order_mgr.configure(gen_config, bbox)
        _sim_engine.attach(_entity_mgr, _order_mgr)

        # ── 预定义动态订单（spawn_sim_s 注入，与算法无关、可复现）──────────────
        _order_mgr.set_scheduled_dynamic_orders(body.get("scheduled_dynamic_orders") or [])

        # ── 加载前端发送的初始订单 ───────────────────────────────────────────
        initial_orders = body.get("initial_orders", [])
        if initial_orders:
            _load_initial_orders(initial_orders, _order_mgr)

        # 更新 telemetry 快照构造函数绑定（新 _sim_engine 实例）
        set_snapshot_builder(_sim_engine.build_full_snapshot)
        _last_init_payload = copy.deepcopy(body)

        logger.info(
            "[sim_init] 初始化完成：scene_id=%s，%d 仓库，%d 换电站，%d 卡车，%d 无人机，%d 初始订单",
            body.get("scene_id", "unknown"),
            len(_entity_mgr.depots),
            len(_entity_mgr.stations),
            len(_entity_mgr.trucks),
            len(_entity_mgr.drones),
            len(_order_mgr.pending_orders),
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        logger.exception("[sim_init] 初始化失败")
        return jsonify({"error": str(exc)}), 422

    return jsonify({
        "status":  "initialized",
        "summary": {
            "depots":   len(_entity_mgr.depots),
            "stations": len(_entity_mgr.stations),
            "trucks":   len(_entity_mgr.trucks),
            "drones":   len(_entity_mgr.drones),
            "pending_orders": len(_order_mgr.pending_orders),
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /control
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/control", methods=["POST"])
def sim_control():
    """
    控制仿真启停与速率。

    请求体：
      {"action": "start" | "pause" | "reset", "speed": <float>}

    Returns:
      JSON {"status": "ok", "is_running": bool, "sim_time": float, "speed_ratio": float}
    """
    body   = request.get_json(silent=True) or {}
    action = body.get("action", "")
    speed  = body.get("speed")
    runtime = _get_active_runtime()

    try:
        if action == "start":
            if runtime is _sim_engine and not _sim_engine.is_running:
                logger.info(
                    "[sim_control] 启动 classic 仿真，_entity_mgr=%s, _order_mgr=%s",
                    "OK" if _sim_engine._entity_mgr else "None",
                    "OK" if _sim_engine._order_mgr else "None",
                )
            runtime.start()
        elif action == "pause":
            runtime.pause()
        elif action == "reset":
            runtime.reset()
        elif action == "set_speed":
            pass  # 仅调整速率，不改变运行状态
        else:
            return jsonify({"error": f"未知 action: '{action}'，支持 start/pause/reset/set_speed"}), 400

        if speed is not None:
            runtime.set_speed(float(speed))

    except (RuntimeError, ValueError) as exc:
        logger.exception("[sim_control] 控制异常：action=%s, error=%s", action, str(exc))
        return jsonify({"error": str(exc)}), 409

    return jsonify({
        "status":      "ok",
        "is_running":  runtime.is_running,
        "sim_time":    round(runtime.current_time, 3),
        "speed_ratio": runtime.speed_ratio,
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /state
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/state", methods=["GET"])
def sim_state():
    """
    返回后端当前完整快照，供 F5 刷新 / 重新连接时恢复前端状态。

    响应结构与 WebSocket FULL_SNAPSHOT payload 完全对齐，
    但不包含 type 字段（直接返回 payload 内容）。
    """
    snapshot = _get_active_runtime().build_full_snapshot()
    return jsonify(snapshot["payload"])


# ══════════════════════════════════════════════════════════════════════════════
# GET /orders
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/orders", methods=["GET"])
def sim_orders():
    """
    返回最近 N 条订单详情，供 OrderTask 页面列表展示。

    Query params:
      limit (int, default=100): 最多返回条数
    """
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 1000))   # 防范滥用，限[1, 1000]
    except (TypeError, ValueError):
        limit = 100

    runtime = _get_active_runtime()
    if hasattr(runtime, "get_recent_orders"):
        orders = runtime.get_recent_orders(limit)
    else:
        orders = _order_mgr.get_recent_orders(limit)
    return jsonify({
        "total":  len(orders),
        "orders": orders,
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /training/start
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/training/start", methods=["POST"])
def sim_training_start():
    """启动默认场景 PPO 训练，并把训练环境遥测推到前端。"""
    global _training_runtime
    body = request.get_json(silent=True) or {}

    try:
        if _training_runtime is not None and _training_runtime.is_running:
            return jsonify({"error": "PPO 训练已经在运行"}), 409

        activation = build_frontend_training_activation(body)
        scene_ctx = resolve_scene_context(
            config_path=activation.config_path,
            scene_id=activation.scene_id,
            scene_bundle_dir=None,
        )
        if scene_ctx.scene_id != DEFAULT_TRAINING_SCENE_ID:
            raise ValueError(
                "第一版 PPO 训练可视化仅支持默认场景: "
                f"config_default={scene_ctx.scene_id}, supported={DEFAULT_TRAINING_SCENE_ID}"
            )

        require_current_init = bool(body.get("require_current_init_scene", True))
        if require_current_init:
            if _last_init_payload is None:
                raise ValueError("请先用默认场景初始化前端仿真，再启动 PPO 训练可视化")
            init_scene_id = str(_last_init_payload.get("scene_id", "")).strip()
            if init_scene_id != DEFAULT_TRAINING_SCENE_ID:
                raise ValueError(
                    "当前前端初始化场景与 PPO 训练场景不一致: "
                    f"init={init_scene_id or 'unknown'}, required={DEFAULT_TRAINING_SCENE_ID}"
                )

        if _policy_runtime is not None:
            _shutdown_policy_runtime()
        _sim_engine.pause()

        runtime = FrontendPPOTrainingRuntime(activation=activation)
        _switch_to_training_runtime(runtime)
        runtime.start()

        return jsonify(
            {
                "status": "started",
                "runtime": runtime.status(),
            }
        )
    except Exception as exc:
        logger.exception("[sim_training_start] 启动失败")
        return jsonify({"error": str(exc)}), 422


# ══════════════════════════════════════════════════════════════════════════════
# GET /training/state
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/training/state", methods=["GET"])
def sim_training_state():
    """返回当前 PPO 训练遥测运行时状态。"""
    if _training_runtime is None:
        return jsonify({"active": False})
    return jsonify(_training_runtime.status())


# ══════════════════════════════════════════════════════════════════════════════
# POST /policy/activate
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/policy/activate", methods=["POST"])
def sim_policy_activate():
    """
    激活基于 TrainingEnvAdapter 的在线 PPO 运行时。

    说明：
      - 不替换 classic SimulationEngine 代码路径；
      - 仅通过活跃运行时切换，让现有 `/control` `/state` `/ws` 自动复用。
    """
    body = request.get_json(silent=True) or {}
    started_at = time.perf_counter()
    logger.info(
        "[sim_policy_activate] 收到激活请求: policy_path=%s config_path=%s scene_id=%s "
        "order_source_mode=%s use_current_init_payload=%s deterministic=%s",
        body.get("policy_path"),
        body.get("config_path"),
        body.get("scene_id"),
        body.get("order_source_mode"),
        body.get("use_current_init_payload", True),
        body.get("deterministic", True),
    )

    try:
        if _training_runtime is not None and _training_runtime.is_running:
            return jsonify({"error": "PPO 训练正在运行，不能激活在线策略"}), 409
        logger.info("[sim_policy_activate] 解析激活配置...")
        activation = build_policy_activation_config(body)
        logger.info(
            "[sim_policy_activate] 激活配置解析完成: policy=%s checkpoint=%s config=%s device=%s",
            activation.policy_name,
            activation.policy_path,
            activation.config_path,
            activation.device,
        )
        logger.info("[sim_policy_activate] 解析场景上下文...")
        base_scene_ctx = resolve_scene_context(
            config_path=activation.config_path,
            scene_id=activation.scene_id or None,
            scene_bundle_dir=activation.scene_bundle_dir,
        )
        logger.info(
            "[sim_policy_activate] 基础场景解析完成: scene_id=%s depots=%d stations=%d "
            "trucks=%d drones=%d static_orders=%d dynamic_orders=%d",
            base_scene_ctx.scene_id,
            len(base_scene_ctx.depots),
            len(base_scene_ctx.stations),
            len(base_scene_ctx.trucks),
            len(base_scene_ctx.drones),
            len(base_scene_ctx.static_orders),
            len(base_scene_ctx.dynamic_orders),
        )

        use_current_init_payload = bool(body.get("use_current_init_payload", True))
        if use_current_init_payload and _last_init_payload is not None:
            logger.info("[sim_policy_activate] 使用最近一次前端 init payload 重建在线场景...")
            init_scene_id = str(_last_init_payload.get("scene_id", "")).strip()
            requested_scene_id = activation.scene_id or base_scene_ctx.scene_id
            if init_scene_id and requested_scene_id and init_scene_id != requested_scene_id:
                raise ValueError(
                    "当前 init scene_id 与 policy 激活 scene_id 不一致: "
                    f"init={init_scene_id}, requested={requested_scene_id}"
                )
            scene_ctx = _build_policy_scene_context_from_last_init(base_scene_ctx, _last_init_payload)
        else:
            logger.info(
                "[sim_policy_activate] 使用配置默认场景: use_current_init_payload=%s has_last_init_payload=%s",
                use_current_init_payload,
                _last_init_payload is not None,
            )
            scene_ctx = base_scene_ctx

        logger.info("[sim_policy_activate] 暂停 classic sim engine，开始创建 PPO runtime...")
        _sim_engine.pause()
        runtime = OnlinePolicyRuntimePlayer(
            activation=activation,
            scene_ctx=scene_ctx,
        )
        logger.info("[sim_policy_activate] PPO runtime 创建完成，切换 snapshot builder...")
        _switch_to_policy_runtime(runtime)

        try:
            logger.info("[sim_policy_activate] 构造并广播激活后首帧...")
            broadcast_tick(runtime.build_tick_payload())
            logger.info("[sim_policy_activate] 首帧广播完成")
        except Exception:
            logger.exception("[sim_policy_activate] 激活后首帧广播失败")

        elapsed_sec = time.perf_counter() - started_at
        logger.info(
            "[sim_policy_activate] 激活成功: scene_id=%s sim_time=%.3f speed_ratio=%.3f elapsed=%.2fs",
            scene_ctx.scene_id,
            runtime.current_time,
            runtime.speed_ratio,
            elapsed_sec,
        )
        return jsonify(
            {
                "status": "activated",
                "runtime": {
                    "type": "training_env_adapter_policy",
                    "policy_name": runtime.policy_name,
                    "checkpoint": runtime.checkpoint_path,
                    "scene_id": scene_ctx.scene_id,
                    "is_running": runtime.is_running,
                    "sim_time": round(runtime.current_time, 3),
                    "speed_ratio": runtime.speed_ratio,
                },
            }
        )
    except Exception as exc:
        logger.exception(
            "[sim_policy_activate] 激活失败: elapsed=%.2fs",
            time.perf_counter() - started_at,
        )
        return jsonify({"error": str(exc)}), 422


# ══════════════════════════════════════════════════════════════════════════════
# POST /policy/deactivate
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/policy/deactivate", methods=["POST"])
def sim_policy_deactivate():
    """停用 PPO 在线运行时，恢复 classic SimulationEngine。"""
    active = _policy_runtime is not None
    _shutdown_policy_runtime()
    return jsonify(
        {
            "status": "deactivated" if active else "noop",
            "active_runtime": "classic_sim_engine",
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /policy/state
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/policy/state", methods=["GET"])
def sim_policy_state():
    """返回当前 PPO 在线运行时状态。"""
    if _policy_runtime is None:
        return jsonify({"active": False})

    snapshot = _policy_runtime.build_full_snapshot()["payload"]
    return jsonify(
        {
            "active": True,
            "runtime_type": "training_env_adapter_policy",
            "policy_name": _policy_runtime.policy_name,
            "checkpoint": _policy_runtime.checkpoint_path,
            "snapshot": snapshot,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /policy/events
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/policy/events", methods=["GET"])
def sim_policy_events():
    """按 event_seq 拉取 PPO 在线决策事件。"""
    if _policy_runtime is None:
        return jsonify(
            {
                "active": False,
                "events": [],
                "latest_event_seq": 0,
            }
        )
    try:
        after_seq = int(request.args.get("after_seq", 0))
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        return jsonify({"error": "after_seq 和 limit 必须为整数"}), 400

    events = _policy_runtime.get_decision_events(after_seq=after_seq, limit=limit)
    return jsonify(
        {
            "active": True,
            "events": events,
            "latest_event_seq": _policy_runtime.latest_decision_event_seq(),
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /dispatch
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/dispatch", methods=["POST"])
def sim_dispatch():
    """
    触发调度决策（支持可切换求解器）。

    将所有待分配订单（pending_orders）传递给当前求解器，
    返回分配方案并将订单状态更新为 assigned。

    请求体：
      {"solver": "greedy", "bbox": {"minx": float, "miny": float, "maxx": float, "maxy": float}}

    返回：
      {
        "status": "ok",
        "plan": {
          "total_orders": 5,
          "feasible": 4,
          "modes": {"B": 2, "C": 2},
          "cost_total": 12345.67,
          "allocations": [...]
        },
        "pending_count": 1,
        "assigned_count": 4,
        "timestamp": <float>
      }
    """
    if _policy_runtime is not None:
        return jsonify({"error": "当前处于 PPO 在线策略模式，/api/sim/dispatch 不可用"}), 409
    if _training_runtime is not None and _training_runtime.is_running:
        return jsonify({"error": "当前处于 PPO 训练可视化模式，/api/sim/dispatch 不可用"}), 409

    if not _dispatch_engine:
        return jsonify({"error": "调度引擎未初始化，请先调用 /api/sim/init"}), 409

    body = request.get_json(silent=True) or {}
    solver = body.get("solver", "greedy")
    bbox = body.get("bbox")
    scene_id = body.get("scene_id")  # 可选：预设场景 ID
    reuse_static_ga_plan = _optional_bool(body.get("reuse_static_ga_plan"))

    logger.debug(f"[sim_dispatch] 收到请求体: {body}")

    try:
        _dispatch_engine.set_solver(str(solver))
    except ValueError as exc:
        return jsonify({
            "error": str(exc),
            "available_solvers": _dispatch_engine.get_available_solvers(),
        }), 400

    solver_impl = getattr(_dispatch_engine, "solver", None)
    if hasattr(solver_impl, "set_static_plan_cache_reuse"):
        solver_impl.set_static_plan_cache_reuse(reuse_static_ga_plan)

    if not bbox or not all(k in bbox for k in ["minx", "miny", "maxx", "maxy"]):
        logger.error(f"[sim_dispatch] bbox 缺失或格式错误: {bbox}")
        return jsonify({"error": "缺少 bbox 参数"}), 400

    try:
        current_time = _sim_engine.current_time
        plan = _dispatch_engine.execute(current_time, bbox, scene_id=scene_id)

        # 首次调度后注册自动增量调度，使后续动态订单能被自动处理
        if _sim_engine._dispatch_engine is None:
            _sim_engine.attach_dispatch_engine(_dispatch_engine, bbox, scene_id=scene_id)
            # 初始化快照为当前已分配的订单，避免下一帧重复调度
            _sim_engine._pending_snapshot = set(_order_mgr.pending_orders.keys())

        truck_routes = {
            truck_id: _serialize_truck_route(route)
            for truck_id, route in plan.truck_routes.items()
        }

        drone_routes = []
        for alloc in plan.allocations:
            # 首先确保订单从 pending 移动到 assigned
            if alloc.order_id in _order_mgr.pending_orders and alloc.feasible:
                order = _order_mgr.pending_orders.pop(alloc.order_id)
                _order_mgr.assigned_orders[alloc.order_id] = order
            
            order = _order_mgr.assigned_orders.get(alloc.order_id) or _order_mgr.pending_orders.get(alloc.order_id)
            if order is None:
                logger.warning(f"[sim_dispatch] 找不到订单 {alloc.order_id}，跳过该分配")
                continue

            # 优先使用执行层真实航线，确保与动态仿真一致。
            runtime_route = plan.drone_routes.get(alloc.drone_id) if alloc.drone_id else None
            route = _serialize_runtime_drone_route(alloc, runtime_route)
            if route is None:
                route = _serialize_drone_route(alloc, order, current_time, plan.truck_routes)
            if route is not None:
                drone_routes.append(route)

        runtime_metrics = {}
        if hasattr(_dispatch_engine, "get_runtime_metrics"):
            try:
                runtime_metrics = _dispatch_engine.get_runtime_metrics()
            except Exception:
                runtime_metrics = {}

        return jsonify({
            "status": "ok",
            "plan": {
                "total_orders":  plan.summary.get("total_orders", 0),
                "feasible":      plan.summary.get("feasible", 0),
                "modes":         plan.summary.get("modes", {}),
                "cost_total":    round(plan.cost_total, 2),
                "cost_breakdown": plan.summary.get("cost_breakdown", {}),
                "static_cache_reused": bool(plan.summary.get("static_cache_reused", False)),
                "static_cache_path": plan.summary.get("static_cache_path", ""),
                "allocations": [
                    {
                        "order_id": a.order_id,
                        "vehicle_id": a.vehicle_id,
                        "mode": a.mode,
                        "distance": round(a.distance, 2),
                        "feasible": a.feasible,
                        "reason": a.reason,
                        "recovery_station_id": a.recovery_station_id,
                        "launch_station_id": getattr(a, "launch_station_id", ""),
                        "launch_time": float(getattr(a, "launch_time", 0.0) or 0.0),
                        "wait_duration": float(getattr(a, "wait_duration", 0.0) or 0.0),
                        "drone_id": a.drone_id,
                    }
                    for a in plan.allocations
                ],
                "truck_routes": truck_routes,
                "drone_routes": drone_routes,
            },
            "runtime_metrics": runtime_metrics,
            "pending_count":  len(_order_mgr.pending_orders),
            "assigned_count": len(_order_mgr.assigned_orders),
            "timestamp":      current_time,
        })

    except Exception as exc:
        logger.exception("[sim_dispatch] 调度失败")
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# GET /preset/entities/<preset_id>
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/preset/entities/<preset_id>", methods=["GET"])
def api_get_preset_entities(preset_id: str):
    """
    返回预设场景的实体配置（仓库、充电站、无人机、任务点）。
    
    路径参数:
      preset_id : 预设场景 ID，如 'default_test_4x4km'
    
    响应:
      200 — JSON 包含 depots, stations, trucks, drones, orders
      404 — 预设场景不存在
    """
    try:
        preset = get_preset_scene(preset_id)
        if not preset:
            return jsonify({"error": f"预设场景 '{preset_id}' 不存在"}), 404
        
        od = preset.get("orders") or {}
        result = {
            "depots": preset.get("entities", {}).get("depots", []),
            "stations": preset.get("entities", {}).get("stations", []),
            "trucks": preset.get("entities", {}).get("trucks", []),
            "drones": preset.get("entities", {}).get("drones", []),
            "orders": od.get("static_orders", []),
            "dynamic_orders": od.get("dynamic_orders", []),
        }
        return jsonify(result)
    except Exception as exc:
        logger.exception(f"[api_get_preset_entities] 加载预设实体失败")
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# POST /preset/entities/<preset_id>（保存调整后的预设场景）
# ══════════════════════════════════════════════════════════════════════════════

@sim_bp.route("/preset/entities/<preset_id>", methods=["POST"])
def api_save_preset_entities(preset_id: str):
    """
    保存调整后的预设场景实体和任务点到磁盘
    
    路径参数:
      preset_id : 预设场景 ID，如 'default_test_4x4km'
    
    请求体 (JSON):
      entities : 包含 depots, stations, trucks, drones 的对象
      orders   : 包含 static_orders 的对象
    
    响应:
      200 — {"success": true}
      400 — 请求体格式错误
      404 — 预设场景不存在
      500 — 保存失败
    """
    try:
        payload = request.json or {}
        entities = payload.get("entities", {})
        orders = payload.get("orders", {})
        
        if not entities or not orders:
            return jsonify({"error": "缺少 entities 或 orders"}), 400
        
        success = save_preset_entities(preset_id, entities, orders)
        if not success:
            return jsonify({"error": f"预设场景 '{preset_id}' 不存在"}), 404
        
        return jsonify({"success": True, "message": f"预设场景 '{preset_id}' 已保存"})
    except Exception as exc:
        logger.exception(f"[api_save_preset_entities] 保存预设场景失败")
        return jsonify({"error": str(exc)}), 500

