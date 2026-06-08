#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 实体管理器 (Section 2.1)

EntityManager 是后端后端 Single Source of Truth 的实体容器层，持有并驱动所有
Depot / SwapStation / Truck / Drone 实例，并维护前端 UI 字段的 Sidecar 元数据。

生命周期：
  1. POST /api/sim/init   → load_from_config(config_json) 重建所有实体
  2. SimEngine 每 100ms  → tick_all(current_time, dt, order_mgr)  物理步进（订单归档需传入 OrderManager）
  3. WebSocket 推送      → get_telemetry()                TICK 帧数据
  4. 建连/请求           → get_static_snapshot()          FULL_SNAPSHOT 数据

导入规则（依赖 app.py 已将 BASE_DIR 注入 sys.path）：
  from core.entities.xxx import Xxx
  from utils.coord_utils import wgs84_to_utm
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from config.loader import load_solver_energy_params
from core.entities.depot import Depot
from core.entities.drone import Drone, HeavyDrone, LightDrone
from core.entities.primitives import Position3D, SourceType
from core.entities.swap_station import SwapStation
from core.entities.truck import Truck
from utils.coord_utils import wgs84_to_utm

if TYPE_CHECKING:
    from order_manager import OrderManager

logger = logging.getLogger(__name__)

# ── 调试开关 ────────────────────────────────────────────────────────
# 设为 False 以禁用冗长的卡车位置日志
DEBUG_TRUCK_POSITION = False

# 无人机类型名称 → 类映射
_DRONE_CLASS_MAP: dict[str, type[Drone]] = {
    "LightDrone": LightDrone,
    "HeavyDrone": HeavyDrone,
}


class EntityManager:
    """
    全局实体容器。

    存储结构：
      depots   : {depot_id:   Depot}
      stations : {station_id: SwapStation}
      trucks   : {truck_id:   Truck}
      drones   : {drone_id:   Drone}
      _metadata: {entity_id: {name, type, [home_depot_id]}}
                 存储 core.entities 构造函数不接受的 UI 字段
    """

    def __init__(self) -> None:
        self.depots:   dict[str, Depot]       = {}
        self.stations: dict[str, SwapStation] = {}
        self.trucks:   dict[str, Truck]       = {}
        self.drones:   dict[str, Drone]       = {}
        self._metadata: dict[str, dict]       = {}

        runtime_cfg = load_solver_energy_params()
        self.TRUCK_SERVICE_TIME_ORDER = runtime_cfg.truck_service_time_order_s
        self.DRONE_SERVICE_TIME_ORDER = runtime_cfg.drone_service_time_order_s
        self.TRUCK_DRONE_LAUNCH_TIME = runtime_cfg.truck_drone_launch_time_s
        self.TRUCK_DRONE_RECOVER_TIME = runtime_cfg.truck_drone_recover_time_s
        # 停靠触发容差：recovery/station 采用更宽容阈值，降低路网几何误差导致的漏触发。
        self.STOP_PROXIMITY_BASE_M = 15.0
        self.STOP_PROXIMITY_SPEED_FACTOR = 1.5
        self.RECOVERY_PROXIMITY_BASE_M = 45.0
        self.RECOVERY_PROXIMITY_SPEED_FACTOR = 4.0
        # 超过该时长仍未命中的 recovery/station 视作过期，跳过以避免阻塞后续 customer 完成事件。
        self.STALE_RECOVERY_SKIP_S = 20.0

    # ══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ══════════════════════════════════════════════════════════════════════════

    def load_from_config(self, config_json: dict) -> None:
        """
        从 /api/sim/init 请求体中的 entities 字段实例化所有实体。

        必须严格按四步顺序执行（详见 Section 2.1 设计文档）：
          1. 基础设施（Depot / SwapStation）
          2a. Truck（先于 Drone，供后者读取初始坐标）
          2b. Drone（按 home_type 从 Depot/Truck 取初始坐标）
          3. 关联注册（register_drone / docked_drones.append / register_truck）
          4. Sidecar 元数据填充

        Args:
            config_json: /api/sim/init 请求体（包含 entities 子字典）
        """
        self.depots.clear()
        self.stations.clear()
        self.trucks.clear()
        self.drones.clear()
        self._metadata.clear()

        entities = config_json.get("entities", {})

        # ── 步骤 1：实例化基础设施 ───────────────────────────────────────────
        for d_cfg in entities.get("depots", []):
            x, y = wgs84_to_utm(d_cfg["lng"], d_cfg["lat"])
            loc   = Position3D(x=x, y=y, z=float(d_cfg.get("altitude", 0)))
            depot = Depot(
                depot_id=d_cfg["depot_id"],
                location=loc,
                swap_time=float(d_cfg["swap_time"]),
                parking_slots=int(d_cfg["parking_slots"]),
                capacity=int(d_cfg.get("capacity", 1000)),
            )
            self.depots[d_cfg["depot_id"]] = depot
            logger.debug("[EntityManager] 创建 Depot %s", d_cfg["depot_id"])

        for s_cfg in entities.get("stations", []):
            x, y = wgs84_to_utm(s_cfg["lng"], s_cfg["lat"])
            loc     = Position3D(x=x, y=y, z=float(s_cfg.get("altitude", 0)))
            station = SwapStation(
                station_id=s_cfg["station_id"],
                location=loc,
                swap_time=float(s_cfg["swap_time"]),
                parking_slots=int(s_cfg["parking_slots"]),
            )
            self.stations[s_cfg["station_id"]] = station
            logger.debug("[EntityManager] 创建 SwapStation %s (输入: lng=%.6f, lat=%.6f; UTM转换后: x=%.2f, y=%.2f)", 
                         s_cfg["station_id"], s_cfg["lng"], s_cfg["lat"], x, y)

        # ── 步骤 2a：实例化 Truck（必须先于 Drone）──────────────────────────
        for t_cfg in entities.get("trucks", []):
            home_depot_id = t_cfg["home_depot_id"]
            if home_depot_id not in self.depots:
                raise ValueError(
                    f"Truck {t_cfg['truck_id']} 引用的 home_depot_id '{home_depot_id}' 不存在，"
                    "请检查 entities.depots 配置。"
                )
            depot_loc = self.depots[home_depot_id].location
            init_loc  = Position3D(x=depot_loc.x, y=depot_loc.y, z=0.0)
            truck = Truck(
                truck_id=t_cfg["truck_id"],
                speed=float(t_cfg["speed"]),
                max_inventory=int(t_cfg["max_inventory"]),
                swap_time=float(t_cfg["swap_time"]),
                parking_slots=int(t_cfg["parking_slots"]),
                init_loc=init_loc,
            )
            self.trucks[t_cfg["truck_id"]] = truck
            logger.debug("[EntityManager] 创建 Truck %s", t_cfg["truck_id"])

        # ── 步骤 2b：实例化 Drone（Truck 已就绪）────────────────────────────
        for dr_cfg in entities.get("drones", []):
            drone_type_name = dr_cfg.get("drone_type", "LightDrone")
            drone_class = _DRONE_CLASS_MAP.get(drone_type_name)
            if drone_class is None:
                raise ValueError(
                    f"未知无人机类型 '{drone_type_name}'，"
                    f"支持: {list(_DRONE_CLASS_MAP.keys())}"
                )
            home_id   = dr_cfg["home_id"]
            home_type = dr_cfg["home_type"]   # "DEPOT" | "TRUCK"

            if home_type == "DEPOT":
                if home_id not in self.depots:
                    raise ValueError(
                        f"Drone {dr_cfg['drone_id']} 的 home_id '{home_id}' 不在 depots 中。"
                    )
                init_loc = self.depots[home_id].location
            elif home_type == "TRUCK":
                if home_id not in self.trucks:
                    raise ValueError(
                        f"Drone {dr_cfg['drone_id']} 的 home_id '{home_id}' 不在 trucks 中。"
                    )
                init_loc = self.trucks[home_id].current_loc
            else:
                raise ValueError(
                    f"Drone {dr_cfg['drone_id']} 的 home_type '{home_type}' 无效，"
                    "必须为 'DEPOT' 或 'TRUCK'。"
                )

            drone = drone_class(
                drone_id=dr_cfg["drone_id"],
                home_id=home_id,
                home_type=SourceType(home_type),
                init_loc=init_loc,
            )
            self.drones[dr_cfg["drone_id"]] = drone
            logger.debug("[EntityManager] 创建 %s %s", drone_type_name, dr_cfg["drone_id"])

        # ── 步骤 3：关联注册 ─────────────────────────────────────────────────
        for dr_cfg in entities.get("drones", []):
            drone_id  = dr_cfg["drone_id"]
            home_id   = dr_cfg["home_id"]
            home_type = dr_cfg["home_type"]
            if home_type == "DEPOT":
                self.depots[home_id].register_drone(drone_id, is_idle=True)
            else:  # TRUCK
                if drone_id not in self.trucks[home_id].docked_drones:
                    self.trucks[home_id].docked_drones.append(drone_id)

        for t_cfg in entities.get("trucks", []):
            self.depots[t_cfg["home_depot_id"]].register_truck(t_cfg["truck_id"])

        # ── 步骤 4：填充 Sidecar 元数据 ──────────────────────────────────────
        for d_cfg in entities.get("depots", []):
            self._metadata[d_cfg["depot_id"]] = {
                "name": d_cfg.get("name", d_cfg["depot_id"]),
                "type": "DEPOT",
            }
        for s_cfg in entities.get("stations", []):
            self._metadata[s_cfg["station_id"]] = {
                "name": s_cfg.get("name", s_cfg["station_id"]),
                "type": "STATION",
            }
        for t_cfg in entities.get("trucks", []):
            self._metadata[t_cfg["truck_id"]] = {
                "name":          t_cfg.get("name", t_cfg["truck_id"]),
                "type":          "TRUCK",
                "home_depot_id": t_cfg["home_depot_id"],
            }
        for dr_cfg in entities.get("drones", []):
            self._metadata[dr_cfg["drone_id"]] = {
                "type": "DRONE",
                # home_id / home_type 已在 Drone 实例构造函数中存储，无需 sidecar
            }

        logger.info(
            "[EntityManager] 加载完成：%d 仓库，%d 换电站，%d 卡车，%d 无人机",
            len(self.depots), len(self.stations), len(self.trucks), len(self.drones),
        )
        
        # 诊断输出：打印所有充电站的坐标
        if self.stations:
            logger.info("[EntityManager] 充电站坐标清单（诊断用）：")
            for station_id, station in self.stations.items():
                logger.info(f"  {station_id}: UTM({station.location.x:.2f}, {station.location.y:.2f})")
        if self.depots:
            logger.info("[EntityManager] 仓库坐标清单（诊断用）：")
            for depot_id, depot in self.depots.items():
                logger.info(f"  {depot_id}: UTM({depot.location.x:.2f}, {depot.location.y:.2f})")

    # ══════════════════════════════════════════════════════════════════════════
    # 快照序列化
    # ══════════════════════════════════════════════════════════════════════════

    def get_static_snapshot(self) -> dict:
        """
        返回所有实体的完整静态元数据 + 运行时状态，用于 FULL_SNAPSHOT 首帧。

        实现：对每个实体调用 to_telemetry_dict()，再与 _metadata 合并，
        使 name / home_depot_id 等 UI 字段出现在输出中。

        Returns:
            dict with keys: depots, stations, trucks, drones（各为列表）
        """
        trucks_list = []
        for tid, truck in self.trucks.items():
            try:
                truck_data = {**truck.to_telemetry_dict(), **self._metadata.get(tid, {})}
                trucks_list.append(truck_data)
            except Exception as e:
                logger.error(f"[get_static_snapshot] 序列化卡车 {tid} 失败: {e}", exc_info=True)
                # 继续处理其他卡车
        
        return {
            "depots": [
                {**depot.to_telemetry_dict(), **self._metadata.get(did, {})}
                for did, depot in self.depots.items()
            ],
            "stations": [
                {**station.to_telemetry_dict(), **self._metadata.get(sid, {})}
                for sid, station in self.stations.items()
            ],
            "trucks": trucks_list,
            "drones": [
                {**drone.to_telemetry_dict(), **self._metadata.get(did, {})}
                for did, drone in self.drones.items()
            ],
        }

    def get_telemetry(self) -> dict:
        """
        返回所有实体的动态运行时字段，用于 TICK 帧（100ms 广播）。

        实现：对每个实体调用 to_dynamic_state()，再与 _metadata 合并，
        确保 TruckConfig.name / home_depot_id 等非 Optional TypeScript 字段
        不因 setRuntimeAll 全量替换而变成 undefined。（v4.9 修正）

        Returns:
            dict with keys: depots, stations, trucks, drones（各为列表）
        """
        try:
            trucks_list = []
            for tid, truck in self.trucks.items():
                try:
                    truck_state = {**truck.to_dynamic_state(), **self._metadata.get(tid, {})}
                    trucks_list.append(truck_state)
                except Exception as e:
                    logger.error(f"[get_telemetry] 序列化卡车 {tid} 失败: {e}", exc_info=True)
                    # 返回至少有 truck_id 的不完整状态
                    trucks_list.append({"truck_id": tid, "error": str(e)})

            return {
                "depots": [
                    {**depot.to_dynamic_state(), **self._metadata.get(did, {})}
                    for did, depot in self.depots.items()
                ],
                "stations": [
                    {**station.to_dynamic_state(), **self._metadata.get(sid, {})}
                    for sid, station in self.stations.items()
                ],
                "trucks": trucks_list,
                "drones": [
                    {**drone.to_dynamic_state(), **self._metadata.get(did, {})}
                    for did, drone in self.drones.items()
                ],
            }
        except Exception as e:
            logger.error(f"[get_telemetry] 构建遥测数据失败: {e}", exc_info=True)
            return {"depots": [], "stations": [], "trucks": [], "drones": []}

    # ══════════════════════════════════════════════════════════════════════════
    # 物理步进
    # ══════════════════════════════════════════════════════════════════════════

    def tick_all(
        self,
        current_time: float,
        dt: float,
        order_mgr: Optional["OrderManager"] = None,
    ) -> None:
        """
        驱动所有实体完成一个物理时间步。

        仅调用已在各实体类中实现的 tick_update()。
        无人机与卡车的主动运动逻辑（move_step / consume_energy）由调度引擎
        在 Phase 4 接管后注入；当前阶段各实体停止在初始位置，仅基础设施充换电队列推进。

        Args:
            current_time: 仿真累计时间（秒）
            dt:           本步推进的仿真时长（秒）= 0.1 × speed_ratio
            order_mgr:    订单管理器；无人机/卡车完成配送归档时写入，与求解器类型无关。
                          未传入时无法将订单移入 completed（会打日志告警）。
        """
        for depot in self.depots.values():
            try:
                depot.tick_update(current_time)
            except Exception:
                logger.exception("[EntityManager.tick_all] Depot %s tick 异常", depot.depot_id)

        for station in self.stations.values():
            try:
                station.tick_update(current_time)
            except Exception:
                logger.exception("[EntityManager.tick_all] Station %s tick 异常", station.station_id)

        # 卡车提供了 tick_update（继承自 ChargingHost），无人机暂无独立 tick
        for truck in self.trucks.values():
            try:
                truck.tick_update(current_time)
                # 按计划停靠时段冻结卡车位置，确保“到达-离开”时间在物理上生效。
                wait_stop = self._get_truck_wait_stop(truck, current_time)
                if wait_stop is not None:
                    from core.entities.primitives import TruckStatus

                    wait_pos = wait_stop.get("position")
                    if wait_pos is not None:
                        truck.current_loc = wait_pos
                    # 通过平移 departure_time 冻结里程推进，避免恢复后瞬移。
                    truck._departure_time += dt
                    truck.status = TruckStatus.WAITING
                else:
                    if truck._route_data and truck.status.value == "WAITING":
                        from core.entities.primitives import TruckStatus
                        truck.status = TruckStatus.DRIVING
                    # 推进卡车位置（如果已设置路由）
                    truck.move_step(dt, current_time)
                # 基于调度器下发的关键节点时序，执行 customer/recovery 事件。
                self._process_truck_route_events(truck, current_time, order_mgr)
                if DEBUG_TRUCK_POSITION and truck.status.value == "DRIVING" and truck._route_data:
                    logger.debug(
                        "[EntityManager] Truck %s: status=%s, pos=(%f, %f), time=%.2f",
                        truck.truck_id, truck.status.value,
                        truck.current_loc.x, truck.current_loc.y, current_time
                    )
            except Exception:
                logger.exception("[EntityManager.tick_all] Truck %s tick 异常", truck.truck_id)

        # ── 无人机物理步进 ────────────────────────────────────────────────────
        from core.entities.primitives import WaypointAction, DroneStatus
        
        for drone in self.drones.values():
            try:
                # 无人机配送点服务停留：到达 DELIVER 后暂停一段时间再继续后续航路。
                delivery_service_end = float(getattr(drone, "delivery_service_end_time", 0.0))
                if delivery_service_end > 0.0:
                    if current_time + 1e-6 < delivery_service_end:
                        continue
                    pending_order = getattr(drone, "pending_release_order_id", None)
                    if pending_order:
                        released = drone.release_order()
                        if released:
                            self._complete_assigned_order(
                                released,
                                current_time,
                                source=f"drone {drone.drone_id}",
                                order_mgr=order_mgr,
                            )
                    drone.delivery_service_end_time = 0.0
                    drone.pending_release_order_id = None
                    # 服务刚结束这一拍不推进位移，避免视觉上瞬移。
                    continue

                # B_WAIT：无人机由卡车运输到起飞站前，位置与卡车绑定且不耗电。
                transport_truck_id = getattr(drone, "transport_truck_id", None)
                if transport_truck_id:
                    carrier = self.trucks.get(transport_truck_id)
                    if carrier is not None:
                        drone.current_loc = carrier.current_loc
                        launch_time = float(getattr(drone, "scheduled_launch_time", 0.0))
                        if current_time + 1e-6 < launch_time:
                            continue

                        # 起飞前必须校验卡车已到达起飞站，防止增量改路由后“无人机瞬移放飞”。
                        launch_loc = None
                        launch_station_id = getattr(drone, "launch_station_id", "")
                        launch_station = self.stations.get(launch_station_id) if launch_station_id else None
                        if launch_station is not None:
                            launch_loc = launch_station.location
                        if launch_loc is None and launch_station_id:
                            launch_depot = self.depots.get(launch_station_id)
                            if launch_depot is not None:
                                launch_loc = launch_depot.location
                        if launch_loc is None and drone.has_pending_route:
                            launch_idx = drone.current_waypoint_index
                            if 0 <= launch_idx < len(drone.route_plan):
                                launch_loc = drone.route_plan[launch_idx].loc

                        if launch_loc is None:
                            continue

                        carrier_speed = float(getattr(carrier, "speed", 0.0))
                        if str(getattr(drone, "_runtime_solver", "") or "").lower() == "ga_mmce":
                            launch_tol_m = max(
                                self.RECOVERY_PROXIMITY_BASE_M,
                                carrier_speed * self.RECOVERY_PROXIMITY_SPEED_FACTOR,
                            )
                        else:
                            launch_tol_m = max(15.0, carrier_speed * 1.5)
                        dist_to_launch = carrier.current_loc.distance_2d(launch_loc)
                        if dist_to_launch > launch_tol_m:
                            last_log_t = float(getattr(drone, "_last_launch_wait_log_time", -1e9))
                            if current_time - last_log_t >= 10.0:
                                logger.info(
                                    "[EntityManager.tick_all] 无人机 %s 等待卡车 %s 到达起飞站 %s "
                                    "(dist=%.1fm, tol=%.1fm)",
                                    drone.drone_id,
                                    carrier.truck_id,
                                    launch_station_id or "-",
                                    dist_to_launch,
                                    launch_tol_m,
                                )
                                drone._last_launch_wait_log_time = current_time
                            continue

                        drone.current_loc = launch_loc

                        if drone.drone_id in carrier.docked_drones:
                            carrier.docked_drones.remove(drone.drone_id)
                        drone.transport_truck_id = None
                        drone.status = DroneStatus.FLYING_TO_PICKUP
                        logger.info(
                            "[EntityManager.tick_all] 无人机 %s 在 t=%.1fs 从卡车 %s 放飞",
                            drone.drone_id, current_time, carrier.truck_id,
                        )
                        # 放飞后下一拍再推进无人机飞行，避免同拍消耗电量并造成视觉突变。
                        continue
                    else:
                        # 兜底：承运卡车不存在时，解除运输绑定避免永久卡住。
                        drone.transport_truck_id = None

                # 非飞行状态不推进航路，也不消耗飞行电量。
                if not drone.status.is_flying:
                    continue

                # 推进无人机位置并消耗电量。
                action = drone.move_step(dt)
                drone.consume_energy(dt)
                
                if action is not None:
                    logger.debug(
                        "[EntityManager.tick_all] 无人机 %s 到达航路点，动作: %s, 位置: %s",
                        drone.drone_id, action.value, drone.current_loc
                    )

                    reached_wp = None
                    if 0 < drone.current_waypoint_index <= len(drone.route_plan):
                        reached_wp = drone.route_plan[drone.current_waypoint_index - 1]
                    reached_target = reached_wp.target_entity_id if reached_wp else None

                    runtime_handler = getattr(drone, "_runtime_action_handler", None)
                    if callable(runtime_handler) and runtime_handler(
                        entity_mgr=self,
                        order_mgr=order_mgr,
                        drone=drone,
                        action=action,
                        reached_wp=reached_wp,
                        reached_target=reached_target,
                        current_time=current_time,
                    ):
                        continue

                    # ── 处理航路点 action ──────────────────────────────────────
                    if action == WaypointAction.DOCK_DEPOT or action == WaypointAction.DOCK_TRUCK:
                        dock_target = reached_target or "-"
                        self._recharge_drone_to_full(drone, f"回收到 {dock_target}")
                        # 到达回收点后，若落在充电站则等待卡车回收；否则直接停靠空闲。
                        if reached_target and reached_target in self.stations:
                            drone.waiting_recovery_station_id = reached_target
                            logger.info(
                                "[EntityManager.tick_all] 无人机 %s 已落地充电站 %s，等待卡车回收",
                                drone.drone_id, reached_target,
                            )
                        else:
                            drone.waiting_recovery_station_id = ""
                            logger.info(
                                "[EntityManager.tick_all] 无人机 %s 已到达回收点 %s，状态转为 IDLE",
                                drone.drone_id, reached_target or "-",
                            )
                        drone.status = DroneStatus.IDLE
                    
                    elif action == WaypointAction.DELIVER:
                        # 无人机到达配送点后执行卸货停留，再完成订单。
                        if drone.carrying_order_id:
                            drone.pending_release_order_id = drone.carrying_order_id
                            drone.delivery_service_end_time = current_time + self.DRONE_SERVICE_TIME_ORDER
                            logger.info(
                                "[EntityManager.tick_all] 无人机 %s 到达配送点，开始卸货停留 %.1fs",
                                drone.drone_id,
                                self.DRONE_SERVICE_TIME_ORDER,
                            )
                    
                    elif action == WaypointAction.PICKUP:
                        # 无人机到达取货点（订单状态已在 decision_engine 中转为 DELIVERING）
                        logger.debug(
                            "[EntityManager.tick_all] 无人机 %s 到达取货点，订单: %s",
                            drone.drone_id, drone.carrying_order_id or "无"
                        )
            except Exception:
                logger.exception("[EntityManager.tick_all] Drone %s tick 异常", drone.drone_id)

    def _process_truck_route_events(
        self,
        truck: Truck,
        current_time: float,
        order_mgr: Optional["OrderManager"],
    ) -> None:
        """按时间顺序执行卡车关键节点事件（customer/recovery/station/depot）。"""
        planned_stops = getattr(truck, "_planned_route_stops", None)
        if not planned_stops:
            return

        cursor = int(getattr(truck, "_planned_route_cursor", 0))
        while cursor < len(planned_stops):
            stop = planned_stops[cursor]
            node_type = stop.get("node_type", "")
            # recovery 的回收动作应在卡车到达后立即尝试执行；是否继续等待由
            # departure_time 控制，避免已落站无人机被旧等待窗口拖到离站时才上车。
            # GA-MMCE 的 station/depot 节点承载放飞/回收服务，仍按服务结束触发；
            # customer 保持既有语义：到达即完成订单，随后继续执行服务停留。
            ga_service_stop = (
                self._is_ga_truck_route(truck)
                and node_type in {"station", "depot"}
            )

            if node_type == "recovery":
                arrival_time = float(stop.get("arrival_time", float("inf")))
                departure_time = float(stop.get("departure_time", arrival_time))
                recovery_done = bool(stop.get("_recovery_event_done", False))

                if not recovery_done:
                    if arrival_time > current_time + 1e-6:
                        break
                    if not self._truck_is_near_stop(truck, stop):
                        overdue_s = current_time - departure_time
                        if overdue_s >= self.STALE_RECOVERY_SKIP_S:
                            logger.warning(
                                "[EntityManager] 卡车 %s 跳过过期%s停靠 %s（overdue=%.1fs），避免阻塞后续事件",
                                truck.truck_id,
                                node_type,
                                stop.get("node_id", ""),
                                overdue_s,
                            )
                            cursor += 1
                            continue
                        break
                    if self._handle_truck_stop_event(truck, stop, current_time, order_mgr):
                        stop["_recovery_event_done"] = True
                        station_id = str(stop.get("node_id", "") or "")
                        order_id = str(stop.get("order_id", "") or "")
                        if station_id and not self._truck_has_pending_recovery_at_station(
                            truck,
                            station_id,
                            order_id,
                        ):
                            service_departure = current_time + max(
                                self.TRUCK_DRONE_RECOVER_TIME,
                                self.TRUCK_DRONE_LAUNCH_TIME,
                            )
                            if departure_time > service_departure + 1e-6:
                                old_departure = departure_time
                                logger.info(
                                    "[EntityManager] 卡车 %s recovery %s 已完成且无其它待回收无人机，"
                                    "收缩离开时刻 %.1fs -> %.1fs",
                                    truck.truck_id,
                                    station_id,
                                    old_departure,
                                    service_departure,
                                )
                                stop["departure_time"] = service_departure
                                departure_time = service_departure
                                self._shift_future_truck_stops(
                                    planned_stops,
                                    cursor + 1,
                                    old_departure - service_departure,
                                )
                    if departure_time > current_time + 1e-6:
                        break

                if departure_time > current_time + 1e-6:
                    break
                if not self._truck_is_near_stop(truck, stop):
                    overdue_s = current_time - departure_time
                    if overdue_s >= self.STALE_RECOVERY_SKIP_S:
                        logger.warning(
                            "[EntityManager] 卡车 %s 跳过过期%s停靠 %s（overdue=%.1fs），避免阻塞后续事件",
                            truck.truck_id,
                            node_type,
                            stop.get("node_id", ""),
                            overdue_s,
                        )
                        cursor += 1
                        continue
                    break
                if not bool(stop.get("_recovery_event_done", False)):
                    self._handle_truck_stop_event(truck, stop, current_time, order_mgr)
                    stop["_recovery_event_done"] = True
                cursor += 1
                continue

            if ga_service_stop:
                event_time = float(stop.get("departure_time", stop.get("arrival_time", float("inf"))))
            else:
                event_time = float(stop.get("arrival_time", float("inf")))

            if event_time > current_time + 1e-6:
                break
            if not self._truck_is_near_stop(truck, stop):
                if node_type in {"recovery", "station", "depot"}:
                    departure_time = float(stop.get("departure_time", event_time))
                    overdue_s = current_time - departure_time
                    if overdue_s >= self.STALE_RECOVERY_SKIP_S:
                        logger.warning(
                            "[EntityManager] 卡车 %s 跳过过期%s停靠 %s（overdue=%.1fs），避免阻塞后续事件",
                            truck.truck_id,
                            node_type,
                            stop.get("node_id", ""),
                            overdue_s,
                        )
                        cursor += 1
                        continue
                # 到时但尚未到点：等待卡车物理位置追上，避免“按时间瞬移触发”。
                break
            self._handle_truck_stop_event(truck, stop, current_time, order_mgr)
            cursor += 1

        truck._planned_route_cursor = cursor

    @staticmethod
    def _shift_future_truck_stops(
        planned_stops: list[dict],
        start_idx: int,
        delta_s: float,
    ) -> None:
        """Shift later stop event times earlier after a recovery wait is shortened."""
        if delta_s <= 1e-6:
            return
        for future_stop in planned_stops[start_idx:]:
            for key in ("arrival_time", "departure_time"):
                if key in future_stop:
                    future_stop[key] = max(0.0, float(future_stop[key]) - delta_s)

    def _get_truck_wait_stop(self, truck: Truck, current_time: float) -> Optional[dict]:
        """返回当前时刻卡车应等待的节点；GA-MMCE 额外支持 station/depot 服务停留。"""
        planned_stops = getattr(truck, "_planned_route_stops", None)
        if not planned_stops:
            return None

        wait_node_types = {"customer", "recovery"}
        if self._is_ga_truck_route(truck):
            wait_node_types.add("station")
            wait_node_types.add("depot")

        for stop in planned_stops:
            node_type = stop.get("node_type", "")
            if node_type not in wait_node_types:
                continue
            arrival = float(stop.get("arrival_time", float("inf")))
            departure = float(stop.get("departure_time", arrival))
            if departure <= arrival + 1e-6:
                continue
            if arrival <= current_time + 1e-6 < departure - 1e-6:
                if not self._truck_is_near_stop(truck, stop):
                    continue
                return stop
        return None

    def _is_ga_truck_route(self, truck: Truck) -> bool:
        """判断当前卡车计划是否来自 GA-MMCE，避免影响其它调度算法的站点语义。"""
        return str(getattr(truck, "_planned_route_solver", "") or "").lower() == "ga_mmce"

    def _truck_is_near_stop(self, truck: Truck, stop: dict) -> bool:
        """判定卡车是否已接近停靠点，防止仅按时刻触发导致位置跳变。"""
        stop_pos = stop.get("position")
        if stop_pos is None:
            return False
        node_type = str(stop.get("node_type", ""))
        speed = float(getattr(truck, "speed", 0.0))
        if node_type in {"recovery", "station", "depot"}:
            tol_m = max(self.RECOVERY_PROXIMITY_BASE_M, speed * self.RECOVERY_PROXIMITY_SPEED_FACTOR)
        else:
            tol_m = max(self.STOP_PROXIMITY_BASE_M, speed * self.STOP_PROXIMITY_SPEED_FACTOR)
        return truck.current_loc.distance_2d(stop_pos) <= tol_m

    def _handle_truck_stop_event(
        self,
        truck: Truck,
        stop: dict,
        current_time: float,
        order_mgr: Optional["OrderManager"],
    ) -> bool:
        """处理单个卡车停靠事件。"""
        node_type = stop.get("node_type", "")
        if node_type == "customer":
            order_id = stop.get("order_id", "")
            if order_id:
                self._complete_assigned_order(
                    order_id,
                    current_time,
                    source=f"truck {truck.truck_id}",
                    order_mgr=order_mgr,
                )
            return bool(order_id)

        if node_type in {"recovery", "station", "depot"}:
            station_id = stop.get("node_id", "")
            if station_id:
                return self._recover_drones_from_station_to_truck(truck, station_id, current_time)
        return False

    def _recover_drones_from_station_to_truck(
        self,
        truck: Truck,
        station_id: str,
        current_time: float,
    ) -> bool:
        """卡车到达回收站点后，将在站点等待的无人机回收到车上。"""
        from core.entities.primitives import DroneStatus

        recovered: list[str] = []
        for drone in self.drones.values():
            waiting_station_id = getattr(drone, "waiting_recovery_station_id", "")
            if waiting_station_id != station_id:
                continue
            if drone.status != DroneStatus.IDLE:
                continue

            drone.waiting_recovery_station_id = ""
            drone.transport_truck_id = truck.truck_id
            drone.scheduled_launch_time = float("inf")
            drone.current_loc = truck.current_loc
            self._recharge_drone_to_full(drone, f"卡车 {truck.truck_id} 在站点 {station_id} 回收")
            if drone.drone_id not in truck.docked_drones:
                truck.docked_drones.append(drone.drone_id)
            runtime_recovery_handler = getattr(drone, "_runtime_recovery_handler", None)
            if callable(runtime_recovery_handler):
                runtime_recovery_handler(
                    entity_mgr=self,
                    truck=truck,
                    station_id=station_id,
                    drone=drone,
                    current_time=current_time,
                )
            recovered.append(drone.drone_id)

        if recovered:
            logger.info(
                "[EntityManager.tick_all] 卡车 %s 在站点 %s 回收无人机: %s",
                truck.truck_id, station_id, recovered,
            )
        return bool(recovered)

    def _truck_has_pending_recovery_at_station(
        self,
        truck: Truck,
        station_id: str,
        stop_order_id: str = "",
    ) -> bool:
        """判断该站点是否还有已知的、需要当前卡车等待的无人机回收任务。"""
        from core.entities.primitives import WaypointAction

        def resolve_drone_order_id(drone: Drone) -> str:
            for attr_name in ("carrying_order_id", "pending_release_order_id"):
                order_id = str(getattr(drone, attr_name, "") or "")
                if order_id:
                    return order_id

            route_plan = getattr(drone, "route_plan", None) or []
            start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            for wp in route_plan[start_idx:]:
                if wp.action in (WaypointAction.PICKUP, WaypointAction.DELIVER):
                    order_id = str(wp.target_entity_id or "")
                    if order_id:
                        return order_id
            for wp in reversed(route_plan[:start_idx]):
                if wp.action in (WaypointAction.PICKUP, WaypointAction.DELIVER):
                    order_id = str(wp.target_entity_id or "")
                    if order_id:
                        return order_id
            return ""

        def order_matches(drone: Drone) -> bool:
            if not stop_order_id:
                return True
            return resolve_drone_order_id(drone) == stop_order_id

        def route_targets_station(drone: Drone) -> bool:
            route_plan = getattr(drone, "route_plan", None) or []
            start_idx = int(getattr(drone, "current_waypoint_index", 0) or 0)
            for wp in route_plan[start_idx:]:
                if wp.action in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                    return (wp.target_entity_id or "") == station_id and order_matches(drone)
            return False

        for drone in self.drones.values():
            if getattr(drone, "waiting_recovery_station_id", "") == station_id and order_matches(drone):
                return True

            transport_truck_id = getattr(drone, "transport_truck_id", "")
            owned_by_truck = (
                transport_truck_id == truck.truck_id
                or drone.drone_id in getattr(truck, "docked_drones", [])
            )
            if not owned_by_truck:
                continue

            if route_targets_station(drone):
                return True

        return False

    def _recharge_drone_to_full(self, drone: Drone, reason: str) -> None:
        """统一补能入口：无人机回收到卡车/仓库/站点后立即满电。"""
        prev_energy = drone.battery_current
        drone.recharge_to_full()
        if prev_energy < drone.battery_max - 1e-6:
            logger.info(
                "[EntityManager] 无人机 %s %s，电量补满: %.0fJ -> %.0fJ",
                drone.drone_id,
                reason,
                prev_energy,
                drone.battery_current,
            )

    def _complete_assigned_order(
        self,
        order_id: str,
        current_time: float,
        source: str,
        order_mgr: Optional["OrderManager"] = None,
    ) -> None:
        """将 assigned 订单推进到 COMPLETED 并归档。"""
        if order_mgr is None:
            logger.warning("[EntityManager] %s 完成订单 %s 失败：未传入 order_mgr", source, order_id)
            return

        order = order_mgr.assigned_orders.get(order_id)
        if order is None:
            for co in order_mgr.completed_orders:
                if co.order_id == order_id:
                    logger.info(
                        "[EntityManager] %s 回调完成订单 %s：已在 completed 池（幂等忽略）",
                        source,
                        order_id,
                    )
                    return
            logger.warning(
                "[EntityManager] %s 完成订单 %s：订单不在 assigned 且未在 completed，无法归档",
                source,
                order_id,
            )
            return

        from core.entities.primitives import TaskStatus

        try:
            if order.status == TaskStatus.ASSIGNED:
                order.update_status(TaskStatus.PICKED_UP)
                order.update_status(TaskStatus.DELIVERING)
            elif order.status == TaskStatus.PICKED_UP:
                order.update_status(TaskStatus.DELIVERING)

            if order.status not in {TaskStatus.DELIVERING, TaskStatus.TIMEOUT}:
                logger.warning(
                    "[EntityManager] %s 完成订单 %s 时状态异常: %s",
                    source, order_id, order.status.value,
                )
                return

            order.actual_deliver_time = current_time
            order.update_status(TaskStatus.COMPLETED)
        except ValueError as exc:
            logger.warning("[EntityManager] %s 完成订单 %s 失败: %s", source, order_id, exc)
            return

        order_mgr.assigned_orders.pop(order_id, None)
        if order not in order_mgr.completed_orders:
            order_mgr.completed_orders.append(order)
        logger.info("[EntityManager] %s 完成订单 %s，状态更新为 COMPLETED", source, order_id)
