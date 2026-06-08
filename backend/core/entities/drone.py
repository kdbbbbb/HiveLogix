#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 无人机实体 (Section 2)

本模块实现异构无人机的完整物理模型，包括：
  - Drone         : 基类，持有全部物理属性与通用方法
  - LightDrone    : 轻型无人机（2.5kg 载重，18m/s 巡航）
  - HeavyDrone    : 重型无人机（10kg 载重，14m/s 巡航）

功率模型（多旋翼 induced + parasitic drag 功耗方程）：
  P = k1 × (m_empty + payload)^1.5 + k2 × v^3

路由设计：
  route_plan 是 RouteWaypoint 的**只读列表**，current_waypoint_index 为当前
  执行指针，move_step() 推进指针而非弹出元素，保留完整序列以支持回放与诊断。

状态语义边界：
  - `QUEUING` / `CHARGING` 只表示无人机已经到达 ChargingHost
    （station / depot / truck）后的补能服务阶段；
  - 不表示 mode C 中的 rendezvous 或等待被卡车回收阶段。
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from config.loader import load_drone_params
from core.entities.primitives import (
    DroneStatus,
    Position3D,
    RouteWaypoint,
    SourceType,
    WaypointAction,
)

logger = logging.getLogger(__name__)


class Drone:
    """
    无人机物理实体基类。

    子类（LightDrone / HeavyDrone）在 __init__ 中固化气动参数与飞行性能，
    调度器与仿真引擎通过基类接口操作，无需感知具体型号（里氏替换原则）。

    Args:
        drone_id:   全局唯一 ID
        home_id:    归属母体 ID（仓库或卡车），返航基准
        home_type:  归属类型（DEPOT / TRUCK）
        init_loc:   初始UTM三维坐标
    """

    def __init__(
        self,
        *,
        drone_id: str,
        home_id: str,
        home_type: SourceType,
        init_loc: Position3D,
        # 气动参数（由子类赋值，基类可传入供覆盖）
        k1: float,
        k2: float,
        cruise_speed: float,
        payload_capacity: float,
        empty_weight: float,
        battery_max: float,
    ) -> None:
        if not drone_id:
            raise ValueError("drone_id 不能为空")
        if payload_capacity <= 0:
            raise ValueError(f"[{drone_id}] payload_capacity 必须为正数")
        if battery_max <= 0:
            raise ValueError(f"[{drone_id}] battery_max 必须为正数")

        # ── 身份信息 ──────────────────────────────────────────────────────────
        self.drone_id: str = drone_id
        self.home_id: str = home_id
        self.home_type: SourceType = home_type

        # ── 气动 & 飞行参数（运行期不可变，子类在 __init__ 中固化）──────────
        self.k1: float = k1           # 诱导功率系数 [W/kg^1.5]
        self.k2: float = k2           # 废阻功率系数 [W/(m/s)^3]
        self.cruise_speed: float = cruise_speed         # [m/s]
        self.payload_capacity: float = payload_capacity # [kg]
        self.empty_weight: float = empty_weight         # [kg]
        self.battery_max: float = battery_max           # [J]

        # ── 动态物理状态 ──────────────────────────────────────────────────────
        self.current_loc: Position3D = init_loc
        self.status: DroneStatus = DroneStatus.IDLE
        self.battery_current: float = battery_max       # 初始满电
        self.current_payload: float = 0.0               # [kg]
        self.carrying_order_id: Optional[str] = None

        # ── 路由（只读序列 + 指针）────────────────────────────────────────────
        self.route_plan: list[RouteWaypoint] = []
        self.current_waypoint_index: int = 0

        # ── 统计 ──────────────────────────────────────────────────────────────
        self.cumulative_distance: float = 0.0     # 累计飞行距离 [m]
        self.cumulative_energy_j: float = 0.0     # 累计飞行耗能 [J]

    # ══════════════════════════════════════════════════════════════════════════
    # 属性访问
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def battery_ratio(self) -> float:
        """当前电量占满电比例 [0, 1]。"""
        return self.battery_current / self.battery_max

    @property
    def is_loaded(self) -> bool:
        """是否携带货物。"""
        return self.carrying_order_id is not None

    @property
    def total_mass(self) -> float:
        """当前总质量 = 机体自重 + 当前载重 [kg]。"""
        return self.empty_weight + self.current_payload

    @property
    def current_waypoint(self) -> Optional[RouteWaypoint]:
        """当前目标航路点；路由已完成时返回 None。"""
        if 0 <= self.current_waypoint_index < len(self.route_plan):
            return self.route_plan[self.current_waypoint_index]
        return None

    @property
    def has_pending_route(self) -> bool:
        """是否还有未执行的航路点。"""
        return self.current_waypoint_index < len(self.route_plan)

    # ══════════════════════════════════════════════════════════════════════════
    # 气动功率模型
    # ══════════════════════════════════════════════════════════════════════════

    def calculate_power(self, payload: float, v: float) -> float:
        """
        多旋翼功耗方程（诱导功率 + 废阻功率）。

        公式：P = k1 × (m_empty + payload)^1.5 + k2 × v^3

        适用场景：巡航飞行功耗估算。悬停时取 v=0 退化为纯诱导功率。
        本模型假设匀速水平飞行；爬升/下降阶段可对 k1 项乘以修正系数（外部传入）。

        Args:
            payload: 当前挂载重量 [kg]
            v:       飞行速度 [m/s]

        Returns:
            瞬时功耗 [W]
        """
        mass = self.empty_weight + max(0.0, payload)
        induced = self.k1 * (mass ** 1.5)
        parasitic = self.k2 * (v ** 3)
        return induced + parasitic

    def consume_energy(self, dt: float) -> None:
        """
        按当前飞行状态消耗电量（仿真心跳调用）。

        仅在 status.is_flying 时消耗巡航功率；地面状态（IDLE/LOADING/等）
        不消耗（或可扩展为极低待机功率，当前设计省略）。

        这里的地面状态包含补能服务阶段的 `QUEUING` / `CHARGING`。
        mode C 的等待回收阶段若后续被建模为独立状态，其能耗语义应与
        “已到达充换电宿主后的排队/服务”分开处理。

        若电量耗尽（battery_current ≤ 0），状态自动转为 DEAD 并记录告警。

        Args:
            dt: 时间步长 [s]
        """
        if not self.status.is_flying:
            return

        power = self.calculate_power(self.current_payload, self.cruise_speed)
        energy_j = max(0.0, power * dt)
        self.battery_current -= energy_j
        self.cumulative_distance += self.cruise_speed * dt
        self.cumulative_energy_j += energy_j

        if self.battery_current <= 0:
            self.battery_current = 0.0
            self.status = DroneStatus.DEAD
            logger.critical(
                "[Drone %s] 电量耗尽！位置: %s，累计飞行距离: %.0f m",
                self.drone_id, self.current_loc, self.cumulative_distance,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 路由推进
    # ══════════════════════════════════════════════════════════════════════════

    def move_step(self, dt: float) -> Optional[WaypointAction]:
        """
        沿 route_plan[current_waypoint_index] 方向移动一个时间步。

        到达当前目标航路点后：
          1. 触发该航路点的 action，作为返回值交由仿真引擎处理
          2. current_waypoint_index += 1（**不弹出列表**）
          3. 若已到达最后一个航路点，返回最后一个 action 后停止

        注意：本方法只负责位置推演与指针推进，action 的实际业务逻辑（如
        assign_order、arrive 等）由仿真引擎根据返回值调用对应实体方法执行。

        Args:
            dt: 时间步长 [s]

        Returns:
            若本时间步到达了一个航路点，返回该点的 WaypointAction；
            否则返回 None。
        """
        if not self.has_pending_route or self.status == DroneStatus.DEAD:
            return None

        waypoint = self.route_plan[self.current_waypoint_index]
        target = waypoint.loc
        dist_remaining = self.current_loc.distance_3d(target)

        step_dist = self.cruise_speed * dt

        if step_dist >= dist_remaining:
            # ── 本步到达航路点 ──────────────────────────────────────────────
            self.current_loc = target
            self.current_waypoint_index += 1
            logger.debug(
                "[Drone %s] 到达航路点[%d] action=%s target_entity=%s",
                self.drone_id,
                self.current_waypoint_index - 1,
                waypoint.action.value,
                waypoint.target_entity_id,
            )
            return waypoint.action
        else:
            # ── 朝目标方向前进 ──────────────────────────────────────────────
            ratio = step_dist / dist_remaining
            self.current_loc = self.current_loc.interpolate(target, ratio)
            return None

    def set_route(self, waypoints: list[RouteWaypoint]) -> None:
        """
        设置新的路由计划，重置指针。

        调度器在派单时调用此方法，覆盖旧路由（若有）。
        调用前应确保无人机处于 IDLE 状态，否则记录警告。

        Args:
            waypoints: 完整的航路点序列
        """
        if self.status not in {DroneStatus.IDLE, DroneStatus.QUEUING, DroneStatus.CHARGING}:
            logger.warning(
                "[Drone %s] set_route() 在非空闲状态(%s)下被调用，路由已覆盖。",
                self.drone_id, self.status.value,
            )
        self.route_plan = list(waypoints)
        self.current_waypoint_index = 0

    def append_route(self, waypoints: list[RouteWaypoint]) -> None:
        """将新航路点追加到当前路径末尾（路径拼接，不覆盖已执行部分）。

        用于串联任务场景：无人机在完成当前配送后继续执行新任务，
        而非返回仓库后重新起飞。最后一个航路点（原 DOCK）在拼接时被
        新路径替代。
        """
        if not self.route_plan or self.current_waypoint_index >= len(self.route_plan):
            self.route_plan = list(waypoints)
            self.current_waypoint_index = 0
            return

        keep = self.route_plan[:len(self.route_plan)]
        if keep and keep[-1].action == WaypointAction.DOCK_DEPOT:
            keep = keep[:-1]
        keep.extend(waypoints)
        self.route_plan = keep
        logger.info(
            "[Drone %s] 路径拼接：保留 %d 已有航点 + %d 新航点",
            self.drone_id,
            len(self.route_plan) - len(waypoints),
            len(waypoints),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 订单绑定
    # ══════════════════════════════════════════════════════════════════════════

    def assign_order(self, order_id: str, payload_weight: float) -> None:
        """
        绑定订单至本无人机，更新载重状态。

        Args:
            order_id:       要绑定的订单 ID
            payload_weight: 货物重量 [kg]

        Raises:
            ValueError: 无人机已有在途订单，或载重超限
        """
        if self.carrying_order_id is not None:
            raise ValueError(
                f"[Drone {self.drone_id}] 已携带订单 {self.carrying_order_id}，"
                f"无法再绑定 {order_id}"
            )
        if payload_weight > self.payload_capacity:
            raise ValueError(
                f"[Drone {self.drone_id}] 货物重量 {payload_weight}kg 超过载重上限 "
                f"{self.payload_capacity}kg"
            )
        self.carrying_order_id = order_id
        self.current_payload = payload_weight
        logger.debug("[Drone %s] 绑定订单 %s，载重 %.2fkg", self.drone_id, order_id, payload_weight)

    def release_order(self) -> Optional[str]:
        """
        解除订单绑定，清空载重。

        Returns:
            被解除的订单 ID；若原本无挂载则返回 None 并记录警告。
        """
        if self.carrying_order_id is None:
            logger.warning("[Drone %s] release_order() 调用时无在途订单。", self.drone_id)
            return None
        released_id = self.carrying_order_id
        self.carrying_order_id = None
        self.current_payload = 0.0
        logger.debug("[Drone %s] 释放订单 %s。", self.drone_id, released_id)
        return released_id

    def recharge_to_full(self) -> None:
        """将电量立即补满到 battery_max。"""
        self.battery_current = self.battery_max

    # ══════════════════════════════════════════════════════════════════════════
    # 电量与可达性
    # ══════════════════════════════════════════════════════════════════════════

    def can_reach(
        self,
        target: Position3D,
        payload: float,
        safe_margin: float,
    ) -> bool:
        """
        判断当前电量能否飞到目标点并保留指定安全余量。

        调度器派单前**必须调用**此方法，避免产生空中电量耗尽事故。

        能耗估算：E = P × t = (k1·m^1.5 + k2·v^3) × (d / v)
        其中 d 为三维欧氏距离（近似水平飞行），v 为巡航速度。

        Args:
            target:      目标点 UTM 坐标
            payload:     飞行途中的挂载重量 [kg]
            safe_margin: 到达后须保留的最小电量 [J]

        Returns:
            True 表示电量充足；False 表示不足以到达
        """
        distance = self.current_loc.distance_3d(target)
        if distance < 1e-3:
            # 已在目标点附近，认为可达
            return self.battery_current >= safe_margin

        flight_time = distance / self.cruise_speed
        power = self.calculate_power(payload, self.cruise_speed)
        energy_needed = power * flight_time

        return self.battery_current >= energy_needed + safe_margin

    def get_remaining_range(self) -> float:
        """
        快速预估当前电量支持的最大飞行距离（空载，巡航速度）。

        公式：range = battery_current / P_empty × cruise_speed
        其中 P_empty = k1 × m_empty^1.5 + k2 × v^3

        Returns:
            最大续航距离 [m]
        """
        power = self.calculate_power(0.0, self.cruise_speed)
        if power <= 0:
            return 0.0
        flight_time = self.battery_current / power
        return flight_time * self.cruise_speed

    # ══════════════════════════════════════════════════════════════════════════
    # 序列化
    # ══════════════════════════════════════════════════════════════════════════

    def to_telemetry_dict(self) -> dict:
        """
        序列化当前状态为 WebSocket 推送字典（坐标转 WGS84）。

        Returns:
            JSON 可序列化的无人机状态字典
        """
        lon, lat = self.current_loc.to_wgs84()
        waypoint = self.current_waypoint
        return {
            "entity_type":          "drone",
            "drone_id":             self.drone_id,
            "drone_type":           self.__class__.__name__,
            "status":               self.status.value,
            "home_id":              self.home_id,
            "home_type":            self.home_type.value,
            "lng":                  lon,
            "lat":                  lat,
            "altitude":             self.current_loc.z,
            "battery_ratio":        round(self.battery_ratio, 4),
            "battery_current_j":    round(self.battery_current, 2),
            "battery_max_j":        self.battery_max,
            "payload_weight":       self.current_payload,
            "carrying_order_id":    self.carrying_order_id,
            "current_waypoint_idx": self.current_waypoint_index,
            "current_action":       waypoint.action.value if waypoint else None,
            "cumulative_distance_m": round(self.cumulative_distance, 2),
            "cumulative_energy_wh": round(self.cumulative_energy_j / 3600.0, 3),
            "remaining_range_m":    round(self.get_remaining_range(), 2),
        }

    def to_dynamic_state(self) -> dict:
        """
        轻量序列化，仅含 TICK 帧所需字段。
        包含所有 DroneConfig 非 Optional 字段，防止 setRuntimeAll 全量替换后字段丢失。
        drone_type / home_id / home_type 已内嵌，无需 _metadata 合并。
        """
        lon, lat = self.current_loc.to_wgs84()
        return {
            "drone_id":              self.drone_id,
            # ── 静态 DroneConfig 必填字段（防止 TICK 替换后丢失渲染信息）────────
            "drone_type":            self.__class__.__name__,
            "home_id":               self.home_id,
            "home_type":             self.home_type.value,
            # ── 动态字段 ─────────────────────────────────────────────────────
            "lng":                   lon,
            "lat":                   lat,
            "altitude":              self.current_loc.z,
            "status":                self.status.value,
            "battery_ratio":         round(self.battery_ratio, 4),
            "carrying_order_id":     self.carrying_order_id,
            "cumulative_distance_m": round(self.cumulative_distance, 2),
            "cumulative_energy_wh":  round(self.cumulative_energy_j / 3600.0, 3),
            "remaining_range_m":     round(self.get_remaining_range(), 2),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.drone_id!r}, "
            f"status={self.status.value}, "
            f"battery={self.battery_ratio:.1%}, "
            f"payload={self.current_payload}kg)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 异构子类
# ══════════════════════════════════════════════════════════════════════════════

class LightDrone(Drone):
    """
    轻型无人机（对标美团 FP400 V4）。

    当前运行参数从 config/drone_params.yaml 的 light_drone 读取：
      - 巡航速度: 18 m/s
      - 最大载重: 2.5 kg
      - 机体自重: 7 kg
      - 电池容量: 约 414 Wh（1,490,400 J）
      - 安全余量: 电池容量的 10%
      - k1: 22.0 W/kg^1.5
      - k2: 0.06 W/(m/s)^3

    气动参数由 config/drone_params.yaml 载入，子类实例化时读取并固化。
    """

    def __init__(
        self,
        *,
        drone_id: str,
        home_id: str,
        home_type: SourceType,
        init_loc: Position3D,
    ) -> None:
        params = load_drone_params().light
        super().__init__(
            drone_id=drone_id,
            home_id=home_id,
            home_type=home_type,
            init_loc=init_loc,
            k1=params.k1,
            k2=params.k2,
            cruise_speed=params.cruise_speed,
            payload_capacity=params.payload_capacity,
            empty_weight=params.empty_weight,
            battery_max=params.battery_capacity_j,
        )
        # 将安全余量配置固化供外部查询
        self._safe_margin_j: float = params.safe_margin_j

    @property
    def safe_margin_j(self) -> float:
        """建议安全余量（焦耳），调度器调用 can_reach() 时传入。"""
        return self._safe_margin_j


class HeavyDrone(Drone):
    """
    重型无人机（对标丰翼方舟 40）。

    当前运行参数从 config/drone_params.yaml 的 heavy_drone 读取：
      - 巡航速度: 14 m/s
      - 最大载重: 10 kg
      - 机体自重: 36 kg
      - 电池容量: 约 3953 Wh（14,230,800 J）
      - 安全余量: 电池容量的 10%
      - k1: 22.0 W/kg^1.5
      - k2: 0.06 W/(m/s)^3

    气动参数由 config/drone_params.yaml 载入，子类实例化时读取并固化。
    """

    def __init__(
        self,
        *,
        drone_id: str,
        home_id: str,
        home_type: SourceType,
        init_loc: Position3D,
    ) -> None:
        params = load_drone_params().heavy
        super().__init__(
            drone_id=drone_id,
            home_id=home_id,
            home_type=home_type,
            init_loc=init_loc,
            k1=params.k1,
            k2=params.k2,
            cruise_speed=params.cruise_speed,
            payload_capacity=params.payload_capacity,
            empty_weight=params.empty_weight,
            battery_max=params.battery_capacity_j,
        )
        self._safe_margin_j: float = params.safe_margin_j

    @property
    def safe_margin_j(self) -> float:
        """建议安全余量（焦耳），调度器调用 can_reach() 时传入。"""
        return self._safe_margin_j
