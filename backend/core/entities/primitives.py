#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 基础共享原语 (Section 0.1)

本模块为整个仿真引擎的底层语义基石，包含：
  - Position3D       : UTM 三维坐标（内部统一坐标系）
  - SourceType       : 取货源/归属宿主类型枚举
  - WaypointAction   : 航路点动作语义枚举
  - RouteWaypoint    : 带动作的航路点（只读，不弹出的完整路由序列元素）
  - TaskStatus       : 订单生命周期状态机
  - DroneStatus      : 无人机物理状态机
  - TruckStatus      : 卡车运行状态机

设计原则：
  - 本模块 **零外部依赖**（仅用 stdlib），确保可在任何上下文中独立导入
  - 所有 Enum 使用字符串 value，便于 JSON 序列化和日志可读性
  - Position3D 提供常用几何运算方法，避免调用侧重复计算
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 空间原语
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position3D:
    """
    UTM Zone 51N 三维坐标（内部坐标系，单位：米）。

    Attributes:
        x: UTM 东向坐标 [m]，等价于 SUMO 坐标系东向偏移 + net_offset.ox
        y: UTM 北向坐标 [m]，等价于 SUMO 坐标系北向偏移 + net_offset.oy
        z: 高度，相对大地水准面 [m]，用于禁飞区判断与三维路径规划

    坐标约定：
        WGS84 (lon, lat) ↔ UTM (x, y) 转换由 utils.coord_utils 负责。
        对外（WebSocket 帧 / REST）统一转换为 WGS84 再输出。
    """
    x: float  # UTM 东向 [m]
    y: float  # UTM 北向 [m]
    z: float  # 高度    [m]

    # ── 几何运算 ──────────────────────────────────────────────────────────────

    def distance_2d(self, other: "Position3D") -> float:
        """XY 平面欧氏距离（忽略高度差），单位：米。"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_3d(self, other: "Position3D") -> float:
        """三维欧氏距离，单位：米。"""
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def interpolate(self, other: "Position3D", t: float) -> "Position3D":
        """
        在 self 和 other 之间线性插值。

        Args:
            other: 目标点
            t:     插值因子，0.0 = self，1.0 = other，值域 [0, 1] 内裁剪

        Returns:
            插值后的新 Position3D
        """
        t = max(0.0, min(1.0, t))
        return Position3D(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t,
            z=self.z + (other.z - self.z) * t,
        )

    def to_wgs84(self) -> tuple[float, float]:
        """
        转换为 WGS84 经纬度 (lon, lat)，供对外序列化使用。

        注意：该方法在调用时引入 coord_transform 依赖，确保 pyproj 已安装。
        """
        from utils.coord_utils import utm_to_wgs84  # 延迟导入，隔离依赖
        return utm_to_wgs84(self.x, self.y)

    def __repr__(self) -> str:
        return f"Position3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# 类型枚举
# ══════════════════════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    """
    取货源 / 归属宿主的类型。

    继承 str 使 Enum value 可直接用于 JSON 序列化，无需额外 .value 调用。
    避免在调度逻辑中使用裸字符串比较，防止大小写不一致引发的逻辑错误。
    """
    DEPOT = "DEPOT"   # 固定仓库节点
    TRUCK = "TRUCK"   # 移动卡车节点


class WaypointAction(str, Enum):
    """
    无人机到达某航路点后应执行的动作语义。

    赋予 route_plan 语义信息，使仿真引擎可在到达时派发对应的事件而非依赖
    外部状态机判断。每个 WaypointAction 对应一个具体的实体方法调用：

      PICKUP       → drone.assign_order() + 从宿主弹出货物
      DELIVER      → drone.release_order() + 订单状态 → COMPLETED
      SWAP_BATTERY → charging_host.arrive()，触发换电队列
      RENDEZVOUS   → 飞往动态汇合点（坐标由调度器实时插值给出）
      DOCK_TRUCK   → 降落至卡车平台，触发 truck.recover_drone()
      DOCK_DEPOT   → 降落至仓库，触发 depot.receive_drone()
    """
    PICKUP       = "PICKUP"
    DELIVER      = "DELIVER"
    SWAP_BATTERY = "SWAP_BATTERY"
    RENDEZVOUS   = "RENDEZVOUS"
    DOCK_TRUCK   = "DOCK_TRUCK"
    DOCK_DEPOT   = "DOCK_DEPOT"


# ── 航路点（不可变，route_plan 以该对象构成的列表为整体只读结构）─────────────────

@dataclass(frozen=True)
class RouteWaypoint:
    """
    带动作语义的三维航路点。

    frozen=True 保证航路点一旦规划完成不可被意外修改，支持哈希与集合操作。
    route_plan 使用指针（current_waypoint_index）遍历，列表本身不弹出，
    以支持路径回放和调试诊断。

    Attributes:
        loc:              航路点三维坐标（UTM）
        action:           到达后执行的动作
        target_entity_id: 关联实体 ID（可选），供动作执行时快速查找对象。
                          例如：换电站 ID、卡车 ID、订单 ID 等。
    """
    loc: Position3D
    action: WaypointAction
    target_entity_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# 状态机枚举
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    """
    订单生命周期状态机。

    状态转移规则（单向，不可逆回退）：
      PENDING → ASSIGNED → PICKED_UP → DELIVERING → COMPLETED
                                                  ↘ TIMEOUT（并非终止，仍需继续履约）
      PENDING → REJECTED（当前场景 100% 履约率下不应触发）
    """
    PENDING    = "PENDING"    # 已生成，尚未分配给任何载具
    ASSIGNED   = "ASSIGNED"   # 已分配给载具，等待取货
    PICKED_UP  = "PICKED_UP"  # 无人机/卡车已完成取货，货物脱离源载具
    DELIVERING = "DELIVERING" # 货物在途，飞往或行驶至客户点
    COMPLETED  = "COMPLETED"  # 货物已成功送达客户
    TIMEOUT    = "TIMEOUT"    # 超过软时间窗，产生惩罚成本，仍需继续履约
    REJECTED   = "REJECTED"   # 订单被系统拒绝（保留状态，生产环境应告警）

    # ── 合法的前驱状态映射，用于 update_status() 的校验 ──────────────────────
    _VALID_TRANSITIONS: dict  # 类型注解占位，避免 Enum 将其视为成员

    @classmethod
    def valid_transitions(cls) -> dict["TaskStatus", set["TaskStatus"]]:
        """返回合法的状态转移图（每个状态 → 允许流转到的下一状态集合）。"""
        return {
            cls.PENDING:    {cls.ASSIGNED, cls.REJECTED, cls.TIMEOUT},
            cls.ASSIGNED:   {cls.PICKED_UP},
            cls.PICKED_UP:  {cls.DELIVERING},
            cls.DELIVERING: {cls.COMPLETED, cls.TIMEOUT},
            cls.TIMEOUT:    {cls.COMPLETED},
            cls.COMPLETED:  set(),   # 终态
            cls.REJECTED:   set(),   # 终态
        }


class DroneStatus(str, Enum):
    """
    无人机物理状态机。

    每个状态对应仿真引擎中不同的行为分支（运动学更新、能耗计算、可调度性判断）。
    IDLE 是唯一可被调度器选中的状态；DEAD 需要人工介入恢复，仿真引擎应发出告警。
    """
    IDLE               = "IDLE"               # 空闲停靠，可被调度
    FLYING_TO_PICKUP   = "FLYING_TO_PICKUP"   # 飞往取货点
    FLYING_TO_DELIVER  = "FLYING_TO_DELIVER"  # 飞往客户投递点
    FLYING_TO_STATION  = "FLYING_TO_STATION"  # 飞往充换电站补能
    FLYING_TO_TRUCK    = "FLYING_TO_TRUCK"    # 追赶移动卡车
    RETURNING_TO_DEPOT = "RETURNING_TO_DEPOT" # 返回仓库
    QUEUING            = "QUEUING"            # 已到达充换电宿主，等待补能服务空闲槽位
    CHARGING           = "CHARGING"           # 已占用补能服务槽位，正在执行换电
    LOADING            = "LOADING"            # 装载货物，短暂不可调度
    UNLOADING          = "UNLOADING"          # 卸载货物，短暂不可调度
    DEAD               = "DEAD"               # 异常终止，需人工介入

    @property
    def is_flying(self) -> bool:
        """是否处于任意飞行状态（用于能耗计算分支判断）。"""
        return self in {
            DroneStatus.FLYING_TO_PICKUP,
            DroneStatus.FLYING_TO_DELIVER,
            DroneStatus.FLYING_TO_STATION,
            DroneStatus.FLYING_TO_TRUCK,
            DroneStatus.RETURNING_TO_DEPOT,
        }

    @property
    def is_dispatchable(self) -> bool:
        """是否可被调度器选中派单。"""
        return self == DroneStatus.IDLE


class TruckStatus(str, Enum):
    """
    卡车运行状态机。

    WAITING 用于 Mode B（无人机追车汇合）场景，卡车主动停车等待无人机飞回。
    LOADING_DRONE / UNLOADING_DRONE 期间卡车短暂停止行驶，仿真引擎不推进位置。
    """
    IDLE            = "IDLE"            # 静止待命
    DRIVING         = "DRIVING"         # 沿路网行驶
    WAITING         = "WAITING"         # 停车等待无人机汇合（Mode B）
    LOADING_DRONE   = "LOADING_DRONE"   # 正在回收无人机并换电
    UNLOADING_DRONE = "UNLOADING_DRONE" # 正在释放无人机起飞

    @property
    def is_mobile(self) -> bool:
        """是否正在行驶（用于位置推演分支判断）。"""
        return self == TruckStatus.DRIVING
