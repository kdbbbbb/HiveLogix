"""
HiveLogix core.entities — 实体层公共 API

统一导出所有实体类，使上层模块（调度器、仿真引擎、API 层）
仅需从此一处导入，无需感知内部文件结构：

    from core.entities import (
        Position3D, SourceType, WaypointAction, RouteWaypoint,
        TaskStatus, DroneStatus, TruckStatus,
        Order,
        Drone, LightDrone, HeavyDrone,
        ChargingHost,
        SwapStation,
        Truck,
        Depot,
    )
"""

from core.entities.primitives import (
    DroneStatus,
    Position3D,
    RouteWaypoint,
    SourceType,
    TaskStatus,
    TruckStatus,
    WaypointAction,
)
from core.entities.order import Order
from core.entities.drone import Drone, HeavyDrone, LightDrone
from core.entities.charging_host import ChargingHost
from core.entities.swap_station import SwapStation
from core.entities.truck import Truck
from core.entities.depot import Depot

__all__ = [
    # 基础原语
    "Position3D",
    "SourceType",
    "WaypointAction",
    "RouteWaypoint",
    "TaskStatus",
    "DroneStatus",
    "TruckStatus",
    # 实体类
    "Order",
    "Drone",
    "LightDrone",
    "HeavyDrone",
    "ChargingHost",
    "SwapStation",
    "Truck",
    "Depot",
]
