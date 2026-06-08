#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 移动基站：卡车实体 (Section 4)

Truck 是整个调度系统中最复杂的实体，同时承担：
  1. 地面运输载具：按路网节点序列行驶，Mode A 直递到客户
  2. 无人机母港：装载、释放、回收无人机，Vehicle-Drone 协同核心
  3. 充换电宿主：重写 arrive() 加入起降平台物理占位逻辑

路网位置模型：
  卡车持有 route_nodes（SUMO 节点 ID 列表）和 _route_positions（对应 UTM 坐标列表）。
  通过 set_route() 一次性绑定，之后 move_step() 推进位置，
  get_location(t) 通过 _departure_time + speed 估算任意时刻的位移。

设计决策（与原始 design doc 的补充）：
  - parking_slots 语义等同于原 max_drone_slots（起降平台并发数）
  - docked_drones 表示已停靠在车顶的无人机，arrive() 需先检查此列表
  - get_future_trajectory() 供调度器预测汇合点（Mode B / D / E）
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from core.entities.charging_host import ChargingHost
from core.entities.order import Order
from core.entities.primitives import Position3D, TaskStatus, TruckStatus

logger = logging.getLogger(__name__)


# ── 路网节点（内部辅助数据类，不对外暴露）────────────────────────────────────────
class _RouteNode:
    """卡车路网节点：SUMO 节点 ID + UTM 坐标 + 累计路径长度（用于高效位置插值）。"""
    __slots__ = ("node_id", "position", "cumulative_dist")

    def __init__(self, node_id: str, position: Position3D, cumulative_dist: float) -> None:
        self.node_id = node_id
        self.position = position
        self.cumulative_dist = cumulative_dist  # 从起点到本节点的路径长度 [m]


class Truck(ChargingHost):
    """
    移动基站：卡车。

    Args:
        truck_id:      全局唯一 ID，格式建议：'T-{序号}'
        speed:         地面行驶速度 [m/s]
        max_inventory: 车厢最大装载包裹数量上限
        swap_time:     车载换电耗时（秒）
        parking_slots: 起降平台并发数（即最多同时停靠的无人机数）
        init_loc:      初始 UTM 三维坐标
    """

    def __init__(
        self,
        *,
        truck_id: str,
        speed: float,
        max_inventory: int,
        swap_time: float,
        parking_slots: int,
        init_loc: Position3D,
    ) -> None:
        super().__init__(
            swap_time=swap_time,
            parking_slots=parking_slots,
            host_id=truck_id,
        )
        if speed <= 0:
            raise ValueError(f"[{truck_id}] speed 必须为正数: {speed}")
        if max_inventory < 1:
            raise ValueError(f"[{truck_id}] max_inventory 至少为 1: {max_inventory}")

        self.truck_id: str = truck_id
        self.speed: float = speed
        self.max_inventory: int = max_inventory

        # ── 动态状态 ──────────────────────────────────────────────────────────
        self.status: TruckStatus = TruckStatus.IDLE
        self.current_loc: Position3D = init_loc

        # ── 订单库存 ──────────────────────────────────────────────────────────
        self.inventory: dict[str, Order] = {}        # {order_id: Order}
        self.docked_drones: list[str] = []           # 停靠在车顶的无人机 ID

        # ── 路网路由（通过 set_route() 绑定）────────────────────────────────
        self.route_nodes: list[str] = []             # SUMO 节点 ID 列表
        self._route_data: list[_RouteNode] = []      # 内部带累计距离的路网数据
        self._departure_time: float = 0.0            # 开始沿当前路由行驶的仿真时间
        self._current_node_idx: int = 0              # 当前正在前往的节点下标
        self.cumulative_distance_m: float = 0.0      # 累计地面行驶距离 [m]

    # ══════════════════════════════════════════════════════════════════════════
    # 路网路由设置
    # ══════════════════════════════════════════════════════════════════════════

    def set_route(
        self,
        route_nodes: list[str],
        route_positions: list[Position3D],
        departure_time: float,
        geometry: list[Position3D] | None = None,
    ) -> None:
        """
        绑定新的行驶路由。

        Args:
            route_nodes:     SUMO 节点 ID 列表（关键节点）
            route_positions: 每个关键节点对应的 UTM 三维坐标列表
            departure_time:  开始行驶的仿真时间（秒）
            geometry:        完整的路径几何（包含所有中间OSM节点），若提供则用其替代 route_positions

        Raises:
            ValueError: 节点 ID 列表与坐标列表长度不一致
        """
        if len(route_nodes) != len(route_positions):
            raise ValueError(
                f"[{self.truck_id}] route_nodes({len(route_nodes)}) 与 "
                f"route_positions({len(route_positions)}) 长度不一致"
            )
        if len(route_nodes) < 2:
            raise ValueError(f"[{self.truck_id}] 路由至少需要 2 个节点")

        self.route_nodes = list(route_nodes)
        self._departure_time = departure_time
        self._current_node_idx = 1  # 从第一段开始前进

        # ── 预计算各节点的累计路径长度（便于后续 O(log n) 位置插值）──────────
        # 优先使用完整几何路径；若几何点不足（<2）则回退关键节点，避免“零长度路线”
        positions_to_use = geometry if geometry and len(geometry) >= 2 else route_positions
        
        self._route_data = []
        cumulative = 0.0
        prev_pos: Optional[Position3D] = None
        for pos in positions_to_use:
            if prev_pos is not None:
                cumulative += prev_pos.distance_2d(pos)  # 地面路由取2D距离
            self._route_data.append(_RouteNode("", pos, cumulative))
            prev_pos = pos

        # ── 立即更新为路由起始点 ─────────────────────────────────────────────
        if self._route_data:
            self.current_loc = self._route_data[0].position

        self.status = TruckStatus.DRIVING
        logger.debug(
            "[Truck %s] 设置路由：%d 个关键节点，几何路径 %d 点，总路径长 %.0f m，起始位置 (%.2f, %.2f)",
            self.truck_id, len(route_nodes), len(positions_to_use), self._route_data[-1].cumulative_dist,
            self.current_loc.x, self.current_loc.y,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 位置推演（实现 ChargingHost 抽象方法）
    # ══════════════════════════════════════════════════════════════════════════

    def get_location(self, current_time: float) -> Position3D:
        """
        按当前路由和行驶速度，估算给定仿真时刻的 UTM 坐标（线性插值）。

        仅在 DRIVING 状态下推演位置；其他状态返回 current_loc（当前停车位置）。

        Args:
            current_time: 查询时刻的仿真时间（秒）

        Returns:
            UTM 三维坐标
        """
        if not self.status.is_mobile or not self._route_data:
            return self.current_loc

        elapsed = max(0.0, current_time - self._departure_time)
        dist_traveled = elapsed * self.speed

        total_dist = self._route_data[-1].cumulative_dist
        if dist_traveled >= total_dist:
            return self._route_data[-1].position

        return self._interpolate_position(dist_traveled)

    def _interpolate_position(self, dist_traveled: float) -> Position3D:
        """
        在路网线段上按距离进行线性插值。

        使用二分查找定位所在线段，时间复杂度 O(log n)。

        Args:
            dist_traveled: 从路由起点行驶的距离 [m]

        Returns:
            插值后的 UTM 三维坐标
        """
        # 二分查找 dist_traveled 所在的路段 [lo, hi)
        lo, hi = 0, len(self._route_data) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._route_data[mid].cumulative_dist <= dist_traveled:
                lo = mid
            else:
                hi = mid

        seg_start = self._route_data[lo]
        seg_end = self._route_data[hi]
        seg_len = seg_end.cumulative_dist - seg_start.cumulative_dist

        if seg_len < 1e-6:
            return seg_end.position

        t = (dist_traveled - seg_start.cumulative_dist) / seg_len
        return seg_start.position.interpolate(seg_end.position, t)

    # ══════════════════════════════════════════════════════════════════════════
    # 仿真推进
    # ══════════════════════════════════════════════════════════════════════════

    def move_step(self, dt: float, current_time: float) -> bool:
        """
        沿地面路网推演位置一个时间步。

        Args:
            dt:           时间步长 [s]
            current_time: 当前仿真时间（秒），用于更新 current_loc

        Returns:
            True 表示本步到达路由终点（已行驶完毕）
        """
        if not self.status.is_mobile or not self._route_data:
            return False

        prev_loc = self.current_loc
        self.current_loc = self.get_location(current_time)
        self.cumulative_distance_m += max(0.0, prev_loc.distance_2d(self.current_loc))

        # 检查是否到达终点
        elapsed = max(0.0, current_time - self._departure_time)
        dist_traveled = elapsed * self.speed
        if dist_traveled >= self._route_data[-1].cumulative_dist:
            self.current_loc = self._route_data[-1].position
            self.status = TruckStatus.IDLE
            logger.debug("[Truck %s] 到达路由终点，切换至 IDLE。", self.truck_id)
            return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # 充换电宿主（重写基类 arrive()）
    # ══════════════════════════════════════════════════════════════════════════

    def arrive(self, drone_id: str, current_time: float) -> None:
        """
        无人机降落至卡车起降平台，进入车载充换电流程。

        重写基类 arrive()，增加起降平台物理占位检查：
          1. 若 docked_drones 数量 < parking_slots → 停靠并进入换电队列
          2. 否则 → 加入 wait_queue 等待车载补能服务空位

        注意：起降平台占位（docked_drones）与换电槽位（serving_drones）均受
        parking_slots 约束，两者不做独立计数，平台占位即服务开始。
        这里的等待队列表示“已经被卡车成功回收后的车载服务等待”，
        不表示 mode C 中等待与卡车汇合的阶段。

        Args:
            drone_id:    降落的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        with self._lock:
            if drone_id in self.docked_drones:
                logger.warning(
                    "[Truck %s] arrive(): drone %s 已在 docked_drones 中，忽略。",
                    self.truck_id, drone_id,
                )
                return

            if len(self.docked_drones) >= self.parking_slots:
                # 平台已满，进入等待队列
                if drone_id not in self.wait_queue:
                    self.wait_queue.append(drone_id)
                    logger.debug(
                        "[Truck %s] 平台已满(%d/%d)，drone %s 加入等待队列。",
                        self.truck_id, len(self.docked_drones),
                        self.parking_slots, drone_id,
                    )
                return

            # 有空位：停靠 + 开始换电
            self.docked_drones.append(drone_id)
            # 调用基类换电逻辑（直接将 serving_drones 写入，跳过重复 arrive 检查）
            finish_time = current_time + self.swap_time
            self.serving_drones[drone_id] = finish_time
            logger.debug(
                "[Truck %s] drone %s 已停靠并开始换电，预计完成: %.2f",
                self.truck_id, drone_id, finish_time,
            )

    def depart(self, drone_id: str, current_time: float) -> None:
        """
        换电完成，无人机从平台离机（等待起飞指令）。

        移除 docked_drones 和 serving_drones 中的记录；
        若 wait_queue 非空，从队列拉取下一架无人机停靠并进入车载补能服务。

        Args:
            drone_id:    完成换电的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        with self._lock:
            if drone_id in self.docked_drones:
                self.docked_drones.remove(drone_id)

            # 调用基类 depart()（会触发 _try_serve_next）
            # 但基类的 _try_serve_next 只检查 serving_drones，我们需要同时更新 docked_drones
            if drone_id in self.serving_drones:
                del self.serving_drones[drone_id]
                logger.debug("[Truck %s] drone %s 换电完成并离机。", self.truck_id, drone_id)
                self._truck_serve_next(current_time)

    def _truck_serve_next(self, current_time: float) -> None:
        """
        从等待队列拉取下一架无人机停靠。

        调用者须持有 self._lock。与基类 _try_serve_next 不同,
        卡车版本同时更新 docked_drones。
        """
        if self.wait_queue and len(self.docked_drones) < self.parking_slots:
            next_drone = self.wait_queue.pop(0)
            self.docked_drones.append(next_drone)
            finish_time = current_time + self.swap_time
            self.serving_drones[next_drone] = finish_time
            logger.debug(
                "[Truck %s] 从队列拉取 drone %s 停靠，预计换电完成: %.2f",
                self.truck_id, next_drone, finish_time,
            )

    def tick_update(self, current_time: float) -> list[str]:
        """
        重写 tick_update，使用 Truck 版本的 depart() 以保持 docked_drones 一致性。

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            本轮完成换电的无人机 ID 列表
        """
        completed: list[str] = []
        with self._lock:
            finished = [
                did for did, ft in list(self.serving_drones.items())
                if ft <= current_time
            ]

        for drone_id in finished:
            self.depart(drone_id, current_time)
            completed.append(drone_id)

        return completed

    # ══════════════════════════════════════════════════════════════════════════
    # 无人机起飞 / 回收
    # ══════════════════════════════════════════════════════════════════════════

    def launch_drone(self, drone_id: str) -> None:
        """
        从起降平台释放无人机起飞。

        卡车状态短暂切换为 UNLOADING_DRONE，由仿真引擎在后续心跳中恢复为 DRIVING。
        本方法只执行平台侧的资源释放，无人机侧的状态更新（IDLE → FLYING）由调度器负责。

        Args:
            drone_id: 起飞的无人机 ID

        Raises:
            ValueError: 无人机不在 docked_drones 中
        """
        with self._lock:
            if drone_id not in self.docked_drones:
                raise ValueError(
                    f"[Truck {self.truck_id}] drone {drone_id} 不在停靠列表中，"
                    f"无法起飞。实际停靠: {self.docked_drones}"
                )
            self.docked_drones.remove(drone_id)
            # 若该无人机恰好还在换电队列，也清除（紧急起飞场景）
            self.serving_drones.pop(drone_id, None)

        self.status = TruckStatus.UNLOADING_DRONE
        logger.debug("[Truck %s] drone %s 已起飞，平台进入 UNLOADING_DRONE 状态。",
                     self.truck_id, drone_id)

    def recover_drone(self, drone_id: str, current_time: float) -> None:
        """
        回收无人机降落至起降平台，触发换电流程。

        卡车状态切换为 LOADING_DRONE，由仿真引擎在后续心跳中恢复。

        Args:
            drone_id:    回收的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        self.status = TruckStatus.LOADING_DRONE
        self.arrive(drone_id, current_time)
        logger.debug("[Truck %s] 开始回收 drone %s，切换至 LOADING_DRONE。",
                     self.truck_id, drone_id)

    # ══════════════════════════════════════════════════════════════════════════
    # 订单库存管理
    # ══════════════════════════════════════════════════════════════════════════

    def load_order(self, order: Order) -> None:
        """
        将订单装入车厢库存。

        Args:
            order: 要装载的订单对象

        Raises:
            ValueError: 库存已满
        """
        if len(self.inventory) >= self.max_inventory:
            raise ValueError(
                f"[Truck {self.truck_id}] 车厢已满（{self.max_inventory} 件），"
                f"无法再装载订单 {order.order_id}"
            )
        self.inventory[order.order_id] = order
        logger.debug("[Truck %s] 装载订单 %s，当前库存 %d/%d。",
                     self.truck_id, order.order_id, len(self.inventory), self.max_inventory)

    def deliver_order(self, order_id: str, current_time: float) -> Order:
        """
        Mode A 直递：卡车直接将订单送达客户。

        从 inventory 移除订单，更新订单状态为 COMPLETED，记录 actual_deliver_time。

        Args:
            order_id:    要送达的订单 ID
            current_time: 当前仿真时间（秒）

        Returns:
            已完成的 Order 对象

        Raises:
            KeyError: 订单不在车载库存中
        """
        if order_id not in self.inventory:
            raise KeyError(
                f"[Truck {self.truck_id}] 订单 {order_id} 不在车载库存中。"
                f"当前库存: {list(self.inventory.keys())}"
            )
        order = self.inventory.pop(order_id)
        order.actual_deliver_time = current_time
        order.update_status(TaskStatus.COMPLETED)
        logger.info("[Truck %s] Mode A 送达订单 %s，耗时 %.0f s。",
                    self.truck_id, order_id, current_time - order.create_time)
        return order

    # ══════════════════════════════════════════════════════════════════════════
    # 轨迹预测（供调度器规划 RENDEZVOUS 汇合点）
    # ══════════════════════════════════════════════════════════════════════════

    def get_future_trajectory(
        self,
        current_time: float,
        time_window: float,
        interval: float = 10.0,
    ) -> list[Position3D]:
        """
        预测未来一段时间内的卡车位置序列，供调度器规划无人机汇合点（RENDEZVOUS）。

        按 interval 步长对 [current_time, current_time + time_window] 区间采样，
        使用 get_location() 插值计算每个时刻的预测坐标。

        Args:
            current_time: 查询起始仿真时间（秒）
            time_window:  预测时间窗口长度（秒）
            interval:     采样间隔（秒），默认 10 秒

        Returns:
            位置点序列（含起始点），若路由未设置则返回 [current_loc]
        """
        if not self._route_data:
            return [self.current_loc]

        trajectory: list[Position3D] = []
        t = current_time
        end_time = current_time + time_window
        while t <= end_time + 1e-9:
            trajectory.append(self.get_location(t))
            t += interval

        return trajectory

    # ══════════════════════════════════════════════════════════════════════════
    # 序列化（实现 ChargingHost 抽象方法）
    # ══════════════════════════════════════════════════════════════════════════

    def to_telemetry_dict(self) -> dict:
        """
        序列化卡车状态为 WebSocket 推送字典（坐标转 WGS84）。

        Returns:
            JSON 可序列化的卡车状态字典
        """
        try:
            lon, lat = self.current_loc.to_wgs84()
        except Exception as e:
            logger.error(f"[Truck.to_telemetry_dict] 坐标转换失败: {e}, current_loc={self.current_loc}")
            lon, lat = 0.0, 0.0
        
        with self._lock:
            docked = list(self.docked_drones)
            waiting = list(self.wait_queue)

        return {
            "entity_type":      "truck",
            "truck_id":         self.truck_id,
            "status":           self.status.value,
            "lng":              lon,
            "lat":              lat,
            "altitude":         self.current_loc.z,
            "speed":            self.speed,
            "inventory_count":  len(self.inventory),
            "max_inventory":    self.max_inventory,
            "inventory_ids":    list(self.inventory.keys()),
            "docked_drones":    docked,
            "waiting_drones":   waiting,
            "parking_slots":    self.parking_slots,
            "available_slots":  self.available_slots,
            "swap_time":        self.swap_time,
            "cumulative_distance_m": round(self.cumulative_distance_m, 2),
        }

    def to_dynamic_state(self) -> dict:
        """
        轻量序列化，仅含 TICK 帧所需字段。
        包含所有 TruckConfig 非 Optional 字段，防止 setRuntimeAll 全量替换后字段丢失。
        name / home_depot_id 由 EntityManager.get_telemetry() 从 _metadata 合并。
        """
        try:
            lon, lat = self.current_loc.to_wgs84()
        except Exception as e:
            logger.error(f"[Truck.to_dynamic_state] 坐标转换失败: {e}, current_loc={self.current_loc}")
            # 返回临时坐标避免序列化失败
            lon, lat = 0.0, 0.0
        
        return {
            "truck_id":        self.truck_id,
            # ── 静态 TruckConfig 必填字段 ─────────────────────────────────────
            "speed":           self.speed,
            "max_inventory":   self.max_inventory,
            "swap_time":       self.swap_time,
            "parking_slots":   self.parking_slots,
            # ── 动态字段 ─────────────────────────────────────────────────────
            "lng":             lon,
            "lat":             lat,
            "status":          self.status.value,
            "inventory_count": len(self.inventory),
            "available_slots": self.available_slots,
            "docked_drones":   list(self.docked_drones),
            "cumulative_distance_m": round(self.cumulative_distance_m, 2),
        }

    def __repr__(self) -> str:
        return (
            f"Truck(id={self.truck_id!r}, "
            f"status={self.status.value}, "
            f"inventory={len(self.inventory)}/{self.max_inventory}, "
            f"docked={len(self.docked_drones)})"
        )
