#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 充换电宿主抽象基类 (Section 0.2)

充换电站（SwapStation）、仓库（Depot）、卡车（Truck）均继承本基类。
调度器通过统一接口查询所有补能节点，无需感知具体类型，实现对扩展开放、
对修改封闭（OCP）的架构。

换电服务队列模型：
  - parking_slots (K) 个并发槽位：类似 K 服务台 M/D/K 排队论模型
  - wait_queue：FIFO **充换电服务等待队列**，加入后按顺序占用释放的槽位
  - serving_drones：{drone_id: finish_time} 正在服务的无人机

语义边界：
  - 本基类只建模“无人机已经到达 ChargingHost 之后”的充换电服务排队与并发服务。
  - 它不建模 mode C 中的 rendezvous / 等待被卡车回收阶段。
  - 因此，wait_queue 不应被解释为“等待被回收队列”，只表示后续补能服务等待队列。

线程安全说明：
  仿真引擎预期以单线程时间步推进，_lock 提供防御性保护以供未来并发场景扩展。
  若切换为多线程/协程架构，建议将 _lock 替换为对应的异步原语。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

from core.entities.primitives import Position3D

logger = logging.getLogger(__name__)


class ChargingHost(ABC):
    """
    所有可对无人机执行充换电服务的宿主基类。

    子类必须在 __init__ 中调用 super().__init__(...) 完成共享属性初始化，
    并实现 get_location() 与 to_telemetry_dict() 两个抽象方法。

    Args:
        swap_time:     单次换电耗时（秒）
        parking_slots: 并发服务槽位数 K，同一时刻最多可执行补能服务的无人机数量
        host_id:       宿主实体 ID，用于日志与调试
    """

    def __init__(
        self,
        *,
        swap_time: float,
        parking_slots: int,
        host_id: str,
    ) -> None:
        if swap_time <= 0:
            raise ValueError(f"[{host_id}] swap_time 必须为正数，当前值: {swap_time}")
        if parking_slots < 1:
            raise ValueError(f"[{host_id}] parking_slots 至少为 1，当前值: {parking_slots}")

        self.swap_time: float = swap_time
        self.parking_slots: int = parking_slots
        self._host_id: str = host_id

        # ── 队列状态（受 _lock 保护）─────────────────────────────────────────
        self.wait_queue: list[str] = []
        self.serving_drones: dict[str, float] = {}  # {drone_id: finish_time}

        self._lock: threading.RLock = threading.RLock()

    # ══════════════════════════════════════════════════════════════════════════
    # 共享方法（提供默认实现，子类按需重写）
    # ══════════════════════════════════════════════════════════════════════════

    def arrive(self, drone_id: str, current_time: float) -> None:
        """
        无人机请求进入充换电服务。

        流程：
          1. 若当前占用槽位数 < parking_slots → 立即开始服务
          2. 否则 → 加入 wait_queue 末尾

        重复到达防护：若同一 drone_id 在 serving_drones 或 wait_queue 中已存在，
        记录警告并忽略本次调用（防止仿真引擎双重推进导致状态腐化）。

        语义边界：
          - 本方法表示无人机**已经到达宿主并可进入补能流程**
          - 不表示 mode C 中“等待被回收”或“等待卡车到站”

        子类（如 Truck）应先检查物理平台占位后再调用 super().arrive()。

        Args:
            drone_id:    请求服务的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        with self._lock:
            if drone_id in self.serving_drones:
                logger.warning(
                    "[%s] arrive() 被重复调用：drone %s 已在服务槽位中，忽略。",
                    self._host_id, drone_id,
                )
                return
            if drone_id in self.wait_queue:
                logger.warning(
                    "[%s] arrive() 被重复调用：drone %s 已在等待队列中，忽略。",
                    self._host_id, drone_id,
                )
                return

            if len(self.serving_drones) < self.parking_slots:
                finish_time = current_time + self.swap_time
                self.serving_drones[drone_id] = finish_time
                logger.debug(
                    "[%s] drone %s 开始换电，预计完成时间: %.2f",
                    self._host_id, drone_id, finish_time,
                )
            else:
                self.wait_queue.append(drone_id)
                logger.debug(
                    "[%s] drone %s 加入等待队列，当前队列长度: %d",
                    self._host_id, drone_id, len(self.wait_queue),
                )

    def depart(self, drone_id: str, current_time: float) -> None:
        """
        换电完成，释放该无人机占用的槽位。

        若等待队列非空，立即拉取队首无人机进入服务。

        Args:
            drone_id:    完成换电的无人机 ID
            current_time: 当前仿真时间（秒）
        """
        with self._lock:
            if drone_id not in self.serving_drones:
                logger.warning(
                    "[%s] depart() 调用异常：drone %s 不在服务槽位中。当前服务: %s",
                    self._host_id, drone_id, list(self.serving_drones.keys()),
                )
                return

            del self.serving_drones[drone_id]
            logger.debug("[%s] drone %s 换电完成，槽位已释放。", self._host_id, drone_id)

            # ── 从等待队列中拉取下一架无人机 ──────────────────────────────────
            self._try_serve_next(current_time)

    def tick_update(self, current_time: float) -> list[str]:
        """
        仿真心跳调用（每个时间步执行一次）。

        遍历 serving_drones，找出预计完成时间 ≤ current_time 的无人机，
        调用 depart() 释放槽位并触发后续队列调度。

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            本轮完成换电的无人机 ID 列表，供仿真引擎更新这些无人机的状态。
        """
        completed: list[str] = []
        with self._lock:
            # 使用快照迭代，避免在迭代中修改 dict
            finished = [
                did for did, ft in list(self.serving_drones.items())
                if ft <= current_time
            ]

        for drone_id in finished:
            self.depart(drone_id, current_time)
            completed.append(drone_id)

        if completed:
            logger.debug(
                "[%s] tick_update t=%.2f，本轮完成换电: %s",
                self._host_id, current_time, completed,
            )
        return completed

    def estimate_wait_time(self, current_time: float) -> float:
        """
        预估一架无人机此刻飞来后需等待多久才能开始换电。

        算法：
          - 若当前有空闲槽位 → 返回 0
          - 否则：
            1. 找出 serving_drones 中最早完成时间，作为第一个空出的槽位时间
            2. 每个提前排队的无人机占用该槽位 swap_time 秒
            3. 累加等待时间

        注意：
          - 这是补能服务阶段的等待估计，不含 mode C rendezvous 等待。
          - 调度器应叠加飞行时间后再比较多个宿主的总等待成本。

        Args:
            current_time: 当前仿真时间（秒）

        Returns:
            预估等待时间（秒）；若有空位则为 0。
        """
        with self._lock:
            free_slots = self.parking_slots - len(self.serving_drones)
            if free_slots > 0:
                return 0.0

            if not self.serving_drones:
                return 0.0

            # 最早可用槽位时间
            earliest_slot_time = min(self.serving_drones.values())

            # 队列中每架无人机依次占用 swap_time
            queue_delay = len(self.wait_queue) * self.swap_time

            wait = max(0.0, earliest_slot_time - current_time) + queue_delay
            return wait

    @property
    def available_slots(self) -> int:
        """当前空闲的服务槽位数量。"""
        with self._lock:
            return max(0, self.parking_slots - len(self.serving_drones))

    @property
    def queue_length(self) -> int:
        """当前充换电服务等待队列长度。"""
        with self._lock:
            return len(self.wait_queue)

    # ══════════════════════════════════════════════════════════════════════════
    # 抽象方法（子类必须实现）
    # ══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def get_location(self, current_time: float) -> Position3D:
        """
        返回宿主在给定仿真时刻的三维坐标（UTM）。

        固定节点（SwapStation / Depot）直接返回存储坐标，忽略时间参数；
        移动节点（Truck）需按 route_nodes 序列对 current_time 做线性插值。
        调度器选择最优补能宿主时统一调用此接口计算飞行距离与时间成本。

        Args:
            current_time: 查询时刻的仿真时间（秒）

        Returns:
            UTM 三维坐标
        """
        ...

    @abstractmethod
    def to_telemetry_dict(self) -> dict:
        """
        将宿主当前状态序列化为字典，坐标转换为 WGS84 格式。

        返回字典供 WebSocket 帧推送至前端地图渲染，字段应包括：
          - entity_id / entity_type
          - lng, lat（WGS84）
          - queue_length, available_slots
          - 子类特有字段（如卡车库存、无人机列表等）

        Returns:
            JSON 可序列化的状态字典
        """
        ...

    # ══════════════════════════════════════════════════════════════════════════
    # 内部工具方法
    # ══════════════════════════════════════════════════════════════════════════

    def _try_serve_next(self, current_time: float) -> None:
        """
        内部方法：若队列非空且有空闲槽位，将队首无人机移入服务。

        调用者须持有 self._lock。
        """
        if self.wait_queue and len(self.serving_drones) < self.parking_slots:
            next_drone = self.wait_queue.pop(0)
            finish_time = current_time + self.swap_time
            self.serving_drones[next_drone] = finish_time
            logger.debug(
                "[%s] 从等待队列中拉取 drone %s，预计完成时间: %.2f",
                self._host_id, next_drone, finish_time,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self._host_id!r}, "
            f"slots={self.parking_slots}, "
            f"serving={len(self.serving_drones)}, "
            f"queue={len(self.wait_queue)})"
        )
