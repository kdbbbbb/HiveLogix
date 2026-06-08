#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 充换电站实体 (Section 3)

SwapStation 是固定地面节点，提供无人机电池快速更换服务。
直接继承 ChargingHost，无需重写换电队列逻辑，仅实现两个抽象方法：
  - get_location()：固定坐标，忽略时间参数
  - to_telemetry_dict()：序列化为 WGS84 供前端渲染

语义边界：
  - 电站的 queue_length / wait_queue 只表示进入电站后的充换电服务等待；
  - 不表示 mode C 中等待被卡车回收的阶段。

根据前端规划，充换电站在地图上需要可视化：
  - 当前槽位占用状态（available_slots / parking_slots）
  - 等待队列长度（queue_length）
  - 预计排队时间（estimate_wait_time）
"""

from __future__ import annotations

import logging

from core.entities.charging_host import ChargingHost
from core.entities.primitives import Position3D

logger = logging.getLogger(__name__)


class SwapStation(ChargingHost):
    """
    固定充换电站。

    Args:
        station_id:    全局唯一 ID，格式建议：'S-{序号}'
        location:      电站固定 UTM 三维坐标
        swap_time:     单次换电耗时（秒）
        parking_slots: 最大并发服务槽位数 K
    """

    def __init__(
        self,
        *,
        station_id: str,
        location: Position3D,
        swap_time: float,
        parking_slots: int,
    ) -> None:
        super().__init__(
            swap_time=swap_time,
            parking_slots=parking_slots,
            host_id=station_id,
        )
        self.station_id: str = station_id
        self.location: Position3D = location

    # ══════════════════════════════════════════════════════════════════════════
    # 实现抽象方法
    # ══════════════════════════════════════════════════════════════════════════

    def get_location(self, current_time: float = 0.0) -> Position3D:
        """
        返回电站固定位置（忽略时间参数）。

        Args:
            current_time: 仿真时间（忽略，仅保持接口一致）

        Returns:
            电站 UTM 三维坐标
        """
        return self.location

    def to_telemetry_dict(self) -> dict:
        """
        序列化电站状态为 WebSocket 推送字典。

        字段说明：
          - available_slots: 当前空闲槽位数
          - queue_length:    当前等待队列长度
          - serving_drone_ids: 正在换电的无人机 ID 列表（供调试）

        Returns:
            JSON 可序列化的电站状态字典
        """
        lon, lat = self.location.to_wgs84()
        with self._lock:
            serving_ids = list(self.serving_drones.keys())
            queue_ids = list(self.wait_queue)

        return {
            "entity_type":       "swap_station",
            "station_id":        self.station_id,
            "lng":               lon,
            "lat":               lat,
            "altitude":          self.location.z,
            "parking_slots":     self.parking_slots,
            "available_slots":   self.available_slots,
            "queue_length":      self.queue_length,
            "swap_time":         self.swap_time,
            "serving_drone_ids": serving_ids,
            "waiting_drone_ids": queue_ids,
        }

    def to_dynamic_state(self) -> dict:
        """
        轻量序列化，仅含 TICK 帧所需字段。
        包含所有 StationConfig 非 Optional 字段，防止 setRuntimeAll 全量替换后字段丢失。
        name 由 EntityManager.get_telemetry() 从 _metadata 合并。
        """
        lon, lat = self.location.to_wgs84()
        with self._lock:
            serving_ids = list(self.serving_drones.keys())
        return {
            "station_id":        self.station_id,
            # ── 静态 StationConfig 必填字段 ───────────────────────────────────
            "lng":               lon,
            "lat":               lat,
            "altitude":          self.location.z,
            "swap_time":         self.swap_time,
            "parking_slots":     self.parking_slots,
            # ── 动态字段 ─────────────────────────────────────────────────────
            "available_slots":   self.available_slots,
            "queue_length":      self.queue_length,
            "serving_drone_ids": serving_ids,
        }

    def __repr__(self) -> str:
        return (
            f"SwapStation(id={self.station_id!r}, "
            f"slots={self.parking_slots}, "
            f"serving={len(self.serving_drones)}, "
            f"queue={len(self.wait_queue)})"
        )
