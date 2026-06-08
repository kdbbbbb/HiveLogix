#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标系互转工具

UTM Zone 51N (EPSG:32651) ↔ WGS84 (EPSG:4326) 纯函数封装。

设计原则：
  - 无状态，无副作用，可直接在任何模块中 import 使用
  - Transformer 对象在模块加载时初始化一次（线程安全），避免高频调用时重复建立投影对象
  - 后端 physics engine 内部坐标统一使用 UTM（米制），
    对外（WebSocket 帧、REST 响应）统一使用 WGS84

坐标约定：
  - WGS84:  (lon, lat)  — 经度在前，纬度在后
  - UTM:    (x_m, y_m)  — 东向距离、北向距离（米）

UTM Zone 51N 适用范围：上海所在区域（东经 120°E–126°E，北半球）
"""

from pyproj import Transformer

# ── 模块级单例：仅初始化一次，避免高频调用开销 ─────────────────────────────────
_TR_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
_TR_TO_WGS = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)


def wgs84_to_utm(lon: float, lat: float) -> tuple[float, float]:
    """
    WGS84 经纬度 → UTM Zone 51N 米制坐标。

    Args:
        lon: 经度（度），东经为正
        lat: 纬度（度），北纬为正

    Returns:
        (x_m, y_m) UTM 东向/北向坐标（米）
    """
    return _TR_TO_UTM.transform(lon, lat)


def utm_to_wgs84(x_m: float, y_m: float) -> tuple[float, float]:
    """
    UTM Zone 51N 米制坐标 → WGS84 经纬度。

    Args:
        x_m: UTM 东向坐标（米）
        y_m: UTM 北向坐标（米）

    Returns:
        (lon, lat) WGS84 经纬度
    """
    return _TR_TO_WGS.transform(x_m, y_m)
