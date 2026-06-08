#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 坐标转换工具包装层

职责：
  将 environment/scene/coord_transformer.py 中的纯函数重新暴露，
  使 core/ 层代码无需感知物理文件路径，也无需修改 sys.path。

  后端内部坐标系：UTM Zone 51N (EPSG:32651)，单位：米
  对外（WebSocket 帧 / REST）：WGS84 (EPSG:4326)，(lon, lat)

使用方式：
  from utils.coord_utils import utm_to_wgs84, wgs84_to_utm
"""

from __future__ import annotations

import os
import sys

# ── 确保 coord_transformer 所在目录在 sys.path 中 ─────────────────────────────
# 本文件位于 backend/utils/，coord_transformer.py 位于 backend/environment/scene/
_SCENE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "environment", "scene")
)
if _SCENE_DIR not in sys.path:
    sys.path.insert(0, _SCENE_DIR)

try:
    from coord_transformer import utm_to_wgs84, wgs84_to_utm  # type: ignore[import]
except ImportError as _exc:
    # 仅在依赖 pyproj 未安装时走此分支；其他错误应直接暴露
    _ERR_MSG = (
        f"导入 coord_transformer 失败: {_exc}\n"
        "请确认已安装 pyproj：pip install pyproj"
    )

    def utm_to_wgs84(x_m: float, y_m: float) -> tuple[float, float]:  # type: ignore[misc]
        raise RuntimeError(_ERR_MSG)

    def wgs84_to_utm(lon: float, lat: float) -> tuple[float, float]:  # type: ignore[misc]
        raise RuntimeError(_ERR_MSG)


__all__ = ["utm_to_wgs84", "wgs84_to_utm"]
