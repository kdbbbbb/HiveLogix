#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建筑查询与禁飞区分类服务
负责：空间裁剪、高度阈值分类、GeoJSON FeatureCollection 生成
"""

import json

import pandas as pd
import geopandas as gpd


def query_buildings(
    gdf: gpd.GeoDataFrame,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    threshold: float,
    h_col: str | None,
    max_feat: int = 30000,
) -> dict:
    """
    裁剪选区内建筑，按高度阈值分类，返回 GeoJSON FeatureCollection。
    结果包含 stats 字段：total / shown / no_fly / fly / truncated。
    """
    sub   = gdf.cx[minx:maxx, miny:maxy].copy()
    total = len(sub)

    if total == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "stats": {"total": 0, "no_fly": 0, "fly": 0, "shown": 0, "truncated": False},
        }

    if h_col and h_col in sub.columns:
        heights = pd.to_numeric(sub[h_col], errors="coerce").fillna(0)
    else:
        heights = pd.Series(0.0, index=sub.index)

    nf_mask      = heights >= threshold
    no_fly_count = int(nf_mask.sum())

    # 超出上限时优先保留禁飞区建筑
    truncated = total > max_feat
    if truncated:
        nf_idx  = nf_mask[nf_mask].index[: max_feat // 2]
        fly_idx = nf_mask[~nf_mask].index[: max_feat // 2]
        keep    = nf_idx.append(fly_idx)
        sub     = sub.loc[keep]
        heights = heights.loc[keep]
        nf_mask = nf_mask.loc[keep]

    # 构建精简 GeoDataFrame（仅保留必要字段）并简化几何
    # 使用 geometry= 参数确保正确设置几何列，按 index 对齐
    slim = gpd.GeoDataFrame(
        {
            "h":  heights.round(1),
            "nf": nf_mask.astype(bool),
        },
        geometry=sub.geometry.simplify(0.00001, preserve_topology=True),
        crs=sub.crs,
    )
    # 极少数建筑在简化后退化为空几何（如细长条），过滤掉
    slim = slim[slim.geometry.notna() & ~slim.geometry.is_empty]

    # 向量化序列化：利用 geopandas C 扩展，远快于 iterrows + mapping
    result: dict = json.loads(slim.to_json())
    result["stats"] = {
        "total":     total,
        "shown":     len(slim),
        "no_fly":    no_fly_count,
        "fly":       total - no_fly_count,
        "truncated": truncated,
    }
    return result


def prepare_sub(gdf: gpd.GeoDataFrame, minx, miny, maxx, maxy, threshold, h_col):
    """
    裁剪并附加 _h（高度）和 _nf（是否禁飞）字段，供导出器使用。
    """
    sub = gdf.cx[minx:maxx, miny:maxy].copy()
    if h_col and h_col in sub.columns:
        heights = pd.to_numeric(sub[h_col], errors="coerce").fillna(0)
    else:
        heights = pd.Series(0.0, index=sub.index)
    sub["_h"]  = heights
    sub["_nf"] = heights >= threshold
    return sub
