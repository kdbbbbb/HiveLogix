#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoJSON 与 CSV 导出器
"""

import pandas as pd
import geopandas as gpd


def export_geojson(sub: gpd.GeoDataFrame, path: str):
    """导出建筑物 GeoJSON，包含 height_m 和 no_fly_zone 字段。"""
    out = sub[["geometry", "_h", "_nf"]].copy()
    out.columns = ["geometry", "height_m", "no_fly_zone"]
    out.to_file(path, driver="GeoJSON")
    print(f"[EXPORT] GeoJSON → {path}")


def export_csv(sub: gpd.GeoDataFrame, path: str):
    """导出建筑物 CSV，包含质心坐标、高度和禁飞标记。"""
    cen = sub.geometry.centroid
    pd.DataFrame({
        "id":          range(len(sub)),
        "height_m":    sub["_h"].values,
        "no_fly_zone": sub["_nf"].values,
        "lon":         cen.x.values,
        "lat":         cen.y.values,
    }).to_csv(path, index=False)
    print(f"[EXPORT] CSV → {path}")
