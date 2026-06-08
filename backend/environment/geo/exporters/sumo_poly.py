#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMO polygon additional file (.add.xml) 导出器
将建筑物多边形导出为 SUMO 可识别的禁飞区/可飞区叠加层。
坐标系：EPSG:32651 (UTM Zone 51N)，以选区左下角为原点。
"""

import geopandas as gpd


def export_sumo_poly(gdf: gpd.GeoDataFrame, threshold: float, path: str):
    """
    将含 _h（高度）和 _nf（是否禁飞）字段的 GeoDataFrame 导出为 SUMO .add.xml。
    用法：sumo-gui -n <net.xml> --additional-files no_fly_zones.add.xml
    """
    gdf_utm       = gdf.to_crs("EPSG:32651")
    b             = gdf_utm.total_bounds
    ox, oy        = float(b[0]), float(b[1])
    width_m       = float(b[2] - b[0])
    height_m_area = float(b[3] - b[1])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!--",
        f"  UAV 禁飞区地图 | 高度阈值: {threshold} m",
        f"  坐标系: EPSG:32651 (UTM Zone 51N)",
        f"  原点(左下): E={ox:.2f} N={oy:.2f}",
        f"  范围: {width_m:.1f} m × {height_m_area:.1f} m",
        f"  红色 (no_fly_zone): 建筑高度 >= {threshold} m",
        f"  绿色 (fly_zone)   : 建筑高度 <  {threshold} m",
        "-->",
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '            xsi:noNamespaceSchemaLocation='
        '"http://sumo.dlr.de/xsd/additional_file.xsd">',
    ]

    def _write_poly(poly_geom, pid, h, is_nf):
        pts   = list(poly_geom.exterior.coords)
        shape = " ".join(f"{x - ox:.2f},{y - oy:.2f}" for x, y in pts)
        color = "255,0,0,160" if is_nf else "0,180,0,80"
        layer = "20"          if is_nf else "10"
        ptype = "no_fly_zone" if is_nf else "fly_zone"
        lines.append(
            f'    <poly id="{pid}" type="{ptype}" color="{color}" '
            f'fill="1" layer="{layer}" height="{h:.2f}" shape="{shape}"/>'
        )

    for idx, row in gdf_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        h  = float(row.get("_h", 0))
        nf = bool(row.get("_nf", False))
        if geom.geom_type == "Polygon":
            _write_poly(geom, f"b_{idx}", h, nf)
        elif geom.geom_type == "MultiPolygon":
            for i, part in enumerate(geom.geoms):
                _write_poly(part, f"b_{idx}_{i}", h, nf)

    lines.append("</additional>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[EXPORT] sumo_poly: {len(gdf_utm)} 栋建筑 → {path}")
