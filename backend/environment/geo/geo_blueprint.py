#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 禁飞区地图 — Flask Blueprint

所有业务路由注册在此 Blueprint 中。
- 独立运行：由 app.py 挂载到 /api
- 集成运行：由主入口 backend/app.py 挂载到 /api/geo
"""

import io
import os
import traceback
import zipfile

from flask import Blueprint, request, jsonify, send_file

from data_loader      import load_shapefile_async, get_state, get_gdf  # noqa: E402
from building_service import query_buildings, prepare_sub               # noqa: E402
from osm_service      import download_osm, osm_to_geojson               # noqa: E402
from preset_scenes    import get_buildings_from_cache                   # noqa: E402

from exporters.sumo_poly     import export_sumo_poly      # noqa: E402
from exporters.sumo_net_grid import export_grid_net       # noqa: E402
from exporters.sumo_net_osm  import osm_to_sumo_net       # noqa: E402
from exporters.geofile       import export_geojson, export_csv  # noqa: E402

# ── 常量 ──────────────────────────────────────────────────────────────────────
GEO_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(GEO_DIR, "output")

# ── Blueprint 实例 ─────────────────────────────────────────────────────────────
geo_bp = Blueprint("geo", __name__)


# ── 路由 ──────────────────────────────────────────────────────────────────────

@geo_bp.route("/status")
def api_status():
    return jsonify(get_state())


@geo_bp.route("/roads", methods=["POST"])
def api_roads():
    body = request.get_json(force=True)
    try:
        osm_xml = download_osm(
            float(body["minx"]), float(body["miny"]),
            float(body["maxx"]), float(body["maxy"]),
        )
        return jsonify(osm_to_geojson(osm_xml))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@geo_bp.route("/query", methods=["POST"])
def api_query():
    body      = request.get_json(force=True)
    scene_id  = body.get("scene_id")
    
    # 优先检查预设场景缓存
    if scene_id:
        cached = get_buildings_from_cache(scene_id)
        if cached:
            print(f"[GEO] 从缓存加载预设场景 {scene_id}")
            return jsonify(cached)
    
    # 正常流程：从 Shapefile 加载
    state = get_state()
    if not state["loaded"]:
        return jsonify({"error": "数据尚未加载完毕，请稍候"}), 503

    threshold = float(body.get("threshold", 50))
    h_col     = body.get("height_column") or state["height_column"]
    max_feat  = min(int(body.get("max", 30000)), 60000)

    try:
        result = query_buildings(
            get_gdf(),
            float(body["minx"]), float(body["miny"]),
            float(body["maxx"]), float(body["maxy"]),
            threshold, h_col, max_feat,
        )
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@geo_bp.route("/export", methods=["POST"])
def api_export():
    state = get_state()
    if not state["loaded"]:
        return jsonify({"error": "数据尚未加载完毕"}), 503

    body      = request.get_json(force=True)
    minx      = float(body["minx"])
    miny      = float(body["miny"])
    maxx      = float(body["maxx"])
    maxy      = float(body["maxy"])
    threshold = float(body.get("threshold", 50))
    h_col     = body.get("height_column") or state["height_column"]
    fmt       = body.get("format", "sumo_poly")

    try:
        sub = prepare_sub(get_gdf(), minx, miny, maxx, maxy, threshold, h_col)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if fmt == "sumo_poly":
            path = os.path.join(OUTPUT_DIR, "no_fly_zones.add.xml")
            export_sumo_poly(sub, threshold, path)
            return send_file(path, as_attachment=True,
                             download_name="no_fly_zones.add.xml",
                             mimetype="application/xml")

        if fmt == "geojson":
            path = os.path.join(OUTPUT_DIR, "buildings.geojson")
            export_geojson(sub, path)
            return send_file(path, as_attachment=True,
                             download_name="buildings.geojson",
                             mimetype="application/geo+json")

        if fmt == "csv":
            path = os.path.join(OUTPUT_DIR, "buildings.csv")
            export_csv(sub, path)
            return send_file(path, as_attachment=True,
                             download_name="buildings.csv",
                             mimetype="text/csv")

        if fmt == "sumo_zip":
            spacing   = float(body.get("grid_spacing", 200))
            poly_path = os.path.join(OUTPUT_DIR, "no_fly_zones.add.xml")
            net_path  = os.path.join(OUTPUT_DIR, "grid.net.xml")
            gdf_utm   = sub.to_crs("EPSG:32651")
            b         = gdf_utm.total_bounds
            export_sumo_poly(sub, threshold, poly_path)
            export_grid_net(
                ox=float(b[0]), oy=float(b[1]),
                width_m=float(b[2] - b[0]), height_m=float(b[3] - b[1]),
                grid_spacing=spacing, orig_bounds=(minx, miny, maxx, maxy),
                path=net_path,
            )
            zip_buf = _make_zip(
                files=[(poly_path, "no_fly_zones.add.xml"), (net_path, "grid.net.xml")],
                readme=_readme_grid(threshold, spacing, minx, miny, maxx, maxy),
            )
            return send_file(zip_buf, as_attachment=True,
                             download_name="uav_no_fly_sumo.zip",
                             mimetype="application/zip")

        if fmt == "sumo_zip_osm":
            poly_path = os.path.join(OUTPUT_DIR, "no_fly_zones.add.xml")
            net_path  = os.path.join(OUTPUT_DIR, "roads.net.xml")
            osm_path  = os.path.join(OUTPUT_DIR, "area.osm")
            export_sumo_poly(sub, threshold, poly_path)
            print(f"[OSM] 下载道路数据 ({minx:.4f},{miny:.4f})→({maxx:.4f},{maxy:.4f}) ...")
            osm_xml = download_osm(minx, miny, maxx, maxy)
            with open(osm_path, "w", encoding="utf-8") as f:
                f.write(osm_xml)
            n_nodes, n_edges = osm_to_sumo_net(
                osm_xml=osm_xml, orig_bounds=(minx, miny, maxx, maxy), path=net_path,
            )
            zip_buf = _make_zip(
                files=[
                    (poly_path, "no_fly_zones.add.xml"),
                    (net_path,  "roads.net.xml"),
                    (osm_path,  "area.osm"),
                ],
                readme=_readme_osm(threshold, minx, miny, maxx, maxy, n_nodes, n_edges),
            )
            return send_file(zip_buf, as_attachment=True,
                             download_name="uav_no_fly_sumo_osm.zip",
                             mimetype="application/zip")

        return jsonify({"error": f"未知格式: {fmt}"}), 400

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ── 私有辅助函数 ───────────────────────────────────────────────────────────────

def _make_zip(files: list, readme: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, name in files:
            zf.write(src, name)
        zf.writestr("README.txt", readme)
    buf.seek(0)
    return buf


def _readme_grid(threshold, spacing, minx, miny, maxx, maxy) -> str:
    return (
        "UAV 禁飞区 SUMO 完整包\n"
        "=======================\n"
        f"高度阈值: {threshold} m\n"
        f"网格间距: {spacing:.0f} m\n"
        f"选区范围: ({minx:.6f}, {miny:.6f}) → ({maxx:.6f}, {maxy:.6f})\n"
        "坐标系:   EPSG:32651 (UTM Zone 51N)\n\n"
        "使用方法:\n"
        "  sumo-gui -n grid.net.xml --additional-files no_fly_zones.add.xml\n\n"
        "文件说明:\n"
        "  grid.net.xml          — 覆盖选区的网格路网（自动生成）\n"
        "  no_fly_zones.add.xml  — 建筑多边形禁飞区叠加层\n\n"
        "在 sumo-gui 中，红色多边形 = 禁飞区，绿色 = 可飞区。\n"
    )


def _readme_osm(threshold, minx, miny, maxx, maxy, n_nodes, n_edges) -> str:
    return (
        "UAV 禁飞区 SUMO 真实路网包\n"
        "===========================\n"
        f"高度阈值: {threshold} m\n"
        f"选区范围: ({minx:.6f}, {miny:.6f}) → ({maxx:.6f}, {maxy:.6f})\n"
        "路网来源: OpenStreetMap (Overpass API)\n"
        f"节点数:   {n_nodes}\n"
        f"边数:     {n_edges}\n"
        "坐标系:   EPSG:32651 (UTM Zone 51N)\n\n"
        "使用方法:\n"
        "  sumo-gui -n roads.net.xml --additional-files no_fly_zones.add.xml\n\n"
        "文件说明:\n"
        "  roads.net.xml         — 基于 OSM 真实道路的 SUMO 路网\n"
        "  no_fly_zones.add.xml  — 建筑多边形禁飞区叠加层\n"
        "  area.osm              — 原始 OSM 数据\n\n"
        "在 sumo-gui 中，红色多边形 = 禁飞区，绿色 = 可飞区。\n"
    )
