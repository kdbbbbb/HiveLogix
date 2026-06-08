#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真场景上下文打包服务

职责：
  1. 调用 geo 模块的 OSM 下载能力，获取选区路网（OSM 优先，无网络时降级为网格路网标注）
  2. 直接从 OSM XML 提取路网节点的原始 WGS84 经纬度（不经 UTM 投影，消除精度损失）
  3. 计算 SUMO netOffset（选区左下角的 UTM 坐标），供 physics engine 还原 SUMO 坐标用
  4. 打包为 SceneContext 字典（JSON 可序列化），仅包含轻量级 meta 与路网拓扑
  5. 对相同参数的请求做幂等缓存（MD5 哈希），防止重复计算

SceneContext 字段说明（对外接口协议）：
  scene_id      : UUID，唯一标识本次场景
  sel_bounds    : 选区经纬度 {minx, miny, maxx, maxy}
  threshold     : 飞行高度阈值（米）
  height_column : 使用的高度字段名
  meta:
    road_source : 'osm' | 'grid'（路网数据来源）
    road_nodes  : 参与路网的 OSM 节点数量（仅统计量）
    road_edges  : 参与路网的道路段数量（仅统计量）
    created_at  : ISO 8601 UTC 时间戳
    utm_zone    : 51（UTM Zone 51N）
    utm_band    : 'N'
    net_offset  : {ox, oy} — 选区左下角的 UTM 绝对坐标（米）
                  physics engine 还原公式：utm_x = sumo_node_x + ox
  road_network:
    nodes : [{id, lng, lat}, ...]  WGS84，直接来自 OSM 原始数据
    edges : [{shape: [[lng,lat], ...]}, ...]  WGS84
    bounds: {min_lng, min_lat, max_lng, max_lat}

注意：SceneContext 不包含 buildings GeoJSON；
     建筑数据由前端 UnifiedMapView 直接调用 /api/geo/query 并行加载。
"""

import datetime
import hashlib
import json
import sys
import uuid
import xml.etree.ElementTree as ET
from typing import Optional

from pyproj import Transformer

# ── sys.path 中 geo 目录已由 app.py 注入，可直接 import ──────────────────────
import osm_service as _osm_svc


# ── 内存缓存：params_hash → SceneContext ────────────────────────────────────
# 以 MD5(sel_bounds + threshold) 为键，相同参数直接返回已有结果
_cache: dict[str, dict] = {}


def _params_hash(sel_bounds: dict, threshold: float) -> str:
    """生成请求参数的 MD5 摘要，用于幂等缓存键。"""
    key = json.dumps({**sel_bounds, "thr": round(threshold, 2)}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:16]


def prepare_scene(
    sel_bounds: dict,
    threshold: float,
    height_column: Optional[str],
) -> dict:
    """
    打包仿真场景上下文。

    Args:
        sel_bounds     : 选区经纬度边界，dict with keys: minx, miny, maxx, maxy
        threshold      : 无人机飞行高度阈值（米），用于建筑禁飞区分类
        height_column  : Shapefile 高度字段名，None 时使用服务端默认值

    Returns:
        SceneContext dict（JSON 可序列化）

    Raises:
        ValueError: sel_bounds 缺少必要字段时抛出
    """
    # ── 幂等检查 ──────────────────────────────────────────────────────────────
    h = _params_hash(sel_bounds, threshold)
    if h in _cache:
        return _cache[h]

    minx: float = sel_bounds["minx"]
    miny: float = sel_bounds["miny"]
    maxx: float = sel_bounds["maxx"]
    maxy: float = sel_bounds["maxy"]

    # ── 1. 下载 OSM 路网（失败时记录来源为 grid，由前端知悉但不中断流程） ────────
    road_source = "osm"
    osm_xml: Optional[str] = None
    try:
        osm_xml = _osm_svc.download_osm(minx, miny, maxx, maxy)
    except Exception as exc:
        print(f"[scene] OSM 下载失败，将标注 road_source=grid: {exc}", file=sys.stderr)
        road_source = "grid"

    # ── 2. 从 OSM XML 直接读取节点 WGS84 坐标（不经 UTM，消除投影精度损失） ─────
    road_nodes: list[dict] = []
    road_edges: list[dict] = []

    if road_source == "osm" and osm_xml:
        root = ET.fromstring(osm_xml)

        # 2a. 收集所有 OSM 节点（经纬度）
        osm_nodes: dict[str, tuple[float, float]] = {}
        for nd in root.iter("node"):
            nid = nd.get("id")
            if nid:
                osm_nodes[nid] = (
                    float(nd.get("lon", 0)),
                    float(nd.get("lat", 0)),
                )

        # 2b. 收集道路 Way，过滤非机动车道路类型
        hw_speeds = _osm_svc.HW_SPEED  # 复用 geo 模块的道路类型白名单
        for way in root.iter("way"):
            tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
            if tags.get("highway", "") not in hw_speeds:
                continue
            coords: list[list[float]] = []
            for ref in way.iter("nd"):
                ref_id = ref.get("ref")
                if ref_id and ref_id in osm_nodes:
                    coords.append(list(osm_nodes[ref_id]))  # [lon, lat]
            if len(coords) >= 2:
                road_edges.append({"shape": coords})

        road_nodes = [
            {"id": nid, "lng": lon, "lat": lat}
            for nid, (lon, lat) in osm_nodes.items()
        ]

    else:
        # ── OSM 不可用时降级为合成网格路网，确保前端始终有路网可渲染 ─────────────
        # 生成 6×6 纵横路网：6 条纵向线 + 6 条横向线，覆盖整个选区
        n_lines = 6
        lng_step = (maxx - minx) / (n_lines + 1)
        lat_step = (maxy - miny) / (n_lines + 1)
        for i in range(1, n_lines + 1):
            lng = minx + lng_step * i
            road_edges.append({"shape": [[lng, miny], [lng, maxy]]})
        for j in range(1, n_lines + 1):
            lat = miny + lat_step * j
            road_edges.append({"shape": [[minx, lat], [maxx, lat]]})
        print(f"[scene] Overpass 不可达，已生成合成网格路网 {len(road_edges)} 条")

    # ── 3. 计算 netOffset（与 sumo_net_osm.py 保持一致，以选区左下角为 UTM 原点）
    _tr = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    ox, oy = _tr.transform(minx, miny)

    # ── 4. 组装 SceneContext ──────────────────────────────────────────────────
    scene_id = str(uuid.uuid4())
    ctx: dict = {
        "scene_id": scene_id,
        "sel_bounds": sel_bounds,
        "threshold": threshold,
        "height_column": height_column,
        "meta": {
            "road_source":  road_source,
            "road_nodes":   len(road_nodes),
            "road_edges":   len(road_edges),
            "created_at":   datetime.datetime.utcnow().isoformat() + "Z",
            # ── physics engine 必须字段 ────────────────────────────────────
            # SUMO net.xml 中节点坐标为相对于选区左下角的米制偏移量
            # 还原公式：utm_x = sumo_node_x + ox；utm_y = sumo_node_y + oy
            "utm_zone":     51,
            "utm_band":     "N",
            "net_offset": {
                "ox": round(float(ox), 2),
                "oy": round(float(oy), 2),
            },
        },
        "road_network": {
            "nodes": road_nodes,
            "edges": road_edges,
            "bounds": {
                "min_lng": minx,
                "min_lat": miny,
                "max_lng": maxx,
                "max_lat": maxy,
            },
        },
    }

    # ── 5. 写入缓存 ───────────────────────────────────────────────────────────
    _cache[h] = ctx
    print(
        f"[scene] 场景已创建 scene_id={scene_id} "
        f"road_source={road_source} "
        f"nodes={len(road_nodes)} edges={len(road_edges)}"
    )
    return ctx


def get_scene_by_id(scene_id: str) -> Optional[dict]:
    """按 scene_id 从缓存中检索已有场景，不存在时返回 None。"""
    for ctx in _cache.values():
        if ctx["scene_id"] == scene_id:
            return ctx
    return None


def load_preset_scene(preset_id: str) -> Optional[dict]:
    """
    加载预设场景（从磁盘缓存）。
    
    Args:
        preset_id: 预设场景 ID，如 'default_test_4x4km'
    
    Returns:
        SceneContext dict，格式与 prepare_scene() 返回值相同
        或 None 如果预设场景不存在
    """
    try:
        from preset_scenes import get_preset_scene
        preset = get_preset_scene(preset_id)
        if not preset:
            return None
        
        # 将预设场景转换为 SceneContext 格式
        bounds = preset.get("bounds", {})
        config = preset.get("config", {})
        osm_data = preset.get("osm_network", {})
        
        sel_bounds = {
            "minx": bounds.get("min_lng", 0),
            "miny": bounds.get("min_lat", 0),
            "maxx": bounds.get("max_lng", 0),
            "maxy": bounds.get("max_lat", 0),
        }
        
        # 使用配置中的阈值，或默认 80
        threshold = config.get("threshold", 80)
        
        # 从 OSM GeoJSON 提取特征（道路）
        geojson_features = osm_data.get("features", []) if isinstance(osm_data, dict) else []
        
        # 过滤出 LineString 类型作为道路
        road_edges = [
            {"shape": f["geometry"]["coordinates"]} 
            for f in geojson_features 
            if f.get("geometry", {}).get("type") == "LineString"
        ]
        
        # 计算 netOffset
        _tr = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        ox, oy = _tr.transform(sel_bounds["minx"], sel_bounds["miny"])
        
        # 组装 SceneContext，预设场景使用 preset_id 作为稳定标识符
        scene_id = preset_id
        ctx: dict = {
            "scene_id": scene_id,
            "sel_bounds": sel_bounds,
            "threshold": threshold,
            # 预设场景不强制指定高度字段，交给 /api/geo/query 使用数据加载器自动检测值。
            # 这样可避免 Shapefile 实际列名为 "Height" 等大小写差异时全部变成 0m。
            "height_column": None,
            "meta": {
                "road_source": "osm_preset",
                "road_nodes": 0,
                "road_edges": len(road_edges),
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "utm_zone": 51,
                "utm_band": "N",
                "net_offset": {
                    "ox": round(float(ox), 2),
                    "oy": round(float(oy), 2),
                },
            },
            "road_network": {
                "nodes": [],
                "edges": road_edges,
                "bounds": {
                    "min_lng": sel_bounds["minx"],
                    "min_lat": sel_bounds["miny"],
                    "max_lng": sel_bounds["maxx"],
                    "max_lat": sel_bounds["maxy"],
                },
            },
        }
        
        # 写入缓存
        h = _params_hash(sel_bounds, threshold)
        _cache[h] = ctx
        print(f"[scene] 预设场景已加载 preset_id={preset_id} scene_id={scene_id}")
        return ctx
        
    except Exception as e:
        print(f"[ERROR] 加载预设场景失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

