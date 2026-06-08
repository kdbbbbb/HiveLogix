#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSM 数据获取与道路 GeoJSON 转换服务
负责：通过 Overpass API 下载道路数据、转换为 GeoJSON 前端叠加层
道路类型常量（HW_*）同时供 exporters/sumo_net_osm.py 使用
"""

import urllib.request as _ureq
import urllib.parse  as _uparse
import xml.etree.ElementTree as _ET
import networkx as nx
from pyproj import Transformer

# ── 道路类型参数表 ─────────────────────────────────────────────────────────────
HW_SPEED = {
    "motorway": 33.33,       "trunk": 27.78,         "primary": 22.22,
    "secondary": 16.67,      "tertiary": 13.89,
    "residential": 8.33,     "service": 5.56,         "unclassified": 11.11,
    "motorway_link": 22.22,  "trunk_link": 16.67,
    "primary_link": 13.89,   "secondary_link": 11.11, "tertiary_link": 8.33,
    "living_street": 2.78,   "road": 11.11,
}
HW_LANES = {
    "motorway": 3,  "trunk": 2,     "primary": 2,     "secondary": 2,
    "tertiary": 1,  "residential": 1, "service": 1,   "unclassified": 1,
    "motorway_link": 1,  "trunk_link": 1,  "primary_link": 1,
    "secondary_link": 1, "tertiary_link": 1, "living_street": 1, "road": 1,
}
HW_PRIO = {
    "motorway": 14, "trunk": 13,    "primary": 12,    "secondary": 11,
    "tertiary": 10, "residential": 5, "service": 3,   "unclassified": 4,
    "motorway_link": 9,  "trunk_link": 8,  "primary_link": 7,
    "secondary_link": 6, "tertiary_link": 5, "living_street": 4, "road": 5,
}
ONEWAY_HW = {"motorway", "motorway_link"}


def download_osm(minx: float, miny: float, maxx: float, maxy: float) -> str:
    """通过 Overpass API 下载选区内道路 OSM 数据，返回 XML 字符串。"""
    query = (
        f"[out:xml][timeout:90];\n"
        f"(\n"
        f'  way["highway"]({miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f});\n'
        f"  node(w);\n"
        f");\n"
        f"out body;\n"
        f">;\n"
        f"out skel qt;\n"
    )
    data = _uparse.urlencode({"data": query}).encode("utf-8")
    req  = _ureq.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "UAV-NoFlyMap/1.0"},
    )
    with _ureq.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8")


def osm_to_geojson(osm_xml: str) -> dict:
    """将 OSM XML 转换为 GeoJSON FeatureCollection（道路线段），供前端叠加层使用。"""
    root   = _ET.fromstring(osm_xml)
    nodes  = {}
    for nd in root.iter("node"):
        nodes[nd.get("id")] = (float(nd.get("lon", 0)), float(nd.get("lat", 0)))

    features = []
    for way in root.iter("way"):
        tags   = {t.get("k"): t.get("v") for t in way.iter("tag")}
        hw     = tags.get("highway", "")
        if hw not in HW_SPEED:
            continue
        coords = [nodes[r.get("ref")] for r in way.iter("nd") if r.get("ref") in nodes]
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"highway": hw, "name": tags.get("name", "")},
        })

    return {"type": "FeatureCollection", "features": features}


def build_road_graph(
    osm_xml: str,
    *,
    respect_osm_oneway: bool = False,
) -> tuple[nx.DiGraph, dict]:
    """
    从 OSM XML 构建道路图，用于路径计算。
    
    Returns:
        G: networkx.DiGraph，节点为 OSM node id，边权重为距离（米）
        nodes: dict[node_id: (lon, lat)]
    """
    root = _ET.fromstring(osm_xml)
    nodes = {}
    for nd in root.iter("node"):
        nodes[nd.get("id")] = (float(nd.get("lon", 0)), float(nd.get("lat", 0)))

    G = nx.DiGraph()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651")  # WGS84 to UTM Zone 51N

    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        hw = tags.get("highway", "")
        if hw not in HW_SPEED:
            continue
        is_oneway = (
            tags.get("oneway", "no") in ("yes", "1", "true") or hw in ONEWAY_HW
        )
        nd_refs = [r.get("ref") for r in way.iter("nd") if r.get("ref") in nodes]
        for i in range(len(nd_refs) - 1):
            u = nd_refs[i]
            v = nd_refs[i + 1]
            lon1, lat1 = nodes[u]
            lon2, lat2 = nodes[v]
            x1, y1 = transformer.transform(lat1, lon1)
            x2, y2 = transformer.transform(lat2, lon2)
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            G.add_edge(u, v, weight=dist)
            if respect_osm_oneway:
                add_reverse = not is_oneway
            else:
                # 兼容仓库内既有求解器：历史上仅把 motorway 系列视为单向。
                add_reverse = hw not in ONEWAY_HW
            if add_reverse:
                G.add_edge(v, u, weight=dist)

    return G, nodes


def find_nearest_node(G: nx.DiGraph, nodes: dict, pos_x: float, pos_y: float) -> str:
    """
    找到最近的 OSM 节点（必须在图中）。
    
    Args:
        pos_x, pos_y: UTM 坐标
    """
    if not G or not G.nodes():
        return ""
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651")
    min_dist = float('inf')
    nearest = ""
    # 只在图中已有的节点里查找，避免孤立节点
    for nid in G.nodes():
        if nid not in nodes:
            continue
        lon, lat = nodes[nid]
        x, y = transformer.transform(lat, lon)
        dist = ((x - pos_x)**2 + (y - pos_y)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = nid
    return nearest


def shortest_path(G: nx.DiGraph, start_node: str, end_node: str) -> list[str]:
    """计算最短路径，返回节点列表。"""
    try:
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def shortest_path_length(G: nx.DiGraph, start_node: str, end_node: str) -> float | None:
    """返回最短路径总距离（米），不可达时返回 None。"""
    try:
        return float(nx.shortest_path_length(G, start_node, end_node, weight='weight'))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
