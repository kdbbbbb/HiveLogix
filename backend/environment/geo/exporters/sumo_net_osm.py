#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSM → SUMO net.xml 转换器
将 Overpass 下载的 OSM XML 直接解析为 SUMO 路网文件（不依赖 netconvert）。
坐标系：EPSG:32651 (UTM Zone 51N)，以选区左下角为 SUMO 原点。
"""

import math
import xml.etree.ElementTree as _ET
from collections import defaultdict as _dd
from dataclasses import dataclass

from osm_service import HW_SPEED, HW_LANES, HW_PRIO, ONEWAY_HW


@dataclass(frozen=True)
class SumoNetEdge:
    """SUMO net.xml 中的一条有向 edge。"""

    edge_id: str
    from_node: str
    to_node: str
    osm_node_ids: tuple[str, ...]
    priority: int
    num_lanes: int
    speed: float
    length: float
    shape_coords_utm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SumoNetArtifacts:
    """导出 net.xml 与生成 route 文件共用的中间结果。"""

    ox: float
    oy: float
    conv_boundary: tuple[float, float, float, float]
    orig_bounds: tuple[float, float, float, float]
    utm_nodes: dict[str, tuple[float, float]]
    junction_nodes: tuple[str, ...]
    edges: tuple[SumoNetEdge, ...]
    connections: tuple[tuple[str, str], ...]
    directed_step_to_edge: dict[tuple[str, str], str]


def build_sumo_net_artifacts(
    osm_xml: str,
    orig_bounds: tuple[float, float, float, float],
) -> SumoNetArtifacts:
    """
    解析 OSM XML，构建 SUMO 路网导出与 route 映射共用的中间结果。
    """
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    root = _ET.fromstring(osm_xml)

    osm_nodes = {}
    for nd in root.iter("node"):
        nid = nd.get("id")
        osm_nodes[nid] = (float(nd.get("lon", 0)), float(nd.get("lat", 0)))

    class _Way:
        __slots__ = ["wid", "nodes", "hw", "oneway", "lanes", "speed", "prio"]

    ways = []
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        hw = tags.get("highway", "")
        if hw not in HW_SPEED:
            continue
        w = _Way()
        w.wid = way.get("id")
        w.nodes = [r.get("ref") for r in way.iter("nd") if r.get("ref") in osm_nodes]
        if len(w.nodes) < 2:
            continue
        w.hw = hw
        w.oneway = tags.get("oneway", "no") in ("yes", "1", "true") or hw in ONEWAY_HW
        try:
            w.lanes = max(1, int(tags.get("lanes", HW_LANES.get(hw, 1))))
        except ValueError:
            w.lanes = HW_LANES.get(hw, 1)
        spd = tags.get("maxspeed", "")
        w.speed = float(spd) / 3.6 if spd.isdigit() else HW_SPEED.get(hw, 11.11)
        w.prio = HW_PRIO.get(hw, 5)
        ways.append(w)

    if not ways:
        raise ValueError("选区内未找到 OSM 道路数据，请尝试扩大选区范围")

    nref = _dd(int)
    for w in ways:
        nref[w.nodes[0]] += 10
        nref[w.nodes[-1]] += 10
        for n in w.nodes[1:-1]:
            nref[n] += 1
    junctions = {nid for nid, cnt in nref.items() if cnt >= 2}

    all_nids = {n for w in ways for n in w.nodes}
    utm = {}
    for nid in all_nids:
        if nid in osm_nodes:
            x, y = tr.transform(*osm_nodes[nid])
            utm[nid] = (float(x), float(y))

    ox, oy = tr.transform(orig_bounds[0], orig_bounds[1])
    ox, oy = float(ox), float(oy)

    segments = []
    for w in ways:
        seg_start = 0
        seg_idx = 0
        for i in range(1, len(w.nodes)):
            if w.nodes[i] in junctions or i == len(w.nodes) - 1:
                chunk = w.nodes[seg_start: i + 1]
                if len(chunk) >= 2:
                    segments.append((f"e{w.wid}s{seg_idx}", chunk, w))
                    seg_idx += 1
                seg_start = i

    used_j = set()
    for _, node_list, _ in segments:
        used_j.add(node_list[0])
        used_j.add(node_list[-1])

    all_xy = [utm[n] for n in utm]
    conv_boundary = (
        min(x - ox for x, y in all_xy),
        min(y - oy for x, y in all_xy),
        max(x - ox for x, y in all_xy),
        max(y - oy for x, y in all_xy),
    )

    edges: list[SumoNetEdge] = []
    directed_step_to_edge: dict[tuple[str, str], str] = {}

    def _build_edge(
        edge_id: str,
        node_list: list[str],
        w: _Way,
    ) -> SumoNetEdge | None:
        coords = [utm[n] for n in node_list if n in utm]
        if len(coords) < 2:
            return None
        length = sum(
            math.hypot(coords[k + 1][0] - coords[k][0], coords[k + 1][1] - coords[k][1])
            for k in range(len(coords) - 1)
        )
        if length < 0.1:
            return None
        return SumoNetEdge(
            edge_id=edge_id,
            from_node=node_list[0],
            to_node=node_list[-1],
            osm_node_ids=tuple(node_list),
            priority=w.prio,
            num_lanes=w.lanes,
            speed=w.speed,
            length=length,
            shape_coords_utm=tuple(coords),
        )

    for sid, node_list, w in segments:
        fwd_edge = _build_edge(sid, node_list, w)
        if fwd_edge is not None:
            edges.append(fwd_edge)
            for idx in range(len(node_list) - 1):
                directed_step_to_edge[(node_list[idx], node_list[idx + 1])] = sid

        if not w.oneway:
            rev_id = f"r{sid}"
            rev_nodes = list(reversed(node_list))
            rev_edge = _build_edge(rev_id, rev_nodes, w)
            if rev_edge is not None:
                edges.append(rev_edge)
                for idx in range(len(rev_nodes) - 1):
                    directed_step_to_edge[(rev_nodes[idx], rev_nodes[idx + 1])] = rev_id

    outgoing = _dd(list)
    incoming = _dd(list)
    for edge in edges:
        outgoing[edge.from_node].append(edge.edge_id)
        incoming[edge.to_node].append(edge.edge_id)

    # 允许双向道路在同一节点上掉头。当前 Phase 4 路由会把 stop snap 到 OSM
    # 路径中的中间节点；映射回 SUMO chunk edge 后，可能出现 “反向进入该 chunk，
    # 再沿正向离开该 chunk” 的相邻 edge。若这里强行禁掉回头连接，SUMO 会在
    # route 载入阶段直接报 no valid route。
    edge_by_id = {edge.edge_id: edge for edge in edges}
    connections = []
    seen_connections = set()
    for node in used_j:
        for ie in incoming.get(node, []):
            for oe in outgoing.get(node, []):
                incoming_edge = edge_by_id[ie]
                outgoing_edge = edge_by_id[oe]
                if incoming_edge == outgoing_edge:
                    continue
                pair = (ie, oe)
                if pair in seen_connections:
                    continue
                seen_connections.add(pair)
                connections.append(pair)

    return SumoNetArtifacts(
        ox=ox,
        oy=oy,
        conv_boundary=conv_boundary,
        orig_bounds=orig_bounds,
        utm_nodes=utm,
        junction_nodes=tuple(sorted(used_j)),
        edges=tuple(edges),
        connections=tuple(connections),
        directed_step_to_edge=directed_step_to_edge,
    )


def write_sumo_net_artifacts(artifacts: SumoNetArtifacts, path: str) -> tuple[int, int]:
    """将中间结果写成 SUMO net.xml。"""

    conv_xmin, conv_ymin, conv_xmax, conv_ymax = artifacts.conv_boundary
    orig_bounds = artifacts.orig_bounds

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">',
        f'    <location netOffset="{-artifacts.ox:.2f},{-artifacts.oy:.2f}"',
        f'              convBoundary="{conv_xmin:.2f},{conv_ymin:.2f},{conv_xmax:.2f},{conv_ymax:.2f}"',
        f'              origBoundary="{orig_bounds[0]:.6f},{orig_bounds[1]:.6f},{orig_bounds[2]:.6f},{orig_bounds[3]:.6f}"',
        '              projParameter="+proj=utm +zone=51 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>',
    ]

    for edge in artifacts.edges:
        shape = " ".join(
            f"{x - artifacts.ox:.2f},{y - artifacts.oy:.2f}"
            for x, y in edge.shape_coords_utm
        )
        lines.append(
            f'    <edge id="{edge.edge_id}" from="n{edge.from_node}" to="n{edge.to_node}"'
            f' priority="{edge.priority}" numLanes="{edge.num_lanes}" speed="{edge.speed:.2f}">'
        )
        for lane_idx in range(edge.num_lanes):
            lines.append(
                f'        <lane id="{edge.edge_id}_{lane_idx}" index="{lane_idx}"'
                f' speed="{edge.speed:.2f}" length="{edge.length:.2f}" width="3.20"'
                f' shape="{shape}"/>'
            )
        lines.append("    </edge>")

    incoming_lanes = _dd(list)
    for edge in artifacts.edges:
        incoming_lanes[edge.to_node].extend(
            f"{edge.edge_id}_{lane_idx}" for lane_idx in range(edge.num_lanes)
        )

    for nid in artifacts.junction_nodes:
        x, y = artifacts.utm_nodes[nid]
        incoming = " ".join(incoming_lanes.get(nid, []))
        lines.append(
            f'    <junction id="n{nid}" type="priority" x="{x - artifacts.ox:.2f}"'
            f' y="{y - artifacts.oy:.2f}" incLanes="{incoming}" intLanes=""'
            f' shape="{x - artifacts.ox:.2f},{y - artifacts.oy:.2f}"/>'
        )

    for from_edge, to_edge in artifacts.connections:
        lines.append(
            f'    <connection from="{from_edge}" to="{to_edge}" fromLane="0"'
            f' toLane="0" dir="s" state="M"/>'
        )

    lines.append("</net>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    edge_count = len(artifacts.edges)
    node_count = len(artifacts.junction_nodes)
    print(f"[OSM→SUMO] {node_count} 节点, {edge_count} 条边 → {path}")
    return node_count, edge_count


def osm_to_sumo_net(osm_xml: str, orig_bounds: tuple, path: str):
    """
    解析 OSM XML，生成 SUMO net.xml。
    orig_bounds: (minx, miny, maxx, maxy) WGS84 经纬度，用于设置 netOffset。
    返回 (节点数, 边数)。
    """
    artifacts = build_sumo_net_artifacts(osm_xml, orig_bounds)
    return write_sumo_net_artifacts(artifacts, path)
