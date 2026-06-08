#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMO 网格路网生成器
纯 Python 生成覆盖选区的 SUMO grid net.xml，无需安装 SUMO。
坐标系与 no_fly_zones.add.xml 一致（EPSG:32651，以选区左下角为原点）。
"""

import math


def export_grid_net(
    ox: float,
    oy: float,
    width_m: float,
    height_m: float,
    grid_spacing: float,
    orig_bounds: tuple,
    path: str,
):
    """
    生成网格状 SUMO net.xml。
    ox, oy: UTM 坐标系下选区左下角的绝对坐标（用作 netOffset）。
    """
    nx    = int(math.ceil(width_m  / grid_spacing)) + 1
    ny    = int(math.ceil(height_m / grid_spacing)) + 1
    xs    = [min(i * grid_spacing, width_m)  for i in range(nx)]
    ys    = [min(j * grid_spacing, height_m) for j in range(ny)]
    speed = 13.89  # 50 km/h

    L = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">',
        f'    <location netOffset="{-ox:.2f},{-oy:.2f}"',
        f'              convBoundary="0.00,0.00,{width_m:.2f},{height_m:.2f}"',
        f'              origBoundary="{orig_bounds[0]:.6f},{orig_bounds[1]:.6f},'
        f'{orig_bounds[2]:.6f},{orig_bounds[3]:.6f}"',
        '              projParameter="+proj=utm +zone=51 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>',
    ]

    # ── 节点 ──
    for j in range(ny):
        for i in range(nx):
            L.append(f'    <node id="n{i}_{j}" x="{xs[i]:.2f}" y="{ys[j]:.2f}" type="priority"/>')

    edges = {}  # eid -> (from_node, to_node)

    def _add_edge(eid, fn, tn, sx0, sy0, sx1, sy1, ln):
        edges[eid] = (fn, tn)
        L += [
            f'    <edge id="{eid}" from="{fn}" to="{tn}" priority="2" numLanes="1" speed="{speed}">',
            f'        <lane id="{eid}_0" index="0" speed="{speed}" length="{ln:.2f}" width="3.20"'
            f' shape="{sx0:.2f},{sy0:.2f} {sx1:.2f},{sy1:.2f}"/>',
            "    </edge>",
        ]

    # ── 水平方向（双向） ──
    for j in range(ny):
        for i in range(nx - 1):
            ln = xs[i + 1] - xs[i]
            _add_edge(f"h{i}_{j}_f", f"n{i}_{j}",   f"n{i+1}_{j}", xs[i],   ys[j], xs[i+1], ys[j], ln)
            _add_edge(f"h{i}_{j}_b", f"n{i+1}_{j}", f"n{i}_{j}",   xs[i+1], ys[j], xs[i],   ys[j], ln)

    # ── 垂直方向（双向） ──
    for j in range(ny - 1):
        for i in range(nx):
            ln = ys[j + 1] - ys[j]
            _add_edge(f"v{i}_{j}_f", f"n{i}_{j}",   f"n{i}_{j+1}", xs[i], ys[j],   xs[i], ys[j+1], ln)
            _add_edge(f"v{i}_{j}_b", f"n{i}_{j+1}", f"n{i}_{j}",   xs[i], ys[j+1], xs[i], ys[j],   ln)

    # ── 连接（跳过 U 形掉头） ──
    incoming = {}
    outgoing = {}
    for eid, (fn, tn) in edges.items():
        outgoing.setdefault(fn, []).append(eid)
        incoming.setdefault(tn, []).append(eid)

    for node, in_edges in incoming.items():
        for ie in in_edges:
            ie_from = edges[ie][0]
            for oe in outgoing.get(node, []):
                if ie_from == edges[oe][1]:
                    continue
                L.append(f'    <connection from="{ie}" to="{oe}" fromLane="0" toLane="0"/>')

    L.append("</net>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"[EXPORT] grid.net.xml: {nx}×{ny} 节点, {len(edges)} 条边 → {path}")
