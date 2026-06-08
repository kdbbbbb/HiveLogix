#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a generated SUMO/OSM folder as a backend-readable scene bundle."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pyproj import Transformer


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from environment.geo.osm_service import osm_to_geojson  # noqa: E402
from environment.geo.osm_service import build_road_graph  # noqa: E402
from utils.coord_utils import wgs84_to_utm  # noqa: E402
from config.loader import load_drone_params  # noqa: E402


DEFAULT_SOURCE_TEMPLATE = BACKEND_DIR / "test_data" / "default_scene"
DEFAULT_ORDERS_FILE = "orders_ppo_training_intensity.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON top-level must be an object: {path}")
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _parse_area_osm_bounds(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    lons: list[float] = []
    lats: list[float] = []
    for node in root.iter("node"):
        lat = node.get("lat")
        lon = node.get("lon")
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))
    if not lons or not lats:
        raise ValueError(f"No OSM node coordinates found in {path}")
    return {
        "min_lng": min(lons),
        "max_lng": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats),
    }


def _parse_net_orig_boundary(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    location = root.find("location")
    if location is None:
        return None
    raw = location.get("origBoundary")
    if not raw:
        return None
    parts = [float(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid origBoundary in {path}: {raw}")
    min_lng, min_lat, max_lng, max_lat = parts
    return {
        "min_lng": min_lng,
        "max_lng": max_lng,
        "min_lat": min_lat,
        "max_lat": max_lat,
    }


def _center(bounds: dict[str, float]) -> dict[str, float]:
    return {
        "lng": (bounds["min_lng"] + bounds["max_lng"]) / 2.0,
        "lat": (bounds["min_lat"] + bounds["max_lat"]) / 2.0,
    }


def _scale_lng_lat(
    lng: float,
    lat: float,
    *,
    source_bounds: dict[str, float],
    target_bounds: dict[str, float],
) -> tuple[float, float]:
    sx = (lng - source_bounds["min_lng"]) / (
        source_bounds["max_lng"] - source_bounds["min_lng"]
    )
    sy = (lat - source_bounds["min_lat"]) / (
        source_bounds["max_lat"] - source_bounds["min_lat"]
    )
    target_lng = target_bounds["min_lng"] + sx * (
        target_bounds["max_lng"] - target_bounds["min_lng"]
    )
    target_lat = target_bounds["min_lat"] + sy * (
        target_bounds["max_lat"] - target_bounds["min_lat"]
    )
    return target_lng, target_lat


def _scale_entity_or_order_positions(
    payload: Any,
    *,
    source_bounds: dict[str, float],
    target_bounds: dict[str, float],
) -> Any:
    cloned = copy.deepcopy(payload)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "lng" in value and "lat" in value:
                value["lng"], value["lat"] = _scale_lng_lat(
                    float(value["lng"]),
                    float(value["lat"]),
                    source_bounds=source_bounds,
                    target_bounds=target_bounds,
                )
            if "delivery_lng" in value and "delivery_lat" in value:
                value["delivery_lng"], value["delivery_lat"] = _scale_lng_lat(
                    float(value["delivery_lng"]),
                    float(value["delivery_lat"]),
                    source_bounds=source_bounds,
                    target_bounds=target_bounds,
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(cloned)
    return cloned


def _convert_osm(scene_dir: Path) -> None:
    area_osm = scene_dir / "area.osm"
    if not area_osm.is_file():
        raise FileNotFoundError(f"Missing required source file: {area_osm}")

    osm_xml = area_osm.read_text(encoding="utf-8")
    shutil.copyfile(area_osm, scene_dir / "osm_network.xml")
    geojson = osm_to_geojson(osm_xml)
    _write_json(scene_dir / "osm_network.geojson", geojson)


def _parse_poly_origin(raw_xml: str, path: Path) -> tuple[float, float]:
    match = re.search(r"原点\(左下\):\s*E=([0-9.+-]+)\s+N=([0-9.+-]+)", raw_xml)
    if not match:
        raise ValueError(
            f"Cannot find SUMO poly origin in {path}; expected comment like "
            "'原点(左下): E=... N=...'"
        )
    return float(match.group(1)), float(match.group(2))


def _convert_sumo_polys(scene_dir: Path) -> None:
    poly_path = scene_dir / "no_fly_zones.add.xml"
    if not poly_path.is_file():
        return

    raw_xml = poly_path.read_text(encoding="utf-8")
    origin_e, origin_n = _parse_poly_origin(raw_xml, poly_path)
    transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    root = ET.fromstring(raw_xml)

    features: list[dict[str, Any]] = []
    for poly in root.iter("poly"):
        shape_raw = poly.get("shape", "").strip()
        if not shape_raw:
            continue
        coords: list[list[float]] = []
        for pair in shape_raw.split():
            x_raw, y_raw = pair.split(",", 1)
            lng, lat = transformer.transform(
                origin_e + float(x_raw),
                origin_n + float(y_raw),
            )
            coords.append([lng, lat])
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        height = float(poly.get("height", 0.0))
        is_no_fly = poly.get("type") == "no_fly_zone"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": poly.get("id", ""),
                    "type": poly.get("type", ""),
                    "h": height,
                    "height_m": height,
                    "nf": is_no_fly,
                    "no_fly_zone": is_no_fly,
                },
            }
        )

    full_collection = {"type": "FeatureCollection", "features": features}
    no_fly_collection = {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in features
            if feature.get("properties", {}).get("nf") is True
        ],
    }
    _write_json(scene_dir / "buildings.geojson", full_collection)
    _write_json(scene_dir / "no_fly_zones.geojson", no_fly_collection)


def _write_scene_config(
    *,
    scene_dir: Path,
    scene_id: str,
    bounds: dict[str, float],
    height_threshold: float | None,
) -> None:
    scene_config = {
        "scene_id": scene_id,
        "name": f"测试场景 - {scene_id}",
        "description": "Generated scene bundle for HiveLogix backend training/eval.",
        "region": "shanghai_pudong",
        "bounds": bounds,
        "center": _center(bounds),
        "road_network": {
            "source": "osm",
            "file": "osm_network.geojson",
            "bounds": bounds,
        },
        "buildings_file": "buildings.geojson",
        "entities_file": "entities.json",
        "orders_file": DEFAULT_ORDERS_FILE,
    }
    if height_threshold is not None:
        scene_config["height_threshold"] = height_threshold
        scene_config["threshold"] = height_threshold
    _write_json(scene_dir / "scene_config.json", scene_config)


def _extract_height_threshold(scene_dir: Path) -> float | None:
    readme = scene_dir / "README.txt"
    if readme.is_file():
        match = re.search(r"高度阈值:\s*([0-9.]+)\s*m", readme.read_text(encoding="utf-8"))
        if match:
            return float(match.group(1))
    poly_path = scene_dir / "no_fly_zones.add.xml"
    if poly_path.is_file():
        match = re.search(
            r"高度阈值:\s*([0-9.]+)\s*m",
            poly_path.read_text(encoding="utf-8"),
        )
        if match:
            return float(match.group(1))
    return None


def _write_scaled_templates(
    *,
    scene_dir: Path,
    source_template: Path,
    target_bounds: dict[str, float],
) -> None:
    source_config = _load_json(source_template / "scene_config.json")
    source_bounds = dict(source_config["bounds"])

    entities = _load_json(source_template / "entities.json")
    scaled_entities = _scale_entity_or_order_positions(
        entities,
        source_bounds=source_bounds,
        target_bounds=target_bounds,
    )
    _write_json(scene_dir / "entities.json", scaled_entities)

    orders_path = source_template / DEFAULT_ORDERS_FILE
    orders = _load_json(orders_path)
    scaled_orders = _scale_entity_or_order_positions(
        orders,
        source_bounds=source_bounds,
        target_bounds=target_bounds,
    )
    _write_json(scene_dir / DEFAULT_ORDERS_FILE, scaled_orders)
    _snap_truck_only_orders_to_road_network(scene_dir / DEFAULT_ORDERS_FILE, scene_dir)


def _snap_truck_only_orders_to_road_network(orders_path: Path, scene_dir: Path) -> None:
    """Move truck-only order locations to reachable OSM road nodes."""

    if not orders_path.is_file():
        raise FileNotFoundError(f"Orders file does not exist: {orders_path}")
    osm_xml_path = scene_dir / "osm_network.xml"
    if not osm_xml_path.is_file():
        raise FileNotFoundError(f"OSM XML file does not exist: {osm_xml_path}")
    entities_path = scene_dir / "entities.json"
    if not entities_path.is_file():
        raise FileNotFoundError(f"Entities file does not exist: {entities_path}")

    heavy_payload_capacity = float(load_drone_params().heavy.payload_capacity)
    orders = _load_json(orders_path)
    entities = _load_json(entities_path)
    depots = entities.get("depots", [])
    if not isinstance(depots, list) or not depots:
        raise ValueError(f"entities.json missing depots: {entities_path}")

    depot = depots[0]
    depot_x, depot_y = wgs84_to_utm(float(depot["lng"]), float(depot["lat"]))
    road_graph, road_nodes = build_road_graph(
        osm_xml_path.read_text(encoding="utf-8"),
        respect_osm_oneway=True,
    )
    if len(road_graph.nodes) == 0:
        raise ValueError(f"OSM road graph is empty: {osm_xml_path}")

    component_nodes = _reachable_component_containing_point(
        road_graph=road_graph,
        road_nodes=road_nodes,
        point_x=depot_x,
        point_y=depot_y,
    )
    if not component_nodes:
        raise ValueError("No reachable OSM component found for depot")
    component_node_positions = {
        node_id: wgs84_to_utm(float(road_nodes[node_id][0]), float(road_nodes[node_id][1]))
        for node_id in component_nodes
        if node_id in road_nodes
    }
    if not component_node_positions:
        raise ValueError("Reachable OSM component has no coordinates")

    changed = 0
    for section in ("static_orders", "dynamic_orders"):
        items = orders.get(section, [])
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if float(entry.get("payload_weight", 0.0)) <= heavy_payload_capacity:
                continue
            order_x, order_y = wgs84_to_utm(
                float(entry["delivery_lng"]),
                float(entry["delivery_lat"]),
            )
            nearest_node = min(
                component_node_positions,
                key=lambda node_id: (
                    (component_node_positions[node_id][0] - order_x) ** 2
                    + (component_node_positions[node_id][1] - order_y) ** 2
                ),
            )
            nearest_lng, nearest_lat = road_nodes[nearest_node]
            if (
                float(entry["delivery_lng"]) != float(nearest_lng)
                or float(entry["delivery_lat"]) != float(nearest_lat)
            ):
                entry["delivery_lng"] = float(nearest_lng)
                entry["delivery_lat"] = float(nearest_lat)
                changed += 1

    if changed:
        _write_json(orders_path, orders)


def _reachable_component_containing_point(
    *,
    road_graph: Any,
    road_nodes: dict[str, tuple[float, float]],
    point_x: float,
    point_y: float,
) -> set[str]:
    try:
        import networkx as nx  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("networkx is required to find reachable road components") from exc

    nearest_node = _nearest_road_node(
        road_graph=road_graph,
        road_nodes=road_nodes,
        point_x=point_x,
        point_y=point_y,
    )
    if not nearest_node:
        return set()
    for component in nx.strongly_connected_components(road_graph):
        component_ids = {str(node_id) for node_id in component}
        if nearest_node in component_ids:
            return component_ids
    return {str(node_id) for node_id in max(nx.strongly_connected_components(road_graph), key=len)}


def _nearest_road_node(
    *,
    road_graph: Any,
    road_nodes: dict[str, tuple[float, float]],
    point_x: float,
    point_y: float,
) -> str:
    best_node = ""
    best_dist = float("inf")
    for node_id in road_graph.nodes:
        node_key = str(node_id)
        if node_key not in road_nodes:
            continue
        node_x, node_y = wgs84_to_utm(float(road_nodes[node_key][0]), float(road_nodes[node_key][1]))
        dist = (node_x - point_x) ** 2 + (node_y - point_y) ** 2
        if dist < best_dist:
            best_node = node_key
            best_dist = dist
    return best_node


def prepare_scene_bundle(
    *,
    scene_dir: Path,
    scene_id: str,
    source_template: Path,
) -> None:
    scene_dir = scene_dir.resolve()
    source_template = source_template.resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    if not source_template.is_dir():
        raise FileNotFoundError(f"Template scene directory does not exist: {source_template}")

    _convert_osm(scene_dir)
    _convert_sumo_polys(scene_dir)

    bounds = _parse_net_orig_boundary(scene_dir / "roads.net.xml")
    if bounds is None:
        bounds = _parse_area_osm_bounds(scene_dir / "area.osm")

    _write_scene_config(
        scene_dir=scene_dir,
        scene_id=scene_id,
        bounds=bounds,
        height_threshold=_extract_height_threshold(scene_dir),
    )
    _write_scaled_templates(
        scene_dir=scene_dir,
        source_template=source_template,
        target_bounds=bounds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_dir",
        nargs="?",
        default=str(BACKEND_DIR / "test_data" / "scene-2"),
        help="Directory containing area.osm, roads.net.xml and no_fly_zones.add.xml.",
    )
    parser.add_argument("--scene-id", default="scene_2_7x7")
    parser.add_argument(
        "--source-template",
        default=str(DEFAULT_SOURCE_TEMPLATE),
        help="Scene bundle whose entities/orders are scaled into the target bounds.",
    )
    args = parser.parse_args()

    prepare_scene_bundle(
        scene_dir=Path(args.scene_dir),
        scene_id=str(args.scene_id),
        source_template=Path(args.source_template),
    )
    print(f"[OK] Prepared backend scene bundle: {Path(args.scene_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
