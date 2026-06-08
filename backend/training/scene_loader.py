#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 2 静态场景装载器。

职责边界：
  - 只负责读取并标准化 default_scene 的静态资产；
  - 输出训练/验证可直接复用的 TrainingSceneContext；
  - 不在此阶段决定 benchmark / poisson / hybrid 订单源模式。

实现原则：
  - 复用现有 EntityManager 与 Order 语义，不引入第二套实体命名；
  - 订单装载统一规范为训练侧内部对象；
  - 保留 Phase 4 需要的 OSM XML 文件路径，但不在此阶段加载 XML 全量内容。
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from core.entities.order import Order
from core.entities.primitives import Position3D, SourceType
from environment.state.entity_manager import EntityManager
from utils.coord_utils import wgs84_to_utm


DEFAULT_CONFIG_PATH = BACKEND_DIR / "config" / "rh_alns_cmrappo.yaml"


@dataclass(frozen=True)
class BenchmarkDynamicOrder:
    """
    benchmark 动态订单的训练侧标准化视图。

    说明：
      - `order` 提供统一的运行时 Order 对象；
      - `spawn_sim_s` / `deadline_offset_s` / `deadline_sim_s` 保留 benchmark
        回放所需的确定性字段，避免后续阶段退回裸字典。
    """

    order: Order
    spawn_sim_s: float
    deadline_offset_s: float | None
    deadline_sim_s: float | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingRoadNetwork:
    """训练侧路网引用。"""

    geojson: Mapping[str, Any]
    geojson_path: str
    xml_path: str | None
    fmt: str


@dataclass(frozen=True)
class TrainingSceneContext:
    """
    Phase 2 统一输出对象。

    注意：
      - `scene_id` / `scene_bundle_dir` 字段名与 TrainingRunMeta 严格对齐；
      - 这是运行期装载结果，不属于 Phase 1 的冻结契约。
    """

    scene_id: str
    scene_bundle_dir: str
    scene_config_path: str
    entities_json_path: str
    orders_json_path: str
    scene_config: Mapping[str, Any]
    bounds: Mapping[str, float]
    depots: Mapping[str, Any]
    stations: Mapping[str, Any]
    trucks: Mapping[str, Any]
    drones: Mapping[str, Any]
    static_orders: tuple[Order, ...]
    dynamic_orders: tuple[BenchmarkDynamicOrder, ...]
    entities_raw: Mapping[str, Any]
    orders_raw: Mapping[str, Any]
    road_network: TrainingRoadNetwork
    entity_manager: EntityManager

    def build_summary(self) -> dict[str, Any]:
        """返回 Phase 2 验收所需的核心摘要。"""

        return {
            "scene_id": self.scene_id,
            "scene_bundle_dir": self.scene_bundle_dir,
            "orders_json_path": self.orders_json_path,
            "depots": len(self.depots),
            "stations": len(self.stations),
            "trucks": len(self.trucks),
            "drones": len(self.drones),
            "static_orders": len(self.static_orders),
            "dynamic_orders": len(self.dynamic_orders),
            "osm_geojson_path": self.road_network.geojson_path,
            "osm_xml_path": self.road_network.xml_path,
        }


def load_default_scene(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TrainingSceneContext:
    """
    从 `rh_alns_cmrappo.yaml` 读取默认场景配置并装载。
    """

    cfg_path = _resolve_repo_path(config_path)
    cfg = _load_yaml_file(cfg_path)
    scene_cfg = cfg.get("scene")
    if not isinstance(scene_cfg, Mapping):
        raise ValueError(f"配置缺少 scene 段: {cfg_path}")

    bundle_dir = _resolve_repo_path(str(scene_cfg["scene_bundle_dir"]))
    expected_counts = {
        "depots": int(scene_cfg["expected_depot_count"]),
        "stations": int(scene_cfg["expected_station_count"]),
        "trucks": int(scene_cfg["expected_truck_count"]),
        "drones": int(scene_cfg["expected_drone_count"]),
    }

    return load_training_scene(
        scene_bundle_dir=bundle_dir,
        scene_config_file=str(scene_cfg["scene_config_file"]),
        entities_file=str(scene_cfg["entities_file"]),
        orders_file=str(scene_cfg["orders_file"]),
        osm_network_file=str(scene_cfg["osm_network_file"]),
        osm_network_format=str(scene_cfg.get("osm_network_format", "geojson")),
        expected_scene_id=str(scene_cfg["scene_id"]),
        expected_counts=expected_counts,
    )


def load_training_scene(
    *,
    scene_bundle_dir: str | Path,
    scene_config_file: str = "scene_config.json",
    entities_file: str = "entities.json",
    orders_file: str = "orders.json",
    osm_network_file: str = "osm_network.geojson",
    osm_network_format: str = "geojson",
    expected_scene_id: str | None = None,
    expected_counts: Mapping[str, int] | None = None,
) -> TrainingSceneContext:
    """
    装载训练侧静态场景资产，输出统一 TrainingSceneContext。
    """

    bundle_dir = _resolve_repo_path(scene_bundle_dir)
    scene_config_path = bundle_dir / scene_config_file
    entities_path = bundle_dir / entities_file
    orders_path = bundle_dir / orders_file
    osm_geojson_path = bundle_dir / osm_network_file

    _require_file(scene_config_path)
    _require_file(entities_path)
    _require_file(orders_path)
    _require_file(osm_geojson_path)

    osm_xml_candidate = bundle_dir / "osm_network.xml"
    osm_xml_path = str(osm_xml_candidate) if osm_xml_candidate.is_file() else None

    scene_config = _load_json_file(scene_config_path)
    entities_raw = _load_json_file(entities_path)
    orders_raw = _load_json_file(orders_path)
    osm_geojson = _load_json_file(osm_geojson_path)

    scene_id = str(scene_config.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError(f"scene_config.json 缺少 scene_id: {scene_config_path}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            "scene_id 不一致: "
            f"yaml={expected_scene_id}, scene_config.json={scene_id}"
        )

    entity_manager = EntityManager()
    entity_manager.load_from_config({"entities": entities_raw})

    static_orders = tuple(
        _build_static_order(entry) for entry in orders_raw.get("static_orders", [])
    )
    dynamic_orders = tuple(
        _build_dynamic_order(entry) for entry in orders_raw.get("dynamic_orders", [])
    )

    context = TrainingSceneContext(
        scene_id=scene_id,
        scene_bundle_dir=str(bundle_dir),
        scene_config_path=str(scene_config_path),
        entities_json_path=str(entities_path),
        orders_json_path=str(orders_path),
        scene_config=copy.deepcopy(scene_config),
        bounds=copy.deepcopy(scene_config.get("bounds", {})),
        depots=dict(entity_manager.depots),
        stations=dict(entity_manager.stations),
        trucks=dict(entity_manager.trucks),
        drones=dict(entity_manager.drones),
        static_orders=static_orders,
        dynamic_orders=dynamic_orders,
        entities_raw=copy.deepcopy(entities_raw),
        orders_raw=copy.deepcopy(orders_raw),
        road_network=TrainingRoadNetwork(
            geojson=copy.deepcopy(osm_geojson),
            geojson_path=str(osm_geojson_path),
            xml_path=osm_xml_path,
            fmt=osm_network_format,
        ),
        entity_manager=entity_manager,
    )

    _validate_expected_counts(context, expected_counts or {})
    return context


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须为对象: {path}")
    return data


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "缺少 PyYAML 依赖，无法读取 rh_alns_cmrappo.yaml"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须为对象: {path}")
    return data


def _build_static_order(entry: Mapping[str, Any]) -> Order:
    create_time = float(entry["create_time"])
    deadline = float(entry["deadline"])
    time_domain = entry.get("time_domain")
    if time_domain not in (None, "sim_s"):
        raise ValueError(
            f"静态订单 {entry.get('order_id', '?')} 仅支持 sim_s 时间域: {time_domain}"
        )

    return _build_order(
        entry=entry,
        create_time=create_time,
        deadline=deadline,
    )


def _build_dynamic_order(entry: Mapping[str, Any]) -> BenchmarkDynamicOrder:
    spawn_sim_s = float(entry["spawn_sim_s"])
    deadline_sim_s: float | None = None
    deadline_offset_s: float | None = None

    if "deadline_sim_s" in entry:
        deadline_sim_s = float(entry["deadline_sim_s"])
        deadline = deadline_sim_s
        deadline_offset_s = deadline_sim_s - spawn_sim_s
    else:
        deadline_offset_s = float(entry.get("deadline_offset_s", 900.0))
        deadline = spawn_sim_s + deadline_offset_s

    order = _build_order(
        entry=entry,
        create_time=spawn_sim_s,
        deadline=deadline,
    )
    return BenchmarkDynamicOrder(
        order=order,
        spawn_sim_s=spawn_sim_s,
        deadline_offset_s=deadline_offset_s,
        deadline_sim_s=deadline_sim_s,
        raw=copy.deepcopy(dict(entry)),
    )


def _build_order(
    *,
    entry: Mapping[str, Any],
    create_time: float,
    deadline: float,
) -> Order:
    order_id = str(entry["order_id"])
    lng = float(entry["delivery_lng"])
    lat = float(entry["delivery_lat"])
    x, y = wgs84_to_utm(lng, lat)
    delivery_loc = Position3D(
        x=x,
        y=y,
        z=float(entry.get("delivery_z", 0.0)),
    )

    source_type_raw = entry.get("source_type")
    source_type = (
        None if source_type_raw is None else SourceType(str(source_type_raw))
    )

    return Order(
        order_id=order_id,
        create_time=create_time,
        deadline=deadline,
        delivery_loc=delivery_loc,
        pickup_source_id=entry.get("pickup_source_id"),
        source_type=source_type,
        payload_weight=float(entry.get("payload_weight", 1.0)),
    )


def _validate_expected_counts(
    context: TrainingSceneContext,
    expected_counts: Mapping[str, int],
) -> None:
    actual = {
        "depots": len(context.depots),
        "stations": len(context.stations),
        "trucks": len(context.trucks),
        "drones": len(context.drones),
    }
    for key, expected_value in expected_counts.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"{key} 数量不匹配: expected={expected_value}, actual={actual_value}"
            )


__all__ = [
    "BenchmarkDynamicOrder",
    "TrainingRoadNetwork",
    "TrainingSceneContext",
    "load_default_scene",
    "load_training_scene",
]
