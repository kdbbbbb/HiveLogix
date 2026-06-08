#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 3 订单源适配器。

职责边界：
  - 统一 `benchmark / poisson / hybrid` 三种订单源模式；
  - 复用 `OrderManager` 现有的泊松生成与 benchmark 回放语义；
  - 输出可直接写入 `meta.json` 的 `TrainingInputMeta` 快照；
  - 为后续 `train_cmrappo.py` / `validate_benchmark.py` /
    `validate_stochastic.py` 提供稳定入口。
"""

from __future__ import annotations

import copy
import hashlib
import random
import sys
from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from core.entities.order import Order
from config.loader import load_drone_params
from environment.geo.osm_service import build_road_graph, find_nearest_node
from environment.state.order_manager import OrderManager

from .contracts import BenchmarkMeta, TrainingInputMeta
from .scene_loader import (
    DEFAULT_CONFIG_PATH,
    BenchmarkDynamicOrder,
    TrainingSceneContext,
)


class OrderSourceMode(StrEnum):
    """运行时订单注入模式。"""

    BENCHMARK = "benchmark"
    POISSON = "poisson"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class PoissonOrderGenConfig:
    """与 `OrderManager.configure()` 对齐的泊松订单生成配置。"""

    arrival_rate: float
    weight_min_kg: float
    weight_max_kg: float
    geo_mode: str
    burst_enabled: bool
    burst_multiplier: float
    max_orders_per_episode: int
    window_min_min: int
    window_max_min: int
    weight_bands: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate 不能为负数")
        if self.weight_min_kg <= 0:
            raise ValueError("weight_min_kg 必须为正数")
        if self.weight_max_kg < self.weight_min_kg:
            raise ValueError("weight_max_kg 不能小于 weight_min_kg")
        _validate_weight_bands(
            self.weight_bands,
            weight_min_kg=self.weight_min_kg,
            weight_max_kg=self.weight_max_kg,
        )
        if self.burst_multiplier <= 0:
            raise ValueError("burst_multiplier 必须为正数")
        if self.max_orders_per_episode <= 0:
            raise ValueError("max_orders_per_episode 必须为正数")
        if self.window_min_min <= 0:
            raise ValueError("window_min_min 必须为正数")
        if self.window_max_min < self.window_min_min:
            raise ValueError("window_max_min 不能小于 window_min_min")

    def to_order_manager_config(self) -> dict[str, Any]:
        """转换为 `OrderManager.configure()` 所需配置字典。"""

        config = {
            "arrival_rate": self.arrival_rate,
            "weight_min": self.weight_min_kg,
            "weight_max": self.weight_max_kg,
            "geo_mode": self.geo_mode,
            "burst_enabled": self.burst_enabled,
            "burst_multiplier": self.burst_multiplier,
            "max_orders": self.max_orders_per_episode,
            "window_min": self.window_min_min,
            "window_max": self.window_max_min,
        }
        if self.weight_bands:
            config["weight_bands"] = [dict(band) for band in self.weight_bands]
        return config


@dataclass(frozen=True)
class OrderSourceConfig:
    """
    统一订单源输出。

    说明：
      - `background_static_orders` 始终保留 `static_orders`，供卡车骨架 /
        mode A 背景订单使用；
      - `initial_static_uav_orders` 在 benchmark / poisson / hybrid episode
        起点注入 UAV-capable static orders，使训练重开一轮时与卡车静态
        背景订单处在同一初始场景里；
      - `scheduled_dynamic_orders` 包含 benchmark/hybrid 回放动态单，以及
        poisson/hybrid 下显式加入的 truck-only 动态扰动单；
      - `poisson_gen_config` 始终存在；benchmark 模式通过 `arrival_rate=0`
        显式关闭泊松流，而不是省略配置。
    """

    mode: OrderSourceMode
    background_static_orders: tuple[Order, ...]
    scheduled_dynamic_orders: tuple[Mapping[str, Any], ...]
    poisson_gen_config: PoissonOrderGenConfig
    arrival_rate: float
    seed: int
    benchmark: BenchmarkMeta
    training_input_meta: TrainingInputMeta
    initial_static_uav_orders: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple
    )

    @property
    def has_scheduled_dynamic_orders(self) -> bool:
        return bool(self.scheduled_dynamic_orders)

    @property
    def has_poisson_stream(self) -> bool:
        return self.arrival_rate > 0.0

    def build_summary(self) -> dict[str, Any]:
        """返回可直接打印的配置摘要。"""

        return {
            "mode": self.mode.value,
            "background_static_orders": len(self.background_static_orders),
            "initial_static_uav_orders": len(self.initial_static_uav_orders),
            "scheduled_dynamic_orders": len(self.scheduled_dynamic_orders),
            "arrival_rate": self.arrival_rate,
            "seed": self.seed,
            "poisson_gen_config": self.poisson_gen_config.to_order_manager_config(),
            "benchmark": asdict(self.benchmark),
            "training_input_meta": asdict(self.training_input_meta),
        }


def build_order_source(
    scene_ctx: TrainingSceneContext,
    mode: str | OrderSourceMode | None = None,
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> OrderSourceConfig:
    """
    构建统一订单源配置。

    Args:
        scene_ctx:   Phase 2 输出的静态场景上下文
        mode:        订单源模式；为空时回落到 yaml `data.order_source_mode`
        seed:        显式随机种子；为空时根据模式选择 yaml 默认值
        overrides:   对 yaml 字段的局部覆盖，键名使用配置字段名即可
        config_path: 训练配置文件路径
    """

    cfg = _load_yaml_file(_resolve_repo_path(config_path))
    scene_cfg = _require_mapping(cfg, "scene")
    data_cfg = _require_mapping(cfg, "data")
    training_cfg = _require_mapping(cfg, "training")
    overrides = dict(overrides or {})

    selected_mode = _coerce_mode(
        mode if mode is not None else overrides.get("order_source_mode", data_cfg["order_source_mode"])
    )
    selected_seed = _resolve_seed(selected_mode, data_cfg, seed, overrides)

    benchmark_use_dynamic_orders = bool(
        overrides.get(
            "benchmark_use_dynamic_orders",
            data_cfg.get("benchmark_use_dynamic_orders", True),
        )
    )
    hybrid_background_dynamic_orders = bool(
        overrides.get(
            "hybrid_background_dynamic_orders",
            data_cfg.get("hybrid_background_dynamic_orders", True),
        )
    )

    order_window_min_min = int(
        overrides.get(
            "order_window_min_min",
            scene_cfg["order_window_min_min"],
        )
    )
    order_window_max_min = int(
        overrides.get(
            "order_window_max_min",
            scene_cfg["order_window_max_min"],
        )
    )

    arrival_rate = _resolve_arrival_rate(selected_mode, data_cfg, overrides)
    poisson_gen_config = PoissonOrderGenConfig(
        arrival_rate=arrival_rate,
        weight_min_kg=float(
            overrides.get(
                "poisson_weight_min_kg",
                data_cfg["poisson_weight_min_kg"],
            )
        ),
        weight_max_kg=float(
            overrides.get(
                "poisson_weight_max_kg",
                data_cfg["poisson_weight_max_kg"],
            )
        ),
        geo_mode=str(
            overrides.get(
                "poisson_geo_mode",
                data_cfg.get("poisson_geo_mode", "bbox_uniform"),
            )
        ),
        burst_enabled=bool(
            overrides.get(
                "poisson_burst_enabled",
                data_cfg.get("poisson_burst_enabled", False),
            )
        ),
        burst_multiplier=float(
            overrides.get(
                "poisson_burst_multiplier",
                data_cfg.get("poisson_burst_multiplier", 1.0),
            )
        ),
        max_orders_per_episode=int(
            overrides.get(
                "poisson_max_orders_per_episode",
                data_cfg["poisson_max_orders_per_episode"],
            )
        ),
        window_min_min=order_window_min_min,
        window_max_min=order_window_max_min,
        weight_bands=_resolve_weight_bands(data_cfg, overrides),
    )

    scheduled_dynamic_orders = _build_scheduled_dynamic_orders(
        scene_ctx=scene_ctx,
        mode=selected_mode,
        benchmark_use_dynamic_orders=benchmark_use_dynamic_orders,
        hybrid_background_dynamic_orders=hybrid_background_dynamic_orders,
        data_cfg=data_cfg,
        overrides=overrides,
        seed=selected_seed,
        upper_horizon_sec=float(_require_mapping(cfg, "planner")["upper_horizon_sec"]),
    )
    initial_static_uav_orders = _build_initial_static_uav_orders(
        scene_ctx=scene_ctx,
        mode=selected_mode,
    )

    benchmark = _build_benchmark_meta(
        scene_ctx=scene_ctx,
        benchmark_use_dynamic_orders=benchmark_use_dynamic_orders,
    )
    training_input_meta = TrainingInputMeta(
        order_source_mode=selected_mode.value,
        benchmark=benchmark,
        poisson_arrival_rate=poisson_gen_config.arrival_rate,
        poisson_weight_max_kg=poisson_gen_config.weight_max_kg,
        order_window_min_min=poisson_gen_config.window_min_min,
        order_window_max_min=poisson_gen_config.window_max_min,
        poisson_seed=selected_seed,
        training_seed=int(training_cfg["training_seed"]),
        total_timesteps=int(training_cfg["total_timesteps"]),
    )

    return OrderSourceConfig(
        mode=selected_mode,
        background_static_orders=tuple(scene_ctx.static_orders),
        scheduled_dynamic_orders=scheduled_dynamic_orders,
        poisson_gen_config=poisson_gen_config,
        arrival_rate=poisson_gen_config.arrival_rate,
        seed=selected_seed,
        benchmark=benchmark,
        training_input_meta=training_input_meta,
        initial_static_uav_orders=initial_static_uav_orders,
    )


def configure_order_manager_for_source(
    manager: OrderManager,
    scene_ctx: TrainingSceneContext,
    order_source: OrderSourceConfig,
) -> None:
    """将统一订单源配置注入 `OrderManager`。"""

    manager.configure(
        order_source.poisson_gen_config.to_order_manager_config(),
        dict(scene_ctx.bounds),
    )
    manager.set_scheduled_dynamic_orders(
        [
            dict(entry)
            for entry in (
                tuple(order_source.initial_static_uav_orders)
                + tuple(order_source.scheduled_dynamic_orders)
            )
        ]
    )


def preview_dynamic_order_stream(
    scene_ctx: TrainingSceneContext,
    order_source: OrderSourceConfig,
    *,
    horizon_sec: float,
    tick_step_sec: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    """
    在不污染全局随机态的前提下，预览订单源在给定时域内注入的动态订单流。

    该函数直接复用 `OrderManager.tick()` 与 scheduled replay 逻辑，
    用于 Phase 3 自测和后续离线验证基线构建。
    """

    if horizon_sec < 0:
        raise ValueError("horizon_sec 不能为负数")
    if tick_step_sec <= 0:
        raise ValueError("tick_step_sec 必须为正数")

    manager = OrderManager()
    configure_order_manager_for_source(manager, scene_ctx, order_source)

    random_state = random.getstate()
    try:
        random.seed(order_source.seed)
        current_time = 0.0
        while current_time <= horizon_sec:
            manager.tick(current_time, scene_ctx.entity_manager)
            current_time += tick_step_sec
    finally:
        random.setstate(random_state)

    orders = list(manager.pending_orders.values())
    orders.extend(manager.assigned_orders.values())
    orders.extend(manager.completed_orders)
    orders.sort(key=lambda order: (order.create_time, order.order_id))

    return tuple(_snapshot_order(order) for order in orders)


def build_order_source_preview_summary(
    scene_ctx: TrainingSceneContext,
    order_source: OrderSourceConfig,
    *,
    horizon_sec: float,
    preview_limit: int = 5,
) -> dict[str, Any]:
    """输出包含预览订单流的摘要，便于脚本打印。"""

    preview_orders = preview_dynamic_order_stream(
        scene_ctx,
        order_source,
        horizon_sec=horizon_sec,
    )
    return {
        **order_source.build_summary(),
        "preview_horizon_sec": horizon_sec,
        "preview_dynamic_order_count": len(preview_orders),
        "preview_orders_head": list(preview_orders[:preview_limit]),
    }


def ensure_mode_allowed(
    mode: str | OrderSourceMode,
    allowed_modes: tuple[str, ...] | list[str],
    consumer_name: str,
) -> OrderSourceMode:
    """供训练/验证入口复用的模式白名单校验。"""

    normalized_mode = _coerce_mode(mode)
    normalized_allowed = {_coerce_mode(item) for item in allowed_modes}
    if normalized_mode not in normalized_allowed:
        allowed_values = ", ".join(sorted(item.value for item in normalized_allowed))
        raise ValueError(
            f"{consumer_name} 不允许订单源模式 {normalized_mode.value}，"
            f"仅允许: {allowed_values}"
        )
    return normalized_mode


def _snapshot_order(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "create_time": round(float(order.create_time), 6),
        "deadline": round(float(order.deadline), 6),
        "payload_weight": round(float(order.payload_weight), 6),
        "delivery_x": round(float(order.delivery_loc.x), 6),
        "delivery_y": round(float(order.delivery_loc.y), 6),
    }


def _build_benchmark_meta(
    *,
    scene_ctx: TrainingSceneContext,
    benchmark_use_dynamic_orders: bool,
) -> BenchmarkMeta:
    orders_path = Path(scene_ctx.orders_json_path)
    _require_file(orders_path)
    return BenchmarkMeta(
        orders_json=str(orders_path),
        orders_json_sha256=_sha256_file(orders_path),
        static_order_count=len(scene_ctx.static_orders),
        dynamic_order_count=len(scene_ctx.dynamic_orders),
        benchmark_use_dynamic_orders=benchmark_use_dynamic_orders,
    )


def _build_scheduled_dynamic_orders(
    *,
    scene_ctx: TrainingSceneContext,
    mode: OrderSourceMode,
    benchmark_use_dynamic_orders: bool,
    hybrid_background_dynamic_orders: bool,
    data_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any],
    seed: int,
    upper_horizon_sec: float,
) -> tuple[Mapping[str, Any], ...]:
    scheduled: list[Mapping[str, Any]] = []
    if mode != OrderSourceMode.POISSON:
        if mode == OrderSourceMode.BENCHMARK and benchmark_use_dynamic_orders:
            scheduled.extend(
                _normalize_scheduled_dynamic_entry(dynamic_order)
                for dynamic_order in scene_ctx.dynamic_orders
            )
        elif mode == OrderSourceMode.HYBRID and hybrid_background_dynamic_orders:
            scheduled.extend(
                _normalize_scheduled_dynamic_entry(dynamic_order)
                for dynamic_order in scene_ctx.dynamic_orders
            )

    scheduled.extend(
        _build_truck_only_dynamic_orders(
            scene_ctx=scene_ctx,
            mode=mode,
            data_cfg=data_cfg,
            overrides=overrides,
            seed=seed,
            upper_horizon_sec=upper_horizon_sec,
        )
    )
    return tuple(
        sorted(
            scheduled,
            key=lambda item: (float(item.get("spawn_sim_s", 0.0)), str(item.get("order_id", ""))),
        )
    )


def _build_initial_static_uav_orders(
    *,
    scene_ctx: TrainingSceneContext,
    mode: OrderSourceMode,
) -> tuple[Mapping[str, Any], ...]:
    if mode not in {
        OrderSourceMode.BENCHMARK,
        OrderSourceMode.POISSON,
        OrderSourceMode.HYBRID,
    }:
        return ()
    heavy_payload_capacity = float(load_drone_params().heavy.payload_capacity)
    entries = [
        _normalize_initial_static_uav_entry(order)
        for order in scene_ctx.static_orders
        if float(order.payload_weight) <= heavy_payload_capacity
    ]
    return tuple(
        sorted(
            entries,
            key=lambda item: (float(item.get("spawn_sim_s", 0.0)), str(item.get("order_id", ""))),
        )
    )


def _build_truck_only_dynamic_orders(
    *,
    scene_ctx: TrainingSceneContext,
    mode: OrderSourceMode,
    data_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any],
    seed: int,
    upper_horizon_sec: float,
) -> tuple[Mapping[str, Any], ...]:
    """为 PPO 随机动态流生成少量 truck-only 预定义订单。"""

    if mode not in {OrderSourceMode.POISSON, OrderSourceMode.HYBRID}:
        return ()

    enabled = bool(
        overrides.get(
            "truck_only_dynamic_enabled",
            data_cfg.get("truck_only_dynamic_enabled", False),
        )
    )
    if not enabled:
        return ()

    count = int(
        overrides.get(
            "truck_only_dynamic_orders_per_episode",
            data_cfg.get("truck_only_dynamic_orders_per_episode", 0),
        )
    )
    if count <= 0:
        return ()

    bounds = scene_ctx.bounds
    required_bounds = ("min_lng", "max_lng", "min_lat", "max_lat")
    missing_bounds = [key for key in required_bounds if key not in bounds]
    if missing_bounds:
        raise ValueError(
            "truck-only 动态订单需要 scene bounds 字段: "
            + ", ".join(missing_bounds)
        )

    heavy_payload_capacity = float(load_drone_params().heavy.payload_capacity)
    weight_min = float(
        overrides.get(
            "truck_only_dynamic_weight_min_kg",
            data_cfg.get(
                "truck_only_dynamic_weight_min_kg",
                heavy_payload_capacity + 0.5,
            ),
        )
    )
    weight_max = float(
        overrides.get(
            "truck_only_dynamic_weight_max_kg",
            data_cfg.get(
                "truck_only_dynamic_weight_max_kg",
                max(weight_min, heavy_payload_capacity + 2.0),
            ),
        )
    )
    if weight_min <= heavy_payload_capacity:
        raise ValueError(
            "truck_only_dynamic_weight_min_kg 必须大于 heavy UAV 承载上限 "
            f"{heavy_payload_capacity:.3f}kg，当前为 {weight_min:.3f}kg"
        )
    if weight_max < weight_min:
        raise ValueError("truck_only_dynamic_weight_max_kg 不能小于 min")

    spawn_start = float(
        overrides.get(
            "truck_only_dynamic_spawn_start_s",
            data_cfg.get("truck_only_dynamic_spawn_start_s", 0.15 * upper_horizon_sec),
        )
    )
    spawn_end = float(
        overrides.get(
            "truck_only_dynamic_spawn_end_s",
            data_cfg.get("truck_only_dynamic_spawn_end_s", 0.70 * upper_horizon_sec),
        )
    )
    if spawn_start < 0.0:
        raise ValueError("truck_only_dynamic_spawn_start_s 不能为负数")
    if spawn_end < spawn_start:
        raise ValueError("truck_only_dynamic_spawn_end_s 不能小于 start")

    deadline_min_min = float(
        overrides.get(
            "truck_only_dynamic_deadline_window_min_min",
            data_cfg.get("truck_only_dynamic_deadline_window_min_min", 80),
        )
    )
    deadline_max_min = float(
        overrides.get(
            "truck_only_dynamic_deadline_window_max_min",
            data_cfg.get("truck_only_dynamic_deadline_window_max_min", 120),
        )
    )
    if deadline_min_min <= 0.0:
        raise ValueError("truck_only_dynamic_deadline_window_min_min 必须为正数")
    if deadline_max_min < deadline_min_min:
        raise ValueError("truck_only_dynamic_deadline_window_max_min 不能小于 min")

    rng = random.Random(int(seed) ^ 0xC0FFEE)
    reachable_delivery_nodes = _truck_reachable_delivery_nodes(scene_ctx)
    entries: list[Mapping[str, Any]] = []
    for idx in range(count):
        if count == 1:
            spawn_t = spawn_start
        else:
            ratio = idx / float(count - 1)
            base_spawn_t = spawn_start + (spawn_end - spawn_start) * ratio
            jitter_span = max(0.0, (spawn_end - spawn_start) / max(1, count - 1) * 0.20)
            spawn_t = min(
                spawn_end,
                max(spawn_start, base_spawn_t + rng.uniform(-jitter_span, jitter_span)),
            )
        deadline_offset_s = rng.uniform(deadline_min_min, deadline_max_min) * 60.0
        delivery_lng, delivery_lat = rng.choice(reachable_delivery_nodes)
        entries.append(
            {
                "order_id": f"TRUCKONLY-{int(seed)}-{idx + 1:02d}",
                "spawn_sim_s": float(round(spawn_t, 6)),
                "deadline_offset_s": float(round(deadline_offset_s, 6)),
                "payload_weight": float(round(rng.uniform(weight_min, weight_max), 6)),
                "delivery_lng": float(delivery_lng),
                "delivery_lat": float(delivery_lat),
                "delivery_z": 0.0,
                "pickup_source_id": None,
                "source_type": None,
            }
        )
    return tuple(entries)


def _truck_reachable_delivery_nodes(
    scene_ctx: TrainingSceneContext,
) -> tuple[tuple[float, float], ...]:
    """返回 depot 所在强连通路网分量内的 WGS84 节点坐标，供 truck-only 订单采样。"""

    xml_path = scene_ctx.road_network.xml_path
    if not xml_path:
        raise ValueError("truck-only 动态订单需要 scene_ctx.road_network.xml_path")

    try:
        import networkx as nx  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 networkx，无法按路网生成 truck-only 动态订单") from exc

    road_graph, road_nodes = build_road_graph(
        Path(xml_path).read_text(encoding="utf-8"),
        respect_osm_oneway=True,
    )
    if len(road_graph.nodes) == 0:
        raise ValueError("truck-only 动态订单需要非空 OSM 路网")

    depot_nodes: list[str] = []
    for depot in scene_ctx.depots.values():
        location = depot.get_location(0.0)
        nearest = find_nearest_node(road_graph, road_nodes, location.x, location.y)
        if nearest:
            depot_nodes.append(str(nearest))
    if not depot_nodes:
        raise ValueError("无法将 depot 映射到 OSM 路网节点")

    selected_component: set[str] | None = None
    for component in nx.strongly_connected_components(road_graph):
        component_ids = {str(node_id) for node_id in component}
        if any(node_id in component_ids for node_id in depot_nodes):
            selected_component = component_ids
            break
    if selected_component is None:
        selected_component = {
            str(node_id)
            for node_id in max(nx.strongly_connected_components(road_graph), key=len)
        }

    delivery_nodes = tuple(
        (float(road_nodes[node_id][0]), float(road_nodes[node_id][1]))
        for node_id in sorted(selected_component)
        if node_id in road_nodes
    )
    if not delivery_nodes:
        raise ValueError("depot 所在 OSM 强连通分量没有可采样节点")
    return delivery_nodes


def _normalize_scheduled_dynamic_entry(
    dynamic_order: BenchmarkDynamicOrder,
) -> Mapping[str, Any]:
    raw = copy.deepcopy(dict(dynamic_order.raw))
    raw["order_id"] = dynamic_order.order.order_id
    raw["spawn_sim_s"] = dynamic_order.spawn_sim_s
    raw["payload_weight"] = dynamic_order.order.payload_weight
    raw["pickup_source_id"] = dynamic_order.order.pickup_source_id
    raw["source_type"] = (
        None
        if dynamic_order.order.source_type is None
        else str(dynamic_order.order.source_type.value)
    )
    if "delivery_lng" not in raw or "delivery_lat" not in raw:
        delivery_lng, delivery_lat = dynamic_order.order.delivery_loc.to_wgs84()
        raw["delivery_lng"] = float(delivery_lng)
        raw["delivery_lat"] = float(delivery_lat)
    else:
        raw["delivery_lng"] = float(raw["delivery_lng"])
        raw["delivery_lat"] = float(raw["delivery_lat"])
    raw["delivery_z"] = float(raw.get("delivery_z", dynamic_order.order.delivery_loc.z))

    if dynamic_order.deadline_sim_s is not None:
        raw["deadline_sim_s"] = dynamic_order.deadline_sim_s
        raw.pop("deadline_offset_s", None)
    else:
        raw["deadline_offset_s"] = float(dynamic_order.deadline_offset_s or 0.0)
        raw.pop("deadline_sim_s", None)
    return raw


def _normalize_initial_static_uav_entry(order: Order) -> Mapping[str, Any]:
    delivery_lng, delivery_lat = order.delivery_loc.to_wgs84()
    return {
        "order_id": str(order.order_id),
        "spawn_sim_s": 0.0,
        "deadline_sim_s": float(order.deadline),
        "payload_weight": float(order.payload_weight),
        "delivery_lng": float(delivery_lng),
        "delivery_lat": float(delivery_lat),
        "delivery_z": float(order.delivery_loc.z),
        "pickup_source_id": order.pickup_source_id,
        "source_type": (
            None
            if order.source_type is None
            else str(order.source_type.value)
        ),
    }


def _resolve_weight_bands(
    data_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw = overrides.get("poisson_weight_bands", data_cfg.get("poisson_weight_bands", ()))
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("poisson_weight_bands 必须为列表")

    bands: list[Mapping[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"poisson_weight_bands[{idx}] 必须为对象")
        if "weight_min_kg" not in item or "weight_max_kg" not in item:
            raise ValueError(
                f"poisson_weight_bands[{idx}] 必须包含 weight_min_kg / weight_max_kg"
            )
        if "probability" not in item:
            raise ValueError(f"poisson_weight_bands[{idx}] 必须包含 probability")

        band: dict[str, Any] = {
            "weight_min": float(item["weight_min_kg"]),
            "weight_max": float(item["weight_max_kg"]),
            "probability": float(item["probability"]),
        }
        if "name" in item:
            band["name"] = str(item["name"])
        bands.append(band)
    return tuple(bands)


def _validate_weight_bands(
    bands: tuple[Mapping[str, Any], ...],
    *,
    weight_min_kg: float,
    weight_max_kg: float,
) -> None:
    if not bands:
        return

    probability_sum = 0.0
    for idx, band in enumerate(bands, start=1):
        band_min = float(band["weight_min"])
        band_max = float(band["weight_max"])
        probability = float(band["probability"])
        if probability <= 0.0:
            raise ValueError(f"poisson_weight_bands[{idx}].probability 必须为正数")
        if band_min <= 0.0:
            raise ValueError(f"poisson_weight_bands[{idx}].weight_min_kg 必须为正数")
        if band_max < band_min:
            raise ValueError(
                f"poisson_weight_bands[{idx}].weight_max_kg 不能小于 weight_min_kg"
            )
        if band_min < weight_min_kg or band_max > weight_max_kg:
            raise ValueError(
                "poisson_weight_bands 必须落在 "
                f"[{weight_min_kg}, {weight_max_kg}] kg 范围内"
            )
        probability_sum += probability

    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(
            "poisson_weight_bands.probability 之和必须等于 1.0，"
            f"当前为 {probability_sum:.6f}"
        )


def _resolve_arrival_rate(
    mode: OrderSourceMode,
    data_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> float:
    if mode == OrderSourceMode.BENCHMARK:
        replay_rate = float(
            overrides.get(
                "benchmark_replay_arrival_rate",
                data_cfg.get("benchmark_replay_arrival_rate", 0.0),
            )
        )
        if replay_rate != 0.0:
            raise ValueError(
                "benchmark 模式必须显式关闭泊松流，"
                f"当前 benchmark_replay_arrival_rate={replay_rate}"
            )
        return 0.0
    default_rate = data_cfg["poisson_arrival_rate"]
    if mode == OrderSourceMode.HYBRID:
        default_rate = data_cfg["poisson_arrival_rate"]
    return float(overrides.get("poisson_arrival_rate", default_rate))


def _resolve_seed(
    mode: OrderSourceMode,
    data_cfg: Mapping[str, Any],
    explicit_seed: int | None,
    overrides: Mapping[str, Any],
) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    if "seed" in overrides:
        return int(overrides["seed"])
    if "poisson_seed" in overrides:
        return int(overrides["poisson_seed"])

    if mode == OrderSourceMode.BENCHMARK:
        return int(data_cfg["benchmark_eval_seed"])
    if mode == OrderSourceMode.HYBRID:
        return int(data_cfg["stochastic_eval_seed_base"])
    return int(data_cfg["poisson_seed"])


def _coerce_mode(mode: str | OrderSourceMode) -> OrderSourceMode:
    if isinstance(mode, OrderSourceMode):
        return mode
    try:
        return OrderSourceMode(str(mode).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in OrderSourceMode)
        raise ValueError(f"未知订单源模式: {mode}，仅支持 {allowed}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML 依赖，无法读取训练配置") from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须为对象: {path}")
    return data


def _require_mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置缺少 {key} 段或类型错误")
    return value


__all__ = [
    "OrderSourceConfig",
    "OrderSourceMode",
    "PoissonOrderGenConfig",
    "build_order_source",
    "build_order_source_preview_summary",
    "configure_order_manager_for_source",
    "ensure_mode_allowed",
    "preview_dynamic_order_stream",
]
