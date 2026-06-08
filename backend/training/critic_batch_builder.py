#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 7 CriticBatchBuilder.

职责：
  - 基于同一份 `DecisionContext` pre-action snapshot 构造 centralized critic 输入；
  - 固化 CriticTensorSchemaV1 的字段顺序、排序、截断、padding 与归一化规则；
  - 输出 materialized numpy tensors，供 rollout buffer / model 再转换为 torch。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    CriticBatch,
    CriticNormalizationMeta,
    CriticTensorSchemaMeta,
)
from .scene_loader import DEFAULT_CONFIG_PATH, TrainingSceneContext, load_default_scene


_FLOAT_DTYPE = np.float32
_BOOL_DTYPE = np.bool_
_TIME_EPS = 1e-6

_STATUS_CODE = {"pending": 0, "assigned": 1}
_TRAINING_STATE_CODE = {
    "idle": 0,
    "flying_to_deliver": 1,
    "delivered": 2,
    "return_to_rendezvous": 3,
    "waiting_for_truck": 4,
    "return_to_station": 5,
    "return_to_depot": 6,
    "queueing_at_host": 7,
    "charging_or_swap": 8,
    "active_wait": 9,
    "fallback_recovery": 10,
    "charging_on_truck": 11,
    "riding_with_truck": 12,
    "airborne_energy_failure": 13,
}
_HOME_TYPE_CODE = {"depot": 0, "truck": 1, "station": 2, "none": 3}
_NODE_TYPE_CODE = {"depot": 0, "station": 1}
_DISPATCH_MODE_CODE = {"NONE": 0, "A": 1, "B": 2, "C": 3}
_PAYLOAD_CLASS_CODE = {"light": 0, "heavy": 1}

CRITIC_ORDER_TOKEN_FIELDS = (
    "is_valid",
    "is_uav_primary_scope",
    "is_mode_a_background",
    "is_assigned",
    "is_overdue",
    "status_code_norm",
    "deadline_slack_norm",
    "age_norm",
    "payload_norm",
    "delivery_x_norm",
    "delivery_y_norm",
    "delivery_z_norm",
    "distance_to_truck_norm",
    "assigned_mode_code_norm",
    "priority_band_norm",
)

CRITIC_UAV_TOKEN_FIELDS = (
    "is_valid",
    "is_current_actor",
    "training_state_code_norm",
    "battery_current_norm",
    "battery_max_norm",
    "battery_ratio",
    "eta_to_available_norm",
    "is_riding_with_truck",
    "has_reservation",
    "reservation_remaining_sec_norm",
    "carrying_order_exists",
    "dispatch_mode_code_norm",
    "payload_class_code_norm",
    "home_type_code_norm",
)

CRITIC_STATION_TOKEN_FIELDS = (
    "is_valid",
    "node_type_code_norm",
    "x_norm",
    "y_norm",
    "z_norm",
    "queue_length_norm",
    "available_slots_norm",
    "parking_slots_norm",
    "predicted_queue_time_est_norm",
    "reservation_count_norm",
    "node_charge_load_budget_norm",
    "truck_eta_remaining_norm",
    "has_truck_eta",
    "is_future_truck_stop",
    "is_active_launch_station",
)

CRITIC_COARSE_PLAN_SUMMARY_FIELDS = (
    "plan_version_norm",
    "future_backbone_node_count_norm",
    "launch_candidate_station_count_norm",
    "authorized_order_count_norm",
    "route_drift_ratio",
    "backlog_new_orders_norm",
    "fallback_count_window_norm",
    "hard_failure_count_window_norm",
)

CRITIC_GLOBAL_SYSTEM_SUMMARY_FIELDS = (
    "active_order_count_norm",
    "pending_order_count_norm",
    "assigned_order_count_norm",
    "overdue_order_count_norm",
    "mode_a_background_count_norm",
    "station_queue_total_norm",
    "reservation_total_norm",
    "fallback_count_window_norm",
    "hard_failure_count_window_norm",
    "truncated_order_count_norm",
)

_ORDERING_RULES = (
    "global_order_pool_tokens: scope_rank -> overdue_rank -> status_rank -> deadline_slack_sec -> priority_band -> spawn_time_sec -> stable_order_id",
    "global_uav_tokens: slot0=current_actor; remaining sorted by stable_drone_id",
    "global_station_tokens: slot0=depot; remaining stations sorted by stable_node_id",
)
_TRUNCATION_RULES = (
    "orders: top-K urgent active orders",
    "uavs: keep current actor, then stable_drone_id",
    "stations: keep depot first, then stable_node_id",
)
_PADDING_RULES = (
    "padding token is all zeros",
    "padding_mask=True means slot is padding",
)
_CAUSAL_BLACKLIST = (
    "future_poisson_orders",
    "future_benchmark_dynamic_orders",
    "future_queue_outcomes",
    "post_action_state_mutations",
)


@dataclass(frozen=True)
class _BuilderConfig:
    upper_horizon_sec: float
    queue_norm_cap: float
    payload_norm_kg: float
    max_global_orders: int
    max_global_uavs: int
    max_global_stations: int


@dataclass(frozen=True)
class _SceneNorm:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    diagonal_m: float
    light_battery_capacity_j: float
    heavy_battery_capacity_j: float
    heavy_payload_capacity_kg: float


class CriticBatchBuilder:
    """基于 pre-action DecisionContext snapshot 构造 CriticBatch。"""

    def __init__(
        self,
        *,
        scene_ctx: TrainingSceneContext | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self._scene_ctx = scene_ctx or load_default_scene(config_path=config_path)
        self._cfg = _load_builder_config(Path(config_path))
        self._scene_norm = _build_scene_norm(self._scene_ctx)
        self._default_schema_meta = build_default_critic_tensor_schema_meta(
            scene_ctx=self._scene_ctx,
            config_path=config_path,
        )

    @property
    def default_schema_meta(self) -> CriticTensorSchemaMeta:
        return self._default_schema_meta

    def build(
        self,
        *,
        decision_context: Any,
        critic_tensor_schema_meta: CriticTensorSchemaMeta,
    ) -> CriticBatch:
        if critic_tensor_schema_meta.schema_hash != self._default_schema_meta.schema_hash:
            _assert_compatible_schema(critic_tensor_schema_meta, self._default_schema_meta)

        order_tokens, order_padding_mask, order_stats = self._build_order_tokens(decision_context)
        uav_tokens, uav_padding_mask = self._build_uav_tokens(decision_context)
        station_tokens, station_padding_mask, station_queue_total = self._build_station_tokens(
            decision_context
        )
        coarse_plan_summary_vec = self._build_coarse_plan_summary_vec(decision_context)
        global_system_summary_vec = self._build_global_system_summary_vec(
            decision_context=decision_context,
            order_stats=order_stats,
            station_queue_total=station_queue_total,
        )
        return CriticBatch(
            global_order_pool_tokens=order_tokens,
            global_uav_tokens=uav_tokens,
            global_station_tokens=station_tokens,
            coarse_plan_summary_vec=coarse_plan_summary_vec,
            global_system_summary_vec=global_system_summary_vec,
            global_order_padding_mask=order_padding_mask,
            global_uav_padding_mask=uav_padding_mask,
            global_station_padding_mask=station_padding_mask,
        )

    def _build_order_tokens(
        self,
        decision_context: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        runtime_state = decision_context.runtime_state
        coarse_plan = decision_context.coarse_plan
        t_now = float(decision_context.t_decision)
        truck_loc = runtime_state.truck_current_loc
        order_map: dict[str, Any] = {}
        order_map.update(runtime_state.pending_orders)
        order_map.update(runtime_state.assigned_orders)
        ordered = []
        for order_id, order in order_map.items():
            is_mode_a_background = bool(
                float(order.payload_weight) > self._scene_norm.heavy_payload_capacity_kg
                or str(order.assigned_mode or "") == "A"
            )
            is_uav_primary_scope = not is_mode_a_background
            is_assigned = order_id in runtime_state.assigned_orders
            is_overdue = float(order.deadline) < t_now - _TIME_EPS
            priority_band = int(coarse_plan.order_priority_band.get(order_id, 3))
            status_rank = 1 if is_assigned else 0
            ordered.append(
                (
                    (
                        0 if is_uav_primary_scope else 1,
                        0 if is_overdue else 1,
                        status_rank,
                        float(order.deadline) - t_now,
                        priority_band,
                        float(order.create_time),
                        str(order_id),
                    ),
                    order,
                    {
                        "is_uav_primary_scope": is_uav_primary_scope,
                        "is_mode_a_background": is_mode_a_background,
                        "is_assigned": is_assigned,
                        "is_overdue": is_overdue,
                        "priority_band": priority_band,
                    },
                )
            )
        ordered.sort(key=lambda item: item[0])
        kept = ordered[: self._cfg.max_global_orders]

        tokens = np.zeros(
            (self._cfg.max_global_orders, len(CRITIC_ORDER_TOKEN_FIELDS)),
            dtype=_FLOAT_DTYPE,
        )
        padding_mask = np.ones((self._cfg.max_global_orders,), dtype=_BOOL_DTYPE)
        active_order_count = len(ordered)
        pending_order_count = len(runtime_state.pending_orders)
        assigned_order_count = len(runtime_state.assigned_orders)
        overdue_order_count = 0
        mode_a_background_count = 0
        for slot, (_key, order, flags) in enumerate(kept):
            padding_mask[slot] = False
            if flags["is_overdue"]:
                overdue_order_count += 1
            if flags["is_mode_a_background"]:
                mode_a_background_count += 1
            tokens[slot, :] = np.asarray(
                [
                    1.0,
                    _bool(flags["is_uav_primary_scope"]),
                    _bool(flags["is_mode_a_background"]),
                    _bool(flags["is_assigned"]),
                    _bool(flags["is_overdue"]),
                    _code_norm(_STATUS_CODE, "assigned" if flags["is_assigned"] else "pending"),
                    _norm_time_signed(float(order.deadline) - t_now, self._cfg.upper_horizon_sec),
                    _norm_time_nonneg(t_now - float(order.create_time), self._cfg.upper_horizon_sec),
                    _clip01(float(order.payload_weight) / max(self._cfg.payload_norm_kg, _TIME_EPS)),
                    _norm_axis(float(order.delivery_loc.x), self._scene_norm.min_x, self._scene_norm.max_x),
                    _norm_axis(float(order.delivery_loc.y), self._scene_norm.min_y, self._scene_norm.max_y),
                    _norm_axis(float(order.delivery_loc.z), self._scene_norm.min_z, self._scene_norm.max_z),
                    _norm_distance(
                        truck_loc.distance_3d(order.delivery_loc),
                        self._scene_norm.diagonal_m,
                    ),
                    _code_norm(_DISPATCH_MODE_CODE, str(order.assigned_mode or "NONE")),
                    _clip01(float(flags["priority_band"]) / 3.0),
                ],
                dtype=_FLOAT_DTYPE,
            )

        if len(kept) < len(ordered):
            overdue_order_count += sum(1 for _, _, flags in ordered[len(kept) :] if flags["is_overdue"])
            mode_a_background_count += sum(
                1 for _, _, flags in ordered[len(kept) :] if flags["is_mode_a_background"]
            )
        return tokens, padding_mask, {
            "active_order_count": int(active_order_count),
            "pending_order_count": int(pending_order_count),
            "assigned_order_count": int(assigned_order_count),
            "overdue_order_count": int(overdue_order_count),
            "mode_a_background_count": int(mode_a_background_count),
            "truncated_order_count": int(max(0, len(ordered) - len(kept))),
        }

    def _build_uav_tokens(
        self,
        decision_context: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        runtime_state = decision_context.runtime_state
        current_actor = str(decision_context.deciding_drone_id)
        ordered_ids = [current_actor] + sorted(
            drone_id for drone_id in runtime_state.drone_states if drone_id != current_actor
        )
        kept_ids = ordered_ids[: self._cfg.max_global_uavs]

        tokens = np.zeros(
            (self._cfg.max_global_uavs, len(CRITIC_UAV_TOKEN_FIELDS)),
            dtype=_FLOAT_DTYPE,
        )
        padding_mask = np.ones((self._cfg.max_global_uavs,), dtype=_BOOL_DTYPE)
        for slot, drone_id in enumerate(kept_ids):
            state = runtime_state.drone_states[drone_id]
            padding_mask[slot] = False
            tokens[slot, :] = np.asarray(
                [
                    1.0,
                    _bool(drone_id == current_actor),
                    _code_norm(_TRAINING_STATE_CODE, str(state.training_state)),
                    _norm_energy_absolute(float(state.battery_current), float(state.battery_max)),
                    _norm_energy_absolute(float(state.battery_max), float(state.battery_max)),
                    _clip01(float(state.battery_ratio)),
                    _norm_time_nonneg(
                        float(decision_context.execution_snapshot.uav_eta_to_available.get(drone_id, 0.0)),
                        self._cfg.upper_horizon_sec,
                    ),
                    _bool(str(state.training_state) == "riding_with_truck"),
                    _bool(state.reservation is not None),
                    _norm_time_nonneg(
                        (
                            max(
                                0.0,
                                float(
                                    decision_context.coarse_plan.truck_eta_map.get(
                                        state.reservation.recover_node,
                                        decision_context.t_decision,
                                    )
                                )
                                - float(decision_context.t_decision),
                            )
                            if state.reservation is not None
                            else 0.0
                        ),
                        self._cfg.upper_horizon_sec,
                    ),
                    _bool(bool(state.carrying_order_id)),
                    _code_norm(
                        _DISPATCH_MODE_CODE,
                        str(decision_context.execution_snapshot.uav_dispatch_mode.get(drone_id, "NONE")),
                    ),
                    _code_norm(_PAYLOAD_CLASS_CODE, _payload_class(float(state.payload_capacity))),
                    _code_norm(_HOME_TYPE_CODE, str(state.home_type)),
                ],
                dtype=_FLOAT_DTYPE,
            )
        return tokens, padding_mask

    def _build_station_tokens(
        self,
        decision_context: Any,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        runtime_state = decision_context.runtime_state
        coarse_plan = decision_context.coarse_plan
        active_launch = set(decision_context.planner_snapshot.active_launch_stations)

        depot_ids = sorted(
            node_id for node_id, node in runtime_state.node_states.items() if node.node_type == "depot"
        )
        station_ids = sorted(
            node_id for node_id, node in runtime_state.node_states.items() if node.node_type == "station"
        )
        ordered_ids = depot_ids + station_ids
        kept_ids = ordered_ids[: self._cfg.max_global_stations]

        tokens = np.zeros(
            (self._cfg.max_global_stations, len(CRITIC_STATION_TOKEN_FIELDS)),
            dtype=_FLOAT_DTYPE,
        )
        padding_mask = np.ones((self._cfg.max_global_stations,), dtype=_BOOL_DTYPE)
        station_queue_total = 0
        for slot, node_id in enumerate(kept_ids):
            node = runtime_state.node_states[node_id]
            padding_mask[slot] = False
            station_queue_total += int(node.queue_length)
            truck_eta_remaining = (
                max(0.0, float(coarse_plan.truck_eta_map[node_id]) - float(decision_context.t_decision))
                if node_id in coarse_plan.truck_eta_map
                else 0.0
            )
            predicted_queue_time_est = (
                0.0
                if int(node.available_slots) > 0
                else (int(node.queue_length) + 1) * float(node.swap_time)
            )
            tokens[slot, :] = np.asarray(
                [
                    1.0,
                    _code_norm(_NODE_TYPE_CODE, str(node.node_type)),
                    _norm_axis(float(node.position.x), self._scene_norm.min_x, self._scene_norm.max_x),
                    _norm_axis(float(node.position.y), self._scene_norm.min_y, self._scene_norm.max_y),
                    _norm_axis(float(node.position.z), self._scene_norm.min_z, self._scene_norm.max_z),
                    _clip01(float(node.queue_length) / max(self._cfg.queue_norm_cap, _TIME_EPS)),
                    _clip01(float(node.available_slots) / max(float(node.parking_slots), 1.0)),
                    _clip01(float(node.parking_slots) / max(self._cfg.queue_norm_cap, 1.0)),
                    _norm_time_nonneg(predicted_queue_time_est, self._cfg.upper_horizon_sec),
                    _clip01(
                        float(runtime_state.reservation_count.get(node_id, 0))
                        / max(self._cfg.queue_norm_cap, _TIME_EPS)
                    ),
                    _clip01(
                        float(coarse_plan.node_charge_load_budget.get(node_id, 0))
                        / max(self._cfg.queue_norm_cap, _TIME_EPS)
                    ),
                    _norm_time_nonneg(truck_eta_remaining, self._cfg.upper_horizon_sec),
                    _bool(node_id in coarse_plan.truck_eta_map),
                    _bool(node_id in coarse_plan.truck_eta_map),
                    _bool(node_id in active_launch),
                ],
                dtype=_FLOAT_DTYPE,
            )
        return tokens, padding_mask, int(station_queue_total)

    def _build_coarse_plan_summary_vec(self, decision_context: Any) -> np.ndarray:
        coarse_plan = decision_context.coarse_plan
        planner_snapshot = decision_context.planner_snapshot
        return np.asarray(
            [
                _clip01(float(coarse_plan.plan_version) / 32.0),
                _clip01(float(len(coarse_plan.truck_backbone_route)) / 16.0),
                _clip01(float(len(planner_snapshot.active_launch_stations)) / 16.0),
                _clip01(float(len(coarse_plan.authorized_orders)) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(planner_snapshot.route_drift_ratio)),
                _clip01(float(planner_snapshot.backlog_new_orders) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(planner_snapshot.fallback_count_in_window) / 8.0),
                _clip01(float(planner_snapshot.hard_failure_count_in_window) / 8.0),
            ],
            dtype=_FLOAT_DTYPE,
        )

    def _build_global_system_summary_vec(
        self,
        *,
        decision_context: Any,
        order_stats: Mapping[str, int],
        station_queue_total: int,
    ) -> np.ndarray:
        planner_snapshot = decision_context.planner_snapshot
        return np.asarray(
            [
                _clip01(float(order_stats["active_order_count"]) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(order_stats["pending_order_count"]) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(order_stats["assigned_order_count"]) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(order_stats["overdue_order_count"]) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(order_stats["mode_a_background_count"]) / max(self._cfg.max_global_orders, 1)),
                _clip01(float(station_queue_total) / max(self._cfg.queue_norm_cap * self._cfg.max_global_stations, 1.0)),
                _clip01(
                    float(sum(decision_context.runtime_state.reservation_count.values()))
                    / max(self._cfg.queue_norm_cap * self._cfg.max_global_stations, 1.0)
                ),
                _clip01(float(planner_snapshot.fallback_count_in_window) / 8.0),
                _clip01(float(planner_snapshot.hard_failure_count_in_window) / 8.0),
                _clip01(float(order_stats["truncated_order_count"]) / max(self._cfg.max_global_orders, 1)),
            ],
            dtype=_FLOAT_DTYPE,
        )


def build_default_critic_tensor_schema_meta(
    *,
    scene_ctx: TrainingSceneContext | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> CriticTensorSchemaMeta:
    scene_ctx = scene_ctx or load_default_scene(config_path=config_path)
    cfg = _load_builder_config(Path(config_path))
    scene_norm = _build_scene_norm(scene_ctx)
    return CriticTensorSchemaMeta(
        name=str(_load_policy(config_path)["critic_schema_name"]),
        schema_version=str(_load_policy(config_path)["critic_schema_version"]),
        max_global_orders=int(cfg.max_global_orders),
        max_global_uavs=int(cfg.max_global_uavs),
        max_global_stations=int(cfg.max_global_stations),
        order_token_fields=CRITIC_ORDER_TOKEN_FIELDS,
        uav_token_fields=CRITIC_UAV_TOKEN_FIELDS,
        station_token_fields=CRITIC_STATION_TOKEN_FIELDS,
        coarse_plan_summary_fields=CRITIC_COARSE_PLAN_SUMMARY_FIELDS,
        global_system_summary_fields=CRITIC_GLOBAL_SYSTEM_SUMMARY_FIELDS,
        ordering_rules=_ORDERING_RULES,
        truncation_rules=_TRUNCATION_RULES,
        padding_rules=_PADDING_RULES,
        normalization=CriticNormalizationMeta(
            time_norm_sec=float(cfg.upper_horizon_sec),
            distance_norm_m=float(scene_norm.diagonal_m),
            payload_norm_kg=float(cfg.payload_norm_kg),
            eta_norm_sec=float(cfg.upper_horizon_sec),
            queue_norm_cap=float(cfg.queue_norm_cap),
            energy_norm_strategy=str(_load_policy(config_path)["energy_norm_strategy"]),
            light_battery_capacity_j=float(scene_norm.light_battery_capacity_j),
            heavy_battery_capacity_j=float(scene_norm.heavy_battery_capacity_j),
        ),
        snapshot_rule="pre_action_decision_context",
        storage_mode="materialized_tensors_in_rollout_buffer",
        causal_blacklist=_CAUSAL_BLACKLIST,
    )


def _assert_compatible_schema(
    runtime_meta: CriticTensorSchemaMeta,
    default_meta: CriticTensorSchemaMeta,
) -> None:
    if runtime_meta.schema_hash != default_meta.schema_hash:
        raise ValueError(
            "传入的 critic_tensor_schema_meta 与当前代码默认 schema 不一致: "
            f"{runtime_meta.schema_hash} != {default_meta.schema_hash}"
        )


def _payload_class(payload_capacity: float) -> str:
    return "light" if payload_capacity <= 2.0 + _TIME_EPS else "heavy"


def _norm_axis(value: float, min_value: float, max_value: float) -> float:
    span = max(max_value - min_value, _TIME_EPS)
    return _clip01((value - min_value) / span)


def _norm_distance(value: float, diagonal_m: float) -> float:
    return _clip_signed(float(value) / max(diagonal_m, _TIME_EPS), -1.0, 1.0)


def _norm_time_nonneg(value: float, upper_horizon_sec: float) -> float:
    return _clip01(float(value) / max(upper_horizon_sec, _TIME_EPS))


def _norm_time_signed(value: float, upper_horizon_sec: float) -> float:
    return _clip_signed(float(value) / max(upper_horizon_sec, _TIME_EPS), -1.0, 1.0)


def _norm_energy_absolute(value: float, battery_max: float) -> float:
    return _clip01(float(value) / max(float(battery_max), _TIME_EPS))


def _code_norm(mapping: Mapping[str, int], value: str) -> float:
    denom = max(len(mapping) - 1, 1)
    return float(mapping.get(value, 0)) / float(denom)


def _bool(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _clip_signed(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _load_policy(config_path: str | Path) -> Mapping[str, Any]:
    raw = _load_yaml(Path(config_path))
    policy = raw.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"配置缺少 policy 段: {config_path}")
    return policy


def _load_builder_config(config_path: Path) -> _BuilderConfig:
    raw = _load_yaml(config_path)
    planner = _require_mapping(raw, "planner")
    policy = _require_mapping(raw, "policy")
    data = _require_mapping(raw, "data")
    return _BuilderConfig(
        upper_horizon_sec=float(planner["upper_horizon_sec"]),
        queue_norm_cap=float(policy.get("queue_norm_cap", 8.0)),
        payload_norm_kg=float(data["poisson_weight_max_kg"]),
        max_global_orders=int(policy["max_global_orders"]),
        max_global_uavs=int(policy["max_global_uavs"]),
        max_global_stations=int(policy["max_global_stations"]),
    )


def _iter_scene_positions(scene_ctx: TrainingSceneContext) -> Sequence[tuple[float, float, float]]:
    positions = []
    for depot in scene_ctx.depots.values():
        positions.append((float(depot.location.x), float(depot.location.y), float(depot.location.z)))
    for station in scene_ctx.stations.values():
        positions.append((float(station.location.x), float(station.location.y), float(station.location.z)))
    for truck in scene_ctx.trucks.values():
        positions.append((float(truck.current_loc.x), float(truck.current_loc.y), float(truck.current_loc.z)))
    for order in scene_ctx.static_orders:
        positions.append((float(order.delivery_loc.x), float(order.delivery_loc.y), float(order.delivery_loc.z)))
    for item in scene_ctx.dynamic_orders:
        positions.append((float(item.order.delivery_loc.x), float(item.order.delivery_loc.y), float(item.order.delivery_loc.z)))
    return positions


def _build_scene_norm(scene_ctx: TrainingSceneContext) -> _SceneNorm:
    positions = list(_iter_scene_positions(scene_ctx))
    xs = [item[0] for item in positions]
    ys = [item[1] for item in positions]
    zs = [item[2] for item in positions]
    drones = list(scene_ctx.drones.values())
    light_candidates = [drone for drone in drones if float(drone.payload_capacity) <= 2.0 + _TIME_EPS]
    heavy_candidates = [drone for drone in drones if float(drone.payload_capacity) > 2.0 + _TIME_EPS]
    light_battery = min((float(drone.battery_max) for drone in light_candidates), default=min(float(drone.battery_max) for drone in drones))
    heavy_battery = max((float(drone.battery_max) for drone in heavy_candidates), default=max(float(drone.battery_max) for drone in drones))
    heavy_payload_capacity = max(float(drone.payload_capacity) for drone in drones)
    return _SceneNorm(
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
        min_z=min(zs),
        max_z=max(zs),
        diagonal_m=max(float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))), 1.0),
        light_battery_capacity_j=float(light_battery),
        heavy_battery_capacity_j=float(heavy_battery),
        heavy_payload_capacity_kg=float(heavy_payload_capacity),
    )


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取 critic builder 配置") from exc

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML 顶层必须为 mapping: {config_path}")
    return raw


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置缺少 mapping 段: {key}")
    return value


__all__ = [
    "CRITIC_COARSE_PLAN_SUMMARY_FIELDS",
    "CRITIC_GLOBAL_SYSTEM_SUMMARY_FIELDS",
    "CRITIC_ORDER_TOKEN_FIELDS",
    "CRITIC_STATION_TOKEN_FIELDS",
    "CRITIC_UAV_TOKEN_FIELDS",
    "CriticBatchBuilder",
    "build_default_critic_tensor_schema_meta",
]
