#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 7 ObservationTensorizer.

职责：
  - 将 Phase 6 `CandidateOutput` 物化为固定 shape 的 observation tensors；
  - 将 rollout 侧维护的 `TransitionSummary` 历史压成 `history_tokens`；
  - 独立输出 actor 需要的结构化 `FactorizedActionMask`。

实现边界：
  - 本模块只做 schema 固化与数值编码，不依赖 torch；
  - 输出使用 numpy，后续 `model.py` / `rollout_buffer.py` 可再转换为 torch tensor。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    CandidateOutput,
    FactorizedActionMask,
    ObservationBatch,
    TransitionSummary,
)
from .scene_loader import DEFAULT_CONFIG_PATH, TrainingSceneContext, load_default_scene


_FLOAT_DTYPE = np.float32
_BOOL_DTYPE = np.bool_
_TIME_EPS = 1e-6

_TRAINING_STATE_CODE: Mapping[str, int] = {
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
_HOME_TYPE_CODE: Mapping[str, int] = {"depot": 0, "truck": 1, "station": 2, "none": 3}
_HOST_TYPE_CODE: Mapping[str, int] = {"": 0, "none": 0, "station": 1, "depot": 2}
_TRIGGER_TYPE_CODE: Mapping[str, int] = {
    "initial_idle": 0,
    "inline_idle": 1,
    "test_idle": 2,
    "riding_with_truck": 3,
    "truck_station_arrival": 4,
    "idle_ready": 5,
    "wait_resume": 6,
    "order_arrival_wake": 7,
}
_ROOT_BRANCH_CODE: Mapping[str, int] = {"WAIT": 0, "DISPATCH": 1}
_DISPATCH_MODE_CODE: Mapping[str, int] = {"NONE": 0, "B": 1, "C": 2}
_PAYLOAD_CLASS_CODE: Mapping[str, int] = {"light": 0, "heavy": 1}

UAV_SELF_TOKEN_FIELDS = (
    "x_norm",
    "y_norm",
    "z_norm",
    "battery_current_norm",
    "battery_max_norm",
    "battery_ratio",
    "training_state_code_norm",
    "has_reservation",
    "reservation_remaining_norm",
    "plan_version_delta_norm",
    "is_riding_truck",
    "home_type_code_norm",
    "cruise_speed_norm",
    "payload_capacity_norm",
)

ORDER_TOKEN_FIELDS = (
    "is_valid",
    "weight_norm",
    "deadline_remaining_time_norm",
    "estimated_delivery_finish_slack_norm",
    "delivery_x_norm",
    "delivery_y_norm",
    "delivery_z_norm",
    "distance_to_order_norm",
    "order_pre_score_norm",
    "priority_band_norm",
    "has_mode_b_action",
    "best_mode_b_return_score_norm",
    "best_mode_b_recovery_flight_time_norm",
    "best_mode_b_host_type_code_norm",
    "best_mode_b_queue_time_est_norm",
    "has_mode_c_action",
    "mode_c_candidate_count_norm",
    "best_mode_c_rendezvous_margin_norm",
    "best_mode_c_wait_time_norm",
    "best_mode_c_uav_flight_time_norm",
    "best_mode_c_energy_margin_ratio",
    "delivery_energy_ratio",
    "best_mode_b_recovery_energy_ratio",
    "best_mode_c_recovery_energy_ratio",
    "best_mode_b_total_energy_ratio",
    "best_mode_c_total_energy_ratio",
    "mode_c_energy_saving_ratio",
    "best_mode_c_node_type_code_norm",
    "best_mode_c_truck_eta_remaining_norm",
    "best_mode_c_timeout_risk_norm",
    "local_teacher_has_order_choice",
    "local_teacher_prefers_order",
    "local_teacher_order_cost_norm",
    "local_teacher_best_mode_code_norm",
    "local_teacher_peer_prefer_count_norm",
    "local_teacher_peer_best_other_cost_norm",
    "local_teacher_cost_gap_to_order_best_norm",
    "local_teacher_is_order_best",
    "local_teacher_mode_b_prefer_count_norm",
    "local_teacher_mode_c_prefer_count_norm",
)

INFRA_TOKEN_FIELDS = (
    "node_type_code_norm",
    "x_norm",
    "y_norm",
    "z_norm",
    "queue_length_norm",
    "available_slots_norm",
    "parking_slots_norm",
    "swap_time_norm",
    "truck_eta_remaining_norm",
    "has_truck_eta",
    "node_charge_load_budget_norm",
    "is_in_backbone",
    "is_launch_candidate_station",
    "future_backbone_node_count_norm",
    "authorized_order_count_norm",
    "plan_version_norm",
    "truck_x_norm",
    "truck_y_norm",
    "truck_z_norm",
)

HISTORY_TOKEN_FIELDS = (
    "dt_since_event_norm",
    "is_self_event",
    "actor_rel_x_norm",
    "actor_rel_y_norm",
    "actor_training_state_before_code_norm",
    "actor_training_state_after_code_norm",
    "actor_home_type_code_norm",
    "actor_payload_class_code_norm",
    "trigger_type_code_norm",
    "root_branch_code_norm",
    "dispatch_mode_code_norm",
    "selected_recover_node_type_code_norm",
    "has_selected_order",
    "selected_order_slot_rank_norm",
    "selected_order_deadline_slack_norm",
    "selected_eta_to_deliver_norm",
    "selected_rendezvous_margin_norm",
    "energy_ratio_before",
    "energy_ratio_after",
    "queue_after_norm",
    "plan_version_delta_norm",
    "delivered",
    "rendezvous_success",
    "reservation_timeout",
    "fallback_started",
    "hard_failure",
    "queue_entered",
    "service_completed",
)


@dataclass(frozen=True)
class _TensorizerConfig:
    hist_len: int
    upper_horizon_sec: float
    payload_norm_kg: float
    queue_norm_cap: float
    max_order_tokens: int
    max_candidate_recovery_per_order: int


@dataclass(frozen=True)
class _SceneNorm:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    diagonal_m: float
    max_speed: float
    max_payload_capacity: float
    max_drone_count: int


class ObservationTensorizer:
    """将 `CandidateOutput` 固定编码为 ObservationBatch。"""

    def __init__(
        self,
        *,
        scene_ctx: TrainingSceneContext | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self._scene_ctx = scene_ctx or load_default_scene(config_path=config_path)
        self._cfg = _load_tensorizer_config(Path(config_path))
        self._scene_norm = _build_scene_norm(self._scene_ctx)

    @property
    def history_length(self) -> int:
        return self._cfg.hist_len

    def build_action_mask(self, candidate_out: CandidateOutput) -> FactorizedActionMask:
        return FactorizedActionMask(
            root_branch_mask=np.asarray(candidate_out.root_branch_mask, dtype=_BOOL_DTYPE),
            order_mask=np.asarray(candidate_out.order_mask, dtype=_BOOL_DTYPE),
            mode_mask=np.asarray(candidate_out.mode_mask, dtype=_BOOL_DTYPE),
        )

    def build(
        self,
        *,
        decision_context: Any,
        candidate_out: CandidateOutput,
        transition_history: Sequence[TransitionSummary],
    ) -> ObservationBatch:
        features = candidate_out.candidate_features
        uav_self_token = self._build_uav_self_token(features.uav_self)
        order_tokens, order_padding_mask = self._build_order_tokens(features.order_features)
        infra_tokens = self._build_infra_tokens(
            features.infra_features,
            decision_context=decision_context,
        )
        history_tokens, history_padding_mask = self._build_history_tokens(
            transition_history=transition_history,
            decision_context=decision_context,
        )
        return ObservationBatch(
            uav_self_token=uav_self_token,
            order_tokens=order_tokens,
            infra_tokens=infra_tokens,
            history_tokens=history_tokens,
            history_padding_mask=history_padding_mask,
            padding_mask=order_padding_mask,
        )

    def build_transition_summary(
        self,
        *,
        decision_context: Any,
        candidate_out: CandidateOutput,
        action_indices: Any,
        step_result: Any,
    ) -> TransitionSummary:
        actor_id = str(decision_context.deciding_drone_id)
        trigger_type = str(decision_context.trigger_type)
        self._code_norm(
            _TRIGGER_TYPE_CODE,
            trigger_type,
            strict=True,
            field_name="trigger_type",
        )
        pre_drone = decision_context.runtime_state.drone_states[actor_id]
        post_drone = step_result.runtime_state.drone_states[actor_id]
        reward_breakdown = dict(step_result.info.get("reward_breakdown", {}))

        root_branch = "WAIT" if int(action_indices.root_branch_idx) == 0 else "DISPATCH"
        dispatch_mode = "NONE"
        selected_recover_node_type = "NONE"
        has_selected_order = False
        selected_order_slot_rank = -1
        selected_order_deadline_slack_norm = 0.0
        selected_eta_to_deliver_norm = 0.0
        selected_rendezvous_margin_norm = 0.0

        if root_branch == "DISPATCH":
            has_selected_order = True
            selected_order_slot_rank = int(action_indices.order_idx)
            order_feature = candidate_out.candidate_features.order_features[action_indices.order_idx]
            selected_order_deadline_slack_norm = self._norm_time_signed(order_feature.remaining_time)
            selected_eta_to_deliver_norm = self._norm_time_nonneg(
                order_feature.distance_to_order / max(float(pre_drone.cruise_speed), _TIME_EPS)
            )
            dispatch_mode = (
                candidate_out.factorized_action_schema.mode_order[action_indices.mode_idx]
            )
            if dispatch_mode == "C":
                selected_rendezvous_margin_norm = self._norm_time_signed(
                    order_feature.best_mode_c_rendezvous_margin
                )
                selected_recover_node_type = order_feature.best_mode_c_node_type

        queue_after_norm = self._infer_queue_after_norm(
            runtime_state=step_result.runtime_state,
            drone_state=post_drone,
        )
        delivered = float(reward_breakdown.get("delivery_bonus", 0.0)) > 0.0
        hard_failure = (
            float(reward_breakdown.get("hard_failure", 0.0)) < 0.0
            or post_drone.training_state == "airborne_energy_failure"
        )
        fallback_started = post_drone.training_state == "fallback_recovery"
        fallback_cause = "none"
        active_fallback_causes = step_result.info.get(
            "active_fallback_causes_by_drone",
            {},
        )
        if isinstance(active_fallback_causes, dict):
            fallback_cause = str(active_fallback_causes.get(actor_id, "none"))
        reservation_timeout = fallback_cause == "rendezvous_wait_timeout"
        rendezvous_success = post_drone.training_state in {"charging_on_truck", "riding_with_truck"}
        queue_entered = post_drone.training_state == "queueing_at_host"
        service_completed = (
            pre_drone.training_state in {"queueing_at_host", "charging_or_swap"}
            and post_drone.training_state == "idle"
        )

        return TransitionSummary(
            event_time=float(step_result.runtime_state.t_now),
            actor_drone_id=actor_id,
            actor_pos_x=float(post_drone.current_loc.x),
            actor_pos_y=float(post_drone.current_loc.y),
            actor_training_state_before=str(pre_drone.training_state),
            actor_training_state_after=str(post_drone.training_state),
            actor_home_type=str(pre_drone.home_type),
            actor_payload_class=_payload_class(float(pre_drone.payload_capacity)),
            trigger_type=trigger_type,
            root_branch=root_branch,
            dispatch_mode=dispatch_mode,
            selected_recover_node_type=selected_recover_node_type,
            has_selected_order=has_selected_order,
            selected_order_slot_rank=selected_order_slot_rank,
            selected_order_deadline_slack_norm=float(selected_order_deadline_slack_norm),
            selected_eta_to_deliver_norm=float(selected_eta_to_deliver_norm),
            selected_rendezvous_margin_norm=float(selected_rendezvous_margin_norm),
            energy_ratio_before=float(pre_drone.battery_ratio),
            energy_ratio_after=float(post_drone.battery_ratio),
            queue_after_norm=float(queue_after_norm),
            plan_version_delta_at_event=int(candidate_out.candidate_features.uav_self.plan_version_delta),
            delivered=bool(delivered),
            rendezvous_success=bool(rendezvous_success),
            reservation_timeout=bool(reservation_timeout),
            fallback_started=bool(fallback_started),
            hard_failure=bool(hard_failure),
            queue_entered=bool(queue_entered),
            service_completed=bool(service_completed),
            fallback_cause=fallback_cause if fallback_started else "none",
        )

    def _build_uav_self_token(self, uav_self: Any) -> np.ndarray:
        return np.asarray(
            [
                self._norm_x(float(uav_self.x)),
                self._norm_y(float(uav_self.y)),
                self._norm_z(float(uav_self.z)),
                self._norm_energy_absolute(float(uav_self.battery_current), float(uav_self.battery_max)),
                self._norm_energy_absolute(float(uav_self.battery_max), float(uav_self.battery_max)),
                float(uav_self.battery_ratio),
                self._code_norm(_TRAINING_STATE_CODE, str(uav_self.training_state)),
                self._bool(float(uav_self.has_reservation)),
                self._norm_time_nonneg(float(uav_self.reservation_remaining_sec)),
                self._norm_plan_version_delta(int(uav_self.plan_version_delta)),
                self._bool(bool(uav_self.is_riding_truck)),
                self._code_norm(_HOME_TYPE_CODE, str(uav_self.drone_source_type)),
                self._clip01(float(uav_self.cruise_speed) / max(self._scene_norm.max_speed, _TIME_EPS)),
                self._clip01(
                    float(uav_self.payload_capacity)
                    / max(self._scene_norm.max_payload_capacity, _TIME_EPS)
                ),
            ],
            dtype=_FLOAT_DTYPE,
        )

    def _build_order_tokens(
        self,
        order_features: Sequence[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.zeros((len(order_features), len(ORDER_TOKEN_FIELDS)), dtype=_FLOAT_DTYPE)
        padding_mask = np.ones((len(order_features),), dtype=_BOOL_DTYPE)
        for idx, item in enumerate(order_features):
            if not bool(item.is_valid):
                continue
            padding_mask[idx] = False
            tokens[idx, :] = np.asarray(
                [
                    1.0,
                    self._clip01(float(item.weight) / max(self._cfg.payload_norm_kg, _TIME_EPS)),
                    self._norm_time_signed(float(item.remaining_time)),
                    self._norm_time_signed(float(item.estimated_delivery_finish_slack_sec)),
                    self._norm_x(float(item.delivery_x)),
                    self._norm_y(float(item.delivery_y)),
                    self._norm_z(float(item.delivery_z)),
                    self._norm_distance(float(item.distance_to_order)),
                    self._norm_time_nonneg(float(item.order_pre_score)),
                    self._clip01(float(item.priority_band) / 2.0),
                    self._bool(bool(item.has_mode_b_action)),
                    self._norm_time_nonneg(float(item.best_mode_b_return_score)),
                    self._norm_time_nonneg(float(item.best_mode_b_recovery_flight_time)),
                    self._code_norm(_HOST_TYPE_CODE, str(item.best_mode_b_host_type)),
                    self._norm_time_nonneg(float(item.best_mode_b_queue_time_est)),
                    self._bool(bool(item.has_mode_c_action)),
                    self._clip01(
                        float(item.mode_c_candidate_count)
                        / max(float(self._cfg.max_candidate_recovery_per_order), 1.0)
                    ),
                    self._norm_time_signed(float(item.best_mode_c_rendezvous_margin)),
                    self._norm_time_nonneg(float(item.best_mode_c_wait_time)),
                    self._norm_time_nonneg(float(item.best_mode_c_uav_flight_time)),
                    self._clip01(max(0.0, float(item.best_mode_c_energy_margin_ratio))),
                    self._energy_ratio_feature(float(item.delivery_energy_ratio)),
                    self._energy_ratio_feature(
                        float(item.best_mode_b_recovery_energy_ratio)
                    ),
                    self._energy_ratio_feature(
                        float(item.best_mode_c_recovery_energy_ratio)
                    ),
                    self._energy_ratio_feature(
                        float(item.best_mode_b_total_energy_ratio)
                    ),
                    self._energy_ratio_feature(
                        float(item.best_mode_c_total_energy_ratio)
                    ),
                    self._clip_signed(
                        self._finite_or_zero(float(item.mode_c_energy_saving_ratio)),
                        -1.0,
                        1.0,
                    ),
                    self._code_norm(_HOST_TYPE_CODE, str(item.best_mode_c_node_type)),
                    self._norm_time_nonneg(float(item.best_mode_c_truck_eta_remaining)),
                    self._clip01(float(item.best_mode_c_timeout_risk)),
                    self._bool(bool(item.local_teacher_has_order_choice)),
                    self._bool(bool(item.local_teacher_prefers_order)),
                    self._norm_time_signed(float(item.local_teacher_order_cost)),
                    self._code_norm(
                        _DISPATCH_MODE_CODE,
                        str(item.local_teacher_best_mode),
                    ),
                    self._norm_peer_count(int(item.local_teacher_peer_prefer_count)),
                    self._norm_time_signed(float(item.local_teacher_peer_best_other_cost)),
                    self._norm_time_signed(float(item.local_teacher_cost_gap_to_order_best)),
                    self._bool(bool(item.local_teacher_is_order_best)),
                    self._norm_peer_count(int(item.local_teacher_mode_b_prefer_count)),
                    self._norm_peer_count(int(item.local_teacher_mode_c_prefer_count)),
                ],
                dtype=_FLOAT_DTYPE,
            )
        return tokens, padding_mask

    def _build_infra_tokens(
        self,
        infra_features: Any,
        *,
        decision_context: Any,
    ) -> np.ndarray:
        tokens = np.zeros(
            (len(infra_features.node_features), len(INFRA_TOKEN_FIELDS)),
            dtype=_FLOAT_DTYPE,
        )
        t_now = float(decision_context.t_decision)
        for idx, node in enumerate(infra_features.node_features):
            truck_eta_remaining = (
                max(0.0, float(node.truck_eta) - t_now)
                if node.truck_eta is not None
                else 0.0
            )
            tokens[idx, :] = np.asarray(
                [
                    self._code_norm(_HOST_TYPE_CODE, str(node.node_type)),
                    self._norm_x(float(node.x)),
                    self._norm_y(float(node.y)),
                    self._norm_z(float(node.z)),
                    self._clip01(float(node.queue_length) / max(self._cfg.queue_norm_cap, _TIME_EPS)),
                    self._clip01(
                        float(node.available_slots) / max(float(node.parking_slots), 1.0)
                    ),
                    self._clip01(float(node.parking_slots) / max(self._cfg.queue_norm_cap, 1.0)),
                    self._norm_time_nonneg(float(node.swap_time)),
                    self._norm_time_nonneg(truck_eta_remaining),
                    self._bool(node.truck_eta is not None),
                    self._clip01(float(node.node_charge_load_budget) / max(self._cfg.queue_norm_cap, 1.0)),
                    self._bool(bool(node.is_in_backbone)),
                    self._bool(bool(node.is_launch_candidate_station)),
                    self._clip01(float(infra_features.future_backbone_node_count) / 16.0),
                    self._clip01(float(infra_features.authorized_order_count) / max(self._cfg.max_order_tokens, 1)),
                    self._clip01(float(infra_features.plan_version) / 32.0),
                    self._norm_x(float(infra_features.truck_x)),
                    self._norm_y(float(infra_features.truck_y)),
                    self._norm_z(float(infra_features.truck_z)),
                ],
                dtype=_FLOAT_DTYPE,
            )
        return tokens

    def _build_history_tokens(
        self,
        *,
        transition_history: Sequence[TransitionSummary],
        decision_context: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        hist_len = self._cfg.hist_len
        tokens = np.zeros((hist_len, len(HISTORY_TOKEN_FIELDS)), dtype=_FLOAT_DTYPE)
        padding_mask = np.ones((hist_len,), dtype=_BOOL_DTYPE)
        ego = decision_context.runtime_state.drone_states[decision_context.deciding_drone_id]
        history_tail = list(transition_history[-hist_len:])
        offset = hist_len - len(history_tail)
        for local_idx, item in enumerate(history_tail):
            slot = offset + local_idx
            padding_mask[slot] = False
            tokens[slot, :] = np.asarray(
                [
                    self._norm_time_nonneg(float(decision_context.t_decision) - float(item.event_time)),
                    self._bool(item.actor_drone_id == decision_context.deciding_drone_id),
                    self._norm_distance(float(item.actor_pos_x) - float(ego.current_loc.x)),
                    self._norm_distance(float(item.actor_pos_y) - float(ego.current_loc.y)),
                    self._code_norm(_TRAINING_STATE_CODE, item.actor_training_state_before),
                    self._code_norm(_TRAINING_STATE_CODE, item.actor_training_state_after),
                    self._code_norm(_HOME_TYPE_CODE, item.actor_home_type),
                    self._code_norm(_PAYLOAD_CLASS_CODE, item.actor_payload_class),
                    self._code_norm(
                        _TRIGGER_TYPE_CODE,
                        item.trigger_type,
                        strict=True,
                        field_name="trigger_type",
                    ),
                    self._code_norm(_ROOT_BRANCH_CODE, item.root_branch),
                    self._code_norm(_DISPATCH_MODE_CODE, item.dispatch_mode),
                    self._code_norm(_HOST_TYPE_CODE, item.selected_recover_node_type),
                    self._bool(item.has_selected_order),
                    self._clip_signed(float(item.selected_order_slot_rank) / max(self._cfg.max_order_tokens, 1), -1.0, 1.0),
                    self._clip_signed(float(item.selected_order_deadline_slack_norm), -1.0, 1.0),
                    self._clip01(float(item.selected_eta_to_deliver_norm)),
                    self._clip_signed(float(item.selected_rendezvous_margin_norm), -1.0, 1.0),
                    self._clip01(float(item.energy_ratio_before)),
                    self._clip01(float(item.energy_ratio_after)),
                    self._clip01(float(item.queue_after_norm)),
                    self._norm_plan_version_delta(int(item.plan_version_delta_at_event)),
                    self._bool(item.delivered),
                    self._bool(item.rendezvous_success),
                    self._bool(item.reservation_timeout),
                    self._bool(item.fallback_started),
                    self._bool(item.hard_failure),
                    self._bool(item.queue_entered),
                    self._bool(item.service_completed),
                ],
                dtype=_FLOAT_DTYPE,
            )
        return tokens, padding_mask

    def _infer_queue_after_norm(self, *, runtime_state: Any, drone_state: Any) -> float:
        for node in runtime_state.node_states.values():
            if (
                abs(float(node.position.x) - float(drone_state.current_loc.x)) <= 1e-3
                and abs(float(node.position.y) - float(drone_state.current_loc.y)) <= 1e-3
                and abs(float(node.position.z) - float(drone_state.current_loc.z)) <= 1e-3
            ):
                return self._clip01(float(node.queue_length) / max(self._cfg.queue_norm_cap, _TIME_EPS))
        return 0.0

    def _norm_x(self, value: float) -> float:
        span = max(self._scene_norm.max_x - self._scene_norm.min_x, _TIME_EPS)
        return self._clip01((value - self._scene_norm.min_x) / span)

    def _norm_y(self, value: float) -> float:
        span = max(self._scene_norm.max_y - self._scene_norm.min_y, _TIME_EPS)
        return self._clip01((value - self._scene_norm.min_y) / span)

    def _norm_z(self, value: float) -> float:
        span = max(self._scene_norm.max_z - self._scene_norm.min_z, 1.0)
        return self._clip01((value - self._scene_norm.min_z) / span)

    def _norm_distance(self, value: float) -> float:
        return self._clip_signed(
            float(value) / max(self._scene_norm.diagonal_m, _TIME_EPS),
            -1.0,
            1.0,
        )

    def _norm_energy_absolute(self, value: float, battery_max: float) -> float:
        norm_base = max(float(battery_max), _TIME_EPS)
        return self._clip01(float(value) / norm_base)

    def _energy_ratio_feature(self, value: float) -> float:
        return self._clip01(max(0.0, self._finite_or_zero(float(value))))

    def _norm_time_nonneg(self, value: float) -> float:
        return self._clip01(float(value) / max(self._cfg.upper_horizon_sec, _TIME_EPS))

    def _norm_time_signed(self, value: float) -> float:
        return self._clip_signed(
            float(value) / max(self._cfg.upper_horizon_sec, _TIME_EPS),
            -1.0,
            1.0,
        )

    def _norm_peer_count(self, value: int) -> float:
        return self._clip01(
            float(value) / max(float(self._scene_norm.max_drone_count), 1.0)
        )

    @staticmethod
    def _norm_plan_version_delta(value: int) -> float:
        return ObservationTensorizer._clip_signed(float(value) / 16.0, -1.0, 1.0)

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        result = float(value)
        return result if math.isfinite(result) else 0.0

    @staticmethod
    def _code_norm(
        mapping: Mapping[str, int],
        value: str,
        *,
        strict: bool = False,
        field_name: str = "category",
    ) -> float:
        if not mapping:
            return 0.0
        if strict and value not in mapping:
            raise ValueError(f"未知 {field_name}: {value}")
        denom = max(len(mapping) - 1, 1)
        return float(mapping.get(value, 0)) / float(denom)

    @staticmethod
    def _bool(value: bool | float) -> float:
        return 1.0 if bool(value) else 0.0

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _clip_signed(value: float, lower: float, upper: float) -> float:
        return float(max(lower, min(upper, value)))


def _payload_class(payload_capacity: float) -> str:
    return "light" if payload_capacity <= 2.0 + _TIME_EPS else "heavy"


def _iter_scene_positions(scene_ctx: TrainingSceneContext) -> Iterable[tuple[float, float, float]]:
    for depot in scene_ctx.depots.values():
        yield float(depot.location.x), float(depot.location.y), float(depot.location.z)
    for station in scene_ctx.stations.values():
        yield float(station.location.x), float(station.location.y), float(station.location.z)
    for truck in scene_ctx.trucks.values():
        yield float(truck.current_loc.x), float(truck.current_loc.y), float(truck.current_loc.z)
    for drone in scene_ctx.drones.values():
        yield float(drone.current_loc.x), float(drone.current_loc.y), float(drone.current_loc.z)
    for order in scene_ctx.static_orders:
        yield float(order.delivery_loc.x), float(order.delivery_loc.y), float(order.delivery_loc.z)
    for item in scene_ctx.dynamic_orders:
        yield (
            float(item.order.delivery_loc.x),
            float(item.order.delivery_loc.y),
            float(item.order.delivery_loc.z),
        )


def _build_scene_norm(scene_ctx: TrainingSceneContext) -> _SceneNorm:
    positions = list(_iter_scene_positions(scene_ctx))
    xs = [item[0] for item in positions]
    ys = [item[1] for item in positions]
    zs = [item[2] for item in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    diagonal_m = float(np.hypot(max_x - min_x, max_y - min_y))
    max_speed = max(float(drone.cruise_speed) for drone in scene_ctx.drones.values())
    max_payload_capacity = max(float(drone.payload_capacity) for drone in scene_ctx.drones.values())
    return _SceneNorm(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
        diagonal_m=max(diagonal_m, 1.0),
        max_speed=max(max_speed, 1.0),
        max_payload_capacity=max(max_payload_capacity, 1.0),
        max_drone_count=max(len(scene_ctx.drones), 1),
    )


def _load_tensorizer_config(config_path: Path) -> _TensorizerConfig:
    raw = _load_yaml(config_path)
    planner = _require_mapping(raw, "planner")
    policy = _require_mapping(raw, "policy")
    data = _require_mapping(raw, "data")
    candidate = _require_mapping(raw, "candidate")
    return _TensorizerConfig(
        hist_len=int(policy["hist_len"]),
        upper_horizon_sec=float(planner["upper_horizon_sec"]),
        payload_norm_kg=float(data["poisson_weight_max_kg"]),
        queue_norm_cap=float(policy.get("queue_norm_cap", 8.0)),
        max_order_tokens=int(policy["max_order_tokens"]),
        max_candidate_recovery_per_order=int(candidate["max_candidate_recovery_per_order"]),
    )


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取 observation tensorizer 配置") from exc

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
    "HISTORY_TOKEN_FIELDS",
    "INFRA_TOKEN_FIELDS",
    "ObservationTensorizer",
    "ORDER_TOKEN_FIELDS",
    "UAV_SELF_TOKEN_FIELDS",
]
