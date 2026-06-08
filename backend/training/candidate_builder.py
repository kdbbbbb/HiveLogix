#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 6 候选动作生成器。

输入固定为：
  - RuntimeStateView
  - CoarsePlanView
  - 本次决策的局部上下文

输出固定为：
  - CandidateFeatures
  - factorized masks
  - resolved action lookup
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.entities.primitives import Position3D
from config.loader import load_solver_energy_params

from .actions import DispatchAction, WAIT_ACTION
from .contracts import (
    CandidateFeatures,
    CandidateOutput,
    CandidateTeacherEnergy,
    CoarsePlanView,
    FactorizedActionSchema,
    InfraFeatures,
    InfraNodeFeatures,
    OrderFeatures,
    PolicyMode,
    ResolvedActionLookup,
    UavSelfFeatures,
)
from .scene_loader import DEFAULT_CONFIG_PATH, TrainingSceneContext, load_default_scene
from .uav_path_service import TrainingUavPathService


_TIME_EPS = 1e-6
_MODE_B_IDX = 0
_MODE_C_IDX = 1
_DEADLINE_LEAD_TIME_SEC = 480.0
_OVERDUE_SCORE_CAP_SEC = 600.0
_OLD_OVERDUE_RESERVE_SLOTS = 2


@dataclass(frozen=True)
class _CandidateConfig:
    max_candidate_orders: int
    max_candidate_actions: int
    station_wait_threshold_sec: float
    rendezvous_filter_margin_sec: float
    rendezvous_execution_margin_sec: float
    rendezvous_max_wait_sec: float
    upper_horizon_sec: float


@dataclass(frozen=True)
class _ModeCSummary:
    candidate_count: int
    best_node_id: str
    best_truck_arrival_time: float
    best_uav_arrival_time: float
    best_rendezvous_margin: float
    best_wait_time: float
    best_uav_flight_time: float
    best_recovery_energy_j: float
    best_recovery_energy_ratio: float
    best_total_energy_ratio: float
    best_energy_margin_ratio: float
    best_node_type: str
    best_truck_eta_remaining: float
    timeout_risk: float


class CandidateBuilder:
    """无状态候选构建器。"""

    def __init__(
        self,
        *,
        scene_ctx: TrainingSceneContext | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        uav_path_service: TrainingUavPathService | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._cfg = _load_candidate_config(self._config_path)
        self._scene_ctx = scene_ctx or load_default_scene(config_path=self._config_path)
        self._uav_path_service = uav_path_service or TrainingUavPathService(
            scene_ctx=self._scene_ctx
        )
        solver_params = load_solver_energy_params()
        self._truck_drone_launch_time_s = float(solver_params.truck_drone_launch_time_s)
        self._drone_service_time_order_s = float(solver_params.drone_service_time_order_s)
        self._safe_margin_j_by_drone = {
            drone_id: float(drone.safe_margin_j)
            for drone_id, drone in self._scene_ctx.entity_manager.drones.items()
        }

    def build(
        self,
        runtime_state: Any,
        coarse_plan: CoarsePlanView,
        deciding_drone_id: str,
        trigger_type: str,
        trigger_station_id: str | None,
        last_seen_plan_version: int,
    ) -> CandidateOutput:
        _validate_trigger_context(
            runtime_state=runtime_state,
            trigger_type=trigger_type,
            trigger_station_id=trigger_station_id,
        )
        drone_view = runtime_state.drone_states[deciding_drone_id]
        launch_pos = self._resolve_launch_position(
            runtime_state=runtime_state,
            drone_view=drone_view,
            trigger_type=trigger_type,
            trigger_station_id=trigger_station_id,
        )
        effective_launch_time = self._resolve_effective_launch_time(
            runtime_state=runtime_state,
            trigger_type=trigger_type,
        )

        mode_c_order_filter_counts = _new_counter(
            (
                "authorized_orders",
                "pending_missing",
                "policy_mode_missing",
                "policy_c_allowed",
                "payload_over_uav_capacity",
                "delivery_energy_insufficient",
                "delivery_feasible_orders",
                "no_recovery_pool",
                "no_feasible_recovery_node",
                "feasible_mode_c_order_pre_trunc",
                "feasible_mode_c_order_post_trunc",
                "feasible_mode_c_order_truncated",
                "actionable_order_pre_trunc",
                "actionable_order_post_trunc",
            )
        )
        mode_c_node_filter_counts = _new_counter(
            (
                "recovery_node_considered",
                "node_missing",
                "truck_eta_missing",
                "truck_before_delivery_finish",
                "truck_too_early",
                "truck_too_late",
                "energy_to_recover_insufficient",
                "feasible",
            )
        )
        teacher_energy_debug: list[str] = []

        actionable_orders: list[dict[str, Any]] = []
        for order_id in coarse_plan.authorized_orders:
            mode_c_order_filter_counts["authorized_orders"] += 1
            order = runtime_state.pending_orders.get(order_id)
            if order is None:
                mode_c_order_filter_counts["pending_missing"] += 1
                continue
            allowed_policy_modes = coarse_plan.get_policy_modes(order_id)
            if not allowed_policy_modes:
                mode_c_order_filter_counts["policy_mode_missing"] += 1
                continue
            if PolicyMode.C in allowed_policy_modes:
                mode_c_order_filter_counts["policy_c_allowed"] += 1
            if float(order.payload_weight) > float(drone_view.payload_capacity):
                mode_c_order_filter_counts["payload_over_uav_capacity"] += 1
                continue

            delivery_leg = self._estimate_uav_leg(
                drone_view=drone_view,
                from_pos=launch_pos,
                to_pos=order.delivery_loc,
                payload=float(order.payload_weight),
            )
            t_deliver_arrive = effective_launch_time + delivery_leg.flight_time_sec
            t_deliver_finish = (
                t_deliver_arrive + self._drone_service_time_order_s
            )
            energy_to_deliver = delivery_leg.energy_j
            if energy_to_deliver > float(drone_view.battery_current) + _TIME_EPS:
                mode_c_order_filter_counts["delivery_energy_insufficient"] += 1
                continue
            mode_c_order_filter_counts["delivery_feasible_orders"] += 1

            energy_after_delivery = float(drone_view.battery_current) - energy_to_deliver
            battery_max = float(getattr(drone_view, "battery_max", 0.0))
            delivery_energy_ratio = _teacher_energy_ratio(
                energy_to_deliver,
                battery_max,
                debug_messages=teacher_energy_debug,
                label=f"order={order_id}:delivery",
            )
            mode_b_summary = None
            if PolicyMode.B in allowed_policy_modes:
                mode_b_summary = self._select_best_mode_b_host(
                    runtime_state=runtime_state,
                    drone_view=drone_view,
                    deliver_pos=order.delivery_loc,
                    t_deliver_finish=t_deliver_finish,
                    battery_after_delivery=energy_after_delivery,
                )

            mode_c_summary: _ModeCSummary | None = None
            if PolicyMode.C in allowed_policy_modes:
                mode_c_summary = self._select_best_mode_c_recovery(
                    runtime_state=runtime_state,
                    coarse_plan=coarse_plan,
                    drone_view=drone_view,
                    order=order,
                    deliver_pos=order.delivery_loc,
                    t_deliver_finish=t_deliver_finish,
                    energy_after_delivery=energy_after_delivery,
                    delivery_energy_ratio=delivery_energy_ratio,
                    trigger_type=trigger_type,
                    trigger_station_id=trigger_station_id,
                    t_now=float(runtime_state.t_now),
                    mode_c_node_filter_counts=mode_c_node_filter_counts,
                )
                if mode_c_summary is None:
                    if _recovery_pool_for_context(
                        coarse_plan=coarse_plan,
                        order_id=order.order_id,
                        trigger_type=trigger_type,
                        trigger_station_id=trigger_station_id,
                    ):
                        mode_c_order_filter_counts["no_feasible_recovery_node"] += 1
                    else:
                        mode_c_order_filter_counts["no_recovery_pool"] += 1

            if mode_b_summary is None and mode_c_summary is None:
                continue
            mode_c_order_filter_counts["actionable_order_pre_trunc"] += 1
            if mode_c_summary is not None:
                mode_c_order_filter_counts["feasible_mode_c_order_pre_trunc"] += 1
            mode_b_recovery_energy_ratio = (
                _teacher_energy_ratio(
                    float(mode_b_summary["recovery_energy_j"]),
                    battery_max,
                    debug_messages=teacher_energy_debug,
                    label=f"order={order_id}:mode_b_recovery",
                )
                if mode_b_summary is not None
                else 0.0
            )
            mode_c_recovery_energy_ratio = (
                _teacher_energy_ratio(
                    float(mode_c_summary.best_recovery_energy_j),
                    battery_max,
                    debug_messages=teacher_energy_debug,
                    label=f"order={order_id}:mode_c_recovery",
                )
                if mode_c_summary is not None
                else 0.0
            )
            teacher_energy = CandidateTeacherEnergy(
                delivery_energy_ratio=float(delivery_energy_ratio),
                best_mode_b_recovery_energy_ratio=float(mode_b_recovery_energy_ratio),
                best_mode_c_recovery_energy_ratio=float(mode_c_recovery_energy_ratio),
                best_mode_b_total_energy_ratio=_sum_teacher_energy_ratios(
                    delivery_energy_ratio,
                    mode_b_recovery_energy_ratio,
                    debug_messages=teacher_energy_debug,
                    label=f"order={order_id}:mode_b_total",
                )
                if mode_b_summary is not None
                else 0.0,
                best_mode_c_total_energy_ratio=_sum_teacher_energy_ratios(
                    delivery_energy_ratio,
                    mode_c_recovery_energy_ratio,
                    debug_messages=teacher_energy_debug,
                    label=f"order={order_id}:mode_c_total",
                )
                if mode_c_summary is not None
                else 0.0,
                mode_c_energy_saving_ratio=_teacher_energy_saving_ratio(
                    mode_b_summary is not None,
                    mode_c_summary is not None,
                    _sum_teacher_energy_ratios(
                        delivery_energy_ratio,
                        mode_b_recovery_energy_ratio,
                        debug_messages=teacher_energy_debug,
                        label=f"order={order_id}:mode_b_total_for_saving",
                    ),
                    _sum_teacher_energy_ratios(
                        delivery_energy_ratio,
                        mode_c_recovery_energy_ratio,
                        debug_messages=teacher_energy_debug,
                        label=f"order={order_id}:mode_c_total_for_saving",
                    ),
                    debug_messages=teacher_energy_debug,
                    label=f"order={order_id}:mode_c_saving",
                ),
            )

            order_feature = OrderFeatures(
                order_id=order_id,
                weight=float(order.payload_weight),
                deadline=float(order.deadline),
                remaining_time=float(order.deadline) - float(runtime_state.t_now),
                delivery_x=float(order.delivery_loc.x),
                delivery_y=float(order.delivery_loc.y),
                delivery_z=float(order.delivery_loc.z),
                distance_to_order=float(delivery_leg.distance_m),
                order_pre_score=float(coarse_plan.order_pre_score[order_id]),
                priority_band=int(coarse_plan.order_priority_band[order_id]),
                has_mode_b_action=mode_b_summary is not None,
                best_mode_b_return_score=(
                    float(mode_b_summary["score"]) if mode_b_summary is not None else 0.0
                ),
                best_mode_b_recovery_flight_time=(
                    float(mode_b_summary["recovery_flight_time"])
                    if mode_b_summary is not None
                    else 0.0
                ),
                best_mode_b_host_type=(
                    str(mode_b_summary["host_type"]) if mode_b_summary is not None else ""
                ),
                best_mode_b_queue_time_est=(
                    float(mode_b_summary["queue_time_est"])
                    if mode_b_summary is not None
                    else 0.0
                ),
                has_mode_c_action=mode_c_summary is not None,
                mode_c_candidate_count=(
                    int(mode_c_summary.candidate_count)
                    if mode_c_summary is not None
                    else 0
                ),
                best_mode_c_rendezvous_margin=(
                    float(mode_c_summary.best_rendezvous_margin)
                    if mode_c_summary is not None
                    else 0.0
                ),
                best_mode_c_wait_time=(
                    float(mode_c_summary.best_wait_time)
                    if mode_c_summary is not None
                    else 0.0
                ),
                best_mode_c_uav_flight_time=(
                    float(mode_c_summary.best_uav_flight_time)
                    if mode_c_summary is not None
                    else 0.0
                ),
                best_mode_c_energy_margin_ratio=(
                    float(mode_c_summary.best_energy_margin_ratio)
                    if mode_c_summary is not None
                    else 0.0
                ),
                best_mode_c_node_type=(
                    str(mode_c_summary.best_node_type)
                    if mode_c_summary is not None
                    else ""
                ),
                best_mode_c_truck_eta_remaining=(
                    float(mode_c_summary.best_truck_eta_remaining)
                    if mode_c_summary is not None
                    else 0.0
                ),
                best_mode_c_timeout_risk=(
                    float(mode_c_summary.timeout_risk)
                    if mode_c_summary is not None
                    else 0.0
                ),
                is_valid=True,
                delivery_energy_ratio=float(teacher_energy.delivery_energy_ratio),
                best_mode_b_recovery_energy_ratio=float(
                    teacher_energy.best_mode_b_recovery_energy_ratio
                ),
                best_mode_c_recovery_energy_ratio=float(
                    teacher_energy.best_mode_c_recovery_energy_ratio
                ),
                best_mode_b_total_energy_ratio=float(
                    teacher_energy.best_mode_b_total_energy_ratio
                ),
                best_mode_c_total_energy_ratio=float(
                    teacher_energy.best_mode_c_total_energy_ratio
                ),
                mode_c_energy_saving_ratio=float(
                    teacher_energy.mode_c_energy_saving_ratio
                ),
                estimated_delivery_finish_slack_sec=float(order.deadline) - t_deliver_finish,
            )
            actionable_orders.append(
                {
                    "order_id": order_id,
                    "order_feature": order_feature,
                    "has_mode_b": mode_b_summary is not None,
                    "has_mode_c": mode_c_summary is not None,
                    "mode_c_recover_node_id": (
                        mode_c_summary.best_node_id
                        if mode_c_summary is not None
                        else None
                    ),
                    "teacher_energy": teacher_energy,
                }
            )

        pre_trunc_mode_c_order_count = sum(
            1 for item in actionable_orders if bool(item["has_mode_c"])
        )
        actionable_orders = _select_candidate_orders(
            actionable_orders,
            max_candidate_orders=self._cfg.max_candidate_orders,
        )
        post_trunc_mode_c_order_count = sum(
            1 for item in actionable_orders if bool(item["has_mode_c"])
        )
        mode_c_order_filter_counts["actionable_order_post_trunc"] += len(actionable_orders)
        mode_c_order_filter_counts["feasible_mode_c_order_post_trunc"] += (
            post_trunc_mode_c_order_count
        )
        mode_c_order_filter_counts["feasible_mode_c_order_truncated"] += max(
            0,
            int(pre_trunc_mode_c_order_count) - int(post_trunc_mode_c_order_count),
        )

        order_features: list[OrderFeatures] = []
        order_mask: list[bool] = []
        mode_mask: list[tuple[bool, bool]] = []
        dispatch_actions: dict[tuple[int, int], DispatchAction] = {}
        teacher_energy_by_order_idx: dict[int, CandidateTeacherEnergy] = {}

        for order_slot, item in enumerate(actionable_orders):
            order_features.append(item["order_feature"])
            order_mask.append(True)
            teacher_energy_by_order_idx[int(order_slot)] = item["teacher_energy"]
            has_mode_b = bool(item["has_mode_b"])
            has_mode_c = bool(item["has_mode_c"])
            mode_mask.append((has_mode_b, has_mode_c))

            if has_mode_b:
                dispatch_actions[(order_slot, _MODE_B_IDX)] = DispatchAction(
                    order_id=item["order_id"],
                    mode=PolicyMode.B,
                )
            if has_mode_c:
                recover_node_id = item["mode_c_recover_node_id"]
                if recover_node_id is None:
                    raise RuntimeError(
                        "mode C 候选缺少已解析的 best recover node: "
                        f"order_id={item['order_id']}"
                    )
                dispatch_actions[(order_slot, _MODE_C_IDX)] = DispatchAction(
                    order_id=item["order_id"],
                    mode=PolicyMode.C,
                    recover_node_id=str(recover_node_id),
                )

        while len(order_features) < self._cfg.max_candidate_orders:
            order_features.append(_padding_order_feature())
            order_mask.append(False)
            mode_mask.append((False, False))

        action_count = 1 + len(dispatch_actions)
        if action_count > self._cfg.max_candidate_actions:
            raise RuntimeError(
                "resolved_action_lookup 展开规模超出预算: "
                f"{action_count} > {self._cfg.max_candidate_actions}"
            )

        candidate_features = CandidateFeatures(
            uav_self=self._build_uav_self_features(
                runtime_state=runtime_state,
                coarse_plan=coarse_plan,
                drone_view=drone_view,
                last_seen_plan_version=last_seen_plan_version,
            ),
            order_features=tuple(order_features),
            infra_features=self._build_infra_features(runtime_state, coarse_plan),
        )
        return CandidateOutput(
            candidate_features=candidate_features,
            root_branch_mask=(True, bool(dispatch_actions)),
            has_wait_action=True,
            order_mask=tuple(order_mask),
            mode_mask=tuple(mode_mask),
            factorized_action_schema=FactorizedActionSchema(
                root_branch_order=("WAIT", "DISPATCH"),
                mode_order=("B", "C"),
                max_order_slots=self._cfg.max_candidate_orders,
            ),
            resolved_action_lookup=ResolvedActionLookup(
                wait_action=WAIT_ACTION,
                dispatch_actions=dispatch_actions,
            ),
            teacher_energy_by_order_idx=teacher_energy_by_order_idx,
            diagnostics={
                "mode_c_order_filter_counts": dict(mode_c_order_filter_counts),
                "mode_c_node_filter_counts": dict(mode_c_node_filter_counts),
                "teacher_energy_debug": tuple(teacher_energy_debug),
            },
        )

    def build_from_decision_context(
        self,
        decision_context: Any,
        *,
        last_seen_plan_version: int,
    ) -> CandidateOutput:
        """基于一次既有决策快照重建 `CandidateOutput`。

        该入口保持 `CandidateOutput` 仍由 env 外部调用方持有，不挂在
        `DecisionContext` / `EnvStepResult` 上；同时避免外部重新手拼
        `runtime_state` / `coarse_plan` / trigger 字段。
        """
        try:
            runtime_state = decision_context.runtime_state
            coarse_plan = decision_context.coarse_plan
            deciding_drone_id = decision_context.deciding_drone_id
            trigger_type = decision_context.trigger_type
            trigger_station_id = decision_context.trigger_station_id
        except AttributeError as exc:
            raise TypeError(
                "decision_context 必须暴露 runtime_state / coarse_plan / "
                "deciding_drone_id / trigger_type / trigger_station_id"
            ) from exc

        return self.build(
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            deciding_drone_id=deciding_drone_id,
            trigger_type=trigger_type,
            trigger_station_id=trigger_station_id,
            last_seen_plan_version=last_seen_plan_version,
        )

    def _resolve_launch_position(
        self,
        *,
        runtime_state: Any,
        drone_view: Any,
        trigger_type: str,
        trigger_station_id: str | None,
    ) -> Position3D:
        if (
            _is_riding_with_truck_trigger(trigger_type)
            and trigger_station_id
            and trigger_station_id in runtime_state.node_states
        ):
            return runtime_state.node_states[trigger_station_id].position
        return drone_view.current_loc

    def _resolve_effective_launch_time(
        self,
        *,
        runtime_state: Any,
        trigger_type: str,
    ) -> float:
        effective_launch_time = float(runtime_state.t_now)
        if _is_riding_with_truck_trigger(trigger_type):
            effective_launch_time += self._truck_drone_launch_time_s
        return effective_launch_time

    def _estimate_uav_leg(
        self,
        *,
        drone_view: Any,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
    ):
        return self._uav_path_service.estimate(
            drone=drone_view,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=payload,
        )

    def _estimate_flight_time(
        self,
        *,
        drone_view: Any,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> float:
        return self._estimate_uav_leg(
            drone_view=drone_view,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=0.0,
        ).flight_time_sec

    def _can_reach(
        self,
        *,
        drone_view: Any,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
        safe_margin: float,
        battery_current: float,
    ) -> bool:
        return self._uav_path_service.can_reach(
            drone=drone_view,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=payload,
            safe_margin=safe_margin,
            battery_current=battery_current,
        )

    def _path_distance(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> float:
        return self._uav_path_service.path_distance(
            from_pos=from_pos,
            to_pos=to_pos,
        )

    def _select_best_mode_c_recovery(
        self,
        *,
        runtime_state: Any,
        coarse_plan: CoarsePlanView,
        drone_view: Any,
        order: Any,
        deliver_pos: Position3D,
        t_deliver_finish: float,
        energy_after_delivery: float,
        delivery_energy_ratio: float,
        trigger_type: str,
        trigger_station_id: str | None,
        t_now: float,
        mode_c_node_filter_counts: dict[str, int] | None = None,
    ) -> _ModeCSummary | None:
        if _is_riding_with_truck_trigger(trigger_type) and trigger_station_id:
            recovery_pool = self._runtime_recovery_pool(
                coarse_plan=coarse_plan,
                trigger_station_id=trigger_station_id,
            )
        else:
            recovery_pool = coarse_plan.get_recovery_candidates(order.order_id)

        safe_margin = self._safe_margin_j_by_drone[drone_view.drone_id]
        battery_max = max(float(getattr(drone_view, "battery_max", 0.0)), _TIME_EPS)
        feasible_count = 0
        best_item: tuple[float, float, float, str, dict[str, Any]] | None = None
        for node_id in recovery_pool:
            _increment_counter(mode_c_node_filter_counts, "recovery_node_considered")
            node_state = runtime_state.node_states.get(node_id)
            t_arrive_truck = coarse_plan.truck_eta_map.get(node_id)
            if node_state is None:
                _increment_counter(mode_c_node_filter_counts, "node_missing")
                continue
            if t_arrive_truck is None:
                _increment_counter(mode_c_node_filter_counts, "truck_eta_missing")
                continue
            if float(t_arrive_truck) <= float(t_deliver_finish) + _TIME_EPS:
                _increment_counter(mode_c_node_filter_counts, "truck_before_delivery_finish")
                continue

            recover_leg = self._estimate_uav_leg(
                drone_view=drone_view,
                from_pos=deliver_pos,
                to_pos=node_state.position,
                payload=0.0,
            )
            uav_flight_time = float(recover_leg.flight_time_sec)
            energy_to_recover = float(recover_leg.energy_j)
            recovery_energy_ratio = (
                float(energy_to_recover) / battery_max
                if battery_max > _TIME_EPS
                else 0.0
            )
            if not math.isfinite(recovery_energy_ratio) or recovery_energy_ratio < 0.0:
                recovery_energy_ratio = 0.0
            total_energy_ratio = float(delivery_energy_ratio) + float(recovery_energy_ratio)
            if not math.isfinite(total_energy_ratio) or total_energy_ratio < 0.0:
                total_energy_ratio = 0.0
            energy_margin_j = (
                float(energy_after_delivery)
                - energy_to_recover
                - float(safe_margin)
            )
            if energy_margin_j < -_TIME_EPS:
                _increment_counter(mode_c_node_filter_counts, "energy_to_recover_insufficient")
                continue

            t_arrive_uav = t_deliver_finish + uav_flight_time
            planned_wait = float(t_arrive_truck) - float(t_arrive_uav)
            if planned_wait < self._cfg.rendezvous_execution_margin_sec - _TIME_EPS:
                _increment_counter(mode_c_node_filter_counts, "truck_too_early")
                continue
            if planned_wait > self._cfg.rendezvous_max_wait_sec + _TIME_EPS:
                _increment_counter(mode_c_node_filter_counts, "truck_too_late")
                continue

            recovery_time = float(t_arrive_truck)
            rendezvous_margin = float(
                planned_wait - self._cfg.rendezvous_execution_margin_sec
            )
            feasible_count += 1
            _increment_counter(mode_c_node_filter_counts, "feasible")
            item = (
                recovery_time,
                float(planned_wait),
                float(uav_flight_time),
                str(node_id),
                {
                    "node_type": str(node_state.node_type),
                    "node_id": str(node_id),
                    "truck_arrival_time": float(t_arrive_truck),
                    "uav_arrival_time": float(t_arrive_uav),
                    "truck_eta_remaining": max(
                        0.0,
                        float(t_arrive_truck) - float(t_now),
                    ),
                    "rendezvous_margin": rendezvous_margin,
                    "wait_time": float(planned_wait),
                    "uav_flight_time": float(uav_flight_time),
                    "recovery_energy_j": float(energy_to_recover),
                    "recovery_energy_ratio": max(0.0, float(recovery_energy_ratio)),
                    "total_energy_ratio": max(0.0, float(total_energy_ratio)),
                    "energy_margin_ratio": max(0.0, energy_margin_j / battery_max),
                },
            )
            if best_item is None or item[:4] < best_item[:4]:
                best_item = item

        if best_item is None:
            return None

        best = best_item[4]
        best_margin = float(best["rendezvous_margin"])
        timeout_risk = 1.0 - min(
            1.0,
            max(0.0, best_margin)
            / max(float(self._cfg.rendezvous_max_wait_sec), _TIME_EPS),
        )
        return _ModeCSummary(
            candidate_count=int(feasible_count),
            best_node_id=str(best["node_id"]),
            best_truck_arrival_time=float(best["truck_arrival_time"]),
            best_uav_arrival_time=float(best["uav_arrival_time"]),
            best_rendezvous_margin=best_margin,
            best_wait_time=float(best["wait_time"]),
            best_uav_flight_time=float(best["uav_flight_time"]),
            best_recovery_energy_j=float(best["recovery_energy_j"]),
            best_recovery_energy_ratio=float(best["recovery_energy_ratio"]),
            best_total_energy_ratio=float(best["total_energy_ratio"]),
            best_energy_margin_ratio=float(best["energy_margin_ratio"]),
            best_node_type=str(best["node_type"]),
            best_truck_eta_remaining=float(best["truck_eta_remaining"]),
            timeout_risk=float(timeout_risk),
        )

    def _runtime_recovery_pool(
        self,
        *,
        coarse_plan: CoarsePlanView,
        trigger_station_id: str,
    ) -> tuple[str, ...]:
        if trigger_station_id not in coarse_plan.route_drift_ref:
            return coarse_plan.truck_backbone_route
        trigger_idx = coarse_plan.get_route_position(trigger_station_id)
        return tuple(
            node_id
            for node_id in coarse_plan.truck_backbone_route
            if coarse_plan.get_route_position(node_id) >= trigger_idx
        )

    def _select_best_mode_b_host(
        self,
        *,
        runtime_state: Any,
        drone_view: Any,
        deliver_pos: Position3D,
        t_deliver_finish: float,
        battery_after_delivery: float,
    ) -> dict[str, Any] | None:
        safe_margin = self._safe_margin_j_by_drone[drone_view.drone_id]
        depot_nodes = [
            node_state
            for node_state in runtime_state.node_states.values()
            if str(node_state.node_type) == "depot"
        ]
        depot_nodes.sort(key=lambda item: str(item.node_id))
        if depot_nodes:
            depot_node = depot_nodes[0]
            recover_leg = self._estimate_uav_leg(
                drone_view=drone_view,
                from_pos=deliver_pos,
                to_pos=depot_node.position,
                payload=0.0,
            )
            if (
                float(battery_after_delivery) + _TIME_EPS
                >= float(recover_leg.energy_j) + float(safe_margin)
            ):
                fly_time = float(recover_leg.flight_time_sec)
                return {
                    "score": float(t_deliver_finish + fly_time),
                    "recovery_flight_time": float(fly_time),
                    "recovery_energy_j": float(recover_leg.energy_j),
                    "queue_time_est": 0.0,
                    "host_type": "depot",
                }

        if not depot_nodes:
            return None
        depot_node = depot_nodes[0]
        scored_hosts: list[tuple[float, float, float, str, str, float, float]] = []
        for node_state in runtime_state.node_states.values():
            if str(node_state.node_type) != "station":
                continue
            station_leg = self._estimate_uav_leg(
                drone_view=drone_view,
                from_pos=deliver_pos,
                to_pos=node_state.position,
                payload=0.0,
            )
            if (
                float(battery_after_delivery) + _TIME_EPS
                < float(station_leg.energy_j) + float(safe_margin)
            ):
                continue
            depot_leg = self._estimate_uav_leg(
                drone_view=drone_view,
                from_pos=node_state.position,
                to_pos=depot_node.position,
                payload=0.0,
            )
            if (
                float(drone_view.battery_max) + _TIME_EPS
                < float(depot_leg.energy_j) + float(safe_margin)
            ):
                continue

            fly_time = float(station_leg.flight_time_sec)
            depot_fly_time = float(depot_leg.flight_time_sec)
            queue_time_est = _predicted_queue_time_est(node_state)
            service_time = float(node_state.swap_time)
            score = (
                t_deliver_finish
                + fly_time
                + queue_time_est
                + service_time
                + depot_fly_time
            )
            recovery_flight_time = float(fly_time) + float(depot_fly_time)
            recovery_energy_j = float(station_leg.energy_j) + float(depot_leg.energy_j)
            scored_hosts.append(
                (
                    self._path_distance(
                        from_pos=deliver_pos,
                        to_pos=node_state.position,
                    ),
                    score,
                    queue_time_est,
                    str(node_state.node_id),
                    str(node_state.node_type),
                    recovery_flight_time,
                    recovery_energy_j,
                )
            )

        if not scored_hosts:
            return None

        scored_hosts.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        (
            _distance,
            score,
            queue_time_est,
            _node_id,
            host_type,
            recovery_flight_time,
            recovery_energy_j,
        ) = scored_hosts[0]
        return {
            "score": float(score),
            "recovery_flight_time": float(recovery_flight_time),
            "recovery_energy_j": float(recovery_energy_j),
            "queue_time_est": float(queue_time_est),
            "host_type": host_type,
        }

    def _build_uav_self_features(
        self,
        *,
        runtime_state: Any,
        coarse_plan: CoarsePlanView,
        drone_view: Any,
        last_seen_plan_version: int,
    ) -> UavSelfFeatures:
        reservation = drone_view.reservation
        return UavSelfFeatures(
            drone_id=str(drone_view.drone_id),
            x=float(drone_view.current_loc.x),
            y=float(drone_view.current_loc.y),
            z=float(drone_view.current_loc.z),
            battery_current=float(drone_view.battery_current),
            battery_max=float(drone_view.battery_max),
            battery_ratio=float(drone_view.battery_ratio),
            training_state=str(drone_view.training_state),
            has_reservation=reservation is not None,
            reservation_remaining_sec=(
                max(
                    0.0,
                    float(coarse_plan.truck_eta_map.get(reservation.recover_node, runtime_state.t_now))
                    - float(runtime_state.t_now),
                )
                if reservation is not None
                else 0.0
            ),
            plan_version_delta=int(coarse_plan.plan_version - last_seen_plan_version),
            is_riding_truck=str(drone_view.training_state) == "riding_with_truck",
            drone_source_type=str(drone_view.home_type),
            cruise_speed=float(drone_view.cruise_speed),
            payload_capacity=float(drone_view.payload_capacity),
        )

    def _build_infra_features(
        self,
        runtime_state: Any,
        coarse_plan: CoarsePlanView,
    ) -> InfraFeatures:
        node_features = tuple(
            InfraNodeFeatures(
                node_id=str(node_id),
                node_type=str(node_state.node_type),
                x=float(node_state.position.x),
                y=float(node_state.position.y),
                z=float(node_state.position.z),
                queue_length=int(node_state.queue_length),
                available_slots=int(node_state.available_slots),
                parking_slots=int(node_state.parking_slots),
                swap_time=float(node_state.swap_time),
                truck_eta=(
                    float(coarse_plan.truck_eta_map[node_id])
                    if node_id in coarse_plan.truck_eta_map
                    else None
                ),
                node_charge_load_budget=int(
                    coarse_plan.node_charge_load_budget.get(node_id, 0)
                ),
                is_in_backbone=node_id in coarse_plan.truck_eta_map,
                is_launch_candidate_station=coarse_plan.is_launch_candidate_station(
                    node_id
                ),
            )
            for node_id, node_state in sorted(runtime_state.node_states.items())
        )
        return InfraFeatures(
            truck_x=float(runtime_state.truck_current_loc.x),
            truck_y=float(runtime_state.truck_current_loc.y),
            truck_z=float(runtime_state.truck_current_loc.z),
            plan_version=int(coarse_plan.plan_version),
            future_backbone_node_count=len(coarse_plan.truck_backbone_route),
            authorized_order_count=len(coarse_plan.authorized_orders),
            node_features=node_features,
        )


def _predicted_queue_time_est(node_state: Any) -> float:
    if int(node_state.available_slots) > 0:
        return 0.0
    return (int(node_state.queue_length) + 1) * float(node_state.swap_time)


def _teacher_energy_ratio(
    energy_j: float,
    battery_max: float,
    *,
    debug_messages: list[str],
    label: str,
) -> float:
    energy = float(energy_j)
    battery = float(battery_max)
    if not math.isfinite(battery) or battery <= 0.0:
        debug_messages.append(
            f"{label}: invalid battery_max={battery!r}; teacher energy ratio set to 0"
        )
        return 0.0
    if not math.isfinite(energy):
        debug_messages.append(
            f"{label}: non-finite energy_j={energy!r}; teacher energy ratio set to 0"
        )
        return 0.0
    if energy < 0.0:
        debug_messages.append(
            f"{label}: negative energy_j={energy!r}; teacher energy ratio set to 0"
        )
        return 0.0
    ratio = energy / battery
    if not math.isfinite(ratio):
        debug_messages.append(
            f"{label}: non-finite energy ratio={ratio!r}; teacher energy ratio set to 0"
        )
        return 0.0
    return float(max(0.0, ratio))


def _sum_teacher_energy_ratios(
    *ratios: float,
    debug_messages: list[str],
    label: str,
) -> float:
    total = 0.0
    for ratio in ratios:
        value = float(ratio)
        if not math.isfinite(value) or value < 0.0:
            debug_messages.append(
                f"{label}: invalid component ratio={value!r}; total ratio set to 0"
            )
            return 0.0
        total += value
    if not math.isfinite(total):
        debug_messages.append(
            f"{label}: non-finite total ratio={total!r}; total ratio set to 0"
        )
        return 0.0
    return float(max(0.0, total))


def _teacher_energy_saving_ratio(
    has_mode_b: bool,
    has_mode_c: bool,
    mode_b_total_energy_ratio: float,
    mode_c_total_energy_ratio: float,
    *,
    debug_messages: list[str],
    label: str,
) -> float:
    if not has_mode_b or not has_mode_c:
        return 0.0
    b_total = float(mode_b_total_energy_ratio)
    c_total = float(mode_c_total_energy_ratio)
    if (
        not math.isfinite(b_total)
        or not math.isfinite(c_total)
        or b_total < 0.0
        or c_total < 0.0
    ):
        debug_messages.append(
            f"{label}: invalid total ratios b={b_total!r}, c={c_total!r}; saving set to 0"
        )
        return 0.0
    saving = b_total - c_total
    if not math.isfinite(saving):
        debug_messages.append(
            f"{label}: non-finite saving ratio={saving!r}; saving set to 0"
        )
        return 0.0
    return float(saving)


def _new_counter(keys: tuple[str, ...]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _increment_counter(counter: dict[str, int] | None, key: str, amount: int = 1) -> None:
    if counter is None:
        return
    counter[key] = int(counter.get(key, 0)) + int(amount)


def _recovery_pool_for_context(
    *,
    coarse_plan: CoarsePlanView,
    order_id: str,
    trigger_type: str,
    trigger_station_id: str | None,
) -> tuple[str, ...]:
    if _is_riding_with_truck_trigger(trigger_type) and trigger_station_id:
        if trigger_station_id not in coarse_plan.route_drift_ref:
            return coarse_plan.truck_backbone_route
        trigger_idx = coarse_plan.get_route_position(trigger_station_id)
        return tuple(
            node_id
            for node_id in coarse_plan.truck_backbone_route
            if coarse_plan.get_route_position(node_id) >= trigger_idx
        )
    return coarse_plan.get_recovery_candidates(str(order_id))


def _exact_mode_c_score(
    *,
    rendezvous_margin: float,
    truck_eta_remaining: float,
    upper_horizon_sec: float,
) -> float:
    norm_base = max(float(upper_horizon_sec), _TIME_EPS)
    normalized_margin = float(rendezvous_margin) / norm_base
    normalized_eta = float(truck_eta_remaining) / norm_base
    return normalized_margin - 0.05 * normalized_eta


def _is_riding_with_truck_trigger(trigger_type: str) -> bool:
    """兼容文档对外语义名与 env 内部事件名。"""
    return trigger_type in {"riding_with_truck", "truck_station_arrival"}


def _candidate_deadline_sort_key(
    item: Mapping[str, Any],
) -> tuple[float, int, int, float, str]:
    order_feature = item["order_feature"]
    slack = float(order_feature.remaining_time)
    lead = _DEADLINE_LEAD_TIME_SEC
    if slack >= 0.0:
        deadline_score = abs(slack - lead)
        overdue_rank = 1
        tail_score = max(0.0, slack - lead)
    else:
        lateness = -slack
        deadline_score = lead + min(lateness, _OVERDUE_SCORE_CAP_SEC)
        overdue_rank = 0
        tail_score = lateness
    return (
        deadline_score,
        overdue_rank,
        int(order_feature.priority_band),
        tail_score,
        str(item["order_id"]),
    )


def _select_candidate_orders(
    actionable_orders: list[dict[str, Any]],
    *,
    max_candidate_orders: int,
) -> list[dict[str, Any]]:
    sorted_actionable_orders = sorted(
        actionable_orders,
        key=_candidate_deadline_sort_key,
    )
    if len(sorted_actionable_orders) <= max_candidate_orders:
        return sorted_actionable_orders
    if max_candidate_orders <= 0:
        return []

    reserve_slots = min(_OLD_OVERDUE_RESERVE_SLOTS, max_candidate_orders)
    main_slots = max_candidate_orders - reserve_slots
    main_selected = sorted_actionable_orders[:main_slots]
    selected_order_ids = {str(item["order_id"]) for item in main_selected}

    overdue_candidates = [
        item
        for item in sorted_actionable_orders[main_slots:]
        if float(item["order_feature"].remaining_time) < 0.0
        and str(item["order_id"]) not in selected_order_ids
    ]
    reserve_selected = overdue_candidates[:reserve_slots]

    selected = list(main_selected)
    for item in reserve_selected:
        selected.append(item)
        selected_order_ids.add(str(item["order_id"]))

    for item in sorted_actionable_orders[main_slots:]:
        if len(selected) >= max_candidate_orders:
            break
        order_id = str(item["order_id"])
        if order_id in selected_order_ids:
            continue
        selected.append(item)
        selected_order_ids.add(order_id)

    return selected[:max_candidate_orders]


def _validate_trigger_context(
    *,
    runtime_state: Any,
    trigger_type: str,
    trigger_station_id: str | None,
) -> None:
    if not _is_riding_with_truck_trigger(trigger_type):
        return
    if not trigger_station_id:
        raise ValueError(
            "riding_with_truck / truck_station_arrival 触发必须提供 trigger_station_id"
        )
    if trigger_station_id not in runtime_state.node_states:
        raise ValueError(
            "trigger_station_id 不存在于 runtime_state.node_states: "
            f"{trigger_station_id}"
        )


def _padding_order_feature() -> OrderFeatures:
    return OrderFeatures(
        order_id="",
        weight=0.0,
        deadline=0.0,
        remaining_time=0.0,
        delivery_x=0.0,
        delivery_y=0.0,
        delivery_z=0.0,
        distance_to_order=0.0,
        order_pre_score=0.0,
        priority_band=0,
        has_mode_b_action=False,
        best_mode_b_return_score=0.0,
        best_mode_b_recovery_flight_time=0.0,
        best_mode_b_host_type="",
        best_mode_b_queue_time_est=0.0,
        has_mode_c_action=False,
        mode_c_candidate_count=0,
        best_mode_c_rendezvous_margin=0.0,
        best_mode_c_wait_time=0.0,
        best_mode_c_uav_flight_time=0.0,
        best_mode_c_energy_margin_ratio=0.0,
        best_mode_c_node_type="",
        best_mode_c_truck_eta_remaining=0.0,
        best_mode_c_timeout_risk=0.0,
        is_valid=False,
        delivery_energy_ratio=0.0,
        best_mode_b_recovery_energy_ratio=0.0,
        best_mode_c_recovery_energy_ratio=0.0,
        best_mode_b_total_energy_ratio=0.0,
        best_mode_c_total_energy_ratio=0.0,
        mode_c_energy_saving_ratio=0.0,
        estimated_delivery_finish_slack_sec=0.0,
        local_teacher_has_order_choice=False,
        local_teacher_prefers_order=False,
        local_teacher_order_cost=0.0,
        local_teacher_best_mode="NONE",
        local_teacher_peer_prefer_count=0,
        local_teacher_peer_best_other_cost=0.0,
        local_teacher_cost_gap_to_order_best=0.0,
        local_teacher_is_order_best=False,
        local_teacher_mode_b_prefer_count=0,
        local_teacher_mode_c_prefer_count=0,
    )


def _load_candidate_config(config_path: Path) -> _CandidateConfig:
    raw = _load_yaml(config_path)
    candidate = _require_mapping(raw, "candidate")
    cfg = _CandidateConfig(
        max_candidate_orders=int(candidate["max_candidate_orders"]),
        max_candidate_actions=int(candidate["max_candidate_actions"]),
        station_wait_threshold_sec=float(candidate["station_wait_threshold_sec"]),
        rendezvous_filter_margin_sec=float(
            candidate.get(
                "rendezvous_filter_margin_sec",
                candidate.get("rendezvous_eta_safe_margin_sec"),
            )
        ),
        rendezvous_execution_margin_sec=float(
            candidate.get(
                "rendezvous_execution_margin_sec",
                candidate.get(
                    "rendezvous_filter_margin_sec",
                    candidate.get("rendezvous_eta_safe_margin_sec"),
                ),
            )
        ),
        rendezvous_max_wait_sec=float(
            candidate.get(
                "rendezvous_max_wait_sec",
                candidate["station_wait_threshold_sec"],
            )
        ),
        upper_horizon_sec=float(_require_mapping(raw, "planner")["upper_horizon_sec"]),
    )
    if cfg.rendezvous_execution_margin_sec < 0.0:
        raise ValueError("candidate.rendezvous_execution_margin_sec 不能为负数")
    if cfg.rendezvous_max_wait_sec < cfg.rendezvous_execution_margin_sec:
        raise ValueError(
            "candidate.rendezvous_max_wait_sec 不能小于 "
            "rendezvous_execution_margin_sec"
        )
    return cfg


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取 candidate 配置") from exc

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


__all__ = ["CandidateBuilder"]
