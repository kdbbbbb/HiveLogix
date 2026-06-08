#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch matching teacher for behavior-cloning labels.

This module is read-only with respect to the environment.  It consumes a
same-time decision batch plus the corresponding CandidateOutput objects and
returns per-UAV BC labels/actions.  It does not call env.step(), does not submit
actions, and does not advance simulation time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from config.loader import load_solver_energy_params

from .actions import DispatchAction, EnvAction, WAIT_ACTION
from .contracts import CandidateOutput, ResolvedActionIndices


_INF = 1.0e18
_TIME_EPS = 1.0e-6
_DEFAULT_WAIT_DISPATCH_MARGIN = 1_000_000.0
_MODE_C_WAIT_COST_WEIGHT = 1.0
_DEADLINE_WARNING_WINDOW_SEC = 900.0
_DEADLINE_CRITICAL_WINDOW_SEC = 300.0
_DEADLINE_WARNING_WEIGHT = 1.0
_DEADLINE_CRITICAL_WEIGHT = 4.0
_DEADLINE_LATE_WEIGHT = 10.0
_TEACHER_ENERGY_COST_SEC_PER_BATTERY = 240.0


DispatchCostFn = Callable[[Any, CandidateOutput, DispatchAction, int, int], float]
WaitCostFn = Callable[[Any, CandidateOutput, float | None], float]


@dataclass(frozen=True)
class BatchTeacherAssignment:
    """Teacher choice for one UAV in a same-time decision batch."""

    drone_id: str
    action: EnvAction
    action_indices: ResolvedActionIndices
    cost: float


@dataclass(frozen=True)
class BatchMatchingTeacherResult:
    """Batch teacher output for BC training."""

    assignments_by_drone: Mapping[str, BatchTeacherAssignment]
    labels_by_drone: Mapping[str, ResolvedActionIndices]
    actions_by_drone: Mapping[str, EnvAction]
    total_cost: float


@dataclass(frozen=True)
class _DispatchChoice:
    order_id: str
    action: DispatchAction
    action_indices: ResolvedActionIndices
    cost: float


def build_batch_matching_teacher_labels(
    *,
    decision_contexts: tuple[Any, ...],
    candidate_outputs_by_drone: Mapping[str, CandidateOutput],
    dispatch_cost_fn: DispatchCostFn | None = None,
    wait_cost_fn: WaitCostFn | None = None,
    require_shared_snapshot: bool = True,
) -> BatchMatchingTeacherResult:
    """Build mutually-exclusive per-UAV BC labels for a same-time batch.

    Constraints enforced:
      - each UAV receives exactly one label;
      - each order can be assigned to at most one UAV;
      - WAIT is represented by one private dummy column per UAV.
    """

    if not decision_contexts:
        return BatchMatchingTeacherResult(
            assignments_by_drone={},
            labels_by_drone={},
            actions_by_drone={},
            total_cost=0.0,
        )

    if require_shared_snapshot:
        _validate_contexts(
            decision_contexts,
            require_shared_snapshot=require_shared_snapshot,
        )
    dispatch_cost = dispatch_cost_fn or _default_dispatch_cost
    wait_cost = wait_cost_fn or _default_wait_cost

    drone_ids = [str(context.deciding_drone_id) for context in decision_contexts]
    if len(set(drone_ids)) != len(drone_ids):
        raise ValueError(f"decision batch 中存在重复 UAV: {drone_ids}")

    missing = [drone_id for drone_id in drone_ids if drone_id not in candidate_outputs_by_drone]
    if missing:
        raise ValueError(f"candidate_outputs_by_drone 缺少 UAV: {missing}")

    choices_by_drone: dict[str, dict[str, _DispatchChoice]] = {}
    ordered_order_ids: list[str] = []
    seen_order_ids: set[str] = set()
    wait_cost_by_drone: dict[str, float] = {}

    for context in decision_contexts:
        drone_id = str(context.deciding_drone_id)
        candidate_out = candidate_outputs_by_drone[drone_id]
        dispatch_choices = _best_dispatch_choice_by_order(
            context=context,
            candidate_out=candidate_out,
            dispatch_cost_fn=dispatch_cost,
        )
        choices_by_drone[drone_id] = dispatch_choices
        for order_id in sorted(dispatch_choices):
            if order_id not in seen_order_ids:
                seen_order_ids.add(order_id)
                ordered_order_ids.append(order_id)

        best_cost = (
            min(choice.cost for choice in dispatch_choices.values())
            if dispatch_choices
            else None
        )
        if candidate_out.root_branch_mask[0] and candidate_out.has_wait_action:
            cost = float(wait_cost(context, candidate_out, best_cost))
            if not math.isfinite(cost):
                raise ValueError(f"WAIT cost 非有限值: drone_id={drone_id}, cost={cost}")
            wait_cost_by_drone[drone_id] = cost

    columns: list[tuple[str, str]] = [
        ("DISPATCH", order_id) for order_id in ordered_order_ids
    ]
    columns.extend(("WAIT", drone_id) for drone_id in drone_ids)

    cost_matrix: list[list[float]] = []
    for drone_id in drone_ids:
        row: list[float] = []
        for kind, key in columns:
            if kind == "DISPATCH":
                choice = choices_by_drone[drone_id].get(key)
                row.append(_INF if choice is None else float(choice.cost))
            elif key == drone_id and drone_id in wait_cost_by_drone:
                row.append(wait_cost_by_drone[drone_id])
            else:
                row.append(_INF)
        cost_matrix.append(row)

    selected_columns = _solve_rectangular_assignment(cost_matrix)
    assignments: dict[str, BatchTeacherAssignment] = {}
    total_cost = 0.0
    for row_idx, col_idx in enumerate(selected_columns):
        drone_id = drone_ids[row_idx]
        kind, key = columns[col_idx]
        selected_cost = float(cost_matrix[row_idx][col_idx])
        if not math.isfinite(selected_cost) or selected_cost >= _INF / 2.0:
            raise RuntimeError(f"无法为 UAV {drone_id} 求得合法 teacher 动作")

        if kind == "WAIT":
            action = WAIT_ACTION
            action_indices = ResolvedActionIndices(root_branch_idx=0)
        else:
            choice = choices_by_drone[drone_id][key]
            action = choice.action
            action_indices = choice.action_indices

        assignments[drone_id] = BatchTeacherAssignment(
            drone_id=drone_id,
            action=action,
            action_indices=action_indices,
            cost=selected_cost,
        )
        total_cost += selected_cost

    return BatchMatchingTeacherResult(
        assignments_by_drone=assignments,
        labels_by_drone={
            drone_id: assignment.action_indices
            for drone_id, assignment in assignments.items()
        },
        actions_by_drone={
            drone_id: assignment.action
            for drone_id, assignment in assignments.items()
        },
        total_cost=float(total_cost),
    )


def attach_local_teacher_competition_hints(
    *,
    decision_contexts: tuple[Any, ...],
    candidate_outputs_by_drone: Mapping[str, CandidateOutput],
    dispatch_cost_fn: DispatchCostFn | None = None,
    require_shared_snapshot: bool = True,
) -> dict[str, CandidateOutput]:
    """Attach pre-matching local teacher hints to candidate order features.

    The hints are derived from each UAV's local dispatch choices before the
    order-mutual-exclusion assignment is solved.  No selected Hungarian column,
    final BC label, or final batch action is used here.
    """

    if not decision_contexts:
        return dict(candidate_outputs_by_drone)

    if require_shared_snapshot:
        _validate_contexts(
            decision_contexts,
            require_shared_snapshot=require_shared_snapshot,
        )
    dispatch_cost = dispatch_cost_fn or _default_dispatch_cost
    drone_ids = [str(context.deciding_drone_id) for context in decision_contexts]
    missing = [drone_id for drone_id in drone_ids if drone_id not in candidate_outputs_by_drone]
    if missing:
        raise ValueError(f"candidate_outputs_by_drone 缺少 UAV: {missing}")

    choices_by_drone: dict[str, dict[str, _DispatchChoice]] = {}
    preferred_choice_by_drone: dict[str, _DispatchChoice] = {}
    for context in decision_contexts:
        drone_id = str(context.deciding_drone_id)
        candidate_out = candidate_outputs_by_drone[drone_id]
        choices = _best_dispatch_choice_by_order(
            context=context,
            candidate_out=candidate_out,
            dispatch_cost_fn=dispatch_cost,
        )
        choices_by_drone[drone_id] = choices
        if choices:
            preferred_choice_by_drone[drone_id] = min(
                choices.values(),
                key=lambda item: (float(item.cost), str(item.order_id), str(item.action.mode)),
            )

    prefer_count_by_order: dict[str, int] = {}
    mode_b_prefer_count_by_order: dict[str, int] = {}
    mode_c_prefer_count_by_order: dict[str, int] = {}
    for choice in preferred_choice_by_drone.values():
        order_id = str(choice.order_id)
        prefer_count_by_order[order_id] = int(prefer_count_by_order.get(order_id, 0)) + 1
        mode = str(choice.action.mode)
        if mode == "B":
            mode_b_prefer_count_by_order[order_id] = (
                int(mode_b_prefer_count_by_order.get(order_id, 0)) + 1
            )
        elif mode == "C":
            mode_c_prefer_count_by_order[order_id] = (
                int(mode_c_prefer_count_by_order.get(order_id, 0)) + 1
            )

    order_costs: dict[str, list[tuple[str, _DispatchChoice]]] = {}
    for drone_id, choices in choices_by_drone.items():
        for order_id, choice in choices.items():
            order_costs.setdefault(str(order_id), []).append((str(drone_id), choice))

    annotated: dict[str, CandidateOutput] = dict(candidate_outputs_by_drone)
    for drone_id in drone_ids:
        candidate_out = candidate_outputs_by_drone[drone_id]
        choices = choices_by_drone.get(drone_id, {})
        preferred = preferred_choice_by_drone.get(drone_id)
        updated_orders = []
        for order_feature in candidate_out.candidate_features.order_features:
            if not bool(order_feature.is_valid):
                updated_orders.append(order_feature)
                continue

            order_id = str(order_feature.order_id)
            choice = choices.get(order_id)
            all_order_choices = order_costs.get(order_id, [])
            best_cost = (
                min(float(item.cost) for _peer_id, item in all_order_choices)
                if all_order_choices
                else None
            )
            other_costs = [
                float(item.cost)
                for peer_id, item in all_order_choices
                if str(peer_id) != str(drone_id)
            ]
            best_other_cost = min(other_costs) if other_costs else 0.0
            has_choice = choice is not None
            order_cost = float(choice.cost) if choice is not None else 0.0
            cost_gap = (
                float(order_cost) - float(best_cost)
                if choice is not None and best_cost is not None
                else 0.0
            )
            is_order_best = (
                choice is not None
                and best_cost is not None
                and float(choice.cost) <= float(best_cost) + _TIME_EPS
            )
            updated_orders.append(
                replace(
                    order_feature,
                    local_teacher_has_order_choice=bool(has_choice),
                    local_teacher_prefers_order=bool(
                        preferred is not None and str(preferred.order_id) == order_id
                    ),
                    local_teacher_order_cost=float(order_cost),
                    local_teacher_best_mode=(
                        str(choice.action.mode) if choice is not None else "NONE"
                    ),
                    local_teacher_peer_prefer_count=int(
                        prefer_count_by_order.get(order_id, 0)
                    ),
                    local_teacher_peer_best_other_cost=float(best_other_cost),
                    local_teacher_cost_gap_to_order_best=float(cost_gap),
                    local_teacher_is_order_best=bool(is_order_best),
                    local_teacher_mode_b_prefer_count=int(
                        mode_b_prefer_count_by_order.get(order_id, 0)
                    ),
                    local_teacher_mode_c_prefer_count=int(
                        mode_c_prefer_count_by_order.get(order_id, 0)
                    ),
                )
            )

        annotated[drone_id] = replace(
            candidate_out,
            candidate_features=replace(
                candidate_out.candidate_features,
                order_features=tuple(updated_orders),
            ),
        )

    return annotated


def _validate_contexts(
    decision_contexts: tuple[Any, ...],
    *,
    require_shared_snapshot: bool,
) -> None:
    first = decision_contexts[0]
    first_time = float(first.t_decision)
    runtime_state = first.runtime_state
    coarse_plan = first.coarse_plan
    for context in decision_contexts:
        if abs(float(context.t_decision) - first_time) > _TIME_EPS:
            raise ValueError("decision_contexts 混入了不同 t_decision")
        if require_shared_snapshot and context.runtime_state is not runtime_state:
            raise ValueError("decision_contexts 必须共享同一份 runtime_state snapshot")
        if require_shared_snapshot and context.coarse_plan is not coarse_plan:
            raise ValueError("decision_contexts 必须共享同一份 coarse_plan snapshot")


def _best_dispatch_choice_by_order(
    *,
    context: Any,
    candidate_out: CandidateOutput,
    dispatch_cost_fn: DispatchCostFn,
) -> dict[str, _DispatchChoice]:
    best_by_order: dict[str, _DispatchChoice] = {}
    for (order_idx, mode_idx), action in sorted(
        candidate_out.resolved_action_lookup.dispatch_actions.items()
    ):
        cost = float(dispatch_cost_fn(context, candidate_out, action, order_idx, mode_idx))
        if not math.isfinite(cost):
            continue
        order_id = str(action.order_id)
        candidate = _DispatchChoice(
            order_id=order_id,
            action=action,
            action_indices=ResolvedActionIndices(
                root_branch_idx=1,
                order_idx=int(order_idx),
                mode_idx=int(mode_idx),
            ),
            cost=cost,
        )
        previous = best_by_order.get(order_id)
        if previous is None or (candidate.cost, candidate.action.mode) < (
            previous.cost,
            previous.action.mode,
        ):
            best_by_order[order_id] = candidate
    return best_by_order


def _default_dispatch_cost(
    context: Any,
    candidate_out: CandidateOutput,
    action: DispatchAction,
    order_idx: int,
    mode_idx: int,
) -> float:
    del mode_idx
    del context
    order_feature = candidate_out.candidate_features.order_features[order_idx]
    delivery_flight_time = _estimate_delivery_flight_time(candidate_out, order_idx)
    service_time = float(load_solver_energy_params().drone_service_time_order_s)
    # Use precomputed delivery-finish slack from CandidateBuilder, which accounts for
    # effective_launch_time + flight_time + drone_service_time.  This matches the
    # env_adapter reward criterion: actual_deliver_time = service_leg.finish_time.
    deadline_risk_penalty = _deadline_risk_penalty(
        slack_sec=float(order_feature.estimated_delivery_finish_slack_sec)
    )

    mode = str(action.mode)
    if mode == "B":
        recovery_cost = float(order_feature.best_mode_b_recovery_flight_time)
        energy_cost = _teacher_energy_cost(candidate_out, order_idx, mode)
        return _finite_or_inf(
            delivery_flight_time
            + service_time
            + recovery_cost
            + deadline_risk_penalty
            + energy_cost
        )
    if mode == "C":
        recovery_cost = (
            float(order_feature.best_mode_c_uav_flight_time)
            + _MODE_C_WAIT_COST_WEIGHT
            * float(order_feature.best_mode_c_wait_time)
        )
        energy_cost = _teacher_energy_cost(candidate_out, order_idx, mode)
        return _finite_or_inf(
            delivery_flight_time
            + service_time
            + recovery_cost
            + deadline_risk_penalty
            + energy_cost
        )
    raise ValueError(f"未知 dispatch mode: {action.mode}")


def _estimate_delivery_flight_time(
    candidate_out: CandidateOutput,
    order_idx: int,
) -> float:
    order_feature = candidate_out.candidate_features.order_features[order_idx]
    cruise_speed = max(
        float(candidate_out.candidate_features.uav_self.cruise_speed),
        _TIME_EPS,
    )
    return float(order_feature.distance_to_order) / cruise_speed


def _deadline_risk_penalty(*, slack_sec: float) -> float:
    """Return a DDL-aware cost correction for minimizing teacher assignment.

    Negative values intentionally prioritize saveable urgent orders.  Once the
    delivery ETA has already crossed the deadline, lateness gradually removes
    that urgency bonus so already-late orders do not crowd out still-saveable
    critical orders.
    """

    slack = float(slack_sec)
    if not math.isfinite(slack):
        return 0.0
    warning_window = _DEADLINE_WARNING_WINDOW_SEC
    critical_window = _DEADLINE_CRITICAL_WINDOW_SEC
    if critical_window <= 0.0 or warning_window <= critical_window:
        raise ValueError("DDL 窗口配置非法")

    if slack >= warning_window:
        return 0.0

    warning_bonus_cap = (
        _DEADLINE_WARNING_WEIGHT * (warning_window - critical_window)
    )
    critical_bonus_cap = _DEADLINE_CRITICAL_WEIGHT * critical_window
    if slack >= critical_window:
        return -_DEADLINE_WARNING_WEIGHT * (warning_window - slack)
    if slack >= 0.0:
        return -warning_bonus_cap - _DEADLINE_CRITICAL_WEIGHT * (
            critical_window - slack
        )
    return (
        -warning_bonus_cap
        - critical_bonus_cap
        + _DEADLINE_LATE_WEIGHT * (-slack)
    )


def _teacher_energy_cost(
    candidate_out: CandidateOutput,
    order_idx: int,
    mode: str,
) -> float:
    energy = candidate_out.teacher_energy_by_order_idx.get(int(order_idx))
    if energy is None:
        return 0.0
    if mode == "B":
        ratio = float(energy.best_mode_b_total_energy_ratio)
    elif mode == "C":
        ratio = float(energy.best_mode_c_total_energy_ratio)
    else:
        raise ValueError(f"未知 dispatch mode: {mode}")
    if not math.isfinite(ratio) or ratio < 0.0:
        return 0.0
    return float(_TEACHER_ENERGY_COST_SEC_PER_BATTERY * ratio)


def _finite_or_inf(value: float) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.inf


def _default_wait_cost(
    _context: Any,
    _candidate_out: CandidateOutput,
    best_dispatch_cost: float | None,
) -> float:
    if best_dispatch_cost is None:
        return 0.0
    return float(best_dispatch_cost) + _DEFAULT_WAIT_DISPATCH_MARGIN


def _solve_rectangular_assignment(cost_matrix: list[list[float]]) -> list[int]:
    """Return selected column index for each row using Hungarian minimization."""

    row_count = len(cost_matrix)
    if row_count == 0:
        return []
    col_count = len(cost_matrix[0])
    if col_count < row_count:
        raise ValueError(
            f"assignment 矩阵列数必须不少于行数: rows={row_count}, cols={col_count}"
        )
    if any(len(row) != col_count for row in cost_matrix):
        raise ValueError("assignment 矩阵行长度不一致")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (col_count + 1)
    p = [0] * (col_count + 1)
    way = [0] * (col_count + 1)

    for i in range(1, row_count + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (col_count + 1)
        used = [False] * (col_count + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, col_count + 1):
                if used[j]:
                    continue
                cur = float(cost_matrix[i0 - 1][j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not math.isfinite(delta):
                raise RuntimeError("assignment 矩阵不存在完整可行匹配")
            for j in range(0, col_count + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * row_count
    for j in range(1, col_count + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    if any(col_idx < 0 for col_idx in assignment):
        raise RuntimeError("assignment 求解失败")
    return assignment


__all__ = [
    "attach_local_teacher_competition_hints",
    "BatchMatchingTeacherResult",
    "BatchTeacherAssignment",
    "DispatchCostFn",
    "WaitCostFn",
    "build_batch_matching_teacher_labels",
]
