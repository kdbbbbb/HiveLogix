#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 7 PPO 训练主循环。

当前实现目标：
  - 复用 Phase 2~6 已冻结的 `DecisionContext -> CandidateOutput -> EnvAction` 链路；
  - 在训练前强制校验 `order_source_mode == poisson`；
  - 使用 `ObservationTensorizer` / `CriticBatchBuilder` / `RolloutBuffer`
    固化 Phase 7 的 pre-action snapshot 训练数据流；
  - 训练产物写出 `policy.pt` / `meta.json` / `train_metrics.jsonl` / 配置快照。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from config.loader import load_drone_params

from .actions import DispatchAction
from .batch_matching_teacher import (
    attach_local_teacher_competition_hints,
    build_batch_matching_teacher_labels,
)
from .contracts import (
    ActionSpaceMeta,
    CandidateMeta,
    CriticTensorSchemaMeta,
    DroneRuntimeParamsSnapshot,
    EnvSemanticContractMeta,
    MetaJson,
    OnlineLockParams,
    PlannerMeta,
    PolicyMeta,
    ResolvedActionIndices,
    RewardMeta,
    SharedRuntimeParamsSnapshot,
    SolverEnergyRuntimeSnapshot,
    TrainingRunMeta,
    build_meta_json_dict,
)
from .critic_batch_builder import CriticBatchBuilder
from .model import SharedPPOActorCritic
from .observation_tensorizer import (
    HISTORY_TOKEN_FIELDS,
    INFRA_TOKEN_FIELDS,
    ORDER_TOKEN_FIELDS,
    UAV_SELF_TOKEN_FIELDS,
    ObservationTensorizer,
)
from .order_source_adapter import (
    OrderSourceConfig,
    OrderSourceMode,
    build_order_source,
    ensure_mode_allowed,
)
from .export_sumo_truck_route import export_phase4_truck_route
from .rollout_buffer import (
    RolloutTransition,
    build_batch_view_from_transitions,
    compute_gae,
)
from .scene_loader import DEFAULT_CONFIG_PATH, load_default_scene
from .env_adapter import TrainingDroneState, TrainingEnvAdapter


try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class _BCWarmStartStageConfig:
    name: str
    order_source_mode: str
    updates: int
    learning_rate: float


@dataclass(frozen=True)
class _TrainingConfig:
    device: str
    total_timesteps: int
    rollout_steps: int
    recurrent_ppo: bool
    sequence_len: int
    burn_in_len: int
    train_len: int
    sequence_minibatch_size: int
    target_minibatch_timesteps: int
    ppo_learning_rate: float
    bc_learning_rate: float
    batch_size_alias: int | None
    ppo_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    reward_scale: float
    rendezvous_arrive_bonus: float
    rendezvous_bonus: float
    mode_c_attempt_bonus: float
    value_loss_coef: float
    value_loss_type: str
    value_huber_delta: float
    max_grad_norm: float
    normalize_advantage: bool
    target_kl: float
    wait_without_dispatch_loss_weight: float
    wait_with_dispatch_loss_weight: float
    mode_b_dispatch_loss_weight: float
    mode_c_dispatch_loss_weight: float
    training_seed: int
    log_interval_updates: int
    save_interval_updates: int
    eval_interval_updates: int
    benchmark_eval_episodes: int
    stochastic_eval_seeds: int
    early_stop_enabled: bool
    early_stop_min_evals: int
    early_stop_stochastic_high_patience: int
    early_stop_value_loss_window: int
    early_stop_value_loss_min_delta: float
    allowed_train_order_source_mode: tuple[str, ...]
    bc_warm_start_stages: tuple[_BCWarmStartStageConfig, ...]
    bc_warm_start_enabled: bool = False
    bc_warm_start_updates: int = 0
    bc_warm_start_batches_per_update: int = 1
    bc_warm_start_max_decision_batches_per_rollout: int = 0
    bc_warm_start_decision_batches_per_update: int = 0
    bc_warm_start_reset_each_update: bool = True
    bc_warm_start_checkpoint_path: str | None = None
    bc_warm_start_reuse_existing: bool = True
    bc_warm_start_checkpoint_interval_updates: int = 10
    bc_wait_no_dispatch_loss_weight: float = 0.0
    bc_wait_with_dispatch_loss_weight: float = 0.5
    bc_mode_b_dispatch_loss_weight: float = 1.0
    bc_mode_c_dispatch_loss_weight: float = 2.0


@dataclass(frozen=True)
class _PolicyConfig:
    d_model: int
    ff_dim: int
    lstm_hidden: int
    lstm_layers: int
    hist_len: int


@dataclass(frozen=True)
class _EvalConfig:
    benchmark_seed: int
    stochastic_seed_base: int
    stochastic_arrival_rates: tuple[tuple[str, float], ...]
    train_arrival_curriculum_enabled: bool
    train_arrival_base_rate: float
    train_arrival_high_start_update: int
    train_arrival_high_full_update: int
    train_arrival_high_max_probability: float
    c_sensitive_enabled: bool
    c_sensitive_seed: int
    c_sensitive_min_legal_mode_c_recovery_nodes: int


@dataclass(frozen=True)
class _MaterializedSequence:
    sequence_id: tuple[int, str, int, int]
    episode_id: int
    actor_drone_id: str
    recurrent_segment_id: int
    local_chunk_idx: int
    transitions: tuple[RolloutTransition, ...]
    advantages: np.ndarray
    returns: np.ndarray
    sample_loss_weights: np.ndarray
    valid_timestep_mask: np.ndarray


@dataclass(frozen=True)
class _SequenceMiniBatch:
    observation_batch: Any
    critic_batch: Any
    action_mask: Any
    action_indices: dict[str, Any]
    old_log_probs: Any
    old_values: Any
    returns: Any
    advantages: Any
    sample_loss_weights: Any
    valid_timestep_mask: Any
    lstm_state_in: Any | None
    sequence_count: int
    padded_timesteps: int


@dataclass(frozen=True)
class _BestCheckpointState:
    path: Path
    update_idx: int
    global_step: int
    selection_key: tuple[Any, ...]
    eval_report: dict[str, Any]


def _extract_attributed_step_reward_parts(*, step_result: Any) -> tuple[float, float, float]:
    reward_breakdown = dict(getattr(step_result, "info", {}).get("reward_breakdown", {}))
    carry_in_reward = float(reward_breakdown.get("attributed_carry_in", 0.0))
    if "attributed_post_action" in reward_breakdown:
        post_action_reward = float(reward_breakdown["attributed_post_action"])
    else:
        post_action_reward = float(step_result.reward) - carry_in_reward
    mode_c_attempt_reward = float(reward_breakdown.get("mode_c_attempt_bonus", 0.0))
    return carry_in_reward, post_action_reward, mode_c_attempt_reward


def _is_rendezvous_success_state(training_state: str) -> bool:
    return str(training_state) in {"charging_on_truck", "riding_with_truck"}


def _is_mode_c_action_indices(action_indices: Any) -> bool:
    return (
        int(getattr(action_indices, "root_branch_idx", 0)) == 1
        and int(getattr(action_indices, "mode_idx", 0)) == 1
    )


def _shape_post_action_reward_for_rendezvous(
    *,
    post_action_reward: float,
    action_indices: Any,
    transition_summary: Any,
    rendezvous_arrive_bonus: float,
    rendezvous_bonus: float,
) -> tuple[float, bool, bool]:
    if not _is_mode_c_action_indices(action_indices):
        return float(post_action_reward), False, False

    shaped_reward = float(post_action_reward)
    arrive_applied = False
    success_applied = False
    rendezvous_success = bool(getattr(transition_summary, "rendezvous_success", False))

    success_bonus = float(rendezvous_bonus)
    if rendezvous_success and success_bonus > 0.0:
        shaped_reward += success_bonus
        success_applied = True

    return shaped_reward, arrive_applied, success_applied


def _shape_pending_transition_reward_for_rendezvous(
    *,
    pending_transition: RolloutTransition,
    reward_so_far: float,
    runtime_state: Any | None,
    rendezvous_arrive_bonus: float,
    rendezvous_bonus: float,
) -> tuple[float, bool, bool]:
    if not _is_mode_c_action_indices(pending_transition.action_indices):
        return (
            float(reward_so_far),
            bool(getattr(pending_transition, "rendezvous_arrive_bonus_applied", False)),
            bool(getattr(pending_transition, "rendezvous_success_bonus_applied", False)),
        )
    if runtime_state is None:
        return (
            float(reward_so_far),
            bool(getattr(pending_transition, "rendezvous_arrive_bonus_applied", False)),
            bool(getattr(pending_transition, "rendezvous_success_bonus_applied", False)),
        )
    drone_states = getattr(runtime_state, "drone_states", {})
    drone_state = drone_states.get(str(pending_transition.actor_drone_id))
    if drone_state is None:
        return (
            float(reward_so_far),
            bool(getattr(pending_transition, "rendezvous_arrive_bonus_applied", False)),
            bool(getattr(pending_transition, "rendezvous_success_bonus_applied", False)),
        )

    shaped_reward = float(reward_so_far)
    training_state = str(getattr(drone_state, "training_state", ""))
    arrive_applied = bool(
        getattr(pending_transition, "rendezvous_arrive_bonus_applied", False)
    )
    success_applied = bool(
        getattr(pending_transition, "rendezvous_success_bonus_applied", False)
    )

    success_bonus = float(rendezvous_bonus)
    if (
        not success_applied
        and success_bonus > 0.0
        and _is_rendezvous_success_state(training_state)
    ):
        shaped_reward += success_bonus
        success_applied = True

    return shaped_reward, arrive_applied, success_applied


def _insert_finalized_transition_in_order(
    *,
    rollout_backlog: list[RolloutTransition],
    transition: RolloutTransition,
) -> None:
    insert_at = len(rollout_backlog)
    target_index = int(transition.global_decision_index)
    while insert_at > 0:
        prev_index = int(rollout_backlog[insert_at - 1].global_decision_index)
        if prev_index <= target_index:
            break
        insert_at -= 1
    rollout_backlog.insert(insert_at, transition)


def _finalize_pending_transition_for_next_decision(
    *,
    pending_transition_by_drone: dict[str, RolloutTransition],
    rollout_backlog: list[RolloutTransition],
    successor_bootstrap_values: dict[tuple[int, str, int], float],
    drone_id: str,
    carry_in_reward: float,
    next_value: float,
    decision_context: Any | None = None,
    rendezvous_arrive_bonus: float = 0.0,
    rendezvous_bonus: float = 0.0,
    reward_scale: float = 1.0,
) -> None:
    pending = pending_transition_by_drone.pop(drone_id, None)
    if pending is None:
        return
    reward_so_far = 0.0 if pending.reward is None else float(pending.reward)
    runtime_state = (
        getattr(decision_context, "runtime_state", None)
        if decision_context is not None
        else None
    )
    reward_scale_value = float(reward_scale)
    (
        reward_so_far,
        arrive_bonus_applied,
        success_bonus_applied,
    ) = _shape_pending_transition_reward_for_rendezvous(
        pending_transition=pending,
        reward_so_far=reward_so_far,
        runtime_state=runtime_state,
        rendezvous_arrive_bonus=float(rendezvous_arrive_bonus),
        rendezvous_bonus=float(rendezvous_bonus),
    )
    finalized = replace(
        pending,
        reward=reward_so_far + float(carry_in_reward) * reward_scale_value,
        done=False,
        rendezvous_arrive_bonus_applied=arrive_bonus_applied,
        rendezvous_success_bonus_applied=success_bonus_applied,
    )
    _insert_finalized_transition_in_order(
        rollout_backlog=rollout_backlog,
        transition=finalized,
    )
    key = (
        int(pending.episode_id),
        str(pending.actor_drone_id),
        int(pending.recurrent_segment_id),
    )
    successor_bootstrap_values[key] = float(next_value)


def _flush_terminal_pending_transitions(
    *,
    pending_transition_by_drone: dict[str, RolloutTransition],
    rollout_backlog: list[RolloutTransition],
    terminal_reward_by_drone: Mapping[str, float],
    runtime_state: Any | None = None,
    rendezvous_arrive_bonus: float = 0.0,
    rendezvous_bonus: float = 0.0,
    reward_scale: float = 1.0,
) -> None:
    reward_scale_value = float(reward_scale)
    for drone_id, pending in tuple(pending_transition_by_drone.items()):
        reward_so_far = 0.0 if pending.reward is None else float(pending.reward)
        (
            reward_so_far,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_pending_transition_reward_for_rendezvous(
            pending_transition=pending,
            reward_so_far=reward_so_far,
            runtime_state=runtime_state,
            rendezvous_arrive_bonus=float(rendezvous_arrive_bonus),
            rendezvous_bonus=float(rendezvous_bonus),
        )
        finalized = replace(
            pending,
            reward=reward_so_far
            + float(terminal_reward_by_drone.get(drone_id, 0.0)) * reward_scale_value,
            done=True,
            rendezvous_arrive_bonus_applied=arrive_bonus_applied,
            rendezvous_success_bonus_applied=success_bonus_applied,
        )
        _insert_finalized_transition_in_order(
            rollout_backlog=rollout_backlog,
            transition=finalized,
        )
    pending_transition_by_drone.clear()


def _build_rollout_prefix_boundary_bootstrap_values(
    *,
    current_result: Any,
    episode_id: int,
    env: TrainingEnvAdapter,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    model: Any,
    history_buffer: Sequence[Any],
    last_seen_plan_version_by_drone: Mapping[str, int],
    lstm_state_by_drone: Mapping[str, Any],
    recurrent_segment_id_by_drone: Mapping[str, int],
    successor_bootstrap_values: Mapping[tuple[int, str, int], float],
) -> dict[tuple[int, str, int], float]:
    return {
        **_build_current_boundary_bootstrap_values(
            current_result=current_result,
            episode_id=episode_id,
            env=env,
            tensorizer=tensorizer,
            critic_builder=critic_builder,
            critic_schema=critic_schema,
            model=model,
            history_buffer=history_buffer,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
            lstm_state_by_drone=lstm_state_by_drone,
            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
        ),
        **successor_bootstrap_values,
    }


def _maybe_record_probe_successor_bootstrap_value(
    *,
    successor_bootstrap_values: dict[tuple[int, str, int], float],
    episode_id: int,
    drone_id: str,
    recurrent_segment_id: int,
    value_old: float,
) -> None:
    key = (int(episode_id), str(drone_id), int(recurrent_segment_id))
    successor_bootstrap_values.setdefault(key, float(value_old))


@dataclass
class _EpisodeAccumulator:
    phase: str
    episode_id: int
    order_source_mode: str
    order_source_seed: int
    train_arrival_band: str | None = None
    train_arrival_rate_per_min: float | None = None
    global_step_start: int | None = None
    global_step_end: int | None = None
    update_start: int | None = None
    update_end: int | None = None
    decision_count: int = 0
    total_reward: float = 0.0


def _sample_training_arrival_band(
    eval_cfg: _EvalConfig,
    *,
    update_idx: int = 0,
) -> tuple[str, float]:
    if bool(getattr(eval_cfg, "train_arrival_curriculum_enabled", False)):
        rates = dict(getattr(eval_cfg, "stochastic_arrival_rates", ()))
        base_rate = float(getattr(eval_cfg, "train_arrival_base_rate"))
        medium_rate = float(rates.get("medium", base_rate))
        high_rate = float(rates.get("high", medium_rate))
        high_start = int(getattr(eval_cfg, "train_arrival_high_start_update", 0))
        high_full = int(getattr(eval_cfg, "train_arrival_high_full_update", high_start))
        high_max_p = float(getattr(eval_cfg, "train_arrival_high_max_probability", 0.0))
        if update_idx < high_start or high_max_p <= 0.0:
            high_p = 0.0
        elif high_full <= high_start:
            high_p = high_max_p
        else:
            progress = min(
                1.0,
                max(0.0, (float(update_idx) - float(high_start)) / float(high_full - high_start)),
            )
            high_p = high_max_p * progress

        if high_p > 0.0 and random.random() < high_p:
            return "high", high_rate
        if random.random() < 0.5:
            return "base", base_rate
        return "medium", medium_rate

    band_name, arrival_rate = random.choice(tuple(eval_cfg.stochastic_arrival_rates))
    return str(band_name), float(arrival_rate)


def _build_training_episode_order_source(
    *,
    scene_ctx: Any,
    config_path: str | Path,
    eval_cfg: _EvalConfig,
    base_order_source: OrderSourceConfig,
    update_idx: int = 0,
) -> tuple[str, float, OrderSourceConfig]:
    band_name, arrival_rate = _sample_training_arrival_band(
        eval_cfg,
        update_idx=update_idx,
    )
    order_source = build_order_source(
        scene_ctx,
        mode=OrderSourceMode.POISSON,
        seed=base_order_source.seed,
        overrides={"poisson_arrival_rate": arrival_rate},
        config_path=config_path,
    )
    return band_name, arrival_rate, order_source


def _compute_transition_loss_weight(
    *,
    train_cfg: _TrainingConfig,
    action_indices: ResolvedActionIndices,
    candidate_out: Any,
) -> float:
    if int(action_indices.root_branch_idx) == 0:
        has_dispatch = bool(candidate_out.resolved_action_lookup.dispatch_actions)
        return (
            float(train_cfg.wait_with_dispatch_loss_weight)
            if has_dispatch
            else float(train_cfg.wait_without_dispatch_loss_weight)
        )
    if int(action_indices.mode_idx or 0) == 1:
        return float(train_cfg.mode_c_dispatch_loss_weight)
    return float(train_cfg.mode_b_dispatch_loss_weight)


def _candidate_output_has_legal_mode_c(candidate_out: Any) -> bool:
    return any(
        isinstance(action, DispatchAction) and str(action.mode) == "C"
        for action in candidate_out.resolved_action_lookup.dispatch_actions.values()
    )


def _compute_bc_label_loss_weight(
    *,
    train_cfg: _TrainingConfig,
    action: Any,
    candidate_out: Any,
) -> float:
    has_dispatch = bool(candidate_out.resolved_action_lookup.dispatch_actions)
    if not isinstance(action, DispatchAction):
        return (
            float(train_cfg.bc_wait_with_dispatch_loss_weight)
            if has_dispatch
            else float(train_cfg.bc_wait_no_dispatch_loss_weight)
        )
    if str(action.mode) == "C":
        return float(train_cfg.bc_mode_c_dispatch_loss_weight)
    return float(train_cfg.bc_mode_b_dispatch_loss_weight)


def _bc_warm_start_stage_total_updates(
    stages: Sequence[_BCWarmStartStageConfig],
) -> int:
    return sum(int(stage.updates) for stage in stages)


def _bc_warm_start_stage_payloads(
    stages: Sequence[_BCWarmStartStageConfig],
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(stage.name),
            "order_source_mode": str(stage.order_source_mode),
            "updates": int(stage.updates),
            "learning_rate": float(stage.learning_rate),
        }
        for stage in stages
    ]


def _bc_warm_start_stage_stats_payload(
    stats: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_stages = stats.get("stages", ())
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        return []
    return [dict(stage) for stage in raw_stages if isinstance(stage, Mapping)]


def _normalize_bc_warm_start_stage_config_payload(
    raw_stages: Any,
) -> Any:
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        return raw_stages
    return [
        dict(stage) if isinstance(stage, Mapping) else stage
        for stage in raw_stages
    ]


def _bc_warm_start_checkpoint_stage_payload(
    *,
    train_cfg: _TrainingConfig,
    stats: Mapping[str, Any],
    legacy_checkpoint: bool = False,
) -> dict[str, Any]:
    return {
        "bc_warm_start_stages": _bc_warm_start_stage_payloads(
            train_cfg.bc_warm_start_stages
        ),
        "bc_warm_start_stage_total_updates": _bc_warm_start_stage_total_updates(
            train_cfg.bc_warm_start_stages
        ),
        "bc_warm_start_stage_stats": _bc_warm_start_stage_stats_payload(stats),
        "bc_warm_start_legacy_checkpoint": bool(legacy_checkpoint),
    }


def _safe_divide_metric(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _masked_argmax_index(*, logits: Any, mask: Any, device: Any) -> int:
    logits_tensor = torch.as_tensor(logits, device=device).detach()
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if logits_tensor.dim() == 2:
        logits_tensor = logits_tensor.reshape(-1, logits_tensor.shape[-1])[0]
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.reshape(-1, mask_tensor.shape[-1])[0]
    if logits_tensor.shape != mask_tensor.shape:
        raise ValueError(
            "BC head metric logits/mask shape 不一致: "
            f"{tuple(logits_tensor.shape)} vs {tuple(mask_tensor.shape)}"
        )
    if not bool(mask_tensor.any().detach().cpu().item()):
        return -1
    masked_logits = logits_tensor.masked_fill(~mask_tensor, -1e9)
    return int(torch.argmax(masked_logits, dim=-1).detach().cpu().item())


def _select_mask_row(*, mask: Any, index: int, device: Any) -> Any:
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.reshape(
            -1,
            mask_tensor.shape[-2],
            mask_tensor.shape[-1],
        )[0]
    if mask_tensor.dim() != 2:
        raise ValueError(
            f"BC mode mask 期望为 2D/3D，实际 shape={tuple(mask_tensor.shape)}"
        )
    return mask_tensor[int(index)]


def _select_logits_row(*, logits: Any, index: int, device: Any) -> Any:
    logits_tensor = torch.as_tensor(logits, device=device).detach()
    if logits_tensor.dim() == 3:
        logits_tensor = logits_tensor.reshape(
            -1,
            logits_tensor.shape[-2],
            logits_tensor.shape[-1],
        )[0]
    if logits_tensor.dim() != 2:
        raise ValueError(
            f"BC mode logits 期望为 2D/3D，实际 shape={tuple(logits_tensor.shape)}"
        )
    return logits_tensor[int(index)]


def train_cmrappo(
    *,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_version: str | None = None,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require_torch()
    scene_ctx = load_default_scene(config_path=config_path)
    train_cfg = _load_training_config(Path(config_path))
    policy_cfg = _load_policy_config(Path(config_path))
    eval_cfg = _load_eval_config(Path(config_path))
    _set_training_seed(train_cfg.training_seed)
    order_source = build_order_source(
        scene_ctx,
        mode=OrderSourceMode.POISSON,
        config_path=config_path,
    )
    ensure_mode_allowed(
        order_source.mode,
        train_cfg.allowed_train_order_source_mode,
        "train_cmrappo",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(config_path), out_dir / "training_config_snapshot.yaml")
    export_phase4_truck_route(config_path=config_path)

    env = TrainingEnvAdapter(
        scene_ctx=scene_ctx,
        order_source=order_source,
        config_path=config_path,
    )
    tensorizer = ObservationTensorizer(scene_ctx=scene_ctx, config_path=config_path)
    critic_builder = CriticBatchBuilder(scene_ctx=scene_ctx, config_path=config_path)
    critic_schema = critic_builder.default_schema_meta

    global_step = 0
    update_idx = 0
    current_train_arrival_band, current_train_arrival_rate, current_train_order_source = (
        _build_training_episode_order_source(
            scene_ctx=scene_ctx,
            config_path=config_path,
            eval_cfg=eval_cfg,
            base_order_source=order_source,
            update_idx=update_idx,
        )
    )
    env.set_order_source(current_train_order_source)
    initial_result = env.reset()
    if initial_result.decision_context is None:
        raise RuntimeError("reset 后没有可用的 decision_context，无法启动训练")

    last_seen_plan_version_by_drone: dict[str, int] = {}
    history_buffer: deque[Any] = deque(maxlen=policy_cfg.hist_len)
    lstm_state_by_drone: dict[str, Any] = {}
    recurrent_segment_id_by_drone: dict[str, int] = {}
    local_decision_cursor: dict[tuple[int, str, int], int] = {}
    episode_id = 0
    bootstrap_candidate = _build_candidate_output(
        env=env,
        decision_context=initial_result.decision_context,
        last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
    )
    bootstrap_observation = tensorizer.build(
        decision_context=initial_result.decision_context,
        candidate_out=bootstrap_candidate,
        transition_history=tuple(history_buffer),
    )
    bootstrap_action_mask = tensorizer.build_action_mask(bootstrap_candidate)
    bootstrap_critic = critic_builder.build(
        decision_context=initial_result.decision_context,
        critic_tensor_schema_meta=critic_schema,
    )
    model = SharedPPOActorCritic(
        uav_feat_dim=int(np.asarray(bootstrap_observation.uav_self_token).shape[-1]),
        order_feat_dim=int(np.asarray(bootstrap_observation.order_tokens).shape[-1]),
        infra_feat_dim=int(np.asarray(bootstrap_observation.infra_tokens).shape[-1]),
        history_feat_dim=int(np.asarray(bootstrap_observation.history_tokens).shape[-1]),
        critic_order_feat_dim=int(np.asarray(bootstrap_critic.global_order_pool_tokens).shape[-1]),
        critic_uav_feat_dim=int(np.asarray(bootstrap_critic.global_uav_tokens).shape[-1]),
        critic_station_feat_dim=int(np.asarray(bootstrap_critic.global_station_tokens).shape[-1]),
        critic_plan_feat_dim=int(np.asarray(bootstrap_critic.coarse_plan_summary_vec).shape[-1]),
        critic_sys_feat_dim=int(np.asarray(bootstrap_critic.global_system_summary_vec).shape[-1]),
        d_model=policy_cfg.d_model,
        ff_dim=policy_cfg.ff_dim,
        lstm_hidden=policy_cfg.lstm_hidden,
        lstm_layers=policy_cfg.lstm_layers,
    )
    device = _resolve_device(train_cfg.device)
    model.to(device)
    if getattr(device, "type", str(device)) == "mps":
        model.enable_lstm_cpu_fallback()
        print("[train_cmrappo] MPS LSTM CPU fallback enabled")
    _emit_runtime_device_debug(
        requested_device=train_cfg.device,
        resolved_device=device,
    )

    metrics_path = out_dir / "train_metrics.jsonl"
    episode_metrics_path = out_dir / "episode_metrics.jsonl"
    planner_replan_events_path = out_dir / "planner_replan_events.jsonl"
    eval_metrics_path = out_dir / "eval_metrics.jsonl"
    benchmark_report_path = out_dir / "benchmark_report.json"
    stochastic_report_path = out_dir / "stochastic_report.json"
    c_sensitive_report_path = out_dir / "c_sensitive_report.json"
    policy_path = out_dir / "policy.pt"
    policy_last_path = out_dir / "policy_last.pt"
    policy_best_path = out_dir / "policy_best.pt"
    bc_warm_start_policy_path = out_dir / "bc_warm_start_policy.pt"
    bc_warm_start_meta_path = out_dir / "bc_warm_start_meta.json"
    meta_path = out_dir / "meta.json"
    config_snapshot_path = out_dir / "training_config_snapshot.yaml"
    model_version = model_version or datetime.now(timezone.utc).strftime("cmrappo_%Y%m%dT%H%M%SZ")
    learning_rate_payload = {
        "phase": "optimizer_config",
        "bc_learning_rate": float(train_cfg.bc_learning_rate),
        "ppo_learning_rate": float(train_cfg.ppo_learning_rate),
        "bc_warm_start_stages": _bc_warm_start_stage_payloads(
            train_cfg.bc_warm_start_stages
        ),
        "bc_warm_start_stage_total_updates": _bc_warm_start_stage_total_updates(
            train_cfg.bc_warm_start_stages
        ),
    }
    _append_metrics(metrics_path, learning_rate_payload)
    _emit_event_hook(event_hook, "optimizer_config", learning_rate_payload)

    bc_optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.bc_learning_rate)
    bc_warm_start_stats = _load_or_run_bc_warm_start(
        model=model,
        optimizer=bc_optimizer,
        scene_ctx=scene_ctx,
        order_source=current_train_order_source,
        config_path=Path(config_path),
        tensorizer=tensorizer,
        critic_builder=critic_builder,
        critic_schema=critic_schema,
        train_cfg=train_cfg,
        device=device,
        metrics_path=metrics_path,
        checkpoint_path=bc_warm_start_policy_path,
        checkpoint_meta_path=bc_warm_start_meta_path,
        model_version=model_version,
        event_hook=event_hook,
    )
    del bc_optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.ppo_learning_rate)

    current_result = initial_result
    episode_acc = _EpisodeAccumulator(
        phase="train",
        episode_id=episode_id,
        order_source_mode=str(order_source.mode.value),
        order_source_seed=env.current_episode_order_source_seed(),
        train_arrival_band=current_train_arrival_band,
        train_arrival_rate_per_min=current_train_arrival_rate,
        global_step_start=0,
        update_start=0,
    )
    _emit_event_hook(
        event_hook,
        "train_reset",
        {
            "phase": "train",
            "episode_id": episode_id,
            "global_step": global_step,
            "update_idx": update_idx,
            "train_arrival_band": current_train_arrival_band,
            "train_arrival_rate_per_min": current_train_arrival_rate,
            "env": env,
            "result": current_result,
        },
    )
    rollout_backlog: list[RolloutTransition] = []
    pending_transition_by_drone: dict[str, RolloutTransition] = {}
    successor_bootstrap_values: dict[tuple[int, str, int], float] = {}
    best_checkpoint: _BestCheckpointState | None = None
    latest_eval_report: dict[str, Any] | None = None
    eval_count = 0
    best_stochastic_high_key: tuple[Any, ...] | None = None
    stochastic_high_no_improve_evals = 0
    recent_eval_value_losses: deque[float] = deque(
        maxlen=train_cfg.early_stop_value_loss_window
    )
    stop_training_early = False
    stop_reason: str | None = None
    while (
        global_step < train_cfg.total_timesteps
        or rollout_backlog
        or pending_transition_by_drone
    ):
        while global_step < train_cfg.total_timesteps:
            if (
                len(rollout_backlog) >= train_cfg.rollout_steps
                and (
                    _resolve_rollout_prefix_bootstrap_values(
                        transitions=rollout_backlog,
                        prefix_len=len(rollout_backlog),
                        boundary_bootstrap_values=_build_rollout_prefix_boundary_bootstrap_values(
                            current_result=current_result,
                            episode_id=episode_id,
                            env=env,
                            tensorizer=tensorizer,
                            critic_builder=critic_builder,
                            critic_schema=critic_schema,
                            model=model,
                            history_buffer=history_buffer,
                            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
                            lstm_state_by_drone=lstm_state_by_drone,
                            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
                            successor_bootstrap_values=successor_bootstrap_values,
                        ),
                        current_runtime_state=current_result.runtime_state,
                        episode_terminated_at_boundary=bool(current_result.done),
                    )
                    is not None
                )
            ):
                break
            if current_result.done:
                if pending_transition_by_drone:
                    raise RuntimeError("episode 已结束，但仍有未 finalize 的 pending transition")
                (
                    current_train_arrival_band,
                    current_train_arrival_rate,
                    current_train_order_source,
                ) = _build_training_episode_order_source(
                    scene_ctx=scene_ctx,
                    config_path=config_path,
                    eval_cfg=eval_cfg,
                    base_order_source=order_source,
                    update_idx=update_idx,
                )
                env.set_order_source(current_train_order_source)
                current_result = env.reset()
                episode_id += 1
                episode_acc = _EpisodeAccumulator(
                    phase="train",
                    episode_id=episode_id,
                    order_source_mode=str(order_source.mode.value),
                    order_source_seed=env.current_episode_order_source_seed(),
                    train_arrival_band=current_train_arrival_band,
                    train_arrival_rate_per_min=current_train_arrival_rate,
                    global_step_start=global_step,
                    update_start=update_idx,
                )
                history_buffer.clear()
                lstm_state_by_drone.clear()
                recurrent_segment_id_by_drone.clear()
                local_decision_cursor.clear()
                _emit_event_hook(
                    event_hook,
                    "train_reset",
                    {
                        "phase": "train",
                        "episode_id": episode_id,
                        "global_step": global_step,
                        "update_idx": update_idx,
                        "train_arrival_band": current_train_arrival_band,
                        "train_arrival_rate_per_min": current_train_arrival_rate,
                        "env": env,
                        "result": current_result,
                    },
                )
            decision_context = current_result.decision_context
            if decision_context is None:
                if pending_transition_by_drone:
                    raise RuntimeError("decision_context 丢失时仍有未 finalize 的 pending transition")
                (
                    current_train_arrival_band,
                    current_train_arrival_rate,
                    current_train_order_source,
                ) = _build_training_episode_order_source(
                    scene_ctx=scene_ctx,
                    config_path=config_path,
                    eval_cfg=eval_cfg,
                    base_order_source=order_source,
                    update_idx=update_idx,
                )
                env.set_order_source(current_train_order_source)
                current_result = env.reset()
                episode_id += 1
                episode_acc = _EpisodeAccumulator(
                    phase="train",
                    episode_id=episode_id,
                    order_source_mode=str(order_source.mode.value),
                    order_source_seed=env.current_episode_order_source_seed(),
                    train_arrival_band=current_train_arrival_band,
                    train_arrival_rate_per_min=current_train_arrival_rate,
                    global_step_start=global_step,
                    update_start=update_idx,
                )
                history_buffer.clear()
                lstm_state_by_drone.clear()
                recurrent_segment_id_by_drone.clear()
                local_decision_cursor.clear()
                _emit_event_hook(
                    event_hook,
                    "train_reset",
                    {
                        "phase": "train",
                        "episode_id": episode_id,
                        "global_step": global_step,
                        "update_idx": update_idx,
                        "train_arrival_band": current_train_arrival_band,
                        "train_arrival_rate_per_min": current_train_arrival_rate,
                        "env": env,
                        "result": current_result,
                    },
                )
                continue

            drone_id = str(decision_context.deciding_drone_id)
            lstm_state_in = lstm_state_by_drone.get(drone_id)
            recurrent_segment_id = recurrent_segment_id_by_drone.get(drone_id, 0)
            decision_key = (episode_id, drone_id, recurrent_segment_id)
            local_decision_index = local_decision_cursor.get(decision_key, 0)
            candidate_out = _build_candidate_output(
                env=env,
                decision_context=decision_context,
                last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
            )
            observation_batch = tensorizer.build(
                decision_context=decision_context,
                candidate_out=candidate_out,
                transition_history=tuple(history_buffer),
            )
            action_mask = tensorizer.build_action_mask(candidate_out)
            critic_batch = critic_builder.build(
                decision_context=decision_context,
                critic_tensor_schema_meta=critic_schema,
            )
            with torch.no_grad():
                policy_out, next_lstm_state = model.forward(
                    observation_batch=observation_batch,
                    action_mask=action_mask,
                    critic_batch=critic_batch,
                    lstm_state=lstm_state_in,
                )
                sampled_action, log_prob = model.sample_action(
                    policy_out=policy_out,
                    action_mask=action_mask,
                    deterministic=False,
                )
            value_old = float(policy_out.value.detach().cpu().item())
            action_indices = ResolvedActionIndices(**sampled_action)
            env_action = candidate_out.resolved_action_lookup.resolve(
                root_branch_idx=action_indices.root_branch_idx,
                order_idx=action_indices.order_idx,
                mode_idx=action_indices.mode_idx,
            )
            step_result = env.step(env_action)
            (
                carry_in_reward,
                post_action_reward,
                mode_c_attempt_reward,
            ) = _extract_attributed_step_reward_parts(
                step_result=step_result
            )
            _finalize_pending_transition_for_next_decision(
                pending_transition_by_drone=pending_transition_by_drone,
                rollout_backlog=rollout_backlog,
                successor_bootstrap_values=successor_bootstrap_values,
                drone_id=drone_id,
                carry_in_reward=carry_in_reward,
                next_value=value_old,
                decision_context=decision_context,
                rendezvous_arrive_bonus=train_cfg.rendezvous_arrive_bonus,
                rendezvous_bonus=train_cfg.rendezvous_bonus,
                reward_scale=train_cfg.reward_scale,
            )
            transition_summary = tensorizer.build_transition_summary(
                decision_context=decision_context,
                candidate_out=candidate_out,
                action_indices=action_indices,
                step_result=step_result,
            )
            (
                shaped_post_action_reward,
                arrive_bonus_applied,
                success_bonus_applied,
            ) = _shape_post_action_reward_for_rendezvous(
                post_action_reward=(
                    post_action_reward * train_cfg.reward_scale
                    + (
                        train_cfg.mode_c_attempt_bonus
                        if mode_c_attempt_reward > 0.0
                        else 0.0
                    )
                    - mode_c_attempt_reward * train_cfg.reward_scale
                ),
                action_indices=action_indices,
                transition_summary=transition_summary,
                rendezvous_arrive_bonus=train_cfg.rendezvous_arrive_bonus,
                rendezvous_bonus=train_cfg.rendezvous_bonus,
            )
            current_transition = RolloutTransition(
                observation_batch=observation_batch,
                critic_batch=critic_batch,
                action_mask=action_mask,
                action_indices=action_indices,
                log_prob_old=float(log_prob.detach().cpu().item()),
                value_old=value_old,
                critic_schema_hash=critic_schema.schema_hash,
                reward=shaped_post_action_reward,
                done=True if step_result.done else None,
                lstm_state_in=_detach_lstm_state(lstm_state=lstm_state_in),
                lstm_state_out=_detach_lstm_state(lstm_state=next_lstm_state),
                episode_id=episode_id,
                actor_drone_id=drone_id,
                recurrent_segment_id=recurrent_segment_id,
                local_decision_index=local_decision_index,
                global_decision_index=global_step,
                decision_context_debug_snapshot=decision_context,
                sample_loss_weight=_compute_transition_loss_weight(
                    train_cfg=train_cfg,
                    action_indices=action_indices,
                    candidate_out=candidate_out,
                ),
                rendezvous_arrive_bonus_applied=arrive_bonus_applied,
                rendezvous_success_bonus_applied=success_bonus_applied,
                fallback_cause=str(getattr(transition_summary, "fallback_cause", "none")),
            )
            terminal_reward_by_drone: dict[str, float] = {}
            if step_result.done:
                _insert_finalized_transition_in_order(
                    rollout_backlog=rollout_backlog,
                    transition=current_transition,
                )
                terminal_reward_by_drone = env.consume_terminal_agent_costs()
                _flush_terminal_pending_transitions(
                    pending_transition_by_drone=pending_transition_by_drone,
                    rollout_backlog=rollout_backlog,
                    terminal_reward_by_drone=terminal_reward_by_drone,
                    runtime_state=step_result.runtime_state,
                    rendezvous_arrive_bonus=train_cfg.rendezvous_arrive_bonus,
                    rendezvous_bonus=train_cfg.rendezvous_bonus,
                    reward_scale=train_cfg.reward_scale,
                )
            else:
                pending_transition_by_drone[drone_id] = current_transition
            history_buffer.append(transition_summary)
            last_seen_plan_version_by_drone[decision_context.deciding_drone_id] = (
                decision_context.coarse_plan.plan_version
            )
            local_decision_cursor[decision_key] = local_decision_index + 1
            lstm_state_by_drone[drone_id] = _detach_lstm_state(lstm_state=next_lstm_state)
            _reset_recurrent_state_for_failed_drones(
                runtime_state=step_result.runtime_state,
                lstm_state_by_drone=lstm_state_by_drone,
                recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
            )
            current_result = step_result
            global_step += 1
            _record_episode_step(
                episode_acc,
                reward=float(step_result.reward),
                reward_scale=train_cfg.reward_scale,
            )
            _emit_event_hook(
                event_hook,
                "train_step",
                {
                    "phase": "train",
                    "episode_id": episode_id,
                    "global_step": global_step,
                    "update_idx": update_idx,
                    "env": env,
                    "decision_context": decision_context,
                    "env_action": env_action,
                    "step_result": step_result,
                },
            )
            if step_result.done:
                _record_terminal_episode_rewards(
                    accumulator=episode_acc,
                    terminal_reward_by_drone=terminal_reward_by_drone,
                    reward_scale=train_cfg.reward_scale,
                )
                episode_metrics = _finalize_episode_metrics(
                    accumulator=episode_acc,
                    episode_snapshot=env.build_episode_metrics_snapshot(),
                    global_step_end=global_step,
                    update_idx_end=update_idx,
                )
                _append_metrics(
                    episode_metrics_path,
                    episode_metrics,
                )
                _append_planner_replan_events(
                    planner_replan_events_path,
                    episode_metrics=episode_metrics,
                    events=env.build_planner_replan_events_snapshot(),
                )
                _emit_event_hook(
                    event_hook,
                    "train_episode_end",
                    {
                        "phase": "train",
                        "episode_id": episode_id,
                        "global_step": global_step,
                        "update_idx": update_idx,
                        "env": env,
                        "episode_metrics": episode_metrics,
                    },
                )

        if global_step >= train_cfg.total_timesteps and pending_transition_by_drone:
            current_result = _drain_pending_transitions_after_collection(
                current_result=current_result,
                env=env,
                model=model,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
                history_buffer=history_buffer,
                lstm_state_by_drone=lstm_state_by_drone,
                recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
                pending_transition_by_drone=pending_transition_by_drone,
                rollout_backlog=rollout_backlog,
                successor_bootstrap_values=successor_bootstrap_values,
                train_cfg=train_cfg,
            )
        if global_step >= train_cfg.total_timesteps and rollout_backlog:
            current_result = _advance_until_rollout_bootstraps_resolved_after_collection(
                current_result=current_result,
                episode_id=episode_id,
                env=env,
                model=model,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
                history_buffer=history_buffer,
                lstm_state_by_drone=lstm_state_by_drone,
                recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
                rollout_backlog=rollout_backlog,
                successor_bootstrap_values=successor_bootstrap_values,
            )

        if not rollout_backlog:
            break
        batch_size = len(rollout_backlog)
        boundary_bootstrap_values = _build_rollout_prefix_boundary_bootstrap_values(
            current_result=current_result,
            episode_id=episode_id,
            env=env,
            tensorizer=tensorizer,
            critic_builder=critic_builder,
            critic_schema=critic_schema,
            model=model,
            history_buffer=history_buffer,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
            lstm_state_by_drone=lstm_state_by_drone,
            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
            successor_bootstrap_values=successor_bootstrap_values,
        )
        tail_bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=rollout_backlog,
            prefix_len=batch_size,
            boundary_bootstrap_values=boundary_bootstrap_values,
            current_runtime_state=current_result.runtime_state,
            episode_terminated_at_boundary=bool(current_result.done),
        )
        if tail_bootstrap_values is None:
            raise RuntimeError("rollout prefix bootstrap 未解析完成，不能执行 PPO update")
        batch_view = build_batch_view_from_transitions(rollout_backlog[:batch_size])
        ppo_stats = _ppo_update(
            model=model,
            optimizer=optimizer,
            batch_view=batch_view,
            train_cfg=train_cfg,
            device=device,
            tail_bootstrap_values=tail_bootstrap_values,
        )
        rollout_backlog.clear()
        successor_bootstrap_values.clear()
        update_idx += 1
        _append_metrics(
            metrics_path,
            {
                "update": update_idx,
                "global_step": global_step,
                "rollout_size": batch_size,
                **ppo_stats,
            },
        )
        should_break_after_update = False
        if (
            train_cfg.eval_interval_updates > 0
            and (
                update_idx % train_cfg.eval_interval_updates == 0
                or global_step >= train_cfg.total_timesteps
            )
        ):
            eval_report = _run_periodic_evaluation(
                scene_ctx=scene_ctx,
                config_path=Path(config_path),
                model=model,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                policy_cfg=policy_cfg,
                eval_cfg=eval_cfg,
                train_cfg=train_cfg,
                device=device,
                update_idx=update_idx,
                global_step=global_step,
            )
            latest_eval_report = eval_report
            eval_count += 1
            _append_metrics(
                eval_metrics_path,
                {
                    "update": update_idx,
                    "global_step": global_step,
                    "benchmark": _strip_episode_details(eval_report["benchmark"]),
                    **(
                        {
                            "c_sensitive": _strip_episode_details(
                                eval_report["c_sensitive"]
                            )
                        }
                        if "c_sensitive" in eval_report
                        else {}
                    ),
                    "stochastic": {
                        name: _strip_episode_details(payload)
                        for name, payload in eval_report["stochastic"].items()
                    },
                },
            )
            _write_json(
                path=benchmark_report_path,
                payload=eval_report["benchmark"],
            )
            _write_json(
                path=stochastic_report_path,
                payload=eval_report["stochastic_report"],
            )
            if "c_sensitive" in eval_report:
                _write_json(
                    path=c_sensitive_report_path,
                    payload=eval_report["c_sensitive"],
                )
            selection_key = _build_eval_selection_key(eval_report)
            if best_checkpoint is None or selection_key > best_checkpoint.selection_key:
                _save_policy_checkpoint(
                    path=policy_best_path,
                    model=model,
                    optimizer=optimizer,
                    critic_schema_hash=critic_schema.schema_hash,
                    model_version=model_version,
                    global_step=global_step,
                    update_idx=update_idx,
                )
                best_checkpoint = _BestCheckpointState(
                    path=policy_best_path,
                    update_idx=update_idx,
                    global_step=global_step,
                    selection_key=selection_key,
                    eval_report=eval_report,
                )
                _emit_event_hook(
                    event_hook,
                    "train_best_checkpoint_updated",
                    {
                        "update": update_idx,
                        "global_step": global_step,
                        "selection_key": list(selection_key),
                        "path": str(policy_best_path),
                    },
                )

            stochastic_high_key = _build_stochastic_high_improvement_key(eval_report)
            if best_stochastic_high_key is None or stochastic_high_key > best_stochastic_high_key:
                best_stochastic_high_key = stochastic_high_key
                stochastic_high_no_improve_evals = 0
            else:
                stochastic_high_no_improve_evals += 1

            recent_eval_value_losses.append(float(ppo_stats["value_loss"]))
            if _should_stop_early(
                train_cfg=train_cfg,
                eval_count=eval_count,
                stochastic_high_no_improve_evals=stochastic_high_no_improve_evals,
                recent_eval_value_losses=tuple(recent_eval_value_losses),
            ):
                stop_training_early = True
                stop_reason = (
                    "early_stop:"
                    f"stochastic_high_no_improve={stochastic_high_no_improve_evals},"
                    f"recent_eval_value_losses={[round(value, 6) for value in recent_eval_value_losses]}"
                )
                should_break_after_update = True
                _emit_event_hook(
                    event_hook,
                    "train_early_stop_triggered",
                    {
                        "update": update_idx,
                        "global_step": global_step,
                        "reason": stop_reason,
                    },
                )

        if update_idx % train_cfg.save_interval_updates == 0 or global_step >= train_cfg.total_timesteps:
            _save_policy_checkpoint(
                path=policy_last_path,
                model=model,
                optimizer=optimizer,
                critic_schema_hash=critic_schema.schema_hash,
                model_version=model_version,
                global_step=global_step,
                update_idx=update_idx,
            )
        if should_break_after_update:
            break

    _save_policy_checkpoint(
        path=policy_last_path,
        model=model,
        optimizer=optimizer,
        critic_schema_hash=critic_schema.schema_hash,
        model_version=model_version,
        global_step=global_step,
        update_idx=update_idx,
    )
    selected_checkpoint = best_checkpoint.path if best_checkpoint is not None else policy_last_path
    if selected_checkpoint != policy_path:
        shutil.copy2(selected_checkpoint, policy_path)
    final_eval_report = (
        best_checkpoint.eval_report
        if best_checkpoint is not None
        else latest_eval_report
    )
    if final_eval_report is not None:
        _write_json(
            path=benchmark_report_path,
            payload=final_eval_report["benchmark"],
        )
        _write_json(
            path=stochastic_report_path,
            payload=final_eval_report["stochastic_report"],
        )
        if "c_sensitive" in final_eval_report:
            _write_json(
                path=c_sensitive_report_path,
                payload=final_eval_report["c_sensitive"],
            )
    meta_payload = _build_meta_payload(
        scene_ctx=scene_ctx,
        order_source=order_source,
        critic_schema=critic_schema,
        config_path=Path(config_path),
        model_version=model_version,
        effective_total_timesteps=global_step,
        train_cfg=train_cfg,
    )
    _validate_meta_payload_before_write(
        meta_payload=meta_payload,
        scene_ctx=scene_ctx,
        order_source=order_source,
        critic_schema=critic_schema,
        train_cfg=train_cfg,
        model_version=model_version,
        global_step=global_step,
        policy_path=policy_path,
        config_snapshot_path=config_snapshot_path,
        metrics_path=metrics_path,
    )
    _write_json(path=meta_path, payload=meta_payload)

    return {
        "model_version": model_version,
        "global_step": global_step,
        "updates": update_idx,
        "policy_path": str(policy_path),
        "policy_last_path": str(policy_last_path),
        "policy_best_path": str(policy_best_path) if best_checkpoint is not None else None,
        "bc_warm_start_policy_path": str(bc_warm_start_policy_path),
        "bc_warm_start_meta_path": str(bc_warm_start_meta_path),
        "meta_path": str(meta_path),
        "metrics_path": str(metrics_path),
        "episode_metrics_path": str(episode_metrics_path),
        "planner_replan_events_path": str(planner_replan_events_path),
        "eval_metrics_path": str(eval_metrics_path),
        "benchmark_report_path": str(benchmark_report_path),
        "stochastic_report_path": str(stochastic_report_path),
        "c_sensitive_report_path": str(c_sensitive_report_path),
        "stopped_early": bool(stop_training_early),
        "stop_reason": stop_reason,
        "selected_policy_source": str(selected_checkpoint),
        "bc_warm_start": bc_warm_start_stats,
    }


def _record_episode_step(
    accumulator: _EpisodeAccumulator,
    *,
    reward: float,
    reward_scale: float = 1.0,
) -> None:
    accumulator.decision_count += 1
    accumulator.total_reward += float(reward) * float(reward_scale)


def _record_terminal_episode_rewards(
    *,
    accumulator: _EpisodeAccumulator,
    terminal_reward_by_drone: Mapping[str, float],
    reward_scale: float = 1.0,
) -> float:
    """将 episode 结束时残留的尾部归因奖励按训练尺度补记回 total_reward。"""
    terminal_reward_total = 0.0
    for reward in terminal_reward_by_drone.values():
        terminal_reward_total += float(reward)
    scaled_terminal_reward_total = terminal_reward_total * float(reward_scale)
    accumulator.total_reward += scaled_terminal_reward_total
    return scaled_terminal_reward_total


def _finalize_episode_metrics(
    *,
    accumulator: _EpisodeAccumulator,
    episode_snapshot: Mapping[str, Any],
    global_step_end: int | None,
    update_idx_end: int | None,
) -> dict[str, Any]:
    payload = {
        "phase": accumulator.phase,
        "episode_id": int(accumulator.episode_id),
        "order_source_mode": str(accumulator.order_source_mode),
        "order_source_seed": int(accumulator.order_source_seed),
        "total_reward": float(accumulator.total_reward),
        "episode_length_decisions": int(accumulator.decision_count),
        "global_step_start": accumulator.global_step_start,
        "global_step_end": global_step_end,
        "update_start": accumulator.update_start,
        "update_end": update_idx_end,
    }
    if accumulator.train_arrival_band is not None:
        payload["train_arrival_band"] = str(accumulator.train_arrival_band)
    if accumulator.train_arrival_rate_per_min is not None:
        payload["train_arrival_rate_per_min"] = float(
            accumulator.train_arrival_rate_per_min
        )
    payload.update(dict(episode_snapshot))
    return payload


def _build_c_sensitive_eval_order_source(
    *,
    scene_ctx: Any,
    config_path: Path,
    eval_cfg: _EvalConfig,
) -> tuple[OrderSourceConfig, dict[str, Any]]:
    """基于现有 env 语义筛出一个额外的 benchmark-like `C-sensitive` replay split。"""
    full_benchmark_order_source = build_order_source(
        scene_ctx,
        mode=OrderSourceMode.BENCHMARK,
        seed=eval_cfg.c_sensitive_seed,
        overrides={"benchmark_use_dynamic_orders": True},
        config_path=config_path,
    )
    analysis_order_source = build_order_source(
        scene_ctx,
        mode=OrderSourceMode.BENCHMARK,
        seed=eval_cfg.c_sensitive_seed,
        overrides={"benchmark_use_dynamic_orders": False},
        config_path=config_path,
    )
    analysis_env = TrainingEnvAdapter(
        scene_ctx=scene_ctx,
        order_source=analysis_order_source,
        config_path=config_path,
    )
    analysis_env.reset()
    order_mgr = analysis_env._require_order_manager()
    entity_mgr = analysis_env._require_entity_manager()
    depot = analysis_env._require_depot()
    depot_home_drone_id = next(
        (
            drone_id
            for drone_id, state in analysis_env._drone_state.items()
            if state == TrainingDroneState.IDLE
        ),
        None,
    )
    if depot_home_drone_id is None:
        raise ValueError("C-sensitive eval 需要至少一架 depot-home idle UAV")
    drone = entity_mgr.drones[depot_home_drone_id]

    selected_order_ids: list[str] = []
    candidate_count_by_order_id: dict[str, int] = {}
    skipped_after_horizon = 0
    for dynamic_order in scene_ctx.dynamic_orders:
        if float(dynamic_order.spawn_sim_s) > analysis_env._cfg.upper_horizon_sec:
            skipped_after_horizon += 1
            continue
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        analysis_env._t_now = float(dynamic_order.spawn_sim_s)
        drone.current_loc = copy.deepcopy(depot.location)
        drone.battery_current = float(drone.battery_max)
        analysis_env._drone_state[depot_home_drone_id] = TrainingDroneState.IDLE

        order = copy.deepcopy(dynamic_order.order)
        order_mgr.pending_orders[order.order_id] = order
        coarse_plan = analysis_env._build_coarse_plan_view(analysis_env._t_now)
        legal_mode_c_nodes = analysis_env._iter_feasible_mode_c_recovery_nodes(
            drone=drone,
            order=order,
            coarse_plan=coarse_plan,
        )
        if len(legal_mode_c_nodes) < eval_cfg.c_sensitive_min_legal_mode_c_recovery_nodes:
            continue
        if not analysis_env._has_mode_b_return_host(drone, order):
            continue
        selected_order_ids.append(order.order_id)
        candidate_count_by_order_id[order.order_id] = len(legal_mode_c_nodes)

    if not selected_order_ids:
        raise ValueError(
            "C-sensitive eval 没有筛出任何动态订单；"
            "请检查默认场景或降低最小 legal mode C 候选数阈值"
        )

    normalized_dynamic_by_id = {
        str(item["order_id"]): dict(item)
        for item in full_benchmark_order_source.scheduled_dynamic_orders
    }
    selected_dynamic_orders = tuple(
        dict(normalized_dynamic_by_id[order_id]) for order_id in selected_order_ids
    )
    return (
        replace(
            full_benchmark_order_source,
            scheduled_dynamic_orders=selected_dynamic_orders,
            seed=int(eval_cfg.c_sensitive_seed),
        ),
        {
            "selected_dynamic_order_count": int(len(selected_order_ids)),
            "selected_dynamic_order_ids": tuple(selected_order_ids),
            "selected_dynamic_order_legal_mode_c_node_counts": dict(
                candidate_count_by_order_id
            ),
            "skipped_dynamic_order_count_after_horizon": int(skipped_after_horizon),
            "c_sensitive_min_legal_mode_c_recovery_nodes": int(
                eval_cfg.c_sensitive_min_legal_mode_c_recovery_nodes
            ),
            "selection_drone_id": str(depot_home_drone_id),
        },
    )


def _run_periodic_evaluation(
    *,
    scene_ctx: Any,
    config_path: Path,
    model: Any,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    policy_cfg: _PolicyConfig,
    eval_cfg: _EvalConfig,
    train_cfg: _TrainingConfig,
    device: Any,
    update_idx: int,
    global_step: int,
) -> dict[str, Any]:
    rng_state = _capture_rng_state()
    was_training = bool(model.training)
    model.eval()
    try:
        generated_at = datetime.now(timezone.utc).isoformat()
        c_sensitive_report: dict[str, Any] | None = None
        # benchmark 当前是确定性 replay：固定订单流 + deterministic policy。
        # 因此重复跑多次不会提供额外信息，统一折叠为单次评估，并显式记录
        # configured 次数与 effective 次数，避免报表误导 postmortem。
        benchmark_episode_count = 1
        benchmark_episodes = [
            _evaluate_policy_episode(
                scene_ctx=scene_ctx,
                config_path=config_path,
                order_source=build_order_source(
                    scene_ctx,
                    mode=OrderSourceMode.BENCHMARK,
                    seed=eval_cfg.benchmark_seed,
                    config_path=config_path,
                ),
                model=model,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                policy_cfg=policy_cfg,
                device=device,
                eval_phase="benchmark",
                episode_id=0,
                reward_scale=train_cfg.reward_scale,
            )
        ]
        benchmark_report = _summarize_episode_records(
            split="benchmark",
            order_source_mode=OrderSourceMode.BENCHMARK.value,
            episodes=benchmark_episodes,
            update_idx=update_idx,
            global_step=global_step,
            generated_at=generated_at,
            extra={
                "benchmark_seed": int(eval_cfg.benchmark_seed),
                "benchmark_eval_episodes_configured": int(train_cfg.benchmark_eval_episodes),
                "benchmark_effective_episode_count": int(benchmark_episode_count),
                "benchmark_deterministic_replay": True,
            },
        )
        if eval_cfg.c_sensitive_enabled:
            c_sensitive_order_source, c_sensitive_meta = _build_c_sensitive_eval_order_source(
                scene_ctx=scene_ctx,
                config_path=config_path,
                eval_cfg=eval_cfg,
            )
            c_sensitive_episode = _evaluate_policy_episode(
                scene_ctx=scene_ctx,
                config_path=config_path,
                order_source=c_sensitive_order_source,
                model=model,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                policy_cfg=policy_cfg,
                device=device,
                eval_phase="c_sensitive",
                episode_id=0,
                reward_scale=train_cfg.reward_scale,
            )
            c_sensitive_report = _summarize_episode_records(
                split="c_sensitive",
                order_source_mode=OrderSourceMode.BENCHMARK.value,
                episodes=[c_sensitive_episode],
                update_idx=update_idx,
                global_step=global_step,
                generated_at=generated_at,
                extra={
                    "c_sensitive_seed": int(eval_cfg.c_sensitive_seed),
                    **c_sensitive_meta,
                },
            )

        stochastic_reports: dict[str, dict[str, Any]] = {}
        for band_name, arrival_rate in eval_cfg.stochastic_arrival_rates:
            band_episodes = []
            for seed_offset in range(train_cfg.stochastic_eval_seeds):
                order_source = build_order_source(
                    scene_ctx,
                    mode=OrderSourceMode.POISSON,
                    seed=eval_cfg.stochastic_seed_base + seed_offset,
                    overrides={"poisson_arrival_rate": arrival_rate},
                    config_path=config_path,
                )
                band_episodes.append(
                    _evaluate_policy_episode(
                        scene_ctx=scene_ctx,
                        config_path=config_path,
                        order_source=order_source,
                        model=model,
                        tensorizer=tensorizer,
                        critic_builder=critic_builder,
                        critic_schema=critic_schema,
                        policy_cfg=policy_cfg,
                        device=device,
                        eval_phase=f"stochastic_{band_name}",
                        episode_id=seed_offset,
                        reward_scale=train_cfg.reward_scale,
                    )
                )
            stochastic_reports[band_name] = _summarize_episode_records(
                split=f"stochastic_{band_name}",
                order_source_mode=OrderSourceMode.POISSON.value,
                episodes=band_episodes,
                update_idx=update_idx,
                global_step=global_step,
                generated_at=generated_at,
                extra={
                    "arrival_rate_per_min": float(arrival_rate),
                    "seed_base": int(eval_cfg.stochastic_seed_base),
                    "seed_count": int(train_cfg.stochastic_eval_seeds),
                },
            )

        payload = {
            "benchmark": benchmark_report,
            "stochastic": stochastic_reports,
            "stochastic_report": {
                "generated_at": generated_at,
                "update": int(update_idx),
                "global_step": int(global_step),
                "order_source_mode": OrderSourceMode.POISSON.value,
                "profiles": stochastic_reports,
            },
        }
        if c_sensitive_report is not None:
            payload["c_sensitive"] = c_sensitive_report
        return payload
    finally:
        _restore_rng_state(rng_state)
        if was_training:
            model.train()


def _evaluate_policy_episode(
    *,
    scene_ctx: Any,
    config_path: Path,
    order_source: Any,
    model: Any,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    policy_cfg: _PolicyConfig,
    device: Any,
    eval_phase: str,
    episode_id: int,
    reward_scale: float,
) -> dict[str, Any]:
    env = TrainingEnvAdapter(
        scene_ctx=scene_ctx,
        order_source=order_source,
        config_path=config_path,
    )
    result = env.reset()
    if result.decision_context is None:
        raise RuntimeError("eval reset 后没有可用 decision_context")

    last_seen_plan_version_by_drone: dict[str, int] = {}
    history_buffer: deque[Any] = deque(maxlen=policy_cfg.hist_len)
    lstm_state_by_drone: dict[str, Any] = {}
    recurrent_segment_id_by_drone: dict[str, int] = {}
    episode_acc = _EpisodeAccumulator(
        phase=eval_phase,
        episode_id=episode_id,
        order_source_mode=str(order_source.mode.value),
        order_source_seed=env.current_episode_order_source_seed(),
    )

    while not result.done:
        decision_context = result.decision_context
        if decision_context is None:
            raise RuntimeError("eval 过程中 decision_context 丢失")

        drone_id = str(decision_context.deciding_drone_id)
        candidate_out = _build_candidate_output(
            env=env,
            decision_context=decision_context,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
        )
        observation_batch = tensorizer.build(
            decision_context=decision_context,
            candidate_out=candidate_out,
            transition_history=tuple(history_buffer),
        )
        action_mask = tensorizer.build_action_mask(candidate_out)
        critic_batch = critic_builder.build(
            decision_context=decision_context,
            critic_tensor_schema_meta=critic_schema,
        )
        with torch.no_grad():
            policy_out, next_lstm_state = model.forward(
                observation_batch=observation_batch,
                action_mask=action_mask,
                critic_batch=critic_batch,
                lstm_state=lstm_state_by_drone.get(drone_id),
            )
            sampled_action, _ = model.sample_action(
                policy_out=policy_out,
                action_mask=action_mask,
                deterministic=True,
            )
        action_indices = ResolvedActionIndices(**sampled_action)
        env_action = candidate_out.resolved_action_lookup.resolve(
            root_branch_idx=action_indices.root_branch_idx,
            order_idx=action_indices.order_idx,
            mode_idx=action_indices.mode_idx,
        )
        step_result = env.step(env_action)
        history_buffer.append(
            tensorizer.build_transition_summary(
                decision_context=decision_context,
                candidate_out=candidate_out,
                action_indices=action_indices,
                step_result=step_result,
            )
        )
        last_seen_plan_version_by_drone[decision_context.deciding_drone_id] = (
            decision_context.coarse_plan.plan_version
        )
        lstm_state_by_drone[drone_id] = _detach_lstm_state(lstm_state=next_lstm_state)
        _reset_recurrent_state_for_failed_drones(
            runtime_state=step_result.runtime_state,
            lstm_state_by_drone=lstm_state_by_drone,
            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
        )
        _record_episode_step(
            episode_acc,
            reward=float(step_result.reward),
            reward_scale=reward_scale,
        )
        result = step_result

    terminal_reward_by_drone = env.consume_terminal_agent_costs()
    _record_terminal_episode_rewards(
        accumulator=episode_acc,
        terminal_reward_by_drone=terminal_reward_by_drone,
        reward_scale=reward_scale,
    )
    return _finalize_episode_metrics(
        accumulator=episode_acc,
        episode_snapshot=env.build_episode_metrics_snapshot(),
        global_step_end=None,
        update_idx_end=None,
    )


def _summarize_episode_records(
    *,
    split: str,
    order_source_mode: str,
    episodes: Sequence[Mapping[str, Any]],
    update_idx: int,
    global_step: int,
    generated_at: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError(f"{split} 评估结果不能为空")

    episode_count = len(episodes)
    mode_b_total = float(sum(float(item["dispatch_mode_b_count"]) for item in episodes))
    mode_c_total = float(sum(float(item["dispatch_mode_c_count"]) for item in episodes))
    dispatch_total = mode_b_total + mode_c_total
    dispatch_decision_total = int(
        sum(int(item.get("dispatch_decision_count", 0)) for item in episodes)
    )
    dispatch_decision_with_legal_mode_c_total = int(
        sum(
            int(item.get("dispatch_decision_with_legal_mode_c_count", 0))
            for item in episodes
        )
    )
    feasible_mode_c_recover_node_total = int(
        sum(
            int(item.get("feasible_mode_c_recover_node_count_total", 0))
            for item in episodes
        )
    )
    mode_c_revalidation_fail_reason_totals: dict[str, int] = {}
    mode_c_timeout_from_state_totals: dict[str, int] = {}
    mode_c_fallback_from_state_totals: dict[str, int] = {}
    mode_c_candidate_order_filter_totals: dict[str, int] = {}
    mode_c_candidate_node_filter_totals: dict[str, int] = {}
    fallback_cause_totals: dict[str, int] = {}
    reservation_release_cause_totals: dict[str, int] = {}
    completed_with_timing_total = int(
        sum(int(item.get("completed_with_timing_order_count", 0)) for item in episodes)
    )
    order_delay_sum_min_total = float(
        sum(float(item.get("order_delay_sum_min", 0.0)) for item in episodes)
    )
    total_tardiness_sec = float(
        sum(float(item.get("total_tardiness_sec", 0.0)) for item in episodes)
    )
    completed_with_tardiness_total = int(
        sum(int(item.get("completed_with_timing_order_count", 0)) for item in episodes)
    )
    for item in episodes:
        for reason_key, count in dict(
            item.get("mode_c_post_delivery_revalidation_fail_reasons", {})
        ).items():
            mode_c_revalidation_fail_reason_totals[str(reason_key)] = (
                mode_c_revalidation_fail_reason_totals.get(str(reason_key), 0)
                + int(count)
            )
        for state_key, count in dict(item.get("mode_c_timeout_from_state", {})).items():
            mode_c_timeout_from_state_totals[str(state_key)] = (
                mode_c_timeout_from_state_totals.get(str(state_key), 0) + int(count)
            )
        for state_key, count in dict(item.get("mode_c_fallback_from_state", {})).items():
            mode_c_fallback_from_state_totals[str(state_key)] = (
                mode_c_fallback_from_state_totals.get(str(state_key), 0) + int(count)
            )
        for reason_key, count in dict(
            item.get("mode_c_candidate_order_filter_counts", {})
        ).items():
            mode_c_candidate_order_filter_totals[str(reason_key)] = (
                mode_c_candidate_order_filter_totals.get(str(reason_key), 0)
                + int(count)
            )
        for reason_key, count in dict(
            item.get("mode_c_candidate_node_filter_counts", {})
        ).items():
            mode_c_candidate_node_filter_totals[str(reason_key)] = (
                mode_c_candidate_node_filter_totals.get(str(reason_key), 0)
                + int(count)
            )
        for cause_key, count in dict(item.get("fallback_cause_counts", {})).items():
            fallback_cause_totals[str(cause_key)] = (
                fallback_cause_totals.get(str(cause_key), 0) + int(count)
            )
        for cause_key, count in dict(
            item.get("reservation_release_cause_counts", {})
        ).items():
            reservation_release_cause_totals[str(cause_key)] = (
                reservation_release_cause_totals.get(str(cause_key), 0) + int(count)
            )
    payload = {
        "generated_at": generated_at,
        "update": int(update_idx),
        "global_step": int(global_step),
        "split": str(split),
        "order_source_mode": str(order_source_mode),
        "episode_count": int(episode_count),
        "mean_total_reward": float(np.mean([float(item["total_reward"]) for item in episodes])),
        "mean_episode_length_decisions": float(
            np.mean([float(item["episode_length_decisions"]) for item in episodes])
        ),
        "mean_episode_end_t_sec": float(
            np.mean([float(item["episode_end_t_sec"]) for item in episodes])
        ),
        "mean_delivery_count": float(np.mean([float(item["delivery_count"]) for item in episodes])),
        "sum_delivery_count": int(sum(int(item["delivery_count"]) for item in episodes)),
        "mean_on_time_rate": float(np.mean([float(item["on_time_rate"]) for item in episodes])),
        "mean_avg_order_delay_min": float(
            np.mean([float(item.get("avg_order_delay_min", 0.0)) for item in episodes])
        ),
        "sum_completed_with_timing_order_count": int(completed_with_timing_total),
        "sum_order_delay_min": float(order_delay_sum_min_total),
        "sum_total_tardiness_sec": float(total_tardiness_sec),
        "mean_tardiness_sec": (
            float(total_tardiness_sec) / float(completed_with_tardiness_total)
            if completed_with_tardiness_total > 0
            else 0.0
        ),
        "max_tardiness_sec": float(
            max(float(item.get("max_tardiness_sec", 0.0)) for item in episodes)
        ),
        "sum_late_delivery_count": int(
            sum(int(item.get("late_delivery_count", 0)) for item in episodes)
        ),
        "weighted_avg_order_delay_min": (
            float(order_delay_sum_min_total) / float(completed_with_timing_total)
            if completed_with_timing_total > 0
            else 0.0
        ),
        "sum_required_primary_order_count": int(
            sum(int(item.get("required_primary_order_count", 0)) for item in episodes)
        ),
        "mean_completion_rate": float(
            np.mean([float(item.get("completion_rate", 0.0)) for item in episodes])
        ),
        "mean_required_on_time_rate": float(
            np.mean([float(item.get("required_on_time_rate", 0.0)) for item in episodes])
        ),
        "sum_timeout_order_count": int(sum(int(item["timeout_order_count"]) for item in episodes)),
        "sum_unserved_primary_order_count": int(
            sum(int(item.get("unserved_primary_order_count", 0)) for item in episodes)
        ),
        "sum_fallback_count": int(sum(int(item["fallback_count"]) for item in episodes)),
        "sum_hard_failure_count": int(sum(int(item["hard_failure_count"]) for item in episodes)),
        "sum_reservation_timeout_count": int(
            sum(int(item["reservation_timeout_count"]) for item in episodes)
        ),
        "sum_hard_overdue_count": int(
            sum(int(item["hard_overdue_count"]) for item in episodes)
        ),
        "sum_dispatch_decision_count": int(dispatch_decision_total),
        "sum_dispatch_decision_with_legal_mode_c_count": int(
            dispatch_decision_with_legal_mode_c_total
        ),
        "sum_feasible_mode_c_recover_node_count_total": int(
            feasible_mode_c_recover_node_total
        ),
        "sum_mode_c_candidate_order_filter_counts": dict(
            mode_c_candidate_order_filter_totals
        ),
        "sum_mode_c_candidate_node_filter_counts": dict(
            mode_c_candidate_node_filter_totals
        ),
        "avg_feasible_mode_c_nodes_per_dispatch_decision": (
            float(feasible_mode_c_recover_node_total) / float(dispatch_decision_total)
            if dispatch_decision_total > 0
            else 0.0
        ),
        "sum_mode_c_selected_count": int(
            sum(int(item.get("mode_c_selected_count", 0)) for item in episodes)
        ),
        "sum_mode_c_success_count": int(
            sum(int(item.get("mode_c_success_count", 0)) for item in episodes)
        ),
        "sum_mode_c_post_delivery_revalidation_fail_count": int(
            sum(
                int(item.get("mode_c_post_delivery_revalidation_fail_count", 0))
                for item in episodes
            )
        ),
        "sum_mode_c_post_delivery_revalidation_fail_reasons": dict(
            mode_c_revalidation_fail_reason_totals
        ),
        "sum_mode_c_selected_node_expired_count": int(
            sum(int(item.get("mode_c_selected_node_expired_count", 0)) for item in episodes)
        ),
        "sum_mode_c_selected_filter_margin_sum": float(
            sum(float(item.get("mode_c_selected_filter_margin_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_execution_slack_sum": float(
            sum(float(item.get("mode_c_selected_execution_slack_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_reservation_count_sum": float(
            sum(float(item.get("mode_c_selected_reservation_count_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_truck_eta_remaining_sum": float(
            sum(float(item.get("mode_c_selected_truck_eta_remaining_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_planned_truck_eta_sum": float(
            sum(float(item.get("mode_c_selected_planned_truck_eta_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_planned_uav_eta_sum": float(
            sum(float(item.get("mode_c_selected_planned_uav_eta_sum", 0.0)) for item in episodes)
        ),
        "sum_mode_c_selected_planned_slack_sum": float(
            sum(float(item.get("mode_c_selected_planned_slack_sum", 0.0)) for item in episodes)
        ),
        "mode_c_timeout_from_state": dict(mode_c_timeout_from_state_totals),
        "mode_c_fallback_from_state": dict(mode_c_fallback_from_state_totals),
        "fallback_cause_counts": dict(fallback_cause_totals),
        "sum_ppo_attributed_fallback_count": int(
            sum(int(item.get("ppo_attributed_fallback_count", 0)) for item in episodes)
        ),
        "sum_system_attributed_fallback_count": int(
            sum(int(item.get("system_attributed_fallback_count", 0)) for item in episodes)
        ),
        "reservation_release_cause_counts": dict(reservation_release_cause_totals),
        "sum_t_wait_sec": float(sum(float(item["t_wait_sec"]) for item in episodes)),
        "sum_t_idle_sec": float(sum(float(item["t_idle_sec"]) for item in episodes)),
        "sum_t_queue_sec": float(sum(float(item["t_queue_sec"]) for item in episodes)),
        "sum_t_fallback_sec": float(sum(float(item["t_fallback_sec"]) for item in episodes)),
        "sum_t_ppo_attributed_fallback_sec": float(
            sum(float(item.get("t_ppo_attributed_fallback_sec", 0.0)) for item in episodes)
        ),
        "sum_t_system_attributed_fallback_sec": float(
            sum(float(item.get("t_system_attributed_fallback_sec", 0.0)) for item in episodes)
        ),
        "sum_t_overdue_sec": float(sum(float(item["t_overdue_sec"]) for item in episodes)),
        "sum_t_reservation_timeout_cost_sec": float(
            sum(float(item["t_reservation_timeout_cost_sec"]) for item in episodes)
        ),
        "sum_episode_uav_energy_reward_penalty": float(
            sum(float(item.get("episode_uav_energy_reward_penalty", 0.0)) for item in episodes)
        ),
        "sum_episode_uav_energy_ratio_sum": float(
            sum(float(item.get("episode_uav_energy_ratio_sum", 0.0)) for item in episodes)
        ),
        "sum_episode_uav_energy_penalty_events": int(
            sum(int(item.get("episode_uav_energy_penalty_events", 0)) for item in episodes)
        ),
        "mode_b_dispatch_ratio": (
            float(mode_b_total / dispatch_total) if dispatch_total > 0.0 else 0.0
        ),
        "mode_c_dispatch_ratio": (
            float(mode_c_total / dispatch_total) if dispatch_total > 0.0 else 0.0
        ),
        "episodes": [dict(item) for item in episodes],
    }
    decision_batch_lengths = [
        float(item["episode_length_decision_batches"])
        for item in episodes
        if "episode_length_decision_batches" in item
    ]
    if len(decision_batch_lengths) == episode_count:
        payload["mean_episode_length_decision_batches"] = float(
            np.mean(decision_batch_lengths)
        )
        payload["sum_episode_length_decision_batches"] = int(
            sum(int(item["episode_length_decision_batches"]) for item in episodes)
        )
    if extra:
        payload.update(dict(extra))
    return payload


def _strip_episode_details(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.pop("episodes", None)
    return payload


def _count_done_reason(report: Mapping[str, Any], *, reason: str) -> int:
    episodes = report.get("episodes", ())
    if not isinstance(episodes, Sequence):
        return 0
    return sum(1 for episode in episodes if str(episode.get("done_reason")) == str(reason))


def _sum_undelivered_order_count(report: Mapping[str, Any]) -> int:
    return int(report.get("sum_timeout_order_count", 0)) + int(
        report.get("sum_unserved_primary_order_count", 0)
    )


def _build_benchmark_guardrail_key(eval_report: Mapping[str, Any]) -> tuple[Any, ...]:
    benchmark = dict(eval_report["benchmark"])
    return (
        -_sum_undelivered_order_count(benchmark),
        int(_count_done_reason(benchmark, reason="all_orders_cleared")),
    )


def _build_eval_selection_key(eval_report: Mapping[str, Any]) -> tuple[Any, ...]:
    stochastic = dict(eval_report["stochastic"])
    stochastic_high = dict(stochastic["high"])
    stochastic_medium = dict(stochastic["medium"])
    return (
        *_build_benchmark_guardrail_key(eval_report),
        -_sum_undelivered_order_count(stochastic_high),
        float(stochastic_high["mean_total_reward"]),
        -int(stochastic_high.get("sum_fallback_count", 0)),
        float(stochastic_medium["mean_total_reward"]),
        -_sum_undelivered_order_count(stochastic_medium),
        -int(stochastic_medium.get("sum_fallback_count", 0)),
    )


def _build_stochastic_high_improvement_key(eval_report: Mapping[str, Any]) -> tuple[Any, ...]:
    stochastic_high = dict(eval_report["stochastic"]["high"])
    return (
        -_sum_undelivered_order_count(stochastic_high),
        float(stochastic_high["mean_total_reward"]),
        -int(stochastic_high.get("sum_fallback_count", 0)),
    )


def _latest_value_loss_shows_meaningful_decline(
    *,
    recent_eval_value_losses: Sequence[float],
    min_delta: float,
) -> bool:
    if len(recent_eval_value_losses) < 2:
        return False
    baseline = min(float(value) for value in recent_eval_value_losses[:-1])
    latest = float(recent_eval_value_losses[-1])
    target = baseline * (1.0 - float(min_delta))
    return latest <= target


def _should_stop_early(
    *,
    train_cfg: _TrainingConfig,
    eval_count: int,
    stochastic_high_no_improve_evals: int,
    recent_eval_value_losses: Sequence[float],
) -> bool:
    if not train_cfg.early_stop_enabled:
        return False
    if eval_count < train_cfg.early_stop_min_evals:
        return False
    if stochastic_high_no_improve_evals < train_cfg.early_stop_stochastic_high_patience:
        return False
    if len(recent_eval_value_losses) < train_cfg.early_stop_value_loss_window:
        return False
    return not _latest_value_loss_shows_meaningful_decline(
        recent_eval_value_losses=recent_eval_value_losses,
        min_delta=train_cfg.early_stop_value_loss_min_delta,
    )


def _compute_value_loss_masked(
    *,
    values: Any,
    returns: Any,
    valid_mask: Any,
    loss_type: str,
    huber_delta: float,
    sample_weights: Any | None = None,
) -> Any:
    normalized_loss_type = str(loss_type).strip().lower()
    diff = values - returns
    if normalized_loss_type == "mse":
        per_timestep_loss = 0.5 * (diff ** 2)
    elif normalized_loss_type == "huber":
        abs_diff = torch.abs(diff)
        delta = float(huber_delta)
        quadratic = 0.5 * (diff ** 2)
        linear = delta * (abs_diff - 0.5 * delta)
        per_timestep_loss = torch.where(abs_diff <= delta, quadratic, linear)
    else:
        raise ValueError(f"未知 value_loss_type: {loss_type}")
    return _masked_weighted_mean_tensor(
        per_timestep_loss,
        valid_mask,
        sample_weights=sample_weights,
    )


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _build_current_boundary_bootstrap_values(
    *,
    current_result: Any,
    episode_id: int,
    env: TrainingEnvAdapter,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: Any,
    model: Any,
    history_buffer: Sequence[Any],
    last_seen_plan_version_by_drone: Mapping[str, int],
    lstm_state_by_drone: Mapping[str, Any],
    recurrent_segment_id_by_drone: Mapping[str, int],
) -> dict[tuple[int, str, int], float]:
    decision_context = current_result.decision_context
    if current_result.done or decision_context is None:
        return {}

    drone_id = str(decision_context.deciding_drone_id)
    candidate_out = _build_candidate_output(
        env=env,
        decision_context=decision_context,
        last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
    )
    observation_batch = tensorizer.build(
        decision_context=decision_context,
        candidate_out=candidate_out,
        transition_history=tuple(history_buffer),
    )
    action_mask = tensorizer.build_action_mask(candidate_out)
    critic_batch = critic_builder.build(
        decision_context=decision_context,
        critic_tensor_schema_meta=critic_schema,
    )
    with torch.no_grad():
        policy_out, _ = model.forward(
            observation_batch=observation_batch,
            action_mask=action_mask,
            critic_batch=critic_batch,
            lstm_state=lstm_state_by_drone.get(drone_id),
        )
    key = (
        int(episode_id),
        drone_id,
        int(recurrent_segment_id_by_drone.get(drone_id, 0)),
    )
    return {key: float(policy_out.value.detach().cpu().item())}


def _resolve_rollout_prefix_bootstrap_values(
    *,
    transitions: Sequence[RolloutTransition],
    prefix_len: int,
    boundary_bootstrap_values: Mapping[tuple[int, str, int], float],
    current_runtime_state: Any,
    episode_terminated_at_boundary: bool,
) -> dict[tuple[int, str, int], float] | None:
    if prefix_len <= 0:
        raise ValueError("prefix_len 必须为正数")
    if prefix_len > len(transitions):
        raise ValueError("prefix_len 不能超过当前 rollout backlog 长度")

    tail_index_by_key: dict[tuple[int, str, int], int] = {}
    for idx, transition in enumerate(transitions[:prefix_len]):
        key = (
            int(transition.episode_id),
            str(transition.actor_drone_id),
            int(transition.recurrent_segment_id),
        )
        tail_index_by_key[key] = idx

    resolved: dict[tuple[int, str, int], float] = {}
    for key, last_idx in tail_index_by_key.items():
        last_transition = transitions[last_idx]
        if bool(last_transition.done):
            resolved[key] = 0.0
            continue

        bootstrap_value = _find_rollout_tail_bootstrap_value(
            transitions=transitions,
            start_idx=last_idx + 1,
            key=key,
        )
        if bootstrap_value is not None:
            resolved[key] = bootstrap_value
            continue
        if key in boundary_bootstrap_values:
            resolved[key] = float(boundary_bootstrap_values[key])
            continue
        if _is_failed_drone_in_runtime_state(
            runtime_state=current_runtime_state,
            drone_id=key[1],
        ):
            resolved[key] = 0.0
            continue
        if episode_terminated_at_boundary:
            resolved[key] = 0.0
            continue
        return None
    return resolved


def _find_rollout_tail_bootstrap_value(
    *,
    transitions: Sequence[RolloutTransition],
    start_idx: int,
    key: tuple[int, str, int],
) -> float | None:
    episode_id, drone_id, recurrent_segment_id = key
    for later in transitions[start_idx:]:
        later_episode_id = int(later.episode_id)
        if later_episode_id > episode_id:
            return 0.0
        if later_episode_id < episode_id:
            continue
        if str(later.actor_drone_id) == drone_id:
            later_segment_id = int(later.recurrent_segment_id)
            if later_segment_id == recurrent_segment_id:
                return float(later.value_old)
            if later_segment_id > recurrent_segment_id:
                return 0.0
        if bool(later.done):
            return 0.0
    return None


def _is_failed_drone_in_runtime_state(*, runtime_state: Any, drone_id: str) -> bool:
    if runtime_state is None:
        return False
    drone_states = getattr(runtime_state, "drone_states", None)
    if drone_states is None or drone_id not in drone_states:
        return False
    return str(drone_states[drone_id].training_state) == "airborne_energy_failure"


def _bc_label_to_action_indices_tensor(
    label: ResolvedActionIndices,
    *,
    device: Any,
) -> dict[str, Any]:
    root_branch_idx = int(label.root_branch_idx)
    order_idx = 0 if label.order_idx is None else int(label.order_idx)
    mode_idx = 0 if label.mode_idx is None else int(label.mode_idx)
    return {
        "root_branch_idx": torch.as_tensor([root_branch_idx], dtype=torch.long, device=device),
        "order_idx": torch.as_tensor([order_idx], dtype=torch.long, device=device),
        "mode_idx": torch.as_tensor([mode_idx], dtype=torch.long, device=device),
    }


def _load_or_run_bc_warm_start(
    *,
    model: Any,
    optimizer: Any,
    scene_ctx: Any,
    order_source: OrderSourceConfig,
    config_path: Path,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    train_cfg: _TrainingConfig,
    device: Any,
    metrics_path: Path,
    checkpoint_path: Path | None = None,
    checkpoint_meta_path: Path | None = None,
    model_version: str | None = None,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    if not train_cfg.bc_warm_start_enabled:
        return _run_bc_warm_start(
            model=model,
            optimizer=optimizer,
            scene_ctx=scene_ctx,
            order_source=order_source,
            config_path=config_path,
            tensorizer=tensorizer,
            critic_builder=critic_builder,
            critic_schema=critic_schema,
            train_cfg=train_cfg,
            device=device,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            checkpoint_meta_path=checkpoint_meta_path,
            model_version=model_version,
            event_hook=event_hook,
        )

    load_path = _resolve_bc_warm_start_load_path(
        train_cfg=train_cfg,
        default_checkpoint_path=checkpoint_path,
    )
    if load_path is not None:
        stats = _load_bc_warm_start_checkpoint(
            path=load_path,
            model=model,
            bc_optimizer=optimizer,
            device=device,
            critic_schema_hash=critic_schema.schema_hash,
            train_cfg=train_cfg,
        )
        _emit_event_hook(
            event_hook,
            "bc_warm_start_checkpoint_loaded",
            {
                "phase": "bc_warm_start",
                "path": str(load_path),
                "target_checkpoint_path": str(checkpoint_path),
                "samples": int(stats.get("samples", 0)),
            },
        )
        if load_path.resolve() != checkpoint_path.resolve():
            shutil.copy2(load_path, checkpoint_path)
        _write_json(
            path=checkpoint_meta_path,
            payload={
                "phase": "bc_warm_start",
                "loaded_from_checkpoint": True,
                "source_checkpoint_path": str(load_path),
                "checkpoint_path": str(checkpoint_path),
                "model_version": str(model_version),
                "critic_schema_hash": str(critic_schema.schema_hash),
                "actor_observation_schema_hash": _actor_observation_schema_hash(),
                "actor_observation_schema": _actor_observation_schema_payload(),
                "config_path": str(config_path),
                **_bc_warm_start_checkpoint_stage_payload(
                    train_cfg=train_cfg,
                    stats=stats,
                    legacy_checkpoint=bool(
                        stats.get("bc_warm_start_legacy_checkpoint", False)
                    ),
                ),
                "stats": dict(stats),
            },
        )
        stats = {
            **dict(stats),
            "checkpoint_path": str(checkpoint_path),
            "source_checkpoint_path": str(load_path),
            "checkpoint_meta_path": str(checkpoint_meta_path),
        }
        return stats

    stats = _run_bc_warm_start(
        model=model,
        optimizer=optimizer,
        scene_ctx=scene_ctx,
        order_source=order_source,
        config_path=config_path,
        tensorizer=tensorizer,
        critic_builder=critic_builder,
        critic_schema=critic_schema,
        train_cfg=train_cfg,
        device=device,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        checkpoint_meta_path=checkpoint_meta_path,
        model_version=model_version,
        event_hook=event_hook,
    )
    if int(stats.get("updates", 0)) > 0:
        _save_policy_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            critic_schema_hash=critic_schema.schema_hash,
            model_version=str(model_version or ""),
            global_step=0,
            update_idx=0,
            extra_payload={
                "phase": "bc_warm_start",
                "bc_warm_start": dict(stats),
                **_bc_warm_start_checkpoint_stage_payload(
                    train_cfg=train_cfg,
                    stats=stats,
                ),
            },
        )
        _write_json(
            path=checkpoint_meta_path,
            payload={
                "phase": "bc_warm_start",
                "checkpoint_path": str(checkpoint_path),
                "model_version": str(model_version),
                "critic_schema_hash": str(critic_schema.schema_hash),
                "actor_observation_schema_hash": _actor_observation_schema_hash(),
                "actor_observation_schema": _actor_observation_schema_payload(),
                "config_path": str(config_path),
                **_bc_warm_start_checkpoint_stage_payload(
                    train_cfg=train_cfg,
                    stats=stats,
                ),
                "stats": dict(stats),
            },
        )
        stats = {
            **dict(stats),
            "loaded_from_checkpoint": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_meta_path": str(checkpoint_meta_path),
        }
        _emit_event_hook(
            event_hook,
            "bc_warm_start_checkpoint_saved",
            {
                "phase": "bc_warm_start",
                "path": str(checkpoint_path),
                "meta_path": str(checkpoint_meta_path),
                "samples": int(stats.get("samples", 0)),
            },
        )
    return stats


def _resolve_bc_warm_start_load_path(
    *,
    train_cfg: _TrainingConfig,
    default_checkpoint_path: Path,
) -> Path | None:
    explicit_path = train_cfg.bc_warm_start_checkpoint_path
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"BC warm start checkpoint 不存在: {path}")
        return path
    if train_cfg.bc_warm_start_reuse_existing and default_checkpoint_path.is_file():
        return default_checkpoint_path
    return None


def _load_bc_warm_start_checkpoint(
    *,
    path: Path,
    model: Any,
    bc_optimizer: Any,
    device: Any,
    critic_schema_hash: str,
    train_cfg: _TrainingConfig,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"BC warm start checkpoint 缺少 model_state_dict: {path}")
    checkpoint_schema_hash = checkpoint.get("critic_schema_hash")
    if (
        checkpoint_schema_hash is not None
        and str(checkpoint_schema_hash) != str(critic_schema_hash)
    ):
        raise ValueError(
            "BC warm start checkpoint critic schema 不匹配: "
            f"{checkpoint_schema_hash} != {critic_schema_hash}"
        )
    checkpoint_actor_schema_hash = checkpoint.get("actor_observation_schema_hash")
    expected_actor_schema_hash = _actor_observation_schema_hash()
    if (
        checkpoint_actor_schema_hash is not None
        and str(checkpoint_actor_schema_hash) != expected_actor_schema_hash
    ):
        raise ValueError(
            "BC warm start checkpoint actor observation schema 不匹配: "
            f"{checkpoint_actor_schema_hash} != {expected_actor_schema_hash}"
        )
    order_proj_weight = state_dict.get("order_proj.weight")
    if order_proj_weight is not None:
        checkpoint_order_dim = int(order_proj_weight.shape[1])
        model_order_dim = int(model.order_proj.in_features)
        if checkpoint_order_dim != model_order_dim:
            raise ValueError(
                "BC warm start checkpoint order token dim 不匹配: "
                f"{checkpoint_order_dim} != {model_order_dim}"
            )
    raw_checkpoint_stage_config = checkpoint.get("bc_warm_start_stages")
    legacy_checkpoint = raw_checkpoint_stage_config is None
    checkpoint_stage_config = (
        None
        if legacy_checkpoint
        else _normalize_bc_warm_start_stage_config_payload(raw_checkpoint_stage_config)
    )
    current_stage_config = _bc_warm_start_stage_payloads(
        train_cfg.bc_warm_start_stages
    )
    if not legacy_checkpoint and checkpoint_stage_config != current_stage_config:
        raise ValueError(
            "BC warm start checkpoint stages 与当前配置不一致，禁止复用: "
            f"checkpoint={checkpoint_stage_config}, current={current_stage_config}"
        )

    model.load_state_dict(state_dict)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        bc_optimizer.load_state_dict(optimizer_state)
    raw_stats = checkpoint.get("bc_warm_start")
    stats = dict(raw_stats) if isinstance(raw_stats, Mapping) else {}
    stats.update(
        {
            "enabled": True,
            "loaded_from_checkpoint": True,
            "checkpoint_path": str(path),
            "checkpoint_model_version": checkpoint.get("model_version"),
            "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
            "checkpoint_update": int(checkpoint.get("update", 0)),
            "bc_warm_start_legacy_checkpoint": bool(legacy_checkpoint),
            "checkpoint_bc_warm_start_stages": (
                None if legacy_checkpoint else checkpoint_stage_config
            ),
            "bc_warm_start_stages": current_stage_config,
            "bc_warm_start_stage_total_updates": _bc_warm_start_stage_total_updates(
                train_cfg.bc_warm_start_stages
            ),
            "bc_warm_start_stage_stats": _bc_warm_start_stage_stats_payload(
                dict(raw_stats) if isinstance(raw_stats, Mapping) else {}
            ),
        }
    )
    return stats


def _run_bc_warm_start(
    *,
    model: Any,
    optimizer: Any,
    scene_ctx: Any,
    order_source: OrderSourceConfig,
    config_path: Path,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    train_cfg: _TrainingConfig,
    device: Any,
    metrics_path: Path,
    checkpoint_path: Path | None = None,
    checkpoint_meta_path: Path | None = None,
    model_version: str | None = None,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    target_global_updates = _bc_warm_start_stage_total_updates(
        train_cfg.bc_warm_start_stages
    )
    stage_stats: list[dict[str, Any]] = [
        {
            "name": str(stage.name),
            "order_source_mode": str(stage.order_source_mode),
            "target_updates": int(stage.updates),
            "completed_updates": 0,
            "samples": 0,
            "dispatch": 0,
            "wait": 0,
            "forced_wait_no_dispatch": 0,
            "dispatch_label_count": 0,
            "wait_label_count": 0,
            "last_loss": None,
            "skipped": int(stage.updates) == 0,
        }
        for stage in train_cfg.bc_warm_start_stages
    ]
    if not train_cfg.bc_warm_start_enabled:
        return {
            "enabled": False,
            "updates": 0,
            "stage_total_updates": target_global_updates,
            "stages": stage_stats,
            "samples": 0,
            "wait_no_dispatch": 0,
            "wait_with_dispatch": 0,
            "forced_wait_no_dispatch": 0,
            "informative_samples": 0,
            "loss_sample_count": 0,
            "loss_skipped_samples": 0,
            "root_acc": 0.0,
            "bc_root_acc": 0.0,
            "root_dispatch_acc": 0.0,
            "bc_root_dispatch_acc": 0.0,
            "root_wait_acc": 0.0,
            "bc_root_wait_acc": 0.0,
            "order_acc_given_dispatch": 0.0,
            "bc_order_acc_given_dispatch": 0.0,
            "mode_acc_given_dispatch": 0.0,
            "bc_mode_acc_given_dispatch": 0.0,
            "mode_b_acc_given_dispatch": 0.0,
            "bc_mode_b_acc_given_dispatch": 0.0,
            "mode_c_acc_given_dispatch": 0.0,
            "bc_mode_c_acc_given_dispatch": 0.0,
            "dispatch_label_count": 0,
            "wait_label_count": 0,
            "bc_dispatch_label_count": 0,
            "bc_wait_label_count": 0,
            "dispatch_B": 0,
            "dispatch_C": 0,
            "legal_C_count": 0,
            "selected_C_given_legal_C": 0.0,
            "decision_batches_per_update": int(
                train_cfg.bc_warm_start_decision_batches_per_update
            ),
            "reset_each_update": bool(train_cfg.bc_warm_start_reset_each_update),
            "max_decision_batches_per_rollout": int(
                train_cfg.bc_warm_start_max_decision_batches_per_rollout
            ),
            "last_loss": None,
        }

    total_samples = 0
    total_wait = 0
    total_wait_no_dispatch = 0
    total_wait_with_dispatch = 0
    total_dispatch = 0
    total_forced_wait_no_dispatch = 0
    total_informative_samples = 0
    total_loss_sample_count = 0
    total_loss_skipped_samples = 0
    total_root_correct = 0
    total_root_count = 0
    total_root_dispatch_correct = 0
    total_root_dispatch_count = 0
    total_root_wait_correct = 0
    total_root_wait_count = 0
    total_order_correct = 0
    total_order_count = 0
    total_mode_correct = 0
    total_mode_count = 0
    total_mode_b_correct = 0
    total_mode_b_count = 0
    total_mode_c_correct = 0
    total_mode_c_count = 0
    total_dispatch_label_count = 0
    total_wait_label_count = 0
    total_mode_b = 0
    total_mode_c = 0
    total_legal_c = 0
    total_selected_c_given_legal_c = 0
    total_decision_batches = 0
    total_rollout_episodes = 0
    total_completed_rollouts = 0
    last_loss: float | None = None
    global_update = 0
    model.train()
    partial_checkpoint_path = (
        checkpoint_path.with_name(
            f"{checkpoint_path.stem}_partial{checkpoint_path.suffix}"
        )
        if checkpoint_path is not None
        else None
    )
    partial_checkpoint_meta_path = (
        checkpoint_meta_path.with_name(
            f"{checkpoint_meta_path.stem}_partial{checkpoint_meta_path.suffix}"
        )
        if checkpoint_meta_path is not None
        else None
    )

    def _set_optimizer_learning_rate(learning_rate: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(learning_rate)

    def _current_bc_stats(*, completed: bool, current_update: int) -> dict[str, Any]:
        return {
            "enabled": True,
            "completed": bool(completed),
            "updates": int(current_update),
            "target_updates": int(train_cfg.bc_warm_start_updates),
            "stage_total_updates": target_global_updates,
            "stages": [dict(stage_stat) for stage_stat in stage_stats],
            "batches_per_update": int(train_cfg.bc_warm_start_batches_per_update),
            "samples": int(total_samples),
            "wait": int(total_wait),
            "wait_no_dispatch": int(total_wait_no_dispatch),
            "wait_with_dispatch": int(total_wait_with_dispatch),
            "dispatch": int(total_dispatch),
            "forced_wait_no_dispatch": int(total_forced_wait_no_dispatch),
            "informative_samples": int(total_informative_samples),
            "loss_sample_count": int(total_loss_sample_count),
            "loss_skipped_samples": int(total_loss_skipped_samples),
            "root_acc": _safe_divide_metric(total_root_correct, total_root_count),
            "bc_root_acc": _safe_divide_metric(total_root_correct, total_root_count),
            "root_dispatch_acc": _safe_divide_metric(
                total_root_dispatch_correct,
                total_root_dispatch_count,
            ),
            "bc_root_dispatch_acc": _safe_divide_metric(
                total_root_dispatch_correct,
                total_root_dispatch_count,
            ),
            "root_wait_acc": _safe_divide_metric(
                total_root_wait_correct,
                total_root_wait_count,
            ),
            "bc_root_wait_acc": _safe_divide_metric(
                total_root_wait_correct,
                total_root_wait_count,
            ),
            "order_acc_given_dispatch": _safe_divide_metric(
                total_order_correct,
                total_order_count,
            ),
            "bc_order_acc_given_dispatch": _safe_divide_metric(
                total_order_correct,
                total_order_count,
            ),
            "mode_acc_given_dispatch": _safe_divide_metric(
                total_mode_correct,
                total_mode_count,
            ),
            "bc_mode_acc_given_dispatch": _safe_divide_metric(
                total_mode_correct,
                total_mode_count,
            ),
            "mode_b_acc_given_dispatch": _safe_divide_metric(
                total_mode_b_correct,
                total_mode_b_count,
            ),
            "bc_mode_b_acc_given_dispatch": _safe_divide_metric(
                total_mode_b_correct,
                total_mode_b_count,
            ),
            "mode_c_acc_given_dispatch": _safe_divide_metric(
                total_mode_c_correct,
                total_mode_c_count,
            ),
            "bc_mode_c_acc_given_dispatch": _safe_divide_metric(
                total_mode_c_correct,
                total_mode_c_count,
            ),
            "dispatch_label_count": int(total_dispatch_label_count),
            "wait_label_count": int(total_wait_label_count),
            "bc_dispatch_label_count": int(total_dispatch_label_count),
            "bc_wait_label_count": int(total_wait_label_count),
            "dispatch_B": int(total_mode_b),
            "dispatch_C": int(total_mode_c),
            "mode_b": int(total_mode_b),
            "mode_c": int(total_mode_c),
            "legal_C_count": int(total_legal_c),
            "selected_C_given_legal_C_count": int(total_selected_c_given_legal_c),
            "selected_C_given_legal_C": (
                float(total_selected_c_given_legal_c) / float(total_legal_c)
                if total_legal_c
                else 0.0
            ),
            "rollout_episodes": int(total_rollout_episodes),
            "completed_rollouts": int(total_completed_rollouts),
            "decision_batches": int(total_decision_batches),
            "decision_batches_per_update": int(
                train_cfg.bc_warm_start_decision_batches_per_update
            ),
            "reset_each_update": bool(train_cfg.bc_warm_start_reset_each_update),
            "max_decision_batches_per_rollout": int(
                train_cfg.bc_warm_start_max_decision_batches_per_rollout
            ),
            "last_loss": last_loss,
        }

    def _maybe_save_partial_checkpoint(*, current_update: int) -> None:
        interval = int(train_cfg.bc_warm_start_checkpoint_interval_updates)
        if interval <= 0:
            return
        if partial_checkpoint_path is None or partial_checkpoint_meta_path is None:
            return
        if (
            current_update % interval != 0
            and current_update != int(target_global_updates)
        ):
            return
        stats = _current_bc_stats(completed=False, current_update=current_update)
        _save_policy_checkpoint(
            path=partial_checkpoint_path,
            model=model,
            optimizer=optimizer,
            critic_schema_hash=critic_schema.schema_hash,
            model_version=model_version,
            global_step=0,
            update_idx=0,
            extra_payload={
                "phase": "bc_warm_start",
                "bc_warm_start": dict(stats),
                **_bc_warm_start_checkpoint_stage_payload(
                    train_cfg=train_cfg,
                    stats=stats,
                ),
                "bc_warm_start_partial": True,
                "bc_warm_start_partial_update": int(current_update),
            },
        )
        _write_json(
            path=partial_checkpoint_meta_path,
            payload={
                "phase": "bc_warm_start",
                "partial": True,
                "checkpoint_path": str(partial_checkpoint_path),
                "target_final_checkpoint_path": str(checkpoint_path),
                "model_version": str(model_version or ""),
                "critic_schema_hash": str(critic_schema.schema_hash),
                "actor_observation_schema_hash": _actor_observation_schema_hash(),
                "actor_observation_schema": _actor_observation_schema_payload(),
                "config_path": str(config_path),
                **_bc_warm_start_checkpoint_stage_payload(
                    train_cfg=train_cfg,
                    stats=stats,
                ),
                "stats": dict(stats),
            },
        )
        _emit_event_hook(
            event_hook,
            "bc_warm_start_partial_checkpoint_saved",
            {
                "phase": "bc_warm_start",
                "path": str(partial_checkpoint_path),
                "meta_path": str(partial_checkpoint_meta_path),
                "bc_update": int(current_update),
                "samples": int(stats.get("samples", 0)),
            },
        )

    for stage_index, stage in enumerate(train_cfg.bc_warm_start_stages, start=1):
        stage_stat = stage_stats[stage_index - 1]
        stage_updates = int(stage.updates)
        if stage_updates <= 0:
            stage_stat["skipped"] = True
            continue

        stage_mode = OrderSourceMode(str(stage.order_source_mode))
        stage_order_source = build_order_source(
            scene_ctx,
            mode=stage_mode,
            config_path=config_path,
        )
        bc_env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=stage_order_source,
            config_path=config_path,
        )
        stage_learning_rate = float(stage.learning_rate)
        _set_optimizer_learning_rate(stage_learning_rate)
        continuous_episode_active = False
        continuous_rollout_decision_batches = 0

        for stage_update_idx in range(1, stage_updates + 1):
            global_update += 1
            loss_terms: list[Any] = []
            update_wait = 0
            update_wait_no_dispatch = 0
            update_wait_with_dispatch = 0
            update_dispatch = 0
            update_forced_wait_no_dispatch = 0
            update_loss_sample_count = 0
            update_loss_skipped_samples = 0
            update_root_correct = 0
            update_root_count = 0
            update_root_dispatch_correct = 0
            update_root_dispatch_count = 0
            update_root_wait_correct = 0
            update_root_wait_count = 0
            update_order_correct = 0
            update_order_count = 0
            update_mode_correct = 0
            update_mode_count = 0
            update_mode_b_correct = 0
            update_mode_b_count = 0
            update_mode_c_correct = 0
            update_mode_c_count = 0
            update_dispatch_label_count = 0
            update_wait_label_count = 0
            update_mode_b = 0
            update_mode_c = 0
            update_legal_c = 0
            update_selected_c_given_legal_c = 0
            update_decision_batches = 0
            update_completed_rollouts = 0
            update_started_rollouts = 0
            update_samples = 0
            update_loss_weight_sum = 0.0
            reset_each_update = bool(train_cfg.bc_warm_start_reset_each_update)
            decision_batches_per_update = int(
                train_cfg.bc_warm_start_decision_batches_per_update
            )
            if decision_batches_per_update <= 0:
                decision_batches_per_update = int(
                    train_cfg.bc_warm_start_max_decision_batches_per_rollout
                )
            rollout_count = (
                int(train_cfg.bc_warm_start_batches_per_update)
                if reset_each_update
                else 1
            )

            for _rollout_idx in range(rollout_count):
                if reset_each_update:
                    reset_result = bc_env.reset()
                    update_started_rollouts += 1
                    if reset_result.done:
                        update_completed_rollouts += 1
                        continue
                    rollout_decision_batches = 0
                else:
                    if not continuous_episode_active or bc_env.is_done():
                        reset_result = bc_env.reset()
                        update_started_rollouts += 1
                        continuous_rollout_decision_batches = 0
                        continuous_episode_active = not bool(reset_result.done)
                        if reset_result.done:
                            update_completed_rollouts += 1
                            break
                    rollout_decision_batches = continuous_rollout_decision_batches

                while not bc_env.is_done():
                    if reset_each_update:
                        max_batches = int(
                            train_cfg.bc_warm_start_max_decision_batches_per_rollout
                        )
                        if (
                            max_batches > 0
                            and rollout_decision_batches >= max_batches
                        ):
                            break
                    elif (
                        decision_batches_per_update > 0
                        and update_decision_batches >= decision_batches_per_update
                    ):
                        break

                    decision_batch = bc_env.peek_current_decision_batch()
                    if not decision_batch:
                        if bc_env.is_done():
                            break
                        raise RuntimeError(
                            "BC teacher rollout 遇到非终止且无 decision batch 的状态"
                        )

                    candidate_outputs_by_drone = {
                        str(context.deciding_drone_id): bc_env.build_candidate_output(
                            context,
                            last_seen_plan_version=int(
                                context.coarse_plan.plan_version
                            ),
                        )
                        for context in decision_batch
                    }
                    if all(
                        hasattr(candidate_out, "candidate_features")
                        for candidate_out in candidate_outputs_by_drone.values()
                    ):
                        candidate_outputs_by_drone = (
                            attach_local_teacher_competition_hints(
                                decision_contexts=tuple(decision_batch),
                                candidate_outputs_by_drone=candidate_outputs_by_drone,
                                require_shared_snapshot=all(
                                    hasattr(context, "t_decision")
                                    and hasattr(context, "runtime_state")
                                    and hasattr(context, "coarse_plan")
                                    for context in decision_batch
                                ),
                            )
                        )
                    teacher_result = build_batch_matching_teacher_labels(
                        decision_contexts=tuple(decision_batch),
                        candidate_outputs_by_drone=candidate_outputs_by_drone,
                    )

                    for context in decision_batch:
                        drone_id = str(context.deciding_drone_id)
                        candidate_out = candidate_outputs_by_drone[drone_id]
                        label = teacher_result.labels_by_drone[drone_id]
                        assignment = teacher_result.assignments_by_drone[drone_id]
                        has_dispatch = bool(
                            candidate_out.resolved_action_lookup.dispatch_actions
                        )
                        has_legal_c = _candidate_output_has_legal_mode_c(candidate_out)
                        selected_c = (
                            isinstance(assignment.action, DispatchAction)
                            and str(assignment.action.mode) == "C"
                        )
                        if has_legal_c:
                            update_legal_c += 1
                            if selected_c:
                                update_selected_c_given_legal_c += 1
                        update_samples += 1

                        if isinstance(assignment.action, DispatchAction):
                            update_dispatch_label_count += 1
                            update_dispatch += 1
                            if selected_c:
                                update_mode_c += 1
                            else:
                                update_mode_b += 1
                        else:
                            update_wait_label_count += 1
                            update_wait += 1
                            if has_dispatch:
                                update_wait_with_dispatch += 1
                            else:
                                update_wait_no_dispatch += 1
                        is_forced_wait_no_dispatch = (
                            not isinstance(assignment.action, DispatchAction)
                            and not has_dispatch
                        )
                        if is_forced_wait_no_dispatch:
                            update_forced_wait_no_dispatch += 1
                        sample_weight = _compute_bc_label_loss_weight(
                            train_cfg=train_cfg,
                            action=assignment.action,
                            candidate_out=candidate_out,
                        )
                        if sample_weight <= 0.0:
                            update_loss_skipped_samples += 1
                            continue

                        observation_batch = tensorizer.build(
                            decision_context=context,
                            candidate_out=candidate_out,
                            transition_history=(),
                        )
                        action_mask = tensorizer.build_action_mask(candidate_out)
                        critic_batch = critic_builder.build(
                            decision_context=context,
                            critic_tensor_schema_meta=critic_schema,
                        )
                        policy_out, _ = model.forward(
                            observation_batch=observation_batch,
                            action_mask=action_mask,
                            critic_batch=critic_batch,
                            lstm_state=None,
                        )
                        log_prob, _entropy = model.evaluate_actions(
                            policy_out=policy_out,
                            action_mask=action_mask,
                            action_indices=_bc_label_to_action_indices_tensor(
                                label,
                                device=device,
                            ),
                        )
                        label_root_idx = int(label.root_branch_idx)
                        predicted_root_idx = _masked_argmax_index(
                            logits=policy_out.root_branch_logits,
                            mask=action_mask.root_branch_mask,
                            device=device,
                        )
                        if not is_forced_wait_no_dispatch:
                            update_root_count += 1
                            if predicted_root_idx == label_root_idx:
                                update_root_correct += 1
                            if isinstance(assignment.action, DispatchAction):
                                update_root_dispatch_count += 1
                                if predicted_root_idx == label_root_idx:
                                    update_root_dispatch_correct += 1
                            else:
                                update_root_wait_count += 1
                                if predicted_root_idx == label_root_idx:
                                    update_root_wait_correct += 1
                        if isinstance(assignment.action, DispatchAction):
                            label_order_idx = int(label.order_idx or 0)
                            label_mode_idx = int(label.mode_idx or 0)
                            predicted_order_idx = _masked_argmax_index(
                                logits=policy_out.order_logits,
                                mask=action_mask.order_mask,
                                device=device,
                            )
                            update_order_count += 1
                            if predicted_order_idx == label_order_idx:
                                update_order_correct += 1
                            predicted_mode_idx = _masked_argmax_index(
                                logits=_select_logits_row(
                                    logits=policy_out.mode_logits,
                                    index=label_order_idx,
                                    device=device,
                                ),
                                mask=_select_mask_row(
                                    mask=action_mask.mode_mask,
                                    index=label_order_idx,
                                    device=device,
                                ),
                                device=device,
                            )
                            update_mode_count += 1
                            if predicted_mode_idx == label_mode_idx:
                                update_mode_correct += 1
                            if selected_c:
                                update_mode_c_count += 1
                                if predicted_mode_idx == label_mode_idx:
                                    update_mode_c_correct += 1
                            else:
                                update_mode_b_count += 1
                                if predicted_mode_idx == label_mode_idx:
                                    update_mode_b_correct += 1
                        loss_terms.append(
                            -log_prob.reshape(-1)[0] * float(sample_weight)
                        )
                        update_loss_weight_sum += float(sample_weight)
                        update_loss_sample_count += 1

                    step_result = bc_env.apply_decision_batch(
                        teacher_result.actions_by_drone
                    )
                    rollout_decision_batches += 1
                    if not reset_each_update:
                        continuous_rollout_decision_batches = rollout_decision_batches
                    update_decision_batches += 1
                    if step_result.done:
                        update_completed_rollouts += 1
                        if not reset_each_update:
                            continuous_episode_active = False
                            continuous_rollout_decision_batches = 0
                        break

            total_rollout_episodes += update_started_rollouts

            if not loss_terms:
                update_last_loss = None
            else:
                if update_loss_weight_sum <= 0.0:
                    raise RuntimeError("BC warm start 存在 loss_terms 但权重和为 0")
                loss = torch.stack(loss_terms).sum() / float(update_loss_weight_sum)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.max_grad_norm,
                )
                optimizer.step()
                update_last_loss = float(loss.detach().cpu().item())
                last_loss = update_last_loss

            total_samples += update_samples
            total_wait += update_wait
            total_wait_no_dispatch += update_wait_no_dispatch
            total_wait_with_dispatch += update_wait_with_dispatch
            total_dispatch += update_dispatch
            total_forced_wait_no_dispatch += update_forced_wait_no_dispatch
            total_informative_samples += (
                update_samples - update_forced_wait_no_dispatch
            )
            total_loss_sample_count += update_loss_sample_count
            total_loss_skipped_samples += update_loss_skipped_samples
            total_root_correct += update_root_correct
            total_root_count += update_root_count
            total_root_dispatch_correct += update_root_dispatch_correct
            total_root_dispatch_count += update_root_dispatch_count
            total_root_wait_correct += update_root_wait_correct
            total_root_wait_count += update_root_wait_count
            total_order_correct += update_order_correct
            total_order_count += update_order_count
            total_mode_correct += update_mode_correct
            total_mode_count += update_mode_count
            total_mode_b_correct += update_mode_b_correct
            total_mode_b_count += update_mode_b_count
            total_mode_c_correct += update_mode_c_correct
            total_mode_c_count += update_mode_c_count
            total_dispatch_label_count += update_dispatch_label_count
            total_wait_label_count += update_wait_label_count
            total_mode_b += update_mode_b
            total_mode_c += update_mode_c
            total_legal_c += update_legal_c
            total_selected_c_given_legal_c += update_selected_c_given_legal_c
            total_decision_batches += update_decision_batches
            total_completed_rollouts += update_completed_rollouts

            stage_stat["completed_updates"] = int(stage_stat["completed_updates"]) + 1
            stage_stat["samples"] = int(stage_stat["samples"]) + update_samples
            stage_stat["dispatch"] = int(stage_stat["dispatch"]) + update_dispatch
            stage_stat["wait"] = int(stage_stat["wait"]) + update_wait
            stage_stat["forced_wait_no_dispatch"] = (
                int(stage_stat["forced_wait_no_dispatch"])
                + update_forced_wait_no_dispatch
            )
            stage_stat["dispatch_label_count"] = (
                int(stage_stat["dispatch_label_count"])
                + update_dispatch_label_count
            )
            stage_stat["wait_label_count"] = (
                int(stage_stat["wait_label_count"]) + update_wait_label_count
            )
            if update_last_loss is not None:
                stage_stat["last_loss"] = update_last_loss
            stage_stat["skipped"] = False

            metrics_payload = {
                "phase": "bc_warm_start",
                "bc_update": global_update,
                "bc_global_update": global_update,
                "bc_stage_name": str(stage.name),
                "bc_stage_index": stage_index,
                "bc_stage_update": stage_update_idx,
                "bc_stage_order_source_mode": str(stage_mode.value),
                "bc_stage_learning_rate": stage_learning_rate,
                "bc_loss": update_last_loss,
                "bc_samples": update_samples,
                "bc_loss_weight_sum": update_loss_weight_sum,
                "bc_wait": update_wait,
                "bc_wait_no_dispatch": update_wait_no_dispatch,
                "bc_wait_with_dispatch": update_wait_with_dispatch,
                "bc_dispatch": update_dispatch,
                "bc_forced_wait_no_dispatch": update_forced_wait_no_dispatch,
                "bc_informative_samples": (
                    update_samples - update_forced_wait_no_dispatch
                ),
                "bc_loss_sample_count": update_loss_sample_count,
                "bc_loss_skipped_samples": update_loss_skipped_samples,
                "bc_root_acc": _safe_divide_metric(
                    update_root_correct,
                    update_root_count,
                ),
                "bc_root_dispatch_acc": _safe_divide_metric(
                    update_root_dispatch_correct,
                    update_root_dispatch_count,
                ),
                "bc_root_wait_acc": _safe_divide_metric(
                    update_root_wait_correct,
                    update_root_wait_count,
                ),
                "bc_order_acc_given_dispatch": _safe_divide_metric(
                    update_order_correct,
                    update_order_count,
                ),
                "bc_mode_acc_given_dispatch": _safe_divide_metric(
                    update_mode_correct,
                    update_mode_count,
                ),
                "bc_mode_b_acc_given_dispatch": _safe_divide_metric(
                    update_mode_b_correct,
                    update_mode_b_count,
                ),
                "bc_mode_c_acc_given_dispatch": _safe_divide_metric(
                    update_mode_c_correct,
                    update_mode_c_count,
                ),
                "bc_dispatch_label_count": update_dispatch_label_count,
                "bc_wait_label_count": update_wait_label_count,
                "bc_dispatch_B": update_mode_b,
                "bc_dispatch_C": update_mode_c,
                "bc_mode_b": update_mode_b,
                "bc_mode_c": update_mode_c,
                "bc_legal_C_count": update_legal_c,
                "bc_selected_C_given_legal_C_count": update_selected_c_given_legal_c,
                "bc_selected_C_given_legal_C": (
                    float(update_selected_c_given_legal_c) / float(update_legal_c)
                    if update_legal_c
                    else 0.0
                ),
                "bc_rollout_episodes": update_started_rollouts,
                "bc_completed_rollouts": update_completed_rollouts,
                "bc_decision_batches": update_decision_batches,
                "bc_decision_batches_per_update": decision_batches_per_update,
                "bc_reset_each_update": reset_each_update,
            }
            _append_metrics(metrics_path, metrics_payload)
            _emit_event_hook(event_hook, "bc_warm_start_update", metrics_payload)
            _maybe_save_partial_checkpoint(current_update=global_update)

    return _current_bc_stats(completed=True, current_update=global_update)


def _ppo_update(
    *,
    model: Any,
    optimizer: Any,
    batch_view: Any,
    train_cfg: _TrainingConfig,
    device: Any,
    tail_bootstrap_values: Mapping[tuple[int, str, int], float],
) -> dict[str, float]:
    sequences = _materialize_recurrent_sequences(
        batch_view=batch_view,
        sequence_len=train_cfg.sequence_len,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        tail_bootstrap_values=tail_bootstrap_values,
    )
    if not sequences:
        raise RuntimeError("当前 rollout 未物化出任何 recurrent sequence")

    total_valid_timesteps = int(
        sum(int(sequence.valid_timestep_mask.sum()) for sequence in sequences)
    )
    total_padded_timesteps = int(
        sum(train_cfg.sequence_len - len(sequence.transitions) for sequence in sequences)
    )
    latest_approx_kl = 0.0
    latest_policy_loss = 0.0
    latest_value_loss = 0.0
    latest_entropy = 0.0
    sequence_indices = np.arange(len(sequences), dtype=np.int64)

    stop_early = False
    for _epoch in range(train_cfg.ppo_epochs):
        np.random.shuffle(sequence_indices)
        for start in range(0, len(sequence_indices), train_cfg.sequence_minibatch_size):
            picked = sequence_indices[start : start + train_cfg.sequence_minibatch_size]
            minibatch = _build_sequence_minibatch(
                sequences=[sequences[int(idx)] for idx in picked],
                train_cfg=train_cfg,
                model=model,
                device=device,
            )
            valid_mask = minibatch.valid_timestep_mask
            advantages_t = minibatch.advantages
            if train_cfg.normalize_advantage:
                advantages_t = _normalize_advantages_masked(
                    advantages=advantages_t,
                    valid_mask=valid_mask,
                )

            policy_seq_out, _ = model.forward_sequence(
                observation_batch=minibatch.observation_batch,
                action_mask=minibatch.action_mask,
                critic_batch=minibatch.critic_batch,
                lstm_state=minibatch.lstm_state_in,
            )
            flat_policy_out = _flatten_sequence_policy_output(policy_seq_out)
            flat_action_mask = _flatten_sequence_action_mask(minibatch.action_mask)
            flat_action_indices = _flatten_sequence_action_indices(minibatch.action_indices)
            new_log_probs, entropy = model.evaluate_actions(
                policy_out=flat_policy_out,
                action_mask=flat_action_mask,
                action_indices=flat_action_indices,
            )

            old_log_probs = minibatch.old_log_probs.reshape(-1)
            old_values_flat = minibatch.old_values.reshape(-1)
            returns = minibatch.returns.reshape(-1)
            advantages_flat = advantages_t.reshape(-1)
            valid_mask_flat = valid_mask.reshape(-1)
            values_flat = policy_seq_out.value.reshape(-1)
            sample_loss_weights = getattr(minibatch, "sample_loss_weights", None)
            sample_loss_weights_flat = (
                None
                if sample_loss_weights is None
                else sample_loss_weights.reshape(-1)
            )

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages_flat
            clipped = torch.clamp(
                ratio,
                1.0 - train_cfg.clip_coef,
                1.0 + train_cfg.clip_coef,
            ) * advantages_flat
            policy_loss = -_masked_mean_tensor(
                torch.min(unclipped, clipped),
                valid_mask_flat,
                sample_weights=sample_loss_weights_flat,
            )
            # 动作类型样本权重只用于 actor 的 policy 梯度；critic 学 V(s)，
            # entropy 也先保持原始状态分布上的平均，避免放大特定动作类型。
            value_loss = _compute_value_loss_masked(
                values=values_flat,
                returns=returns,
                valid_mask=valid_mask_flat,
                loss_type=train_cfg.value_loss_type,
                huber_delta=train_cfg.value_huber_delta,
                sample_weights=None,
            )
            entropy_loss = _masked_mean_tensor(
                entropy,
                valid_mask_flat,
                sample_weights=None,
            )
            approx_kl_t = _compute_masked_approx_kl(
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                valid_mask=valid_mask_flat,
            )
            latest_approx_kl = float(approx_kl_t.detach().cpu().item())
            latest_policy_loss = float(policy_loss.detach().cpu().item())
            latest_value_loss = float(value_loss.detach().cpu().item())
            latest_entropy = float(entropy_loss.detach().cpu().item())
            if latest_approx_kl > train_cfg.target_kl:
                stop_early = True
                break
            loss = (
                policy_loss
                + train_cfg.value_loss_coef * value_loss
                - train_cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
        if stop_early:
            break

    return {
        "policy_loss": latest_policy_loss,
        "value_loss": latest_value_loss,
        "entropy": latest_entropy,
        "approx_kl": latest_approx_kl,
        "adv_mean": float(
            np.mean(
                np.concatenate(
                    [sequence.advantages[sequence.valid_timestep_mask] for sequence in sequences]
                )
            )
        ),
        "return_mean": float(
            np.mean(
                np.concatenate(
                    [sequence.returns[sequence.valid_timestep_mask] for sequence in sequences]
                )
            )
        ),
        "reward_mean": float(np.mean(batch_view.rewards)),
        "sample_loss_weight_mean": float(
            np.mean(
                np.concatenate(
                    [
                        getattr(
                            sequence,
                            "sample_loss_weights",
                            np.ones_like(sequence.advantages, dtype=np.float32),
                        )[sequence.valid_timestep_mask]
                        for sequence in sequences
                    ]
                )
            )
        ),
        "sequence_count": float(len(sequences)),
        "valid_timestep_count": float(total_valid_timesteps),
        "padded_timestep_count": float(total_padded_timesteps),
    }


def _build_candidate_output(
    *,
    env: TrainingEnvAdapter,
    decision_context: Any,
    last_seen_plan_version_by_drone: Mapping[str, int],
) -> Any:
    drone_id = str(decision_context.deciding_drone_id)
    decision_batch = env.peek_current_decision_batch()
    batch_by_decision_id = {
        int(context.decision_id): context
        for context in decision_batch
    }
    if int(decision_context.decision_id) in batch_by_decision_id:
        candidate_outputs_by_drone = {
            str(context.deciding_drone_id): env.build_candidate_output(
                context,
                last_seen_plan_version=last_seen_plan_version_by_drone.get(
                    str(context.deciding_drone_id),
                    int(context.coarse_plan.plan_version),
                ),
            )
            for context in decision_batch
        }
        hinted_outputs = attach_local_teacher_competition_hints(
            decision_contexts=tuple(decision_batch),
            candidate_outputs_by_drone=candidate_outputs_by_drone,
        )
        return hinted_outputs[drone_id]

    last_seen = last_seen_plan_version_by_drone.get(
        drone_id,
        int(decision_context.coarse_plan.plan_version),
    )
    return env.build_candidate_output(
        decision_context,
        last_seen_plan_version=last_seen,
    )


def _drain_pending_transitions_after_collection(
    *,
    current_result: Any,
    env: TrainingEnvAdapter,
    model: Any,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    last_seen_plan_version_by_drone: dict[str, int],
    history_buffer: deque[Any],
    lstm_state_by_drone: dict[str, Any],
    recurrent_segment_id_by_drone: dict[str, int],
    pending_transition_by_drone: dict[str, RolloutTransition],
    rollout_backlog: list[RolloutTransition],
    successor_bootstrap_values: dict[tuple[int, str, int], float],
    train_cfg: _TrainingConfig,
) -> Any:
    while pending_transition_by_drone:
        if current_result.done:
            terminal_reward_by_drone = env.consume_terminal_agent_costs()
            _flush_terminal_pending_transitions(
                pending_transition_by_drone=pending_transition_by_drone,
                rollout_backlog=rollout_backlog,
                terminal_reward_by_drone=terminal_reward_by_drone,
                runtime_state=current_result.runtime_state,
                rendezvous_arrive_bonus=train_cfg.rendezvous_arrive_bonus,
                rendezvous_bonus=train_cfg.rendezvous_bonus,
                reward_scale=train_cfg.reward_scale,
            )
            return current_result

        decision_context = current_result.decision_context
        if decision_context is None:
            raise RuntimeError("drain pending transitions 时缺少 decision_context")

        drone_id = str(decision_context.deciding_drone_id)
        lstm_state_in = lstm_state_by_drone.get(drone_id)
        candidate_out = _build_candidate_output(
            env=env,
            decision_context=decision_context,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
        )
        observation_batch = tensorizer.build(
            decision_context=decision_context,
            candidate_out=candidate_out,
            transition_history=tuple(history_buffer),
        )
        action_mask = tensorizer.build_action_mask(candidate_out)
        critic_batch = critic_builder.build(
            decision_context=decision_context,
            critic_tensor_schema_meta=critic_schema,
        )
        with torch.no_grad():
            policy_out, next_lstm_state = model.forward(
                observation_batch=observation_batch,
                action_mask=action_mask,
                critic_batch=critic_batch,
                lstm_state=lstm_state_in,
            )
            sampled_action, _ = model.sample_action(
                policy_out=policy_out,
                action_mask=action_mask,
                deterministic=False,
            )
        value_old = float(policy_out.value.detach().cpu().item())
        action_indices = ResolvedActionIndices(**sampled_action)
        env_action = candidate_out.resolved_action_lookup.resolve(
            root_branch_idx=action_indices.root_branch_idx,
            order_idx=action_indices.order_idx,
            mode_idx=action_indices.mode_idx,
        )
        step_result = env.step(env_action)
        carry_in_reward, _post_action_reward, _mode_c_attempt_reward = _extract_attributed_step_reward_parts(
            step_result=step_result
        )
        _finalize_pending_transition_for_next_decision(
            pending_transition_by_drone=pending_transition_by_drone,
            rollout_backlog=rollout_backlog,
            successor_bootstrap_values=successor_bootstrap_values,
            drone_id=drone_id,
            carry_in_reward=carry_in_reward,
            next_value=value_old,
            decision_context=decision_context,
            rendezvous_arrive_bonus=train_cfg.rendezvous_arrive_bonus,
            rendezvous_bonus=train_cfg.rendezvous_bonus,
            reward_scale=train_cfg.reward_scale,
        )
        history_buffer.append(
            tensorizer.build_transition_summary(
                decision_context=decision_context,
                candidate_out=candidate_out,
                action_indices=action_indices,
                step_result=step_result,
            )
        )
        last_seen_plan_version_by_drone[decision_context.deciding_drone_id] = (
            decision_context.coarse_plan.plan_version
        )
        lstm_state_by_drone[drone_id] = _detach_lstm_state(lstm_state=next_lstm_state)
        _reset_recurrent_state_for_failed_drones(
            runtime_state=step_result.runtime_state,
            lstm_state_by_drone=lstm_state_by_drone,
            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
        )
        current_result = step_result

    return current_result


def _advance_until_rollout_bootstraps_resolved_after_collection(
    *,
    current_result: Any,
    episode_id: int,
    env: TrainingEnvAdapter,
    model: Any,
    tensorizer: ObservationTensorizer,
    critic_builder: CriticBatchBuilder,
    critic_schema: CriticTensorSchemaMeta,
    last_seen_plan_version_by_drone: dict[str, int],
    history_buffer: deque[Any],
    lstm_state_by_drone: dict[str, Any],
    recurrent_segment_id_by_drone: dict[str, int],
    rollout_backlog: Sequence[RolloutTransition],
    successor_bootstrap_values: dict[tuple[int, str, int], float],
) -> Any:
    while True:
        resolved = _resolve_rollout_prefix_bootstrap_values(
            transitions=rollout_backlog,
            prefix_len=len(rollout_backlog),
            boundary_bootstrap_values=_build_rollout_prefix_boundary_bootstrap_values(
                current_result=current_result,
                episode_id=episode_id,
                env=env,
                tensorizer=tensorizer,
                critic_builder=critic_builder,
                critic_schema=critic_schema,
                model=model,
                history_buffer=history_buffer,
                last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
                lstm_state_by_drone=lstm_state_by_drone,
                recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
                successor_bootstrap_values=successor_bootstrap_values,
            ),
            current_runtime_state=current_result.runtime_state,
            episode_terminated_at_boundary=bool(current_result.done),
        )
        if resolved is not None:
            return current_result
        if current_result.done:
            raise RuntimeError("episode 已结束，但 rollout prefix bootstrap 仍未解析完成")

        decision_context = current_result.decision_context
        if decision_context is None:
            raise RuntimeError("bootstrap probe 时缺少 decision_context")

        drone_id = str(decision_context.deciding_drone_id)
        lstm_state_in = lstm_state_by_drone.get(drone_id)
        recurrent_segment_id = int(recurrent_segment_id_by_drone.get(drone_id, 0))
        candidate_out = _build_candidate_output(
            env=env,
            decision_context=decision_context,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
        )
        observation_batch = tensorizer.build(
            decision_context=decision_context,
            candidate_out=candidate_out,
            transition_history=tuple(history_buffer),
        )
        action_mask = tensorizer.build_action_mask(candidate_out)
        critic_batch = critic_builder.build(
            decision_context=decision_context,
            critic_tensor_schema_meta=critic_schema,
        )
        with torch.no_grad():
            policy_out, next_lstm_state = model.forward(
                observation_batch=observation_batch,
                action_mask=action_mask,
                critic_batch=critic_batch,
                lstm_state=lstm_state_in,
            )
            sampled_action, _ = model.sample_action(
                policy_out=policy_out,
                action_mask=action_mask,
                deterministic=False,
            )
        value_old = float(policy_out.value.detach().cpu().item())
        _maybe_record_probe_successor_bootstrap_value(
            successor_bootstrap_values=successor_bootstrap_values,
            episode_id=episode_id,
            drone_id=drone_id,
            recurrent_segment_id=recurrent_segment_id,
            value_old=value_old,
        )
        action_indices = ResolvedActionIndices(**sampled_action)
        env_action = candidate_out.resolved_action_lookup.resolve(
            root_branch_idx=action_indices.root_branch_idx,
            order_idx=action_indices.order_idx,
            mode_idx=action_indices.mode_idx,
        )
        step_result = env.step(env_action)
        history_buffer.append(
            tensorizer.build_transition_summary(
                decision_context=decision_context,
                candidate_out=candidate_out,
                action_indices=action_indices,
                step_result=step_result,
            )
        )
        last_seen_plan_version_by_drone[decision_context.deciding_drone_id] = (
            decision_context.coarse_plan.plan_version
        )
        lstm_state_by_drone[drone_id] = _detach_lstm_state(lstm_state=next_lstm_state)
        _reset_recurrent_state_for_failed_drones(
            runtime_state=step_result.runtime_state,
            lstm_state_by_drone=lstm_state_by_drone,
            recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
        )
        current_result = step_result


def _detach_lstm_state(
    *,
    lstm_state: Any | None,
) -> Any | None:
    if lstm_state is None:
        return None
    hidden, cell = lstm_state
    return hidden.detach().cpu().clone(), cell.detach().cpu().clone()


def _materialize_recurrent_sequences(
    *,
    batch_view: Any,
    sequence_len: int,
    gamma: float,
    gae_lambda: float,
    tail_bootstrap_values: Mapping[tuple[int, str, int], float],
) -> list[_MaterializedSequence]:
    # 按 (episode_id, drone_id, segment_id) 分组，收集每个 drone 在本 rollout 中的
    # 全部 transition 索引，并按 local_decision_index 排序。
    grouped_indices: dict[tuple[int, str, int], list[int]] = {}
    for idx, transition in enumerate(batch_view.transitions):
        key = (
            int(transition.episode_id),
            str(transition.actor_drone_id),
            int(transition.recurrent_segment_id),
        )
        grouped_indices.setdefault(key, []).append(idx)

    sequences: list[_MaterializedSequence] = []
    for key, indices in grouped_indices.items():
        indices.sort(
            key=lambda item_idx: int(batch_view.transitions[item_idx].local_decision_index)
        )

        # 提取该 drone 序列的 rewards / dones / values
        drone_rewards = np.asarray(
            [float(batch_view.transitions[i].reward) for i in indices],
            dtype=np.float32,
        )
        drone_dones = np.asarray(
            [bool(batch_view.transitions[i].done) for i in indices],
            dtype=np.bool_,
        )
        drone_values = np.asarray(
            [float(batch_view.transitions[i].value_old) for i in indices],
            dtype=np.float32,
        )

        # bootstrap：若序列末尾 done=True（episode 结束），bootstrap=0；
        # 否则必须显式使用该 drone/segment 在 rollout 边界之后解析出的同 actor value。
        last_done = bool(drone_dones[-1])
        if last_done:
            drone_last_value = 0.0
        else:
            if key not in tail_bootstrap_values:
                raise RuntimeError(
                    "缺少 rollout 尾部 bootstrap value: "
                    f"episode={key[0]}, drone={key[1]}, segment={key[2]}"
                )
            drone_last_value = float(tail_bootstrap_values[key])

        # 对该 drone 的完整序列独立计算 GAE
        drone_advantages, drone_returns = compute_gae(
            rewards=drone_rewards,
            dones=drone_dones,
            values=drone_values,
            last_value=drone_last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # 按 sequence_len 切块
        for local_chunk_idx, start in enumerate(range(0, len(indices), sequence_len)):
            chunk_indices = indices[start : start + sequence_len]
            chunk_slice = slice(start, start + len(chunk_indices))
            transitions = tuple(batch_view.transitions[i] for i in chunk_indices)
            advantages = drone_advantages[chunk_slice].copy()
            returns = drone_returns[chunk_slice].copy()
            sample_loss_weights = np.asarray(
                [
                    float(batch_view.transitions[i].sample_loss_weight)
                    for i in chunk_indices
                ],
                dtype=np.float32,
            )
            valid_mask = np.ones((len(chunk_indices),), dtype=np.bool_)
            sequences.append(
                _MaterializedSequence(
                    sequence_id=(key[0], key[1], key[2], local_chunk_idx),
                    episode_id=key[0],
                    actor_drone_id=key[1],
                    recurrent_segment_id=key[2],
                    local_chunk_idx=local_chunk_idx,
                    transitions=transitions,
                    advantages=advantages,
                    returns=returns,
                    sample_loss_weights=sample_loss_weights,
                    valid_timestep_mask=valid_mask,
                )
            )
    sequences.sort(
        key=lambda item: (
            item.episode_id,
            min(
                transition.local_decision_index for transition in item.transitions
            ),
            item.actor_drone_id,
            item.recurrent_segment_id,
            item.local_chunk_idx,
        )
    )
    return sequences


def _stack_lstm_states_for_update(
    transitions: Sequence[RolloutTransition],
    *,
    model: Any,
    device: Any,
) -> Any | None:
    if not transitions:
        return None

    num_layers = int(model.recurrent_core.num_layers)
    hidden_size = int(model.recurrent_core.hidden_size)
    has_any_state = any(item.lstm_state_in is not None for item in transitions)
    if not has_any_state:
        return None

    hidden_parts = []
    cell_parts = []
    for idx, transition in enumerate(transitions):
        state = transition.lstm_state_in
        if state is None:
            hidden_parts.append(torch.zeros((num_layers, hidden_size), dtype=torch.float32, device=device))
            cell_parts.append(torch.zeros((num_layers, hidden_size), dtype=torch.float32, device=device))
            continue

        hidden, cell = state
        hidden = torch.as_tensor(hidden, dtype=torch.float32, device=device)
        cell = torch.as_tensor(cell, dtype=torch.float32, device=device)
        expected_shape = (num_layers, 1, hidden_size)
        if tuple(hidden.shape) != expected_shape or tuple(cell.shape) != expected_shape:
            raise ValueError(
                "rollout 中保存的 lstm_state_in 形状不合法: "
                f"idx={idx}, hidden={tuple(hidden.shape)}, cell={tuple(cell.shape)}, "
                f"expected={expected_shape}"
            )
        hidden_parts.append(hidden.squeeze(1))
        cell_parts.append(cell.squeeze(1))

    return torch.stack(hidden_parts, dim=1), torch.stack(cell_parts, dim=1)


def _build_sequence_minibatch(
    *,
    sequences: Sequence[_MaterializedSequence],
    train_cfg: _TrainingConfig,
    model: Any,
    device: Any,
) -> _SequenceMiniBatch:
    if not sequences:
        raise ValueError("sequence minibatch 不能为空")

    template_transition = sequences[0].transitions[0]
    pad_observation = _zero_observation_batch_like(template_transition.observation_batch)
    pad_critic = _zero_critic_batch_like(template_transition.critic_batch)
    pad_action_mask = _zero_action_mask_like(template_transition.action_mask)

    batch_size = len(sequences)
    seq_len = train_cfg.sequence_len

    observation_rows = []
    critic_rows = []
    action_mask_rows = []
    root_branch_idx = np.zeros((batch_size, seq_len), dtype=np.int64)
    order_idx = np.zeros((batch_size, seq_len), dtype=np.int64)
    mode_idx = np.zeros((batch_size, seq_len), dtype=np.int64)
    old_log_probs = np.zeros((batch_size, seq_len), dtype=np.float32)
    old_values = np.zeros((batch_size, seq_len), dtype=np.float32)
    returns = np.zeros((batch_size, seq_len), dtype=np.float32)
    advantages = np.zeros((batch_size, seq_len), dtype=np.float32)
    sample_loss_weights = np.zeros((batch_size, seq_len), dtype=np.float32)
    valid_timestep_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
    padded_timesteps = 0

    for seq_row, sequence in enumerate(sequences):
        step_transitions = list(sequence.transitions)
        pad_count = seq_len - len(step_transitions)
        padded_timesteps += pad_count

        obs_steps = [item.observation_batch for item in step_transitions]
        critic_steps = [item.critic_batch for item in step_transitions]
        mask_steps = [item.action_mask for item in step_transitions]
        if pad_count > 0:
            obs_steps.extend([pad_observation] * pad_count)
            critic_steps.extend([pad_critic] * pad_count)
            mask_steps.extend([pad_action_mask] * pad_count)

        observation_rows.append(_stack_observation_batches_from_steps(obs_steps))
        critic_rows.append(_stack_critic_batches_from_steps(critic_steps))
        action_mask_rows.append(_stack_action_masks_from_steps(mask_steps))

        for step_idx, transition in enumerate(step_transitions):
            action = transition.action_indices
            root_branch_idx[seq_row, step_idx] = int(action.root_branch_idx)
            order_idx[seq_row, step_idx] = 0 if action.order_idx is None else int(action.order_idx)
            mode_idx[seq_row, step_idx] = 0 if action.mode_idx is None else int(action.mode_idx)
            old_log_probs[seq_row, step_idx] = float(transition.log_prob_old)
            old_values[seq_row, step_idx] = float(transition.value_old)
            returns[seq_row, step_idx] = float(sequence.returns[step_idx])
            advantages[seq_row, step_idx] = float(sequence.advantages[step_idx])
            sample_loss_weights[seq_row, step_idx] = float(
                sequence.sample_loss_weights[step_idx]
            )
            valid_timestep_mask[seq_row, step_idx] = bool(sequence.valid_timestep_mask[step_idx])

    observation_batch = _stack_sequence_observation_rows(observation_rows)
    critic_batch = _stack_sequence_critic_rows(critic_rows)
    action_mask = _stack_sequence_action_mask_rows(action_mask_rows)
    first_transitions = tuple(sequence.transitions[0] for sequence in sequences)
    lstm_state_in = _stack_lstm_states_for_update(
        first_transitions,
        model=model,
        device=device,
    )

    return _SequenceMiniBatch(
        observation_batch=observation_batch,
        critic_batch=critic_batch,
        action_mask=action_mask,
        action_indices={
            "root_branch_idx": torch.as_tensor(root_branch_idx, dtype=torch.long, device=device),
            "order_idx": torch.as_tensor(order_idx, dtype=torch.long, device=device),
            "mode_idx": torch.as_tensor(mode_idx, dtype=torch.long, device=device),
        },
        old_log_probs=torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
        old_values=torch.as_tensor(old_values, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        sample_loss_weights=torch.as_tensor(sample_loss_weights, dtype=torch.float32, device=device),
        valid_timestep_mask=torch.as_tensor(valid_timestep_mask, dtype=torch.bool, device=device),
        lstm_state_in=lstm_state_in,
        sequence_count=batch_size,
        padded_timesteps=padded_timesteps,
    )


def _stack_observation_batches_from_steps(step_batches: Sequence[Any]) -> Any:
    cls = step_batches[0].__class__
    return cls(
        uav_self_token=np.stack([item.uav_self_token for item in step_batches], axis=0),
        order_tokens=np.stack([item.order_tokens for item in step_batches], axis=0),
        infra_tokens=np.stack([item.infra_tokens for item in step_batches], axis=0),
        history_tokens=np.stack([item.history_tokens for item in step_batches], axis=0),
        history_padding_mask=np.stack([item.history_padding_mask for item in step_batches], axis=0),
        padding_mask=np.stack([item.padding_mask for item in step_batches], axis=0),
    )


def _stack_critic_batches_from_steps(step_batches: Sequence[Any]) -> Any:
    cls = step_batches[0].__class__
    return cls(
        global_order_pool_tokens=np.stack([item.global_order_pool_tokens for item in step_batches], axis=0),
        global_uav_tokens=np.stack([item.global_uav_tokens for item in step_batches], axis=0),
        global_station_tokens=np.stack([item.global_station_tokens for item in step_batches], axis=0),
        coarse_plan_summary_vec=np.stack([item.coarse_plan_summary_vec for item in step_batches], axis=0),
        global_system_summary_vec=np.stack([item.global_system_summary_vec for item in step_batches], axis=0),
        global_order_padding_mask=np.stack([item.global_order_padding_mask for item in step_batches], axis=0),
        global_uav_padding_mask=np.stack([item.global_uav_padding_mask for item in step_batches], axis=0),
        global_station_padding_mask=np.stack([item.global_station_padding_mask for item in step_batches], axis=0),
    )


def _stack_action_masks_from_steps(step_masks: Sequence[Any]) -> Any:
    cls = step_masks[0].__class__
    return cls(
        root_branch_mask=np.stack([item.root_branch_mask for item in step_masks], axis=0),
        order_mask=np.stack([item.order_mask for item in step_masks], axis=0),
        mode_mask=np.stack([item.mode_mask for item in step_masks], axis=0),
    )


def _stack_sequence_observation_rows(rows: Sequence[Any]) -> Any:
    cls = rows[0].__class__
    return cls(
        uav_self_token=np.stack([item.uav_self_token for item in rows], axis=0),
        order_tokens=np.stack([item.order_tokens for item in rows], axis=0),
        infra_tokens=np.stack([item.infra_tokens for item in rows], axis=0),
        history_tokens=np.stack([item.history_tokens for item in rows], axis=0),
        history_padding_mask=np.stack([item.history_padding_mask for item in rows], axis=0),
        padding_mask=np.stack([item.padding_mask for item in rows], axis=0),
    )


def _stack_sequence_critic_rows(rows: Sequence[Any]) -> Any:
    cls = rows[0].__class__
    return cls(
        global_order_pool_tokens=np.stack([item.global_order_pool_tokens for item in rows], axis=0),
        global_uav_tokens=np.stack([item.global_uav_tokens for item in rows], axis=0),
        global_station_tokens=np.stack([item.global_station_tokens for item in rows], axis=0),
        coarse_plan_summary_vec=np.stack([item.coarse_plan_summary_vec for item in rows], axis=0),
        global_system_summary_vec=np.stack([item.global_system_summary_vec for item in rows], axis=0),
        global_order_padding_mask=np.stack([item.global_order_padding_mask for item in rows], axis=0),
        global_uav_padding_mask=np.stack([item.global_uav_padding_mask for item in rows], axis=0),
        global_station_padding_mask=np.stack([item.global_station_padding_mask for item in rows], axis=0),
    )


def _stack_sequence_action_mask_rows(rows: Sequence[Any]) -> Any:
    cls = rows[0].__class__
    return cls(
        root_branch_mask=np.stack([item.root_branch_mask for item in rows], axis=0),
        order_mask=np.stack([item.order_mask for item in rows], axis=0),
        mode_mask=np.stack([item.mode_mask for item in rows], axis=0),
    )


def _zero_observation_batch_like(template: Any) -> Any:
    cls = template.__class__
    return cls(
        uav_self_token=np.zeros_like(template.uav_self_token),
        order_tokens=np.zeros_like(template.order_tokens),
        infra_tokens=np.zeros_like(template.infra_tokens),
        history_tokens=np.zeros_like(template.history_tokens),
        history_padding_mask=np.ones_like(template.history_padding_mask, dtype=np.bool_),
        padding_mask=np.ones_like(template.padding_mask, dtype=np.bool_),
    )


def _zero_critic_batch_like(template: Any) -> Any:
    cls = template.__class__
    return cls(
        global_order_pool_tokens=np.zeros_like(template.global_order_pool_tokens),
        global_uav_tokens=np.zeros_like(template.global_uav_tokens),
        global_station_tokens=np.zeros_like(template.global_station_tokens),
        coarse_plan_summary_vec=np.zeros_like(template.coarse_plan_summary_vec),
        global_system_summary_vec=np.zeros_like(template.global_system_summary_vec),
        global_order_padding_mask=np.ones_like(template.global_order_padding_mask, dtype=np.bool_),
        global_uav_padding_mask=np.ones_like(template.global_uav_padding_mask, dtype=np.bool_),
        global_station_padding_mask=np.ones_like(template.global_station_padding_mask, dtype=np.bool_),
    )


def _zero_action_mask_like(template: Any) -> Any:
    cls = template.__class__
    root_branch_mask = np.zeros_like(template.root_branch_mask, dtype=np.bool_)
    if root_branch_mask.size != 2:
        raise ValueError(f"root_branch_mask 形状异常，无法构造 WAIT-only padding: {root_branch_mask.shape}")
    root_branch_mask[0] = True
    return cls(
        root_branch_mask=root_branch_mask,
        order_mask=np.zeros_like(template.order_mask, dtype=np.bool_),
        mode_mask=np.zeros_like(template.mode_mask, dtype=np.bool_),
    )


def _flatten_sequence_policy_output(policy_out: Any) -> Any:
    cls = policy_out.__class__
    return cls(
        root_branch_logits=policy_out.root_branch_logits.reshape(-1, policy_out.root_branch_logits.shape[-1]),
        order_logits=policy_out.order_logits.reshape(-1, policy_out.order_logits.shape[-1]),
        mode_logits=policy_out.mode_logits.reshape(
            -1,
            policy_out.mode_logits.shape[-2],
            policy_out.mode_logits.shape[-1],
        ),
        value=policy_out.value.reshape(-1),
    )


def _flatten_sequence_action_mask(action_mask: Any) -> Any:
    cls = action_mask.__class__
    return cls(
        root_branch_mask=action_mask.root_branch_mask.reshape(-1, action_mask.root_branch_mask.shape[-1]),
        order_mask=action_mask.order_mask.reshape(-1, action_mask.order_mask.shape[-1]),
        mode_mask=action_mask.mode_mask.reshape(
            -1,
            action_mask.mode_mask.shape[-2],
            action_mask.mode_mask.shape[-1],
        ),
    )


def _flatten_sequence_action_indices(action_indices: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_branch_idx": action_indices["root_branch_idx"].reshape(-1),
        "order_idx": action_indices["order_idx"].reshape(-1),
        "mode_idx": action_indices["mode_idx"].reshape(-1),
    }


def _masked_mean_tensor(
    values: Any,
    valid_mask: Any,
    *,
    sample_weights: Any | None = None,
) -> Any:
    return _masked_weighted_mean_tensor(
        values,
        valid_mask,
        sample_weights=sample_weights,
    )


def _masked_weighted_mean_tensor(
    values: Any,
    valid_mask: Any,
    *,
    sample_weights: Any | None = None,
) -> Any:
    weights = valid_mask.to(dtype=values.dtype)
    if sample_weights is not None:
        weights = weights * sample_weights.to(dtype=values.dtype)
    denom = weights.sum().clamp(min=1.0)
    return (values * weights).sum() / denom


def _compute_masked_approx_kl(
    *,
    old_log_probs: Any,
    new_log_probs: Any,
    valid_mask: Any,
) -> Any:
    return _masked_mean_tensor(old_log_probs - new_log_probs, valid_mask)


def _normalize_advantages_masked(*, advantages: Any, valid_mask: Any) -> Any:
    weights = valid_mask.to(dtype=advantages.dtype)
    valid_count = weights.sum().clamp(min=1.0)
    mean = (advantages * weights).sum() / valid_count
    centered = (advantages - mean) * weights
    variance = (centered * centered).sum() / valid_count
    normalized = centered / torch.sqrt(variance.clamp(min=1e-8))
    return normalized * weights


def _reset_recurrent_state_for_failed_drones(
    *,
    runtime_state: Any,
    lstm_state_by_drone: dict[str, Any],
    recurrent_segment_id_by_drone: dict[str, int],
) -> None:
    for drone_id, drone_state in runtime_state.drone_states.items():
        if str(drone_state.training_state) != "airborne_energy_failure":
            continue
        if drone_id not in lstm_state_by_drone:
            continue
        lstm_state_by_drone.pop(drone_id, None)
        recurrent_segment_id_by_drone[drone_id] = int(
            recurrent_segment_id_by_drone.get(drone_id, 0)
        ) + 1


def _build_meta_payload(
    *,
    scene_ctx: Any,
    order_source: Any,
    critic_schema: CriticTensorSchemaMeta,
    config_path: Path,
    model_version: str,
    effective_total_timesteps: int,
    train_cfg: _TrainingConfig,
) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    action_space = raw["action_space"]
    candidate = raw["candidate"]
    planner = raw["planner"]
    policy = raw["policy"]
    reward = raw["reward"]
    reservation = raw["reservation"]

    params = load_drone_params()
    meta = MetaJson(
        schema_version="phase7_v1",
        coarse_plan_view_contract_version="v1",
        env_semantic_contract_version="v1",
        policy=PolicyMeta(
            encoder_type=str(policy["encoder_type"]),
            d_model=int(policy["d_model"]),
            nhead=int(policy["nhead"]) if policy.get("nhead") is not None else None,
            ff_dim=int(policy["ff_dim"]),
            dropout=float(policy["dropout"]),
            lstm_hidden=int(policy["lstm_hidden"]),
            lstm_layers=int(policy["lstm_layers"]),
            hist_len=int(policy["hist_len"]),
            max_order_tokens=int(policy["max_order_tokens"]),
            use_plan_version_delta=bool(policy["use_plan_version_delta"]),
            use_is_riding_truck_flag=bool(policy["use_is_riding_truck_flag"]),
            use_drone_source_type_flag=bool(policy["use_drone_source_type_flag"]),
            critic_mode=str(policy["critic_mode"]),
            inference_mode=str(policy["inference_mode"]),
            uav_self_token_fields=tuple(UAV_SELF_TOKEN_FIELDS),
            order_token_fields=tuple(ORDER_TOKEN_FIELDS),
            infra_token_fields=tuple(INFRA_TOKEN_FIELDS),
            history_token_fields=tuple(HISTORY_TOKEN_FIELDS),
        ),
        action_space=ActionSpaceMeta(
            type=str(action_space["type"]),
            factorized_head_order=tuple(action_space["factorized_head_order"]),
            policy_modes=tuple(action_space["policy_modes"]),
            planner_modes=tuple(action_space["planner_modes"]),
            enable_wait_action=bool(action_space["enable_wait_action"]),
            include_mode_a_in_policy=bool(action_space["include_mode_a_in_policy"]),
        ),
        candidate=CandidateMeta(
            max_candidate_orders=int(candidate["max_candidate_orders"]),
            max_candidate_recovery_per_order=int(candidate["max_candidate_recovery_per_order"]),
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
            energy_safe_margin_ratio=float(candidate["energy_safe_margin_ratio"]),
        ),
        planner=PlannerMeta(
            coarse_replan_interval_sec=float(planner["coarse_replan_interval_sec"]),
            coarse_new_order_trigger=int(planner["coarse_new_order_trigger"]),
            route_drift_trigger_ratio=float(planner["route_drift_trigger_ratio"]),
            fallback_burst_trigger_count=int(planner["fallback_burst_trigger_count"]),
            fallback_burst_window_sec=float(planner["fallback_burst_window_sec"]),
            hard_failure_trigger_count=int(planner["hard_failure_trigger_count"]),
            upper_horizon_sec=float(planner["upper_horizon_sec"]),
            support_radius_km=float(planner["support_radius_km"]),
            min_orders_to_trigger=int(planner["min_orders_to_trigger"]),
        ),
        reward=RewardMeta(
            wait_idle_penalty_coef=float(reward["wait_idle_penalty_coef"]),
            lambda_miss=float(reward["lambda_miss"]),
            lambda_res_timeout=float(reward["lambda_res_timeout"]),
            lambda_overdue=float(reward.get("lambda_overdue", 0.0)),
            R_delivery_bonus=float(reward["R_delivery_bonus"]),
            late_delivery_penalty_coef=float(
                reward.get("late_delivery_penalty_coef", 0.0)
            ),
            late_delivery_penalty_cap=float(
                reward.get("late_delivery_penalty_cap", 0.0)
            ),
            min_late_delivery_reward=float(
                reward.get("min_late_delivery_reward", 1.0)
            ),
            rendezvous_arrive_bonus=float(reward.get("rendezvous_arrive_bonus", 0.0)),
            rendezvous_bonus=float(reward.get("rendezvous_bonus", 0.0)),
            mode_c_attempt_bonus=float(reward.get("mode_c_attempt_bonus", 0.0)),
            uav_energy_penalty_coef=float(
                reward.get("uav_energy_penalty_coef", 0.0)
            ),
            uav_energy_penalty_cap_ratio=float(
                reward.get("uav_energy_penalty_cap_ratio", 0.0)
            ),
            max_overdue_sec=float(reward["max_overdue_sec"]),
            hard_overdue_penalty_sec=float(reward.get("hard_overdue_penalty_sec", 0.0)),
            hard_failure_penalty_sec=float(reward["hard_failure_penalty_sec"]),
            primary_metrics_scope=str(reward["primary_metrics_scope"]),
            include_mode_a_in_primary_metrics=bool(reward["include_mode_a_in_primary_metrics"]),
        ),
        critic_schema=critic_schema,
        shared_runtime_params_snapshot=SharedRuntimeParamsSnapshot(
            source_config="backend/config/drone_params.yaml",
            light_drone=DroneRuntimeParamsSnapshot(
                k1=float(params.light.k1),
                k2=float(params.light.k2),
                cruise_speed=float(params.light.cruise_speed),
                payload_capacity=float(params.light.payload_capacity),
                empty_weight=float(params.light.empty_weight),
                battery_capacity_j=float(params.light.battery_capacity_j),
                safe_margin_ratio=float(params.light.safe_margin_ratio),
            ),
            heavy_drone=DroneRuntimeParamsSnapshot(
                k1=float(params.heavy.k1),
                k2=float(params.heavy.k2),
                cruise_speed=float(params.heavy.cruise_speed),
                payload_capacity=float(params.heavy.payload_capacity),
                empty_weight=float(params.heavy.empty_weight),
                battery_capacity_j=float(params.heavy.battery_capacity_j),
                safe_margin_ratio=float(params.heavy.safe_margin_ratio),
            ),
            solver_energy=SolverEnergyRuntimeSnapshot(
                c_dist_et=float(params.solver_energy.c_dist_et),
                c_dist_uav=float(params.solver_energy.c_dist_uav),
                c_energy_et=float(params.solver_energy.c_energy_et),
                c_energy_uav=float(params.solver_energy.c_energy_uav),
                lambda_time=float(params.solver_energy.lambda_time),
                truck_energy_kwh_per_km=float(params.solver_energy.truck_energy_kwh_per_km),
                uav_energy_model=str(params.solver_energy.uav_energy_model),
                uav_alpha_wh_per_kg_km=float(params.solver_energy.uav_alpha_wh_per_kg_km),
                allow_moving_truck_launch=bool(params.solver_energy.allow_moving_truck_launch),
                truck_service_time_order_s=float(params.solver_energy.truck_service_time_order_s),
                drone_service_time_order_s=float(params.solver_energy.drone_service_time_order_s),
                truck_drone_launch_time_s=float(params.solver_energy.truck_drone_launch_time_s),
                truck_drone_recover_time_s=float(params.solver_energy.truck_drone_recover_time_s),
            ),
        ),
        env_semantic_contract=EnvSemanticContractMeta(
            mode_c_recovery_nodes=("station", "depot"),
            reservation_timeout_enabled=bool(reservation["enable"]),
            reservation_alpha=float(reservation["alpha"]),
            reservation_beta=float(reservation["beta"]),
            reservation_gamma=float(reservation["gamma"]),
            overdue_penalty_mode="delivery_time_lateness_discount",
            fifo_queue_enabled=True,
            riding_with_truck_enabled=True,
            allow_empty_backbone_route=bool(planner["allow_empty_backbone_route"]),
            hard_failure_type="airborne_energy_failure_or_no_safe_host",
        ),
        online_lock_params=OnlineLockParams(
            locked_fields=(
                "policy.d_model",
                "policy.nhead",
                "policy.lstm_hidden",
                "policy.max_global_orders",
                "policy.max_global_uavs",
                "policy.max_global_stations",
            ),
            tunable_fields=(
                "reward.lambda_miss",
                "reward.late_delivery_penalty_coef",
                "reward.late_delivery_penalty_cap",
                "reward.min_late_delivery_reward",
                "reward.uav_energy_penalty_coef",
                "reward.uav_energy_penalty_cap_ratio",
            ),
        ),
    )
    training_input_meta = replace(
        order_source.training_input_meta,
        total_timesteps=int(effective_total_timesteps),
    )
    run = TrainingRunMeta(
        model_version=model_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        scene_id=str(scene_ctx.scene_id),
        scene_bundle_dir=str(scene_ctx.scene_bundle_dir),
        training_input=training_input_meta,
    )
    payload = build_meta_json_dict(meta, run)
    payload["actor_observation_schema_hash"] = _actor_observation_schema_hash()
    payload["actor_observation_schema"] = _actor_observation_schema_payload()
    payload["bc_warm_start_stages"] = _bc_warm_start_stage_payloads(
        train_cfg.bc_warm_start_stages
    )
    payload["bc_warm_start_stage_total_updates"] = _bc_warm_start_stage_total_updates(
        train_cfg.bc_warm_start_stages
    )
    return payload


def _append_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _append_planner_replan_events(
    path: Path,
    *,
    episode_metrics: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> None:
    prefix = {
        "phase": str(episode_metrics.get("phase", "")),
        "episode_id": int(episode_metrics.get("episode_id", -1)),
        "order_source_mode": str(episode_metrics.get("order_source_mode", "")),
        "order_source_seed": int(episode_metrics.get("order_source_seed", -1)),
        "global_step_start": episode_metrics.get("global_step_start"),
        "global_step_end": episode_metrics.get("global_step_end"),
        "update_start": episode_metrics.get("update_start"),
        "update_end": episode_metrics.get("update_end"),
    }
    for event in events:
        _append_metrics(path, {**prefix, **dict(event)})


def _emit_event_hook(
    event_hook: Callable[[str, Mapping[str, Any]], None] | None,
    event_name: str,
    payload: Mapping[str, Any],
) -> None:
    if event_hook is None:
        return
    event_hook(str(event_name), payload)


def _write_json(*, path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_policy_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: Any,
    critic_schema_hash: str,
    model_version: str,
    global_step: int,
    update_idx: int,
    extra_payload: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "critic_schema_hash": critic_schema_hash,
        "actor_observation_schema_hash": _actor_observation_schema_hash(),
        "actor_observation_schema": _actor_observation_schema_payload(),
        "model_version": model_version,
        "global_step": int(global_step),
        "update": int(update_idx),
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _actor_observation_schema_payload() -> dict[str, tuple[str, ...]]:
    return {
        "uav_self_token_fields": tuple(UAV_SELF_TOKEN_FIELDS),
        "order_token_fields": tuple(ORDER_TOKEN_FIELDS),
        "infra_token_fields": tuple(INFRA_TOKEN_FIELDS),
        "history_token_fields": tuple(HISTORY_TOKEN_FIELDS),
    }


def _actor_observation_schema_hash() -> str:
    payload = _actor_observation_schema_payload()
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_module = getattr(torch, "mps", None)
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_module is not None
        and hasattr(mps_module, "manual_seed")
        and mps_backend is not None
        and mps_backend.is_available()
    ):
        mps_module.manual_seed(seed)


def _validate_meta_payload_before_write(
    *,
    meta_payload: Mapping[str, Any],
    scene_ctx: Any,
    order_source: Any,
    critic_schema: CriticTensorSchemaMeta,
    train_cfg: _TrainingConfig,
    model_version: str,
    global_step: int,
    policy_path: Path,
    config_snapshot_path: Path,
    metrics_path: Path,
) -> None:
    if global_step <= 0:
        raise RuntimeError(f"训练未产生任何有效 step，禁止写出 meta.json: global_step={global_step}")
    if global_step > train_cfg.total_timesteps:
        raise RuntimeError(
            "训练步数超过配置 total_timesteps，禁止写出 meta.json: "
            f"global_step={global_step}, total_timesteps={train_cfg.total_timesteps}"
        )
    if not policy_path.is_file():
        raise FileNotFoundError(f"缺少 policy.pt，禁止写出 meta.json: {policy_path}")
    if not config_snapshot_path.is_file():
        raise FileNotFoundError(
            f"缺少 training_config_snapshot.yaml，禁止写出 meta.json: {config_snapshot_path}"
        )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"缺少 train_metrics.jsonl，禁止写出 meta.json: {metrics_path}")

    required_root_keys = (
        "model_version",
        "trained_at",
        "scene_id",
        "scene_bundle_dir",
        "training_input",
        "policy",
        "action_space",
        "candidate",
        "planner",
        "reward",
        "critic_schema",
        "shared_runtime_params_snapshot",
        "env_semantic_contract",
        "online_lock_params",
    )
    for key in required_root_keys:
        if key not in meta_payload:
            raise ValueError(f"meta.json 缺少一级字段: {key}")

    payload_model_version = str(meta_payload["model_version"])
    if payload_model_version != model_version:
        raise ValueError(
            "meta.json model_version 与本次 run 不一致: "
            f"payload={payload_model_version}, expected={model_version}"
        )
    _validate_iso8601_timestamp(str(meta_payload["trained_at"]))

    payload_scene_id = str(meta_payload["scene_id"])
    expected_scene_id = str(scene_ctx.scene_id)
    if payload_scene_id != expected_scene_id:
        raise ValueError(
            "meta.json scene_id 与 scene_ctx 不一致: "
            f"payload={payload_scene_id}, expected={expected_scene_id}"
        )

    payload_scene_bundle_dir = str(meta_payload["scene_bundle_dir"])
    expected_scene_bundle_dir = str(scene_ctx.scene_bundle_dir)
    if payload_scene_bundle_dir != expected_scene_bundle_dir:
        raise ValueError(
            "meta.json scene_bundle_dir 与 scene_ctx 不一致: "
            f"payload={payload_scene_bundle_dir}, expected={expected_scene_bundle_dir}"
        )

    training_input = _require_mapping(meta_payload, "training_input")
    payload_mode = str(training_input["order_source_mode"])
    expected_mode = str(order_source.mode.value)
    if payload_mode != expected_mode:
        raise ValueError(
            "meta.json training_input.order_source_mode 不一致: "
            f"payload={payload_mode}, expected={expected_mode}"
        )
    _require_exact_int(
        name="training_input.training_seed",
        actual=training_input["training_seed"],
        expected=train_cfg.training_seed,
    )
    _require_exact_int(
        name="training_input.total_timesteps",
        actual=training_input["total_timesteps"],
        expected=global_step,
    )
    _require_exact_int(
        name="training_input.poisson_seed",
        actual=training_input["poisson_seed"],
        expected=order_source.seed,
    )
    payload_bc_stages = meta_payload.get("bc_warm_start_stages")
    expected_bc_stages = _bc_warm_start_stage_payloads(train_cfg.bc_warm_start_stages)
    if payload_bc_stages != expected_bc_stages:
        raise ValueError("meta.json bc_warm_start_stages 与训练配置不一致")
    _require_exact_int(
        name="bc_warm_start_stage_total_updates",
        actual=meta_payload.get("bc_warm_start_stage_total_updates"),
        expected=_bc_warm_start_stage_total_updates(train_cfg.bc_warm_start_stages),
    )

    expected_training_input = order_source.training_input_meta
    _require_exact_float(
        name="training_input.poisson_arrival_rate",
        actual=training_input["poisson_arrival_rate"],
        expected=expected_training_input.poisson_arrival_rate,
    )
    _require_exact_float(
        name="training_input.poisson_weight_max_kg",
        actual=training_input["poisson_weight_max_kg"],
        expected=expected_training_input.poisson_weight_max_kg,
    )
    _require_exact_int(
        name="training_input.order_window_min_min",
        actual=training_input["order_window_min_min"],
        expected=expected_training_input.order_window_min_min,
    )
    _require_exact_int(
        name="training_input.order_window_max_min",
        actual=training_input["order_window_max_min"],
        expected=expected_training_input.order_window_max_min,
    )

    benchmark = _require_mapping(training_input, "benchmark")
    expected_benchmark = order_source.benchmark
    if str(benchmark["orders_json"]) != expected_benchmark.orders_json:
        raise ValueError("meta.json training_input.benchmark.orders_json 不一致")
    if str(benchmark["orders_json_sha256"]) != expected_benchmark.orders_json_sha256:
        raise ValueError("meta.json training_input.benchmark.orders_json_sha256 不一致")
    _require_exact_int(
        name="training_input.benchmark.static_order_count",
        actual=benchmark["static_order_count"],
        expected=expected_benchmark.static_order_count,
    )
    _require_exact_int(
        name="training_input.benchmark.dynamic_order_count",
        actual=benchmark["dynamic_order_count"],
        expected=expected_benchmark.dynamic_order_count,
    )
    if bool(benchmark["benchmark_use_dynamic_orders"]) != bool(
        expected_benchmark.benchmark_use_dynamic_orders
    ):
        raise ValueError("meta.json training_input.benchmark.benchmark_use_dynamic_orders 不一致")

    critic_schema_meta = _require_mapping(meta_payload, "critic_schema")
    if str(critic_schema_meta["schema_hash"]) != str(critic_schema.schema_hash):
        raise ValueError(
            "meta.json critic_schema.schema_hash 与训练时 schema 不一致: "
            f"payload={critic_schema_meta['schema_hash']}, expected={critic_schema.schema_hash}"
        )
    if not str(critic_schema_meta["name"]):
        raise ValueError("meta.json critic_schema.name 不能为空")
    if not str(critic_schema_meta["schema_version"]):
        raise ValueError("meta.json critic_schema.schema_version 不能为空")

    shared_runtime = _require_mapping(meta_payload, "shared_runtime_params_snapshot")
    if not str(shared_runtime["source_config"]):
        raise ValueError("meta.json shared_runtime_params_snapshot.source_config 不能为空")
    _require_mapping(shared_runtime, "light_drone")
    _require_mapping(shared_runtime, "heavy_drone")
    _require_mapping(shared_runtime, "solver_energy")


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必须为 mapping")
    return value


def _require_exact_int(*, name: str, actual: Any, expected: int) -> None:
    actual_int = int(actual)
    expected_int = int(expected)
    if actual_int != expected_int:
        raise ValueError(f"{name} 不一致: payload={actual_int}, expected={expected_int}")


def _require_exact_float(*, name: str, actual: Any, expected: float) -> None:
    actual_float = float(actual)
    expected_float = float(expected)
    if actual_float != expected_float:
        raise ValueError(f"{name} 不一致: payload={actual_float}, expected={expected_float}")


def _validate_iso8601_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"trained_at 不是合法 ISO 8601 时间戳: {value}") from exc


def _emit_runtime_device_debug(
    *,
    requested_device: str,
    resolved_device: Any,
) -> None:
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    print(
        "[train_cmrappo] device debug | "
        f"requested={requested_device} | "
        f"resolved={resolved_device} | "
        f"torch={torch.__version__} | "
        f"cuda_available={cuda_available} | "
        f"mps_built={mps_built} | "
        f"mps_available={mps_available}"
    )


def _resolve_device(device_name: str) -> Any:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求使用 {device_name}，但当前环境 CUDA 不可用")
    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("请求使用 mps，但当前环境 MPS 不可用")
    return torch.device(device_name)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "缺少 torch，无法运行 Phase 7 train_cmrappo.py。"
            "当前代码已实现训练入口，但本地环境尚未安装训练依赖。"
        )


def _parse_bc_warm_start_stages(
    *,
    training: Mapping[str, Any],
) -> tuple[_BCWarmStartStageConfig, ...]:
    bc_learning_rate = float(
        training.get("bc_learning_rate", training["ppo_learning_rate"])
    )
    raw_stages = training.get("bc_warm_start_stages")
    if raw_stages is None:
        return (
            _BCWarmStartStageConfig(
                name="legacy",
                order_source_mode=OrderSourceMode.POISSON.value,
                updates=int(training.get("bc_warm_start_updates", 0)),
                learning_rate=bc_learning_rate,
            ),
        )
    if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, Sequence):
        raise ValueError("training.bc_warm_start_stages 必须为 stage 列表")

    stages: list[_BCWarmStartStageConfig] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise ValueError("training.bc_warm_start_stages 中每个 stage 必须为 mapping")
        stages.append(
            _BCWarmStartStageConfig(
                name=str(raw_stage.get("name", "")).strip(),
                order_source_mode=str(raw_stage.get("order_source_mode", "")).strip(),
                updates=int(raw_stage.get("updates", 0)),
                learning_rate=float(raw_stage.get("learning_rate", bc_learning_rate)),
            )
        )
    return tuple(stages)


def _load_training_config(config_path: Path) -> _TrainingConfig:
    raw = _load_yaml(config_path)
    training = raw["training"]
    reward = raw["reward"]
    batch_size_alias = training.get("batch_size")
    target_minibatch_timesteps = training.get("target_minibatch_timesteps")
    if target_minibatch_timesteps is None and batch_size_alias is None:
        raise ValueError("training.target_minibatch_timesteps 与 training.batch_size 不能同时缺失")
    if target_minibatch_timesteps is None:
        target_minibatch_timesteps = batch_size_alias
    if (
        batch_size_alias is not None
        and int(batch_size_alias) != int(target_minibatch_timesteps)
    ):
        raise ValueError(
            "recurrent PPO 配置错误：batch_size 与 target_minibatch_timesteps 不一致"
        )

    bc_warm_start_stages = _parse_bc_warm_start_stages(training=training)
    bc_warm_start_updates = (
        _bc_warm_start_stage_total_updates(bc_warm_start_stages)
        if "bc_warm_start_stages" in training
        else int(training.get("bc_warm_start_updates", 0))
    )
    cfg = _TrainingConfig(
        device=str(training["device"]),
        total_timesteps=int(training["total_timesteps"]),
        rollout_steps=int(training["rollout_steps"]),
        recurrent_ppo=bool(training["recurrent_ppo"]),
        sequence_len=int(training["sequence_len"]),
        burn_in_len=int(training["burn_in_len"]),
        train_len=int(training["train_len"]),
        sequence_minibatch_size=int(training["sequence_minibatch_size"]),
        target_minibatch_timesteps=int(target_minibatch_timesteps),
        ppo_learning_rate=float(training["ppo_learning_rate"]),
        bc_learning_rate=float(
            training.get("bc_learning_rate", training["ppo_learning_rate"])
        ),
        batch_size_alias=None if batch_size_alias is None else int(batch_size_alias),
        ppo_epochs=int(training["ppo_epochs"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_coef=float(training["clip_coef"]),
        entropy_coef=float(training["entropy_coef"]),
        reward_scale=float(training.get("reward_scale", 1.0)),
        rendezvous_arrive_bonus=float(
            training.get(
                "rendezvous_arrive_bonus_scaled",
                float(reward.get("rendezvous_arrive_bonus", 0.0))
                * float(training.get("reward_scale", 1.0)),
            )
        ),
        rendezvous_bonus=float(
            training.get(
                "rendezvous_bonus_scaled",
                float(reward.get("rendezvous_bonus", 0.0))
                * float(training.get("reward_scale", 1.0)),
            )
        ),
        mode_c_attempt_bonus=float(
            training.get(
                "mode_c_attempt_bonus_scaled",
                float(reward.get("mode_c_attempt_bonus", 0.0))
                * float(training.get("reward_scale", 1.0)),
            )
        ),
        value_loss_coef=float(training["value_loss_coef"]),
        value_loss_type=str(training.get("value_loss_type", "mse")),
        value_huber_delta=float(training.get("value_huber_delta", 1.0)),
        max_grad_norm=float(training["max_grad_norm"]),
        normalize_advantage=bool(training["normalize_advantage"]),
        target_kl=float(training["target_kl"]),
        wait_without_dispatch_loss_weight=float(
            training.get("wait_without_dispatch_loss_weight", 1.0)
        ),
        wait_with_dispatch_loss_weight=float(
            training.get("wait_with_dispatch_loss_weight", 1.0)
        ),
        mode_b_dispatch_loss_weight=float(
            training.get("mode_b_dispatch_loss_weight", 1.0)
        ),
        mode_c_dispatch_loss_weight=float(
            training.get("mode_c_dispatch_loss_weight", 1.0)
        ),
        training_seed=int(training["training_seed"]),
        log_interval_updates=int(training["log_interval_updates"]),
        save_interval_updates=int(training["save_interval_updates"]),
        eval_interval_updates=int(training["eval_interval_updates"]),
        benchmark_eval_episodes=int(training["benchmark_eval_episodes"]),
        stochastic_eval_seeds=int(training["stochastic_eval_seeds"]),
        early_stop_enabled=bool(training.get("early_stop_enabled", False)),
        early_stop_min_evals=int(training.get("early_stop_min_evals", 1)),
        early_stop_stochastic_high_patience=int(
            training.get("early_stop_stochastic_high_patience", 5)
        ),
        early_stop_value_loss_window=int(
            training.get("early_stop_value_loss_window", 5)
        ),
        early_stop_value_loss_min_delta=float(
            training.get("early_stop_value_loss_min_delta", 0.0)
        ),
        allowed_train_order_source_mode=tuple(training["allowed_train_order_source_mode"]),
        bc_warm_start_stages=bc_warm_start_stages,
        bc_warm_start_enabled=bool(training.get("bc_warm_start_enabled", False)),
        bc_warm_start_updates=bc_warm_start_updates,
        bc_warm_start_batches_per_update=int(
            training.get("bc_warm_start_batches_per_update", 1)
        ),
        bc_warm_start_max_decision_batches_per_rollout=int(
            training.get("bc_warm_start_max_decision_batches_per_rollout", 0)
        ),
        bc_warm_start_decision_batches_per_update=int(
            training.get("bc_warm_start_decision_batches_per_update", 0)
        ),
        bc_warm_start_reset_each_update=bool(
            training.get("bc_warm_start_reset_each_update", True)
        ),
        bc_warm_start_checkpoint_path=(
            str(training.get("bc_warm_start_checkpoint_path", "")).strip() or None
        ),
        bc_warm_start_reuse_existing=bool(
            training.get("bc_warm_start_reuse_existing", True)
        ),
        bc_warm_start_checkpoint_interval_updates=int(
            training.get("bc_warm_start_checkpoint_interval_updates", 10)
        ),
        bc_wait_no_dispatch_loss_weight=float(
            training.get("bc_wait_no_dispatch_loss_weight", 0.0)
        ),
        bc_wait_with_dispatch_loss_weight=float(
            training.get("bc_wait_with_dispatch_loss_weight", 0.5)
        ),
        bc_mode_b_dispatch_loss_weight=float(
            training.get("bc_mode_b_dispatch_loss_weight", 1.0)
        ),
        bc_mode_c_dispatch_loss_weight=float(
            training.get("bc_mode_c_dispatch_loss_weight", 2.0)
        ),
    )
    _validate_training_config(cfg)
    return cfg


def _validate_training_config(cfg: _TrainingConfig) -> None:
    if not cfg.recurrent_ppo:
        raise RuntimeError(
            "当前 train_cmrappo.py 仅接受 recurrent_ppo=true 的 Phase 7 V1 正式口径"
        )
    if cfg.burn_in_len != 0:
        raise ValueError("Phase 7 V1 固定要求 burn_in_len=0")
    if cfg.sequence_len <= 0:
        raise ValueError("sequence_len 必须为正数")
    if cfg.train_len != cfg.sequence_len - cfg.burn_in_len:
        raise ValueError("train_len 必须满足 sequence_len - burn_in_len")
    if cfg.sequence_minibatch_size <= 0:
        raise ValueError("sequence_minibatch_size 必须为正数")
    if cfg.target_minibatch_timesteps <= 0:
        raise ValueError("target_minibatch_timesteps 必须为正数")
    if cfg.bc_learning_rate <= 0.0 or not math.isfinite(cfg.bc_learning_rate):
        raise ValueError("bc_learning_rate 必须为有限正数")
    if cfg.eval_interval_updates < 0:
        raise ValueError("eval_interval_updates 不能为负数")
    if cfg.early_stop_enabled and cfg.eval_interval_updates <= 0:
        raise ValueError("early_stop_enabled=true 时 eval_interval_updates 必须为正数")
    if cfg.benchmark_eval_episodes <= 0:
        raise ValueError("benchmark_eval_episodes 必须为正数")
    if cfg.stochastic_eval_seeds <= 0:
        raise ValueError("stochastic_eval_seeds 必须为正数")
    if cfg.early_stop_min_evals <= 0:
        raise ValueError("early_stop_min_evals 必须为正数")
    if cfg.early_stop_stochastic_high_patience <= 0:
        raise ValueError("early_stop_stochastic_high_patience 必须为正数")
    if cfg.early_stop_value_loss_window <= 0:
        raise ValueError("early_stop_value_loss_window 必须为正数")
    if cfg.early_stop_value_loss_min_delta < 0.0:
        raise ValueError("early_stop_value_loss_min_delta 不能为负数")
    if cfg.bc_warm_start_updates < 0:
        raise ValueError("bc_warm_start_updates 不能为负数")
    if cfg.bc_warm_start_batches_per_update <= 0:
        raise ValueError("bc_warm_start_batches_per_update 必须为正数")
    if cfg.bc_warm_start_max_decision_batches_per_rollout < 0:
        raise ValueError("bc_warm_start_max_decision_batches_per_rollout 不能为负数")
    if cfg.bc_warm_start_decision_batches_per_update < 0:
        raise ValueError("bc_warm_start_decision_batches_per_update 不能为负数")
    if (
        cfg.bc_warm_start_enabled
        and not cfg.bc_warm_start_reset_each_update
        and cfg.bc_warm_start_decision_batches_per_update <= 0
        and cfg.bc_warm_start_max_decision_batches_per_rollout <= 0
    ):
        raise ValueError(
            "bc_warm_start_reset_each_update=false 时必须设置正数 "
            "bc_warm_start_decision_batches_per_update "
            "或 bc_warm_start_max_decision_batches_per_rollout"
        )
    if cfg.bc_warm_start_checkpoint_interval_updates < 0:
        raise ValueError("bc_warm_start_checkpoint_interval_updates 不能为负数")
    for stage in cfg.bc_warm_start_stages:
        if not str(stage.name).strip():
            raise ValueError("bc_warm_start_stages[].name 不能为空")
        if int(stage.updates) < 0:
            raise ValueError("bc_warm_start_stages[].updates 不能为负数")
        if float(stage.learning_rate) <= 0.0 or not math.isfinite(
            float(stage.learning_rate)
        ):
            raise ValueError("bc_warm_start_stages[].learning_rate 必须为有限正数")
        try:
            OrderSourceMode(str(stage.order_source_mode))
        except ValueError as exc:
            raise ValueError(
                "bc_warm_start_stages[].order_source_mode 不受支持: "
                f"{stage.order_source_mode}"
            ) from exc
    if cfg.bc_wait_no_dispatch_loss_weight < 0.0:
        raise ValueError("bc_wait_no_dispatch_loss_weight 不能为负数")
    if cfg.bc_wait_with_dispatch_loss_weight < 0.0:
        raise ValueError("bc_wait_with_dispatch_loss_weight 不能为负数")
    if cfg.bc_mode_b_dispatch_loss_weight < 0.0:
        raise ValueError("bc_mode_b_dispatch_loss_weight 不能为负数")
    if cfg.bc_mode_c_dispatch_loss_weight < 0.0:
        raise ValueError("bc_mode_c_dispatch_loss_weight 不能为负数")
    if (
        cfg.bc_warm_start_enabled
        and _bc_warm_start_stage_total_updates(cfg.bc_warm_start_stages) <= 0
    ):
        raise ValueError(
            "bc_warm_start_enabled=true 时 bc_warm_start_stages updates 总和必须为正数"
        )
    if cfg.wait_without_dispatch_loss_weight < 0.0:
        raise ValueError("wait_without_dispatch_loss_weight 不能为负数")
    if cfg.wait_with_dispatch_loss_weight < 0.0:
        raise ValueError("wait_with_dispatch_loss_weight 不能为负数")
    if cfg.mode_b_dispatch_loss_weight <= 0.0:
        raise ValueError("mode_b_dispatch_loss_weight 必须为正数")
    if cfg.mode_c_dispatch_loss_weight <= 0.0:
        raise ValueError("mode_c_dispatch_loss_weight 必须为正数")
    if cfg.reward_scale <= 0.0:
        raise ValueError("reward_scale 必须为正数")
    normalized_value_loss_type = str(cfg.value_loss_type).strip().lower()
    if normalized_value_loss_type not in {"mse", "huber"}:
        raise ValueError("value_loss_type 仅支持 mse 或 huber")
    if cfg.value_huber_delta <= 0.0:
        raise ValueError("value_huber_delta 必须为正数")
    if cfg.rendezvous_arrive_bonus < 0.0:
        raise ValueError("rendezvous_arrive_bonus 不能为负数")
    if cfg.rendezvous_bonus < 0.0:
        raise ValueError("rendezvous_bonus 不能为负数")
    if cfg.mode_c_attempt_bonus < 0.0:
        raise ValueError("mode_c_attempt_bonus 不能为负数")
    expected_target = cfg.sequence_minibatch_size * cfg.train_len
    if expected_target != cfg.target_minibatch_timesteps:
        raise ValueError(
            "Phase 7 V1 推荐满载关系被破坏："
            f"sequence_minibatch_size * train_len = {expected_target}, "
            f"target_minibatch_timesteps = {cfg.target_minibatch_timesteps}"
        )


def _load_eval_config(config_path: Path) -> _EvalConfig:
    raw = _load_yaml(config_path)
    data = raw["data"]
    training = raw["training"]
    stochastic_eval = data["stochastic_eval_arrival_rate_per_min"]
    cfg = _EvalConfig(
        benchmark_seed=int(data["benchmark_eval_seed"]),
        stochastic_seed_base=int(data["stochastic_eval_seed_base"]),
        stochastic_arrival_rates=(
            ("low", float(stochastic_eval["low"])),
            ("medium", float(stochastic_eval["medium"])),
            ("high", float(stochastic_eval["high"])),
        ),
        train_arrival_curriculum_enabled=bool(
            training.get("train_arrival_curriculum_enabled", False)
        ),
        train_arrival_base_rate=float(
            training.get("train_arrival_base_rate_per_min", data["poisson_arrival_rate"])
        ),
        train_arrival_high_start_update=int(
            training.get("train_arrival_high_start_update", 0)
        ),
        train_arrival_high_full_update=int(
            training.get("train_arrival_high_full_update", 0)
        ),
        train_arrival_high_max_probability=float(
            training.get("train_arrival_high_max_probability", 0.0)
        ),
        c_sensitive_enabled=bool(data.get("c_sensitive_eval_enabled", True)),
        c_sensitive_seed=int(
            data.get("c_sensitive_eval_seed", data["benchmark_eval_seed"])
        ),
        c_sensitive_min_legal_mode_c_recovery_nodes=int(
            data.get("c_sensitive_eval_min_legal_mode_c_recovery_nodes", 1)
        ),
    )
    if cfg.c_sensitive_min_legal_mode_c_recovery_nodes <= 0:
        raise ValueError("c_sensitive_eval_min_legal_mode_c_recovery_nodes 必须为正数")
    if cfg.train_arrival_base_rate <= 0.0:
        raise ValueError("train_arrival_base_rate_per_min 必须为正数")
    if cfg.train_arrival_high_start_update < 0:
        raise ValueError("train_arrival_high_start_update 不能为负数")
    if cfg.train_arrival_high_full_update < 0:
        raise ValueError("train_arrival_high_full_update 不能为负数")
    if not 0.0 <= cfg.train_arrival_high_max_probability <= 1.0:
        raise ValueError("train_arrival_high_max_probability 必须位于 [0, 1]")
    return cfg


def _load_policy_config(config_path: Path) -> _PolicyConfig:
    raw = _load_yaml(config_path)
    policy = raw["policy"]
    return _PolicyConfig(
        d_model=int(policy["d_model"]),
        ff_dim=int(policy["ff_dim"]),
        lstm_hidden=int(policy["lstm_hidden"]),
        lstm_layers=int(policy["lstm_layers"]),
        hist_len=int(policy["hist_len"]),
    )


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取训练配置") from exc

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML 顶层必须为 mapping: {config_path}")
    return raw


__all__ = ["train_cmrappo"]
