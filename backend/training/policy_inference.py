#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 已训练 CMRAPPO 策略的加载与单 episode 推理运行时。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import CriticTensorSchemaMeta, ResolvedActionIndices
from .critic_batch_builder import CriticBatchBuilder
from .env_adapter import EnvStepResult, TrainingEnvAdapter
from .model import SharedPPOActorCritic
from .observation_tensorizer import ObservationTensorizer
from .order_source_adapter import OrderSourceMode, build_order_source
from .scene_loader import DEFAULT_CONFIG_PATH, TrainingSceneContext, load_default_scene
from .train_cmrappo import (
    _EpisodeAccumulator,
    _actor_observation_schema_hash,
    _build_candidate_output,
    _detach_lstm_state,
    _finalize_episode_metrics,
    _load_policy_config,
    _record_episode_step,
    _record_terminal_episode_rewards,
    _require_torch,
    _reset_recurrent_state_for_failed_drones,
    _resolve_device,
)


try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class LoadedPolicyRuntime:
    config_path: Path
    policy_path: Path
    scene_ctx: TrainingSceneContext
    tensorizer: ObservationTensorizer
    critic_builder: CriticBatchBuilder
    critic_schema: CriticTensorSchemaMeta
    model: Any
    policy_cfg: Any
    device: Any
    checkpoint_meta: dict[str, Any]


def load_trained_policy(
    *,
    policy_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "auto",
) -> LoadedPolicyRuntime:
    _require_torch()
    cfg_path = Path(config_path)
    scene_ctx = load_default_scene(config_path=cfg_path)
    policy_cfg = _load_policy_config(cfg_path)
    tensorizer = ObservationTensorizer(scene_ctx=scene_ctx, config_path=cfg_path)
    critic_builder = CriticBatchBuilder(scene_ctx=scene_ctx, config_path=cfg_path)
    critic_schema = critic_builder.default_schema_meta

    bootstrap_source = build_order_source(
        scene_ctx,
        mode=OrderSourceMode.POISSON,
        config_path=cfg_path,
    )
    bootstrap_env = TrainingEnvAdapter(
        scene_ctx=scene_ctx,
        order_source=bootstrap_source,
        config_path=cfg_path,
    )
    bootstrap_result = bootstrap_env.reset()
    if bootstrap_result.decision_context is None:
        raise RuntimeError("bootstrap env 没有可用的 decision_context")
    bootstrap_candidate = _build_candidate_output(
        env=bootstrap_env,
        decision_context=bootstrap_result.decision_context,
        last_seen_plan_version_by_drone={},
    )
    bootstrap_observation = tensorizer.build(
        decision_context=bootstrap_result.decision_context,
        candidate_out=bootstrap_candidate,
        transition_history=(),
    )
    bootstrap_critic = critic_builder.build(
        decision_context=bootstrap_result.decision_context,
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

    device_obj = _resolve_device(device)
    model.to(device_obj)

    resolved_policy_path = Path(policy_path).resolve()
    checkpoint = torch.load(resolved_policy_path, map_location=device_obj)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"policy checkpoint 缺少 model_state_dict: {resolved_policy_path}")
    checkpoint_actor_schema_hash = checkpoint.get("actor_observation_schema_hash")
    expected_actor_schema_hash = _actor_observation_schema_hash()
    if (
        checkpoint_actor_schema_hash is not None
        and str(checkpoint_actor_schema_hash) != expected_actor_schema_hash
    ):
        raise ValueError(
            "policy checkpoint actor observation schema 不匹配: "
            f"{checkpoint_actor_schema_hash} != {expected_actor_schema_hash}"
        )
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedPolicyRuntime(
        config_path=cfg_path,
        policy_path=resolved_policy_path,
        scene_ctx=scene_ctx,
        tensorizer=tensorizer,
        critic_builder=critic_builder,
        critic_schema=critic_schema,
        model=model,
        policy_cfg=policy_cfg,
        device=device_obj,
        checkpoint_meta={
            "model_version": checkpoint.get("model_version"),
            "global_step": checkpoint.get("global_step"),
            "update": checkpoint.get("update"),
            "critic_schema_hash": checkpoint.get("critic_schema_hash"),
            "actor_observation_schema_hash": checkpoint.get(
                "actor_observation_schema_hash"
            ),
        },
    )


def run_policy_episode(
    *,
    runtime: LoadedPolicyRuntime,
    order_source_mode: str | OrderSourceMode = OrderSourceMode.BENCHMARK,
    seed: int | None = None,
    arrival_rate_per_min: float | None = None,
    deterministic: bool = True,
    eval_phase: str = "live_validation",
    episode_id: int = 0,
    on_reset: Callable[[TrainingEnvAdapter, EnvStepResult], None] | None = None,
    on_step: Callable[[TrainingEnvAdapter, Any, Any, EnvStepResult], None] | None = None,
) -> dict[str, Any]:
    selected_mode = (
        order_source_mode
        if isinstance(order_source_mode, OrderSourceMode)
        else OrderSourceMode(str(order_source_mode).strip().lower())
    )
    overrides: dict[str, Any] = {}
    if arrival_rate_per_min is not None:
        overrides["poisson_arrival_rate"] = float(arrival_rate_per_min)

    order_source = build_order_source(
        runtime.scene_ctx,
        mode=selected_mode,
        seed=seed,
        overrides=overrides,
        config_path=runtime.config_path,
    )
    env = TrainingEnvAdapter(
        scene_ctx=runtime.scene_ctx,
        order_source=order_source,
        config_path=runtime.config_path,
    )
    result = env.reset()
    if result.decision_context is None:
        raise RuntimeError("validation reset 后没有可用 decision_context")
    if on_reset is not None:
        on_reset(env, result)

    last_seen_plan_version_by_drone: dict[str, int] = {}
    history_buffer: deque[Any] = deque(maxlen=runtime.policy_cfg.hist_len)
    lstm_state_by_drone: dict[str, Any] = {}
    recurrent_segment_id_by_drone: dict[str, int] = {}
    episode_acc = _EpisodeAccumulator(
        phase=eval_phase,
        episode_id=episode_id,
        order_source_mode=str(order_source.mode.value),
        order_source_seed=int(order_source.seed),
    )

    while not result.done:
        decision_context = result.decision_context
        if decision_context is None:
            raise RuntimeError("validation 过程中 decision_context 丢失")

        drone_id = str(decision_context.deciding_drone_id)
        candidate_out = _build_candidate_output(
            env=env,
            decision_context=decision_context,
            last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
        )
        observation_batch = runtime.tensorizer.build(
            decision_context=decision_context,
            candidate_out=candidate_out,
            transition_history=tuple(history_buffer),
        )
        action_mask = runtime.tensorizer.build_action_mask(candidate_out)
        critic_batch = runtime.critic_builder.build(
            decision_context=decision_context,
            critic_tensor_schema_meta=runtime.critic_schema,
        )
        with torch.no_grad():
            policy_out, next_lstm_state = runtime.model.forward(
                observation_batch=observation_batch,
                action_mask=action_mask,
                critic_batch=critic_batch,
                lstm_state=lstm_state_by_drone.get(drone_id),
            )
            sampled_action, _ = runtime.model.sample_action(
                policy_out=policy_out,
                action_mask=action_mask,
                deterministic=deterministic,
            )
        action_indices = ResolvedActionIndices(**sampled_action)
        env_action = candidate_out.resolved_action_lookup.resolve(
            root_branch_idx=action_indices.root_branch_idx,
            order_idx=action_indices.order_idx,
            mode_idx=action_indices.mode_idx,
        )
        step_result = env.step(env_action)
        if on_step is not None:
            on_step(env, decision_context, env_action, step_result)
        history_buffer.append(
            runtime.tensorizer.build_transition_summary(
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
        _record_episode_step(episode_acc, reward=float(step_result.reward))
        result = step_result

    terminal_reward_by_drone = env.consume_terminal_agent_costs()
    _record_terminal_episode_rewards(
        accumulator=episode_acc,
        terminal_reward_by_drone=terminal_reward_by_drone,
    )
    return {
        "episode_metrics": _finalize_episode_metrics(
            accumulator=episode_acc,
            episode_snapshot=env.build_episode_metrics_snapshot(),
            global_step_end=None,
            update_idx_end=None,
        ),
        "order_source_summary": order_source.build_summary(),
        "env": env,
        "result": result,
    }
