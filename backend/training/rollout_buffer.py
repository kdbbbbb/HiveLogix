#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 7 rollout buffer.

设计目标：
  - 存 materialized tensors，而不是存原始对象后续再现算；
  - 支持“先固化 pre-action actor/critic 输入，再在 step 后补 reward/done”；
  - 以 numpy 维护 GAE/returns，后续训练代码可再转 torch。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .contracts import (
    CriticBatch,
    FactorizedActionMask,
    ObservationBatch,
    ResolvedActionIndices,
)


_FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class RolloutTransition:
    observation_batch: ObservationBatch
    critic_batch: CriticBatch
    action_mask: FactorizedActionMask
    action_indices: ResolvedActionIndices
    log_prob_old: float
    value_old: float
    critic_schema_hash: str
    reward: float | None = None
    done: bool | None = None
    lstm_state_in: Any | None = None
    lstm_state_out: Any | None = None
    episode_id: int = 0
    actor_drone_id: str = ""
    recurrent_segment_id: int = 0
    local_decision_index: int = 0
    global_decision_index: int = 0
    decision_context_debug_snapshot: Any | None = None
    sample_loss_weight: float = 1.0
    rendezvous_arrive_bonus_applied: bool = False
    rendezvous_success_bonus_applied: bool = False
    fallback_cause: str = "none"

    @property
    def is_finalized(self) -> bool:
        return self.reward is not None and self.done is not None


@dataclass(frozen=True)
class RolloutBatchView:
    transitions: tuple[RolloutTransition, ...]
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray


class RolloutBuffer:
    """Phase 7 单环境 PPO rollout buffer。"""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("rollout buffer capacity 必须为正数")
        self._capacity = int(capacity)
        self._transitions: list[RolloutTransition] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def begin_transition(
        self,
        *,
        observation_batch: ObservationBatch,
        critic_batch: CriticBatch,
        action_mask: FactorizedActionMask,
        action_indices: ResolvedActionIndices,
        log_prob_old: float,
        value_old: float,
        critic_schema_hash: str,
        lstm_state_in: Any | None = None,
        episode_id: int = 0,
        actor_drone_id: str = "",
        recurrent_segment_id: int = 0,
        local_decision_index: int = 0,
        global_decision_index: int = 0,
        decision_context_debug_snapshot: Any | None = None,
        sample_loss_weight: float = 1.0,
    ) -> int:
        if self.size >= self._capacity:
            raise RuntimeError("rollout buffer 已满，不能再 begin_transition")
        self._transitions.append(
            RolloutTransition(
                observation_batch=observation_batch,
                critic_batch=critic_batch,
                action_mask=action_mask,
                action_indices=action_indices,
                log_prob_old=float(log_prob_old),
                value_old=float(value_old),
                critic_schema_hash=str(critic_schema_hash),
                lstm_state_in=lstm_state_in,
                episode_id=int(episode_id),
                actor_drone_id=str(actor_drone_id),
                recurrent_segment_id=int(recurrent_segment_id),
                local_decision_index=int(local_decision_index),
                global_decision_index=int(global_decision_index),
                decision_context_debug_snapshot=decision_context_debug_snapshot,
                sample_loss_weight=float(sample_loss_weight),
            )
        )
        return self.size - 1

    def finalize_transition(
        self,
        slot: int,
        *,
        reward: float,
        done: bool,
        lstm_state_out: Any | None = None,
    ) -> None:
        transition = self._require_slot(slot)
        if transition.is_finalized:
            raise RuntimeError(f"transition slot {slot} 已 finalize")
        self._transitions[slot] = replace(
            transition,
            reward=float(reward),
            done=bool(done),
            lstm_state_out=lstm_state_out,
        )

    def append_transition(
        self,
        *,
        observation_batch: ObservationBatch,
        critic_batch: CriticBatch,
        action_mask: FactorizedActionMask,
        action_indices: ResolvedActionIndices,
        log_prob_old: float,
        value_old: float,
        critic_schema_hash: str,
        reward: float,
        done: bool,
        lstm_state_in: Any | None = None,
        lstm_state_out: Any | None = None,
        episode_id: int = 0,
        actor_drone_id: str = "",
        recurrent_segment_id: int = 0,
        local_decision_index: int = 0,
        global_decision_index: int = 0,
        decision_context_debug_snapshot: Any | None = None,
        sample_loss_weight: float = 1.0,
    ) -> int:
        slot = self.begin_transition(
            observation_batch=observation_batch,
            critic_batch=critic_batch,
            action_mask=action_mask,
            action_indices=action_indices,
            log_prob_old=log_prob_old,
            value_old=value_old,
            critic_schema_hash=critic_schema_hash,
            lstm_state_in=lstm_state_in,
            episode_id=episode_id,
            actor_drone_id=actor_drone_id,
            recurrent_segment_id=recurrent_segment_id,
            local_decision_index=local_decision_index,
            global_decision_index=global_decision_index,
            decision_context_debug_snapshot=decision_context_debug_snapshot,
            sample_loss_weight=sample_loss_weight,
        )
        self.finalize_transition(
            slot,
            reward=reward,
            done=done,
            lstm_state_out=lstm_state_out,
        )
        return slot

    def build_batch_view(self) -> RolloutBatchView:
        return build_batch_view_from_transitions(self._transitions)

    def assert_all_finalized(self) -> None:
        for idx, transition in enumerate(self._transitions):
            if not transition.is_finalized:
                raise RuntimeError(f"rollout transition {idx} 尚未 finalize")

    def _require_slot(self, slot: int) -> RolloutTransition:
        if slot < 0 or slot >= self.size:
            raise IndexError(f"rollout slot 越界: {slot}")
        return self._transitions[slot]


def compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(rewards) == len(dones) == len(values)):
        raise ValueError("rewards/dones/values 长度必须一致")
    advantages = np.zeros_like(rewards, dtype=_FLOAT_DTYPE)
    last_advantage = 0.0
    next_value = float(last_value)
    for idx in reversed(range(len(rewards))):
        not_done = 0.0 if bool(dones[idx]) else 1.0
        delta = float(rewards[idx]) + gamma * next_value * not_done - float(values[idx])
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[idx] = last_advantage
        next_value = float(values[idx])
    returns = advantages + values.astype(_FLOAT_DTYPE, copy=False)
    return advantages, returns


def build_batch_view_from_transitions(
    transitions: Sequence[RolloutTransition],
) -> RolloutBatchView:
    materialized = tuple(transitions)
    for idx, transition in enumerate(materialized):
        if not transition.is_finalized:
            raise RuntimeError(f"rollout transition {idx} 尚未 finalize")
    rewards = np.asarray([float(item.reward) for item in materialized], dtype=_FLOAT_DTYPE)
    dones = np.asarray([bool(item.done) for item in materialized], dtype=np.bool_)
    values = np.asarray([float(item.value_old) for item in materialized], dtype=_FLOAT_DTYPE)
    return RolloutBatchView(
        transitions=materialized,
        rewards=rewards,
        dones=dones,
        values=values,
    )


__all__ = [
    "build_batch_view_from_transitions",
    "RolloutBatchView",
    "RolloutBuffer",
    "RolloutTransition",
    "compute_gae",
]
