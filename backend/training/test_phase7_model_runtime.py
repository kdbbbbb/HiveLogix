#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 model runtime tests.

运行方式：
  python -m unittest backend.training.test_phase7_model_runtime
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from .actions import WAIT_ACTION
from .contracts import CriticBatch, FactorizedActionMask, ObservationBatch, ResolvedActionIndices
from .model import PolicyForwardOutput, SharedPPOActorCritic
from .rollout_buffer import RolloutTransition, compute_gae
from .train_cmrappo import (
    _EpisodeAccumulator,
    _advance_until_rollout_bootstraps_resolved_after_collection,
    _build_training_episode_order_source,
    _build_benchmark_guardrail_key,
    _build_eval_selection_key,
    _build_sequence_minibatch,
    _compute_value_loss_masked,
    _compute_masked_approx_kl,
    _extract_attributed_step_reward_parts,
    _finalize_episode_metrics,
    _finalize_pending_transition_for_next_decision,
    _flush_terminal_pending_transitions,
    _insert_finalized_transition_in_order,
    _latest_value_loss_shows_meaningful_decline,
    _load_bc_warm_start_checkpoint,
    _materialize_recurrent_sequences,
    _ppo_update,
    _record_episode_step,
    _record_terminal_episode_rewards,
    _run_bc_warm_start,
    _run_periodic_evaluation,
    _sample_training_arrival_band,
    _shape_pending_transition_reward_for_rendezvous,
    _shape_post_action_reward_for_rendezvous,
    _should_stop_early,
    _build_stochastic_high_improvement_key,
    _BCWarmStartStageConfig,
    _TrainingConfig,
    _resolve_device,
    _resolve_rollout_prefix_bootstrap_values,
    _save_policy_checkpoint,
    _set_training_seed,
    _strip_episode_details,
    _summarize_episode_records,
    _stack_lstm_states_for_update,
    _validate_meta_payload_before_write,
    torch,
)


_TORCH_AVAILABLE = torch is not None


@unittest.skipUnless(_TORCH_AVAILABLE, "缺少 torch，跳过 model runtime tests")
class TestPhase7ModelRuntime(unittest.TestCase):
    def _build_single_observation(self) -> ObservationBatch:
        return ObservationBatch(
            uav_self_token=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            order_tokens=np.asarray(
                [
                    [1.0, 0.1, 0.2, 0.3, 0.4],
                    [1.0, 0.5, 0.6, 0.7, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            infra_tokens=np.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                ],
                dtype=np.float32,
            ),
            history_tokens=np.asarray(
                [
                    [0.0] * 8,
                    [0.1] * 8,
                    [0.2] * 8,
                    [0.3] * 8,
                    [0.4] * 8,
                    [0.5] * 8,
                ],
                dtype=np.float32,
            ),
            history_padding_mask=np.asarray([True, False, False, False, False, False], dtype=np.bool_),
            padding_mask=np.asarray([False, False, True], dtype=np.bool_),
        )

    def _build_single_critic_batch(self) -> CriticBatch:
        return CriticBatch(
            global_order_pool_tokens=np.asarray(
                [
                    [1.0, 0.1, 0.2, 0.3],
                    [1.0, 0.4, 0.5, 0.6],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            global_uav_tokens=np.asarray(
                [
                    [1.0, 0.1, 0.2],
                    [1.0, 0.3, 0.4],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            global_station_tokens=np.asarray(
                [
                    [1.0, 0.1, 0.2, 0.3, 0.4],
                    [1.0, 0.5, 0.6, 0.7, 0.8],
                ],
                dtype=np.float32,
            ),
            coarse_plan_summary_vec=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            global_system_summary_vec=np.asarray([0.4, 0.5], dtype=np.float32),
            global_order_padding_mask=np.asarray([False, False, True, True], dtype=np.bool_),
            global_uav_padding_mask=np.asarray([False, False, True], dtype=np.bool_),
            global_station_padding_mask=np.asarray([False, False], dtype=np.bool_),
        )

    def _build_single_action_mask(self) -> FactorizedActionMask:
        return FactorizedActionMask(
            root_branch_mask=np.asarray([True, True], dtype=np.bool_),
            order_mask=np.asarray([True, True, False], dtype=np.bool_),
            mode_mask=np.asarray(
                [
                    [True, True],
                    [True, False],
                    [False, False],
                ],
                dtype=np.bool_,
            ),
        )

    def _build_model(self, device: torch.device) -> SharedPPOActorCritic:
        model = SharedPPOActorCritic(
            uav_feat_dim=4,
            order_feat_dim=5,
            infra_feat_dim=7,
            history_feat_dim=8,
            critic_order_feat_dim=4,
            critic_uav_feat_dim=3,
            critic_station_feat_dim=5,
            critic_plan_feat_dim=3,
            critic_sys_feat_dim=2,
            d_model=16,
            ff_dim=32,
            lstm_hidden=12,
            lstm_layers=1,
        )
        return model.to(device)

    def _build_training_config(self) -> _TrainingConfig:
        return _TrainingConfig(
            device="cpu",
            total_timesteps=128,
            rollout_steps=32,
            recurrent_ppo=True,
            sequence_len=2,
            burn_in_len=0,
            train_len=2,
            sequence_minibatch_size=2,
            target_minibatch_timesteps=4,
            ppo_learning_rate=3e-4,
            bc_learning_rate=3e-4,
            batch_size_alias=4,
            ppo_epochs=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            entropy_coef=0.01,
            reward_scale=1.0,
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=0.2,
            mode_c_attempt_bonus=0.0,
            value_loss_coef=0.5,
            value_loss_type="huber",
            value_huber_delta=1.0,
            max_grad_norm=0.5,
            normalize_advantage=True,
            target_kl=0.02,
            wait_without_dispatch_loss_weight=0.1,
            wait_with_dispatch_loss_weight=0.35,
            mode_b_dispatch_loss_weight=1.25,
            mode_c_dispatch_loss_weight=1.75,
            training_seed=2026,
            log_interval_updates=1,
            save_interval_updates=1,
            eval_interval_updates=2,
            benchmark_eval_episodes=2,
            stochastic_eval_seeds=3,
            early_stop_enabled=True,
            early_stop_min_evals=3,
            early_stop_stochastic_high_patience=3,
            early_stop_value_loss_window=3,
            early_stop_value_loss_min_delta=0.0,
            allowed_train_order_source_mode=("poisson",),
            bc_warm_start_stages=(
                _BCWarmStartStageConfig(
                    name="test_bc",
                    order_source_mode="benchmark",
                    updates=1,
                    learning_rate=3e-4,
                ),
            ),
        )

    def test_bc_warm_start_reads_batch_labels_without_env_step(self) -> None:
        device = torch.device("cpu")

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.as_tensor(1.0, dtype=torch.float32))

            def forward(self, **_kwargs: object) -> tuple[SimpleNamespace, None]:
                return SimpleNamespace(), None

            def evaluate_actions(self, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
                return self.weight.reshape(1), torch.zeros(1, dtype=torch.float32)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_cfg = replace(
            self._build_training_config(),
            bc_warm_start_enabled=True,
            bc_warm_start_updates=1,
            bc_warm_start_batches_per_update=1,
        )
        context_1 = SimpleNamespace(
            deciding_drone_id="d1",
            coarse_plan=SimpleNamespace(plan_version=3),
        )
        context_2 = SimpleNamespace(
            deciding_drone_id="d2",
            coarse_plan=SimpleNamespace(plan_version=4),
        )
        candidate_out_1 = SimpleNamespace(
            resolved_action_lookup=SimpleNamespace(dispatch_actions={})
        )
        candidate_out_2 = SimpleNamespace(
            resolved_action_lookup=SimpleNamespace(dispatch_actions={})
        )
        fake_env = mock.Mock()
        fake_env.reset.return_value = SimpleNamespace(done=False)
        fake_env.is_done.side_effect = [False, False]
        fake_env.peek_current_decision_batch.side_effect = [(context_1,), (context_2,)]
        fake_env.build_candidate_output.side_effect = [candidate_out_1, candidate_out_2]
        fake_env.step.side_effect = AssertionError("BC warm start 不应执行 env.step")
        fake_env.apply_decision_batch.side_effect = [
            SimpleNamespace(done=False),
            SimpleNamespace(done=True),
        ]
        teacher_result_1 = SimpleNamespace(
            labels_by_drone={
                "d1": ResolvedActionIndices(root_branch_idx=0),
            },
            assignments_by_drone={
                "d1": SimpleNamespace(action=WAIT_ACTION),
            },
            actions_by_drone={
                "d1": WAIT_ACTION,
            },
        )
        teacher_result_2 = SimpleNamespace(
            labels_by_drone={
                "d2": ResolvedActionIndices(root_branch_idx=0),
            },
            assignments_by_drone={
                "d2": SimpleNamespace(action=WAIT_ACTION),
            },
            actions_by_drone={
                "d2": WAIT_ACTION,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            with mock.patch(
                "backend.training.train_cmrappo.TrainingEnvAdapter",
                return_value=fake_env,
            ):
                with mock.patch(
                    "backend.training.train_cmrappo.build_order_source",
                    return_value=SimpleNamespace(),
                ):
                    with mock.patch(
                        "backend.training.train_cmrappo.build_batch_matching_teacher_labels",
                        side_effect=[teacher_result_1, teacher_result_2],
                    ) as teacher_mock:
                        stats = _run_bc_warm_start(
                            model=model,
                            optimizer=optimizer,
                            scene_ctx=SimpleNamespace(),
                            order_source=SimpleNamespace(),
                            config_path=Path("backend/config/rh_alns_cmrappo.yaml"),
                            tensorizer=SimpleNamespace(
                                build=mock.Mock(return_value=SimpleNamespace()),
                                build_action_mask=mock.Mock(return_value=SimpleNamespace()),
                            ),
                            critic_builder=SimpleNamespace(
                                build=mock.Mock(return_value=SimpleNamespace())
                            ),
                            critic_schema=SimpleNamespace(),
                            train_cfg=train_cfg,
                            device=device,
                            metrics_path=metrics_path,
                            event_hook=None,
                        )

            self.assertEqual(stats["samples"], 2)
            self.assertEqual(stats["wait"], 2)
            self.assertEqual(stats["wait_no_dispatch"], 2)
            self.assertEqual(stats["wait_with_dispatch"], 0)
            self.assertEqual(stats["decision_batches"], 2)
            self.assertTrue(metrics_path.read_text(encoding="utf-8"))
        fake_env.step.assert_not_called()
        self.assertEqual(
            fake_env.apply_decision_batch.call_args_list,
            [
                mock.call({"d1": WAIT_ACTION}),
                mock.call({"d2": WAIT_ACTION}),
            ],
        )
        self.assertEqual(teacher_mock.call_count, 2)

    def test_bc_warm_start_can_continue_rollout_across_updates(self) -> None:
        device = torch.device("cpu")

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.as_tensor(1.0, dtype=torch.float32))

            def forward(self, **_kwargs: object) -> tuple[SimpleNamespace, None]:
                return SimpleNamespace(), None

            def evaluate_actions(self, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
                return self.weight.reshape(1), torch.zeros(1, dtype=torch.float32)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_cfg = replace(
            self._build_training_config(),
            bc_warm_start_enabled=True,
            bc_warm_start_updates=2,
            bc_warm_start_stages=(
                _BCWarmStartStageConfig(
                    name="test_bc",
                    order_source_mode="benchmark",
                    updates=2,
                    learning_rate=3e-4,
                ),
            ),
            bc_warm_start_batches_per_update=1,
            bc_warm_start_decision_batches_per_update=1,
            bc_warm_start_reset_each_update=False,
        )
        contexts = [
            SimpleNamespace(deciding_drone_id="d1", coarse_plan=SimpleNamespace(plan_version=1)),
            SimpleNamespace(deciding_drone_id="d2", coarse_plan=SimpleNamespace(plan_version=2)),
        ]
        candidate_outputs = [
            SimpleNamespace(resolved_action_lookup=SimpleNamespace(dispatch_actions={})),
            SimpleNamespace(resolved_action_lookup=SimpleNamespace(dispatch_actions={})),
        ]
        fake_env = mock.Mock()
        fake_env.reset.return_value = SimpleNamespace(done=False)
        fake_env.is_done.side_effect = [False, False, False, False]
        fake_env.peek_current_decision_batch.side_effect = [(contexts[0],), (contexts[1],)]
        fake_env.build_candidate_output.side_effect = candidate_outputs
        fake_env.step.side_effect = AssertionError("BC warm start 不应执行 env.step")
        fake_env.apply_decision_batch.side_effect = [
            SimpleNamespace(done=False),
            SimpleNamespace(done=True),
        ]
        teacher_results = [
            SimpleNamespace(
                labels_by_drone={"d1": ResolvedActionIndices(root_branch_idx=0)},
                assignments_by_drone={"d1": SimpleNamespace(action=WAIT_ACTION)},
                actions_by_drone={"d1": WAIT_ACTION},
            ),
            SimpleNamespace(
                labels_by_drone={"d2": ResolvedActionIndices(root_branch_idx=0)},
                assignments_by_drone={"d2": SimpleNamespace(action=WAIT_ACTION)},
                actions_by_drone={"d2": WAIT_ACTION},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            with mock.patch(
                "backend.training.train_cmrappo.TrainingEnvAdapter",
                return_value=fake_env,
            ):
                with mock.patch(
                    "backend.training.train_cmrappo.build_order_source",
                    return_value=SimpleNamespace(),
                ):
                    with mock.patch(
                        "backend.training.train_cmrappo.build_batch_matching_teacher_labels",
                        side_effect=teacher_results,
                    ):
                        stats = _run_bc_warm_start(
                            model=model,
                            optimizer=optimizer,
                            scene_ctx=SimpleNamespace(),
                            order_source=SimpleNamespace(),
                            config_path=Path("backend/config/rh_alns_cmrappo.yaml"),
                            tensorizer=SimpleNamespace(
                                build=mock.Mock(return_value=SimpleNamespace()),
                                build_action_mask=mock.Mock(return_value=SimpleNamespace()),
                            ),
                            critic_builder=SimpleNamespace(
                                build=mock.Mock(return_value=SimpleNamespace())
                            ),
                            critic_schema=SimpleNamespace(),
                            train_cfg=train_cfg,
                            device=device,
                            metrics_path=metrics_path,
                            event_hook=None,
                        )

            rows = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(fake_env.reset.call_count, 1)
        fake_env.step.assert_not_called()
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["decision_batches"], 2)
        self.assertEqual(stats["rollout_episodes"], 1)
        self.assertEqual(stats["completed_rollouts"], 1)
        self.assertFalse(stats["reset_each_update"])
        self.assertEqual(rows[0]["bc_decision_batches"], 1)
        self.assertEqual(rows[0]["bc_rollout_episodes"], 1)
        self.assertEqual(rows[1]["bc_decision_batches"], 1)
        self.assertEqual(rows[1]["bc_rollout_episodes"], 0)

    def test_bc_warm_start_checkpoint_round_trips_model_optimizer_and_stats(self) -> None:
        device = torch.device("cpu")
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(0.5)
        loss = model(torch.as_tensor([[1.0]])).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bc_warm_start_policy.pt"
            _save_policy_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                critic_schema_hash="schema-1",
                model_version="test-version",
                global_step=0,
                update_idx=0,
                extra_payload={
                    "phase": "bc_warm_start",
                    "bc_warm_start": {"enabled": True, "samples": 7},
                },
            )

            restored = torch.nn.Linear(1, 1)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
            stats = _load_bc_warm_start_checkpoint(
                path=path,
                model=restored,
                optimizer=restored_optimizer,
                device=device,
                critic_schema_hash="schema-1",
            )

        self.assertTrue(stats["loaded_from_checkpoint"])
        self.assertEqual(stats["samples"], 7)
        self.assertEqual(stats["checkpoint_model_version"], "test-version")
        self.assertTrue(torch.allclose(restored.weight, model.weight))
        self.assertTrue(torch.allclose(restored.bias, model.bias))
        self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_forward_accepts_single_and_batched_inputs_without_extra_unsqueeze(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()

        single_out, _ = model.forward(
            observation_batch=obs,
            action_mask=action_mask,
            critic_batch=critic,
            lstm_state=None,
        )
        self.assertEqual(tuple(single_out.root_branch_logits.shape), (1, 2))
        self.assertEqual(tuple(single_out.order_logits.shape), (1, 3))
        self.assertEqual(tuple(single_out.mode_logits.shape), (1, 3, 2))
        self.assertEqual(tuple(single_out.value.shape), (1,))

        batched_obs = ObservationBatch(
            uav_self_token=np.stack([obs.uav_self_token, obs.uav_self_token], axis=0),
            order_tokens=np.stack([obs.order_tokens, obs.order_tokens], axis=0),
            infra_tokens=np.stack([obs.infra_tokens, obs.infra_tokens], axis=0),
            history_tokens=np.stack([obs.history_tokens, obs.history_tokens], axis=0),
            history_padding_mask=np.stack([obs.history_padding_mask, obs.history_padding_mask], axis=0),
            padding_mask=np.stack([obs.padding_mask, obs.padding_mask], axis=0),
        )
        batched_critic = CriticBatch(
            global_order_pool_tokens=np.stack(
                [critic.global_order_pool_tokens, critic.global_order_pool_tokens],
                axis=0,
            ),
            global_uav_tokens=np.stack([critic.global_uav_tokens, critic.global_uav_tokens], axis=0),
            global_station_tokens=np.stack(
                [critic.global_station_tokens, critic.global_station_tokens],
                axis=0,
            ),
            coarse_plan_summary_vec=np.stack(
                [critic.coarse_plan_summary_vec, critic.coarse_plan_summary_vec],
                axis=0,
            ),
            global_system_summary_vec=np.stack(
                [critic.global_system_summary_vec, critic.global_system_summary_vec],
                axis=0,
            ),
            global_order_padding_mask=np.stack(
                [critic.global_order_padding_mask, critic.global_order_padding_mask],
                axis=0,
            ),
            global_uav_padding_mask=np.stack(
                [critic.global_uav_padding_mask, critic.global_uav_padding_mask],
                axis=0,
            ),
            global_station_padding_mask=np.stack(
                [critic.global_station_padding_mask, critic.global_station_padding_mask],
                axis=0,
            ),
        )
        batched_action_mask = FactorizedActionMask(
            root_branch_mask=np.stack([action_mask.root_branch_mask, action_mask.root_branch_mask], axis=0),
            order_mask=np.stack([action_mask.order_mask, action_mask.order_mask], axis=0),
            mode_mask=np.stack([action_mask.mode_mask, action_mask.mode_mask], axis=0),
        )

        batch_out, _ = model.forward(
            observation_batch=batched_obs,
            action_mask=batched_action_mask,
            critic_batch=batched_critic,
            lstm_state=None,
        )
        self.assertEqual(tuple(batch_out.root_branch_logits.shape), (2, 2))
        self.assertEqual(tuple(batch_out.order_logits.shape), (2, 3))
        self.assertEqual(tuple(batch_out.mode_logits.shape), (2, 3, 2))
        self.assertEqual(tuple(batch_out.value.shape), (2,))

    def test_forward_and_evaluate_actions_keep_outputs_on_model_device(self) -> None:
        device = _resolve_device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()

        policy_out, _ = model.forward(
            observation_batch=obs,
            action_mask=action_mask,
            critic_batch=critic,
            lstm_state=None,
        )
        self.assertEqual(policy_out.root_branch_logits.device.type, device.type)
        self.assertEqual(policy_out.value.device.type, device.type)

        action_indices = {
            "root_branch_idx": torch.as_tensor([1, 0], dtype=torch.long, device=device),
            "order_idx": torch.as_tensor([0, 0], dtype=torch.long, device=device),
            "mode_idx": torch.as_tensor([1, 0], dtype=torch.long, device=device),
        }
        batched_obs = ObservationBatch(
            uav_self_token=np.stack([obs.uav_self_token, obs.uav_self_token], axis=0),
            order_tokens=np.stack([obs.order_tokens, obs.order_tokens], axis=0),
            infra_tokens=np.stack([obs.infra_tokens, obs.infra_tokens], axis=0),
            history_tokens=np.stack([obs.history_tokens, obs.history_tokens], axis=0),
            history_padding_mask=np.stack([obs.history_padding_mask, obs.history_padding_mask], axis=0),
            padding_mask=np.stack([obs.padding_mask, obs.padding_mask], axis=0),
        )
        batched_critic = CriticBatch(
            global_order_pool_tokens=np.stack(
                [critic.global_order_pool_tokens, critic.global_order_pool_tokens],
                axis=0,
            ),
            global_uav_tokens=np.stack([critic.global_uav_tokens, critic.global_uav_tokens], axis=0),
            global_station_tokens=np.stack(
                [critic.global_station_tokens, critic.global_station_tokens],
                axis=0,
            ),
            coarse_plan_summary_vec=np.stack(
                [critic.coarse_plan_summary_vec, critic.coarse_plan_summary_vec],
                axis=0,
            ),
            global_system_summary_vec=np.stack(
                [critic.global_system_summary_vec, critic.global_system_summary_vec],
                axis=0,
            ),
            global_order_padding_mask=np.stack(
                [critic.global_order_padding_mask, critic.global_order_padding_mask],
                axis=0,
            ),
            global_uav_padding_mask=np.stack(
                [critic.global_uav_padding_mask, critic.global_uav_padding_mask],
                axis=0,
            ),
            global_station_padding_mask=np.stack(
                [critic.global_station_padding_mask, critic.global_station_padding_mask],
                axis=0,
            ),
        )
        batched_action_mask = FactorizedActionMask(
            root_branch_mask=np.stack([action_mask.root_branch_mask, action_mask.root_branch_mask], axis=0),
            order_mask=np.stack([action_mask.order_mask, action_mask.order_mask], axis=0),
            mode_mask=np.stack([action_mask.mode_mask, action_mask.mode_mask], axis=0),
        )
        batch_out, _ = model.forward(
            observation_batch=batched_obs,
            action_mask=batched_action_mask,
            critic_batch=batched_critic,
            lstm_state=None,
        )
        log_prob, entropy = model.evaluate_actions(
            policy_out=batch_out,
            action_mask=batched_action_mask,
            action_indices=action_indices,
        )
        self.assertEqual(log_prob.device.type, device.type)
        self.assertEqual(entropy.device.type, device.type)

    def test_stack_lstm_states_for_update_uses_saved_transition_states(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()

        first_state = (
            torch.full((1, 1, 12), 1.5, dtype=torch.float32),
            torch.full((1, 1, 12), -2.0, dtype=torch.float32),
        )
        transitions = (
            RolloutTransition(
                observation_batch=obs,
                critic_batch=critic,
                action_mask=action_mask,
                action_indices=mock.Mock(),
                log_prob_old=0.0,
                value_old=0.0,
                critic_schema_hash="schema",
                reward=0.0,
                done=False,
                lstm_state_in=None,
            ),
            RolloutTransition(
                observation_batch=obs,
                critic_batch=critic,
                action_mask=action_mask,
                action_indices=mock.Mock(),
                log_prob_old=0.0,
                value_old=0.0,
                critic_schema_hash="schema",
                reward=0.0,
                done=False,
                lstm_state_in=first_state,
            ),
        )

        stacked = _stack_lstm_states_for_update(
            transitions,
            model=model,
            device=device,
        )
        self.assertIsNotNone(stacked)
        hidden, cell = stacked
        self.assertEqual(tuple(hidden.shape), (1, 2, 12))
        self.assertEqual(tuple(cell.shape), (1, 2, 12))
        self.assertTrue(torch.allclose(hidden[:, 0, :], torch.zeros((1, 12), dtype=torch.float32)))
        self.assertTrue(torch.allclose(cell[:, 0, :], torch.zeros((1, 12), dtype=torch.float32)))
        self.assertTrue(torch.allclose(hidden[:, 1, :], torch.full((1, 12), 1.5, dtype=torch.float32)))
        self.assertTrue(torch.allclose(cell[:, 1, :], torch.full((1, 12), -2.0, dtype=torch.float32)))

    def test_evaluate_actions_entropy_uses_order_expectation_for_wait_samples(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        policy_out = PolicyForwardOutput(
            root_branch_logits=torch.as_tensor([[0.0, 0.0]], dtype=torch.float32, device=device),
            order_logits=torch.as_tensor([[0.0, 0.0, -1e9]], dtype=torch.float32, device=device),
            mode_logits=torch.as_tensor(
                [[[20.0, -20.0], [0.0, 0.0], [-1e9, -1e9]]],
                dtype=torch.float32,
                device=device,
            ),
            value=torch.as_tensor([0.0], dtype=torch.float32, device=device),
        )
        action_mask = FactorizedActionMask(
            root_branch_mask=np.asarray([[True, True]], dtype=np.bool_),
            order_mask=np.asarray([[True, True, False]], dtype=np.bool_),
            mode_mask=np.asarray(
                [[[True, True], [True, True], [False, False]]],
                dtype=np.bool_,
            ),
        )
        action_indices = {
            "root_branch_idx": torch.as_tensor([0], dtype=torch.long, device=device),
            "order_idx": torch.as_tensor([0], dtype=torch.long, device=device),
            "mode_idx": torch.as_tensor([0], dtype=torch.long, device=device),
        }

        _log_prob, entropy = model.evaluate_actions(
            policy_out=policy_out,
            action_mask=action_mask,
            action_indices=action_indices,
        )

        ln2 = float(np.log(2.0))
        expected_entropy = ln2 + 0.5 * (ln2 + 0.5 * ln2)
        self.assertAlmostEqual(float(entropy.item()), expected_entropy, places=6)

    def test_evaluate_actions_entropy_has_no_recovery_component(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        policy_out = PolicyForwardOutput(
            root_branch_logits=torch.as_tensor([[0.0, 0.0]], dtype=torch.float32, device=device),
            order_logits=torch.as_tensor([[0.0, 0.0, -1e9]], dtype=torch.float32, device=device),
            mode_logits=torch.as_tensor(
                [[[20.0, -20.0], [0.0, 0.0], [-1e9, -1e9]]],
                dtype=torch.float32,
                device=device,
            ),
            value=torch.as_tensor([0.0], dtype=torch.float32, device=device),
        )
        action_mask = FactorizedActionMask(
            root_branch_mask=np.asarray([[True, True]], dtype=np.bool_),
            order_mask=np.asarray([[True, True, False]], dtype=np.bool_),
            mode_mask=np.asarray(
                [[[True, True], [True, True], [False, False]]],
                dtype=np.bool_,
            ),
        )
        action_indices = {
            "root_branch_idx": torch.as_tensor([0], dtype=torch.long, device=device),
            "order_idx": torch.as_tensor([0], dtype=torch.long, device=device),
            "mode_idx": torch.as_tensor([0], dtype=torch.long, device=device),
        }

        _log_prob, entropy = model.evaluate_actions(
            policy_out=policy_out,
            action_mask=action_mask,
            action_indices=action_indices,
        )

        ln2 = float(np.log(2.0))
        expected_entropy = ln2 + 0.5 * (ln2 + 0.5 * ln2)
        self.assertAlmostEqual(
            float(entropy.item()),
            expected_entropy,
            places=6,
        )

    def test_evaluate_actions_accepts_wait_only_masks_in_batch(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        policy_out = PolicyForwardOutput(
            root_branch_logits=torch.as_tensor(
                [[0.0, 0.0], [0.0, -1e9]],
                dtype=torch.float32,
                device=device,
            ),
            order_logits=torch.as_tensor(
                [[0.0, 0.0, -1e9], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=device,
            ),
            mode_logits=torch.as_tensor(
                [
                    [[0.0, 0.0], [0.0, -1e9], [-1e9, -1e9]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
                device=device,
            ),
            value=torch.as_tensor([0.0, 0.0], dtype=torch.float32, device=device),
        )
        action_mask = FactorizedActionMask(
            root_branch_mask=np.asarray([[True, True], [True, False]], dtype=np.bool_),
            order_mask=np.asarray(
                [[True, True, False], [False, False, False]],
                dtype=np.bool_,
            ),
            mode_mask=np.asarray(
                [
                    [[True, True], [True, False], [False, False]],
                    [[False, False], [False, False], [False, False]],
                ],
                dtype=np.bool_,
            ),
        )
        action_indices = {
            "root_branch_idx": torch.as_tensor([1, 0], dtype=torch.long, device=device),
            "order_idx": torch.as_tensor([0, 0], dtype=torch.long, device=device),
            "mode_idx": torch.as_tensor([0, 0], dtype=torch.long, device=device),
        }

        log_prob, entropy = model.evaluate_actions(
            policy_out=policy_out,
            action_mask=action_mask,
            action_indices=action_indices,
        )

        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(entropy.shape), (2,))
        self.assertTrue(torch.isfinite(log_prob).all().item())
        self.assertTrue(torch.isfinite(entropy).all().item())

    def test_forward_accepts_batched_lstm_state(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()

        batched_obs = ObservationBatch(
            uav_self_token=np.stack([obs.uav_self_token, obs.uav_self_token], axis=0),
            order_tokens=np.stack([obs.order_tokens, obs.order_tokens], axis=0),
            infra_tokens=np.stack([obs.infra_tokens, obs.infra_tokens], axis=0),
            history_tokens=np.stack([obs.history_tokens, obs.history_tokens], axis=0),
            history_padding_mask=np.stack([obs.history_padding_mask, obs.history_padding_mask], axis=0),
            padding_mask=np.stack([obs.padding_mask, obs.padding_mask], axis=0),
        )
        batched_critic = CriticBatch(
            global_order_pool_tokens=np.stack(
                [critic.global_order_pool_tokens, critic.global_order_pool_tokens],
                axis=0,
            ),
            global_uav_tokens=np.stack([critic.global_uav_tokens, critic.global_uav_tokens], axis=0),
            global_station_tokens=np.stack(
                [critic.global_station_tokens, critic.global_station_tokens],
                axis=0,
            ),
            coarse_plan_summary_vec=np.stack(
                [critic.coarse_plan_summary_vec, critic.coarse_plan_summary_vec],
                axis=0,
            ),
            global_system_summary_vec=np.stack(
                [critic.global_system_summary_vec, critic.global_system_summary_vec],
                axis=0,
            ),
            global_order_padding_mask=np.stack(
                [critic.global_order_padding_mask, critic.global_order_padding_mask],
                axis=0,
            ),
            global_uav_padding_mask=np.stack(
                [critic.global_uav_padding_mask, critic.global_uav_padding_mask],
                axis=0,
            ),
            global_station_padding_mask=np.stack(
                [critic.global_station_padding_mask, critic.global_station_padding_mask],
                axis=0,
            ),
        )
        batched_action_mask = FactorizedActionMask(
            root_branch_mask=np.stack([action_mask.root_branch_mask, action_mask.root_branch_mask], axis=0),
            order_mask=np.stack([action_mask.order_mask, action_mask.order_mask], axis=0),
            mode_mask=np.stack([action_mask.mode_mask, action_mask.mode_mask], axis=0),
        )
        lstm_state = (
            torch.zeros((1, 2, 12), dtype=torch.float32, device=device),
            torch.zeros((1, 2, 12), dtype=torch.float32, device=device),
        )

        batch_out, next_state = model.forward(
            observation_batch=batched_obs,
            action_mask=batched_action_mask,
            critic_batch=batched_critic,
            lstm_state=lstm_state,
        )
        self.assertEqual(tuple(batch_out.value.shape), (2,))
        self.assertEqual(tuple(next_state[0].shape), (1, 2, 12))
        self.assertEqual(tuple(next_state[1].shape), (1, 2, 12))

    def test_forward_sequence_preserves_batch_and_time_axes(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()

        sequence_obs = ObservationBatch(
            uav_self_token=np.stack(
                [
                    np.stack([obs.uav_self_token, obs.uav_self_token], axis=0),
                    np.stack([obs.uav_self_token, obs.uav_self_token], axis=0),
                ],
                axis=0,
            ),
            order_tokens=np.stack(
                [
                    np.stack([obs.order_tokens, obs.order_tokens], axis=0),
                    np.stack([obs.order_tokens, obs.order_tokens], axis=0),
                ],
                axis=0,
            ),
            infra_tokens=np.stack(
                [
                    np.stack([obs.infra_tokens, obs.infra_tokens], axis=0),
                    np.stack([obs.infra_tokens, obs.infra_tokens], axis=0),
                ],
                axis=0,
            ),
            history_tokens=np.stack(
                [
                    np.stack([obs.history_tokens, obs.history_tokens], axis=0),
                    np.stack([obs.history_tokens, obs.history_tokens], axis=0),
                ],
                axis=0,
            ),
            history_padding_mask=np.stack(
                [
                    np.stack([obs.history_padding_mask, obs.history_padding_mask], axis=0),
                    np.stack([obs.history_padding_mask, obs.history_padding_mask], axis=0),
                ],
                axis=0,
            ),
            padding_mask=np.stack(
                [
                    np.stack([obs.padding_mask, obs.padding_mask], axis=0),
                    np.stack([obs.padding_mask, obs.padding_mask], axis=0),
                ],
                axis=0,
            ),
        )
        sequence_critic = CriticBatch(
            global_order_pool_tokens=np.stack(
                [
                    np.stack([critic.global_order_pool_tokens, critic.global_order_pool_tokens], axis=0),
                    np.stack([critic.global_order_pool_tokens, critic.global_order_pool_tokens], axis=0),
                ],
                axis=0,
            ),
            global_uav_tokens=np.stack(
                [
                    np.stack([critic.global_uav_tokens, critic.global_uav_tokens], axis=0),
                    np.stack([critic.global_uav_tokens, critic.global_uav_tokens], axis=0),
                ],
                axis=0,
            ),
            global_station_tokens=np.stack(
                [
                    np.stack([critic.global_station_tokens, critic.global_station_tokens], axis=0),
                    np.stack([critic.global_station_tokens, critic.global_station_tokens], axis=0),
                ],
                axis=0,
            ),
            coarse_plan_summary_vec=np.stack(
                [
                    np.stack([critic.coarse_plan_summary_vec, critic.coarse_plan_summary_vec], axis=0),
                    np.stack([critic.coarse_plan_summary_vec, critic.coarse_plan_summary_vec], axis=0),
                ],
                axis=0,
            ),
            global_system_summary_vec=np.stack(
                [
                    np.stack([critic.global_system_summary_vec, critic.global_system_summary_vec], axis=0),
                    np.stack([critic.global_system_summary_vec, critic.global_system_summary_vec], axis=0),
                ],
                axis=0,
            ),
            global_order_padding_mask=np.stack(
                [
                    np.stack([critic.global_order_padding_mask, critic.global_order_padding_mask], axis=0),
                    np.stack([critic.global_order_padding_mask, critic.global_order_padding_mask], axis=0),
                ],
                axis=0,
            ),
            global_uav_padding_mask=np.stack(
                [
                    np.stack([critic.global_uav_padding_mask, critic.global_uav_padding_mask], axis=0),
                    np.stack([critic.global_uav_padding_mask, critic.global_uav_padding_mask], axis=0),
                ],
                axis=0,
            ),
            global_station_padding_mask=np.stack(
                [
                    np.stack([critic.global_station_padding_mask, critic.global_station_padding_mask], axis=0),
                    np.stack([critic.global_station_padding_mask, critic.global_station_padding_mask], axis=0),
                ],
                axis=0,
            ),
        )
        sequence_action_mask = FactorizedActionMask(
            root_branch_mask=np.stack(
                [
                    np.stack([action_mask.root_branch_mask, action_mask.root_branch_mask], axis=0),
                    np.stack([action_mask.root_branch_mask, action_mask.root_branch_mask], axis=0),
                ],
                axis=0,
            ),
            order_mask=np.stack(
                [
                    np.stack([action_mask.order_mask, action_mask.order_mask], axis=0),
                    np.stack([action_mask.order_mask, action_mask.order_mask], axis=0),
                ],
                axis=0,
            ),
            mode_mask=np.stack(
                [
                    np.stack([action_mask.mode_mask, action_mask.mode_mask], axis=0),
                    np.stack([action_mask.mode_mask, action_mask.mode_mask], axis=0),
                ],
                axis=0,
            ),
        )
        policy_out, next_state = model.forward_sequence(
            observation_batch=sequence_obs,
            action_mask=sequence_action_mask,
            critic_batch=sequence_critic,
            lstm_state=None,
        )
        self.assertEqual(tuple(policy_out.root_branch_logits.shape), (2, 2, 2))
        self.assertEqual(tuple(policy_out.order_logits.shape), (2, 2, 3))
        self.assertEqual(tuple(policy_out.mode_logits.shape), (2, 2, 3, 2))
        self.assertEqual(tuple(policy_out.value.shape), (2, 2))
        self.assertEqual(tuple(next_state[0].shape), (1, 2, 12))

    def test_materialized_sequences_group_by_actor_and_pad_tail(self) -> None:
        device = torch.device("cpu")
        model = self._build_model(device)
        obs = self._build_single_observation()
        critic = self._build_single_critic_batch()
        action_mask = self._build_single_action_mask()
        transitions = (
            RolloutTransition(
                observation_batch=obs,
                critic_batch=critic,
                action_mask=action_mask,
                action_indices=SimpleNamespace(
                    root_branch_idx=1,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=1.0,
                done=False,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                local_decision_index=0,
            ),
            RolloutTransition(
                observation_batch=obs,
                critic_batch=critic,
                action_mask=action_mask,
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=None,
                    mode_idx=None,
                ),
                log_prob_old=-0.2,
                value_old=0.4,
                reward=0.5,
                done=False,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_2",
                recurrent_segment_id=0,
                local_decision_index=0,
            ),
            RolloutTransition(
                observation_batch=obs,
                critic_batch=critic,
                action_mask=action_mask,
                action_indices=SimpleNamespace(
                    root_branch_idx=1,
                    order_idx=1,
                    mode_idx=1,
                ),
                log_prob_old=-0.3,
                value_old=0.3,
                reward=0.2,
                done=True,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                local_decision_index=1,
            ),
        )
        batch_view = SimpleNamespace(
            transitions=transitions,
            rewards=np.asarray([1.0, 0.5, 0.2], dtype=np.float32),
            dones=np.asarray([False, False, True], dtype=np.bool_),
            values=np.asarray([0.5, 0.4, 0.3], dtype=np.float32),
        )

        sequences = _materialize_recurrent_sequences(
            batch_view=batch_view,
            sequence_len=2,
            gamma=0.99,
            gae_lambda=0.95,
            tail_bootstrap_values={
                (0, "uav_2", 0): 0.25,
            },
        )
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0].sequence_id, (0, "uav_1", 0, 0))
        self.assertEqual(len(sequences[0].transitions), 2)
        self.assertEqual(sequences[1].sequence_id, (0, "uav_2", 0, 0))
        self.assertEqual(len(sequences[1].transitions), 1)
        expected_uav1_adv, expected_uav1_ret = compute_gae(
            rewards=np.asarray([1.0, 0.2], dtype=np.float32),
            dones=np.asarray([False, True], dtype=np.bool_),
            values=np.asarray([0.5, 0.3], dtype=np.float32),
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        expected_uav2_adv, expected_uav2_ret = compute_gae(
            rewards=np.asarray([0.5], dtype=np.float32),
            dones=np.asarray([False], dtype=np.bool_),
            values=np.asarray([0.4], dtype=np.float32),
            last_value=0.25,
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertTrue(np.allclose(sequences[0].advantages, expected_uav1_adv))
        self.assertTrue(np.allclose(sequences[0].returns, expected_uav1_ret))
        self.assertTrue(np.allclose(sequences[1].advantages, expected_uav2_adv))
        self.assertTrue(np.allclose(sequences[1].returns, expected_uav2_ret))

        minibatch = _build_sequence_minibatch(
            sequences=sequences,
            train_cfg=self._build_training_config(),
            model=model,
            device=device,
        )
        self.assertEqual(tuple(minibatch.valid_timestep_mask.shape), (2, 2))
        self.assertTrue(bool(minibatch.valid_timestep_mask[0, 0].item()))
        self.assertTrue(bool(minibatch.valid_timestep_mask[0, 1].item()))
        self.assertTrue(bool(minibatch.valid_timestep_mask[1, 0].item()))
        self.assertFalse(bool(minibatch.valid_timestep_mask[1, 1].item()))
        self.assertEqual(minibatch.padded_timesteps, 1)
        self.assertEqual(int(minibatch.action_indices["root_branch_idx"][1, 1].item()), 0)

    def test_extract_attributed_step_reward_parts_prefers_explicit_breakdown(self) -> None:
        step_result = SimpleNamespace(
            reward=-9.0,
            info={
                "reward_breakdown": {
                    "attributed_carry_in": -3.0,
                    "attributed_post_action": -2.5,
                }
            },
        )
        carry_in, post_action, mode_c_attempt = _extract_attributed_step_reward_parts(
            step_result=step_result
        )
        self.assertAlmostEqual(carry_in, -3.0, places=6)
        self.assertAlmostEqual(post_action, -2.5, places=6)
        self.assertAlmostEqual(mode_c_attempt, 0.0, places=6)

    def test_shape_post_action_reward_for_rendezvous_adds_success_bonus_only(self) -> None:
        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_post_action_reward_for_rendezvous(
            post_action_reward=-1.5,
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            ),
            transition_summary=SimpleNamespace(rendezvous_success=True),
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=0.2,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertTrue(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -1.3, places=6)

    def test_shape_post_action_reward_for_rendezvous_does_not_add_arrive_bonus_on_waiting_state(self) -> None:
        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_post_action_reward_for_rendezvous(
            post_action_reward=-1.5,
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            ),
            transition_summary=SimpleNamespace(
                rendezvous_success=False,
                actor_training_state_after="waiting_for_truck",
            ),
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=5.0,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertFalse(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -1.5, places=6)

    def test_shape_post_action_reward_for_rendezvous_does_not_add_attempt_bonus_at_dispatch(self) -> None:
        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_post_action_reward_for_rendezvous(
            post_action_reward=-1.5,
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            ),
            transition_summary=SimpleNamespace(
                rendezvous_success=False,
                actor_training_state_after="flying",
            ),
            rendezvous_arrive_bonus=4.0,
            rendezvous_bonus=12.0,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertFalse(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -1.5, places=6)

    def test_shape_post_action_reward_for_rendezvous_leaves_non_mode_c_unchanged(
        self,
    ) -> None:
        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_post_action_reward_for_rendezvous(
            post_action_reward=-1.5,
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=0,
            ),
            transition_summary=SimpleNamespace(
                rendezvous_success=False,
                actor_training_state_after="flying",
            ),
            rendezvous_arrive_bonus=4.0,
            rendezvous_bonus=12.0,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertFalse(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -1.5, places=6)

    def test_shape_pending_transition_reward_for_rendezvous_uses_success_state(self) -> None:
        pending_transition = RolloutTransition(
            observation_batch=self._build_single_observation(),
            critic_batch=self._build_single_critic_batch(),
            action_mask=self._build_single_action_mask(),
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            ),
            log_prob_old=-0.1,
            value_old=0.5,
            reward=-1.0,
            done=None,
            critic_schema_hash="schema",
            actor_drone_id="uav_1",
        )

        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_pending_transition_reward_for_rendezvous(
            pending_transition=pending_transition,
            reward_so_far=-1.0,
            runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="riding_with_truck"),
                }
            ),
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=0.2,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertTrue(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -0.8, places=6)

    def test_shape_pending_transition_reward_for_rendezvous_does_not_add_arrive_on_waiting_state(self) -> None:
        pending_transition = RolloutTransition(
            observation_batch=self._build_single_observation(),
            critic_batch=self._build_single_critic_batch(),
            action_mask=self._build_single_action_mask(),
            action_indices=SimpleNamespace(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            ),
            log_prob_old=-0.1,
            value_old=0.5,
            reward=-1.0,
            done=None,
            critic_schema_hash="schema",
            actor_drone_id="uav_1",
        )

        (
            shaped_reward,
            arrive_bonus_applied,
            success_bonus_applied,
        ) = _shape_pending_transition_reward_for_rendezvous(
            pending_transition=pending_transition,
            reward_so_far=-1.0,
            runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="waiting_for_truck"),
                }
            ),
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=5.0,
        )

        self.assertFalse(arrive_bonus_applied)
        self.assertFalse(success_bonus_applied)
        self.assertAlmostEqual(shaped_reward, -1.0, places=6)

    def test_finalize_pending_transition_records_successor_bootstrap(self) -> None:
        pending = {
            "uav_1": RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=-1.25,
                done=None,
                critic_schema_hash="schema",
                episode_id=3,
                actor_drone_id="uav_1",
                recurrent_segment_id=2,
                local_decision_index=4,
                global_decision_index=11,
            )
        }
        backlog: list[RolloutTransition] = []
        successor_bootstrap_values: dict[tuple[int, str, int], float] = {}

        _finalize_pending_transition_for_next_decision(
            pending_transition_by_drone=pending,
            rollout_backlog=backlog,
            successor_bootstrap_values=successor_bootstrap_values,
            drone_id="uav_1",
            carry_in_reward=-0.75,
            next_value=1.75,
        )

        self.assertFalse(pending)
        self.assertEqual(len(backlog), 1)
        self.assertAlmostEqual(float(backlog[0].reward), -2.0, places=6)
        self.assertFalse(bool(backlog[0].done))
        self.assertEqual(successor_bootstrap_values, {(3, "uav_1", 2): 1.75})

    def test_finalize_pending_transition_applies_success_bonus_to_mode_c_pending(self) -> None:
        pending = {
            "uav_1": RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=1,
                    order_idx=0,
                    mode_idx=1,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=-1.25,
                done=None,
                critic_schema_hash="schema",
                episode_id=3,
                actor_drone_id="uav_1",
                recurrent_segment_id=2,
                local_decision_index=4,
                global_decision_index=11,
            )
        }
        backlog: list[RolloutTransition] = []
        successor_bootstrap_values: dict[tuple[int, str, int], float] = {}

        _finalize_pending_transition_for_next_decision(
            pending_transition_by_drone=pending,
            rollout_backlog=backlog,
            successor_bootstrap_values=successor_bootstrap_values,
            drone_id="uav_1",
            carry_in_reward=-0.75,
            next_value=1.75,
            decision_context=SimpleNamespace(
                runtime_state=SimpleNamespace(
                    drone_states={
                        "uav_1": SimpleNamespace(training_state="charging_on_truck"),
                    }
                )
            ),
            rendezvous_arrive_bonus=2.0,
            rendezvous_bonus=0.2,
        )

        self.assertFalse(pending)
        self.assertEqual(len(backlog), 1)
        self.assertAlmostEqual(float(backlog[0].reward), -1.8, places=6)
        self.assertFalse(bool(backlog[0].rendezvous_arrive_bonus_applied))
        self.assertTrue(bool(backlog[0].rendezvous_success_bonus_applied))

    def test_finalize_pending_transition_scales_late_success_bonus(self) -> None:
        pending = {
            "uav_1": RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=1,
                    order_idx=0,
                    mode_idx=1,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=-0.0125,
                done=None,
                critic_schema_hash="schema",
                episode_id=3,
                actor_drone_id="uav_1",
                recurrent_segment_id=2,
                local_decision_index=4,
                global_decision_index=11,
            )
        }
        backlog: list[RolloutTransition] = []
        successor_bootstrap_values: dict[tuple[int, str, int], float] = {}

        _finalize_pending_transition_for_next_decision(
            pending_transition_by_drone=pending,
            rollout_backlog=backlog,
            successor_bootstrap_values=successor_bootstrap_values,
            drone_id="uav_1",
            carry_in_reward=-0.75,
            next_value=1.75,
            decision_context=SimpleNamespace(
                runtime_state=SimpleNamespace(
                    drone_states={
                        "uav_1": SimpleNamespace(training_state="charging_on_truck"),
                    }
                )
            ),
            rendezvous_arrive_bonus=0.02,
            rendezvous_bonus=0.002,
            reward_scale=0.01,
        )

        self.assertEqual(len(backlog), 1)
        self.assertAlmostEqual(float(backlog[0].reward), -0.018, places=6)
        self.assertFalse(bool(backlog[0].rendezvous_arrive_bonus_applied))
        self.assertTrue(bool(backlog[0].rendezvous_success_bonus_applied))

    def test_flush_terminal_pending_transitions_marks_done_and_preserves_order(self) -> None:
        first = RolloutTransition(
            observation_batch=self._build_single_observation(),
            critic_batch=self._build_single_critic_batch(),
            action_mask=self._build_single_action_mask(),
            action_indices=SimpleNamespace(
                root_branch_idx=0,
                order_idx=0,
                mode_idx=0,
            ),
            log_prob_old=-0.1,
            value_old=0.5,
            reward=1.0,
            done=None,
            critic_schema_hash="schema",
            episode_id=0,
            actor_drone_id="uav_1",
            recurrent_segment_id=0,
            local_decision_index=0,
            global_decision_index=1,
        )
        second = RolloutTransition(
            observation_batch=self._build_single_observation(),
            critic_batch=self._build_single_critic_batch(),
            action_mask=self._build_single_action_mask(),
            action_indices=SimpleNamespace(
                root_branch_idx=0,
                order_idx=0,
                mode_idx=0,
            ),
            log_prob_old=-0.2,
            value_old=0.4,
            reward=2.0,
            done=None,
            critic_schema_hash="schema",
            episode_id=0,
            actor_drone_id="uav_2",
            recurrent_segment_id=0,
            local_decision_index=0,
            global_decision_index=3,
        )
        backlog = [
            RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.3,
                value_old=0.3,
                reward=0.0,
                done=True,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_0",
                recurrent_segment_id=0,
                local_decision_index=0,
                global_decision_index=5,
            )
        ]
        pending = {"uav_1": first, "uav_2": second}

        _flush_terminal_pending_transitions(
            pending_transition_by_drone=pending,
            rollout_backlog=backlog,
            terminal_reward_by_drone={"uav_2": -0.5},
        )

        self.assertFalse(pending)
        self.assertEqual([item.actor_drone_id for item in backlog], ["uav_1", "uav_2", "uav_0"])
        self.assertTrue(all(item.done for item in backlog[:2]))
        self.assertAlmostEqual(float(backlog[0].reward), 1.0, places=6)
        self.assertAlmostEqual(float(backlog[1].reward), 1.5, places=6)

    def test_flush_terminal_pending_transitions_scales_late_success_bonus(self) -> None:
        pending = {
            "uav_1": RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=1,
                    order_idx=0,
                    mode_idx=1,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=-0.0125,
                done=None,
                critic_schema_hash="schema",
                episode_id=3,
                actor_drone_id="uav_1",
                recurrent_segment_id=2,
                local_decision_index=4,
                global_decision_index=11,
            )
        }
        backlog: list[RolloutTransition] = []

        _flush_terminal_pending_transitions(
            pending_transition_by_drone=pending,
            rollout_backlog=backlog,
            terminal_reward_by_drone={"uav_1": -0.75},
            runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="charging_on_truck"),
                }
            ),
            rendezvous_arrive_bonus=0.02,
            rendezvous_bonus=0.002,
            reward_scale=0.01,
        )

        self.assertFalse(pending)
        self.assertEqual(len(backlog), 1)
        self.assertAlmostEqual(float(backlog[0].reward), -0.018, places=6)
        self.assertTrue(bool(backlog[0].done))
        self.assertFalse(bool(backlog[0].rendezvous_arrive_bonus_applied))
        self.assertTrue(bool(backlog[0].rendezvous_success_bonus_applied))

    def test_insert_finalized_transition_in_order_uses_global_decision_index(self) -> None:
        backlog = [
            RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.2,
                reward=0.0,
                done=True,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                local_decision_index=0,
                global_decision_index=2,
            ),
            RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.2,
                reward=0.0,
                done=True,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_3",
                recurrent_segment_id=0,
                local_decision_index=0,
                global_decision_index=6,
            ),
        ]

        _insert_finalized_transition_in_order(
            rollout_backlog=backlog,
            transition=RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.2,
                reward=0.0,
                done=True,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_2",
                recurrent_segment_id=0,
                local_decision_index=0,
                global_decision_index=4,
            ),
        )

        self.assertEqual(
            [item.global_decision_index for item in backlog],
            [2, 4, 6],
        )

    def test_resolve_rollout_prefix_bootstrap_values_reads_same_actor_successor(self) -> None:
        transitions = (
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                reward=1.0,
                done=False,
                value_old=0.5,
            ),
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_2",
                recurrent_segment_id=0,
                reward=0.5,
                done=False,
                value_old=0.4,
            ),
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                reward=0.2,
                done=True,
                value_old=0.3,
            ),
        )
        bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=transitions,
            prefix_len=2,
            boundary_bootstrap_values={},
            current_runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                    "uav_2": SimpleNamespace(training_state="idle"),
                }
            ),
            episode_terminated_at_boundary=False,
        )
        self.assertEqual(
            bootstrap_values,
            {
                (0, "uav_1", 0): 0.3,
                (0, "uav_2", 0): 0.0,
            },
        )

    def test_resolve_rollout_prefix_bootstrap_values_uses_boundary_value_when_needed(self) -> None:
        transitions = (
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                reward=1.0,
                done=False,
                value_old=0.5,
            ),
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_2",
                recurrent_segment_id=0,
                reward=0.5,
                done=False,
                value_old=0.4,
            ),
        )
        bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=transitions,
            prefix_len=2,
            boundary_bootstrap_values={(0, "uav_2", 0): 0.25},
            current_runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                    "uav_2": SimpleNamespace(training_state="idle"),
                }
            ),
            episode_terminated_at_boundary=False,
        )
        self.assertIsNone(bootstrap_values)

        bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=transitions,
            prefix_len=2,
            boundary_bootstrap_values={
                (0, "uav_1", 0): 0.35,
                (0, "uav_2", 0): 0.25,
            },
            current_runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                    "uav_2": SimpleNamespace(training_state="idle"),
                }
            ),
            episode_terminated_at_boundary=False,
        )
        self.assertEqual(
            bootstrap_values,
            {
                (0, "uav_1", 0): 0.35,
                (0, "uav_2", 0): 0.25,
            },
        )

    def test_resolve_rollout_prefix_bootstrap_values_uses_zero_only_for_true_terminal_boundary(self) -> None:
        transitions = (
            SimpleNamespace(
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                reward=1.0,
                done=False,
                value_old=0.5,
            ),
        )
        bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=transitions,
            prefix_len=1,
            boundary_bootstrap_values={},
            current_runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                }
            ),
            episode_terminated_at_boundary=False,
        )
        self.assertIsNone(bootstrap_values)

        bootstrap_values = _resolve_rollout_prefix_bootstrap_values(
            transitions=transitions,
            prefix_len=1,
            boundary_bootstrap_values={},
            current_runtime_state=SimpleNamespace(
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                }
            ),
            episode_terminated_at_boundary=True,
        )
        self.assertEqual(bootstrap_values, {(0, "uav_1", 0): 0.0})

    def test_advance_until_rollout_bootstraps_resolved_after_collection_probes_live_successor(self) -> None:
        current_result = SimpleNamespace(
            done=False,
            runtime_state=SimpleNamespace(
                t_now=10.0,
                drone_states={
                    "uav_1": SimpleNamespace(training_state="idle"),
                    "uav_2": SimpleNamespace(training_state="idle"),
                },
            ),
            decision_context=SimpleNamespace(
                deciding_drone_id="uav_2",
                coarse_plan=SimpleNamespace(plan_version=7),
            ),
        )
        rollout_backlog = [
            RolloutTransition(
                observation_batch=self._build_single_observation(),
                critic_batch=self._build_single_critic_batch(),
                action_mask=self._build_single_action_mask(),
                action_indices=SimpleNamespace(
                    root_branch_idx=0,
                    order_idx=0,
                    mode_idx=0,
                ),
                log_prob_old=-0.1,
                value_old=0.5,
                reward=1.0,
                done=False,
                critic_schema_hash="schema",
                episode_id=0,
                actor_drone_id="uav_1",
                recurrent_segment_id=0,
                local_decision_index=0,
                global_decision_index=0,
            )
        ]
        fake_policy_out = SimpleNamespace(value=torch.as_tensor([1.75], dtype=torch.float32))
        fake_model = SimpleNamespace(
            forward=mock.Mock(return_value=(fake_policy_out, None)),
            sample_action=mock.Mock(
                return_value=(
                    {
                        "root_branch_idx": 1,
                        "order_idx": 0,
                        "mode_idx": 0,
                    },
                    torch.as_tensor(0.0),
                )
            ),
        )
        fake_tensorizer = SimpleNamespace(
            build=mock.Mock(return_value=SimpleNamespace()),
            build_action_mask=mock.Mock(return_value=SimpleNamespace()),
            build_transition_summary=mock.Mock(return_value=SimpleNamespace()),
        )
        fake_critic_builder = SimpleNamespace(build=mock.Mock(return_value=SimpleNamespace()))
        fake_env = SimpleNamespace(
            step=mock.Mock(
                return_value=SimpleNamespace(
                    done=False,
                    runtime_state=SimpleNamespace(
                        t_now=12.0,
                        drone_states={
                            "uav_1": SimpleNamespace(training_state="idle"),
                            "uav_2": SimpleNamespace(training_state="idle"),
                        },
                    ),
                    decision_context=SimpleNamespace(
                        deciding_drone_id="uav_1",
                        coarse_plan=SimpleNamespace(plan_version=8),
                    ),
                    info={"reward_breakdown": {}},
                )
            )
        )
        history_buffer = []
        last_seen_plan_version_by_drone: dict[str, int] = {}
        lstm_state_by_drone: dict[str, object] = {}
        recurrent_segment_id_by_drone: dict[str, int] = {}
        successor_bootstrap_values: dict[tuple[int, str, int], float] = {}

        with mock.patch(
            "backend.training.train_cmrappo._build_candidate_output",
            return_value=SimpleNamespace(
                resolved_action_lookup=SimpleNamespace(
                    resolve=mock.Mock(return_value=SimpleNamespace())
                )
            ),
        ):
            result_after_probe = _advance_until_rollout_bootstraps_resolved_after_collection(
                current_result=current_result,
                episode_id=0,
                env=fake_env,
                model=fake_model,
                tensorizer=fake_tensorizer,
                critic_builder=fake_critic_builder,
                critic_schema=SimpleNamespace(),
                last_seen_plan_version_by_drone=last_seen_plan_version_by_drone,
                history_buffer=history_buffer,
                lstm_state_by_drone=lstm_state_by_drone,
                recurrent_segment_id_by_drone=recurrent_segment_id_by_drone,
                rollout_backlog=rollout_backlog,
                successor_bootstrap_values=successor_bootstrap_values,
            )

        self.assertIs(result_after_probe, fake_env.step.return_value)
        self.assertEqual(successor_bootstrap_values, {(0, "uav_2", 0): 1.75})
        fake_env.step.assert_called_once()

    def test_compute_masked_approx_kl_uses_signed_mean(self) -> None:
        old_log_probs = torch.as_tensor([0.0, 0.0], dtype=torch.float32)
        new_log_probs = torch.as_tensor([1.0, -0.5], dtype=torch.float32)
        valid_mask = torch.as_tensor([True, True], dtype=torch.bool)
        approx_kl = _compute_masked_approx_kl(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            valid_mask=valid_mask,
        )
        self.assertAlmostEqual(float(approx_kl.item()), -0.25, places=6)

    def test_compute_value_loss_masked_supports_mse_and_huber(self) -> None:
        values = torch.as_tensor([3.0, 0.0], dtype=torch.float32)
        returns = torch.as_tensor([1.0, 0.0], dtype=torch.float32)
        valid_mask = torch.as_tensor([True, False], dtype=torch.bool)

        mse_loss = _compute_value_loss_masked(
            values=values,
            returns=returns,
            valid_mask=valid_mask,
            loss_type="mse",
            huber_delta=1.0,
        )
        huber_loss = _compute_value_loss_masked(
            values=values,
            returns=returns,
            valid_mask=valid_mask,
            loss_type="huber",
            huber_delta=1.0,
        )

        self.assertAlmostEqual(float(mse_loss.item()), 2.0, places=6)
        self.assertAlmostEqual(float(huber_loss.item()), 1.5, places=6)

    def test_ppo_update_weights_policy_loss_only(self) -> None:
        train_cfg = self._build_training_config()
        train_cfg = _TrainingConfig(
            **{
                **train_cfg.__dict__,
                "ppo_epochs": 1,
                "sequence_minibatch_size": 1,
                "normalize_advantage": False,
                "target_kl": 10.0,
                "value_loss_type": "mse",
            }
        )
        fake_sequence = SimpleNamespace(
            valid_timestep_mask=np.asarray([True, True], dtype=np.bool_),
            transitions=(SimpleNamespace(), SimpleNamespace()),
            advantages=np.asarray([1.0, 3.0], dtype=np.float32),
            returns=np.asarray([0.0, 0.0], dtype=np.float32),
            sample_loss_weights=np.asarray([10.0, 1.0], dtype=np.float32),
        )
        fake_minibatch = SimpleNamespace(
            observation_batch=SimpleNamespace(),
            critic_batch=SimpleNamespace(),
            action_mask=SimpleNamespace(),
            action_indices={},
            old_log_probs=torch.as_tensor([[0.0, 0.0]], dtype=torch.float32),
            old_values=torch.as_tensor([[0.0, 0.0]], dtype=torch.float32),
            returns=torch.as_tensor([[0.0, 0.0]], dtype=torch.float32),
            advantages=torch.as_tensor([[1.0, 3.0]], dtype=torch.float32),
            sample_loss_weights=torch.as_tensor([[10.0, 1.0]], dtype=torch.float32),
            valid_timestep_mask=torch.as_tensor([[True, True]], dtype=torch.bool),
            lstm_state_in=None,
            sequence_count=1,
            padded_timesteps=0,
        )
        value_param = torch.nn.Parameter(torch.as_tensor(0.0, dtype=torch.float32))
        fake_policy_out = SimpleNamespace(
            value=torch.stack((value_param * 0.0, value_param + 2.0)).reshape(1, 2)
        )
        fake_model = SimpleNamespace(
            forward_sequence=mock.Mock(return_value=(fake_policy_out, None)),
            evaluate_actions=mock.Mock(
                return_value=(
                    torch.as_tensor([0.0, 0.0], dtype=torch.float32),
                    torch.as_tensor([1.0, 3.0], dtype=torch.float32),
                )
            ),
            parameters=mock.Mock(return_value=[value_param]),
        )
        fake_optimizer = mock.Mock()
        batch_view = SimpleNamespace(rewards=np.asarray([0.0, 0.0], dtype=np.float32))
        with mock.patch(
            "backend.training.train_cmrappo._materialize_recurrent_sequences",
            return_value=[fake_sequence],
        ):
            with mock.patch(
                "backend.training.train_cmrappo._build_sequence_minibatch",
                return_value=fake_minibatch,
            ):
                with mock.patch(
                    "backend.training.train_cmrappo._flatten_sequence_policy_output",
                    return_value=SimpleNamespace(),
                ):
                    with mock.patch(
                        "backend.training.train_cmrappo._flatten_sequence_action_mask",
                        return_value=SimpleNamespace(),
                    ):
                        with mock.patch(
                            "backend.training.train_cmrappo._flatten_sequence_action_indices",
                            return_value={},
                        ):
                            stats = _ppo_update(
                                model=fake_model,
                                optimizer=fake_optimizer,
                                batch_view=batch_view,
                                train_cfg=train_cfg,
                                device=torch.device("cpu"),
                                tail_bootstrap_values={},
                            )

        self.assertAlmostEqual(float(stats["policy_loss"]), -13.0 / 11.0, places=6)
        self.assertAlmostEqual(float(stats["value_loss"]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["entropy"]), 2.0, places=6)
        self.assertAlmostEqual(float(stats["sample_loss_weight_mean"]), 5.5, places=6)
        fake_optimizer.step.assert_called_once()


    def test_ppo_update_skips_optimizer_step_when_target_kl_exceeded(self) -> None:
        train_cfg = self._build_training_config()
        train_cfg = _TrainingConfig(
            **{
                **train_cfg.__dict__,
                "ppo_epochs": 1,
                "sequence_minibatch_size": 1,
                "normalize_advantage": False,
                "target_kl": 0.1,
            }
        )
        fake_sequence = SimpleNamespace(
            valid_timestep_mask=np.asarray([True], dtype=np.bool_),
            transitions=(SimpleNamespace(),),
            advantages=np.asarray([1.0], dtype=np.float32),
            returns=np.asarray([1.0], dtype=np.float32),
        )
        fake_minibatch = SimpleNamespace(
            observation_batch=SimpleNamespace(),
            critic_batch=SimpleNamespace(),
            action_mask=SimpleNamespace(),
            action_indices={},
            old_log_probs=torch.as_tensor([[0.0]], dtype=torch.float32),
            old_values=torch.as_tensor([[0.0]], dtype=torch.float32),
            returns=torch.as_tensor([[1.0]], dtype=torch.float32),
            advantages=torch.as_tensor([[1.0]], dtype=torch.float32),
            valid_timestep_mask=torch.as_tensor([[True]], dtype=torch.bool),
            lstm_state_in=None,
            sequence_count=1,
            padded_timesteps=0,
        )
        fake_policy_out = SimpleNamespace(value=torch.as_tensor([[1.0]], dtype=torch.float32))
        fake_model = SimpleNamespace(
            forward_sequence=mock.Mock(return_value=(fake_policy_out, None)),
            evaluate_actions=mock.Mock(
                return_value=(
                    torch.as_tensor([-1.0], dtype=torch.float32),
                    torch.as_tensor([0.2], dtype=torch.float32),
                )
            ),
        )
        fake_optimizer = mock.Mock()
        batch_view = SimpleNamespace(rewards=np.asarray([0.0], dtype=np.float32))
        with mock.patch(
            "backend.training.train_cmrappo._materialize_recurrent_sequences",
            return_value=[fake_sequence],
        ):
            with mock.patch(
                "backend.training.train_cmrappo._build_sequence_minibatch",
                return_value=fake_minibatch,
            ):
                with mock.patch(
                    "backend.training.train_cmrappo._flatten_sequence_policy_output",
                    return_value=SimpleNamespace(),
                ):
                    with mock.patch(
                        "backend.training.train_cmrappo._flatten_sequence_action_mask",
                        return_value=SimpleNamespace(),
                    ):
                        with mock.patch(
                            "backend.training.train_cmrappo._flatten_sequence_action_indices",
                            return_value={},
                        ):
                            stats = _ppo_update(
                                model=fake_model,
                                optimizer=fake_optimizer,
                                batch_view=batch_view,
                                train_cfg=train_cfg,
                                device=torch.device("cpu"),
                                tail_bootstrap_values={},
                            )
        self.assertEqual(
            set(fake_model.evaluate_actions.call_args.kwargs),
            {"policy_out", "action_mask", "action_indices"},
        )
        fake_optimizer.step.assert_not_called()
        self.assertAlmostEqual(float(stats["approx_kl"]), 1.0, places=6)

    def test_resolve_device_auto_prefers_mps_after_cuda(self) -> None:
        if not hasattr(torch.backends, "mps"):
            self.skipTest("当前 torch 构建不包含 MPS backend")
        with mock.patch("backend.training.train_cmrappo.torch.cuda.is_available", return_value=False):
            with mock.patch("backend.training.train_cmrappo.torch.backends.mps.is_available", return_value=True):
                device = _resolve_device("auto")
        self.assertEqual(device.type, "mps")

    def test_set_training_seed_replays_python_numpy_and_torch_streams(self) -> None:
        _set_training_seed(20260427)
        python_seq_1 = [random.random() for _ in range(3)]
        numpy_seq_1 = np.random.rand(3)
        torch_seq_1 = torch.rand(3)

        _set_training_seed(20260427)
        python_seq_2 = [random.random() for _ in range(3)]
        numpy_seq_2 = np.random.rand(3)
        torch_seq_2 = torch.rand(3)

        self.assertEqual(python_seq_1, python_seq_2)
        self.assertTrue(np.allclose(numpy_seq_1, numpy_seq_2))
        self.assertTrue(torch.allclose(torch_seq_1, torch_seq_2))

    def test_finalize_episode_metrics_merges_runtime_snapshot(self) -> None:
        payload = _finalize_episode_metrics(
            accumulator=SimpleNamespace(
                phase="train",
                episode_id=7,
                order_source_mode="poisson",
                order_source_seed=20260424,
                train_arrival_band="high",
                train_arrival_rate_per_min=0.6,
                total_reward=12.5,
                decision_count=9,
                global_step_start=128,
                update_start=3,
            ),
            episode_snapshot={
                "episode_end_t_sec": 3600.0,
                "delivery_count": 4,
                "fallback_count": 1,
                "hard_failure_count": 0,
                "reservation_timeout_count": 2,
            },
            global_step_end=137,
            update_idx_end=4,
        )
        self.assertEqual(payload["phase"], "train")
        self.assertEqual(payload["episode_id"], 7)
        self.assertEqual(payload["order_source_mode"], "poisson")
        self.assertEqual(payload["train_arrival_band"], "high")
        self.assertEqual(payload["train_arrival_rate_per_min"], 0.6)
        self.assertEqual(payload["episode_length_decisions"], 9)
        self.assertEqual(payload["delivery_count"], 4)
        self.assertEqual(payload["global_step_end"], 137)
        self.assertEqual(payload["update_end"], 4)

    def test_build_training_episode_order_source_samples_stochastic_band(self) -> None:
        base_order_source = SimpleNamespace(seed=20260424)
        sampled_order_source = SimpleNamespace(seed=20260424, arrival_rate=0.6)

        with mock.patch(
            "backend.training.train_cmrappo.random.choice",
            return_value=("high", 0.6),
        ):
            with mock.patch(
                "backend.training.train_cmrappo.build_order_source",
                return_value=sampled_order_source,
            ) as build_order_source_mock:
                band_name, arrival_rate, order_source = _build_training_episode_order_source(
                    scene_ctx=SimpleNamespace(),
                    config_path=Path("dummy.yaml"),
                    eval_cfg=SimpleNamespace(
                        stochastic_arrival_rates=(
                            ("low", 0.2),
                            ("medium", 0.4),
                            ("high", 0.6),
                        )
                    ),
                    base_order_source=base_order_source,
                )

        self.assertEqual(band_name, "high")
        self.assertEqual(arrival_rate, 0.6)
        self.assertIs(order_source, sampled_order_source)
        self.assertEqual(build_order_source_mock.call_args.kwargs["seed"], 20260424)
        self.assertEqual(
            build_order_source_mock.call_args.kwargs["overrides"],
            {"poisson_arrival_rate": 0.6},
        )

    def test_sample_training_arrival_band_curriculum_delays_high_load(self) -> None:
        eval_cfg = SimpleNamespace(
            stochastic_arrival_rates=(("low", 0.2), ("medium", 0.4), ("high", 0.6)),
            train_arrival_curriculum_enabled=True,
            train_arrival_base_rate=0.35,
            train_arrival_high_start_update=40,
            train_arrival_high_full_update=120,
            train_arrival_high_max_probability=0.25,
        )

        with mock.patch("backend.training.train_cmrappo.random.random", return_value=0.1):
            band_name, arrival_rate = _sample_training_arrival_band(
                eval_cfg,
                update_idx=0,
            )
        self.assertEqual(band_name, "base")
        self.assertEqual(arrival_rate, 0.35)

        with mock.patch(
            "backend.training.train_cmrappo.random.random",
            side_effect=(0.01, 0.9),
        ):
            band_name, arrival_rate = _sample_training_arrival_band(
                eval_cfg,
                update_idx=120,
            )
        self.assertEqual(band_name, "high")
        self.assertEqual(arrival_rate, 0.6)

    def test_run_periodic_evaluation_collapses_benchmark_to_single_effective_episode(self) -> None:
        fake_model = mock.Mock()
        fake_model.training = True
        fake_order_source = SimpleNamespace(mode=SimpleNamespace(value="benchmark"), seed=20260425)
        fake_benchmark_episode = {
            "total_reward": 10.0,
            "episode_length_decisions": 5,
            "episode_end_t_sec": 120.0,
            "delivery_count": 2,
            "on_time_rate": 1.0,
            "timeout_order_count": 0,
            "fallback_count": 0,
            "hard_failure_count": 0,
            "reservation_timeout_count": 0,
            "hard_overdue_count": 0,
            "t_wait_sec": 0.0,
            "t_idle_sec": 0.0,
            "t_queue_sec": 0.0,
            "t_fallback_sec": 0.0,
            "t_overdue_sec": 0.0,
            "t_reservation_timeout_cost_sec": 0.0,
            "dispatch_decision_count": 1,
            "dispatch_decision_with_legal_mode_c_count": 0,
            "dispatch_mode_b_count": 1,
            "dispatch_mode_c_count": 0,
            "mode_c_selected_count": 0,
            "mode_c_success_count": 0,
            "mode_c_post_delivery_revalidation_fail_count": 0,
            "mode_c_post_delivery_revalidation_fail_reasons": {
                "energy_feasible": 0,
                "rendezvous_time_feasible": 0,
                "node_still_valid": 0,
            },
            "feasible_mode_c_recover_node_count_total": 0,
            "avg_feasible_mode_c_nodes_per_dispatch_decision": 0.0,
            "order_source_seed": 20260425,
        }
        fake_stochastic_episode = {
            **fake_benchmark_episode,
            "order_source_seed": 20260501,
        }

        with mock.patch(
            "backend.training.train_cmrappo.build_order_source",
            side_effect=[
                fake_order_source,
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
            ],
        ) as build_order_source_mock:
            with mock.patch(
                "backend.training.train_cmrappo._evaluate_policy_episode",
                side_effect=[
                    fake_benchmark_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                    fake_stochastic_episode,
                ],
            ) as eval_episode_mock:
                report = _run_periodic_evaluation(
                    scene_ctx=SimpleNamespace(),
                    config_path=Path("dummy.yaml"),
                    model=fake_model,
                    tensorizer=SimpleNamespace(),
                    critic_builder=SimpleNamespace(),
                    critic_schema=SimpleNamespace(),
                    policy_cfg=SimpleNamespace(hist_len=6),
                    eval_cfg=SimpleNamespace(
                        benchmark_seed=20260425,
                        stochastic_seed_base=20260501,
                        stochastic_arrival_rates=(("low", 0.2), ("medium", 0.4), ("high", 0.6)),
                        c_sensitive_enabled=False,
                        c_sensitive_seed=20260426,
                        c_sensitive_min_legal_mode_c_recovery_nodes=1,
                    ),
                    train_cfg=self._build_training_config(),
                    device=torch.device("cpu"),
                    update_idx=7,
                    global_step=256,
                )

        self.assertEqual(report["benchmark"]["episode_count"], 1)
        self.assertEqual(report["benchmark"]["benchmark_seed"], 20260425)
        self.assertEqual(report["benchmark"]["benchmark_eval_episodes_configured"], 2)
        self.assertEqual(report["benchmark"]["benchmark_effective_episode_count"], 1)
        self.assertTrue(report["benchmark"]["benchmark_deterministic_replay"])
        self.assertEqual(eval_episode_mock.call_args_list[0].kwargs["episode_id"], 0)
        self.assertEqual(eval_episode_mock.call_count, 10)
        self.assertEqual(build_order_source_mock.call_args_list[0].kwargs["seed"], 20260425)

    def test_run_periodic_evaluation_includes_c_sensitive_split_when_enabled(self) -> None:
        fake_model = mock.Mock()
        fake_model.training = True
        fake_benchmark_episode = {
            "total_reward": 10.0,
            "episode_length_decisions": 5,
            "episode_end_t_sec": 120.0,
            "delivery_count": 2,
            "on_time_rate": 1.0,
            "timeout_order_count": 0,
            "fallback_count": 0,
            "hard_failure_count": 0,
            "reservation_timeout_count": 0,
            "hard_overdue_count": 0,
            "dispatch_decision_count": 1,
            "dispatch_decision_with_legal_mode_c_count": 1,
            "dispatch_mode_b_count": 1,
            "dispatch_mode_c_count": 1,
            "mode_c_selected_count": 1,
            "mode_c_success_count": 1,
            "mode_c_post_delivery_revalidation_fail_count": 0,
            "mode_c_post_delivery_revalidation_fail_reasons": {
                "energy_feasible": 0,
                "rendezvous_time_feasible": 0,
                "node_still_valid": 0,
            },
            "feasible_mode_c_recover_node_count_total": 2,
            "avg_feasible_mode_c_nodes_per_dispatch_decision": 2.0,
            "t_wait_sec": 0.0,
            "t_idle_sec": 0.0,
            "t_queue_sec": 0.0,
            "t_fallback_sec": 0.0,
            "t_overdue_sec": 0.0,
            "t_reservation_timeout_cost_sec": 0.0,
            "order_source_seed": 20260425,
        }
        fake_stochastic_episode = {
            **fake_benchmark_episode,
            "dispatch_mode_b_count": 2,
            "dispatch_mode_c_count": 0,
            "mode_c_selected_count": 0,
            "mode_c_success_count": 0,
            "dispatch_decision_with_legal_mode_c_count": 0,
            "feasible_mode_c_recover_node_count_total": 0,
            "avg_feasible_mode_c_nodes_per_dispatch_decision": 0.0,
            "order_source_seed": 20260501,
        }
        fake_c_sensitive_order_source = SimpleNamespace(
            mode=SimpleNamespace(value="benchmark"),
            seed=20260426,
        )
        with mock.patch(
            "backend.training.train_cmrappo.build_order_source",
            side_effect=[
                SimpleNamespace(mode=SimpleNamespace(value="benchmark"), seed=20260425),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260501),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260502),
                SimpleNamespace(mode=SimpleNamespace(value="poisson"), seed=20260503),
            ],
        ):
            with mock.patch(
                "backend.training.train_cmrappo._build_c_sensitive_eval_order_source",
                return_value=(
                    fake_c_sensitive_order_source,
                    {
                        "selected_dynamic_order_count": 2,
                        "selected_dynamic_order_ids": ("DYN-01", "DYN-02"),
                        "selected_dynamic_order_legal_mode_c_node_counts": {
                            "DYN-01": 2,
                            "DYN-02": 1,
                        },
                        "c_sensitive_min_legal_mode_c_recovery_nodes": 1,
                        "selection_drone_id": "DRN-TEST-01",
                    },
                ),
            ):
                with mock.patch(
                    "backend.training.train_cmrappo._evaluate_policy_episode",
                    side_effect=[
                        fake_benchmark_episode,
                        fake_benchmark_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                        fake_stochastic_episode,
                    ],
                ) as eval_episode_mock:
                    report = _run_periodic_evaluation(
                        scene_ctx=SimpleNamespace(),
                        config_path=Path("dummy.yaml"),
                        model=fake_model,
                        tensorizer=SimpleNamespace(),
                        critic_builder=SimpleNamespace(),
                        critic_schema=SimpleNamespace(),
                        policy_cfg=SimpleNamespace(hist_len=6),
                        eval_cfg=SimpleNamespace(
                            benchmark_seed=20260425,
                            stochastic_seed_base=20260501,
                            stochastic_arrival_rates=(
                                ("low", 0.2),
                                ("medium", 0.4),
                                ("high", 0.6),
                            ),
                            c_sensitive_enabled=True,
                            c_sensitive_seed=20260426,
                            c_sensitive_min_legal_mode_c_recovery_nodes=1,
                        ),
                        train_cfg=self._build_training_config(),
                        device=torch.device("cpu"),
                        update_idx=7,
                        global_step=256,
                    )

        self.assertIn("c_sensitive", report)
        self.assertEqual(report["c_sensitive"]["c_sensitive_seed"], 20260426)
        self.assertEqual(report["c_sensitive"]["selected_dynamic_order_count"], 2)
        self.assertEqual(
            report["c_sensitive"]["selected_dynamic_order_ids"],
            ("DYN-01", "DYN-02"),
        )
        self.assertEqual(eval_episode_mock.call_count, 11)

    def test_summarize_episode_records_aggregates_counts_and_ratios(self) -> None:
        report = _summarize_episode_records(
            split="benchmark",
            order_source_mode="benchmark",
            episodes=[
                {
                    "total_reward": 10.0,
                    "episode_length_decisions": 5,
                    "episode_length_decision_batches": 2,
                    "episode_end_t_sec": 120.0,
                    "delivery_count": 2,
                    "on_time_rate": 1.0,
                    "timeout_order_count": 0,
                    "fallback_count": 1,
                    "hard_failure_count": 0,
                    "reservation_timeout_count": 0,
                    "hard_overdue_count": 0,
                    "dispatch_decision_count": 3,
                    "dispatch_decision_with_legal_mode_c_count": 1,
                    "t_wait_sec": 3.0,
                    "t_idle_sec": 1.0,
                    "t_queue_sec": 2.0,
                    "t_fallback_sec": 4.0,
                    "t_ppo_attributed_fallback_sec": 4.0,
                    "t_system_attributed_fallback_sec": 0.0,
                    "t_overdue_sec": 0.0,
                    "t_reservation_timeout_cost_sec": 0.0,
                    "ppo_attributed_fallback_count": 1,
                    "system_attributed_fallback_count": 0,
                    "fallback_cause_counts": {
                        "rendezvous_wait_timeout": 1,
                    },
                    "reservation_release_cause_counts": {
                        "rendezvous_wait_timeout": 1,
                    },
                    "dispatch_mode_b_count": 3,
                    "dispatch_mode_c_count": 1,
                    "mode_c_selected_count": 1,
                    "mode_c_success_count": 1,
                    "mode_c_post_delivery_revalidation_fail_count": 0,
                    "mode_c_post_delivery_revalidation_fail_reasons": {
                        "energy_feasible": 0,
                        "rendezvous_time_feasible": 0,
                        "node_still_valid": 0,
                    },
                    "feasible_mode_c_recover_node_count_total": 2,
                },
                {
                    "total_reward": 14.0,
                    "episode_length_decisions": 7,
                    "episode_length_decision_batches": 3,
                    "episode_end_t_sec": 150.0,
                    "delivery_count": 3,
                    "on_time_rate": 0.5,
                    "timeout_order_count": 1,
                    "fallback_count": 2,
                    "hard_failure_count": 1,
                    "reservation_timeout_count": 1,
                    "hard_overdue_count": 1,
                    "dispatch_decision_count": 5,
                    "dispatch_decision_with_legal_mode_c_count": 3,
                    "t_wait_sec": 5.0,
                    "t_idle_sec": 2.0,
                    "t_queue_sec": 1.0,
                    "t_fallback_sec": 6.0,
                    "t_ppo_attributed_fallback_sec": 2.0,
                    "t_system_attributed_fallback_sec": 4.0,
                    "t_overdue_sec": 7.0,
                    "t_reservation_timeout_cost_sec": 8.0,
                    "ppo_attributed_fallback_count": 1,
                    "system_attributed_fallback_count": 1,
                    "fallback_cause_counts": {
                        "no_post_delivery_c_node": 1,
                        "planner_invalidated_for_truck_order": 1,
                    },
                    "reservation_release_cause_counts": {
                        "no_post_delivery_c_node": 1,
                        "planner_invalidated_for_truck_order": 1,
                    },
                    "dispatch_mode_b_count": 1,
                    "dispatch_mode_c_count": 3,
                    "mode_c_selected_count": 3,
                    "mode_c_success_count": 2,
                    "mode_c_post_delivery_revalidation_fail_count": 2,
                    "mode_c_post_delivery_revalidation_fail_reasons": {
                        "energy_feasible": 1,
                        "rendezvous_time_feasible": 2,
                        "node_still_valid": 1,
                    },
                    "feasible_mode_c_recover_node_count_total": 7,
                },
            ],
            update_idx=5,
            global_step=1024,
            generated_at="2026-04-27T10:20:30+00:00",
        )
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["sum_delivery_count"], 5)
        self.assertEqual(report["sum_fallback_count"], 3)
        self.assertEqual(report["sum_hard_failure_count"], 1)
        self.assertEqual(report["sum_dispatch_decision_count"], 8)
        self.assertEqual(report["sum_dispatch_decision_with_legal_mode_c_count"], 4)
        self.assertEqual(report["sum_feasible_mode_c_recover_node_count_total"], 9)
        self.assertAlmostEqual(report["avg_feasible_mode_c_nodes_per_dispatch_decision"], 1.125)
        self.assertEqual(report["sum_mode_c_selected_count"], 4)
        self.assertEqual(report["sum_mode_c_success_count"], 3)
        self.assertEqual(report["sum_mode_c_post_delivery_revalidation_fail_count"], 2)
        self.assertEqual(
            report["sum_mode_c_post_delivery_revalidation_fail_reasons"],
            {
                "energy_feasible": 1,
                "rendezvous_time_feasible": 2,
                "node_still_valid": 1,
            },
        )
        self.assertEqual(report["sum_ppo_attributed_fallback_count"], 2)
        self.assertEqual(report["sum_system_attributed_fallback_count"], 1)
        self.assertEqual(
            report["fallback_cause_counts"],
            {
                "rendezvous_wait_timeout": 1,
                "no_post_delivery_c_node": 1,
                "planner_invalidated_for_truck_order": 1,
            },
        )
        self.assertEqual(
            report["reservation_release_cause_counts"],
            {
                "rendezvous_wait_timeout": 1,
                "no_post_delivery_c_node": 1,
                "planner_invalidated_for_truck_order": 1,
            },
        )
        self.assertAlmostEqual(report["sum_t_ppo_attributed_fallback_sec"], 6.0)
        self.assertAlmostEqual(report["sum_t_system_attributed_fallback_sec"], 4.0)
        self.assertAlmostEqual(report["mean_total_reward"], 12.0)
        self.assertAlmostEqual(report["mean_episode_length_decision_batches"], 2.5)
        self.assertEqual(report["sum_episode_length_decision_batches"], 5)
        self.assertAlmostEqual(report["mode_b_dispatch_ratio"], 0.5)
        self.assertAlmostEqual(report["mode_c_dispatch_ratio"], 0.5)
        self.assertEqual(len(report["episodes"]), 2)

    def test_strip_episode_details_removes_verbose_episode_list(self) -> None:
        compact = _strip_episode_details(
            {
                "split": "benchmark",
                "mean_total_reward": 1.0,
                "episodes": [{"episode_id": 0}],
            }
        )
        self.assertEqual(compact["split"], "benchmark")
        self.assertEqual(compact["mean_total_reward"], 1.0)
        self.assertNotIn("episodes", compact)

    def test_eval_selection_key_uses_benchmark_as_guardrail_and_stochastic_as_objective(self) -> None:
        baseline = {
            "benchmark": {
                "sum_timeout_order_count": 0,
                "mean_episode_end_t_sec": 900.0,
                "mean_total_reward": -80.0,
                "episodes": [{"done_reason": "all_orders_cleared"}],
            },
            "stochastic": {
                "medium": {"mean_total_reward": 300.0},
                "high": {
                    "sum_timeout_order_count": 4,
                    "mean_total_reward": 500.0,
                    "sum_fallback_count": 0,
                },
            },
        }
        better_stochastic = {
            "benchmark": {
                "sum_timeout_order_count": 0,
                "mean_episode_end_t_sec": 980.0,
                "mean_total_reward": -140.0,
                "episodes": [{"done_reason": "all_orders_cleared"}],
            },
            "stochastic": {
                "medium": {"mean_total_reward": 320.0},
                "high": {
                    "sum_timeout_order_count": 2,
                    "mean_total_reward": 450.0,
                    "sum_fallback_count": 0,
                },
            },
        }
        benchmark_broken = {
            "benchmark": {
                "sum_timeout_order_count": 1,
                "mean_episode_end_t_sec": 850.0,
                "mean_total_reward": 10.0,
                "episodes": [{"done_reason": "timeout"}],
            },
            "stochastic": {
                "medium": {"mean_total_reward": 340.0},
                "high": {
                    "sum_timeout_order_count": 1,
                    "mean_total_reward": 650.0,
                    "sum_fallback_count": 0,
                },
            },
        }
        self.assertGreater(
            _build_eval_selection_key(better_stochastic),
            _build_eval_selection_key(baseline),
        )
        self.assertGreater(
            _build_eval_selection_key(baseline),
            _build_eval_selection_key(benchmark_broken),
        )
        self.assertEqual(
            _build_benchmark_guardrail_key(better_stochastic),
            _build_benchmark_guardrail_key(baseline),
        )
        self.assertGreater(
            _build_stochastic_high_improvement_key(better_stochastic),
            _build_stochastic_high_improvement_key(baseline),
        )

    def test_should_stop_early_depends_on_stochastic_patience_and_value_loss(self) -> None:
        cfg = self._build_training_config()
        self.assertFalse(
            _should_stop_early(
                train_cfg=cfg,
                eval_count=2,
                stochastic_high_no_improve_evals=3,
                recent_eval_value_losses=(10.0, 9.0, 8.0),
            )
        )
        self.assertFalse(
            _should_stop_early(
                train_cfg=cfg,
                eval_count=3,
                stochastic_high_no_improve_evals=2,
                recent_eval_value_losses=(10.0, 9.0, 9.5),
            )
        )
        self.assertFalse(
            _should_stop_early(
                train_cfg=cfg,
                eval_count=3,
                stochastic_high_no_improve_evals=3,
                recent_eval_value_losses=(10.0, 9.0, 8.0),
            )
        )
        self.assertTrue(
            _should_stop_early(
                train_cfg=cfg,
                eval_count=3,
                stochastic_high_no_improve_evals=3,
                recent_eval_value_losses=(10.0, 9.0, 9.5),
            )
        )
        self.assertTrue(
            _latest_value_loss_shows_meaningful_decline(
                recent_eval_value_losses=(10.0, 9.0, 8.0),
                min_delta=0.0,
            )
        )

    def test_validate_meta_payload_before_write_accepts_consistent_runtime_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            policy_path = tmp_dir / "policy.pt"
            policy_path.write_bytes(b"policy")
            config_snapshot_path = tmp_dir / "training_config_snapshot.yaml"
            config_snapshot_path.write_text("training: {}\n", encoding="utf-8")
            metrics_path = tmp_dir / "train_metrics.jsonl"
            metrics_path.write_text("{\"update\": 1}\n", encoding="utf-8")

            scene_ctx = SimpleNamespace(
                scene_id="scene_a",
                scene_bundle_dir="/tmp/scene_bundle",
            )
            benchmark = SimpleNamespace(
                orders_json="backend/data/orders.json",
                orders_json_sha256="abc123",
                static_order_count=10,
                dynamic_order_count=5,
                benchmark_use_dynamic_orders=True,
            )
            training_input_meta = SimpleNamespace(
                poisson_arrival_rate=2.5,
                poisson_weight_max_kg=10.0,
                order_window_min_min=20,
                order_window_max_min=60,
            )
            order_source = SimpleNamespace(
                mode=SimpleNamespace(value="poisson"),
                seed=314159,
                benchmark=benchmark,
                training_input_meta=training_input_meta,
            )
            critic_schema = SimpleNamespace(schema_hash="schema_hash_v1")
            train_cfg = self._build_training_config()
            meta_payload = {
                "model_version": "cmrappo_test",
                "trained_at": "2026-04-27T10:20:30+00:00",
                "scene_id": "scene_a",
                "scene_bundle_dir": "/tmp/scene_bundle",
                "training_input": {
                    "order_source_mode": "poisson",
                    "benchmark": {
                        "orders_json": "backend/data/orders.json",
                        "orders_json_sha256": "abc123",
                        "static_order_count": 10,
                        "dynamic_order_count": 5,
                        "benchmark_use_dynamic_orders": True,
                    },
                    "poisson_arrival_rate": 2.5,
                    "poisson_weight_max_kg": 10.0,
                    "order_window_min_min": 20,
                    "order_window_max_min": 60,
                    "poisson_seed": 314159,
                    "training_seed": 2026,
                    "total_timesteps": 128,
                },
                "policy": {"encoder_type": "mlp"},
                "action_space": {"type": "factorized"},
                "candidate": {"max_candidate_orders": 32},
                "planner": {"coarse_replan_interval_sec": 60.0},
                "reward": {"wait_idle_penalty_coef": 0.03},
                "critic_schema": {
                    "name": "critic_tensor_v1",
                    "schema_version": "v1",
                    "schema_hash": "schema_hash_v1",
                },
                "shared_runtime_params_snapshot": {
                    "source_config": "backend/config/drone_params.yaml",
                    "light_drone": {"k1": 1.0},
                    "heavy_drone": {"k1": 2.0},
                    "solver_energy": {"lambda_time": 1.0},
                },
                "env_semantic_contract": {"fifo_queue_enabled": True},
                "online_lock_params": {"locked_fields": [], "tunable_fields": []},
            }

            _validate_meta_payload_before_write(
                meta_payload=meta_payload,
                scene_ctx=scene_ctx,
                order_source=order_source,
                critic_schema=critic_schema,
                train_cfg=train_cfg,
                model_version="cmrappo_test",
                global_step=128,
                policy_path=policy_path,
                config_snapshot_path=config_snapshot_path,
                metrics_path=metrics_path,
            )

    def test_record_terminal_episode_rewards_closes_tail_reward_gap(self) -> None:
        accumulator = _EpisodeAccumulator(
            phase="benchmark",
            episode_id=0,
            order_source_mode="benchmark",
            order_source_seed=20260425,
            decision_count=16,
            total_reward=-79.4354995,
        )

        terminal_reward_total = _record_terminal_episode_rewards(
            accumulator=accumulator,
            terminal_reward_by_drone={
                "UAV-TEST-03": 100.0,
                "UAV-TEST-10": 100.0,
                "UAV-TEST-12": 100.0,
            },
        )

        self.assertAlmostEqual(terminal_reward_total, 300.0)
        self.assertEqual(accumulator.decision_count, 16)
        self.assertAlmostEqual(accumulator.total_reward, 220.5645005)

    def test_episode_reward_recording_uses_reward_scale(self) -> None:
        accumulator = _EpisodeAccumulator(
            phase="train",
            episode_id=0,
            order_source_mode="poisson",
            order_source_seed=20260425,
            total_reward=-0.5,
        )

        _record_episode_step(accumulator, reward=100.0, reward_scale=0.01)
        terminal_reward_total = _record_terminal_episode_rewards(
            accumulator=accumulator,
            terminal_reward_by_drone={
                "UAV-TEST-03": 100.0,
                "UAV-TEST-10": 50.0,
            },
            reward_scale=0.01,
        )

        self.assertAlmostEqual(terminal_reward_total, 1.5)
        self.assertEqual(accumulator.decision_count, 1)
        self.assertAlmostEqual(accumulator.total_reward, 2.0)

    def test_validate_meta_payload_before_write_rejects_mismatched_effective_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            policy_path = tmp_dir / "policy.pt"
            policy_path.write_bytes(b"policy")
            config_snapshot_path = tmp_dir / "training_config_snapshot.yaml"
            config_snapshot_path.write_text("training: {}\n", encoding="utf-8")
            metrics_path = tmp_dir / "train_metrics.jsonl"
            metrics_path.write_text("{\"update\": 1}\n", encoding="utf-8")

            scene_ctx = SimpleNamespace(scene_id="scene_a", scene_bundle_dir="/tmp/scene_bundle")
            order_source = SimpleNamespace(
                mode=SimpleNamespace(value="poisson"),
                seed=1,
                benchmark=SimpleNamespace(
                    orders_json="o",
                    orders_json_sha256="h",
                    static_order_count=1,
                    dynamic_order_count=1,
                    benchmark_use_dynamic_orders=True,
                ),
                training_input_meta=SimpleNamespace(
                    poisson_arrival_rate=1.0,
                    poisson_weight_max_kg=10.0,
                    order_window_min_min=20,
                    order_window_max_min=60,
                ),
            )
            critic_schema = SimpleNamespace(schema_hash="schema_hash_v1")
            train_cfg = self._build_training_config()
            meta_payload = {
                "model_version": "cmrappo_test",
                "trained_at": "2026-04-27T10:20:30+00:00",
                "scene_id": "scene_a",
                "scene_bundle_dir": "/tmp/scene_bundle",
                "training_input": {
                    "order_source_mode": "poisson",
                    "benchmark": {
                        "orders_json": "o",
                        "orders_json_sha256": "h",
                        "static_order_count": 1,
                        "dynamic_order_count": 1,
                        "benchmark_use_dynamic_orders": True,
                    },
                    "poisson_arrival_rate": 1.0,
                    "poisson_weight_max_kg": 10.0,
                    "order_window_min_min": 20,
                    "order_window_max_min": 60,
                    "poisson_seed": 1,
                    "training_seed": 2026,
                    "total_timesteps": 128,
                },
                "policy": {"encoder_type": "mlp"},
                "action_space": {"type": "factorized"},
                "candidate": {"max_candidate_orders": 32},
                "planner": {"coarse_replan_interval_sec": 60.0},
                "reward": {"wait_idle_penalty_coef": 0.03},
                "critic_schema": {
                    "name": "critic_tensor_v1",
                    "schema_version": "v1",
                    "schema_hash": "schema_hash_v1",
                },
                "shared_runtime_params_snapshot": {
                    "source_config": "backend/config/drone_params.yaml",
                    "light_drone": {"k1": 1.0},
                    "heavy_drone": {"k1": 2.0},
                    "solver_energy": {"lambda_time": 1.0},
                },
                "env_semantic_contract": {"fifo_queue_enabled": True},
                "online_lock_params": {"locked_fields": [], "tunable_fields": []},
            }

            with self.assertRaises(ValueError):
                _validate_meta_payload_before_write(
                    meta_payload=meta_payload,
                    scene_ctx=scene_ctx,
                    order_source=order_source,
                    critic_schema=critic_schema,
                    train_cfg=train_cfg,
                    model_version="cmrappo_test",
                    global_step=127,
                    policy_path=policy_path,
                    config_snapshot_path=config_snapshot_path,
                    metrics_path=metrics_path,
                )

    def test_validate_meta_payload_before_write_accepts_early_stopped_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            policy_path = tmp_dir / "policy.pt"
            policy_path.write_bytes(b"policy")
            config_snapshot_path = tmp_dir / "training_config_snapshot.yaml"
            config_snapshot_path.write_text("training: {}\n", encoding="utf-8")
            metrics_path = tmp_dir / "train_metrics.jsonl"
            metrics_path.write_text("{\"update\": 1}\n", encoding="utf-8")

            scene_ctx = SimpleNamespace(scene_id="scene_a", scene_bundle_dir="/tmp/scene_bundle")
            order_source = SimpleNamespace(
                mode=SimpleNamespace(value="poisson"),
                seed=1,
                benchmark=SimpleNamespace(
                    orders_json="o",
                    orders_json_sha256="h",
                    static_order_count=1,
                    dynamic_order_count=1,
                    benchmark_use_dynamic_orders=True,
                ),
                training_input_meta=SimpleNamespace(
                    poisson_arrival_rate=1.0,
                    poisson_weight_max_kg=10.0,
                    order_window_min_min=20,
                    order_window_max_min=60,
                ),
            )
            critic_schema = SimpleNamespace(schema_hash="schema_hash_v1")
            train_cfg = self._build_training_config()
            meta_payload = {
                "model_version": "cmrappo_test",
                "trained_at": "2026-04-27T10:20:30+00:00",
                "scene_id": "scene_a",
                "scene_bundle_dir": "/tmp/scene_bundle",
                "training_input": {
                    "order_source_mode": "poisson",
                    "benchmark": {
                        "orders_json": "o",
                        "orders_json_sha256": "h",
                        "static_order_count": 1,
                        "dynamic_order_count": 1,
                        "benchmark_use_dynamic_orders": True,
                    },
                    "poisson_arrival_rate": 1.0,
                    "poisson_weight_max_kg": 10.0,
                    "order_window_min_min": 20,
                    "order_window_max_min": 60,
                    "poisson_seed": 1,
                    "training_seed": 2026,
                    "total_timesteps": 96,
                },
                "policy": {"encoder_type": "mlp"},
                "action_space": {"type": "factorized"},
                "candidate": {"max_candidate_orders": 32},
                "planner": {"coarse_replan_interval_sec": 60.0},
                "reward": {"wait_idle_penalty_coef": 0.03},
                "critic_schema": {
                    "name": "critic_tensor_v1",
                    "schema_version": "v1",
                    "schema_hash": "schema_hash_v1",
                },
                "shared_runtime_params_snapshot": {
                    "source_config": "backend/config/drone_params.yaml",
                    "light_drone": {"k1": 1.0},
                    "heavy_drone": {"k1": 2.0},
                    "solver_energy": {"lambda_time": 1.0},
                },
                "env_semantic_contract": {"fifo_queue_enabled": True},
                "online_lock_params": {"locked_fields": [], "tunable_fields": []},
            }

            _validate_meta_payload_before_write(
                meta_payload=meta_payload,
                scene_ctx=scene_ctx,
                order_source=order_source,
                critic_schema=critic_schema,
                train_cfg=train_cfg,
                model_version="cmrappo_test",
                global_step=96,
                policy_path=policy_path,
                config_snapshot_path=config_snapshot_path,
                metrics_path=metrics_path,
            )


if __name__ == "__main__":
    unittest.main()
