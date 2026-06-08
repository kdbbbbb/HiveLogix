#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 snapshot / tensorizer / critic buffer tests.

运行方式：
  python -m unittest backend.training.test_phase7_snapshot_and_tensorizers
"""

from __future__ import annotations

import unittest

import numpy as np

from .critic_batch_builder import CriticBatchBuilder
from .contracts import ResolvedActionIndices, TransitionSummary
from .env_adapter import TrainingEnvAdapter
from .observation_tensorizer import (
    HISTORY_TOKEN_FIELDS,
    ORDER_TOKEN_FIELDS,
    ObservationTensorizer,
)
from .rollout_buffer import RolloutBuffer, compute_gae


class TestPhase7SnapshotAndTensorizers(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TrainingEnvAdapter()
        self.reset_result = self.env.reset()
        self.assertIsNotNone(self.reset_result.decision_context)
        self.decision = self.reset_result.decision_context
        self.tensorizer = ObservationTensorizer()
        self.critic_builder = CriticBatchBuilder()

    def test_decision_context_exposes_phase7_pre_action_snapshot(self) -> None:
        planner_snapshot = self.decision.planner_snapshot
        execution_snapshot = self.decision.execution_snapshot

        self.assertEqual(self.decision.t_decision, self.decision.runtime_state.t_now)
        self.assertIn(self.decision.deciding_drone_id, execution_snapshot.uav_eta_to_available)
        self.assertIn(self.decision.deciding_drone_id, execution_snapshot.uav_dispatch_mode)
        self.assertGreaterEqual(planner_snapshot.backlog_new_orders, 0)
        self.assertGreaterEqual(planner_snapshot.fallback_count_in_window, 0)
        self.assertGreaterEqual(planner_snapshot.hard_failure_count_in_window, 0)
        self.assertGreaterEqual(planner_snapshot.total_backbone_count, 0)
        self.assertIsInstance(planner_snapshot.active_launch_stations, tuple)

    def test_observation_tensorizer_builds_fixed_shape_batches(self) -> None:
        candidate_out = self.env.build_candidate_output(
            self.decision,
            last_seen_plan_version=self.decision.coarse_plan.plan_version,
        )
        batch = self.tensorizer.build(
            decision_context=self.decision,
            candidate_out=candidate_out,
            transition_history=(),
        )
        action_mask = self.tensorizer.build_action_mask(candidate_out)

        self.assertEqual(batch.order_tokens.shape[0], len(candidate_out.order_mask))
        self.assertEqual(batch.order_tokens.shape[1], len(ORDER_TOKEN_FIELDS))
        self.assertEqual(batch.history_tokens.shape[0], self.tensorizer.history_length)
        self.assertEqual(batch.history_padding_mask.dtype, np.bool_)
        self.assertEqual(action_mask.root_branch_mask.shape, (2,))
        self.assertEqual(action_mask.mode_mask.shape[0], len(candidate_out.mode_mask))
        self.assertIn("deadline_remaining_time_norm", ORDER_TOKEN_FIELDS)
        self.assertIn("estimated_delivery_finish_slack_norm", ORDER_TOKEN_FIELDS)
        self.assertIn("best_mode_b_recovery_flight_time_norm", ORDER_TOKEN_FIELDS)
        self.assertIn("local_teacher_order_cost_norm", ORDER_TOKEN_FIELDS)

    def test_history_trigger_types_use_distinct_codes_for_runtime_resume_paths(self) -> None:
        candidate_out = self.env.build_candidate_output(
            self.decision,
            last_seen_plan_version=self.decision.coarse_plan.plan_version,
        )
        trigger_field_idx = HISTORY_TOKEN_FIELDS.index("trigger_type_code_norm")
        history = (
            TransitionSummary(
                event_time=1.0,
                actor_drone_id=str(self.decision.deciding_drone_id),
                actor_pos_x=0.0,
                actor_pos_y=0.0,
                actor_training_state_before="idle",
                actor_training_state_after="idle",
                actor_home_type="depot",
                actor_payload_class="light",
                trigger_type="initial_idle",
                root_branch="WAIT",
                dispatch_mode="NONE",
                selected_recover_node_type="none",
                has_selected_order=False,
                selected_order_slot_rank=-1,
                selected_order_deadline_slack_norm=0.0,
                selected_eta_to_deliver_norm=0.0,
                selected_rendezvous_margin_norm=0.0,
                energy_ratio_before=1.0,
                energy_ratio_after=1.0,
                queue_after_norm=0.0,
                plan_version_delta_at_event=0,
                delivered=False,
                rendezvous_success=False,
                reservation_timeout=False,
                fallback_started=False,
                hard_failure=False,
                queue_entered=False,
                service_completed=False,
            ),
            TransitionSummary(
                event_time=2.0,
                actor_drone_id=str(self.decision.deciding_drone_id),
                actor_pos_x=0.0,
                actor_pos_y=0.0,
                actor_training_state_before="idle",
                actor_training_state_after="idle",
                actor_home_type="depot",
                actor_payload_class="light",
                trigger_type="idle_ready",
                root_branch="WAIT",
                dispatch_mode="NONE",
                selected_recover_node_type="none",
                has_selected_order=False,
                selected_order_slot_rank=-1,
                selected_order_deadline_slack_norm=0.0,
                selected_eta_to_deliver_norm=0.0,
                selected_rendezvous_margin_norm=0.0,
                energy_ratio_before=1.0,
                energy_ratio_after=1.0,
                queue_after_norm=0.0,
                plan_version_delta_at_event=0,
                delivered=False,
                rendezvous_success=False,
                reservation_timeout=False,
                fallback_started=False,
                hard_failure=False,
                queue_entered=False,
                service_completed=False,
            ),
            TransitionSummary(
                event_time=3.0,
                actor_drone_id=str(self.decision.deciding_drone_id),
                actor_pos_x=0.0,
                actor_pos_y=0.0,
                actor_training_state_before="idle",
                actor_training_state_after="idle",
                actor_home_type="depot",
                actor_payload_class="light",
                trigger_type="wait_resume",
                root_branch="WAIT",
                dispatch_mode="NONE",
                selected_recover_node_type="none",
                has_selected_order=False,
                selected_order_slot_rank=-1,
                selected_order_deadline_slack_norm=0.0,
                selected_eta_to_deliver_norm=0.0,
                selected_rendezvous_margin_norm=0.0,
                energy_ratio_before=1.0,
                energy_ratio_after=1.0,
                queue_after_norm=0.0,
                plan_version_delta_at_event=0,
                delivered=False,
                rendezvous_success=False,
                reservation_timeout=False,
                fallback_started=False,
                hard_failure=False,
                queue_entered=False,
                service_completed=False,
            ),
            TransitionSummary(
                event_time=4.0,
                actor_drone_id=str(self.decision.deciding_drone_id),
                actor_pos_x=0.0,
                actor_pos_y=0.0,
                actor_training_state_before="idle",
                actor_training_state_after="idle",
                actor_home_type="depot",
                actor_payload_class="light",
                trigger_type="order_arrival_wake",
                root_branch="WAIT",
                dispatch_mode="NONE",
                selected_recover_node_type="none",
                has_selected_order=False,
                selected_order_slot_rank=-1,
                selected_order_deadline_slack_norm=0.0,
                selected_eta_to_deliver_norm=0.0,
                selected_rendezvous_margin_norm=0.0,
                energy_ratio_before=1.0,
                energy_ratio_after=1.0,
                queue_after_norm=0.0,
                plan_version_delta_at_event=0,
                delivered=False,
                rendezvous_success=False,
                reservation_timeout=False,
                fallback_started=False,
                hard_failure=False,
                queue_entered=False,
                service_completed=False,
            ),
        )

        batch = self.tensorizer.build(
            decision_context=self.decision,
            candidate_out=candidate_out,
            transition_history=history,
        )

        trigger_codes = batch.history_tokens[-4:, trigger_field_idx]
        self.assertEqual(len(set(float(item) for item in trigger_codes)), 4)

    def test_history_build_raises_for_unknown_trigger_type(self) -> None:
        candidate_out = self.env.build_candidate_output(
            self.decision,
            last_seen_plan_version=self.decision.coarse_plan.plan_version,
        )
        history = (
            TransitionSummary(
                event_time=1.0,
                actor_drone_id=str(self.decision.deciding_drone_id),
                actor_pos_x=0.0,
                actor_pos_y=0.0,
                actor_training_state_before="idle",
                actor_training_state_after="idle",
                actor_home_type="depot",
                actor_payload_class="light",
                trigger_type="unknown_trigger",
                root_branch="WAIT",
                dispatch_mode="NONE",
                selected_recover_node_type="none",
                has_selected_order=False,
                selected_order_slot_rank=-1,
                selected_order_deadline_slack_norm=0.0,
                selected_eta_to_deliver_norm=0.0,
                selected_rendezvous_margin_norm=0.0,
                energy_ratio_before=1.0,
                energy_ratio_after=1.0,
                queue_after_norm=0.0,
                plan_version_delta_at_event=0,
                delivered=False,
                rendezvous_success=False,
                reservation_timeout=False,
                fallback_started=False,
                hard_failure=False,
                queue_entered=False,
                service_completed=False,
            ),
        )

        with self.assertRaisesRegex(ValueError, "未知 trigger_type"):
            self.tensorizer.build(
                decision_context=self.decision,
                candidate_out=candidate_out,
                transition_history=history,
            )

    def test_critic_batch_builder_respects_schema_capacity(self) -> None:
        schema = self.critic_builder.default_schema_meta
        critic_batch = self.critic_builder.build(
            decision_context=self.decision,
            critic_tensor_schema_meta=schema,
        )

        self.assertEqual(
            critic_batch.global_order_pool_tokens.shape,
            (schema.max_global_orders, len(schema.order_token_fields)),
        )
        self.assertEqual(
            critic_batch.global_uav_tokens.shape,
            (schema.max_global_uavs, len(schema.uav_token_fields)),
        )
        self.assertEqual(
            critic_batch.global_station_tokens.shape,
            (schema.max_global_stations, len(schema.station_token_fields)),
        )
        self.assertEqual(
            critic_batch.coarse_plan_summary_vec.shape,
            (len(schema.coarse_plan_summary_fields),),
        )
        self.assertEqual(
            critic_batch.global_system_summary_vec.shape,
            (len(schema.global_system_summary_fields),),
        )
        self.assertTrue(schema.schema_hash)

    def test_rollout_buffer_two_phase_finalize_and_gae(self) -> None:
        candidate_out = self.env.build_candidate_output(
            self.decision,
            last_seen_plan_version=self.decision.coarse_plan.plan_version,
        )
        observation_batch = self.tensorizer.build(
            decision_context=self.decision,
            candidate_out=candidate_out,
            transition_history=(),
        )
        action_mask = self.tensorizer.build_action_mask(candidate_out)
        critic_batch = self.critic_builder.build(
            decision_context=self.decision,
            critic_tensor_schema_meta=self.critic_builder.default_schema_meta,
        )

        buffer = RolloutBuffer(capacity=2)
        slot = buffer.begin_transition(
            observation_batch=observation_batch,
            critic_batch=critic_batch,
            action_mask=action_mask,
            action_indices=ResolvedActionIndices(root_branch_idx=0),
            log_prob_old=-0.1,
            value_old=1.0,
            critic_schema_hash=self.critic_builder.default_schema_meta.schema_hash,
        )
        buffer.finalize_transition(slot, reward=2.0, done=False)

        self.assertEqual(buffer.size, 1)
        view = buffer.build_batch_view()
        self.assertEqual(view.rewards.shape, (1,))
        self.assertEqual(view.dones.shape, (1,))
        self.assertEqual(view.values.shape, (1,))
        self.assertAlmostEqual(float(view.rewards[0]), 2.0, places=6)
        self.assertFalse(bool(view.dones[0]))
        self.assertAlmostEqual(float(view.values[0]), 1.0, places=6)

        advantages, returns = compute_gae(
            rewards=view.rewards,
            dones=view.dones,
            values=view.values,
            last_value=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertEqual(advantages.shape, (1,))
        self.assertEqual(returns.shape, (1,))
        self.assertAlmostEqual(float(advantages[0]), 1.495, places=6)
        self.assertAlmostEqual(float(returns[0]), 2.495, places=6)

    def test_poisson_env_exposes_actual_episode_order_source_seed(self) -> None:
        self.assertEqual(self.env.current_episode_order_source_seed(), 20260424)

        second_result = self.env.reset()
        self.assertIsNotNone(second_result.decision_context)
        self.assertEqual(self.env.current_episode_order_source_seed(), 20260425)


if __name__ == "__main__":
    unittest.main()
