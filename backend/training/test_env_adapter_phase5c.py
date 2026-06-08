#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5c env adapter tests.

运行方式：
  python -m unittest backend.training.test_env_adapter_phase5c
"""

from __future__ import annotations

import math
import unittest

from core.entities.order import Order
from core.entities.primitives import Position3D, TaskStatus

from .batch_matching_teacher import build_batch_matching_teacher_labels
from .actions import DispatchAction as SharedDispatchAction
from .contracts import PolicyMode
from .env_adapter import (
    BackboneVisit,
    DecisionTrigger,
    DispatchAction,
    FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE,
    TrainingDroneState,
    TrainingEnvAdapter,
    WAIT_ACTION,
)


class TestTrainingEnvAdapterPhase5c(unittest.TestCase):
    def _make_env(self) -> TrainingEnvAdapter:
        return TrainingEnvAdapter()

    def _first_idle_drone_id(self, env: TrainingEnvAdapter) -> str:
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.IDLE:
                return drone_id
        self.fail("默认场景中至少应有一架 depot-home 无人机处于 idle")

    def _second_drone_id(self, env: TrainingEnvAdapter, exclude: str) -> str:
        for drone_id in env._drone_state:
            if drone_id != exclude:
                return drone_id
        self.fail("默认场景中至少应有两架无人机")

    def _first_riding_drone_id(self, env: TrainingEnvAdapter) -> str:
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.RIDING_WITH_TRUCK:
                return drone_id
        self.fail("默认场景中至少应有一架 truck-home 无人机处于 riding_with_truck")

    def _reset_controlled_env(self) -> tuple[TrainingEnvAdapter, str]:
        env = self._make_env()
        env.reset()

        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        order_mgr._next_order_time = math.inf
        order_mgr._scheduled_dynamic = []
        order_mgr._scheduled_dynamic_i = 0

        env._decision_queue.clear()
        env._background_mode_a_pending.clear()
        drone_id = self._first_idle_drone_id(env)
        return env, drone_id

    def _inject_order(
        self,
        env: TrainingEnvAdapter,
        *,
        drone_id: str,
        order_id: str,
        offset_x: float = 10.0,
        create_time: float | None = None,
        deadline: float | None = None,
        payload_weight: float = 1.0,
    ) -> Order:
        drone = env._require_entity_manager().drones[drone_id]
        spawn_time = env._t_now if create_time is None else create_time
        due_time = env._t_now + 3600.0 if deadline is None else deadline
        order = Order(
            order_id=order_id,
            create_time=spawn_time,
            deadline=due_time,
            delivery_loc=Position3D(
                x=drone.current_loc.x + offset_x,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
            payload_weight=payload_weight,
        )
        env._require_order_manager().pending_orders[order.order_id] = order
        return order

    def test_enqueue_decision_records_trigger_enqueue_time(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._t_now = 123.456
        self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-TRIGGER-TIME",
            create_time=120.0,
            deadline=3600.0,
        )

        env._enqueue_decision(drone_id, "test_idle", None)

        self.assertEqual(len(env._decision_queue), 1)
        trigger = env._decision_queue[0]
        self.assertAlmostEqual(trigger.t_enqueued, 123.456)

        env._t_now = 200.0
        self.assertAlmostEqual(trigger.t_enqueued, 123.456)

    def test_invalid_action_does_not_pop_decision_queue(self) -> None:
        env, drone_id = self._reset_controlled_env()
        self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-VALID-ACTION-SETUP",
        )
        env._enqueue_decision(drone_id, "test_idle", None)
        self.assertEqual(len(env._decision_queue), 1)
        original_trigger = env._decision_queue[0]

        with self.assertRaises(ValueError):
            env._apply_decision_core(
                DispatchAction(order_id="ORDER-P5C-NOT-IN-LOOKUP", mode=PolicyMode.B)
            )

        self.assertEqual(len(env._decision_queue), 1)
        self.assertEqual(env._decision_queue[0], original_trigger)

    def test_peek_current_decision_batch_uses_same_enqueue_time_prefix(self) -> None:
        env, drone_id = self._reset_controlled_env()
        idle_drone_ids = [
            item
            for item, state in env._drone_state.items()
            if state == TrainingDroneState.IDLE
        ]
        self.assertGreaterEqual(len(idle_drone_ids), 3)
        env._t_now = 321.0
        self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-BATCH-PEEK",
            create_time=300.0,
            deadline=3600.0,
        )

        first_id, second_id, later_id = idle_drone_ids[:3]
        env._enqueue_decision(first_id, "test_idle", None)
        env._enqueue_decision(second_id, "test_idle", None)
        env._decision_queue.append(
            DecisionTrigger(
                decision_id=env._next_decision_id,
                drone_id=later_id,
                trigger_type="test_idle",
                trigger_station_id=None,
                t_enqueued=env._t_now + 1.0,
            )
        )
        env._next_decision_id += 1

        batch = env.peek_current_decision_batch()

        self.assertEqual([ctx.deciding_drone_id for ctx in batch], [first_id, second_id])
        self.assertIs(batch[0].runtime_state, batch[1].runtime_state)
        self.assertIs(batch[0].coarse_plan, batch[1].coarse_plan)
        self.assertEqual(len(env._decision_queue), 3)

    def test_batch_teacher_assigns_conflicting_order_to_only_one_uav(self) -> None:
        env, drone_id = self._reset_controlled_env()
        idle_drone_ids = [
            item
            for item, state in env._drone_state.items()
            if state == TrainingDroneState.IDLE
        ]
        self.assertGreaterEqual(len(idle_drone_ids), 2)
        first_id, second_id = idle_drone_ids[:2]
        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-BATCH-CONFLICT",
            payload_weight=1.0,
        )
        env._enqueue_decision(first_id, "test_idle", None)
        env._enqueue_decision(second_id, "test_idle", None)

        batch = env.peek_current_decision_batch()
        self.assertEqual([ctx.deciding_drone_id for ctx in batch], [first_id, second_id])

        candidate_outputs = {
            ctx.deciding_drone_id: env.build_candidate_output(
                ctx,
                last_seen_plan_version=ctx.coarse_plan.plan_version,
            )
            for ctx in batch
        }
        for ctx in batch:
            candidate_out = candidate_outputs[ctx.deciding_drone_id]
            dispatch_order_ids = {
                action.order_id
                for action in candidate_out.resolved_action_lookup.dispatch_actions.values()
            }
            self.assertIn(order.order_id, dispatch_order_ids)

        result = build_batch_matching_teacher_labels(
            decision_contexts=batch,
            candidate_outputs_by_drone=candidate_outputs,
        )

        selected = [
            action
            for action in result.actions_by_drone.values()
            if isinstance(action, SharedDispatchAction)
            and action.order_id == order.order_id
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            sum(
                1
                for action in result.actions_by_drone.values()
                if action == WAIT_ACTION
            ),
            1,
        )

    def test_apply_decision_batch_rejects_duplicate_dispatch_order_without_mutation(self) -> None:
        env, drone_id = self._reset_controlled_env()
        idle_drone_ids = [
            item
            for item, state in env._drone_state.items()
            if state == TrainingDroneState.IDLE
        ]
        self.assertGreaterEqual(len(idle_drone_ids), 2)
        first_id, second_id = idle_drone_ids[:2]
        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-BATCH-DUPLICATE",
            payload_weight=1.0,
        )
        env._enqueue_decision(first_id, "test_idle", None)
        env._enqueue_decision(second_id, "test_idle", None)
        batch = env.peek_current_decision_batch()
        self.assertEqual([ctx.deciding_drone_id for ctx in batch], [first_id, second_id])

        actions_by_drone = {}
        for ctx in batch:
            actions_by_drone[ctx.deciding_drone_id] = next(
                action
                for action in ctx.action_lookup
                if isinstance(action, DispatchAction)
                and action.order_id == order.order_id
                and action.mode == PolicyMode.B
            )
        original_queue = tuple(env._decision_queue)
        original_t = env._t_now

        with self.assertRaises(ValueError):
            env.apply_decision_batch(actions_by_drone)

        self.assertEqual(tuple(env._decision_queue), original_queue)
        self.assertAlmostEqual(env._t_now, original_t)
        self.assertIn(order.order_id, env._require_order_manager().pending_orders)
        self.assertNotIn(order.order_id, env._require_order_manager().assigned_orders)

    def test_apply_decision_batch_submits_wait_and_dispatch_before_advancing_time(self) -> None:
        env, drone_id = self._reset_controlled_env()
        idle_drone_ids = [
            item
            for item, state in env._drone_state.items()
            if state == TrainingDroneState.IDLE
        ]
        self.assertGreaterEqual(len(idle_drone_ids), 2)
        wait_drone_id, dispatch_drone_id = idle_drone_ids[:2]
        order = self._inject_order(
            env,
            drone_id=dispatch_drone_id,
            order_id="ORDER-P5C-BATCH-WAIT-DISPATCH",
            payload_weight=1.0,
        )
        env._enqueue_decision(wait_drone_id, "test_idle", None)
        env._enqueue_decision(dispatch_drone_id, "test_idle", None)
        batch = env.peek_current_decision_batch()
        self.assertEqual(
            [ctx.deciding_drone_id for ctx in batch],
            [wait_drone_id, dispatch_drone_id],
        )
        dispatch_context = batch[1]
        dispatch_action = next(
            action
            for action in dispatch_context.action_lookup
            if isinstance(action, DispatchAction)
            and action.order_id == order.order_id
            and action.mode == PolicyMode.B
        )

        result = env.apply_decision_batch(
            {
                wait_drone_id: WAIT_ACTION,
                dispatch_drone_id: dispatch_action,
            }
        )

        self.assertEqual(result.info["event"], "apply_decision_batch")
        self.assertEqual(result.info["batch_size"], 2)
        self.assertIn(wait_drone_id, result.info["per_drone_rewards"])
        self.assertIn(dispatch_drone_id, result.info["per_drone_rewards"])
        order_mgr = env._require_order_manager()
        self.assertNotIn(order.order_id, order_mgr.pending_orders)
        progressed_order = order_mgr.assigned_orders.get(
            order.order_id
        ) or next(
            (
                item
                for item in order_mgr.completed_orders
                if item.order_id == order.order_id
            ),
            None,
        )
        self.assertIsNotNone(progressed_order)
        self.assertEqual(progressed_order.assigned_vehicle_id, dispatch_drone_id)

    def test_runtime_state_exposes_reservation_and_count(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-01")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_deliver + 600.0,
                departure_time=t_deliver + 600.0 + 1e-6,
            )
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )

        runtime_state = env.build_runtime_state_view()
        reservation = runtime_state.drone_states[drone_id].reservation
        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.recover_node, recover_node_id)
        self.assertIn(recover_node_id, runtime_state.reservation_count)
        self.assertEqual(runtime_state.reservation_count[recover_node_id], 1)

    def test_mode_c_dispatch_commit_records_planned_commitment_metrics(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-ETA-01",
            offset_x=3000.0,
        )
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_deliver + 900.0,
                departure_time=t_deliver + 900.0 + 1e-6,
            )
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )

        commit = env._dispatch_commit[drone_id]
        recover_host = env._resolve_fixed_node(recover_node_id)
        expected_delivery_finish = (
            env._estimate_delivery_arrival_time(drone, order)
            + env._scene_solver_params().drone_service_time_order_s
        )
        expected_uav_arrival_lb = expected_delivery_finish + env._estimate_flight_time(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=recover_host.get_location(expected_delivery_finish),
        )
        expected_truck_arrival = t_deliver + 900.0

        self.assertAlmostEqual(
            commit.planned_truck_arrival_time,
            expected_truck_arrival,
            places=6,
        )
        self.assertAlmostEqual(
            commit.planned_uav_arrival_time_lb,
            expected_uav_arrival_lb,
            places=6,
        )
        self.assertAlmostEqual(
            commit.planned_execution_slack_sec,
            expected_truck_arrival - expected_uav_arrival_lb,
            places=6,
        )

    def test_mode_c_dispatch_without_node_selects_recovery_after_delivery_service(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_id = sorted(entity_mgr.stations)[0]
        depot_id = env._require_depot().depot_id

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-POST-01")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        t_service_finish = (
            t_deliver + env._scene_solver_params().drone_service_time_order_s
        )
        depot = env._require_depot()
        depot_uav_arrival = t_service_finish + env._estimate_flight_time(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=depot.get_location(t_service_finish),
        )
        depot_truck_eta = (
            depot_uav_arrival + env._cfg.rendezvous_execution_margin_sec + 60.0
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=t_service_finish + 1.0,
                departure_time=t_service_finish + 1.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=depot_id,
                arrival_time=depot_truck_eta,
                departure_time=depot_truck_eta + 1e-6,
            ),
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(order_id=order.order_id, mode=PolicyMode.C),
        )
        self.assertIsNone(env._dispatch_commit[drone_id].selected_recover_node)
        self.assertNotIn(drone_id, env._reservations)

        deliver_leg = env._flight_legs[drone_id]
        env._advance_to_event(deliver_leg.arrival_time)
        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        reward_before_service = env._agent_cost_accum.get(drone_id, 0.0)
        env._advance_to_event(service_finish_time)

        commit = env._dispatch_commit[drone_id]
        self.assertEqual(commit.selected_recover_node, depot_id)
        self.assertAlmostEqual(commit.planned_truck_arrival_time, depot_truck_eta, places=6)
        self.assertIn(drone_id, env._reservations)
        self.assertEqual(env._reservations[drone_id].recover_node, depot_id)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.RETURN_TO_RENDEZVOUS)
        self.assertEqual(env._flight_legs[drone_id].target_node_id, depot_id)
        self.assertAlmostEqual(
            env._agent_cost_accum.get(drone_id, 0.0) - reward_before_service,
            env._cfg.R_delivery_bonus + env._cfg.mode_c_attempt_bonus,
            places=6,
        )

    def test_mode_c_dispatch_without_post_delivery_node_enters_fallback(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-POST-FAIL")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        t_service_finish = (
            t_deliver + env._scene_solver_params().drone_service_time_order_s
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=t_service_finish + 1.0,
                departure_time=t_service_finish + 1.0 + 1e-6,
            )
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(order_id=order.order_id, mode=PolicyMode.C),
        )

        deliver_leg = env._flight_legs[drone_id]
        env._advance_to_event(deliver_leg.arrival_time)
        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        env._advance_to_event(service_finish_time)

        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.FALLBACK_RECOVERY)
        self.assertNotIn(drone_id, env._reservations)
        self.assertIsNone(env._dispatch_commit[drone_id].selected_recover_node)
        self.assertEqual(env._episode_mode_c_post_delivery_revalidation_fail_count, 1)
        self.assertEqual(env._fallback_leg[drone_id].cause, FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE)
        self.assertEqual(
            env._episode_fallback_cause_counts[FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE],
            1,
        )
        self.assertEqual(env._episode_ppo_attributed_fallback_count, 1)

    def test_mode_c_riding_launch_commitment_uses_decision_time_truck_eta(self) -> None:
        env = self._make_env()
        env.reset()

        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        order_mgr._next_order_time = math.inf
        order_mgr._scheduled_dynamic = []
        order_mgr._scheduled_dynamic_i = 0
        env._decision_queue.clear()
        env._background_mode_a_pending.clear()

        drone_id = self._first_riding_drone_id(env)
        recover_node_id = sorted(env._require_entity_manager().stations)[0]
        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-ETA-TRUCK",
            offset_x=1000.0,
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=env._t_now + 1200.0,
                departure_time=env._t_now + 1200.0 + 1e-6,
            )
        ]

        trigger = DecisionTrigger(
            decision_id=0,
            drone_id=drone_id,
            trigger_type="truck_station_arrival",
            trigger_station_id=recover_node_id,
            t_enqueued=float(env._t_now),
        )
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )

        commit = env._dispatch_commit[drone_id]

        self.assertAlmostEqual(
            float(commit.planned_truck_arrival_time),
            env._t_now + 1200.0,
            places=6,
        )
        self.assertIsNotNone(commit.planned_uav_arrival_time_lb)
        self.assertIsNotNone(commit.planned_execution_slack_sec)

    def test_runtime_state_reservation_view_does_not_expose_fixed_expiry(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-ETA-TRUCK-GUARD",
            offset_x=50.0,
        )
        t_arrive_truck = env._t_now + 1800.0
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_arrive_truck,
                departure_time=t_arrive_truck + 1e-6,
            )
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )

        runtime_state = env.build_runtime_state_view()
        reservation = runtime_state.drone_states[drone_id].reservation
        self.assertIsNotNone(reservation)
        self.assertFalse(hasattr(reservation, "expires_at"))

    def test_revalidation_rejects_locked_node_beyond_max_wait(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-02")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_deliver + 600.0,
                departure_time=t_deliver + 600.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=env._require_depot().depot_id,
                arrival_time=t_deliver + 1200.0,
                departure_time=t_deliver + 1200.0 + 1e-6,
            ),
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )
        deliver_leg = env._flight_legs[drone_id]
        env._advance_to_event(deliver_leg.arrival_time)
        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        reward = env._advance_to_event(service_finish_time)

        self.assertEqual(reward, env._cfg.R_delivery_bonus)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.FALLBACK_RECOVERY)
        self.assertNotIn(drone_id, env._reservations)
        self.assertEqual(env._flight_legs[drone_id].kind, "fallback_recovery")

    def test_revalidation_failure_reselects_available_post_delivery_node(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-02B")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_deliver + 600.0,
                departure_time=t_deliver + 600.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=env._require_depot().depot_id,
                arrival_time=t_deliver + 1200.0,
                departure_time=t_deliver + 1200.0 + 1e-6,
            ),
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.C,
                recover_node_id=recover_node_id,
            ),
        )
        deliver_leg = env._flight_legs[drone_id]
        env._advance_to_event(deliver_leg.arrival_time)
        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        missed_truck_arrival = service_finish_time - 1.0
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=missed_truck_arrival,
                departure_time=missed_truck_arrival + 1e-6,
            ),
            BackboneVisit(
                node_id=env._require_depot().depot_id,
                arrival_time=service_finish_time + 100.0,
                departure_time=service_finish_time + 100.0 + 1e-6,
            ),
        ]

        env._advance_to_event(service_finish_time)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.RETURN_TO_RENDEZVOUS,
        )
        self.assertEqual(
            env._dispatch_commit[drone_id].selected_recover_node,
            env._require_depot().depot_id,
        )
        self.assertIn(drone_id, env._reservations)
        self.assertEqual(env._flight_legs[drone_id].kind, "return_to_rendezvous")

    def test_hard_overdue_records_metric_without_removing_pending_order(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._planned_route_stop_i = len(env._planned_route_stops)

        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-03",
            create_time=-2000.0,
            deadline=-1000.0,
        )

        reward = env._advance_to_event(env._t_now + 1.0)

        self.assertAlmostEqual(reward, 0.0, places=6)
        self.assertIn(order.order_id, env._require_order_manager().pending_orders)
        self.assertFalse(env._require_order_manager().completed_orders)
        self.assertEqual(order.status, TaskStatus.PENDING)
        snapshot = env.build_episode_metrics_snapshot()
        self.assertEqual(snapshot["hard_overdue_count"], 1)
        self.assertEqual(snapshot["timeout_order_count"], 0)

    def test_late_delivery_discount_keeps_delivery_reward_positive(self) -> None:
        env, drone_id = self._reset_controlled_env()
        order = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-LATE-REWARD",
            create_time=0.0,
            deadline=10.0,
        )
        order.actual_deliver_time = 10.0 + 10_000.0

        delivery_bonus, lateness_discount = env._compute_delivery_reward(order)

        self.assertAlmostEqual(delivery_bonus, env._cfg.R_delivery_bonus, places=6)
        self.assertLess(lateness_discount, 0.0)
        self.assertGreater(
            delivery_bonus + lateness_discount,
            0.0,
        )
        self.assertGreaterEqual(
            delivery_bonus + lateness_discount,
            env._cfg.min_late_delivery_reward,
        )

    def test_episode_snapshot_reports_frontend_style_average_order_delay(self) -> None:
        env, drone_id = self._reset_controlled_env()
        order_mgr = env._require_order_manager()

        on_time = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-AVG-DELAY-ON-TIME",
            create_time=0.0,
            deadline=100.0,
        )
        late = self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-AVG-DELAY-LATE",
            create_time=0.0,
            deadline=100.0,
        )
        for order, deliver_time in ((on_time, 90.0), (late, 190.0)):
            order_mgr.pending_orders.pop(order.order_id)
            order.actual_deliver_time = deliver_time
            order.update_status(TaskStatus.ASSIGNED)
            order.update_status(TaskStatus.PICKED_UP)
            order.update_status(TaskStatus.DELIVERING)
            order.update_status(TaskStatus.COMPLETED)
            order_mgr.completed_orders.append(order)

        snapshot = env.build_episode_metrics_snapshot()

        self.assertEqual(snapshot["completed_with_timing_order_count"], 2)
        self.assertAlmostEqual(snapshot["order_delay_sum_min"], 1.5, places=6)
        self.assertAlmostEqual(snapshot["avg_order_delay_min"], 0.75, places=6)

    def test_wait_action_uses_exact_idle_penalty_and_gap_cap(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._planned_route_stop_i = len(env._planned_route_stops)
        self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-04")
        env._enqueue_decision(drone_id, "test_idle", None)

        result = env.step(WAIT_ACTION)

        expected_opportunity_penalty = -env._cfg.wait_opportunity_penalty_coef * 0.1
        self.assertAlmostEqual(
            result.reward,
            -env._cfg.wait_idle_penalty_coef * env._cfg.max_wait_decision_gap_sec
            + expected_opportunity_penalty,
            places=6,
        )
        self.assertAlmostEqual(
            result.info["wait_opportunity_penalty"],
            expected_opportunity_penalty,
            places=6,
        )
        self.assertAlmostEqual(
            result.info["reward_breakdown"]["wait_opportunity"],
            expected_opportunity_penalty,
            places=6,
        )
        self.assertAlmostEqual(
            result.runtime_state.t_now,
            env._cfg.max_wait_decision_gap_sec,
            places=6,
        )
        self.assertEqual(result.info["wait_delta"], env._cfg.max_wait_decision_gap_sec)
        self.assertIsNotNone(result.decision_context)
        self.assertTrue(
            any(
                trigger.drone_id == drone_id and trigger.trigger_type == "wait_resume"
                for trigger in env._decision_queue
            )
        )

    def test_online_interfaces_apply_wait_without_implicit_advance(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._planned_route_stop_i = len(env._planned_route_stops)
        self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-ONLINE-WAIT")
        env._enqueue_decision(drone_id, "test_idle", None)

        self.assertAlmostEqual(env.peek_next_decision_time(), env._t_now, places=6)
        applied = env.apply_decision(WAIT_ACTION)

        self.assertAlmostEqual(applied.runtime_state.t_now, 0.0, places=6)
        self.assertIsNone(applied.decision_context)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.ACTIVE_WAIT)
        self.assertAlmostEqual(
            env._active_wait_until[drone_id],
            env._cfg.max_wait_decision_gap_sec,
            places=6,
        )

        advanced = env.advance_to_time(env._cfg.upper_horizon_sec)

        self.assertAlmostEqual(
            advanced.runtime_state.t_now,
            env._cfg.max_wait_decision_gap_sec,
            places=6,
        )
        self.assertIsNotNone(advanced.decision_context)
        self.assertTrue(
            any(
                trigger.drone_id == drone_id and trigger.trigger_type == "wait_resume"
                for trigger in env._decision_queue
            )
        )

    def test_step_preserves_delayed_attribution_carry_in_before_current_window(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._planned_route_stop_i = len(env._planned_route_stops)
        self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-04-CARRY")
        env._enqueue_decision(drone_id, "test_idle", None)

        carried_reward = -7.5
        env._agent_cost_accum[drone_id] = carried_reward

        result = env.step(WAIT_ACTION)

        expected_opportunity_penalty = -env._cfg.wait_opportunity_penalty_coef * 0.1
        expected_post_action = (
            -env._cfg.wait_idle_penalty_coef * env._cfg.max_wait_decision_gap_sec
            + expected_opportunity_penalty
        )
        self.assertAlmostEqual(
            result.reward,
            carried_reward + expected_post_action,
            places=6,
        )
        self.assertAlmostEqual(
            result.info["reward_breakdown"]["attributed_carry_in"],
            carried_reward,
            places=6,
        )
        self.assertAlmostEqual(
            result.info["reward_breakdown"]["attributed_post_action"],
            expected_post_action,
            places=6,
        )
        self.assertAlmostEqual(
            result.info["reward_breakdown"]["attributed_total"],
            carried_reward + expected_post_action,
            places=6,
        )
        self.assertNotIn(drone_id, env._agent_cost_accum)

    def test_wait_opportunity_penalty_requires_legal_dispatch(self) -> None:
        env, drone_id = self._reset_controlled_env()
        env._planned_route_stop_i = len(env._planned_route_stops)
        self._inject_order(
            env,
            drone_id=drone_id,
            order_id="ORDER-P5C-WAIT-INFEASIBLE",
            offset_x=1_000_000.0,
        )
        env._decision_queue.append(
            DecisionTrigger(
                decision_id=env._next_decision_id,
                drone_id=drone_id,
                trigger_type="test_idle",
                trigger_station_id=None,
                t_enqueued=float(env._t_now),
            )
        )
        env._next_decision_id += 1

        result = env.step(WAIT_ACTION)

        self.assertNotIn("wait_opportunity_penalty", result.info)
        self.assertAlmostEqual(
            result.reward,
            -env._cfg.wait_idle_penalty_coef * env._cfg.max_wait_decision_gap_sec,
            places=6,
        )

    def test_consume_terminal_agent_costs_clears_tail_accumulator(self) -> None:
        env, drone_id = self._reset_controlled_env()
        other_drone_id = self._second_drone_id(env, drone_id)
        env._agent_cost_accum[drone_id] = -3.5
        env._agent_cost_accum[other_drone_id] = 1.25
        env._t_now = env._cfg.upper_horizon_sec

        pending = env.consume_terminal_agent_costs()

        self.assertEqual(
            pending,
            {
                drone_id: -3.5,
                other_drone_id: 1.25,
            },
        )
        self.assertFalse(env._agent_cost_accum)

    def test_no_authorized_orders_does_not_enqueue_decision(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        env._decision_queue.clear()

        env._enqueue_initial_idle_decisions()

        self.assertFalse(env._decision_queue)
        self.assertIsNone(env.current_decision_context)

    def test_wake_stranded_idle_drones_requeues_idle_uavs_after_new_order_arrives(self) -> None:
        env, drone_id = self._reset_controlled_env()
        sleeping_drone_id = self._second_drone_id(env, drone_id)
        env._decision_queue.clear()

        env._enqueue_initial_idle_decisions()
        self.assertFalse(env._decision_queue)

        env._drone_state[sleeping_drone_id] = TrainingDroneState.ACTIVE_WAIT
        env._active_wait_resume[sleeping_drone_id] = TrainingDroneState.IDLE
        self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5C-WAKE-01")

        env._wake_stranded_idle_drones()

        queued_ids = [item.drone_id for item in env._decision_queue]
        queued_trigger_types = {item.trigger_type for item in env._decision_queue}
        self.assertIn(drone_id, queued_ids)
        self.assertNotIn(sleeping_drone_id, queued_ids)
        self.assertEqual(queued_trigger_types, {"order_arrival_wake"})
        self.assertIsNotNone(env.current_decision_context)

    def test_settle_per_dt_rewards_tracks_wait_and_queue_without_penalty(self) -> None:
        env, drone_id = self._reset_controlled_env()
        other_drone_id = self._second_drone_id(env, drone_id)

        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        env._drone_state[other_drone_id] = TrainingDroneState.QUEUEING_AT_HOST

        reward, breakdown = env._settle_per_dt_rewards(t_prev=0.0, t_next=5.0)

        self.assertAlmostEqual(reward, 0.0, places=6)
        self.assertNotIn("wait", breakdown)
        self.assertNotIn("queue", breakdown)
        self.assertAlmostEqual(env._episode_wait_time_sec, 5.0, places=6)
        self.assertAlmostEqual(env._episode_queue_time_sec, 5.0, places=6)

    def test_reservation_timeout_enters_fallback_without_reward_penalty(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        t_now = float(env._cfg.rendezvous_max_wait_sec + 5.0)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=t_now + 100.0,
                departure_time=t_now + 100.0 + 1e-6,
            )
        ]
        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        env._acquire_reservation(
            drone_id=drone_id,
            recover_node_id=station_id,
        )
        env._rendezvous_wait_started_at[drone_id] = 0.0

        reward = env._process_reservation_timeouts(t_now)

        self.assertAlmostEqual(reward, 0.0, places=6)
        self.assertNotIn(drone_id, env._reservations)
        self.assertAlmostEqual(env._agent_cost_accum.get(drone_id, 0.0), 0.0, places=6)
        self.assertEqual(env._episode_reservation_timeout_count, 1)
        self.assertAlmostEqual(env._episode_reservation_timeout_cost_sec, 5.0, places=6)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.FALLBACK_RECOVERY)

    def test_mode_a_background_completion_updates_stats_without_reward(self) -> None:
        env = self._make_env()
        result = env.reset()
        self.assertIn("system_context_stats", result.info)

        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        order_mgr._next_order_time = math.inf
        order_mgr._scheduled_dynamic = []
        order_mgr._scheduled_dynamic_i = 0
        env._decision_queue.clear()

        background_stop = next(
            stop
            for stop in env._planned_route_stops
            if stop.node_type == "customer" and stop.order_id in env._background_mode_a_pending
        )

        reward = env._advance_to_event(background_stop.arrival_time)

        self.assertEqual(reward, 0.0)
        self.assertNotIn("delivery_bonus", env._last_reward_breakdown)

        stats = env.build_system_context_stats()
        self.assertGreaterEqual(stats["mode_a_background_order_count"], 1)
        self.assertEqual(stats["mode_a_background_completed_count"], 1)
        self.assertAlmostEqual(
            stats["mode_a_background_completion_time_sum"],
            background_stop.arrival_time,
            places=6,
        )
        self.assertEqual(len(stats["truck_background_order_completion_events"]), 1)
        self.assertEqual(
            stats["truck_background_order_completion_events"][0]["order_id"],
            background_stop.order_id,
        )


if __name__ == "__main__":
    unittest.main()
