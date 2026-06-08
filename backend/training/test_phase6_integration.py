#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 integration tests.

运行方式：
  python -m unittest backend.training.test_phase6_integration
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from core.entities.order import Order
from core.entities.primitives import Position3D
from environment.geo.osm_service import build_road_graph, find_nearest_node, shortest_path
from utils.coord_utils import wgs84_to_utm

from .candidate_builder import (
    CandidateBuilder,
    _candidate_deadline_sort_key,
    _select_candidate_orders,
)
from .contracts import (
    PlannerTriggerContext,
    PolicyMode,
    ReservationConstraintState,
    ReservationPlanOutcome,
    ReservationPlanStatus,
    RouteDriftRef,
    TruckReservationConstraint,
)
from .env_adapter import (
    BackboneVisit,
    DispatchCommit,
    PlannedStop,
    ReservationState,
    TrainingDroneState,
    TrainingEnvAdapter,
    WAIT_ACTION,
)
from .order_source_adapter import OrderSourceMode, build_order_source
from .observation_tensorizer import ORDER_TOKEN_FIELDS, ObservationTensorizer
from .scene_loader import load_default_scene


class TestPhase6Integration(unittest.TestCase):
    def _make_env(self) -> TrainingEnvAdapter:
        return TrainingEnvAdapter()

    def _first_idle_drone_id(self, env: TrainingEnvAdapter) -> str:
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.IDLE:
                return drone_id
        self.fail("默认场景中至少应有一架 idle 无人机")

    def _first_riding_drone_id(self, env: TrainingEnvAdapter) -> str:
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.RIDING_WITH_TRUCK:
                return drone_id
        self.fail("默认场景中至少应有一架 riding_with_truck 无人机")

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

    def _inject_order_at(
        self,
        env: TrainingEnvAdapter,
        *,
        order_id: str,
        position: Position3D,
        deadline_offset: float = 3600.0,
        payload_weight: float = 1.0,
    ) -> Order:
        order = Order(
            order_id=order_id,
            create_time=env._t_now,
            deadline=env._t_now + deadline_offset,
            delivery_loc=Position3D(x=position.x, y=position.y, z=position.z),
            payload_weight=payload_weight,
        )
        env._require_order_manager().pending_orders[order.order_id] = order
        return order

    def _candidate_sort_item(
        self,
        order_id: str,
        remaining_time: float,
        *,
        priority_band: int = 1,
    ) -> dict[str, object]:
        return {
            "order_id": order_id,
            "order_feature": SimpleNamespace(
                remaining_time=remaining_time,
                priority_band=priority_band,
            ),
        }

    def test_background_mode_a_pending_is_seeded_from_static_truck_only_orders(self) -> None:
        env = self._make_env()
        env.reset()

        expected_order_ids = {
            order.order_id
            for order in env._scene_ctx.static_orders
            if order.payload_weight > env._heavy_payload_capacity
        }

        self.assertEqual(expected_order_ids, env._background_mode_a_pending)
        self.assertEqual(
            len(expected_order_ids),
            env.build_system_context_stats()["mode_a_background_order_count"],
        )

    def test_stale_cached_plan_does_not_suppress_new_idle_decision(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]

        stale_order = self._inject_order_at(
            env,
            order_id="ORDER-P6-STALE-01",
            position=Position3D(
                x=drone.current_loc.x + 20.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        runtime_state = env.build_runtime_state_view()
        self.assertTrue(env._has_authorized_orders_now(runtime_state))

        stale_plan = env._current_coarse_plan
        self.assertIsNotNone(stale_plan)
        self.assertTrue(stale_plan.is_order_authorized(stale_order.order_id))
        stale_plan_version = stale_plan.plan_version

        env._require_order_manager().pending_orders.clear()
        new_order = self._inject_order_at(
            env,
            order_id="ORDER-P6-STALE-02",
            position=Position3D(
                x=drone.current_loc.x + 40.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        self.assertFalse(
            stale_plan.is_order_authorized(new_order.order_id),
            "测试前置条件不成立：旧 coarse plan 不应提前包含新订单",
        )

        env._enqueue_initial_idle_decisions()

        refreshed_plan = env._current_coarse_plan
        self.assertIsNotNone(refreshed_plan)
        self.assertGreater(
            refreshed_plan.plan_version,
            stale_plan_version,
            "当旧 coarse plan 不含新订单时，应触发 replan 而不是继续沿用旧缓存",
        )
        self.assertTrue(refreshed_plan.is_order_authorized(new_order.order_id))
        self.assertFalse(refreshed_plan.is_order_authorized(stale_order.order_id))

        self.assertTrue(
            any(
                trigger.drone_id == drone_id and trigger.trigger_type == "initial_idle"
                for trigger in env._decision_queue
            ),
            "新单出现时，不应因为旧 coarse plan 仍缓存而压住 idle 决策点",
        )

    def test_poisson_truck_only_dynamic_orders_are_not_uav_authorized(self) -> None:
        scene_ctx = load_default_scene()
        order_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.POISSON,
            overrides={
                "poisson_arrival_rate": 0.0,
                "truck_only_dynamic_enabled": True,
                "truck_only_dynamic_orders_per_episode": 1,
                "truck_only_dynamic_spawn_start_s": 0.0,
                "truck_only_dynamic_spawn_end_s": 0.0,
                "truck_only_dynamic_weight_min_kg": 10.5,
                "truck_only_dynamic_weight_max_kg": 10.5,
                "truck_only_dynamic_deadline_window_min_min": 90,
                "truck_only_dynamic_deadline_window_max_min": 90,
            },
        )
        static_uav_ids = {
            str(entry["order_id"])
            for entry in order_source.initial_static_uav_orders
        }
        self.assertTrue(static_uav_ids)
        truck_only_entries = [
            entry
            for entry in order_source.scheduled_dynamic_orders
            if str(entry["order_id"]).startswith("TRUCKONLY-")
        ]
        self.assertEqual(len(truck_only_entries), 1)
        self.assertGreater(float(truck_only_entries[0]["payload_weight"]), 10.0)
        road_graph, road_nodes = build_road_graph(
            Path(scene_ctx.road_network.xml_path).read_text(encoding="utf-8"),
            respect_osm_oneway=True,
        )
        depot = next(iter(scene_ctx.depots.values()))
        depot_node = find_nearest_node(
            road_graph,
            road_nodes,
            depot.location.x,
            depot.location.y,
        )
        delivery_x, delivery_y = wgs84_to_utm(
            float(truck_only_entries[0]["delivery_lng"]),
            float(truck_only_entries[0]["delivery_lat"]),
        )
        delivery_node = find_nearest_node(
            road_graph,
            road_nodes,
            delivery_x,
            delivery_y,
        )
        self.assertTrue(shortest_path(road_graph, depot_node, delivery_node))
        self.assertTrue(shortest_path(road_graph, delivery_node, depot_node))

        env = TrainingEnvAdapter(scene_ctx=scene_ctx, order_source=order_source)
        env.reset()
        order_mgr = env._require_order_manager()
        self.assertTrue(static_uav_ids.issubset(set(order_mgr.pending_orders)))
        truck_only_order_id = str(truck_only_entries[0]["order_id"])
        if truck_only_order_id in order_mgr.pending_orders:
            coarse_plan = env._refresh_coarse_plan_if_needed(
                env.build_runtime_state_view(),
                allow_truck_route_replan=True,
            )
            self.assertFalse(coarse_plan.is_order_authorized(truck_only_order_id))
            self.assertNotIn(truck_only_order_id, coarse_plan.policy_mode_mask)
        else:
            completed_ids = {order.order_id for order in order_mgr.completed_orders}
            self.assertIn(truck_only_order_id, completed_ids)
            stats = env.build_system_context_stats()
            truck_only_events = [
                event
                for event in stats["truck_background_order_completion_events"]
                if event.get("truck_only_dynamic")
            ]
            self.assertEqual(len(truck_only_events), 1)
            self.assertEqual(truck_only_events[0]["order_id"], truck_only_order_id)

    def test_poisson_reset_injects_static_uav_without_poisson_id_overwrite(self) -> None:
        config_path = Path(
            "backend/config/"
            "rh_alns_cmrappo_bc_warm_start_stage1_benchmark_only_overnight_20k_default_poisson.yaml"
        )
        scene_ctx = load_default_scene(config_path=config_path)
        order_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.POISSON,
            config_path=config_path,
        )
        static_uav_by_id = {
            str(entry["order_id"]): entry
            for entry in order_source.initial_static_uav_orders
        }
        self.assertTrue(static_uav_by_id)

        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=order_source,
            config_path=config_path,
        )
        env.reset()
        order_mgr = env._require_order_manager()

        self.assertTrue(set(static_uav_by_id).issubset(set(order_mgr.pending_orders)))
        self.assertGreater(len(order_mgr.pending_orders), len(static_uav_by_id))
        for order_id, entry in static_uav_by_id.items():
            self.assertAlmostEqual(
                order_mgr.pending_orders[order_id].deadline,
                float(entry["deadline_sim_s"]),
            )

    def test_manual_truck_only_order_is_not_uav_authorized(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        order = self._inject_order_at(
            env,
            order_id="TRUCKONLY-MANUAL-P6",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
            payload_weight=env._heavy_payload_capacity + 1.0,
        )

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=True,
        )

        self.assertFalse(coarse_plan.is_order_authorized(order.order_id))
        self.assertNotIn(order.order_id, coarse_plan.policy_mode_mask)
        self.assertTrue(
            any(
                stop.node_type == "customer" and stop.order_id == order.order_id
                for stop in coarse_plan.truck_plan_stops
            )
        )

    def test_dynamic_truck_replan_keeps_pending_background_mode_a_order(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        background_order_id = "ORD-STATIC-07"
        self.assertTrue(
            any(
                order.order_id == background_order_id
                for order in env._scene_ctx.static_orders
            )
        )
        env._background_mode_a_pending.add(background_order_id)

        truck_only_order = self._inject_order_at(
            env,
            order_id="TRUCKONLY-P6-KEEP-BACKGROUND",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            payload_weight=env._heavy_payload_capacity + 1.0,
        )

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=True,
        )
        planned_customer_order_ids = {
            stop.order_id
            for stop in coarse_plan.truck_plan_stops
            if stop.node_type == "customer"
        }

        self.assertIn(background_order_id, planned_customer_order_ids)
        self.assertIn(truck_only_order.order_id, planned_customer_order_ids)
        self.assertFalse(coarse_plan.is_order_authorized(background_order_id))
        self.assertNotIn(background_order_id, coarse_plan.policy_mode_mask)
        self.assertNotIn(background_order_id, env._require_order_manager().pending_orders)

    def test_truck_only_replan_is_deferred_until_truck_stop(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        original_route_order_ids = [
            stop.order_id
            for stop in env._planned_route_stops
            if stop.node_type == "customer"
        ]
        order = self._inject_order_at(
            env,
            order_id="TRUCKONLY-P6-DEFERRED",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            payload_weight=env._heavy_payload_capacity + 1.0,
        )

        deferred_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view()
        )

        self.assertTrue(env._truck_replan_pending)
        self.assertNotIn(
            order.order_id,
            [stop.order_id for stop in deferred_plan.truck_plan_stops],
        )
        self.assertEqual(
            original_route_order_ids,
            [
                stop.order_id
                for stop in env._planned_route_stops
                if stop.node_type == "customer"
            ],
        )

        applied_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=True,
        )

        self.assertFalse(env._truck_replan_pending)
        self.assertIn(
            order.order_id,
            {
                stop.order_id
                for stop in applied_plan.truck_plan_stops
                if stop.node_type == "customer"
            },
        )
        self.assertIn(
            order.order_id,
            [
                stop.order_id
                for stop in env._planned_route_stops
                if stop.node_type == "customer"
            ],
        )

    def test_planner_bridge_marks_all_future_stations_as_active_launch(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 3)

        station_1 = entity_mgr.stations[station_ids[0]]
        station_2 = entity_mgr.stations[station_ids[1]]
        station_3 = entity_mgr.stations[station_ids[2]]
        env._full_backbone_cache = [
            BackboneVisit(node_id=station_1.station_id, arrival_time=60.0, departure_time=60.0 + 1e-6),
            BackboneVisit(node_id=station_2.station_id, arrival_time=120.0, departure_time=120.0 + 1e-6),
            BackboneVisit(node_id=station_3.station_id, arrival_time=180.0, departure_time=180.0 + 1e-6),
        ]

        self._inject_order_at(
            env,
            order_id="ORDER-P6-PLAN-01",
            position=station_1.location,
        )
        runtime_state = env.build_runtime_state_view()
        self.assertTrue(env._has_authorized_orders_now(runtime_state))
        plan_v0 = env._current_coarse_plan
        self.assertIsNotNone(plan_v0)
        self.assertGreaterEqual(plan_v0.plan_version, 0)
        self.assertEqual(
            env._active_launch_stations,
            {station_1.station_id, station_2.station_id, station_3.station_id},
        )

        env._require_order_manager().pending_orders.clear()
        self._inject_order_at(
            env,
            order_id="ORDER-P6-PLAN-02",
            position=station_2.location,
        )
        env._fallback_event_times[:] = [env._t_now, env._t_now]

        runtime_state = env.build_runtime_state_view()
        plan_v1 = env._refresh_coarse_plan_if_needed(runtime_state)
        self.assertGreater(plan_v1.plan_version, plan_v0.plan_version)
        self.assertEqual(
            env._active_launch_stations,
            {station_1.station_id, station_2.station_id, station_3.station_id},
        )

    def test_planner_bridge_recovery_pool_exposes_future_backbone_nodes(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 5)

        preferred_station_id = station_ids[4]
        preferred_station = entity_mgr.stations[preferred_station_id]
        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-PLAN-RECOVERY-01",
            position=preferred_station.location,
        )

        for station_id in station_ids[:4]:
            station = entity_mgr.stations[station_id]
            station.serving_drones = {
                f"busy-{station_id}-{idx}": env._t_now + 600.0 + idx
                for idx in range(station.parking_slots)
            }
            station.wait_queue = [f"wait-{station_id}"]

        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=env._t_now + 60.0 * (idx + 1),
                departure_time=env._t_now + 60.0 * (idx + 1) + 1e-6,
            )
            for idx, station_id in enumerate(station_ids[:5])
        ]

        runtime_state = env.build_runtime_state_view()
        env._planner_bridge.reset_episode()
        coarse_plan = env._planner_bridge.maybe_replan(
            runtime_state,
            PlannerTriggerContext(
                t_now=env._t_now,
                backlog_new_orders=0,
                fallback_count_in_window=0,
                hard_failure_count_in_window=0,
                route_drift_ratio=0.0,
            ),
        )
        recovery_nodes = coarse_plan.recovery_pool[order.order_id]

        self.assertLessEqual(
            len(recovery_nodes),
            env._cfg.max_candidate_recovery_per_order,
        )
        self.assertTrue(set(recovery_nodes).issubset(set(station_ids[:5])))
        self.assertEqual(recovery_nodes[0], preferred_station_id)
        self.assertIn(
            preferred_station_id,
            recovery_nodes,
            "PlannerBridge 应与 env 内联 coarse plan 一致，优先保留靠近订单的未来固定节点",
        )

    def test_poisson_planner_bridge_rejects_empty_backbone(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        env._full_backbone_cache = []
        runtime_state = env.build_runtime_state_view()
        env._planner_bridge.reset_episode()

        with self.assertRaises(ValueError):
            env._planner_bridge.maybe_replan(
                runtime_state,
                PlannerTriggerContext(
                    t_now=env._t_now,
                    backlog_new_orders=0,
                    fallback_count_in_window=0,
                    hard_failure_count_in_window=0,
                    route_drift_ratio=0.0,
                ),
            )

    def test_benchmark_planner_bridge_allows_empty_backbone(self) -> None:
        scene_ctx = load_default_scene()
        benchmark_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.BENCHMARK,
        )
        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=benchmark_source,
        )
        env.reset()
        env._full_backbone_cache = []
        runtime_state = env.build_runtime_state_view()
        env._planner_bridge.reset_episode()

        coarse_plan = env._planner_bridge.maybe_replan(
            runtime_state,
            PlannerTriggerContext(
                t_now=env._t_now,
                backlog_new_orders=0,
                fallback_count_in_window=0,
                hard_failure_count_in_window=0,
                route_drift_ratio=0.0,
            ),
        )
        self.assertTrue(coarse_plan.allow_empty_backbone_route)
        self.assertEqual(coarse_plan.truck_backbone_route, ())

    def test_planner_bridge_reports_reservation_outcomes(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        station_ids = sorted(env._require_entity_manager().stations)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_ids[0],
                arrival_time=100.0,
                departure_time=100.0,
            ),
            BackboneVisit(
                node_id=station_ids[1],
                arrival_time=230.0,
                departure_time=230.0,
            ),
        ]
        runtime_state = env.build_runtime_state_view()
        env._planner_bridge.reset_episode()
        entity_mgr = env._require_entity_manager()
        fixed_service = max(
            env._scene_solver_params().truck_drone_launch_time_s,
            env._scene_solver_params().truck_drone_recover_time_s,
        )
        station_03_eta = env._estimate_truck_road_travel_time(
            runtime_state.truck_current_loc,
            entity_mgr.stations[station_ids[2]].location,
        )
        station_02_eta = (
            station_03_eta
            + fixed_service
            + env._estimate_truck_road_travel_time(
                entity_mgr.stations[station_ids[2]].location,
                entity_mgr.stations[station_ids[1]].location,
            )
        )

        coarse_plan = env._planner_bridge.maybe_replan(
            runtime_state,
            PlannerTriggerContext(
                t_now=0.0,
                backlog_new_orders=0,
                fallback_count_in_window=0,
                hard_failure_count_in_window=0,
                route_drift_ratio=0.0,
            ),
            reservation_constraints=(
                TruckReservationConstraint(
                    reservation_id="res-kept",
                    drone_id="drone-kept",
                    node_id=station_ids[2],
                    state=ReservationConstraintState.HARD,
                    eta_ref=station_03_eta,
                    max_eta_drift_sec=60.0,
                    issued_at=0.0,
                    related_order_id="order-kept",
                ),
                TruckReservationConstraint(
                    reservation_id="res-drifted",
                    drone_id="drone-drifted",
                    node_id=station_ids[1],
                    state=ReservationConstraintState.HARD,
                    eta_ref=station_02_eta - 30.0,
                    max_eta_drift_sec=60.0,
                    issued_at=0.0,
                    related_order_id="order-drifted",
                ),
                TruckReservationConstraint(
                    reservation_id="res-invalidated",
                    drone_id="drone-invalidated",
                    node_id=station_ids[0],
                    state=ReservationConstraintState.STRONG_HARD,
                    eta_ref=100.0,
                    max_eta_drift_sec=60.0,
                    issued_at=0.0,
                    related_order_id="order-invalidated",
                ),
            ),
        )

        outcomes = coarse_plan.reservation_outcomes
        self.assertEqual(outcomes["res-kept"].status, ReservationPlanStatus.KEPT)
        self.assertEqual(
            outcomes["res-drifted"].status,
            ReservationPlanStatus.DRIFTED,
        )
        self.assertEqual(outcomes["res-drifted"].eta_drift_sec, 30.0)
        self.assertEqual(
            outcomes["res-invalidated"].status,
            ReservationPlanStatus.INVALIDATED,
        )
        self.assertEqual(
            outcomes["res-invalidated"].invalidate_cause,
            "reservation_window_missed",
        )

    def test_env_adapter_builds_post_delivery_reservation_constraints(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=12.0,
        )
        env._reservation_count[station_id] = 1
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-RES-CONSTRAINT",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=180.0,
        )

        runtime_state = env.build_runtime_state_view()
        env._drone_state[drone_id] = TrainingDroneState.FLYING_TO_DELIVER
        constraints = env._build_truck_reservation_constraints(runtime_state)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(
            constraints[0].state,
            ReservationConstraintState.HARD,
        )
        self.assertEqual(constraints[0].node_id, station_id)

        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        constraints = env._build_truck_reservation_constraints(runtime_state)

        self.assertEqual(len(constraints), 1)
        self.assertEqual(
            constraints[0].state,
            ReservationConstraintState.STRONG_HARD,
        )
        self.assertEqual(constraints[0].node_id, station_id)
        self.assertEqual(constraints[0].eta_ref, 180.0)
        self.assertEqual(
            constraints[0].related_order_id,
            "ORDER-C-RES-CONSTRAINT",
        )

    def test_mode_c_revalidation_uses_commit_eta_when_execution_backbone_empty(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        station = env._require_entity_manager().stations[station_id]
        drone = env._require_entity_manager().drones[drone_id]
        t_now = 120.0
        flight_time = env._estimate_flight_time(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=station.location,
        )
        truck_eta = (
            t_now
            + flight_time
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        env._full_backbone_cache = []
        env._current_coarse_plan = None
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-REVALIDATE-COMMIT-ETA",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=truck_eta,
        )

        is_valid, checks = env._revalidate_mode_c_recover_node(
            drone_id=drone_id,
            node_id=station_id,
            t_now=t_now,
        )

        self.assertTrue(is_valid)
        self.assertTrue(all(checks.values()))
        debug = env.build_episode_metrics_snapshot()["mode_c_revalidation_debug_last"]
        self.assertEqual(debug["eta_source"], "dispatch_commit")
        self.assertEqual(debug["future_backbone_count"], 0)
        self.assertAlmostEqual(debug["resolved_truck_eta"], truck_eta)

    def test_mode_c_revalidation_falls_back_to_current_coarse_plan_eta(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        station = env._require_entity_manager().stations[station_id]
        drone = env._require_entity_manager().drones[drone_id]
        t_now = 120.0
        flight_time = env._estimate_flight_time(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=station.location,
        )
        truck_eta = (
            t_now
            + flight_time
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-REVALIDATE-PLAN-ETA",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=t_now - 10.0,
        )
        coarse_plan = env._build_coarse_plan_view(0.0)
        env._full_backbone_cache = []
        env._current_coarse_plan = replace(
            coarse_plan,
            truck_backbone_route=(station_id,),
            truck_eta_map={station_id: truck_eta},
            route_drift_ref={
                station_id: RouteDriftRef(
                    eta_ref=truck_eta,
                    route_index_ref=0,
                )
            },
            launch_candidate_stations=(station_id,),
        )

        is_valid, checks = env._revalidate_mode_c_recover_node(
            drone_id=drone_id,
            node_id=station_id,
            t_now=t_now,
        )

        self.assertTrue(is_valid)
        self.assertTrue(all(checks.values()))
        debug = env.build_episode_metrics_snapshot()["mode_c_revalidation_debug_last"]
        self.assertEqual(debug["eta_source"], "current_coarse_plan")
        self.assertAlmostEqual(debug["current_coarse_plan_eta"], truck_eta)

    def test_planner_kept_outcome_syncs_reservation_commit_eta(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        reservation_id = f"{drone_id}:{station_id}:10.000000"
        old_eta = 180.0
        new_eta = 220.0
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=10.0,
        )
        env._reservation_count[station_id] = 1
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-KEPT-SYNC",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=old_eta,
        )
        constraint = TruckReservationConstraint(
            reservation_id=reservation_id,
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.HARD,
            eta_ref=old_eta,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
        )
        coarse_plan = replace(
            env._build_coarse_plan_view(0.0),
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=old_eta,
                    new_eta=new_eta,
                    eta_drift_sec=new_eta - old_eta,
                    status=ReservationPlanStatus.KEPT,
                )
            },
        )

        env_decisions = env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=0.0,
        )

        self.assertEqual(env_decisions[reservation_id], "kept_commit_eta_synced")
        self.assertAlmostEqual(
            env._dispatch_commit[drone_id].planned_truck_arrival_time,
            new_eta,
        )

    def test_planner_invalidated_node_not_in_plan_releases_reservation(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        reservation_id = f"{drone_id}:{station_id}:10.000000"
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=10.0,
        )
        env._reservation_count[station_id] = 1
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-NODE-MISSING",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=180.0,
        )
        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        env._rendezvous_wait_started_at[drone_id] = 100.0
        constraint = TruckReservationConstraint(
            reservation_id=reservation_id,
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.STRONG_HARD,
            eta_ref=180.0,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
        )
        coarse_plan = replace(
            env._build_coarse_plan_view(120.0),
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=180.0,
                    new_eta=None,
                    eta_drift_sec=None,
                    status=ReservationPlanStatus.INVALIDATED,
                    invalidate_cause="node_not_in_truck_plan",
                )
            },
        )

        env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=120.0,
        )

        self.assertNotIn(drone_id, env._reservations)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.FALLBACK_RECOVERY)
        self.assertIsNone(env._dispatch_commit[drone_id].selected_recover_node)

    def test_arrived_before_eta_ref_keeps_reservation_when_uav_can_catch_new_eta(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        reservation_id = f"{drone_id}:{station_id}:10.000000"
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=10.0,
        )
        env._reservation_count[station_id] = 1
        env._schedule_return_to_rendezvous(drone_id, station_id, start_time=0.0)
        leg = env._flight_legs[drone_id]
        new_eta = (
            float(leg.arrival_time)
            + float(env._cfg.rendezvous_execution_margin_sec)
            + 5.0
        )
        old_eta = new_eta + 30.0
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=new_eta,
                departure_time=new_eta + 1e-6,
            )
        ]
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-ARRIVED-BEFORE",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=old_eta,
            planned_uav_arrival_time_lb=leg.arrival_time,
            planned_execution_slack_sec=old_eta - leg.arrival_time,
        )
        constraint = TruckReservationConstraint(
            reservation_id=reservation_id,
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.HARD,
            eta_ref=old_eta,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
        )
        coarse_plan = replace(
            env._build_coarse_plan_view(0.0),
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=old_eta,
                    new_eta=new_eta,
                    eta_drift_sec=0.0,
                    status=ReservationPlanStatus.INVALIDATED,
                    invalidate_cause="arrived_before_eta_ref",
                )
            },
        )

        env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=0.0,
        )

        self.assertIn(drone_id, env._reservations)
        self.assertNotIn(drone_id, env._fallback_leg)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.RETURN_TO_RENDEZVOUS)
        self.assertAlmostEqual(
            env._dispatch_commit[drone_id].planned_truck_arrival_time,
            new_eta,
            places=6,
        )

    def test_planner_waits_for_reservation_window_instead_of_invalidating_early_arrival(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        runtime_state = env.build_runtime_state_view()
        node_state = runtime_state.node_states[station_id]
        travel_time = env._estimate_truck_road_travel_time(
            runtime_state.truck_current_loc,
            node_state.position,
        )
        earliest_eta = travel_time + 120.0
        constraint = TruckReservationConstraint(
            reservation_id=f"{drone_id}:{station_id}:10.000000",
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.HARD,
            eta_ref=earliest_eta + 60.0,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
            earliest_eta=earliest_eta,
            latest_eta=earliest_eta + 240.0,
            preferred_eta=earliest_eta,
        )
        env._planner_bridge.reset_episode(allow_empty_backbone_route=False)

        coarse_plan = env._planner_bridge.maybe_replan(
            runtime_state,
            PlannerTriggerContext(
                t_now=env._t_now,
                backlog_new_orders=env._cfg.coarse_new_order_trigger,
                fallback_count_in_window=0,
                hard_failure_count_in_window=0,
                route_drift_ratio=0.0,
            ),
            reservation_constraints=(constraint,),
        )

        outcome = coarse_plan.reservation_outcomes[constraint.reservation_id]
        self.assertNotEqual(outcome.status, ReservationPlanStatus.INVALIDATED)
        self.assertAlmostEqual(coarse_plan.truck_eta_map[station_id], earliest_eta)

    def test_eta_late_exceeds_threshold_uses_total_wait_limit_for_waiting_uav(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        reservation_id = f"{drone_id}:{station_id}:10.000000"
        wait_started_at = 100.0
        t_now = 150.0
        old_eta = 160.0
        kept_eta = wait_started_at + float(env._cfg.rendezvous_max_wait_sec) - 1.0
        late_eta = wait_started_at + float(env._cfg.rendezvous_max_wait_sec) + 1.0
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=10.0,
        )
        env._reservation_count[station_id] = 1
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-LATE-RECHECK",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=old_eta,
        )
        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        env._rendezvous_wait_started_at[drone_id] = wait_started_at
        constraint = TruckReservationConstraint(
            reservation_id=reservation_id,
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.STRONG_HARD,
            eta_ref=old_eta,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
        )

        coarse_plan = replace(
            env._build_coarse_plan_view(t_now),
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=old_eta,
                    new_eta=kept_eta,
                    eta_drift_sec=kept_eta - old_eta,
                    status=ReservationPlanStatus.INVALIDATED,
                    invalidate_cause="eta_late_exceeds_threshold",
                )
            },
        )
        env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=t_now,
        )

        self.assertIn(drone_id, env._reservations)
        self.assertNotIn(drone_id, env._fallback_leg)
        self.assertAlmostEqual(
            env._dispatch_commit[drone_id].planned_truck_arrival_time,
            kept_eta,
            places=6,
        )

        coarse_plan = replace(
            coarse_plan,
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=old_eta,
                    new_eta=late_eta,
                    eta_drift_sec=late_eta - old_eta,
                    status=ReservationPlanStatus.INVALIDATED,
                    invalidate_cause="eta_late_exceeds_threshold",
                )
            },
        )
        env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=t_now,
        )

        self.assertNotIn(drone_id, env._reservations)
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.FALLBACK_RECOVERY)

    def test_planner_replan_event_records_reservation_env_decision(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        reservation_id = f"{drone_id}:{station_id}:10.000000"
        env._reservations[drone_id] = ReservationState(
            recover_node=station_id,
            issued_at=10.0,
        )
        env._reservation_count[station_id] = 1
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-REPLAN-LOG",
            mode=PolicyMode.C,
            selected_recover_node=station_id,
            trigger_station_id=None,
            planned_truck_arrival_time=180.0,
        )
        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
        env._rendezvous_wait_started_at[drone_id] = 100.0
        constraint = TruckReservationConstraint(
            reservation_id=reservation_id,
            drone_id=drone_id,
            node_id=station_id,
            state=ReservationConstraintState.STRONG_HARD,
            eta_ref=180.0,
            max_eta_drift_sec=60.0,
            issued_at=10.0,
            related_order_id="ORDER-C-REPLAN-LOG",
        )
        coarse_plan = replace(
            env._build_coarse_plan_view(120.0),
            reservation_outcomes={
                reservation_id: ReservationPlanOutcome(
                    reservation_id=reservation_id,
                    node_id=station_id,
                    old_eta=180.0,
                    new_eta=220.0,
                    eta_drift_sec=40.0,
                    status=ReservationPlanStatus.DRIFTED,
                )
            },
        )
        trigger_ctx = PlannerTriggerContext(
            t_now=120.0,
            backlog_new_orders=3,
            fallback_count_in_window=1,
            hard_failure_count_in_window=0,
            route_drift_ratio=0.2,
        )
        event_count_before = len(env.build_planner_replan_events_snapshot())

        env_decisions = env._apply_planner_reservation_outcomes(
            coarse_plan=coarse_plan,
            reservation_constraints=(constraint,),
            t_now=120.0,
        )
        env._record_planner_replan_event(
            coarse_plan=coarse_plan,
            trigger_ctx=trigger_ctx,
            reservation_constraints=(constraint,),
            reservation_env_decisions=env_decisions,
            applied_to_truck_route=False,
            allow_truck_route_replan=True,
            truck_route_ready_at=120.0,
        )

        events = env.build_planner_replan_events_snapshot()
        self.assertEqual(len(events), event_count_before + 1)
        self.assertEqual(
            env.build_episode_metrics_snapshot()["planner_replan_event_count"],
            event_count_before + 1,
        )
        event = events[-1]
        self.assertEqual(event["plan_version"], coarse_plan.plan_version)
        self.assertEqual(event["trigger"]["backlog_new_orders"], 3)
        self.assertEqual(
            event["reservation_outcomes"][0]["env_decision"],
            "drifted_commit_eta_synced",
        )
        self.assertEqual(event["reservation_outcomes"][0]["drone_id"], drone_id)
        self.assertAlmostEqual(
            event["reservation_outcomes"][0]["commit_planned_truck_arrival_time"],
            220.0,
        )

    def test_future_backbone_visits_exclude_after_upper_horizon(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        station_id = sorted(env._require_entity_manager().stations)[0]
        depot_id = env._require_depot().depot_id
        in_horizon_t = float(env._cfg.upper_horizon_sec) - 1.0
        out_of_horizon_t = float(env._cfg.upper_horizon_sec) + 1.0
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=in_horizon_t,
                departure_time=in_horizon_t + 1e-6,
            ),
            BackboneVisit(
                node_id=depot_id,
                arrival_time=out_of_horizon_t,
                departure_time=out_of_horizon_t + 1e-6,
            ),
        ]

        visits = env._future_backbone_visits(0.0)

        self.assertEqual([visit.node_id for visit in visits], [station_id])
        self.assertEqual(
            env._next_backbone_arrival_time_for_node(
                depot_id,
                0.0,
                include_current=False,
            ),
            None,
        )

    def test_future_backbone_visits_respect_committed_rendezvous_wait_limit(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_ids = sorted(env._require_entity_manager().stations)
        recover_node_id = station_ids[0]
        unrelated_station_id = station_ids[1]
        planned_truck_eta = 100.0
        late_recover_eta = (
            planned_truck_eta + float(env._cfg.rendezvous_max_wait_sec) + 1.0
        )
        env._reservations[drone_id] = ReservationState(
            recover_node=recover_node_id,
            issued_at=10.0,
        )
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-FUTURE-LIMIT",
            mode=PolicyMode.C,
            selected_recover_node=recover_node_id,
            trigger_station_id=None,
            planned_truck_arrival_time=planned_truck_eta,
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=late_recover_eta,
                departure_time=late_recover_eta + 1e-6,
            ),
            BackboneVisit(
                node_id=unrelated_station_id,
                arrival_time=late_recover_eta,
                departure_time=late_recover_eta + 1e-6,
            ),
        ]

        visits = env._future_backbone_visits(0.0)

        self.assertEqual([visit.node_id for visit in visits], [unrelated_station_id])
        self.assertEqual(
            env._next_backbone_arrival_time_for_node(
                recover_node_id,
                0.0,
                include_current=False,
            ),
            None,
        )

    def test_advance_to_upper_horizon_does_not_rebuild_empty_backbone_plan(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        self._inject_order_at(
            env,
            order_id="ORDER-P6-HORIZON-PENDING",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        env._full_backbone_cache = []
        env._decision_queue.clear()
        env._t_now = float(env._cfg.upper_horizon_sec) - 1.0

        env._advance_to_event(float(env._cfg.upper_horizon_sec))

        self.assertTrue(env.is_done())
        self.assertEqual(env._episode_done_reason(), "upper_horizon_reached")

    def test_idle_wait_to_upper_horizon_does_not_resume_decision(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_id = sorted(entity_mgr.stations)[0]
        self._inject_order_at(
            env,
            order_id="ORDER-P6-HORIZON-WAIT",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        env._t_now = float(env._cfg.upper_horizon_sec) - 1.0
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=float(env._cfg.upper_horizon_sec),
                departure_time=float(env._cfg.upper_horizon_sec) + 1e-6,
            )
        ]
        env._decision_queue.clear()
        env._enqueue_decision(drone_id, "test_idle_near_horizon", None)
        self.assertTrue(env._decision_queue)

        result = env.step(WAIT_ACTION)

        self.assertTrue(result.done)
        self.assertEqual(env._episode_done_reason(), "upper_horizon_reached")
        self.assertFalse(env._decision_queue)

    def test_tail_empty_backbone_without_orders_degrades_to_empty_plan(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        env._t_now = float(env._cfg.upper_horizon_sec) - 5.0
        env._full_backbone_cache = []

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            force_replan=True,
        )

        self.assertTrue(coarse_plan.allow_empty_backbone_route)
        self.assertEqual(coarse_plan.truck_backbone_route, ())
        self.assertEqual(coarse_plan.policy_mode_mask, {})

    def test_empty_backbone_allowed_when_no_orders_and_no_mode_c_obligation(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        env._full_backbone_cache = []

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            force_replan=True,
        )

        self.assertTrue(coarse_plan.allow_empty_backbone_route)
        self.assertEqual(coarse_plan.truck_backbone_route, ())
        self.assertEqual(env._episode_done_reason(), "all_orders_cleared")

    def test_all_orders_cleared_waits_for_active_mode_c_recovery_obligation(self) -> None:
        env, drone_id = self._reset_controlled_env()
        recover_node_id = sorted(env._require_entity_manager().stations)[0]
        env._reservations[drone_id] = ReservationState(
            recover_node=recover_node_id,
            issued_at=10.0,
        )
        env._dispatch_commit[drone_id] = DispatchCommit(
            order_id="ORDER-C-ACTIVE-OBLIGATION",
            mode=PolicyMode.C,
            selected_recover_node=recover_node_id,
            trigger_station_id=None,
            planned_truck_arrival_time=180.0,
        )
        env._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK

        self.assertIsNone(env._episode_done_reason())

        env._release_reservation(drone_id, cause="test_clear")
        env._dispatch_commit.pop(drone_id, None)
        env._drone_state[drone_id] = TrainingDroneState.IDLE
        self.assertEqual(env._episode_done_reason(), "all_orders_cleared")

    def test_all_orders_cleared_waits_for_future_scheduled_dynamic_orders(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr._next_order_time = math.inf
        order_mgr._scheduled_dynamic = [
            {
                "order_id": "ORDER-FUTURE-DYNAMIC",
                "spawn_sim_s": env._t_now + 60.0,
            }
        ]
        order_mgr._scheduled_dynamic_i = 0

        self.assertIsNone(env._episode_done_reason())
        self.assertFalse(env.is_done())

        order_mgr._scheduled_dynamic_i = len(order_mgr._scheduled_dynamic)
        self.assertEqual(env._episode_done_reason(), "all_orders_cleared")
        self.assertTrue(env.is_done())

    def test_recovery_only_truck_plan_skips_patrol_station_coverage(self) -> None:
        env, drone_id = self._reset_controlled_env()
        station_ids = sorted(env._require_entity_manager().stations)
        recover_node_id = station_ids[0]
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_id,
                arrival_time=100.0 + idx * 100.0,
                departure_time=100.0 + idx * 100.0 + 1e-6,
            )
            for idx, station_id in enumerate(station_ids[1:7], start=1)
        ]
        runtime_state = env.build_runtime_state_view()
        constraint = TruckReservationConstraint(
            reservation_id=f"{drone_id}:{recover_node_id}:10.0",
            drone_id=drone_id,
            node_id=recover_node_id,
            state=ReservationConstraintState.STRONG_HARD,
            eta_ref=0.0,
            max_eta_drift_sec=1.0e9,
            issued_at=10.0,
            related_order_id="ORDER-C-RECOVERY-ONLY",
        )
        env._planner_bridge.reset_episode(allow_empty_backbone_route=False)

        coarse_plan = env._planner_bridge.maybe_replan(
            runtime_state,
            PlannerTriggerContext(
                t_now=env._t_now,
                backlog_new_orders=env._cfg.coarse_new_order_trigger,
                fallback_count_in_window=0,
                hard_failure_count_in_window=0,
                route_drift_ratio=0.0,
            ),
            reservation_constraints=(constraint,),
        )

        station_stops = [
            stop for stop in coarse_plan.truck_plan_stops if stop.node_type == "station"
        ]
        self.assertEqual([stop.node_id for stop in station_stops], [recover_node_id])
        self.assertEqual(coarse_plan.truck_plan_stops[-1].node_type, "depot")

    def test_station_backbone_replan_inserts_stations_before_depot_when_orders_remain(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        truck = env._require_truck()
        depot = env._require_depot()
        current_station = entity_mgr.stations[sorted(entity_mgr.stations)[0]]
        drone = entity_mgr.drones[drone_id]
        t_now = float(env._cfg.upper_horizon_sec) - 140.0
        depot_after_horizon_t = float(env._cfg.upper_horizon_sec) + 30.0
        truck.current_loc = current_station.location
        env._t_now = t_now
        self._inject_order_at(
            env,
            order_id="ORDER-P6-REBUILD-STATION-BACKBONE",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        env._planned_route_stops = [
            PlannedStop(
                seq=0,
                node_type="truck_current",
                node_id="truck_current",
                position=truck.current_loc,
                order_id=None,
                arrival_time=t_now,
                departure_time=t_now,
            ),
            PlannedStop(
                seq=1,
                node_type="station",
                node_id=current_station.station_id,
                position=current_station.location,
                order_id=None,
                arrival_time=t_now,
                departure_time=t_now + 10.0,
            ),
            PlannedStop(
                seq=2,
                node_type="depot",
                node_id=depot.depot_id,
                position=depot.location,
                order_id=None,
                arrival_time=depot_after_horizon_t,
                departure_time=depot_after_horizon_t,
            ),
        ]
        env._planned_route_segments = []
        env._planned_route_stop_i = 1
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=current_station.station_id,
                arrival_time=t_now,
                departure_time=t_now + 10.0,
            ),
            BackboneVisit(
                node_id=depot.depot_id,
                arrival_time=depot_after_horizon_t,
                departure_time=depot_after_horizon_t + 1e-6,
            ),
        ]
        env._planner_bridge.reset_episode(allow_empty_backbone_route=False)
        env._current_coarse_plan = None

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=True,
        )

        future_station_stops = [
            stop
            for stop in coarse_plan.truck_plan_stops
            if stop.node_type == "station" and stop.arrival_time > t_now + 1e-6
        ]
        self.assertTrue(future_station_stops)
        self.assertEqual(coarse_plan.truck_plan_stops[-1].node_type, "depot")
        self.assertTrue(env._future_backbone_visits(t_now))

    def test_future_station_capacity_trigger_counts_station_only(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 3)
        depot = env._require_depot()
        drone = entity_mgr.drones[drone_id]
        t_now = 100.0
        env._t_now = t_now
        self._inject_order_at(
            env,
            order_id="ORDER-P6-STATION-CAPACITY-STATION-ONLY",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_ids[0],
                arrival_time=t_now + 60.0,
                departure_time=t_now + 60.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_ids[1],
                arrival_time=t_now + 120.0,
                departure_time=t_now + 120.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=depot.depot_id,
                arrival_time=t_now + 180.0,
                departure_time=t_now + 180.0 + 1e-6,
            ),
        ]
        env._truck_replan_pending = False
        env._truck_replan_pending_reasons.clear()

        require_station_backbone = env._ensure_future_backbone_capacity(
            env.build_runtime_state_view()
        )

        self.assertTrue(require_station_backbone)
        self.assertTrue(env._truck_replan_pending)
        self.assertIn("future_backbone_capacity", env._truck_replan_pending_reasons)
        self.assertEqual(len(env._future_station_backbone_visits(t_now)), 2)
        self.assertEqual(len(env._future_backbone_visits(t_now)), 3)

    def test_future_station_capacity_trigger_is_shared_by_benchmark_flow(self) -> None:
        scene_ctx = load_default_scene()
        benchmark_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.BENCHMARK,
        )
        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=benchmark_source,
        )
        env.reset()

        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        order_mgr._next_order_time = math.inf
        order_mgr._scheduled_dynamic = []
        order_mgr._scheduled_dynamic_i = 0

        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 3)
        depot = env._require_depot()
        drone_id = self._first_idle_drone_id(env)
        drone = entity_mgr.drones[drone_id]
        t_now = 100.0
        env._t_now = t_now
        self._inject_order_at(
            env,
            order_id="ORDER-P6-BENCHMARK-STATION-CAPACITY",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_ids[0],
                arrival_time=t_now + 60.0,
                departure_time=t_now + 60.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_ids[1],
                arrival_time=t_now + 120.0,
                departure_time=t_now + 120.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=depot.depot_id,
                arrival_time=t_now + 180.0,
                departure_time=t_now + 180.0 + 1e-6,
            ),
        ]
        self.assertTrue(env._allow_empty_backbone_route)

        require_station_backbone = env._ensure_future_backbone_capacity(
            env.build_runtime_state_view()
        )

        self.assertTrue(require_station_backbone)
        self.assertTrue(env._truck_replan_pending)
        self.assertIn("future_backbone_capacity", env._truck_replan_pending_reasons)

    def test_truck_replan_after_station_arrival_starts_after_current_stop_service(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        truck = env._require_truck()
        depot = env._require_depot()
        current_station = entity_mgr.stations[sorted(entity_mgr.stations)[0]]
        t_now = 100.0
        service_finish = t_now + float(
            max(
                env._scene_solver_params().truck_drone_launch_time_s,
                env._scene_solver_params().truck_drone_recover_time_s,
            )
        )
        truck.current_loc = current_station.location
        env._t_now = t_now
        env._planned_route_stops = [
            PlannedStop(
                seq=0,
                node_type="truck_current",
                node_id="truck_current",
                position=truck.current_loc,
                order_id=None,
                arrival_time=t_now,
                departure_time=t_now,
            ),
            PlannedStop(
                seq=1,
                node_type="station",
                node_id=current_station.station_id,
                position=current_station.location,
                order_id=None,
                arrival_time=t_now,
                departure_time=service_finish,
            ),
            PlannedStop(
                seq=2,
                node_type="depot",
                node_id=depot.depot_id,
                position=depot.location,
                order_id=None,
                arrival_time=env._cfg.upper_horizon_sec + 1.0,
                departure_time=env._cfg.upper_horizon_sec + 1.0,
            ),
        ]
        env._planned_route_segments = []
        # Simulate the event loop state after the current station arrival was collected.
        env._planned_route_stop_i = 2
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=depot.depot_id,
                arrival_time=env._cfg.upper_horizon_sec + 1.0,
                departure_time=env._cfg.upper_horizon_sec + 1.0 + 1e-6,
            )
        ]
        self._inject_order_at(
            env,
            order_id="ORDER-P6-TRUCK-READY-AFTER-SERVICE",
            position=current_station.location,
            deadline_offset=7200.0,
            payload_weight=env._heavy_payload_capacity + 1.0,
        )
        env._planner_bridge.reset_episode(allow_empty_backbone_route=False)
        env._current_coarse_plan = None

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=True,
            force_replan=True,
        )

        self.assertGreaterEqual(
            coarse_plan.truck_plan_stops[0].arrival_time,
            service_finish,
        )
        self.assertAlmostEqual(
            coarse_plan.truck_plan_stops[0].departure_time,
            coarse_plan.truck_plan_stops[0].arrival_time
            + float(env._scene_solver_params().truck_service_time_order_s),
            places=6,
        )
        self.assertAlmostEqual(
            env._planned_route_segments[0].start_time,
            service_finish,
            places=6,
        )

    def test_deferred_truck_replan_allows_temporary_empty_backbone(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        normal_order = self._inject_order_at(
            env,
            order_id="ORDER-P6-DEFERRED-B-ONLY",
            position=Position3D(
                x=drone.current_loc.x + 100.0,
                y=drone.current_loc.y + 50.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=1.0,
        )
        truck_only_order = self._inject_order_at(
            env,
            order_id="TRUCKONLY-P6-DEFERRED-EMPTY-BACKBONE",
            position=Position3D(
                x=drone.current_loc.x + 200.0,
                y=drone.current_loc.y + 100.0,
                z=drone.current_loc.z,
            ),
            deadline_offset=7200.0,
            payload_weight=env._heavy_payload_capacity + 1.0,
        )
        env._full_backbone_cache = []
        env._planner_bridge.reset_episode(allow_empty_backbone_route=False)
        env._current_coarse_plan = None

        coarse_plan = env._refresh_coarse_plan_if_needed(
            env.build_runtime_state_view(),
            allow_truck_route_replan=False,
        )

        self.assertTrue(env._truck_replan_pending)
        self.assertTrue(coarse_plan.allow_empty_backbone_route)
        self.assertEqual(coarse_plan.truck_backbone_route, ())
        self.assertEqual(
            coarse_plan.policy_mode_mask[normal_order.order_id],
            frozenset({PolicyMode.B}),
        )
        self.assertNotIn(truck_only_order.order_id, coarse_plan.policy_mode_mask)

    def test_candidate_builder_keeps_masks_and_lookup_aligned(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        station_fast = entity_mgr.stations[station_ids[0]]
        station_safe = entity_mgr.stations[station_ids[1]]

        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-CAND-01",
            position=Position3D(
                x=drone.current_loc.x + 10.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        t_deliver_finish = t_deliver + env._scene_solver_params().drone_service_time_order_s
        safe_recover_fly = env._estimate_flight_time(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=station_safe.location,
        )
        t_safe_truck = (
            t_deliver_finish
            + safe_recover_fly
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_fast.station_id,
                arrival_time=t_deliver + 1.0,
                departure_time=t_deliver + 1.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_safe.station_id,
                arrival_time=t_safe_truck,
                departure_time=t_safe_truck + 1e-6,
            ),
        ]

        runtime_state = env.build_runtime_state_view()
        coarse_plan = env._build_coarse_plan_view(env._t_now)
        candidate_out = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            deciding_drone_id=drone_id,
            trigger_type="test_idle",
            trigger_station_id=None,
            last_seen_plan_version=-1,
        )

        self.assertEqual(candidate_out.root_branch_mask, (True, True))
        self.assertTrue(candidate_out.has_wait_action)
        self.assertEqual(candidate_out.candidate_features.uav_self.plan_version_delta, 1)
        self.assertTrue(candidate_out.order_mask[0])
        self.assertFalse(any(candidate_out.order_mask[1:]))
        self.assertTrue(candidate_out.mode_mask[0][0], "mode B 应保留")
        self.assertTrue(candidate_out.mode_mask[0][1], "mode C 应保留")
        order_feature = candidate_out.candidate_features.order_features[0]
        self.assertTrue(order_feature.has_mode_c_action)
        self.assertEqual(order_feature.mode_c_candidate_count, 1)
        self.assertAlmostEqual(
            order_feature.best_mode_c_rendezvous_margin,
            30.0,
        )
        self.assertAlmostEqual(
            order_feature.best_mode_c_wait_time,
            env._cfg.rendezvous_execution_margin_sec + 30.0,
        )
        self.assertAlmostEqual(
            order_feature.best_mode_c_uav_flight_time,
            safe_recover_fly,
        )
        self.assertGreater(order_feature.best_mode_c_energy_margin_ratio, 0.0)
        self.assertEqual(
            order_feature.best_mode_c_node_type,
            "station",
        )
        self.assertAlmostEqual(
            order_feature.best_mode_c_truck_eta_remaining,
            t_safe_truck - runtime_state.t_now,
        )
        self.assertAlmostEqual(
            order_feature.best_mode_c_timeout_risk,
            1.0 - 30.0 / env._cfg.rendezvous_max_wait_sec,
        )
        tensorizer = ObservationTensorizer()
        order_tokens, order_padding_mask = tensorizer._build_order_tokens((order_feature,))
        self.assertFalse(order_padding_mask[0])
        self.assertEqual(order_tokens.shape[1], len(ORDER_TOKEN_FIELDS))
        self.assertAlmostEqual(
            order_tokens[
                0,
                ORDER_TOKEN_FIELDS.index("mode_c_candidate_count_norm"),
            ],
            1.0 / tensorizer._cfg.max_candidate_recovery_per_order,
        )
        self.assertAlmostEqual(
            order_tokens[
                0,
                ORDER_TOKEN_FIELDS.index("best_mode_c_wait_time_norm"),
            ],
            (
                env._cfg.rendezvous_execution_margin_sec + 30.0
            )
            / env._cfg.upper_horizon_sec,
        )
        self.assertAlmostEqual(
            order_tokens[
                0,
                ORDER_TOKEN_FIELDS.index("best_mode_c_energy_margin_ratio"),
            ],
            min(1.0, order_feature.best_mode_c_energy_margin_ratio),
        )
        self.assertAlmostEqual(
            order_tokens[
                0,
                ORDER_TOKEN_FIELDS.index("best_mode_c_timeout_risk_norm"),
            ],
            order_feature.best_mode_c_timeout_risk,
        )

        resolved_wait = candidate_out.resolved_action_lookup.resolve(root_branch_idx=0)
        resolved_mode_c = candidate_out.resolved_action_lookup.resolve(
            root_branch_idx=1,
            order_idx=0,
            mode_idx=1,
        )
        self.assertEqual(resolved_wait, WAIT_ACTION)
        self.assertEqual(resolved_mode_c.order_id, order.order_id)
        self.assertEqual(resolved_mode_c.mode, PolicyMode.C)
        self.assertEqual(resolved_mode_c.recover_node_id, station_safe.station_id)

    def test_candidate_builder_deadline_sort_centers_lead_time(self) -> None:
        items = [
            self._candidate_sort_item("A", 480.0),
            self._candidate_sort_item("B", 300.0),
            self._candidate_sort_item("C", 600.0),
            self._candidate_sort_item("D", 60.0),
            self._candidate_sort_item("E", -60.0),
            self._candidate_sort_item("F", -600.0),
            self._candidate_sort_item("G", -1800.0),
            self._candidate_sort_item("H", 1800.0),
        ]

        sorted_ids = [
            item["order_id"] for item in sorted(items, key=_candidate_deadline_sort_key)
        ]

        self.assertEqual(sorted_ids[0], "A")
        self.assertLess(sorted_ids.index("B"), 4)
        self.assertLess(sorted_ids.index("C"), 4)
        self.assertLess(sorted_ids.index("E"), sorted_ids.index("F"))
        self.assertLess(sorted_ids.index("E"), sorted_ids.index("G"))
        self.assertLess(sorted_ids.index("D"), sorted_ids.index("G"))
        self.assertEqual(sorted_ids[-1], "H")

    def test_candidate_builder_overdue_reserve_keeps_limited_visibility(self) -> None:
        items = [
            self._candidate_sort_item("A", 480.0),
            self._candidate_sort_item("B", 500.0),
            self._candidate_sort_item("C", 460.0),
            self._candidate_sort_item("D", 450.0),
            self._candidate_sort_item("E", -60.0),
            self._candidate_sort_item("F", -600.0),
            self._candidate_sort_item("G", -1800.0),
            self._candidate_sort_item("H", 1800.0),
        ]

        selected = _select_candidate_orders(items, max_candidate_orders=5)
        selected_ids = [item["order_id"] for item in selected]

        self.assertEqual(len(selected_ids), 5)
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertEqual(set(selected_ids[:3]), {"A", "B", "C"})
        self.assertIn("E", selected_ids)
        self.assertIn("F", selected_ids)
        self.assertNotIn("G", selected_ids)

        all_selected = _select_candidate_orders(
            items,
            max_candidate_orders=len(items),
        )
        self.assertEqual(
            {item["order_id"] for item in all_selected},
            {item["order_id"] for item in items},
        )

    def test_candidate_builder_mode_c_prefers_earliest_recovery_time(self) -> None:
        builder = CandidateBuilder.__new__(CandidateBuilder)
        builder._cfg = SimpleNamespace(
            rendezvous_execution_margin_sec=10.0,
            rendezvous_max_wait_sec=1000.0,
        )
        builder._safe_margin_j_by_drone = {"DRN-1": 0.0}

        flight_time_by_node = {
            "early": 100.0,
            "late": 400.0,
        }

        def estimate_leg(*, drone_view, from_pos, to_pos, payload):  # noqa: ANN001
            del drone_view, from_pos, payload
            return SimpleNamespace(
                flight_time_sec=flight_time_by_node[to_pos.node_id],
                energy_j=10.0,
            )

        builder._estimate_uav_leg = estimate_leg
        runtime_state = SimpleNamespace(
            t_now=0.0,
            node_states={
                "early": SimpleNamespace(
                    node_id="early",
                    node_type="station",
                    position=SimpleNamespace(node_id="early"),
                ),
                "late": SimpleNamespace(
                    node_id="late",
                    node_type="station",
                    position=SimpleNamespace(node_id="late"),
                ),
            },
        )
        coarse_plan = SimpleNamespace(
            truck_eta_map={
                "early": 500.0,
                "late": 550.0,
            },
            get_recovery_candidates=lambda _order_id: ("early", "late"),
        )

        summary = builder._select_best_mode_c_recovery(
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            drone_view=SimpleNamespace(drone_id="DRN-1", battery_max=100.0),
            order=SimpleNamespace(order_id="ORDER-1"),
            deliver_pos=SimpleNamespace(),
            t_deliver_finish=0.0,
            energy_after_delivery=100.0,
            delivery_energy_ratio=0.2,
            trigger_type="test_idle",
            trigger_station_id=None,
            t_now=0.0,
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.candidate_count, 2)
        self.assertAlmostEqual(summary.best_truck_eta_remaining, 500.0)
        self.assertAlmostEqual(summary.best_uav_flight_time, 100.0)
        self.assertAlmostEqual(summary.best_recovery_energy_j, 10.0)
        self.assertAlmostEqual(summary.best_recovery_energy_ratio, 0.1)
        self.assertAlmostEqual(summary.best_total_energy_ratio, 0.3)
        self.assertAlmostEqual(summary.best_wait_time, 400.0)

    def test_public_candidate_output_entry_rebuilds_from_decision_context(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        station_fast = entity_mgr.stations[station_ids[0]]
        station_safe = entity_mgr.stations[station_ids[1]]

        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-PUBLIC-01",
            position=Position3D(
                x=drone.current_loc.x + 10.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_fast.station_id,
                arrival_time=t_deliver + 1.0,
                departure_time=t_deliver + 1.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_safe.station_id,
                arrival_time=t_deliver + 600.0,
                departure_time=t_deliver + 600.0 + 1e-6,
            ),
        ]

        env._enqueue_initial_idle_decisions()
        decision_context = env.current_decision_context
        self.assertIsNotNone(decision_context)

        candidate_out = env.build_candidate_output(
            decision_context,
            last_seen_plan_version=-1,
        )

        self.assertEqual(
            candidate_out.resolved_action_lookup.as_action_lookup(),
            decision_context.action_lookup,
        )
        self.assertEqual(
            candidate_out.candidate_features.order_features[0].order_id,
            order.order_id,
        )
        self.assertEqual(
            candidate_out.candidate_features.uav_self.plan_version_delta,
            decision_context.coarse_plan.plan_version + 1,
        )

    def test_riding_with_truck_alias_uses_runtime_recovery_pool(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        station_1 = entity_mgr.stations[station_ids[0]]
        station_2 = entity_mgr.stations[station_ids[1]]

        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-RIDE-ALIAS-01",
            position=Position3D(
                x=station_2.location.x + 10.0,
                y=station_2.location.y,
                z=station_2.location.z,
            ),
        )

        runtime_state = env.build_runtime_state_view()
        coarse_plan = env._build_coarse_plan_view(env._t_now)
        t_deliver_finish = env._estimate_delivery_finish_time(drone, order)
        station_1_eta = (
            t_deliver_finish
            + env._estimate_flight_time(
                drone=drone,
                from_pos=order.delivery_loc,
                to_pos=station_1.location,
            )
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        station_2_eta = runtime_state.t_now
        custom_plan = replace(
            coarse_plan,
            truck_backbone_route=(station_1.station_id, station_2.station_id),
            truck_eta_map={
                station_1.station_id: station_1_eta,
                station_2.station_id: station_2_eta,
            },
            route_drift_ref={
                station_1.station_id: RouteDriftRef(
                    eta_ref=station_1_eta,
                    route_index_ref=0,
                ),
                station_2.station_id: RouteDriftRef(
                    eta_ref=station_2_eta,
                    route_index_ref=1,
                ),
            },
            recovery_pool={
                **coarse_plan.recovery_pool,
                order.order_id: (station_1.station_id, station_2.station_id),
            },
            launch_candidate_stations=(station_1.station_id, station_2.station_id),
        )

        candidate_out_alias = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=custom_plan,
            deciding_drone_id=drone_id,
            trigger_type="riding_with_truck",
            trigger_station_id=station_2.station_id,
            last_seen_plan_version=custom_plan.plan_version,
        )
        candidate_out_internal = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=custom_plan,
            deciding_drone_id=drone_id,
            trigger_type="truck_station_arrival",
            trigger_station_id=station_2.station_id,
            last_seen_plan_version=custom_plan.plan_version,
        )
        candidate_out_idle = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=custom_plan,
            deciding_drone_id=drone_id,
            trigger_type="test_idle",
            trigger_station_id=None,
            last_seen_plan_version=custom_plan.plan_version,
        )

        alias_order_feature = next(
            feature
            for feature in candidate_out_alias.candidate_features.order_features
            if feature.order_id == order.order_id
        )
        internal_order_feature = next(
            feature
            for feature in candidate_out_internal.candidate_features.order_features
            if feature.order_id == order.order_id
        )
        idle_order_feature = next(
            feature
            for feature in candidate_out_idle.candidate_features.order_features
            if feature.order_id == order.order_id
        )
        self.assertFalse(alias_order_feature.has_mode_c_action)
        self.assertFalse(internal_order_feature.has_mode_c_action)
        self.assertTrue(idle_order_feature.has_mode_c_action)
        self.assertAlmostEqual(
            idle_order_feature.best_mode_c_truck_eta_remaining,
            station_1_eta - runtime_state.t_now,
        )

    def test_riding_with_truck_trigger_requires_station_id(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        self._inject_order_at(
            env,
            order_id="ORDER-P6-RIDE-REQ-01",
            position=Position3D(
                x=drone.current_loc.x + 10.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        runtime_state = env.build_runtime_state_view()
        coarse_plan = env._build_coarse_plan_view(env._t_now)

        with self.assertRaises(ValueError):
            env._candidate_builder.build(
                runtime_state=runtime_state,
                coarse_plan=coarse_plan,
                deciding_drone_id=drone_id,
                trigger_type="riding_with_truck",
                trigger_station_id=None,
                last_seen_plan_version=coarse_plan.plan_version,
            )

    def test_candidate_builder_respects_policy_mode_mask(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        station_1 = entity_mgr.stations[station_ids[0]]
        station_2 = entity_mgr.stations[station_ids[1]]

        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-POLICY-01",
            position=Position3D(
                x=drone.current_loc.x + 10.0,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
        )
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_1.station_id,
                arrival_time=t_deliver + 300.0,
                departure_time=t_deliver + 300.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_2.station_id,
                arrival_time=t_deliver + 600.0,
                departure_time=t_deliver + 600.0 + 1e-6,
            ),
        ]

        runtime_state = env.build_runtime_state_view()
        coarse_plan = env._build_coarse_plan_view(env._t_now)
        restricted_plan = replace(
            coarse_plan,
            policy_mode_mask={
                **coarse_plan.policy_mode_mask,
                order.order_id: frozenset({PolicyMode.B}),
            },
        )
        candidate_out = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=restricted_plan,
            deciding_drone_id=drone_id,
            trigger_type="test_idle",
            trigger_station_id=None,
            last_seen_plan_version=restricted_plan.plan_version,
        )

        self.assertTrue(candidate_out.order_mask[0])
        self.assertEqual(candidate_out.mode_mask[0], (True, False))
        self.assertEqual(
            candidate_out.candidate_features.order_features[0].order_id,
            order.order_id,
        )
        self.assertTrue(
            candidate_out.candidate_features.order_features[0].has_mode_b_action
        )
        with self.assertRaises(KeyError):
            candidate_out.resolved_action_lookup.resolve(
                root_branch_idx=1,
                order_idx=0,
                mode_idx=1,
            )

    def test_riding_wait_tracks_actual_elapsed_until_trigger_station(self) -> None:
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
        env._planner_bridge = None
        env._current_coarse_plan = None

        drone_id = self._first_riding_drone_id(env)
        entity_mgr = env._require_entity_manager()
        truck = env._require_truck()
        depot = env._require_depot()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 2)
        station_1 = entity_mgr.stations[station_ids[0]]
        station_2 = entity_mgr.stations[station_ids[1]]

        env._planned_route_stops = [
            PlannedStop(
                seq=0,
                node_type="depot",
                node_id=depot.depot_id,
                position=Position3D(
                    x=truck.current_loc.x,
                    y=truck.current_loc.y,
                    z=truck.current_loc.z,
                ),
                order_id=None,
                arrival_time=0.0,
                departure_time=1e-6,
            ),
            PlannedStop(
                seq=1,
                node_type="station",
                node_id=station_1.station_id,
                position=station_1.location,
                order_id=None,
                arrival_time=60.0,
                departure_time=60.0 + 1e-6,
            ),
            PlannedStop(
                seq=2,
                node_type="station",
                node_id=station_2.station_id,
                position=station_2.location,
                order_id=None,
                arrival_time=120.0,
                departure_time=120.0 + 1e-6,
            ),
        ]
        env._planned_route_stop_i = 1
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_1.station_id,
                arrival_time=60.0,
                departure_time=60.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_2.station_id,
                arrival_time=120.0,
                departure_time=120.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=depot.depot_id,
                arrival_time=180.0,
                departure_time=180.0 + 1e-6,
            ),
        ]

        self._inject_order_at(
            env,
            order_id="ORDER-P6-RIDE-WAIT-01",
            position=Position3D(
                x=truck.current_loc.x + 20.0,
                y=truck.current_loc.y,
                z=truck.current_loc.z,
            ),
            deadline_offset=7200.0,
        )
        for other_drone_id, state in list(env._drone_state.items()):
            if other_drone_id == drone_id:
                continue
            if state == TrainingDroneState.IDLE:
                env._drone_state[other_drone_id] = TrainingDroneState.ACTIVE_WAIT
                env._active_wait_resume[other_drone_id] = TrainingDroneState.IDLE

        env._active_launch_stations = {station_2.station_id}
        env._enqueue_decision(
            drone_id,
            trigger_type="truck_station_arrival",
            trigger_station_id=station_1.station_id,
        )
        self.assertTrue(env._decision_queue)

        t_start = env._t_now
        result = env.step(WAIT_ACTION)
        elapsed = env._t_now - t_start
        expected_elapsed = 120.0 - t_start

        self.assertAlmostEqual(elapsed, expected_elapsed, places=6)
        self.assertEqual(
            result.info["wait_mode"],
            "deferred_riding_with_truck",
        )
        self.assertNotIn("wait_delta", result.info)
        self.assertAlmostEqual(
            result.reward,
            -env._cfg.wait_idle_penalty_coef * expected_elapsed,
            places=6,
        )
        self.assertIn(
            env._drone_state[drone_id],
            {TrainingDroneState.RIDING_WITH_TRUCK, TrainingDroneState.ACTIVE_WAIT},
        )
        self.assertTrue(
            any(
                trigger.drone_id == drone_id
                and trigger.trigger_type == "truck_station_arrival"
                and trigger.trigger_station_id == station_2.station_id
                for trigger in env._decision_queue
            )
        )

    def test_candidate_builder_riding_trigger_applies_launch_delay_and_delivery_service(self) -> None:
        env = self._make_env()
        env.reset()

        drone_id = self._first_riding_drone_id(env)
        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        launch_station = entity_mgr.stations[station_ids[0]]
        recover_station = entity_mgr.stations[station_ids[1]]
        order = self._inject_order_at(
            env,
            order_id="ORDER-P6-LAUNCH-SVC-01",
            position=Position3D(
                x=launch_station.location.x + 10.0,
                y=launch_station.location.y,
                z=launch_station.location.z,
            ),
        )

        runtime_state = env.build_runtime_state_view()
        launch_time = (
            runtime_state.t_now + env._scene_solver_params().truck_drone_launch_time_s
        )
        deliver_fly = env._estimate_flight_time(
            drone=entity_mgr.drones[drone_id],
            from_pos=launch_station.location,
            to_pos=order.delivery_loc,
        )
        recover_fly = env._estimate_flight_time(
            drone=entity_mgr.drones[drone_id],
            from_pos=order.delivery_loc,
            to_pos=recover_station.location,
        )
        truck_eta = (
            launch_time
            + deliver_fly
            + recover_fly
            + env._cfg.rendezvous_execution_margin_sec
            + env._scene_solver_params().drone_service_time_order_s
            - 1.0
        )
        coarse_plan = env._build_coarse_plan_view(env._t_now)
        custom_plan = replace(
            coarse_plan,
            truck_backbone_route=(launch_station.station_id, recover_station.station_id),
            truck_eta_map={
                launch_station.station_id: runtime_state.t_now,
                recover_station.station_id: truck_eta,
            },
            authorized_orders=(order.order_id,),
            order_priority_band={order.order_id: coarse_plan.order_priority_band[order.order_id]},
            order_pre_score={order.order_id: coarse_plan.order_pre_score[order.order_id]},
            planner_mode_cap={order.order_id: coarse_plan.planner_mode_cap[order.order_id]},
            policy_mode_mask={order.order_id: frozenset({PolicyMode.B, PolicyMode.C})},
            route_drift_ref={
                launch_station.station_id: RouteDriftRef(
                    eta_ref=runtime_state.t_now,
                    route_index_ref=0,
                ),
                recover_station.station_id: RouteDriftRef(
                    eta_ref=truck_eta,
                    route_index_ref=1,
                )
            },
            recovery_pool={
                order.order_id: (recover_station.station_id,),
            },
            launch_candidate_stations=(launch_station.station_id,),
        )

        candidate_out = env._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=custom_plan,
            deciding_drone_id=drone_id,
            trigger_type="truck_station_arrival",
            trigger_station_id=launch_station.station_id,
            last_seen_plan_version=custom_plan.plan_version,
        )

        self.assertTrue(candidate_out.order_mask[0])
        self.assertEqual(candidate_out.mode_mask[0], (True, False))


if __name__ == "__main__":
    unittest.main()
