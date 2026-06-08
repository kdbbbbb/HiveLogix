#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5b env adapter tests.

运行方式：
  python -m unittest backend.training.test_env_adapter_phase5b
"""

from __future__ import annotations

import math
import unittest

import networkx as nx

from core.entities.order import Order
from core.entities.primitives import Position3D
from environment.geo.osm_service import find_nearest_node
from utils.coord_utils import wgs84_to_utm

from .contracts import PolicyMode
from .env_adapter import (
    ActiveTruckRouteContext,
    BackboneVisit,
    DispatchAction,
    FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
    _build_route_from_active_truck_context,
    TrainingDroneState,
    TrainingEnvAdapter,
)
from .uav_path_service import UavPathEstimate
from .actions import WAIT_ACTION


class TestTrainingEnvAdapterPhase5b(unittest.TestCase):
    def _make_env(self) -> TrainingEnvAdapter:
        return TrainingEnvAdapter()

    def test_active_truck_route_context_uses_downstream_osm_path_not_nearest_trap(self) -> None:
        road_nodes = {
            "route_a": (121.00000, 31.00000),
            "route_b": (121.00010, 31.00000),
            "target": (121.00020, 31.00000),
            "trap": (121.000055, 31.000005),
        }
        graph = nx.DiGraph()
        graph.add_edge("route_a", "route_b", weight=10.0)
        graph.add_edge("route_b", "target", weight=10.0)
        graph.add_edge("target", "route_b", weight=10.0)
        graph.add_node("trap")

        route_a_x, route_a_y = wgs84_to_utm(*road_nodes["route_a"])
        route_b_x, route_b_y = wgs84_to_utm(*road_nodes["route_b"])
        target_x, target_y = wgs84_to_utm(*road_nodes["target"])
        from_pos = Position3D(
            x=(route_a_x + route_b_x) / 2.0,
            y=(route_a_y + route_b_y) / 2.0,
            z=0.0,
        )
        to_pos = Position3D(x=target_x, y=target_y, z=0.0)
        context = ActiveTruckRouteContext(
            position=from_pos,
            segment_id=0,
            traveled_m=5.0,
            remaining_osm_node_path=("route_b", "target"),
        )
        self.assertEqual(
            find_nearest_node(graph, road_nodes, from_pos.x, from_pos.y),
            "trap",
        )

        route = _build_route_from_active_truck_context(
            context=context,
            from_pos=from_pos,
            to_pos=to_pos,
            road_graph=graph,
            road_nodes=road_nodes,
        )

        self.assertIsNotNone(route)
        self.assertEqual(route.osm_node_path, ("route_b", "target"))

    def _first_idle_drone_id(self, env: TrainingEnvAdapter) -> str:
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.IDLE:
                return drone_id
        self.fail("默认场景中至少应有一架 depot-home 无人机处于 idle")

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
        drone_id = self._first_idle_drone_id(env)
        env._enqueue_decision(drone_id, "test_idle", None)
        return env, drone_id

    def _inject_order(
        self,
        env: TrainingEnvAdapter,
        *,
        drone_id: str,
        order_id: str,
        offset_x: float = 10.0,
        payload_weight: float = 1.0,
    ) -> Order:
        drone = env._require_entity_manager().drones[drone_id]
        order = Order(
            order_id=order_id,
            create_time=env._t_now,
            deadline=env._t_now + 3600.0,
            delivery_loc=Position3D(
                x=drone.current_loc.x + offset_x,
                y=drone.current_loc.y,
                z=drone.current_loc.z,
            ),
            payload_weight=payload_weight,
        )
        env._require_order_manager().pending_orders[order.order_id] = order
        return order

    def _inject_order_at_position(
        self,
        env: TrainingEnvAdapter,
        *,
        order_id: str,
        position: Position3D,
        payload_weight: float = 1.0,
    ) -> Order:
        order = Order(
            order_id=order_id,
            create_time=env._t_now,
            deadline=env._t_now + 3600.0,
            delivery_loc=Position3D(x=position.x, y=position.y, z=position.z),
            payload_weight=payload_weight,
        )
        env._require_order_manager().pending_orders[order.order_id] = order
        return order

    def test_mode_c_action_mask_filters_by_timestamp_and_energy(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 2)

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5B-01")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        t_deliver_finish = (
            t_deliver + env._scene_solver_params().drone_service_time_order_s
        )
        recover_flight_time = env._estimate_flight_time(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=entity_mgr.stations[station_ids[1]].location,
        )
        feasible_truck_arrival = (
            t_deliver_finish
            + recover_flight_time
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_ids[0],
                arrival_time=t_deliver + 1.0,
                departure_time=t_deliver + 1.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_ids[1],
                arrival_time=feasible_truck_arrival,
                departure_time=feasible_truck_arrival + 1e-6,
            ),
        ]

        coarse_plan = env._build_coarse_plan_view(env._t_now)
        action_lookup = env._build_action_lookup(
            drone_id=drone_id,
            coarse_plan=coarse_plan,
        )
        mode_c_actions = [
            action
            for action in action_lookup
            if isinstance(action, DispatchAction) and action.mode == PolicyMode.C
        ]
        candidate_out = env._candidate_builder.build(
            runtime_state=env.build_runtime_state_view(),
            coarse_plan=coarse_plan,
            deciding_drone_id=drone_id,
            trigger_type="test_idle",
            trigger_station_id=None,
            last_seen_plan_version=coarse_plan.plan_version,
        )
        order_feature = next(
            feature
            for feature in candidate_out.candidate_features.order_features
            if feature.order_id == order.order_id
        )

        self.assertEqual(len(mode_c_actions), 1)
        self.assertEqual(mode_c_actions[0].recover_node_id, station_ids[1])
        self.assertTrue(order_feature.has_mode_c_action)
        self.assertAlmostEqual(
            order_feature.best_mode_c_truck_eta_remaining,
            coarse_plan.truck_eta_map[station_ids[1]] - env._t_now,
        )

    def test_mode_b_return_host_prefers_depot_when_reachable(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station = next(iter(entity_mgr.stations.values()))
        depot = env._require_depot()

        drone.current_loc = Position3D(
            x=station.location.x,
            y=station.location.y,
            z=station.location.z,
        )

        selected_host = env._select_return_host_mode_b(drone_id, current_time=env._t_now)

        self.assertIsNotNone(selected_host)
        self.assertEqual(selected_host.depot_id, depot.depot_id)

    def test_mode_b_return_host_uses_station_only_when_depot_unreachable(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station = next(iter(entity_mgr.stations.values()))

        selected_host = env._select_return_host_mode_b(
            drone_id,
            from_pos=station.location,
            battery_current=drone.safe_margin_j + 1.0,
            current_time=env._t_now,
        )

        self.assertIsNotNone(selected_host)
        self.assertEqual(selected_host.station_id, station.station_id)

    def test_mode_b_station_charge_completion_auto_returns_to_depot(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station = next(iter(entity_mgr.stations.values()))
        depot = env._require_depot()

        drone.current_loc = Position3D(
            x=station.location.x,
            y=station.location.y,
            z=station.location.z,
        )
        env._drone_state[drone_id] = TrainingDroneState.CHARGING_OR_SWAP
        env._mode_b_pending_depot_return[drone_id] = depot.depot_id
        station.serving_drones[drone_id] = env._t_now

        env._advance_to_event(env._t_now)

        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.RETURN_TO_DEPOT)
        self.assertNotIn(drone_id, env._mode_b_pending_depot_return)
        return_leg = env._flight_legs[drone_id]
        self.assertEqual(return_leg.kind, "mode_b_return_to_depot")
        self.assertEqual(return_leg.target_node_id, depot.depot_id)

        env._advance_to_event(return_leg.arrival_time)

        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.IDLE)
        self.assertNotIn(drone_id, env._flight_legs)

    def test_idle_wait_at_depot_enters_charge_queue_when_battery_not_full(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        depot = env._require_depot()
        drone.current_loc = Position3D(
            x=depot.location.x,
            y=depot.location.y,
            z=depot.location.z,
        )
        drone.battery_current = drone.battery_max * 0.5
        self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5B-WAIT-DEPOT")
        env._decision_queue.clear()
        env._enqueue_decision(drone_id, "test_idle", None)

        applied = env._apply_decision_core(WAIT_ACTION)

        self.assertEqual(applied.auto_advance_kind, "until_decision")
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.CHARGING_OR_SWAP)
        self.assertIn(drone_id, depot.serving_drones)
        self.assertLess(drone.battery_current, drone.battery_max)

    def test_riding_wait_enters_truck_charge_queue_and_follows_truck(self) -> None:
        env = self._make_env()
        env.reset()
        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr.completed_orders.clear()
        order_mgr._next_order_time = math.inf
        env._decision_queue.clear()
        env._planned_route_stops.clear()
        env._planned_route_segments.clear()

        drone_id = self._first_riding_drone_id(env)
        drone = env._require_entity_manager().drones[drone_id]
        truck = env._require_truck()
        station = next(iter(env._require_entity_manager().stations.values()))
        truck.current_loc = Position3D(
            x=station.location.x,
            y=station.location.y,
            z=station.location.z,
        )
        drone.current_loc = Position3D(
            x=truck.current_loc.x,
            y=truck.current_loc.y,
            z=truck.current_loc.z,
        )
        drone.battery_current = drone.battery_max * 0.5
        self._inject_order_at_position(
            env,
            order_id="ORDER-P5B-WAIT-TRUCK",
            position=truck.current_loc,
            payload_weight=0.5,
        )
        env._enqueue_decision(
            drone_id,
            trigger_type="truck_station_arrival",
            trigger_station_id=station.station_id,
        )

        applied = env._apply_decision_core(WAIT_ACTION)

        self.assertEqual(applied.auto_advance_kind, "until_decision")
        self.assertEqual(env._drone_state[drone_id], TrainingDroneState.CHARGING_ON_TRUCK)
        self.assertNotEqual(env._drone_state[drone_id], TrainingDroneState.RIDING_WITH_TRUCK)
        self.assertIn(drone_id, truck.serving_drones)

        truck.current_loc = Position3D(
            x=station.location.x + 123.0,
            y=station.location.y + 45.0,
            z=station.location.z,
        )
        env._sync_in_transit_positions(env._t_now)
        self.assertAlmostEqual(drone.current_loc.x, truck.current_loc.x)
        self.assertAlmostEqual(drone.current_loc.y, truck.current_loc.y)

    def test_coarse_plan_recovery_pool_exposes_future_backbone_nodes(self) -> None:
        env, _drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 5)

        preferred_station_id = station_ids[4]
        preferred_station = entity_mgr.stations[preferred_station_id]
        order = self._inject_order_at_position(
            env,
            order_id="ORDER-P5B-RECOVERY-POOL-01",
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

        coarse_plan = env._build_coarse_plan_view(env._t_now)
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
            "coarse plan 应优先保留靠近订单的未来固定节点，具体可行性由 CandidateBuilder 过滤",
        )

    def test_post_delivery_revalidation_failure_enters_fallback(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        recover_node_id = sorted(entity_mgr.stations)[0]

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5B-02")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=t_deliver + 450.0,
                departure_time=t_deliver + 450.0 + 1e-6,
            )
        ]
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        decision = env._build_decision_context(trigger)
        action = DispatchAction(
            order_id=order.order_id,
            mode=PolicyMode.C,
            recover_node_id=recover_node_id,
        )

        env._apply_dispatch_action(trigger, action)
        deliver_leg = env._flight_legs[drone_id]

        # 送达前把卡车到站时刻改成“已晚于当前复核窗口”，验证 5b 的时间复核会拒绝原节点。
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=recover_node_id,
                arrival_time=max(0.0, deliver_leg.arrival_time - 1.0),
                departure_time=deliver_leg.arrival_time + 120.0,
            ),
            BackboneVisit(
                node_id=env._require_depot().depot_id,
                arrival_time=deliver_leg.arrival_time + 600.0,
                departure_time=deliver_leg.arrival_time + 600.0 + 1e-6,
            ),
        ]

        reward = env._advance_to_event(deliver_leg.arrival_time)

        self.assertEqual(reward, env._cfg.R_delivery_bonus)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.DELIVERY_SERVICE,
        )
        self.assertIn(drone_id, env._delivery_service_legs)

        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        reward = env._advance_to_event(service_finish_time)

        self.assertEqual(reward, 0.0)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.FALLBACK_RECOVERY,
        )
        self.assertIn(drone_id, env._fallback_leg)
        self.assertEqual(env._flight_legs[drone_id].kind, "fallback_recovery")
        episode_snapshot = env.build_episode_metrics_snapshot()
        self.assertEqual(
            episode_snapshot["mode_c_post_delivery_revalidation_fail_count"],
            1,
        )
        self.assertEqual(episode_snapshot["mode_c_success_count"], 0)
        self.assertEqual(
            episode_snapshot["mode_c_post_delivery_revalidation_fail_reasons"],
            {
                "energy_feasible": 1,
                "rendezvous_time_feasible": 1,
                "node_still_valid": 1,
            },
        )

    def test_exposed_decision_metrics_count_dispatch_and_legal_mode_c_once(self) -> None:
        env, drone_id = self._reset_controlled_env()
        entity_mgr = env._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        station_ids = sorted(entity_mgr.stations)
        self.assertGreaterEqual(len(station_ids), 2)

        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5B-DIAG-01")
        t_deliver = env._estimate_delivery_arrival_time(drone, order)
        t_deliver_finish = (
            t_deliver + env._scene_solver_params().drone_service_time_order_s
        )
        recover_flight_time = env._estimate_flight_time(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=entity_mgr.stations[station_ids[1]].location,
        )
        feasible_truck_arrival = (
            t_deliver_finish
            + recover_flight_time
            + env._cfg.rendezvous_execution_margin_sec
            + 30.0
        )
        env._full_backbone_cache = [
            BackboneVisit(
                node_id=station_ids[0],
                arrival_time=t_deliver + 1.0,
                departure_time=t_deliver + 1.0 + 1e-6,
            ),
            BackboneVisit(
                node_id=station_ids[1],
                arrival_time=feasible_truck_arrival,
                departure_time=feasible_truck_arrival + 1e-6,
            ),
        ]
        env._decision_queue.clear()
        env._enqueue_decision(drone_id, "test_idle", None)
        env._planner_bridge = None
        env._current_coarse_plan = None
        env._episode_dispatch_decision_count = 0
        env._episode_dispatch_decision_with_legal_mode_c_count = 0
        env._episode_feasible_mode_c_recover_node_count_total = 0
        env._last_exposed_decision_id = None

        first_result = env._build_step_result(reward=0.0, info={"event": "probe"})
        self.assertIsNotNone(first_result.decision_context)
        first_snapshot = env.build_episode_metrics_snapshot()
        self.assertEqual(first_snapshot["dispatch_decision_count"], 1)
        self.assertEqual(
            first_snapshot["dispatch_decision_with_legal_mode_c_count"],
            1,
        )
        self.assertEqual(
            first_snapshot["feasible_mode_c_recover_node_count_total"],
            1,
        )
        self.assertAlmostEqual(
            first_snapshot["avg_feasible_mode_c_nodes_per_dispatch_decision"],
            1.0,
        )

        second_result = env._build_step_result(reward=0.0, info={"event": "probe-again"})
        self.assertIsNotNone(second_result.decision_context)
        second_snapshot = env.build_episode_metrics_snapshot()
        self.assertEqual(second_snapshot["dispatch_decision_count"], 1)
        self.assertEqual(
            second_snapshot["dispatch_decision_with_legal_mode_c_count"],
            1,
        )
        self.assertEqual(
            second_snapshot["feasible_mode_c_recover_node_count_total"],
            1,
        )

    def test_delivery_service_finishes_before_mode_b_return_leg_is_scheduled(self) -> None:
        env, drone_id = self._reset_controlled_env()
        order = self._inject_order(env, drone_id=drone_id, order_id="ORDER-P5B-SVC-01")
        env._enqueue_decision(drone_id, "test_idle", None)

        trigger = env._decision_queue.pop(0)
        env._apply_dispatch_action(
            trigger,
            DispatchAction(
                order_id=order.order_id,
                mode=PolicyMode.B,
            ),
        )
        deliver_leg = env._flight_legs[drone_id]

        env._advance_to_event(deliver_leg.arrival_time)

        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.DELIVERY_SERVICE,
        )
        self.assertIn(drone_id, env._delivery_service_legs)
        self.assertNotIn(drone_id, env._flight_legs)

        service_finish_time = env._delivery_service_legs[drone_id].finish_time
        env._advance_to_event(service_finish_time)

        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.RETURN_TO_DEPOT
            if env._flight_legs[drone_id].target_node_type == "depot"
            else TrainingDroneState.RETURN_TO_STATION,
        )
        self.assertGreaterEqual(
            env._flight_legs[drone_id].start_time,
            service_finish_time - 1e-6,
        )

    def test_fallback_reward_accumulates_during_recovery(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        depot = env._require_depot()

        drone.current_loc = Position3D(
            x=depot.location.x + 300.0,
            y=depot.location.y,
            z=depot.location.z,
        )
        env._drone_state[drone_id] = TrainingDroneState.DELIVERED
        env._planned_route_stop_i = len(env._planned_route_stops)

        entered = env._enter_fallback_recovery(drone_id)
        self.assertTrue(entered)

        leg = env._fallback_leg[drone_id]
        delta_t = min(5.0, max(1.0, (leg.arrival_time - env._t_now) / 2.0))

        reward = env._advance_to_event(env._t_now + delta_t)

        self.assertAlmostEqual(reward, -env._cfg.lambda_miss * delta_t, places=6)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.FALLBACK_RECOVERY,
        )
        self.assertAlmostEqual(
            env._last_reward_breakdown["fallback"],
            -env._cfg.lambda_miss * delta_t,
            places=6,
        )

    def test_scheduled_uav_leg_uses_path_service_geometry_for_execution(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        start_pos = Position3D(
            x=drone.current_loc.x,
            y=drone.current_loc.y,
            z=drone.current_loc.z,
        )
        target_pos = Position3D(
            x=start_pos.x + 100.0,
            y=start_pos.y + 100.0,
            z=start_pos.z,
        )
        bend_pos = Position3D(
            x=start_pos.x + 100.0,
            y=start_pos.y,
            z=start_pos.z,
        )

        class BentPathService:
            has_obstacle_planner = True

            def estimate(self, *, drone, from_pos, to_pos, payload):
                path = (
                    Position3D(x=from_pos.x, y=from_pos.y, z=from_pos.z),
                    Position3D(x=bend_pos.x, y=bend_pos.y, z=bend_pos.z),
                    Position3D(x=to_pos.x, y=to_pos.y, z=to_pos.z),
                )
                return UavPathEstimate(
                    path_points=path,
                    cumulative_distances_m=(0.0, 100.0, 200.0),
                    distance_m=200.0,
                    flight_time_sec=20.0,
                    energy_j=1234.0,
                    motion_mode="uav_avoidance_path",
                )

            def plan_path(self, *, from_pos, to_pos, altitude=None):
                return (
                    Position3D(x=from_pos.x, y=from_pos.y, z=from_pos.z),
                    Position3D(x=bend_pos.x, y=bend_pos.y, z=bend_pos.z),
                    Position3D(x=to_pos.x, y=to_pos.y, z=to_pos.z),
                )

            def path_distance(self, *, from_pos, to_pos, altitude=None):
                return 200.0

            def can_reach(
                self,
                *,
                drone,
                from_pos,
                to_pos,
                payload,
                safe_margin,
                battery_current,
                altitude=None,
            ):
                return True

        env._uav_path_service = BentPathService()
        env._schedule_flight_leg(
            drone_id=drone_id,
            kind="test_bent_leg",
            target_pos=target_pos,
            payload=1.0,
        )

        leg = env._flight_legs[drone_id]
        self.assertEqual(leg.motion_mode, "uav_avoidance_path")
        self.assertEqual(len(leg.path_points), 3)
        self.assertAlmostEqual(leg.distance_m, 200.0, places=6)
        self.assertAlmostEqual(leg.energy_cost_j, 1234.0, places=6)
        self.assertAlmostEqual(leg.arrival_time, env._t_now + 20.0, places=6)

        env._sync_in_transit_positions(env._t_now + 10.0)
        self.assertAlmostEqual(drone.current_loc.x, bend_pos.x, places=6)
        self.assertAlmostEqual(drone.current_loc.y, bend_pos.y, places=6)

        entry = env._build_active_drone_path_entry(
            drone_id=drone_id,
            leg=leg,
            t_now=env._t_now + 10.0,
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry["motion_mode"], "uav_avoidance_path")
        self.assertAlmostEqual(entry["distance_m"], 100.0, places=6)

    def test_planner_invalidated_fallback_does_not_charge_ppo_reward(self) -> None:
        env, drone_id = self._reset_controlled_env()
        drone = env._require_entity_manager().drones[drone_id]
        depot = env._require_depot()

        drone.current_loc = Position3D(
            x=depot.location.x + 300.0,
            y=depot.location.y,
            z=depot.location.z,
        )
        env._drone_state[drone_id] = TrainingDroneState.DELIVERED
        env._planned_route_stop_i = len(env._planned_route_stops)

        entered = env._enter_fallback_recovery(
            drone_id,
            cause=FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
        )
        self.assertTrue(entered)

        leg = env._fallback_leg[drone_id]
        delta_t = min(5.0, max(1.0, (leg.arrival_time - env._t_now) / 2.0))
        reward = env._advance_to_event(env._t_now + delta_t)

        self.assertAlmostEqual(reward, -env._cfg.lambda_miss * delta_t, places=6)
        self.assertAlmostEqual(env._agent_cost_accum.get(drone_id, 0.0), 0.0, places=6)
        self.assertEqual(env._episode_system_attributed_fallback_count, 1)
        self.assertEqual(env._episode_ppo_attributed_fallback_count, 0)
        self.assertAlmostEqual(
            env._episode_system_attributed_fallback_time_sec,
            delta_t,
            places=6,
        )
        self.assertEqual(
            env._episode_fallback_cause_counts[
                FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER
            ],
            1,
        )
        self.assertAlmostEqual(
            env._last_reward_breakdown["fallback_system_attributed"],
            -env._cfg.lambda_miss * delta_t,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
