#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainingEnvAdapter 前端在线运行时适配器测试。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from core.entities.order import Order
from core.entities.primitives import Position3D
from training.env_adapter import (
    DeliveryServiceLeg,
    PlannedStop,
    PlannedTruckSegment,
    TrainingDroneState,
    TrainingEnvAdapter,
)
from training.export_sumo_truck_route import export_phase4_truck_route_for_scene
from training.frontend_runtime_adapter import (
    OnlinePolicyRuntimePlayer,
    TrainingTelemetryBridge,
    build_orders_raw_from_sim_init_payload,
    build_runtime_scene_context_from_payload,
    resolve_scene_context,
)
from training.order_source_adapter import OrderSourceMode, build_order_source
from training.scene_loader import DEFAULT_CONFIG_PATH


class TestFrontendRuntimeAdapter(unittest.TestCase):
    def _build_env_with_orders_cleared(self) -> TrainingEnvAdapter:
        scene_ctx = resolve_scene_context(config_path=DEFAULT_CONFIG_PATH)
        order_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.BENCHMARK,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=order_source,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env.reset()
        order_mgr = env._require_order_manager()
        order_mgr.pending_orders.clear()
        order_mgr.assigned_orders.clear()
        order_mgr._scheduled_dynamic.clear()
        order_mgr._scheduled_dynamic_i = 0
        order_mgr._gen_config = None
        order_mgr._next_order_time = float("inf")
        env._background_mode_a_pending.clear()
        env._decision_queue.clear()
        env._flight_legs.clear()
        env._delivery_service_legs.clear()
        env._fallback_leg.clear()
        env._truck_charge_until.clear()
        env._active_wait_until.clear()
        env._active_wait_resume.clear()

        entity_mgr = env._require_entity_manager()
        depot = env._require_depot()
        truck = env._require_truck()
        truck.current_loc = Position3D(
            x=depot.location.x,
            y=depot.location.y,
            z=depot.location.z,
        )
        for host in list(entity_mgr.depots.values()) + list(entity_mgr.stations.values()) + [truck]:
            host.serving_drones.clear()
            host.wait_queue.clear()
        for drone_id, drone in entity_mgr.drones.items():
            if env._drone_state.get(drone_id) == TrainingDroneState.RIDING_WITH_TRUCK:
                drone.current_loc = Position3D(
                    x=truck.current_loc.x,
                    y=truck.current_loc.y,
                    z=truck.current_loc.z,
                )
                if drone_id not in truck.docked_drones:
                    truck.docked_drones.append(drone_id)
            else:
                env._drone_state[drone_id] = TrainingDroneState.IDLE
                drone.current_loc = Position3D(
                    x=depot.location.x,
                    y=depot.location.y,
                    z=depot.location.z,
                )
        return env

    def _runtime_shell(self, env: TrainingEnvAdapter) -> OnlinePolicyRuntimePlayer:
        runtime = object.__new__(OnlinePolicyRuntimePlayer)
        runtime._env = env
        return runtime

    def test_build_orders_raw_from_sim_init_payload_normalizes_wall_ms_initial_orders(self) -> None:
        orders_raw = build_orders_raw_from_sim_init_payload(
            initial_orders=[
                {
                    "order_id": "ORD-1",
                    "create_time": 1_710_000_000_000,
                    "deadline": 1_710_000_060_000,
                    "delivery_lng": 121.20,
                    "delivery_lat": 31.03,
                    "delivery_z": 0,
                    "payload_weight": 1.5,
                    "source_type": "DEPOT",
                    "pickup_source_id": None,
                    "time_domain": "wall_ms",
                },
                {
                    "order_id": "ORD-2",
                    "create_time": 1_710_000_030_000,
                    "deadline": 1_710_000_120_000,
                    "delivery_lng": 121.21,
                    "delivery_lat": 31.04,
                    "delivery_z": 0,
                    "payload_weight": 2.0,
                    "source_type": "DEPOT",
                    "pickup_source_id": None,
                    "time_domain": "wall_ms",
                },
            ],
            scheduled_dynamic_orders=[],
        )

        static_orders = orders_raw["static_orders"]
        self.assertEqual(len(static_orders), 2)
        self.assertEqual(static_orders[0]["order_id"], "ORD-1")
        self.assertAlmostEqual(float(static_orders[0]["create_time"]), 0.0, places=6)
        self.assertAlmostEqual(float(static_orders[0]["deadline"]), 60.0, places=6)
        dynamic_orders = orders_raw["dynamic_orders"]
        self.assertEqual(len(dynamic_orders), 2)
        self.assertEqual(dynamic_orders[0]["order_id"], "ORD-1")
        self.assertAlmostEqual(float(dynamic_orders[0]["spawn_sim_s"]), 0.0, places=6)
        self.assertAlmostEqual(float(dynamic_orders[0]["deadline_sim_s"]), 60.0, places=6)
        self.assertAlmostEqual(float(dynamic_orders[1]["spawn_sim_s"]), 30.0, places=6)
        self.assertAlmostEqual(float(dynamic_orders[1]["deadline_sim_s"]), 120.0, places=6)

    def test_runtime_scene_can_export_phase4_from_initial_orders(self) -> None:
        base_scene_ctx = resolve_scene_context(config_path=DEFAULT_CONFIG_PATH)
        orders_raw = build_orders_raw_from_sim_init_payload(
            initial_orders=[
                {
                    "order_id": "ORD-HEAVY-1",
                    "create_time": 0.0,
                    "deadline": 1800.0,
                    "delivery_lng": 121.224,
                    "delivery_lat": 31.049,
                    "delivery_z": 0,
                    "payload_weight": 20.0,
                    "source_type": "DEPOT",
                    "pickup_source_id": None,
                    "fulfillment_mode": "DRONE_TRUCK_DEPOT",
                    "time_domain": "sim_s",
                }
            ],
            scheduled_dynamic_orders=[],
        )
        runtime_scene_ctx = build_runtime_scene_context_from_payload(
            scene_ctx=base_scene_ctx,
            entities_raw=base_scene_ctx.entities_raw,
            orders_raw=orders_raw,
        )

        with tempfile.TemporaryDirectory(prefix="hl-phase4-test-") as tmp_dir:
            result = export_phase4_truck_route_for_scene(
                scene_ctx=runtime_scene_ctx,
                config_path=DEFAULT_CONFIG_PATH,
                output_dir=tmp_dir,
            )

            self.assertTrue(result.execution_route.stops)
            self.assertTrue((Path(tmp_dir) / "truck_execution_route.json").is_file())
            self.assertTrue((Path(tmp_dir) / "truck_backbone_route.json").is_file())

    def test_training_telemetry_bridge_builds_frontend_payload(self) -> None:
        scene_ctx = resolve_scene_context(config_path=DEFAULT_CONFIG_PATH)
        order_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.BENCHMARK,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=order_source,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env.reset()

        bridge = TrainingTelemetryBridge(
            policy_name="test_policy",
            checkpoint_path="backend/weights/test/policy.pt",
        )
        tick = bridge.build_tick_payload(
            env=env,
            speed_ratio=1.0,
            deterministic=True,
            order_source_mode="benchmark",
            recent_decision_events=(
                {
                    "decision_id": 1,
                    "event_seq": 1,
                    "status": "DECISION_PENDING",
                    "sim_time": 0.0,
                    "drone_id": "DRONE-TEST",
                },
            ),
            latest_event_seq=1,
        )
        snapshot = bridge.build_full_snapshot(
            env=env,
            is_running=False,
            speed_ratio=1.0,
            sim_start_wall_ms=int(time.time() * 1000),
            deterministic=True,
            order_source_mode="benchmark",
        )

        self.assertIn("sim_time", tick)
        self.assertIn("entities", tick)
        self.assertIn("orders", tick)
        self.assertIn("stats", tick)
        self.assertEqual(len(tick["entities"]["trucks"]), 1)
        self.assertEqual(len(tick["entities"]["drones"]), len(scene_ctx.drones))
        self.assertEqual(len(tick["entities"]["stations"]), len(scene_ctx.stations))
        self.assertEqual(tick["stats"]["active_policy"], "test_policy")
        self.assertIn("current_decision", tick["stats"])
        self.assertEqual(snapshot["type"], "FULL_SNAPSHOT")
        self.assertIn("payload", snapshot)
        self.assertIn("is_running", snapshot["payload"])
        self.assertIn("sim_start_wall_ms", snapshot["payload"])
        self.assertIn("paths", tick)
        self.assertIn("trucks", tick["paths"])
        self.assertGreaterEqual(len(tick["paths"]["trucks"]), 1)
        self.assertIn("path", tick["paths"]["trucks"][0])
        self.assertIn("dispatch_chains", tick)
        self.assertEqual(tick["dispatch_chains"], [])
        self.assertIn("dispatch_chains", snapshot["payload"])
        self.assertEqual(tick["latest_event_seq"], 1)
        self.assertEqual(tick["recent_decision_events"][0]["status"], "DECISION_PENDING")

    def test_truck_runtime_position_uses_segment_geometry_instead_of_stop_to_stop_straight_line(self) -> None:
        scene_ctx = resolve_scene_context(config_path=DEFAULT_CONFIG_PATH)
        order_source = build_order_source(
            scene_ctx,
            mode=OrderSourceMode.BENCHMARK,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env = TrainingEnvAdapter(
            scene_ctx=scene_ctx,
            order_source=order_source,
            config_path=DEFAULT_CONFIG_PATH,
        )
        env._planned_route_stops = [
            PlannedStop(
                seq=0,
                node_type="depot",
                node_id="DEP-0",
                position=Position3D(x=0.0, y=0.0, z=0.0),
                order_id=None,
                arrival_time=0.0,
                departure_time=0.0,
            ),
            PlannedStop(
                seq=1,
                node_type="station",
                node_id="STA-1",
                position=Position3D(x=10.0, y=10.0, z=0.0),
                order_id=None,
                arrival_time=20.0,
                departure_time=20.0,
            ),
        ]
        env._planned_route_segments = [
            PlannedTruckSegment(
                segment_id=0,
                from_node_id="DEP-0",
                to_node_id="STA-1",
                from_node_type="depot",
                to_node_type="station",
                start_time=0.0,
                end_time=20.0,
                distance_m=20.0,
                geometry=(
                    Position3D(x=0.0, y=0.0, z=0.0),
                    Position3D(x=0.0, y=10.0, z=0.0),
                    Position3D(x=10.0, y=10.0, z=0.0),
                ),
                cumulative_distances_m=(0.0, 10.0, 20.0),
            )
        ]

        pos_t5 = env._truck_position_at_time(5.0)
        pos_t15 = env._truck_position_at_time(15.0)

        self.assertAlmostEqual(pos_t5.x, 0.0, places=6)
        self.assertAlmostEqual(pos_t5.y, 5.0, places=6)
        self.assertAlmostEqual(pos_t15.x, 5.0, places=6)
        self.assertAlmostEqual(pos_t15.y, 10.0, places=6)

    def test_online_advance_can_continue_after_training_orders_cleared_done(self) -> None:
        env = self._build_env_with_orders_cleared()
        self.assertTrue(env.is_done())

        target_time = env._t_now + 10.0
        default_result = env.advance_to_time(target_time)
        self.assertAlmostEqual(default_result.runtime_state.t_now, env._t_now, places=6)

        online_result = env.advance_to_time(
            target_time,
            stop_on_training_done=False,
        )
        self.assertAlmostEqual(online_result.runtime_state.t_now, target_time, places=6)

    def test_online_advance_can_continue_beyond_training_horizon(self) -> None:
        env = self._build_env_with_orders_cleared()
        env._t_now = env._cfg.upper_horizon_sec - 1.0
        env._require_order_manager().pending_orders["ORDER-HORIZON-PROBE"] = Order(
            order_id="ORDER-HORIZON-PROBE",
            create_time=env._t_now,
            deadline=env._cfg.upper_horizon_sec + 100.0,
            delivery_loc=Position3D(
                x=env._require_depot().location.x,
                y=env._require_depot().location.y,
                z=env._require_depot().location.z,
            ),
            payload_weight=999.0,
        )
        target_time = env._cfg.upper_horizon_sec + 5.0

        default_result = env.advance_to_time(target_time)
        self.assertAlmostEqual(default_result.runtime_state.t_now, target_time, places=6)

        env._t_now = env._cfg.upper_horizon_sec - 1.0
        env._decision_queue.clear()
        online_result = env.advance_to_time(
            target_time,
            stop_on_training_done=False,
        )
        self.assertAlmostEqual(online_result.runtime_state.t_now, target_time, places=6)

    def test_online_done_does_not_stop_at_training_horizon(self) -> None:
        env = self._build_env_with_orders_cleared()
        runtime = self._runtime_shell(env)
        env._t_now = env._cfg.upper_horizon_sec + 1.0
        env._require_order_manager()._scheduled_dynamic = [
            {"spawn_sim_s": env._cfg.upper_horizon_sec + 10.0}
        ]
        env._require_order_manager()._scheduled_dynamic_i = 0

        self.assertFalse(runtime._is_online_done_locked())

    def test_online_done_requires_orders_home_and_closed_execution_chain(self) -> None:
        env = self._build_env_with_orders_cleared()
        runtime = self._runtime_shell(env)
        self.assertTrue(runtime._is_online_done_locked())

        drone_id = next(iter(env._require_entity_manager().drones))
        env._drone_state[drone_id] = TrainingDroneState.DELIVERY_SERVICE
        env._delivery_service_legs[drone_id] = DeliveryServiceLeg(
            order_id="ORDER-DONE-CHECK",
            start_time=env._t_now,
            finish_time=env._t_now + 30.0,
            service_pos=Position3D(
                x=env._require_depot().location.x + 100.0,
                y=env._require_depot().location.y,
                z=env._require_depot().location.z,
            ),
        )
        self.assertFalse(runtime._is_online_done_locked())

        env._delivery_service_legs.clear()
        env._drone_state[drone_id] = TrainingDroneState.IDLE
        env._require_entity_manager().drones[drone_id].current_loc = Position3D(
            x=env._require_depot().location.x,
            y=env._require_depot().location.y,
            z=env._require_depot().location.z,
        )
        env._require_order_manager()._scheduled_dynamic = [{"spawn_sim_s": 100.0}]
        env._require_order_manager()._scheduled_dynamic_i = 0
        self.assertFalse(runtime._is_online_done_locked())

        env._require_order_manager()._scheduled_dynamic = []
        env._require_truck().current_loc = Position3D(
            x=env._require_depot().location.x + 100.0,
            y=env._require_depot().location.y,
            z=env._require_depot().location.z,
        )
        self.assertFalse(runtime._is_online_done_locked())

    def test_online_done_counts_drones_on_truck_when_truck_is_at_depot(self) -> None:
        env = self._build_env_with_orders_cleared()
        runtime = self._runtime_shell(env)
        truck = env._require_truck()
        depot = env._require_depot()
        drone_id = next(iter(env._require_entity_manager().drones))
        drone = env._require_entity_manager().drones[drone_id]

        truck.current_loc = Position3D(
            x=depot.location.x,
            y=depot.location.y,
            z=depot.location.z,
        )
        drone.current_loc = Position3D(
            x=depot.location.x + 1000.0,
            y=depot.location.y,
            z=depot.location.z,
        )
        env._drone_state[drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
        if drone_id not in truck.docked_drones:
            truck.docked_drones.append(drone_id)

        self.assertTrue(runtime._is_online_done_locked())


if __name__ == "__main__":
    unittest.main()
