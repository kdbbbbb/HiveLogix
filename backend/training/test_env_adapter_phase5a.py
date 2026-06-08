#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5a env adapter smoke tests.

运行方式：
  python -m unittest backend.training.test_env_adapter_phase5a
"""

from __future__ import annotations

import unittest

from core.entities.primitives import Position3D

from .contracts import PolicyMode
from .env_adapter import (
    BackboneVisit,
    DispatchAction,
    GlobalWaitAction,
    TrainingDroneState,
    TrainingEnvAdapter,
)


class TestTrainingEnvAdapterPhase5a(unittest.TestCase):
    def _make_env(self) -> TrainingEnvAdapter:
        """构造一个使用默认场景的 5a 环境实例。"""
        return TrainingEnvAdapter()

    def _first_idle_drone_id(self, env: TrainingEnvAdapter) -> str:
        """返回 reset 后首个处于 idle 的无人机。"""
        for drone_id, state in env._drone_state.items():
            if state == TrainingDroneState.IDLE:
                return drone_id
        self.fail("默认场景中至少应有一架 depot-home 无人机处于 idle")

    def test_default_scene_episode_smoke(self) -> None:
        env = self._make_env()
        result = env.reset()

        self.assertIsNotNone(result.decision_context)
        self.assertFalse(result.done)
        self.assertIn("system_context_stats", result.info)

        max_steps = 384
        for _ in range(max_steps):
            if result.done:
                break
            decision = result.decision_context
            self.assertIsNotNone(decision)

            # smoke test 只验证环境连通性：有派送动作就优先选派送动作，
            # 否则退回 WAIT，避免把测试本身绑死在某个特定策略上。
            action = next(
                (
                    candidate
                    for candidate in decision.action_lookup
                    if not isinstance(candidate, GlobalWaitAction)
                ),
                GlobalWaitAction(),
            )
            result = env.step(action)

        self.assertTrue(result.done, "episode 未在步数预算内结束，可能存在状态机卡死")
        self.assertLessEqual(result.runtime_state.t_now, env._cfg.upper_horizon_sec + 1e-6)
        self.assertFalse(env._fallback_leg, "episode 结束后 fallback 账本应为空")
        self.assertIn("system_context_stats", result.info)

    def test_fallback_leg_clears_on_host_arrival(self) -> None:
        env = self._make_env()
        env.reset()

        drone_id = "UAV-TEST-03"
        drone = env._require_entity_manager().drones[drone_id]
        station = env._require_entity_manager().stations["STA-TEST-02"]

        drone.current_loc = Position3D(
            x=station.location.x,
            y=station.location.y,
            z=station.location.z,
        )
        env._drone_state[drone_id] = TrainingDroneState.DELIVERED

        # 这里直接把无人机放到站点位置，是为了稳定触发“fallback 飞行时长为 0 的到达分支”；
        # 断言目标是账本清空和统一 charging host 入口是否生效。
        entered = env._enter_fallback_recovery(drone_id)
        self.assertTrue(entered)
        self.assertIn(drone_id, env._fallback_leg)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.FALLBACK_RECOVERY,
        )

        arrival_time = env._fallback_leg[drone_id].arrival_time
        env._advance_to_event(arrival_time)

        self.assertNotIn(drone_id, env._fallback_leg)
        self.assertIn(
            env._drone_state[drone_id],
            {
                TrainingDroneState.CHARGING_OR_SWAP,
                TrainingDroneState.QUEUEING_AT_HOST,
            },
        )

    def test_mode_c_action_shape_is_fixed(self) -> None:
        env = self._make_env()
        result = env.reset()
        decision = result.decision_context
        self.assertIsNotNone(decision)

        # mode C 动作仍只占用 C intent 的模型维度；recover node 是候选阶段
        # 解析出的确定性执行参数。
        mode_c_actions = [
            action
            for action in decision.action_lookup
            if isinstance(action, DispatchAction) and action.mode == PolicyMode.C
        ]
        self.assertTrue(
            all(action.recover_node_id is not None for action in mode_c_actions)
        )

    def test_airborne_energy_failure_interrupts_flight_before_arrival(self) -> None:
        env = self._make_env()
        env.reset()

        drone_id = self._first_idle_drone_id(env)
        drone = env._require_entity_manager().drones[drone_id]
        # 人工写入一段长距离返程飞行，再把剩余电量压到整段能耗的一半，
        # 用来验证“真实空中耗尽”会在到达前被事件队列截断。
        env._drone_state[drone_id] = TrainingDroneState.RETURN_TO_RENDEZVOUS
        far_target = Position3D(
            x=drone.current_loc.x + 2000.0,
            y=drone.current_loc.y,
            z=drone.current_loc.z,
        )
        env._schedule_flight_leg(
            drone_id=drone_id,
            kind="return_to_rendezvous",
            target_pos=far_target,
            payload=0.0,
            target_node_id="STA-TEST-01",
            target_node_type="station",
        )
        leg = env._flight_legs[drone_id]
        drone.battery_current = leg.energy_cost_j / 2.0

        next_event_time = env._compute_airborne_failure_time(drone_id, leg)
        self.assertIsNotNone(next_event_time)
        self.assertLess(next_event_time, leg.arrival_time)

        reward = env._advance_to_event(next_event_time)

        self.assertEqual(reward, -env._cfg.hard_failure_penalty_sec)
        self.assertEqual(
            env._drone_state[drone_id],
            TrainingDroneState.AIRBORNE_ENERGY_FAILURE,
        )
        self.assertEqual(drone.battery_current, 0.0)
        self.assertNotIn(drone_id, env._flight_legs)
        self.assertFalse(
            any(item.drone_id == drone_id for item in env._decision_queue),
            "硬失败无人机应从后续决策集合中移除",
        )

    def test_on_arrive_charging_host_uses_same_queue_logic_for_station_and_depot(self) -> None:
        for host_type in ("station", "depot"):
            env = self._make_env()
            env.reset()

            drone_id = self._first_idle_drone_id(env)
            if host_type == "station":
                host = next(iter(env._require_entity_manager().stations.values()))
            else:
                host = next(iter(env._require_entity_manager().depots.values()))

            # 先按宿主真实并发容量占满所有槽位，再让测试无人机到达，
            # 确认两类宿主都会走同一套 wait_queue 逻辑。
            for idx in range(host.parking_slots):
                host.arrive(f"BLOCKER-{idx}", env._t_now)
            env._on_arrive_charging_host(drone_id, host, env._t_now)

            self.assertIn(
                drone_id,
                host.wait_queue,
                f"{host_type} 到达后应进入 wait_queue，而不是绕过排队",
            )
            self.assertEqual(
                env._drone_state[drone_id],
                TrainingDroneState.QUEUEING_AT_HOST,
            )

    def test_coarse_plan_drops_node_once_truck_has_arrived(self) -> None:
        env = self._make_env()
        env.reset()

        station_id = next(iter(env._require_entity_manager().stations))
        depot_id = next(iter(env._require_entity_manager().depots))
        env._full_backbone_cache = [
            BackboneVisit(node_id=station_id, arrival_time=10.0, departure_time=10.0 + 1e-6),
            BackboneVisit(node_id=depot_id, arrival_time=20.0, departure_time=20.0 + 1e-6),
        ]

        # 新语义要求 future backbone 按 truck arrival 过滤；
        # arrival_time == t_now 时，该站点不再视作未来 recovery 节点。
        coarse_plan = env._build_coarse_plan_view(10.0)
        self.assertNotIn(station_id, coarse_plan.truck_backbone_route)
        self.assertNotIn(station_id, coarse_plan.launch_candidate_stations)
        self.assertIn(depot_id, coarse_plan.truck_backbone_route)

    def test_poisson_reset_appends_patrol_loop_when_phase4_route_returns_early(self) -> None:
        env = self._make_env()
        raw_artifacts = env._load_phase4_artifacts()
        raw_last_stop = raw_artifacts["planned_stops"][-1]

        self.assertEqual(env._order_source.mode.value, "poisson")
        self.assertEqual(raw_last_stop.node_type, "depot")
        self.assertLess(
            raw_last_stop.arrival_time,
            env._cfg.upper_horizon_sec - env._cfg.patrol_min_remaining_sec,
        )

        env.reset()

        self.assertGreater(
            env._planned_route_stops[-1].arrival_time,
            raw_last_stop.arrival_time,
        )
        self.assertGreater(
            len(env._full_backbone_cache),
            len(raw_artifacts["backbone_cache"]),
        )

    def test_poisson_patrol_loop_station_stops_have_fixed_hold_window(self) -> None:
        env = self._make_env()
        raw_artifacts = env._load_phase4_artifacts()
        raw_stop_count = len(raw_artifacts["planned_stops"])
        env.reset()

        appended_stops = env._planned_route_stops[raw_stop_count:]
        appended_station_stops = [
            stop for stop in appended_stops if stop.node_type == "station"
        ]
        self.assertTrue(appended_station_stops)

        solver_params = env._scene_solver_params()
        expected_hold = max(
            solver_params.truck_drone_launch_time_s,
            solver_params.truck_drone_recover_time_s,
        )
        first_station = appended_station_stops[0]
        self.assertAlmostEqual(
            first_station.departure_time - first_station.arrival_time,
            expected_hold,
            places=6,
        )

    def test_poisson_patrol_loop_rotates_station_coverage(self) -> None:
        env = self._make_env()
        raw_artifacts = env._load_phase4_artifacts()
        raw_stop_count = len(raw_artifacts["planned_stops"])
        env.reset()

        appended_station_ids = [
            stop.node_id
            for stop in env._planned_route_stops[raw_stop_count:]
            if stop.node_type == "station"
        ]

        self.assertGreater(
            len(set(appended_station_ids)),
            env._cfg.patrol_stations_per_loop,
            "poisson 巡站不应每轮固定覆盖同一批最近站点",
        )

    def test_poisson_patrol_loop_filters_future_backbone_after_upper_horizon(self) -> None:
        env = self._make_env()
        env.reset()

        future_visits = env._future_backbone_visits(0.0)
        self.assertTrue(
            future_visits,
            "poisson 巡站循环应在 upper_horizon 内提供 future backbone 节点",
        )
        self.assertTrue(
            all(
                visit.arrival_time <= env._cfg.upper_horizon_sec + 1e-6
                for visit in future_visits
            )
        )
        self.assertEqual(
            env._future_backbone_visits(env._cfg.upper_horizon_sec - 1.0),
            (),
        )


if __name__ == "__main__":
    unittest.main()
