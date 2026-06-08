#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_bp 运行时适配路由测试。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from flask import Flask


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
GEO_DIR = BACKEND_DIR / "environment" / "geo"
STATE_DIR = BACKEND_DIR / "environment" / "state"
ROUTES_DIR = BACKEND_DIR / "api" / "routes"

for path in (BACKEND_DIR, GEO_DIR, STATE_DIR, ROUTES_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import simulation_bp as sim_routes  # noqa: E402


class _FakePolicyRuntime:
    def __init__(self) -> None:
        self.current_time = 12.345
        self.is_running = False
        self.speed_ratio = 1.0
        self.started = 0
        self.paused = 0
        self.reset_called = 0
        self.speed_updates: list[float] = []

    @property
    def policy_name(self) -> str:
        return "fake_policy"

    @property
    def checkpoint_path(self) -> str:
        return "/tmp/fake.pt"

    def start(self) -> None:
        self.started += 1
        self.is_running = True

    def pause(self) -> None:
        self.paused += 1
        self.is_running = False

    def reset(self) -> None:
        self.reset_called += 1
        self.is_running = False
        self.current_time = 0.0

    def set_speed(self, speed_ratio: float) -> None:
        self.speed_ratio = float(speed_ratio)
        self.speed_updates.append(float(speed_ratio))

    def get_recent_orders(self, limit: int = 100):
        return [{"order_id": "ORD-FAKE", "status": "PENDING"}][:limit]

    def build_full_snapshot(self):
        return {
            "type": "FULL_SNAPSHOT",
            "payload": {
                "sim_time": self.current_time,
                "is_running": self.is_running,
                "speed_ratio": self.speed_ratio,
                "sim_start_wall_ms": 0,
                "entities": {
                    "depots": [],
                    "stations": [],
                    "trucks": [],
                    "drones": [],
                },
                "orders": self.get_recent_orders(),
                "stats": {"active_policy": self.policy_name},
            },
        }


class TestSimulationPolicyRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(sim_routes.sim_bp, url_prefix="/api/sim")
        self.client = self.app.test_client()
        self.original_policy_runtime = sim_routes._policy_runtime

    def tearDown(self) -> None:
        sim_routes._policy_runtime = self.original_policy_runtime
        sim_routes._restore_classic_runtime()

    def test_state_orders_and_control_delegate_to_active_policy_runtime(self) -> None:
        fake = _FakePolicyRuntime()
        sim_routes._policy_runtime = fake

        state_resp = self.client.get("/api/sim/state")
        self.assertEqual(state_resp.status_code, 200)
        state_data = state_resp.get_json()
        self.assertEqual(state_data["sim_time"], 12.345)
        self.assertEqual(state_data["stats"]["active_policy"], "fake_policy")

        orders_resp = self.client.get("/api/sim/orders?limit=10")
        self.assertEqual(orders_resp.status_code, 200)
        orders_data = orders_resp.get_json()
        self.assertEqual(orders_data["total"], 1)
        self.assertEqual(orders_data["orders"][0]["order_id"], "ORD-FAKE")

        control_resp = self.client.post("/api/sim/control", json={"action": "start", "speed": 2.0})
        self.assertEqual(control_resp.status_code, 200)
        control_data = control_resp.get_json()
        self.assertTrue(control_data["is_running"])
        self.assertEqual(fake.started, 1)
        self.assertEqual(fake.speed_updates[-1], 2.0)


if __name__ == "__main__":
    unittest.main()
