#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for PlannerBridge station opportunity scoring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.entities.order import Order
from core.entities.primitives import Position3D

from .planner_bridge import (
    PlannerBridge,
    _TruckPlanNode,
    _station_sequence_violates_anti_pingpong,
)


class TestPlannerBridgeStationBonus(unittest.TestCase):
    def _make_bridge(self) -> PlannerBridge:
        return PlannerBridge(
            future_backbone_provider=lambda _t_now: (),
            config_path=(
                "backend/config/"
                "rh_alns_cmrappo_bc_warm_start_poisson_two_stage_formal_30k_ddl_sorted.yaml"
            ),
            heavy_payload_capacity=10.0,
            truck_speed_provider=lambda: 13.0,
            truck_travel_time_provider=lambda from_pos, to_pos: (
                from_pos.distance_2d(to_pos) / 13.0
            ),
        )

    def _make_runtime(self, order_positions: list[Position3D]) -> SimpleNamespace:
        orders = {
            f"ORDER-{idx:02d}": Order(
                order_id=f"ORDER-{idx:02d}",
                create_time=0.0,
                deadline=3600.0,
                delivery_loc=position,
                payload_weight=1.0,
            )
            for idx, position in enumerate(order_positions)
        }
        drone = SimpleNamespace(
            payload_capacity=10.0,
            cruise_speed=10.0,
            current_loc=Position3D(x=0.0, y=0.0, z=0.0),
        )
        return SimpleNamespace(
            t_now=0.0,
            pending_orders=orders,
            drone_states={"DRONE-01": drone},
        )

    def _station(self, node_id: str, x: float, y: float = 0.0) -> _TruckPlanNode:
        return _TruckPlanNode(
            node_id=node_id,
            node_type="station",
            order_id=None,
            position=Position3D(x=x, y=y, z=0.0),
            service_time_sec=0.0,
        )

    def test_station_density_bonus_prefers_larger_uncovered_order_share(self) -> None:
        bridge = self._make_bridge()
        runtime_state = self._make_runtime(
            [
                Position3D(x=0.0, y=0.0, z=0.0),
                Position3D(x=80.0, y=0.0, z=0.0),
                Position3D(x=120.0, y=0.0, z=0.0),
                Position3D(x=2500.0, y=0.0, z=0.0),
            ]
        )
        cache = bridge._build_station_bonus_order_cache(runtime_state)

        dense_bonus = bridge._station_recovery_opportunity_bonus(
            runtime_state=runtime_state,
            station=self._station("STA-DENSE", 0.0),
            station_eta=120.0,
            bonus_order_cache=cache,
        )
        sparse_bonus = bridge._station_recovery_opportunity_bonus(
            runtime_state=runtime_state,
            station=self._station("STA-SPARSE", 2500.0),
            station_eta=370.0,
            bonus_order_cache=cache,
        )

        self.assertGreater(dense_bonus, sparse_bonus)

    def test_selected_station_overlap_discounts_redundant_neighbor_bonus(self) -> None:
        bridge = self._make_bridge()
        runtime_state = self._make_runtime(
            [
                Position3D(x=0.0, y=0.0, z=0.0),
                Position3D(x=60.0, y=0.0, z=0.0),
                Position3D(x=120.0, y=0.0, z=0.0),
            ]
        )
        cache = bridge._build_station_bonus_order_cache(runtime_state)
        candidate = self._station("STA-NEARBY", 50.0)
        selected = (self._station("STA-SELECTED", 0.0),)

        no_overlap_bonus = bridge._station_recovery_opportunity_bonus(
            runtime_state=runtime_state,
            station=candidate,
            station_eta=120.0,
            bonus_order_cache=cache,
        )
        overlap_discounted_bonus = bridge._station_recovery_opportunity_bonus(
            runtime_state=runtime_state,
            station=candidate,
            station_eta=120.0,
            bonus_order_cache=cache,
            selected_station_nodes=selected,
        )

        self.assertLess(overlap_discounted_bonus, no_overlap_bonus)

    def test_station_anti_pingpong_rejects_recent_reversals(self) -> None:
        self.assertTrue(
            _station_sequence_violates_anti_pingpong(
                ("STA-03", "STA-04", "STA-03"),
                min_gap=3,
            )
        )
        self.assertTrue(
            _station_sequence_violates_anti_pingpong(
                ("STA-03", "STA-04", "STA-05", "STA-04", "STA-03"),
                min_gap=3,
            )
        )
        self.assertFalse(
            _station_sequence_violates_anti_pingpong(
                ("STA-03", "STA-04", "STA-05", "STA-06", "STA-07", "STA-03"),
                min_gap=3,
            )
        )

    def test_station_anti_pingpong_only_checks_new_suffix_when_requested(self) -> None:
        self.assertFalse(
            _station_sequence_violates_anti_pingpong(
                ("STA-03", "STA-04", "STA-03", "STA-05"),
                min_gap=3,
                check_from_index=3,
            )
        )
        self.assertTrue(
            _station_sequence_violates_anti_pingpong(
                ("STA-03", "STA-04", "STA-03", "STA-04"),
                min_gap=3,
                check_from_index=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
