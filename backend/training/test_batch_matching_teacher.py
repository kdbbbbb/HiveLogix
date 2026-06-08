#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for batch matching BC teacher."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from .actions import DispatchAction, WAIT_ACTION
from .batch_matching_teacher import (
    attach_local_teacher_competition_hints,
    build_batch_matching_teacher_labels,
)
from .contracts import (
    CandidateFeatures,
    CandidateOutput,
    CandidateTeacherEnergy,
    FactorizedActionSchema,
    InfraFeatures,
    OrderFeatures,
    ResolvedActionLookup,
    UavSelfFeatures,
)


class TestBatchMatchingTeacher(unittest.TestCase):
    def test_duplicate_order_is_assigned_to_lowest_cost_uav(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (
            _context("DRN-1", runtime_state, coarse_plan),
            _context("DRN-2", runtime_state, coarse_plan),
        )
        candidates = {
            "DRN-1": _candidate_output([("ORDER-1", ("B",))], "DRN-1"),
            "DRN-2": _candidate_output([("ORDER-1", ("B",))], "DRN-2"),
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
            dispatch_cost_fn=_cost({("DRN-1", "ORDER-1", "B"): 10.0, ("DRN-2", "ORDER-1", "B"): 1.0}),
        )

        self.assertEqual(result.actions_by_drone["DRN-2"], DispatchAction("ORDER-1", "B"))
        self.assertEqual(result.actions_by_drone["DRN-1"], WAIT_ACTION)
        selected_orders = [
            action.order_id
            for action in result.actions_by_drone.values()
            if isinstance(action, DispatchAction)
        ]
        self.assertEqual(selected_orders, ["ORDER-1"])

    def test_assignment_is_global_not_per_drone_greedy(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (
            _context("DRN-1", runtime_state, coarse_plan),
            _context("DRN-2", runtime_state, coarse_plan),
        )
        candidates = {
            "DRN-1": _candidate_output(
                [("ORDER-1", ("B",)), ("ORDER-2", ("B",))],
                "DRN-1",
            ),
            "DRN-2": _candidate_output([("ORDER-1", ("B",))], "DRN-2"),
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
            dispatch_cost_fn=_cost(
                {
                    ("DRN-1", "ORDER-1", "B"): 0.0,
                    ("DRN-1", "ORDER-2", "B"): 5.0,
                    ("DRN-2", "ORDER-1", "B"): 1.0,
                }
            ),
        )

        self.assertEqual(result.actions_by_drone["DRN-1"], DispatchAction("ORDER-2", "B"))
        self.assertEqual(result.labels_by_drone["DRN-1"].order_idx, 1)
        self.assertEqual(result.actions_by_drone["DRN-2"], DispatchAction("ORDER-1", "B"))

    def test_no_dispatch_action_returns_wait_label(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (_context("DRN-1", runtime_state, coarse_plan),)
        candidates = {"DRN-1": _candidate_output([], "DRN-1")}

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
        )

        self.assertEqual(result.actions_by_drone["DRN-1"], WAIT_ACTION)
        self.assertEqual(result.labels_by_drone["DRN-1"].root_branch_idx, 0)
        self.assertIsNone(result.labels_by_drone["DRN-1"].order_idx)
        self.assertIsNone(result.labels_by_drone["DRN-1"].mode_idx)

    def test_shared_snapshot_is_required_by_default(self) -> None:
        coarse_plan = object()
        contexts = (
            _context("DRN-1", object(), coarse_plan),
            _context("DRN-2", object(), coarse_plan),
        )
        candidates = {
            "DRN-1": _candidate_output([], "DRN-1"),
            "DRN-2": _candidate_output([], "DRN-2"),
        }

        with self.assertRaises(ValueError):
            build_batch_matching_teacher_labels(
                decision_contexts=contexts,
                candidate_outputs_by_drone=candidates,
            )

    def test_default_cost_combines_delivery_service_and_recovery_cost(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (_context("DRN-1", runtime_state, coarse_plan),)
        candidates = {
            "DRN-1": _candidate_output(
                [
                    ("ORDER-B", ("B",)),
                    ("ORDER-C", ("C",)),
                ],
                "DRN-1",
                order_feature_overrides_by_order={
                    "ORDER-B": {
                        "distance_to_order": 100.0,
                        "deadline": 2000.0,
                        "best_mode_b_recovery_flight_time": 80.0,
                        "best_mode_b_queue_time_est": 10000.0,
                    },
                    "ORDER-C": {
                        "distance_to_order": 10000.0,
                        "deadline": 20000.0,
                        "best_mode_c_wait_time": 20.0,
                        "best_mode_c_uav_flight_time": 30.0,
                        "best_mode_c_energy_margin_ratio": 0.0,
                        "best_mode_c_timeout_risk": 10000.0,
                    },
                },
            )
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
        )

        self.assertEqual(result.actions_by_drone["DRN-1"], DispatchAction("ORDER-B", "B"))
        self.assertAlmostEqual(result.assignments_by_drone["DRN-1"].cost, 120.0)

    def test_default_cost_prioritizes_saveable_urgent_deadline(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (_context("DRN-1", runtime_state, coarse_plan),)
        candidates = {
            "DRN-1": _candidate_output(
                [
                    ("RELAXED", ("B",)),
                    ("URGENT", ("B",)),
                ],
                "DRN-1",
                order_feature_overrides_by_order={
                    "RELAXED": {
                        "distance_to_order": 100.0,
                        "deadline": 2000.0,
                        "best_mode_b_recovery_flight_time": 1.0,
                    },
                    "URGENT": {
                        "distance_to_order": 2000.0,
                        "deadline": 350.0,
                        "best_mode_b_recovery_flight_time": 500.0,
                    },
                },
            )
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
        )

        self.assertEqual(result.actions_by_drone["DRN-1"], DispatchAction("URGENT", "B"))
        self.assertAlmostEqual(result.assignments_by_drone["DRN-1"].cost, -990.0)

    def test_deadline_risk_uses_service_finish_slack_not_arrival_slack(self) -> None:
        """Regression: deadline slack must be based on delivery *finish* time
        (arrival + service_time), not bare customer arrival time.

        Scenario (t_decision=100, cruise_speed=10, service_time=30):
            distance      = 800  → flight_time = 80
            arrival       = 100 + 80 = 180
            finish        = 180 + 30 = 210
            deadline      = 190

            old slack (arrival-based) = 190 - 180 = +10  (appears on-time)
            new slack (finish-based)  = 190 - 210 = -20  (correctly late)

        With finish-based slack=-20:
            penalty = -1800 + 10*20 = -1600
            total   = flight(80) + service(30) + recovery(0) + (-1600) = -1490

        With old arrival-based slack=+10 the total would have been -1650.
        The exact value -1490 proves the stored finish-slack field is used.
        """
        runtime_state = object()
        coarse_plan = object()
        contexts = (_context("DRN-1", runtime_state, coarse_plan),)
        candidates = {
            "DRN-1": _candidate_output(
                [("ORDER-B", ("B",))],
                "DRN-1",
                order_feature_overrides_by_order={
                    "ORDER-B": {
                        "distance_to_order": 800.0,
                        "deadline": 190.0,
                        "best_mode_b_recovery_flight_time": 0.0,
                        # finish-based slack: deadline(190) - finish(210) = -20
                        "estimated_delivery_finish_slack_sec": -20.0,
                    },
                },
            )
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
        )

        self.assertAlmostEqual(
            result.assignments_by_drone["DRN-1"].cost,
            -1490.0,
            msg="Cost must use delivery finish slack (-20), not arrival slack (+10)",
        )

    def test_default_cost_adds_teacher_only_energy_ratio(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (_context("DRN-1", runtime_state, coarse_plan),)
        candidates = {
            "DRN-1": _candidate_output(
                [("ORDER-1", ("B", "C"))],
                "DRN-1",
                order_feature_overrides_by_order={
                    "ORDER-1": {
                        "deadline": 2000.0,
                        "distance_to_order": 100.0,
                        "best_mode_b_recovery_flight_time": 10.0,
                        "best_mode_c_uav_flight_time": 10.0,
                        "best_mode_c_wait_time": 0.0,
                    },
                },
                teacher_energy_by_order_idx={
                    0: CandidateTeacherEnergy(
                        delivery_energy_ratio=0.10,
                        best_mode_b_recovery_energy_ratio=0.40,
                        best_mode_c_recovery_energy_ratio=0.05,
                        best_mode_b_total_energy_ratio=0.50,
                        best_mode_c_total_energy_ratio=0.15,
                    )
                },
            )
        }

        result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
        )

        self.assertEqual(
            result.actions_by_drone["DRN-1"],
            DispatchAction("ORDER-1", "C"),
        )
        self.assertAlmostEqual(result.assignments_by_drone["DRN-1"].cost, 86.0)

    def test_local_teacher_competition_hints_are_pre_matching(self) -> None:
        runtime_state = object()
        coarse_plan = object()
        contexts = (
            _context("DRN-1", runtime_state, coarse_plan),
            _context("DRN-2", runtime_state, coarse_plan),
        )
        candidates = {
            "DRN-1": _candidate_output(
                [("ORDER-1", ("B",)), ("ORDER-2", ("B",))],
                "DRN-1",
            ),
            "DRN-2": _candidate_output([("ORDER-1", ("C",))], "DRN-2"),
        }

        hinted = attach_local_teacher_competition_hints(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
            dispatch_cost_fn=_cost(
                {
                    ("DRN-1", "ORDER-1", "B"): 0.0,
                    ("DRN-1", "ORDER-2", "B"): 5.0,
                    ("DRN-2", "ORDER-1", "C"): 1.0,
                }
            ),
        )
        final_result = build_batch_matching_teacher_labels(
            decision_contexts=contexts,
            candidate_outputs_by_drone=candidates,
            dispatch_cost_fn=_cost(
                {
                    ("DRN-1", "ORDER-1", "B"): 0.0,
                    ("DRN-1", "ORDER-2", "B"): 5.0,
                    ("DRN-2", "ORDER-1", "C"): 1.0,
                }
            ),
        )

        drn1_order1 = hinted["DRN-1"].candidate_features.order_features[0]
        drn1_order2 = hinted["DRN-1"].candidate_features.order_features[1]
        drn2_order1 = hinted["DRN-2"].candidate_features.order_features[0]
        self.assertEqual(final_result.actions_by_drone["DRN-1"], DispatchAction("ORDER-2", "B"))
        self.assertTrue(drn1_order1.local_teacher_prefers_order)
        self.assertFalse(drn1_order2.local_teacher_prefers_order)
        self.assertEqual(drn1_order1.local_teacher_peer_prefer_count, 2)
        self.assertEqual(drn1_order1.local_teacher_mode_b_prefer_count, 1)
        self.assertEqual(drn1_order1.local_teacher_mode_c_prefer_count, 1)
        self.assertTrue(drn1_order1.local_teacher_is_order_best)
        self.assertFalse(drn2_order1.local_teacher_is_order_best)
        self.assertAlmostEqual(drn2_order1.local_teacher_cost_gap_to_order_best, 1.0)


def _context(drone_id: str, runtime_state: object, coarse_plan: object) -> SimpleNamespace:
    return SimpleNamespace(
        deciding_drone_id=drone_id,
        t_decision=100.0,
        runtime_state=runtime_state,
        coarse_plan=coarse_plan,
    )


def _cost(costs: dict[tuple[str, str, str], float]):
    def cost_fn(
        context,
        _candidate_out: CandidateOutput,
        action: DispatchAction,
        _order_idx: int,
        _mode_idx: int,
    ) -> float:
        return costs[(str(context.deciding_drone_id), str(action.order_id), str(action.mode))]

    return cost_fn


def _candidate_output(
    order_specs: list[tuple[str, tuple[str, ...]]],
    drone_id: str,
    *,
    order_feature_overrides_by_order: dict[str, dict[str, float]] | None = None,
    teacher_energy_by_order_idx: dict[int, CandidateTeacherEnergy] | None = None,
) -> CandidateOutput:
    order_features: list[OrderFeatures] = []
    order_mask: list[bool] = []
    mode_mask: list[tuple[bool, bool]] = []
    dispatch_actions: dict[tuple[int, int], DispatchAction] = {}
    order_feature_overrides_by_order = dict(order_feature_overrides_by_order or {})

    for order_idx, (order_id, modes) in enumerate(order_specs):
        order_features.append(
            _order_feature(
                order_id,
                **order_feature_overrides_by_order.get(order_id, {}),
            )
        )
        order_mask.append(True)
        has_b = "B" in modes
        has_c = "C" in modes
        mode_mask.append((has_b, has_c))
        if has_b:
            dispatch_actions[(order_idx, 0)] = DispatchAction(order_id=order_id, mode="B")
        if has_c:
            dispatch_actions[(order_idx, 1)] = DispatchAction(order_id=order_id, mode="C")

    return CandidateOutput(
        candidate_features=CandidateFeatures(
            uav_self=_uav_self(drone_id),
            order_features=tuple(order_features),
            infra_features=InfraFeatures(
                truck_x=0.0,
                truck_y=0.0,
                truck_z=0.0,
                plan_version=0,
                future_backbone_node_count=0,
                authorized_order_count=len(order_specs),
                node_features=(),
            ),
        ),
        root_branch_mask=(True, bool(dispatch_actions)),
        has_wait_action=True,
        order_mask=tuple(order_mask),
        mode_mask=tuple(mode_mask),
        factorized_action_schema=FactorizedActionSchema(
            root_branch_order=("WAIT", "DISPATCH"),
            mode_order=("B", "C"),
            max_order_slots=max(1, len(order_specs)),
        ),
        resolved_action_lookup=ResolvedActionLookup(
            wait_action=WAIT_ACTION,
            dispatch_actions=dispatch_actions,
        ),
        teacher_energy_by_order_idx=dict(teacher_energy_by_order_idx or {}),
    )


def _uav_self(drone_id: str) -> UavSelfFeatures:
    return UavSelfFeatures(
        drone_id=drone_id,
        x=0.0,
        y=0.0,
        z=0.0,
        battery_current=100.0,
        battery_max=100.0,
        battery_ratio=1.0,
        training_state="idle",
        has_reservation=False,
        reservation_remaining_sec=0.0,
        plan_version_delta=0,
        is_riding_truck=False,
        drone_source_type="depot",
        cruise_speed=10.0,
        payload_capacity=10.0,
    )


def _order_feature(
    order_id: str,
    **overrides: float,
) -> OrderFeatures:
    values = {
        "deadline": 1000.0,
        "distance_to_order": 100.0,
        "remaining_time": 900.0,
        "best_mode_b_return_score": 200.0,
        "best_mode_b_recovery_flight_time": 30.0,
        "best_mode_b_queue_time_est": 0.0,
        "best_mode_c_wait_time": 20.0,
        "best_mode_c_uav_flight_time": 30.0,
        "best_mode_c_energy_margin_ratio": 0.5,
        "best_mode_c_timeout_risk": 0.0,
    }
    values.update(overrides)
    # Compute estimated_delivery_finish_slack_sec from the other values unless
    # explicitly overridden.  Constants match _context (t_decision=100),
    # _uav_self (cruise_speed=10), and drone_params.yaml (service_time=30).
    _T_DECISION = 100.0
    _CRUISE_SPEED = 10.0
    _SERVICE_TIME = 30.0
    flight_time = float(values["distance_to_order"]) / _CRUISE_SPEED
    values.setdefault(
        "estimated_delivery_finish_slack_sec",
        float(values["deadline"]) - (_T_DECISION + flight_time + _SERVICE_TIME),
    )
    return OrderFeatures(
        order_id=order_id,
        weight=1.0,
        deadline=float(values["deadline"]),
        remaining_time=float(values["remaining_time"]),
        delivery_x=0.0,
        delivery_y=0.0,
        delivery_z=0.0,
        distance_to_order=float(values["distance_to_order"]),
        order_pre_score=0.0,
        priority_band=0,
        has_mode_b_action=True,
        best_mode_b_return_score=float(values["best_mode_b_return_score"]),
        best_mode_b_recovery_flight_time=float(
            values["best_mode_b_recovery_flight_time"]
        ),
        best_mode_b_host_type="depot",
        best_mode_b_queue_time_est=float(values["best_mode_b_queue_time_est"]),
        has_mode_c_action=True,
        mode_c_candidate_count=1,
        best_mode_c_rendezvous_margin=30.0,
        best_mode_c_wait_time=float(values["best_mode_c_wait_time"]),
        best_mode_c_uav_flight_time=float(values["best_mode_c_uav_flight_time"]),
        best_mode_c_energy_margin_ratio=float(values["best_mode_c_energy_margin_ratio"]),
        best_mode_c_node_type="station",
        best_mode_c_truck_eta_remaining=100.0,
        best_mode_c_timeout_risk=float(values["best_mode_c_timeout_risk"]),
        is_valid=True,
        estimated_delivery_finish_slack_sec=float(
            values["estimated_delivery_finish_slack_sec"]
        ),
    )


if __name__ == "__main__":
    unittest.main()
