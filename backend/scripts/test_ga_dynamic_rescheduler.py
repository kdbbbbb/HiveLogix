#!/usr/bin/env python3
"""Smoke test for GA-MMCE static solve followed by dynamic rescheduling.

Run from the repository root:
    python backend/scripts/test_ga_dynamic_rescheduler.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
GEO = BACKEND / "environment" / "geo"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(GEO))

from core.entities.order import Order
from core.entities.primitives import Position3D, SourceType
from environment.state.entity_manager import EntityManager
from solver.ga_mmce.config import GAConfig
from solver.ga_mmce.solver import GAMMCESolver
from utils.coord_utils import wgs84_to_utm


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_order(raw: dict, order_id: str | None = None, create_time: float | None = None) -> Order:
    x, y = wgs84_to_utm(float(raw["delivery_lng"]), float(raw["delivery_lat"]))
    return Order(
        order_id=order_id or raw["order_id"],
        create_time=float(raw.get("create_time", 0.0) if create_time is None else create_time),
        deadline=float(raw["deadline"]),
        delivery_loc=Position3D(x=x, y=y, z=float(raw.get("delivery_z", 0.0))),
        pickup_source_id=raw.get("pickup_source_id") or "DEP-TEST-01",
        source_type=SourceType(raw.get("source_type") or "DEPOT"),
        payload_weight=float(raw["payload_weight"]),
    )


def _make_dynamic_order(seed_order: dict) -> Order:
    dynamic = dict(seed_order)
    dynamic.update(
        {
            "order_id": "ORD-DYN-TEST-01",
            "create_time": 600,
            "deadline": 3600,
            "delivery_lng": 121.2125,
            "delivery_lat": 31.0435,
            "payload_weight": 2.0,
        }
    )
    return _make_order(dynamic, order_id="ORD-DYN-TEST-01", create_time=600)


def main() -> None:
    random.seed(42)
    scene_dir = ROOT / "backend" / "test_data" / "default_scene"
    entities = _load_json(scene_dir / "entities.json")
    orders_data = _load_json(scene_dir / "orders.json")
    static_orders = {
        raw["order_id"]: _make_order(raw)
        for raw in orders_data["static_orders"]
    }

    entity_mgr = EntityManager()
    entity_mgr.load_from_config({"entities": entities})

    bbox = {
        "minx": 121.195602,
        "miny": 31.023747,
        "maxx": 121.237661,
        "maxy": 31.059783,
    }
    config = GAConfig(
        population_size=18,
        generations=18,
        min_generations=5,
        early_stopping_patience=6,
        random_seed=42,
        save_evolution_csv=False,
        save_evolution_plots=False,
        diagnostics_enabled=False,
        max_runtime_seconds=8.0,
    )
    solver = GAMMCESolver(entity_mgr, config=config)

    static_state = SimpleNamespace(
        entity_mgr=entity_mgr,
        orders=static_orders,
        current_time=0.0,
        bbox=bbox,
        scene_id="default_test_4x4km",
    )
    static_plan = solver.solve(static_state, dispatch_type="full")
    assert solver.last_best_individual is not None
    assert static_plan.summary.get("total_orders") == len(static_orders)
    assert static_plan.summary.get("feasible") == len(static_orders)

    dynamic_order = _make_dynamic_order(orders_data["static_orders"][0])
    dynamic_state = SimpleNamespace(
        entity_mgr=entity_mgr,
        orders=static_orders,
        current_time=600.0,
        bbox=bbox,
        scene_id="default_test_4x4km",
    )
    dynamic_plan = solver.reschedule_on_event(
        dynamic_state,
        {"ORD-DYN-TEST-01": dynamic_order},
        600.0,
    )
    dynamic_alloc_ids = [allocation.order_id for allocation in dynamic_plan.allocations]

    assert dynamic_plan.summary.get("dispatch_type") == "dynamic_replan"
    assert dynamic_plan.summary.get("solver") == "ga_mmce"
    assert dynamic_plan.summary.get("dynamic_solver") == "greedy_mmce_bi"
    assert dynamic_plan.summary.get("ga_dynamic_skipped") is True
    assert "ORD-DYN-TEST-01" in dynamic_plan.summary.get("new_order_ids", [])
    assert "ORD-DYN-TEST-01" in dynamic_alloc_ids
    assert dynamic_plan.summary.get("warm_start_count", 0) == 0
    assert dynamic_plan.summary.get("reoptimized_order_count", 0) > 0
    assert dynamic_plan.summary.get("feasible") == dynamic_plan.summary.get("total_orders")
    assert dynamic_plan.summary.get("unserved_order_ids") == []

    print(
        "STATIC",
        static_plan.summary.get("dispatch_type"),
        static_plan.summary.get("total_orders"),
        static_plan.summary.get("feasible"),
        static_plan.summary.get("ga_feasible"),
    )
    print(
        "DYNAMIC",
        dynamic_plan.summary.get("dispatch_type"),
        dynamic_plan.summary.get("total_orders"),
        dynamic_plan.summary.get("feasible"),
        dynamic_plan.summary.get("ga_feasible"),
        dynamic_plan.summary.get("fallback_used"),
        dynamic_plan.summary.get("fallback_level"),
    )
    print("DYNAMIC_NEW_IDS", dynamic_plan.summary.get("new_order_ids"))
    print(
        "DYNAMIC_COUNTS",
        "reopt",
        dynamic_plan.summary.get("reoptimized_order_count"),
        "frozen",
        dynamic_plan.summary.get("frozen_future_order_count"),
        "warm",
        dynamic_plan.summary.get("warm_start_count"),
    )
    print("DYNAMIC_MODES", dynamic_plan.summary.get("modes"))
    print("DYNAMIC_UNSERVED", dynamic_plan.summary.get("unserved_order_ids"))


if __name__ == "__main__":
    main()
