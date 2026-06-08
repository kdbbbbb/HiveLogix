#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 订单源适配器自测脚本。

运行方式：
  python backend/scripts/inspect_order_sources.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.order_source_adapter import (
    OrderSourceMode,
    build_order_source,
    build_order_source_preview_summary,
)
from training.scene_loader import load_default_scene


def main() -> int:
    scene_ctx = load_default_scene()
    horizon_sec = 3600.0

    summaries: dict[str, dict] = {}
    sources = {}
    for mode in (
        OrderSourceMode.BENCHMARK,
        OrderSourceMode.POISSON,
        OrderSourceMode.HYBRID,
    ):
        order_source = build_order_source(scene_ctx, mode=mode)
        sources[mode.value] = order_source
        summaries[mode.value] = build_order_source_preview_summary(
            scene_ctx,
            order_source,
            horizon_sec=horizon_sec,
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))

    benchmark = summaries["benchmark"]
    poisson = summaries["poisson"]
    hybrid = summaries["hybrid"]
    benchmark_dynamic_ids = {item.order.order_id for item in scene_ctx.dynamic_orders}
    poisson_scheduled_ids = {
        str(entry["order_id"]) for entry in sources["poisson"].scheduled_dynamic_orders
    }

    assert benchmark["arrival_rate"] == 0.0, "benchmark 模式必须关闭泊松流"
    assert benchmark["scheduled_dynamic_orders"] == len(scene_ctx.dynamic_orders)
    assert benchmark["initial_static_uav_orders"] > 0, "benchmark 必须注入静态 UAV 单"

    assert poisson["initial_static_uav_orders"] > 0, "poisson 训练必须注入静态 UAV 单"
    assert not (
        benchmark_dynamic_ids & poisson_scheduled_ids
    ), "poisson 模式不得回放 benchmark dynamic_orders"
    assert poisson["arrival_rate"] > 0.0, "poisson 模式必须启用泊松流"

    assert hybrid["scheduled_dynamic_orders"] >= len(scene_ctx.dynamic_orders)
    assert hybrid["initial_static_uav_orders"] > 0, "hybrid 必须注入静态 UAV 单"
    assert hybrid["arrival_rate"] > 0.0, "hybrid 模式必须同时包含泊松流"

    same_seed_a = build_order_source_preview_summary(
        scene_ctx,
        build_order_source(scene_ctx, mode=OrderSourceMode.POISSON, seed=20260424),
        horizon_sec=horizon_sec,
        preview_limit=16,
    )
    same_seed_b = build_order_source_preview_summary(
        scene_ctx,
        build_order_source(scene_ctx, mode=OrderSourceMode.POISSON, seed=20260424),
        horizon_sec=horizon_sec,
        preview_limit=16,
    )
    diff_seed = build_order_source_preview_summary(
        scene_ctx,
        build_order_source(scene_ctx, mode=OrderSourceMode.POISSON, seed=20260425),
        horizon_sec=horizon_sec,
        preview_limit=16,
    )

    assert (
        same_seed_a["preview_dynamic_order_count"]
        == same_seed_b["preview_dynamic_order_count"]
    )
    assert (
        same_seed_a["preview_orders_head"] == same_seed_b["preview_orders_head"]
    ), "相同 seed 下泊松订单流必须完全一致"
    assert (
        same_seed_a["preview_orders_head"] != diff_seed["preview_orders_head"]
    ), "不同 seed 下泊松订单流应发生变化"

    print("Phase 3 order_source_adapter self-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
