#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale benchmark dynamic order spawn times without changing service windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def compress_dynamic_spawn_times(
    *,
    input_path: Path,
    output_path: Path,
    target_last_spawn_s: float,
) -> dict[str, Any]:
    if target_last_spawn_s <= 0.0:
        raise ValueError("target_last_spawn_s must be positive")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    dynamic_orders = data.get("dynamic_orders")
    if not isinstance(dynamic_orders, list) or not dynamic_orders:
        raise ValueError(f"{input_path} has no dynamic_orders list")

    spawn_values = [float(item["spawn_sim_s"]) for item in dynamic_orders]
    current_last_spawn_s = max(spawn_values)
    if current_last_spawn_s <= 0.0:
        raise ValueError("current max spawn_sim_s must be positive")

    scale = float(target_last_spawn_s) / float(current_last_spawn_s)
    for item in dynamic_orders:
        # Keep deadline_offset_s unchanged so each order's own service window is preserved.
        item["spawn_sim_s"] = round(float(item["spawn_sim_s"]) * scale, 6)

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dynamic_order_count": len(dynamic_orders),
        "current_last_spawn_s": current_last_spawn_s,
        "target_last_spawn_s": float(target_last_spawn_s),
        "scale": scale,
        "new_last_spawn_s": max(float(item["spawn_sim_s"]) for item in dynamic_orders),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compress dynamic_orders[*].spawn_sim_s to a target last spawn time "
            "without changing deadline_offset_s."
        )
    )
    parser.add_argument("orders_json", type=Path, help="Orders JSON path")
    parser.add_argument(
        "--target-last-spawn-s",
        type=float,
        required=True,
        help="Desired maximum dynamic_orders[*].spawn_sim_s after scaling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Omit with --in-place to overwrite orders_json.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite orders_json in place.",
    )
    args = parser.parse_args(argv)

    input_path = args.orders_json.resolve()
    if bool(args.in_place) == (args.output is not None):
        raise ValueError("Use exactly one of --in-place or --output")
    output_path = input_path if args.in_place else args.output.resolve()

    summary = compress_dynamic_spawn_times(
        input_path=input_path,
        output_path=output_path,
        target_last_spawn_s=float(args.target_last_spawn_s),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
