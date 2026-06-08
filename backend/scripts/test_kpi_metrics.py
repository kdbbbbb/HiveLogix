#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KPI 计算测试脚本。

用法示例：
  python scripts/test_kpi_metrics.py
  python scripts/test_kpi_metrics.py --orders-json ./tmp/orders_sample.json --energy-wh 1234.5

JSON 输入格式示例：
{
  "orders": [
    {"order_id": "ORD-1", "status": "COMPLETED", "deadline": 300, "actual_deliver_time": 280, "time_domain": "sim_s"},
    {"order_id": "ORD-2", "status": "COMPLETED", "deadline": 500, "actual_deliver_time": 620, "time_domain": "sim_s"},
    {"order_id": "ORD-3", "status": "ASSIGNED",  "deadline": 700, "time_domain": "sim_s"}
  ],
  "energy_wh": 800.0
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solver.kpi_metrics import compute_dispatch_kpi


def _default_orders() -> list[dict]:
    return [
        {
            "order_id": "ORD-DEMO-1",
            "status": "COMPLETED",
            "deadline": 300.0,
            "actual_deliver_time": 280.0,
            "time_domain": "sim_s",
        },
        {
            "order_id": "ORD-DEMO-2",
            "status": "COMPLETED",
            "deadline": 500.0,
            "actual_deliver_time": 620.0,
            "time_domain": "sim_s",
        },
        {
            "order_id": "ORD-DEMO-3",
            "status": "ASSIGNED",
            "deadline": 700.0,
            "time_domain": "sim_s",
        },
        {
            "order_id": "ORD-DEMO-4",
            "status": "PENDING",
            "deadline": 900.0,
            "time_domain": "sim_s",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="计算并打印调度 KPI 指标")
    parser.add_argument(
        "--orders-json",
        type=str,
        default="",
        help="订单 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--energy-wh",
        type=float,
        default=0.0,
        help="总体能耗成本（Wh）",
    )
    args = parser.parse_args()

    orders: list[dict]
    energy_wh = args.energy_wh

    if args.orders_json:
        path = Path(args.orders_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        orders = list(data.get("orders", []))
        if "energy_wh" in data and args.energy_wh == 0.0:
            energy_wh = float(data["energy_wh"])
    else:
        orders = _default_orders()

    kpi = compute_dispatch_kpi(orders=orders, total_energy_cost_wh=energy_wh)

    print("=== Dispatch KPI ===")
    print(f"total_orders: {kpi.total_orders}")
    print(f"completed_orders: {kpi.completed_orders}")
    print(f"completion_rate: {kpi.completion_rate:.2f}%")
    print(f"on_time_rate: {kpi.on_time_rate:.2f}%")
    print(f"avg_delay_min: {kpi.avg_delay_min:.3f}")
    print(f"total_energy_cost_wh: {kpi.total_energy_cost_wh:.3f}")


if __name__ == "__main__":
    main()
