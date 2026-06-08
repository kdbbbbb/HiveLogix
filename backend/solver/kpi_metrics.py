#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 KPI 指标计算模块（算法无关）。

用于对任意调度算法输出的订单结果做统一评估，便于横向对比：
  1. 综合任务完成率
  2. 准时送达率
  3. 平均订单延迟（分钟）
  4. 总体能耗成本（Wh）
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class DispatchKpi:
    """调度 KPI 聚合结果。"""

    total_orders: int
    completed_orders: int
    completed_with_timing: int
    on_time_completed: int

    completion_rate: float
    on_time_rate: float
    avg_delay_min: float
    total_energy_cost_wh: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _to_seconds(value: Any, time_domain: str | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if time_domain == "wall_ms":
        return v / 1000.0
    return v


def compute_dispatch_kpi(
    orders: Iterable[Mapping[str, Any]],
    total_energy_cost_wh: float = 0.0,
) -> DispatchKpi:
    """
    计算调度 KPI。

    Args:
        orders: 订单序列。每项需包含 status，建议包含 deadline/actual_deliver_time/time_domain。
        total_energy_cost_wh: 累计能耗成本（Wh）。可由求解器 cost_breakdown.energy 累加获得。
    """
    order_list = list(orders)
    total_orders = len(order_list)

    completed = [o for o in order_list if str(o.get("status", "")).upper() == "COMPLETED"]
    completed_orders = len(completed)

    completed_with_timing = 0
    on_time_completed = 0
    delays_min: list[float] = []

    for order in completed:
        time_domain = order.get("time_domain")
        deadline_sec = _to_seconds(order.get("deadline"), time_domain)
        deliver_sec = _to_seconds(order.get("actual_deliver_time"), time_domain)
        if deadline_sec is None or deliver_sec is None:
            continue

        completed_with_timing += 1
        if deliver_sec <= deadline_sec:
            on_time_completed += 1

        delays_min.append(max(0.0, deliver_sec - deadline_sec) / 60.0)

    completion_rate = (completed_orders / total_orders * 100.0) if total_orders > 0 else 0.0
    on_time_rate = (
        on_time_completed / completed_with_timing * 100.0
        if completed_with_timing > 0 else 0.0
    )
    avg_delay_min = sum(delays_min) / len(delays_min) if delays_min else 0.0

    return DispatchKpi(
        total_orders=total_orders,
        completed_orders=completed_orders,
        completed_with_timing=completed_with_timing,
        on_time_completed=on_time_completed,
        completion_rate=completion_rate,
        on_time_rate=on_time_rate,
        avg_delay_min=avg_delay_min,
        total_energy_cost_wh=max(0.0, float(total_energy_cost_wh)),
    )
