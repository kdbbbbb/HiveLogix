from __future__ import annotations

import math
from typing import Any

from .config import GAConfig


def compute_fitness(decode_result: Any, config: GAConfig) -> float:
    """Return the scalar GA fitness for a decoded Individual.

    Smaller is better. This function is intentionally pure: it does not repair
    routes, mutate state, or reinterpret penalty names. Decoder has already
    converted every penalty value into a weighted cost.
    """
    big_m = _safe_float(getattr(config, "big_m", 1e9), 1e9)

    objective = _safe_float(getattr(decode_result, "objective", 0.0), 0.0)
    if not math.isfinite(objective):
        return big_m

    penalties = getattr(decode_result, "penalties", {}) or {}
    penalty_cost = _sum_penalty_costs(penalties)
    if not math.isfinite(penalty_cost):
        return big_m

    # penalty_counts is diagnostic only. If an infeasible result forgot to
    # provide penalty costs, add one fallback cost rather than silently ranking
    # it as cheap.
    feasible = bool(getattr(decode_result, "feasible", False))
    if not feasible and not penalties:
        penalty_cost += _safe_float(getattr(config, "weight_infeasible", big_m), big_m)

    fitness = objective + penalty_cost
    return fitness if math.isfinite(fitness) else big_m


def _sum_penalty_costs(penalties: Any) -> float:
    if not isinstance(penalties, dict):
        return 0.0

    total = 0.0
    for value in penalties.values():
        cost = _safe_float(value, 0.0)
        if not math.isfinite(cost):
            return math.inf
        total += cost
    return total


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)
