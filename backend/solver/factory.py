#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 求解器工厂。

统一管理算法名称到实现类的映射，供 DecisionEngine 按配置创建求解器。
"""

from __future__ import annotations

from typing import Any, Callable

from solver.greedy_baseline import GreedyBaseline
from solver.ga_mmce.solver import GAMMCESolver
from solver.greedy_mmce import GreedyMMCE
from solver.greedy_mmce_bi import GreedyMMCEBackboneInsertion
from solver.market_based_solver import MarketBasedSolver
from solver.interfaces import DispatchSolver


SolverBuilder = Callable[[Any], DispatchSolver]


_SOLVER_BUILDERS: dict[str, SolverBuilder] = {
    "greedy": lambda entity_mgr: GreedyBaseline(entity_mgr),
    "ga_mmce": lambda entity_mgr: GAMMCESolver(entity_mgr),
    "genetic": lambda entity_mgr: GAMMCESolver(entity_mgr),
    "greedy_mmce": lambda entity_mgr: GreedyMMCE(entity_mgr),
    "greedy_mmce_bi": lambda entity_mgr: GreedyMMCEBackboneInsertion(entity_mgr),
    "mmce_backbone_insertion": lambda entity_mgr: GreedyMMCEBackboneInsertion(entity_mgr),
    "mmce-bi": lambda entity_mgr: GreedyMMCEBackboneInsertion(entity_mgr),
    "market": lambda entity_mgr: MarketBasedSolver(entity_mgr),
}


def register_solver(name: str, builder: SolverBuilder) -> None:
    """注册新的求解器构造器。"""
    key = name.strip().lower()
    if not key:
        raise ValueError("solver 名称不能为空")
    _SOLVER_BUILDERS[key] = builder


def create_solver(name: str, entity_mgr: Any) -> DispatchSolver:
    """按名称创建求解器实例。"""
    key = name.strip().lower()
    builder = _SOLVER_BUILDERS.get(key)
    if builder is None:
        available = ", ".join(sorted(_SOLVER_BUILDERS))
        raise ValueError(f"未知求解器 '{name}'，可用: {available}")
    return builder(entity_mgr)


def list_solvers() -> list[str]:
    """返回当前可用求解器名称列表。"""
    return sorted(_SOLVER_BUILDERS.keys())
