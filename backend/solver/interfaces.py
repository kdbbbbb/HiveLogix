#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 求解器接口定义。

用于将调度编排层（DecisionEngine）与具体算法实现解耦。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from solver.greedy_mmce import DispatchPlan

if TYPE_CHECKING:
    from core.entities.order import Order


class DispatchSolver(Protocol):
    """调度求解器协议。"""

    def dispatch(
        self,
        pending_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """执行一轮求解并返回标准化 DispatchPlan。"""
        ...

    def dispatch_incremental(
        self,
        new_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """执行增量调度并返回标准化 DispatchPlan。"""
        ...

    def should_replan_unfinished(self) -> bool:
        """是否默认启用“新单+未完成单”滚动重优化。"""
        ...

    def dispatch_replan_current_state(
        self,
        replan_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """按当前实体状态执行重优化（不中断物理连续性）。"""
        ...

    def get_active_contracts(self) -> list[Any]:
        """返回当前活跃契约列表；无契约机制时返回空列表。"""
        ...

    def fulfill_contract(self, contract_id: str) -> None:
        """标记契约已兑现；无契约机制时可为 no-op。"""
        ...

    def build_incremental_route_from_stops(
        self,
        truck: Any,
        ordered_stops: list[dict],
        current_time: float,
    ) -> Any:
        """按给定停靠顺序重建增量后缀路线；失败返回 None。"""
        ...
