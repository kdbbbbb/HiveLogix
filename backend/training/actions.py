#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 训练侧共享动作定义。

Phase 6 起，`candidate_builder` 与 `env_adapter` 都需要构造和消费同一套
`EnvAction`，因此把动作类型从 `env_adapter.py` 中独立出来，避免循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class DispatchAction:
    """一次派送动作。"""

    order_id: str
    mode: str
    recover_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"B", "C"}:
            raise ValueError(f"dispatch_action.mode 仅允许 B/C，实际={self.mode}")
        if self.mode == "B" and self.recover_node_id is not None:
            raise ValueError("mode B 不应携带 recover_node_id")


@dataclass(frozen=True)
class GlobalWaitAction:
    """全局 WAIT 动作。"""

    name: str = "WAIT"


WAIT_ACTION = GlobalWaitAction()

EnvAction: TypeAlias = DispatchAction | GlobalWaitAction


__all__ = [
    "DispatchAction",
    "EnvAction",
    "GlobalWaitAction",
    "WAIT_ACTION",
]
