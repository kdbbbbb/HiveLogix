#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 离线训练子包。

当前阶段先固化跨模块共享的训练期契约类型，后续逐步补充：
  - scene_loader
  - order_source_adapter
  - planner_bridge
  - candidate_builder
  - env_adapter
"""

from .scene_loader import (
    BenchmarkDynamicOrder,
    TrainingRoadNetwork,
    TrainingSceneContext,
    load_default_scene,
    load_training_scene,
)
from .order_source_adapter import (
    OrderSourceConfig,
    OrderSourceMode,
    PoissonOrderGenConfig,
    build_order_source,
    build_order_source_preview_summary,
    configure_order_manager_for_source,
    ensure_mode_allowed,
    preview_dynamic_order_stream,
)
from .env_adapter import (
    DecisionContext,
    EnvStepResult,
    NodeStateView,
    ReservationStateView,
    RuntimeStateView,
    TrainingDroneState,
    TrainingEnvAdapter,
)
from .actions import DispatchAction, GlobalWaitAction, WAIT_ACTION
from .candidate_builder import CandidateBuilder
from .contracts import CandidateOutput
from .critic_batch_builder import (
    CriticBatchBuilder,
    build_default_critic_tensor_schema_meta,
)
from .observation_tensorizer import ObservationTensorizer
from .planner_bridge import PlannerBridge
from .rollout_buffer import RolloutBuffer
from .train_cmrappo import train_cmrappo
from .export_sumo_truck_route import (
    Phase4ExportResult,
    TruckExecutionRoute,
    export_phase4_truck_route,
)

__all__ = [
    "BenchmarkDynamicOrder",
    "OrderSourceConfig",
    "OrderSourceMode",
    "PoissonOrderGenConfig",
    "Phase4ExportResult",
    "TrainingRoadNetwork",
    "TrainingSceneContext",
    "TruckExecutionRoute",
    "build_order_source",
    "build_order_source_preview_summary",
    "build_default_critic_tensor_schema_meta",
    "CandidateBuilder",
    "CandidateOutput",
    "configure_order_manager_for_source",
    "CriticBatchBuilder",
    "DecisionContext",
    "DispatchAction",
    "EnvStepResult",
    "export_phase4_truck_route",
    "ensure_mode_allowed",
    "GlobalWaitAction",
    "load_default_scene",
    "load_training_scene",
    "NodeStateView",
    "ObservationTensorizer",
    "preview_dynamic_order_stream",
    "PlannerBridge",
    "ReservationStateView",
    "RolloutBuffer",
    "RuntimeStateView",
    "TrainingDroneState",
    "TrainingEnvAdapter",
    "WAIT_ACTION",
    "train_cmrappo",
]
