#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — Phase 5c 训练环境适配器。

职责边界：
  - 读取 Phase 2/3/4 产物，提供事件驱动的最小可运行训练环境；
  - 维护训练侧无人机状态机覆盖层，不污染 core 层物理枚举；
  - 在 Phase 5 内部内联实现最小 coarse plan / action mask；
  - 保证 reset -> step -> done 连通，且为 Phase 5b/5c 保留稳定接口。

本文件当前实现到 Phase 5c：
  - 完整 action_mask：mode C 候选按时序/能量精细过滤
  - 完整 post_delivery_revalidation：送达后对 mode C 原选回收点做三条件复核
  - mode B 优先返仓；电量不足返仓时先到可达 station 补能再自动返仓
  - reservation 状态机与 timeout 触发 fallback
  - 完整 per-dt reward：T_idle / T_fallback；overdue 仅作为统计与特征
  - WAIT 动作的 T_idle 一次性精确结算
  - hard overdue 作为 severe late 指标，不强制移除订单
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, replace
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from core.entities.charging_host import ChargingHost
from core.entities.depot import Depot
from core.entities.drone import Drone
from core.entities.order import Order
from core.entities.primitives import Position3D, SourceType, TaskStatus
from core.entities.swap_station import SwapStation
from core.entities.truck import Truck
from environment.geo.osm_service import build_road_graph, find_nearest_node, shortest_path
from environment.state.entity_manager import EntityManager
from environment.state.order_manager import OrderManager

from .actions import DispatchAction, EnvAction, GlobalWaitAction, WAIT_ACTION
from .candidate_builder import CandidateBuilder
from .contracts import (
    CandidateOutput,
    CoarsePlanView,
    DecisionExecutionSnapshot,
    DecisionPlannerSnapshot,
    PlannerMode,
    PlannerTriggerContext,
    PolicyMode,
    ReservationConstraintState,
    ReservationPlanOutcome,
    ReservationPlanStatus,
    RouteDriftRef,
    TruckPlanStopView,
    TruckReservationConstraint,
)
from .order_source_adapter import (
    OrderSourceConfig,
    OrderSourceMode,
    build_order_source,
    configure_order_manager_for_source,
)
from .planner_bridge import PlannerBridge
from .recovery_pool_selector import select_recovery_pool_for_order
from .scene_loader import DEFAULT_CONFIG_PATH, TrainingSceneContext, load_default_scene
from .uav_path_service import TrainingUavPathService


NodeId: TypeAlias = str
OrderId: TypeAlias = str
DroneId: TypeAlias = str

_TIME_EPS = 1e-6
_BACKBONE_DEPARTURE_EPS = 1e-6
_MIN_FUTURE_STATION_BACKBONE_VISITS = 3
_MODE_C_REVALIDATION_REASON_KEYS = (
    "energy_feasible",
    "rendezvous_time_feasible",
    "node_still_valid",
)
FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER = "planner_invalidated_for_truck_order"
FALLBACK_CAUSE_C_REVALIDATION_FAILED = "c_revalidation_failed"
FALLBACK_CAUSE_RENDEZVOUS_WAIT_TIMEOUT = "rendezvous_wait_timeout"
FALLBACK_CAUSE_ENERGY_OR_NODE_INVALID = "energy_or_node_invalid"
FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE = "no_post_delivery_c_node"
FALLBACK_CAUSE_HARD_FAILURE_FALLBACK = "hard_failure_fallback"
FALLBACK_CAUSE_NONE = "none"
_FALLBACK_CAUSE_KEYS = (
    FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
    FALLBACK_CAUSE_C_REVALIDATION_FAILED,
    FALLBACK_CAUSE_RENDEZVOUS_WAIT_TIMEOUT,
    FALLBACK_CAUSE_ENERGY_OR_NODE_INVALID,
    FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE,
    FALLBACK_CAUSE_HARD_FAILURE_FALLBACK,
)
_SYSTEM_ATTRIBUTED_FALLBACK_CAUSES = {
    FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
}


def _normalize_fallback_cause(cause: str | None) -> str:
    if cause is None:
        return FALLBACK_CAUSE_HARD_FAILURE_FALLBACK
    normalized = str(cause)
    if normalized in _FALLBACK_CAUSE_KEYS:
        return normalized
    return FALLBACK_CAUSE_HARD_FAILURE_FALLBACK


def _is_ppo_attributed_fallback_cause(cause: str | None) -> bool:
    return _normalize_fallback_cause(cause) not in _SYSTEM_ATTRIBUTED_FALLBACK_CAUSES


class TrainingDroneState(StrEnum):
    """训练环境内部使用的无人机状态枚举。"""
    # 训练侧状态覆盖层；不修改 core 层 DroneStatus。
    IDLE = "idle"
    FLYING_TO_DELIVER = "flying_to_deliver"
    DELIVERY_SERVICE = "delivery_service"
    DELIVERED = "delivered"
    RETURN_TO_RENDEZVOUS = "return_to_rendezvous"
    WAITING_FOR_TRUCK = "waiting_for_truck"
    RETURN_TO_STATION = "return_to_station"
    RETURN_TO_DEPOT = "return_to_depot"
    QUEUEING_AT_HOST = "queueing_at_host"
    CHARGING_OR_SWAP = "charging_or_swap"
    ACTIVE_WAIT = "active_wait"
    FALLBACK_RECOVERY = "fallback_recovery"
    CHARGING_ON_TRUCK = "charging_on_truck"
    RIDING_WITH_TRUCK = "riding_with_truck"
    AIRBORNE_ENERGY_FAILURE = "airborne_energy_failure"


@dataclass(frozen=True)
class ReservationStateView:
    """reservation 只读视图。
    """
    recover_node: str
    issued_at: float


@dataclass(frozen=True)
class ReservationState:
    """训练环境内部维护的 mode C reservation 真值。"""
    recover_node: str
    issued_at: float


@dataclass(frozen=True)
class DroneStateView:
    """单架无人机在 runtime_state 中的只读快照。"""
    drone_id: str
    training_state: str
    current_loc: Position3D
    battery_current: float
    battery_max: float
    battery_ratio: float
    carrying_order_id: str | None
    home_type: str
    cruise_speed: float
    payload_capacity: float
    empty_weight: float
    k1: float
    k2: float
    reservation: ReservationStateView | None


@dataclass(frozen=True)
class NodeStateView:
    """固定充换电节点在 runtime_state 中的只读快照。"""
    node_id: str
    node_type: str
    position: Position3D
    parking_slots: int
    swap_time: float
    queue_length: int
    available_slots: int


@dataclass(frozen=True)
class TruckRoadRoute:
    """一次卡车 OSM 路网寻路结果。"""

    geometry: tuple[Position3D, ...]
    osm_node_path: tuple[str, ...]


@dataclass(frozen=True)
class ActiveTruckRouteContext:
    """卡车正在既有 OSM 路径上行驶时的起点上下文。"""

    position: Position3D
    segment_id: int
    traveled_m: float
    remaining_osm_node_path: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeStateView:
    """当前时刻暴露给训练侧的运行时状态快照。"""
    # Phase 5 已固定对外 schema；后续阶段只能填充实现，不能改字段形状。
    t_now: float
    truck_current_loc: Position3D
    drone_states: Mapping[str, DroneStateView]
    pending_orders: Mapping[str, Order]
    assigned_orders: Mapping[str, Order]
    node_states: Mapping[str, NodeStateView]
    reservation_count: Mapping[str, int]


@dataclass(frozen=True)
class PlannerRuntimeStateView:
    """PlannerBridge 内部视图：保留 PPO runtime schema，并显式携带卡车必经订单。"""

    t_now: float
    truck_current_loc: Position3D
    drone_states: Mapping[str, DroneStateView]
    pending_orders: Mapping[str, Order]
    assigned_orders: Mapping[str, Order]
    node_states: Mapping[str, NodeStateView]
    reservation_count: Mapping[str, int]
    truck_mandatory_orders: Mapping[str, Order]
    require_station_backbone: bool = False
    truck_route_ready_at: float | None = None


@dataclass(frozen=True)
class DecisionContext:
    """一次 PPO 决策点对应的上下文。"""
    # 当前实现只返回 action_lookup；还没有单独暴露 dense action_mask 张量。
    decision_id: int
    t_decision: float
    deciding_drone_id: str
    trigger_type: str
    trigger_station_id: str | None
    runtime_state: RuntimeStateView
    coarse_plan: CoarsePlanView
    planner_snapshot: DecisionPlannerSnapshot
    execution_snapshot: DecisionExecutionSnapshot
    action_lookup: tuple[EnvAction, ...]


@dataclass(frozen=True)
class EnvStepResult:
    """一次 `reset()` 或 `step()` 调用返回的统一结果对象。"""
    reward: float
    done: bool
    runtime_state: RuntimeStateView
    decision_context: DecisionContext | None
    info: Mapping[str, Any]


@dataclass(frozen=True)
class FallbackLeg:
    """fallback_recovery 执行段的内部账本。"""
    # fallback_recovery 的退出完全依赖这份账本；到达宿主后必须清空。
    host_node_id: str
    host_node_type: str
    arrival_time: float
    cause: str = FALLBACK_CAUSE_HARD_FAILURE_FALLBACK


@dataclass(frozen=True)
class BackboneVisit:
    """卡车对固定节点的一次未来访问记录。"""
    # _full_backbone_cache 的单条访问记录；允许同一 node_id 多次出现。
    node_id: str
    arrival_time: float
    departure_time: float


@dataclass(frozen=True)
class PlannedStop:
    """卡车事件队列中的一个停靠点。"""
    # 统一承接 Phase 4 stops 和 poisson 巡站追加停靠点。
    seq: int
    node_type: str
    node_id: str
    position: Position3D
    order_id: str | None
    arrival_time: float
    departure_time: float


@dataclass(frozen=True)
class PlannedTruckSegment:
    """卡车运行时的一段真实路网几何。"""
    segment_id: int
    from_node_id: str
    to_node_id: str
    from_node_type: str
    to_node_type: str
    start_time: float
    end_time: float
    distance_m: float
    geometry: tuple[Position3D, ...]
    cumulative_distances_m: tuple[float, ...]
    osm_node_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlightLeg:
    """无人机当前正在执行的一段飞行。"""
    # 当前实现把所有空中过程统一收敛为绝对时刻的飞行段，事件推进只看 arrival_time。
    kind: str
    start_time: float
    arrival_time: float
    start_pos: Position3D
    target_pos: Position3D
    path_points: tuple[Position3D, ...] = ()
    cumulative_distances_m: tuple[float, ...] = ()
    distance_m: float = 0.0
    route_version: int = 0
    motion_mode: str = "straight_line"
    order_id: str | None = None
    target_node_id: str | None = None
    target_node_type: str | None = None
    energy_cost_j: float = 0.0


@dataclass(frozen=True)
class DeliveryServiceLeg:
    """无人机送达客户后在客户点执行的显式服务停留。"""
    order_id: str
    start_time: float
    finish_time: float
    service_pos: Position3D


@dataclass(frozen=True)
class DispatchCommit:
    """记录某架无人机当前订单的 dispatch 承诺。"""
    order_id: str
    mode: PolicyMode
    selected_recover_node: str | None
    trigger_station_id: str | None
    planned_truck_arrival_time: float | None = None
    planned_uav_arrival_time_lb: float | None = None
    planned_execution_slack_sec: float | None = None


@dataclass(frozen=True)
class ModeCPostDeliverySelection:
    """送达后按当前 truck future backbone 选出的 mode C 回收点。"""

    recover_node_id: str
    planned_truck_arrival_time: float
    planned_uav_arrival_time: float
    planned_execution_slack_sec: float
    uav_flight_time_sec: float
    wait_time_sec: float
    eta_source: str = "unknown"


@dataclass(frozen=True)
class _ModeCRevalidationEtaResolution:
    truck_eta: float | None
    eta_source: str
    commit_eta: float | None
    current_coarse_plan_eta: float | None
    current_coarse_plan_version: int | None
    execution_backbone_eta: float | None
    future_backbone_count: int
    truck_replan_pending: bool
    truck_replan_pending_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AppliedDecision:
    """一次已提交到环境的决策。

    该对象只拆分 `step()` 内部语义，不改变 PPO transition 的奖励归因口径。
    """
    drone_id: str
    carried_reward: float
    info: Mapping[str, Any]
    auto_advance_kind: str
    wait_until: float | None = None


@dataclass(frozen=True)
class DecisionTrigger:
    """内部决策队列中的一个待处理触发点。"""
    decision_id: int
    drone_id: str
    trigger_type: str
    trigger_station_id: str | None = None
    t_enqueued: float = 0.0


@dataclass(frozen=True)
class _YamlConfig:
    """从训练 YAML 中提取出的 5b 运行参数子集。"""
    upper_horizon_sec: float
    patrol_min_remaining_sec: float
    patrol_stations_per_loop: int
    allow_empty_backbone_route: bool
    max_wait_decision_gap_sec: float
    max_candidate_recovery_per_order: int
    recovery_pool_future_scan_limit: int
    rendezvous_filter_margin_sec: float
    rendezvous_execution_margin_sec: float
    rendezvous_max_wait_sec: float
    reservation_enabled: bool
    reservation_alpha: float
    reservation_beta: float
    reservation_gamma: float
    reservation_drift_eta_abs_threshold_sec: float
    wait_idle_penalty_coef: float
    wait_opportunity_penalty_coef: float
    lambda_miss: float
    lambda_res_timeout: float
    lambda_overdue: float
    R_delivery_bonus: float
    late_delivery_penalty_coef: float
    late_delivery_penalty_cap: float
    min_late_delivery_reward: float
    mode_c_attempt_bonus: float
    uav_energy_penalty_coef: float
    uav_energy_penalty_cap_ratio: float
    max_overdue_sec: float
    hard_overdue_penalty_sec: float
    hard_failure_penalty_sec: float
    support_radius_km: float
    fallback_burst_window_sec: float
    coarse_new_order_trigger: int


class TrainingEnvAdapter:
    """
    Phase 5b 训练环境。

    当前只支持单 truck / 单 depot 场景，与 Phase 4 路线导出约束保持一致。
    """

    def __init__(
        self,
        *,
        scene_ctx: TrainingSceneContext | None = None,
        order_source: OrderSourceConfig | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        phase4_route_dir: str | Path | None = None,
        enable_phase6: bool = True,
        planner_bridge: PlannerBridge | None = None,
        candidate_builder: CandidateBuilder | None = None,
    ) -> None:
        """构造环境对象并装载静态依赖。

        这里只准备配置、scene 和订单源；真正的 episode 运行态在 `reset()` 里创建。
        """
        self._config_path = Path(config_path)
        self._cfg = _load_env_yaml(self._config_path)

        self._scene_ctx = scene_ctx or load_default_scene(config_path=self._config_path)
        self._order_source = order_source or build_order_source(
            self._scene_ctx,
            config_path=self._config_path,
        )
        self._uav_path_service = TrainingUavPathService(scene_ctx=self._scene_ctx)
        self._candidate_builder = candidate_builder or CandidateBuilder(
            scene_ctx=self._scene_ctx,
            config_path=self._config_path,
            uav_path_service=self._uav_path_service,
        )
        self._phase4_route_dir = (
            Path(phase4_route_dir)
            if phase4_route_dir is not None
            else Path(self._scene_ctx.scene_bundle_dir) / "sumo" / "phase4_truck_route"
        )

        self._entity_manager: EntityManager | None = None
        self._order_manager: OrderManager | None = None
        self._truck: Truck | None = None
        self._truck_id: str | None = None
        self._depot: Depot | None = None
        self._depot_id: str | None = None
        self._heavy_payload_capacity = self._resolve_heavy_payload_capacity()
        self._recovery_pool_drone_cruise_speed = (
            self._resolve_recovery_pool_drone_cruise_speed()
        )

        self._t_now = 0.0
        self._reset_count = 0
        self._current_episode_order_source_seed = int(self._order_source.seed)
        self._planned_route_stops: list[PlannedStop] = []
        self._planned_route_segments: list[PlannedTruckSegment] = []
        self._planned_route_stop_i = 1
        self._full_backbone_cache: list[BackboneVisit] = []
        self._allow_empty_backbone_route = False
        self._truck_route_version = 0
        self._drone_path_version: dict[DroneId, int] = {}
        self._road_graph: Any | None = None
        self._road_nodes: Mapping[str, tuple[float, float]] | None = None
        self._truck_road_route_geometry_cache: dict[
            tuple[float, float, float, float],
            TruckRoadRoute,
        ] = {}
        self._active_truck_route_context_cache: tuple[
            int,
            float,
            ActiveTruckRouteContext | None,
        ] | None = None

        self._drone_state: dict[DroneId, TrainingDroneState] = {}
        # 记录 active_wait 结束后应恢复到哪个基础状态。
        # idle 路径仍在 step(WAIT) 入口一次性结算；riding_with_truck 路径按真实 dt 累计 T_idle。
        self._active_wait_resume: dict[DroneId, TrainingDroneState] = {}
        self._flight_legs: dict[DroneId, FlightLeg] = {}
        self._delivery_service_legs: dict[DroneId, DeliveryServiceLeg] = {}
        self._fallback_leg: dict[DroneId, FallbackLeg] = {}
        # 记录当前订单对应的 dispatch 承诺，供 delivered 后执行层继续转移。
        self._dispatch_commit: dict[DroneId, DispatchCommit] = {}
        # Mode B 在 station 补能后必须继续返仓；key 为无人机，value 为目标 depot_id。
        self._mode_b_pending_depot_return: dict[DroneId, NodeId] = {}
        self._reservations: dict[DroneId, ReservationState] = {}
        self._reservation_count: dict[NodeId, int] = {}
        self._rendezvous_wait_started_at: dict[DroneId, float] = {}
        # 车载充换电完成时刻；到点后 charging_on_truck -> riding_with_truck。
        self._truck_charge_until: dict[DroneId, float] = {}
        self._decision_queue: list[DecisionTrigger] = []
        self._next_decision_id = 0
        self._last_exposed_decision_id: int | None = None
        # IDLE WAIT 的显式唤醒截止时刻。riding_with_truck WAIT 仍完全由卡车到站事件恢复。
        self._active_wait_until: dict[DroneId, float] = {}

        # 只跟踪 truck_execution_route 中的 mode A 背景订单完成情况。
        self._background_mode_a_order_count = 0
        self._background_mode_a_pending: set[str] = set()
        self._background_mode_a_completed: set[str] = set()
        self._background_mode_a_completion_time_sum = 0.0
        self._truck_background_order_completion_events: list[dict[str, Any]] = []
        self._last_reward_breakdown: dict[str, float] = {}
        # 每个无人机自上次决策以来累积的归因成本（方案一强化版）。
        # key = drone_id，value = 该无人机应承担的负奖励总量（已含符号，即负值）。
        self._agent_cost_accum: dict[DroneId, float] = {}
        self._current_coarse_plan: CoarsePlanView | None = None
        self._truck_replan_pending = False
        self._truck_replan_pending_reasons: set[str] = set()
        self._planner_replan_events: list[dict[str, Any]] = []
        self._mode_c_revalidation_events: list[dict[str, Any]] = []
        self._mode_c_post_delivery_selection_events: list[dict[str, Any]] = []
        self._active_launch_stations: set[str] = set()
        self._fallback_event_times: list[float] = []
        self._hard_failure_event_times: list[float] = []
        self._completed_backbone_nodes_since_plan: set[str] = set()
        self._episode_delivery_count = 0
        self._episode_fallback_count = 0
        self._episode_hard_failure_count = 0
        self._episode_reservation_timeout_count = 0
        self._episode_hard_overdue_count = 0
        self._episode_wait_action_count = 0
        self._episode_dispatch_mode_b_count = 0
        self._episode_dispatch_mode_c_count = 0
        self._episode_dispatch_decision_count = 0
        self._episode_dispatch_decision_with_legal_mode_c_count = 0
        self._episode_feasible_mode_c_recover_node_count_total = 0
        self._episode_mode_c_candidate_order_filter_counts: dict[str, int] = {}
        self._episode_mode_c_candidate_node_filter_counts: dict[str, int] = {}
        self._episode_mode_c_success_count = 0
        self._episode_mode_c_post_delivery_revalidation_fail_count = 0
        self._episode_mode_c_post_delivery_revalidation_fail_reasons = {
            key: 0 for key in _MODE_C_REVALIDATION_REASON_KEYS
        }
        self._episode_mode_c_selected_node_expired_count = 0
        self._episode_mode_c_selected_filter_margin_sum = 0.0
        self._episode_mode_c_selected_execution_slack_sum = 0.0
        self._episode_mode_c_selected_reservation_count_sum = 0.0
        self._episode_mode_c_selected_truck_eta_remaining_sum = 0.0
        self._episode_mode_c_selected_planned_truck_eta_sum = 0.0
        self._episode_mode_c_selected_planned_uav_eta_sum = 0.0
        self._episode_mode_c_selected_planned_slack_sum = 0.0
        self._episode_mode_c_timeout_from_state = {
            "delivered": 0,
            "return_to_rendezvous": 0,
            "waiting_for_truck": 0,
        }
        self._episode_mode_c_fallback_from_state = {
            "delivered": 0,
            "return_to_rendezvous": 0,
            "waiting_for_truck": 0,
        }
        self._episode_fallback_cause_counts = {
            key: 0 for key in _FALLBACK_CAUSE_KEYS
        }
        self._episode_ppo_attributed_fallback_count = 0
        self._episode_system_attributed_fallback_count = 0
        self._episode_reservation_release_cause_counts: dict[str, int] = {}
        self._episode_wait_time_sec = 0.0
        self._episode_idle_time_sec = 0.0
        self._episode_queue_time_sec = 0.0
        self._episode_fallback_time_sec = 0.0
        self._episode_ppo_attributed_fallback_time_sec = 0.0
        self._episode_system_attributed_fallback_time_sec = 0.0
        self._episode_overdue_time_sec = 0.0
        self._episode_hard_overdue_time_sec = 0.0
        self._episode_hard_overdue_order_ids: set[str] = set()
        self._episode_lateness_discount_total = 0.0
        self._episode_reservation_timeout_cost_sec = 0.0
        self._episode_uav_energy_reward_penalty = 0.0
        self._episode_uav_energy_ratio_sum = 0.0
        self._episode_uav_energy_penalty_events = 0
        self._runtime_uav_completed_distance_m = 0.0
        self._runtime_uav_completed_energy_j = 0.0
        self._planner_bridge = (
            planner_bridge
            if planner_bridge is not None
            else (
                PlannerBridge(
                    future_backbone_provider=self._future_backbone_visits,
                    config_path=self._config_path,
                    heavy_payload_capacity=self._heavy_payload_capacity,
                    truck_speed_provider=lambda: self._require_truck().speed,
                    truck_travel_time_provider=self._estimate_truck_road_travel_time,
                )
                if enable_phase6
                else None
            )
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_order_source(self, order_source: OrderSourceConfig) -> None:
        """更新下一次 reset() 使用的订单源配置。"""

        self._order_source = order_source
        self._current_episode_order_source_seed = int(order_source.seed)

    def reset(self) -> EnvStepResult:
        """重建一个全新的 episode，并返回首个可见状态。

        当前实现会：
          1. 重建实体与订单管理器；
          2. 读取 Phase 4 产物并按需追加 poisson 巡站循环；
          3. 初始化训练侧状态机；
          4. 注入 t=0 的订单并生成初始决策点。
        """
        episode_seed = int(self._order_source.seed)
        if self._order_source.mode == OrderSourceMode.POISSON:
            episode_seed += int(self._reset_count)
            random.seed(episode_seed)
        else:
            random.seed(episode_seed)
        self._current_episode_order_source_seed = episode_seed

        self._entity_manager = EntityManager()
        self._entity_manager.load_from_config({"entities": self._scene_ctx.entities_raw})
        self._order_manager = OrderManager()
        configure_order_manager_for_source(
            self._order_manager,
            self._scene_ctx,
            self._order_source,
        )

        self._truck_id, self._truck = _require_singleton(self._entity_manager.trucks, "truck")
        self._depot_id, self._depot = _require_singleton(self._entity_manager.depots, "depot")

        self._t_now = 0.0
        self._planned_route_stop_i = 1
        self._planned_route_segments.clear()
        self._flight_legs.clear()
        self._delivery_service_legs.clear()
        self._fallback_leg.clear()
        self._dispatch_commit.clear()
        self._mode_b_pending_depot_return.clear()
        self._reservations.clear()
        self._reservation_count.clear()
        self._rendezvous_wait_started_at.clear()
        self._truck_charge_until.clear()
        self._decision_queue.clear()
        self._next_decision_id = 0
        self._last_exposed_decision_id = None
        self._active_wait_until.clear()
        self._active_wait_resume.clear()
        self._background_mode_a_order_count = 0
        self._background_mode_a_pending.clear()
        self._background_mode_a_completed.clear()
        self._background_mode_a_completion_time_sum = 0.0
        self._truck_background_order_completion_events.clear()
        self._last_reward_breakdown = {}
        self._agent_cost_accum.clear()
        self._current_coarse_plan = None
        self._truck_replan_pending = False
        self._truck_replan_pending_reasons.clear()
        self._planner_replan_events.clear()
        self._mode_c_revalidation_events.clear()
        self._mode_c_post_delivery_selection_events.clear()
        self._active_launch_stations.clear()
        self._fallback_event_times.clear()
        self._hard_failure_event_times.clear()
        self._completed_backbone_nodes_since_plan.clear()
        self._truck_route_version = 1
        self._drone_path_version.clear()
        self._truck_road_route_geometry_cache.clear()
        self._active_truck_route_context_cache = None
        self._episode_delivery_count = 0
        self._episode_fallback_count = 0
        self._episode_hard_failure_count = 0
        self._episode_reservation_timeout_count = 0
        self._episode_hard_overdue_count = 0
        self._episode_wait_action_count = 0
        self._episode_dispatch_mode_b_count = 0
        self._episode_dispatch_mode_c_count = 0
        self._episode_dispatch_decision_count = 0
        self._episode_dispatch_decision_with_legal_mode_c_count = 0
        self._episode_feasible_mode_c_recover_node_count_total = 0
        self._episode_mode_c_candidate_order_filter_counts = {}
        self._episode_mode_c_candidate_node_filter_counts = {}
        self._episode_mode_c_success_count = 0
        self._episode_mode_c_post_delivery_revalidation_fail_count = 0
        self._episode_mode_c_post_delivery_revalidation_fail_reasons = {
            key: 0 for key in _MODE_C_REVALIDATION_REASON_KEYS
        }
        self._episode_mode_c_selected_node_expired_count = 0
        self._episode_mode_c_selected_filter_margin_sum = 0.0
        self._episode_mode_c_selected_execution_slack_sum = 0.0
        self._episode_mode_c_selected_reservation_count_sum = 0.0
        self._episode_mode_c_selected_truck_eta_remaining_sum = 0.0
        self._episode_mode_c_selected_planned_truck_eta_sum = 0.0
        self._episode_mode_c_selected_planned_uav_eta_sum = 0.0
        self._episode_mode_c_selected_planned_slack_sum = 0.0
        self._episode_mode_c_timeout_from_state = {
            "delivered": 0,
            "return_to_rendezvous": 0,
            "waiting_for_truck": 0,
        }
        self._episode_mode_c_fallback_from_state = {
            "delivered": 0,
            "return_to_rendezvous": 0,
            "waiting_for_truck": 0,
        }
        self._episode_fallback_cause_counts = {
            key: 0 for key in _FALLBACK_CAUSE_KEYS
        }
        self._episode_ppo_attributed_fallback_count = 0
        self._episode_system_attributed_fallback_count = 0
        self._episode_reservation_release_cause_counts = {}
        self._episode_wait_time_sec = 0.0
        self._episode_idle_time_sec = 0.0
        self._episode_queue_time_sec = 0.0
        self._episode_fallback_time_sec = 0.0
        self._episode_ppo_attributed_fallback_time_sec = 0.0
        self._episode_system_attributed_fallback_time_sec = 0.0
        self._episode_overdue_time_sec = 0.0
        self._episode_hard_overdue_time_sec = 0.0
        self._episode_hard_overdue_order_ids = set()
        self._episode_lateness_discount_total = 0.0
        self._episode_reservation_timeout_cost_sec = 0.0
        self._episode_uav_energy_reward_penalty = 0.0
        self._episode_uav_energy_ratio_sum = 0.0
        self._episode_uav_energy_penalty_events = 0
        self._runtime_uav_completed_distance_m = 0.0
        self._runtime_uav_completed_energy_j = 0.0

        artifacts = self._load_phase4_artifacts()
        self._planned_route_stops = list(artifacts["planned_stops"])
        self._planned_route_segments = list(artifacts["planned_segments"])
        self._full_backbone_cache = list(artifacts["backbone_cache"])
        self._background_mode_a_pending = set(self._static_truck_only_orders())
        self._background_mode_a_order_count = len(self._background_mode_a_pending)

        if self._order_source.mode == OrderSourceMode.POISSON:
            # 仅 poisson 训练模式追加巡站循环，benchmark/hybrid 保持 Phase 4 原路线。
            self._append_patrol_loop_if_needed()
            self._allow_empty_backbone_route = False
        else:
            self._allow_empty_backbone_route = True
        if self._planner_bridge is not None:
            self._planner_bridge.reset_episode(
                allow_empty_backbone_route=self._allow_empty_backbone_route
            )

        self._bind_truck_route()
        self._initialize_drone_states()

        # 允许 poisson 在 t=0 生成首单，使 reset 后的初始决策上下文可见新单。
        self._order_manager.tick(0.0, self._entity_manager)
        self._enqueue_initial_idle_decisions()
        if not self._decision_queue:
            self._advance_until_decision_or_done()

        self._reset_count += 1
        return self._build_step_result(reward=0.0, info={"event": "reset"})

    def step(self, action: EnvAction) -> EnvStepResult:
        """消费当前决策点的一个动作，并推进到下一个决策点或 episode 结束。

        奖励归因（方案一强化版）：
          - 先取走当前决策无人机自“上一次它自己决策”以来累计的 carry-in 奖励；
          - 再结算本次动作窗口内该无人机新产生的归因成本；
          - 最终 PPO reward = carry-in + 本次动作窗口新增奖励；
          - 其他无人机在同一时间窗口内产生的成本留在各自的累积器中，
            等到它们自己的决策点时再被取走。
        """
        applied = self._apply_decision_core(action)
        deciding_drone_id = applied.drone_id
        reward_breakdown: dict[str, float] = {}
        info: dict[str, Any] = dict(applied.info)

        if applied.auto_advance_kind == "idle_wait":
            if applied.wait_until is None:
                raise RuntimeError("idle WAIT 缺少 wait_until")
            self._advance_to_event(applied.wait_until)
            _merge_reward_breakdown(reward_breakdown, self._last_reward_breakdown)
            if not self.is_done():
                self._resume_capped_wait_if_needed(deciding_drone_id)
            if not self._decision_queue and not self.is_done():
                self._advance_until_decision_or_done()
                _merge_reward_breakdown(reward_breakdown, self._last_reward_breakdown)
        elif applied.auto_advance_kind == "until_decision":
            if not self._decision_queue:
                self._advance_until_decision_or_done()
                _merge_reward_breakdown(reward_breakdown, self._last_reward_breakdown)
        elif applied.auto_advance_kind == "none":
            pass
        else:
            raise RuntimeError(f"未知自动推进类型: {applied.auto_advance_kind}")

        # 本次 transition 的 reward 由两部分组成：
        # 1. carry-in：该无人机在别人动作期间已累计、延迟到本次决策点取走的成本；
        # 2. post-action：该无人机在本次动作窗口内新产生的归因成本。
        post_action_reward = self._agent_cost_accum.pop(deciding_drone_id, 0.0)
        reward = applied.carried_reward + post_action_reward
        reward_breakdown["attributed_carry_in"] = applied.carried_reward
        reward_breakdown["attributed_post_action"] = post_action_reward
        reward_breakdown["attributed_total"] = reward
        if "wait_opportunity_penalty" in info:
            reward_breakdown["wait_opportunity"] = float(
                info["wait_opportunity_penalty"]
            )

        self._last_reward_breakdown = reward_breakdown

        return self._build_step_result(reward=reward, info=info)

    def peek_next_event_time(self) -> float:
        """查看下一个内部事件时刻；若无事件则返回 `math.inf`。

        该接口只暴露现有事件队列真值，不生成或重排任何事件。
        """
        if self.is_done():
            return math.inf
        return float(self._next_event_time())

    def peek_next_decision_time(self) -> float:
        """查看下一次可能暴露决策的时刻。

        当前环境的决策只会由事件推进产生；因此无待处理决策时返回下一事件时刻。
        真实推进仍由 `advance_to_time()` 负责，并会在决策队列出现时停止。
        """
        if self.current_decision_context is not None:
            return float(self._t_now)
        if self.is_done():
            return math.inf
        return float(self._next_event_time())

    def advance_to_time(
        self,
        t_target: float,
        *,
        stop_on_training_done: bool = True,
    ) -> EnvStepResult:
        """按现有环境真值推进到指定仿真时刻或决策边界。

        与训练 `step()` 的区别是：本接口不消费动作，也不会为了离线 transition
        自动滚到下一个决策点；一旦决策队列非空即停止。

        `stop_on_training_done=False` 仅供在线播放器使用，用于在订单池清空后继续
        推进车辆返仓和无人机补能收尾；默认值保持训练/验证原语义。
        """
        target = float(t_target)
        if target < self._t_now - _TIME_EPS:
            raise ValueError(f"不能回退时间: t_target={target}, t_now={self._t_now}")
        if (stop_on_training_done and self.is_done()) or self._decision_queue:
            return self._build_step_result(
                reward=0.0,
                info={"event": "advance_to_time", "target_time": target},
            )

        if stop_on_training_done and self._enforces_upper_horizon():
            target = min(target, float(self._cfg.upper_horizon_sec))
        reward_breakdown: dict[str, float] = {}
        while (
            (not stop_on_training_done or not self.is_done())
            and not self._decision_queue
            and self._t_now < target - _TIME_EPS
        ):
            next_time = self._next_event_time()
            if math.isinf(next_time):
                next_time = target
            else:
                next_time = min(float(next_time), target)
            if next_time <= self._t_now + _TIME_EPS:
                next_time = min(target, self._t_now + 1.0)
            self._advance_to_event(
                next_time,
                clamp_to_horizon=stop_on_training_done,
            )
            _merge_reward_breakdown(reward_breakdown, self._last_reward_breakdown)

        self._last_reward_breakdown = reward_breakdown
        return self._build_step_result(
            reward=0.0,
            info={"event": "advance_to_time", "target_time": target},
        )

    def apply_decision(self, action: EnvAction) -> EnvStepResult:
        """在当前决策点提交动作，但不隐式推进到下一决策点。

        该接口和 `step()` 共用同一个动作提交核心；在线模式只能通过
        `advance_to_time()` 继续推进仿真时间。
        """
        applied = self._apply_decision_core(action)
        info = dict(applied.info)
        info["event"] = "apply_decision"
        return self._build_step_result(reward=applied.carried_reward, info=info)

    def apply_decision_batch(
        self,
        actions_by_drone: Mapping[str, EnvAction],
    ) -> EnvStepResult:
        """提交当前同刻 decision batch，并在 batch 全部提交后统一推进。

        该入口用于 teacher rollout / 同刻匹配执行。batch 内所有动作先基于同一份
        decision snapshot 做全量校验，提交阶段不允许任何单个动作推进时间。
        """
        if self.is_done():
            raise RuntimeError("episode 已结束，不能继续提交 batch 决策")

        decision_batch = self.peek_current_decision_batch()
        if not decision_batch:
            raise RuntimeError("当前没有可执行的 decision batch")

        batch_drone_ids = [str(ctx.deciding_drone_id) for ctx in decision_batch]
        expected_drone_ids = set(batch_drone_ids)
        actual_drone_ids = {str(drone_id) for drone_id in actions_by_drone}
        if actual_drone_ids != expected_drone_ids:
            raise ValueError(
                "actions_by_drone 与当前 decision batch 不一致: "
                f"expected={sorted(expected_drone_ids)}, actual={sorted(actual_drone_ids)}"
            )

        selected_orders: list[str] = []
        normalized_actions: dict[str, EnvAction] = {}
        for ctx in decision_batch:
            drone_id = str(ctx.deciding_drone_id)
            action = actions_by_drone[drone_id]
            if not self._is_action_allowed(action, ctx.action_lookup):
                raise ValueError(f"非法 batch action: drone_id={drone_id}, action={action}")
            normalized_actions[drone_id] = action
            if isinstance(action, DispatchAction):
                selected_orders.append(str(action.order_id))

        if len(selected_orders) != len(set(selected_orders)):
            raise ValueError("batch 内同一个订单被多个 UAV 选择")

        t0 = float(self._t_now)
        applied_items: list[AppliedDecision] = []
        for ctx in decision_batch:
            if abs(float(self._t_now) - t0) > _TIME_EPS:
                raise RuntimeError("batch 内单个动作提交时发生了时间推进")
            if not self._decision_queue:
                raise RuntimeError("decision_queue 在 batch 提交期间被提前清空")
            queue_head = self._decision_queue[0]
            if int(queue_head.decision_id) != int(ctx.decision_id):
                raise RuntimeError(
                    "decision_queue 顺序与预校验 batch 不一致: "
                    f"expected={ctx.decision_id}, actual={queue_head.decision_id}"
                )
            applied = self._apply_decision_core(
                normalized_actions[str(ctx.deciding_drone_id)],
                prevalidated_decision=ctx,
            )
            applied_items.append(applied)
            if abs(float(self._t_now) - t0) > _TIME_EPS:
                raise RuntimeError("batch 内单个动作提交后发生了时间推进")

        advance_reward_breakdown: dict[str, float] = {}
        if not self._decision_queue and not self.is_done():
            advance_reward_breakdown = (
                self._advance_until_decision_or_done_collect_reward_breakdown()
            )

        per_drone_rewards: dict[str, float] = {}
        per_drone_reward_breakdown: dict[str, dict[str, float]] = {}
        total_reward = 0.0
        total_carry_in = 0.0
        total_post_action = 0.0
        total_wait_opportunity = 0.0
        for applied in applied_items:
            drone_id = str(applied.drone_id)
            post_action_reward = self._agent_cost_accum.pop(drone_id, 0.0)
            reward = float(applied.carried_reward) + float(post_action_reward)
            per_drone_rewards[drone_id] = reward
            per_drone_reward_breakdown[drone_id] = {
                "attributed_carry_in": float(applied.carried_reward),
                "attributed_post_action": float(post_action_reward),
                "attributed_total": reward,
            }
            if "wait_opportunity_penalty" in applied.info:
                wait_opportunity = float(applied.info["wait_opportunity_penalty"])
                per_drone_reward_breakdown[drone_id]["wait_opportunity"] = wait_opportunity
                total_wait_opportunity += wait_opportunity
            total_reward += reward
            total_carry_in += float(applied.carried_reward)
            total_post_action += float(post_action_reward)

        reward_breakdown = dict(advance_reward_breakdown)
        reward_breakdown["attributed_carry_in"] = total_carry_in
        reward_breakdown["attributed_post_action"] = total_post_action
        reward_breakdown["attributed_total"] = total_reward
        if total_wait_opportunity:
            reward_breakdown["wait_opportunity"] = total_wait_opportunity
        self._last_reward_breakdown = reward_breakdown

        return self._build_step_result(
            reward=total_reward,
            info={
                "event": "apply_decision_batch",
                "batch_size": len(applied_items),
                "applied": [dict(item.info) for item in applied_items],
                "per_drone_rewards": per_drone_rewards,
                "per_drone_reward_breakdown": per_drone_reward_breakdown,
            },
        )

    def build_runtime_snapshot(self) -> dict[str, Any]:
        """构建与当前 `t_now` 对齐的运行时可视化快照。"""
        return self.build_visualization_snapshot()

    def is_done(self) -> bool:
        """检查 episode 是否满足当前实现中的任一终止条件。"""
        if (
            self._enforces_upper_horizon()
            and self._t_now >= self._cfg.upper_horizon_sec - _TIME_EPS
        ):
            return True
        if (
            self._order_manager is not None
            and not self._order_manager.pending_orders
            and not self._order_manager.assigned_orders
            and not self._background_mode_a_pending
            and not self._has_future_order_arrivals()
            and not self._has_active_mode_c_recovery_obligation()
        ):
            return True
        if self._drone_state and all(
            state == TrainingDroneState.AIRBORNE_ENERGY_FAILURE
            for state in self._drone_state.values()
        ):
            return True
        return False

    def _enforces_upper_horizon(self) -> bool:
        """benchmark 回放用于清空固定订单流，不使用 t_now 上限终止。"""
        return self._order_source.mode != OrderSourceMode.BENCHMARK

    def current_episode_order_source_seed(self) -> int:
        """返回当前 episode 实际使用的订单源随机种子。"""
        return int(self._current_episode_order_source_seed)

    def consume_terminal_agent_costs(self) -> dict[str, float]:
        """在 episode 结束后取走各无人机尚未结算的尾部归因成本。"""
        if not self.is_done():
            raise RuntimeError("episode 尚未结束，不能消费 terminal agent costs")
        pending = {
            str(drone_id): float(reward)
            for drone_id, reward in self._agent_cost_accum.items()
        }
        self._agent_cost_accum.clear()
        return pending

    @property
    def current_decision_context(self) -> DecisionContext | None:
        """返回队首决策点对应的上下文；若当前无需决策则返回 `None`。"""
        if not self._decision_queue or self.is_done():
            return None
        return self._build_decision_context(self._decision_queue[0])

    def peek_current_decision_batch(self) -> tuple[DecisionContext, ...]:
        """返回当前队首同刻决策 batch 的上下文快照。

        batch 边界由 `DecisionTrigger.t_enqueued` 定义，只取队首开始的同刻连续
        前缀；构造出的所有 `DecisionContext` 共享同一份 runtime/coarse snapshot。
        """
        if not self._decision_queue or self.is_done():
            return ()

        batch_t = float(self._decision_queue[0].t_enqueued)
        batch_triggers: list[DecisionTrigger] = []
        for trigger in self._decision_queue:
            if abs(float(trigger.t_enqueued) - batch_t) > _TIME_EPS:
                break
            batch_triggers.append(trigger)

        runtime_state = self.build_runtime_state_view()
        coarse_plan = self._refresh_coarse_plan_if_needed(runtime_state)
        planner_snapshot = self._build_decision_planner_snapshot(runtime_state)
        execution_snapshot = self._build_decision_execution_snapshot(runtime_state)
        return tuple(
            self._build_decision_context_from_snapshot(
                trigger=trigger,
                runtime_state=runtime_state,
                coarse_plan=coarse_plan,
                planner_snapshot=planner_snapshot,
                execution_snapshot=execution_snapshot,
            )
            for trigger in batch_triggers
        )

    def build_candidate_output(
        self,
        decision_context: DecisionContext,
        *,
        last_seen_plan_version: int,
    ) -> CandidateOutput:
        """基于一次既有决策快照重建外部可消费的 `CandidateOutput`。

        该入口显式保持：
          - `CandidateOutput` 仍由 env 外部调用方持有；
          - 重建所依据的状态快照来自 `decision_context`，而不是 env 当前真值；
          - 与兼容字段 `DecisionContext.action_lookup` 的可执行动作集合保持一致。
        """
        candidate_out = self._candidate_builder.build_from_decision_context(
            decision_context,
            last_seen_plan_version=last_seen_plan_version,
        )
        rebuilt_action_lookup = candidate_out.resolved_action_lookup.as_action_lookup()
        if rebuilt_action_lookup != decision_context.action_lookup:
            raise RuntimeError(
                "CandidateOutput 与 DecisionContext.action_lookup 不一致；"
                "请检查 decision_context 是否由同一 env / candidate_builder 生成"
            )
        return candidate_out

    # ---------------------------------------------------------------------
    # Core builders
    # ---------------------------------------------------------------------

    def build_runtime_state_view(self) -> RuntimeStateView:
        """基于当前内部真值构造一次 runtime_state 快照。"""
        entity_mgr = self._require_entity_manager()
        order_mgr = self._require_order_manager()
        truck = self._require_truck()

        drone_states = {
            drone_id: DroneStateView(
                drone_id=drone_id,
                training_state=self._drone_state[drone_id].value,
                current_loc=_clone_position(drone.current_loc),
                battery_current=self._effective_battery_current(drone_id, self._t_now),
                battery_max=float(drone.battery_max),
                battery_ratio=(
                    self._effective_battery_current(drone_id, self._t_now)
                    / max(_TIME_EPS, float(drone.battery_max))
                ),
                carrying_order_id=drone.carrying_order_id,
                home_type=drone.home_type.value,
                cruise_speed=float(drone.cruise_speed),
                payload_capacity=float(drone.payload_capacity),
                empty_weight=float(drone.empty_weight),
                k1=float(drone.k1),
                k2=float(drone.k2),
                reservation=(
                    ReservationStateView(
                        recover_node=reservation.recover_node,
                        issued_at=float(reservation.issued_at),
                    )
                    if (reservation := self._reservations.get(drone_id)) is not None
                    else None
                ),
            )
            for drone_id, drone in entity_mgr.drones.items()
        }

        node_states: dict[str, NodeStateView] = {}
        for node_id, station in entity_mgr.stations.items():
            node_states[node_id] = NodeStateView(
                node_id=node_id,
                node_type="station",
                position=_clone_position(station.location),
                parking_slots=int(station.parking_slots),
                swap_time=float(station.swap_time),
                queue_length=int(station.queue_length),
                available_slots=int(station.available_slots),
            )
        for node_id, depot in entity_mgr.depots.items():
            node_states[node_id] = NodeStateView(
                node_id=node_id,
                node_type="depot",
                position=_clone_position(depot.location),
                parking_slots=int(depot.parking_slots),
                swap_time=float(depot.swap_time),
                queue_length=int(depot.queue_length),
                available_slots=int(depot.available_slots),
            )

        return RuntimeStateView(
            t_now=float(self._t_now),
            truck_current_loc=_clone_position(truck.current_loc),
            drone_states=drone_states,
            pending_orders=copy.copy(order_mgr.pending_orders),
            assigned_orders=copy.copy(order_mgr.assigned_orders),
            node_states=node_states,
            reservation_count=copy.copy(self._reservation_count),
        )

    def build_system_context_stats(self) -> dict[str, Any]:
        """返回不进入 PPO 主奖励的系统上下文统计。"""
        order_mgr = self._require_order_manager()
        truck_only_events = [
            event
            for event in self._truck_background_order_completion_events
            if event.get("truck_only_dynamic")
        ]
        return {
            "mode_a_background_order_count": int(self._background_mode_a_order_count),
            "mode_a_background_completed_count": len(self._background_mode_a_completed),
            "mode_a_background_pending_count": len(self._background_mode_a_pending),
            "mode_a_background_completion_time_sum": float(
                self._background_mode_a_completion_time_sum
            ),
            "truck_only_dynamic_pending_count": sum(
                1
                for order in order_mgr.pending_orders.values()
                if self._is_truck_only_order(order)
            ),
            "truck_only_dynamic_completed_count": len(truck_only_events),
            "truck_background_order_completion_events": tuple(
                dict(event) for event in self._truck_background_order_completion_events
            ),
        }

    def build_runtime_energy_metrics(self) -> dict[str, float]:
        """按当前真实执行进度返回运行时总能耗统计。"""
        truck_distance_m = self._truck_traveled_distance_m(self._t_now)
        truck_energy_wh = (
            truck_distance_m * self._scene_solver_params().truck_energy_wh_per_meter
        )

        active_uav_distance_m = 0.0
        active_uav_energy_j = 0.0
        for leg in self._flight_legs.values():
            ratio = self._flight_leg_progress_ratio(leg=leg, t_now=self._t_now)
            active_uav_distance_m += max(0.0, float(leg.distance_m)) * ratio
            active_uav_energy_j += max(0.0, float(leg.energy_cost_j)) * ratio

        uav_distance_m = max(
            0.0,
            float(self._runtime_uav_completed_distance_m) + active_uav_distance_m,
        )
        uav_energy_wh = max(
            0.0,
            (float(self._runtime_uav_completed_energy_j) + active_uav_energy_j) / 3600.0,
        )
        total_energy_wh = max(0.0, truck_energy_wh + uav_energy_wh)
        return {
            "truck_distance_m": float(max(0.0, truck_distance_m)),
            "truck_energy_wh": float(max(0.0, truck_energy_wh)),
            "uav_distance_m": float(uav_distance_m),
            "uav_energy_wh": float(uav_energy_wh),
            "total_energy_cost_wh": float(total_energy_wh),
        }

    def build_episode_metrics_snapshot(self) -> dict[str, Any]:
        """返回当前 episode 的聚合指标快照。"""
        order_mgr = self._require_order_manager()
        completed_primary_orders = [
            order
            for order in order_mgr.completed_orders
            if self._is_uav_primary_order(order)
        ]
        delivered_primary_orders = [
            order
            for order in completed_primary_orders
            if order.status == TaskStatus.COMPLETED
        ]
        timed_out_primary_orders = [
            order
            for order in completed_primary_orders
            if order.status == TaskStatus.TIMEOUT
        ]
        on_time_delivery_count = sum(
            1
            for order in delivered_primary_orders
            if order.actual_deliver_time is not None
            and float(order.actual_deliver_time) <= float(order.deadline) + _TIME_EPS
        )
        overdue_delivery_count = len(delivered_primary_orders) - on_time_delivery_count
        delivered_primary_orders_with_timing = [
            order
            for order in delivered_primary_orders
            if order.actual_deliver_time is not None
        ]
        tardiness_values_sec = [
            max(0.0, float(order.actual_deliver_time) - float(order.deadline))
            for order in delivered_primary_orders_with_timing
        ]
        total_tardiness_sec = sum(tardiness_values_sec)
        mean_tardiness_sec = (
            total_tardiness_sec / float(len(tardiness_values_sec))
            if tardiness_values_sec
            else 0.0
        )
        max_tardiness_sec = max(tardiness_values_sec) if tardiness_values_sec else 0.0
        order_delay_sum_min = sum(
            max(0.0, float(order.actual_deliver_time) - float(order.deadline)) / 60.0
            for order in delivered_primary_orders_with_timing
        )
        avg_order_delay_min = (
            order_delay_sum_min / float(len(delivered_primary_orders_with_timing))
            if delivered_primary_orders_with_timing
            else 0.0
        )
        pending_primary_order_count = sum(
            1
            for order in order_mgr.pending_orders.values()
            if self._is_uav_primary_order(order)
        )
        assigned_primary_order_count = sum(
            1
            for order in order_mgr.assigned_orders.values()
            if self._is_uav_primary_order(order)
        )
        required_primary_order_count = (
            len(delivered_primary_orders)
            + len(timed_out_primary_orders)
            + pending_primary_order_count
            + assigned_primary_order_count
        )
        completion_rate = (
            float(len(delivered_primary_orders)) / float(required_primary_order_count)
            if required_primary_order_count > 0
            else 0.0
        )
        required_on_time_rate = (
            float(on_time_delivery_count) / float(required_primary_order_count)
            if required_primary_order_count > 0
            else 0.0
        )
        on_time_rate = (
            float(on_time_delivery_count) / float(len(delivered_primary_orders))
            if delivered_primary_orders
            else 0.0
        )
        metrics = {
            "done_reason": self._episode_done_reason(),
            "episode_end_t_sec": float(self._t_now),
            "delivery_count": int(self._episode_delivery_count),
            "on_time_delivery_count": int(on_time_delivery_count),
            "overdue_delivery_count": int(overdue_delivery_count),
            "late_delivery_count": int(overdue_delivery_count),
            "on_time_rate": float(on_time_rate),
            "completed_with_timing_order_count": int(
                len(delivered_primary_orders_with_timing)
            ),
            "order_delay_sum_min": float(order_delay_sum_min),
            "avg_order_delay_min": float(avg_order_delay_min),
            "total_tardiness_sec": float(total_tardiness_sec),
            "mean_tardiness_sec": float(mean_tardiness_sec),
            "max_tardiness_sec": float(max_tardiness_sec),
            "required_primary_order_count": int(required_primary_order_count),
            "completion_rate": float(completion_rate),
            "required_on_time_rate": float(required_on_time_rate),
            "timeout_order_count": int(len(timed_out_primary_orders)),
            "unserved_primary_order_count": int(
                pending_primary_order_count + assigned_primary_order_count
            ),
            "fallback_count": int(self._episode_fallback_count),
            "hard_failure_count": int(self._episode_hard_failure_count),
            "reservation_timeout_count": int(self._episode_reservation_timeout_count),
            "hard_overdue_count": int(self._episode_hard_overdue_count),
            "severe_overdue_order_count": int(self._episode_hard_overdue_count),
            "wait_action_count": int(self._episode_wait_action_count),
            "dispatch_decision_count": int(self._episode_dispatch_decision_count),
            "dispatch_decision_with_legal_mode_c_count": int(
                self._episode_dispatch_decision_with_legal_mode_c_count
            ),
            "dispatch_mode_b_count": int(self._episode_dispatch_mode_b_count),
            "dispatch_mode_c_count": int(self._episode_dispatch_mode_c_count),
            "mode_c_selected_count": int(self._episode_dispatch_mode_c_count),
            "mode_c_success_count": int(self._episode_mode_c_success_count),
            "mode_c_post_delivery_revalidation_fail_count": int(
                self._episode_mode_c_post_delivery_revalidation_fail_count
            ),
            "mode_c_post_delivery_revalidation_fail_reasons": dict(
                self._episode_mode_c_post_delivery_revalidation_fail_reasons
            ),
            "mode_c_selected_node_expired_count": int(
                self._episode_mode_c_selected_node_expired_count
            ),
            "mode_c_revalidation_debug_last": (
                copy.deepcopy(self._mode_c_revalidation_events[-1])
                if self._mode_c_revalidation_events
                else None
            ),
            "mode_c_post_delivery_selection_debug_last": (
                copy.deepcopy(self._mode_c_post_delivery_selection_events[-1])
                if self._mode_c_post_delivery_selection_events
                else None
            ),
            "mode_c_selected_filter_margin_sum": float(
                self._episode_mode_c_selected_filter_margin_sum
            ),
            "mode_c_selected_execution_slack_sum": float(
                self._episode_mode_c_selected_execution_slack_sum
            ),
            "mode_c_selected_reservation_count_sum": float(
                self._episode_mode_c_selected_reservation_count_sum
            ),
            "mode_c_selected_truck_eta_remaining_sum": float(
                self._episode_mode_c_selected_truck_eta_remaining_sum
            ),
            "mode_c_selected_planned_truck_eta_sum": float(
                self._episode_mode_c_selected_planned_truck_eta_sum
            ),
            "mode_c_selected_planned_uav_eta_sum": float(
                self._episode_mode_c_selected_planned_uav_eta_sum
            ),
            "mode_c_selected_planned_slack_sum": float(
                self._episode_mode_c_selected_planned_slack_sum
            ),
            "mode_c_timeout_from_state": dict(self._episode_mode_c_timeout_from_state),
            "mode_c_fallback_from_state": dict(self._episode_mode_c_fallback_from_state),
            "fallback_cause_counts": dict(self._episode_fallback_cause_counts),
            "ppo_attributed_fallback_count": int(
                self._episode_ppo_attributed_fallback_count
            ),
            "system_attributed_fallback_count": int(
                self._episode_system_attributed_fallback_count
            ),
            "reservation_release_cause_counts": dict(
                self._episode_reservation_release_cause_counts
            ),
            "planner_replan_event_count": int(len(self._planner_replan_events)),
            "feasible_mode_c_recover_node_count_total": int(
                self._episode_feasible_mode_c_recover_node_count_total
            ),
            "mode_c_candidate_order_filter_counts": dict(
                self._episode_mode_c_candidate_order_filter_counts
            ),
            "mode_c_candidate_node_filter_counts": dict(
                self._episode_mode_c_candidate_node_filter_counts
            ),
            "avg_feasible_mode_c_nodes_per_dispatch_decision": (
                float(self._episode_feasible_mode_c_recover_node_count_total)
                / float(self._episode_dispatch_decision_count)
                if self._episode_dispatch_decision_count > 0
                else 0.0
            ),
            "t_wait_sec": float(self._episode_wait_time_sec),
            "t_idle_sec": float(self._episode_idle_time_sec),
            "t_queue_sec": float(self._episode_queue_time_sec),
            "t_fallback_sec": float(self._episode_fallback_time_sec),
            "t_ppo_attributed_fallback_sec": float(
                self._episode_ppo_attributed_fallback_time_sec
            ),
            "t_system_attributed_fallback_sec": float(
                self._episode_system_attributed_fallback_time_sec
            ),
            "t_overdue_sec": float(self._episode_overdue_time_sec),
            "t_hard_overdue_sec": float(self._episode_hard_overdue_time_sec),
            "lateness_discount_total": float(self._episode_lateness_discount_total),
            "t_reservation_timeout_cost_sec": float(
                self._episode_reservation_timeout_cost_sec
            ),
            "episode_uav_energy_reward_penalty": float(
                self._episode_uav_energy_reward_penalty
            ),
            "episode_uav_energy_ratio_sum": float(
                self._episode_uav_energy_ratio_sum
            ),
            "episode_uav_energy_penalty_events": int(
                self._episode_uav_energy_penalty_events
            ),
            "pending_primary_order_count": int(pending_primary_order_count),
            "assigned_primary_order_count": int(assigned_primary_order_count),
            "system_context_stats": self.build_system_context_stats(),
        }
        metrics.update(self.build_runtime_energy_metrics())
        return metrics

    def build_planner_replan_events_snapshot(self) -> tuple[dict[str, Any], ...]:
        """返回当前 episode 的 planner replan 结构化事件。"""
        return tuple(copy.deepcopy(event) for event in self._planner_replan_events)

    def build_visualization_snapshot(self) -> dict[str, Any]:
        """返回供实时可视化使用的轻量级状态快照。"""
        runtime_state = self.build_runtime_state_view()
        order_mgr = self._require_order_manager()
        truck = self._require_truck()
        decision = self.current_decision_context

        visual_order_by_id: dict[str, dict[str, Any]] = {}
        for order in self._scene_ctx.static_orders:
            visual_order_by_id[str(order.order_id)] = self._serialize_visual_order(order)

        active_orders = list(runtime_state.pending_orders.values()) + list(runtime_state.assigned_orders.values())
        for order in active_orders:
            visual_order_by_id[str(order.order_id)] = self._serialize_visual_order(order)
        for order in order_mgr.completed_orders:
            visual_order_by_id[str(order.order_id)] = self._serialize_visual_order(order)
        orders = sorted(
            visual_order_by_id.values(),
            key=lambda item: (float(item["create_time"]), item["order_id"]),
        )

        drones = [
            {
                "drone_id": drone_state.drone_id,
                "status": drone_state.training_state,
                "x": float(drone_state.current_loc.x),
                "y": float(drone_state.current_loc.y),
                "z": float(drone_state.current_loc.z),
                "battery_ratio": float(drone_state.battery_ratio),
                "battery_current": float(drone_state.battery_current),
                "battery_max": float(drone_state.battery_max),
                "carrying_order_id": drone_state.carrying_order_id,
                "reservation_node_id": (
                    drone_state.reservation.recover_node
                    if drone_state.reservation is not None
                    else None
                ),
            }
            for drone_state in sorted(
                runtime_state.drone_states.values(),
                key=lambda item: item.drone_id,
            )
        ]
        dispatch_chains = self._build_dispatch_chain_snapshot(
            runtime_state=runtime_state,
            t_now=float(runtime_state.t_now),
        )

        return {
            "t_now": float(runtime_state.t_now),
            "truck": {
                "truck_id": str(self._truck_id or ""),
                "x": float(truck.current_loc.x),
                "y": float(truck.current_loc.y),
                "z": float(truck.current_loc.z),
            },
            "drones": drones,
            "orders": orders,
            "dispatch_chains": dispatch_chains,
            "current_decision": (
                {
                    "drone_id": str(decision.deciding_drone_id),
                    "trigger_type": str(decision.trigger_type),
                    "trigger_station_id": decision.trigger_station_id,
                }
                if decision is not None
                else None
            ),
            "paths": self._build_runtime_path_snapshot(float(runtime_state.t_now)),
            "last_reward_breakdown": dict(self._last_reward_breakdown),
        }

    def _build_dispatch_chain_snapshot(
        self,
        *,
        runtime_state: RuntimeStateView,
        t_now: float,
    ) -> list[dict[str, Any]]:
        """暴露 PPO dispatch 意图到执行层承诺的运行时链路。"""
        chains: list[dict[str, Any]] = []
        for drone_id, commit in sorted(self._dispatch_commit.items()):
            drone_state = runtime_state.drone_states.get(drone_id)
            reservation = self._reservations.get(drone_id)
            active_leg = self._flight_legs.get(drone_id)
            mode = str(commit.mode.value)
            selected_node = commit.selected_recover_node
            if mode != PolicyMode.C.value:
                recovery_stage = "not_applicable"
            elif selected_node is None:
                recovery_stage = "pending_post_delivery_selection"
            elif reservation is not None:
                recovery_stage = "reservation_active"
            else:
                recovery_stage = "rendezvous_selected"

            chains.append(
                {
                    "drone_id": str(drone_id),
                    "order_id": str(commit.order_id),
                    "mode": mode,
                    "trigger_station_id": commit.trigger_station_id,
                    "selected_recover_node_id": selected_node,
                    "reservation_node_id": (
                        None if reservation is None else str(reservation.recover_node)
                    ),
                    "recovery_stage": recovery_stage,
                    "planned_truck_arrival_time": commit.planned_truck_arrival_time,
                    "planned_uav_arrival_time_lb": commit.planned_uav_arrival_time_lb,
                    "planned_execution_slack_sec": commit.planned_execution_slack_sec,
                    "training_state": (
                        None if drone_state is None else str(drone_state.training_state)
                    ),
                    "carrying_order_id": (
                        None if drone_state is None else drone_state.carrying_order_id
                    ),
                    "active_leg_kind": None if active_leg is None else str(active_leg.kind),
                    "active_leg_target_node_id": (
                        None if active_leg is None else active_leg.target_node_id
                    ),
                    "active_leg_target_node_type": (
                        None if active_leg is None else active_leg.target_node_type
                    ),
                    "snapshot_time": float(t_now),
                }
            )
        return chains

    def _build_runtime_path_snapshot(self, t_now: float) -> dict[str, list[dict[str, Any]]]:
        """返回当前时刻仍有效的 truck / drone 路径账本快照。"""
        truck_paths: list[dict[str, Any]] = []
        active_truck_path = self._build_active_truck_path_entry(t_now)
        if active_truck_path is not None:
            truck_paths.append(active_truck_path)

        drone_paths: list[dict[str, Any]] = []
        for drone_id, leg in sorted(self._flight_legs.items()):
            entry = self._build_active_drone_path_entry(drone_id=drone_id, leg=leg, t_now=t_now)
            if entry is not None:
                drone_paths.append(entry)
        for drone_id, depot_id in sorted(self._mode_b_pending_depot_return.items()):
            if drone_id in self._flight_legs:
                continue
            entry = self._build_pending_mode_b_depot_path_entry(
                drone_id=drone_id,
                depot_id=depot_id,
                t_now=t_now,
            )
            if entry is not None:
                drone_paths.append(entry)

        return {
            "trucks": truck_paths,
            "drones": drone_paths,
        }

    def _build_active_truck_path_entry(self, t_now: float) -> dict[str, Any] | None:
        if not self._planned_route_segments:
            return None

        first_stop = self._planned_route_stops[0]
        if t_now <= first_stop.departure_time + _TIME_EPS:
            return self._serialize_truck_segment_entry(
                segment=self._planned_route_segments[0],
                start_pos=_clone_position(first_stop.position),
                traveled_m=0.0,
            )

        for idx, segment in enumerate(self._planned_route_segments):
            if t_now < segment.end_time - _TIME_EPS:
                segment_dt = max(_TIME_EPS, segment.end_time - segment.start_time)
                traveled_m = float(segment.distance_m) * max(
                    0.0,
                    min(1.0, (t_now - segment.start_time) / segment_dt),
                )
                current_pos = _interpolate_position_on_geometry(
                    geometry=segment.geometry,
                    cumulative_distances_m=segment.cumulative_distances_m,
                    traveled_m=traveled_m,
                )
                return self._serialize_truck_segment_entry(
                    segment=segment,
                    start_pos=current_pos,
                    traveled_m=traveled_m,
                )

            next_stop = self._planned_route_stops[idx + 1]
            if t_now <= next_stop.departure_time + _TIME_EPS:
                if idx + 1 >= len(self._planned_route_segments):
                    return None
                next_segment = self._planned_route_segments[idx + 1]
                return self._serialize_truck_segment_entry(
                    segment=next_segment,
                    start_pos=_clone_position(next_stop.position),
                    traveled_m=0.0,
                )

        return None

    def _serialize_truck_segment_entry(
        self,
        *,
        segment: PlannedTruckSegment,
        start_pos: Position3D,
        traveled_m: float,
    ) -> dict[str, Any]:
        remaining_points = _remaining_geometry_from_distance(
            geometry=segment.geometry,
            cumulative_distances_m=segment.cumulative_distances_m,
            traveled_m=traveled_m,
        )
        if not remaining_points:
            remaining_points = (_clone_position(start_pos), _clone_position(segment.geometry[-1]))
        return {
            "entity_id": str(self._truck_id or ""),
            "route_version": int(self._truck_route_version),
            "segment_id": int(segment.segment_id),
            "status": "active",
            "motion_mode": "road_network",
            "start_time": float(segment.start_time),
            "end_time": float(segment.end_time),
            "from_node_id": segment.from_node_id,
            "to_node_id": segment.to_node_id,
            "distance_m": float(max(0.0, segment.distance_m - traveled_m)),
            "osm_node_path": list(segment.osm_node_path),
            "path_utm": [_position_to_payload(pos) for pos in remaining_points],
        }

    def _build_active_drone_path_entry(
        self,
        *,
        drone_id: str,
        leg: FlightLeg,
        t_now: float,
    ) -> dict[str, Any] | None:
        if t_now >= leg.arrival_time - _TIME_EPS:
            return None

        remaining_points = self._flight_leg_remaining_points(leg=leg, t_now=t_now)
        if not remaining_points:
            remaining_points = (_clone_position(leg.target_pos),)
        return {
            "entity_id": drone_id,
            "route_version": int(leg.route_version),
            "segment_id": str(leg.kind),
            "status": "active",
            "motion_mode": str(leg.motion_mode),
            "start_time": float(leg.start_time),
            "end_time": float(leg.arrival_time),
            "target_node_id": leg.target_node_id,
            "target_node_type": leg.target_node_type,
            "order_id": leg.order_id,
            "distance_m": float(_polyline_distance_2d(remaining_points)),
            "path_utm": [_position_to_payload(pos) for pos in remaining_points],
        }

    def _flight_leg_geometry(self, leg: FlightLeg) -> tuple[Position3D, ...]:
        geometry = tuple(_clone_position(pos) for pos in leg.path_points)
        if len(geometry) >= 2:
            return geometry
        return (_clone_position(leg.start_pos), _clone_position(leg.target_pos))

    def _flight_leg_cumulative_distances(self, leg: FlightLeg) -> tuple[float, ...]:
        geometry = self._flight_leg_geometry(leg)
        if len(leg.cumulative_distances_m) == len(geometry):
            return tuple(float(item) for item in leg.cumulative_distances_m)
        return _build_cumulative_distances(geometry)

    def _flight_leg_traveled_distance(self, *, leg: FlightLeg, t_now: float) -> float:
        cumulative = self._flight_leg_cumulative_distances(leg)
        if not cumulative:
            return 0.0
        if t_now <= leg.start_time + _TIME_EPS:
            return 0.0
        if t_now >= leg.arrival_time - _TIME_EPS:
            return float(cumulative[-1])
        ratio = max(
            0.0,
            min(
                1.0,
                (float(t_now) - float(leg.start_time))
                / max(_TIME_EPS, float(leg.arrival_time) - float(leg.start_time)),
            ),
        )
        return float(cumulative[-1]) * ratio

    def _flight_leg_progress_ratio(self, *, leg: FlightLeg, t_now: float) -> float:
        if t_now <= leg.start_time + _TIME_EPS:
            return 0.0
        if t_now >= leg.arrival_time - _TIME_EPS:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                (float(t_now) - float(leg.start_time))
                / max(_TIME_EPS, float(leg.arrival_time) - float(leg.start_time)),
            ),
        )

    def _record_uav_leg_energy(self, *, leg: FlightLeg, progress_ratio: float = 1.0) -> None:
        ratio = max(0.0, min(1.0, float(progress_ratio)))
        self._runtime_uav_completed_distance_m += max(0.0, float(leg.distance_m)) * ratio
        self._runtime_uav_completed_energy_j += max(0.0, float(leg.energy_cost_j)) * ratio

    def _settle_uav_leg_energy_penalty(
        self,
        *,
        drone_id: str,
        leg: FlightLeg,
        progress_ratio: float = 1.0,
    ) -> dict[str, float]:
        ratio = max(0.0, min(1.0, float(progress_ratio)))
        leg_energy_j = max(0.0, float(leg.energy_cost_j)) * ratio
        drone = self._require_entity_manager().drones[drone_id]
        battery_max = float(getattr(drone, "battery_max", 0.0))
        if battery_max <= _TIME_EPS or not math.isfinite(battery_max):
            return {}
        energy_ratio = leg_energy_j / battery_max
        if not math.isfinite(energy_ratio) or energy_ratio <= _TIME_EPS:
            return {}

        cap_ratio = max(0.0, float(self._cfg.uav_energy_penalty_cap_ratio))
        coef = max(0.0, float(self._cfg.uav_energy_penalty_coef))
        clamped_ratio = min(max(0.0, energy_ratio), cap_ratio)
        penalty = -coef * clamped_ratio
        if not math.isfinite(penalty):
            return {}

        self._agent_cost_accum[drone_id] = (
            self._agent_cost_accum.get(drone_id, 0.0) + penalty
        )
        self._episode_uav_energy_reward_penalty += penalty
        self._episode_uav_energy_ratio_sum += energy_ratio
        self._episode_uav_energy_penalty_events += 1
        return {
            "uav_energy_penalty": float(penalty),
            "uav_energy_ratio_sum": float(energy_ratio),
            "uav_energy_penalty_events": 1.0,
        }

    def _flight_leg_position_at(self, *, leg: FlightLeg, t_now: float) -> Position3D:
        geometry = self._flight_leg_geometry(leg)
        cumulative = self._flight_leg_cumulative_distances(leg)
        traveled_m = self._flight_leg_traveled_distance(leg=leg, t_now=t_now)
        return _interpolate_position_on_geometry(
            geometry=geometry,
            cumulative_distances_m=cumulative,
            traveled_m=traveled_m,
        )

    def _flight_leg_remaining_points(
        self,
        *,
        leg: FlightLeg,
        t_now: float,
    ) -> tuple[Position3D, ...]:
        geometry = self._flight_leg_geometry(leg)
        cumulative = self._flight_leg_cumulative_distances(leg)
        traveled_m = self._flight_leg_traveled_distance(leg=leg, t_now=t_now)
        return _remaining_geometry_from_distance(
            geometry=geometry,
            cumulative_distances_m=cumulative,
            traveled_m=traveled_m,
        )

    def _build_pending_mode_b_depot_path_entry(
        self,
        *,
        drone_id: str,
        depot_id: str,
        t_now: float,
    ) -> dict[str, Any] | None:
        """Mode B station 补能等待期间，继续暴露后续返仓折线。"""
        state = self._drone_state.get(drone_id)
        if state not in {
            TrainingDroneState.QUEUEING_AT_HOST,
            TrainingDroneState.CHARGING_OR_SWAP,
        }:
            return None
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones.get(drone_id)
        depot = entity_mgr.depots.get(depot_id)
        if drone is None or depot is None:
            return None
        depot_pos = _clone_position(depot.get_location(t_now))
        path_points = self._uav_path_service.plan_path(
            from_pos=_clone_position(drone.current_loc),
            to_pos=depot_pos,
        )
        return {
            "entity_id": drone_id,
            "route_version": int(self._drone_path_version.get(drone_id, 0)),
            "segment_id": "mode_b_pending_return_to_depot",
            "status": "active",
            "motion_mode": (
                "uav_avoidance_path"
                if self._uav_path_service.has_obstacle_planner
                else "straight_line"
            ),
            "start_time": float(t_now),
            "end_time": None,
            "target_node_id": str(depot_id),
            "target_node_type": "depot",
            "order_id": None,
            "distance_m": float(_polyline_distance_2d(path_points)),
            "path_utm": [_position_to_payload(pos) for pos in path_points],
        }

    def _build_coarse_plan_view(self, t_now: float) -> CoarsePlanView:
        """在给定时刻构造 Phase 5 内联 coarse plan。

        当前版本不做重规划，`plan_version` 固定为 0。
        """
        order_mgr = self._require_order_manager()

        deduped = list(self._dedup_backbone_visits(self._future_backbone_visits(t_now)))

        truck_backbone_route = tuple(visit.node_id for visit in deduped)
        if not truck_backbone_route and not self._allow_empty_backbone_route:
            raise RuntimeError("poisson 模式下不允许空骨架，请检查 patrol loop 生成逻辑")

        truck_eta_map = {visit.node_id: float(visit.arrival_time) for visit in deduped}
        route_drift_ref = {
            visit.node_id: RouteDriftRef(
                eta_ref=float(visit.arrival_time),
                route_index_ref=idx,
            )
            for idx, visit in enumerate(deduped)
        }
        launch_candidate_stations = tuple(
            node_id
            for node_id in truck_backbone_route
            if node_id in self._require_entity_manager().stations
        )
        node_states = self.build_runtime_state_view().node_states

        authorized_orders: list[str] = []
        order_priority_band: dict[str, int] = {}
        order_pre_score: dict[str, float] = {}
        planner_mode_cap: dict[str, frozenset[PlannerMode]] = {}
        policy_mode_mask: dict[str, frozenset[PolicyMode]] = {}
        recovery_pool: dict[str, tuple[str, ...]] = {}

        for order_id, order in sorted(
            order_mgr.pending_orders.items(),
            key=lambda item: (item[1].deadline, item[0]),
        ):
            # 超过 heavy payload 上限的订单只保留 planner 语义边界，不进入 UAV 授权集合。
            if float(order.payload_weight) > self._heavy_payload_capacity:
                planner_mode_cap[order_id] = frozenset({PlannerMode.A})
                continue

            remaining = max(0.0, float(order.deadline) - t_now)
            time_window = max(_TIME_EPS, float(order.time_window_seconds))
            ratio = remaining / time_window
            if ratio <= 1.0 / 3.0:
                band = 0
            elif ratio <= 2.0 / 3.0:
                band = 1
            else:
                band = 2

            authorized_orders.append(order_id)
            order_priority_band[order_id] = band
            order_pre_score[order_id] = remaining
            planner_mode_cap[order_id] = frozenset({PlannerMode.B, PlannerMode.C})

            if not truck_backbone_route and self._allow_empty_backbone_route:
                policy_mode_mask[order_id] = frozenset({PolicyMode.B})
                recovery_pool[order_id] = ()
            else:
                policy_mode_mask[order_id] = frozenset({PolicyMode.B, PolicyMode.C})
                # coarse plan 只暴露 recovery pool；最终动作合法性由 5b 运行时过滤负责。
                recovery_pool[order_id] = select_recovery_pool_for_order(
                    order=order,
                    truck_backbone_route=truck_backbone_route,
                    truck_eta_map=truck_eta_map,
                    node_states=node_states,
                    max_candidates=self._cfg.max_candidate_recovery_per_order,
                    future_scan_limit=self._cfg.recovery_pool_future_scan_limit,
                    drone_cruise_speed=self._recovery_pool_drone_cruise_speed,
                    upper_horizon_sec=self._cfg.upper_horizon_sec,
                )

        node_charge_load_budget = {
            **{node_id: 0 for node_id in self._require_entity_manager().stations},
            **{node_id: 0 for node_id in self._require_entity_manager().depots},
        }

        return CoarsePlanView(
            plan_version=0,
            issued_at=float(t_now),
            valid_until=max(float(t_now), float(self._cfg.upper_horizon_sec)),
            truck_backbone_route=truck_backbone_route,
            truck_eta_map=truck_eta_map,
            authorized_orders=tuple(authorized_orders),
            order_priority_band=order_priority_band,
            order_pre_score=order_pre_score,
            planner_mode_cap=planner_mode_cap,
            policy_mode_mask=policy_mode_mask,
            recovery_pool=recovery_pool,
            node_charge_load_budget=node_charge_load_budget,
            route_drift_ref=route_drift_ref,
            launch_candidate_stations=launch_candidate_stations,
            allow_empty_backbone_route=self._allow_empty_backbone_route,
        )

    def _backbone_visit_arrival_upper_bound(self, node_id: str | None) -> float:
        """返回某个 fixed node 仍可作为 future backbone 的最晚到达时间。"""
        upper_bound = float(self._cfg.upper_horizon_sec)
        if node_id is None:
            return upper_bound

        reservation_deadlines: list[float] = []
        for reservation_drone_id, reservation in self._reservations.items():
            if str(reservation.recover_node) != str(node_id):
                continue
            commit = self._dispatch_commit.get(reservation_drone_id)
            if commit is None or commit.planned_truck_arrival_time is None:
                continue
            reservation_deadlines.append(
                float(commit.planned_truck_arrival_time)
                + float(self._cfg.rendezvous_max_wait_sec)
            )
        if reservation_deadlines:
            upper_bound = min(upper_bound, min(reservation_deadlines))
        return upper_bound

    def _is_future_backbone_visit_available(
        self,
        visit: BackboneVisit,
        t_now: float,
        *,
        include_current: bool,
    ) -> bool:
        """按时间窗和 active reservation 承诺判断 backbone visit 是否仍可用。"""
        threshold = float(t_now) - (_TIME_EPS if include_current else -_TIME_EPS)
        arrival_time = float(visit.arrival_time)
        if arrival_time < threshold:
            return False
        upper_bound = self._backbone_visit_arrival_upper_bound(str(visit.node_id))
        return arrival_time <= upper_bound + _TIME_EPS

    def _future_backbone_visits(self, t_now: float) -> tuple[BackboneVisit, ...]:
        """返回当前时刻之后仍在未来骨架中的固定节点访问记录。"""
        return tuple(
            visit
            for visit in self._full_backbone_cache
            if self._is_future_backbone_visit_available(
                visit,
                t_now,
                include_current=False,
            )
        )

    def _future_station_backbone_visits(self, t_now: float) -> tuple[BackboneVisit, ...]:
        """返回当前时刻之后仍在未来骨架中的 station 访问记录；不包含 depot。"""
        station_ids = self._require_entity_manager().stations
        return tuple(
            visit
            for visit in self._future_backbone_visits(t_now)
            if str(visit.node_id) in station_ids
        )

    def _dedup_backbone_visits(
        self,
        visits: tuple[BackboneVisit, ...],
    ) -> tuple[BackboneVisit, ...]:
        deduped: list[BackboneVisit] = []
        seen_nodes: set[str] = set()
        for visit in visits:
            if visit.node_id in seen_nodes:
                continue
            seen_nodes.add(visit.node_id)
            deduped.append(visit)
        return tuple(deduped)

    def _refresh_coarse_plan_if_needed(
        self,
        runtime_state: RuntimeStateView | None = None,
        *,
        allow_truck_route_replan: bool = False,
        force_replan: bool = False,
    ) -> CoarsePlanView:
        """统一 coarse plan 刷新入口。"""
        if runtime_state is None:
            runtime_state = self.build_runtime_state_view()

        require_station_backbone = self._ensure_future_backbone_capacity(runtime_state)
        if require_station_backbone and self._planner_bridge is None:
            runtime_state = self.build_runtime_state_view()

        if self._planner_bridge is None:
            return self._build_coarse_plan_view(self._t_now)

        truck_route_ready_at = self._truck_route_ready_time_for_replan(
            float(runtime_state.t_now)
        )
        reservation_constraints = self._build_truck_reservation_constraints(
            runtime_state
        )
        truck_mandatory_orders = self._build_truck_mandatory_order_input(runtime_state)
        allow_tail_empty_backbone = self._should_allow_tail_empty_backbone(
            runtime_state=runtime_state,
            reservation_constraints=reservation_constraints,
            truck_mandatory_orders=truck_mandatory_orders,
        )
        truck_replan_pressure = self._has_truck_replan_pressure(
            runtime_state=runtime_state,
            reservation_constraints=reservation_constraints,
            truck_mandatory_orders=truck_mandatory_orders,
        )
        allow_deferred_empty_backbone = False
        if truck_replan_pressure and not allow_truck_route_replan:
            self._mark_truck_replan_pending("truck_replan_pressure")
            allow_deferred_empty_backbone = not self._future_backbone_visits(
                float(runtime_state.t_now)
            )
            planner_runtime_state = self._build_planner_runtime_state(
                runtime_state=self._build_coarse_only_runtime_state(runtime_state),
                truck_mandatory_orders={},
                require_station_backbone=False,
                truck_route_ready_at=truck_route_ready_at,
            )
            planner_reservation_constraints: tuple[TruckReservationConstraint, ...] = ()
        else:
            if not truck_replan_pressure and not require_station_backbone:
                self._clear_truck_replan_pending()
            planning_orders = (
                truck_mandatory_orders
                if (truck_replan_pressure or self._truck_replan_pending)
                else {}
            )
            planner_runtime_state = self._build_planner_runtime_state(
                runtime_state=runtime_state,
                truck_mandatory_orders=planning_orders,
                require_station_backbone=require_station_backbone,
                truck_route_ready_at=truck_route_ready_at,
            )
            planner_reservation_constraints = reservation_constraints

        trigger_ctx = self._build_planner_trigger_context(planner_runtime_state)
        if (
            require_station_backbone
            or force_replan
            or (
                allow_truck_route_replan
                and (self._truck_replan_pending or truck_replan_pressure)
            )
        ):
            trigger_ctx = PlannerTriggerContext(
                t_now=trigger_ctx.t_now,
                backlog_new_orders=max(
                    trigger_ctx.backlog_new_orders,
                    self._cfg.coarse_new_order_trigger,
                ),
                fallback_count_in_window=trigger_ctx.fallback_count_in_window,
                hard_failure_count_in_window=trigger_ctx.hard_failure_count_in_window,
                route_drift_ratio=trigger_ctx.route_drift_ratio,
            )
        coarse_plan = self._planner_bridge.maybe_replan(
            planner_runtime_state,
            trigger_ctx,
            reservation_constraints=planner_reservation_constraints,
            allow_empty_backbone_route=(
                self._allow_empty_backbone_route
                or allow_tail_empty_backbone
                or allow_deferred_empty_backbone
            ),
        )
        if (
            self._current_coarse_plan is None
            or coarse_plan.plan_version > self._current_coarse_plan.plan_version
        ):
            self._current_coarse_plan = coarse_plan
            if coarse_plan.truck_plan_stops:
                if allow_truck_route_replan:
                    self._apply_dynamic_truck_plan(
                        coarse_plan.truck_plan_stops,
                        route_start_time=truck_route_ready_at,
                    )
                    applied_to_truck_route = True
                    self._clear_truck_replan_pending()
                else:
                    applied_to_truck_route = False
                    self._mark_truck_replan_pending("truck_plan_deferred")
            else:
                applied_to_truck_route = False
            reservation_env_decisions = self._apply_planner_reservation_outcomes(
                coarse_plan=coarse_plan,
                reservation_constraints=planner_reservation_constraints,
                t_now=float(runtime_state.t_now),
            )
            self._record_planner_replan_event(
                coarse_plan=coarse_plan,
                trigger_ctx=trigger_ctx,
                reservation_constraints=planner_reservation_constraints,
                reservation_env_decisions=reservation_env_decisions,
                applied_to_truck_route=applied_to_truck_route,
                allow_truck_route_replan=allow_truck_route_replan,
                truck_route_ready_at=truck_route_ready_at,
            )
            self._active_launch_stations = set(coarse_plan.launch_candidate_stations)
            self._completed_backbone_nodes_since_plan.clear()

        self._prune_active_launch_stations(runtime_state)
        return self._current_coarse_plan

    def _ensure_future_backbone_capacity(
        self,
        runtime_state: RuntimeStateView,
    ) -> bool:
        """订单流未结束时保持卡车未来 station backbone 供给。"""
        if not self._has_unfinished_order_flow(runtime_state):
            return False

        future_station_count = len(
            self._future_station_backbone_visits(float(runtime_state.t_now))
        )
        if future_station_count >= _MIN_FUTURE_STATION_BACKBONE_VISITS:
            return False

        if self._planner_bridge is not None:
            self._mark_truck_replan_pending("future_backbone_capacity")
            return True

        extended = self._append_patrol_loop_if_needed(force=True)
        if extended and len(self._planned_route_stops) >= 2:
            self._bind_truck_route()
        return extended

    def _has_unfinished_order_flow(self, runtime_state: RuntimeStateView) -> bool:
        if runtime_state.pending_orders or runtime_state.assigned_orders:
            return True
        if self._background_mode_a_pending:
            return True

        return self._has_future_order_arrivals()

    def _has_future_order_arrivals(self) -> bool:
        """是否仍存在尚未进入 pending 池的订单源事件。"""
        order_mgr = self._order_manager
        if order_mgr is None:
            return False

        next_poisson = float(getattr(order_mgr, "_next_order_time", math.inf))
        if next_poisson <= self._cfg.upper_horizon_sec + _TIME_EPS:
            return True

        scheduled_dynamic = list(getattr(order_mgr, "_scheduled_dynamic", []))
        scheduled_i = int(getattr(order_mgr, "_scheduled_dynamic_i", 0))
        return scheduled_i < len(scheduled_dynamic)

    def _has_active_mode_c_recovery_obligation(self) -> bool:
        """是否仍存在需要卡车兑现或执行层继续解析的 Mode C 回收义务。"""
        for drone_id, reservation in self._reservations.items():
            if not reservation.recover_node:
                continue
            state = self._drone_state.get(drone_id)
            if state in {
                TrainingDroneState.FLYING_TO_DELIVER,
                TrainingDroneState.DELIVERY_SERVICE,
                TrainingDroneState.DELIVERED,
                TrainingDroneState.RETURN_TO_RENDEZVOUS,
                TrainingDroneState.WAITING_FOR_TRUCK,
            }:
                return True

        for drone_id, service_leg in self._delivery_service_legs.items():
            if not service_leg.order_id:
                continue
            commit = self._dispatch_commit.get(drone_id)
            if commit is not None and commit.mode == PolicyMode.C:
                return True
        return False

    def _should_allow_tail_empty_backbone(
        self,
        *,
        runtime_state: RuntimeStateView,
        reservation_constraints: tuple[TruckReservationConstraint, ...],
        truck_mandatory_orders: Mapping[str, Order],
    ) -> bool:
        """episode 尾段无卡车强约束时，允许 coarse plan 退化为空骨架。"""
        has_unfinished_order_flow = self._has_unfinished_order_flow(runtime_state)
        if self._allow_empty_backbone_route and not has_unfinished_order_flow:
            return True
        if (
            reservation_constraints
            or truck_mandatory_orders
            or self._has_active_mode_c_recovery_obligation()
        ):
            return False
        if not has_unfinished_order_flow:
            return True
        if self._future_backbone_visits(float(runtime_state.t_now)):
            return False
        tail_remaining_sec = (
            float(self._cfg.upper_horizon_sec) - float(runtime_state.t_now)
        )
        return tail_remaining_sec <= float(self._cfg.max_wait_decision_gap_sec) + _TIME_EPS

    def _has_truck_replan_pressure(
        self,
        *,
        runtime_state: RuntimeStateView,
        reservation_constraints: tuple[TruckReservationConstraint, ...],
        truck_mandatory_orders: Mapping[str, Order],
    ) -> bool:
        if reservation_constraints:
            return True
        dynamic_truck_orders = set(truck_mandatory_orders) - set(
            self._background_mode_a_pending
        )
        if dynamic_truck_orders:
            return True
        return bool(self._background_mode_a_orders_missing_from_physical_route())

    def _background_mode_a_orders_missing_from_physical_route(self) -> set[str]:
        if not self._background_mode_a_pending:
            return set()
        future_customer_order_ids = {
            str(stop.order_id)
            for stop in self._planned_route_stops[
                max(0, int(self._planned_route_stop_i)) :
            ]
            if stop.node_type == "customer" and stop.order_id is not None
        }
        return set(self._background_mode_a_pending) - future_customer_order_ids

    def _mark_truck_replan_pending(self, reason: str) -> None:
        self._truck_replan_pending = True
        self._truck_replan_pending_reasons.add(str(reason))

    def _clear_truck_replan_pending(self) -> None:
        self._truck_replan_pending = False
        self._truck_replan_pending_reasons.clear()

    def _truck_route_ready_time_for_replan(self, t_now: float) -> float:
        """返回卡车完成当前 stop 服务、可驶向新计划首站的时刻。"""
        current_stop_idx = int(self._planned_route_stop_i) - 1
        if 0 <= current_stop_idx < len(self._planned_route_stops):
            stop = self._planned_route_stops[current_stop_idx]
            if (
                abs(float(stop.arrival_time) - float(t_now)) <= _TIME_EPS
                and float(stop.departure_time) > float(t_now) + _TIME_EPS
            ):
                return float(stop.departure_time)
        return float(t_now)

    def _build_coarse_only_runtime_state(
        self,
        runtime_state: RuntimeStateView,
    ) -> RuntimeStateView:
        filtered_pending = {
            order_id: order
            for order_id, order in runtime_state.pending_orders.items()
            if not self._is_truck_only_order(order)
        }
        if len(filtered_pending) == len(runtime_state.pending_orders):
            return runtime_state
        return replace(runtime_state, pending_orders=filtered_pending)

    def _build_planner_runtime_state(
        self,
        *,
        runtime_state: RuntimeStateView,
        truck_mandatory_orders: Mapping[str, Order],
        require_station_backbone: bool = False,
        truck_route_ready_at: float | None = None,
    ) -> PlannerRuntimeStateView:
        """构造 PlannerBridge 专用输入，显式区分 PPO pending 与卡车必经订单。"""
        return PlannerRuntimeStateView(
            t_now=float(runtime_state.t_now),
            truck_current_loc=_clone_position(runtime_state.truck_current_loc),
            drone_states=runtime_state.drone_states,
            pending_orders=runtime_state.pending_orders,
            assigned_orders=runtime_state.assigned_orders,
            node_states=runtime_state.node_states,
            reservation_count=runtime_state.reservation_count,
            truck_mandatory_orders=dict(truck_mandatory_orders),
            require_station_backbone=bool(require_station_backbone),
            truck_route_ready_at=(
                None if truck_route_ready_at is None else float(truck_route_ready_at)
            ),
        )

    def _build_truck_mandatory_order_input(
        self,
        runtime_state: RuntimeStateView,
    ) -> dict[str, Order]:
        """统一收集所有卡车必须服务订单：静态背景重订单 + 动态 truck-only 订单。"""
        mandatory_orders = dict(self._pending_background_mode_a_orders())
        for order_id, order in sorted(runtime_state.pending_orders.items()):
            if self._is_truck_only_order(order):
                mandatory_orders[str(order_id)] = order
        return mandatory_orders

    def _pending_background_mode_a_orders(self) -> dict[str, Order]:
        if not self._background_mode_a_pending:
            return {}
        static_orders = self._static_truck_only_orders()
        return {
            order_id: order
            for order_id in sorted(self._background_mode_a_pending)
            if (order := static_orders.get(order_id)) is not None
        }

    def _static_truck_only_orders(self) -> dict[str, Order]:
        return {
            str(order.order_id): order
            for order in self._scene_ctx.static_orders
            if self._is_truck_only_order(order)
        }

    def _build_truck_reservation_constraints(
        self,
        runtime_state: RuntimeStateView,
    ) -> tuple[TruckReservationConstraint, ...]:
        """把环境内的 post-delivery reservation 转成 PlannerBridge 约束。"""
        constraints: list[TruckReservationConstraint] = []
        hard_states = {
            TrainingDroneState.FLYING_TO_DELIVER,
            TrainingDroneState.DELIVERY_SERVICE,
            TrainingDroneState.RETURN_TO_RENDEZVOUS,
        }
        strong_hard_states = {
            TrainingDroneState.WAITING_FOR_TRUCK,
        }
        for drone_id, reservation in sorted(self._reservations.items()):
            drone_state = self._drone_state.get(drone_id)
            if drone_state in strong_hard_states:
                reservation_state = ReservationConstraintState.STRONG_HARD
            elif drone_state in hard_states:
                reservation_state = ReservationConstraintState.HARD
            else:
                continue

            commit = self._dispatch_commit.get(drone_id)
            eta_ref = (
                None
                if commit is None
                else commit.planned_truck_arrival_time
            )
            if eta_ref is None and self._current_coarse_plan is not None:
                eta_ref = self._current_coarse_plan.truck_eta_map.get(
                    reservation.recover_node
                )
            if eta_ref is None:
                eta_ref = self._next_backbone_arrival_time_for_node(
                    reservation.recover_node,
                    runtime_state.t_now,
                    include_current=True,
                )
            if eta_ref is None:
                continue
            window = self._build_reservation_constraint_window(
                drone_id=drone_id,
                node_id=reservation.recover_node,
                state=reservation_state,
                eta_ref=float(eta_ref),
                t_now=float(runtime_state.t_now),
            )

            constraints.append(
                TruckReservationConstraint(
                    reservation_id=(
                        f"{drone_id}:{reservation.recover_node}:"
                        f"{float(reservation.issued_at):.6f}"
                    ),
                    drone_id=drone_id,
                    node_id=reservation.recover_node,
                    state=reservation_state,
                    eta_ref=float(eta_ref),
                    max_eta_drift_sec=float(
                        self._cfg.reservation_drift_eta_abs_threshold_sec
                    ),
                    issued_at=float(reservation.issued_at),
                    related_order_id=(
                        None if commit is None else commit.order_id
                    ),
                    earliest_eta=window[0],
                    latest_eta=window[1],
                    preferred_eta=window[2],
                )
            )
        return tuple(constraints)

    def _build_reservation_constraint_window(
        self,
        *,
        drone_id: str,
        node_id: str,
        state: ReservationConstraintState,
        eta_ref: float,
        t_now: float,
    ) -> tuple[float, float, float]:
        """构造 planner 侧使用的动态 rendezvous 时间窗。"""

        max_wait = float(self._cfg.rendezvous_max_wait_sec)
        execution_margin = float(self._cfg.rendezvous_execution_margin_sec)
        if state == ReservationConstraintState.STRONG_HARD:
            wait_started_at = self._rendezvous_wait_started_at.get(drone_id)
            latest = (
                float(wait_started_at) + max_wait
                if wait_started_at is not None
                else float(eta_ref) + float(self._cfg.reservation_drift_eta_abs_threshold_sec)
            )
            earliest = min(max(float(t_now), 0.0), latest)
            preferred = earliest
            return (float(earliest), float(latest), float(preferred))

        uav_arrival = self._estimate_rendezvous_arrival_for_recheck(
            drone_id=drone_id,
            node_id=node_id,
            t_now=float(t_now),
        )
        if uav_arrival is None:
            earliest = float(eta_ref)
            latest = float(eta_ref) + float(self._cfg.reservation_drift_eta_abs_threshold_sec)
            preferred = earliest
            return (float(earliest), float(latest), float(preferred))

        earliest = float(uav_arrival) + execution_margin
        latest = float(uav_arrival) + max_wait
        preferred = earliest
        return (float(earliest), float(latest), float(preferred))

    def _apply_planner_reservation_outcomes(
        self,
        *,
        coarse_plan: CoarsePlanView,
        reservation_constraints: tuple[TruckReservationConstraint, ...],
        t_now: float,
    ) -> dict[str, str]:
        constraints_by_id = {
            constraint.reservation_id: constraint
            for constraint in reservation_constraints
        }
        env_decisions: dict[str, str] = {}
        for reservation_id, outcome in coarse_plan.reservation_outcomes.items():
            if outcome.status != ReservationPlanStatus.INVALIDATED:
                env_decisions[reservation_id] = self._sync_active_reservation_eta_from_outcome(
                    reservation_id=reservation_id,
                    outcome=outcome,
                    constraints_by_id=constraints_by_id,
                    t_now=float(t_now),
                )
                continue
            constraint = constraints_by_id.get(reservation_id)
            if constraint is None:
                env_decisions[reservation_id] = "ignored_missing_constraint"
                continue
            drone_id = constraint.drone_id
            if drone_id not in self._reservations:
                env_decisions[reservation_id] = "ignored_no_active_reservation"
                continue
            if self._can_keep_invalidated_reservation_after_execution_recheck(
                drone_id=drone_id,
                constraint=constraint,
                outcome=outcome,
                t_now=float(t_now),
            ):
                self._update_reservation_commit_for_new_truck_eta(
                    drone_id=drone_id,
                    node_id=constraint.node_id,
                    new_truck_eta=float(outcome.new_eta),
                    t_now=float(t_now),
                )
                env_decisions[reservation_id] = "kept_after_execution_recheck"
                continue
            self._release_reservation(
                drone_id,
                cause=FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
            )
            commit = self._dispatch_commit.get(drone_id)
            if commit is not None:
                self._dispatch_commit[drone_id] = replace(
                    commit,
                    selected_recover_node=None,
                )
            if self._drone_state.get(drone_id) in {
                TrainingDroneState.RETURN_TO_RENDEZVOUS,
                TrainingDroneState.WAITING_FOR_TRUCK,
            }:
                if not self._enter_fallback_recovery(
                    drone_id,
                    start_time=float(t_now),
                    cause=FALLBACK_CAUSE_PLANNER_INVALIDATED_FOR_TRUCK_ORDER,
                ):
                    self._mark_hard_failure(drone_id)
                    env_decisions[reservation_id] = "released_and_hard_failure"
                else:
                    env_decisions[reservation_id] = "released_and_fallback"
            else:
                env_decisions[reservation_id] = "released_only"
        return env_decisions

    def _sync_active_reservation_eta_from_outcome(
        self,
        *,
        reservation_id: str,
        outcome: ReservationPlanOutcome,
        constraints_by_id: Mapping[str, TruckReservationConstraint],
        t_now: float,
    ) -> str:
        """把 kept/drifted reservation 的 planner ETA 同步到 dispatch commit。"""
        if outcome.new_eta is None:
            return "planner_status_no_env_action"
        constraint = constraints_by_id.get(reservation_id)
        if constraint is None:
            return "ignored_missing_constraint"
        drone_id = constraint.drone_id
        reservation = self._reservations.get(drone_id)
        if reservation is None:
            return "ignored_no_active_reservation"
        if (
            reservation.recover_node != constraint.node_id
            or str(outcome.node_id) != str(constraint.node_id)
        ):
            return "ignored_reservation_node_mismatch"
        commit = self._dispatch_commit.get(drone_id)
        if commit is None:
            return "ignored_missing_commit"
        if (
            commit.selected_recover_node is not None
            and str(commit.selected_recover_node) != str(constraint.node_id)
        ):
            return "ignored_commit_node_mismatch"
        self._update_reservation_commit_for_new_truck_eta(
            drone_id=drone_id,
            node_id=constraint.node_id,
            new_truck_eta=float(outcome.new_eta),
            t_now=float(t_now),
        )
        return f"{outcome.status.value}_commit_eta_synced"

    def _record_planner_replan_event(
        self,
        *,
        coarse_plan: CoarsePlanView,
        trigger_ctx: PlannerTriggerContext,
        reservation_constraints: tuple[TruckReservationConstraint, ...],
        reservation_env_decisions: Mapping[str, str],
        applied_to_truck_route: bool,
        allow_truck_route_replan: bool,
        truck_route_ready_at: float,
    ) -> None:
        constraints_by_id = {
            constraint.reservation_id: constraint
            for constraint in reservation_constraints
        }
        reservation_outcomes: list[dict[str, Any]] = []
        for reservation_id, outcome in sorted(coarse_plan.reservation_outcomes.items()):
            constraint = constraints_by_id.get(reservation_id)
            commit = (
                None
                if constraint is None
                else self._dispatch_commit.get(constraint.drone_id)
            )
            reservation_outcomes.append(
                {
                    "reservation_id": str(reservation_id),
                    "drone_id": (
                        None if constraint is None else str(constraint.drone_id)
                    ),
                    "node_id": str(outcome.node_id),
                    "status": str(outcome.status.value),
                    "invalidate_cause": (
                        None
                        if outcome.invalidate_cause is None
                        else str(outcome.invalidate_cause)
                    ),
                    "old_eta": float(outcome.old_eta),
                    "new_eta": (
                        None if outcome.new_eta is None else float(outcome.new_eta)
                    ),
                    "eta_drift_sec": (
                        None
                        if outcome.eta_drift_sec is None
                        else float(outcome.eta_drift_sec)
                    ),
                    "env_decision": str(
                        reservation_env_decisions.get(
                            reservation_id,
                            "not_evaluated",
                        )
                    ),
                    "constraint_state": (
                        None if constraint is None else str(constraint.state.value)
                    ),
                    "related_order_id": (
                        None
                        if constraint is None or constraint.related_order_id is None
                        else str(constraint.related_order_id)
                    ),
                    "commit_planned_truck_arrival_time": (
                        None if commit is None else commit.planned_truck_arrival_time
                    ),
                }
            )

        self._planner_replan_events.append(
            {
                "t_now": float(trigger_ctx.t_now),
                "truck_route_ready_at": float(truck_route_ready_at),
                "plan_version": int(coarse_plan.plan_version),
                "applied_to_truck_route": bool(applied_to_truck_route),
                "allow_truck_route_replan": bool(allow_truck_route_replan),
                "truck_replan_pending": bool(self._truck_replan_pending),
                "truck_replan_pending_reasons": tuple(
                    sorted(self._truck_replan_pending_reasons)
                ),
                "trigger": {
                    "backlog_new_orders": int(trigger_ctx.backlog_new_orders),
                    "fallback_count_in_window": int(
                        trigger_ctx.fallback_count_in_window
                    ),
                    "hard_failure_count_in_window": int(
                        trigger_ctx.hard_failure_count_in_window
                    ),
                    "route_drift_ratio": float(trigger_ctx.route_drift_ratio),
                },
                "truck_backbone_route": tuple(
                    str(node_id) for node_id in coarse_plan.truck_backbone_route
                ),
                "truck_eta_map": {
                    str(node_id): float(eta)
                    for node_id, eta in sorted(coarse_plan.truck_eta_map.items())
                },
                "truck_plan_stops": tuple(
                    {
                        "seq": int(stop.seq),
                        "node_type": str(stop.node_type),
                        "node_id": str(stop.node_id),
                        "order_id": (
                            None if stop.order_id is None else str(stop.order_id)
                        ),
                        "arrival_time": float(stop.arrival_time),
                        "departure_time": float(stop.departure_time),
                    }
                    for stop in coarse_plan.truck_plan_stops
                ),
                "reservation_outcomes": tuple(reservation_outcomes),
            }
        )

    def _can_keep_invalidated_reservation_after_execution_recheck(
        self,
        *,
        drone_id: str,
        constraint: TruckReservationConstraint,
        outcome: ReservationPlanOutcome,
        t_now: float,
    ) -> bool:
        """把 planner 层 invalidated 复核为执行层是否仍可兑现。"""
        cause = str(outcome.invalidate_cause or "")
        if cause == "node_not_in_truck_plan":
            return False
        if cause not in {"arrived_before_eta_ref", "eta_late_exceeds_threshold"}:
            return False
        if outcome.new_eta is None:
            return False

        reservation = self._reservations.get(drone_id)
        if reservation is None or reservation.recover_node != constraint.node_id:
            return False

        new_eta = float(outcome.new_eta)
        state = self._drone_state.get(drone_id)
        if state == TrainingDroneState.WAITING_FOR_TRUCK:
            wait_started_at = self._rendezvous_wait_started_at.get(drone_id)
            if wait_started_at is None:
                return False
            if new_eta < float(t_now) - _TIME_EPS:
                return False
            total_wait = new_eta - float(wait_started_at)
            return total_wait <= float(self._cfg.rendezvous_max_wait_sec) + _TIME_EPS

        uav_arrival = self._estimate_rendezvous_arrival_for_recheck(
            drone_id=drone_id,
            node_id=constraint.node_id,
            t_now=float(t_now),
        )
        if uav_arrival is None:
            return False
        if (
            float(uav_arrival) + float(self._cfg.rendezvous_execution_margin_sec)
            > new_eta + _TIME_EPS
        ):
            return False
        wait_time = new_eta - float(uav_arrival)
        if cause == "eta_late_exceeds_threshold":
            return wait_time <= float(self._cfg.rendezvous_max_wait_sec) + _TIME_EPS
        return True

    def _estimate_rendezvous_arrival_for_recheck(
        self,
        *,
        drone_id: str,
        node_id: str,
        t_now: float,
    ) -> float | None:
        """按当前状态估计 UAV 到达 reservation 节点的时刻。"""
        state = self._drone_state.get(drone_id)
        if state == TrainingDroneState.FLYING_TO_DELIVER:
            leg = self._flight_legs.get(drone_id)
            commit = self._dispatch_commit.get(drone_id)
            if (
                leg is not None
                and leg.kind == "deliver"
                and commit is not None
                and leg.order_id == commit.order_id
            ):
                service_finish = (
                    float(leg.arrival_time)
                    + float(self._scene_solver_params().drone_service_time_order_s)
                )
                drone = self._require_entity_manager().drones[drone_id]
                host = self._resolve_fixed_node(node_id)
                return service_finish + self._estimate_flight_time(
                    drone=drone,
                    from_pos=leg.target_pos,
                    to_pos=host.get_location(service_finish),
                )
            if commit is not None and commit.planned_uav_arrival_time_lb is not None:
                return max(float(t_now), float(commit.planned_uav_arrival_time_lb))

        if state == TrainingDroneState.RETURN_TO_RENDEZVOUS:
            leg = self._flight_legs.get(drone_id)
            if (
                leg is not None
                and leg.kind == "return_to_rendezvous"
                and leg.target_node_id == node_id
            ):
                return max(float(t_now), float(leg.arrival_time))

        if state == TrainingDroneState.DELIVERY_SERVICE:
            service_leg = self._delivery_service_legs.get(drone_id)
            if service_leg is not None:
                drone = self._require_entity_manager().drones[drone_id]
                host = self._resolve_fixed_node(node_id)
                service_finish = float(service_leg.finish_time)
                return service_finish + self._estimate_flight_time(
                    drone=drone,
                    from_pos=service_leg.service_pos,
                    to_pos=host.get_location(service_finish),
                )

        if state == TrainingDroneState.DELIVERED:
            drone = self._require_entity_manager().drones[drone_id]
            host = self._resolve_fixed_node(node_id)
            return float(t_now) + self._estimate_flight_time(
                drone=drone,
                from_pos=drone.current_loc,
                to_pos=host.get_location(float(t_now)),
            )

        return None

    def _update_reservation_commit_for_new_truck_eta(
        self,
        *,
        drone_id: str,
        node_id: str,
        new_truck_eta: float,
        t_now: float,
    ) -> None:
        """保留 reservation 时，把 dispatch commit 同步到新的 truck ETA。"""
        commit = self._dispatch_commit.get(drone_id)
        if commit is None:
            return

        if self._drone_state.get(drone_id) == TrainingDroneState.WAITING_FOR_TRUCK:
            uav_arrival = self._rendezvous_wait_started_at.get(drone_id, float(t_now))
        else:
            uav_arrival = self._estimate_rendezvous_arrival_for_recheck(
                drone_id=drone_id,
                node_id=node_id,
                t_now=float(t_now),
            )
        planned_execution_slack_sec = (
            None if uav_arrival is None else float(new_truck_eta) - float(uav_arrival)
        )
        self._dispatch_commit[drone_id] = replace(
            commit,
            selected_recover_node=node_id,
            planned_truck_arrival_time=float(new_truck_eta),
            planned_uav_arrival_time_lb=(
                commit.planned_uav_arrival_time_lb
                if uav_arrival is None
                else float(uav_arrival)
            ),
            planned_execution_slack_sec=planned_execution_slack_sec,
        )

    def _apply_dynamic_truck_plan(
        self,
        truck_plan_stops: tuple[TruckPlanStopView, ...],
        *,
        route_start_time: float | None = None,
    ) -> None:
        if not truck_plan_stops:
            return

        effective_route_start_time = float(
            self._t_now if route_start_time is None else route_start_time
        )
        anchor = PlannedStop(
            seq=0,
            node_type="truck_current",
            node_id="truck_current",
            position=self._truck_position_at_time(self._t_now),
            order_id=None,
            arrival_time=float(self._t_now),
            departure_time=effective_route_start_time,
        )
        planned_stops = [anchor]
        planned_segments: list[PlannedTruckSegment] = []
        prev_stop = anchor
        for idx, stop_view in enumerate(truck_plan_stops, start=1):
            position = self._resolve_truck_plan_stop_position(stop_view)
            planned_stop = PlannedStop(
                seq=idx,
                node_type=stop_view.node_type,
                node_id=stop_view.node_id,
                position=position,
                order_id=stop_view.order_id,
                arrival_time=float(stop_view.arrival_time),
                departure_time=float(stop_view.departure_time),
            )
            route = self._build_truck_road_route(
                from_pos=prev_stop.position,
                to_pos=position,
            )
            geometry = route.geometry
            distance_m = _polyline_distance_2d(geometry)
            planned_segments.append(
                PlannedTruckSegment(
                    segment_id=len(planned_segments),
                    from_node_id=prev_stop.node_id,
                    to_node_id=planned_stop.node_id,
                    from_node_type=prev_stop.node_type,
                    to_node_type=planned_stop.node_type,
                    start_time=float(prev_stop.departure_time),
                    end_time=float(planned_stop.arrival_time),
                    distance_m=float(distance_m),
                    geometry=geometry,
                    cumulative_distances_m=_build_cumulative_distances(geometry),
                    osm_node_path=route.osm_node_path,
                )
            )
            planned_stops.append(planned_stop)
            prev_stop = planned_stop

        self._planned_route_stops = planned_stops
        self._planned_route_segments = planned_segments
        self._planned_route_stop_i = 1
        self._full_backbone_cache = [
            BackboneVisit(
                node_id=stop.node_id,
                arrival_time=float(stop.arrival_time),
                departure_time=float(stop.departure_time) + _BACKBONE_DEPARTURE_EPS,
            )
            for stop in planned_stops[1:]
            if stop.node_type in {"station", "depot"}
        ]
        self._truck_route_version += 1
        self._append_patrol_loop_if_needed()
        if len(self._planned_route_stops) >= 2:
            self._bind_truck_route()

    def _resolve_truck_plan_stop_position(
        self,
        stop: TruckPlanStopView,
    ) -> Position3D:
        if stop.node_type == "customer":
            order = self._require_order_manager().pending_orders.get(stop.order_id or "")
            if order is None:
                order = self._require_order_manager().assigned_orders.get(
                    stop.order_id or ""
                )
            if order is None and stop.order_id in self._background_mode_a_pending:
                order = self._pending_background_mode_a_orders().get(
                    str(stop.order_id)
                )
            if order is None:
                raise RuntimeError(f"truck plan 缺少 customer order: {stop.order_id}")
            return _clone_position(order.delivery_loc)

        fixed_node = self._resolve_fixed_node(stop.node_id)
        return _clone_position(fixed_node.get_location(float(stop.arrival_time)))

    def _estimate_truck_road_travel_time(
        self,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> float:
        """按 OSM 路网最短路径估算卡车行驶时间。"""
        geometry = self._build_truck_road_route_geometry(
            from_pos=from_pos,
            to_pos=to_pos,
        )
        distance_m = _polyline_distance_2d(geometry)
        return distance_m / max(_TIME_EPS, float(self._require_truck().speed))

    def _build_truck_road_route_geometry(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> tuple[Position3D, ...]:
        return self._build_truck_road_route(
            from_pos=from_pos,
            to_pos=to_pos,
        ).geometry

    def _build_truck_road_route(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> TruckRoadRoute:
        road_graph, road_nodes = self._require_road_graph()
        active_context = self._active_truck_route_context_for_position(from_pos)
        if active_context is not None:
            anchored_route = _build_route_from_active_truck_context(
                context=active_context,
                from_pos=from_pos,
                to_pos=to_pos,
                road_graph=road_graph,
                road_nodes=road_nodes,
            )
            if anchored_route is not None:
                return anchored_route

        key = (
            round(float(from_pos.x), 2),
            round(float(from_pos.y), 2),
            round(float(to_pos.x), 2),
            round(float(to_pos.y), 2),
        )
        cached = self._truck_road_route_geometry_cache.get(key)
        if cached is not None:
            return _clone_truck_road_route(cached)

        route = _build_route_between_positions(
            from_pos=from_pos,
            to_pos=to_pos,
            road_graph=road_graph,
            road_nodes=road_nodes,
            nearest_cache={},
        )
        self._truck_road_route_geometry_cache[key] = _clone_truck_road_route(route)
        return route

    def _build_planner_trigger_context(
        self,
        runtime_state: RuntimeStateView,
    ) -> PlannerTriggerContext:
        metrics = self._compute_planner_snapshot_metrics(runtime_state)
        self._prune_window_events(self._fallback_event_times, runtime_state.t_now)
        self._prune_window_events(self._hard_failure_event_times, runtime_state.t_now)
        return PlannerTriggerContext(
            t_now=float(runtime_state.t_now),
            backlog_new_orders=int(metrics["backlog_new_orders"]),
            fallback_count_in_window=len(self._fallback_event_times),
            hard_failure_count_in_window=len(self._hard_failure_event_times),
            route_drift_ratio=float(metrics["route_drift_ratio"]),
        )

    def _build_decision_planner_snapshot(
        self,
        runtime_state: RuntimeStateView,
    ) -> DecisionPlannerSnapshot:
        metrics = self._compute_planner_snapshot_metrics(runtime_state)
        self._prune_window_events(self._fallback_event_times, runtime_state.t_now)
        self._prune_window_events(self._hard_failure_event_times, runtime_state.t_now)
        return DecisionPlannerSnapshot(
            backlog_new_orders=int(metrics["backlog_new_orders"]),
            fallback_count_in_window=len(self._fallback_event_times),
            hard_failure_count_in_window=len(self._hard_failure_event_times),
            route_drift_ratio=float(metrics["route_drift_ratio"]),
            completed_backbone_count=int(metrics["completed_backbone_count"]),
            expected_backbone_count=int(metrics["expected_backbone_count"]),
            total_backbone_count=int(metrics["total_backbone_count"]),
            active_launch_stations=tuple(sorted(self._active_launch_stations)),
        )

    def _build_decision_execution_snapshot(
        self,
        runtime_state: RuntimeStateView,
    ) -> DecisionExecutionSnapshot:
        return DecisionExecutionSnapshot(
            uav_eta_to_available={
                drone_id: float(
                    self._estimate_eta_to_available_for_snapshot(
                        drone_id=drone_id,
                        t_now=float(runtime_state.t_now),
                    )
                )
                for drone_id in runtime_state.drone_states
            },
            uav_dispatch_mode={
                drone_id: (
                    self._dispatch_commit[drone_id].mode.value
                    if drone_id in self._dispatch_commit
                    else "NONE"
                )
                for drone_id in runtime_state.drone_states
            },
        )

    def _compute_planner_snapshot_metrics(
        self,
        runtime_state: RuntimeStateView,
    ) -> dict[str, float | int]:
        backlog_new_orders = 0
        route_drift_ratio = 0.0
        completed_count = len(self._completed_backbone_nodes_since_plan)
        expected_count = 0
        total_backbone_count = 0
        if self._current_coarse_plan is not None:
            planned_truck_only_orders = {
                str(stop.order_id)
                for stop in self._current_coarse_plan.truck_plan_stops
                if stop.node_type == "customer" and stop.order_id is not None
            }
            uav_backlog_count = sum(
                1
                for order_id, order in runtime_state.pending_orders.items()
                if self._is_uav_primary_order(order)
                and not self._current_coarse_plan.is_order_authorized(order_id)
            )
            truck_only_backlog_count = sum(
                1
                for order_id, order in runtime_state.pending_orders.items()
                if (
                    self._is_truck_only_order(order)
                    and order_id not in planned_truck_only_orders
                )
            )
            backlog_new_orders = uav_backlog_count
            if truck_only_backlog_count:
                backlog_new_orders = max(
                    backlog_new_orders,
                    self._cfg.coarse_new_order_trigger,
                )
            total_backbone_count = len(self._current_coarse_plan.route_drift_ref)
            if total_backbone_count > 0:
                expected_count = sum(
                    1
                    for item in self._current_coarse_plan.route_drift_ref.values()
                    if float(item.eta_ref) <= float(runtime_state.t_now) + _TIME_EPS
                )
                route_drift_ratio = abs(completed_count - expected_count) / total_backbone_count
        return {
            "backlog_new_orders": int(backlog_new_orders),
            "route_drift_ratio": float(route_drift_ratio),
            "completed_backbone_count": int(completed_count),
            "expected_backbone_count": int(expected_count),
            "total_backbone_count": int(total_backbone_count),
        }

    def _estimate_eta_to_available_for_snapshot(
        self,
        *,
        drone_id: str,
        t_now: float,
    ) -> float:
        state = self._drone_state.get(drone_id)
        if state is None:
            return 0.0
        if state in {
            TrainingDroneState.IDLE,
            TrainingDroneState.RIDING_WITH_TRUCK,
            TrainingDroneState.AIRBORNE_ENERGY_FAILURE,
        }:
            return 0.0
        if state == TrainingDroneState.DELIVERY_SERVICE:
            service_leg = self._delivery_service_legs.get(drone_id)
            if service_leg is not None:
                return max(0.0, float(service_leg.finish_time) - t_now)
        if drone_id in self._flight_legs:
            return max(0.0, float(self._flight_legs[drone_id].arrival_time) - t_now)
        if state == TrainingDroneState.WAITING_FOR_TRUCK:
            commit = self._dispatch_commit.get(drone_id)
            if commit is not None and commit.selected_recover_node is not None:
                arrival = self._future_arrival_time_for_node(commit.selected_recover_node, t_now)
                if arrival is not None:
                    return max(0.0, arrival - t_now)
        if state == TrainingDroneState.CHARGING_ON_TRUCK:
            eta = self._estimate_truck_charge_remaining_sec(drone_id, t_now)
            if eta is not None:
                return eta
        return 0.0

    def _future_arrival_time_for_node(
        self,
        node_id: str,
        t_now: float,
    ) -> float | None:
        return self._next_backbone_arrival_time_for_node(
            node_id,
            t_now,
            include_current=False,
        )

    def _next_backbone_arrival_time_for_node(
        self,
        node_id: str | None,
        t_now: float,
        *,
        include_current: bool,
    ) -> float | None:
        """返回某节点在给定时刻之后的下一次卡车到达时间。"""
        if node_id is None:
            return None
        for visit in self._full_backbone_cache:
            if visit.node_id != node_id:
                continue
            if self._is_future_backbone_visit_available(
                visit,
                t_now,
                include_current=include_current,
            ):
                return float(visit.arrival_time)
        return None

    def _prune_window_events(self, events: list[float], t_now: float) -> None:
        window_start = t_now - self._cfg.fallback_burst_window_sec
        while events and events[0] < window_start - _TIME_EPS:
            events.pop(0)

    def _prune_active_launch_stations(self, runtime_state: RuntimeStateView) -> None:
        # 新语义：只要 station 在卡车未来骨架中，到站时就必须暴露
        # riding-with-truck UAV 的放飞/等待决策，不再按附近 pending 订单裁剪。
        _ = runtime_state

    def _build_action_lookup(
        self,
        *,
        drone_id: str,
        coarse_plan: CoarsePlanView,
        runtime_state: RuntimeStateView | None = None,
        trigger_type: str | None = None,
        trigger_station_id: str | None = None,
    ) -> tuple[EnvAction, ...]:
        """为当前决策 UAV 构造可执行动作列表。"""
        if runtime_state is None:
            runtime_state = self.build_runtime_state_view()
        effective_trigger_type = (
            trigger_type
            if trigger_type is not None
            else (
                "truck_station_arrival"
                if self._drone_state.get(drone_id) == TrainingDroneState.RIDING_WITH_TRUCK
                else "inline_idle"
            )
        )
        candidate_out = self._candidate_builder.build(
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            deciding_drone_id=drone_id,
            trigger_type=effective_trigger_type,
            trigger_station_id=trigger_station_id,
            last_seen_plan_version=coarse_plan.plan_version,
        )
        return candidate_out.resolved_action_lookup.as_action_lookup()

    # ---------------------------------------------------------------------
    # Step helpers
    # ---------------------------------------------------------------------

    def _apply_wait_action(self, drone_id: str) -> None:
        """把当前 UAV 切入 `active_wait` 占位状态。"""
        current_state = self._drone_state[drone_id]
        if current_state == TrainingDroneState.IDLE:
            self._active_wait_resume[drone_id] = TrainingDroneState.IDLE
        elif current_state == TrainingDroneState.RIDING_WITH_TRUCK:
            self._active_wait_resume[drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
        else:
            raise RuntimeError(f"状态 {current_state.value} 不允许执行 WAIT")
        # idle 路径在 step(WAIT) 入口一次性结算；riding_with_truck 路径后续按真实 dt 累计。
        self._drone_state[drone_id] = TrainingDroneState.ACTIVE_WAIT

    def _apply_decision_core(
        self,
        action: EnvAction,
        *,
        prevalidated_decision: DecisionContext | None = None,
    ) -> AppliedDecision:
        """消费当前队首决策并提交动作，不负责自动推进到下一决策点。"""
        if self.is_done():
            raise RuntimeError("episode 已结束，不能继续提交决策")
        if not self._decision_queue:
            raise RuntimeError("当前没有可执行的 decision context")

        trigger = self._decision_queue[0]
        deciding_drone_id = trigger.drone_id
        if prevalidated_decision is None:
            decision = self._build_decision_context(trigger)
        else:
            decision = prevalidated_decision
            if int(decision.decision_id) != int(trigger.decision_id):
                raise RuntimeError(
                    "预校验 decision 与队首 trigger 不一致: "
                    f"decision_id={decision.decision_id}, trigger_id={trigger.decision_id}"
                )
            if str(decision.deciding_drone_id) != str(trigger.drone_id):
                raise RuntimeError(
                    "预校验 decision 与队首 trigger 的 UAV 不一致: "
                    f"decision_drone={decision.deciding_drone_id}, trigger_drone={trigger.drone_id}"
                )
        if not self._is_action_allowed(action, decision.action_lookup):
            raise ValueError(f"非法动作: {action}")
        trigger = self._decision_queue.pop(0)

        # 与原 step() 语义一致：当前 drone 在别人动作窗口内累计的自身成本，
        # 在它自己的下一次决策被取走。
        carried_reward = self._agent_cost_accum.pop(deciding_drone_id, 0.0)
        self._last_reward_breakdown = {}
        info: dict[str, Any] = {
            "drone_id": deciding_drone_id,
            "trigger_type": trigger.trigger_type,
            "trigger_station_id": trigger.trigger_station_id,
        }

        if isinstance(action, GlobalWaitAction):
            self._episode_wait_action_count += 1
            current_state = self._drone_state[deciding_drone_id]
            info["applied_action"] = "WAIT"
            if current_state == TrainingDroneState.IDLE:
                opportunity_penalty = self._compute_wait_opportunity_penalty(
                    decision=decision
                )
                if opportunity_penalty:
                    self._agent_cost_accum[deciding_drone_id] = (
                        self._agent_cost_accum.get(deciding_drone_id, 0.0)
                        + opportunity_penalty
                    )
                    info["wait_opportunity_penalty"] = opportunity_penalty
                if self._try_enter_wait_charging(deciding_drone_id, current_state):
                    info["wait_mode"] = "charge_or_swap"
                    return AppliedDecision(
                        drone_id=deciding_drone_id,
                        carried_reward=carried_reward,
                        info=info,
                        auto_advance_kind="until_decision",
                    )
                delta_wait = self._compute_wait_delta(deciding_drone_id)
                wait_until = self._t_now + delta_wait
                self._apply_wait_action(deciding_drone_id)
                self._active_wait_until[deciding_drone_id] = float(wait_until)
                info["wait_delta"] = delta_wait
                # 保持原训练语义：IDLE WAIT 的 T_idle 在动作提交时按完整等待窗结算。
                idle_penalty = -self._cfg.wait_idle_penalty_coef * delta_wait
                self._agent_cost_accum[deciding_drone_id] = (
                    self._agent_cost_accum.get(deciding_drone_id, 0.0) + idle_penalty
                )
                return AppliedDecision(
                    drone_id=deciding_drone_id,
                    carried_reward=carried_reward,
                    info=info,
                    auto_advance_kind="idle_wait",
                    wait_until=float(wait_until),
                )
            if current_state == TrainingDroneState.RIDING_WITH_TRUCK:
                if self._try_enter_wait_charging(deciding_drone_id, current_state):
                    self._active_wait_until.pop(deciding_drone_id, None)
                    info["wait_mode"] = "truck_charge_or_swap"
                    return AppliedDecision(
                        drone_id=deciding_drone_id,
                        carried_reward=carried_reward,
                        info=info,
                        auto_advance_kind="until_decision",
                    )
                self._apply_wait_action(deciding_drone_id)
                self._active_wait_until.pop(deciding_drone_id, None)
                info["wait_mode"] = "deferred_riding_with_truck"
                return AppliedDecision(
                    drone_id=deciding_drone_id,
                    carried_reward=carried_reward,
                    info=info,
                    auto_advance_kind="until_decision",
                )
            raise RuntimeError(f"状态 {current_state.value} 不允许执行 WAIT")

        if action.mode == PolicyMode.B:
            self._episode_dispatch_mode_b_count += 1
        elif action.mode == PolicyMode.C:
            self._episode_dispatch_mode_c_count += 1
        self._active_wait_until.pop(deciding_drone_id, None)
        self._apply_dispatch_action(
            trigger,
            action,
            coarse_plan=decision.coarse_plan,
        )
        info["applied_action"] = {
            "order_id": action.order_id,
            "mode": action.mode.value,
            "recover_node_id": action.recover_node_id,
        }
        return AppliedDecision(
            drone_id=deciding_drone_id,
            carried_reward=carried_reward,
            info=info,
            auto_advance_kind=("none" if self._decision_queue else "until_decision"),
        )

    def _apply_dispatch_action(
        self,
        trigger: DecisionTrigger,
        action: DispatchAction,
        *,
        coarse_plan: CoarsePlanView | None = None,
    ) -> None:
        """把 dispatch 动作落到订单池、无人机绑定和飞行账本上。"""
        entity_mgr = self._require_entity_manager()
        order_mgr = self._require_order_manager()
        drone = entity_mgr.drones[trigger.drone_id]
        order = order_mgr.pending_orders.pop(action.order_id)

        order.assigned_vehicle_id = drone.drone_id
        order.assigned_mode = action.mode.value
        order.update_status(TaskStatus.ASSIGNED)
        order.update_status(TaskStatus.PICKED_UP)
        order.update_status(TaskStatus.DELIVERING)
        order_mgr.assigned_orders[order.order_id] = order

        drone.assign_order(order.order_id, float(order.payload_weight))
        if self._drone_state[drone.drone_id] == TrainingDroneState.RIDING_WITH_TRUCK:
            if drone.drone_id in self._require_truck().docked_drones:
                self._require_truck().docked_drones.remove(drone.drone_id)

        launch_time = self._resolve_dispatch_launch_time(
            drone_id=drone.drone_id,
            trigger=trigger,
        )
        planned_truck_arrival_time: float | None = None
        planned_uav_arrival_time_lb: float | None = None
        planned_execution_slack_sec: float | None = None
        if action.mode == PolicyMode.C and action.recover_node_id is not None:
            (
                planned_truck_arrival_time,
                planned_uav_arrival_time_lb,
                planned_execution_slack_sec,
            ) = self._build_mode_c_commitment(
                drone=drone,
                order=order,
                recover_node_id=action.recover_node_id,
                launch_time=launch_time,
                coarse_plan=coarse_plan,
            )
        self._dispatch_commit[drone.drone_id] = DispatchCommit(
            order_id=order.order_id,
            mode=action.mode,
            selected_recover_node=action.recover_node_id,
            trigger_station_id=trigger.trigger_station_id,
            planned_truck_arrival_time=planned_truck_arrival_time,
            planned_uav_arrival_time_lb=planned_uav_arrival_time_lb,
            planned_execution_slack_sec=planned_execution_slack_sec,
        )
        if action.mode == PolicyMode.C and action.recover_node_id is not None:
            self._record_mode_c_selection_metrics(
                recover_node_id=action.recover_node_id,
                planned_truck_arrival_time=planned_truck_arrival_time,
                planned_uav_arrival_time_lb=planned_uav_arrival_time_lb,
                planned_execution_slack_sec=planned_execution_slack_sec,
            )
            self._acquire_reservation(
                drone_id=drone.drone_id,
                recover_node_id=action.recover_node_id,
            )
        self._schedule_flight_leg(
            drone_id=drone.drone_id,
            kind="deliver",
            target_pos=_clone_position(order.delivery_loc),
            payload=float(order.payload_weight),
            target_order_id=order.order_id,
            start_time=launch_time,
        )
        self._drone_state[drone.drone_id] = TrainingDroneState.FLYING_TO_DELIVER

    def _build_decision_context(self, trigger: DecisionTrigger) -> DecisionContext:
        """把内部 trigger 展开成给调用方可直接消费的决策上下文。"""
        runtime_state = self.build_runtime_state_view()
        coarse_plan = self._refresh_coarse_plan_if_needed(runtime_state)
        planner_snapshot = self._build_decision_planner_snapshot(runtime_state)
        execution_snapshot = self._build_decision_execution_snapshot(runtime_state)
        return self._build_decision_context_from_snapshot(
            trigger=trigger,
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            planner_snapshot=planner_snapshot,
            execution_snapshot=execution_snapshot,
        )

    def _build_decision_context_from_snapshot(
        self,
        *,
        trigger: DecisionTrigger,
        runtime_state: RuntimeStateView,
        coarse_plan: CoarsePlanView,
        planner_snapshot: DecisionPlannerSnapshot,
        execution_snapshot: DecisionExecutionSnapshot,
    ) -> DecisionContext:
        """基于共享 snapshot 构造单个 trigger 的决策上下文。"""
        action_lookup = self._build_action_lookup(
            drone_id=trigger.drone_id,
            coarse_plan=coarse_plan,
            runtime_state=runtime_state,
            trigger_type=trigger.trigger_type,
            trigger_station_id=trigger.trigger_station_id,
        )
        return DecisionContext(
            decision_id=int(trigger.decision_id),
            t_decision=float(runtime_state.t_now),
            deciding_drone_id=trigger.drone_id,
            trigger_type=trigger.trigger_type,
            trigger_station_id=trigger.trigger_station_id,
            runtime_state=runtime_state,
            coarse_plan=coarse_plan,
            planner_snapshot=planner_snapshot,
            execution_snapshot=execution_snapshot,
            action_lookup=action_lookup,
        )

    def _build_step_result(
        self,
        *,
        reward: float,
        info: Mapping[str, Any],
    ) -> EnvStepResult:
        """把当前内部状态封装成一次统一返回结果。"""
        decision_context = self.current_decision_context
        self._record_exposed_decision_metrics(decision_context)
        runtime_state = self.build_runtime_state_view()
        merged_info = dict(info)
        if self._last_reward_breakdown:
            merged_info["reward_breakdown"] = dict(self._last_reward_breakdown)
        if self._fallback_leg:
            merged_info["active_fallback_causes_by_drone"] = {
                str(drone_id): str(leg.cause)
                for drone_id, leg in self._fallback_leg.items()
            }
        merged_info["system_context_stats"] = self.build_system_context_stats()
        return EnvStepResult(
            reward=float(reward),
            done=self.is_done(),
            runtime_state=runtime_state,
            decision_context=decision_context,
            info=merged_info,
        )

    def _record_exposed_decision_metrics(
        self,
        decision_context: DecisionContext | None,
    ) -> None:
        """对外暴露一次新决策点时，记录 dispatch / mode C 供给诊断指标。"""
        if decision_context is None or not self._decision_queue:
            return
        trigger = self._decision_queue[0]
        if trigger.decision_id == self._last_exposed_decision_id:
            return
        self._last_exposed_decision_id = trigger.decision_id

        candidate_out = self._candidate_builder.build_from_decision_context(
            decision_context,
            last_seen_plan_version=int(decision_context.coarse_plan.plan_version),
        )
        rebuilt_action_lookup = candidate_out.resolved_action_lookup.as_action_lookup()
        if rebuilt_action_lookup != decision_context.action_lookup:
            raise RuntimeError(
                "记录 decision diagnostics 时 CandidateOutput 与 DecisionContext 不一致"
            )
        diagnostics = dict(candidate_out.diagnostics)
        _merge_int_counts(
            self._episode_mode_c_candidate_order_filter_counts,
            dict(diagnostics.get("mode_c_order_filter_counts", {})),
        )
        _merge_int_counts(
            self._episode_mode_c_candidate_node_filter_counts,
            dict(diagnostics.get("mode_c_node_filter_counts", {})),
        )

        dispatch_actions = [
            action
            for action in rebuilt_action_lookup
            if isinstance(action, DispatchAction)
        ]
        if not dispatch_actions:
            return

        self._episode_dispatch_decision_count += 1
        entity_mgr = self._require_entity_manager()
        legal_mode_c_nodes: set[str] = set()
        for action in dispatch_actions:
            if action.mode != PolicyMode.C:
                continue
            order = decision_context.runtime_state.pending_orders.get(action.order_id)
            if order is None:
                continue
            drone = entity_mgr.drones.get(decision_context.deciding_drone_id)
            if drone is None:
                continue
            legal_mode_c_nodes.update(
                self._iter_feasible_mode_c_recovery_nodes(
                    drone=drone,
                    order=order,
                    coarse_plan=decision_context.coarse_plan,
                )
            )
        self._episode_feasible_mode_c_recover_node_count_total += len(legal_mode_c_nodes)
        if legal_mode_c_nodes:
            self._episode_dispatch_decision_with_legal_mode_c_count += 1

    # ---------------------------------------------------------------------
    # Event loop
    # ---------------------------------------------------------------------

    def _advance_until_decision_or_done(self) -> None:
        """持续推进事件，直到出现新的决策点或 episode 结束。"""
        self._advance_until_decision_or_done_collect_reward_breakdown()

    def _advance_until_decision_or_done_collect_reward_breakdown(self) -> dict[str, float]:
        """持续推进到下一决策点，并返回推进期间聚合的 reward breakdown。"""
        reward_breakdown: dict[str, float] = {}
        while not self.is_done() and not self._decision_queue:
            next_time = self._next_event_time()
            if math.isinf(next_time):
                if self._enforces_upper_horizon():
                    next_time = self._cfg.upper_horizon_sec
                else:
                    raise RuntimeError(
                        "benchmark episode 存在未完成状态，但没有可推进的未来事件"
                    )
            if next_time <= self._t_now + _TIME_EPS:
                next_time = (
                    min(self._cfg.upper_horizon_sec, self._t_now + 1.0)
                    if self._enforces_upper_horizon()
                    else self._t_now + 1.0
                )
            self._advance_to_event(next_time)
            _merge_reward_breakdown(reward_breakdown, self._last_reward_breakdown)
        return reward_breakdown

    def _advance_to_event(
        self,
        t_next: float,
        *,
        clamp_to_horizon: bool = True,
    ) -> float:
        """推进到给定绝对时刻，并结算该区间内跨过的事件。

        返回值为全局事件奖励（仅用于 _last_reward_breakdown 记录）。
        per-dt 成本和事件奖励均已写入 _agent_cost_accum，step() 从中按 drone 取值。
        """
        t_next = float(t_next)
        if clamp_to_horizon and self._enforces_upper_horizon():
            t_next = min(t_next, float(self._cfg.upper_horizon_sec))
        if t_next < self._t_now - _TIME_EPS:
            raise ValueError(f"不能回退时间: t_next={t_next}, t_now={self._t_now}")

        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        order_mgr = self._require_order_manager()

        self._sync_in_transit_positions(t_next)

        # per-dt 成本已写入 _agent_cost_accum，global_reward 仅用于 breakdown 记录
        _global_per_dt, reward_breakdown = self._settle_per_dt_rewards(
            t_prev=self._t_now,
            t_next=t_next,
        )
        hard_failure_ready = self._collect_airborne_failure_events(t_next)
        failed_drones = {drone_id for drone_id, _, _ in hard_failure_ready}
        # delivery 先处理，保持"送达奖励优先于后续到达交互"的顺序。
        delivery_ready = [
            (drone_id, leg)
            for drone_id, leg in self._collect_flight_events(t_next, kind="deliver")
            if drone_id not in failed_drones
        ]
        non_delivery_ready = [
            (drone_id, leg)
            for drone_id, leg in self._collect_flight_events(
                t_next,
                kind=None,
                exclude={"deliver"},
            )
            if drone_id not in failed_drones
        ]
        delivery_service_ready = self._collect_delivery_service_events(t_next)
        truck_stops = self._collect_truck_stops(t_next)

        event_reward = 0.0
        delivery_reward = 0.0
        lateness_discount = 0.0
        mode_c_selection_reward = 0.0
        uav_energy_reward = 0.0
        hard_failure_reward = 0.0
        reservation_timeout_hard_failure_reward = 0.0

        # 1. 硬失败事件：半空停电会在到达事件之前截断当前飞行段。
        for drone_id, leg, failure_time in hard_failure_ready:
            penalty, energy_breakdown = self._process_airborne_failure_event(
                drone_id,
                leg=leg,
                failure_time=failure_time,
            )
            hard_failure_reward += penalty
            uav_energy_reward += float(energy_breakdown.get("uav_energy_penalty", 0.0))
            _merge_reward_breakdown(reward_breakdown, energy_breakdown)
            # 归因给失败的无人机
            self._agent_cost_accum[drone_id] = (
                self._agent_cost_accum.get(drone_id, 0.0) + penalty
            )

        # 2. UAV 订单送达 / mode A 背景完成（只记系统上下文统计，不进 PPO reward）
        for drone_id, leg in delivery_ready:
            delivery_bonus, delivery_lateness_discount, service_bonus, energy_breakdown = (
                self._process_delivery_event(drone_id, leg)
            )
            delivery_reward += delivery_bonus
            lateness_discount += delivery_lateness_discount
            mode_c_selection_reward += service_bonus
            uav_energy_reward += float(energy_breakdown.get("uav_energy_penalty", 0.0))
            _merge_reward_breakdown(reward_breakdown, energy_breakdown)
            # 送达奖励归因给完成送达的无人机
            self._agent_cost_accum[drone_id] = (
                self._agent_cost_accum.get(drone_id, 0.0)
                + delivery_bonus
                + delivery_lateness_discount
            )
        for drone_id, service_leg in delivery_service_ready:
            delivery_bonus, delivery_lateness_discount, service_bonus = (
                self._process_delivery_service_event(
                    drone_id,
                    service_leg,
                )
            )
            delivery_reward += delivery_bonus
            lateness_discount += delivery_lateness_discount
            mode_c_selection_reward += service_bonus
            self._agent_cost_accum[drone_id] = (
                self._agent_cost_accum.get(drone_id, 0.0)
                + delivery_bonus
                + delivery_lateness_discount
            )
        for stop in truck_stops:
            if stop.node_type == "customer" and stop.order_id:
                if stop.order_id in self._background_mode_a_pending:
                    self._background_mode_a_pending.remove(stop.order_id)
                    self._background_mode_a_completed.add(stop.order_id)
                    self._background_mode_a_completion_time_sum += float(
                        stop.arrival_time
                    )
                    self._truck_background_order_completion_events.append(
                        {
                            "order_id": str(stop.order_id),
                            "stop_seq": int(stop.seq),
                            "node_id": str(stop.node_id),
                            "arrival_time_sec": float(stop.arrival_time),
                            "departure_time_sec": float(stop.departure_time),
                        }
                    )
                else:
                    self._complete_truck_only_order_at_stop(stop)

        event_reward += (
            delivery_reward
            + lateness_discount
            + hard_failure_reward
            + mode_c_selection_reward
            + uav_energy_reward
        )
        if delivery_reward:
            reward_breakdown["delivery_bonus"] = delivery_reward
        if lateness_discount:
            reward_breakdown["lateness_discount"] = lateness_discount
        if mode_c_selection_reward:
            reward_breakdown["mode_c_attempt_bonus"] = mode_c_selection_reward
        if hard_failure_reward:
            reward_breakdown["hard_failure"] = hard_failure_reward

        # 3a. 卡车到站
        station_arrivals = [stop for stop in truck_stops if stop.node_type == "station"]
        if truck_stops:
            last_stop = truck_stops[-1]
            truck.current_loc = _clone_position(last_stop.position)
        if self._current_coarse_plan is not None:
            for stop in station_arrivals:
                if stop.node_id in self._current_coarse_plan.route_drift_ref:
                    self._completed_backbone_nodes_since_plan.add(stop.node_id)

        # 3b. UAV 到站 + host charge complete + truck charge complete
        newly_idle_from_host: list[str] = []
        mode_b_depot_return_ready: list[str] = []
        for drone_id, leg in non_delivery_ready:
            energy_breakdown = self._process_non_delivery_arrival(drone_id, leg)
            non_delivery_energy_reward = float(
                energy_breakdown.get("uav_energy_penalty", 0.0)
            )
            uav_energy_reward += non_delivery_energy_reward
            event_reward += non_delivery_energy_reward
            _merge_reward_breakdown(reward_breakdown, energy_breakdown)

        self._process_truck_charge_host(t_next)

        for host in list(entity_mgr.stations.values()) + list(entity_mgr.depots.values()):
            completed = host.tick_update(t_next)
            if completed:
                if isinstance(host, SwapStation):
                    for drone_id in completed:
                        if drone_id in self._mode_b_pending_depot_return:
                            mode_b_depot_return_ready.append(drone_id)
                        else:
                            newly_idle_from_host.append(drone_id)
                else:
                    newly_idle_from_host.extend(completed)
            self._sync_host_service_states(host)

        for drone_id in mode_b_depot_return_ready:
            drone = entity_mgr.drones[drone_id]
            drone.recharge_to_full()
            depot_id = self._mode_b_pending_depot_return.get(drone_id)
            depot = entity_mgr.depots.get(depot_id) if depot_id is not None else None
            if depot is None:
                self._mark_hard_failure(drone_id)
                continue
            self._schedule_mode_b_return_to_depot(
                drone_id,
                depot,
                start_time=t_next,
            )

        for drone_id in newly_idle_from_host:
            drone = entity_mgr.drones[drone_id]
            drone.recharge_to_full()
            if drone_id in self._dispatch_commit:
                self._dispatch_commit.pop(drone_id, None)
            self._drone_state[drone_id] = TrainingDroneState.IDLE

        # 3c. 由到达触发的交互
        if truck_stops:
            for stop in truck_stops:
                self._process_rendezvous_recovery(stop.node_id, t_next)

        # 4. severe overdue metrics + reservation timeout（timeout 本身只转移状态和记录指标）
        self._record_hard_overdue_metrics(t_next)
        reservation_timeout_hard_failure_reward += self._process_reservation_timeouts(t_next)
        event_reward += reservation_timeout_hard_failure_reward
        if reservation_timeout_hard_failure_reward:
            reward_breakdown["hard_failure"] = (
                reward_breakdown.get("hard_failure", 0.0)
                + reservation_timeout_hard_failure_reward
            )

        # 5. poisson / benchmark 动态订单注入
        order_mgr.tick(t_next, entity_mgr)

        # 6. 生成 decision context / done
        self._t_now = float(t_next)
        if self.is_done():
            self._last_reward_breakdown = reward_breakdown
            # 返回全局总奖励（per-dt + 事件），供测试和 breakdown 观察。
            # step() 不使用此返回值，而是从 _agent_cost_accum 按 drone 取归因奖励。
            return _global_per_dt + event_reward

        station_trigger: str | None = None
        refreshed_runtime_state = self.build_runtime_state_view()
        truck_stop_at_event_time = any(
            abs(float(stop.arrival_time) - float(t_next)) <= _TIME_EPS
            for stop in truck_stops
            if stop.node_type in {"customer", "station", "depot"}
        )
        self._refresh_coarse_plan_if_needed(
            refreshed_runtime_state,
            allow_truck_route_replan=truck_stop_at_event_time,
        )
        for stop in station_arrivals:
            if stop.node_id in self._active_launch_stations:
                station_trigger = stop.node_id
            self._active_launch_stations.discard(stop.node_id)
        if station_trigger is not None:
            # 车到站先恢复 riding_with_truck 的 WAIT 占位，再为当前仍在车上的 UAV 产生触发。
            self._resume_active_wait_for_station_trigger()
            for drone_id, state in self._drone_state.items():
                if state == TrainingDroneState.RIDING_WITH_TRUCK:
                    self._enqueue_decision(
                        drone_id,
                        trigger_type="truck_station_arrival",
                        trigger_station_id=station_trigger,
                    )
        if newly_idle_from_host:
            # 固定节点充换电完成会恢复 idle 上的 WAIT 占位，并为真正空闲的 UAV 重新建决策点。
            self._resume_active_wait_for_idle_trigger()
            for drone_id in newly_idle_from_host:
                if self._drone_state.get(drone_id) == TrainingDroneState.IDLE:
                    self._enqueue_decision(
                        drone_id,
                        trigger_type="idle_ready",
                        trigger_station_id=None,
                    )

        self._resume_due_active_waits(t_next)

        # order_mgr.tick() 可能注入了新 Poisson 订单；唤醒因无单而"睡死"的 idle UAV。
        self._wake_stranded_idle_drones()

        self._last_reward_breakdown = reward_breakdown
        # 返回全局总奖励（per-dt + 事件），供测试和 breakdown 观察。
        # step() 不使用此返回值，而是从 _agent_cost_accum 按 drone 取归因奖励。
        return _global_per_dt + event_reward

    # ---------------------------------------------------------------------
    # Event processing
    # ---------------------------------------------------------------------

    def _complete_truck_only_order_at_stop(self, stop: PlannedStop) -> None:
        if not stop.order_id:
            return
        order_mgr = self._require_order_manager()
        order = order_mgr.pending_orders.pop(stop.order_id, None)
        if order is None:
            return
        if not self._is_truck_only_order(order):
            order_mgr.pending_orders[order.order_id] = order
            return

        order.assigned_vehicle_id = self._require_truck().truck_id
        order.assigned_mode = "A"
        if order.status == TaskStatus.PENDING:
            order.update_status(TaskStatus.ASSIGNED)
        if order.status == TaskStatus.ASSIGNED:
            order.update_status(TaskStatus.PICKED_UP)
        if order.status == TaskStatus.PICKED_UP:
            order.update_status(TaskStatus.DELIVERING)
        order.actual_deliver_time = float(stop.arrival_time)
        if order.status in {TaskStatus.DELIVERING, TaskStatus.TIMEOUT}:
            order.update_status(TaskStatus.COMPLETED)
        order_mgr.completed_orders.append(order)
        self._truck_background_order_completion_events.append(
            {
                "order_id": str(stop.order_id),
                "stop_seq": int(stop.seq),
                "node_id": str(stop.node_id),
                "arrival_time_sec": float(stop.arrival_time),
                "departure_time_sec": float(stop.departure_time),
                "truck_only_dynamic": True,
            }
        )

    def _process_delivery_event(
        self,
        drone_id: str,
        leg: FlightLeg,
    ) -> tuple[float, float, float, dict[str, float]]:
        """处理一次到达客户点事件；真实完成与奖励在 delivery service 结束时结算。"""
        entity_mgr = self._require_entity_manager()
        order_mgr = self._require_order_manager()
        drone = entity_mgr.drones[drone_id]
        commit = self._dispatch_commit[drone_id]

        drone.current_loc = _clone_position(leg.target_pos)
        self._record_uav_leg_energy(leg=leg)
        energy_breakdown = self._settle_uav_leg_energy_penalty(
            drone_id=drone_id,
            leg=leg,
        )
        drone.battery_current = max(0.0, drone.battery_current - leg.energy_cost_j)
        released_order_id = drone.release_order()
        if released_order_id != commit.order_id:
            raise RuntimeError("delivery_event 订单绑定不一致")

        if commit.order_id not in order_mgr.assigned_orders:
            raise RuntimeError(f"delivery_event 找不到已指派订单: {commit.order_id}")
        self._flight_legs.pop(drone_id, None)
        self._drone_state[drone_id] = TrainingDroneState.DELIVERY_SERVICE

        service_duration = float(self._scene_solver_params().drone_service_time_order_s)
        if service_duration <= _TIME_EPS:
            delivery_bonus, lateness_discount, service_bonus = self._process_delivery_service_event(
                drone_id,
                DeliveryServiceLeg(
                    order_id=commit.order_id,
                    start_time=float(leg.arrival_time),
                    finish_time=float(leg.arrival_time),
                    service_pos=_clone_position(leg.target_pos),
                ),
            )
            return delivery_bonus, lateness_discount, service_bonus, energy_breakdown

        self._delivery_service_legs[drone_id] = DeliveryServiceLeg(
            order_id=commit.order_id,
            start_time=float(leg.arrival_time),
            finish_time=float(leg.arrival_time + service_duration),
            service_pos=_clone_position(leg.target_pos),
        )

        return 0.0, 0.0, 0.0, energy_breakdown

    def _process_delivery_service_event(
        self,
        drone_id: str,
        service_leg: DeliveryServiceLeg,
    ) -> tuple[float, float, float]:
        """处理客户点 delivery service 结束后的完成结算与后续转移。"""
        self._delivery_service_legs.pop(drone_id, None)
        order_mgr = self._require_order_manager()
        drone = self._require_entity_manager().drones[drone_id]
        commit = self._dispatch_commit[drone_id]
        order = order_mgr.assigned_orders.pop(service_leg.order_id)
        order.actual_deliver_time = float(service_leg.finish_time)
        order.update_status(TaskStatus.COMPLETED)
        order_mgr.completed_orders.append(order)
        self._episode_delivery_count += 1
        delivery_bonus, lateness_discount = self._compute_delivery_reward(order)
        drone.current_loc = _clone_position(service_leg.service_pos)
        self._drone_state[drone_id] = TrainingDroneState.DELIVERED

        if commit.mode == PolicyMode.B:
            target = self._select_return_host_mode_b(
                drone_id,
                current_time=service_leg.finish_time,
            )
            if target is None:
                self._mark_hard_failure(drone_id)
                return delivery_bonus, lateness_discount, 0.0
            if isinstance(target, Depot):
                self._schedule_mode_b_return_to_depot(
                    drone_id,
                    target,
                    start_time=service_leg.finish_time,
                )
            elif isinstance(target, SwapStation):
                self._schedule_mode_b_return_to_station(
                    drone_id,
                    target,
                    start_time=service_leg.finish_time,
                )
            else:
                raise TypeError(f"不支持的 mode B 返程宿主类型: {type(target)!r}")
            return delivery_bonus, lateness_discount, 0.0

        had_revalidation_failure = False
        selected_node = commit.selected_recover_node
        if selected_node is not None:
            is_valid, checks = self._revalidate_mode_c_recover_node(
                drone_id=drone_id,
                node_id=selected_node,
                t_now=service_leg.finish_time,
            )
            if is_valid:
                self._schedule_return_to_rendezvous(
                    drone_id,
                    selected_node,
                    start_time=service_leg.finish_time,
                )
                bonus = float(self._cfg.mode_c_attempt_bonus)
                if bonus > 0.0:
                    self._agent_cost_accum[drone_id] = (
                        self._agent_cost_accum.get(drone_id, 0.0) + bonus
                    )
                return delivery_bonus, lateness_discount, bonus

            had_revalidation_failure = True
            self._episode_mode_c_post_delivery_revalidation_fail_count += 1
            for reason_key, passed in checks.items():
                if not passed:
                    self._episode_mode_c_post_delivery_revalidation_fail_reasons[
                        reason_key
                    ] += 1
            self._release_reservation(
                drone_id,
                cause=FALLBACK_CAUSE_C_REVALIDATION_FAILED,
            )
            commit = replace(
                commit,
                selected_recover_node=None,
            )
            self._dispatch_commit[drone_id] = commit

        selection = self._select_post_delivery_mode_c_recover_node(
            drone_id=drone_id,
            t_now=service_leg.finish_time,
        )
        if selection is not None:
            selected_node = selection.recover_node_id
            self._dispatch_commit[drone_id] = replace(
                commit,
                selected_recover_node=selected_node,
                planned_truck_arrival_time=selection.planned_truck_arrival_time,
                planned_uav_arrival_time_lb=selection.planned_uav_arrival_time,
                planned_execution_slack_sec=selection.planned_execution_slack_sec,
            )
            self._record_mode_c_selection_metrics(
                recover_node_id=selected_node,
                planned_truck_arrival_time=selection.planned_truck_arrival_time,
                planned_uav_arrival_time_lb=selection.planned_uav_arrival_time,
                planned_execution_slack_sec=selection.planned_execution_slack_sec,
                t_now=service_leg.finish_time,
            )
            self._acquire_reservation(
                drone_id=drone_id,
                recover_node_id=selected_node,
            )
            self._schedule_return_to_rendezvous(
                drone_id,
                selected_node,
                start_time=service_leg.finish_time,
            )
            bonus = float(self._cfg.mode_c_attempt_bonus)
            if bonus > 0.0:
                self._agent_cost_accum[drone_id] = (
                    self._agent_cost_accum.get(drone_id, 0.0) + bonus
                )
            return delivery_bonus, lateness_discount, bonus

        if not had_revalidation_failure:
            self._episode_mode_c_post_delivery_revalidation_fail_count += 1
            for reason_key in _MODE_C_REVALIDATION_REASON_KEYS:
                self._episode_mode_c_post_delivery_revalidation_fail_reasons[
                    reason_key
                ] += 1

        self._release_reservation(
            drone_id,
            cause=FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE,
        )
        if not self._enter_fallback_recovery(
            drone_id,
            start_time=service_leg.finish_time,
            cause=FALLBACK_CAUSE_NO_POST_DELIVERY_C_NODE,
        ):
            self._mark_hard_failure(drone_id)
        return delivery_bonus, lateness_discount, 0.0

    def _compute_delivery_reward(self, order: Order) -> tuple[float, float]:
        """返回送达基础奖励和一次性 lateness discount（负值）。"""
        gross_reward = float(self._cfg.R_delivery_bonus)
        delivered_time = float(
            order.actual_deliver_time
            if order.actual_deliver_time is not None
            else self._t_now
        )
        tardiness = max(0.0, delivered_time - float(order.deadline))
        if tardiness <= _TIME_EPS:
            return gross_reward, 0.0

        raw_discount = max(0.0, float(self._cfg.late_delivery_penalty_coef) * tardiness)
        cap = max(0.0, float(self._cfg.late_delivery_penalty_cap))
        min_reward = max(0.0, float(self._cfg.min_late_delivery_reward))
        max_discount_for_positive_delivery = max(0.0, gross_reward - min_reward)
        discount = min(raw_discount, cap, max_discount_for_positive_delivery)
        self._episode_lateness_discount_total += discount
        return gross_reward, -discount

    def _process_non_delivery_arrival(
        self,
        drone_id: str,
        leg: FlightLeg,
    ) -> dict[str, float]:
        """处理除送达以外的飞行到达事件。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        drone.current_loc = _clone_position(leg.target_pos)
        self._record_uav_leg_energy(leg=leg)
        energy_breakdown = self._settle_uav_leg_energy_penalty(
            drone_id=drone_id,
            leg=leg,
        )
        drone.battery_current = max(0.0, drone.battery_current - leg.energy_cost_j)
        self._flight_legs.pop(drone_id, None)

        if leg.kind == "return_to_rendezvous":
            next_truck_arrival = self._next_backbone_arrival_time_for_node(
                leg.target_node_id,
                leg.arrival_time,
                include_current=True,
            )
            if next_truck_arrival is None:
                self._episode_mode_c_selected_node_expired_count += 1
                self._release_reservation(
                    drone_id,
                    cause=FALLBACK_CAUSE_ENERGY_OR_NODE_INVALID,
                )
                if not self._enter_fallback_recovery(
                    drone_id,
                    start_time=leg.arrival_time,
                    cause=FALLBACK_CAUSE_ENERGY_OR_NODE_INVALID,
                ):
                    self._mark_hard_failure(drone_id)
                return energy_breakdown
            self._drone_state[drone_id] = TrainingDroneState.WAITING_FOR_TRUCK
            self._rendezvous_wait_started_at[drone_id] = float(leg.arrival_time)
            return energy_breakdown

        if leg.kind == "mode_b_return_to_depot":
            self._mode_b_pending_depot_return.pop(drone_id, None)
            self._dispatch_commit.pop(drone_id, None)
            self._drone_state[drone_id] = TrainingDroneState.IDLE
            return energy_breakdown

        host = self._resolve_host(leg.target_node_id, leg.target_node_type)
        if leg.kind == "fallback_recovery":
            # fallback 到达后直接进入统一 charging host 入口，不再转成 return_to_station/depot 二次飞行。
            self._fallback_leg.pop(drone_id, None)
            self._on_arrive_charging_host(drone_id, host, leg.arrival_time)
            return energy_breakdown

        if leg.kind in {"return_to_station", "return_to_depot"}:
            self._on_arrive_charging_host(drone_id, host, leg.arrival_time)
            return energy_breakdown

        raise RuntimeError(f"未知飞行段类型: {leg.kind}")

    def _process_rendezvous_recovery(self, node_id: str, t_now: float) -> None:
        """处理"卡车到达某个 rendezvous 节点"时的回收配对。"""
        truck = self._require_truck()
        recovered: list[str] = []
        for drone_id, state in self._drone_state.items():
            if state != TrainingDroneState.WAITING_FOR_TRUCK:
                continue
            commit = self._dispatch_commit.get(drone_id)
            if commit is None or commit.selected_recover_node != node_id:
                continue
            recovered.append(drone_id)

        for drone_id in recovered:
            self._release_reservation(drone_id, cause="rendezvous_success")
            self._rendezvous_wait_started_at.pop(drone_id, None)
            self._drone_state[drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
            self._episode_mode_c_success_count += 1
            drone = self._require_entity_manager().drones[drone_id]
            drone.current_loc = _clone_position(truck.current_loc)
            if drone_id not in truck.docked_drones:
                truck.docked_drones.append(drone_id)

    def _process_airborne_failure_event(
        self,
        drone_id: str,
        *,
        leg: FlightLeg,
        failure_time: float,
    ) -> tuple[float, dict[str, float]]:
        """处理一次真实的空中电量耗尽事件。"""
        drone = self._require_entity_manager().drones[drone_id]
        progress_ratio = self._flight_leg_progress_ratio(
            leg=leg,
            t_now=float(failure_time),
        )
        self._record_uav_leg_energy(
            leg=leg,
            progress_ratio=progress_ratio,
        )
        energy_breakdown = self._settle_uav_leg_energy_penalty(
            drone_id=drone_id,
            leg=leg,
            progress_ratio=progress_ratio,
        )
        drone.battery_current = 0.0
        return self._mark_hard_failure(drone_id), energy_breakdown

    # ---------------------------------------------------------------------
    # Scheduling helpers
    # ---------------------------------------------------------------------

    def _schedule_flight_leg(
        self,
        *,
        drone_id: str,
        kind: str,
        target_pos: Position3D,
        payload: float,
        start_time: float | None = None,
        target_order_id: str | None = None,
        target_node_id: str | None = None,
        target_node_type: str | None = None,
    ) -> None:
        """为 UAV 写入一段新的飞行账本。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        effective_start_time = float(self._t_now if start_time is None else start_time)
        start_pos = _clone_position(drone.current_loc)
        estimate = self._uav_path_service.estimate(
            drone=drone,
            from_pos=start_pos,
            to_pos=target_pos,
            payload=payload,
        )
        route_version = int(self._drone_path_version.get(drone_id, 0) + 1)
        self._drone_path_version[drone_id] = route_version
        visual_path_points = tuple(
            _clone_position(pos) for pos in estimate.path_points
        )
        self._flight_legs[drone_id] = FlightLeg(
            kind=kind,
            start_time=effective_start_time,
            arrival_time=float(effective_start_time + estimate.flight_time_sec),
            start_pos=start_pos,
            target_pos=_clone_position(target_pos),
            path_points=visual_path_points,
            cumulative_distances_m=tuple(
                float(item) for item in estimate.cumulative_distances_m
            ),
            distance_m=float(estimate.distance_m),
            route_version=route_version,
            motion_mode=str(estimate.motion_mode),
            order_id=target_order_id,
            target_node_id=target_node_id,
            target_node_type=target_node_type,
            energy_cost_j=float(estimate.energy_j),
        )

    def _schedule_return_to_host(
        self,
        drone_id: str,
        host: ChargingHost,
        *,
        start_time: float | None = None,
    ) -> None:
        """为 mode B 或其他固定宿主返程写入飞行段。"""
        drone = self._require_entity_manager().drones[drone_id]
        effective_start_time = float(self._t_now if start_time is None else start_time)
        if isinstance(host, Depot):
            kind = "return_to_depot"
            self._drone_state[drone_id] = TrainingDroneState.RETURN_TO_DEPOT
            node_type = "depot"
            node_id = host.depot_id
        elif isinstance(host, SwapStation):
            kind = "return_to_station"
            self._drone_state[drone_id] = TrainingDroneState.RETURN_TO_STATION
            node_type = "station"
            node_id = host.station_id
        else:
            raise TypeError(f"不支持的 host 类型: {type(host)!r}")

        self._schedule_flight_leg(
            drone_id=drone_id,
            kind=kind,
            target_pos=_clone_position(host.get_location(effective_start_time)),
            payload=0.0,
            start_time=effective_start_time,
            target_node_id=node_id,
            target_node_type=node_type,
        )

    def _schedule_mode_b_return_to_depot(
        self,
        drone_id: str,
        depot: Depot,
        *,
        start_time: float | None = None,
    ) -> None:
        """Mode B 的最终返仓飞行；到达 depot 后直接回到 IDLE。"""
        effective_start_time = float(self._t_now if start_time is None else start_time)
        self._mode_b_pending_depot_return.pop(drone_id, None)
        self._drone_state[drone_id] = TrainingDroneState.RETURN_TO_DEPOT
        self._schedule_flight_leg(
            drone_id=drone_id,
            kind="mode_b_return_to_depot",
            target_pos=_clone_position(depot.get_location(effective_start_time)),
            payload=0.0,
            start_time=effective_start_time,
            target_node_id=depot.depot_id,
            target_node_type="depot",
        )

    def _schedule_mode_b_return_to_station(
        self,
        drone_id: str,
        station: SwapStation,
        *,
        start_time: float | None = None,
    ) -> None:
        """Mode B 电量不足返仓时，先到 station 补能，补能后自动返 depot。"""
        depot = self._require_depot()
        effective_start_time = float(self._t_now if start_time is None else start_time)
        station_pos = _clone_position(station.get_location(effective_start_time))
        self._mode_b_pending_depot_return[drone_id] = depot.depot_id
        self._drone_state[drone_id] = TrainingDroneState.RETURN_TO_STATION
        self._schedule_flight_leg(
            drone_id=drone_id,
            kind="return_to_station",
            target_pos=station_pos,
            payload=0.0,
            start_time=effective_start_time,
            target_node_id=station.station_id,
            target_node_type="station",
        )

    def _schedule_return_to_rendezvous(
        self,
        drone_id: str,
        node_id: str,
        *,
        start_time: float | None = None,
    ) -> None:
        """为 mode C 的 rendezvous 返程写入飞行段。"""
        host = self._resolve_fixed_node(node_id)
        effective_start_time = float(self._t_now if start_time is None else start_time)
        self._drone_state[drone_id] = TrainingDroneState.RETURN_TO_RENDEZVOUS
        self._schedule_flight_leg(
            drone_id=drone_id,
            kind="return_to_rendezvous",
            target_pos=_clone_position(host.get_location(effective_start_time)),
            payload=0.0,
            start_time=effective_start_time,
            target_node_id=node_id,
            target_node_type=_node_type_of_host(host),
        )

    def _build_mode_c_commitment(
        self,
        *,
        drone: Drone,
        order: Order,
        recover_node_id: str,
        launch_time: float,
        coarse_plan: CoarsePlanView | None = None,
    ) -> tuple[float, float, float]:
        effective_coarse_plan = (
            coarse_plan
            if coarse_plan is not None
            else self._build_coarse_plan_view(self._t_now)
        )
        planned_truck_arrival_time = effective_coarse_plan.truck_eta_map.get(
            recover_node_id
        )
        if planned_truck_arrival_time is None:
            raise RuntimeError(
                "mode C dispatch 缺少 recover node 的 truck ETA: "
                f"{recover_node_id}"
            )
        launch_pos = _clone_position(drone.current_loc)
        delivery_estimate = self._uav_path_service.estimate(
            drone=drone,
            from_pos=launch_pos,
            to_pos=order.delivery_loc,
            payload=float(order.payload_weight),
        )
        delivery_arrival_time = float(launch_time) + float(
            delivery_estimate.flight_time_sec
        )
        delivery_finish_time = (
            delivery_arrival_time
            + float(self._scene_solver_params().drone_service_time_order_s)
        )
        recover_host = self._resolve_fixed_node(recover_node_id)
        recover_estimate = self._uav_path_service.estimate(
            drone=drone,
            from_pos=order.delivery_loc,
            to_pos=recover_host.get_location(delivery_finish_time),
            payload=0.0,
        )
        planned_uav_arrival_time_lb = delivery_finish_time + float(
            recover_estimate.flight_time_sec
        )
        planned_execution_slack_sec = (
            float(planned_truck_arrival_time) - float(planned_uav_arrival_time_lb)
        )
        return (
            float(planned_truck_arrival_time),
            float(planned_uav_arrival_time_lb),
            float(planned_execution_slack_sec),
        )

    def _select_post_delivery_mode_c_recover_node(
        self,
        *,
        drone_id: str,
        t_now: float,
    ) -> ModeCPostDeliverySelection | None:
        """送达 service 后，优先基于当前 coarse plan ETA 选择 C 回收点。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        t_select = float(t_now)
        candidates: list[tuple[float, float, float, str, float, str]] = []
        candidate_eta_by_node: dict[str, tuple[float, str]] = {}
        if self._current_coarse_plan is not None:
            for node_id in self._current_coarse_plan.truck_backbone_route:
                if node_id not in self._current_coarse_plan.truck_eta_map:
                    continue
                truck_eta = float(self._current_coarse_plan.truck_eta_map[node_id])
                if truck_eta >= t_select - _TIME_EPS:
                    candidate_eta_by_node[str(node_id)] = (
                        truck_eta,
                        "current_coarse_plan",
                    )

        if not candidate_eta_by_node:
            future_visits = self._dedup_backbone_visits(
                self._future_backbone_visits(t_select)
            )
            for visit in future_visits:
                candidate_eta_by_node[str(visit.node_id)] = (
                    float(visit.arrival_time),
                    "execution_backbone",
                )

        for node_id, (truck_eta, eta_source) in candidate_eta_by_node.items():
            if node_id not in entity_mgr.stations and node_id not in entity_mgr.depots:
                continue
            host = self._resolve_fixed_node(node_id)
            host_pos = host.get_location(t_select)
            if not self._can_reach_from(
                drone=drone,
                from_pos=drone.current_loc,
                to_pos=host_pos,
                payload=0.0,
                safe_margin=drone.safe_margin_j,
                battery_current=drone.battery_current,
            ):
                continue
            uav_flight_time = self._estimate_flight_time(
                drone=drone,
                from_pos=drone.current_loc,
                to_pos=host_pos,
            )
            uav_arrival = t_select + float(uav_flight_time)
            wait_time = truck_eta - uav_arrival
            if wait_time < -_TIME_EPS:
                continue
            if (
                uav_arrival + float(self._cfg.rendezvous_execution_margin_sec)
                > truck_eta + _TIME_EPS
            ):
                continue
            if wait_time > float(self._cfg.rendezvous_max_wait_sec) + _TIME_EPS:
                continue
            candidates.append(
                (
                    truck_eta,
                    float(wait_time),
                    float(uav_flight_time),
                    node_id,
                    uav_arrival,
                    eta_source,
                )
            )

        if not candidates:
            self._record_mode_c_post_delivery_selection_debug(
                drone_id=drone_id,
                t_now=t_select,
                eta_source="missing",
                selected_node=None,
                resolved_truck_eta=None,
                candidate_count=0,
            )
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        truck_eta, wait_time, uav_flight_time, node_id, uav_arrival, eta_source = (
            candidates[0]
        )
        self._record_mode_c_post_delivery_selection_debug(
            drone_id=drone_id,
            t_now=t_select,
            eta_source=str(eta_source),
            selected_node=node_id,
            resolved_truck_eta=float(truck_eta),
            candidate_count=len(candidates),
        )
        return ModeCPostDeliverySelection(
            recover_node_id=node_id,
            planned_truck_arrival_time=float(truck_eta),
            planned_uav_arrival_time=float(uav_arrival),
            planned_execution_slack_sec=float(wait_time),
            uav_flight_time_sec=float(uav_flight_time),
            wait_time_sec=float(wait_time),
            eta_source=str(eta_source),
        )

    def _record_mode_c_post_delivery_selection_debug(
        self,
        *,
        drone_id: str,
        t_now: float,
        eta_source: str,
        selected_node: str | None,
        resolved_truck_eta: float | None,
        candidate_count: int,
    ) -> None:
        event = {
            "drone_id": str(drone_id),
            "t_now": float(t_now),
            "eta_source": str(eta_source),
            "selected_recover_node": (
                None if selected_node is None else str(selected_node)
            ),
            "resolved_truck_eta": (
                None if resolved_truck_eta is None else float(resolved_truck_eta)
            ),
            "candidate_count": int(candidate_count),
            "current_coarse_plan_version": (
                None
                if self._current_coarse_plan is None
                else int(self._current_coarse_plan.plan_version)
            ),
            "future_backbone_count": int(
                len(self._future_backbone_visits(float(t_now)))
            ),
            "truck_replan_pending": bool(self._truck_replan_pending),
            "truck_replan_pending_reasons": tuple(
                sorted(self._truck_replan_pending_reasons)
            ),
        }
        self._mode_c_post_delivery_selection_events.append(event)
        if len(self._mode_c_post_delivery_selection_events) > 100:
            del self._mode_c_post_delivery_selection_events[:-100]

    def _record_mode_c_selection_metrics(
        self,
        *,
        recover_node_id: str,
        planned_truck_arrival_time: float | None,
        planned_uav_arrival_time_lb: float | None,
        planned_execution_slack_sec: float | None,
        t_now: float | None = None,
    ) -> None:
        reservation_count = int(self._reservation_count.get(recover_node_id, 0))
        metric_t_now = float(self._t_now if t_now is None else t_now)
        truck_eta_remaining = (
            max(0.0, float(planned_truck_arrival_time) - metric_t_now)
            if planned_truck_arrival_time is not None
            else 0.0
        )
        filter_margin = (
            float(planned_execution_slack_sec) - float(self._cfg.rendezvous_filter_margin_sec)
            if planned_execution_slack_sec is not None
            else 0.0
        )
        execution_slack = (
            float(planned_execution_slack_sec) - float(self._cfg.rendezvous_execution_margin_sec)
            if planned_execution_slack_sec is not None
            else 0.0
        )
        self._episode_mode_c_selected_filter_margin_sum += float(filter_margin)
        self._episode_mode_c_selected_execution_slack_sum += float(execution_slack)
        self._episode_mode_c_selected_reservation_count_sum += float(reservation_count)
        self._episode_mode_c_selected_truck_eta_remaining_sum += float(truck_eta_remaining)
        self._episode_mode_c_selected_planned_truck_eta_sum += float(
            planned_truck_arrival_time or 0.0
        )
        self._episode_mode_c_selected_planned_uav_eta_sum += float(
            planned_uav_arrival_time_lb or 0.0
        )
        self._episode_mode_c_selected_planned_slack_sum += float(
            planned_execution_slack_sec or 0.0
        )

    def _record_mode_c_timeout_from_state(self, state: TrainingDroneState | None) -> None:
        state_name = _mode_c_state_label(state)
        if state_name is not None:
            self._episode_mode_c_timeout_from_state[state_name] += 1

    def _record_mode_c_fallback_from_state(self, state: TrainingDroneState | None) -> None:
        state_name = _mode_c_state_label(state)
        if state_name is not None:
            self._episode_mode_c_fallback_from_state[state_name] += 1

    def _enter_fallback_recovery(
        self,
        drone_id: str,
        *,
        start_time: float | None = None,
        cause: str | None = None,
    ) -> bool:
        """让 UAV 进入 fallback_recovery，并建立对应账本。

        返回值表示是否找到了可落地的兜底宿主。
        """
        host = self._select_deterministic_fallback_host(drone_id)
        if host is None:
            return False

        effective_start_time = float(self._t_now if start_time is None else start_time)
        normalized_cause = _normalize_fallback_cause(cause)
        self._record_mode_c_fallback_from_state(self._drone_state.get(drone_id))
        drone = self._require_entity_manager().drones[drone_id]
        target_pos = _clone_position(host.get_location(effective_start_time))
        arrival_time = effective_start_time + self._estimate_flight_time(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=target_pos,
        )
        self._fallback_leg[drone_id] = FallbackLeg(
            host_node_id=_host_id(host),
            host_node_type=_node_type_of_host(host),
            arrival_time=float(arrival_time),
            cause=normalized_cause,
        )
        self._episode_fallback_count += 1
        self._episode_fallback_cause_counts[normalized_cause] = (
            self._episode_fallback_cause_counts.get(normalized_cause, 0) + 1
        )
        if _is_ppo_attributed_fallback_cause(normalized_cause):
            self._episode_ppo_attributed_fallback_count += 1
        else:
            self._episode_system_attributed_fallback_count += 1
        self._drone_state[drone_id] = TrainingDroneState.FALLBACK_RECOVERY
        self._fallback_event_times.append(effective_start_time)
        self._schedule_flight_leg(
            drone_id=drone_id,
            kind="fallback_recovery",
            target_pos=target_pos,
            payload=0.0,
            start_time=effective_start_time,
            target_node_id=_host_id(host),
            target_node_type=_node_type_of_host(host),
        )
        return True

    def _acquire_reservation(
        self,
        *,
        drone_id: str,
        recover_node_id: str,
    ) -> None:
        """为一次 mode C dispatch 建立 reservation。"""
        if not self._cfg.reservation_enabled:
            return

        self._release_reservation(drone_id, cause="new_reservation_replace")
        reservation = ReservationState(
            recover_node=recover_node_id,
            issued_at=float(self._t_now),
        )
        self._reservations[drone_id] = reservation
        self._reservation_count[recover_node_id] = self._reservation_count.get(recover_node_id, 0) + 1

    def _release_reservation(self, drone_id: str, *, cause: str | None = None) -> None:
        """释放某架 UAV 当前 reservation，并回写节点计数。"""
        self._rendezvous_wait_started_at.pop(drone_id, None)
        reservation = self._reservations.pop(drone_id, None)
        if reservation is None:
            return
        release_cause = str(cause or FALLBACK_CAUSE_NONE)
        self._episode_reservation_release_cause_counts[release_cause] = (
            self._episode_reservation_release_cause_counts.get(release_cause, 0) + 1
        )
        node_id = reservation.recover_node
        current = self._reservation_count.get(node_id, 0)
        if current <= 1:
            self._reservation_count.pop(node_id, None)
        else:
            self._reservation_count[node_id] = current - 1

    def _process_reservation_timeouts(self, t_now: float) -> float:
        """扫描所有 rendezvous 等待 timeout，并执行 5c 兜底转移。

        reservation timeout 是 fallback 路径的一部分；timeout 本身不再产生 reward
        penalty。返回值仅包含 timeout 后无法进入 fallback 时的 hard failure 惩罚。
        """
        reward = 0.0
        for drone_id, reservation in list(self._reservations.items()):
            timeout_cost = self._reservation_timeout_cost(drone_id, reservation, t_now)
            if timeout_cost is None:
                continue
            self._record_mode_c_timeout_from_state(self._drone_state.get(drone_id))

            self._release_reservation(
                drone_id,
                cause=FALLBACK_CAUSE_RENDEZVOUS_WAIT_TIMEOUT,
            )
            commit = self._dispatch_commit.get(drone_id)
            if commit is not None:
                self._dispatch_commit[drone_id] = DispatchCommit(
                    order_id=commit.order_id,
                    mode=commit.mode,
                    selected_recover_node=None,
                    trigger_station_id=commit.trigger_station_id,
                    planned_truck_arrival_time=commit.planned_truck_arrival_time,
                    planned_uav_arrival_time_lb=commit.planned_uav_arrival_time_lb,
                    planned_execution_slack_sec=commit.planned_execution_slack_sec,
                )

            self._episode_reservation_timeout_count += 1
            self._episode_reservation_timeout_cost_sec += float(timeout_cost)
            if self._drone_state.get(drone_id) == TrainingDroneState.WAITING_FOR_TRUCK:
                if not self._enter_fallback_recovery(
                    drone_id,
                    start_time=t_now,
                    cause=FALLBACK_CAUSE_RENDEZVOUS_WAIT_TIMEOUT,
                ):
                    hard_penalty = self._mark_hard_failure(drone_id)
                    reward += hard_penalty
                    self._agent_cost_accum[drone_id] = (
                        self._agent_cost_accum.get(drone_id, 0.0) + hard_penalty
                    )
        return reward

    def _reservation_timeout_cost(
        self,
        drone_id: str,
        reservation: ReservationState,
        t_now: float,
    ) -> float | None:
        """返回 reservation timeout 的违规量；无 timeout 时返回 None。

        返回非 None 表示 timeout，上层会释放 reservation 并触发 fallback。
        返回 0.0 表示 timeout 但无额外惩罚（如卡车已离开）。
        """
        if self._drone_state.get(drone_id) != TrainingDroneState.WAITING_FOR_TRUCK:
            return None

        next_truck_arrival = self._next_backbone_arrival_time_for_node(
            reservation.recover_node,
            t_now,
            include_current=True,
        )
        if next_truck_arrival is None:
            return 0.0

        wait_started_at = self._rendezvous_wait_started_at.get(drone_id)
        if wait_started_at is None:
            return None
        wait_overrun = (
            float(t_now)
            - float(wait_started_at)
            - float(self._cfg.rendezvous_max_wait_sec)
        )
        if wait_overrun > _TIME_EPS:
            return wait_overrun
        return None

    def _record_hard_overdue_metrics(self, t_now: float) -> None:
        """记录 severe overdue 订单；不移除订单、不标记 TIMEOUT、不产生奖励惩罚。"""
        order_mgr = self._require_order_manager()
        active_orders = (
            list(order_mgr.pending_orders.values())
            + list(order_mgr.assigned_orders.values())
        )
        for order in active_orders:
            if not self._is_hard_overdue_candidate(order, t_now):
                continue
            order_id = str(order.order_id)
            if order_id in self._episode_hard_overdue_order_ids:
                continue
            self._episode_hard_overdue_order_ids.add(order_id)
            self._episode_hard_overdue_count = len(self._episode_hard_overdue_order_ids)

    def _is_hard_overdue_candidate(self, order: Order, t_now: float) -> bool:
        """检查订单是否达到 severe overdue 统计阈值。"""
        if not self._is_uav_primary_order(order):
            return False
        return (t_now - float(order.deadline)) > self._cfg.max_overdue_sec + _TIME_EPS

    def _compute_wait_opportunity_penalty(self, *, decision: DecisionContext) -> float:
        """计算 IDLE WAIT 跳过当前合法派单集合时的即时机会成本。

        只基于当前 `action_lookup` 中真实存在的 DispatchAction 计算，避免把
        当前无人机无法执行的 pending 订单误算进 WAIT 惩罚。
        """
        coef = float(self._cfg.wait_opportunity_penalty_coef)
        if coef <= _TIME_EPS:
            return 0.0

        dispatch_order_ids = {
            str(action.order_id)
            for action in decision.action_lookup
            if isinstance(action, DispatchAction)
        }
        if not dispatch_order_ids:
            return 0.0

        pending_orders = decision.runtime_state.pending_orders
        eligible_orders = [
            pending_orders[order_id]
            for order_id in sorted(dispatch_order_ids)
            if order_id in pending_orders
            and self._is_uav_primary_order(pending_orders[order_id])
        ]
        if not eligible_orders:
            return 0.0

        t_now = float(decision.t_decision)
        urgency = 0.0
        for order in eligible_orders:
            remaining = max(0.0, float(order.deadline) - t_now)
            window = max(_TIME_EPS, float(order.time_window_seconds))
            urgency = max(urgency, 1.0 - min(remaining / window, 1.0))

        backlog_factor = min(float(len(eligible_orders)), 5.0) / 5.0
        pressure = 0.5 * urgency + 0.5 * backlog_factor
        if pressure <= _TIME_EPS:
            return 0.0
        return -coef * pressure

    def _settle_per_dt_rewards(
        self,
        *,
        t_prev: float,
        t_next: float,
    ) -> tuple[float, dict[str, float]]:
        """结算 Phase 5c 的持续型奖励项，并将成本归因到各无人机的累积器。

        返回值仍保留 (global_reward, breakdown) 供 _advance_to_event 记录系统指标，
        但实际 PPO 奖励通过 _agent_cost_accum 按无人机归因，不再直接用 global_reward。
        """
        if t_next <= t_prev + _TIME_EPS:
            return 0.0, {}

        global_reward = 0.0
        breakdown: dict[str, float] = {}
        dt = t_next - t_prev

        # ── overdue / severe overdue：只统计时长，不进入 PPO reward ──
        overdue_dt_total = 0.0
        hard_overdue_dt_total = 0.0
        for order in self._active_uav_orders():
            overdue_dt = max(0.0, t_next - max(t_prev, float(order.deadline)))
            if overdue_dt > _TIME_EPS:
                overdue_dt_total += overdue_dt
            hard_threshold = float(order.deadline) + float(self._cfg.max_overdue_sec)
            hard_overdue_dt = max(0.0, t_next - max(t_prev, hard_threshold))
            if hard_overdue_dt > _TIME_EPS:
                hard_overdue_dt_total += hard_overdue_dt

        self._episode_overdue_time_sec += overdue_dt_total
        self._episode_hard_overdue_time_sec += hard_overdue_dt_total

        # ── T_idle / T_fallback：按状态归因给对应无人机 ──
        # T_wait 与 T_queue 仍统计时长，但不再作为 reward 惩罚项。
        wait_dt_total = 0.0
        idle_wait_dt_total = 0.0
        queue_dt_total = 0.0
        fallback_dt_total = 0.0
        ppo_fallback_dt_total = 0.0
        system_fallback_dt_total = 0.0

        for drone_id, state in self._drone_state.items():
            if state == TrainingDroneState.WAITING_FOR_TRUCK:
                wait_dt_total += dt
            elif (
                state == TrainingDroneState.ACTIVE_WAIT
                and self._active_wait_resume.get(drone_id)
                == TrainingDroneState.RIDING_WITH_TRUCK
            ):
                # T_idle（riding_with_truck 路径）：显式 WAIT 的无人机自己承担
                penalty = -self._cfg.wait_idle_penalty_coef * dt
                self._agent_cost_accum[drone_id] = (
                    self._agent_cost_accum.get(drone_id, 0.0) + penalty
                )
                idle_wait_dt_total += dt
            elif state == TrainingDroneState.QUEUEING_AT_HOST:
                queue_dt_total += dt
            elif state == TrainingDroneState.FALLBACK_RECOVERY:
                fallback_cause = (
                    self._fallback_leg[drone_id].cause
                    if drone_id in self._fallback_leg
                    else FALLBACK_CAUSE_HARD_FAILURE_FALLBACK
                )
                if _is_ppo_attributed_fallback_cause(fallback_cause):
                    penalty = -self._cfg.lambda_miss * dt
                    self._agent_cost_accum[drone_id] = (
                        self._agent_cost_accum.get(drone_id, 0.0) + penalty
                    )
                    ppo_fallback_dt_total += dt
                else:
                    system_fallback_dt_total += dt
                fallback_dt_total += dt

        self._episode_wait_time_sec += wait_dt_total
        self._episode_idle_time_sec += idle_wait_dt_total
        self._episode_queue_time_sec += queue_dt_total
        self._episode_fallback_time_sec += fallback_dt_total
        self._episode_ppo_attributed_fallback_time_sec += ppo_fallback_dt_total
        self._episode_system_attributed_fallback_time_sec += system_fallback_dt_total

        if idle_wait_dt_total > _TIME_EPS:
            penalty = -self._cfg.wait_idle_penalty_coef * idle_wait_dt_total
            global_reward += penalty
            breakdown["idle"] = penalty
        if fallback_dt_total > _TIME_EPS:
            penalty = -self._cfg.lambda_miss * fallback_dt_total
            global_reward += penalty
            breakdown["fallback"] = penalty
        if ppo_fallback_dt_total > _TIME_EPS:
            breakdown["fallback_ppo_attributed"] = (
                -self._cfg.lambda_miss * ppo_fallback_dt_total
            )
        if system_fallback_dt_total > _TIME_EPS:
            breakdown["fallback_system_attributed"] = (
                -self._cfg.lambda_miss * system_fallback_dt_total
            )

        return global_reward, breakdown

    # ---------------------------------------------------------------------
    # Selection helpers
    # ---------------------------------------------------------------------

    def _deliver_leg_feasible(self, drone: Drone, order: Order) -> bool:
        """检查当前电量是否足以把订单送到客户点。"""
        energy_need = self._estimate_energy_needed(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=order.delivery_loc,
            payload=float(order.payload_weight),
        )
        return energy_need <= drone.battery_current + _TIME_EPS

    def _estimate_delivery_arrival_time(self, drone: Drone, order: Order) -> float:
        """估算当前时刻派送该订单的送达绝对时刻。"""
        return self._estimate_delivery_launch_time(drone.drone_id) + self._estimate_flight_time(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=order.delivery_loc,
        )

    def _estimate_delivery_finish_time(self, drone: Drone, order: Order) -> float:
        """估算当前时刻派送该订单后完成 customer service 的绝对时刻。"""
        return (
            self._estimate_delivery_arrival_time(drone, order)
            + float(self._scene_solver_params().drone_service_time_order_s)
        )

    def _estimate_energy_after_delivery(self, drone: Drone, order: Order) -> float:
        """估算无人机送达订单后的剩余电量。"""
        return drone.battery_current - self._estimate_energy_needed(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=order.delivery_loc,
            payload=float(order.payload_weight),
        )

    def _iter_feasible_mode_c_recovery_nodes(
        self,
        *,
        drone: Drone,
        order: Order,
        coarse_plan: CoarsePlanView,
    ) -> tuple[str, ...]:
        """返回当前订单在 5b 语义下合法的 mode C 回收节点。"""
        t_deliver_finish = self._estimate_delivery_finish_time(drone, order)
        energy_after_delivery = self._estimate_energy_after_delivery(drone, order)
        if energy_after_delivery <= 0.0:
            return ()

        feasible_nodes: list[str] = []
        for recover_node_id in coarse_plan.get_recovery_candidates(order.order_id):
            if self._is_mode_c_recovery_feasible(
                drone=drone,
                deliver_pos=order.delivery_loc,
                recover_node_id=recover_node_id,
                t_deliver_finish=t_deliver_finish,
                energy_after_delivery=energy_after_delivery,
                coarse_plan=coarse_plan,
            ):
                feasible_nodes.append(recover_node_id)
        return tuple(feasible_nodes)

    def _is_mode_c_recovery_feasible(
        self,
        *,
        drone: Drone,
        deliver_pos: Position3D,
        recover_node_id: str,
        t_deliver_finish: float,
        energy_after_delivery: float,
        coarse_plan: CoarsePlanView,
    ) -> bool:
        """按 5b 的时序与能量口径判定单个 mode C 回收点是否合法。"""
        t_arrive_truck = coarse_plan.truck_eta_map.get(recover_node_id)
        if t_arrive_truck is None or t_arrive_truck <= t_deliver_finish + _TIME_EPS:
            return False

        host = self._resolve_fixed_node(recover_node_id)
        recover_pos = host.get_location(t_deliver_finish)
        t_arrive_uav = t_deliver_finish + self._estimate_flight_time(
            drone=drone,
            from_pos=deliver_pos,
            to_pos=recover_pos,
        )
        planned_wait = float(t_arrive_truck) - float(t_arrive_uav)
        if planned_wait < self._cfg.rendezvous_execution_margin_sec - _TIME_EPS:
            return False
        if planned_wait > self._cfg.rendezvous_max_wait_sec + _TIME_EPS:
            return False

        return self._can_reach_from(
            drone=drone,
            from_pos=deliver_pos,
            to_pos=recover_pos,
            payload=0.0,
            safe_margin=drone.safe_margin_j,
            battery_current=energy_after_delivery,
        )

    def _has_mode_b_return_host(self, drone: Drone, order: Order) -> bool:
        """检查送达后是否至少存在一个可达的 mode B 返程宿主。"""
        energy_after_delivery = self._estimate_energy_after_delivery(drone, order)
        if energy_after_delivery <= 0:
            return False

        return (
            self._select_return_host_mode_b(
                drone.drone_id,
                from_pos=order.delivery_loc,
                battery_current=energy_after_delivery,
                current_time=self._estimate_delivery_finish_time(drone, order),
            )
            is not None
        )

    def _select_return_host_mode_b(
        self,
        drone_id: str,
        *,
        from_pos: Position3D | None = None,
        battery_current: float | None = None,
        current_time: float | None = None,
    ) -> ChargingHost | None:
        """按 5b 规则选择 mode B 的确定性返程宿主。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        eval_pos = _clone_position(from_pos) if from_pos is not None else _clone_position(drone.current_loc)
        eval_battery = float(drone.battery_current if battery_current is None else battery_current)
        eval_time = float(self._t_now if current_time is None else current_time)

        depot = self._require_depot()
        depot_pos = depot.get_location(eval_time)
        if self._can_reach_from(
            drone=drone,
            from_pos=eval_pos,
            to_pos=depot_pos,
            payload=0.0,
            safe_margin=drone.safe_margin_j,
            battery_current=eval_battery,
        ):
            return depot

        reachable_stations: list[tuple[float, str, SwapStation]] = []
        for station in entity_mgr.stations.values():
            station_pos = station.get_location(eval_time)
            if not self._can_reach_from(
                drone=drone,
                from_pos=eval_pos,
                to_pos=station_pos,
                payload=0.0,
                safe_margin=drone.safe_margin_j,
                battery_current=eval_battery,
            ):
                continue
            if not self._can_reach_from(
                drone=drone,
                from_pos=station_pos,
                to_pos=depot_pos,
                payload=0.0,
                safe_margin=drone.safe_margin_j,
                battery_current=drone.battery_max,
            ):
                continue
            reachable_stations.append(
                (
                    self._estimate_path_distance(
                        from_pos=eval_pos,
                        to_pos=station_pos,
                    ),
                    station.station_id,
                    station,
                )
            )

        if not reachable_stations:
            return None

        reachable_stations.sort(key=lambda item: (item[0], item[1]))
        return reachable_stations[0][2]

    def _revalidate_mode_c_recover_node(
        self,
        *,
        drone_id: str,
        node_id: str,
        t_now: float,
    ) -> tuple[bool, dict[str, bool]]:
        """对 mode C 原选回收点执行 5b 的送达后复核。"""
        drone = self._require_entity_manager().drones[drone_id]
        eta_resolution = self._resolve_mode_c_revalidation_truck_eta(
            drone_id=drone_id,
            node_id=node_id,
            t_now=t_now,
        )
        truck_eta = eta_resolution.truck_eta

        node_still_valid = truck_eta is not None
        if not node_still_valid:
            checks = {
                "energy_feasible": False,
                "rendezvous_time_feasible": False,
                "node_still_valid": False,
            }
            self._record_mode_c_revalidation_debug(
                drone_id=drone_id,
                node_id=node_id,
                t_now=t_now,
                eta_resolution=eta_resolution,
                checks=checks,
                t_arrive_uav=None,
                wait_time=None,
            )
            return False, checks

        host = self._resolve_fixed_node(node_id)
        host_pos = host.get_location(t_now)
        energy_feasible = self._can_reach_from(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=host_pos,
            payload=0.0,
            safe_margin=drone.safe_margin_j,
            battery_current=drone.battery_current,
        )
        t_arrive_uav = t_now + self._estimate_flight_time(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=host_pos,
        )
        wait_time = float(truck_eta) - float(t_arrive_uav)
        rendezvous_time_feasible = (
            t_arrive_uav + self._cfg.rendezvous_execution_margin_sec
            <= float(truck_eta) + _TIME_EPS
            and wait_time <= self._cfg.rendezvous_max_wait_sec + _TIME_EPS
        )
        checks = {
            "energy_feasible": energy_feasible,
            "rendezvous_time_feasible": rendezvous_time_feasible,
            "node_still_valid": node_still_valid,
        }
        self._record_mode_c_revalidation_debug(
            drone_id=drone_id,
            node_id=node_id,
            t_now=t_now,
            eta_resolution=eta_resolution,
            checks=checks,
            t_arrive_uav=t_arrive_uav,
            wait_time=wait_time,
        )
        return all(checks.values()), checks

    def _resolve_mode_c_revalidation_truck_eta(
        self,
        *,
        drone_id: str,
        node_id: str,
        t_now: float,
    ) -> _ModeCRevalidationEtaResolution:
        """解析 Mode C 送达后复核使用的计划承诺 ETA。"""
        t_ref = float(t_now)
        commit = self._dispatch_commit.get(drone_id)
        commit_eta = None
        if (
            commit is not None
            and commit.selected_recover_node == node_id
            and commit.planned_truck_arrival_time is not None
        ):
            raw_commit_eta = float(commit.planned_truck_arrival_time)
            if raw_commit_eta >= t_ref - _TIME_EPS:
                commit_eta = raw_commit_eta

        current_coarse_plan_eta = None
        current_coarse_plan_version = None
        if self._current_coarse_plan is not None:
            current_coarse_plan_version = int(self._current_coarse_plan.plan_version)
            raw_plan_eta = self._current_coarse_plan.truck_eta_map.get(node_id)
            if raw_plan_eta is not None and float(raw_plan_eta) >= t_ref - _TIME_EPS:
                current_coarse_plan_eta = float(raw_plan_eta)

        execution_backbone_eta = self._next_backbone_arrival_time_for_node(
            node_id,
            t_ref,
            include_current=True,
        )
        future_backbone_count = len(self._future_backbone_visits(t_ref))

        if commit_eta is not None:
            truck_eta = commit_eta
            eta_source = "dispatch_commit"
        elif current_coarse_plan_eta is not None:
            truck_eta = current_coarse_plan_eta
            eta_source = "current_coarse_plan"
        elif execution_backbone_eta is not None:
            truck_eta = float(execution_backbone_eta)
            eta_source = "execution_backbone"
        else:
            truck_eta = None
            eta_source = "missing"

        return _ModeCRevalidationEtaResolution(
            truck_eta=truck_eta,
            eta_source=eta_source,
            commit_eta=commit_eta,
            current_coarse_plan_eta=current_coarse_plan_eta,
            current_coarse_plan_version=current_coarse_plan_version,
            execution_backbone_eta=execution_backbone_eta,
            future_backbone_count=int(future_backbone_count),
            truck_replan_pending=bool(self._truck_replan_pending),
            truck_replan_pending_reasons=tuple(
                sorted(self._truck_replan_pending_reasons)
            ),
        )

    def _record_mode_c_revalidation_debug(
        self,
        *,
        drone_id: str,
        node_id: str,
        t_now: float,
        eta_resolution: _ModeCRevalidationEtaResolution,
        checks: Mapping[str, bool],
        t_arrive_uav: float | None,
        wait_time: float | None,
    ) -> None:
        commit = self._dispatch_commit.get(drone_id)
        event = {
            "drone_id": str(drone_id),
            "recover_node": str(node_id),
            "t_now": float(t_now),
            "eta_source": str(eta_resolution.eta_source),
            "resolved_truck_eta": eta_resolution.truck_eta,
            "commit_planned_truck_arrival_time": (
                None if commit is None else commit.planned_truck_arrival_time
            ),
            "current_coarse_plan_version": eta_resolution.current_coarse_plan_version,
            "current_coarse_plan_eta": eta_resolution.current_coarse_plan_eta,
            "execution_backbone_eta": eta_resolution.execution_backbone_eta,
            "future_backbone_count": int(eta_resolution.future_backbone_count),
            "truck_replan_pending": bool(eta_resolution.truck_replan_pending),
            "truck_replan_pending_reasons": tuple(
                eta_resolution.truck_replan_pending_reasons
            ),
            "t_arrive_uav": None if t_arrive_uav is None else float(t_arrive_uav),
            "wait_time": None if wait_time is None else float(wait_time),
            "checks": {str(key): bool(value) for key, value in checks.items()},
        }
        self._mode_c_revalidation_events.append(event)
        if len(self._mode_c_revalidation_events) > 100:
            del self._mode_c_revalidation_events[:-100]

    def _select_deterministic_fallback_host(self, drone_id: str) -> ChargingHost | None:
        """按当前实现的 deterministic fallback 规则选宿主。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]

        depot = self._require_depot()
        depot_pos = depot.get_location(self._t_now)
        if self._can_reach_from(
            drone=drone,
            from_pos=drone.current_loc,
            to_pos=depot_pos,
            payload=0.0,
            safe_margin=drone.safe_margin_j,
            battery_current=drone.battery_current,
        ):
            return depot

        reachable_stations: list[tuple[float, SwapStation]] = []
        for station in entity_mgr.stations.values():
            station_pos = station.get_location(self._t_now)
            if self._can_reach_from(
                drone=drone,
                from_pos=drone.current_loc,
                to_pos=station_pos,
                payload=0.0,
                safe_margin=drone.safe_margin_j,
                battery_current=drone.battery_current,
            ):
                reachable_stations.append(
                    (
                        self._estimate_path_distance(
                            from_pos=drone.current_loc,
                            to_pos=station_pos,
                        ),
                        station,
                    )
                )
        if not reachable_stations:
            return None
        reachable_stations.sort(key=lambda item: (item[0], item[1].station_id))
        return reachable_stations[0][1]

    def _active_uav_orders(self) -> tuple[Order, ...]:
        """返回进入 PPO 主指标口径的所有未送达订单。"""
        order_mgr = self._require_order_manager()
        active = [
            order
            for order in list(order_mgr.pending_orders.values()) + list(order_mgr.assigned_orders.values())
            if self._is_uav_primary_order(order)
        ]
        active.sort(key=lambda item: item.order_id)
        return tuple(active)

    def _serialize_visual_order(self, order: Order) -> dict[str, Any]:
        return {
            "order_id": str(order.order_id),
            "status": str(order.status.value),
            "x": float(order.delivery_loc.x),
            "y": float(order.delivery_loc.y),
            "z": float(order.delivery_loc.z),
            "create_time": float(order.create_time),
            "deadline": float(order.deadline),
            "payload_weight": float(order.payload_weight),
            "assigned_mode": order.assigned_mode,
            "assigned_vehicle_id": order.assigned_vehicle_id,
            "actual_deliver_time": (
                None
                if order.actual_deliver_time is None
                else float(order.actual_deliver_time)
            ),
        }

    def _is_uav_primary_order(self, order: Order) -> bool:
        """判断订单是否属于 UAV 主指标统计范围。"""
        return float(order.payload_weight) <= self._heavy_payload_capacity + _TIME_EPS

    def _is_truck_only_order(self, order: Order) -> bool:
        """判断订单是否只能由卡车侧粗规划处理。"""
        return float(order.payload_weight) > self._heavy_payload_capacity + _TIME_EPS

    # ---------------------------------------------------------------------
    # Position / time helpers
    # ---------------------------------------------------------------------

    def _compute_wait_delta(self, drone_id: str) -> float:
        """按文档定义计算本次 WAIT 的精确推进量。"""
        state = self._drone_state[drone_id]
        if state == TrainingDroneState.IDLE:
            next_decision_t = self._next_global_decision_event_time()
            target_time = min(
                next_decision_t,
                self._t_now + self._cfg.max_wait_decision_gap_sec,
                self._cfg.upper_horizon_sec,
            )
            return max(_TIME_EPS, target_time - self._t_now)

        if state == TrainingDroneState.RIDING_WITH_TRUCK:
            next_station_t = self._next_active_launch_station_arrival_time_on_route()
            if not math.isfinite(next_station_t):
                return max(_TIME_EPS, self._cfg.upper_horizon_sec - self._t_now)
            return max(_TIME_EPS, min(next_station_t, self._cfg.upper_horizon_sec) - self._t_now)

        raise RuntimeError(f"状态 {state.value} 不允许执行 WAIT")

    def _next_global_decision_event_time(self) -> float:
        """返回下一个会触发 PPO 决策的全局事件时刻。"""
        candidates: list[float] = []
        coarse_plan = self._refresh_coarse_plan_if_needed()
        launch_stations = (
            self._active_launch_stations
            if self._planner_bridge is not None
            else set(coarse_plan.launch_candidate_stations)
        )

        for stop in self._planned_route_stops[self._planned_route_stop_i :]:
            if (
                stop.node_type == "station"
                and stop.node_id in launch_stations
                and stop.arrival_time > self._t_now + _TIME_EPS
            ):
                candidates.append(stop.arrival_time)

        for host in list(self._require_entity_manager().stations.values()) + list(self._require_entity_manager().depots.values()):
            candidates.extend(
                finish_time
                for finish_time in host.serving_drones.values()
                if finish_time > self._t_now + _TIME_EPS
            )

        if not candidates:
            return math.inf
        return min(candidates)

    def _next_station_arrival_time_on_route(self) -> float:
        """返回卡车未来下一个 station 停靠时刻。"""
        for stop in self._planned_route_stops[self._planned_route_stop_i :]:
            if stop.node_type == "station" and stop.arrival_time > self._t_now + _TIME_EPS:
                return stop.arrival_time
        return math.inf

    def _next_active_launch_station_arrival_time_on_route(self) -> float:
        """返回卡车未来下一个真实触发站的到达时刻。"""
        for stop in self._planned_route_stops[self._planned_route_stop_i :]:
            if (
                stop.node_type == "station"
                and stop.node_id in self._active_launch_stations
                and stop.arrival_time > self._t_now + _TIME_EPS
            ):
                return stop.arrival_time
        return math.inf

    def _resume_capped_wait_if_needed(self, drone_id: str) -> None:
        """idle WAIT 被 gap 上限截断且未被事件恢复时，手动恢复到新决策点。"""
        if self._drone_state.get(drone_id) != TrainingDroneState.ACTIVE_WAIT:
            return
        resume_state = self._active_wait_resume.get(drone_id)
        if resume_state != TrainingDroneState.IDLE:
            return

        self._active_wait_resume.pop(drone_id, None)
        self._active_wait_until.pop(drone_id, None)
        self._drone_state[drone_id] = TrainingDroneState.IDLE
        self._enqueue_decision(drone_id, "wait_resume", None)

    def _resume_due_active_waits(self, t_now: float) -> None:
        """恢复已到显式 WAIT 截止时刻的 IDLE UAV。"""
        due = [
            drone_id
            for drone_id, wait_until in list(self._active_wait_until.items())
            if wait_until <= t_now + _TIME_EPS
        ]
        for drone_id in due:
            self._resume_capped_wait_if_needed(drone_id)

    def _effective_battery_current(self, drone_id: str, t_now: float) -> float:
        """返回给定时刻的有效剩余电量（含飞行中线性耗电近似）。"""
        drone = self._require_entity_manager().drones[drone_id]
        leg = self._flight_legs.get(drone_id)
        if leg is None:
            return float(drone.battery_current)
        if t_now <= leg.start_time + _TIME_EPS:
            return float(drone.battery_current)
        if t_now >= leg.arrival_time - _TIME_EPS:
            return max(0.0, float(drone.battery_current) - float(leg.energy_cost_j))

        ratio = (t_now - leg.start_time) / max(_TIME_EPS, leg.arrival_time - leg.start_time)
        return max(0.0, float(drone.battery_current) - float(leg.energy_cost_j) * ratio)

    def _sync_in_transit_positions(self, t_now: float) -> None:
        """把 truck / UAV 的物理位置同步到给定时刻。"""
        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        truck.current_loc = self._truck_position_at_time(t_now)

        for drone_id, state in self._drone_state.items():
            drone = entity_mgr.drones[drone_id]
            if state in {
                TrainingDroneState.RIDING_WITH_TRUCK,
                TrainingDroneState.CHARGING_ON_TRUCK,
            }:
                # 这两个状态下无人机位置直接跟随卡车当前位置。
                drone.current_loc = _clone_position(truck.current_loc)
            elif state == TrainingDroneState.ACTIVE_WAIT:
                resume_state = self._active_wait_resume.get(drone_id)
                if resume_state == TrainingDroneState.RIDING_WITH_TRUCK:
                    drone.current_loc = _clone_position(truck.current_loc)

        for drone_id, leg in self._flight_legs.items():
            drone = entity_mgr.drones[drone_id]
            drone.current_loc = self._flight_leg_position_at(leg=leg, t_now=t_now)

    def _next_event_time(self) -> float:
        """从当前所有已知事件源里取下一个未来事件时刻。"""
        candidates = (
            [self._cfg.upper_horizon_sec]
            if self._enforces_upper_horizon()
            else []
        )

        if self._planned_route_stop_i < len(self._planned_route_stops):
            candidates.append(self._planned_route_stops[self._planned_route_stop_i].arrival_time)

        for leg in self._flight_legs.values():
            candidates.append(leg.arrival_time)

        for service_leg in self._delivery_service_legs.values():
            candidates.append(service_leg.finish_time)

        failure_time = self._next_airborne_failure_time()
        if math.isfinite(failure_time):
            candidates.append(failure_time)

        truck = self._require_truck()
        candidates.extend(float(done_time) for done_time in truck.serving_drones.values())

        for wait_until in self._active_wait_until.values():
            candidates.append(float(wait_until))

        reservation_probe_time = self._next_reservation_timeout_probe_time()
        if math.isfinite(reservation_probe_time):
            candidates.append(reservation_probe_time)

        for host in list(self._require_entity_manager().stations.values()) + list(self._require_entity_manager().depots.values()):
            for finish_time in host.serving_drones.values():
                candidates.append(float(finish_time))

        order_mgr = self._require_order_manager()
        next_poisson = float(getattr(order_mgr, "_next_order_time", math.inf))
        if math.isfinite(next_poisson):
            candidates.append(next_poisson)

        scheduled_dynamic = list(getattr(order_mgr, "_scheduled_dynamic", []))
        idx = int(getattr(order_mgr, "_scheduled_dynamic_i", 0))
        if idx < len(scheduled_dynamic):
            candidates.append(float(scheduled_dynamic[idx]["spawn_sim_s"]))

        future = [value for value in candidates if value > self._t_now + _TIME_EPS]
        if not future:
            return math.inf
        return min(future)

    def _next_reservation_timeout_probe_time(self) -> float:
        probe_times: list[float] = []
        for drone_id, reservation in self._reservations.items():
            state = self._drone_state.get(drone_id)
            if state != TrainingDroneState.WAITING_FOR_TRUCK:
                continue
            next_truck_arrival = self._next_backbone_arrival_time_for_node(
                reservation.recover_node,
                self._t_now,
                include_current=True,
            )
            if next_truck_arrival is None:
                probe_times.append(self._t_now + _TIME_EPS)
                continue
            wait_started_at = self._rendezvous_wait_started_at.get(drone_id)
            if wait_started_at is None:
                continue
            trigger_time = float(wait_started_at) + float(
                self._cfg.rendezvous_max_wait_sec
            )
            if trigger_time > self._t_now + _TIME_EPS:
                probe_times.append(trigger_time + _TIME_EPS)
            else:
                probe_times.append(self._t_now + _TIME_EPS)
        return min(probe_times) if probe_times else math.inf

    def _next_airborne_failure_time(self) -> float:
        """返回当前最早的真实空中电量耗尽时刻。"""
        failure_times = [
            failure_time
            for drone_id, leg in self._flight_legs.items()
            if (failure_time := self._compute_airborne_failure_time(drone_id, leg)) is not None
        ]
        if not failure_times:
            return math.inf
        return min(failure_times)

    def _collect_flight_events(
        self,
        t_now: float,
        *,
        kind: str | None,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, FlightLeg]]:
        """收集在 `t_now` 之前已经到达的飞行段事件。"""
        events: list[tuple[str, FlightLeg]] = []
        for drone_id, leg in list(self._flight_legs.items()):
            if leg.arrival_time > t_now + _TIME_EPS:
                continue
            if kind is not None and leg.kind != kind:
                continue
            if exclude is not None and leg.kind in exclude:
                continue
            events.append((drone_id, leg))
        events.sort(key=lambda item: (item[1].arrival_time, item[0]))
        return events

    def _collect_airborne_failure_events(
        self,
        t_now: float,
    ) -> list[tuple[str, FlightLeg, float]]:
        """收集在 `t_now` 前已经发生的空中电量耗尽事件。"""
        events: list[tuple[str, FlightLeg, float]] = []
        for drone_id, leg in list(self._flight_legs.items()):
            failure_time = self._compute_airborne_failure_time(drone_id, leg)
            if failure_time is None or failure_time > t_now + _TIME_EPS:
                continue
            events.append((drone_id, leg, failure_time))
        events.sort(key=lambda item: (item[2], item[0]))
        return events

    def _collect_truck_stops(self, t_now: float) -> list[PlannedStop]:
        """收集卡车在 `t_now` 之前已经到站的停靠点事件。"""
        stops: list[PlannedStop] = []
        while self._planned_route_stop_i < len(self._planned_route_stops):
            stop = self._planned_route_stops[self._planned_route_stop_i]
            if stop.arrival_time > t_now + _TIME_EPS:
                break
            stops.append(stop)
            self._planned_route_stop_i += 1
        return stops

    def _collect_delivery_service_events(
        self,
        t_now: float,
    ) -> list[tuple[str, DeliveryServiceLeg]]:
        """收集在 `t_now` 前已完成的 delivery service 事件。"""
        events: list[tuple[str, DeliveryServiceLeg]] = []
        for drone_id, service_leg in list(self._delivery_service_legs.items()):
            if service_leg.finish_time > t_now + _TIME_EPS:
                continue
            events.append((drone_id, service_leg))
        events.sort(key=lambda item: (item[1].finish_time, item[0]))
        return events

    # ---------------------------------------------------------------------
    # Charging / queue helpers
    # ---------------------------------------------------------------------

    def _try_enter_wait_charging(
        self,
        drone_id: str,
        current_state: TrainingDroneState,
    ) -> bool:
        """WAIT 时，如果当前位置宿主可补能且电量未满，则进入补能队列。"""
        drone = self._require_entity_manager().drones[drone_id]
        if drone.battery_current >= float(drone.battery_max) - _TIME_EPS:
            return False

        if current_state == TrainingDroneState.RIDING_WITH_TRUCK:
            return self._enter_truck_charging_queue(drone_id, self._t_now)

        if current_state == TrainingDroneState.IDLE:
            depot = self._depot_at_drone_location(drone_id, self._t_now)
            if depot is None:
                return False
            self._on_arrive_charging_host(drone_id, depot, self._t_now)
            return True

        return False

    def _depot_at_drone_location(self, drone_id: str, t_now: float) -> Depot | None:
        """返回无人机当前贴近的 depot；避免 WAIT 补能把无人机传送回仓。"""
        entity_mgr = self._require_entity_manager()
        drone = entity_mgr.drones[drone_id]
        depots = sorted(
            entity_mgr.depots.values(),
            key=lambda depot: drone.current_loc.distance_2d(depot.get_location(t_now)),
        )
        if not depots:
            return None
        depot = depots[0]
        if drone.current_loc.distance_2d(depot.get_location(t_now)) <= 5.0 + _TIME_EPS:
            return depot
        return None

    def _enter_truck_charging_queue(self, drone_id: str, t_now: float) -> bool:
        """把车载 WAIT 的无人机放入卡车补能队列，完成前不再视为可放飞。"""
        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        drone = entity_mgr.drones[drone_id]
        drone.current_loc = _clone_position(truck.current_loc)
        if drone_id not in truck.docked_drones:
            truck.docked_drones.append(drone_id)

        if drone_id in truck.serving_drones or drone_id in truck.wait_queue:
            self._drone_state[drone_id] = TrainingDroneState.CHARGING_ON_TRUCK
            return True

        if len(truck.serving_drones) < int(truck.parking_slots):
            truck.serving_drones[drone_id] = float(t_now) + float(truck.swap_time)
        else:
            truck.wait_queue.append(drone_id)
        self._drone_state[drone_id] = TrainingDroneState.CHARGING_ON_TRUCK
        return True

    def _process_truck_charge_host(self, t_now: float) -> list[str]:
        """推进 PPO 训练侧的车载补能；完成后无人机仍停靠在卡车上。"""
        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        completed: list[str] = []

        finished = [
            drone_id
            for drone_id, finish_time in list(truck.serving_drones.items())
            if float(finish_time) <= float(t_now) + _TIME_EPS
        ]
        for drone_id in finished:
            truck.serving_drones.pop(drone_id, None)
            drone = entity_mgr.drones.get(drone_id)
            if drone is None:
                continue
            drone.recharge_to_full()
            drone.current_loc = _clone_position(truck.current_loc)
            if drone_id not in truck.docked_drones:
                truck.docked_drones.append(drone_id)
            if self._drone_state.get(drone_id) == TrainingDroneState.CHARGING_ON_TRUCK:
                self._drone_state[drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
            completed.append(drone_id)

        while truck.wait_queue and len(truck.serving_drones) < int(truck.parking_slots):
            next_drone_id = truck.wait_queue.pop(0)
            drone = entity_mgr.drones.get(next_drone_id)
            if drone is None:
                continue
            drone.current_loc = _clone_position(truck.current_loc)
            if next_drone_id not in truck.docked_drones:
                truck.docked_drones.append(next_drone_id)
            if drone.battery_current >= float(drone.battery_max) - _TIME_EPS:
                if self._drone_state.get(next_drone_id) == TrainingDroneState.CHARGING_ON_TRUCK:
                    self._drone_state[next_drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
                continue
            truck.serving_drones[next_drone_id] = float(t_now) + float(truck.swap_time)
            self._drone_state[next_drone_id] = TrainingDroneState.CHARGING_ON_TRUCK

        return completed

    def _estimate_truck_charge_remaining_sec(
        self,
        drone_id: str,
        t_now: float,
    ) -> float | None:
        """估算车载补能完成剩余时间，供 runtime snapshot 使用。"""
        truck = self._require_truck()
        if drone_id in truck.serving_drones:
            return max(0.0, float(truck.serving_drones[drone_id]) - float(t_now))
        if drone_id not in truck.wait_queue:
            return None
        if truck.serving_drones:
            first_slot_time = max(0.0, min(truck.serving_drones.values()) - float(t_now))
        else:
            first_slot_time = 0.0
        queue_index = truck.wait_queue.index(drone_id)
        return first_slot_time + float(queue_index + 1) * float(truck.swap_time)

    def _on_arrive_charging_host(
        self,
        drone_id: str,
        host: ChargingHost,
        t_now: float,
    ) -> None:
        """统一处理 UAV 到达 depot/station 后的入队或入服。"""
        drone = self._require_entity_manager().drones[drone_id]
        drone.current_loc = _clone_position(host.get_location(t_now))
        host.arrive(drone_id, t_now)
        # 状态分流完全以 host.arrive() 的真实队列结果为准，不手写第二套判定。
        if drone_id in host.serving_drones:
            self._drone_state[drone_id] = TrainingDroneState.CHARGING_OR_SWAP
        elif drone_id in host.wait_queue:
            self._drone_state[drone_id] = TrainingDroneState.QUEUEING_AT_HOST
        else:
            raise RuntimeError("charging host arrival state inconsistent")

    def _sync_host_service_states(self, host: ChargingHost) -> None:
        """把 host 内部队列真值回写到训练状态层。"""
        for drone_id in list(host.serving_drones):
            if self._drone_state.get(drone_id) in {
                TrainingDroneState.QUEUEING_AT_HOST,
                TrainingDroneState.CHARGING_OR_SWAP,
            }:
                self._drone_state[drone_id] = TrainingDroneState.CHARGING_OR_SWAP
        for drone_id in list(host.wait_queue):
            if self._drone_state.get(drone_id) in {
                TrainingDroneState.QUEUEING_AT_HOST,
                TrainingDroneState.CHARGING_OR_SWAP,
            }:
                self._drone_state[drone_id] = TrainingDroneState.QUEUEING_AT_HOST

    # ---------------------------------------------------------------------
    # Decision queue helpers
    # ---------------------------------------------------------------------

    def _wake_stranded_idle_drones(self) -> None:
        """为既不在决策队列也不在 WAIT 中的 idle UAV 补入决策点。

        idle UAV 在无授权订单时会被 _enqueue_decision 的早返回"睡死"——
        既无队列条目也无 WAIT 占位。order_mgr.tick() 注入新单后调用此方法
        可确保这些 UAV 被重新唤醒。
        """
        queued = {item.drone_id for item in self._decision_queue}
        for drone_id, state in self._drone_state.items():
            # ACTIVE_WAIT 的 UAV 已有 WAIT 占位，到期后会自行恢复，无需处理。
            if state == TrainingDroneState.IDLE and drone_id not in queued:
                self._enqueue_decision(drone_id, "order_arrival_wake", None)

    def _enqueue_initial_idle_decisions(self) -> None:
        """为 reset 后处于 idle 的 UAV 批量创建初始决策点。"""
        for drone_id, state in sorted(self._drone_state.items()):
            if state == TrainingDroneState.IDLE:
                self._enqueue_decision(drone_id, "initial_idle", None)

    def _enqueue_decision(
        self,
        drone_id: str,
        trigger_type: str,
        trigger_station_id: str | None,
    ) -> None:
        """向内部决策队列追加一个触发点。"""
        if self.is_done():
            return
        state = self._drone_state.get(drone_id)
        if state not in {TrainingDroneState.IDLE, TrainingDroneState.RIDING_WITH_TRUCK}:
            return
        runtime_state = self.build_runtime_state_view()
        if not self._has_authorized_orders_now(runtime_state):
            return
        if any(item.drone_id == drone_id for item in self._decision_queue):
            return
        self._decision_queue.append(
            DecisionTrigger(
                decision_id=self._next_decision_id,
                drone_id=drone_id,
                trigger_type=trigger_type,
                trigger_station_id=trigger_station_id,
                t_enqueued=float(self._t_now),
            )
        )
        self._next_decision_id += 1

    def _resume_active_wait_for_idle_trigger(self) -> None:
        """在"固定节点空闲触发"发生时恢复 idle 上的 WAIT 占位。"""
        resumable = [
            drone_id
            for drone_id, state in self._active_wait_resume.items()
            if state == TrainingDroneState.IDLE
        ]
        for drone_id in resumable:
            self._active_wait_resume.pop(drone_id, None)
            self._active_wait_until.pop(drone_id, None)
            self._drone_state[drone_id] = TrainingDroneState.IDLE
            self._enqueue_decision(drone_id, "wait_resume", None)

    def _resume_active_wait_for_station_trigger(self) -> None:
        """在"卡车到站触发"发生时恢复 WAIT 占位。"""
        resumable = list(self._active_wait_resume.items())
        for drone_id, state in resumable:
            self._active_wait_resume.pop(drone_id, None)
            self._active_wait_until.pop(drone_id, None)
            self._drone_state[drone_id] = state
            if state == TrainingDroneState.IDLE:
                self._enqueue_decision(drone_id, "wait_resume", None)

    def _has_authorized_orders_now(
        self,
        runtime_state: RuntimeStateView | None = None,
    ) -> bool:
        """当前时刻是否存在 coarse plan 授权订单。"""
        if runtime_state is None:
            runtime_state = self.build_runtime_state_view()

        coarse_plan = self._refresh_coarse_plan_if_needed(runtime_state)
        live_authorized = any(
            order_id in runtime_state.pending_orders
            for order_id in coarse_plan.authorized_orders
        )
        if live_authorized:
            return True

        if self._planner_bridge is not None:
            pending_uav_orders = any(
                self._is_uav_primary_order(order)
                for order in runtime_state.pending_orders.values()
            )
            if pending_uav_orders:
                coarse_plan = self._refresh_coarse_plan_if_needed(
                    runtime_state,
                    force_replan=True,
                )
                return any(
                    order_id in runtime_state.pending_orders
                    for order_id in coarse_plan.authorized_orders
                )

        return False

    # ---------------------------------------------------------------------
    # Phase 4 / reset helpers
    # ---------------------------------------------------------------------

    def _load_phase4_artifacts(self) -> dict[str, Any]:
        """读取 Phase 4 路线产物，并转成当前环境内部结构。"""
        route_dir = self._phase4_route_dir
        execution_payload = _load_json(route_dir / "truck_execution_route.json")
        planned_stops: list[PlannedStop] = []
        planned_segments: list[PlannedTruckSegment] = []
        backbone_cache: list[BackboneVisit] = []
        mode_a_order_ids: list[str] = []

        for raw_stop in execution_payload["stops"]:
            stop = PlannedStop(
                seq=int(raw_stop["seq"]),
                node_type=str(raw_stop["node_type"]),
                node_id=str(raw_stop["node_id"]),
                position=Position3D(
                    x=float(raw_stop["x"]),
                    y=float(raw_stop["y"]),
                    z=float(raw_stop.get("z", 0.0)),
                ),
                order_id=None if raw_stop.get("order_id") is None else str(raw_stop["order_id"]),
                arrival_time=float(raw_stop["arrival_time_sec"]),
                departure_time=float(raw_stop["departure_time_sec"]),
            )
            planned_stops.append(stop)

            if stop.node_type in {"station", "depot"} and stop.seq > 0:
                backbone_cache.append(
                    BackboneVisit(
                        node_id=stop.node_id,
                        arrival_time=stop.arrival_time,
                        departure_time=stop.departure_time + _BACKBONE_DEPARTURE_EPS,
                    )
                )
            if stop.node_type == "customer" and stop.order_id:
                # truck_execution_route 里的 customer stop 对应 mode A 背景完成事件。
                mode_a_order_ids.append(stop.order_id)

        raw_segments = execution_payload.get("segments")
        if raw_segments is not None:
            if not isinstance(raw_segments, list):
                raise ValueError("truck_execution_route.segments 必须为 list")
            if len(raw_segments) != max(0, len(planned_stops) - 1):
                raise ValueError(
                    "truck_execution_route.segments 与 stops 数量不一致: "
                    f"segments={len(raw_segments)}, stops={len(planned_stops)}"
                )
            road_nodes = None
            for idx, raw_segment in enumerate(raw_segments, start=1):
                if not isinstance(raw_segment, Mapping):
                    raise ValueError("truck_execution_route.segments 元素必须为对象")
                from_stop = planned_stops[idx - 1]
                to_stop = planned_stops[idx]
                osm_node_path_raw = raw_segment.get("osm_node_path")
                osm_node_path = (
                    tuple(str(item) for item in osm_node_path_raw)
                    if isinstance(osm_node_path_raw, list)
                    else ()
                )
                geometry = _parse_segment_geometry_payload(raw_segment)
                if geometry is None:
                    if not osm_node_path:
                        raise ValueError("缺少 segment.geometry，且 osm_node_path 非法")
                    if road_nodes is None:
                        _, road_nodes = self._require_road_graph()
                    geometry = _build_geometry_from_osm_node_path(
                        from_pos=from_stop.position,
                        to_pos=to_stop.position,
                        osm_node_path=list(osm_node_path),
                        road_nodes=road_nodes,
                    )
                planned_segments.append(
                    PlannedTruckSegment(
                        segment_id=idx - 1,
                        from_node_id=str(raw_segment.get("from_node_id", from_stop.node_id)),
                        to_node_id=str(raw_segment.get("to_node_id", to_stop.node_id)),
                        from_node_type=str(raw_segment.get("from_node_type", from_stop.node_type)),
                        to_node_type=str(raw_segment.get("to_node_type", to_stop.node_type)),
                        start_time=float(from_stop.departure_time),
                        end_time=float(to_stop.arrival_time),
                        distance_m=float(raw_segment.get("distance_m", _polyline_distance_2d(geometry))),
                        geometry=geometry,
                        cumulative_distances_m=_build_cumulative_distances(geometry),
                        osm_node_path=osm_node_path,
                    )
                )

        return {
            "planned_stops": planned_stops,
            "planned_segments": planned_segments,
            "backbone_cache": backbone_cache,
            "mode_a_order_ids": tuple(mode_a_order_ids),
        }

    def _append_patrol_loop_if_needed(self, *, force: bool = False) -> bool:
        """在 poisson 模式下按当前实现追加巡站循环。"""
        if not self._planned_route_stops:
            return False
        if self._order_source.mode != OrderSourceMode.POISSON:
            return False

        last_stop = self._planned_route_stops[-1]
        if last_stop.node_type != "depot":
            return False
        if not force and last_stop.arrival_time >= (
            self._cfg.upper_horizon_sec - self._cfg.patrol_min_remaining_sec
        ):
            return False

        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        depot = self._require_depot()
        road_graph, road_nodes = self._require_road_graph()
        nearest_cache: dict[tuple[float, float], tuple[str, float]] = {}
        stations = sorted(entity_mgr.stations.values(), key=lambda item: item.station_id)
        if not stations:
            return False

        seq = self._planned_route_stops[-1].seq
        t_cursor = max(float(last_stop.departure_time), float(self._t_now))
        patrol_k = min(len(stations), self._cfg.patrol_stations_per_loop)
        if patrol_k <= 0:
            return False
        patrol_offset = 0
        station_hold_time = max(
            float(self._scene_solver_params().truck_drone_launch_time_s),
            float(self._scene_solver_params().truck_drone_recover_time_s),
        )
        initial_stop_count = len(self._planned_route_stops)

        # 一旦决定进入 poisson 巡站补全路径，就持续追加到 upper horizon，
        # 避免 episode 尾段再次出现"未来 backbone 耗尽"的契约破口。
        while t_cursor < self._cfg.upper_horizon_sec:
            current_pos = _clone_position(depot.location)
            pending_batch = [
                stations[(patrol_offset + idx) % len(stations)]
                for idx in range(patrol_k)
            ]
            patrol_offset = (patrol_offset + patrol_k) % len(stations)
            chosen: list[SwapStation] = []

            while pending_batch:
                next_station = min(
                    pending_batch,
                    key=lambda station: (
                        current_pos.distance_2d(station.location),
                        station.station_id,
                    ),
                )
                pending_batch.remove(next_station)
                chosen.append(next_station)
                current_pos = next_station.location

            current_pos = _clone_position(depot.location)
            for station in chosen:
                prev_stop = self._planned_route_stops[-1]
                segment_route = _build_route_between_positions(
                    from_pos=current_pos,
                    to_pos=station.location,
                    road_graph=road_graph,
                    road_nodes=road_nodes,
                    nearest_cache=nearest_cache,
                )
                segment_geometry = segment_route.geometry
                travel_distance = _polyline_distance_2d(segment_geometry)
                segment_start_time = t_cursor
                travel_time = travel_distance / max(_TIME_EPS, truck.speed)
                t_cursor += travel_time
                seq += 1
                departure_time = t_cursor + station_hold_time
                planned_stop = PlannedStop(
                    seq=seq,
                    node_type="station",
                    node_id=station.station_id,
                    position=_clone_position(station.location),
                    order_id=None,
                    arrival_time=t_cursor,
                    departure_time=departure_time,
                )
                self._planned_route_stops.append(planned_stop)
                self._planned_route_segments.append(
                    PlannedTruckSegment(
                        segment_id=len(self._planned_route_segments),
                        from_node_id=prev_stop.node_id,
                        to_node_id=station.station_id,
                        from_node_type=prev_stop.node_type,
                        to_node_type="station",
                        start_time=segment_start_time,
                        end_time=t_cursor,
                        distance_m=travel_distance,
                        geometry=segment_geometry,
                        cumulative_distances_m=_build_cumulative_distances(segment_geometry),
                        osm_node_path=segment_route.osm_node_path,
                    )
                )
                self._full_backbone_cache.append(
                    BackboneVisit(
                        node_id=station.station_id,
                        arrival_time=t_cursor,
                        departure_time=departure_time + _BACKBONE_DEPARTURE_EPS,
                    )
                )
                current_pos = station.location
                t_cursor = departure_time

            segment_route = _build_route_between_positions(
                from_pos=current_pos,
                to_pos=depot.location,
                road_graph=road_graph,
                road_nodes=road_nodes,
                nearest_cache=nearest_cache,
            )
            segment_geometry = segment_route.geometry
            travel_distance_back = _polyline_distance_2d(segment_geometry)
            segment_start_time = t_cursor
            travel_time_back = travel_distance_back / max(_TIME_EPS, truck.speed)
            t_cursor += travel_time_back
            seq += 1
            # depot 同样属于 future fixed node；若不写入骨架缓存，
            # 则尾段会出现“物理路线仍在继续，但 coarse backbone 已耗尽”的语义裂缝。
            depot_stop = PlannedStop(
                seq=seq,
                node_type="depot",
                node_id=depot.depot_id,
                position=_clone_position(depot.location),
                order_id=None,
                arrival_time=t_cursor,
                departure_time=t_cursor,
            )
            self._planned_route_stops.append(depot_stop)
            prev_stop = self._planned_route_stops[-2]
            self._planned_route_segments.append(
                PlannedTruckSegment(
                    segment_id=len(self._planned_route_segments),
                    from_node_id=prev_stop.node_id,
                    to_node_id=depot.depot_id,
                    from_node_type=prev_stop.node_type,
                    to_node_type="depot",
                    start_time=segment_start_time,
                    end_time=t_cursor,
                    distance_m=travel_distance_back,
                    geometry=segment_geometry,
                    cumulative_distances_m=_build_cumulative_distances(segment_geometry),
                    osm_node_path=segment_route.osm_node_path,
                )
            )
            self._full_backbone_cache.append(
                BackboneVisit(
                    node_id=depot.depot_id,
                    arrival_time=t_cursor,
                    departure_time=t_cursor + _BACKBONE_DEPARTURE_EPS,
                )
            )

        self._full_backbone_cache.sort(key=lambda item: (item.arrival_time, item.node_id))
        return len(self._planned_route_stops) > initial_stop_count

    def _bind_truck_route(self) -> None:
        """把当前 `_planned_route_stops` 绑定到 Truck 实体的路线模型上。"""
        truck = self._require_truck()
        route_nodes = [stop.node_id for stop in self._planned_route_stops]
        route_positions = [_clone_position(stop.position) for stop in self._planned_route_stops]
        geometry: list[Position3D] = []
        for segment in self._planned_route_segments:
            if not geometry:
                geometry.extend(_clone_position(pos) for pos in segment.geometry)
                continue
            if geometry[-1].distance_2d(segment.geometry[0]) <= 0.5:
                geometry.extend(_clone_position(pos) for pos in segment.geometry[1:])
            else:
                geometry.extend(_clone_position(pos) for pos in segment.geometry)
        truck.set_route(
            route_nodes=route_nodes,
            route_positions=route_positions,
            departure_time=0.0,
            geometry=geometry if len(geometry) >= 2 else None,
        )

    def _require_road_graph(self) -> tuple[Any, Mapping[str, tuple[float, float]]]:
        """按需加载场景 OSM 路网图，供运行时真实路径计算复用。"""
        if self._road_graph is not None and self._road_nodes is not None:
            return self._road_graph, self._road_nodes

        xml_path = self._scene_ctx.road_network.xml_path
        if not xml_path:
            raise ValueError("scene_ctx 缺少 osm_network.xml 路径，无法构造真实路网几何")
        osm_xml = Path(xml_path).read_text(encoding="utf-8")
        road_graph, road_nodes = build_road_graph(osm_xml, respect_osm_oneway=True)
        self._road_graph = road_graph
        self._road_nodes = road_nodes
        return road_graph, road_nodes

    def _initialize_drone_states(self) -> None:
        """根据 drone.home_type 建立训练侧初始状态。"""
        entity_mgr = self._require_entity_manager()
        truck = self._require_truck()
        for drone_id, drone in entity_mgr.drones.items():
            if drone.home_type == SourceType.TRUCK:
                self._drone_state[drone_id] = TrainingDroneState.RIDING_WITH_TRUCK
                drone.current_loc = _clone_position(truck.current_loc)
            else:
                self._drone_state[drone_id] = TrainingDroneState.IDLE
                drone.current_loc = _clone_position(self._require_depot().location)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _mark_hard_failure(
        self,
        drone_id: str,
    ) -> float:
        """把 UAV 标记为硬失败，并返回一次性惩罚。"""
        drone = self._require_entity_manager().drones[drone_id]
        truck = self._require_truck()

        self._flight_legs.pop(drone_id, None)
        self._delivery_service_legs.pop(drone_id, None)
        self._fallback_leg.pop(drone_id, None)
        self._mode_b_pending_depot_return.pop(drone_id, None)
        self._truck_charge_until.pop(drone_id, None)
        truck.serving_drones.pop(drone_id, None)
        truck.wait_queue = [item for item in truck.wait_queue if item != drone_id]
        self._active_wait_resume.pop(drone_id, None)
        self._release_reservation(drone_id, cause=FALLBACK_CAUSE_HARD_FAILURE_FALLBACK)
        self._decision_queue = [
            item for item in self._decision_queue if item.drone_id != drone_id
        ]
        if drone_id in truck.docked_drones:
            truck.docked_drones.remove(drone_id)
        drone.battery_current = 0.0
        self._drone_state[drone_id] = TrainingDroneState.AIRBORNE_ENERGY_FAILURE
        self._hard_failure_event_times.append(float(self._t_now))
        self._episode_hard_failure_count += 1
        return -self._cfg.hard_failure_penalty_sec

    def _episode_done_reason(self) -> str | None:
        if (
            self._enforces_upper_horizon()
            and self._t_now >= self._cfg.upper_horizon_sec - _TIME_EPS
        ):
            return "upper_horizon_reached"
        if (
            self._order_manager is not None
            and not self._order_manager.pending_orders
            and not self._order_manager.assigned_orders
            and not self._background_mode_a_pending
            and not self._has_future_order_arrivals()
            and not self._has_active_mode_c_recovery_obligation()
        ):
            return "all_orders_cleared"
        if self._drone_state and all(
            state == TrainingDroneState.AIRBORNE_ENERGY_FAILURE
            for state in self._drone_state.values()
        ):
            return "all_drones_hard_failed"
        return None

    def _compute_airborne_failure_time(
        self,
        drone_id: str,
        leg: FlightLeg,
    ) -> float | None:
        """按当前飞行账本估算"在到达前耗尽电量"的最早时刻。"""
        drone = self._require_entity_manager().drones[drone_id]
        if leg.energy_cost_j <= drone.battery_current + _TIME_EPS:
            return None
        if leg.arrival_time <= leg.start_time + _TIME_EPS:
            return self._t_now + _TIME_EPS

        failure_ratio = max(0.0, min(1.0, drone.battery_current / leg.energy_cost_j))
        failure_time = leg.start_time + (leg.arrival_time - leg.start_time) * failure_ratio
        # 若外部测试/调试在飞行中途手动改了电量，failure_time 可能已经落在当前时刻之前；
        # 此时把失败事件钳到"当前推进区间的下一个瞬间"。
        return max(self._t_now + _TIME_EPS, failure_time)

    def _estimate_flight_time(
        self,
        *,
        drone: Drone,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> float:
        """按训练侧 UAV 路径服务估算飞行时长。"""
        return self._uav_path_service.estimate(
            drone=drone,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=0.0,
        ).flight_time_sec

    def _estimate_energy_needed(
        self,
        *,
        drone: Drone,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
    ) -> float:
        """按训练侧 UAV 路径服务估算一段飞行能耗。"""
        return self._uav_path_service.estimate(
            drone=drone,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=payload,
        ).energy_j

    def _can_reach_from(
        self,
        *,
        drone: Drone,
        from_pos: Position3D,
        to_pos: Position3D,
        payload: float,
        safe_margin: float,
        battery_current: float,
    ) -> bool:
        """在给定起点和剩余电量假设下判断 UAV 是否可达。"""
        return self._uav_path_service.can_reach(
            drone=drone,
            from_pos=from_pos,
            to_pos=to_pos,
            payload=payload,
            safe_margin=safe_margin,
            battery_current=battery_current,
        )

    def _estimate_path_distance(
        self,
        *,
        from_pos: Position3D,
        to_pos: Position3D,
    ) -> float:
        """按训练侧 UAV 路径服务估算水平路径距离。"""
        return self._uav_path_service.path_distance(
            from_pos=from_pos,
            to_pos=to_pos,
        )

    def _all_return_hosts(self) -> tuple[ChargingHost, ...]:
        """返回当前实现里所有可能的固定返程宿主。"""
        entity_mgr = self._require_entity_manager()
        return tuple(entity_mgr.depots.values()) + tuple(entity_mgr.stations.values())

    def _resolve_host(self, node_id: str | None, node_type: str | None) -> ChargingHost:
        """按 `(node_id, node_type)` 解析固定宿主对象。"""
        if not node_id or not node_type:
            raise ValueError("host 节点信息不完整")
        entity_mgr = self._require_entity_manager()
        if node_type == "depot":
            return entity_mgr.depots[node_id]
        if node_type == "station":
            return entity_mgr.stations[node_id]
        raise ValueError(f"不支持的 host node_type: {node_type}")

    def _resolve_fixed_node(self, node_id: str) -> ChargingHost:
        """按固定节点 ID 解析 station/depot。"""
        entity_mgr = self._require_entity_manager()
        if node_id in entity_mgr.stations:
            return entity_mgr.stations[node_id]
        if node_id in entity_mgr.depots:
            return entity_mgr.depots[node_id]
        raise KeyError(f"未知固定节点: {node_id}")

    def _scene_solver_params(self):
        """按需读取共享 solver 参数。"""
        from config.loader import load_solver_energy_params

        return load_solver_energy_params()

    def _resolve_dispatch_launch_time(
        self,
        *,
        drone_id: str,
        trigger: DecisionTrigger,
    ) -> float:
        """返回本次 dispatch 的真实起飞时刻。"""
        if (
            trigger.trigger_type == "truck_station_arrival"
            and trigger.trigger_station_id
            and self._drone_state.get(drone_id) == TrainingDroneState.RIDING_WITH_TRUCK
        ):
            return (
                self._t_now
                + float(self._scene_solver_params().truck_drone_launch_time_s)
            )
        return float(self._t_now)

    def _estimate_delivery_launch_time(self, drone_id: str) -> float:
        """估算当前状态下若立即 dispatch，其真实起飞时刻。"""
        if self._drone_state.get(drone_id) == TrainingDroneState.RIDING_WITH_TRUCK:
            return (
                self._t_now
                + float(self._scene_solver_params().truck_drone_launch_time_s)
            )
        return float(self._t_now)

    def _truck_position_at_time(self, t_now: float) -> Position3D:
        """按 planned stop 的 arrival/departure 窗口推演 truck 位置。"""
        if not self._planned_route_stops:
            return _clone_position(self._require_truck().current_loc)

        first_stop = self._planned_route_stops[0]
        if t_now <= first_stop.departure_time + _TIME_EPS:
            return _clone_position(first_stop.position)

        for idx, segment in enumerate(self._planned_route_segments):
            if t_now < segment.start_time - _TIME_EPS:
                return _clone_position(self._planned_route_stops[idx].position)
            if t_now < segment.end_time - _TIME_EPS:
                segment_dt = max(_TIME_EPS, segment.end_time - segment.start_time)
                ratio = max(
                    0.0,
                    min(1.0, (t_now - segment.start_time) / segment_dt),
                )
                traveled_m = float(segment.distance_m) * ratio
                return _interpolate_position_on_geometry(
                    geometry=segment.geometry,
                    cumulative_distances_m=segment.cumulative_distances_m,
                    traveled_m=traveled_m,
                )
            next_stop = self._planned_route_stops[idx + 1]
            if t_now <= next_stop.departure_time + _TIME_EPS:
                return _clone_position(next_stop.position)

        return _clone_position(self._planned_route_stops[-1].position)

    def _active_truck_route_context_for_position(
        self,
        from_pos: Position3D,
    ) -> ActiveTruckRouteContext | None:
        """若 from_pos 是当前卡车行驶中位置，返回其下游 OSM 路径上下文。"""
        cache_key = (int(self._truck_route_version), round(float(self._t_now), 6))
        cached = self._active_truck_route_context_cache
        if (
            cached is not None
            and cached[0] == cache_key[0]
            and cached[1] == cache_key[1]
        ):
            context = cached[2]
        else:
            context = self._active_truck_route_context_at_time(self._t_now)
            self._active_truck_route_context_cache = (
                cache_key[0],
                cache_key[1],
                context,
            )
        if context is None:
            return None
        if context.position.distance_2d(from_pos) > 1.0:
            return None
        return context

    def _active_truck_route_context_at_time(
        self,
        t_now: float,
    ) -> ActiveTruckRouteContext | None:
        if not self._planned_route_stops:
            return None

        first_stop = self._planned_route_stops[0]
        if t_now <= first_stop.departure_time + _TIME_EPS:
            return None

        for idx, segment in enumerate(self._planned_route_segments):
            if t_now < segment.start_time - _TIME_EPS:
                return None
            if t_now < segment.end_time - _TIME_EPS:
                if not segment.osm_node_path:
                    return None
                segment_dt = max(_TIME_EPS, segment.end_time - segment.start_time)
                ratio = max(
                    0.0,
                    min(1.0, (t_now - segment.start_time) / segment_dt),
                )
                traveled_m = float(segment.distance_m) * ratio
                position = _interpolate_position_on_geometry(
                    geometry=segment.geometry,
                    cumulative_distances_m=segment.cumulative_distances_m,
                    traveled_m=traveled_m,
                )
                _, road_nodes = self._require_road_graph()
                remaining_path = _remaining_osm_node_path_from_distance(
                    segment=segment,
                    traveled_m=traveled_m,
                    road_nodes=road_nodes,
                )
                if not remaining_path:
                    return None
                return ActiveTruckRouteContext(
                    position=position,
                    segment_id=int(segment.segment_id),
                    traveled_m=float(traveled_m),
                    remaining_osm_node_path=remaining_path,
                )
            next_stop = self._planned_route_stops[idx + 1]
            if t_now <= next_stop.departure_time + _TIME_EPS:
                return None

        return None

    def _truck_traveled_distance_m(self, t_now: float) -> float:
        """按 planned route segments 统计当前时刻已真实行驶的路网距离。"""
        traveled = 0.0
        for segment in self._planned_route_segments:
            if t_now <= segment.start_time + _TIME_EPS:
                break
            if t_now >= segment.end_time - _TIME_EPS:
                traveled += max(0.0, float(segment.distance_m))
                continue
            segment_dt = max(_TIME_EPS, float(segment.end_time) - float(segment.start_time))
            ratio = max(
                0.0,
                min(1.0, (float(t_now) - float(segment.start_time)) / segment_dt),
            )
            traveled += max(0.0, float(segment.distance_m)) * ratio
            break
        return float(max(0.0, traveled))

    def _resolve_heavy_payload_capacity(self) -> float:
        """读取 heavy drone 的载重上限，供 coarse plan 做 mode A 边界判断。"""
        from config.loader import load_drone_params

        return float(load_drone_params().heavy.payload_capacity)

    def _resolve_recovery_pool_drone_cruise_speed(self) -> float:
        """读取 recovery-pool 粗筛评分使用的轻型无人机巡航速度。"""
        from config.loader import load_drone_params

        return float(load_drone_params().light.cruise_speed)

    def _is_action_allowed(
        self,
        action: EnvAction,
        action_lookup: tuple[EnvAction, ...],
    ) -> bool:
        """检查给定动作是否出现在当前候选动作列表中。"""
        return action in action_lookup

    def _require_entity_manager(self) -> EntityManager:
        """确保 entity_manager 已在 reset() 后初始化。"""
        if self._entity_manager is None:
            raise RuntimeError("reset() 前不能访问 entity_manager")
        return self._entity_manager

    def _require_order_manager(self) -> OrderManager:
        """确保 order_manager 已在 reset() 后初始化。"""
        if self._order_manager is None:
            raise RuntimeError("reset() 前不能访问 order_manager")
        return self._order_manager

    def _require_truck(self) -> Truck:
        """确保 truck 已在 reset() 后解析完成。"""
        if self._truck is None:
            raise RuntimeError("reset() 前不能访问 truck")
        return self._truck

    def _require_depot(self) -> Depot:
        """确保 depot 已在 reset() 后解析完成。"""
        if self._depot is None:
            raise RuntimeError("reset() 前不能访问 depot")
        return self._depot


def _load_env_yaml(config_path: Path) -> _YamlConfig:
    """读取训练 YAML，并抽取 env_adapter 当前实际使用的字段。"""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取训练配置") from exc

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML 顶层必须为 mapping: {config_path}")

    planner = _require_mapping(raw, "planner")
    reward = _require_mapping(raw, "reward")
    candidate = _require_mapping(raw, "candidate")
    reservation = _require_mapping(raw, "reservation")
    max_candidate_recovery_per_order = int(
        candidate["max_candidate_recovery_per_order"]
    )
    recovery_pool_future_scan_limit = int(
        candidate.get(
            "recovery_pool_future_scan_limit",
            max_candidate_recovery_per_order,
        )
    )
    if max_candidate_recovery_per_order <= 0:
        raise ValueError("candidate.max_candidate_recovery_per_order 必须为正数")
    if recovery_pool_future_scan_limit < max_candidate_recovery_per_order:
        raise ValueError(
            "candidate.recovery_pool_future_scan_limit 不能小于 "
            "max_candidate_recovery_per_order"
        )

    cfg = _YamlConfig(
        upper_horizon_sec=float(planner["upper_horizon_sec"]),
        patrol_min_remaining_sec=float(planner["patrol_min_remaining_sec"]),
        patrol_stations_per_loop=int(planner["patrol_stations_per_loop"]),
        allow_empty_backbone_route=bool(planner.get("allow_empty_backbone_route", False)),
        support_radius_km=float(planner["support_radius_km"]),
        fallback_burst_window_sec=float(planner["fallback_burst_window_sec"]),
        coarse_new_order_trigger=int(planner["coarse_new_order_trigger"]),
        max_wait_decision_gap_sec=float(planner["max_wait_decision_gap_sec"]),
        max_candidate_recovery_per_order=max_candidate_recovery_per_order,
        recovery_pool_future_scan_limit=recovery_pool_future_scan_limit,
        rendezvous_filter_margin_sec=float(
            candidate.get(
                "rendezvous_filter_margin_sec",
                candidate.get("rendezvous_eta_safe_margin_sec"),
            )
        ),
        rendezvous_execution_margin_sec=float(
            candidate.get(
                "rendezvous_execution_margin_sec",
                candidate.get(
                    "rendezvous_filter_margin_sec",
                    candidate.get("rendezvous_eta_safe_margin_sec"),
                ),
            )
        ),
        rendezvous_max_wait_sec=float(
            candidate.get(
                "rendezvous_max_wait_sec",
                candidate["station_wait_threshold_sec"],
            )
        ),
        reservation_enabled=bool(reservation.get("enable", True)),
        reservation_alpha=float(reservation["alpha"]),
        reservation_beta=float(reservation["beta"]),
        reservation_gamma=float(reservation["gamma"]),
        reservation_drift_eta_abs_threshold_sec=float(
            reservation["drift_eta_abs_threshold_sec"]
        ),
        wait_idle_penalty_coef=float(reward["wait_idle_penalty_coef"]),
        wait_opportunity_penalty_coef=float(
            reward.get("wait_opportunity_penalty_coef", 0.0)
        ),
        lambda_miss=float(reward["lambda_miss"]),
        lambda_res_timeout=float(reward["lambda_res_timeout"]),
        lambda_overdue=float(reward.get("lambda_overdue", 0.0)),
        R_delivery_bonus=float(reward["R_delivery_bonus"]),
        late_delivery_penalty_coef=float(reward.get("late_delivery_penalty_coef", 0.0)),
        late_delivery_penalty_cap=float(reward.get("late_delivery_penalty_cap", 0.0)),
        min_late_delivery_reward=float(
            reward.get("min_late_delivery_reward", 1.0)
        ),
        mode_c_attempt_bonus=float(reward.get("mode_c_attempt_bonus", 0.0)),
        uav_energy_penalty_coef=float(
            reward.get("uav_energy_penalty_coef", 0.0)
        ),
        uav_energy_penalty_cap_ratio=float(
            reward.get("uav_energy_penalty_cap_ratio", 0.0)
        ),
        max_overdue_sec=float(reward["max_overdue_sec"]),
        hard_overdue_penalty_sec=float(reward.get("hard_overdue_penalty_sec", 0.0)),
        hard_failure_penalty_sec=float(reward["hard_failure_penalty_sec"]),
    )
    if cfg.rendezvous_execution_margin_sec < 0.0:
        raise ValueError("candidate.rendezvous_execution_margin_sec 不能为负数")
    if cfg.rendezvous_max_wait_sec < cfg.rendezvous_execution_margin_sec:
        raise ValueError(
            "candidate.rendezvous_max_wait_sec 不能小于 "
            "rendezvous_execution_margin_sec"
        )
    if cfg.late_delivery_penalty_coef < 0.0:
        raise ValueError("reward.late_delivery_penalty_coef 不能为负数")
    if cfg.late_delivery_penalty_cap < 0.0:
        raise ValueError("reward.late_delivery_penalty_cap 不能为负数")
    if cfg.min_late_delivery_reward < 0.0:
        raise ValueError("reward.min_late_delivery_reward 不能为负数")
    if cfg.min_late_delivery_reward > cfg.R_delivery_bonus:
        raise ValueError("reward.min_late_delivery_reward 不能大于 R_delivery_bonus")
    if cfg.uav_energy_penalty_coef < 0.0:
        raise ValueError("reward.uav_energy_penalty_coef 不能为负数")
    if cfg.uav_energy_penalty_cap_ratio < 0.0:
        raise ValueError("reward.uav_energy_penalty_cap_ratio 不能为负数")
    return cfg


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件，并断言顶层对象类型。"""
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象: {path}")
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """从字典中取出指定 mapping 字段，否则抛错。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置缺少 mapping 段: {key}")
    return value


def _require_singleton(mapping: Mapping[str, Any], label: str) -> tuple[str, Any]:
    """断言某类实体当前只有一个，并返回该唯一元素。"""
    if len(mapping) != 1:
        raise ValueError(f"当前仅支持单 {label} 场景，实际数量={len(mapping)}")
    return next(iter(mapping.items()))


def _clone_position(pos: Position3D) -> Position3D:
    """复制 Position3D，避免直接复用实体上的同一对象引用。"""
    return Position3D(x=float(pos.x), y=float(pos.y), z=float(pos.z))


def _position_to_payload(pos: Position3D) -> dict[str, float]:
    return {
        "x": float(pos.x),
        "y": float(pos.y),
        "z": float(pos.z),
    }


def _build_cumulative_distances(geometry: tuple[Position3D, ...]) -> tuple[float, ...]:
    cumulative = [0.0]
    for idx in range(1, len(geometry)):
        cumulative.append(cumulative[-1] + geometry[idx - 1].distance_2d(geometry[idx]))
    return tuple(cumulative)


def _polyline_distance_2d(points: tuple[Position3D, ...] | list[Position3D]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(points[idx - 1].distance_2d(points[idx]) for idx in range(1, len(points)))


def _interpolate_position_on_geometry(
    *,
    geometry: tuple[Position3D, ...],
    cumulative_distances_m: tuple[float, ...],
    traveled_m: float,
) -> Position3D:
    if not geometry:
        raise ValueError("geometry 不能为空")
    if len(geometry) == 1:
        return _clone_position(geometry[0])

    clamped = max(0.0, min(float(traveled_m), float(cumulative_distances_m[-1])))
    for idx in range(1, len(geometry)):
        seg_end = cumulative_distances_m[idx]
        if clamped > seg_end + _TIME_EPS:
            continue
        seg_start = cumulative_distances_m[idx - 1]
        seg_length = max(_TIME_EPS, seg_end - seg_start)
        ratio = max(0.0, min(1.0, (clamped - seg_start) / seg_length))
        return geometry[idx - 1].interpolate(geometry[idx], ratio)
    return _clone_position(geometry[-1])


def _remaining_geometry_from_distance(
    *,
    geometry: tuple[Position3D, ...],
    cumulative_distances_m: tuple[float, ...],
    traveled_m: float,
) -> tuple[Position3D, ...]:
    if not geometry:
        return ()
    if len(geometry) == 1:
        return (_clone_position(geometry[0]),)

    clamped = max(0.0, min(float(traveled_m), float(cumulative_distances_m[-1])))
    if clamped <= _TIME_EPS:
        return tuple(_clone_position(pos) for pos in geometry)
    if clamped >= cumulative_distances_m[-1] - _TIME_EPS:
        return (_clone_position(geometry[-1]),)

    current_pos = _interpolate_position_on_geometry(
        geometry=geometry,
        cumulative_distances_m=cumulative_distances_m,
        traveled_m=clamped,
    )
    remaining: list[Position3D] = [_clone_position(current_pos)]
    for idx in range(1, len(geometry)):
        if cumulative_distances_m[idx] <= clamped + _TIME_EPS:
            continue
        if remaining[-1].distance_2d(geometry[idx]) > 0.5:
            remaining.append(_clone_position(geometry[idx]))
    if len(remaining) == 1:
        remaining.append(_clone_position(geometry[-1]))
    return tuple(remaining)


def _remaining_osm_node_path_from_distance(
    *,
    segment: PlannedTruckSegment,
    traveled_m: float,
    road_nodes: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    if not segment.osm_node_path:
        return ()

    remaining: list[str] = []
    threshold_m = max(0.0, float(traveled_m) - 0.5)
    for osm_node_id in segment.osm_node_path:
        node_pos = _road_node_position(road_nodes, str(osm_node_id))
        node_progress_m = _project_position_progress_on_geometry(
            geometry=segment.geometry,
            cumulative_distances_m=segment.cumulative_distances_m,
            position=node_pos,
        )
        if node_progress_m + 0.5 >= threshold_m:
            remaining.append(str(osm_node_id))

    if remaining:
        return tuple(remaining)
    return (str(segment.osm_node_path[-1]),)


def _project_position_progress_on_geometry(
    *,
    geometry: tuple[Position3D, ...],
    cumulative_distances_m: tuple[float, ...],
    position: Position3D,
) -> float:
    if len(geometry) <= 1:
        return 0.0
    if len(cumulative_distances_m) != len(geometry):
        cumulative_distances_m = _build_cumulative_distances(geometry)

    best_progress = 0.0
    best_distance_sq = float("inf")
    for idx in range(1, len(geometry)):
        start = geometry[idx - 1]
        end = geometry[idx]
        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= _TIME_EPS:
            ratio = 0.0
        else:
            ratio = (
                ((float(position.x) - float(start.x)) * dx)
                + ((float(position.y) - float(start.y)) * dy)
            ) / seg_len_sq
            ratio = max(0.0, min(1.0, ratio))
        proj_x = float(start.x) + dx * ratio
        proj_y = float(start.y) + dy * ratio
        dist_sq = (
            (float(position.x) - proj_x) ** 2
            + (float(position.y) - proj_y) ** 2
        )
        if dist_sq < best_distance_sq:
            best_distance_sq = dist_sq
            best_progress = float(cumulative_distances_m[idx - 1]) + (
                float(cumulative_distances_m[idx])
                - float(cumulative_distances_m[idx - 1])
            ) * ratio
    return best_progress


def _parse_segment_geometry_payload(raw_segment: Mapping[str, Any]) -> tuple[Position3D, ...] | None:
    raw_geometry = raw_segment.get("geometry")
    if raw_geometry is None:
        return None
    if not isinstance(raw_geometry, list):
        raise ValueError("segment.geometry 必须为 list")
    geometry: list[Position3D] = []
    for raw_point in raw_geometry:
        if not isinstance(raw_point, Mapping):
            raise ValueError("segment.geometry 点必须为对象")
        geometry.append(
            Position3D(
                x=float(raw_point["x"]),
                y=float(raw_point["y"]),
                z=float(raw_point.get("z", 0.0)),
            )
        )
    if len(geometry) < 2:
        raise ValueError("segment.geometry 至少需要 2 个点")
    return tuple(geometry)


def _build_geometry_from_osm_node_path(
    *,
    from_pos: Position3D,
    to_pos: Position3D,
    osm_node_path: list[str],
    road_nodes: Mapping[str, tuple[float, float]],
) -> tuple[Position3D, ...]:
    geometry: list[Position3D] = [_clone_position(from_pos)]
    for osm_node_id in osm_node_path:
        pos = _road_node_position(road_nodes, osm_node_id)
        if geometry[-1].distance_2d(pos) > 0.5:
            geometry.append(pos)
    if geometry[-1].distance_2d(to_pos) > 0.5:
        geometry.append(_clone_position(to_pos))
    return tuple(geometry)


def _clone_truck_road_route(route: TruckRoadRoute) -> TruckRoadRoute:
    return TruckRoadRoute(
        geometry=tuple(_clone_position(pos) for pos in route.geometry),
        osm_node_path=tuple(route.osm_node_path),
    )


def _build_route_between_positions(
    *,
    from_pos: Position3D,
    to_pos: Position3D,
    road_graph: Any,
    road_nodes: Mapping[str, tuple[float, float]],
    nearest_cache: dict[tuple[float, float], tuple[str, float]],
) -> TruckRoadRoute:
    from_node_id, _ = _find_nearest_node_cached(
        road_graph=road_graph,
        road_nodes=road_nodes,
        position=from_pos,
        cache=nearest_cache,
    )
    to_node_id, _ = _find_nearest_node_cached(
        road_graph=road_graph,
        road_nodes=road_nodes,
        position=to_pos,
        cache=nearest_cache,
    )
    if from_node_id == to_node_id:
        osm_path = [from_node_id]
    else:
        osm_path = shortest_path(road_graph, from_node_id, to_node_id)
        if not osm_path:
            raise ValueError(f"OSM 路网不可达: {from_node_id} -> {to_node_id}")
    normalized_path = tuple(str(node_id) for node_id in osm_path)
    return TruckRoadRoute(
        geometry=_build_geometry_from_osm_node_path(
            from_pos=from_pos,
            to_pos=to_pos,
            osm_node_path=list(normalized_path),
            road_nodes=road_nodes,
        ),
        osm_node_path=normalized_path,
    )


def _build_route_geometry_between_positions(
    *,
    from_pos: Position3D,
    to_pos: Position3D,
    road_graph: Any,
    road_nodes: Mapping[str, tuple[float, float]],
    nearest_cache: dict[tuple[float, float], tuple[str, float]],
) -> tuple[Position3D, ...]:
    return _build_route_between_positions(
        from_pos=from_pos,
        to_pos=to_pos,
        road_graph=road_graph,
        road_nodes=road_nodes,
        nearest_cache=nearest_cache,
    ).geometry


def _build_route_from_active_truck_context(
    *,
    context: ActiveTruckRouteContext,
    from_pos: Position3D,
    to_pos: Position3D,
    road_graph: Any,
    road_nodes: Mapping[str, tuple[float, float]],
) -> TruckRoadRoute | None:
    to_node_id, _ = _find_nearest_node_cached(
        road_graph=road_graph,
        road_nodes=road_nodes,
        position=to_pos,
        cache={},
    )
    remaining_path = tuple(str(node_id) for node_id in context.remaining_osm_node_path)
    if not remaining_path:
        return None

    for idx, start_node_id in enumerate(remaining_path):
        if start_node_id == to_node_id:
            suffix_path = [start_node_id]
        else:
            suffix_path = shortest_path(road_graph, start_node_id, to_node_id)
        if not suffix_path:
            continue
        prefix_path = remaining_path[: idx + 1]
        route_path = tuple(prefix_path) + tuple(
            str(node_id) for node_id in suffix_path[1:]
        )
        return TruckRoadRoute(
            geometry=_build_geometry_from_osm_node_path(
                from_pos=from_pos,
                to_pos=to_pos,
                osm_node_path=list(route_path),
                road_nodes=road_nodes,
            ),
            osm_node_path=route_path,
        )

    return None


def _find_nearest_node_cached(
    *,
    road_graph: Any,
    road_nodes: Mapping[str, tuple[float, float]],
    position: Position3D,
    cache: dict[tuple[float, float], tuple[str, float]],
) -> tuple[str, float]:
    key = (round(float(position.x), 2), round(float(position.y), 2))
    cached = cache.get(key)
    if cached is not None:
        return cached
    nearest_node_id = find_nearest_node(road_graph, road_nodes, position.x, position.y)
    if not nearest_node_id:
        raise ValueError(f"无法将坐标映射到 OSM 节点: ({position.x}, {position.y})")
    nearest_pos = _road_node_position(road_nodes, str(nearest_node_id))
    snapped = (str(nearest_node_id), position.distance_2d(nearest_pos))
    cache[key] = snapped
    return snapped


def _road_node_position(
    road_nodes: Mapping[str, tuple[float, float]],
    osm_node_id: str,
) -> Position3D:
    from pyproj import Transformer

    lon, lat = road_nodes[osm_node_id]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return Position3D(x=float(x), y=float(y), z=0.0)


def _host_id(host: ChargingHost) -> str:
    """把 depot/station 宿主对象映射回其节点 ID。"""
    if isinstance(host, Depot):
        return host.depot_id
    if isinstance(host, SwapStation):
        return host.station_id
    raise TypeError(f"未知 host 类型: {type(host)!r}")


def _node_type_of_host(host: ChargingHost) -> str:
    """把 depot/station 宿主对象映射回当前环境使用的节点类型字符串。"""
    if isinstance(host, Depot):
        return "depot"
    if isinstance(host, SwapStation):
        return "station"
    raise TypeError(f"未知 host 类型: {type(host)!r}")


def _mode_c_state_label(state: TrainingDroneState | None) -> str | None:
    if state == TrainingDroneState.DELIVERED:
        return "delivered"
    if state == TrainingDroneState.RETURN_TO_RENDEZVOUS:
        return "return_to_rendezvous"
    if state == TrainingDroneState.WAITING_FOR_TRUCK:
        return "waiting_for_truck"
    return None


def _merge_reward_breakdown(target: dict[str, float], source: Mapping[str, float]) -> None:
    """把一段 reward breakdown 累加合并到目标字典。"""
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _merge_int_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    """把结构化诊断计数字典累加到 episode 级别。"""
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)
