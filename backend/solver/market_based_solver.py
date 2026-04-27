#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 市场拍卖调度算法（支持动态增量调度与时空契约）

设计原则：
    1. 复用 GreedyMMCE 中已经稳定的能耗模型、卡车路径构建与 DispatchPlan 契约
  2. 将逐单"直接选模式"替换为"单任务顺序拍卖"
  3. 由协调器统一收集无人机 / 卡车投标，按综合成本最低授标
  4. 中标后即刻生成 RendezvousContract，锁定卡车锚点与无人机资源
  5. 支持 dispatch_incremental 实现事件驱动的动态新单接入

当前实现覆盖三类投标：
  - A: 卡车直递（卡车作为兜底竞标者）
  - B_WAIT: 基于卡车锚点时刻表的站点起飞 / 站点回收
  - C: 仓库直发 / 仓库回收

动态协议：
  - 每次 B_WAIT 中标，生成 RendezvousContract（不可移动的硬约束）
  - 卡车路径包含"自由段"（可插单）和"锁定段"（必须按时到达锚点）
  - 新单拍卖时，卡车必须校验插单不会导致任何已锁定契约超时
  - 评分函数引入非对称等待惩罚（机等车低权重，车等机高权重）
  - 接近生死线的锚点被选中时附加风险惩罚分

工程说明（与实现同步维护）：solver/market_based.md

订单池解耦：需读取 `assigned_orders` / `pending_orders` 的约束逻辑使用本类 `bind_order_manager(order_mgr)`
（由 `DispatchDecisionEngine` 注入），**不**再依赖 `EntityManager.order_mgr`。
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.entities.primitives import DroneStatus, SourceType
from solver.greedy_mmce import (
    AllocationResult,
    DispatchPlan,
    GreedyMMCE,
    _osm_svc,
    load_osm_from_cache,
)

if TYPE_CHECKING:
    from core.entities.drone import Drone
    from core.entities.order import Order
    from core.entities.primitives import Position3D
    from core.entities.truck import Truck

logger = logging.getLogger(__name__)
_ENTITY_MANAGER_LOGGER = logging.getLogger("entity_manager")
_DECISION_ENGINE_LOGGER = logging.getLogger("solver.decision_engine")
_MARKET_DECISION_ENGINE_LOGGER = logging.getLogger("solver.decision_engine_market")

# 与本模块在终端上的 logging 输出一致，追加写入 solver 目录下的 market_debug_log
_MARKET_DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_debug_log")


# ══════════════════════════════════════════════════════════════════════════════
# 自动flush的FileHandler
# ══════════════════════════════════════════════════════════════════════════════

class _AutoFlushFileHandler(logging.FileHandler):
    """自定义FileHandler：每条日志立即flush到磁盘，避免程序pause时日志仅在缓冲区"""
    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)


def _attach_market_debug_file_handler(target_logger: logging.Logger) -> None:
    abs_target = os.path.abspath(_MARKET_DEBUG_LOG_PATH)
    for h in target_logger.handlers:
        if isinstance(h, (logging.FileHandler, _AutoFlushFileHandler)):
            try:
                if os.path.abspath(h.baseFilename) == abs_target:
                    return
            except (AttributeError, OSError, ValueError):
                continue
    try:
        fh = _AutoFlushFileHandler(_MARKET_DEBUG_LOG_PATH, mode="a", encoding="utf-8")
    except OSError:
        return
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    fh.setLevel(logging.DEBUG)
    target_logger.addHandler(fh)
    # 同时确保logger本身的级别足够低，不会过滤掉DEBUG日志
    if target_logger.level == logging.NOTSET or target_logger.level > logging.DEBUG:
        target_logger.setLevel(logging.DEBUG)


def _ensure_market_debug_file_handler() -> None:
    # market solver：竞价/授标；solver.decision_engine：下发无人机路由等编排日志；
    # entity_manager：配送点停留与完成归档。
    _attach_market_debug_file_handler(logger)
    _attach_market_debug_file_handler(_DECISION_ENGINE_LOGGER)
    _attach_market_debug_file_handler(_MARKET_DECISION_ENGINE_LOGGER)
    _attach_market_debug_file_handler(_ENTITY_MANAGER_LOGGER)


_ensure_market_debug_file_handler()


# ══════════════════════════════════════════════════════════════════════════════
# 时空契约
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RendezvousContract:
    """中标后生成的时空约束。

    当一个 B_WAIT 订单中标，系统生成 RendezvousContract 锁定：
      - 卡车必须在 latest_departure 之前到达指定回收锚点
      - 关联无人机在 t_sync 之前不可被重新分配

    status 生命周期：
      active   → fulfilled （回收完成）
      active   → expired   （超过 latest_departure 仍未兑现）
      active   → released  （弹性归巢：无人机自主返回仓库，释放卡车锚点锁定）
    """
    contract_id: str
    truck_id: str
    drone_id: str
    order_id: str
    anchor_id: str
    launch_anchor_id: str = ""
    arrival_time: float = 0.0
    latest_departure: float = 0.0
    uav_arrival_time: float = 0.0
    t_sync: float = 0.0
    status: str = "active"


# ══════════════════════════════════════════════════════════════════════════════
# 市场拍卖调度器
# ══════════════════════════════════════════════════════════════════════════════

class MarketBasedSolver(GreedyMMCE):
    """基于锚点时刻表的单任务顺序拍卖调度器，支持动态增量拍卖与时空契约。"""

    T_MAX_WAIT = 300.0

    # 非对称等待惩罚权重（已校正：缩小机等车/车等机的极端不对称性）
    OMEGA_UAV_IDLE = 0.5         # 机等车：含机会成本（无法投标新单）
    OMEGA_TRUCK_IDLE = 1.0       # 车等机：阻塞卡车后续配送
    # 机等车的额外机会成本（每秒等待损失的竞标潜力）
    OPPORTUNITY_COST_PER_SEC = 0.1
    # 时刻表缓冲比例（应对路网拥堵等不确定性）
    TIMETABLE_SLACK_RATIO = 0.10
    # 接近生死线的风险惩罚权重（已校正：降低以减少保守退化为卡车直递）
    DEADLINE_RISK_WEIGHT = 0.8

    # ── 竞标准入约束参数 ──────────────────────────────────────────────
    AUCTION_BUFFER_TIME = 15.0     # DELIVERING 无人机距终点 T_remain 小于此值时允许预调度
    PATH_STITCH_TOLERANCE_M = 1.0  # 路径拼接坐标容差（米）
    AUCTION_COMPUTE_DELTA_T = 0.5  # 拍卖计算预估耗时（用于起点预测）
    MODE_C_DEPOT_TOLERANCE_M = 25.0  # Mode C 起飞时无人机与仓库位置容差

    # ── 路径方向性与多维优化参数 ────────────────────────────────────
    BACKTRACK_PENALTY_WEIGHT = 2.0       # 回头路惩罚权重（方向角越偏离前进方向，惩罚越高）
    DETOUR_ENERGY_MULTIPLIER = 1.5       # 绕路能耗放大系数（反映频繁变向的额外损耗）
    PATH_PROXIMITY_BONUS = 0.3           # 路径邻近度奖励权重（订单在卡车路径附近时减分）
    TWO_OPT_MAX_ITERATIONS = 50          # 2-opt 局部搜索最大迭代次数

    # 单车市场场景：驶出仓时车载 9 架、仓内留 3 架（与 UAV-TEST-xx 编号尾数对应，在此由求解器首调度前写回实体）
    MARKET_TRUCK_DRONE_INDEXES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 11})
    MARKET_DEPOT_DRONE_INDEXES = frozenset({9, 10, 12})

    # 在 `solver_energy.lambda_time` 基础上再放大时间窗迟罚，仅作用于 market 竞价/基线分中的 `cost_penalty`
    TIME_PENALTY_WEIGHT_MULT = 1.5

    def __init__(self, entity_mgr) -> None:
        super().__init__(entity_mgr)
        self.LAMBDA_TIME = self.LAMBDA_TIME * self.TIME_PENALTY_WEIGHT_MULT
        self._auction_bid_count = 0
        self._auction_award_count = 0
        self._anchor_preview_routes: dict[str, object] = {}
        self._active_contracts: list[RendezvousContract] = []
        self._contract_seq = 0
        self._current_round_bdynamic_launches: list[tuple[str, str, float]] = []
        # 订单视图仅用于市场约束（不可撤销、清空校验等），由 DispatchDecisionEngine 注入，避免挂在 EntityManager 上。
        self._order_mgr: Any = None
        self._market_loadout_applied: bool = False

    def bind_order_manager(self, order_mgr: Any) -> None:
        """由编排层在构造/切换求解器后调用，供增量过滤与车队安全校验读取订单池。"""
        self._order_mgr = order_mgr

    @staticmethod
    def _extract_market_drone_id_suffix(drone_id: str) -> int | None:
        m = re.search(r"(\d+)$", str(drone_id))
        if m is None:
            return None
        return int(m.group(1))

    def _resolve_market_fixed_drone_pools(self) -> tuple[str, str, set[str], set[str]] | None:
        """单仓单车且存在 12 架 UAV-TEST-xx 时，解析 9+3 无人机池，否则不生效。"""
        if len(self.entity_mgr.trucks) != 1 or not self.entity_mgr.depots:
            return None
        truck_id = next(iter(self.entity_mgr.trucks.keys()))
        depot_id = next(iter(self.entity_mgr.depots.keys()))
        truck_pool: set[str] = set()
        depot_pool: set[str] = set()
        for drone_id in self.entity_mgr.drones:
            idx = self._extract_market_drone_id_suffix(drone_id)
            if idx is None:
                continue
            if idx in self.MARKET_TRUCK_DRONE_INDEXES:
                truck_pool.add(drone_id)
            elif idx in self.MARKET_DEPOT_DRONE_INDEXES:
                depot_pool.add(drone_id)
        if len(truck_pool) < len(self.MARKET_TRUCK_DRONE_INDEXES):
            return None
        if len(depot_pool) < len(self.MARKET_DEPOT_DRONE_INDEXES):
            return None
        return truck_id, depot_id, truck_pool, depot_pool

    def _sync_depot_registrations_after_market_loadout(
        self,
        depot: Any,
        truck_pool: set[str],
        depot_pool: set[str],
    ) -> None:
        """与 entities 默认「多数机在仓」的登记一致：上车无人机从仓资产表中移除，留仓机保留/登记。"""
        for did in list(depot.drone_fleet):
            if did in truck_pool:
                if did in depot.idle_drones:
                    depot.idle_drones.remove(did)
                depot.drone_fleet.remove(did)
        for did in depot_pool:
            depot.register_drone(did, is_idle=True)

    def _apply_market_initial_drone_loadout(self, current_time: float) -> None:
        """在首次全量/增量市场调度前，将 9+3 归属与 `docked_drones` / `parking_slots` 写回实体，不修改 entities 文件。"""
        if self._market_loadout_applied:
            return
        resolved = self._resolve_market_fixed_drone_pools()
        if resolved is None:
            self._market_loadout_applied = True
            logger.debug(
                "[MarketBasedSolver] 9/3 固装载不生效：非单车/单仓或缺少编号 UAV-TEST-01..12"
            )
            return
        truck_id, depot_id, truck_pool, depot_pool = resolved
        truck = self.entity_mgr.trucks.get(truck_id)
        depot = self.entity_mgr.depots.get(depot_id)
        if truck is None or depot is None:
            self._market_loadout_applied = True
            return
        if int(getattr(truck, "parking_slots", 0)) < len(truck_pool):
            truck.parking_slots = len(truck_pool)
        truck.docked_drones = sorted(truck_pool)
        for other_id, other_truck in self.entity_mgr.trucks.items():
            if other_id == truck_id:
                continue
            other_truck.docked_drones = [
                d for d in other_truck.docked_drones if d not in truck_pool
            ]
        for drone_id, drone in self.entity_mgr.drones.items():
            if drone_id in truck_pool:
                drone.home_type = SourceType.TRUCK
                drone.home_id = truck_id
                drone.transport_truck_id = truck_id
                if not drone.status.is_flying:
                    drone.current_loc = truck.get_location(current_time)
            elif drone_id in depot_pool:
                drone.home_type = SourceType.DEPOT
                drone.home_id = depot_id
                drone.transport_truck_id = None
                if not drone.status.is_flying:
                    drone.current_loc = depot.location
        self._sync_depot_registrations_after_market_loadout(depot, truck_pool, depot_pool)
        self._market_loadout_applied = True
        logger.info(
            "[MarketBasedSolver] 已应用市场 9/3 固装载: truck=%s (%d) depot=%s (%d) parking_slots=%d",
            truck_id,
            len(truck_pool),
            depot_id,
            len(depot_pool),
            getattr(truck, "parking_slots", 0),
        )

    def _is_order_immutable(self, order_id: str, current_time: float = 0.0) -> bool:
        """判断订单是否已进入不可撤销状态（履约临界点）。

        基于地理围栏与物理状态：若持有该订单的实体真实位于客户附近（卸货中）
        或处于飞行中、状态不为 PENDING，均视为已锁定。
        """
        from core.entities.primitives import TaskStatus
        if not self._order_mgr:
            return False

        order = self._order_mgr.assigned_orders.get(order_id)
        if order is None:
            return False
            
        if order.status in (
            TaskStatus.PICKED_UP,
            TaskStatus.DELIVERING,
            TaskStatus.COMPLETED,
        ):
            return True
            
        # 地理围栏补充校验：即使状态未及时同步，若物理上已被携带且到达目的地附近，禁止撤销
        sole_executor = self._get_order_sole_executor(order_id)
        if sole_executor:
            drone = self.entity_mgr.drones.get(sole_executor)
            if drone:
                # 若无人机正持有此单且服务时间倒计时已启动，或正在飞行，或抵达卸货区
                if getattr(drone, "delivery_service_end_time", 0) > current_time:
                    return True
                if drone.status.is_flying:
                    return True
                dist_to_customer = self._dist(drone.get_location(current_time), order.delivery_loc)
                if dist_to_customer < 10.0:  # Geofencing
                    return True
                    
            truck = self.entity_mgr.trucks.get(sole_executor)
            if truck:
                dist_to_customer = self._dist(truck.get_location(current_time), order.delivery_loc)
                if dist_to_customer < 10.0:  # Geofencing
                    return True

        return False

    def _get_order_sole_executor(self, order_id: str) -> str | None:
        """获取订单的唯一履约实体 ID（carrying_order_id 持有者）。

        系统保证：只有当前持有该订单 carrying_order_id 的实体才有权触发
        COMPLETED 归档函数，杜绝"UAV-A 送了货，但后端认为订单还在 UAV-B 身上"。
        """
        for drone in self.entity_mgr.drones.values():
            if getattr(drone, "carrying_order_id", None) == order_id:
                return drone.drone_id
        return None

    def _revoke_stale_drone_assignments(self, plan: DispatchPlan, current_time: float) -> None:
        """本计划将订单授予 winner 无人机时，清理其他无人机上同订单的挂载/卸货计时/航路。

        不可撤销约束：若订单已处于 PICKED_UP / DELIVERING 状态，则跳过撤销，
        保持原执行实体的绑定关系不变。
        """
        winners: list[tuple[str, str]] = [
            (a.order_id, a.drone_id)
            for a in plan.allocations
            if a.feasible and a.drone_id and a.mode != "A"
        ]
        if not winners:
            return

        for order_id, winner_id in winners:
            if self._is_order_immutable(order_id, current_time):
                sole_executor = self._get_order_sole_executor(order_id)
                if sole_executor and sole_executor != winner_id:
                    logger.warning(
                        "[MarketBasedSolver] 订单 %s 已处于履约状态，唯一执行者=%s，"
                        "拒绝改派给 %s（不可撤销约束）",
                        order_id, sole_executor, winner_id,
                    )
                    for a in plan.allocations:
                        if a.order_id == order_id and a.drone_id == winner_id:
                            a.feasible = False
                            a.reason = "订单已进入不可撤销状态，拒绝改派"
                continue

            for drone in self.entity_mgr.drones.values():
                if drone.drone_id == winner_id:
                    continue

                pend = getattr(drone, "pending_release_order_id", None) or ""
                carry = getattr(drone, "carrying_order_id", None) or ""
                idx = max(0, int(getattr(drone, "current_waypoint_index", 0)))
                route = getattr(drone, "route_plan", None) or []
                route_refs_order = any(
                    getattr(wp, "target_entity_id", None) == order_id
                    for wp in route[idx:]
                )

                if carry != order_id and pend != order_id and not route_refs_order:
                    continue

                # 保守保护：只清理“静止残留状态”。
                # 对于仍有活动航路/等待回收/服务停留中的无人机，
                # 禁止在增量拍卖中强制清路由，避免前端轨迹断裂与实体卡死。
                if getattr(drone, "delivery_service_end_time", 0.0) > current_time:
                    logger.warning(
                        "[MarketBasedSolver] 订单 %s 候选改派触发，但无人机 %s 仍在服务停留，"
                        "拒绝撤销其航路",
                        order_id,
                        drone.drone_id,
                    )
                    for a in plan.allocations:
                        if a.order_id == order_id and a.drone_id == winner_id:
                            a.feasible = False
                            a.reason = "原执行无人机仍在服务停留，拒绝改派"
                    break
                if getattr(drone, "waiting_recovery_station_id", ""):
                    logger.warning(
                        "[MarketBasedSolver] 订单 %s 候选改派触发，但无人机 %s 正在等待回收，"
                        "拒绝撤销其航路",
                        order_id,
                        drone.drone_id,
                    )
                    for a in plan.allocations:
                        if a.order_id == order_id and a.drone_id == winner_id:
                            a.feasible = False
                            a.reason = "原执行无人机等待回收中，拒绝改派"
                    break
                if bool(getattr(drone, "has_pending_route", False)) and carry == order_id:
                    logger.warning(
                        "[MarketBasedSolver] 订单 %s 候选改派触发，但无人机 %s 仍有待执行航路，"
                        "拒绝撤销其航路",
                        order_id,
                        drone.drone_id,
                    )
                    for a in plan.allocations:
                        if a.order_id == order_id and a.drone_id == winner_id:
                            a.feasible = False
                            a.reason = "原执行无人机仍有待执行航路，拒绝改派"
                    break

                # 再次检查：若该无人机正在物理执行该订单（飞行中），拒绝撤销
                if carry == order_id and drone.status.is_flying:
                    logger.warning(
                        "[MarketBasedSolver] 订单 %s 正由无人机 %s 飞行执行中，"
                        "拒绝改派给 %s（飞行中不可撤销）",
                        order_id, drone.drone_id, winner_id,
                    )
                    for a in plan.allocations:
                        if a.order_id == order_id and a.drone_id == winner_id:
                            a.feasible = False
                            a.reason = "订单正在飞行执行中，拒绝改派"
                    break

                logger.info(
                    "[MarketBasedSolver] 订单 %s 改由 %s 执行，撤销无人机 %s 上的同单残留状态",
                    order_id,
                    winner_id,
                    drone.drone_id,
                )

                if pend == order_id:
                    drone.pending_release_order_id = None
                    drone.delivery_service_end_time = 0.0

                if carry == order_id:
                    drone.release_order()

                drone.route_plan = []
                drone.current_waypoint_index = 0
                drone.waiting_recovery_station_id = ""
                tid = getattr(drone, "transport_truck_id", None)
                if tid:
                    truck = self.entity_mgr.trucks.get(tid)
                    if truck is not None and drone.drone_id in truck.docked_drones:
                        truck.docked_drones.remove(drone.drone_id)
                    drone.transport_truck_id = None
                drone.scheduled_launch_time = 0.0
                drone.launch_station_id = ""
                if drone.status.is_flying:
                    drone.status = DroneStatus.IDLE

    # ══════════════════════════════════════════════════════════════════════════
    # 调度入口
    # ══════════════════════════════════════════════════════════════════════════

    def dispatch(
        self,
        pending_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """执行一轮市场拍卖调度（全量批次），并在 summary 中附加拍卖统计信息。"""
        self._apply_market_initial_drone_loadout(current_time)
        self._auction_bid_count = 0
        self._auction_award_count = 0
        self._current_round_bdynamic_launches.clear()
        self._anchor_preview_routes = {}
        self._expire_contracts(current_time)
        self._try_flexible_recovery(current_time)
        pending_orders = self._orders_dict_in_market_auction_order(pending_orders, current_time)
        self._prepare_anchor_preview_routes(pending_orders, current_time, bbox, scene_id)

        plan = super().dispatch(pending_orders, current_time, bbox, scene_id=scene_id)
        self._patch_missing_truck_routes(plan, current_time, bbox, scene_id, return_to_depot=True)
        self._recalculate_actual_plan_costs(plan, pending_orders, current_time)
        plan.summary["solver"] = "market"
        plan.summary["auction_stats"] = {
            "bids": self._auction_bid_count,
            "awards": self._auction_award_count,
            "active_contracts": len([c for c in self._active_contracts if c.status == "active"]),
        }
        logger.info(
            "[MarketBasedSolver] 分配完成：orders=%d feasible=%d bids=%d awards=%d contracts=%d",
            plan.summary.get("total_orders", 0),
            plan.summary.get("feasible", 0),
            self._auction_bid_count,
            self._auction_award_count,
            plan.summary["auction_stats"]["active_contracts"],
        )
        self._revoke_stale_drone_assignments(plan, current_time)

        # 全局闭环校验
        self._validate_drone_route_closure(plan)
        self._validate_truck_premature_return(plan, current_time)
        self._ensure_idle_drones_return_to_depot(current_time)

        return plan

    def dispatch_incremental(
        self,
        new_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """事件驱动的增量拍卖：仅对动态新单执行调度，尊重已有契约。

        流程：
          1. 冻结状态：过期旧契约，获取当前卡车位置与活跃契约
          2. 不可变更过滤：将已处于 PICKED_UP/DELIVERING 的订单排除出拍卖池
          3. 构建保证时刻表：基于已锁定契约生成基础时刻表
          4. 增量拍卖：仅对 new_orders 收集投标并授标
        5. 双向锁定：中标后生成新契约，更新无人机/卡车锁定状态
        6. 返回增量计划（可与前序计划合并）
        """
        self._apply_market_initial_drone_loadout(current_time)
        self._auction_bid_count = 0
        self._auction_award_count = 0
        self._current_round_bdynamic_launches.clear()
        self._expire_contracts(current_time)
        self._try_flexible_recovery(current_time)

        # 兜底：避免调用侧漏传最后一批动态单，合并当前 pending 订单参与本轮拍卖。
        candidate_orders = dict(new_orders)
        adopted_pending = 0
        if self._order_mgr:
            for oid, order in self._order_mgr.pending_orders.items():
                if oid not in candidate_orders:
                    candidate_orders[oid] = order
                    adopted_pending += 1
        if adopted_pending > 0:
            logger.info(
                "[MarketBasedSolver] 增量调度补入 %d 个 pending 订单（防止动态单漏拍）",
                adopted_pending,
            )

        # 不可变更过滤：排除已进入履约状态的订单
        filtered_orders = {
            oid: order for oid, order in candidate_orders.items()
            if not self._is_order_immutable(oid, current_time)
        }
        skipped = len(candidate_orders) - len(filtered_orders)
        if skipped > 0:
            logger.info(
                "[MarketBasedSolver] 增量调度过滤 %d 个不可变更订单（已处于 PICKED_UP/DELIVERING）",
                skipped,
            )

        filtered_orders = self._orders_dict_in_market_auction_order(filtered_orders, current_time)
        if not self._anchor_preview_routes:
            self._prepare_anchor_preview_routes(filtered_orders, current_time, bbox, scene_id)

        plan = super().dispatch_incremental(filtered_orders, current_time, bbox, scene_id=scene_id)
        self._patch_missing_truck_routes(plan, current_time, bbox, scene_id, return_to_depot=False)
        self._recalculate_actual_plan_costs(plan, filtered_orders, current_time)
        plan.summary["solver"] = "market"
        plan.summary["dispatch_type"] = "incremental"
        plan.summary["auction_stats"] = {
            "bids": self._auction_bid_count,
            "awards": self._auction_award_count,
            "active_contracts": len([c for c in self._active_contracts if c.status == "active"]),
            "immutable_orders_skipped": skipped,
        }
        logger.info(
            "[MarketBasedSolver] 增量调度完成：new_orders=%d feasible=%d bids=%d awards=%d contracts=%d skipped_immutable=%d",
            plan.summary.get("total_orders", 0),
            plan.summary.get("feasible", 0),
            self._auction_bid_count,
            self._auction_award_count,
            plan.summary["auction_stats"]["active_contracts"],
            skipped,
        )
        self._revoke_stale_drone_assignments(plan, current_time)

        # 全局闭环校验
        self._validate_drone_route_closure(plan)
        self._validate_truck_premature_return(plan, current_time)
        self._ensure_idle_drones_return_to_depot(current_time)

        return plan

    def should_replan_unfinished(self) -> bool:
        """Market 默认保持契约优先，不做未完成单重优化。"""
        return False

    def dispatch_replan_current_state(
        self,
        replan_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None = None,
    ) -> DispatchPlan:
        """接口兼容：market 保持增量语义，直接按增量入口执行。"""
        return self.dispatch_incremental(replan_orders, current_time, bbox, scene_id=scene_id)

    def get_active_contracts(self) -> list[RendezvousContract]:
        """返回当前契约列表，供编排层统一兑现。"""
        return self._active_contracts

    def _patch_missing_truck_routes(
        self,
        plan: DispatchPlan,
        current_time: float,
        bbox: dict,
        scene_id: str | None,
        return_to_depot: bool,
    ) -> None:
        """补充生成只被分配了 B_DYNAMIC 任务导致被 GreedyMMCE 基类误跳过的卡车路线。"""
        missing_truck_ids = set()
        for truck in self.entity_mgr.trucks.values():
            if truck.truck_id in plan.truck_routes:
                continue
            needs_routing = False
            for v_id, s_id, _ in getattr(self, "_current_round_bdynamic_launches", []):
                if v_id == truck.truck_id and s_id:
                    needs_routing = True
                    break
            if not needs_routing:
                for drone in self.entity_mgr.drones.values():
                    if getattr(drone, "transport_truck_id", None) == truck.truck_id:
                        if getattr(drone, "launch_station_id", ""):
                            needs_routing = True
                            break
            if needs_routing:
                missing_truck_ids.add(truck.truck_id)

        if not missing_truck_ids:
            return

        road_graph, nodes = self._load_road_graph_for_market(bbox, scene_id)
        if not road_graph or not nodes:
            return

        for truck_id in missing_truck_ids:
            truck = self.entity_mgr.trucks.get(truck_id)
            if not truck:
                continue
            try:
                route = self._build_truck_route(
                    truck=truck,
                    orders=[],
                    recovery_station_ids=[],
                    current_time=current_time,
                    road_graph=road_graph,
                    nodes=nodes,
                    recovery_station_wait_times=None,
                    start_pos=truck.get_location(current_time),
                    return_to_depot=return_to_depot,
                )
                plan.truck_routes[truck_id] = route
                logger.info("[MarketBasedSolver] 已为被基类跳过的卡车 %s 补建 B_DYNAMIC/无人机 起飞路网", truck_id)
            except Exception as exc:
                logger.warning("[MarketBasedSolver] 补建卡车 %s 路网失败: %s", truck_id, exc)

    # ══════════════════════════════════════════════════════════════════════════
    # 成本重算（总体能耗公式不变）
    # ══════════════════════════════════════════════════════════════════════════

    def _recalculate_actual_plan_costs(
        self,
        plan: DispatchPlan,
        orders_by_id: dict[str, "Order"],
        current_time: float,
    ) -> None:
        """
        按最终真实执行路径重算总成本。

        说明：
          - 市场拍卖的授标阶段允许 B/B_WAIT 使用"边际卡车成本"参与竞价
          - 但最终统计必须计入卡车真实主干路线的全部距离与能耗
        """
        truck_distance_total = sum(route.total_distance for route in plan.truck_routes.values())
        truck_energy_total = sum(
            self._truck_energy_wh(route.total_distance)
            for route in plan.truck_routes.values()
        )

        uav_distance_total = 0.0
        uav_energy_total = 0.0
        penalty_total = sum(
            alloc.cost_penalty for alloc in plan.allocations
            if alloc.feasible and math.isfinite(alloc.cost_penalty)
        )

        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            order = orders_by_id.get(alloc.order_id)
            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if order is None or drone is None:
                continue

            if alloc.mode in ("B_WAIT", "B_DYNAMIC"):
                launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
                recovery = (
                    self.entity_mgr.stations.get(alloc.recovery_station_id)
                    or self.entity_mgr.depots.get(alloc.recovery_station_id)
                )
                if launch_station is None or recovery is None:
                    continue
                launch_loc = launch_station.location
                recovery_loc = recovery.location
            elif alloc.mode == "B":
                truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
                recovery = (
                    self.entity_mgr.stations.get(alloc.recovery_station_id)
                    or self.entity_mgr.depots.get(alloc.recovery_station_id)
                )
                if truck is None or recovery is None:
                    continue
                launch_loc = truck.get_location(current_time)
                recovery_loc = recovery.location
            elif alloc.mode == "C":
                depot = self.entity_mgr.depots.get(alloc.vehicle_id)
                if depot is None:
                    continue
                launch_loc = depot.location
                recovery_loc = depot.location
            else:
                continue

            dist_out = self._dist(launch_loc, order.delivery_loc)
            dist_back = self._dist(order.delivery_loc, recovery_loc)
            uav_distance_total += dist_out + dist_back
            uav_energy_total += self._uav_energy_wh(
                drone, launch_loc, order.delivery_loc, order.payload_weight
            )
            uav_energy_total += self._uav_energy_wh(
                drone, order.delivery_loc, recovery_loc, 0.0
            )

        cost_dist_total = self.C_DIST_ET * truck_distance_total + self.C_DIST_UAV * uav_distance_total
        cost_energy_total = self.C_ENERGY_ET * truck_energy_total + self.C_ENERGY_UAV * uav_energy_total

        plan.summary["cost_breakdown"] = {
            "dist": cost_dist_total,
            "energy": cost_energy_total,
            "penalty": penalty_total,
        }
        plan.summary["actual_route_costs"] = {
            "truck_distance_total": truck_distance_total,
            "truck_energy_total": truck_energy_total,
            "uav_distance_total": uav_distance_total,
            "uav_energy_total": uav_energy_total,
        }
        plan.cost_total = cost_dist_total + cost_energy_total

        logger.info(
            "[MarketBasedSolver] 按真实执行路径重算总成本: "
            "truck_dist=%.2f truck_energy=%.2f uav_dist=%.2f uav_energy=%.2f total=%.2f",
            truck_distance_total,
            truck_energy_total,
            uav_distance_total,
            uav_energy_total,
            plan.cost_total,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 契约管理
    # ══════════════════════════════════════════════════════════════════════════

    def _expire_contracts(self, current_time: float) -> None:
        """将已过生死线的契约标记为 expired。"""
        for contract in self._active_contracts:
            if contract.status == "active" and current_time > contract.latest_departure:
                contract.status = "expired"
                logger.info(
                    "[MarketBasedSolver][Contract] 契约过期: %s truck=%s anchor=%s deadline=%.1f now=%.1f",
                    contract.contract_id, contract.truck_id, contract.anchor_id,
                    contract.latest_departure, current_time,
                )

    def _try_flexible_recovery(self, current_time: float) -> None:
        """弹性归巢：若无人机飞回仓库比等待卡车回收更优，则释放契约。

        触发条件（全部满足）：
          1. 契约仍处于 active 状态
          2. 无人机在锚点等待卡车的剩余时间 T_wait > 飞回仓库时间 T_to_depot
          3. 无人机电量足以安全飞回仓库
        释放后无人机变为可用，可立即参与新订单竞标。
        """
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return
        depot = depots[0]

        for contract in self._active_contracts:
            if contract.status != "active":
                continue
            if current_time >= contract.t_sync:
                continue

            drone = self.entity_mgr.drones.get(contract.drone_id)
            if drone is None or drone.cruise_speed <= 0:
                continue

            anchor = (
                self.entity_mgr.stations.get(contract.anchor_id)
                or self.entity_mgr.depots.get(contract.anchor_id)
            )
            if anchor is None:
                continue

            t_wait = contract.t_sync - current_time
            dist_to_depot = self._dist(anchor.location, depot.location)
            t_to_depot = dist_to_depot / drone.cruise_speed

            if t_wait <= t_to_depot:
                continue

            energy_to_depot = self._flight_energy(drone, anchor.location, depot.location, 0.0)
            if energy_to_depot * self.ENERGY_SAFETY_FACTOR > drone.battery_current:
                continue

            contract.status = "released"
            logger.info(
                "[MarketBasedSolver][Contract] 弹性归巢释放契约 %s: drone=%s "
                "T_wait=%.1fs > T_to_depot=%.1fs, energy_ok=True",
                contract.contract_id, contract.drone_id,
                t_wait, t_to_depot,
            )

    def _create_contract(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
    ) -> RendezvousContract:
        """基于中标结果创建并存储 RendezvousContract。"""
        self._contract_seq += 1

        truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
        drone = self.entity_mgr.drones.get(alloc.drone_id)
        launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
        recovery = (
            self.entity_mgr.stations.get(alloc.recovery_station_id)
            or self.entity_mgr.depots.get(alloc.recovery_station_id)
        )

        # 卡车到达回收锚点的预计时间
        truck_arrival = current_time
        latest_departure = current_time + self.T_MAX_WAIT
        if truck is not None:
            anchors = self._build_anchor_timetable(truck, current_time)
            recovery_anchor = next(
                (a for a in anchors if a["anchor_id"] == alloc.recovery_station_id), None
            )
            if recovery_anchor is not None:
                truck_arrival = recovery_anchor["arrival_time"]
                latest_departure = recovery_anchor["latest_rendezvous_time"]

        # 无人机到达回收锚点的预计时间
        launch_time_est = (
            alloc.launch_time
            if alloc.launch_time > current_time
            else current_time + self.TRUCK_DRONE_LAUNCH_TIME
        )
        uav_arrival_time = launch_time_est
        if drone is not None and drone.cruise_speed > 0 and launch_station and recovery:
            dist_out = self._dist(launch_station.location, order.delivery_loc)
            dist_back = self._dist(order.delivery_loc, recovery.location)
            uav_arrival_time = (
                launch_time_est
                + dist_out / drone.cruise_speed
                + self.delivery_service_time
                + dist_back / drone.cruise_speed
            )

        t_sync = max(uav_arrival_time, truck_arrival) + self.TRUCK_DRONE_RECOVER_TIME

        contract = RendezvousContract(
            contract_id=f"RC-{self._contract_seq:04d}",
            truck_id=alloc.vehicle_id,
            drone_id=alloc.drone_id,
            order_id=alloc.order_id,
            anchor_id=alloc.recovery_station_id,
            launch_anchor_id=alloc.launch_station_id,
            arrival_time=truck_arrival,
            latest_departure=latest_departure,
            uav_arrival_time=uav_arrival_time,
            t_sync=t_sync,
        )
        self._active_contracts.append(contract)
        logger.info(
            "[MarketBasedSolver][Contract] 新建契约 %s: truck=%s drone=%s order=%s "
            "anchor=%s truck_arrival=%.1f deadline=%.1f uav_arrival=%.1f t_sync=%.1f",
            contract.contract_id, contract.truck_id, contract.drone_id,
            contract.order_id, contract.anchor_id,
            contract.arrival_time, contract.latest_departure,
            contract.uav_arrival_time, contract.t_sync,
        )
        return contract

    def _get_truck_contracts(self, truck_id: str) -> list[RendezvousContract]:
        """获取某辆卡车的所有活跃契约。"""
        return [c for c in self._active_contracts if c.truck_id == truck_id and c.status == "active"]

    def _is_drone_locked(self, drone_id: str, current_time: float) -> bool:
        """检查无人机是否被活跃契约锁定（尚未完成回收）。

        released / fulfilled / expired 状态的契约不再锁定无人机。
        """
        for contract in self._active_contracts:
            if contract.status == "active" and contract.drone_id == drone_id:
                if current_time < contract.t_sync:
                    return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # 路径方向性优化工具方法
    # ══════════════════════════════════════════════════════════════════════════

    def _find_best_insertion_for_truck(
        self,
        truck: "Truck",
        new_order: "Order",
        current_time: float,
    ) -> tuple[bool, int, float]:
        """在卡车已规划路径中搜索新订单的最优插入位置。

        遍历计划停靠序列中所有可能的插入间隙，选择使绕路距离最小
        且不违反任何已锁定契约时间窗的最优插入点。

        Returns:
            (is_feasible, best_insert_index, detour_time)
            best_insert_index: 在 remaining_stops 中的插入位置索引
                               （0 = 插在第一个剩余停靠前），-1 = 不可行
        """
        if truck.speed <= 0:
            return False, -1, float("inf")

        planned_stops = getattr(truck, "_planned_route_stops", None) or []
        cursor = int(getattr(truck, "_planned_route_cursor", 0))
        remaining = planned_stops[cursor:]
        contracts = self._get_truck_contracts(truck.truck_id)
        order_pos = new_order.delivery_loc
        truck_loc = truck.get_location(current_time)

        positions: list = [truck_loc]
        arrival_times: list[float] = [current_time]
        for stop in remaining:
            pos = stop.get("position")
            if pos is not None:
                positions.append(pos)
                arrival_times.append(float(stop.get("arrival_time", current_time)))

        n = len(positions)

        contract_deadlines: dict[int, float] = {}
        for ci, stop in enumerate(remaining):
            node_id = stop.get("node_id", "")
            for c in contracts:
                if c.anchor_id == node_id:
                    contract_deadlines[ci + 1] = c.latest_departure

        best_feasible = False
        best_idx = -1
        best_detour_time = float("inf")

        for insert_after in range(n):
            if insert_after < n - 1:
                prev_pos = positions[insert_after]
                next_pos = positions[insert_after + 1]
                orig_dist = self._dist(prev_pos, next_pos)
                new_dist = self._dist(prev_pos, order_pos) + self._dist(order_pos, next_pos)
                detour_dist = max(0.0, new_dist - orig_dist)
            else:
                prev_pos = positions[-1]
                detour_dist = self._dist(prev_pos, order_pos)

            detour_time = detour_dist / truck.speed + self.SERVICE_TIME_CUSTOMER

            feasible = True
            for pos_idx, deadline in contract_deadlines.items():
                if pos_idx > insert_after:
                    shifted_arrival = arrival_times[pos_idx] + detour_time
                    if shifted_arrival > deadline:
                        feasible = False
                        break

            if not feasible:
                continue

            if n == 1 and contracts:
                for c in contracts:
                    anchor = (
                        self.entity_mgr.stations.get(c.anchor_id)
                        or self.entity_mgr.depots.get(c.anchor_id)
                    )
                    if anchor is None:
                        continue
                    dist_to_anchor = self._dist(order_pos, anchor.location)
                    eta = current_time + detour_time + dist_to_anchor / truck.speed
                    if eta > c.latest_departure:
                        feasible = False
                        break

            if feasible and detour_time < best_detour_time:
                best_feasible = True
                best_idx = insert_after
                best_detour_time = detour_time

        return best_feasible, best_idx, best_detour_time

    def _point_to_segment_distance(
        self,
        point: "Position3D",
        seg_start: "Position3D",
        seg_end: "Position3D",
    ) -> float:
        """计算点到线段的最短距离（2D 投影）。"""
        dx = seg_end.x - seg_start.x
        dy = seg_end.y - seg_start.y
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-10:
            return self._dist(point, seg_start)

        t = max(0.0, min(1.0, (
            (point.x - seg_start.x) * dx + (point.y - seg_start.y) * dy
        ) / seg_len_sq))

        from core.entities.primitives import Position3D as Pos3D
        proj = Pos3D(x=seg_start.x + t * dx, y=seg_start.y + t * dy, z=0)
        return self._dist(point, proj)

    def _compute_backtrack_penalty(
        self,
        truck: "Truck",
        order: "Order",
        current_time: float,
    ) -> float:
        """计算新订单相对于卡车前进方向的回头路惩罚。

        使用卡车当前前进方向向量与卡车→订单方向向量的夹角余弦判断：
          cos > 0  → 订单在前进方向上，无惩罚
          cos < 0  → 订单在卡车身后（回头路），惩罚与偏离程度和距离成正比
        """
        truck_loc = truck.get_location(current_time)

        planned_stops = getattr(truck, "_planned_route_stops", None) or []
        cursor = int(getattr(truck, "_planned_route_cursor", 0))
        remaining = planned_stops[cursor:]

        if not remaining:
            return 0.0

        next_pos = remaining[0].get("position")
        if next_pos is None:
            return 0.0

        dx_fwd = next_pos.x - truck_loc.x
        dy_fwd = next_pos.y - truck_loc.y
        mag_fwd = math.sqrt(dx_fwd * dx_fwd + dy_fwd * dy_fwd)
        if mag_fwd < 1.0:
            return 0.0

        dx_ord = order.delivery_loc.x - truck_loc.x
        dy_ord = order.delivery_loc.y - truck_loc.y
        mag_ord = math.sqrt(dx_ord * dx_ord + dy_ord * dy_ord)
        if mag_ord < 1.0:
            return 0.0

        cos_angle = (dx_fwd * dx_ord + dy_fwd * dy_ord) / (mag_fwd * mag_ord)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        if cos_angle >= 0:
            return 0.0

        backtrack_factor = abs(cos_angle)
        distance = self._dist(truck_loc, order.delivery_loc)
        return self.BACKTRACK_PENALTY_WEIGHT * backtrack_factor * distance

    def _compute_path_proximity_bonus(
        self,
        truck: "Truck",
        order: "Order",
        current_time: float,
    ) -> float:
        """计算订单与卡车未来路径的邻近度奖励。

        订单越靠近卡车计划路径线段，奖励越大（作为负成本参与评分）。
        鼓励卡车 "顺路" 处理路径附近的订单，而非执行大范围迂回。
        """
        planned_stops = getattr(truck, "_planned_route_stops", None) or []
        cursor = int(getattr(truck, "_planned_route_cursor", 0))
        remaining = planned_stops[cursor:]

        if not remaining:
            return 0.0

        order_pos = order.delivery_loc
        min_dist = float("inf")
        prev_pos = truck.get_location(current_time)

        for stop in remaining:
            stop_pos = stop.get("position")
            if stop_pos is None:
                continue
            seg_dist = self._point_to_segment_distance(order_pos, prev_pos, stop_pos)
            min_dist = min(min_dist, seg_dist)
            prev_pos = stop_pos

        if min_dist == float("inf"):
            return 0.0

        max_effective_range = 2000.0
        if min_dist >= max_effective_range:
            return 0.0

        proximity_ratio = 1.0 - min(min_dist / max_effective_range, 1.0)
        return self.PATH_PROXIMITY_BONUS * proximity_ratio * max_effective_range

    def _optimize_route_2opt(self, route, locked_anchor_ids: set[str] | None = None):
        """对 TruckRoute 的中间节点执行 2-opt 局部搜索，减少回头路。

        约束：
          - 首尾节点（仓库/origin）固定不可移动
          - locked_anchor_ids 中的锚点节点位置固定不可交换
        """
        if locked_anchor_ids is None:
            locked_anchor_ids = set()

        nodes = route.nodes
        n = len(nodes)
        if n < 4:
            return route

        locked_indices = {0, n - 1}
        for i, node in enumerate(nodes):
            if node.node_id in locked_anchor_ids:
                locked_indices.add(i)

        improved = True
        iterations = 0

        while improved and iterations < self.TWO_OPT_MAX_ITERATIONS:
            improved = False
            iterations += 1

            for i in range(1, n - 2):
                if i in locked_indices:
                    continue
                for j in range(i + 1, n - 1):
                    if j in locked_indices:
                        continue

                    has_locked = any(k in locked_indices for k in range(i, j + 1))
                    if has_locked:
                        continue

                    d_old = (
                        self._dist(nodes[i - 1].position, nodes[i].position)
                        + self._dist(nodes[j].position, nodes[j + 1].position)
                    )
                    d_new = (
                        self._dist(nodes[i - 1].position, nodes[j].position)
                        + self._dist(nodes[i].position, nodes[j + 1].position)
                    )

                    if d_new < d_old - 1.0:
                        nodes[i:j + 1] = list(reversed(nodes[i:j + 1]))
                        improved = True

        total_dist = 0.0
        cur_time = nodes[0].arrival_time
        speed = 0.0
        for truck in self.entity_mgr.trucks.values():
            if truck.truck_id == route.truck_id:
                speed = truck.speed
                break
        if speed <= 0:
            return route

        for idx in range(1, len(nodes)):
            seg_dist = self._dist(nodes[idx - 1].position, nodes[idx].position)
            total_dist += seg_dist
            arrive = cur_time + seg_dist / speed
            nodes[idx].arrival_time = arrive
            nodes[idx].departure_time = arrive
            cur_time = arrive

        route.total_distance = total_dist
        route.geometry = [node.position for node in nodes]
        return route

    def _validate_truck_insertion(
        self,
        truck: "Truck",
        new_order: "Order",
        current_time: float,
    ) -> bool:
        """校验卡车接受新订单后，是否存在至少一个合法插入位置满足所有契约时间窗。

        改进：不再假设新单只能在"当前位置"立即插入，而是搜索计划路径中
        所有可能的插入间隙，只要任一间隙可行即返回 True。这允许 "顺路"
        处理路径附近的订单而非要求立即绕路。
        """
        contracts = self._get_truck_contracts(truck.truck_id)
        if not contracts:
            return True

        feasible, best_idx, detour_time = self._find_best_insertion_for_truck(
            truck, new_order, current_time
        )
        if not feasible:
            logger.debug(
                "[MarketBasedSolver] 卡车 %s 插单 %s 在所有位置均导致契约超时 "
                "(最小绕路时间=%.1f)，该投标不可行",
                truck.truck_id, new_order.order_id, detour_time,
            )
        return feasible

    def fulfill_contract(self, contract_id: str) -> None:
        """外部（仿真引擎/决策引擎）通知契约已兑现。"""
        for contract in self._active_contracts:
            if contract.contract_id == contract_id and contract.status == "active":
                contract.status = "fulfilled"
                logger.info("[MarketBasedSolver][Contract] 契约已兑现: %s", contract_id)
                return

    # ══════════════════════════════════════════════════════════════════════════
    # 锚点预览路线与卡车路线构建
    # ══════════════════════════════════════════════════════════════════════════

    def _build_truck_route(
        self,
        truck: "Truck",
        orders: list["Order"],
        recovery_station_ids: list[str],
        current_time: float,
        road_graph: Any,
        nodes: Any,
        recovery_station_wait_times: dict[str, float] | None = None,
        start_pos: "Position3D | None" = None,
        return_to_depot: bool = True,
    ):
        """重写 _build_truck_route 以修复基类对 B_DYNAMIC 起飞站点的遗漏。"""
        if recovery_station_wait_times is None:
            recovery_station_wait_times = {}

        # 1. 注入本轮新分配给该卡车的 B_DYNAMIC 起飞站点
        added_stations = []
        for v_id, s_id, wait_time in getattr(self, "_current_round_bdynamic_launches", []):
            if v_id == truck.truck_id and s_id:
                if s_id not in recovery_station_ids:
                    recovery_station_ids.append(s_id)
                    added_stations.append(s_id)
                res_wait = recovery_station_wait_times.get(s_id, 0.0)
                recovery_station_wait_times[s_id] = max(res_wait, wait_time)

        # 2. 防御性补丁：将已锁定的活跃契约的停靠锚点强制补充进 recovery 列表，防止部分重排逻辑漏掉
        for contract in self._get_truck_contracts(truck.truck_id):
            if contract.anchor_id and contract.anchor_id not in recovery_station_ids:
                recovery_station_ids.append(contract.anchor_id)
                added_stations.append(contract.anchor_id)
            if contract.launch_anchor_id and contract.launch_anchor_id not in recovery_station_ids:
                recovery_station_ids.append(contract.launch_anchor_id)
                added_stations.append(contract.launch_anchor_id)

        # 3. 防防御性补丁 2：检查当前挂载在本车上尚未起飞的无人机目标站（补偿跨批次 B_DYNAMIC 任务）
        for drone in self.entity_mgr.drones.values():
            if getattr(drone, "transport_truck_id", None) != truck.truck_id:
                continue
            # 仅补偿“真实待起飞”的站点，避免历史脏状态把卡车拉回头路。
            if drone.status != DroneStatus.IDLE:
                continue
            if not getattr(drone, "route_plan", None):
                continue
            s_id = getattr(drone, "launch_station_id", "")
            if s_id and s_id not in recovery_station_ids:
                recovery_station_ids.append(s_id)
                added_stations.append(s_id)

        if added_stations:
            logger.debug(
                "[MarketBasedSolver] 为卡车 %s 补充路径必须途径站点(B_DYNAMIC起飞站/契约锚点): %s",
                truck.truck_id, added_stations
            )

        return super()._build_truck_route(
            truck=truck,
            orders=orders,
            recovery_station_ids=recovery_station_ids,
            current_time=current_time,
            road_graph=road_graph,
            nodes=nodes,
            recovery_station_wait_times=recovery_station_wait_times,
            start_pos=start_pos,
            return_to_depot=return_to_depot,
        )

    def _prepare_anchor_preview_routes(
        self,
        pending_orders: dict[str, "Order"],
        current_time: float,
        bbox: dict,
        scene_id: str | None,
    ) -> None:
        """
        为市场拍卖生成卡车主干路径预览，并据此广播真实锚点时刻表。

        优先级：
          1. 若卡车已有执行中的 planned_route_stops，则直接复用
          2. 否则基于本轮待调度订单，为每辆卡车构建一条预览主干路线
        """
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return

        # 若已有执行中的真实停靠计划，则无需再做预览。
        has_runtime_plan = False
        for truck in trucks:
            planned_stops = getattr(truck, "_planned_route_stops", None)
            if planned_stops:
                has_runtime_plan = True
                break
        if has_runtime_plan:
            logger.info("[MarketBasedSolver] 锚点时刻表复用卡车当前运行计划")
            return

        road_graph, nodes = self._load_road_graph_for_market(bbox, scene_id)
        if road_graph is None or nodes is None:
            logger.warning("[MarketBasedSolver] 无法构建卡车主干预览路线，将回退到运行时计划/启发式锚点")
            return

        orders_by_truck: dict[str, list["Order"]] = {truck.truck_id: [] for truck in trucks}
        for order in pending_orders.values():
            nearest_truck = min(
                trucks,
                key=lambda truck: self._dist(truck.get_location(current_time), order.delivery_loc),
            )
            orders_by_truck[nearest_truck.truck_id].append(order)

        for truck in trucks:
            candidate_orders = orders_by_truck.get(truck.truck_id, [])
            if not candidate_orders:
                continue
            # 将契约锁定的站点作为必经回收点注入预览路线
            contract_stations = [c.anchor_id for c in self._get_truck_contracts(truck.truck_id)]
            try:
                preview_route = self._build_truck_route(
                    truck=truck,
                    orders=candidate_orders,
                    recovery_station_ids=contract_stations,
                    current_time=current_time,
                    road_graph=road_graph,
                    nodes=nodes,
                    recovery_station_wait_times=None,
                )
            except Exception as exc:
                logger.warning(
                    "[MarketBasedSolver] 卡车 %s 主干预览路线构建失败，回退到启发式锚点: %s",
                    truck.truck_id,
                    exc,
                )
                continue

            locked_ids = set(contract_stations)
            old_dist = preview_route.total_distance
            preview_route = self._optimize_route_2opt(preview_route, locked_ids)
            if preview_route.total_distance < old_dist - 1.0:
                logger.info(
                    "[MarketBasedSolver] 卡车 %s 2-opt 优化：%.0f m → %.0f m (节省 %.0f m)",
                    truck.truck_id, old_dist, preview_route.total_distance,
                    old_dist - preview_route.total_distance,
                )

            self._anchor_preview_routes[truck.truck_id] = preview_route
            logger.info(
                "[MarketBasedSolver] 卡车 %s 主干预览路线已生成：%d 节点，经停锚点 %s",
                truck.truck_id,
                len(preview_route.nodes),
                preview_route.charging_stop_ids,
            )

    def _load_road_graph_for_market(
        self,
        bbox: dict,
        scene_id: str | None,
    ):
        """复用贪心基线的 OSM 加载逻辑，为锚点时刻表生成预览路线。"""
        road_graph = None
        nodes = None

        if scene_id and load_osm_from_cache:
            try:
                osm_xml, _ = load_osm_from_cache(scene_id)
                if osm_xml:
                    road_graph, nodes = _osm_svc.build_road_graph(osm_xml)
            except Exception as exc:
                logger.warning("[MarketBasedSolver] 从场景缓存加载 OSM 失败: %s", exc)
                road_graph = None
                nodes = None

        if road_graph is None:
            try:
                osm_xml = _osm_svc.download_osm(
                    bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
                )
                road_graph, nodes = _osm_svc.build_road_graph(osm_xml)
            except Exception as exc:
                logger.warning("[MarketBasedSolver] 下载/构建 OSM 失败: %s", exc)
                return None, None

        return road_graph, nodes

    # ══════════════════════════════════════════════════════════════════════════
    # 单订单拍卖
    # ══════════════════════════════════════════════════════════════════════════

    def _allocate_order(
        self,
        order: "Order",
        current_time: float,
        allocated_drones: set[str],
        truck_last_pos: dict[str, "Position3D"] = None,
    ) -> AllocationResult:
        """
        对单个订单执行单任务拍卖。

        流程：
          1. 先做硬过滤：超重订单直接交由卡车竞标
          2. 收集卡车投标（模式 A）—— 需通过契约合规校验
          3. 收集无人机投标（模式 B_WAIT / B / C）—— 排除契约锁定的无人机
          4. 选择综合评分最低的投标作为中标结果
          5. 若中标模式为 B_WAIT，立即生成 RendezvousContract
        """
        bids: list[AllocationResult] = []
        drone_diag: dict = {}

        if self._must_assign_to_truck(order):
            forced_truck_bid = self._best_truck_bid(order, current_time, truck_last_pos)
            if forced_truck_bid is not None:
                forced_truck_bid.reason = "订单超出无人机载重能力，强制由卡车执行"
                self._auction_bid_count += 1
                self._auction_award_count += 1
                self._log_bid_diagnostics(
                    order=order,
                    bids=[forced_truck_bid],
                    selected=forced_truck_bid,
                    drone_diag={
                        "forced_to_truck": True,
                        "max_drone_payload": max(
                            (d.payload_capacity for d in self.entity_mgr.drones.values()),
                            default=0.0,
                        ),
                    },
                )
                return forced_truck_bid
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="REJECT",
                distance=float("inf"),
                feasible=False,
                reason="订单超重且无可用卡车",
            )

        truck_bid = self._best_truck_bid(order, current_time, truck_last_pos)
        if truck_bid is not None:
            bids.append(truck_bid)

        bids.extend(self._collect_drone_bids(order, current_time, allocated_drones, drone_diag))
        self._auction_bid_count += len(bids)

        if not bids:
            self._log_bid_diagnostics(
                order=order,
                bids=[],
                selected=None,
                drone_diag=drone_diag,
            )
            return AllocationResult(
                order_id=order.order_id,
                vehicle_id="",
                mode="REJECT",
                distance=float("inf"),
                feasible=False,
                reason="拍卖失败：无可用卡车或无人机投标",
            )

        best = min(bids, key=lambda bid: bid.score_total)
        self._auction_award_count += 1

        # 路径拼接约束：预测起点对齐 + 坐标跳变校验
        if best.feasible and best.drone_id and best.mode != "A":
            best = self._apply_path_stitching_constraints(best, order, current_time)

        # 中标后立即生成时空契约
        if best.mode == "B_WAIT" and best.feasible:
            self._create_contract(best, order, current_time)

        self._log_bid_diagnostics(order=order, bids=bids, selected=best, drone_diag=drone_diag)
        logger.info(
            "[MarketBasedSolver] 订单 %s 收到 %d 个投标，授标模式=%s 载体=%s score=%.2f",
            order.order_id,
            len(bids),
            best.mode,
            best.drone_id or best.vehicle_id,
            best.score_total,
        )

        # 把当前中标无人机记录进屏蔽列表，不依赖父类greedy_mmce，避免B_DYNAMIC在父类漏判导致一机多单
        if best.feasible and best.drone_id and best.mode != "A":
            allocated_drones.add(best.drone_id)
            if best.mode == "B_DYNAMIC" and best.launch_station_id:
                self._current_round_bdynamic_launches.append(
                    (best.vehicle_id, best.launch_station_id, self.TRUCK_DRONE_LAUNCH_TIME)
                )

        return best

    def _must_assign_to_truck(self, order: "Order") -> bool:
        """若订单重量超过全局无人机最大载重，则直接走卡车模式。"""
        if not self.entity_mgr.drones:
            return True
        max_payload = max(d.payload_capacity for d in self.entity_mgr.drones.values())
        return order.payload_weight > max_payload

    def _orders_dict_in_market_auction_order(
        self,
        orders: dict[str, "Order"],
        current_time: float,
    ) -> dict[str, "Order"]:
        """在调用基类 dispatch 前重排 dict 插入序：超重（仅模式 A 卡车）单优先，再按紧迫度。

        不修改 greedy_mmce：仅 market 入口对本轮订单重排，避免 B_WAIT 先锁约导致
        卡车专送单 _validate_truck_insertion 恒失败。
        """
        if not orders:
            return orders
        sorted_list = sorted(
            orders.values(),
            key=lambda o: (
                0 if self._must_assign_to_truck(o) else 1,
                o.deadline - current_time,
                o.deadline,
                o.order_id,
            ),
        )
        return {o.order_id: o for o in sorted_list}

    # ── 竞标准入：无人机状态分级 ─────────────────────────────────────

    def _classify_drone_eligibility(
        self,
        drone: "Drone",
        current_time: float,
        allocated_drones: set[str],
        order: "Order",
    ) -> tuple[str, "Position3D | None"]:
        """根据无人机即时物理状态判定竞标准入等级与竞标起点。

        Returns:
            (eligibility, bid_origin)
            eligibility: "IDLE" | "RETURNING" | "DELIVERING" | "LOCKED" | "REJECT"
            bid_origin:  竞标起点坐标（REJECT/LOCKED 时为 None）
        """
        if drone.drone_id in allocated_drones:
            return "REJECT", None
        if self._is_drone_locked(drone.drone_id, current_time):
            return "LOCKED", None
        if drone.payload_capacity < order.payload_weight:
            return "REJECT", None

        # IDLE（含 CHARGING/QUEUING 之后恢复的空闲态）
        if drone.status == DroneStatus.IDLE:
            if getattr(drone, "carrying_order_id", None):
                return "REJECT", None
            # 关键修复：IDLE 但已挂载待执行路径/待起飞时，必须视作锁定，
            # 防止增量拍卖重复占用同一无人机造成轨迹覆盖。
            if bool(getattr(drone, "has_pending_route", False)):
                return "LOCKED", None
            if float(getattr(drone, "scheduled_launch_time", 0.0)) > current_time + 1e-6:
                return "LOCKED", None
            if getattr(drone, "launch_station_id", "") and getattr(drone, "transport_truck_id", None):
                return "LOCKED", None
            if getattr(drone, "waiting_recovery_station_id", ""):
                return "LOCKED", None
            return "IDLE", drone.current_loc

        # RETURNING_TO_DEPOT / FLYING_TO_STATION（归航中）：允许串联/接力模式
        if drone.status in (
            DroneStatus.RETURNING_TO_DEPOT,
            DroneStatus.FLYING_TO_STATION,
            DroneStatus.FLYING_TO_TRUCK,
        ):
            route_end = self._get_drone_route_endpoint(drone)
            if route_end is None:
                return "REJECT", None
            energy_for_next = self._flight_energy(
                drone, route_end, order.delivery_loc, order.payload_weight
            )
            safety_energy = energy_for_next * self.ENERGY_SAFETY_FACTOR
            if drone.battery_current <= safety_energy:
                return "REJECT", None
            return "RETURNING", route_end

        # DELIVERING（配送中）：仅当距终点时间 < AUCTION_BUFFER_TIME 时允许预调度
        if drone.status in (DroneStatus.FLYING_TO_DELIVER,):
            t_remain = self._estimate_remaining_flight_time(drone, current_time)
            if t_remain > self.AUCTION_BUFFER_TIME:
                return "REJECT", None
            deliver_point = self._get_drone_delivery_endpoint(drone)
            if deliver_point is None:
                return "REJECT", None
            return "DELIVERING", deliver_point

        # FLYING_TO_PICKUP / LOADING / UNLOADING — 正在执行关键动作，不可打断
        return "LOCKED", None

    def _get_drone_route_endpoint(self, drone: "Drone") -> "Position3D | None":
        """获取无人机当前路径的最终航路点坐标。"""
        route = getattr(drone, "route_plan", None) or []
        if not route:
            return None
        return route[-1].loc

    def _get_drone_delivery_endpoint(self, drone: "Drone") -> "Position3D | None":
        """获取 DELIVERING 无人机当前任务的送货点坐标。"""
        from core.entities.primitives import WaypointAction
        route = getattr(drone, "route_plan", None) or []
        idx = max(0, int(getattr(drone, "current_waypoint_index", 0)))
        for wp in route[idx:]:
            if wp.action == WaypointAction.DELIVER:
                return wp.loc
        if route:
            return route[-1].loc
        return None

    def _estimate_remaining_flight_time(self, drone: "Drone", current_time: float) -> float:
        """估算无人机完成当前路径剩余飞行的时间。"""
        if drone.cruise_speed <= 0:
            return float("inf")
        route = getattr(drone, "route_plan", None) or []
        idx = max(0, int(getattr(drone, "current_waypoint_index", 0)))
        total_dist = 0.0
        pos = drone.current_loc
        for wp in route[idx:]:
            total_dist += self._dist(pos, wp.loc)
            pos = wp.loc
        return total_dist / drone.cruise_speed

    def _predict_drone_position(
        self, drone: "Drone", current_time: float, delta_t: float
    ) -> "Position3D":
        """预测无人机在 t + delta_t 时刻的位置（线性插值）。

        用于拍卖起点对齐：竞标时的起点必须预测为 P(t + delta_t)，
        严禁使用订单初始坐标或仓库坐标作为移动中实体的起点。
        """
        if not drone.status.is_flying or drone.cruise_speed <= 0:
            return drone.current_loc

        route = getattr(drone, "route_plan", None) or []
        idx = max(0, int(getattr(drone, "current_waypoint_index", 0)))
        if idx >= len(route):
            return drone.current_loc

        pos = drone.current_loc
        remaining_dist = drone.cruise_speed * delta_t
        for wp in route[idx:]:
            seg_dist = self._dist(pos, wp.loc)
            if seg_dist <= 0:
                pos = wp.loc
                continue
            if remaining_dist < seg_dist:
                ratio = remaining_dist / seg_dist
                from core.entities.primitives import Position3D as Pos3D
                return Pos3D(
                    x=pos.x + (wp.loc.x - pos.x) * ratio,
                    y=pos.y + (wp.loc.y - pos.y) * ratio,
                    z=pos.z + (wp.loc.z - pos.z) * ratio,
                )
            remaining_dist -= seg_dist
            pos = wp.loc
        return pos

    def _collect_drone_bids(
        self,
        order: "Order",
        current_time: float,
        allocated_drones: set[str],
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """收集所有可用无人机的竞标结果（基于状态准入分级）。

        准入等级：
          IDLE:       绝对准入，起点为当前精确物理坐标
          RETURNING:  准入（串联/接力模式），起点为当前航段终点
          DELIVERING: 准入（预调度模式），起点为当前任务送货点
          LOCKED:     严禁准入（已锁定/起飞中/回收中）
          REJECT:     不满足基本条件（已分配/载重不足/电量不足）
        """
        excluded_allocated = 0
        excluded_payload = 0
        excluded_locked = 0
        idle_count = 0
        returning_count = 0
        delivering_count = 0

        eligible_idle: list[tuple["Drone", "Position3D"]] = []
        eligible_returning: list[tuple["Drone", "Position3D"]] = []
        eligible_delivering: list[tuple["Drone", "Position3D"]] = []

        for drone in self.entity_mgr.drones.values():
            eligibility, bid_origin = self._classify_drone_eligibility(
                drone, current_time, allocated_drones, order
            )
            if eligibility == "IDLE":
                idle_count += 1
                eligible_idle.append((drone, bid_origin))
            elif eligibility == "RETURNING":
                returning_count += 1
                eligible_returning.append((drone, bid_origin))
            elif eligibility == "DELIVERING":
                delivering_count += 1
                eligible_delivering.append((drone, bid_origin))
            elif eligibility == "LOCKED":
                excluded_locked += 1
            else:  # REJECT
                if drone.drone_id in allocated_drones:
                    excluded_allocated += 1
                elif drone.payload_capacity < order.payload_weight:
                    excluded_payload += 1

        if diagnostics is not None:
            diagnostics["idle_drones"] = idle_count
            diagnostics["returning_drones"] = returning_count
            diagnostics["delivering_drones"] = delivering_count
            diagnostics["excluded_allocated"] = excluded_allocated
            diagnostics["excluded_payload"] = excluded_payload
            diagnostics["excluded_locked"] = excluded_locked
            diagnostics["eligible_drones"] = idle_count + returning_count + delivering_count
            diagnostics["drones"] = []
            diagnostics["relay_candidates"] = 0
            diagnostics["relay_feasible"] = 0

        bids: list[AllocationResult] = []

        # 1. IDLE 无人机 — 标准竞标（B_WAIT / C / B_DYNAMIC）
        for drone, origin in eligible_idle:
            drone_diag = self._make_drone_diag(drone.drone_id)
            bids.extend(self._collect_anchor_bids_for_drone(drone, order, current_time, drone_diag))
            depot_bid = self._build_depot_bid(drone, order, current_time, drone_diag)
            if depot_bid is not None:
                bids.append(depot_bid)
            bids.extend(self._collect_b_dynamic_bids(drone, order, current_time, drone_diag))
            if self.ALLOW_MOVING_TRUCK_LAUNCH:
                bids.extend(self._collect_moving_truck_bids(drone, order, current_time, drone_diag))
            if diagnostics is not None:
                diagnostics["drones"].append(drone_diag)

        # 2. RETURNING 无人机 — 串联/接力竞标，起点为航段终点
        for drone, origin in eligible_returning:
            relay_bids = self._build_returning_relay_bids(
                drone, origin, order, current_time, diagnostics
            )
            bids.extend(relay_bids)

        # 3. DELIVERING 无人机 — 预调度竞标，起点为送货点坐标
        for drone, origin in eligible_delivering:
            relay_bids = self._build_delivering_preschedule_bids(
                drone, origin, order, current_time, diagnostics
            )
            bids.extend(relay_bids)

        return bids

    def _make_drone_diag(self, drone_id: str) -> dict:
        return {
            "drone_id": drone_id,
            "anchor_candidates": 0,
            "anchor_feasible": 0,
            "anchor_energy_rejected": 0,
            "anchor_sync_rejected": 0,
            "depot_feasible": 0,
            "depot_energy_rejected": 0,
            "moving_candidates": 0,
            "moving_feasible": 0,
            "moving_energy_rejected": 0,
            "moving_sync_rejected": 0,
            "b_dynamic_candidates": 0,
            "b_dynamic_feasible": 0,
            "b_dynamic_energy_rejected": 0,
        }

    def _build_returning_relay_bids(
        self,
        drone: "Drone",
        relay_origin: "Position3D",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """RETURNING 无人机串联竞标：以当前航段终点为起点飞往新订单后回仓。"""
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return []
        depot = depots[0]
        bids: list[AllocationResult] = []

        remaining_energy = self._estimate_relay_remaining_energy(drone, relay_origin)
        if remaining_energy <= 0:
            return bids

        dist_to_customer = self._dist(relay_origin, order.delivery_loc)
        dist_to_depot = self._dist(order.delivery_loc, depot.location)
        energy_to_customer = self._flight_energy(
            drone, relay_origin, order.delivery_loc, order.payload_weight
        )
        energy_to_depot = self._flight_energy(
            drone, order.delivery_loc, depot.location, 0.0
        )
        energy_needed = (energy_to_customer + energy_to_depot) * self.ENERGY_SAFETY_FACTOR
        if energy_needed > remaining_energy:
            return bids

        time_to_available = self._estimate_relay_available_time(drone, relay_origin, current_time)
        time_to_customer = dist_to_customer / drone.cruise_speed if drone.cruise_speed > 0 else float("inf")
        eta_at_customer = current_time + time_to_available + time_to_customer

        if eta_at_customer > order.deadline:
            return bids

        if diagnostics is not None:
            diagnostics["relay_candidates"] = diagnostics.get("relay_candidates", 0) + 1

        bid = AllocationResult(
            order_id=order.order_id,
            vehicle_id=depot.depot_id,
            mode="C",
            distance=dist_to_customer + dist_to_depot,
            feasible=True,
            recovery_station_id=depot.depot_id,
            drone_id=drone.drone_id,
            launch_time=current_time + time_to_available,
        )
        bid._relay_origin = relay_origin
        bid._is_relay = True

        scored = self._score_standard_bid(bid, order, current_time)
        if scored is not None:
            scored._relay_origin = relay_origin
            scored._is_relay = True
            bids.append(scored)
            if diagnostics is not None:
                diagnostics["relay_feasible"] = diagnostics.get("relay_feasible", 0) + 1

        return bids

    def _build_delivering_preschedule_bids(
        self,
        drone: "Drone",
        delivery_endpoint: "Position3D",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """DELIVERING 无人机预调度竞标：以当前任务送货点为起点。"""
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return []
        depot = depots[0]
        bids: list[AllocationResult] = []

        remaining_energy = self._estimate_relay_remaining_energy(drone, delivery_endpoint)
        if remaining_energy <= 0:
            return bids

        dist_to_customer = self._dist(delivery_endpoint, order.delivery_loc)
        dist_to_depot = self._dist(order.delivery_loc, depot.location)
        energy_to_customer = self._flight_energy(
            drone, delivery_endpoint, order.delivery_loc, order.payload_weight
        )
        energy_to_depot = self._flight_energy(
            drone, order.delivery_loc, depot.location, 0.0
        )
        energy_needed = (energy_to_customer + energy_to_depot) * self.ENERGY_SAFETY_FACTOR
        if energy_needed > remaining_energy:
            return bids

        t_remain = self._estimate_remaining_flight_time(drone, current_time)
        eta_at_customer = current_time + t_remain + self.delivery_service_time + (
            dist_to_customer / drone.cruise_speed if drone.cruise_speed > 0 else float("inf")
        )
        if eta_at_customer > order.deadline:
            return bids

        if diagnostics is not None:
            diagnostics["relay_candidates"] = diagnostics.get("relay_candidates", 0) + 1

        bid = AllocationResult(
            order_id=order.order_id,
            vehicle_id=depot.depot_id,
            mode="C",
            distance=dist_to_customer + dist_to_depot,
            feasible=True,
            recovery_station_id=depot.depot_id,
            drone_id=drone.drone_id,
            launch_time=current_time + t_remain + self.delivery_service_time,
        )
        bid._relay_origin = delivery_endpoint
        bid._is_relay = True

        scored = self._score_standard_bid(bid, order, current_time)
        if scored is not None:
            scored._relay_origin = delivery_endpoint
            scored._is_relay = True
            bids.append(scored)
            if diagnostics is not None:
                diagnostics["relay_feasible"] = diagnostics.get("relay_feasible", 0) + 1

        return bids

    def _validate_truck_fleet_safety(
        self,
        truck: "Truck",
        order: "Order",
        current_time: float,
    ) -> bool:
        """卡车准入的契约冲突校验：新单路径偏移不得导致已挂载/已起飞无人机错过回收生死线。

        校验范围：
          1. 已有契约的时间窗（_validate_truck_insertion 已覆盖）
          2. 车上正在运输的无人机的起飞站到达时间
          3. 正在空中等待回收的无人机的电量续航
        """
        if not self._validate_truck_insertion(truck, order, current_time):
            return False

        if truck.speed <= 0:
            return False

        _, _, detour_time = self._find_best_insertion_for_truck(truck, order, current_time)
        if not math.isfinite(detour_time):
            detour_time = (
                self._dist(truck.get_location(current_time), order.delivery_loc)
                / truck.speed + self.SERVICE_TIME_CUSTOMER
            )

        for drone in self.entity_mgr.drones.values():
            if getattr(drone, "transport_truck_id", None) != truck.truck_id:
                continue

            launch_time = float(getattr(drone, "scheduled_launch_time", 0.0))
            if launch_time <= 0:
                continue

            # 绕路会延迟到达起飞站，需确保不超过合理范围
            new_launch_time = launch_time + detour_time
            if drone.has_pending_route and drone.route_plan:
                from core.entities.primitives import WaypointAction
                for wp in drone.route_plan:
                    if wp.action == WaypointAction.DELIVER:
                        flight_time = (
                            self._dist(wp.loc, drone.route_plan[-1].loc) / drone.cruise_speed
                            if drone.cruise_speed > 0 else float("inf")
                        )
                        # 检查绕路后无人机是否仍能按时完成任务
                        order_obj = None
                        if self._order_mgr:
                            order_obj = self._order_mgr.assigned_orders.get(
                                getattr(drone, "carrying_order_id", "") or ""
                            )
                        if order_obj and hasattr(order_obj, "deadline"):
                            if new_launch_time + flight_time > order_obj.deadline:
                                logger.debug(
                                    "[MarketBasedSolver] 卡车 %s 接单 %s 导致车载无人机 %s "
                                    "错过订单截止时间",
                                    truck.truck_id, order.order_id, drone.drone_id,
                                )
                                return False
                        break
        return True

    def _best_truck_bid(
        self,
        order: "Order",
        current_time: float,
        truck_last_pos: dict[str, "Position3D"] = None,
    ) -> AllocationResult | None:
        """卡车作为竞标者，对订单提交模式 A 的最低价投标（需通过契约+舰队安全校验）。"""
        trucks = list(self.entity_mgr.trucks.values())
        if not trucks:
            return None

        best_bid: AllocationResult | None = None
        for truck in trucks:
            if not self._validate_truck_fleet_safety(truck, order, current_time):
                continue
            bid = AllocationResult(
                order_id=order.order_id,
                vehicle_id=truck.truck_id,
                mode="A",
                distance=self._dist(truck.get_location(current_time), order.delivery_loc),
                feasible=True,
            )
            scored = self._score_standard_bid(bid, order, current_time, truck_last_pos)
            if scored is None:
                continue
            if best_bid is None or scored.score_total < best_bid.score_total:
                best_bid = scored
        return best_bid

    def _build_depot_bid(
        self,
        drone: "Drone",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> AllocationResult | None:
        """构造模式 C 的无人机投标。"""
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return None

        depot = depots[0]
        if not self._can_drone_start_mode_c_from_depot(drone, depot):
            return None

        energy_out = self._flight_energy(drone, depot.location, order.delivery_loc, order.payload_weight)
        energy_back = self._flight_energy(drone, order.delivery_loc, depot.location, 0.0)
        energy_needed = (energy_out + energy_back) * self.ENERGY_SAFETY_FACTOR
        if energy_needed > drone.battery_current:
            if diagnostics is not None:
                diagnostics["depot_energy_rejected"] += 1
            return None

        bid = AllocationResult(
            order_id=order.order_id,
            vehicle_id=depot.depot_id,
            mode="C",
            distance=self._dist(depot.location, order.delivery_loc),
            feasible=True,
            recovery_station_id=depot.depot_id,
            drone_id=drone.drone_id,
        )
        if diagnostics is not None:
            diagnostics["depot_feasible"] += 1
        return self._score_standard_bid(bid, order, current_time)

    def _is_drone_physically_on_truck(self, drone: "Drone", truck: "Truck") -> bool:
        """与 decision_engine_market 相同语义：docked 或 transport_truck_id 视为在目标卡车上。"""
        if drone.status.is_flying:
            return False
        if getattr(drone, "waiting_recovery_station_id", ""):
            return False
        if drone.drone_id in getattr(truck, "docked_drones", []):
            return True
        return getattr(drone, "transport_truck_id", None) == truck.truck_id

    def _can_drone_start_mode_c_from_depot(self, drone: "Drone", depot: Any) -> bool:
        """Mode C 要求无人机真实可从仓库起飞，防止“仓-空直递”幻觉分配。"""
        if getattr(drone, "transport_truck_id", None):
            return False
        if getattr(drone, "waiting_recovery_station_id", ""):
            return False
        for truck in self.entity_mgr.trucks.values():
            if drone.drone_id in getattr(truck, "docked_drones", []):
                return False
        if self._dist(drone.current_loc, depot.location) > self.MODE_C_DEPOT_TOLERANCE_M:
            return False
        return True

    def _collect_anchor_bids_for_drone(
        self,
        drone: "Drone",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """
        基于卡车广播的锚点时刻表，为单架无人机生成 B_WAIT 投标。

        约束：
          - 起飞锚点与回收锚点均必须属于同一辆卡车的预测锚点序列
          - 回收锚点顺序不得早于起飞锚点
          - 需满足前瞻能量校验
        """
        bids: list[AllocationResult] = []

        for truck in self.entity_mgr.trucks.values():
            # 与 MarketDispatchDecisionEngine 一致：B_WAIT 只能使用当前已在目标卡车
            # 上真实停靠的无人机，否则计划会在执行层被标为不可行。
            if not self._is_drone_physically_on_truck(drone, truck):
                continue
            # B_WAIT / B 模式下卡车不需要绕路到客户，不应使用 _validate_truck_insertion
            # （该校验假设卡车去客户地址，会导致误拒合法的 UAV 协同投标）。
            # 锚点时间可行性已由 _build_anchor_bid 内的 sync 校验覆盖。
            anchors = self._build_anchor_timetable(truck, current_time)
            if not anchors:
                continue

            for launch_idx, launch_anchor in enumerate(anchors):
                for recovery_anchor in anchors[launch_idx:]:
                    if diagnostics is not None:
                        diagnostics["anchor_candidates"] += 1
                    bid = self._build_anchor_bid(
                        drone=drone,
                        truck=truck,
                        order=order,
                        current_time=current_time,
                        launch_anchor=launch_anchor,
                        recovery_anchor=recovery_anchor,
                        diagnostics=diagnostics,
                    )
                    if bid is not None:
                        bids.append(bid)

        return bids

    def _collect_b_dynamic_bids(
        self,
        drone: "Drone",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """站点起飞 + 仓库回收的混合投标（B_DYNAMIC）。

        适用场景：卡车远离仓库，但订单和仓库距离较近时，无人机从卡车锚点
        起飞送达后直飞仓库，无需等待卡车回收，节省等待时间并释放无人机产能。
        """
        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return []
        depot = depots[0]
        bids: list[AllocationResult] = []

        for truck in self.entity_mgr.trucks.values():
            if not self._is_drone_physically_on_truck(drone, truck):
                continue
            anchors = self._build_anchor_timetable(truck, current_time)
            if not anchors:
                continue

            for launch_anchor in anchors:
                if diagnostics is not None:
                    diagnostics["b_dynamic_candidates"] += 1

                launch_loc = launch_anchor["location"]
                launch_time = launch_anchor["eta"] + self.TRUCK_DRONE_LAUNCH_TIME

                dist_out = self._dist(launch_loc, order.delivery_loc)
                dist_to_depot = self._dist(order.delivery_loc, depot.location)

                energy_out_j = self._flight_energy(
                    drone, launch_loc, order.delivery_loc, order.payload_weight
                )
                energy_to_depot_j = self._flight_energy(
                    drone, order.delivery_loc, depot.location, 0.0
                )
                total_energy_j = (energy_out_j + energy_to_depot_j) * self.ENERGY_SAFETY_FACTOR
                if total_energy_j > drone.battery_current:
                    if diagnostics is not None:
                        diagnostics["b_dynamic_energy_rejected"] += 1
                    continue

                bid = AllocationResult(
                    order_id=order.order_id,
                    vehicle_id=truck.truck_id,
                    mode="B_DYNAMIC",
                    distance=dist_out,
                    feasible=True,
                    recovery_station_id=depot.depot_id,
                    drone_id=drone.drone_id,
                    launch_station_id=launch_anchor["anchor_id"],
                    launch_time=launch_time,
                    wait_duration=(
                        self.TRUCK_DRONE_LAUNCH_TIME
                        + dist_out / drone.cruise_speed
                        + self.delivery_service_time
                        + dist_to_depot / drone.cruise_speed
                    ),
                )
                scored = self._score_standard_bid(bid, order, current_time)
                if scored is not None:
                    if diagnostics is not None:
                        diagnostics["b_dynamic_feasible"] += 1
                    bids.append(scored)

        return bids

    def _get_drone_relay_origin(self, drone) -> "Position3D | None":
        """获取无人机串联竞标的起点（当前路径中最后一个 DELIVER 点）。"""
        from core.entities.primitives import WaypointAction
        last_deliver = None
        for wp in drone.route_plan:
            if wp.action == WaypointAction.DELIVER:
                last_deliver = wp.loc
        return last_deliver

    # ══════════════════════════════════════════════════════════════════════════
    # 任务闭环约束 (Global Completeness)
    # ══════════════════════════════════════════════════════════════════════════

    def _validate_drone_route_closure(self, plan: DispatchPlan) -> None:
        """闭环强制要求：任何无人机路径计划的最后一个 Waypoint 必须是 DOCK_DEPOT 或 DOCK_TRUCK。

        若检测到违规分配（最终航路点不是归巢/回车动作），则在日志中警告。
        此校验在计划返回前执行，作为最后的安全网。
        """
        from core.entities.primitives import WaypointAction

        for alloc in plan.allocations:
            if not alloc.feasible or alloc.mode == "A" or not alloc.drone_id:
                continue

            drone = self.entity_mgr.drones.get(alloc.drone_id)
            if drone is None:
                continue

            route = getattr(drone, "route_plan", None) or []
            if not route:
                continue

            last_action = route[-1].action
            if last_action not in (WaypointAction.DOCK_DEPOT, WaypointAction.DOCK_TRUCK):
                logger.warning(
                    "[MarketBasedSolver][Closure] 无人机 %s 路径闭环校验失败: "
                    "最后航路点动作=%s，应为 DOCK_DEPOT 或 DOCK_TRUCK。"
                    "订单=%s 模式=%s",
                    alloc.drone_id, last_action.value,
                    alloc.order_id, alloc.mode,
                )

    def _ensure_idle_drones_return_to_depot(self, current_time: float) -> None:
        """待命逻辑：若无人机在充电站完成配送且无后续任务，自动生成 ReturnToDepot 路径。

        触发条件：
          - 无人机状态为 IDLE
          - 没有挂载订单、没有待执行路径
          - 不在仓库、不在卡车上、不在等待回收
        """
        from core.entities.primitives import WaypointAction, RouteWaypoint

        depots = list(self.entity_mgr.depots.values())
        if not depots:
            return
        depot = depots[0]

        for drone in self.entity_mgr.drones.values():
            if drone.status != DroneStatus.IDLE:
                continue
            if getattr(drone, "carrying_order_id", None):
                continue
            if getattr(drone, "has_pending_route", False):
                continue
            if getattr(drone, "transport_truck_id", None):
                continue
            if getattr(drone, "waiting_recovery_station_id", ""):
                continue
            if getattr(drone, "pending_release_order_id", None):
                continue

            dist_to_depot = self._dist(drone.current_loc, depot.location)
            if dist_to_depot < 5.0:
                continue

            energy_needed = self._flight_energy(
                drone, drone.current_loc, depot.location, 0.0
            ) * self.ENERGY_SAFETY_FACTOR
            if energy_needed > drone.battery_current:
                logger.warning(
                    "[MarketBasedSolver][Closure] 无人机 %s 电量不足以返回仓库 "
                    "(需%.1fJ > 当前%.1fJ)，标记为滞留",
                    drone.drone_id, energy_needed, drone.battery_current,
                )
                continue

            waypoints = [
                RouteWaypoint(depot.location, WaypointAction.DOCK_DEPOT, depot.depot_id),
            ]
            drone.set_route(waypoints)
            drone.status = DroneStatus.RETURNING_TO_DEPOT
            logger.info(
                "[MarketBasedSolver][Closure] 无人机 %s 无后续任务，自动生成归巢路径 "
                "(距仓库 %.1fm)",
                drone.drone_id, dist_to_depot,
            )

    def should_truck_return_to_depot(self, truck_id: str) -> bool:
        """卡车清空校验：判断卡车是否满足返回仓库条件。

        判定公式：
          Status = (Pending_Orders == 0) ∧ (Assigned_Orders_In_Fleet == 0)

        只要还有无人机在执行任务或等待回收，卡车必须保持在动态锚点或行驶状态。
        """
        truck = self.entity_mgr.trucks.get(truck_id)
        if truck is None:
            return False

        # 检查 1：是否还有未完成的待分配订单
        if self._order_mgr:
            if self._order_mgr.pending_orders:
                return False

        # 检查 2：是否还有本车相关的活跃契约（有无人机在飞行中）
        active_contracts = self._get_truck_contracts(truck_id)
        if active_contracts:
            return False

        # 检查 3：是否还有无人机与本卡车绑定（挂载、在途、等待回收）
        for drone in self.entity_mgr.drones.values():
            if getattr(drone, "transport_truck_id", None) == truck_id:
                return False
            if getattr(drone, "waiting_recovery_station_id", ""):
                for contract in self._active_contracts:
                    if (contract.truck_id == truck_id
                            and contract.drone_id == drone.drone_id
                            and contract.status == "active"):
                        return False

        # 检查 4：是否还有任何无人机正在飞行（全局级清空）
        for drone in self.entity_mgr.drones.values():
            if drone.status.is_flying:
                return False
            if getattr(drone, "carrying_order_id", None):
                return False

        # 检查 5：是否还有 assigned 订单未完成
        if self._order_mgr:
            if self._order_mgr.assigned_orders:
                return False

        return True

    def _validate_truck_premature_return(self, plan: DispatchPlan, current_time: float) -> None:
        """在计划返回前校验卡车是否会提前回仓，防止无人机失去回收载体。"""
        for truck_id, route in plan.truck_routes.items():
            if not route.nodes:
                continue
            last_node = route.nodes[-1]
            if last_node.node_type == "depot" and "_return" in last_node.node_id:
                if not self.should_truck_return_to_depot(truck_id):
                    logger.warning(
                        "[MarketBasedSolver][Closure] 卡车 %s 计划提前返回仓库，"
                        "但仍有活跃任务/契约未完成，正在移除回仓信号",
                        truck_id,
                    )
                    # 移除最后一个 _return 节点
                    route.nodes.pop()
                    if plan.allocations:  # 尝试将卡车的总距离等还原一下（粗略）
                        route.total_distance = sum(
                            route.geometry[i - 1].distance_2d(route.geometry[i])
                            for i in range(1, len(route.geometry))
                            # 但完整的卡车几何路径不好裁剪，我们简单处理：后续逻辑由 decision engine 控制物理轨迹
                        )

    def _estimate_relay_remaining_energy(self, drone, relay_origin) -> float:
        """估算无人机到达串联起点时的剩余电量。"""
        current_energy = drone.battery_current
        if not drone.has_pending_route:
            return current_energy

        # 估算完成当前路径的能耗
        pos = drone.current_loc
        for i in range(drone.current_waypoint_index, len(drone.route_plan)):
            wp = drone.route_plan[i]
            energy = self._flight_energy(drone, pos, wp.loc, drone.current_payload)
            current_energy -= energy
            pos = wp.loc
            if current_energy <= 0:
                return 0.0
        return max(0.0, current_energy)

    def _estimate_relay_available_time(self, drone, relay_origin, current_time: float) -> float:
        """估算无人机到达串联起点的时间（完成当前路径所需时间）。"""
        if not drone.has_pending_route:
            return 0.0

        total_dist = 0.0
        pos = drone.current_loc
        for i in range(drone.current_waypoint_index, len(drone.route_plan)):
            wp = drone.route_plan[i]
            total_dist += self._dist(pos, wp.loc)
            pos = wp.loc
        return total_dist / drone.cruise_speed + self.delivery_service_time

    def _collect_moving_truck_bids(
        self,
        drone: "Drone",
        order: "Order",
        current_time: float,
        diagnostics: dict | None = None,
    ) -> list[AllocationResult]:
        """在允许的情况下，生成从卡车当前位置直接起飞的模式 B 投标。"""
        bids: list[AllocationResult] = []

        for truck in self.entity_mgr.trucks.values():
            if not self._is_drone_physically_on_truck(drone, truck):
                continue
            truck_loc = truck.get_location(current_time)
            anchors = self._build_anchor_timetable(truck, current_time)
            if not anchors:
                continue

            truck_distance_to_launch = 0.0
            launch_time = current_time + self.TRUCK_DRONE_LAUNCH_TIME
            dist_out = self._dist(truck_loc, order.delivery_loc)
            energy_out_j = self._flight_energy(drone, truck_loc, order.delivery_loc, order.payload_weight)

            for recovery_anchor in anchors:
                if diagnostics is not None:
                    diagnostics["moving_candidates"] += 1
                recovery_loc = recovery_anchor["location"]
                dist_back = self._dist(order.delivery_loc, recovery_loc)
                energy_back_j = self._flight_energy(drone, order.delivery_loc, recovery_loc, 0.0)
                total_energy_j = (energy_out_j + energy_back_j) * self.ENERGY_SAFETY_FACTOR
                if total_energy_j > drone.battery_current:
                    if diagnostics is not None:
                        diagnostics["moving_energy_rejected"] += 1
                    continue

                uav_arrival_recovery = (
                    launch_time
                    + dist_out / drone.cruise_speed
                    + self.delivery_service_time
                    + dist_back / drone.cruise_speed
                )
                latest_rendezvous = recovery_anchor["latest_rendezvous_time"]
                if uav_arrival_recovery > latest_rendezvous:
                    if diagnostics is not None:
                        diagnostics["moving_sync_rejected"] += 1
                    continue
                bid = AllocationResult(
                    order_id=order.order_id,
                    vehicle_id=truck.truck_id,
                    mode="B",
                    distance=dist_out,
                    feasible=True,
                    recovery_station_id=recovery_anchor["anchor_id"],
                    drone_id=drone.drone_id,
                )
                scored = self._score_standard_bid(bid, order, current_time)
                if scored is not None:
                    if diagnostics is not None:
                        diagnostics["moving_feasible"] += 1
                    bids.append(scored)

        return bids

    # ══════════════════════════════════════════════════════════════════════════
    # 锚点时刻表
    # ══════════════════════════════════════════════════════════════════════════

    def _build_anchor_timetable(
        self,
        truck: "Truck",
        current_time: float,
    ) -> list[dict]:
        """基于卡车真实主干路径或运行时计划生成锚点时刻表，
        并合并活跃契约的硬约束锚点、对非锁定锚点施加缓冲时间。"""
        if truck.speed <= 0:
            return []

        anchors: list[dict] = []

        # ── 第一优先级：预览路线 ─────────────────────────────────────
        preview_route = self._anchor_preview_routes.get(truck.truck_id)
        if preview_route is not None:
            for node in preview_route.nodes:
                if node.node_type != "station":
                    continue
                latest_rendezvous_time = node.departure_time + self.T_MAX_WAIT
                anchors.append(
                    {
                        "index": len(anchors),
                        "anchor_id": node.node_id,
                        "location": node.position,
                        "eta": node.arrival_time,
                        "arrival_time": node.arrival_time,
                        "latest_rendezvous_time": latest_rendezvous_time,
                        "departure_time": latest_rendezvous_time + self.TRUCK_DRONE_RECOVER_TIME,
                        "distance_from_truck": max(
                            0.0,
                            (node.arrival_time - current_time) * truck.speed,
                        ),
                        "locked": False,
                    }
                )
            if anchors:
                return self._merge_contracts_into_timetable(anchors, truck, current_time)

        # ── 第二优先级：运行时停靠计划 ──────────────────────────────
        planned_stops = getattr(truck, "_planned_route_stops", None)
        if planned_stops:
            for stop in planned_stops:
                if stop.get("node_type") != "station":
                    continue
                position = stop.get("position")
                arrival_time = float(stop.get("arrival_time", current_time))
                latest_rendezvous_time = arrival_time + self.T_MAX_WAIT
                anchors.append(
                    {
                        "index": len(anchors),
                        "anchor_id": str(stop.get("node_id")),
                        "location": position,
                        "eta": arrival_time,
                        "arrival_time": arrival_time,
                        "latest_rendezvous_time": latest_rendezvous_time,
                        "departure_time": latest_rendezvous_time + self.TRUCK_DRONE_RECOVER_TIME,
                        "distance_from_truck": max(0.0, (arrival_time - current_time) * truck.speed),
                        "locked": False,
                    }
                )
            if anchors:
                return self._merge_contracts_into_timetable(anchors, truck, current_time)

        # ── 第三优先级：兼容回退（旧启发式） ────────────────────────
        truck_loc = truck.get_location(current_time)
        predicted_stations = self._predict_truck_charging_stations(truck, current_time)

        prev_loc = truck_loc
        cumulative_distance = 0.0
        for index, (anchor_id, anchor_loc) in enumerate(predicted_stations):
            leg_distance = self._dist(prev_loc, anchor_loc)
            cumulative_distance += leg_distance
            arrival_time = current_time + cumulative_distance / truck.speed
            latest_rendezvous_time = arrival_time + self.T_MAX_WAIT
            anchors.append(
                {
                    "index": index,
                    "anchor_id": anchor_id,
                    "location": anchor_loc,
                    "eta": arrival_time,
                    "arrival_time": arrival_time,
                    "latest_rendezvous_time": latest_rendezvous_time,
                    "departure_time": latest_rendezvous_time + self.TRUCK_DRONE_RECOVER_TIME,
                    "distance_from_truck": cumulative_distance,
                    "locked": False,
                }
            )
            prev_loc = anchor_loc

        return self._merge_contracts_into_timetable(anchors, truck, current_time)

    def _merge_contracts_into_timetable(
        self,
        anchors: list[dict],
        truck: "Truck",
        current_time: float,
    ) -> list[dict]:
        """将活跃契约的硬约束锚点合并入时刻表，并对非锁定锚点施加缓冲时间。"""
        contracts = self._get_truck_contracts(truck.truck_id)

        # 合并契约锁定的锚点
        existing_ids = {a["anchor_id"] for a in anchors}
        for contract in contracts:
            if contract.anchor_id in existing_ids:
                for anchor in anchors:
                    if anchor["anchor_id"] == contract.anchor_id:
                        anchor["locked"] = True
                        anchor["latest_rendezvous_time"] = min(
                            anchor["latest_rendezvous_time"],
                            contract.latest_departure,
                        )
                        break
            else:
                station = (
                    self.entity_mgr.stations.get(contract.anchor_id)
                    or self.entity_mgr.depots.get(contract.anchor_id)
                )
                if station is not None:
                    anchors.append(
                        {
                            "index": len(anchors),
                            "anchor_id": contract.anchor_id,
                            "location": station.location,
                            "eta": contract.arrival_time,
                            "arrival_time": contract.arrival_time,
                            "latest_rendezvous_time": contract.latest_departure,
                            "departure_time": contract.latest_departure + self.TRUCK_DRONE_RECOVER_TIME,
                            "distance_from_truck": max(
                                0.0,
                                (contract.arrival_time - current_time) * truck.speed,
                            ),
                            "locked": True,
                        }
                    )

        # 对非锁定锚点施加缓冲时间（Slack Time）
        for anchor in anchors:
            if not anchor.get("locked"):
                anchor["latest_rendezvous_time"] = (
                    anchor["arrival_time"]
                    + self.T_MAX_WAIT * (1 - self.TIMETABLE_SLACK_RATIO)
                )

        anchors.sort(key=lambda a: a["arrival_time"])
        for i, anchor in enumerate(anchors):
            anchor["index"] = i

        return anchors

    # ══════════════════════════════════════════════════════════════════════════
    # 单条投标构建
    # ══════════════════════════════════════════════════════════════════════════

    def _build_anchor_bid(
        self,
        drone: "Drone",
        truck: "Truck",
        order: "Order",
        current_time: float,
        launch_anchor: dict,
        recovery_anchor: dict,
        diagnostics: dict | None = None,
    ) -> AllocationResult | None:
        """对固定的"卡车-起飞锚点-回收锚点-无人机"组合生成一条投标。"""
        launch_loc: "Position3D" = launch_anchor["location"]
        recovery_loc: "Position3D" = recovery_anchor["location"]
        truck_distance_to_launch = launch_anchor["distance_from_truck"]
        launch_time = launch_anchor["eta"] + self.TRUCK_DRONE_LAUNCH_TIME

        dist_out = self._dist(launch_loc, order.delivery_loc)
        dist_back = self._dist(order.delivery_loc, recovery_loc)

        energy_out_j = self._flight_energy(drone, launch_loc, order.delivery_loc, order.payload_weight)
        energy_back_j = self._flight_energy(drone, order.delivery_loc, recovery_loc, 0.0)
        total_energy_j = (energy_out_j + energy_back_j) * self.ENERGY_SAFETY_FACTOR
        if total_energy_j > drone.battery_current:
            if diagnostics is not None:
                diagnostics["anchor_energy_rejected"] += 1
            return None

        uav_arrival_recovery = (
            launch_time
            + dist_out / drone.cruise_speed
            + self.delivery_service_time
            + dist_back / drone.cruise_speed
        )
        latest_rendezvous = recovery_anchor["latest_rendezvous_time"]
        if uav_arrival_recovery > latest_rendezvous:
            if diagnostics is not None:
                diagnostics["anchor_sync_rejected"] += 1
            return None
        bid = AllocationResult(
            order_id=order.order_id,
            vehicle_id=truck.truck_id,
            mode="B_WAIT",
            distance=dist_out,
            feasible=True,
            recovery_station_id=recovery_anchor["anchor_id"],
            drone_id=drone.drone_id,
            launch_station_id=launch_anchor["anchor_id"],
            launch_time=launch_time,
            wait_duration=(
                self.TRUCK_DRONE_LAUNCH_TIME
                + dist_out / drone.cruise_speed
                + self.delivery_service_time
                + dist_back / drone.cruise_speed
            ),
        )
        if diagnostics is not None:
            diagnostics["anchor_feasible"] += 1
        return self._score_standard_bid(bid, order, current_time)

    # ══════════════════════════════════════════════════════════════════════════
    # 路径拼接约束 (Path Stitching Constraints)
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_path_stitching_constraints(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
    ) -> AllocationResult:
        """对中标结果执行路径拼接约束校验。

        1. 预测起点对齐：移动中实体的起点预测为 P(t + delta_t)
        2. 坐标跳变校验：新路径首帧与旧路径末帧误差必须 < PATH_STITCH_TOLERANCE_M
        3. 若误差超标，系统自动标记需要线性插值补全
        """
        drone = self.entity_mgr.drones.get(alloc.drone_id)
        if drone is None:
            return alloc

        is_relay = getattr(alloc, "_is_relay", False)

        if is_relay and drone.status.is_flying:
            # 串联任务：校验新路径起点与旧路径末帧的坐标连续性
            old_endpoint = self._get_drone_route_endpoint(drone)
            relay_origin = getattr(alloc, "_relay_origin", None)
            if old_endpoint is not None and relay_origin is not None:
                gap = self._dist(old_endpoint, relay_origin)
                if gap > self.PATH_STITCH_TOLERANCE_M:
                    logger.info(
                        "[MarketBasedSolver][PathStitch] 无人机 %s 串联路径坐标跳变 %.2fm > %.2fm，"
                        "将在编排层执行线性插值补全",
                        alloc.drone_id, gap, self.PATH_STITCH_TOLERANCE_M,
                    )
                    alloc._needs_interpolation = True
                    alloc._interpolation_from = old_endpoint
                    alloc._interpolation_to = relay_origin

        elif drone.status.is_flying:
            # 非串联的移动中实体：使用预测起点
            predicted_pos = self._predict_drone_position(
                drone, current_time, self.AUCTION_COMPUTE_DELTA_T
            )
            launch_loc = self._resolve_alloc_launch_loc(alloc)
            if launch_loc is not None:
                gap = self._dist(predicted_pos, launch_loc)
                if gap > self.PATH_STITCH_TOLERANCE_M:
                    logger.debug(
                        "[MarketBasedSolver][PathStitch] 无人机 %s 预测位置与发射点偏差 %.2fm",
                        alloc.drone_id, gap,
                    )

        return alloc

    def _resolve_alloc_launch_loc(self, alloc: AllocationResult) -> "Position3D | None":
        """从分配结果中解析发射点坐标。"""
        if alloc.launch_station_id:
            station = self.entity_mgr.stations.get(alloc.launch_station_id)
            if station is not None:
                return station.location
        if alloc.mode == "C":
            depot = self.entity_mgr.depots.get(alloc.vehicle_id)
            if depot is not None:
                return depot.location
        if alloc.mode == "B":
            truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
            if truck is not None:
                return truck.current_loc
        return None

    def validate_path_continuity(
        self,
        drone_id: str,
        new_waypoints: list,
        current_time: float,
    ) -> tuple[bool, float]:
        """外部编排层调用：校验新路径段首帧与无人机当前位置/路径末帧的连续性。

        Returns:
            (is_valid, gap_meters) — 是否满足容差，以及实际间隙距离
        """
        drone = self.entity_mgr.drones.get(drone_id)
        if drone is None or not new_waypoints:
            return True, 0.0

        if drone.status.is_flying:
            reference_pos = self._predict_drone_position(
                drone, current_time, self.AUCTION_COMPUTE_DELTA_T
            )
        else:
            reference_pos = drone.current_loc

        first_wp_loc = new_waypoints[0].loc if hasattr(new_waypoints[0], "loc") else new_waypoints[0]
        gap = self._dist(reference_pos, first_wp_loc)
        return gap <= self.PATH_STITCH_TOLERANCE_M, gap

    # ══════════════════════════════════════════════════════════════════════════
    # 评分
    # ══════════════════════════════════════════════════════════════════════════

    def _score_standard_bid(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
        truck_last_pos: dict[str, "Position3D"] = None,
    ) -> AllocationResult | None:
        """
        统一复用贪心基线中的目标函数打分（距离 + 能耗 + 时间窗迟罚）。

        对市场算法中的 B / B_WAIT，卡车主干路线属于既定基础设施成本，
        因此仅将"额外引入的边际卡车成本"计入协同任务评分，而不再把
        "卡车当前位置 -> 起飞锚点"的整段主干路径全量重复计费。

        注：本阶段不叠加 cost_wait / cost_risk、回头路 / 绕路放大 / 路径邻近奖，
        仅保留加权距离、能耗与 deadline 迟罚（与 B_DYNAMIC 及 Mode A 基线一致）。
        """
        if alloc.mode in ("B", "B_WAIT"):
            score_total, cost_dist, cost_energy, cost_penalty = self._score_market_b_bid(
                alloc, order, current_time
            )
        elif alloc.mode == "B_DYNAMIC":
            score_total, cost_dist, cost_energy, cost_penalty = self._score_b_dynamic_bid(
                alloc, order, current_time
            )
        else:
            score_total, cost_dist, cost_energy, cost_penalty = self._score_allocation(
                alloc, order, current_time, truck_last_pos or {}
            )

        if not math.isfinite(score_total):
            return None
        alloc.score_total = score_total
        alloc.cost_dist = cost_dist
        alloc.cost_energy = cost_energy
        alloc.cost_penalty = cost_penalty
        return alloc

    def _score_market_b_bid(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
    ) -> tuple[float, float, float, float]:
        """
        市场拍卖中的协同模式（B / B_WAIT）评分。

        当前与 B_DYNAMIC 一致：仅 **加权距离 + 能耗 + 时间窗迟罚**；
        暂不纳入 cost_wait / cost_risk、回头路 / 绕路放大 / 路径邻近奖（见 `_score_standard_bid` 说明）。

        基础成本：
          - UAV 飞行距离与飞行能耗按基线全量计入
          - 卡车只计"边际增量成本"，若锚点本就在卡车主干路线上，则增量视为 0
        """
        drone = self.entity_mgr.drones.get(alloc.drone_id)
        if drone is None or drone.cruise_speed <= 0:
            return float("inf"), 0.0, 0.0, 0.0

        truck_distance = self._estimate_incremental_truck_distance(alloc, current_time)
        if not math.isfinite(truck_distance):
            return float("inf"), 0.0, 0.0, 0.0
        truck_energy = self._truck_energy_wh(truck_distance)

        if alloc.mode == "B_WAIT":
            truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
            launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
            if truck is None or launch_station is None:
                return float("inf"), 0.0, 0.0, 0.0
            launch_loc = launch_station.location
            launch_time_est = (
                alloc.launch_time
                if alloc.launch_time > current_time
                else current_time + self.TRUCK_DRONE_LAUNCH_TIME
            )
        else:
            truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
            if truck is None:
                return float("inf"), 0.0, 0.0, 0.0
            launch_loc = truck.get_location(current_time)
            launch_time_est = current_time + self.TRUCK_DRONE_LAUNCH_TIME

        recovery = (
            self.entity_mgr.stations.get(alloc.recovery_station_id)
            or self.entity_mgr.depots.get(alloc.recovery_station_id)
        )
        if recovery is None:
            return float("inf"), 0.0, 0.0, 0.0

        dist_out = self._dist(launch_loc, order.delivery_loc)
        dist_back = self._dist(order.delivery_loc, recovery.location)
        uav_distance = dist_out + dist_back
        uav_energy = self._uav_energy_wh(drone, launch_loc, order.delivery_loc, order.payload_weight)
        uav_energy += self._uav_energy_wh(drone, order.delivery_loc, recovery.location, 0.0)
        delivery_time_est = launch_time_est + dist_out / drone.cruise_speed + self.delivery_service_time

        cost_dist = self.C_DIST_ET * truck_distance + self.C_DIST_UAV * uav_distance
        cost_energy = self.C_ENERGY_ET * truck_energy + self.C_ENERGY_UAV * uav_energy
        lateness = max(0.0, delivery_time_est - order.deadline)
        cost_penalty = self.LAMBDA_TIME * order.penalty_rate * lateness

        score_total = cost_dist + cost_energy + cost_penalty
        return score_total, cost_dist, cost_energy, cost_penalty

    def _score_b_dynamic_bid(
        self,
        alloc: AllocationResult,
        order: "Order",
        current_time: float,
    ) -> tuple[float, float, float, float]:
        """B_DYNAMIC 评分：站点起飞 + 仓库回收，仅距离 + 能耗 + 时间窗迟罚。

        无人机从锚点起飞送达后直飞仓库，无需与卡车同步：
          - 卡车只计 launch 段边际距离（锚点在路线上则为 0）
          - UAV 计全量距离/能耗（launch→customer→depot）
        """
        drone = self.entity_mgr.drones.get(alloc.drone_id)
        if drone is None or drone.cruise_speed <= 0:
            return float("inf"), 0.0, 0.0, 0.0

        depot = self.entity_mgr.depots.get(alloc.recovery_station_id)
        if depot is None:
            return float("inf"), 0.0, 0.0, 0.0

        launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
        if launch_station is None:
            return float("inf"), 0.0, 0.0, 0.0

        launch_loc = launch_station.location

        truck_distance = self._estimate_incremental_truck_distance(alloc, current_time)
        if not math.isfinite(truck_distance):
            return float("inf"), 0.0, 0.0, 0.0
        truck_energy = self._truck_energy_wh(truck_distance)

        dist_out = self._dist(launch_loc, order.delivery_loc)
        dist_to_depot = self._dist(order.delivery_loc, depot.location)
        uav_distance = dist_out + dist_to_depot
        uav_energy = self._uav_energy_wh(drone, launch_loc, order.delivery_loc, order.payload_weight)
        uav_energy += self._uav_energy_wh(drone, order.delivery_loc, depot.location, 0.0)

        launch_time_est = (
            alloc.launch_time
            if alloc.launch_time > current_time
            else current_time + self.TRUCK_DRONE_LAUNCH_TIME
        )
        delivery_time_est = launch_time_est + dist_out / drone.cruise_speed + self.delivery_service_time

        cost_dist = self.C_DIST_ET * truck_distance + self.C_DIST_UAV * uav_distance
        cost_energy = self.C_ENERGY_ET * truck_energy + self.C_ENERGY_UAV * uav_energy
        lateness = max(0.0, delivery_time_est - order.deadline)
        cost_penalty = self.LAMBDA_TIME * order.penalty_rate * lateness

        score_total = cost_dist + cost_energy + cost_penalty
        return score_total, cost_dist, cost_energy, cost_penalty

    def _estimate_incremental_truck_distance(
        self,
        alloc: AllocationResult,
        current_time: float,
    ) -> float:
        """
        估算协同任务给卡车带来的边际增量距离。

        若起飞/回收锚点已属于卡车真实主干路线，则卡车为该协同任务新增的
        行驶距离视为 0；否则退化回旧口径进行保守估算。
        """
        if alloc.mode == "B":
            return 0.0

        if alloc.mode not in ("B_WAIT", "B_DYNAMIC"):
            return 0.0

        truck = self.entity_mgr.trucks.get(alloc.vehicle_id)
        launch_station = self.entity_mgr.stations.get(alloc.launch_station_id)
        if truck is None or launch_station is None:
            return float("inf")

        anchor_ids = {
            anchor["anchor_id"]
            for anchor in self._build_anchor_timetable(truck, current_time)
        }

        if alloc.mode == "B_DYNAMIC":
            if alloc.launch_station_id and alloc.launch_station_id in anchor_ids:
                return 0.0
            return self._dist(truck.get_location(current_time), launch_station.location)

        if (
            alloc.launch_station_id
            and alloc.recovery_station_id
            and alloc.launch_station_id in anchor_ids
            and alloc.recovery_station_id in anchor_ids
        ):
            return 0.0

        return self._dist(truck.get_location(current_time), launch_station.location)

    # ══════════════════════════════════════════════════════════════════════════
    # 诊断日志
    # ══════════════════════════════════════════════════════════════════════════

    def _log_bid_diagnostics(
        self,
        order: "Order",
        bids: list[AllocationResult],
        selected: AllocationResult | None,
        drone_diag: dict,
    ) -> None:
        """打印每单投标明细与无人机候选过滤统计。"""
        active_contracts = len([c for c in self._active_contracts if c.status == "active"])
        logger.info(
            "[MarketBasedSolver][Diag] 订单 %s payload=%.2fkg deadline=%.1f bids=%d contracts=%d",
            order.order_id,
            order.payload_weight,
            order.deadline,
            len(bids),
            active_contracts,
        )

        if drone_diag.get("forced_to_truck"):
            logger.info(
                "[MarketBasedSolver][Diag] 订单 %s 强制卡车：payload=%.2fkg > max_drone_payload=%.2fkg",
                order.order_id,
                order.payload_weight,
                drone_diag.get("max_drone_payload", 0.0),
            )
        else:
            logger.info(
                "[MarketBasedSolver][Diag] 订单 %s 无人机候选：idle=%d eligible=%d "
                "excluded_allocated=%d excluded_payload=%d excluded_locked=%d",
                order.order_id,
                drone_diag.get("idle_drones", 0),
                drone_diag.get("eligible_drones", 0),
                drone_diag.get("excluded_allocated", 0),
                drone_diag.get("excluded_payload", 0),
                drone_diag.get("excluded_locked", 0),
            )

            for item in drone_diag.get("drones", []):
                logger.info(
                    "[MarketBasedSolver][Diag] 订单 %s 无人机 %s: "
                    "anchor %d/%d (energy_rej=%d sync_rej=%d) | "
                    "depot %d (energy_rej=%d) | "
                    "b_dynamic %d/%d (energy_rej=%d) | "
                    "moving %d/%d (energy_rej=%d sync_rej=%d)",
                    order.order_id,
                    item["drone_id"],
                    item["anchor_feasible"],
                    item["anchor_candidates"],
                    item["anchor_energy_rejected"],
                    item["anchor_sync_rejected"],
                    item["depot_feasible"],
                    item["depot_energy_rejected"],
                    item.get("b_dynamic_feasible", 0),
                    item.get("b_dynamic_candidates", 0),
                    item.get("b_dynamic_energy_rejected", 0),
                    item["moving_feasible"],
                    item["moving_candidates"],
                    item["moving_energy_rejected"],
                    item["moving_sync_rejected"],
                )

        for bid in sorted(bids, key=lambda x: x.score_total):
            logger.info(
                "[MarketBasedSolver][Bid] 订单 %s mode=%s vehicle=%s drone=%s "
                "launch=%s recovery=%s score=%.2f dist=%.2f energy=%.2f penalty=%.2f",
                order.order_id,
                bid.mode,
                bid.vehicle_id or "-",
                bid.drone_id or "-",
                bid.launch_station_id or "-",
                bid.recovery_station_id or "-",
                bid.score_total,
                bid.cost_dist,
                bid.cost_energy,
                bid.cost_penalty,
            )

        if selected is not None:
            logger.info(
                "[MarketBasedSolver][Selected] 订单 %s -> mode=%s vehicle=%s drone=%s score=%.2f",
                order.order_id,
                selected.mode,
                selected.vehicle_id or "-",
                selected.drone_id or "-",
                selected.score_total,
            )
