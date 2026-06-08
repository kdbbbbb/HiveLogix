# MMCE 求解器接口总结

本文基于以下文件阅读整理：

- `backend/solver/greedy_mmce.py`
- `backend/solver/greedy_mmce_bi.py`
- `backend/solver/decision_engine.py`
- `backend/solver/factory.py`
- `backend/solver/interfaces.py`
- `backend/api/routes/simulation_bp.py`
- `backend/config/loader.py`
- `backend/core/entities/order.py`
- `backend/core/entities/drone.py`
- `backend/core/entities/truck.py`
- `backend/core/entities/swap_station.py`
- `backend/core/entities/primitives.py`
- `frontend/src/types/index.ts`
- `frontend/src/views/DispatchCenter/components/UnifiedMapView.vue`

## 1. `greedy_mmce.py` 的主调度入口

### 1.1 主类

- 类名：`GreedyMMCE`

### 1.2 对外入口函数

1. `dispatch(pending_orders, current_time, bbox, scene_id=None) -> DispatchPlan`
2. `dispatch_incremental(new_orders, current_time, bbox, scene_id=None) -> DispatchPlan`
3. `dispatch_replan_current_state(replan_orders, current_time, bbox, scene_id=None) -> DispatchPlan`

三者最终都进入统一实现：

- `_dispatch_impl(...) -> DispatchPlan`

### 1.3 `dispatch(...)` 参数

- `pending_orders: dict[str, Order]`
- `current_time: float`
- `bbox: dict`
  - 代码按 `minx / miny / maxx / maxy` 读取
- `scene_id: str | None = None`

### 1.4 返回值 `DispatchPlan`

`DispatchPlan` 是 `greedy_mmce.py` 里的 dataclass，字段为：

- `allocations: list[AllocationResult]`
- `cost_total: float`
- `summary: dict`
- `truck_routes: dict[str, TruckRoute] = {}`
- `drone_routes: dict[str, DroneRoute] = {}`

### 1.5 `AllocationResult` 结构

每个订单对应一个 `AllocationResult`，关键字段：

- `order_id: str`
- `vehicle_id: str`
- `mode: str`
  - 可能值见当前实现：`A` / `B` / `B_WAIT` / `C` / `REJECT`
- `distance: float`
- `feasible: bool`
- `reason: str = ""`
- `recovery_station_id: str = ""`
- `drone_id: str = ""`
- `launch_station_id: str = ""`
- `launch_time: float = 0.0`
- `wait_duration: float = 0.0`
- `score_total: float = inf`
- `cost_dist: float = 0.0`
- `cost_energy: float = 0.0`
- `cost_penalty: float = 0.0`

### 1.6 `summary` 实际内容

`GreedyMMCE._dispatch_impl()` 默认汇总：

- `total_orders`
- `feasible`
- `modes`
- `dispatch_type`
- `cost_breakdown`
  - `dist`
  - `energy`
  - `penalty`

`decision_engine.py` 还会补充：

- `solver`
- 增量调度时：`new_orders`
- 动态重规划时：`replanned_assigned_orders`

### 1.7 `TruckRoute` / `DroneRoute` 结构

`TruckRoute`：

- `truck_id`
- `nodes: list[TruckRouteNode]`
- `total_distance`
- `charging_stop_ids`
- `geometry`

`TruckRouteNode`：

- `node_id`
- `node_type`
  - 常见：`depot` / `customer` / `station` / `recovery`
- `position`
- `arrival_time`
- `departure_time`
- `order_id`

`DroneRoute`：

- `drone_id`
- `order_id`
- `path`
- `mode`
- `launch_loc`
- `delivery_loc`
- `recovery_loc`

说明：

- `decision_engine._build_drone_routes()` 当前会填 `launch_loc / delivery_loc / recovery_loc`
- `path` 在内部 `DroneRoute` 里没有真正填入坐标数组
- 前端最终看到的 `drone_routes[].path` 不是直接取这个 dataclass，而是 `simulation_bp.py` 再序列化出来的 WGS84 路径

## 2. `decision_engine.py` 如何选择和调用 solver

### 2.1 创建方式

`DispatchDecisionEngine.__init__()` 支持两种方式：

1. 直接注入 `solver`
2. 不注入时，按 `solver_name` 调 `create_solver(...)`

对应工厂在 `backend/solver/factory.py`：

- `greedy` -> `GreedyBaseline`
- `greedy_mmce` -> `GreedyMMCE`
- `greedy_mmce_bi` -> `GreedyMMCEBackboneInsertion`
- `mmce_backbone_insertion` -> `GreedyMMCEBackboneInsertion`
- `mmce-bi` -> `GreedyMMCEBackboneInsertion`
- `market` -> `MarketBasedSolver`

### 2.2 运行时切换

`DispatchDecisionEngine.set_solver(solver_name)`：

- 将名字转小写
- 若与当前 solver 相同则直接返回
- 否则重新 `create_solver(...)`
- 若是 `MarketBasedSolver`，额外 `bind_order_manager(order_mgr)`

### 2.3 全量调度调用链

`execute(current_time, bbox, scene_id=None)`：

1. 从 `order_mgr.pending_orders` 拍快照
2. 调 `self.solver.dispatch(pending, current_time, bbox, scene_id=scene_id)`
3. 给 `plan.summary["solver"]` 赋值
4. 校验非 market solver 不应返回 `B_DYNAMIC`
5. `_accumulate_plan_metrics(plan)`
6. `_apply_plan(plan, current_time)`
7. `_build_drone_routes(plan, current_time)`
8. 返回 `plan`

### 2.4 增量调度调用链

`execute_incremental(new_orders, current_time, bbox, scene_id=None, replan_unfinished=None)`：

1. 若 `replan_unfinished is None`，调用 `self.solver.should_replan_unfinished()`
2. 若为 `True`
   - 走 `_execute_replan_unfinished(...)`
   - 最终调用 `self.solver.dispatch_replan_current_state(...)`
3. 若为 `False`
   - 直接调用 `self.solver.dispatch_incremental(...)`
4. 后续同样执行：
   - `summary` 补充
   - `_accumulate_plan_metrics`
   - `_apply_plan`
   - `_build_drone_routes`

### 2.5 solver 的最小协议

`backend/solver/interfaces.py` 定义了统一协议 `DispatchSolver`：

- `dispatch(...) -> DispatchPlan`
- `dispatch_incremental(...) -> DispatchPlan`
- `should_replan_unfinished() -> bool`
- `dispatch_replan_current_state(...) -> DispatchPlan`
- `get_active_contracts() -> list`
- `fulfill_contract(contract_id) -> None`
- `build_incremental_route_from_stops(...)`

结论：

- 新 solver 只要返回标准 `DispatchPlan`，执行层和前端序列化层基本都可以复用

## 3. 前端依赖的调度结果字段

这里区分两层：

1. 求解器内部 `DispatchPlan`
2. 前端真正收到的 `/api/sim/dispatch` 返回 `plan`

前端实际依赖的是后者。

### 3.1 `/api/sim/dispatch` 返回的 `plan` 字段

`simulation_bp.py` 当前返回：

- `total_orders`
- `feasible`
- `modes`
- `cost_total`
- `cost_breakdown`
  - `dist`
  - `energy`
  - `penalty`
- `allocations`
- `truck_routes`
- `drone_routes`

### 3.2 `allocations[]` 前端拿到的字段

序列化时只保留了：

- `order_id`
- `vehicle_id`
- `mode`
- `distance`
- `feasible`
- `reason`
- `recovery_station_id`
- `drone_id`

注意：

- `launch_station_id`
- `launch_time`
- `wait_duration`
- `score_total`
- `cost_dist`
- `cost_energy`
- `cost_penalty`

这些虽然在 `AllocationResult` 里有，但当前 `/dispatch` 响应没有直接透出。

### 3.3 `truck_routes` 前端依赖字段

`_serialize_truck_route()` 输出：

- `truck_id`
- `nodes`
  - `node_id`
  - `node_type`
  - `lng`
  - `lat`
  - `arrival_time`
  - `departure_time`
  - `order_id`
- `total_distance`
- `charging_stop_ids`
- `geometry`
  - `lng`
  - `lat`

`UnifiedMapView.vue` 实际使用：

- `route.geometry`
  - 优先画线
- 若没有 geometry，则退回 `route.nodes`
- `route.truck_id`
- `route.nodes[0]`
- `route.nodes[-1]`

### 3.4 `drone_routes` 前端依赖字段

`simulation_bp.py` 返回的单条 `drone_route` 当前字段：

- `drone_id`
- `order_id`
- `mode`
- `path`
- `recovery_station_id`
- `launch_node_id` 可选
- `launch_node_type` 可选

`UnifiedMapView.vue` 实际画线只直接使用：

- `flight.path`

但类型定义 `frontend/src/types/index.ts` 里还约束了：

- `drone_id`
- `order_id`
- `mode`
- `launch_node_id?`
- `launch_node_type?`
- `recovery_station_id?`
- `path: [number, number][]`

### 3.5 订单侧联动字段

地图上的订单弹窗还依赖订单对象上的：

- `assigned_mode`
- `assigned_vehicle_id`

这两个字段由 `decision_engine._apply_plan()` 写回 `Order`：

- `order.assigned_vehicle_id = alloc.vehicle_id`
- `order.assigned_mode = alloc.mode`

## 4. `Order / Drone / Truck / SwapStation` 的关键字段名

### 4.1 Order

来自 `backend/core/entities/order.py`：

- `order_id`
- `create_time`
- `deadline`
- `delivery_loc`
- `pickup_source_id`
- `source_type`
- `payload_weight`
- `_status`
  - 通过只读属性 `status` 暴露
- `assigned_vehicle_id`
- `assigned_mode`
- `penalty_rate`
- `actual_deliver_time`

对调度最关键的通常是：

- `order_id`
- `create_time`
- `deadline`
- `delivery_loc`
- `pickup_source_id`
- `source_type`
- `payload_weight`
- `status`
- `assigned_vehicle_id`
- `assigned_mode`
- `penalty_rate`

### 4.2 Drone

来自 `backend/core/entities/drone.py`：

- `drone_id`
- `home_id`
- `home_type`
- `k1`
- `k2`
- `cruise_speed`
- `payload_capacity`
- `empty_weight`
- `battery_max`
- `current_loc`
- `status`
- `battery_current`
- `current_payload`
- `carrying_order_id`
- `route_plan`
- `current_waypoint_index`
- `cumulative_distance`
- `cumulative_energy_j`

调度层实际还大量依赖动态附加字段：

- `transport_truck_id`
- `scheduled_launch_time`
- `launch_station_id`
- `waiting_recovery_station_id`

对 MMCE 最关键的是：

- `drone_id`
- `home_id`
- `home_type`
- `current_loc`
- `status`
- `battery_current`
- `battery_max`
- `payload_capacity`
- `cruise_speed`
- `empty_weight`
- `carrying_order_id`
- `route_plan`
- `current_waypoint_index`
- `transport_truck_id`
- `scheduled_launch_time`
- `launch_station_id`
- `waiting_recovery_station_id`

### 4.3 Truck

来自 `backend/core/entities/truck.py`：

- `truck_id`
- `speed`
- `max_inventory`
- `status`
- `current_loc`
- `inventory`
- `docked_drones`
- `route_nodes`
- `_route_data`
- `_departure_time`
- `_current_node_idx`
- `cumulative_distance_m`

继承 `ChargingHost` 后还会用到：

- `swap_time`
- `parking_slots`
- `serving_drones`
- `wait_queue`
- `available_slots`

对调度最关键的是：

- `truck_id`
- `speed`
- `current_loc`
- `status`
- `inventory`
- `docked_drones`
- `swap_time`
- `parking_slots`
- `wait_queue`
- `serving_drones`
- `cumulative_distance_m`

### 4.4 SwapStation

来自 `backend/core/entities/swap_station.py`：

- `station_id`
- `location`

继承 `ChargingHost` 后还会用到：

- `swap_time`
- `parking_slots`
- `serving_drones`
- `wait_queue`
- `available_slots`
- `queue_length`

对调度最关键的是：

- `station_id`
- `location`
- `swap_time`
- `parking_slots`
- `serving_drones`
- `wait_queue`
- `available_slots`
- `queue_length`

### 4.5 `primitives.py` 里和调度强相关的类型

- `Position3D`
  - `x / y / z`
  - `distance_2d()`
  - `distance_3d()`
  - `interpolate()`
  - `to_wgs84()`
- `SourceType`
  - `DEPOT`
  - `TRUCK`
- `WaypointAction`
  - `PICKUP`
  - `DELIVER`
  - `SWAP_BATTERY`
  - `RENDEZVOUS`
  - `DOCK_TRUCK`
  - `DOCK_DEPOT`
- `RouteWaypoint`
  - `loc`
  - `action`
  - `target_entity_id`
- `TaskStatus`
- `DroneStatus`
- `TruckStatus`

## 5. 哪些函数可以复用

下面按你关心的能力拆分。

### 5.1 模式 A 评估

优先复用：

- `GreedyMMCE._try_mode_a(...)`
  - 直接产出 Mode A 的 `AllocationResult`
- `GreedyMMCE._estimate_mode_a_append_distance(...)`
  - 估算 A 模式追加距离
- `GreedyMMCE._score_allocation(...)`
  - 统一算 A 的 `dist / energy / penalty / total score`

如果采用 BI 插入式路线：

- `GreedyMMCEBackboneInsertion._best_truck_only_insertion(...)`
  - 这是 BI 版本更准确的 A 模式边际评估函数

### 5.2 模式 B 评估

基础 MMCE 可复用：

- `GreedyMMCE._try_mode_b(...)`
  - 车机协同，不等待站点发射
- `GreedyMMCE._try_mode_b_with_waiting(...)`
  - 含 `B_WAIT`
- `GreedyMMCE._evaluate_charging_station_departure(...)`
  - 评估从某充电站发射时的完整方案
- `GreedyMMCE._score_allocation(...)`

BI 版本更适合复用：

- `GreedyMMCEBackboneInsertion._best_mode_b_insertion(...)`
  - 结合卡车后续路径做 B/B_WAIT 边际评估
- `GreedyMMCEBackboneInsertion._select_b_wait_station_candidates(...)`
  - 枚举 launch station 候选
- `GreedyMMCEBackboneInsertion._best_station_insertion(...)`
  - 评估站点插入位置
- `GreedyMMCEBackboneInsertion._ensure_recovery_stop(...)`
  - 将 recovery stop 落到卡车 stop 序列中

### 5.3 模式 C 评估

优先复用：

- `GreedyMMCE._try_mode_c(...)`
  - 直接产出 Mode C 的 `AllocationResult`
- `GreedyMMCE._is_drone_ready_for_depot_launch(...)`
  - 判断 depot launch 约束
- `GreedyMMCE._score_allocation(...)`

### 5.4 前瞻能量校验

最核心可复用函数：

- `GreedyMMCE._check_energy_feasible(...)`
  - 当前实现里就是前瞻能量校验核心

支撑函数：

- `GreedyMMCE._flight_energy(...)`
  - 按物理模型算单段飞行焦耳消耗
- `GreedyMMCE._uav_energy_wh(...)`
  - 评分口径的 UAV 能耗
- `Drone.calculate_power(...)`
  - 物理功率模型
- `Drone.can_reach(...)`
  - 单段可达性判断

如果你做“从站点发射后还能否送达并回收”的前瞻评估：

- `GreedyMMCE._evaluate_charging_station_departure(...)`

如果你做“骨干路径沿线插站的能量可达性”：

- `GreedyMMCEBackboneInsertion._is_backbone_station_energy_feasible(...)`

### 5.5 回收点选择

基础 MMCE 可复用：

- `GreedyMMCE._get_recovery_pool()`
  - 构造 recovery 候选池
- `GreedyMMCE._check_energy_feasible(...)`
  - 会参与 recovery feasibility 判断
- `GreedyMMCE._evaluate_charging_station_departure(...)`
  - 同时比较 launch/recovery 组合

可视化层另有一个“从 truck route 上推 launch/recovery”的选择器：

- `simulation_bp._select_mode_b_launch_and_recovery(...)`

但这个主要是前端展示重建，不建议拿来做求解。

### 5.6 换电站排队

如果你说的是“实体层真实排队机制”，复用实体方法：

- `Truck.arrive(...)`
- `Truck.depart(...)`
- `Truck._truck_serve_next(...)`
- `Truck.tick_update(...)`
- `SwapStation`
  - 其排队逻辑主要继承自 `ChargingHost`

如果你说的是“求解阶段对等待站点/等待时间的评估”，复用：

- `GreedyMMCE._try_mode_b_with_waiting(...)`
- `GreedyMMCE._evaluate_charging_station_departure(...)`
- `GreedyMMCE._predict_truck_charging_stations(...)`
- `GreedyMMCEBackboneInsertion._select_b_wait_station_candidates(...)`
- `GreedyMMCEBackboneInsertion._best_station_insertion(...)`

### 5.7 评分函数

统一评分函数就是：

- `GreedyMMCE._score_allocation(...)`

它输出：

- `score_total`
- `cost_dist`
- `cost_energy`
- `cost_penalty`

其依赖的子函数：

- `GreedyMMCE._truck_energy_wh(...)`
- `GreedyMMCE._uav_energy_wh(...)`
- `GreedyMMCE._estimate_mode_a_append_distance(...)`
- `GreedyMMCE._estimate_truck_increment_for_drone_support(...)`

BI 版本在模式选择时还叠加了插入式边际评估逻辑：

- `GreedyMMCEBackboneInsertion._best_truck_only_insertion(...)`
- `GreedyMMCEBackboneInsertion._best_mode_b_insertion(...)`

但底层标量目标口径仍然继承自 MMCE 的距离/能耗/超时罚金体系。

## 6. `greedy_mmce_bi.py` 相对 `greedy_mmce.py` 的扩展点

如果你后面要做 GA/MMCE 结合，这部分很重要。

### 6.1 入口兼容

`GreedyMMCEBackboneInsertion` 保持同样三类入口：

1. `dispatch(...)`
2. `dispatch_incremental(...)`
3. `dispatch_replan_current_state(...)`

但三者统一进入：

- `_dispatch_backbone_insertion(...)`

### 6.2 主要能力差异

相比 `GreedyMMCE`，BI 版本增加了：

- 重货订单先形成 truck backbone
- 其余订单按 stop 序列做增量插入
- B_WAIT 发射站候选按未来路线筛选
- recovery stop 直接写入卡车 stop 序列
- 自动补 backbone station / terminal station

### 6.3 如果以后做 GA 编码，BI 里更值得复用的函数

- `_best_truck_only_insertion(...)`
- `_best_mode_b_insertion(...)`
- `_best_station_insertion(...)`
- `_select_b_wait_station_candidates(...)`
- `_ensure_recovery_stop(...)`
- `_augment_backbone_stations(...)`
- `_prune_nonessential_station_stops(...)`
- `_ensure_terminal_station_stop(...)`

## 7. 结论

如果目标是“只复用现有接口，最小侵入地新增一个求解器/评估器”，推荐直接对齐下面这组契约：

1. 输入接口对齐 `DispatchSolver.dispatch(...)`
2. 输出接口对齐 `DispatchPlan`
3. 单订单评估结果对齐 `AllocationResult`
4. 评分直接复用 `GreedyMMCE._score_allocation(...)`
5. 能量校验复用 `GreedyMMCE._check_energy_feasible(...)`
6. A/B/C 模式候选分别复用 `_try_mode_a / _try_mode_b(_with_waiting) / _try_mode_c`
7. 若需要基于卡车后续路径做插入式评估，优先复用 `GreedyMMCEBackboneInsertion`

