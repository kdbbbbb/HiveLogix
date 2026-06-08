# CMRAPPO（Shared PPO-Lite）离线训练与离线验证阶段实施计划

## 0. 文档目标

本文档只覆盖一个阶段：

- 离线训练
- 离线验证
- SUMO 离线可视化复核

本文档暂不覆盖：

- 现有前端接入
- 在线推理 API 接入
- 前端参数锁定 UI

但本文档要求在本阶段提前产出后续接入所需的核心文件：

- `policy.pt`
- `meta.json`
- benchmark 验证报告
- SUMO 可视化导出文件

### 0.1 本阶段核心策略：先固定 RH-ALNS，单独实现 CMRAPPO 离线训练

根据算法方案文档（`algorithm_scheme_parameter_adjustment_4km_10stations_poisson_modifying.md`）的最新修正，本阶段采用以下策略：

1. **RH-ALNS 先固定**：在本阶段，RH-ALNS 作为低频粗规划器，其输出（`CoarsePlanView`）以固定规则或简化实现提供，不在本阶段优化 ALNS 本身
2. **CMRAPPO（Shared PPO-Lite）单独实现**：本阶段的训练目标是让共享 PPO 策略在固定的 RH-ALNS 骨架约束下，学会高频局部决策
3. **可实现优先**：先跑通固定 benchmark 的完整训练-验证闭环，再扩展到随机泊松流

这样做的原因是：

- 同时优化 ALNS 和 PPO 会引入两个不稳定源，难以定位问题
- 固定 ALNS 骨架后，PPO 的输入分布更稳定，训练更容易收敛
- 先验证 PPO 的局部决策能力，再考虑 ALNS 与 PPO 的联合优化

---

## 1. 阶段目标

本阶段只回答 4 个问题：

1. 训练环境能否稳定构建并重复运行
2. 在固定 RH-ALNS 骨架约束下，CMRAPPO（Shared PPO-Lite）能否在 `backend/test_data/default_scene` 基线场景上跑出可解释结果
3. 训练好的模型能否在固定 benchmark 和随机扩展场景上离线验证
4. 能否把离线结果导出到 SUMO 做复核

---

## 2. 当前阶段唯一主输入

当前阶段统一以 `backend/test_data/default_scene` 作为主输入资产。

直接使用以下文件：

- `backend/test_data/default_scene/scene_config.json`
- `backend/test_data/default_scene/entities.json`
- `backend/test_data/default_scene/orders.json`
- `backend/test_data/default_scene/osm_network.xml`
- `backend/test_data/default_scene/osm_network.geojson`

订单使用策略如下：

1. **卡车路径规划依据（固定，不参与 PPO 训练）**
   - `orders.json.static_orders`（10 个）
   - 其中 4 个超过 HeavyDrone 上限（10kg）的订单由卡车直送（mode A），
     其余 6 个在 HeavyDrone 范围内的静态订单也可作为卡车路径节点依据
   - 第一阶段卡车路径固定，不在训练中优化

2. **PPO 训练订单源（泊松流，完全替换原始动态订单）**
   - `orders.json.dynamic_orders` 在训练阶段**不使用**，不作为背景事件注入
   - 训练时动态订单完全由泊松流生成，`arrival_rate > 0`
   - 泊松流订单重量上限设为 `heavy_drone.payload_capacity = 10kg`，
     确保所有训练订单均可由无人机处理（LightDrone ≤ 2kg，HeavyDrone ≤ 10kg）
   - 这样训练集干净，不混入固定背景订单，有利于泛化

   **泊松流订单 deadline 生成规则**（与 `backend/environment/state/order_manager.py` 保持一致）：
   - 每条泊松流订单的 deadline 由 `OrderManager._create_order` 生成：
     ```text
     deadline = spawn_time + random.uniform(window_min_s, window_max_s)
     window_min_s = window_min * 60   // 默认 window_min=20 分钟 → 1200s
     window_max_s = window_max * 60   // 默认 window_max=60 分钟 → 3600s
     ```
   - 预定义动态订单（`set_scheduled_dynamic_orders`）优先使用 `deadline_sim_s`，
     否则使用 `spawn_sim_s + deadline_offset_s`（默认 `deadline_offset_s=900s`）
   - 训练阶段统一使用 `window_min=20, window_max=60`（分钟），与现有在线参数一致；
     可通过 `rh_alns_cmrappo.yaml` 的 `scene.order_window_min_min` / `order_window_max_min` 覆盖

实施原则：

- 第一阶段：卡车路径固定 + 泊松流训练，验证 PPO 局部决策能力
- 第二阶段（离线验证）：用 `orders.json.dynamic_orders` 作为固定 benchmark 验证集，
  评估模型在真实动态订单分布上的表现

## 2.1 当前阶段必须先冻结的环境语义契约

离线训练与离线验证不能只依赖参数文件，还必须先冻结环境语义。当前阶段至少要先确定以下 4 件事：

1. 模式 C 的合法回收动作定义
2. 错过卡车后的软失败 / 硬失败机制
3. 主目标函数与奖励项的对应关系
4. FIFO、工位容量、等待、终端清场的事件逻辑

### 2.1.1 模式 C 的合法回收动作

模式 C 在离线环境中的正式语义应为：

- 无人机送达后，不允许飞向道路中途的运动中卡车
- “返回卡车”只能解释成返回卡车未来将经过的合法交接节点
- 合法交接节点仅限：
  - `station`
  - `depot`

因此，候选回收动作仅在同时满足以下条件时才能保留在 `action_mask` 中：

1. 该节点在卡车未来路径上，且仍未经过
2. 无人机预计到达时刻早于卡车到达该节点
3. 无人机剩余电量足以飞到该节点并保留安全余量

### 2.1.2 模式 C 返程的保底规则

订单送达后，返程目标按以下顺序选择：

1. 卡车未来合法交接节点
2. 仓库
3. 最近可达充换电站
4. 若上述均不可达，则进入硬失败

这里必须保留的业务语义是：

- 如果无人机返程时，电量不足以支撑到最近汇合点与仓库，则直接飞往最近可达充换电站
- 若连最近可达充换电站也不可达，则不能继续保留该动作

### 2.1.3 错过卡车后的软失败与硬失败

“错过卡车”发生在无人机送达订单之后的返程阶段，因此：

- 订单应保持为已完成
- 失败的是返程回收子任务，不是订单交付失败

软失败定义：

- 错过原定回收节点上的卡车
- 但仍能飞往某个合法安全节点

软惩罚定义：

- `soft_miss_penalty = lambda_miss * T_fallback`

其中 `T_fallback` 为无人机从确认失败时刻飞往下一个目标地所用时间。

硬失败定义：

- 无人机当前不在 `station` / `truck` / `depot`
- 且剩余电量不足以飞往任何下一个合法充换电站或仓库
- 或已经出现半空中停电

### 2.1.4 当前阶段统一采用的主目标

> **注意**：本节为早期版本，完整权威定义见 Section 4.1.4 和 4.1.4.1，以 4.1.4 为准。

当前阶段统一采用以下主目标：

```text
J =
  T_complete
  + lambda_wait * T_wait
  + wait_idle_penalty_coef * T_idle
  + lambda_queue * T_queue
  + lambda_miss * T_fallback
  + lambda_res_timeout * T_res_timeout
  + lambda_overdue * T_overdue
  + hard_failure_penalty_sec * N_hard_fail
  + hard_overdue_penalty_sec * N_hard_overdue
```

含义如下：

- `T_complete`：所有订单从进入可调度池到完成配送的时长总和，定义为
   `T_complete = sum_i (t_delivered_i - t_spawn_i)`，**仅对无人机可配送订单统计**（`weight <= 10kg`）
   其中 `t_spawn_i` 是订单进入可调度池的时刻（对应 `spawn_sim_s`，静态订单取 `0`），
   `t_delivered_i` 是无人机完成配送的时刻（不含 `drone_service_time_order_s`，
   若有服务时长则取服务完成时刻，需与环境实现保持一致）。
   示例：3 个订单分别在进入池后耗时 10 分钟、20 分钟、40 分钟完成配送，则
   `T_complete = 70 分钟`。
   该目标只计入策略可影响的无人机订单时间段，避免将 `mode A` 卡车背景订单混入 PPO 主评估口径
- `T_wait`：物理等待时间总和，仅指 UAV 在节点处被动等待卡车到来的时间，
   不包含显式 WAIT 动作产生的 idle 时间（两者必须分开累计，不允许重复计入）
- `T_idle`：UAV 显式选择 WAIT 动作后产生的 idle 时间总和，定义为
   `T_idle = sum_k delta_t_wait_k`
   其中 `delta_t_wait_k` 是第 k 次 WAIT 动作对应的仿真时间推进量。
   即时惩罚为 `r_wait_step = -wait_idle_penalty_coef * delta_t_wait_k`，
   与 `lambda_wait` 保持同一口径（首轮建议 `wait_idle_penalty_coef = 0.10`，
   与 `lambda_wait` 相同，后续可通过 curriculum 衰减）。
   注意：排队（`T_queue`）、补能宿主容量导致的被动等待不计入 `T_idle`，
   避免与 `T_wait` / `T_queue` 重复扣罚
- `T_queue`：无人机在充换电宿主（`station / depot`）排队总时间，仅统计已到达
   宿主后等待服务槽位的时间；不包含 `charging_on_truck` 的车载充换电等待
- `T_fallback`：错过卡车后飞往下一个安全宿主（充换电站或仓库）的总飞行时间。
   在事件驱动环境中，它不是一次性入场惩罚，而是 `fallback_recovery`（兜底返程飞行态）
   的状态占用时间累计：只要 UAV 仍处于 `fallback_recovery`，每次时间推进区间 `[t_prev, t_next)`
   都累计 `ΔT_fallback = t_next - t_prev`；到达备用宿主后立即停止累计。
- `T_res_timeout`：reservation timeout 造成的总局部损失
- `T_overdue`：所有未送达且已超时订单的累计超时时长，定义为
   `T_overdue = sum_i overdue_duration_i`，其中 `overdue_duration_i` 表示订单 `i`
   在“未送达且已超过 deadline”状态下经历的累计时间。
   在事件驱动环境中，采用**按时间推进区间类实时累计**，不允许等到送达、强制移除或
   episode 结束时再统一结算。对任意一次时间推进区间 `[t_prev, t_next)`，订单 `i` 的
   增量定义为：
   `ΔT_overdue_i = max(0, t_next - max(t_prev, deadline_i))`
   （仅当订单在整个结算时刻仍属于未送达订单集合时才参与累计）。
   这样订单一旦超过 `deadline_i`，就在下一次时间推进时立即开始扣罚；送达后停止累计，
   且送达时不再叠加额外惩罚，避免双重计入。
   首轮建议 `lambda_overdue = 0.20`（高于 `lambda_wait`，体现超时的更高优先级）
- `N_hard_overdue`：超时时长超过 `max_overdue_sec` 仍未送达的订单数。
   超过该阈值后订单从可调度池中强制移除，记一次固定惩罚 `hard_overdue_penalty_sec`，
   避免模型长期持有无法挽回的"死单"。
   首轮建议 `max_overdue_sec = 600`，`hard_overdue_penalty_sec = 600`
- `N_hard_fail`：半空停电或无合法安全节点可达的事件数

要求：

- 训练 reward 必须来自这套主目标及其分解项
- benchmark 与 stochastic 验证也必须回到这套指标，不允许只看 PPO reward
- `mode A` 背景订单不进入这套 PPO 主指标；其执行结果只进入单独的系统上下文统计区
- `T_overdue` 的即时惩罚必须在 `env_adapter` 的 `compute_reward()` 里逐步结算，
   不能只在 episode 末端或强制移除时统一结算，否则梯度信号仍然稀释
- `T_idle` 的即时惩罚对两类 WAIT 分开定义：
   - `idle` 状态下：固定为 `step(WAIT_action)` 调用点的一次性 entry cost（入场成本），
     数值等于本次 WAIT 的精确 `delta_wait`
   - `riding_with_truck` 状态下：不在 `step(WAIT_action)` 入口预付整段未来时长，而是在
     `active_wait` 占位期间按每次真实时间推进区间 `[t_prev, t_next)` 的实际 `dt`
     累计到 `T_idle`

### 2.1.5 当前阶段建议采用事件驱动环境

由于模式 C 的赶上 / 错过、FIFO 队列与站点容量都对时间精度敏感，本阶段的离线环境建议采用事件驱动或至少准事件驱动推进，而不是大时间步粗近似。

否则会直接导致：

- 模式 C 候选动作判定失真
- `action_mask` 不稳定
- miss / catch 统计不可靠
- 奖励噪声异常增大

---

## 2.2 系统运行基础约束（必须先冻结）

以下约束是系统业务语义的基础事实，必须在实现任何环境逻辑之前明确冻结，不允许在实现过程中各自理解。

1. **无人机单次只携带一个订单**
   - 无人机每次出发只能携带并配送一个订单的货物
   - 飞行途中不能接新单，必须完成当前订单的配送与返程后才能进入下一轮决策

2. **卡车与仓库货物均充足，无人机可在卡车上或仓库直接取货**
   - 货物为同质化货物，仅有重量差别
   - 卡车货物充足，无人机在卡车上时可直接取货，无需返回仓库
   - 仓库货物同样充足，从仓库出发的无人机直接取货出发

3. **卡车上配备充换电设施，无人机在卡车上可完成充换电**
   - 无人机被卡车回收后（mode C 成功），可在卡车上充换电
   - 卡车上充换电不占用站点 FIFO 队列，由卡车服务时长（`truck_drone_recover_time_s`）决定
   - 充换电完成后，无人机进入 `riding_with_truck` 状态，等待下一个决策点

4. **"哪些无人机搭车出发"在系统启动时由外部预设决定，不是 PPO 或 ALNS 的决策范围**
   - 由于预订单的存在，卡车在系统启动时即出发
   - 哪些无人机随卡车出发、哪些从仓库直飞，是系统初始化时的外部配置，不进入训练决策空间

5. **无人机在卡车上时，取货点由当前位置自然决定**
   - 处于 `riding_with_truck` 状态的无人机，起飞时直接从卡车取货
   - 处于 `idle_at_depot` 状态的无人机，出发时从仓库取货
   - 不存在"在卡车上但要飞回仓库取货"的场景

---

## 3. 总实施顺序

建议严格按以下顺序推进：

1. 固化数据与参数契约
2. 冻结环境语义契约
3. 实现离线场景装载器
4. 实现训练环境适配器
5. 实现候选动作与上层规划骨架
6. 实现 PPO 训练主循环
7. 实现固定 benchmark 离线验证
8. 实现随机扩展离线验证
9. 实现 SUMO 可视化导出
10. 固化模型产物与阶段验收

不建议跳步，尤其不要在训练环境未稳定前提前做前端接入。

---

## 4. 分阶段实施内容

## 4.1 Phase 1：固化离线阶段的数据、参数契约与架构职责边界

目标：

- 先定义训练到底吃什么、输出什么
- 明确 RH-ALNS 与 CMRAPPO（Shared PPO-Lite）在本阶段的职责边界
- 冻结所有后续实现依赖的环境语义契约

### 4.1.1 本阶段架构职责边界（必须先冻结）

本阶段正式采用以下分工，不允许在实现过程中混淆：

**RH-ALNS（低频粗规划器，本阶段固定）**

在本阶段，RH-ALNS 以简化固定实现提供，其职责仅限于：

1. 生成 truck 的未来骨架路线（`truck_backbone_route`）
2. 给订单池做粗粒度优先级排序与窗口分桶（`order_priority_band`）
3. 生成 station/depot 支持半径与可行 recovery 池（`recovery_pool`）
4. 给未来一段时间内的固定充换电节点分配粗粒度服务软预算（`node_charge_load_budget`）
5. 提供 route drift 参考基线（`route_drift_ref`）

本阶段 RH-ALNS 的触发规则固定为：

```text
Trigger_ALNS =
    periodic(interval = coarse_replan_interval_sec)   // 默认 420s
 OR backlog_new_orders >= coarse_new_order_trigger    // 默认 3 单
 OR route_drift_ratio >= route_drift_trigger_ratio    // 默认 0.18
 OR fallback_count(window) >= fallback_burst_trigger_count  // 默认 2 次/300s
 OR hard_failure_count(window) >= hard_failure_trigger_count // 默认 1 次
```

这些触发条件在 Phase 6 起统一由 `planner_bridge.maybe_replan(...)`（粗规划桥接器：是否重规划）
判定，但**不要求** `planner_bridge` 自己订阅环境事件或维护窗口账本。职责固定为：

- `env_adapter`（训练环境适配器）继续维护运行时真值与原始事件账本
- `planner_bridge` 只消费调用方传入的 `PlannerTriggerContext`（重规划触发上下文）
- `route_drift_ratio`（路线漂移比例）属于基于"当前运行时真值 + 当前 coarse plan 参考基线"
  的派生信号，不属于 `RuntimeStateView`（运行时真值快照）原始字段；
  它**只直接用于** `planner_bridge.maybe_replan(...)`（粗规划桥接器：是否重规划）的
  `Trigger_ALNS`（ALNS 重规划触发条件）判断，不直接进入 `candidate_builder`
  （候选动作构建）、`action_mask`（动作掩码）、`policy.forward(...)`（策略前向）或
  reward（奖励）结算
- `fallback_count(window)`（fallback 窗口计数）与 `hard_failure_count(window)`（硬失败窗口计数）
  来自 `env_adapter` 内部事件账本的窗口聚合结果，不要求 `planner_bridge` 自己维护历史事件流

本阶段不优化 ALNS 本身的求解质量，只要求其输出满足 `CoarsePlanView` 接口契约。

**CMRAPPO（Shared PPO-Lite，本阶段训练目标）**

CMRAPPO 是本阶段唯一需要训练的组件，其职责是：

1. 在每次局部事件触发时，基于共享参数策略在 `action_mask` 过滤后的候选集中选择动作
2. 动作空间采用 `factorized action space`：
   - 对真实派送动作，顺序固定为：先选 `order_i`（候选订单），再选 `mode_i`
     （`B` / `C`）
   - 当 `mode_i = C` 时，第三阶段再选 `recover_node_j`（卡车汇合回收节点）
   - 当 `mode_i = B` 时，不存在第三阶段策略选择；具体返程充换电宿主在订单送达后由
     执行层确定性规则在 `station / depot` 中选择
   - 同时保留一个 `global_wait_action`（全局 WAIT 动作）；它**独立于**
     `DispatchAction`（派送动作）的三阶段分支之外，不绑定任何具体
     `order_i` / `recover_node_j`
   - 训练侧 actor（策略头）因此采用“顶层二分支”结构：
     先在 `{WAIT, DISPATCH}`（等待 / 派送）之间选择；若选择 `DISPATCH`，再进入
     `order -> mode -> recover_node`（订单 -> 模式 -> 回收点）的三阶段 factorized
     dispatch 分支
3. `mode A` 不进入 PPO actor head，由 RH-ALNS / truck-side coarse planner 管理
4. 模式约束采用双层语义：
   - `planner_mode_cap={A,B,C}`：上层规划器语义边界
   - `policy_mode_mask={B,C}`：真正给 PPO actor 的**订单级派送模式掩码**
5. 训练阶段固定使用 `centralized_train_only critic`
6. 推理阶段默认使用 `greedy` 动作选择

**PPO 决策触发时机（两类，结构完全统一）**

PPO 在以下两类时机触发，动作语义相同：

- **`idle` 触发**：无人机在 depot 或 station 完成充换电后进入 idle，触发 PPO 决策
- **`riding_with_truck` 触发**：卡车到达某个站点，且该站点在
  `env_adapter._active_launch_stations`（执行侧实时触发站点集合）中时，
  触发该卡车上所有 `riding_with_truck` 无人机的 PPO 决策。
  该集合的初值来自当前
  `CoarsePlanView.launch_candidate_stations`（粗规划触发站点集合）

两类触发都共享同一套派送动作语义：

- `mode B = autonomous_return_to_charge_host`（自主返程，不依赖卡车；订单送达后由执行层在
  `station / depot` 中确定性选择返程充换电宿主）
- `mode C = rendezvous_return`（卡车汇合返程，需要选择 `recover_node_j`）
- 并始终额外提供一个 `global_wait_action`

`WAIT` 动作在两类触发中的语义：
- `idle` 状态下：保持当前位置不动，等待下一个环境事件
- `riding_with_truck` 状态下：继续搭车到下一站，不起飞（语义等价于 STAY_ON_TRUCK）

**`riding_with_truck` 状态下的候选集构建（实时计算）**

与 `idle` 状态不同，`riding_with_truck` 触发时的回收候选点不由
`coarse_plan.recovery_pool`（粗规划回收候选池）直接约束，而是在 PPO 决策时根据当前时刻实时计算：

```text
当卡车到达站点 S_k（S_k ∈ _active_launch_stations，且其初值来自当前
CoarsePlanView.launch_candidate_stations）时：
  对每个候选订单 order_i：
    t_deliver = t_now + fly_time(S_k → order_i.location)
    runtime_recovery_pool(order_i) = {
      r_j | node_type(r_j) ∈ {station, depot}
          AND t_arrive_truck(r_j) > t_deliver
          AND t_deliver + fly_time(order_i.location → r_j) + rendezvous_eta_safe_margin_sec <= t_arrive_truck(r_j)
          AND E_need(order_i.location → r_j, payload=0) + drone.safe_margin_j <= E_rem
    }
```

这里的 `runtime_recovery_pool(order_i)`（运行时回收候选池）是
`riding_with_truck`（随车搭载触发）专用的决策时局部候选集，不等同于
`CoarsePlanView.recovery_pool[order_i]`（粗规划回收候选池）；两者分别服务于
`riding_with_truck` 与 `idle` 两条 mode C 候选构建路径。

这样设计的原因是：起飞时刻（`t_now`）决定了 `t_deliver`，进而决定了哪些回收节点在时序上合法。ALNS 无法在粗规划时预知精确的起飞时刻，因此回收候选点必须实时计算。

这里需要特别固定一个业务口径：

- **mode C 的等待回收合法性**只由时序、安全可达性和能量可行性决定
- **充换电容量/排队**只在 UAV 被成功回收后的服务阶段生效，不构成“等待被回收”动作本身的非法性条件
- 因此，`predicted_queue_time(node)` 可以作为候选排序、软截断或 reward 估计的输入，但不应作为 mode C 等待回收动作的硬性合法性判据

**ALNS ↔ PPO 正式接口契约（`CoarsePlanView`）**

ALNS 向 PPO 输出的只读边界对象定义如下：

```text
CoarsePlanView {
  plan_version: int
  issued_at: float
  valid_until: float
  truck_backbone_route: [node_id]
  truck_eta_map: {node_id -> t_arrive_truck_sec}
  authorized_orders: [order_id]
  order_priority_band: {order_id -> int}
  order_pre_score: {order_id -> float}
   planner_mode_cap: {order_id -> {A,B,C}}
   policy_mode_mask: {order_id -> {B,C}}
  recovery_pool: {order_id -> [node_id]}
  node_charge_load_budget: {node_id -> int}
  route_drift_ref: {node_id -> (eta_ref, route_index_ref)}
  launch_candidate_stations: [node_id]
  allow_empty_backbone_route: bool
}
```

`launch_candidate_stations` 说明：

- 由 ALNS 基于订单分布粗筛，筛选规则为：卡车未来路线上的站点，且该站点周边 `support_radius_km` 内存在未分配订单
- 仅用于决定"卡车到达哪些站点时触发 `riding_with_truck` 无人机的 PPO 决策点"
- 不预算具体的回收候选点，回收候选点在 PPO 决策时实时计算（见 CMRAPPO 职责描述）
- `plan_version` 更新时，`launch_candidate_stations` 同步更新；已在飞行中的无人机不受影响
- `CoarsePlanView.launch_candidate_stations`（粗规划触发站点集合）在语义上是 planner-issued snapshot upper bound
  （规划器签发的快照上界），不是执行层实时可变集合
- 环境执行层若需要在"卡车已过站点"或"站点周边订单已清空"后停止触发 PPO 决策，
  必须由 `env_adapter`（训练环境适配器）内部维护独立的
  `_active_launch_stations`（执行侧实时触发站点集合）完成；它是
  `CoarsePlanView.launch_candidate_stations` 的**只减不增子集**
- 因此，`plan_version`（规划版本号）只在 ALNS 重规划并重新签发 `CoarsePlanView`
  时递增；执行层对 `_active_launch_stations` 的本地删减**不构成**新的 coarse plan
  版本，也不应回写到当前 `CoarsePlanView`
- **Phase 5 简化版覆盖规则**：见 Section 4.5。为减少上层启发式对训练分布的干预，
  `launch_candidate_stations` 在 Phase 5 直接放宽为
  `future_backbone_station_nodes`（由 `truck_backbone_route` 派生的未来 `station` 节点子集）；
  注意：这不是 Phase 4 独立导出的字段名，而是消费侧根据 `truck_backbone_route`
  （卡车未来骨架路线）和当前路线进度动态过滤得到的 `station` 子序列，**不包含**
  customer 停靠点，也**不包含**场景中卡车未来不会经过的 station；基于订单密度的
  二次筛选延后到 Phase 6
- `allow_empty_backbone_route`：骨架退化开关。poisson 模式下默认 `false`，骨架为空视为
  契约错误；benchmark / hybrid 模式下由 `env_adapter.reset()` 自动设为 `true`，
  允许构造"卡车未来骨架已耗尽"的空骨架 `CoarsePlanView`，为合法退化状态

`node_charge_load_budget` 说明：

- `node_charge_load_budget` 保留在 `CoarsePlanView` 中
- 它只针对固定充换电节点：`station` / `depot`
- 它表示 coarse planner 对未来一段时间内节点**充换电服务阶段**承压的软预算
- 它不是物理工位容量，也不是 mode C 等待回收人数上限
- 其作用是给 `candidate_builder` / baseline / critic 提供固定节点拥挤先验，避免策略在 coarse 层面把过多 UAV 都导向同一个充换电节点
- 因此它仍然应该出现在当前 `CoarsePlanView` 契约中，但只能作为固定节点补能服务的规划侧引导信号，不能被解释为“等待回收容量”

PPO 的权限严格限定为：

1. 只能在 `authorized_orders` 内选订单（仅对 `dispatch_action` 分支）
2. 只能在 `policy_mode_mask[order]` 内选订单级派送模式（`B` / `C`）
3. `planner_mode_cap[order]` 仅作为上层语义边界，不直接作为 actor head 掩码
4. 当 `planner_mode_cap` 仅放行 `A` 时，该订单不进入本轮 PPO 可选订单集
5. `global_wait_action` 独立于 `policy_mode_mask`，由 `enable_wait_action=true`
   全局提供，不绑定订单
6. 对 `mode C`（卡车汇合返程）回收节点的选择权限分两条路径：
   - `idle`（地面空闲触发）时：只能在 `coarse_plan.recovery_pool[order]`（粗规划回收候选池）内选回收节点
   - `riding_with_truck`（随车搭载触发）时：只能在当前决策点实时计算得到的
     `runtime_recovery_pool(order)`（运行时回收候选池）内选回收节点；
     该池不直接复用 `coarse_plan.recovery_pool[order]`
7. 不能改写 `truck_backbone_route`
8. 唯一例外：当所有 coarse-plan 授权动作都非法时，允许进入 `safety fallback action`（飞往 depot 或最近可达 station）

当 `plan_version` 更新时：

1. reservation 节点不再出现在 `truck_backbone_route` 中 → 立即失效
2. 订单不再属于 `authorized_orders` → 相关未执行局部动作立即失效
3. mode C 回收节点边界变化时，相关未执行局部动作立即失效：
   - `idle` 路径下：原回收节点不再属于 `coarse_plan.recovery_pool[order]`
   - `riding_with_truck` 路径下：原回收节点不再属于当前决策点实时计算得到的
     `runtime_recovery_pool(order)`
4. 已起飞且订单已送达的 UAV 不回滚订单状态，只重算返程
5. `LSTM hidden state`（时序隐藏状态）按 `drone_id`（无人机编号）单独维护，
   不做全局硬重置，也不因其他 UAV 的事件直接推进；其他 UAV 的近期行为通过
   `history_tokens`（历史事件摘要张量）进入当前 UAV 观测。
   同时观测中必须注入 `plan_version_delta`（规划版本增量），避免 coarse plan
   更新后时序状态与当前规划语义脱节

### 4.1.2 需要完成的具体工作

1. 固定离线训练阶段主输入：
   - `scene_config.json`
   - `entities.json`
   - `orders.json`
   - `osm_network.xml/geojson`

2. 定义订单使用策略：
   - 卡车路径规划依据：`static_orders`（固定，第一阶段路径不优化）
   - PPO 训练订单源：泊松流（`arrival_rate > 0`，重量上限 ≤ 10kg）
   - `dynamic_orders` 训练阶段不使用，保留作为离线验证 benchmark

3. 定义统一订单源模式：
   - `order_source_mode = benchmark | poisson | hybrid`
   - 训练主循环只允许 `poisson`
   - `validate_benchmark.py` 只允许 `benchmark`
   - `validate_stochastic.py` 允许 `poisson` 或 `hybrid`
   - 泊松流复用 `backend/environment/state/order_manager.py` 的订单生成与 deadline 语义，不再单独重写一套离线规则

4. 定义训练专属配置文件：
   - `backend/config/rh_alns_cmrappo.yaml`
   - 配置文件必须按以下分层组织：`scene` / `data` / `planner` / `candidate` / `action_space` / `reservation` / `policy` / `reward` / `training` / `curriculum`

5. 定义模型元数据文件格式：
   - `backend/weights/rh_alns_cmrappo/<version>/meta.json`

6. 冻结环境语义契约（见 4.1.3 节）

### 4.1.3 必须先冻结的环境语义契约

离线训练与离线验证不能只依赖参数文件，还必须先冻结以下 7 件事：

**（1）模式 C 的合法回收动作**

模式 C 在离线环境中的正式语义：

- 无人机送达后，不允许飞向道路中途的运动中卡车
- "返回卡车"只能解释成返回卡车未来将经过的合法交接节点
- 合法交接节点仅限：`station` 或 `depot`

**时序量统一记号（全文强制）**

- `fly_time(A → B)`（飞行耗时）：duration（持续时间），单位秒
- `t_arrive_uav(A → B)`（无人机到达 `B` 的绝对仿真时刻）：timestamp（绝对时刻）。
  若该飞行段从当前时刻起飞，则 `t_arrive_uav(A → B) = t_now + fly_time(A → B)`；
  若该飞行段从送达时刻起飞，则 `t_arrive_uav(deliver → r_j) = t_deliver + fly_time(deliver → r_j)`
- `t_arrive_truck(r_j)`（卡车到达回收节点 `r_j` 的绝对仿真时刻）：timestamp（绝对时刻），
  固定定义为 `truck_eta_map[r_j]`

禁止在公式里继续混用“无人机 ETA / 卡车 ETA”这类历史写法。所有 mode C 时序判断、
reservation timeout、`T_timeout_cost` 和 WAIT 推进量的计算，都必须只使用
`fly_time`、`t_arrive_uav`、`t_arrive_truck` 这三类量。

候选回收动作仅在同时满足以下条件时才能保留在 `action_mask` 中：

1. 该节点在卡车未来路径上，且仍未经过（`t_arrive_truck(r_j) > t_deliver`）
2. 无人机预计到达时刻早于卡车到达该节点（`t_arrive_uav(deliver -> r_j) + rendezvous_eta_safe_margin_sec <= t_arrive_truck(r_j)`）
3. 无人机剩余电量足以飞到该节点并保留安全余量（`E_need(deliver -> r_j, payload=0) + drone.safe_margin_j <= E_rem`）

其中：

- `rendezvous_eta_safe_margin_sec` 是汇合安全时间裕量参数，对应 `backend/config/rh_alns_cmrappo.yaml.candidate.rendezvous_eta_safe_margin_sec`
- `drone.safe_margin_j` 是运行时绝对安全余量（焦耳），由共享运行时参数 `safe_margin_ratio` 经 loader 换算得到；运行时能量判断以该绝对值为准
- `CandidateMeta.energy_safe_margin_ratio` 仅作为训练配置 / 契约快照保留，不直接作为 `can_reach()` 的入参

`mode B / mode C` 的动作语义在本阶段进一步冻结为：

- `mode B = autonomous_return_to_charge_host`（自主返程，不依赖卡车）：
  `dispatch_action` 只输出 `order_i`（目标订单）与 `mode_i=B`；PPO 不在起飞前锁定具体
  `station / depot`。订单送达后，由执行层按确定性规则选择最优返程充换电宿主
- `mode C = rendezvous_return`（卡车汇合返程）：
  `dispatch_action` 在 PPO 决策时一次性输出 `order_i`（目标订单）、`mode_i=C`、
  `selected_recover_node`（策略原选回收节点）

- `selected_recover_node` 只属于 `mode C`；它必须在起飞前确定，**不允许**在
  `delivered`（已送达待返程）后
  再触发一次 `second_ppo_decision_after_delivery`（送达后二次 PPO 决策）
- 订单送达后，环境必须执行 `post_delivery_revalidation`（送达后执行层复核），对
  `selected_recover_node` 重新检查：
  - `energy_feasible`（送达后剩余电量对原定回收点是否仍可达）
  - `rendezvous_time_feasible`（送达后无人机与卡车在原定回收点的汇合时序是否仍成立）
  - `node_still_valid`（原定回收点是否仍属于卡车未来合法交接节点）
- 若复核通过，则 `effective_recover_node`（执行层最终采用的回收节点）=
  `selected_recover_node`
- 若复核失败，则环境**不重新向 PPO 请求动作**，而是进入
  `deterministic_fallback`（确定性兜底返程），按固定规则改飞仓库或最近可达充换电站，
  并记录对应的 `reservation_timeout`（预留失效）或 `soft_miss`（软失配）惩罚

补充说明：

- UAV 在 `station` / `depot` 等待被卡车回收，本身不受充换电工位容量限制
- `parking_slots`、FIFO 队列和 `predicted_queue_time` 只约束**回收成功后的充换电服务阶段**
- 因此，站点充换电容量不应作为 mode C 等待回收动作的直接非法性条件

**（2）模式 C 返程的保底规则**

订单送达后，返程目标按以下顺序选择：

1. 卡车未来合法交接节点（mode C）
2. 仓库（depot）
3. 最近可达充换电站
4. 若上述均不可达，则进入硬失败

**（3）Reservation Timeout 机制（CCT 映射）**

本阶段必须实现 reservation timeout，这是 CCT 机制在本项目的正式映射：

- reservation 是"软预留、非排他、后续服务排队感知"的局部承诺，不是物理工位锁
- 每个 mode C 动作建立后，启动 timeout 计时器：
- reservation 的建立时机固定为 `dispatch_action`（起飞前一次性派送动作）被选中时；
  其绑定对象是 `selected_recover_node`（策略原选回收节点），不是送达后的二次 PPO 结果
- `reservation timeout`（预留失效）与 `post_delivery_revalidation`（送达后执行层复核）
  属于执行层逻辑，不引入新的 actor 决策点；送达后若原定回收点失效，只允许进入
  `deterministic_fallback`（确定性兜底返程），不允许重新构造 `action_mask` 让 PPO 再选一次

```text
tau_res = alpha * fly_time(deliver -> r_j)
        + beta  * t_hist(node_type_j, mode C)
        + gamma * q_est(r_j)
```

- timeout 判定条件（任一成立即失效）：

```text
timeout(res) =
    (t_now >= expires_at)
 OR (t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j))
 OR (energy_feasible == false)
 OR (route_drift_invalid == true)
 OR (node_available == false)
```

- timeout 后进入 `fallback_recovery`，施加局部软惩罚：

```text
r_timeout = - lambda_res_timeout * T_timeout_cost
```

**（4）软失败与硬失败**

软失败定义：

- 错过原定回收节点上的卡车，但仍能飞往某个合法安全节点
- 订单保持为已完成，失败的是返程回收子任务
- 软惩罚：`soft_miss_penalty = lambda_miss * T_fallback`

硬失败定义：

- 无人机当前不在 `station` / `truck` / `depot`
- 且剩余电量不足以飞往任何下一个合法充换电站或仓库
- 或已经出现半空中停电
- 记为 `airborne_energy_failure`，无人机退出当前 episode 可用集合

**（5）FIFO、工位容量与终端清场**

1. `parking_slots` 只决定**充换电服务**的并行上限
2. 超出服务容量后必须进入 FIFO 队列
3. `swap_time` / `charge_time` 结束后释放工位并唤醒队首
4. 排队时间按"入队时刻到开始服务时刻"精确累计
5. UAV 在节点等待被卡车回收，不属于充换电 FIFO 排队
6. episode 结束时必须清点：已完成订单、超时订单、仍在返程/排队/等待卡车/已硬失败的无人机

**（6）超时订单惩罚机制**

超时订单的惩罚必须采用类实时累加方式，不允许仅在送达时结算，也不允许拖到
episode 结束或强制移除时再统一结算：

- 在事件驱动环境中，对每次时间推进区间 `[t_prev, t_next)`，对所有未送达订单结算：
  ```text
  overdue_dt_i = max(0, t_next - max(t_prev, deadline_i))
  r_overdue_interval = -lambda_overdue * sum_i overdue_dt_i
  ```
- 送达超时订单时，停止该订单的累加，不再叠加额外惩罚（避免双重计入）
- 若订单超时时长超过 `max_overdue_sec` 仍未送达，强制从可调度池移除，记一次 `hard_overdue_penalty_sec` 惩罚

**为什么必须类实时累加**：若仅在送达时结算，或拖到 episode 末端 / 强制移除时统一结算，
模型会学到"不送超时订单"可规避惩罚，导致超时订单被系统性忽略。类实时累加后，
不送的代价持续增大，送达则停止累加，
模型始终有动力去送超时订单。

参数：
- `lambda_overdue = 0.20`（首轮建议值，高于 `lambda_wait`，体现超时的更高优先级）
- `max_overdue_sec = 600`（超过此阈值强制移除）
- `hard_overdue_penalty_sec = 600`（强制移除时的固定惩罚，秒级等价）

**（7）`riding_with_truck` 状态语义**

`riding_with_truck` 是无人机的一个正式状态，进入条件为：

- 系统启动时被预设为搭车出发的无人机，初始状态为 `riding_with_truck`
- 无人机通过 mode C 被卡车成功回收后，完成卡车上充换电，进入 `riding_with_truck`

该状态下的行为规则：

1. **决策触发**：卡车到达某个站点 `S_k` 时，环境先检查
   `_active_launch_stations`（执行侧实时触发站点集合）；仅当 `S_k ∈ _active_launch_stations`
   时，才触发该卡车上所有 `riding_with_truck` 无人机的 PPO 决策点。
   其中 `_active_launch_stations` 的初值来自当前
   `CoarsePlanView.launch_candidate_stations`（粗规划触发站点集合），但会在执行层按规则实时删减

**`launch_candidate_stations` 的选择与更新规则**

选择依据为订单密度，而非卡车路线形态。卡车路线本身已由 RH-ALNS 保证不走回头路、不大绕路，路线上的每个站点卡车都会经过，因此 `launch_candidate_stations` 只需在"卡车已确定会经过的站点"中，按订单密度筛选值得触发 PPO 决策的子集。

筛选规则（每次 ALNS 重规划时执行）：

```text
对卡车未来路线上的每个站点 S_k（卡车尚未经过）：
  如果 count(未分配订单 within support_radius_km of S_k) >= min_orders_to_trigger
  → 加入 launch_candidate_stations
```

参数：
- `support_radius_km = 1.2`（与现有候选集参数一致）
- `min_orders_to_trigger = 1`（首轮默认值；若训练中 WAIT 动作被选得过于频繁，可调至 2）

更新时机（三种）：

1. **ALNS 重规划时**（主要路径）：
   `planner_bridge`（粗规划桥接器）重新计算
   `CoarsePlanView.launch_candidate_stations`（粗规划触发站点集合），与其他 coarse 字段同步刷新，
   `plan_version` 递增；随后 `env_adapter` 用该字段**整体重置**
   `_active_launch_stations`（执行侧实时触发站点集合）
2. **卡车经过某站点后**：
   `env_adapter` 将该站点从 `_active_launch_stations` 中移除，无需等待 ALNS 重规划
3. **某站点周边订单全部被分配后**：
   `env_adapter` 将该站点从 `_active_launch_stations` 中移除，避免无人机在此起飞后无订单可选只能 WAIT

补充约束：

- `_active_launch_stations` 只允许做执行层**删减**，不允许把 planner 未放行的站点加回去
- 若后续因新订单出现，某个已移除站点重新具备触发价值，也必须等下一次 ALNS 重规划后，
  由新的 `CoarsePlanView.launch_candidate_stations` 重新放行；`env_adapter` 不自行增补

**Phase 5 简化版覆盖规则**：

- Phase 5 不再按订单密度筛选 `launch_candidate_stations`
- 直接取 `future_backbone_station_nodes`（由 `truck_backbone_route` 派生的未来 `station`
  节点子集）：即 `truck_backbone_route` 中当前时刻之后仍会经过的全部 `station` 节点
- 由于 Phase 5 的 inline coarse plan 每次都按 `t_now` 重建，且未来骨架按
  `departure_time > t_now` 实时过滤，因此“卡车已过站点的移除”在 Phase 5 中可由
  `_build_coarse_plan_view()` 的动态过滤**隐式实现**
- Phase 6 切换到缓存式 coarse plan 后，上述隐式行为不再成立；
  `env_adapter` 必须显式维护 `_active_launch_stations`，并在“卡车经过站点”或
  “站点周边订单全部被分配”时做执行侧删减，不能继续把缓存对象上的
  `CoarsePlanView.launch_candidate_stations` 直接当作实时触发集合使用
- **不包含** customer 订单停靠点，也**不包含**场景中卡车未来不会经过的 station
- 这样做只放宽"哪些站点触发 `riding_with_truck` 决策"的启发式，不放宽
  `mode C` 的时序、电量与回收节点合法性约束
- 基于 `support_radius_km` / `min_orders_to_trigger` 的订单密度筛选，延后到 Phase 6 恢复

2. **候选集实时构建**：对每个候选订单 `order_i`，实时计算：
   ```text
   t_deliver = t_now + fly_time(S_k → order_i.location)
   runtime_recovery_pool(order_i) = {r_j | 满足时序 + 电量约束，基于 t_deliver}
   ```
3. **动作空间与 idle 完全统一**：
   - `mode B` 分支：`dispatch_action=(order_i, mode=B)`，具体返程宿主不暴露给 PPO，
     由送达后的确定性规则在 `station / depot` 中选择
   - `mode C` 分支：`dispatch_action=(order_i, mode=C, recover_node_j)`
   - 并额外保留一个 `global_wait_action`
4. **WAIT 语义**：在 `riding_with_truck` 状态下，选择 `global_wait_action` =
   继续搭车到下一站，不起飞
5. **不触发决策的站点**：若卡车到达的站点不在 `launch_candidate_stations` 中，无人机继续搭车，不触发 PPO 决策
6. **卡车上充换电**：无人机在卡车上充换电不占用站点 FIFO 队列，由 `truck_drone_recover_time_s` 决定服务时长；充换电完成后重新进入 `riding_with_truck` 状态
7. **取货**：起飞时直接从卡车取货，无需返回仓库

补充说明：

- 对 mode C 而言，`predicted_queue_time(node)` 可以作为“回收后若进入该节点充换电服务，预计会多拥挤”的软信号
- 它可以用于候选排序、soft penalty 估计、critic 特征或 heuristic 打分
- 但不应因为某节点后续充换电队列较长，就把“等待被卡车回收”这个动作直接判为非法

### 4.1.4 主目标函数（本阶段统一采用）

```text
J =
  T_complete
  + lambda_wait * T_wait
  + wait_idle_penalty_coef * T_idle
  + lambda_queue * T_queue
  + lambda_miss * T_fallback
  + lambda_res_timeout * T_res_timeout
  + lambda_overdue * T_overdue
  + hard_failure_penalty_sec * N_hard_fail
  + hard_overdue_penalty_sec * N_hard_overdue
```

含义：

- `T_complete`：无人机可配送订单从进入可调度池到完成配送的时长总和，定义为
   `T_complete = sum_i (t_delivered_i - t_spawn_i)`，其中 `i` 仅遍历 `weight <= 10kg` 的无人机订单，
   `t_spawn_i` 是订单进入可调度池的时刻（对应 `spawn_sim_s`，静态订单取 `0`），
   `t_delivered_i` 是无人机完成配送的时刻（不含 `drone_service_time_order_s`，
   若有服务时长则取服务完成时刻，需与环境实现保持一致）。
   示例：3 个订单分别在进入池后耗时 10 分钟、20 分钟、40 分钟完成配送，则
   `T_complete = 70 分钟`。
   该目标只计入策略可影响的无人机订单时间段，避免把 `mode A` 卡车背景订单混入 PPO 主评估口径
- `T_wait`：UAV 在节点处被动等待卡车到来的物理等待时间总和，
   不包含显式 WAIT 动作产生的 idle 时间（两者必须分开累计，不允许重复计入）
- `T_idle`：UAV 显式选择 WAIT 动作后产生的 idle 时间总和，定义为
   `T_idle = sum_k delta_t_wait_k`（`delta_t_wait_k` 为第 k 次 WAIT 动作的仿真时间推进量）。
   即时惩罚 `r_wait_step = -wait_idle_penalty_coef * delta_t_wait_k` 在每步立即结算，
   不在 episode 末端累计，以保证梯度信号清晰。
   首轮建议 `wait_idle_penalty_coef = 0.10`，与 `lambda_wait` 同口径。
   排队（`T_queue`）、补能宿主容量导致的被动等待不计入 `T_idle`
- `T_queue`：无人机在充换电宿主（`station / depot`）排队总时间，仅统计已到达
   宿主后等待服务槽位的时间；不包含 `charging_on_truck` 的车载充换电等待
- `T_fallback`：错过卡车后飞往下一个站点或仓库的总时间
- `T_res_timeout`：reservation timeout 造成的总局部损失
- `T_overdue`：所有未送达且已超时订单的累计超时时长，定义为
   `T_overdue = sum_i overdue_duration_i`，其中 `overdue_duration_i` 表示订单 `i`
   在“未送达且已超过 deadline”状态下经历的累计时间。
   在事件驱动环境中，采用**按时间推进区间类实时累计**，不允许等到送达、强制移除或
   episode 结束时再统一结算。对任意一次时间推进区间 `[t_prev, t_next)`，订单 `i` 的
   增量定义为：
   `ΔT_overdue_i = max(0, t_next - max(t_prev, deadline_i))`
   （仅当订单在整个结算时刻仍属于未送达订单集合时才参与累计）。
   这样订单一旦超过 `deadline_i`，就在下一次时间推进时立即开始扣罚；送达后停止累计，
   且送达时不再叠加额外惩罚，避免双重计入。
   首轮建议 `lambda_overdue = 0.20`（高于 `lambda_wait`，体现超时的更高优先级）
- `N_hard_overdue`：超时时长超过 `max_overdue_sec` 仍未送达的订单数。
   超过该阈值后订单从可调度池中强制移除，记一次固定惩罚 `hard_overdue_penalty_sec`，
   避免模型长期持有无法挽回的"死单"。
   首轮建议 `max_overdue_sec = 600`，`hard_overdue_penalty_sec = 600`
- `N_hard_fail`：硬失败次数

要求：

- `J` 仅作为评估指标，不直接用于 PPO 梯度更新
- benchmark 与 stochastic 验证必须回到这套 UAV-scope 指标，不允许只看 PPO reward
- `hard_failure_penalty_sec` 首轮建议值为 `1200`（秒级等价损失），必须显著大于一次最坏可恢复扰动
- `T_complete` 仅用于评估，不计入训练 reward（见 4.1.4.1 节）
- `T_overdue` 的即时惩罚必须在 `env_adapter` 的 `compute_reward()` 里按时间推进区间逐步结算，不能只在 episode 末端或强制移除时统一结算
- `T_idle` 的即时惩罚按 WAIT 触发来源分两条路径：
  - `idle` 状态下：仍固定在 `step(WAIT_action)` 调用点按精确 `delta_wait`
    一次性结算，不进入 `_settle_per_dt_rewards(dt)` 的持续累计路径
  - `riding_with_truck` 状态下：改为在 `active_wait` 持续期间按每次真实事件推进的
    实际 `dt` 逐段累计，不在 `step(WAIT_action)` 入口预付整段未来等待时长
- `T_fallback` 的即时惩罚必须在 `env_adapter` 的 `compute_reward()` 里按
  `fallback_recovery` 状态的实际持续时间逐步结算，不允许只在进入 fallback 时扣一次
- `mode A` 背景订单不进入 `J` 与 PPO-vs-baseline 核心比较指标，仅进入单独的系统上下文统计区

### 4.1.4.1 PPO 训练用每步奖励 r_t

`J` 是 episode 级评估指标，PPO 实际用于梯度更新的是每步奖励 `r_t`。
`r_t` 是 `J` 各项的增量版本（去掉 `T_complete`），满足 `sum_t r_t ≈ -J_train`。

`T_complete` 不计入 `r_t`，原因是：送达时一次性扣 `(t_delivered - t_spawn)` 会使送达本身产生大额负奖励，
模型可能学到"慢送或不送"来规避这笔扣分。`T_overdue` 实时累加 + `R_delivery_bonus` 已足够驱动模型及时送达。

```text
r_t =

  [每步累加项，每个仿真步 dt 结算一次]
  - lambda_overdue * sum_i overdue_dt_i                       // 超时订单按时间推进区间类实时惩罚（仅无人机订单）
  - lambda_wait    * delta_wait_t                             // UAV 被动等待卡车
  - wait_idle_penalty_coef * delta_idle_t                     // UAV 主动选 WAIT 动作
  - lambda_queue   * delta_queue_t                            // 充换电宿主排队（station / depot）
  - lambda_miss    * delta_fallback_t                         // UAV 处于 fallback_recovery 时的兜底返程飞行时间

  [事件触发项，事件发生时一次性结算]
  + R_delivery_bonus                                          // 无人机成功送达一单（正奖励）
  - lambda_res_timeout * T_timeout_cost                       // reservation timeout
  - hard_failure_penalty_sec                                  // 硬失败（半空停电等）
  - hard_overdue_penalty_sec                                  // 订单超时被强制移除
```

说明：

- `T_overdue` 只对无人机可配送订单（weight ≤ 10kg）累加；`mode A` 卡车背景订单无超时惩罚，也不进入 `T_complete` 与 PPO 主评估
- `R_delivery_bonus` 是唯一的正奖励项，确保模型始终有动力送达订单，即使订单已超时
- 首轮建议 `R_delivery_bonus = 60`（秒级等价，量级约为一次小惩罚）
- 所有每步累加项必须在 `env_adapter.compute_reward()` 中逐步结算，不允许在 episode 末端批量累计

### 4.1.5 参数分层要求

参数必须按以下分层组织，不允许混用：

- **共享运行时参数**（继续放在 `backend/config/drone_params.yaml`）：
  - 能耗模型、配送时长、操作时长、速度等物理参数

- **算法专属参数**（统一放在 `backend/config/rh_alns_cmrappo.yaml`）：
  - `scene`：地图尺度派生规则
  - `data`：订单源模式与数据注入规则（`order_source_mode`、`benchmark_use_dynamic_orders=true`、`poisson_arrival_rate`、`poisson_seed`、`poisson_weight_max=10kg`、`order_window_min_min=20`、`order_window_max_min=60`、`hybrid_background_dynamic_orders` 等）
  - `planner`：RH-ALNS 低频触发参数（`coarse_replan_interval_sec=420` 等）
  - `candidate`：候选集参数（`max_candidate_orders=32`、`max_candidate_recovery_per_order=4`、`max_candidate_actions=128`、`station_wait_threshold_sec=240` 等）
  - `action_space`：动作空间类型（`factorized`）、`enable_wait_action=true`、`include_mode_a_in_policy=false`
  - `reservation`：timeout 参数（`alpha=1.5`、`beta=1.2`、`gamma=0.3` 等）
  - `policy`：模型结构（`encoder_type=attn_lstm_lite`、`d_model=128`、`nhead=8`、`lstm_hidden=128`、`hist_len=6`、`critic_mode=centralized_train_only`、`inference_mode=greedy` 等）；Phase 7 还必须在这一配置族中补齐 `CriticTensorSchemaV1` 的容量上限、归一化基准与 `schema_version` 来源，避免训练代码另行硬编码
  - `reward`：惩罚权重（`lambda_wait=0.10`、`wait_idle_penalty_coef=0.10`、`lambda_queue=0.10`、`lambda_miss=0.15`、`lambda_res_timeout=0.10`、`lambda_overdue=0.20`、`R_delivery_bonus=60`、`max_overdue_sec=600`、`hard_overdue_penalty_sec=600`、`hard_failure_penalty_sec=1200` 等）
  - `training`：PPO 超参数（`ppo_learning_rate=0.0003`、`rollout_steps=4096` 等）。这里的正式口径需要直接写死为单选结论：
    - `Phase 7 V1 = recurrent_ppo=true`（正式启用 recurrent PPO / simplified TBPTT）
    - `Phase 7 debug / pre-V1 = recurrent_ppo=false`（仅允许临时调试或链路打通，不属于 Phase 7 V1 验收路径）
    - 最终训练产物 `policy.pt`（策略权重）只能来自 `recurrent_ppo=true` 路径
    - 因此，Phase 7 V1 的 `training` 配置族必须进一步拆分并冻结：
    - `sequence_len`：单条训练 sequence 的总长度（total unroll length）
    - `burn_in_len`：sequence 前缀中仅用于推进 recurrent state、不参与 loss 的预热长度；Phase 7 V1 固定为 `0`
    - `train_len`：单条 sequence 中实际参与 PPO loss 的有效训练长度；V1 下固定满足 `train_len = sequence_len - burn_in_len`
    - `sequence_minibatch_size`：每个 PPO minibatch 中包含的 sequence 条数
    - `target_minibatch_timesteps`：期望的有效训练 timestep 数，仅用于配置校验、日志统计与吞吐监控，不作为 sampler 主切分单位
    - 旧字段 `training.batch_size` 不允许继续在 recurrent PPO 下承担“主采样单位”语义；推荐直接重命名为 `target_minibatch_timesteps`
    - 若仓库中某个阶段仍暂时只有平铺 `batch_size` 口径，则该实现最多只能视为
      `recurrent_ppo=false` 的 debug fallback（调试回退路径），不能据此宣称
      “Phase 7 V1 recurrent PPO 已完成”
  - `curriculum`：课程学习噪声（首轮全部为 0）

### 4.1.6 产物

- `rh_alns_cmrappo.yaml` 初版（含完整分层结构）
- `meta.json` schema 初版
- 离线阶段参数清单
- 一份环境语义契约说明（可内嵌于 yaml 注释或单独文档）
- `CoarsePlanView` 字段保留说明：`launch_candidate_stations`、`order_priority_band`、
  `order_pre_score` 在 Phase 1 仍作为正式契约字段保留；Phase 5 对它们的“放宽 /
  降级”仅属于消费侧实现覆盖，不改变 Phase 1 产物结构

### 4.1.7 验收标准

- 任何人只看配置文件就能知道训练输入和训练参数来源
- 不再混用旧共享配置与新算法专属权重/惩罚参数
- RH-ALNS 与 CMRAPPO 的职责边界已明确写入文档，`env_adapter` 的后续实现不会再对模式 C / miss / queue / reservation timeout 语义各自理解
- `CoarsePlanView` 接口契约已定义，后续实现可直接对照
- Phase 5 的简化覆盖规则已与 Phase 1 契约边界对齐：允许保留
  `launch_candidate_stations` / `order_priority_band` / `order_pre_score` 字段，
  但在 Phase 5 中按简化语义消费，不要求回头修改契约结构
- 主目标函数已包含 `lambda_res_timeout * T_res_timeout` 项，与算法方案文档一致

### 4.1.8 Phase 1 当前已完成内容（截至本轮）

本轮已完成以下 Phase 1 落地工作：

1. 已创建算法专属统一配置文件：
   - `backend/config/rh_alns_cmrappo.yaml`
   - 完成了 `scene / data / planner / candidate / action_space / reservation / policy / reward / training / curriculum` 的完整分层
   - 默认参数已按当前 `4×4km` 地图、`10` 个充换电站、`1` 辆卡车、`12` 架无人机的规模进行标定，并显式固化了训练与验证的复现参数（seed、order source mode、评估频率等）

2. 已创建训练期共享契约文件（`CoarsePlanView` 契约）：
   - `backend/training/contracts.py`
   - 已将文档中的 `CoarsePlanView` 正式收敛为代码级只读契约
   - 已定义 `RouteDriftRef`
   - 已定义相关基础类型与模式枚举：`NodeId`、`OrderId`、`PlannerMode`、`PolicyMode` 等
   - 已在契约层加入基础一致性校验，避免后续 `planner_bridge` 产出不完整或自相矛盾的 coarse plan
   - 说明：`launch_candidate_stations`、`order_priority_band`、`order_pre_score`
     仍保留在 Phase 1 契约中；Phase 5 对这三个字段的“全未来骨架站点触发 /
     仅日志排序用途”属于消费侧简化，不要求删除或重命名字段

3. 已创建训练子包入口：
   - `backend/training/__init__.py`
   - 为后续 `scene_loader`、`order_source_adapter`、`planner_bridge`、`candidate_builder`、`env_adapter` 的实现预留统一包路径

4. 已在 `contracts.py` 中固化 `meta.json` schema（代码级契约）：
   - 新增 `MetaJson`：Phase 1 可冻结的结构参数契约，包含八个子契约：
     - `PolicyMeta`：模型结构参数（`d_model`、`nhead`、`lstm_hidden`、`hist_len`、`inference_mode` 等）
     - `ActionSpaceMeta`：动作空间定义（`factorized_head_order`、`policy_modes`、`planner_modes` 等）
     - `CandidateMeta`：候选集参数（`max_candidate_orders`、`max_candidate_actions`、安全余量等）
     - `PlannerMeta`：上层规划触发参数（`coarse_replan_interval_sec`、`route_drift_trigger_ratio`、`fallback_burst_trigger_count`、`fallback_burst_window_sec`、`hard_failure_trigger_count`、`support_radius_km` 等）
     - `RewardMeta`：奖惩权重（`lambda_*`、`R_delivery_bonus`、`hard_failure_penalty_sec` 等）
     - `SharedRuntimeParamsSnapshot`：共享运行时参数快照（对应 `backend/config/drone_params.yaml` 的物理参数、能耗参数、服务时长参数）
     - `EnvSemanticContractMeta`：环境语义契约摘要（mode C 回收节点类型、reservation timeout 开关、overdue 惩罚模式、`allow_empty_backbone_route` 等）
     - `OnlineLockParams`：在线锁参数策略骨架（`locked_fields` / `tunable_fields`）
   - 新增 `TrainingRunMeta`：训练完成后才能填充的运行时字段（`model_version`、`trained_at`、`scene_id`、`scene_bundle_dir`、`training_input`）
   - 新增 `BenchmarkMeta`：benchmark 身份快照（`orders.json` 路径、整体摘要、`static_orders` / `dynamic_orders` 数量、`benchmark_use_dynamic_orders`）
   - 新增 `TrainingInputMeta`：训练输入快照（订单源模式、benchmark 身份、泊松参数、`poisson_seed`、`training_seed`、总步数）
   - 新增 `build_meta_json_dict(meta, run)`：合并两部分生成完整 meta.json 内容
   - 设计原则：结构参数（Phase 1 冻结，`MetaJson`）与运行时字段（Phase 7 填充，`TrainingRunMeta`）分离，避免 `Optional` 字段弱化契约强度
   - 说明：更严格的 `meta.json` 完整性校验（例如训练主循环真实填充值校验、数值范围校验、写盘前终检）放到 Phase 7 的 `train_cmrappo.py` 中继续完善；Phase 1 先冻结字段结构与最小非空约束
   - 说明：这里的 `training_seed` 不只是“写入 `meta.json` 的记录字段”，而是后续 Phase 7 训练入口必须真正消费的复现参数；至少要用于统一固定 `python random / numpy / torch` 的随机态，避免出现“元数据记录了 seed，但训练本身未按该 seed 执行”的伪复现
   - 说明：当前 Phase 1 的 `meta.json` schema 已覆盖 `policy / action_space / candidate / planner / reward / runtime / env semantic contract`
     等主契约，但尚未把 `CriticTensorSchemaV1`（价值网络张量结构契约）单独固化成
     `MetaJson.critic_schema` 一级子契约；这部分必须在 Phase 7 正式训练前补齐，并写入 `meta.json`

5. 当前仍未完成、需要在后续继续推进的项：
   - `CoarsePlanView` 的实际生成逻辑（`planner_bridge.py`，Phase 6）
   - `CoarsePlanView` 消费侧的 `candidate_builder.py`（Phase 6）

说明：

- 本轮只完成了”契约与配置冻结”，尚未实现 `planner_bridge` 或任何真实 coarse planning 逻辑
- `contracts.py` 当前放在 `backend/training/` 下，是因为后续 Phase 2~Phase 7 的新模块都按文档规划落在该目录；若你后续希望将契约层上移到更通用的包路径（例如 `backend/core/contracts/`），建议尽早统一，避免后续导入路径扩散

后续 Phase 注意事项（与 meta.json 契约相关）：

- **Phase 2（scene_loader）**：`TrainingRunMeta.scene_id` 与 `scene_bundle_dir` 必须与 `scene_config.json` 中的实际值对齐；`scene_loader` 加载完成后应输出可直接填入 `TrainingRunMeta` 的字段值，不允许在 `train_cmrappo.py` 中再次硬编码场景路径
- **Phase 3（order_source_adapter）**：`TrainingInputMeta` 的所有字段必须从 `rh_alns_cmrappo.yaml` 的 `data` / `training` 节读取；`order_source_adapter` 应在构建订单源配置时同步输出可填入 `TrainingInputMeta` 的字段快照
- **Phase 7（train_cmrappo）**：训练开始前必须先消费 `training.training_seed`，统一固定 `python random / numpy / torch`（以及当前可用设备 backend）的随机态；训练完成时必须按以下顺序固化产物：先保存最终 `policy.pt`，再构造 `TrainingRunMeta`，再调用 `build_meta_json_dict(MetaJson(...), TrainingRunMeta(...))` 生成 meta.json payload，最后在写盘前执行完整性终检后才允许写出 `meta.json`。`trained_at` 使用 ISO 8601 格式；不允许手写裸字典绕过契约；不允许在训练循环开始前预写一个看起来像“训练已完成”的 `meta.json`；`MetaJson` 的各子契约字段值必须与本次训练实际使用的 yaml 参数严格一致，不允许复制粘贴旧版本值；写盘前终检至少覆盖：benchmark 身份快照、共享运行时参数快照、`scene_id / scene_bundle_dir / model_version / trained_at` 等运行时字段齐全性、`training_input` 与本次实际订单源 / seed / 总步数一致性、`critic_schema_hash` 与本次 checkpoint 一致性，以及 `policy.pt / training_config_snapshot.yaml / train_metrics.jsonl` 等训练产物存在性。Phase 7 必须先在 `contracts.py` / `MetaJson` 中新增 `critic_schema`（价值网络张量结构契约）一级子契约，再写盘训练产物；不允许用临时“等价独立元数据段”替代正式契约，也不允许缺失 `CriticTensorSchemaV1` 的排序/截断/padding/归一化/存储规则
- **Phase 11（模型产物固化）**：meta.json 由 `build_meta_json_dict` 生成后直接写入 `backend/weights/rh_alns_cmrappo/<version>/meta.json`；`benchmark_report.json` / `stochastic_report.json` 等报告文件路径在本阶段不进入 meta.json，但目录结构必须与 §4.11 保持一致，以便后续在线接入时直接定位

## 4.2 Phase 2：实现静态场景装载器

目标：

- 把 `default_scene` 的静态资产变成训练和验证都能直接读取的标准内部对象
- 只负责装载场景资产，不在这一阶段决定运行时采用哪种订单源模式

建议新增文件：

- `backend/training/scene_loader.py`

核心职责：

1. 读取 `scene_config.json`
2. 读取 `entities.json`
3. 读取 `orders.json`
4. 读取 `osm_network.xml/geojson`
5. 输出统一的静态场景上下文对象

建议输出结构：

- 场景边界
- depot / station / truck / drone 内部对象
- 固定静态订单集
- 固定动态订单集
- 路网引用或几何数据引用
- 标准化 `TrainingSceneContext`

实现要求：

- benchmark 订单必须支持按仿真秒回放
- 动态订单必须保留 `spawn_sim_s` / `deadline_offset_s`
- 所有实体字段名与当前项目一致，不引入第二套命名
- 场景对象中必须显式保留站点类型、容量、换电时长与卡车未来路径节点信息，供模式 C 和 FIFO 逻辑使用

产物：

- `scene_loader.py`
- `load_default_scene()` 入口
- 至少 1 个装载自测脚本

验收标准：

- 能打印默认场景摘要：
  - `1` depot
  - `10` stations
  - `1` truck
  - `12` drones
  - `10` static orders
  - `10` dynamic orders
- 多次加载结果一致，可复现

### 4.2.1 仓库现状核对（2026-04-24）

对当前仓库代码的核对结论如下。

**已经存在、可直接复用的能力**

- 场景原始资产读取已经存在：
  - `backend/environment/geo/preset_scenes.py:get_preset_scene()`
    可直接读取 `scene_config.json`、`entities.json`、`orders.json`、`osm_network.geojson`
  - `backend/environment/geo/preset_scenes.py:load_osm_from_cache()`
    可直接读取 `osm_network.xml` 与 `osm_network.geojson`
- 实体实例化已经存在：
  - `backend/environment/state/entity_manager.py:EntityManager.load_from_config()`
    已能把 `depots/stations/trucks/drones` 转成内部对象，包含坐标转换、类型分发与资产注册
- benchmark 动态单按仿真秒回放语义已经存在：
  - `backend/environment/state/order_manager.py:OrderManager.set_scheduled_dynamic_orders()`
  - `backend/environment/state/order_manager.py:OrderManager._order_from_scheduled_entry()`
    已支持 `spawn_sim_s`、`deadline_offset_s`、`deadline_sim_s`
- 泊松订单与 deadline 生成规则已经存在：
  - `backend/environment/state/order_manager.py:OrderManager._create_order()`
    与 `backend/config/rh_alns_cmrappo.yaml` 中
    `scene.order_window_min_min` / `scene.order_window_max_min` 的配置口径一致
- 训练侧元数据契约已经冻结一部分：
  - `backend/training/contracts.py:BenchmarkMeta`
  - `backend/training/contracts.py:TrainingInputMeta`
  - `backend/training/contracts.py:TrainingRunMeta`

**需要明确的边界**

- 当前仓库里还没有训练期入口把“场景文件读取 + 实体对象构建 + 静态订单/动态订单拆分 + 路网引用”组装成统一的 `TrainingSceneContext`
- `backend/environment/scene/scene_service.py:load_preset_scene()` 返回的是前端/场景服务用的 `SceneContext`
  - 它不返回训练期内部对象
  - 不产出 `Depot / SwapStation / Truck / Drone / Order` 这套训练期结构
  - 还会重新生成随机 `scene_id`
  - 因此不能直接代替 `scene_loader`
- 当前训练包 `backend/training/` 中只有契约层文件，尚不存在：
  - `scene_loader.py`
  - `TrainingSceneContext`
  - `load_default_scene()` 训练入口
- 静态订单目前没有训练侧标准装载器
  - 仓库中确实已有在线仿真入口 `backend/api/routes/simulation_bp.py:_load_initial_orders()`
    可把订单字典转成 `Order`
  - 但它属于 API 初始化胶水逻辑，不是训练模块，也不会输出标准化训练场景上下文
- “卡车未来路径节点信息”当前也不是现成的场景装载产物
  - `backend/core/entities/truck.py` 只提供运行时 `route_nodes` 容器
  - `backend/solver/greedy_mmce.py:TruckRoute` 与 `_build_truck_route()` 属于求解器运行时构建逻辑
  - `backend/solver/decision_engine.py` 在应用调度结果时才调用 `truck.set_route()`
  - 因此，文档要求的 `truck_backbone_route` / 未来路径节点信息，当前不能通过静态场景读取直接得到

**对 Phase 2 的评估结论**

- “已有若干可复用片段，但还不能直接满足 Phase 2” 这一判断是正确的
- Phase 2 当前缺的不是底层能力，而是训练侧编排层：
  - 统一场景装载入口
  - 标准化 `TrainingSceneContext`
  - 静态订单/动态订单的训练期拆分与挂接
  - 路网资产引用统一输出
- 需特别注意 `Phase 2` 与 `Phase 4` 的边界：
  - `Phase 2` 至少要保留生成卡车骨架路线所需的静态资产与字段
  - 完整的 `truck_backbone_route`、ETA 与 SUMO 校验产物更接近 `Phase 4` 的职责
  - 实现时不应把这两阶段混成一个“只靠读取 JSON 就自动得到完整卡车未来路径”的错误预期

## 4.3 Phase 3：实现订单源适配（`benchmark` / `poisson` / `hybrid`）

目标：

- 把“场景资产”与“运行时订单注入模式”解耦
- 明确训练、benchmark 验证、stochastic 验证分别使用哪一种订单源
- 复用现有 `OrderManager` 的泊松生成与 deadline 逻辑，不单独复制实现

建议新增文件：

- `backend/training/order_source_adapter.py`

核心职责：

1. 定义统一模式：
   - `benchmark`：`static_orders + dynamic_orders`，并强制 `arrival_rate = 0`
     其中 `dynamic_orders` 作为 PPO 主验证订单，`static_orders` 作为卡车骨架与 `mode A` 背景任务
   - `poisson`：`static_orders` 保留为卡车骨架/`mode A` 背景订单，动态订单仅由泊松流生成，不注入 `dynamic_orders`
   - `hybrid`：`static_orders + dynamic_orders + 泊松流`
2. 统一输出运行时订单源配置：
   - `scheduled_dynamic_orders`
   - `poisson_gen_config`
   - `arrival_rate`
   - `seed`
   - deadline/window 参数快照
3. 复用 `backend/environment/state/order_manager.py` 现有规则：
   - 泊松到达过程复用 `random.expovariate(arrival_rate / 60.0)`
   - 泊松订单 deadline 复用 `_create_order()` 的 `window_min/window_max`
   - benchmark 动态订单回放复用 `_order_from_scheduled_entry()` 的 `spawn_sim_s / deadline_sim_s / deadline_offset_s`
4. 提供统一入口：
   - `build_order_source(scene_ctx, mode, seed, overrides)`
5. 明确模式约束：
   - `train_cmrappo.py` 仅允许 `poisson`
   - `validate_benchmark.py` 仅允许 `benchmark`
   - `validate_stochastic.py` 仅允许 `poisson` 或 `hybrid`

实现要求：

- 同一 `seed` 下，泊松订单流必须可复现
- `poisson` 模式下不得把 `dynamic_orders` 当作背景事件偷偷注入
- `benchmark` 模式下必须显式关闭泊松流（`arrival_rate = 0`）
- 所有订单源配置必须写入训练配置快照和 `meta.json`

产物：

- `order_source_adapter.py`
- 订单源模式说明
- 至少 1 个自测脚本：分别打印三种模式下的订单源摘要

验收标准：

- 三种模式的输入边界明确且互不混淆
- `poisson` 模式能稳定复现订单数、seed 与 deadline 分布
- `benchmark` 模式下结果只受固定订单回放影响，不受泊松流扰动

### 4.3.1 本轮已完成实现记录

本轮已按本节要求完成 `Phase 3` 的第一版落地，实现文件如下：

- `backend/training/order_source_adapter.py`
- `backend/scripts/inspect_order_sources.py`

同时为支撑 `Phase 3` 契约补充了两处配套改动：

- `backend/training/scene_loader.py`
  - `TrainingSceneContext` 新增 `scene_config_path`、`entities_json_path`、`orders_json_path`
  - 其中 `orders_json_path` 供 `BenchmarkMeta.orders_json` 与文件 SHA256 摘要生成使用
- `backend/environment/state/order_manager.py`
  - 将泊松订单 ID 生成从 `uuid4()` 改为基于当前随机流的确定性 `hex8-seq`
  - 目的不是改变业务语义，而是满足“同一 `seed` 下订单流完全可复现”的 Phase 3 验收要求

当前 `order_source_adapter` 已实现的职责：

1. 提供统一入口 `build_order_source(scene_ctx, mode, seed, overrides)`
2. 定义并固化三种订单源模式：
   - `benchmark`：保留 `dynamic_orders` 回放，强制 `arrival_rate = 0`
   - `poisson`：只保留 `static_orders` 作为卡车骨架 / mode A 背景，不注入 `dynamic_orders`
   - `hybrid`：同时保留 `dynamic_orders` 与泊松流
3. 输出统一运行时订单源配置：
   - `background_static_orders`
   - `scheduled_dynamic_orders`
   - `poisson_gen_config`
   - `arrival_rate`
   - `seed`
   - `benchmark`
   - `training_input_meta`
4. 复用 `OrderManager` 现有语义而不是重写一套生成逻辑：
   - scheduled dynamic replay 继续走 `set_scheduled_dynamic_orders()`
   - 泊松流配置继续走 `configure()` + `tick()`
   - deadline/window 参数仍与 `_create_order()` 保持同一配置口径
5. 提供后续入口可直接复用的辅助函数：
   - `configure_order_manager_for_source(...)`
   - `preview_dynamic_order_stream(...)`
   - `build_order_source_preview_summary(...)`
   - `ensure_mode_allowed(...)`

当前 `Phase 3` 已验证通过的边界：

- `benchmark` 模式下 `arrival_rate = 0.0`，不会再混入泊松流
- `poisson` 模式下 `scheduled_dynamic_orders = 0`，不会偷偷注入 `dynamic_orders`
- `hybrid` 模式下同时包含 benchmark 动态单与泊松流
- benchmark 动态单经 `scheduled_dynamic_orders -> OrderManager._order_from_scheduled_entry()`
  回放后，`Order.source_type` 与 `scene_loader` 直读路径保持同一枚举语义
- 相同 `seed` 下，泊松订单预览流的订单数、ID、创建时间、deadline、重量与坐标快照完全一致
- 不同 `seed` 下，泊松订单预览流会发生变化

本轮自测脚本与结果：

- `python backend/scripts/inspect_training_scene.py`
  - 通过，确认 `Phase 2` 场景装载未被破坏
- `python backend/scripts/inspect_order_sources.py`
  - 通过，已打印三种模式摘要并校验上述模式边界与 seed 可复现性
- `python -m compileall backend/training backend/scripts/inspect_order_sources.py backend/environment/state/order_manager.py`
  - 通过，确认新增模块可正常编译

结论：

- `Phase 3` 的“订单源适配层”已具备进入后续 `train_cmrappo.py`、`validate_benchmark.py`、`validate_stochastic.py` 接入的条件
- 后续训练/验证入口应直接复用本节实现，不应再各自维护独立的订单注入分支，以避免 `benchmark` / `poisson` / `hybrid` 语义再次漂移

备注：

- “所有订单源配置必须写入训练配置快照和 `meta.json`”这条要求在当前阶段已完成到“对象准备完成”的程度：
  - `order_source_adapter` 已产出 `benchmark` 与 `training_input_meta`
  - 这两部分可以被后续训练入口直接消费
- 真正的写盘动作仍属于 `Phase 7（train_cmrappo.py）` 的职责：
  - 训练完成后应构造 `TrainingRunMeta`
  - 再调用 `build_meta_json_dict(MetaJson(...), TrainingRunMeta(...))`
  - 最终写入模型目录下的 `meta.json`
- 因此，本阶段不再重复实现一次“临时写盘”，避免 Phase 3 与 Phase 7 各自维护一套不一致的 `meta.json` 生成流程

## 4.4 Phase 4：SUMO 卡车路径验证（env_adapter 前置）

目标：

- 在实现 `env_adapter` 之前，先把卡车物理执行路线、固定节点骨架和 ETA 口径固定下来
- 用同源的 OSM 路网与 SUMO `net.xml` 对卡车路径进行离线导出和 GUI 复核
- 为后续 `CoarsePlanView.truck_backbone_route`、`truck_eta_map`、`route_drift_ref` 提供可直接复用的前置产物

**为什么必须在 env_adapter 之前做**

卡车路径决定了 env_adapter 里三件核心事：

1. mode C 的合法回收节点集合（`recovery_pool`）
2. 各 station/depot 的 `t_arrive_truck`（卡车绝对到达时刻，直接影响 `action_mask`）
3. `CoarsePlanView.truck_backbone_route` 的内容

如果路径有问题，训练时 `action_mask` 会基于错误的 ETA 构建，模型学到错误的 mode C 行为，且无法区分是模型问题还是路径问题。

本阶段已新增/修改的关键文件：

- `backend/training/export_sumo_truck_route.py`
- `backend/environment/geo/osm_service.py`
- `backend/environment/geo/exporters/sumo_net_osm.py`
- `backend/scripts/run_phase4_sumo_gui.py`

从当前代码看，Phase 4 已经不是“准备开始做”的状态，而是已经形成了一条可执行的导出与复核链路，职责可以概括为：

1. 从默认训练场景加载 `depot / truck / static_orders / stations / osm_network.xml`
2. 在严格遵守 OSM 单行线方向的前提下生成卡车完整执行路线 `truck_execution_route`
3. 从完整执行路线投影出后续训练使用的 `truck_backbone_route`、`truck_eta_map`、`route_drift_ref`
4. 生成与 OSM 路径同源的 SUMO `net.xml / rou.xml / poi.add.xml / sumocfg`
5. 输出 JSON 报告、调试轨迹，并可直接启动 `sumo-gui` 做人工复核

### 4.4.1 当前已经落地的路线语义

本阶段已明确把“物理执行路线”和“PPO 可见固定节点骨架”分成两个对象：

1. **`truck_execution_route`**：卡车真实执行路线
   - 节点类型允许为 `depot / customer / station`
   - 路线形态固定为“仓库出发 -> 客户/站点访问 -> 回仓”
   - 每一段 stop-to-stop 都会物化出：
     - OSM 最短路径节点序列
     - SUMO edge 序列
     - 距离、行驶时间、到达时间
   - 该对象只服务于路径验证、导出和 GUI 回放
2. **`truck_backbone_route`**：供 `CoarsePlanView` 使用的未来固定节点骨架
   - 仅允许包含 `station / depot`
   - 不包含任何 `customer`
   - 从 `truck_execution_route.stops[1:]` 中投影得到，因此起始 `depot` 会被排除，回程 `depot` 会保留
   - 不允许重复固定节点，且不能为空

因此，Phase 4 之后应继续坚持：

- `truck_execution_route` 只作为验证和回放产物
- `truck_backbone_route / truck_eta_map / route_drift_ref` 才是后续训练链路消费的粗规划输入

### 4.4.2 `export_sumo_truck_route.py` 当前主流程

`backend/training/export_sumo_truck_route.py` 已经落地了完整的 Phase 4 主流程，当前实现口径如下：

1. 只支持单 `depot`、单 `truck` 的默认训练场景
2. 订单筛选口径不是 `fulfillment_mode`，而是：
   - 从 `static_orders` 中筛出 `payload_weight > heavy_drone.payload_capacity` 的重货订单
   - 这些订单视为必须由卡车直送的订单
3. 客户访问顺序按以下规则确定：
   - 先按 `deadline` 分组
   - 同一 `deadline` 组内，从当前位置出发，按 OSM 路网最短距离做贪心选点
   - 若距离相同，再按 `order_id` 稳定排序
4. 初始执行计划固定为：
   - `depot -> customers -> depot`
5. 为补足未来固定节点，会在执行计划中插入 `station`
   - 第一阶段：反复插入绕路代价 `< 100m` 的顺路站点
   - 第二阶段：若 `station` 数量仍不足 `planner.phase4_min_future_fixed_nodes`，继续按绕路代价从小到大补足
   - 注意：`min_future_fixed_nodes` 统计的是 `station` 数量，不把终点 `depot` 算入补齐目标
6. 路线物化时，所有停靠点会先吸附到最近 OSM 节点，再对每一段 stop-to-stop 计算 OSM 最短路
   - 路段距离会把起终点的 snap 偏移一起计入
   - ETA 由 `segment_distance / truck.speed` 累加得到
   - 当前 `CUSTOMER_SERVICE_TIME_SEC = 0.0`
7. 完整执行路线生成后，再投影得到：
   - `truck_backbone_route`
   - `truck_eta_map`
   - `route_drift_ref`

这里需要特别固定两个口径：

- 排序、插站、ETA 的主距离口径都是 OSM 路网距离
- 曼哈顿距离只作为“路网不可达时”的 fallback，不再是默认启发式

### 4.4.3 `osm_service.py` 与 `sumo_net_osm.py` 当前补齐的底层能力

这两个文件已经把“OSM 求路”和“SUMO 导出”对齐到同一套边语义上：

1. `backend/environment/geo/osm_service.py`
   - `build_road_graph()` 新增 `respect_osm_oneway`
   - Phase 4 导出链路显式使用 `respect_osm_oneway=True`
   - 新增 `shortest_path_length()`，供订单排序、插站绕路代价和 ETA 计算统一复用
   - 默认兼容模式仍保留，避免影响仓库内其他历史求解器
2. `backend/environment/geo/exporters/sumo_net_osm.py`
   - 新增 `SumoNetEdge`、`SumoNetArtifacts`
   - 先构建 artifacts，再写出 `net.xml`
   - 同时导出 `directed_step_to_edge`，把 OSM 有向 step 映射到 SUMO edge
   - 对双向道路生成正反两个 directed edge
   - 在 junction connection 上允许必要的反向切换，降低 SUMO 载入 `no valid route` 的概率

这意味着当前 Phase 4 的 OSM 最短路和 SUMO 回放使用的是同一套有向道路语义，不再是“两套近似上看起来差不多”的实现。

### 4.4.4 当前已完成的导出产物与验证项

`export_phase4_truck_route()` 默认会把产物写到 `scene_bundle_dir/sumo/phase4_truck_route`，当前已稳定导出：

- `truck_execution_route.json`
- `truck_backbone_route.json`
- `truck_eta_map.json`
- `route_drift_ref.json`
- `validation_report.json`
- `phase4_debug_trace.json`
- `phase4_gui.view.xml`
- `poi.add.xml`
- `truck_route.net.xml`
- `truck_route.rou.xml`
- `truck_route.sumocfg`

其中几个关键文件的职责已经比较明确：

- `truck_execution_route.json`
  - 保存完整 stop trace、segment 距离、OSM node path、SUMO edge 序列
- `truck_backbone_route.json`
  - 保存 `station / depot` 组成的 future fixed node 序列
- `truck_eta_map.json`
  - 保存固定节点首次到达 ETA
- `route_drift_ref.json`
  - 保存 `eta_ref + route_index_ref`
- `phase4_debug_trace.json`
  - 保存重货订单排序结果、初始执行计划、最终执行计划、插站原因、最终 stop trace，
    以及 `visited_station_ids`（访问过的站点汇总）/ `inserted_fixed_nodes`
    （插入固定站点记录）等调试摘要

说明：

- Phase 4 当前**没有**独立导出名为 `all_future_backbone_stations` 的 JSON 字段或文件
- 训练侧若需要“未来会经过的 `station` 汇总”，应从 `truck_backbone_route.json` 的
  `truck_backbone_route`（未来固定节点骨架序列）中过滤出 `station` 节点得到
- `phase4_debug_trace.json.visited_station_ids` 仅用于调试与 GUI 摘要，不应作为
  `env_adapter`（训练环境适配器）的正式输入契约

`validation_report.json` 当前已经自动检查：

- `all_segments_connected`
- `all_expected_customers_visited`
- `all_expected_fixed_nodes_visited`
- `eta_monotonic`
- `max_snap_distance_m`
- `snap_within_warn_threshold`
- `bounds_ok`
- `min_future_fixed_nodes_ok`
- `sumo_edge_sequence_non_empty`

因此，Phase 4 当前已经能自动回答以下问题：

- 预期重货客户是否都被卡车实际访问
- 生成的骨架节点是否都真实出现在执行路线中
- 固定节点 ETA 是否严格递增
- 路线吸附误差是否超过告警阈值
- 路线是否仍位于场景边界内
- SUMO 回放所需 edge 序列是否为空

### 4.4.5 `run_phase4_sumo_gui.py` 当前提供的 GUI 复核链路

`backend/scripts/run_phase4_sumo_gui.py` 已经把 Phase 4 的人工复核链路串起来了：

1. 启动前直接调用 `export_phase4_truck_route()`，重新生成最新默认场景产物
2. 自动查找 `sumo-gui`
3. 若同目录存在 `netconvert`，会先对 `net.xml` 做一次规范化
4. 启动前打印 `phase4_debug_trace.json` 摘要与 `truck_execution_route`
5. 默认加载：
   - `backend/test_data/default_scene/sumo/phase4_truck_route/truck_route.sumocfg`

当前已支持的参数包括：

- `--sumocfg`
- `--sumo-gui-bin`
- `--no-start`
- `--delay-ms`
- `--no-debug-print`

因此，Phase 4 现在已经具备“代码生成产物 -> 本地拉起 SUMO GUI -> 人工看路径”的最小闭环。

### 4.4.6 当前 Phase 4 已完成边界与未覆盖边界

当前可以认为已经完成的内容：

1. 基于默认场景重货订单生成单车卡车执行路线
2. 从执行路线稳定投影出 `truck_backbone_route / truck_eta_map / route_drift_ref`
3. 保证 OSM 求路与 SUMO 回放共用同源有向道路语义
4. 自动导出 JSON、SUMO 和调试产物
5. 提供可直接启动的 GUI 复核脚本

当前仍未由 Phase 4 覆盖的部分：

1. `mode C` 的订单时刻级合法性尚未闭环
   - Phase 4 只能保证“路线级存在 future fixed nodes，且 ETA 单调、物理可达”
   - 但 `t_arrive_truck > t_deliver`、安全裕量、电量约束仍需后续 `candidate_builder / env_adapter` 叠加
2. 多车、多仓、多场景泛化尚未覆盖
   - 当前代码明确只支持单 `truck`、单 `depot`
3. GUI 最终验收仍是人工步骤
   - 是否存在视觉跳边、站点经过是否符合预期，仍需人工打开 `sumo-gui` 确认
4. `run_phase4_sumo_gui.py` 对自定义 `--sumocfg` 的“按该路径同步重导”尚未实现
   - 当前导出动作仍固定针对默认场景
5. 还没有自动输出“手工距离/速度估算 vs ETA”误差对照表

## 4.5 Phase 5：实现训练环境适配器

目标：

- 把当前场景和调度问题封装成可训练环境

建议新增文件：

- `backend/training/env_adapter.py`

最小职责：

- `reset()`
- `step(action)`
- `advance_to_decision_event()`
- `build_runtime_state_view()`
- `compute_reward()`
- `is_done()`

职责边界（必须固定）：

- `env_adapter` 是唯一的运行时真相源，负责事件推进、状态转移、奖励结算、终止判定
- `env_adapter` 不再单独重实现一套候选动作筛选或 `action_mask` 逻辑
- 当环境推进到决策事件时：
  1. `env_adapter` 调用 `planner_bridge.maybe_replan(...)` 刷新 `CoarsePlanView`
  2. `env_adapter` 调用 `candidate_builder.build(...)` 生成 `observation + action_mask + action_lookup`
  3. policy 或 baseline 在该上下文上选动作
  4. `env_adapter.step(action)` 执行真实状态转移并结算 reward

**Phase 5 阶段的接口边界明晰**

Phase 5 要求先实现 `env_adapter`，而 `planner_bridge` 和 `candidate_builder` 是 Phase 6 的产物。
为避免 Phase 5 阻塞在 Phase 6 依赖上，采用以下策略：
Phase 5 的实现与验收以 4.5.1 的分层定义为准；4.5 主体仅提供统一语义总则。

**（1）inline planner：Phase 5 内置最小 CoarsePlanView 构建逻辑**

`env_adapter` 内部实现私有方法 `_build_coarse_plan_view(t_now: float) -> CoarsePlanView`，
在每次 `advance_to_decision_event()` 触发决策点时调用，替代 Phase 6 的
`planner_bridge.maybe_replan()`。Phase 5 的语义固定为：

- `_build_coarse_plan_view(t_now)` 每次调用都按当前 `t_now` **重新构建**一份新的 `CoarsePlanView`
- `plan_version = 0`（整局固定，不递增）
- 已过站点依赖 `departure_time > t_now` 的动态过滤自然从未来骨架中消失

Phase 6 接入 `planner_bridge` 后，`CoarsePlanView` 不再是“每次现算的一次性对象”，而变为
`env_adapter` 内部持有的低频刷新缓存对象。因此，切换方式**不能**简化为“只把
`_build_coarse_plan_view()` 的单个调用点替换成 `planner_bridge.maybe_replan(...)`”；
更稳妥的实现约束应为：

- `env_adapter` 新增 `_current_coarse_plan: CoarsePlanView | None`，`reset()` 时显式清空
- `env_adapter` 新增 `_active_launch_stations: set[str]`，作为执行侧实时触发站点集合；
  其初值来自当前 coarse plan 的 `launch_candidate_stations`，但后续允许执行层按规则实时删减
- `env_adapter` 新增统一入口 `_refresh_coarse_plan_if_needed()`：
  - **Phase 5 路径**：当未注入 `planner_bridge` 时，直接返回
    `_build_coarse_plan_view(self._t_now)`，行为保持“每次重建、`plan_version=0` 不变”
  - **Phase 6 路径**：由环境侧先组装 `runtime_state` 与 `trigger_ctx`，再调用
    `planner_bridge.maybe_replan(runtime_state, trigger_ctx)`；若返回对象的 `plan_version`
    相比 `_current_coarse_plan` 发生递增，则整体替换 `_current_coarse_plan`，并用新的
    `CoarsePlanView.launch_candidate_stations` **整体重置** `_active_launch_stations`
- Phase 6 中凡是消费 coarse-plan 语义的 `env_adapter` 内部逻辑（如
  `DecisionContext` 构建、卡车到站触发判断、WAIT 的 `t_next_global_decision_event`
  计算、reservation timeout 检查、mode C 送达后复核等），都应通过该统一入口或当前缓存快照取值，
  不应继续在各处散落直接调用 `_build_coarse_plan_view(t_now)`

其中 `trigger_ctx`（`PlannerTriggerContext`，重规划触发上下文）由环境侧基于内部事件账本与
当前 coarse plan 参考信息组装；它属于 env-side orchestration（环境侧编排层）输入，而不是
`RuntimeStateView` 的一部分。

`_build_coarse_plan_view(t_now)` 构建一个完整合法的 `CoarsePlanView` 对象，所有字段
必须满足 `contracts.py` 的 `__post_init__` 校验。各字段的 Phase 5 临时填充规则如下：

**元数据字段**：

```text
plan_version  = 0（整局固定，Phase 5 不实现重规划）
issued_at     = t_now
valid_until   = upper_horizon_sec（整局有效，不触发重规划）
```

**卡车骨架部分（动态过滤，不整局固定）**：

`reset()` 时从 Phase 4 产出的 `truck_backbone_route.json`、`truck_eta_map.json`、
`route_drift_ref.json` 以及巡站循环追加的节点，合并构造内部缓存 `_full_backbone_cache`。

`_full_backbone_cache` 是一个**有序列表**，每个条目为：

```python
(node_id: str, arrival_time: float, departure_time: float)
```

同一个 `node_id` 可以出现多次，对应卡车多圈经过同一站点的不同访问记录，顺序严格按
`arrival_time` 升序排列。这是与字典结构的关键区别——字典只能存一条记录，无法表达
"S2 在 t=800 和 t=2400 各经过一次"的信息。

每次 `_build_coarse_plan_view(t_now)` 调用时，按以下两步构造骨架字段：

**第一步：时间过滤**

```text
保留条目的条件：departure_time > t_now
```

过滤后得到一个有序子列表，包含所有卡车尚未离开的未来访问记录。

**第二步：去重（保留每个 node_id 的最近一次未来访问）**

遍历过滤后的子列表，每个 `node_id` 只保留首次出现的条目，后续重复丢弃。
由于列表按 `arrival_time` 升序排列，首次出现即为该节点**下一次到站**的记录。

这样当第一圈 S2 的 `departure_time <= t_now` 时，它被过滤掉，第二圈 S2 的条目
（`departure_time > t_now`）成为首次出现，PPO 看到的 S2 的 ETA 自动更新为第二圈
的到站时间。卡车每经过一个站点，该站点的 ETA 在下次 `_build_coarse_plan_view()`
调用时自动切换到下一圈，无需任何额外触发逻辑。

去重后的有序列表直接映射为：

```text
truck_backbone_route = [node_id for (node_id, _, _) in deduped]
truck_eta_map        = {node_id: arrival_time for (node_id, arrival_time, _) in deduped}
route_drift_ref      = {node_id: RouteDriftRef(eta_ref=arrival_time, route_index_ref=i)
                        for i, (node_id, arrival_time, _) in enumerate(deduped)}
```

使用 `departure_time` 而非 `arrival_time` 做过滤的原因：`advance_to_decision_event()` 可能
恰好在卡车到站时刻触发（`arrival_time == t_now`），此时卡车仍停在站点，该节点对
`launch_candidate_stations` 和 `recovery_pool` 仍然有效。若用 `arrival_time > t_now`
（严格大于），会把当前刚到达的站点错误地踢出候选集，导致 `riding_with_truck` 决策点
丢失和 mode C 回收候选丢失。`departure_time > t_now` 语义上等价于"卡车尚未离开该节点"，
是正确的"未来节点"判定条件。

`truck_eta_map` 对外暴露的 `arrival_time` 固定解释为
`t_arrive_truck(node)`（卡车到达该节点的绝对仿真时刻）；它不是 duration（持续时间），
不能与 `fly_time(...)` 直接比较。过滤逻辑内部使用 `departure_time`，
两者分工固定，不混用。

**空骨架退化开关（`allow_empty_backbone_route`）**：

`_build_coarse_plan_view(t_now)` 在过滤后 `deduped` 为空时的行为由该开关控制：

- **poisson 模式**：开关保持 `False`（默认），骨架为空视为契约错误，尽早暴露巡站循环问题
- **benchmark / hybrid 模式**：`env_adapter.reset()` 自动将开关设为 `True`，
  骨架耗尽为合法退化状态，构造以下退化对象：

```text
truck_backbone_route      = ()
truck_eta_map             = {}
route_drift_ref           = {}
launch_candidate_stations = ()
```

同时：

- 所有订单的 `recovery_pool` 置空（`mode C` 不再有 coarse 候选边界）
- 所有 UAV 可配送订单的 `policy_mode_mask` 收缩为 `{B}`，使 PPO 只能选 `mode B` 或全局
  `WAIT`
- `planner_mode_cap` 可继续保留 `{B, C}` 作为 planner 语义边界，但真正暴露给 actor 的
  `policy_mode_mask` 在该退化状态下不再包含 `C`

**巡站循环节点的重复问题**：`CoarsePlanView.truck_backbone_route` 契约要求节点不重复
（`contracts.py:151`）。巡站循环会重复经过同一 station/depot，因此 `_build_coarse_plan_view(t_now)`
在过滤后，对 `truck_backbone_route` 做去重处理：**保留每个节点在过滤后序列中的首次出现**，
后续重复访问丢弃。`truck_eta_map` 和 `route_drift_ref` 同步只保留首次出现的 ETA 和参考位置。
这样 `CoarsePlanView` 始终满足无重复节点的契约，同时巡站循环的物理执行（停靠事件队列）
不受影响，两者独立维护。

**订单相关切片（每次调用时扫描当前订单池）**：

- `weight > heavy_drone.payload_capacity` 的 static_orders → `planner_mode_cap = {A}`，
  不进入 `authorized_orders`
- 其余订单（含 poisson 新单）→ `planner_mode_cap = {B, C}`，`policy_mode_mask = {B, C}`，
  进入 `authorized_orders`
- 若 `allow_empty_backbone_route = true` 且当前 `truck_backbone_route` 为空，
  则上述 UAV 可配送订单的 `policy_mode_mask` 收缩为 `{B}`；`recovery_pool` 保持全空
- `recovery_pool` 填入当前时刻过滤后（去重后）的 `truck_backbone_route` 全部节点，
  作为 `mode C` 的 coarse 候选边界；最终 `action_mask` 仍需在 Phase 5 执行硬合法性过滤
- `launch_candidate_stations` 在 Phase 5 直接放宽为
  `future_backbone_station_nodes`（由过滤后的 `truck_backbone_route` 派生的未来
  `station` 节点子集）
- **不包含** customer 停靠点，也**不包含**场景中卡车未来不会经过的 station
- 不再使用 `support_radius_km` / `min_orders_to_trigger` 做二次筛选；这部分启发式延后到
  Phase 6
- `node_charge_load_budget` 对所有 station/depot 填 `0`（Phase 5 不建模充换电负载预算）

**order_priority_band / order_pre_score 打分规则**：

```text
score(order_i) = max(0, deadline_i - t_now)   // 剩余时间窗，单位秒
```

剩余时间窗越小，分数越低，优先级越高（越紧迫越先处理）。`order_priority_band` 按
剩余时间窗三等分切分为 `0`（紧急，≤ 1/3 窗口）、`1`（正常）、`2`（宽松，> 2/3 窗口）。

Phase 5 对这两个字段采用降级用法：

- `order_priority_band`（优先级分桶）与 `order_pre_score`（预排序分数）**仅用于日志、
  调试输出与稳定排序**
- 它们**不参与强裁剪**，不作为 `authorized_orders`（被上层放行给 PPO 的订单集合）的
  再次截断条件
- 若 Phase 5 里需要做稳定排序，方向固定为：先按 `order_pre_score` **升序**
  （剩余时间窗越小越靠前），再按 `order_id` 做稳定 tie-break
- Phase 6 若需要控制张量预算或做更强的候选集压缩，可再恢复基于分数的硬截断逻辑
  ；若恢复硬截断，也必须优先保留 `order_pre_score` 更小（更紧急）的订单

**重量阈值来源**：`reset()` 时从 `drone_params.yaml` 读取 `heavy_drone.payload_capacity`，
同时断言：

```python
assert cfg.poisson_weight_max_kg <= heavy_drone.payload_capacity
```

不允许硬编码字面量 `10.0`，避免两个配置独立漂移后 mode A 筛选逻辑悄悄失效。

**（1b）inline candidate builder：Phase 5 内置最小 action_mask 构建逻辑**

Phase 5 同样内置私有方法 `_build_action_mask(drone_id, coarse_plan, t_now)`，替代
Phase 6 的 `candidate_builder.build()`。Phase 6 接入时只需替换调用点，`env_adapter`
本身不需要改动。

Phase 5 inline candidate builder 的最小职责：

1. **候选订单**：从 `coarse_plan.authorized_orders` 中取当前 pending 订单，
   Phase 5 不再按 `order_pre_score` 做强截断；`order_priority_band` /
   `order_pre_score` 仅用于日志、调试和稳定排序，不改变可见订单集合
2. **候选模式**：对每个候选订单，从 `coarse_plan.policy_mode_mask[order_id]` 读取，
   只保留订单级派送模式 `B/C`；`WAIT` 不出现在 `policy_mode_mask` 中
3. **候选动作合法性**：
   - 对 `mode B`：仅当 `deliver_leg_feasible`（执行 `drone → order_i` 可达）且
     `exists_mode_b_return_host_feasible`（订单送达后至少存在一个能安全到达的
     `station / depot`）时保留；`mode B` 不读取 `recovery_pool`，也不在 action space
     中暴露具体返程宿主选择
   - 对 `mode C`：从 `coarse_plan.recovery_pool[order_id]` 中过滤，并且**必须同时满足**
     - `truck_eta_map[node] > t_deliver`（该节点仍在卡车未来路径上；即 `t_arrive_truck(node) > t_deliver`）
     - `t_arrive_uav(deliver -> node) + rendezvous_eta_safe_margin_sec <= t_arrive_truck(node)`（汇合安全时间裕量成立）
     - `E_need(deliver -> node, payload=0) + drone.safe_margin_j <= E_rem_after_delivery`（送达后对该节点能量可达）
     - 再截断到 `max_candidate_recovery_per_order`
4. **全局 WAIT 动作**：无论当前是否存在合法派送动作，始终额外保留一个
   `pure WAIT action`（纯 WAIT 动作）；其语义是“本次决策窗口不绑定任何订单，不起飞 /
   不派送”，从而保证 `WAIT` 是真正的全局动作，而不是 `WAIT(order_i)` 的多份等价副本

Phase 6 的 `candidate_builder` 在此基础上增加：`predicted_queue_time` 软截断、
`best_mode_b_return_score`（某订单若选 mode B 时的最优返程总代价估计）、
`best_mode_b_host_type`（该最优返程宿主类型：`station` 或 `depot`）、
更丰富的 runtime feature、`riding_with_truck` 触发的实时 recovery_pool 计算、
observation tensor 生成。

**（1c）`recover_node` 单次承诺 + 送达后执行层复核**

为保持 Phase 5 的训练接口稳定，`env_adapter.step(action)` 的输入仍然是一次性
`dispatch_action`（起飞前一次性派送动作），不拆成"派送阶段动作"与"返程阶段动作"两次 PPO
决策。实现口径固定如下：

- 当 `mode = B` 时，不写入 `selected_recover_node`；订单送达后由环境内部执行
  `_select_return_host_mode_b(...)`，在所有能量可达的 `station / depot` 中按确定性规则选择
  实际返程宿主 `effective_return_host`
- 当 `mode = C` 时，`selected_recover_node`（策略原选回收节点）在 dispatch 时写入
  环境内部 sidecar state，作为该架 UAV 当前订单的返程承诺
- `delivery_event`（订单送达事件）**不触发新的 PPO 决策点**；它只触发执行层的
  `post_delivery_revalidation`（送达后执行层复核）
- `mode B` 的确定性返程宿主选择规则固定为：

  ```text
  score_mode_b(node_j)
    = fly_time(deliver_i -> node_j)
    + predicted_queue_time(node_j, t_arrive_j)
    + service_time(node_j)
  ```

  其中：
  - `t_arrive_uav(deliver_i -> node_j) = t_deliver + fly_time(deliver_i -> node_j)`
  - `predicted_queue_time(node_j, t_arrive_uav(deliver_i -> node_j))` 在 `env_adapter`
    中直接复用
    `ChargingHost.estimate_wait_time(t_arrive_uav(deliver_i -> node_j))`（运行时真实宿主队列估计）
  - `service_time(node_j)` 取该宿主的 `swap_time`
  - 只在 `energy_feasible(deliver_i -> node_j)` 为真的候选宿主中取 `score_mode_b` 最小者

  **执行层真实口径 vs 观测侧近似口径必须区分：**

  - `env_adapter`（训练环境适配器）执行层在 mode B 宿主选择时，使用
    `ChargingHost.estimate_wait_time(...)`（运行时真实宿主队列估计）
  - `candidate_builder.build()`（候选动作构建）在生成
    `best_mode_b_return_score`（mode B 最优返程总代价估计）时，**不得**调用
    `ChargingHost.estimate_wait_time(...)`，因为它只可见 `RuntimeStateView.node_states`
    （运行时真值快照中的节点侧只读字段）
  - 因此，Phase 6 / Phase 7 观测侧使用的 `predicted_queue_time_est`
    （预计排队时间近似）被明确视为**粗近似特征**，只要求提供稳定、可复现的相对量级；
    **不要求**与执行层真实等待时间完全一致，也不要求在 `t_arrive_uav` 到来前对未来
    队列演化做滚动预测

  若最优宿主类型为 `station`，则转入 `return_to_station`；若为 `depot`，则转入
  `return_to_depot`
- `post_delivery_revalidation` 至少复核三个条件：
  - `energy_feasible`（送达后剩余电量对原定回收点是否仍可达）
  - `rendezvous_time_feasible`（送达后对原定回收点的汇合时序是否仍成立）
  - `node_still_valid`（原定回收点是否仍在当前卡车未来合法交接节点集合中）
- 若复核通过，则
  `effective_recover_node`（执行层最终采用的回收节点）继续取 `selected_recover_node`，
  并进入 `return_to_rendezvous`
- 若复核失败，则不做 `second_ppo_decision_after_delivery`（送达后二次 PPO 决策），
  而是直接触发 `reservation_timeout`（预留失效）或 `soft_miss`（软失配），进入
  `fallback_recovery`（兜底返程），并由环境按 `deterministic_fallback`
  （确定性兜底返程规则）选择仓库或最近可达充换电站

这样设计的目的有两点：

- 保持 `recover_node` head（回收节点决策头）的责任归属清晰，避免环境在送达后"偷偷改写"
  策略原决策而稀释训练信号
- 避免把一单任务拆成两个 actor 决策窗口，导致 `reservation` 建立时机、
  `step(action)` 接口和 rollout 采样结构全部改变

**（2）mode A 背景订单的执行主体**

mode A 的职责分两层，不允许混淆：

- **派发标记（inline planner 职责）**：`_build_coarse_plan_view()` 在构建订单切片时，
  把 `weight > heavy_drone.payload_capacity` 的 static_orders 标记为 `planner_mode_cap = {A}`，
  排除出 `authorized_orders`。这是 mode A 的全部"派发决策"，在 `reset()` 时一次性确定，
  整局固定，不需要运行时动态决策。poisson 订单重量上限 ≤ `heavy_drone.payload_capacity`，
  所以 poisson 新单永远不会是 mode A。

- **物理执行（env_adapter 职责）**：`reset()` 时从 Phase 4 的 `truck_execution_route.json`
  读取完整停靠序列，按 `_planned_route_stops` 的数据格式（`node_type`、`node_id`、
  `position`、`arrival_time`、`departure_time`、`order_id`）写入 `env_adapter` 内部事件队列。
  `_advance_to_event(t_next)` 推进时，扫描队列中 `arrival_time <= t_next` 的 customer
  停靠点，触发"mode A 订单完成"事件，更新系统上下文统计，不触发 PPO 决策。

  注意：Phase 4 导出的客户停靠服务时长固定为 `0.0`（`CUSTOMER_SERVICE_TIME_SEC = 0.0`），
  Phase 5 沿用这个简化，不建模额外服务时长。

- **卡车位置同步（env_adapter 职责）**：`reset()` 时调用 `truck.set_route()` 写入
  Phase 4 路线的几何数据，设置 `_departure_time = 0.0`。之后每次 `_advance_to_event(t_next)`
  时，调用 `truck.get_location(t_next)` 更新 `truck.current_loc`。`Truck.get_location()`
  已实现基于 `_route_data` 和 `_departure_time` 的线性插值，不依赖 `tick_all()`，直接按
  仿真时间查询即可。这样 `candidate_builder` 读取 `truck.current_loc` 时始终是正确的物理
  真值，`t_arrive_truck` 计算不会因位置停滞而失真。

**（4）卡车循环巡站机制（poisson 训练模式专用）**

**问题背景**：poisson 训练模式下，卡车路线基于 static_orders 的重货订单一次性规划，
episode 时长 3600s，但卡车可能在 1000s 左右就走完路线回仓库。此后 `truck_backbone_route`
为空，`recovery_pool` 全空，`launch_candidate_stations` 全空，PPO 只能选 mode B 或 WAIT，
训练分布严重偏斜，mode C 的学习信号消失。

**触发条件**：`reset()` 时，在完成 Phase 4 路线读入后，检查以下条件是否同时成立：

```text
触发循环巡站的条件（全部满足）：
  1. order_source_mode == poisson
  2. truck_execution_route 最后一个停靠点是 depot（卡车会回仓）
  3. 最后一个停靠点的 arrival_time < upper_horizon_sec - patrol_min_remaining_sec
     （卡车回仓时刻距 episode 结束仍有足够时间，默认 patrol_min_remaining_sec = 600s）
```

条件 3 的参数 `patrol_min_remaining_sec = 600` 从 `rh_alns_cmrappo.yaml` 的 `planner`
段读取（新增配置项），表示"剩余时间不足此值时不再追加巡站，避免生成极短的无意义路段"。

**执行方案**：

`reset()` 时，若触发条件成立，在 Phase 4 路线末尾追加一段或多段"巡站循环"，直到
填满 `upper_horizon_sec`。具体步骤如下：

1. **起点**：上一段路线的终点 depot，`t_start = 上一段路线最后一个停靠点的 arrival_time`

2. **站点选取**：采用贪心最近邻算法，每选一个站点后更新当前位置，再从剩余站点中
   选距离最近的下一个，构成一次巡站循环：`depot → S_1 → S_2 → ... → S_k → depot`。
   具体步骤：
   - 初始当前位置 = depot 坐标
   - 重复 `min(len(stations), patrol_stations_per_loop)` 次：
     从未选站点中选直线距离当前位置最近的站点，加入序列，更新当前位置
   - **最后一站必须是终点 depot**：在所有 station 停靠点之后、追加下一圈起点之前，
     插入一个 depot 停靠点作为本圈终点；depot 停靠点不写入 `_full_backbone_cache`
     骨架缓存，仅用于维持卡车物理路线的几何连续性与 ETA 单调性
   站点数量 `k` 取 `min(len(stations), patrol_stations_per_loop)`，
   `patrol_stations_per_loop` 默认值为 `planner.phase4_min_future_fixed_nodes`（即 3），
   从配置读取，不硬编码。

3. **ETA 估算**：巡站循环使用站点间直线距离 / `truck.speed` 估算 ETA，不做 OSM 精确规划。
   原因：巡站路段不承载 mode A 订单，只需保证 `truck_backbone_route` 有未来节点，
   ETA 精度要求低于 Phase 4 的执行路线。

4. **循环追加**：每追加一次巡站循环后，检查当前最后一个停靠点的 `arrival_time` 是否
   仍满足条件 3，若满足则继续追加下一次循环，直到不满足为止。

5. **停靠点格式**：追加的巡站停靠点使用与 Phase 4 相同的 `_planned_route_stops` 格式，
   `node_type = "station"`，`order_id = None`，`departure_time = arrival_time`（不停留）。
   这些停靠点不触发 mode A 订单完成事件，只用于维护 `truck_backbone_route` 的未来节点。

6. **骨架路线同步**：追加的巡站停靠点中，`node_type == "station"` 的节点同步写入
   全量骨架缓存（`_full_backbone_cache`），供 `_build_coarse_plan_view(t_now)` 的
   `departure_time > t_now` 过滤逻辑使用。每圈终点的 depot 停靠点**不写入**
   `_full_backbone_cache`，原因是：depot 已经作为 Phase 4 原始路线的终点节点存在于
   骨架缓存中；巡站循环的 depot 停靠点只是物理路线的几何连接点，重复写入会导致
   `_build_coarse_plan_view()` 去重后 depot 的 ETA 被错误地更新为巡站循环的时刻，
   而不是 Phase 4 原始路线的终点时刻。

**与 Phase 4 路线的边界**：

- Phase 4 的 `truck_execution_route.json` 只包含基于 static_orders 的原始路线，不包含巡站循环
- 巡站循环完全在 `env_adapter.reset()` 内部生成，不写回 Phase 4 产物文件
- `truck.set_route()` 调用时传入的是原始路线 + 巡站循环的完整几何序列
- `truck_execution_route` 的语义（物理验证与路径审计）不受影响

**benchmark / hybrid 模式下的行为**：

巡站循环仅在 `order_source_mode == poisson` 时触发。benchmark 和 hybrid 模式下，
卡车路线严格按 Phase 4 产物执行，不追加巡站循环，以保证 benchmark 验证的可复现性。

benchmark / hybrid 模式下，`allow_empty_backbone_route` **自动开启**（无需显式配置），
原因是：这两种模式不追加巡站循环，卡车提前回仓后骨架自然耗尽是预期的合法业务状态，
不应视为契约错误。自动开启后，`_build_coarse_plan_view()` 在骨架耗尽时构造空骨架退化对象：
`truck_backbone_route = ()`、`recovery_pool` 全空、`launch_candidate_stations = ()`、
`policy_mode_mask` 收缩为 `{B}`，PPO 只能选 mode B 或全局 WAIT。

poisson 模式下，`allow_empty_backbone_route` 保持 `false`（默认值）；若骨架为空，
仍视为契约错误，用于尽早暴露巡站循环生成逻辑的问题。

**（3）reservation 状态的存储位置**

reservation 状态存在 `env_adapter` 内部，不挂在 `Drone` 对象上（原因同 `TrainingDroneState`：
不污染在线仿真引擎的实体层）。

`env_adapter` 内部维护两个字典：

```python
_reservations: dict[str, ReservationState]  # drone_id → 当前 reservation
_reservation_count: dict[str, int]          # node_id → 当前预留该节点的无人机数量
```

`ReservationState` 是训练侧专属 dataclass：

```python
@dataclass
class ReservationState:
    recover_node: str   # 预留的回收节点 ID
    issued_at: float    # reservation 建立时刻（仿真秒）
    expires_at: float   # reservation 过期时刻（仿真秒）
```

**命名必须与在线 solver 的 `RendezvousContract` 严格区分**：`RendezvousContract` 是
`B_WAIT` 模式下卡车锚点锁定的时空约束，绑定的是卡车必须在 `latest_departure` 前到达锚点，
生命周期是 `active/fulfilled/expired/released`。`ReservationState` 是 mode C 返程的局部
承诺，timeout 条件是
`t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j)` 等时序约束，两者语义完全不同，
不能混名，也不能复用。

`_reservation_count` 通过 `build_runtime_state_view()` 暴露给外部，作为 `candidate_builder`
构建候选集时的节点拥挤信号，不直接让外部访问 `env_adapter` 内部字典。

生命周期：`reset()` 时两个字典清空；`step()` 时随状态转移更新；reservation timeout 检查
在每次 `_advance_to_event()` 时顺带扫描。timeout 参数（`alpha`、`beta`、`gamma`）从
`rh_alns_cmrappo.yaml` 的 `reservation` 段读取，已在配置文件中预留。

**（5）build_runtime_state_view() 接口 schema**

`build_runtime_state_view()` 返回全局运行时真值快照，供 `candidate_builder` 和外部
观测构建使用。**当前决策上下文（deciding_drone_id（当前决策无人机）、trigger_type（触发类型）、
trigger_station_id（触发站点））不放入此接口**，而是作为 `candidate_builder.build()`
（候选动作构建）的独立参数传入，原因是：
它们不是全局真值，而是"本次 build 面向哪架 UAV"的局部上下文；同一时刻可能有多个
决策 UAV，共用同一个 `runtime_state` 更干净。

因此，Phase 6 起 `candidate_builder.build()` 的推荐接口签名固定为：

```python
candidate_builder.build(
    runtime_state,
    coarse_plan,
    deciding_drone_id,
    trigger_type,
    trigger_station_id,
    last_seen_plan_version,
)
```

其中：
- `deciding_drone_id`（当前决策无人机）用于确定"本次候选集是为哪架 UAV 构建"
- `trigger_type`（触发类型）用于区分 `idle`（地面空闲触发）与 `riding_with_truck`（随车搭载触发）
- `trigger_station_id`（触发站点）在 `idle` 触发时为 `None`，在 `riding_with_truck` 触发时为卡车当前到达站点 `S_k`
- 若 `trigger_type` 表示 `riding_with_truck` 路径，则 `trigger_station_id`
  为**必填**；缺失时实现应直接报错，而不是静默退化到 `idle`/静态 recovery pool 路径
- `last_seen_plan_version`（上次看到的规划版本号）由 **env 外部消费侧调用方** 维护，
  表示该 `deciding_drone_id`（当前决策无人机）在本 episode（回合）上一次**正式消费的**
  `CoarsePlanView.plan_version`（规划版本号）。训练主路径下通常由 rollout collector
  （轨迹收集器）维护；baseline / 验证脚本也应按同一语义维护

`plan_version_delta`（规划版本增量）的来源固定为：

```text
plan_version_delta
  = coarse_plan.plan_version - last_seen_plan_version
```

补充约束：

- 该量属于**跨决策步的 per-UAV 状态**，不是 `RuntimeStateView`（运行时真值快照）
  的原生字段，也不是 `env_adapter`（训练环境适配器）推进仿真所需状态
- 因此不并入 `env_adapter` / `RuntimeStateView`，也不允许把 `candidate_builder`
  变成有状态组件来隐式维护它
- `last_seen_plan_version` 的更新时机固定为：
  由外部调用方在**当前决策点被正式消费**时更新，而不是在
  `candidate_builder.build(...)` 调用后立即更新。这样可避免同一
  `DecisionContext`（决策上下文）被重复 build / 调试 / 对比时，错误地把
  `plan_version_delta` 提前归零
- 某架 UAV 在本 episode 第一次决策时，若调用方尚未记录其历史值，则应取
  `last_seen_plan_version = coarse_plan.plan_version`，使首次
  `plan_version_delta = 0`；这表示“该 UAV 尚未观测到规划版本变化”，为当前推荐初始化语义

`build_runtime_state_view()` 返回的 schema 定义如下：

```python
@dataclass(frozen=True)
class RuntimeStateView:
    t_now: float                              # 当前仿真时间（秒）

    # 卡车侧
    truck_current_loc: Position3D             # 当前物理位置（每次 _advance_to_event 后同步）

    # 无人机侧（每架无人机一条记录）
    drone_states: Mapping[str, DroneStateView]

    # 订单池（只读引用，Phase 5 可用 Order 对象；长期建议只读 snapshot）
    pending_orders: Mapping[str, Order]
    assigned_orders: Mapping[str, Order]

    # 站点侧（station + depot 均包含）
    node_states: Mapping[str, NodeStateView]

    # reservation 拥挤信号（node_id → 当前预留该节点的无人机数量）
    reservation_count: Mapping[str, int]

@dataclass(frozen=True)
class DroneStateView:
    drone_id: str
    training_state: str                       # TrainingDroneState 枚举值
    current_loc: Position3D
    battery_current: float                    # 当前电量（焦耳）
    battery_max: float                        # 满电容量（焦耳）
    battery_ratio: float                      # 当前电量比例
    carrying_order_id: Optional[str]
    home_type: str                            # "DEPOT" 或 "TRUCK"（drone_source_type）
    # 静态能力字段（供 candidate_builder 计算 fly_time / t_arrive_uav / E_need）
    cruise_speed: float                       # m/s
    payload_capacity: float                   # kg
    empty_weight: float                       # kg
    k1: float                                 # 诱导功率系数
    k2: float                                 # 废阻功率系数
    reservation: Optional[ReservationStateView]  # 当前 reservation，无则 None

@dataclass(frozen=True)
class ReservationStateView:
    recover_node: str
    issued_at: float
    expires_at: float

@dataclass(frozen=True)
class NodeStateView:
    node_id: str
    node_type: str                            # "station" 或 "depot"
    position: Position3D
    parking_slots: int                        # 充换电并行槽位数
    swap_time: float                          # 单次充换电服务时长（秒）
    queue_length: int                         # 当前排队数
    available_slots: int                      # 当前空闲服务槽位数
```

说明：
- `truck_eta_map` 和 `truck_backbone_route` 不放入 `RuntimeStateView`，它们属于
  `CoarsePlanView` 语义边界，重复放会形成双真相源
- `DroneStateView` 中的静态能力字段（`k1`、`k2`、`cruise_speed` 等）在 `reset()` 时
  从 `Drone` 实体对象一次性读取并缓存，运行期不变
- `NodeStateView` 中的 `queue_length` 和 `available_slots` 在每次 `_advance_to_event()`
  后从 `ChargingHost` 实体对象实时读取
- 任何依赖节点拥挤度的 runtime feature（如 `best_mode_b_return_score` 的队列项、
  `predicted_queue_time` 软信号）都必须在每次 `candidate_builder.build()` 时基于
  **当前** `RuntimeStateView.node_states` 重新计算，禁止跨 build 缓存旧快照
- Phase 6 首版**不扩充** `NodeStateView` 字段，不额外暴露
  `earliest_slot_release_eta`（最早空槽释放时刻）等服务中细粒度状态；因此观测侧的
  `predicted_queue_time_est`（预计排队时间近似）必须只依赖
  `queue_length`（当前排队数）、`available_slots`（当前空闲槽位数）与
  `swap_time`（单次充换电服务时长）定义

**（5a）PlannerTriggerContext（重规划触发上下文）**

`planner_bridge.maybe_replan(...)` 除 `runtime_state` 外，还需要一份独立的
`PlannerTriggerContext`（重规划触发上下文）。它的职责是承载"是否需要重规划"所需的
planner 侧触发信号，同时保持 `RuntimeStateView` 只表达全局运行时真值。

推荐接口定义如下：

```python
@dataclass(frozen=True)
class PlannerTriggerContext:
    t_now: float
    backlog_new_orders: int
    fallback_count_in_window: int
    hard_failure_count_in_window: int
    route_drift_ratio: float
```

各字段语义固定为：

- `t_now`：当前仿真时刻
- `backlog_new_orders`（新积压订单数）：自上一个 `plan_version`（规划版本号）生效后，
  新进入可调度池、但尚未被当前 coarse plan 吸收的无人机订单数
- `fallback_count_in_window`（fallback 窗口计数）：最近窗口内进入
  `fallback_recovery`（兜底返程飞行状态）的次数
- `hard_failure_count_in_window`（硬失败窗口计数）：最近窗口内进入
  `airborne_energy_failure`（半空电量耗尽硬失败状态）的次数
- `route_drift_ratio`（路线漂移比例）：基于当前卡车运行时位置 / 进度与
  `CoarsePlanView.route_drift_ref`（路线漂移参考基线）计算出的派生量；
  其含义是“当前 coarse plan 认为卡车在骨架路线上的完成进度”与
  “卡车真实执行中已完成的骨架节点进度”之间的归一化偏差。

`route_drift_ratio`（路线漂移比例）的 Phase 6 首版冻结公式如下：

```text
设当前 coarse plan 的 route_drift_ref 包含 N = total_backbone_count（骨架节点总数）个节点，
每个节点只按当前 coarse plan 快照中的首次未来访问计数一次。

expected_count(t_now)（时刻 t_now 的预期已完成骨架节点数）
  = count(node_id | route_drift_ref[node_id].eta_ref <= t_now)

completed_backbone_count（已完成骨架节点数）
  = count(node_id | 自当前 plan_version 生效后，卡车已实际到达该 backbone 节点至少一次)

route_drift_ratio（路线漂移比例）
  = |completed_backbone_count - expected_count(t_now)| / total_backbone_count
```

变量口径固定为：

- `total_backbone_count`（骨架节点总数）：当前 `CoarsePlanView.route_drift_ref`
  中节点数，即当前 coarse plan 快照下用于 route drift 检查的 backbone 节点总数；
  若 `total_backbone_count = 0`（如 `allow_empty_backbone_route=True` 且未来骨架为空），
  则 `route_drift_ratio` 固定定义为 `0.0`
- `expected_count(t_now)`（时刻 `t_now` 的预期已完成骨架节点数）：按
  `route_drift_ref[node_id].eta_ref`（参考到达时刻）统计，到当前时刻理论上“应已到达”的
  backbone 节点个数
- `completed_backbone_count`（已完成骨架节点数）：按卡车真实执行统计，到当前时刻“已实际到达”
  的 backbone 节点个数；这里的“完成”专指**到达该 backbone 节点这一里程碑已发生**，
  不要求卡车已经离站

实现约束固定为：

- `completed_backbone_count` 与 `expected_count(t_now)` 都只统计**当前 coarse plan 快照中的
  backbone 节点**；不统计 customer 停靠点，也不统计该快照之外的重复巡站访问
- 由于 `route_drift_ref` 只存 `eta_ref`（参考到达时刻）与 `route_index_ref`（参考路线顺序索引），
  Phase 6 首版的 route drift 检查统一采用**arrival milestone（到达里程碑）**口径，
  不引入额外的 departure 参考字段
- `route_index_ref`（参考路线顺序索引）在 Phase 6 首版中主要用于保证
  `route_drift_ref`（路线漂移参考基线）的顺序可解释性，并为后续若升级为连续
  `route_progress`（路线进度）版本时提供索引基线；首版冻结公式**不要求**直接使用
  `route_index_ref` 参与数值计算
- 上述 arrival 口径**只用于** `route_drift_ratio` 的重规划触发判断；它不等同于
  `truck_backbone_route` / `launch_candidate_stations` / `recovery_pool` 在其他章节里使用的
  `departure_time > t_now`（离站后才算已过站）过滤口径，两者服务于不同业务目的，不能混用
- `route_drift_ratio` 的直接消费方只有
  `planner_bridge.maybe_replan(...)`（粗规划桥接器：是否重规划）中的
  `route_drift_trigger_ratio`（路线漂移触发阈值）比较；它不直接改变执行层状态转移

实现边界固定为：

- `PlannerTriggerContext` **不并入** `RuntimeStateView`
- `route_drift_ratio` **不由** `env_adapter.build_runtime_state_view()` 直接计算后回填，
  因为它依赖当前 coarse plan 参考基线，不属于原始真值
- `fallback_count_in_window` / `hard_failure_count_in_window` 的原始事件来源于
  `env_adapter` 内部账本，但 `planner_bridge` 不自行维护历史事件流
- `PlannerTriggerContext` 由环境侧调用链组装后传入 `planner_bridge.maybe_replan(...)`；
  它属于 env-side orchestration（环境侧编排层）输入，而不是训练循环私有逻辑
- `launch_candidate_stations` 的执行层实时删减不并入 `PlannerTriggerContext`；该职责由
  `env_adapter` 内部 `_active_launch_stations`（执行侧实时触发站点集合）维护，
  以避免把"是否触发决策点"与"是否触发重规划"混成同一个接口
- Phase 6 首轮若只实现 `periodic(interval)`（周期重规划）这一条触发规则，其余字段允许先按
  `0` 或等价保守值传入，以先稳定接口，再逐步补全触发条件

**（6）WAIT 动作的 delta_wait 唯一化**

`idle` 状态下 WAIT 的 `delta_wait` 定义为：

```text
delta_wait = min(
    t_next_global_decision_event,
    t_now + max_wait_decision_gap_sec,
    upper_horizon_sec
) - t_now
```

其中：
- `t_next_global_decision_event`：下一个会触发 PPO 决策的全局事件时刻，包括：
  任意无人机完成充换电进入 idle、卡车到达 `_active_launch_stations`
  （执行侧实时触发站点集合）中的某个站点。
  不包括：mode A 订单完成、poisson 新订单注入、reservation timeout 等不触发 PPO 的事件。
- `max_wait_decision_gap_sec`：WAIT 推进量上限，防止极长空窗把一次 WAIT 惩罚拉得过大。
  从 `rh_alns_cmrappo.yaml` 的 `planner` 段读取，首轮建议 `60s`。

`riding_with_truck` 状态下 WAIT 的执行语义改为：

```text
进入 active_wait（resume_state = riding_with_truck）
继续随车，不起飞
直到卡车到达下一个真正会触发车上 UAV PPO 决策的站点
```

其中：
- “真正会触发车上 UAV 决策的站点”指卡车到达时刻仍属于
  `_active_launch_stations`（执行侧实时触发站点集合）的站点
- 若下一次环境返回是由其他 UAV 的决策事件触发，而当前 UAV 仍未等到该触发站，
  则当前 UAV 保持 `active_wait`，其 WAIT 承诺不被取消
- 因此 `riding_with_truck` 路径下**不再定义一次性预付的未来 `delta_wait`**；`T_idle`
  只按后续真实事件推进区间的实际 `dt` 逐段累计

**`active_wait` 状态的语义边界**：

WAIT 动作会把无人机切到 `active_wait`，但 `active_wait` 的 reward 语义按来源分两条路径：

- `idle -> active_wait`：
  - 仍表示“本次 WAIT 已经选定、该 UAV 在下一次全局决策事件前不再派送”
  - `T_idle` 固定在 `step(WAIT_action)` 调用点按精确 `delta_wait` 一次性结算
- `riding_with_truck -> active_wait`：
  - 表示“本次 WAIT 已经选定、该 UAV 继续随车等待下一个真正触发车上 PPO 决策的站点”
  - `T_idle` 不在入口一次性预付，而是在 `active_wait` 占位期间按每次真实事件推进区间
    `[t_prev, t_next)` 的实际 `dt` 持续累计

这样定义的原因是：`riding_with_truck` WAIT 可能跨越多个环境返回边界（例如中途先发生其他
UAV 的决策事件），若仍在入口一次性预付整段未来等待时长，会造成 reward 结算边界与环境实际
推进边界脱节。

`max_wait_decision_gap_sec` 新增到配置文件 `rh_alns_cmrappo.yaml` 的 `planner` 段。

**（7）同一时刻多事件的处理顺序**

事件驱动环境中，同一仿真时刻 `t_event` 可能同时发生多个事件。处理顺序分两层：

**第一层：先结算区间成本（连续时间项）**

在处理任何点事件之前，先对区间 `(t_prev, t_event)` 结算所有连续时间成本：

```text
_settle_per_dt_rewards(dt = t_event - t_prev)
```

包含：`T_overdue`、`T_wait`、`T_queue`、`T_fallback`
（fallback_recovery 状态）。

其中 `T_overdue` 的结算口径固定为：按区间 `[t_prev, t_event)` 与每个订单超时区间
`[deadline_i, +∞)` 的重叠长度累计，而不是使用 `max(0, t_now - deadline_i) * dt`
这类近似写法。这样"deadline 恰好到点且同刻送达"不会被误算一次 overdue：区间成本
先精确结算到 `t_event` 之前，送达事件再在 `t_event` 时刻处理，订单从 `_active_uav_orders()`
移除，后续不再累加。

**第二层：按以下顺序处理点事件**

```text
1. 硬失败（airborne_energy_failure）
   → 先处理，避免后续事件基于已失效无人机计算

2. UAV 订单送达，结算 R_delivery_bonus；mode A 完成只更新系统上下文统计
   → 先结算 UAV 送达正奖励，再做超时判定，避免"刚送达就被算超时"

3. 到达类事件（合并处理，再做交互）
   3a. 卡车到站
   3b. UAV 到站（含 `return_to_rendezvous`、`return_to_station`、`return_to_depot`、
       `fallback_recovery` 四类飞行段的到达，以及充换电完成、回收完成）
   3c. 由到达触发的交互：rendezvous 成功、FIFO 入队/出队、
       launch_candidate_stations 决策触发

4. reservation timeout 检查
   → 在到达类事件之后，确保"UAV 与 truck 同刻到达 rendezvous 点"先成功配对，
     再决定是否 timeout

5. poisson 新订单注入
   → 新注入的订单不进入当前时刻的 authorized_orders，
     在下一次 _build_coarse_plan_view() 调用时才被纳入，
     避免"刚出现就要立刻决策"的不合理场景

6. 生成下一个 decision context / 检查 is_done()
```

**（8）reservation timeout 落地规则（Phase 5 简化版）**

`tau_res` 公式在 Phase 5 简化为：

```text
tau_res = alpha * fly_time(deliver -> r_j) + gamma * q_est(r_j)
```

其中：`reservation_beta` 配置字段继续保留在契约 / yaml 中，但 Phase 5 不维护历史统计，
因此 `beta * t_hist(...)` 项在数值上固定为 `0`；Phase 6+ 恢复历史统计后可直接启用，
无需修改接口结构。

各项的 Phase 5 落地规则：

```text
t_hist(node_type_j, mode C) = 0
    → Phase 5 不维护历史统计，固定为 0

q_est(r_j) = host.estimate_wait_time(t_now)
    → 复用 ChargingHost.estimate_wait_time()，已考虑"服务中但队列未排起来"的情况，
      比 queue_length * swap_time 更准确

route_drift_invalid = false
    → Phase 5 不实现 route drift 检测，固定为 false

node_available = true
    → Phase 5 不建模节点不可用，固定为 true
```

timeout 判定保留三个条件（任一成立即失效）：

```text
timeout(res) =
    (t_now >= expires_at)
 OR (t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j))
 OR (energy_feasible == false)
```

这里的实现语义进一步固定为：

- 上述 timeout 检查是 `post_delivery_revalidation`（送达后执行层复核）与飞行途中持续复核
  的一部分，不是新的 PPO 决策触发条件
- 一旦 timeout 成立，环境直接放弃 `selected_recover_node`（策略原选回收节点），转入
  `deterministic_fallback`（确定性兜底返程）；**不重新构造 `action_mask` 让 PPO 二次选点**

`energy_feasible` 的判定时机：**在每次 `_advance_to_event()` 时重新计算**，不只在
reservation 建立时算一次。原因：无人机在飞往客户点途中电量持续消耗，建立时可行的
reservation 在送达后可能已经不可行。

```text
energy_feasible = drone.can_reach(
    target = node_position(r_j),
    payload = 0.0,          # 送达后空载返程
    safe_margin = drone.safe_margin_j
)
```

`T_timeout_cost` 定义为"违规量"，不是 fallback 飞行时间（避免与 `T_fallback` 重复计量）：

```text
若因 t_now >= expires_at 触发：
    T_timeout_cost = t_now - expires_at

若因 t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j) 触发：
    T_timeout_cost = t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec - t_arrive_truck(r_j)

若因 energy_feasible == false 触发：
    T_timeout_cost = 0（电量不足本身不产生时间违规量，惩罚由 hard_failure_penalty_sec 覆盖）
```

timeout 后进入 `fallback_recovery`，施加软惩罚：

```text
r_timeout = -lambda_res_timeout * T_timeout_cost
```

`T_fallback` 继续表示"后续兜底飞行的物理代价"，两者不混用。其落地结算路径固定为：

- UAV 进入 `fallback_recovery`（兜底返程飞行态）后，在每次 `_advance_to_event()` 的
  区间结算里，只要该 UAV 仍处于 `fallback_recovery`，就累计
  `delta_fallback_t = t_next - t_prev`
- 到达备用宿主（`station` 或 `depot`）的事件发生时，先结算到到达时刻之前的
  `T_fallback`，再把状态切到对应的到达后状态，后续不再继续累计 `T_fallback`
- "到达备用宿主"事件的触发依据不是模糊的当前位置比较，而是
  `_fallback_leg[drone_id].arrival_time`（兜底返程预计到达时刻）被 `_advance_to_event()`
  跨过；触发后必须立刻清空 `_fallback_leg[drone_id]`，避免到达后继续累计 `T_fallback`

- 场景固定为 `default_scene`
- 运行时订单源通过 `order_source_adapter` 注入
- 训练阶段默认 `order_source_mode = poisson`
- benchmark 验证阶段默认 `order_source_mode = benchmark`
- stochastic 验证阶段默认 `order_source_mode = poisson | hybrid`
- 每个 `step` 对应一次调度决策窗口
- 不要求一开始就是全精度连续物理仿真，但必须优先保证事件状态转移正确
- 环境推进建议采用事件驱动或准事件驱动，而不是粗时间步近似

需要先定清楚：

1. 观测包含什么
2. 动作代表什么（派送分支 factorized：`order → mode(B/C) → recover_node`，
   另有独立全局 `WAIT` 动作）
3. 奖励在什么时刻结算
4. 终止条件是什么

   episode 终止条件（任一满足即触发 `is_done() = True`）：
   ```text
   1. t_now >= upper_horizon_sec（默认 3600s，仿真时间上限）
   2. 所有订单已送达或已被强制移除（pending + assigned 均为空，N_hard_overdue 覆盖超时死单）
   3. 所有无人机均处于 airborne_energy_failure 状态（无可用无人机）
   ```
   说明：条件 1 是主要终止路径；条件 2 是提前终止（所有任务完成）；
   条件 3 是灾难性终止，应触发大额惩罚后结束。
   三者均不满足时 episode 继续推进。

这一阶段必须明确实现以下状态和事件：

1. 无人机状态

   `env_adapter` 内部维护一个 `_drone_state: dict[str, TrainingDroneState]` 覆盖层，
   使用训练侧自己的枚举，**不修改** `core/entities/primitives.py` 的 `DroneStatus`。
   原因：`DroneStatus` 是在线仿真引擎的物理状态机，`RIDING_WITH_TRUCK` 等训练侧概念
   不应反向污染基础层，否则前端 WebSocket 帧、`is_flying`、`is_dispatchable` 等属性
   都需要感知一个它们不需要处理的状态。

	   `TrainingDroneState` 枚举定义如下（仅在 `env_adapter.py` 内部使用）：

	   - `idle`：在 depot 或 station 完成充换电后空闲，可被 PPO 调度
	   - `flying_to_deliver`：飞往客户投递点
	   - `delivered`：已完成投递，等待按既定返程承诺执行后续转移：
	     `mode B` 在执行层确定性选择 `station / depot` 返程宿主，`mode C` 对
	     `selected_recover_node`（策略原选回收节点）做返程执行层复核；
	     **不是第二次 PPO 决策点**
	   - `return_to_rendezvous`：返程飞往 mode C 预定回收节点（飞行中）
	   - `waiting_for_truck`：已到达 rendezvous 节点，被动等待卡车到来（**T_wait 累计区间**）
	   - `return_to_station`：返程飞往充换电站（`mode B` 执行层已选择 `station`；
	     **不属于** `T_fallback` 的累计状态）
	   - `return_to_depot`：返程飞往仓库（`mode B` 执行层已选择 `depot`）
	   - `queueing_at_host`：已到达充换电宿主（`station / depot`），等待服务槽位
	     （**T_queue 累计区间**）
	   - `charging_or_swap`：正在充换电服务中
	   - `active_wait`：主动选择 WAIT 动作后的占位等待状态；
	     `idle -> active_wait` 路径下，`T_idle` 在 `step(WAIT_action)` 中按精确
	     `delta_wait` 一次性结算；
	     `riding_with_truck -> active_wait` 路径下，`T_idle` 按后续真实事件推进区间的
	     实际 `dt` 持续累计
	   - `fallback_recovery`：错过卡车后飞往备用宿主（`station` 或 `depot`）的整个兜底飞行阶段
	     （**T_fallback 累计区间**）
	   - `charging_on_truck`：mode C 成功回收，正在卡车上充换电（等待 `truck_drone_recover_time_s`）
	   - `riding_with_truck`：卡车上充换电完成，搭车等待下一个决策点
	   - `airborne_energy_failure`：半空中电量耗尽，硬失败终态

   状态转移中涉及的完整路径：

	   ```text
		   idle / riding_with_truck
		     → [PPO 选 dispatch_action] → flying_to_deliver
		     → [到达客户点] → delivered
	         mode C → [post_delivery_revalidation 通过]
	                  → return_to_rendezvous → [到达节点] → waiting_for_truck
	                                         → [卡车到达] → charging_on_truck
	                                                       → [充换电完成] → riding_with_truck
	               → [post_delivery_revalidation 失败 / timeout]
	                  → fallback_recovery
	                     → [到达备用 station/depot] → _on_arrive_charging_host
	                     → queueing_at_host / charging_or_swap → idle
	         mode B → [执行层选择 station] → return_to_station
	                  → [到达宿主] → _on_arrive_charging_host
	                  → queueing_at_host / charging_or_swap → idle
	               → [执行层选择 depot]   → return_to_depot
	                  → [到达宿主] → _on_arrive_charging_host
	                  → queueing_at_host / charging_or_swap → idle
		     → [idle 选 WAIT] → active_wait → [下一个全局决策事件或 gap 截断] → idle
		     → [riding_with_truck 选 WAIT] → active_wait
		                                  → [非触发站到达 / 其他 UAV 先触发决策] → active_wait
		                                  → [触发站到达] → riding_with_truck
	   ```

   说明：`charging_on_truck` 与 `riding_with_truck` 是两个独立状态，不是别名。
   前者表示"已被回收、正在车载充换电服务中"，后者表示"充换电完成、搭车等待决策"。
   `recovered` 不作为正式枚举值，其语义由 `charging_on_truck` 覆盖。

   **`riding_with_truck` 初始化来源**：`reset()` 时读取 `drone.home_type`（存储在
   `Drone` 实例上，由 `EntityManager.load_from_config()` 按 `entities.json` 的
   `home_type` 字段写入）。`home_type == SourceType.TRUCK` 的无人机初始状态设为
   `riding_with_truck`；`home_type == SourceType.DEPOT` 的无人机初始状态设为 `idle`。
   不需要任何额外配置层，`entities.json` 的 `home_type` 字段即为唯一来源。

	   - `waiting_for_truck` 与 `active_wait` 的区分：两者在状态机层面是不同的状态，
	   触发来源不同，`compute_reward()` 只需检查当前状态即可确定 `dt` 计入哪个桶。
	   其中 `active_wait` 的 `T_idle` 结算路径还需区分其恢复目标状态：
	   - `return_to_rendezvous → [到达节点] → waiting_for_truck`（被动，T_wait）
	   - `idle → [选 WAIT 动作] → active_wait(resume_state=idle)`（主动，T_idle 入口一次性结算）
	   - `riding_with_truck → [选 WAIT 动作] → active_wait(resume_state=riding_with_truck)`
	     （主动，T_idle 按真实事件推进区间持续累计）

   - `fallback_recovery` 的退出规则必须显式依赖一段内部飞行账本，而不能只靠文字描述。
   `env_adapter` 在 UAV 进入 `fallback_recovery` 时，必须同步写入一组内部执行字段：

   ```python
   _fallback_leg[drone_id] = FallbackLeg(
       host_node_id=str,          # 备用宿主节点 ID
       host_node_type=str,        # "station" | "depot"
       arrival_time=float,        # 预计到达时刻（绝对仿真秒）
   )
   ```

   这些字段仅用于执行层状态转移，不需要暴露到 `RuntimeStateView`。状态退出口径固定为：

   - 若一次 `_advance_to_event(t_next)` 结束后仍满足
     `t_next < _fallback_leg[drone_id].arrival_time`，
     则 UAV 保持 `fallback_recovery`，本次区间全部计入
     `delta_fallback_t = t_next - t_prev`
   - `env_adapter` 必须统一复用一个宿主到达处理入口，不允许在 fallback 或 mode B
     路径上手写第二套 `available_slots` 判断，也不允许 depot 路径绕过排队判断：

   ```python
   def _on_arrive_charging_host(self, drone_id: str, host: ChargingHost, t_now: float) -> None:
       host.arrive(drone_id, t_now)
       if drone_id in host.serving_drones:
           self._drone_state[drone_id] = TrainingDroneState.CHARGING_OR_SWAP
       elif drone_id in host.wait_queue:
           self._drone_state[drone_id] = TrainingDroneState.QUEUEING_AT_HOST
       else:
           raise RuntimeError("charging host arrival state inconsistent")
   ```

   - 若本次推进满足
     `t_next >= _fallback_leg[drone_id].arrival_time`，
     则先只把区间 `[t_prev, arrival_time)` 计入 `T_fallback`，随后立刻触发
     "到达备用宿主"点事件，并统一调用
     `_on_arrive_charging_host(drone_id, host, arrival_time)`；此处的 `host` 可以是
     `station` 或 `depot`，两者都必须经过同一套入队/入服判断
   - `mode B` 在 `return_to_station` / `return_to_depot` 到达终点后也必须复用
     同一个 `_on_arrive_charging_host(...)` 入口，不允许存在
     `return_to_depot -> charging_or_swap` 这类绕过排队判断的捷径
   - `fallback_recovery` 到达宿主后必须**直接**结束，不允许再转成
     `return_to_station` / `return_to_depot` 二次飞行；否则会把 `T_fallback`
     与普通返程飞行重复计时
   - 一旦完成上述到达事件，必须立即清空 `_fallback_leg[drone_id]`；
     后续同一时刻及之后的奖励结算不再把该 UAV 计入 `T_fallback`

2. 关键事件

   - 订单送达
   - truck-side `mode A` 背景订单完成
   - `mode C` 的返程汇合点在 `dispatch_action` 中一次性确定（第三阶段）；
     `mode B` 的具体返程宿主由 `delivery_event` 后的确定性逻辑选出
   - reservation 建立与 timeout 检查
   - 卡车到站
   - 无人机到站（含 `return_to_rendezvous`、`return_to_station`、`return_to_depot`、
     `fallback_recovery` 四类飞行段的到达事件）
   - 错过卡车（触发 fallback_recovery）
   - FIFO 入队 / 出队
   - 半空中能源硬失败
- 卡车到达 `launch_candidate_stations` 中的某个站点（触发 `riding_with_truck` 无人机的 PPO 决策点）
   - 无人机在卡车上完成充换电（`charging_on_truck` → `riding_with_truck`）
   - 无人机从卡车起飞（`riding_with_truck` → `flying_to_deliver`）

3. 奖励与惩罚来源（必须与主目标逐项对应）
   - `T_complete`（仅无人机订单）
   - `T_wait`（UAV 被动等待卡车的物理等待时间，episode 级累计）
   - `T_idle`（UAV 显式选择 WAIT 动作的 idle 时间）：
     - `idle` WAIT：在 `step(WAIT_action)` 按精确 `delta_wait` 一次性即时结算
     - `riding_with_truck` WAIT：在 `active_wait(resume_state=riding_with_truck)` 期间，
       按每次真实事件推进区间的实际 `dt` 即时累计
   - `T_queue`（UAV 在 `station / depot` 补能宿主等待服务槽位的累计排队时间）
   - `T_fallback`
   - `T_res_timeout`（reservation timeout 造成的局部损失）
   - `T_overdue`（未送达超时订单的累计超时时长，**每次时间推进时按实际 dt 即时结算**）
   - `N_hard_overdue`（超时超过 `max_overdue_sec` 被强制移除的订单数，一次性固定惩罚）
   - `N_hard_fail`

   **事件驱动环境下的奖励结算统一口径**：

   文档中"每个仿真步结算"在事件驱动环境里的正确映射是"每次时间推进时按实际推进量
   `dt` 结算"，而不是固定步长。`env_adapter` 内部实现一个 `_settle_per_dt_rewards(dt)`
   函数，在以下两个时机调用：

   - `_advance_to_event(t_next)`：推进到下一个事件，`dt = t_next - t_now`
   - `step(WAIT_action)`：
     - `idle` 路径：显式推进 `delta_wait`
     - `riding_with_truck` 路径：不预付未来完整 `delta_wait`，而是进入
       `active_wait` 后由后续 `_advance_to_event(...)` 在真实事件边界上逐段结算

   `_settle_per_dt_rewards(dt)` 统一结算所有"按时间流逝累加"的惩罚项。
   其中 `idle` WAIT 的 `T_idle` 仍走 `step(WAIT_action)` 入口一次性结算路径；
   只有 `riding_with_truck` WAIT 的 `T_idle` 会进入该函数的持续累计路径：

   ```python
   def _settle_per_dt_rewards(self, dt: float) -> float:
       reward = 0.0
       t_prev = self._t_now
       t_next = self._t_now + dt
       # T_overdue：按时间推进区间类实时累加，不等送达事件，不允许拖到 episode 末端统一结算
       for order in self._active_uav_orders():
           overdue_dt = max(0.0, t_next - max(t_prev, order.deadline))
           reward -= self._cfg.lambda_overdue * overdue_dt
       # T_wait / T_queue / T_fallback 以及 riding_with_truck-WAIT 的 T_idle：
       # 按当前训练状态分桶，不需要额外标志
       for drone_id, state in self._drone_state.items():
           if state == TrainingDroneState.WAITING_FOR_TRUCK:
               reward -= self._cfg.lambda_wait * dt
           elif state == TrainingDroneState.QUEUEING_AT_HOST:
               reward -= self._cfg.lambda_queue * dt
           elif state == TrainingDroneState.FALLBACK_RECOVERY:
               reward -= self._cfg.lambda_miss * dt
           elif (
               state == TrainingDroneState.ACTIVE_WAIT
               and self._active_wait_resume[drone_id] == TrainingDroneState.RIDING_WITH_TRUCK
           ):
               reward -= self._cfg.wait_idle_penalty_coef * dt
       return reward
   ```

   `T_idle` 的结算按来源分两条路径：

   ```python
   # idle 路径：step(WAIT_action) 内部
   reward += self._settle_per_dt_rewards(delta_wait)
   reward -= self._cfg.wait_idle_penalty_coef * delta_wait

   # riding_with_truck 路径：不在 step(WAIT_action) 入口预付，
   # 而是在 active_wait(resume_state=riding_with_truck) 期间
   # 由后续 _settle_per_dt_rewards(dt) 按真实事件推进区间持续累计
   ```

   这样做的目的，是保证 `riding_with_truck` WAIT 的 reward 结算边界与环境实际推进边界
   保持一致；若中途先发生其他 UAV 的决策事件，当前 UAV 的 WAIT 承诺仍保持有效，其
   `T_idle` 只对已经真实发生的等待时间收费，不对尚未发生的未来等待时长预收费。

   `T_overdue` 的送达处理：订单被送达时，从 `_active_uav_orders()` 中移除，
   后续推进不再对该订单累加，送达时不叠加额外惩罚（避免双重计入）。
   这里采用的是"按时间推进区间类实时累计"，不是"episode 结束或强制移除时统一结算"；
   因此模型始终有动力去送超时订单——不送则惩罚持续增大，送达则立刻止损。

4. 局部承诺状态（reservation）
   - 当前预留的 `recover_node`
   - reservation 发起时刻与过期时刻
   - `reservation_count(node)` 作为拥挤信号
   - timeout 后自动进入 `fallback_recovery` 并施加软惩罚

5. 系统上下文统计（不进入 PPO 主指标）
   - `mode_a_background_order_count`
   - `mode_a_background_completion_time_sum`
   - `truck_total_mileage`
   - `truck_background_order_completion_events`

动作建议：

- 采用 factorized action space：
  `mode B` 分支为 `dispatch_action=(order_i, mode=B)`，
  `mode C` 分支为 `dispatch_action=(order_i, mode=C, recover_node_j)`；
  其中 `mode B` 不暴露 `station / depot` 选择，具体返程宿主由执行层在送达后确定；
  同时保留一个全局 `WAIT` 动作
- 全局 `WAIT` 为必备动作，不允许被 mask 掉
- `mode A` 不进入 PPO 动作空间
- 全程配合 `action_mask`

产物：

- `env_adapter.py`
- 一个最小可运行 episode
- runtime state / reward / done 说明

验收标准：

- `reset -> step -> ... -> done` 能完整跑通
- 不出现非法动作、空 mask、订单状态错乱（由 inline candidate builder 保证：
  WAIT 始终存在，mask 永远非空；Phase 6 接入后此条继续成立）
- 错过卡车时订单不会被错误回滚
- 硬失败时无人机会被正确移出后续可用集合
- 补能宿主排队和工位释放逻辑可复现（`station / depot` 统一复用同一套入队 / 入服规则）
- reservation timeout 能正确触发 fallback 并记录 `T_res_timeout`
- `delivery_event`（订单送达事件）不会触发 `second_ppo_decision_after_delivery`
  （送达后二次 PPO 决策）；原定 `selected_recover_node`（策略原选回收节点）只允许被
  `post_delivery_revalidation`（送达后执行层复核）接受或拒绝
- `mode C` 在 Phase 5 中仍保持三阶段派送语义，但最终 `action_mask` 不能只停留在
  `recovery_pool` 的粗候选边界；必须对每个 `recover_node_j` 同时执行
  `rendezvous_eta_safe_margin_sec`（汇合安全时间裕量）与 `energy_feasible`（能量可达性）过滤，
  不允许退化成仅按 `truck_eta_map` 做粗时序筛选
- `mode B` 在 Phase 5 中表示“送达后自主返回某个可达充换电宿主”，不依赖卡车；
  `action_mask` 只负责检查“送达后至少存在一个可达 `station / depot`”，
  但不向 PPO 暴露具体宿主选择
- `return_to_station` 与 `return_to_depot` 都是 `mode B` 的合法返程飞行状态；
  一旦到达 `station / depot`，必须统一进入 `_on_arrive_charging_host()`，
  后续状态只能是 `queueing_at_host` 或 `charging_or_swap`；
  `fallback_recovery` 只允许由 `mode C` 的复核失败 / timeout 触发，不与 `mode B` 重叠
- `queueing_at_host` 覆盖 `station + depot` 两类补能宿主；
  `T_queue` 统计范围与 `ChargingHost.arrive()` 的实体语义一致，不再只统计 station
- `home_type == TRUCK` 的无人机在 `reset()` 后初始状态为 `riding_with_truck`，
  `home_type == DEPOT` 的无人机初始状态为 `idle`，两类无人机不混淆
- `T_wait` 与 `T_idle` 在同一 episode 内累计值互不重叠：
  `waiting_for_truck` 状态产生的时间只进 `T_wait`，
  `active_wait` 状态产生的时间只进 `T_idle`
- `riding_with_truck` 状态下 WAIT 不再按“任意下一站”预付未来等待时长；
  只有当卡车到达 `_active_launch_stations` 中的真实触发站时，当前 WAIT 承诺才结束
- `T_overdue` 在订单超过 deadline 后的每次时间推进中立刻开始累加，
  不等送达事件；订单送达后停止累加，送达时不叠加额外惩罚
- `_build_coarse_plan_view(t_now)` 在卡车恰好到站时刻（`arrival_time == t_now`）
  仍能正确保留该站点在 `launch_candidate_stations` 和 `recovery_pool` 中，
  不因严格 `>` 过滤而丢失当前站点
- `_build_coarse_plan_view(t_now)` 返回的 `CoarsePlanView` 能通过 `contracts.py`
  的 `__post_init__` 校验（含 `plan_version`、`issued_at`、`valid_until`、
  `truck_backbone_route` 无重复节点等所有字段约束）
- mode A 订单完成事件由 `env_adapter` 内部事件队列驱动，`truck.current_loc`
  在每次 `_advance_to_event()` 后与仿真时间同步，`candidate_builder` 读取的
  卡车位置与订单完成事件时序一致
- `_reservations` 与 `_reservation_count` 在 `reset()` 后均为空，
  reservation timeout 能在 `_advance_to_event()` 中被正确检测并触发 fallback
- `build_runtime_state_view()` 返回的 `RuntimeStateView` 包含 `t_now`、
  `truck_current_loc`、每架无人机的 `DroneStateView`（含静态能力字段和 `home_type`）、
  每个节点的 `NodeStateView`（含 `position`、`parking_slots`、`swap_time`）、
  `reservation_count`；不包含 `truck_eta_map` 和 `truck_backbone_route`（属于 CoarsePlanView）
- `idle` 状态下 WAIT 的 `delta_wait` 不超过 `max_wait_decision_gap_sec`（60s），
  且 `T_idle` 在 `step(WAIT_action)` 中按精确 `delta_wait` 一次性扣罚，不走持续累计路径
- `riding_with_truck` 状态下 WAIT 的 `T_idle` 由后续真实事件推进区间逐段累计，
  不在 `step(WAIT_action)` 调用点一次性预付完整未来等待时长
- 同一时刻多事件按"先区间成本、后点事件；到达类先合并再做 timeout；
  poisson 新订单在下一次 _build_coarse_plan_view() 才纳入 authorized_orders"的顺序处理
- reservation timeout 的 `energy_feasible` 在每次 `_advance_to_event()` 时重新计算，
  不只在建立时算一次；`T_timeout_cost` 为违规量，不与 `T_fallback` 重复计量
- poisson 模式下，`reset()` 后 `truck_backbone_route` 在整个 `upper_horizon_sec`
  内始终有未来节点（巡站循环已追加），不出现卡车提前回仓后骨架为空的情况
- benchmark / hybrid 模式下，`reset()` 后卡车路线严格等于 Phase 4 产物，
  不追加巡站循环，多次运行结果完全一致；`allow_empty_backbone_route` 自动开启，
  骨架耗尽后构造空骨架退化对象，`policy_mode_mask` 收缩为 `{B}`，为合法退化状态
- poisson 模式下，巡站循环的 ETA 单调递增，不出现时序倒退

### 4.5.0 本轮前置代码同步记录（2026-04-25）

为支撑 Phase 5 中"空骨架退化开关"的文档语义，本轮已对
`backend/training/contracts.py` 做以下同步修改：

1. `CoarsePlanView` 新增 `allow_empty_backbone_route: bool = False`
   - 默认保持严格契约（poisson 模式），`truck_backbone_route` 不能为空
   - benchmark / hybrid 模式下由 `env_adapter.reset()` 自动设为 `True`，无需手动配置
2. `CoarsePlanView.__post_init__()` 增加条件校验分支
   - 当 `allow_empty_backbone_route = False` 时，沿用原有"空骨架即报错"
   - 当 `allow_empty_backbone_route = True` 且 `truck_backbone_route` 为空时，要求：
     - `truck_eta_map = {}`
     - `route_drift_ref = {}`
     - `launch_candidate_stations = ()`
     - `recovery_pool` 对所有订单均为空
     - `policy_mode_mask` 不允许再包含 `C`，应收缩为 `{B}`
3. `EnvSemanticContractMeta` 新增 `allow_empty_backbone_route: bool`
   - 用于把这条环境语义开关纳入 Phase 1 已冻结的 `meta.json` 契约摘要

`allow_empty_backbone_route` 的模式绑定规则（由 `env_adapter.reset()` 负责设置）：

| 订单源模式 | `allow_empty_backbone_route` | 说明 |
|---|---|---|
| `poisson` | `False` | 骨架为空视为契约错误，尽早暴露巡站循环问题 |
| `benchmark` | `True`（自动） | 骨架耗尽为合法退化状态，不追加巡站循环 |
| `hybrid` | `True`（自动） | 同 benchmark |

`planner.allow_empty_backbone_route` 配置项保留在 `rh_alns_cmrappo.yaml` 中，
但其值在运行时由 `env_adapter.reset()` 根据 `order_source_mode` 自动覆盖，
手动配置值仅作为 fallback，不建议依赖。

补充说明：当前仓库中不存在 `backend/training/env_adapter.py` 的可编辑源码文件，
因此本轮关于 `queueing_at_host`、`_on_arrive_charging_host()`、`T_queue` 统计范围、
Phase 5 分层（5a/5b/5c）的修复先冻结在 Phase 5 文档语义层；后续新增或恢复
`env_adapter` 源码时，必须严格按本节口径实现。

### 4.5.1 Phase 5 实现分层

Phase 5 的实现规格较重，为避免一次性面对过多复杂度导致 bug 难以定位，拆分为三个子层。
**三层共用同一套接口定义**（`TrainingDroneState` 枚举、`RuntimeStateView` schema、
`step(action)` 输入格式），接口在 5a 一次定义完毕，5b/5c 只填充实现深度，不改接口。

---

#### Phase 5a：状态机骨架 + smoke test 连通性验证

**本轮已完成实现记录（对应当前代码，而非后续目标态）**

当前仓库已新增：

- `backend/training/env_adapter.py`
- `backend/training/test_env_adapter_phase5a.py`
- `backend/training/__init__.py` 已补充 `env_adapter` 相关导出

当前代码中已经落地并可直接从实现验证的内容如下：

1. **环境入口与固定接口**
   - 已实现 `TrainingEnvAdapter.reset()` / `step(action)` / `is_done()`
   - 已定义固定接口：
     - `TrainingDroneState`
     - `RuntimeStateView`
     - `DroneStateView`
     - `NodeStateView`
     - `DecisionContext`
     - `DispatchAction`
     - `GlobalWaitAction`
     - `EnvStepResult`
   - `DecisionContext` 当前暴露的是 `action_lookup`（候选动作列表），**尚未单独暴露 dense `action_mask` 张量**

2. **Phase 2 / 3 / 4 产物对接**
   - `reset()` 已对接 `scene_loader.load_default_scene()`、`order_source_adapter.build_order_source()`
   - 已复用 `configure_order_manager_for_source()` 配置 `OrderManager`
   - 已读取 Phase 4 的 `truck_execution_route.json`，转成内部 `_planned_route_stops`
   - 已从 Phase 4 固定节点停靠记录构造 `_full_backbone_cache`
   - `poisson` 模式下已在 `reset()` 内部按当前文档参数追加巡站循环；
     `benchmark / hybrid` 不追加巡站循环，并自动允许空骨架退化

3. **状态机骨架**
   - 已实现 14 个训练态枚举值：
     - `idle`
     - `flying_to_deliver`
     - `delivered`
     - `return_to_rendezvous`
     - `waiting_for_truck`
     - `return_to_station`
     - `return_to_depot`
     - `queueing_at_host`
     - `charging_or_swap`
     - `active_wait`
     - `fallback_recovery`
     - `charging_on_truck`
     - `riding_with_truck`
     - `airborne_energy_failure`
   - 其中 `home_type == TRUCK` 的 UAV 在 `reset()` 后初始化为 `riding_with_truck`
   - `home_type == DEPOT` 的 UAV 在 `reset()` 后初始化为 `idle`

4. **事件推进**
   - 已实现 `_advance_to_event(t_next)`，当前会处理：
     - 真实空中硬失败事件：若当前飞行段总能耗大于 UAV 剩余电量，则按飞行比例计算
       `airborne_energy_failure` 发生时刻，并在到达事件之前截断该飞行段
     - UAV 送达事件
     - mode A 背景订单完成事件（来自 `truck_execution_route` 的 customer stop）
     - 卡车到站事件
     - `return_to_rendezvous / return_to_station / return_to_depot / fallback_recovery` 到达事件
     - 固定节点充换电完成事件（`ChargingHost.tick_update()`）
     - `charging_on_truck -> riding_with_truck` 的车载充换电完成事件
     - `OrderManager.tick()` 触发的泊松/回放新订单注入

5. **统一 charging host 入口**
   - 已实现 `_on_arrive_charging_host(drone_id, host, t_now)`
   - `station` 与 `depot` 到达均复用该入口
   - `fallback_recovery` 到达后也直接走该入口
   - 当前状态流转以 `host.arrive()` 的真实结果为准：
     - 在 `serving_drones` 中则进入 `charging_or_swap`
     - 在 `wait_queue` 中则进入 `queueing_at_host`

6. **coarse plan 与 5a 候选动作**
   - 已实现 `_build_coarse_plan_view(t_now)`
   - 骨架过滤当前使用 `departure_time > t_now`
   - 已按代码当前实现填充：
     - `truck_backbone_route`
     - `truck_eta_map`
     - `route_drift_ref`
     - `authorized_orders`
     - `planner_mode_cap`
     - `policy_mode_mask`
     - `recovery_pool`
     - `launch_candidate_stations`
     - `node_charge_load_budget`
   - 已实现 `_build_action_lookup(...)`
   - 当前 5a 候选动作过滤包括：
     - 订单必须仍在 `pending_orders`
     - 订单重量不能超过当前 UAV `payload_capacity`
     - 送达腿必须能飞到
     - mode B 还要求存在至少一个可达返程宿主
     - mode C 当前仅要求 `recovery_pool` 中存在 coarse 候选节点，不做 5b 的时序/能量精细过滤
   - `WAIT` 当前始终保留

7. **送达后执行层**
   - mode B 当前**尚未**实现 5b 文档里的完整 `score = fly_time + predicted_queue_time + service_time`
   - 当前代码里，mode B 返程宿主选择临时复用 `deterministic_fallback` 规则：
     - `depot` 优先
     - 若 `depot` 不可达，则选最近可达 `station`
   - mode C 已实现 5a 最小兜底：
     - 若原 `selected_recover_node` 仍在当前 `truck_backbone_route` 中，则进入 `return_to_rendezvous`
     - 否则直接进入 `fallback_recovery`

8. **fallback 内部账本**
   - 已实现 `_fallback_leg: dict[drone_id, FallbackLeg]`
   - 进入 `fallback_recovery` 时写入：
     - `host_node_id`
     - `host_node_type`
     - `arrival_time`
   - 到达 fallback 宿主后会立即清空对应账本

9. **奖励与终止条件**
   - 当前代码里只结算：
     - `+ R_delivery_bonus`
     - `- hard_failure_penalty_sec`
   - 当前 `is_done()` 已实现三个终止条件：
     - `t_now >= upper_horizon_sec`
     - `pending_orders / assigned_orders / background_mode_a_pending` 全空
     - 所有 UAV 都处于 `airborne_energy_failure`

10. **当前 smoke test**
   - 已新增 `backend/training/test_env_adapter_phase5a.py`
   - 当前测试覆盖：
     - default scene 下 `reset -> step -> done` 连通性
     - `_fallback_leg` 在到达宿主后清空
     - mode C 动作接口形状固定为“必须显式携带 `recover_node_id`”
     - 真实空中电量耗尽会在飞行到达前触发 `airborne_energy_failure`
     - `_on_arrive_charging_host()` 对 `station / depot` 统一复用同一套排队逻辑
     - `_build_coarse_plan_view(t_now)` 在 `arrival_time == t_now` 时仍保留当前站点
     - `poisson` 默认场景下，`reset()` 会在 Phase 4 原路线后追加巡站循环

**本轮未完成 / 仍与目标态存在差异的部分**

- 当前代码没有单独输出 dense `action_mask` 张量，只有 `action_lookup`
- 当前代码没有实现 5b 的 mode C 精细过滤
- 当前代码没有实现 reservation 状态机
- 当前代码没有实现 `T_overdue / T_wait / T_queue / T_fallback / T_idle`
- 当前代码没有实现 5b 的 mode B 完整返程评分规则，只保留了 5a 可运行替代口径


**用途**：验证环境状态机正确性，确认 `reset → step → done` 能跑通。
**禁止用途**：**不允许用于正式 PPO 训练**，不允许用于 benchmark 对比。
原因：5a 的奖励只有送达奖励和硬失败惩罚，缺少 `T_overdue`，策略会对超时订单和准时订单
一视同仁，学出错误的优先级偏好，该偏好在 5c 加入完整奖励后需要被覆盖，代价很高。

**实现范围**：

- 完整的 `TrainingDroneState` 枚举（全部 14 个状态，一次定义，5b/5c 不增删）
- 完整的状态转移图（所有路径，含 mode B/C/WAIT、fallback、charging_on_truck、riding_with_truck）
- `_on_arrive_charging_host()` 统一入口（station 和 depot 统一走此处，排队判断一次写对）
- `_fallback_leg` 内部账本（进入 `fallback_recovery` 时写入，到达时清空）——**5a 必须实现**，
  因为 `fallback_recovery` 的退出依赖它，缺失会导致状态机卡死
- 事件队列推进（`_advance_to_event`），含卡车到站、UAV 到站、mode A 完成
- `reset()` 读取 Phase 4 产物 + 巡站循环（poisson 模式触发）
- `is_done()` 三个终止条件
- `build_runtime_state_view()` 完整 schema（一次定义，5b/5c 不改）
- 同一时刻多事件处理顺序（第 7 节定义的 6 步顺序）

**奖励（仅两项）**：
```python
+R_delivery_bonus         # 送达时
- hard_failure_penalty_sec  # 硬失败时
```

**action_mask（最粗过滤）**：
- mode B：`deliver_leg_feasible`（送达可达）且至少一个宿主可达
- mode C：`recovery_pool` 非空（不做时序/电量精细过滤）
- WAIT：始终保留

**最小送达后失败兜底（5a 必须实现，不可省略）**：

5a 不实现完整的 `post_delivery_revalidation`，但必须保留以下最小兜底逻辑，
否则策略可能选出一个送达后根本无法执行的 mode C，导致状态机卡死：

```python
# delivered 状态下的最小兜底（在 delivery_event 处理时执行）
if mode == C:
    if recover_node_j not in current_truck_backbone_route:
        # 卡车已过该节点，直接进 fallback_recovery
        → 写入 _fallback_leg，选最近可达宿主，进入 fallback_recovery
    else:
        → return_to_rendezvous（不做完整三条件复核）
```

此处 `current_truck_backbone_route` 取 `_build_coarse_plan_view(t_now)` 的当前过滤结果，
不需要额外维护。

**不实现**：reservation 状态机、T_overdue、T_wait、T_queue、T_fallback、T_idle、
完整 `post_delivery_revalidation` 三条件复核。

**验收**：
- episode 能跑完，状态不乱
- `_on_arrive_charging_host` 覆盖所有到达路径，无 depot 绕过排队的捷径
- `fallback_recovery` 能正确触发并退出，`_fallback_leg` 账本在到达后清空
- 最小兜底逻辑能拦截"卡车已过的 recover_node"，不出现状态机卡死

---

#### Phase 5b：完整 action_mask + 送达后执行层复核

**用途**：action_mask 语义正确，mode C 候选合法，mode B 返程宿主确定性选择。
**禁止用途**：**不允许用于正式 PPO 训练**，奖励信号仍不完整。

**在 5a 基础上增加**：

- mode C 精细过滤（注意量纲：两边均为 timestamp）：
  - `t_arrive_uav(deliver → node) + rendezvous_eta_safe_margin_sec <= t_arrive_truck(node)`
  - `E_need(deliver → node, payload=0) + drone.safe_margin_j <= E_rem_after_delivery`
- 完整 `post_delivery_revalidation` 三条件复核（替换 5a 的最小兜底）：
  - `energy_feasible`
  - `rendezvous_time_feasible`
  - `node_still_valid`
  - 复核失败 → `fallback_recovery`（复用 `_fallback_leg` 账本）
- mode B 确定性返程宿主选择：
  `score = fly_time + predicted_queue_time + service_time`，选最小者
- `_select_return_host_mode_b()` 实现

**奖励（在 5a 基础上增加）**：
```python
-lambda_miss * dt   # fallback_recovery 状态下持续扣（T_fallback）
```

**不实现**：reservation 状态机、T_overdue、T_wait、T_queue、T_idle。

**验收**：
- mode C 候选集合法，精细过滤后不出现量纲混用
- `post_delivery_revalidation` 能正确拦截复核失败，fallback 路径可复现
- mode B 返程宿主选择结果可复现，score 计算不缓存旧队列快照

**本轮代码同步记录（2026-04-26）**

本轮已在 `backend/training/env_adapter.py` 与
`backend/training/test_env_adapter_phase5b.py` 完成 Phase 5b 的代码同步，
供后续 Phase 5c / Phase 6 / Phase 7 继续接续。当前实际落地情况如下：

1. **mode C 候选的运行时精细过滤已接入**
   - `_build_action_lookup()` 不再把 `recovery_pool` 原样平铺成 mode C 动作
   - 新增 `_iter_feasible_mode_c_recovery_nodes()` /
     `_is_mode_c_recovery_feasible()`，按以下统一口径过滤：
     - `t_arrive_truck(node) > t_deliver`
     - `t_arrive_uav(deliver -> node) + rendezvous_eta_safe_margin_sec <= t_arrive_truck(node)`
     - `E_need(deliver -> node, payload=0) + drone.safe_margin_j <= E_rem_after_delivery`
   - 因此，Phase 5a 中“只要 coarse recovery_pool 非空就放行 mode C”的临时口径已移除

2. **mode B 确定性返程宿主选择已接入**
   - 已新增 `_select_return_host_mode_b()`
   - 评分规则与文档一致：
     `score = fly_time + predicted_queue_time + service_time`
   - `predicted_queue_time` 直接复用 `ChargingHost.estimate_wait_time(current_time)`，
     每次即时读取宿主当前真值，不缓存旧队列快照
   - `_has_mode_b_return_host()` 也已切换为复用这套 5b 规则，而不是 5a 的 depot-first 临时口径

3. **送达后执行层复核（post_delivery_revalidation）已接入**
   - mode C 送达后不再沿用 5a 的“只检查 recover_node 是否还在未来骨架里”的最小兜底
   - 已新增 `_revalidate_mode_c_recover_node()`，统一复核以下三项：
     - `energy_feasible`
     - `rendezvous_time_feasible`
     - `node_still_valid`
   - 任一失败即不重新请求 PPO，而是直接进入 `fallback_recovery`

4. **`T_fallback` 的 5b 级 reward 已接入**
   - 已新增 `_settle_phase5b_per_dt_rewards(t_prev, t_next)`
   - 当前只结算 `fallback_recovery` 状态下的
     `-lambda_miss * dt`
   - 结算时显式截断到 `_fallback_leg[drone_id].arrival_time`，
     并兼容“中途空中耗尽电量导致飞行提前中断”的情况

5. **回归与新增测试已补齐**
   - 原 `backend.training.test_env_adapter_phase5a` 继续通过
   - 新增 `backend.training.test_env_adapter_phase5b`，覆盖：
     - mode C action mask 的时间/能量过滤
     - mode B 返程宿主评分选择
     - post-delivery revalidation 失败后进入 fallback
     - `T_fallback` 的持续扣罚
   - 当前本地验证命令：
     `python -m unittest backend.training.test_env_adapter_phase5a backend.training.test_env_adapter_phase5b`

**与 Phase 1-5 规约的对齐检查**

以下条目表示：从 Phase 1 到 Phase 5 文档中，凡是“应在 Phase 5b 落地”的规约，
当前代码是否已经对齐，以及还有哪些边界需要后续 phase 接续。

1. **`DispatchAction` / `WAIT_ACTION` / `DecisionContext` / `RuntimeStateView` 接口形状**
   - **状态**：已对齐
   - 说明：5a 已冻结接口，本轮未改外部 schema；5b 仅填充实现深度，不改调用方契约

2. **mode C 候选合法性交由运行时 action mask 判定**
   - **状态**：已对齐
   - 说明：现在 mode C 只在“送达后仍能赶上卡车且送达后电量可达”的条件下才进入 `action_lookup`

3. **mode B 不在起飞前绑定返程宿主，送达后由执行层确定性规则选择**
   - **状态**：已对齐
   - 说明：PPO 仍只输出 `order + mode=B`；返程宿主在送达后由 `_select_return_host_mode_b()` 选择

4. **mode C 送达后不允许二次 PPO 决策，只允许执行层复核与兜底**
   - **状态**：已对齐
   - 说明：当前复核失败后直接进入 `fallback_recovery`，没有二次 action mask 或第二次 actor 调用

5. **fallback 退出依赖 `_fallback_leg` 账本，到达后立即清空**
   - **状态**：已对齐
   - 说明：`fallback_recovery` 到达后仍统一走 `_on_arrive_charging_host()`，并在到达分支清空 `_fallback_leg`

6. **mode B 的 `predicted_queue_time` 只作用于回收后补能服务阶段，不用于 mode C 等待回收合法性**
   - **状态**：已对齐
   - 说明：queue 只参与 mode B 返程宿主评分；mode C 合法性仍只看时序与能量，不把站点容量当作非法条件

7. **Phase 5b 允许 reward 只补 `T_fallback`，其余奖励项延后**
   - **状态**：已对齐
   - 说明：当前 reward 由 `delivery_bonus`、`hard_failure`、`fallback` 三部分组成，未越界提前实现 5c 项

8. **“完整 action_mask”当前的对外暴露方式**
   - **状态**：语义已对齐，表示形式保持 5a 契约
   - 说明：当前 env 侧已经实现“合法动作集合”的完整语义过滤，但对外仍只通过
     `DecisionContext.action_lookup` 暴露，不单独输出 dense action_mask 张量。
     这是刻意保持 5a 冻结接口不变；Phase 6 起由 `candidate_builder.build(...)`
     在 env 外部消费侧基于
     `DecisionContext(runtime_state, coarse_plan, deciding_drone_id, trigger_type, trigger_station_id)`
     构造 `CandidateOutput`（候选输出结构）与 dense `action_mask`（稠密动作掩码），
     而不是回改 env API。`DecisionContext.action_lookup` 在 Phase 6 后保留为兼容字段；
     若 `env_adapter.step(...)`（环境执行）仍基于它做合法性校验，则其语义必须与
     `CandidateOutput.resolved_action_lookup`（已解析动作映射）保持一致

**当前 Phase 5b 与全文规约之间仍需明确标注的未实现项**

以下不是本轮遗漏，而是文档本身就要求留到后续 phase 的内容；为了避免后续实现时误判，
在这里显式标注为“未在 5b 落地”：

1. **reservation 状态机**
   - `ReservationState`、`_reservations`、`_reservation_count`
   - `build_runtime_state_view()` 中 `reservation` / `reservation_count` 仍未填真实值
   - **后续归属**：Phase 5c

2. **reservation timeout 与 `T_timeout_cost`**
   - 当前 `_advance_to_event()` 仍未扫描 timeout 条件
   - 当前也未实现 `tau_res = alpha * fly_time + gamma * q_est`
   - 因此本文档中“飞行途中持续复核 / timeout 后直接 fallback”的完整 CCT 映射，
     目前只完成了“送达时复核”这一半，飞行途中 timeout 检查仍待补
   - **后续归属**：Phase 5c

3. **完整 `_settle_per_dt_rewards()`**
   - 当前只实现了 `T_fallback`
   - `T_overdue`、`T_wait`、`T_queue` 尚未接入
   - **后续归属**：Phase 5c

4. **`T_idle` 的 WAIT 即时惩罚**
   - 当前 `WAIT_ACTION` 仍只作为 `active_wait` 的占位语义，不结算 `delta_wait`
   - “在 `step(WAIT_action)` 调用点按精确 `delta_wait` 一次性扣罚”尚未实现
   - **后续归属**：Phase 5c

5. **无订单时不触发 PPO 决策**
   - 当前 env 仍保留 `WAIT_ACTION` 作为兜底动作，未显式落实
     “`authorized_orders` 为空时直接推进到下一事件，不产生 `T_idle`”
   - 由于 `T_idle` 还未接入，这一条暂不影响 5b 正确性，但需要在 5c 一并收口
   - **后续归属**：Phase 5c

6. **reservation 相关观测拥挤信号**
   - 文档中规划的 `reservation_count(node)` 作为拥挤信号，目前仍未对外暴露真实值
   - 当前 mode B 的拥挤感知只来自真实 host queue，而不是 reservation 视角
   - **后续归属**：Phase 5c

7. **benchmark / stochastic 验证口径**
   - 5b 仍不允许正式训练，也不允许作为最终 benchmark 指标基线
   - 因为主目标中的 `T_res_timeout / T_overdue / T_wait / T_queue / T_idle` 仍未齐备
   - **后续归属**：Phase 5c 完整奖励后，Phase 7 / 8 再正式接入验证

**对后续 phase 的实现提醒**

为避免 Phase 5c / Phase 6 重复返工，基于当前 5b 代码状态，后续实现时请保持以下约束不变：

1. `DispatchAction` / `GlobalWaitAction` / `DecisionContext` / `RuntimeStateView`
   的字段形状不要再改，只补真实值
2. mode C 的合法性判定继续沿用“时序 + 能量 + 固定交接节点”三件事，
   不要把 `station/depot` 的补能容量回灌成“等待被回收动作”的非法条件
3. reservation timeout 接入时，不要回到“送达后二次 PPO 决策”；必须复用当前
   `dispatch_commit -> post_delivery_revalidation / timeout -> fallback_recovery`
   这条执行层链路
4. 5c 接 `T_wait / T_queue / T_idle / T_overdue` 时，要在当前 `fallback` 结算路径上继续扩展，
   不要重写 `fallback_recovery` 的账本语义
5. 若 Phase 6/7 需要 dense action mask，请在 policy/candidate builder 消费
   `DecisionContext` 构造 `CandidateOutput` 生成，不要回改 env_adapter 的对外返回格式；
   同时保留 `DecisionContext.action_lookup` 作为兼容字段时，需保证其与
   `CandidateOutput.resolved_action_lookup` 的可执行动作集合一致

---

#### Phase 5c：完整奖励信号（第一个允许正式训练的子层）

**用途**：所有奖励项接入，训练信号完整。**5c 是第一个允许正式 PPO 训练和 benchmark 对比的子层。**
更准确地说，5c 的职责是为 Phase 7 / Phase 8 提供完整环境前置条件；正式训练与 benchmark 对比仍分别属于后续阶段。

**本轮代码同步记录（2026-04-26）**

当前 `backend/training/env_adapter.py` 文件头已明确标注“本文件当前实现到 Phase 5c”，
且对应测试 `backend/training/test_env_adapter_phase5c.py` 已补齐。结合当前代码，5c
实际已完成的内容如下：

1. **reservation 状态机与观测暴露已落地**
   - 已实现 `ReservationState` / `ReservationStateView`
   - `TrainingEnvAdapter` 内部已维护：
     - `_reservations`
     - `_reservation_count`
   - `build_runtime_state_view()` 已把每架 UAV 的 `reservation` 与全局
     `reservation_count` 回填到 `RuntimeStateView`
   - mode C 派送动作在 `_apply_dispatch_action()` 中已通过 `_acquire_reservation()`
     创建 reservation，而不是只保留 5b 的 dispatch 承诺

2. **reservation timeout 检测与执行层 fallback 已落地**
   - `_advance_to_event()` 已接入 `_process_reservation_timeouts(t_next)`
   - timeout 判定由 `_reservation_timeout_cost()` 统一计算，当前覆盖三类触发：
     - `t_now >= expires_at`
     - `t_arrive_uav(current_loc -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j)`
     - 原 reservation 目标不再满足当前时刻的时序约束时产生违规量
   - `tau_res` 已按代码落地为：
     `alpha * fly_time(deliver -> r_j) + gamma * estimate_wait_time(r_j)`
   - timeout 后会立即：
     - 释放 reservation 与节点计数
     - 清空 `dispatch_commit.selected_recover_node`
     - 对 `delivered / return_to_rendezvous / waiting_for_truck` 状态 UAV 直接切入
       `fallback_recovery`
     - 若连 fallback 都不可达，则立即记硬失败

3. **5c 完整持续型奖励结算已落地**
   - `_settle_per_dt_rewards(t_prev, t_next)` 已取代 5b 只结算 `fallback` 的临时版本
   - 当前已按状态分桶结算：
     - `T_overdue` -> `-lambda_overdue * overdue_dt_total`
     - `T_wait` -> `-lambda_wait * wait_dt_total`
     - `T_queue` -> `-lambda_queue * queue_dt_total`
     - `T_fallback` -> `-lambda_miss * fallback_dt_total`
   - 其中 `T_overdue` 已按区间重叠长度即时累计；订单一旦送达或移出活动集合即停止累计，
     不在送达时追加额外一次性惩罚
   - 当前 reward breakdown 也会通过 `EnvStepResult.info["reward_breakdown"]` 对外暴露

4. **WAIT 精确 idle 惩罚与“无订单不触发决策”已落地**
   - `step(WAIT_ACTION)` 已先调用 `_compute_wait_delta()` 计算本次精确推进量，再即时扣除：
     `-wait_idle_penalty_coef * delta_wait`
   - `T_idle` 没有混入 `_settle_per_dt_rewards()`，而是在 WAIT 调用点一次性结算，
     与 `waiting_for_truck` / `queueing_at_host` 的持续惩罚分离
   - `_enqueue_decision()` 已显式检查 `_has_authorized_orders_now()`；
     没有可派送订单时不会入队 PPO 决策，也不会凭空产生 `T_idle`

5. **hard overdue 强制移除与 assigned 订单清场已落地**
   - `_advance_to_event()` 已接入 `_apply_hard_overdue_penalty(t_next)`
   - `pending_orders` 中超出 `max_overdue_sec` 的 UAV 主订单会被强制转为 `TaskStatus.TIMEOUT`
   - `assigned_orders` 中达到 hard overdue 的订单会：
     - 从订单池移除
     - 释放 UAV 当前携带订单
     - 清理飞行段 / reservation / dispatch_commit
     - UAV 切到 `delivered` 后复用现有 fallback 清场链路

6. **Phase 5c 回归测试已补齐并通过**
   - 已新增 `backend/training/test_env_adapter_phase5c.py`
   - 当前测试覆盖：
     - runtime_state 正确暴露 `reservation` 与 `reservation_count`
     - reservation timeout 后进入 `fallback_recovery`
     - hard overdue 订单被移除并记罚
     - WAIT 动作按精确 `delta_wait` 结算 `T_idle`
     - 无 authorized orders 时不创建决策点
     - `T_wait` 与 `T_queue` 分桶结算互不混淆
   - 当前本地验证结果：
     `python -m unittest backend.training.test_env_adapter_phase5a backend.training.test_env_adapter_phase5b backend.training.test_env_adapter_phase5c`
     已通过，共 `17` 个测试

**Phase 5 完成总结（结合当前 `env_adapter.py`）**

1. **Phase 5a 完成了“能跑”的训练环境骨架**
   - 打通了 `reset -> step -> done`
   - 接上了 Phase 2/3/4 的场景、订单与卡车骨架产物
   - 冻结了 `RuntimeStateView / DecisionContext / DispatchAction / WAIT_ACTION` 等对外接口
   - 把训练态状态机、事件推进、充换电宿主统一入口、fallback 账本这些底层骨架先搭起来

2. **Phase 5b 完成了“动作语义正确”的执行层闭环**
   - mode C 不再只看 coarse 候选，而是接入送达后时序/能量合法性过滤
   - mode B 改为按 `fly_time + predicted_queue_time + service_time` 确定性选择返程宿主
   - mode C 送达后增加三条件复核，失败直接走 fallback，不回退到二次 PPO 决策
   - `T_fallback` 开始进入 reward，使“错过卡车”的代价第一次具备持续型训练信号

3. **Phase 5c 完成了“可正式训练”的奖励与约束收口**
   - reservation / timeout / fallback / hard overdue / WAIT idle penalty 已全部接入
   - `T_overdue / T_wait / T_queue / T_fallback / T_idle` 的主要 reward 语义已闭环
   - 无订单不触发决策、reward 分桶不重叠、runtime_state 暴露拥挤相关观测这几项也已收口
   - 结合当前代码状态，Phase 5 的核心目标已经完成；后续 Phase 6 更适合围绕
     `candidate_builder` / `planner_bridge` 做职责拆分，而不是继续回补 env 语义缺口

## 4.6 Phase 6：实现候选动作与上层规划骨架

目标：

- 先把训练真正依赖的动作空间稳定下来
- 实现固定 RH-ALNS 骨架的简化版本，输出 `CoarsePlanView`

建议新增文件：

- `backend/training/candidate_builder.py`
- `backend/training/planner_bridge.py`

职责拆分：

1. `candidate_builder.py`
   - 只读 `env_adapter` 当前运行时状态与 `CoarsePlanView`
   - 构建输入固定为：
     `runtime_state + coarse_plan + deciding_drone_id（当前决策无人机） + trigger_type（触发类型） + trigger_station_id（触发站点）`
   - 其中 `deciding_drone_id` / `trigger_type` / `trigger_station_id` 属于**本次 build 的局部上下文**，
     不属于 `RuntimeStateView`（运行时真值快照）字段
   - `CandidateOutput`（候选输出结构）是 **env 外部消费侧** 的局部中间对象：
     由训练循环 / baseline / 验证脚本在拿到 `DecisionContext`（决策上下文）后调用
     `candidate_builder.build(...)` 构造，生命周期限定在
     “拿到 `DecisionContext` -> 选择动作 -> 调用 `env_adapter.step(env_action)`”这一段。
     它**不挂在** `EnvStepResult`（环境步结果）或 `DecisionContext` 上，也**不由**
     `env_adapter`（训练环境适配器）持有
   - `candidate_builder.build(...)` 继续保持**无状态构建器**语义：
     `plan_version_delta`（规划版本增量）所需的历史值不在 builder 内部缓存，
     而由外部调用方显式以 `last_seen_plan_version` 参数传入
   - 输出 `CandidateOutput`（候选输出结构），其中包含：
     `candidate_features + factorized_action_schema + action_mask + resolved_action_lookup`
   - **不负责**把特征张量化或注入 `history_tokens`；这两件事属于 Phase 7 的 `ObservationTensorizer` 职责
   - 构建候选订单（在 `authorized_orders` 范围内）
   - 构建候选回收点，按触发类型分两条路径：
     - `idle`（地面空闲触发）时：在 `coarse_plan.recovery_pool[order]`（粗规划回收候选池）范围内构建候选
     - `riding_with_truck`（随车搭载触发）时：**不直接使用** `coarse_plan.recovery_pool[order]`，
       而是基于 `trigger_station_id`（触发站点）、`t_now`（当前时刻）和卡车未来 ETA（预计到达时刻）
       实时计算 `runtime_recovery_pool(order_i)`（运行时回收候选池），计算规则见下方
   - 构建派送分支的 factorized action mask（分解式动作掩码，`order -> mode(B/C) -> recover_node` 三阶段）
   - 同时保留一个全局 `WAIT` 动作，不允许在 `max_candidate_actions` 截断时被裁掉
   - **新增**：支持 `riding_with_truck` 触发的候选集构建：
     - 输入：当前站点 `S_k`（即 `trigger_station_id`）、当前时刻 `t_now`、卡车剩余路线与 ETA
     - 对每个候选订单实时计算 `t_deliver = t_now + fly_time(S_k -> order_i.location)`
     - 基于实时 `t_deliver` 计算 `runtime_recovery_pool(order_i)`（运行时回收候选池），
       该池是 `riding_with_truck` 路径专用的决策时局部候选集，不等同于
       `coarse_plan.recovery_pool`（粗规划回收候选池）
     - 构建与 `idle` 状态完全一致的 factorized action mask

   术语约定：

   - 下文中的 `action_mask`（动作掩码）是
     `root_branch_mask + order_mask + mode_mask + recovery_mask` 的统称；
     `CandidateOutput` 不再额外引入一个重复的嵌套 `action_mask` 字段
   - `DecisionContext.action_lookup`（决策上下文中的候选动作列表）在 Phase 6 后保留为兼容字段，
     训练主路径不再直接消费；训练 / baseline / 验证主路径统一消费
     `CandidateOutput.resolved_action_lookup`。若 `env_adapter.step(...)` 仍以内建
     `action_lookup` 做合法性校验，则两者的可执行动作集合必须保持一致
   - `CandidateFeatures.uav_self.plan_version_delta`（无人机自身特征中的规划版本增量）
     由 `candidate_builder.build(...)` 基于
     `coarse_plan.plan_version - last_seen_plan_version` 计算并填充；
     `ObservationTensorizer`（观测张量化器）只负责编码该字段，不负责推导其数值

   **`CandidateOutput`（候选输出结构）推荐定义：**

   ```python
   @dataclass(frozen=True)
   class CandidateOutput:
       # 固定 slot、模型无关、无 history 的特征包；slot 顺序与 action mask 严格对齐
       candidate_features: CandidateFeatures

       # 顶层分支：WAIT / DISPATCH
       root_branch_mask: tuple[bool, bool]   # 顺序固定为 [WAIT, DISPATCH]
       has_wait_action: bool                  # 当前阶段固定应为 True

       # dispatch 分支固定 shape；长度可训练前冻结，batch 友好
       order_mask: list[bool]                # 长度固定为 max_candidate_orders = 32
       mode_mask: list[list[bool]]           # shape [32, 2]，第二维顺序固定为 [B, C]
       recovery_mask: list[list[bool]]       # shape [32, 4]，仅 mode=C 时消费

       # 结构化索引空间定义，供 model.py / rollout_buffer.py 直接消费
       factorized_action_schema: FactorizedActionSchema

       # 已解析动作映射，供 env_adapter.step(...) 或 baseline 直接反查动作对象
       resolved_action_lookup: ResolvedActionLookup
   ```

   **`CandidateFeatures`（候选特征包）推荐定义：**

   ```python
   @dataclass(frozen=True)
   class CandidateFeatures:
       # slot 0：当前决策 UAV 自身特征（单条，不 padding）
       uav_self: UavSelfFeatures

       # slot 1..32：候选订单特征，固定 32 个 slot，不足时 padding，is_valid=False
       # 顺序与 order_mask 严格对齐：order_features[k] 对应 order_mask[k]
       order_features: tuple[OrderFeatures, ...]   # len == max_candidate_orders

       # slot [k][j]：第 k 个订单的第 j 个回收候选特征，固定 [32, 4] slot
       # 顺序与 recovery_mask 严格对齐：recovery_features[k][j] 对应 recovery_mask[k][j]
       recovery_features: tuple[tuple[RecoveryFeatures, ...], ...]  # shape [32, 4]

       # 基础设施摘要（truck 骨架 + station/depot 列表，不 padding）
       infra_features: InfraFeatures
   ```

   `CandidateFeatures` 的设计约束：

   - **模型无关**：所有字段均为 Python 原生类型（`float`、`bool`、`str`），不引入 PyTorch 张量
   - **无 history**：不包含 `history_tokens`；历史状态由 Phase 7 的 `ObservationTensorizer` 在 rollout 侧注入
   - **slot 对齐是核心约束**：`order_features[k]` 与 `order_mask[k]` 必须指向同一个候选订单；`recovery_features[k][j]` 与 `recovery_mask[k][j]` 必须指向同一个回收候选。这个对齐关系由 `candidate_builder` 一次性确定，Phase 7 的 `ObservationTensorizer` 只能按此顺序填充张量，不允许自行重排
   - **padding 语义**：不足 32 个订单时，多余 slot 的 `OrderFeatures.is_valid = False`，`order_mask[k] = False`；不足 4 个回收候选时同理
   - **结构真值与动作合法性分离**：`OrderFeatures.is_valid` / `RecoveryFeatures.is_valid`
     只表达“该 slot 是否承载真实对象”，供 Phase 7 的
     `ObservationTensorizer` 构造结构性 padding mask；`order_mask` /
     `mode_mask` / `recovery_mask` 则只表达“该动作当前是否允许 actor 采样”。
     两者在当前实现中可能经常重合，但契约层语义不得合并

   `OrderFeatures` 中必须包含（由 `candidate_builder.build()` 时计算）：

   - 基础订单字段：`order_id`、`weight`、`deadline`、`remaining_time`、`distance_to_order`、`order_pre_score`、`priority_band`、`is_valid`
   - mode B 摘要特征（**在 `candidate_builder.build()` 时基于当前 `RuntimeStateView.node_states` 计算，不允许跨 build 缓存**）：
     - `best_mode_b_return_score`：估算的最优返程总代价（`fly_time_est + predicted_queue_time_est + service_time`）
     - `best_mode_b_host_type`：最优返程宿主类型（`"station"` / `"depot"`）
     - `predicted_queue_time_est` 的计算口径：基于 `NodeStateView.queue_length`、`available_slots`、`swap_time` 推导，不调用 `ChargingHost.estimate_wait_time()`（后者是 `env_adapter` 内部对象）
       - Phase 6 首版冻结公式：

         ```text
         predicted_queue_time_est(node_j, t_arrive_ij)
           = 0, if available_slots(node_j) > 0
           = (queue_length(node_j) + 1) * swap_time(node_j), otherwise
         ```

       - 物理含义：
         - 若当前存在空闲服务槽位（`available_slots > 0`），则到达后可立即开始充换电，预计排队时间记为 `0`
         - 若当前无空闲服务槽位（`available_slots = 0`），则保守地假设：
           1. 至少还有 **1 个**正在服务中的无人机先占用一个 `swap_time`
           2. 当前等待队列中的 `queue_length` 个无人机各再占用一个 `swap_time`
           因此估计等待时间为 `(queue_length + 1) * swap_time`
       - 适用边界：
         - 这是**观测侧粗近似特征**，只用于 `best_mode_b_return_score`（mode B 最优返程总代价估计）
           与相关排序信号的稳定计算
         - 它**不要求**与执行层 `ChargingHost.estimate_wait_time(...)`（运行时真实宿主队列估计）
           完全一致；后者仍是 mode B 实际返程宿主选择的真实口径
         - 尽管接口形参写作 `predicted_queue_time_est(node_j, t_arrive_ij)`，
           Phase 6 首版**不对** `t_arrive_ij`（无人机预计到达该宿主时刻）期间的队列演化做未来预测；
           该量只基于 `candidate_builder.build()` 时刻的当前节点快照估算
         - 若后续需要更贴近执行层真实等待时间，应通过扩充 `NodeStateView` 字段
           明确暴露服务中槽位释放信息，而不是在不同实现里各自发明新近似公式

   接口语义固定为：

   - `root_branch_mask`（顶层分支掩码）长度固定为 `2`，顺序固定为 `[WAIT, DISPATCH]`
   - `WAIT`（全局等待动作）**不进入** `order_mask` / `mode_mask` / `recovery_mask`
   - `order_mask`（订单阶段掩码）长度固定为 `max_candidate_orders = 32`
   - `mode_mask`（模式阶段掩码）固定 shape 为 `[32, 2]`，第二维顺序固定为 `[B, C]`
   - `recovery_mask`（回收点阶段掩码）固定 shape 为 `[32, 4]`，仅在 `mode = C` 时消费；`mode = B` 路径不读取 recovery head（回收点头）
   - 固定 shape 采用 padding（填充）+ mask（掩码）方式，而不是可变长张量，以保证 `rollout_buffer.py`（轨迹缓存）可以稳定 batch 拼接
   - PPO 主训练路径的张量 shape 由 `max_candidate_orders`（最大候选订单数）与 `max_candidate_recovery_per_order`（每单最大回收候选数）冻结；`max_candidate_actions`（最大候选动作数）**不作为**主训练路径的硬裁剪阈值

   **`factorized_action_schema`（分解式动作结构）与 `resolved_action_lookup`（已解析动作映射）的职责分离固定为：**

   - `factorized_action_schema`：定义训练采样主路径使用的结构化索引空间
   - `resolved_action_lookup`：提供从策略采样结果反查最终 `EnvAction`（环境动作）的映射规则：
     - 若 `root_branch_idx = WAIT`，则直接取 `wait_action`
     - 若 `root_branch_idx = DISPATCH`，则用
       `(order_idx, mode_idx, recovery_idx)` 反查到对应 `DispatchAction`（派送动作）
   - `candidate_builder` 负责同时产出这两者；`model.py`（策略模型）不得自行重建
     三阶段索引空间，`env_adapter.step(...)`（环境执行）也不需要理解三头索引细节
   - `max_candidate_actions`（最大候选动作数）仅作为
     `resolved_action_lookup`（已解析动作映射）展开规模的保护性预算参数；
     若未来调大 `max_candidate_orders` / `max_candidate_recovery_per_order` 导致
     展开规模异常增长，应在 `candidate_builder.build()` 中用于断言 / 告警 / 降级保护，
     而不是用于 PPO 主训练路径的硬裁剪

   推荐最小结构如下：

   ```python
   @dataclass(frozen=True)
   class FactorizedActionSchema:
       root_branch_order: tuple[str, str]          # ("WAIT", "DISPATCH")
       mode_order: tuple[str, str]                 # ("B", "C")
       max_order_slots: int                        # 32
       max_recovery_slots: int                     # 4

   @dataclass(frozen=True)
   class ResolvedActionLookup:
       wait_action: GlobalWaitAction
       dispatch_actions: Mapping[
           tuple[int, int, int | None],           # (order_idx, mode_idx, recovery_idx)
           DispatchAction
       ]
   ```

   说明：

   - `ResolvedActionLookup.dispatch_actions`（已解析派送动作映射）中的 key 不包含
     `WAIT` 分支；`WAIT` 通过独立的 `wait_action` 提供
   - 当 `mode_idx` 对应 `B`（自主返程模式）时，`recovery_idx` 固定为 `None`
   - `rollout_buffer.py`（轨迹缓存）在动作存储上应保持与此一致：
     存储 `root_branch_idx`（顶层分支索引）以及可选的
     `order_idx / mode_idx / recovery_idx`，而不是只存一个扁平 action id（动作编号）

2. `planner_bridge.py`
   - 低频读取全局运行时状态与 `PlannerTriggerContext`（重规划触发上下文），生成或刷新只读 `CoarsePlanView`
   - 承接 RH-ALNS 上层骨架，输出 `CoarsePlanView`
   - 管理重规划周期（`coarse_replan_interval_sec=420`）
   - 管理 `plan_version` 更新与失效规则
   - 对 `backlog_new_orders`（新积压订单数）/ `route_drift_ratio`（路线漂移比例）/
     `fallback_count_in_window`（fallback 窗口计数）/
     `hard_failure_count_in_window`（硬失败窗口计数）做触发判断
   - 不直接推进环境事件，不直接结算奖励
   - 不直接维护环境事件账本；窗口计数与漂移相关输入由环境侧通过
     `PlannerTriggerContext` 传入
   - 本阶段 ALNS 可以简化实现（如基于贪心或固定规则），但接口必须满足 `CoarsePlanView` 契约
   - **新增**：输出 `launch_candidate_stations`，筛选规则为：
     - 卡车未来路线上尚未经过的站点，且该站点周边 `support_radius_km` 内存在至少 `min_orders_to_trigger` 个未分配订单
     - 卡车路线约束（不绕路）已在路线规划阶段保证，此处只做订单密度筛选
     - 更新时机：**仅在 ALNS 重规划时全量重算并写入 `CoarsePlanView`**
     - 卡车经过某站点后的实时移除、以及某站点周边订单全部被分配后的实时移除，
       不由 `planner_bridge` 回写 coarse plan，而由 `env_adapter` 内部
       `_active_launch_stations`（执行侧实时触发站点集合）维护
     - 首轮参数：`min_orders_to_trigger = 1`，可通过调高该值减少触发频率

固定调用链：

1. `env_adapter.advance_to_decision_event()`
2. 环境侧基于当前 `runtime_state`、内部事件账本与当前 coarse plan 参考信息组装
   `trigger_ctx: PlannerTriggerContext`
3. `planner_bridge.maybe_replan(runtime_state, trigger_ctx) -> CoarsePlanView`
4. 若 `plan_version`（规划版本号）发生递增，则 `env_adapter` 用新的
   `coarse_plan.launch_candidate_stations`（粗规划触发站点集合）整体重置
   `_active_launch_stations`（执行侧实时触发站点集合）
5. `candidate_out = candidate_builder.build(runtime_state, coarse_plan, deciding_drone_id, trigger_type, trigger_station_id, last_seen_plan_version)
   -> CandidateOutput(candidate_features, factorized_action_schema, action_mask, resolved_action_lookup)`
6. `ObservationTensorizer.tensorize(candidate_features, history_buffer, ego_drone_id, t_decision) -> ObservationBatch`
   （Phase 7 职责）
   - 其中 `history_buffer`（历史缓冲区）固定表示 rollout 侧维护的
     `global transition ring buffer`（全局转移环形缓冲区），存放最近发生的
     `committed transition summaries`（已提交真实转移摘要）
   - `ObservationTensorizer` 必须以当前 `ego_drone_id`（当前决策 UAV）为参考系，
     将最近 `hist_len` 条全局转移摘要重编码为
     `ego-conditioned history_tokens`（以当前 UAV 视角条件化的历史 token）
7. `CriticBatchBuilder.build(decision_context_snapshot, critic_tensor_schema_meta) -> CriticBatch`
   （仅 Phase 7 PPO 训练路径需要；规则型 baseline / 常规 benchmark greedy eval 不需要）
8. `policy` 或 `baseline` 消费候选语义，输出结构化动作索引：
   - PPO 主训练路径固定消费 `ObservationBatch + action_mask`，并由 critic 额外消费
     同快照下构造的 `CriticBatch`
   - 规则型 baseline 可直接消费 `CandidateOutput` 做打分，也可复用
     `ObservationBatch`；但两条路径都必须遵守同一套 `action_mask`
     与 slot 对齐语义
   - 若 `root_branch_idx = WAIT`，则选择 `WAIT_ACTION`
   - 若 `root_branch_idx = DISPATCH`，则选择
     `(order_idx, mode_idx, recovery_idx)`（订单索引、模式索引、回收点索引）
9. 由 `candidate_out.resolved_action_lookup` 将上述结构化索引反查为最终 `env_action: EnvAction`
10. `env_adapter.step(env_action)` 执行真实状态转移并结算 reward

其中：

- `CandidateOutput` 是训练循环 / baseline / 验证脚本侧的局部变量，不挂在
  `EnvStepResult`（环境步结果）或 `DecisionContext`（决策上下文）上
- `env_adapter`（训练环境适配器）的对外 API 边界保持为
  `EnvStepResult <-> EnvAction`，Phase 6 不改变这一边界
- `last_seen_plan_version`（上次看到的规划版本号）由外部调用方按 `drone_id`（无人机编号）
  维护：`reset()`（环境重置）/ episode 开始时清空；某 UAV 首次决策时默认取当前
  `coarse_plan.plan_version`；并在该 UAV 的**当前决策点被正式消费**时更新为本次
  `coarse_plan.plan_version`。训练主路径下通常由 rollout collector（轨迹收集器）维护，
  但 baseline / 验证脚本也应遵守同一更新语义

**Phase 5 -> Phase 6 的 `env_adapter` 切换约束**

为避免 Phase 6 接入后在 `env_adapter` 内部同时存在“每次现算 coarse plan”和“缓存 coarse plan”
两套并行语义，推荐把切换点收口为一个统一私有入口，例如：

```python
def _refresh_coarse_plan_if_needed(self) -> CoarsePlanView:
    ...
```

接口约束固定为：

- `env_adapter` 持有 `_current_coarse_plan: CoarsePlanView | None`；`reset()` 时显式清空
- `env_adapter` 持有 `_active_launch_stations: set[str]`；首次拿到 coarse plan 时按其
  `launch_candidate_stations` 初始化，后续只允许执行层删减
- 未注入 `planner_bridge` 时，`_refresh_coarse_plan_if_needed()` 直接走 Phase 5 路径：
  返回 `_build_coarse_plan_view(self._t_now)`，保持“每次重建、`plan_version=0` 整局不变”
- 注入 `planner_bridge` 后，`_refresh_coarse_plan_if_needed()` 走 Phase 6 路径：
  基于当前 `runtime_state + trigger_ctx` 调用 `planner_bridge.maybe_replan(...)`
- 若返回 coarse plan 的 `plan_version` 相比当前缓存发生递增，`env_adapter` 必须：
  1. 用新对象整体替换 `_current_coarse_plan`
  2. 用新的 `CoarsePlanView.launch_candidate_stations` **整体重置** `_active_launch_stations`
- 卡车经过某站点后的实时移除、以及某站点周边订单全部被分配后的实时移除，仍只作用于
  `_active_launch_stations`；它们**不回写** `_current_coarse_plan`，也**不构成**新的 `plan_version`

实现注意：

- 依赖“实时触发站点集合”的逻辑（如卡车到站是否触发 `riding_with_truck` 决策、
  `WAIT` 动作中的 `t_next_global_decision_event` 计算）应读取 `_active_launch_stations`，
  而不是直接读取缓存 coarse plan 上的 `launch_candidate_stations`
- 依赖 planner-issued snapshot 的逻辑（如 `truck_eta_map`、`route_drift_ref`、
  `authorized_orders`、`recovery_pool`）应读取 `_current_coarse_plan`
- `env_adapter` 在“是否入队一个新的决策触发点”上，不能被 stale coarse plan 卡死：
  若使用 `authorized_orders` 作为 suppress enqueue 的前置条件，必须先确保当前 coarse plan
  已有机会执行 `maybe_replan(...)`；否则会出现“新订单已进入 `pending_orders`，但旧 coarse plan
  仍未放行任何授权订单，导致不入队、不重规划、最终永远看不到新单”的闭环风险

这一阶段必须先固定以下参数：

- `support_radius_km`（默认 `1.2`）
- `max_candidate_orders`（默认 `32`）
- `max_candidate_recovery_per_order`（默认 `4`）
- `max_candidate_actions`（默认 `128`，用于 `resolved_action_lookup` 展开规模的保护性预算）
- `station_wait_threshold_sec`（默认 `240`）
- `coarse_replan_interval_sec`（默认 `420`）
- `coarse_new_order_trigger`（默认 `3`）
- `route_drift_trigger_ratio`（默认 `0.18`）
- `fallback_burst_trigger_count`（默认 `2`）
- `fallback_burst_window_sec`（默认 `300`）
- `hard_failure_trigger_count`（默认 `1`）
- `upper_horizon_sec`（默认 `3600`）

原因：

- 这些参数直接定义动作空间
- 这些参数同时定义 `planner_bridge.maybe_replan(...)`（粗规划桥接器：是否重规划）的触发语义
- 其中 `max_candidate_orders` / `max_candidate_recovery_per_order` 直接冻结主训练路径的张量 shape
- `max_candidate_actions` 不再定义 PPO 主训练路径 shape，而是约束
  `resolved_action_lookup` 的展开规模保护预算
- `route_drift_trigger_ratio` / `fallback_burst_trigger_count` /
  `fallback_burst_window_sec` / `hard_failure_trigger_count` 必须进入
  `PlannerMeta`（规划器元数据）与 `meta.json`（模型元数据），否则后续训练 run
  即使共享同一套其他参数，也无法完整复现 coarse replan（粗规划重规划）触发分布

同时必须把以下语义写进候选动作生成器：

1. 模式 C 候选回收点只能来自卡车未来合法交接节点或保底合法节点
2. 若 `t_arrive_uav(deliver -> r_j) + rendezvous_eta_safe_margin_sec > t_arrive_truck(r_j)`，该回收动作必须直接裁掉
3. 若电量不足以到达候选回收点并保留安全余量，该动作必须直接裁掉
4. `predicted_queue_time(node)` 可作为排序或软截断信号，但不应单独作为 mode C 等待回收动作的硬性非法判据
5. 若模式 C 原候选失效，fallback 只能指向最近可达充换电站或仓库
6. 全局 `WAIT` 动作必须始终存在，不允许被 mask 掉，也不应复制成多个
   `WAIT(order_i)` 等价动作
7. `max_candidate_actions` 仅用于 `resolved_action_lookup` 的展开规模保护，
   不用于 PPO 主训练路径的硬裁剪；若超出预算，应优先触发断言 / 告警，
   而不是静默改变训练侧 `order_mask / mode_mask / recovery_mask` 的 shape

产物：

- 候选动作生成器（factorized）
- `CandidateOutput`（候选输出结构）生成器：
  `candidate_features / factorized_action_schema / action_mask / resolved_action_lookup`
- 上层规划骨架入口（`CoarsePlanView` 输出）

验收标准：

- 在 `default_scene` 上每一步都能稳定生成候选集
- `resolved_action_lookup` 的展开规模不超过 `max_candidate_actions`
- mask 与候选动作一一对应
- `CoarsePlanView` 输出满足接口契约，PPO 可直接消费

**本轮代码同步记录（2026-04-26）**

本轮已在 `backend/training/` 下完成 Phase 6 的首版落地，当前仓库中的真实实现情况如下。

1. **已新增 Phase 6 共享动作与契约层**
   - 新增 `backend/training/actions.py`
     - 将 `DispatchAction` / `GlobalWaitAction` / `WAIT_ACTION` 从
       `env_adapter.py` 中拆出，供 `candidate_builder` 与 `env_adapter` 共同消费，
       避免循环依赖
   - 扩展 `backend/training/contracts.py`
     - 已新增 `PlannerTriggerContext`
     - 已新增 `UavSelfFeatures` / `OrderFeatures` / `RecoveryFeatures` /
       `InfraFeatures` / `InfraNodeFeatures`
     - 已新增 `CandidateFeatures` / `FactorizedActionSchema` /
       `ResolvedActionLookup` / `CandidateOutput`
   - 当前实现中：
     - `ResolvedActionLookup.resolve(...)` 已支持
       `root_branch_idx -> EnvAction` 的结构化反查
     - `ResolvedActionLookup.as_action_lookup()` 已提供到旧
       `DecisionContext.action_lookup` 的兼容映射

2. **已新增 `planner_bridge.py`，但当前仍为“规则化 coarse planner”而非完整 RH-ALNS**
   - 新增 `backend/training/planner_bridge.py`
   - 已实现：
     - `maybe_replan(runtime_state, trigger_ctx) -> CoarsePlanView`
     - 基于以下触发量做重规划判断：
       - `coarse_replan_interval_sec`
       - `backlog_new_orders`
       - `route_drift_ratio`
       - `fallback_count_in_window`
       - `hard_failure_count_in_window`
     - `plan_version` 递增
     - `launch_candidate_stations` 的订单密度筛选：
       - 仅从未来 backbone 的 `station` 节点中选
       - 以 `support_radius_km` 与 `min_orders_to_trigger` 判断是否放行
   - 当前仍是简化实现：
     - `authorized_orders` 仍直接基于当前 `pending_orders`、重量阈值与
       Phase 5 既有 `order_pre_score` / `priority_band` 规则生成
     - `recovery_pool` 仍按未来 backbone 前缀裁剪，不含更强的 ALNS 优化逻辑
     - 因此它已经满足 `CoarsePlanView` 契约，但**还不是**文档语义里的完整
       RH-ALNS 上层求解器

3. **已新增 `candidate_builder.py`，并切到 factorized 输出**
   - 新增 `backend/training/candidate_builder.py`
   - 已实现：
     - `build(runtime_state, coarse_plan, deciding_drone_id, trigger_type,
       trigger_station_id, last_seen_plan_version) -> CandidateOutput`
     - 固定 shape 的：
       - `root_branch_mask`
       - `order_mask`
       - `mode_mask`
       - `recovery_mask`
     - `CandidateFeatures` 生成
     - `ResolvedActionLookup` 生成
   - 当前候选动作语义已对齐到 Phase 6 约束：
     - `WAIT` 始终存在，且不进入订单/模式/回收点头
     - `mode B` 只暴露订单，不暴露返程宿主
     - `mode C` 只保留同时满足时序与能量约束的回收点
     - `riding_with_truck` 触发时，不直接使用
       `coarse_plan.recovery_pool[order]`，而是按 `trigger_station_id`
       动态生成 runtime recovery pool
     - `candidate_builder` 当前已同时兼容
       `trigger_type="riding_with_truck"`（文档对外语义名）与
       `trigger_type="truck_station_arrival"`（env 内部事件名）；
       二者都会走同一条“车到站触发 -> runtime recovery pool”路径
     - `best_mode_b_return_score` /
       `best_mode_b_host_type` /
       `predicted_queue_time_est`
       已在 build 时按当前 `RuntimeStateView.node_states` 即时计算
   - 当前实现中的一个明确边界：
     - `station_wait_threshold_sec` 当前只作为 recovery 候选排序中的软信号，
       **不作为** mode C 的硬性非法判据；这与文档约束一致
     - `CandidateOutput` 仍然**不挂在**
       `EnvStepResult` / `DecisionContext` 上；但当前已提供
       `candidate_builder.build_from_decision_context(...)` 与
       `env_adapter.build_candidate_output(...)` 这两条公开重建入口，
       外部调用方可基于既有 `DecisionContext` 显式重建并持有
       `CandidateOutput`

4. **`env_adapter.py` 已切到“统一 coarse plan 刷新入口 + Phase 6 候选构建器接线”**
   - `TrainingEnvAdapter` 新增：
     - `_current_coarse_plan`
     - `_active_launch_stations`
     - `_fallback_event_times`
     - `_hard_failure_event_times`
     - `_completed_backbone_nodes_since_plan`
   - 已新增统一入口：
     - `_refresh_coarse_plan_if_needed(...)`
   - 当前行为：
     - `enable_phase6=True` 时默认注入 `PlannerBridge` 与 `CandidateBuilder`
     - `enable_phase6=False` 时仍可回退到 Phase 5 的 inline coarse plan 语义
   - 已完成的 Phase 6 接线点：
     - `DecisionContext` 构造时不再直接依赖旧的 inline action builder，
       而是通过 `candidate_builder.build(...)` 生成兼容的
       `action_lookup`
     - `WAIT` 的下一次全局决策时刻计算改为读取 `_active_launch_stations`
     - 卡车经过站点后，会从 `_active_launch_stations` 中实时移除
     - 若某触发站点周边已无 pending UAV 订单，也会从 `_active_launch_stations`
       中实时移除
   - 当前仍需明确区分的边界：
     - 这里打通的是 **env 内部兼容链路**：`candidate_builder.build(...)`
       -> `resolved_action_lookup.as_action_lookup()` -> `DecisionContext.action_lookup`
     - 同时已补齐 **env 外部主消费入口**：
       `env_adapter.build_candidate_output(decision_context, last_seen_plan_version=...)`
       会基于 `DecisionContext` 快照重建 `CandidateOutput`，并校验其
       `resolved_action_lookup.as_action_lookup()` 与
       `DecisionContext.action_lookup` 完全一致
     - `env_adapter` 内部为了生成兼容 `action_lookup`，当前传给
       `candidate_builder.build(...)` 的 `last_seen_plan_version`
       固定等于 `coarse_plan.plan_version`；因此这条**内部兼容链路**上的
       `plan_version_delta` 不具备训练侧真实语义，真实语义仍应由外部调用方在
       Phase 7 rollout / baseline / validation 侧按 UAV 自行维护

5. **已显式修复“stale coarse plan 卡死新单”的闭环风险**
   - 文档中提到的风险已在 `env_adapter._has_authorized_orders_now(...)` 中做了保护
   - 当前实现口径为：
     - 先检查当前缓存 coarse plan 中是否还存在“仍在 `pending_orders` 中的 live authorized orders”
     - 若没有，但当前 `pending_orders` 中已有 UAV 可配送新单，则构造一个
       `backlog_new_orders >= coarse_new_order_trigger` 的强制触发上下文，
       再次调用 `planner_bridge.maybe_replan(...)`
   - 这样可以避免出现：
     - 旧 coarse plan 已经过时
     - 新订单已进入 `pending_orders`
     - 但由于旧 coarse plan 未授权任何新单，导致不入队、不重规划、永远看不到新单

6. **当前仍保留的一些 Phase 5 真值口径，没有被 Phase 6 错误改写**
   - `post_delivery_revalidation`
   - `reservation timeout`
   - `fallback_recovery`
   - `mode B` 执行层真实返程宿主选择
   - 上述逻辑仍以 `env_adapter` 的运行时真值为准；Phase 6 只是把
     coarse planning 与 candidate building 从 env 内部职责中拆了出来，
     没有改坏 Phase 5 的状态转移与 reward 结算边界

7. **已补充并修正 Phase 6 集成测试**
   - 新增 `backend/training/test_phase6_integration.py`
   - 当前已覆盖：
     - `planner_bridge` 重规划后整体重置 `_active_launch_stations`
     - `candidate_builder` 的 mask / feature / resolved lookup 对齐关系
     - stale cached coarse plan 不会压制新 idle 决策：
       - 该测试最初版本只验证“有新单时能入队”，不是真正的 stale-plan 场景
       - 本轮已修正为：
         1. 先显式生成旧 `_current_coarse_plan`
         2. 再注入旧 plan 不包含的新订单
         3. 验证会触发 replan、`plan_version` 递增、新订单被授权并成功入队
   - 当前**尚未**覆盖的边界：
     - `last_seen_plan_version` 在外部按 UAV 持久化维护后的
       `plan_version_delta` 行为
     - `InfraNodeFeatures.is_launch_candidate_station` 与执行侧
       `_active_launch_stations` 的语义分离是否被后续观测编码正确处理
   - 当前已新增覆盖：
     - `env.current_decision_context -> env.build_candidate_output(...)`
       这条公开入口可稳定基于同一 `DecisionContext` 快照重建
       `CandidateOutput`

8. **当前本地回归结果**
   - 已通过的验证命令：

     ```bash
     python -m unittest \
       backend.training.test_phase6_integration \
       backend.training.test_env_adapter_phase5a \
       backend.training.test_env_adapter_phase5b \
       backend.training.test_env_adapter_phase5c
     ```

   - 当前结果：
     - `20` 个测试全部通过

9. **本轮未完成、仍应保持清晰认知的边界**
   - `planner_bridge.py` 当前是**规则化骨架实现**，不是完整 RH-ALNS
   - `CandidateOutput` 的数据结构、mask 语义、`resolved_action_lookup`
     以及基于 `DecisionContext` 的公共重建入口已经落地；但
     Phase 7 训练主路径以及后续 baseline / validation 脚本本身
     仍未实现，尚未真正接入这条公开主消费链路
   - `last_seen_plan_version` 的跨 step / 按 UAV 持久化维护，当前仍应由
     Phase 7 rollout collector 或后续 baseline / validation 脚本承担；
     `env_adapter` 本身不持有这份外部消费态
   - `CandidateFeatures.infra_features.node_features[*].is_launch_candidate_station`
     当前反映的是 `CoarsePlanView.launch_candidate_stations`
     这一 planner-issued snapshot upper bound，而**不是**执行侧实时集合
     `_active_launch_stations`；因此它目前只能按“planner 快照特征”理解，
     不能误当作“当前时刻一定会触发 riding_with_truck 决策”的实时真值
   - `ObservationTensorizer` / `model.py` / `rollout_buffer.py`
     尚未实现，这部分仍属于 Phase 7
   - 因此，本轮 Phase 6 的准确定位是：
     - **候选动作与 coarse-plan 接口已经落地**
     - **env 内部兼容 `action_lookup` 的编排链路已打通**
     - **面向训练 / baseline / validation 的 `CandidateOutput` 公共重建入口已补齐**
     - **完整 PPO 主循环尚未接入**

## 4.7 Phase 7：实现 PPO 训练主循环

目标：

- 让环境和模型真正连起来，产出第一版权重

建议新增文件：

- `backend/training/train_cmrappo.py`
- `backend/training/model.py`
- `backend/training/rollout_buffer.py`
- `backend/training/observation_tensorizer.py`
- `backend/training/critic_batch_builder.py`

最小训练闭环需要：

- 模型前向（factorized actor head + centralized critic head）
- action mask 采样（三阶段 factorized）
- `CriticBatch`（价值网络批）构造与写入
- rollout 收集
- advantage 计算（GAE，`gae_lambda=0.95`）
- PPO 更新（`clip_coef=0.2`）
- checkpoint 保存

**模型结构（Shared PPO-Lite）**

建议结构：

1. Task encoder：对候选订单集合编码
2. UAV / station / truck encoder：对局部载体与基础设施上下文编码
3. Cross-attention：融合"当前决策无人机"和"候选订单/回收节点"关系
4. history encoder（历史编码器）：编码最近 `hist_len=6` 个
   `ego-conditioned global transition history tokens`（以当前 UAV 视角条件化的全局转移历史 token）。
   这里的 6 步是 6 个“局部决策事件步/已提交真实转移摘要”，不是固定仿真秒步。
   它的职责固定为：把当前决策时刻因果可见的显式历史摘要窗口编码成
   `history_context`（历史上下文向量）；它属于单个 decision step（决策步）内的
   `stateless encoder`（无状态编码器），不消费 rollout 侧按 UAV 维护的
   `lstm_state`（循环隐藏状态），也不向 rollout 持久化自己的跨步状态
5. recurrent core（循环核心，Phase 7 V1 默认为 LSTM）：只负责当前
   `actor_drone_id`（当前决策无人机）自身的跨决策隐式记忆递推。
   其输入应是当前 step 的 fused context（融合上下文，例如 `current_context +
   history_context`），而不是把 `history_tokens` 本身当作 recurrent sequence（循环序列）
   反复展开。`LSTM hidden state` 按 `drone_id` 单独维护，只在该 UAV 自己成为本次
   actor 时推进；其他 UAV 的近期行为通过 `history_tokens` 进入当前观测，而不是共享同一份
   hidden state
6. Actor head：输出 factorized masked action logits（三阶段）
   - 顶层先输出 `root_branch_logits`（顶层分支 logits），顺序固定为 `[WAIT, DISPATCH]`
   - 仅当选择 `DISPATCH` 时，再消费 dispatch 分支的
     `order_logits / mode_logits / recovery_logits`
   - `recovery`（回收点）阶段的条件概率语义固定为：
     `P(recovery_idx | observation, root=DISPATCH, order_idx, mode=C)`。
     也就是说，`recovery_idx` 的采样与 `log_prob` 计算必须条件化在**已选定的**
     `order_idx` 上；当 `mode=B` 时，不存在 recovery 采样
   - `ObservationBatch.recovery_tokens` 的结构固定为
     `[order_slot, recovery_slot] = [32, 4]` 的预分配二维张量，而不是“当前所选订单的
     4 个临时 token”。因此模型实现可以：
     1. 在拿到 `order_idx` 后，只对 `recovery_tokens[order_idx]` 这一行做打分；
     2. 或先计算整个 `[32, 4]` 的 recovery logits grid，再在采样与训练时仅消费
        `order_idx` 对应的那一行。
     两种实现都允许，但对外暴露给 PPO 的语义必须一致：只消费已选 `order_idx`
     对应的 recovery row
7. Critic head：训练时使用 centralized critic，但其额外全局输入必须通过
   显式的 `CriticBatch`（价值网络批）契约提供，不允许在 `model.py` 内部隐式读取
   全局状态。固定接口语义为：
   - `policy_out = actor(observation_batch, action_mask)`
   - `value = critic(observation_batch, critic_batch)`
   - 其中 actor（策略头）只消费 `ObservationBatch + action_mask`；
     critic（价值网络）在此基础上额外消费 `CriticBatch`
   - `CriticBatch` 只在 Phase 7 训练 / rollout update（轨迹更新）路径中使用，
     推理阶段与常规 greedy benchmark eval（贪心基准评估）不需要

`WAIT`（全局等待动作）在训练实现中的接口语义固定为：

- 它是 actor（策略头）的独立顶层分支，不伪装成 `order_idx = WAIT_SENTINEL`
  或 `mode = WAIT`
- 因此 PPO 的 `log_prob`（对数概率）按分支求和：
  - 若选择 `WAIT`：只计算 `root_branch_logits` 上的 `log_prob`
  - 若选择 `DISPATCH`：计算
    `root_branch + order + mode (+ recovery_if_mode_c)`（顶层分支 + 订单 + 模式 + 可选回收点）
- `rollout_buffer.py`（轨迹缓存）必须显式区分这两条动作路径，不能把 `WAIT`
  与派送动作压成同一种扁平编号

关键参数：

- `d_model=128`、`nhead=8`、`ff_dim=256`、`lstm_hidden=128`、`lstm_layers=1`
- `critic_mode=centralized_train_only`（在线推理不需要 centralized critic）
- `critic_schema_name=CriticTensorSchemaV1`，且 `max_global_orders / max_global_uavs /
  max_global_stations / normalizer_meta` 必须来自配置快照并写入 `meta.json`
- `inference_mode=greedy`（推理阶段默认 greedy，不引入额外随机性）

**观测张量规范（5 类 token）与 `ObservationTensorizer`**

Phase 7 新增 `ObservationTensorizer`（`observation_tensorizer.py`），职责固定为：

1. 消费 Phase 6 产出的 `CandidateFeatures`，按固定 slot 顺序将各字段编码为浮点张量
2. 注入 `history_tokens`（最近 `hist_len=6` 个
   `ego-conditioned global transition history tokens`，由 rollout 侧维护的全局历史缓冲区提供）
3. 组装结构性 padding mask：`padding_mask`（订单层）、
   `recovery_padding_mask`（回收候选层）与 `history_padding_mask`（历史层），
   供各自编码层屏蔽 padding slot
4. 输出 `ObservationBatch`，供 `model.py` 直接消费

**边界约束**：`ObservationTensorizer` 的输出 shape 由 token 字段数量决定，不依赖 `d_model` / `nhead`。具体地：

- `uav_self_token`：shape `[uav_feat_dim]`，`uav_feat_dim` 由 `UavSelfFeatures` 的字段数决定
- `order_tokens`：shape `[32, order_feat_dim]`，`order_feat_dim` 由 `OrderFeatures` 的字段数决定
- `recovery_tokens`：shape `[32, 4, recovery_feat_dim]`，`recovery_feat_dim` 由 `RecoveryFeatures` 的字段数决定
- `infra_tokens`：shape `[n_infra_nodes, infra_feat_dim]`，`n_infra_nodes` 为 station + depot 总数
- `history_tokens`：shape `[hist_len, hist_feat_dim]`，`hist_len=6`
- `history_padding_mask`：shape `[hist_len]`，True 表示该 history slot 为 padding

`model.py` 的第一层 linear 负责把各 token 从 `feat_dim` 投影到 `d_model=128`；这个投影属于模型内部，不属于 `ObservationTensorizer`。

`ObservationBatch` 定义：

```python
@dataclass(frozen=True)
class ObservationBatch:
    uav_self_token: Tensor          # [uav_feat_dim]
    order_tokens: Tensor            # [32, order_feat_dim]
    recovery_tokens: Tensor         # [32, 4, recovery_feat_dim]
    infra_tokens: Tensor            # [n_infra_nodes, infra_feat_dim]
    history_tokens: Tensor          # [hist_len, hist_feat_dim]
    history_padding_mask: Tensor    # [hist_len]，True 表示该 history slot 为 padding
    padding_mask: Tensor            # [32]，True 表示该 order slot 为 padding
    recovery_padding_mask: Tensor   # [32, 4]，True 表示该 recovery slot 为 padding
```

**5 类 token 的字段来源**：

1. `uav_self_token`（来自 `CandidateFeatures.uav_self`）：位置、剩余电量、状态、剩余时间、reservation 状态、`plan_version_delta`、`is_riding_truck`、`drone_source_type`
2. `order_tokens`（来自 `CandidateFeatures.order_features`，slot 顺序与 `order_mask` 严格对齐）：基础订单字段 + `best_mode_b_return_score` + `best_mode_b_host_type`（均由 Phase 6 `candidate_builder.build()` 时计算）
3. `recovery_tokens`（来自 `CandidateFeatures.recovery_features`，slot 顺序与 `recovery_mask` 严格对齐）：按 `[order_slot, recovery_slot] = [32, 4]` 预先构建每个候选订单各自的回收节点候选表示；推理时仅在 `mode=C` 且已选定 `order_idx` 后消费对应 `recovery_tokens[order_idx]` 这一行。字段包含回收节点位置、ETA、`rendezvous_margin`、`reservation_count`
4. `infra_tokens`（来自 `CandidateFeatures.infra_features`）：truck 骨架摘要 + station/depot 摘要（queue、ETA、parking_slots、node_charge_load_budget）
5. `history_tokens`（由 rollout 侧历史缓冲区提供，不来自 `CandidateFeatures`）：
   最近 `hist_len=6` 个
   `ego-conditioned global transition history tokens`（以当前 UAV 视角条件化的全局转移历史 token）

**`history_tokens` 契约冻结**

为避免 Phase 7 对“历史”各自理解，`history_tokens` 的语义在此固定为：

- 它不是仅当前 UAV 的私有历史（`self history`）
- 也不是无筛选的裸全局事件日志（`raw global event log`）
- 它表示：
  最近 `hist_len` 个
  `global committed transition summaries`（全局已提交真实转移摘要），
  在当前决策 UAV 视角下重编码后的历史 token 序列

这里的单条 `committed transition summary`（已提交真实转移摘要）固定指：

- 某架 UAV 在某个 `DecisionContext` 下选择了一个动作
- 该动作被 `env_adapter.step(action)` 正式提交
- 环境推进完成该次 transition（转移）的真实状态演化与 reward settlement（奖励结算）
- 再将这段真实演化压缩成一条摘要记录，写入 rollout 侧全局历史缓冲区

因此：

- 一次 `WAIT`（全局等待动作）对应 1 条 history 记录
- 一次 `DISPATCH`（派送动作）对应 1 条 history 记录
- 其间发生的 `delivery`（送达）、`reservation_timeout`（预留超时）、
  `fallback_started`（进入兜底返程）、`queue_entered`（进入排队）等，只作为
  该条 transition summary 的结果字段出现，不单独各占一个 history slot

rollout 侧维护规则固定为：

- 维护一份 `global transition ring buffer`（全局转移环形缓冲区），按真实发生顺序追加
- 每次 `env_adapter.step(action)` 返回后，最多只追加 1 条
  `TransitionSummary`（转移摘要）
- 不为环境内部微事件单独追加 history 记录
- `LSTM hidden state` 按 `drone_id` 单独维护，只在该 UAV 自己成为 actor
  时推进；不允许所有 UAV 共享同一份 hidden state

为避免“显式滑窗历史”和“隐式循环记忆”被同一个模块重复建模，Phase 7 在这里进一步冻结：

- `history_tokens`（显式全局历史摘要窗口）不等于 recurrent sequence（循环序列）本身；
  它表达的是**当前决策时刻因果可见**的最近 `hist_len` 条全局已提交转移摘要
- `history_encoder`（历史编码器）只在单个 decision step（决策步）内对
  `history_tokens` 做无状态编码，输出 `history_context`（历史上下文向量）；
  它不得消费 `lstm_state_in`（输入循环隐藏状态），也不得产出需要跨 decision step
  持久化的隐藏状态
- `lstm_state`（循环隐藏状态）只表示当前 `actor_drone_id`（当前决策无人机）
  自身的跨决策隐式记忆；真正跨 actor 决策序列连续 unroll（展开）的，只能是后续
  `recurrent_core`（循环核心），而不是 `history_encoder`
- 因此，Phase 7 V1 允许“显式历史窗口 + 隐式循环记忆”同时存在，但禁止由同一个
  recurrent module（循环模块）同时承担
  `history window encoder`（历史窗口编码器）和
  `cross-step recurrent core`（跨步循环核心）两种职责
- `history_tokens` 中保留 `self event`（当前 UAV 自身历史事件）是允许的；
  这部分语义仍属于“当前时刻可见的全局事件记录”，不应被实现者误解为
  “因此可以不再维护按 UAV 独立推进的 `lstm_state`”

`ObservationTensorizer` 在消费这份全局历史时必须：

- 以当前 `ego_drone_id`（当前决策 UAV）为参考系重编码最近 `hist_len` 条记录
- 生成：
  - `dt_since_event = t_decision - event_time`
  - `is_self_event = (actor_drone_id == ego_drone_id)`
  - `actor_rel_x / actor_rel_y`（相对于当前 ego UAV 的相对位置）
- 当历史不足 `hist_len` 时，使用固定 padding token 补位，并通过
  `history_padding_mask` 显式标记；不允许依赖全零数值让模型自行猜测

推荐最小原始语义对象可冻结为：

```python
@dataclass(frozen=True)
class TransitionSummary:
    event_time: float
    actor_drone_id: str

    actor_pos_x: float
    actor_pos_y: float

    actor_training_state_before: str
    actor_training_state_after: str
    actor_home_type: str
    actor_payload_class: str

    trigger_type: str

    root_branch: str                # WAIT / DISPATCH
    dispatch_mode: str              # NONE / B / C
    selected_recover_node_type: str # NONE / station / depot

    has_selected_order: bool
    selected_order_slot_rank: int
    selected_order_deadline_slack_norm: float
    selected_eta_to_deliver_norm: float
    selected_rendezvous_margin_norm: float

    energy_ratio_before: float
    energy_ratio_after: float
    queue_after_norm: float
    plan_version_delta_at_event: int

    delivered: bool
    rendezvous_success: bool
    reservation_timeout: bool
    fallback_started: bool
    hard_failure: bool
    queue_entered: bool
    service_completed: bool
```

字段语义固定如下：

- `trigger_type` 必须保留，因为当前文档中 `idle`（地面空闲触发）与
  `riding_with_truck`（随车搭载触发）共享 factorized action structure（分解式动作结构），
  但 `WAIT` 的真实语义不同；若不记录 `trigger_type`，模型会把“原地等待”与“继续搭车”等价化
- `selected_order_slot_rank` / `selected_order_deadline_slack_norm` /
  `selected_eta_to_deliver_norm` / `selected_rendezvous_margin_norm` /
  `energy_ratio_before` / `energy_ratio_after` / `queue_after_norm` /
  `plan_version_delta_at_event` 必须保留，用于表达该次动作的风险结构，而不仅是结果标志
- `actor_home_type` 与 `actor_payload_class` 建议同时保留，不做二选一；
  它们分别表达宿主来源差异与载重类别差异
- `selected_recover_node_type` 只保留 `node type`（节点类型），不保留原始 `node_id`
- 对 `WAIT` 或 `mode=B` 等不适用路径，`dispatch_mode=NONE` /
  `selected_recover_node_type=NONE`；相关数值字段统一填 `0`，并通过
  `has_selected_order` 等布尔字段区分“真实 0”与“不适用”

字段来源边界固定为：

- 来自 `action-time snapshot`（动作提交时快照）的字段：
  `trigger_type`、`root_branch`、`dispatch_mode`、`has_selected_order`、
  `selected_order_slot_rank`、`selected_order_deadline_slack_norm`、
  `selected_eta_to_deliver_norm`、`selected_rendezvous_margin_norm`、
  `energy_ratio_before`、`plan_version_delta_at_event`、
  `actor_training_state_before`
- 来自 `transition-end snapshot`（转移结束时快照）的字段：
  `event_time`、`actor_pos_x / actor_pos_y`、`actor_training_state_after`、
  `energy_ratio_after`、`queue_after_norm` 以及全部 `result_flags`

明确禁止进入 `history_tokens` / `TransitionSummary` 的内容：

- `raw reward`（原始奖励标量）
- `raw order_id / drone_id / node_id`（原始订单/无人机/节点 ID，`actor_drone_id`
  仅允许保留在 rollout 内部缓冲区中，用于派生 `is_self_event`，不得直接入模）
- `mode_a_background_completed`（卡车背景订单完成）
- 未来真实订单
- 当前 step 之后才因果可见的信息
- 裸环境微事件日志

这样冻结的目标是：

- 让当前 UAV 能看见其他 UAV 的近期协作/拥挤/资源竞争上下文
- 但不把历史退化成“噪声过高的全局事件流水”
- 同时保留按 UAV 单独维护的 `LSTM hidden state`，承接个人时间记忆

`ObservationTensorizer` 的结构性 mask 构造规则固定为：

- `padding_mask[i] = (not CandidateFeatures.order_features[i].is_valid)`
- `recovery_padding_mask[i][j] = (not CandidateFeatures.recovery_features[i][j].is_valid)`
- `history_padding_mask[h] = True` 表示该 history slot 仅为补位，不承载真实转移摘要
- 若 `padding_mask[i] = True`，则必须同时满足
  `recovery_padding_mask[i, :] = [True, True, True, True]`，保持二维 slot 结构一致性
- `recovery_tokens[i][j]` 仅在 `recovery_padding_mask[i][j] = False` 时承载真实物理语义；
  否则只是固定 shape 下的补位值，不允许模型赋予其“真实回收候选”含义

Padding/Masking 规则：

- `padding_mask`：只用于 order-level attention（订单层注意力）屏蔽 padding 的 order slot
- `recovery_padding_mask`：只用于 recovery-level attention（回收候选层注意力）屏蔽
  padding 的 recovery slot
- `history_padding_mask`：只用于 history encoder / temporal unit（历史编码层/时序单元）
  屏蔽不存在的历史记录
- `action_mask`：只用于 actor head（策略头）屏蔽物理非法动作（来自
  `CandidateOutput` 的 `order_mask / mode_mask / recovery_mask`）
- `padding_mask / recovery_padding_mask / history_padding_mask` 与 `action_mask`
  不允许混用

`model.py`（策略模型）在消费这些 mask 时的职责固定为：

1. order encoder（订单编码层）只能使用 `padding_mask` 做结构性屏蔽
2. recovery encoder（回收候选编码层）只能使用
   `recovery_padding_mask[order_idx]` 做结构性屏蔽
3. history encoder / temporal unit 只能使用 `history_padding_mask`
   处理“历史不足 `hist_len`”的补位记录，不能把它与动作合法性屏蔽混用
4. actor head（策略头）输出 logits（对数几率）后，才使用
   `order_mask / mode_mask / recovery_mask` 做动作合法性屏蔽
5. 当选择 `mode=C` 时，只允许消费已选 `order_idx` 对应的
   `recovery_tokens[order_idx]` 与 `recovery_padding_mask[order_idx]`
6. 不允许把 `CandidateOutput.recovery_mask` 直接复用为 recovery attention 的结构性 mask；
   前者表达“该动作当前不可采样”，后者表达“该 slot 根本不存在”

关于“哪些信息应该显式提供、哪些信息应该交给模型学习”的边界，Phase 7 固定如下：

- `padding_mask / recovery_padding_mask / history_padding_mask` 属于**数据结构真值**，
  表示哪些 slot 根本不承载真实对象；这部分不应由模型从补位值中自行猜测，
  而应由上游显式提供
- `recovery_tokens` 中真实候选之间的优劣比较，例如更大的 `rendezvous_margin`、
  更小的 `predicted_queue_time_est`、更优的后续拥挤代价，则属于策略应当学习的部分
- 当前 Phase 6 实现中，`RecoveryFeatures.is_valid=False` 与 `recovery_mask=False`
  在许多样本上可能高度重合；但这只是现实现象，不构成两者语义等价。后续若引入
  “结构上存在、但当前动作上不可采样”的 recovery 候选，仍必须能被两套 mask 独立表达

数值安全约束：

- 允许某个真实订单 `order_idx` 出现
  `recovery_padding_mask[order_idx] = [True, True, True, True]`，这表示该订单当前
  没有任何可编码的 recovery 候选，通常对应 `mode_mask[order_idx][C] = False`
- 对这种“全 padding 行”，`model.py` 不得直接执行普通 recovery-level attention
  softmax；必须采用以下两种等价安全策略之一：
  1. 在进入 recovery 分支前由 `mode_mask` 短路，使该订单根本不进入 recovery 编码路径
  2. 显式跳过该行 recovery attention，并返回零向量或固定哨兵向量
- 上述“全 padding 行”的数值安全处理属于 Phase 7 `model.py` 的职责；
  Phase 6 的 `candidate_builder` 只需保证 `is_valid` 与 slot 对齐，不负责 attention 数值实现

**`CriticBatch`（价值网络批）与 `CriticBatchBuilder`**

为避免 `centralized critic` 在 Phase 7 落地时分裂成“一版只吃局部观测”和“另一版偷偷读取全局状态”两套实现，
critic 额外输入必须冻结为独立契约 `CriticBatch`，并由
`CriticBatchBuilder`（`critic_batch_builder.py`）在训练侧显式构造。

Phase 边界固定为：

- **Phase 6（`candidate_builder.py`）**：不构造 `CriticBatch`；仍只负责
  `CandidateOutput(candidate_features, factorized_action_schema, action_mask, resolved_action_lookup)`
- **Phase 7（`critic_batch_builder.py`）**：基于与 `ObservationBatch` 同一份
  `DecisionContext` 快照构造 `CriticBatch`，并与
  `ObservationBatch / action_mask / resolved_action_lookup` 一起写入
  `rollout_buffer.py`
- **Phase 8（`validate_benchmark.py` / `greedy_local_policy.py`）**：
  常规 greedy baseline（贪心基线）与 policy greedy eval（策略贪心评估）不依赖
  `CriticBatch`；若后续实现 benchmark 训练诊断或 value 评估，可复用同一契约，
  但不改变其“训练专用输入”的边界

`CriticBatch` 不允许只冻结到“语义对象”层，还必须冻结到**实际 tensor schema（张量结构契约）**
层。否则即使 `global_order_pool_state / global_uav_state / global_station_state`
这些对象名称保持不变，不同实现仍可能在排序、截断、padding、归一化与写盘方式上各自漂移，
导致 value head（价值头）在不同 run（训练轮次）中拟合的并不是同一个状态空间。

因此，Phase 7 第一版必须正式定义：

**`CriticTensorSchemaV1`（价值网络张量结构契约 V1）**

它至少回答以下 8 件事：

1. `CriticBatch` 具体有哪些 tensor（张量）
2. 每个 tensor 的固定 shape（形状）
3. 每类 token 的字段顺序
4. 全局对象的排序规则（ordering rule）
5. 超过容量时的截断规则（truncation rule）
6. 不足容量时的 padding / mask 规则
7. 数值归一化规则（normalization rule）
8. 写入 `rollout_buffer.py` 与 `meta.json` 的存储/持久化规则

`CriticTensorSchemaV1` 对 critic 的定位固定为：

- actor（策略头）只看当前 UAV 可得的局部候选观测与 `action_mask`
- critic（价值网络）在训练阶段额外看当前时刻**真实可得**的全局系统状态，
  用于更稳定地估计 `V(s_t)`（状态价值）
- inference（推理）不依赖 critic

这属于 `centralized_train_only critic`（仅训练阶段使用的集中式价值网络）边界，
符合 CTDE（集中训练、分布执行）思想；但 critic 虽然可以看全局，
也只能看**当前时刻真实可得**的全局状态，不能看未来真实结果

**`CriticTensorSchemaV1` 推荐最小语义范围**

1. `global_order_pool_state`（全局订单池状态）
   - 范围固定为：当前时刻所有仍会影响未来回报的 active orders（活动订单），
     至少包含 `pending_orders + assigned_orders`
   - 不包含：`completed / timeout removed / hard removed / future unspawned orders`
   - 若后续希望表达“已进入执行中”的细分状态，不新增独立
     `in_progress_orders` 集合，而应通过 token 内部的 `status`（状态）字段表达，
     避免不同实现各自发明集合边界
   - 必须显式区分：
     - `is_uav_primary_scope`（是否属于 UAV 主指标范围）
     - `is_mode_a_background`（是否属于 mode A 背景订单）
     - `is_assigned`
     - `is_overdue`

2. `global_uav_state`（全局 UAV 状态）
   - 范围：所有 UAV 在当前决策时刻的运行状态
   - 至少包含：
     - `training_state`
     - 剩余电量 / 电量比例
     - `eta_to_available`
     - `is_current_actor`
     - `is_riding_with_truck`
     - `has_reservation`
     - `reservation_remaining_sec`
     - `carrying_order_exists`
     - `dispatch_mode`
     - `uav_type / payload class`

3. `global_station_state`（全局站点/仓库状态）
   - 范围：所有 `station / depot` 的系统负载状态
   - 典型字段：`queue_length`、`available_slots`、`predicted_queue_time_est`、
     `reservation_count`、`node_charge_load_budget`、`truck_eta_remaining`
   - 与 `ObservationBatch.infra_tokens` 的关系固定为：
     - `ObservationBatch.infra_tokens` 服务 actor 的局部决策，强调当前候选动作相关的
       基础设施摘要，推理阶段必须可用
     - `CriticBatch.global_station_tokens` 服务 critic 的全局估值，强调所有节点的系统负载、
       队列、reservation 与未来到达压力，只用于训练价值估计
   - 若同时包含 `is_future_truck_stop` 与 `is_active_launch_station`，必须显式区分：
     - 前者来自当前 `coarse_plan` / backbone（规划器快照语义）
     - 后者来自执行侧实时集合 `_active_launch_stations`（运行时真值）

4. `coarse_plan_summary_state`（粗规划摘要）
   - 使用 summary vector（摘要向量）
   - 至少包含：`plan_version`、剩余 truck backbone 长度、未来
     `launch_candidate_stations` 数量、`authorized_orders` 数量、route drift 摘要、
     fallback / hard failure 窗口统计

5. `global_system_summary_state`（全局系统摘要）
   - Phase 7 V1 建议新增独立 summary vector，而不是只依赖 top-K tokens
   - 典型字段：`active_order_count`、`pending_order_count`、`assigned_order_count`、
     `overdue_order_count`、`mode_a_background_count`、`station_queue_total`、
     `reservation_total`、`fallback_count_window`、`hard_failure_count_window` 等
   - 目的：当 active orders 超过 `K_order` 时，即使详细 token 被截断，
     critic 也仍然知道系统整体压力

**`CriticTensorSchemaV1` 默认张量结构**

```python
@dataclass(frozen=True)
class CriticBatch:
    global_order_pool_tokens: Tensor        # [K_order, F_order]
    global_uav_tokens: Tensor               # [K_uav, F_uav]
    global_station_tokens: Tensor           # [K_station, F_station]
    coarse_plan_summary_vec: Tensor         # [F_plan]
    global_system_summary_vec: Tensor       # [F_sys]

    global_order_padding_mask: Tensor       # [K_order]
    global_uav_padding_mask: Tensor         # [K_uav]
    global_station_padding_mask: Tensor     # [K_station]
```

Phase 7 V1 默认容量建议冻结为：

- `K_order = 64`
- `K_uav = 16`
- `K_station = 11`（当前场景为 `10 station + 1 depot`）

说明：

- 上述值是 **Phase 7 V1 默认值**，不是理论常量
- 它们可配置，但一旦某次 run 开始正式训练，必须整局固定，并写入
  `meta.json`
- `global_order_pool_tokens / global_uav_tokens / global_station_tokens`
  使用 token + padding mask 形式是当前默认实现；
  `coarse_plan_summary_vec / global_system_summary_vec` 使用 summary vector
  形式是当前默认实现

**排序规则（ordering rule）**

`global_order_pool_tokens`：

- 排序键固定为：

  ```text
  sort_key(order) =
  (
    scope_rank,         # 0 = is_uav_primary_scope, 1 = is_mode_a_background
    overdue_rank,       # 0 = is_overdue, 1 = not_overdue
    status_rank,        # 0 = pending, 1 = assigned / executing
    deadline_slack_sec, # 越小越靠前
    priority_band,
    spawn_time_sec,
    stable_order_id
  )
  ```

- 这样第 `0` 个 order token 恒表示“当前最紧急、最影响 UAV 主目标的订单”，
  而不是偶然的字典遍历第一个

`global_uav_tokens`：

- `slot 0` 固定为当前决策 UAV
- 其余 UAV 在默认不截断场景下按 `stable_drone_id` 排序
- 若未来 UAV 数量超过 `K_uav`，则：
  - 当前决策 UAV 必保留
  - 其他 UAV 按 `fallback / low_energy / has_reservation / eta_to_available / stable_drone_id`
    的优先级截断保留

`global_station_tokens`：

- 当前阶段默认不做 top-K，直接全量保留
- `slot 0 = depot`
- `slot 1.. = stations sorted by stable_node_id`
- 若未来站点数量超过 `K_station`，则：
  - `depot` 必保留
  - 卡车未来会经过的 `station` 优先
  - `active_launch_station` 优先
  - 剩余按 `stable_node_id` 截断

**截断规则（truncation rule）**

- `global_order_pool_tokens` 固定采用
  `top-K urgent active orders`（前 K 个最紧急活动订单）
- `K_order = critic.max_global_orders`
- 超过 `K_order` 的订单不进入 token 序列，但其数量与聚合统计必须进入
  `global_system_summary_vec`
- 不允许使用“插入顺序截断”或“字典遍历顺序截断”，因为在泊松流高峰期这会让 critic
  看不到当前最危险的超时边缘订单

**padding / mask 规则**

- `padding token` 统一填 `0`
- `padding_mask=True` 表示该 slot 不承载真实对象
- 模型必须在 attention / pooling 时显式屏蔽 padding slot
- 不允许模型从全零 token 自行推断 padding

固定定义：

- `global_order_padding_mask[i] = True  <=> order slot i is padding`
- `global_uav_padding_mask[i] = True    <=> uav slot i is padding`
- `global_station_padding_mask[i] = True <=> station slot i is padding`

空集合时：

- `tokens = all zeros`
- `padding_mask = all True`
- `global_system_summary_vec` / `coarse_plan_summary_vec` 仍然填真实统计值，
  例如 `active_order_count=0`

critic 侧的 padding 语义与 `ObservationBatch` 一致：padding mask 属于结构真值，
不应交给模型自己猜测

**归一化规则（normalization rule）**

必须明确禁止：

- 按当前 batch 最大值归一化
- 按当前 rollout 最大值归一化
- 按本次验证集动态统计归一化

统一原则：

- 所有归一化基准必须来自配置或场景固定量
- 必须写入 `meta.json`

推荐最小 `NormalizerMeta`（归一化元数据）口径：

```text
time_norm_sec      = upper_horizon_sec
distance_norm_m    = map_diagonal_m
payload_norm_kg    = 10.0
energy_norm        = uav_type_capacity_j
eta_norm_sec       = upper_horizon_sec
queue_norm_cap     = configured_queue_cap_or_station_slots
```

推荐公式可以写死为：

- `deadline_slack_norm = clip((deadline_sec - t_now) / upper_horizon_sec, lower, upper)`
- `age_norm = clip((t_now - spawn_time_sec) / upper_horizon_sec, 0, 1)`
- `queue_length_norm = clip(queue_length / queue_norm_cap, 0, 1)`
- `truck_eta_remaining_norm` 若当前无 `truck_eta`，不能只靠 `0` 表示，
  还必须配 `has_truck_eta` 标志位

**快照规则（snapshot rule，最高优先级约束）**

`CriticBatch` 必须从 **pre-action DecisionContext snapshot（动作执行前决策快照）**
构造。完整顺序固定为：

1. `env_adapter` 在当前事件触发点冻结 `DecisionContext`
2. `candidate_builder` 从同一份 `DecisionContext` 构造 `CandidateOutput`
3. `ObservationTensorizer` 构造 `ObservationBatch`
4. `CriticBatchBuilder` 构造 `CriticBatch`
5. `policy.forward(observation_batch, action_mask, critic_batch)`
6. 采样 / greedy 得到动作
7. `env_adapter.step(action)`
8. `reward / done / next_state` 写入 `rollout_buffer.py`

不允许：

- 先 `env.step(action)`，再回头补构造 `CriticBatch`

否则 critic 会混入 post-action 信息。

`CriticBatchBuilder` 的公开接口建议固定为：

```python
CriticBatchBuilder.build(
    decision_context: DecisionContext,
    critic_tensor_schema_meta: CriticTensorSchemaMeta,
) -> CriticBatch
```

不建议再把 `runtime_state` / `coarse_plan` 作为
`DecisionContext` 之外的重复入参，以免调用方传入与快照不一致的对象。

**存储契约（storage contract）**

`rollout_buffer.py` 默认必须存“构造后的张量”（materialized tensors），
而不是依赖 PPO update（PPO 更新）时重新调用 builder 现算。

固定规则为：

- PPO update 路径存：
  `observation_batch_tensor / critic_batch_tensor / action_mask_tensor /
  resolved_action_indices / log_prob_old / value_old / reward / done /
  lstm_state_in / lstm_state_out / critic_schema_hash`
- 原始 `decision_context_debug_snapshot` 可以作为 debug / audit 可选保存，
  但不得作为默认训练更新的重建来源

原因：

- 若 rollout 只存原始对象，恢复训练时 builder 代码一变，旧 rollout 会被重新解释；
  同一条轨迹不再是原训练时的输入，PPO 更新结果会漂

**Recurrent PPO V1 序列训练约束**

为避免本项目的 LSTM 只在“采样时看起来接上了”，而在 PPO update（PPO 更新）
阶段又退化回“按独立 timestep（时间步）重算”的伪 recurrent 实现，Phase 7
必须把第一版 recurrent PPO（循环 PPO）训练约束单独冻结如下。

这里的单选结论再次固定为：

- `Phase 7 V1 = recurrent_ppo=true`（正式验收口径）
- `Phase 7 debug / pre-V1 = recurrent_ppo=false`（仅允许调试链路，不属于正式验收）
- 因此，下文所有 `sequence`（序列）/ `lstm_state`（循环隐藏状态）/
  `valid_timestep_mask`（有效时间步掩码）/ `sequence minibatch`（序列级小批量）
  约束，均不是“未来可选优化项”，而是 `Phase 7 V1` 的正式 contract（契约）
- 任何仍然使用“平铺 rollout -> 平铺 timestep batch -> 直接 PPO update”的实现，
  即使表面上保存了 `lstm_state_in / lstm_state_out`，本质上也只能归类为
  `pseudo recurrent PPO`（伪循环 PPO）或 `recurrent_ppo=false` 的 debug fallback，
  不能作为 `Phase 7 V1` 的正式训练产物生成路径

**Phase 7 V1 必须做**

1. `hidden state`（隐藏状态）按 `drone_id` 单独保存/恢复
   - rollout 收集阶段必须维护 `drone_id -> lstm_state` 的映射
   - 仅当该 UAV 自己成为本次 actor 时，才推进它自己的 hidden state
   - 这里的 `lstm_state` 语义固定为当前 `actor_drone_id`（当前决策无人机）的
     `recurrent core state`（循环核心隐藏状态），不是 `history_encoder`
     的局部编码缓存
   - PPO update 重算 `log_prob / value` 时，必须消费 rollout 中保存的
     `lstm_state_in`，不允许统一传 `None`
   - 若模型实现同时保留 `history_tokens`（显式全局历史摘要窗口）与
     `lstm_state`（隐式循环记忆），则必须满足：
     `history_encoder` 每个 timestep（时间步）从当前观测重新编码，
     `recurrent_core` 才负责 sequence 内连续 unroll；不允许把同一个
     recurrent module（循环模块）同时拿来做 `history_tokens` 编码和
     cross-step hidden state 递推
   - 对 `simplified TBPTT`（简化版截断反向传播）的训练侧语义必须进一步写死：
     在一个 sequence 内，`model.forward_sequence(...)` 必须从该 sequence
     首个真实 timestep 的 `lstm_state_in` 作为初始 recurrent state 开始连续
     unroll；不允许在 sequence 内每个 timestep 上分别取各自保存的
     `lstm_state_in` 独立做 `forward_step(...)`
   - 也就是说，`lstm_state_in` 在 PPO update 中的主作用是提供
     `sequence initial state`（序列起点隐藏状态），而不是把 sequence
     内每个 timestep 都变成“独立重置点”
   - 错误实现示意：
     对同一 sequence 做
     `for t in sequence: output_t = model.forward_step(obs_t, lstm_state_in_t)`
     这种写法虽然表面上“消费了 rollout 中保存的 recurrent state”，
     但本质上仍是伪 recurrent PPO，不符合本节冻结语义
   - 正确实现示意：
     `h0 = first_real_timestep.lstm_state_in`
     `outputs = model.forward_sequence(obs_seq, h0)`
     然后在 sequence 内依时间顺序连续递推 hidden state
   - 因此，Phase 7 V1 明确禁止“只有平铺 timestep tensor，再给每个 timestep
     单独喂一个保存态”的 update 伪实现；sequence 内必须保留真实的 recurrent
     state transition

2. `episode chunking`（按 episode 切块）
   - rollout update 路径必须先按 `done` 边界切分 episode
   - 不允许把跨 episode 的时间步拼进同一条 sequence（序列）
   - 若某个 episode 长度超过单次训练 sequence 长度上限，再在 episode 内继续切块
   - 但对本项目这种“多 UAV 轮流成为 actor”的单环境时间线，
     sequence continuity（序列连续性）不能仅按全局 timestep 顺序定义，
     而必须按以下复合边界冻结：
     `episode_id + actor_drone_id + recurrent_segment_id`
   - 也就是说，Phase 7 V1 的 recurrent sequence 不是“某个 episode 下按全局交错
     actor 时间线直接切出来的一段连续 timestep”，而是“同一 episode 内、
     同一 actor UAV、且属于同一个连续 recurrent 生命周期段”的局部决策子序列
   - 在该边界之内，再按固定 `seq_len` 做 chunk；不允许一条 sequence
     同时包含：
     - 两个不同 `actor_drone_id`
     - 两个不同 `recurrent_segment_id`
     - 或跨 episode 的 timestep
   - `recurrent_segment_id` 的定义固定为：
     同一 `actor_drone_id` 在同一 `episode_id` 内的第几个连续 recurrent
     生命周期段
   - `recurrent_segment_id` 在以下边界必须递增：
     - episode reset 后重新开始时
     - 该 UAV 的 hidden state 被显式清零 / 强制 reset 时
     - 该 UAV 因 hard failure、替换、重新激活等语义导致 recurrent 生命周期中断时
   - 即使 Phase 7 V1 当前实现里未必真的触发上述所有场景，也必须把该字段语义
     先冻结下来，避免后续实现把“同一 drone、同一 episode、但 hidden state
     已重置”的片段错误拼进一条 sequence

3. `sequence minibatch`（序列级小批量）
   - PPO update 的最小采样单元固定为 sequence，而不是“打散后的单 timestep”
   - 同一个 minibatch 内允许包含多条 sequence
   - 但每条 sequence 内的时间顺序必须保留，不能在 sequence 内再次洗牌
   - `recurrent_ppo=true` 时，训练侧配置语义必须冻结为：
     - `sequence_len`：单条 materialized sequence 的总长度
     - `burn_in_len`：sequence 前缀中的 burn-in 长度；Phase 7 V1 固定为 `0`
     - `train_len`：单条 sequence 中实际参与 PPO loss 的长度；V1 下固定满足
       `train_len = sequence_len - burn_in_len`
     - `sequence_minibatch_size`：每个 PPO minibatch 中的 sequence 条数
     - `target_minibatch_timesteps`：期望的有效训练 timestep 数，
       仅用于配置校验、日志统计与吞吐监控，不作为 sampler 主切分单位
   - 因此，当 `recurrent_ppo=true` 时，PPO update 的主采样单位固定是
     sequence，而不是 timestep；sampler 必须以 `sequence_minibatch_size`
     为准，不允许再把 `training.batch_size` 解释成“每个 minibatch 的 timestep 数”
   - 为避免普通 PPO 与 recurrent PPO 共享同名字段造成歧义，
     文档正式口径推荐将旧字段 `training.batch_size` 重命名为
     `target_minibatch_timesteps`
   - 若为了兼容旧配置暂时保留 `training.batch_size`：
     - 它在 `recurrent_ppo=true` 下只能被视为
       `target_minibatch_timesteps` 的过渡别名
     - 不得直接驱动 sampler 切分
     - 若 `batch_size` 与 `target_minibatch_timesteps` 同时存在且取值不一致，
       应视为配置错误，而不是任选其一静默继续
   - 配置校验关系建议固定为：
     - 理想满载情况下，
       `sequence_minibatch_size * train_len ~= target_minibatch_timesteps`
     - 若因尾段 padding / `valid_timestep_mask` 导致实际有效 timestep 低于
       `target_minibatch_timesteps`，这是允许的；但 sampler 的第一主语义仍然是
       “抽多少条 sequence”，而不是“硬凑多少个 timestep”

4. `sequence loss mask`（序列损失掩码）
   - 当一个 episode 尾段不足固定 `seq_len` 时，允许 padding（补位）
   - 但 policy loss / value loss / entropy / KL 统计必须只对 `valid_timestep_mask=True`
     的真实时间步生效
   - 若启用 `advantage normalization`（优势标准化），其 `mean/std`
     也必须只在 `valid_timestep_mask=True` 的真实 timestep 上计算；
     不允许把 padding timestep 一并计入优势归一化统计
   - 因此，`valid_timestep_mask` 的作用域不仅覆盖 PPO loss 本身，
     也覆盖 loss 前的优势标准化、熵/KL 聚合统计以及这些统计项的分母归一化
   - 不允许把 padding timestep 当作真实样本参与 PPO clip、value 回归或熵计算

5. `GAE / returns`（优势与回报目标）计算顺序固定
   - `GAE` 与 `returns` 必须先沿原始 rollout 的完整 episode 时间线计算，
     再做 `actor_drone_id + recurrent_segment_id` 维度上的 sequence
     materialization（序列物化）与 chunking（切块）
   - 不允许先把 trajectory 切成 sequence chunk，再在 chunk 内各自重新定义
     advantage / return 边界
   - 也就是说，`done` 边界首先服务于完整 episode 级的 bootstrapping /
     GAE 终止语义；sequence chunking 是其后的训练组织步骤，而不是反过来决定
     value target 的边界

6. `materialized recurrent contract`（物化循环状态契约）
   - rollout record 默认必须显式存：
     `lstm_state_in / lstm_state_out / episode_id / actor_drone_id /
     recurrent_segment_id / local_decision_index / local_chunk_idx /
     sequence_id / sequence_pos / valid_timestep_mask`
   - 其中 `episode_id / actor_drone_id / recurrent_segment_id /
     local_decision_index / local_chunk_idx / sequence_id / sequence_pos /
     valid_timestep_mask`
     可以作为显式字段存储，也可以在 `RolloutBatchView` 构造阶段从 transition
     顺序和 `done` 边界确定后一次性物化
   - `sequence_id` 不应被理解为“episode 内全局递增编号”，而应理解为一条
     materialized sequence chunk（已物化训练序列块）的唯一标识；
     其语义由以下复合键确定：
     `sequence_id = hash_or_tuple(episode_id, actor_drone_id, recurrent_segment_id, local_chunk_idx)`
   - 各字段语义固定为：
     - `episode_id`：当前训练 episode 编号
     - `actor_drone_id`：该条 transition / sequence 对应的 actor UAV 编号
     - `recurrent_segment_id`：同一 UAV 在同一 episode 内的连续 recurrent 生命周期段编号
     - `local_decision_index`：该 UAV 在当前 `episode_id + recurrent_segment_id`
       范围内的第几个真实 actor 决策步
     - `local_chunk_idx`：将该局部决策子序列按 `seq_len` 切块后得到的 chunk 编号
     - `sequence_pos`：当前 timestep 在所属 sequence chunk 内的位置
   - 但 PPO update 路径不得再退化回“只有平铺 timestep tensor，没有 sequence 边界语义”

**Phase 7 V1 可以不做**

- `burn-in`（预热段）可以先不做，固定为 `burn_in_len = 0`
- 更复杂的 `TBPTT`（Truncated Backpropagation Through Time，长序列截断反向传播）
  可以先不做
- 多 worker / 多 env 的 recurrent replay（循环经验重放）可以先不做
- 基于 value bootstrap 的交错序列采样、跨 env packed sequence、RNN state cache
  压缩等优化都不属于 Phase 7 V1 必要项

**文档必须预留**

- 当前 Phase 7 采用的是
  `simplified TBPTT`（简化版截断反向传播）口径：
  - `burn_in_len = 0`
  - `num_envs = 1`
  - sequence continuity 固定按
    `(episode_id, actor_drone_id, recurrent_segment_id)` 构造
  - sequence 内保留时间顺序，sequence 之间可做 minibatch 级打乱
  - `lstm_state_in` 只作为 sequence 首步的初始 recurrent state；
    sequence 内必须连续 unroll，不允许把每个 timestep 都当独立 reset 点
- 未来若升级到更强版 recurrent PPO，可以在不改变
  `ObservationBatch / CriticBatch / action_mask / resolved_action_indices /
  lstm_state_in` 基础契约的前提下，继续增加：
  - `burn_in_len > 0`
  - 更长 `seq_len`
  - 多 env / 多 worker recurrent rollout
  - 更标准的 TBPTT 调度

**工程落地评估（基于当前代码产物）**

结论需要拆成“文档正式口径”与“当前代码状态”两层表达，不能再混写：

- **文档正式口径**：`Phase 7 V1` 已明确固定为 `recurrent_ppo=true`
  的 `simplified recurrent PPO / simplified TBPTT` 路径；不需要额外新开一个
  Phase 来定义这套 contract（契约）
- **当前代码状态**：截至本轮代码产物，`Recurrent PPO`（循环 PPO）正式验收口径
  仍**未实现完成**；当前实现只能算
  `Phase 7 训练链路已接通 + recurrent 前置语义已部分补齐`，
  不能宣称“Phase 7 V1 已完成”

原因如下：

- 已具备 `num_envs = 1` 的单环境训练前提，当前配置已冻结该前提
- `rollout_buffer.py` 已经是 materialized tensor 路径，并且已显式存
  `lstm_state_in / lstm_state_out`
- `train_cmrappo.py` 已经具备按 `drone_id` 保存/恢复 hidden state 的实现基础
- 当前 PPO update 已能访问完整 transition 序列，因此在 update 前增加
  `episode chunking -> sequence batching -> valid_timestep_mask`
  的派生层，不会破坏现有
  `DecisionContext -> ObservationBatch / CriticBatch -> RolloutBuffer` 主链路
- 但当前代码仍未完成以下 `Phase 7 V1` 必需项：
  - 基于 `episode_id + actor_drone_id + recurrent_segment_id` 的 sequence
    materialization（序列物化）
  - `sequence_len / burn_in_len / train_len / sequence_minibatch_size /
    target_minibatch_timesteps` 的正式训练配置切分
  - 以 sequence 为主采样单位的 PPO update sampler（采样器）
  - `valid_timestep_mask` 作用域下的 loss / KL / entropy / advantage normalization
    统计
  - 明确排除“平铺 timestep batch 直接 update”的伪 recurrent 路径

但也必须明确：

- 当前代码若只做到“按 `drone_id` 恢复 hidden state + update 时重放
  `lstm_state_in`”，仍**不等于**已经完成本节定义的 Recurrent PPO V1
- 真正补齐 V1 还需要：
  - 在 PPO update 前生成 sequence 边界
  - 明确 `sequence_len / burn_in_len / train_len / sequence_minibatch_size /
    target_minibatch_timesteps` 的训练侧配置，并完成 `training.batch_size`
    向 recurrent PPO 新口径的迁移/废弃
  - 为尾段 padding 增加 `valid_timestep_mask`
  - 让 PPO 各项 loss 与统计只消费 mask 后的真实 timestep

因此，本节的推荐执行顺序固定为：

1. 先修正 `hidden state` 按 UAV 保存/恢复与 update 重放问题
2. 再实现 `episode chunking`
3. 再实现 `sequence minibatch + valid_timestep_mask`
4. 最后才考虑 `burn-in` 或更复杂 TBPTT

**`meta.json` 持久化规则**

训练 run 必须把 `CriticTensorSchemaV1` 写入 `MetaJson.critic_schema`，
并随 `meta.json` 一起持久化；不得退化成临时独立字段段或日志注释。

`critic_schema` 至少应记录：

- `name` / `schema_version` / `schema_hash`
- `max_global_orders / max_global_uavs / max_global_stations`
- `ordering_rules`
- `truncation_rules`
- `padding_rules`
- `normalization` 基准
- `snapshot_rule = pre_action_decision_context`
- `storage_mode = materialized_tensors_in_rollout_buffer`
- `causal_blacklist`

其中 `schema_hash` 的计算输入至少必须覆盖：

- 每类 token 的字段顺序
- `K_order / K_uav / K_station`
- `ordering_rules / truncation_rules / padding_rules`
- `normalization` 基准
- `snapshot_rule / storage_mode`

字段清单（如 `order_token_fields / uav_token_fields / station_token_fields`）
建议一并写入；若第一版实现压力较大，至少也要把排序、截断、padding、
归一化与 snapshot/storage 规则完整写入

**checkpoint 兼容规则**

加载 checkpoint 时，必须校验 `critic_schema_hash`。

- 若当前代码生成的 `critic_schema_hash` 与 checkpoint / `meta.json`
  中记录的不一致：
  - 禁止 `resume training`
  - 禁止复用 optimizer state
  - 若要继续训练，只能作为 new run / fine-tune 重新开始，并生成新的
    `model_version`

不允许只靠 shape 判断兼容；`shape` 一样而字段顺序或归一化规则改变时，
语义也可能完全不同

**因果可得信息约束**

`CriticBatch` 只能包含当前决策时刻已经存在、因果可得的信息；不允许把未来真实结果泄漏给 critic。
明确禁止：

- 未来 poisson 订单
- 未来 benchmark 动态订单的完整列表（若在当前时刻尚未 spawn）
- 未来真实排队结果
- 任何依赖 `env.step(action)` 之后状态演化才可得的字段

否则 critic 会退化为“离线 oracle（预言机）价值网络”，导致 advantage 对 actor
不可学习的信息过拟合。

**时间对齐规则（最高优先级约束）**

`ObservationBatch`、`CriticBatch`、`action_mask`、`resolved_action_lookup`
必须来自**同一个 `DecisionContext` 快照**。在事件驱动环境下，这条规则固定为：

- 该快照至少由以下量共同确定：
  `t_decision`（决策时刻）、`deciding_drone_id`、`trigger_type`、
  `trigger_station_id`、`coarse_plan.plan_version`
- `env_adapter.step(action)` 之前，必须完成该 transition 的
  `ObservationBatch / CriticBatch / action_mask / resolved_action_lookup`
  构造，并把它们绑定到当前 transition record；实现上可以先暂存、再在 step 返回后补齐
  `reward / done / next_state`
- 不允许在 `step()` 之后、reward settlement（奖励结算）之后、或 replan
  之后，重新构造当前 transition 的 `CriticBatch`
- 若 actor 采样时所见 `coarse_plan.plan_version = k`，则 critic 评估该 transition 时
  也必须仍基于 `plan_version = k` 的同一份 snapshot，而不是使用 step 后新生成的
  `plan_version = k + 1`
- 同理，若 `ObservationBatch` 对应的是动作执行前的状态 `s_t`，则
  `CriticBatch` 也必须对应同一时刻的 `s_t`；不得错位为 `s_{t+1}` 或
  “`s_t` 被 step/replan 变异后的版本”

这条时间对齐规则的工程落点固定为：`CriticBatchBuilder` 以
`DecisionContext` 快照为输入构造 `CriticBatch`，并由 `rollout_buffer.py`
先把该快照对应的 actor / critic 输入固化到当前 transition 记录，再在
`env_adapter.step(action)` 返回后补齐 `reward / done / next_state`；无论实现采用
“先暂存后补齐”还是“一次性 append”，都不得在 step 之后重新构造当前 transition 的
`CriticBatch`。

训练时必须保证：

- reward 与第 4.1.4 节主目标逐项对应（含 `T_res_timeout`）
- 软失败通过 `T_fallback` 进入小惩罚项
- reservation timeout 通过 `T_res_timeout` 进入小惩罚项
- 硬失败通过 `N_hard_fail` 进入大惩罚项（`hard_failure_penalty_sec=1200`）
- 不允许继续使用与主目标脱节的混杂代理奖励作为默认训练目标

训练建议分两阶段：

1. 固定 seed 的小规模 `poisson` 过拟合阶段
   - 目的：确认模型、奖励、动作空间、订单源接入没有定义错

2. 多 seed `poisson` 扩展阶段
   - 目的：逐步获得对泊松动态订单的泛化能力

订单源要求：

- `train_cmrappo.py` 运行时强制校验 `order_source_mode == poisson`
- 训练配置快照必须记录 `arrival_rate`、`seed`、deadline 窗口参数
- 写入 `meta.json` 时必须同时记录 benchmark 身份快照（当前建议使用 `orders.json` 路径/摘要 + `static_orders` / `dynamic_orders` 数量），并与泊松参数、随机种子一起构成订单源确定性描述
- 写入 `meta.json` 时还必须在 `MetaJson.critic_schema` 中记录本次 run 实际采用的
  `CriticTensorSchemaV1`（含 `schema_hash`、排序/截断/padding/归一化/存储规则），否则该训练 run
  不可严格复现
- 若用户误传 `benchmark` 或 `hybrid` 到训练主循环，应直接报错，避免训练分布漂移

训练输出必须包含：

- `policy.pt`
- `meta.json`
- `train_metrics.jsonl` 或等价训练日志
- 训练配置快照

验收标准：

- 训练前几轮不会直接发散或崩溃
- 能稳定保存权重和元数据
- 固定 benchmark 的 UAV-scope 回放指标出现提升趋势
- reward 提升时，主目标分解项（含 `T_res_timeout`）也能同步解释，而不是只表现为黑盒数值变化

### 4.7.1 本轮实际已落地的代码改动（如实记录）

本轮已在代码中实际补齐以下 Phase 7 相关模块与接口，不涉及 `greedy`、`market`
现有算法实现的复用：

1. **`DecisionContext` 已升级为 Phase 7 的完整 pre-action snapshot**
   - 修改文件：
     - `backend/training/env_adapter.py`
   - 实际改动：
     - 在 `DecisionContext` 中新增：
       - `t_decision`
       - `planner_snapshot`
       - `execution_snapshot`
     - 其中：
       - `planner_snapshot` 当前包含 `backlog_new_orders`、
         `fallback_count_in_window`、`hard_failure_count_in_window`、
         `route_drift_ratio`、`completed_backbone_count`、
         `expected_backbone_count`、`total_backbone_count`、
         `active_launch_stations`
       - `execution_snapshot` 当前包含：
         - `uav_eta_to_available`
         - `uav_dispatch_mode`
     - `RuntimeStateView` 保持为纯运行时真值快照，没有回填 planner / critic 派生信号

2. **正式补齐 Phase 7 契约层**
   - 修改文件：
     - `backend/training/contracts.py`
   - 实际新增契约：
     - `DecisionPlannerSnapshot`
     - `DecisionExecutionSnapshot`
     - `FactorizedActionMask`
     - `ResolvedActionIndices`
     - `ObservationBatch`
     - `TransitionSummary`
     - `CriticBatch`
     - `CriticNormalizationMeta`
     - `CriticTensorSchemaMeta`
   - 实际改动：
     - `MetaJson` 新增正式一级子契约 `critic_schema`
     - `CriticTensorSchemaMeta` 已实现 `schema_hash` 计算逻辑，哈希输入覆盖：
       - token 字段顺序
       - `K_order / K_uav / K_station`
       - 排序 / 截断 / padding 规则
       - normalization 基准
       - snapshot / storage 规则

3. **补齐 Phase 7 配置项，避免训练代码硬编码 critic schema**
   - 修改文件：
     - `backend/config/rh_alns_cmrappo.yaml`
   - 实际新增字段：
     - `policy.critic_schema_name`
     - `policy.critic_schema_version`
     - `policy.max_global_orders`
     - `policy.max_global_uavs`
     - `policy.max_global_stations`
     - `policy.queue_norm_cap`
     - `policy.energy_norm_strategy`

4. **已新增 `ObservationTensorizer` 并接到现有 `CandidateOutput`**
   - 新增文件：
     - `backend/training/observation_tensorizer.py`
   - 实际实现内容：
     - 将 `CandidateOutput.candidate_features` 编码为固定 shape 的：
       - `uav_self_token`
       - `order_tokens`
       - `recovery_tokens`
       - `infra_tokens`
       - `history_tokens`
     - 生成：
       - `padding_mask`
       - `recovery_padding_mask`
       - `history_padding_mask`
       - `FactorizedActionMask`
     - 已提供 `TransitionSummary` 构造逻辑，用于 rollout 后将真实 transition
       压缩写入历史缓冲区

5. **已新增 `CriticBatchBuilder` 并按同一份 `DecisionContext` 构造 critic 输入**
   - 新增文件：
     - `backend/training/critic_batch_builder.py`
   - 实际实现内容：
     - 提供：
       - `build_default_critic_tensor_schema_meta(...)`
       - `CriticBatchBuilder.build(decision_context, critic_tensor_schema_meta)`
     - 已按 Phase 7 当前口径物化：
       - `global_order_pool_tokens`
       - `global_uav_tokens`
       - `global_station_tokens`
       - `coarse_plan_summary_vec`
       - `global_system_summary_vec`
       - 对应 padding mask
     - 当前已实现 ordering / truncation / normalization / padding 的代码级落地

6. **已新增 materialized rollout buffer**
   - 新增文件：
     - `backend/training/rollout_buffer.py`
   - 实际实现内容：
     - 支持“先固化 pre-action 输入，再在 step 后补 reward/done”的两阶段接口：
       - `begin_transition(...)`
       - `finalize_transition(...)`
     - 默认存储的是 materialized tensors 对应对象，而不是事后重建
     - 已实现 `compute_gae(...)`

7. **已新增 Phase 7 模型骨架**
   - 新增文件：
     - `backend/training/model.py`
   - 实际实现内容：
     - 新增 `SharedPPOActorCritic`
     - 已实现：
       - actor forward
       - critic forward
       - factorized action sampling
       - action evaluation（供 PPO update 用）
     - 已修复当前训练主路径所需的 batch/device 基础语义：
       - 不再使用会把 `[B, F] / [B, N, F]` 再次 `unsqueeze` 的通用 batch 包装逻辑
       - 改为按字段 rank（维度阶数）显式规范：
         - 单样本：`[F] / [N, F] / [N, M, F]`
         - batch：`[B, F] / [B, N, F] / [B, N, M, F]`
       - `observation_batch / critic_batch / action_mask / lstm_state`
         在 `model.forward()` / `sample_action()` / `evaluate_actions()` 中
         会统一迁移到模型当前 device，避免后续在 `cuda` / `mps` /
         `cpu` 路径上出现 device mismatch
     - 当前模型是“可训练最小版”，不是文档中完整 cross-attention 设计的最终形态

8. **已新增训练主循环入口，并与前面 phases 的产物对接**
   - 新增文件：
     - `backend/training/train_cmrappo.py`
   - 实际实现内容：
     - 强制 `order_source_mode == poisson`
     - 直接复用既有链路：
       - `DecisionContext`
       - `env.build_candidate_output(...)`
       - `CandidateOutput.resolved_action_lookup`
       - `env.step(env_action)`
     - 已接入：
       - `ObservationTensorizer`
       - `CriticBatchBuilder`
       - `RolloutBuffer`
       - `SharedPPOActorCritic`
       - PPO update
       - `meta.json` 写盘
       - `policy.pt` checkpoint 写盘
       - 训练入口现在会在启动早期消费 `training.training_seed`，
         统一固定 `python random / numpy / torch` 的随机态；
         若设备 backend 可用，则同样对 `cuda` / `mps` 执行对应的 seed 固定，
         使 `training_seed` 成为真实训练行为的一部分，而不是只记录进元数据
       - `meta.json` 的持久化时序已进一步收紧：
         不再在训练开始前预写 `meta.json`，而是改为训练循环完成后，
         先固化最终 `policy.pt`，再构造 `TrainingRunMeta` / `meta payload`，
         再执行写盘前终检，最后才真正写出 `meta.json`
       - 写盘前终检已显式覆盖：
         - `policy.pt` / `training_config_snapshot.yaml` / `train_metrics.jsonl`
           存在性
         - `scene_id / scene_bundle_dir / model_version / trained_at`
           等运行时字段齐全性与一致性
         - `training_input` 中的 `order_source_mode / poisson_seed /
           training_seed / total_timesteps` 与本次训练实际输入的一致性
         - benchmark 身份快照与 `critic_schema_hash` 的一致性
       - 已修复 recurrent（循环状态）与设备选择的关键工程语义：
         - rollout 收集阶段改为按 `drone_id -> lstm_state` 保存/恢复 hidden state，
           不再让所有 UAV 共享单一 `lstm_state`
         - 仅当前 `deciding_drone_id` 对应的 hidden state 会在该 UAV 成为 actor
           时推进，符合文档中“按 UAV 单独维护”的冻结语义
         - `RolloutBuffer` 中已保存的 `lstm_state_in` 现在会在 PPO update 重算
           `log_prob / value` 时重新堆叠并回传 `model.forward()`，
           不再统一传 `None`
         - rollout 采样前向已包进 `torch.no_grad()`，并把保存到
           `lstm_state_by_drone / RolloutBuffer` 的 recurrent state
           统一做 `detach().cpu().clone()`，避免图泄漏与设备耦合
         - `device=auto` 现在按 `cuda -> mps -> cpu` 顺序解析，
           并在训练启动时输出当前实际使用设备的 debug 信息
       - 但必须明确：上述内容只说明“recurrent rollout 前置语义已部分接通”，
         **不等于** `Phase 7 V1 recurrent_ppo=true` 已完成；
         目前的 PPO update 仍是平铺 timestep tensor（时间步张量）重算，
         尚未进入文档正式要求的 sequence materialization / sequence minibatch
         训练口径

9. **已补 Phase 7 自测，并回归现有 Phase 5c / Phase 6**
   - 新增文件：
     - `backend/training/test_phase7_snapshot_and_tensorizers.py`
     - `backend/training/test_phase7_model_runtime.py`
   - 本轮已实际通过的命令：
     - `python -m compileall backend/training`
     - `python -m unittest backend.training.test_phase7_model_runtime`
     - `python -m unittest backend.training.test_phase7_snapshot_and_tensorizers`
     - `python -m unittest backend.training.test_phase6_integration backend.training.test_env_adapter_phase5c`
   - 本轮新增覆盖点：
     - `model.py` 单样本 / batch 前向 shape 语义
     - `model.py` 的 device 一致性（避免 CPU tensor 混入 `cuda/mps` 前向）
     - PPO update 重放 `lstm_state_in` 时的 batch 堆叠逻辑
     - batched recurrent state 传回 `SharedPPOActorCritic.forward()` 的形状正确性

### 4.7.2 本轮尚未完成或尚未验证的项（如实记录）

以下内容目前仍不能宣称“Phase 7 已完全完成”，需明确保留：

在这些未完成项之上，有一条总前提必须先明确：

- 文档正式口径已经固定为 `Phase 7 V1 = recurrent_ppo=true`
- 但当前代码仍未完成：
  - 按 `episode_id + actor_drone_id + recurrent_segment_id` 物化训练 sequence
  - `sequence_len / burn_in_len / train_len / sequence_minibatch_size /
    target_minibatch_timesteps` 的正式配置切分与消费
  - 以 sequence 为主采样单位的 PPO update
  - `valid_timestep_mask` 下的 loss / KL / entropy / advantage normalization
    统计
- 因此，当前实现只能视为
  `Phase 7 训练主链路已接通，但 Recurrent PPO V1 未达标`
- 在这一状态下，任何来自当前平铺 update 路径的 `policy.pt`
  都不应被宣称为 `Phase 7 V1` 正式验收权重

1. **尚未实际跑通训练**
   - 当前开发环境已安装 `torch`，且相关 Phase 7 / Phase 6 / Phase 5c 单测已实际通过
   - 但本轮仍没有真实执行 `train_cmrappo.py` 的完整训练回合
   - 也没有产出经过真实训练得到的 `policy.pt`

2. **`model.py` 当前是最小可训练实现，不等于文档中的最终结构**
   - 当前实现没有完整落地文档里描述的更强版 cross-attention 架构
   - 更准确地说，现状是：
     - 已有可工作的 actor/critic 骨架
     - 但仍属于第一版工程接线实现

3. **训练输出中的“已保存权重并出现收敛趋势”尚未被验证**
   - 本轮没有验证：
     - 训练前几轮是否稳定
     - PPO loss / KL / entropy 是否合理
     - benchmark 指标是否提升
   - 因此原文中关于训练效果的验收标准，目前仍只是目标，不应视为已达成

4. **本轮没有实现 Phase 8/9 的验证与 baseline 闭环**
   - 尚未新增：
     - `validate_benchmark.py`
     - `validate_stochastic.py`
     - `baselines/greedy_local_policy.py`
   - 因而目前还没有用真实训练权重与固定 benchmark 做离线对比

5. **当前 `train_cmrappo.py` / `model.py` 的可运行前提仍依赖额外训练环境**
   - 当前环境已满足最基本的 `torch` 依赖，可运行训练相关单测
   - 但完整训练仍依赖实际可用的训练设备与对应 `torch backend`
   - 训练入口目前已做到：
     - 导入路径闭合
     - 训练 seed 固定闭合
     - 元数据组装与写盘时序闭合
     - 写盘前终检闭合
     - 代码级链路闭合
   - 但尚未做到“当前开发环境下直接完成一次真实训练”

## 4.8 Phase 8：固定 benchmark 离线验证

目标：

- 先做确定性验证，不先做随机场景

建议新增文件：

- `backend/training/validate_benchmark.py`
- `backend/training/baselines/greedy_local_policy.py`

输入：

- `default_scene/entities.json`
- `default_scene/orders.json`
- `policy.pt`
- `meta.json`

验证方式：

- 固定 `static_orders + dynamic_orders`
- 关闭随机泊松流：`arrival_rate = 0`
- 多次重复跑同一组场景
- 与确定性 greedy baseline 对比
- PPO 与 baseline 的核心比较范围仅限无人机订单；`mode A` 背景订单只作为系统上下文统计保留

baseline 设计（本阶段正式采用）：

1. baseline 与 PPO 共用完全相同的：
   - `env_adapter`
   - `planner_bridge`
   - `candidate_builder`
   - `CoarsePlanView`
   - factorized action space 与 `action_mask`
   - 若 baseline 复用 Phase 7 的 token encoder（特征编码器），则也必须遵守
     `padding_mask / recovery_padding_mask / action_mask` 的同一分工；
     若 baseline 直接在 `CandidateOutput` 上做规则打分，则可以不消费
     `ObservationBatch`，但不得改变这些 mask 的契约语义
2. baseline 为确定性策略，不做训练，不引入额外随机性
3. 候选订单选择规则采用加权贪心分数：
   ```text
   score(order_i) =
     - w_priority * order_pre_score_i
     - w_eta      * eta_to_deliver_i
     - w_b_return * best_mode_b_return_score_i
     - w_risk     * miss_risk_i
     - w_overdue  * projected_overdue_cost_i
   ```
4. 模式选择规则：
   - 若存在满足安全时序与电量约束的 `mode C`，且其综合后续代价优于
     `best_mode_b_return_score_i`、同时 `ETA_margin >= greedy_c_margin_min_sec`，优先选 `C`
   - 否则选 `B`
   - 若无合法配送动作，才选 `WAIT`
5. 回收点选择规则：
   - 在合法候选中选择 `ETA_margin` 更大、后续服务拥挤代价更小、额外 detour 更低的点
6. `validate_benchmark.py` 必须支持：
   - `policy_source=ppo`
   - `policy_source=greedy_baseline`

建议至少统计这些指标：

- 无人机订单完成数
- 无人机订单准时率
- 无人机订单超时数
- UAV-scope 主目标值（`J`）及各分解项
- 无人机总里程
- 回收站平均等待
- 模式分布（mode B / C 采用比例）
- 每单平均响应时延（仅无人机订单）
- 无人机等待卡车时间（`T_wait`）
- 补能宿主排队总时间（`T_queue`，含 `station / depot`）
- miss 次数
- fallback 总时间（`T_fallback`）
- reservation timeout 次数（`T_res_timeout`）
- hard failure 次数（`N_hard_fail`）
- 超时订单累计惩罚（`T_overdue`）
- 强制移除订单数（`N_hard_overdue`）
- greedy 推理下的吞吐与稳定性指标

系统上下文统计区至少保留：

- `mode_a_background_order_count`
- `mode_a_background_completion_time_sum`
- `truck_total_mileage`
- `truck_background_order_completion_events`

这一阶段最重要的是：

- 可复现性
- 可解释性
- 公平对比（同一环境、同一 coarse plan、同一 action mask）

产物：

- `benchmark_report.json`
- `benchmark_summary.md`
- baseline 对比结果表
- 系统上下文统计区

验收标准：

- 同一权重、同一输入，多次运行结果一致或波动极小
- PPO 与 greedy baseline 在同一设置下可直接对比
- 至少有一项 UAV-scope 核心指标优于 greedy baseline，或策略行为明显更合理
- 至少能解释模式 C 回收成功率、miss 率与 hard failure 率

## 4.9 Phase 9：随机扩展离线验证

目标：

- 在固定 benchmark 通过后，再验证对泊松动态订单的泛化

建议新增文件：

- `backend/training/validate_stochastic.py`

验证方式：

- 保持 `default_scene` 的设施和地图不变
- 订单源模式限定为：
  - `poisson`
  - `hybrid`
- `validate_stochastic.py` 不允许 `benchmark` 模式
- 按多个随机 seed 评估

建议分档：

- 低强度：`arrival_rate = 0.2`
- 中强度：`arrival_rate = 0.4`
- 高强度：`arrival_rate = 0.6`

重点观察：

- 训练分布内表现是否稳定
- 是否出现明显退化
- 候选集是否经常截断
- action mask 是否失衡
- miss / fallback / hard failure 是否随流量上升而失控
- FIFO 队列长度与等待时间是否出现异常爆炸

产物：

- 随机验证报告
- 不同强度分档结果表
- seed 汇总统计

验收标准：

- 中强度场景下结果稳定
- 高强度下即使性能下降，也不出现明显策略崩坏

## 4.10 Phase 10：SUMO 调度行为回放验证（训练后）

目标：

- 把训练好的模型的调度行为导出到 SUMO，验证 mode C rendezvous、miss、fallback 在时间轴上是否可解释
- 与 Phase 4 的路径验证不同，这里验证的是策略行为，而不是路径本身

建议新增文件：

- `backend/training/export_sumo_replay.py`

职责：

- 把离线验证（Phase 8）产生的路线、事件、订单结果导出成 SUMO 可观察文件

这一阶段主要看：

- 调度行为是否出现明显不合理绕行或聚集
- 模式 C 的 rendezvous、miss 与 fallback 行为是否在时间轴上可解释
- 是否出现明显违反站点容量或空中失效后仍继续派单的错误

产物：

- 回放导出文件
- SUMO 运行说明

验收标准：

- `sumo-gui` 能正确加载
- 至少完成 1 个 benchmark 回放可视化

## 4.11 Phase 11：模型产物固化

每个训练版本建议固化为如下目录：

```text
backend/weights/rh_alns_cmrappo/<version>/
  policy.pt
  meta.json
  benchmark_report.json
  stochastic_report.json
  baseline_benchmark_report.json
  system_context_report.json
  train_metrics.jsonl
  sumo/
    poi.add.xml
    replay.*
```

`meta.json` 至少包含：

- 模型结构参数
- `MetaJson.critic_schema = CriticTensorSchemaV1`（至少含 `schema_version / schema_hash /
  ordering_rules / truncation_rules / padding_rules / normalization / snapshot_rule / storage_mode`）
- 候选集参数
- 上层规划参数（含 `CoarsePlanView` 接口版本）
- 奖励/惩罚参数（含 `lambda_res_timeout` 和 `hard_failure_penalty_sec`）
- 共享运行时参数快照
- 训练所用场景 ID
- 训练所用 benchmark 身份快照（当前建议记录 `orders.json` 路径/摘要、`static_orders` / `dynamic_orders` 数量；再结合泊松参数与随机种子确定订单源）
- 后续在线锁参数策略骨架
- 当前版本冻结的环境语义契约摘要（含 reservation timeout 机制版本）

说明：

- 虽然当前阶段还不接前后端
- 但 `meta.json` 必须在本阶段就做，因为它是下一阶段线上锁参数的基础

---

## 5. 当前最合适的实际开工顺序

建议按以下小顺序开工：

1. 先做 `backend/config/rh_alns_cmrappo.yaml`（含完整分层结构）
2. 再冻结第 4.1.3 节的环境语义契约（含 reservation timeout 机制）
3. 再做 `backend/training/scene_loader.py`（只负责静态场景资产装载）
4. 再做 `backend/training/order_source_adapter.py`（统一 `benchmark / poisson / hybrid` 三种模式）
5. 再做 `backend/training/export_sumo_truck_route.py`，用 SUMO 验证卡车路径和 ETA
6. 再做 `backend/training/env_adapter.py`（含 reservation timeout 状态与事件，ETA 直接使用 Phase 4 产物）
7. 再做 `backend/training/candidate_builder.py` 和 `planner_bridge.py`（固定 RH-ALNS 骨架，输出 `CoarsePlanView`）
8. 再做 `backend/training/train_cmrappo.py`（Shared PPO-Lite，`order_source_mode=poisson`，factorized action space，centralized_train_only critic）
9. 训练出第一版 `policy.pt + meta.json`
10. 做 `backend/training/validate_benchmark.py` 与 `backend/training/baselines/greedy_local_policy.py`（固定 benchmark，对比 PPO 与 greedy baseline）
11. 再做 `backend/training/validate_stochastic.py`
12. 最后做 `backend/training/export_sumo_replay.py`（调度行为回放）

原因：

- 订单源模式必须先固定，否则训练、benchmark、stochastic 三条链路会混用订单注入语义
- SUMO 路径验证（步骤 5）必须在 env_adapter 之前，因为 ETA 是 `action_mask` 的前置输入
- 先把路径正确性确认，再实现环境，避免训练时无法区分路径 bug 和模型 bug
- baseline 对比放在 benchmark 阶段最合适，因为它要求完全相同的环境与动作空间
- 随机泊松流验证和行为回放放在模型能工作之后，避免同时面对多类 bug

---

## 6. 本阶段完成判断标准

离线训练与离线验证阶段完成，至少要满足：

1. 能稳定产出 `policy.pt + meta.json`
2. 能在 `default_scene` 上完成固定 benchmark 回放（greedy inference）
3. 能输出 UAV-scope benchmark 指标报告（含 `T_res_timeout`、`T_overdue`、`N_hard_overdue`、reservation timeout 次数）
4. 能做至少一组随机泊松扩展验证
5. 能导出 SUMO 可视化文件并复核结果
6. 已明确哪些参数属于后续在线必须锁定
7. 模式 C、miss、fallback、reservation timeout、FIFO 与 hard failure 语义已在环境和验证中闭环
8. RH-ALNS 与 CMRAPPO 的职责边界在代码层面已明确分离（`planner_bridge.py` 与 `env_adapter.py` 各司其职）
9. `mode A` 背景订单已从 PPO 主指标中剥离，并单独输出系统上下文统计（订单数、完成时间、卡车里程）

---

## 7. 下一阶段接口预留

虽然本阶段不接现有前后端，但要提前保留以下接口能力：

1. `meta.json` 中保留参数锁定策略字段
2. 训练输出目录结构固定化
3. solver 后续可直接加载：
   - `policy.pt`
   - `meta.json`
4. benchmark / stochastic / SUMO 报告文件命名固定化

这样下一阶段做在线推理接入时，不需要回头重构训练产物格式。
