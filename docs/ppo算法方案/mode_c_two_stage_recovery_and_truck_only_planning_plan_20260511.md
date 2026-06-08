# Mode C 两阶段回收与重订单粗规划改造方案

## 背景

当前 `mode C` 的问题已经不是单纯的候选点数量问题。随着后续动态订单流引入超重订单，系统会出现新的冲突：

- 超重订单只能由卡车派送，卡车路径必须随动态订单滚动调整。
- `mode C` 需要卡车按时经过某个回收站点。
- Phase4 导出的卡车路径不再能被 PPO 后续流程当作整局不变的执行真值，只能作为 reset 时的初始路线、冷启动骨架和无动态冲突时的 fallback baseline。
- 如果 PPO 在 dispatch 时提前选择具体 recovery node，该节点很可能在 UAV 送达后已经不再是最新卡车路径上的合理选择。
- 如果卡车为了重订单重新规划路径，原先被 PPO 选中的 recovery node 会变成脆弱承诺，容易导致 revalidation fail、reservation timeout 或 fallback。

因此，后续改造目标不是继续让 PPO 更精细地选择 recovery node，而是把职责重新分层：

- PPO 只判断某个订单该走 `mode B` 还是 `mode C`。
- 具体 recovery node 在 UAV 送达后，根据最新卡车路径和真实 UAV 状态由规则选择。
- 送达后选中的 recovery node 才成为卡车路径的 hard reservation。
- 卡车粗规划在重订单与 hard reservation 之间做统一滚动规划。

## 目标机制

### PPO 动作空间

PPO 不再选择具体 `recovery_idx`。动作集合收敛为：

- `WAIT`
- `dispatch(order, B)`
- `dispatch(order, C)`

需要删除或停用原先 recovery action head。`mode C` 不再在 dispatch 时绑定 `recover_node_id`。

### PPO 观察特征

虽然 PPO 不再选择具体 recovery node，但 observation 仍需保留 `mode C` 摘要特征，让 PPO 判断当前订单是否适合走 C。

保留的订单级摘要特征：

- 当前订单是否存在任意合法 C 回收点
- 最佳 C 节点预计等待时间
- 最佳 C 节点 ETA margin
- 可行 C 节点数量
- 最佳 C 节点距离

这里不额外引入更复杂的 risk 特征。第一版保持特征边界清晰，降低策略输入变化带来的解释成本。

需要注意：dispatch 时的 C 摘要特征必须尽量使用与送达后选点一致的合法性与评分口径。否则 PPO 看到的“C 可行性”与执行时真实规则不一致，会继续产生学习噪声。

### 送达后 recovery node 选择

如果 dispatch 选择 `mode C`，UAV 先执行取送货链路。订单送达并完成 service 后，再扫描当前卡车 future backbone，选择实际 recovery node。

这里的 `future backbone` 必须来自当前 PlannerBridge / 环境中的最新卡车计划，而不是直接回读 Phase4 静态产物。Phase4 路径只能用于初始化 `_full_backbone_cache` 或在没有动态重排需求时提供 baseline；一旦 truck-only 订单或 reservation 触发 replan，后续 C 摘要、送达后选点、reservation ETA 都必须以动态更新后的 truck plan 为准。

合法条件：

- 节点必须是卡车未来会经过的 station/depot。
- UAV 从当前位置飞到该节点电量可达。
- `uav_arrival + rendezvous_execution_margin_sec <= truck_eta`。
- `0 <= truck_eta - uav_arrival <= rendezvous_max_wait_sec`。

合法节点排序：

```text
score = wait_time + 0.25 * uav_flight_time
```

取 `score` 最小的节点。该规则优先减少空等，同时避免选择等待很短但飞行距离过长、能量边界过紧的节点。

### Hard Reservation 边界

`mode C` 的承诺强度按状态分层：

- `C selected but not delivered`：不 lock。此时还没有决定实际 recovery node，卡车路径可以因重订单变化。
- `delivered and recovery node selected`：hard reservation。卡车不能随意删除该节点，ETA 只能在阈值内漂移。
- `waiting_for_truck`：strong hard lock。UAV 已在回收节点等待卡车，除非重订单严重超时，否则不应主动违约。

这个边界比 dispatch 时立即 lock 更合理，因为送达后 UAV 的真实位置、电量、卡车最新路径都已经确定。

## 重订单与卡车粗规划

### 重订单进入方式

动态订单流中超过 UAV 重载能力的订单进入 truck-only pool：

- 不进入 PPO 可授权订单集合。
- 不生成 UAV dispatch action。
- 由 PlannerBridge / 粗规划模块接管。

第一版动态订单流中，超重订单必须保持极低频率，并使用较长 deadline：

- 超重订单只作为“卡车路径会动态调整”的结构性扰动，不作为训练压力主来源。
- 每个 episode 中超重订单数量应远低于 UAV 可配送订单数量。
- 超重订单 deadline 窗口应显著长于 UAV 动态订单，避免一开始把 planner 推进高冲突状态。
- 后续压力测试也主要增加 UAV 可配送订单的到达强度、deadline 压力或空间分布压力，而不是显著提高超重订单比例。

这样做的目的，是先验证“动态 truck plan + hard reservation + post-delivery C 选点”的机制闭环，而不是把 `mode C` 学习、卡车重排和高压 truck-only VRPTW 同时混在一起。

### PlannerBridge 接口扩展

PlannerBridge 需要接收 reservation 作为约束输入。当前只依赖 runtime state 与 trigger context 不够。

同时，PlannerBridge 不能再被视为 Phase4 静态路径的轻量包装层。Phase4 产物只提供初始 planned stops / segments / backbone seed；PlannerBridge 需要拥有后续动态 truck plan 的生成权，并把每次重排后的 truck backbone、ETA map、reservation outcome 回写给环境侧使用。

建议新增输入结构：

```text
TruckReservationConstraint
  reservation_id
  drone_id
  node_id
  state: hard | strong_hard
  eta_ref
  max_eta_drift_sec
  issued_at
  related_order_id
```

PlannerBridge 每次 replan 时同时接收：

- Phase4 baseline 的剩余路径片段，作为可回退的初始解，不作为硬约束
- 当前 truck-only pending orders
- 当前未完成普通订单上下文
- 当前 hard/strong-hard reservation
- 当前卡车位置与剩余路径
- 站点覆盖下限参数

输出除了原有 `CoarsePlanView` 外，还需要能报告 reservation 处理结果：

```text
ReservationPlanOutcome
  reservation_id
  node_id
  old_eta
  new_eta
  eta_drift_sec
  status: kept | drifted | invalidated
  invalidate_cause
```

### 粗规划目标

重订单到来时，不是简单把订单插到 Phase4 当前路径里，也不是只在原始 Phase4 路径上做局部 patch，而是基于当前全部约束滚动重排卡车路径。Phase4 路径只作为初始可行解和站点覆盖参考，不能压过 truck-only deadline 与 hard reservation。

第一版粗规划不承担“高密度 truck-only 压力测试”的目标。超重订单数量少、deadline 长，因此 planner 应优先做到行为可解释、reservation 处理正确、路径动态更新一致。真正的系统压力仍主要来自 UAV 动态订单流。

规划目标按优先级组织：

1. truck-only 重订单尽可能不超时。
2. hard/strong-hard reservation 尽可能不被删除，ETA 漂移尽可能小。
3. 路径中仍至少经过配置要求的若干 station，用于维持未来 C 机会和车载 UAV 触发机会。
4. 路径总绕行与额外时间尽可能小。

站点覆盖是软目标。如果为了满足站点覆盖会导致重订单超时，站点覆盖应让步。

### 图论与时间窗建模

卡车粗规划不建议第一版就使用大量加权超参数。更合理的口径是把问题建模为小规模的动态时间窗路径问题：

- 图节点：
  - 当前卡车位置
  - truck-only order customer
  - locked reservation station/depot
  - 候选 coverage station
  - depot
- 图边权：
  - 卡车从节点 A 到节点 B 的行驶时间
- 节点服务时间：
  - truck-only order 使用与 UAV delivery 对齐的订单服务时长
  - station/depot 使用卡车交接/停靠时长
- 时间窗：
  - truck-only order：`arrival <= deadline`
  - hard reservation：`eta_ref - early_tol <= arrival <= eta_ref + drift_tol`
  - strong hard reservation：使用更窄时间窗，或在第一版中作为不可放松约束

这样，粗规划首先是在时间窗约束下寻找可行路径，而不是靠一组权重在不可解释的目标里折中。

### 字典序目标

候选路径不使用加权求和排序，而是使用字典序目标：

```text
route_key =
  (
    truck_order_timeout_count,
    strong_reservation_violation_count,
    hard_reservation_violation_count,
    total_truck_order_lateness_sec,
    total_reservation_eta_drift_sec,
    station_coverage_shortfall,
    total_route_time_sec,
  )
```

选择 `route_key` 最小的路径。

这个排序表达的是硬优先级，而不是可调权重：

1. 先减少 truck-only 订单超时数量。
2. 再减少 strong reservation 违约。
3. 再减少 hard reservation 违约。
4. 再最小化 truck-only 订单总迟到秒数。
5. 再最小化 reservation ETA 漂移。
6. 再补 station coverage。
7. 最后才最小化路径总时间。

这能显著减少 planner 超参数，也更容易解释每次重排为什么牺牲了某个目标。

### 第一版搜索算法

第一版建议使用 `Beam Search + 时间窗剪枝`，而不是完整 MILP 或复杂 ALNS。

搜索过程：

1. 必须访问节点集合：
   - truck-only customer
   - hard / strong-hard reservation node
   - depot
2. 从当前卡车位置开始扩展 partial route。
3. 每扩展一个节点，立即计算到达时间、服务完成时间、deadline violation、reservation drift。
4. 对明显不可接受的 partial route 做剪枝。
5. 每层只保留前 `beam_width` 条字典序最好的 partial route。
6. 完成必须节点路径后，再尝试插入 coverage station。

`beam_width` 是主要复杂度控制参数。相比多个评分权重，它的语义更简单：越大越接近全局搜索，越小越接近贪心。

当必须节点数量很小，例如 truck-only orders + reservation 总数不超过 8 到 10，也可以用 DFS 全排列 + 时间窗剪枝做精确枚举。后续节点规模变大再退回 Beam Search。

### Station coverage 后插入

coverage station 不应和 truck-only order / hard reservation 一起竞争主路径搜索。

第一版建议把 station coverage 作为后处理：

1. 先规划必须节点路径。
2. 在不增加 truck-only timeout、不破坏 hard/strong-hard reservation 时间窗的前提下，尝试插入 station。
3. 每次选择额外行驶时间最小的可行插入点。
4. 达到 `patrol_stations_per_loop` 或无可行插入点后停止。

这样 station coverage 不会干扰重订单和 hard lock 的主优先级。

### Reservation invalidation

如果插入或重排重订单不可避免导致某个 reservation 的 truck ETA 漂移超过阈值，则 planner 可以主动 invalidate 该 reservation。

边界：

- 对 `hard` reservation：允许 invalidate，但需要记录系统成本，并触发 UAV fallback 或重新找 C 点。
- 对 `strong_hard` reservation：只有在重订单严重超时风险下才允许 invalidate，且记录更高系统成本。

如果是系统/卡车为了救重订单主动 invalidate，不惩罚 PPO 当初选择 C。该 fallback 属于全局 planner 权衡。

实现上不建议用加权评分决定是否 invalidate，而应使用约束松弛：

1. 第一轮：所有 reservation 都按约束处理，尝试找可行路径。
2. 第二轮：允许 hard reservation 在 soft/hard drift 阈值内漂移。
3. 第三轮：允许 hard reservation invalidate。
4. 第四轮：只有当 truck-only 订单存在严重超时风险时，才允许 strong hard reservation invalidate。

每一轮内部仍使用同一个字典序 `route_key`。这样 invalidation 是可解释的约束放松结果，而不是某个权重刚好压过另一个权重。

## Fallback Cause 与奖励归因

当前 fallback 需要拆分原因。不能简单取消所有 fallback 惩罚，也不能把 planner 主动违约的成本归因给 PPO。

建议新增 fallback cause：

- `planner_invalidated_for_truck_order`
- `c_revalidation_failed`
- `rendezvous_wait_timeout`
- `energy_or_node_invalid`
- `no_post_delivery_c_node`
- `hard_failure_fallback`

实现上需要让 `_release_reservation` 接收 cause 参数，并把 cause 传递到 episode metrics / transition summary。

奖励归因规则：

- `planner_invalidated_for_truck_order`：不惩罚 PPO；系统层记录 reservation invalidation 成本。
- `c_revalidation_failed`：惩罚 PPO，因为 dispatch 时 C 摘要应反映该风险。
- `rendezvous_wait_timeout`：惩罚 PPO 或至少惩罚该 C 链路，因为选择了会超等的 C。
- `energy_or_node_invalid`：惩罚 PPO。
- `no_post_delivery_c_node`：中等惩罚，表示 dispatch 时选择 C 但送达后没有可行节点。
- `hard_failure_fallback`：按现有 hard failure 语义处理。

这能避免两个坏结果：

- planner 为重订单牺牲 C 时错误惩罚 PPO；
- PPO 滥选高风险 C 并把失败全部转嫁给 fallback。

## Mode C Reward 调整

### mode_c_attempt_bonus

`mode_c_attempt_bonus` 不应继续在 dispatch 选择 C 时立即发放。

新的时机：

- dispatch 选择 C：不发 attempt bonus。
- 送达后成功选择到合法 recovery node：发放 `mode_c_attempt_bonus`。
- UAV 到达 recovery node：发放 `rendezvous_arrive_bonus`。
- 卡车成功回收：发放 `rendezvous_bonus`。

这样奖励链路更贴合真实承诺：

- dispatch 时只是 C intent。
- 送达后选点成功才是真正进入 C recovery 链路。

### selected_rendezvous_margin_norm

删除 recovery head 后，原先 action 中的 `selected_rendezvous_margin_norm` 不再来自具体 selected recovery token。

决策：

- dispatch-time transition 中使用 `order_feature` 的 C 摘要值填入。
- 送达后实际选择 recovery node 时，记录真实 selected node 的 margin，进入 execution diagnostics。

这能保证 PPO 训练数据仍有 C 摘要信号，同时不再假装 dispatch 时已经选择了具体 recovery node。

## 需要分几次修改

建议分 7 次推进。每次只改一个语义层，避免重新陷入“策略、环境、reward、planner 同时变化导致无法归因”。

### 第 1 次：调整动态订单流

目标：

- 在订单生成配置中显式区分 UAV 可配送订单与 truck-only 超重订单。
- truck-only 超重订单默认极低频、长 deadline，只用于触发低强度动态卡车重排。
- stochastic / high pressure split 默认只提高 UAV 可配送订单压力，不显著提高 truck-only 比例。
- 先不改变 PPO 动作语义，确保新订单流统计和旧训练链路兼容。

主要影响模块：

- `config/rh_alns_cmrappo.yaml`
- order source / order manager
- training/eval order source 构造
- episode / eval metrics
- 相关测试

验收：

- 超重订单可按配置低频生成。
- 超重订单 deadline window 显著长于 UAV 订单。
- 超重订单不进入 PPO dispatch action。
- high pressure split 中 UAV 订单压力升高，但 truck-only 比例保持低位。

### 第 2 次：动作空间去 recovery head

目标：

- PPO 动作从 `WAIT / order / mode / recovery` 收敛到 `WAIT / order / mode`。
- 删除或停用 recovery action head。
- `DispatchAction(mode=C)` 不再携带 `recover_node_id`。

主要影响模块：

- `contracts.py`
- `candidate_builder.py`
- `model.py`
- `observation_tensorizer.py`
- `policy_inference.py`
- `train_cmrappo.py`
- 相关测试

验收：

- action mask 中仍能区分 B/C。
- C 的可行性摘要仍进入 order token。
- 旧 recovery token 不再参与动作采样、logprob、entropy。

### 第 3 次：送达后规则选 C recovery node

目标：

- dispatch 选 C 后不立即绑定 recovery node。
- `_process_delivery_service_event` 中，如果 commit mode 是 C，则扫描当前动态 truck plan 的 future backbone 并按规则选择 recovery node。
- 选点成功后建立 reservation，并进入 `return_to_rendezvous`。
- 选点失败时按 `no_post_delivery_c_node` fallback。

主要影响模块：

- `env_adapter.py`
- `candidate_builder.py`
- `rollout_buffer.py`
- `train_cmrappo.py`
- tests for mode C execution

验收：

- C dispatch 可以在送达后选择 recovery node。
- 送达后选择的 node 满足时间窗与能量约束。
- `mode_c_attempt_bonus` 只在送达后选点成功时发放。

### 第 4 次：Fallback cause 与奖励归因

目标：

- `_release_reservation(cause=...)`。
- fallback / reservation timeout / revalidation fail / planner invalidate 区分原因。
- reward 根据 cause 分配给 PPO 或系统指标。

主要影响模块：

- `env_adapter.py`
- `rollout_buffer.py`
- `train_cmrappo.py`
- metrics aggregation
- eval reports

验收：

- episode metrics 可统计每类 fallback cause。
- planner invalidate 不惩罚 PPO。
- 自然 C 失败仍惩罚 PPO。

### 第 5 次：PlannerBridge 接收 reservation 约束

目标：

- PlannerBridge 输入 hard/strong-hard reservation。
- PlannerBridge 从 Phase4 静态产物包装层升级为动态 truck plan 生成器；Phase4 只作为初始解 / baseline。
- CoarsePlanView 或旁路 outcome 能报告 reservation ETA drift / invalidation。
- 当前不一定实现完整重订单路径优化，先打通接口与诊断。

主要影响模块：

- `planner_bridge.py`
- `contracts.py`
- `env_adapter.py`
- tests for planner bridge

验收：

- replan 能看到 reservation constraints。
- reservation 被保留、漂移、失效都有结构化输出。

### 第 6 次：truck-only 重订单粗规划

目标：

- 超重动态订单进入 truck-only pool。
- 超重订单在第一版保持极低频、长 deadline，只用于触发低强度动态卡车重排。
- 粗规划按全部 truck-only orders + reservation constraints 重排卡车路径。
- 重排结果替代后续 PPO / env 使用的 truck backbone 和 ETA map，不能继续假设 Phase4 路线整局固定。
- 重订单 deadline 优先于 C reservation，但 reservation invalidation 有系统成本。
- 路径仍尽可能覆盖配置要求的 station 数量。

主要影响模块：

- order source / order manager
- `planner_bridge.py`
- truck route planning utilities
- `env_adapter.py`
- eval diagnostics

验收：

- 超重订单不进入 PPO action。
- 卡车能派送 truck-only 动态订单。
- reservation ETA drift 可控。
- 必要时 planner 可以 invalidate reservation 并触发 cause 正确的 fallback。

### 第 7 次：重新训练与评估

目标：

- 在新 MDP 下重新训练 PPO。
- 建立新评估 split：
  - 无重订单 baseline
  - 极低比例、长 deadline truck-only
  - C-sensitive + truck-only conflict
  - UAV 高压 stochastic，不显著提高 truck-only 比例

验收指标：

- truck-only timeout
- truck-only order count per episode
- truck-only deadline slack
- C selected ratio
- post-delivery C node selection success
- planner invalidated reservation count
- natural C fallback count
- PPO-attributed fallback count
- system-attributed fallback count
- benchmark / stochastic reward

## 粗规划模块规划

### 第一版粗规划：Beam Search 时间窗滚动重排

第一版不建议直接实现复杂 ALNS 或 MILP。建议先做可解释、可测试的 Beam Search 时间窗重排：

1. 读取 Phase4 baseline 的剩余路径片段，作为初始解种子和 coverage station 候选来源。
2. 收集当前 truck-only pending orders。
3. 收集 hard / strong-hard reservation nodes。
4. 生成必须访问节点集合：
   - truck-only customer nodes
   - hard reservation station/depot nodes
   - depot
5. 生成软访问节点集合：
   - 用于 station coverage 的候选 station
6. 以当前卡车位置为起点，对必须访问节点做 Beam Search。
7. 每次扩展 partial route 时做 ETA 仿真和时间窗检查。
8. 用字典序 `route_key` 保留每层前 `beam_width` 条候选路径。
9. 必须节点路径确定后，再做 station coverage 后插入。
10. 输出新的 truck backbone、ETA map、reservation outcome，并替代环境后续使用的 Phase4 静态 backbone。

因为第一版超重订单是极低频、长 deadline，Beam Search 的必须节点规模应保持很小。若某次 episode 中 truck-only 节点数量异常增大，应先通过订单流配置限制，而不是扩大 planner 复杂度来掩盖训练分布失控。

第一版主要参数只需要：

- `beam_width`
- `reservation_eta_drift_soft_threshold_sec`
- `reservation_eta_drift_hard_threshold_sec`
- `patrol_stations_per_loop`

避免引入大量 `w_xxx` 权重。

### 关键约束

重订单服务时间需要与无人机订单服务时间口径对齐，至少在第一版中使用统一的 order service duration。否则 truck-only deadline 评估会和 UAV delivery 评估不一致。

动态订单流约束：

- truck-only 订单比例必须显式配置，并默认极低。
- truck-only deadline window 必须显式配置，并默认长于 UAV 订单 deadline window。
- stochastic / high pressure split 增压时，默认只提高 UAV 可配送订单压力。
- 只有在基础机制稳定后，才单独新增 truck-only stress split。

Phase4 产物在新机制中的约束级别需要降级：

- `truck_execution_route.json` / SUMO route 只代表 reset 时初始卡车路线。
- 动态 replan 后，环境、PPO observation、mode C 摘要、送达后 recovery 选择都必须读取最新 truck plan。
- Phase4 baseline 可以用于无重订单、无 reservation 冲突时保持旧行为，也可以用于 station coverage 的候选顺序，但不能作为 hard route commitment。
- 如果前端或 SUMO 展示仍消费 Phase4 route 文件，需要改为消费运行时 truck path snapshot，否则展示路径会和训练/决策真值分叉。

### Station coverage

当前已有 `patrol_stations_per_loop` 一类参数。新规划中它不应再表示“每轮固定巡最近 K 个站”，而应表示：

- 当没有紧急 truck-only / hard reservation 冲突时，路径至少补充覆盖 K 个 station。
- 当 deadline 冲突时，coverage 可以降级。

实现上作为后插入步骤：

1. 对每个候选 station，枚举插入到主路径每一段之间。
2. 计算插入后的额外行驶时间。
3. 过滤掉会新增 truck-only timeout 或破坏 reservation 时间窗的插入。
4. 选择额外行驶时间最小的 station 插入。
5. 重复直到达到覆盖目标或没有可行插入点。

### Reservation drift threshold

建议新增：

```yaml
planner:
  reservation_eta_drift_soft_threshold_sec: 60
  reservation_eta_drift_hard_threshold_sec: 180
```

语义：

- 小于 soft：正常保留。
- soft 到 hard：保留但记录 drift。
- 超过 hard：允许 invalidate，除非是 `waiting_for_truck` strong hard lock。

### Planner 输出诊断

每次 replan 建议输出：

- `truck_only_order_count`
- `truck_only_arrival_rate_or_ratio`
- `truck_only_deadline_slack_mean_sec`
- `truck_only_timeout_risk_count`
- `reservation_count`
- `reservation_kept_count`
- `reservation_drifted_count`
- `reservation_invalidated_count`
- `station_coverage_count`
- `route_extra_time_sec`
- `planner_selected_route_key`
- `beam_width`
- `constraint_relaxation_level`

这些指标是后续判断“重订单优先是否导致 C 被过度牺牲”的关键。

## 最终判断

这套方案不需要推翻 PPO 框架，但会改变 MDP 与动作语义，因此需要重新训练。

最关键的结构变化是：

- recovery node 不再由 PPO 在 dispatch 时选择。
- `mode C` 从“一次性动作”变成“dispatch intent + post-delivery recovery selection”。
- 卡车粗规划从“提供未来可见站点”升级为“truck-only orders + hard reservation 的滚动约束规划”。
- fallback 奖励从单一惩罚升级为按 cause 归因。

这个拆分能显著降低 `mode C` 的学习噪声，也能让重订单动态插入时的责任边界更清楚。
