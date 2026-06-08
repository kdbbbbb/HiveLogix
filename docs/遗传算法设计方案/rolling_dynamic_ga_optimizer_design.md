# GA-MMCE 持续滚动动态重规划优化器设计方案

## 0. 设计目标

当前动态重规划不应被看成一次次彼此独立的 GA 求解，而应改造成一个**持续滚动的 GA 优化器**：

```text
动态订单到来
  -> 立即复用历史 incumbent / archive 产出可执行方案
  -> 在当前时间预算内继续从历史种群演化
  -> 把本轮优秀个体、候选评估、距离计算沉淀到缓存
  -> 仿真空闲 tick 继续小步优化 archive
  -> 下一次动态触发从已有搜索成果继续，而不是从零初始化
```

工程目标不是严格保证每次触发都得到数学意义上的全局最优，而是：

1. 每次触发先快速得到一个可执行且不差的 incumbent plan。
2. 在有限预算内利用历史种群和缓存减少重复计算。
3. 即使 `actual_generations = 0`，初始可选解里也保留 B/C 空地协同结构。
4. 后续空闲时间继续演化，让下一次重规划复用更好的搜索空间。

---

## 1. 当前动态重规划机制

### 1.1 外层触发链路

动态订单和重规划从前端/接口进入仿真后，当前主要链路是：

```text
frontend scheduled_dynamic_orders
  -> backend/api/routes/simulation_bp.py
  -> OrderManager 注入动态订单
  -> backend/environment/state/sim_engine.py::_try_auto_dispatch()
  -> backend/solver/decision_engine.py::execute_incremental()
  -> solver.should_replan_unfinished()
  -> solver.dispatch_replan_current_state()
  -> backend/solver/ga_mmce/solver.py::dispatch_replan_current_state()
  -> backend/solver/ga_mmce/dynamic_rescheduler.py::reschedule_on_event()
```

相关文件：

```text
backend/api/routes/simulation_bp.py
backend/environment/state/sim_engine.py
backend/solver/decision_engine.py
backend/solver/interfaces.py
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/dynamic_rescheduler.py
```

运行时路线执行和无人机动作回调相关：

```text
backend/solver/ga_mmce/runtime_adapter.py
backend/environment/state/entity_manager.py
```

### 1.2 GA-MMCE 内部动态重规划流程

当前 `dynamic_rescheduler.reschedule_on_event()` 的核心流程是：

```text
1. _advance_or_snapshot_state()
   取得 event_time 下的运行态快照

2. 读取 solver.last_best_individual / last_best_decode_result
   只拿上一轮 best 个体和 best plan

3. _classify_orders()
   将订单分为 completed / locked / pending / new

4. _select_reoptimization_window()
   从 pending + new 中选择 reoptimized_ids
   其余可继续沿用的未来段进入 frozen_future_ids

5. _build_dynamic_snapshot()
   写入 _ga_time / _ga_position / _ga_host_type / _ga_energy 等 GA 虚拟运行态字段

6. build_ga_context()
   根据当前资源快照生成 gene_pool、truck_drone_ids、depot_drone_ids、support_node_ids

7. build_warm_start_population()
   基于 previous_best 生成 warm starts

8. solver.solve(..., dispatch_type="dynamic_replan")
   进入正常 GA 初始化、评估、进化

9. 若 GA 不可行，尝试 fallback
   warm_start_repaired -> greedy_insertion -> previous_plan_unserved_new

10. _annotate_dynamic_summary() / _write_dynamic_replan_csv()
```

核心文件：

```text
backend/solver/ga_mmce/dynamic_rescheduler.py
backend/solver/ga_mmce/adapters.py
backend/solver/ga_mmce/population.py
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/decoder.py
backend/solver/ga_mmce/physical_evaluator.py
backend/solver/ga_mmce/config.py
```

### 1.3 当前动态配置

当前 `DYNAMIC_GA_CONFIG` 大致是：

```text
population_size = 30
max_generations = 60
min_generations = 15
early_stopping_patience = 10
time_budget_seconds = 3.0
warm_start_mutations = 10
reopt_window_size = 8
```

这说明设计上希望动态重规划小种群、短预算，但当前实际瓶颈在于：进入 generation loop 前，初始种群评估、B/C precheck、距离和候选评估已经消耗大量预算。

---

## 2. 当前存在的主要问题

当前动态重规划已经具备“新单 + 可重排旧单 + 当前运行态资源快照”的机制，但它在运行时仍表现出计算阻塞、方案不稳定、A 模式偏置、局部区域服务不足和路线震荡等问题。

### 2.1 动态订单触发同步大规模重规划，前端容易冻结

当前 `reschedule_on_event()` 会在新单到达后同步完成一整套流程：

```text
推进状态
  -> 订单分桶
  -> 构造动态快照
  -> 冻结资源
  -> 构造 warm start
  -> 调用 solver.solve()
  -> fallback
  -> 写日志
```

它是一个同步执行的完整重规划过程，不是轻量插入，也不是异步优化。因此前端卡住的本质链路是：

```text
动态订单到达
  -> 后端同步构造快照 + 资源冻结 + GA 初始化 + 个体 decode + 物理评估
  -> 仿真循环等待重规划返回
  -> 前端表现为整体状态停顿、冻结、长时间无响应
```

当订单数较多、B/C 候选较多、站点较多、无人机较多时，初始种群评估本身就会消耗大量时间。此时还没真正进入有效进化，时间预算可能已经耗尽。

### 2.2 当前不是局部修补，而是把较大订单池重新规划

动态重规划并不只处理新单，而是会把 `new_orders` 和部分 `pending_orders` 放入重优化窗口，同时还会把剩余 pending 作为 `frozen_future` 一起纳入 `planning_orders`。

当前 `planning_orders` 由以下两部分构造：

```text
reoptimized_ids + frozen_future_ids
```

这会导致一个问题：即使部分订单理论上只是“尾部保留”，它们仍然出现在规划问题中，需要参与个体结构修复、fixed tail 检查、decode 或可行性验证。也就是说，当前动态重规划的实际计算范围仍然偏大。

### 2.3 历史计算结果复用不足，每次重规划仍像半重新开始

当前动态入口主要复用：

```text
previous_best = last_best_individual
previous_plan = last_best_decode_result.plan
```

也就是上一轮最优个体和最优解码结果。

问题在于，上一轮 GA 中其实有大量有价值的信息没有被保存和复用，例如：

```text
上一轮 top-K 种群
B 模式较多的个体
C 模式较多的个体
等待时间较低的个体
延迟惩罚较低的个体
优秀 rendezvous 组合
历史 candidate 评估结果
历史距离/能耗计算结果
```

现在只复用 `previous_best`，信息量太少。新单一来，很多 B/C 结构、无人机协同结构、站点回收结构又要重新随机生成、重新评估，因此重复计算严重。

### 2.4 B/C 候选评估成本高，但缺少缓存

`PhysicalEvaluator` 中 A/B/C 模式评估会反复读取 `_ga_position`、`_ga_time`、`_ga_energy`、无人机 host 状态，并计算卡车路程、无人机飞行距离、能耗、等待时间、延迟惩罚等。

B 模式尤其复杂，需要判断：

```text
无人机是否在车上
launch / recover 是否合法
payload 是否超载
电量是否足够
卡车和无人机是否同步
等待是否超过阈值
```

动态重规划中经常会重复计算类似组合：

```text
订单 O1 + UAV-TEST-03 + launch STA-01 + recover STA-04
订单 O2 + UAV-TEST-05 + launch DEPOT + recover STA-02
卡车当前位置 -> launch 点
launch 点 -> recover 点
UAV launch -> customer -> recover
```

这些组合在相邻两次重规划中变化可能不大，但当前没有 `candidate cache` / `distance cache`，因此每次都会重新算，导致初始 population decode 很慢。

### 2.5 时间预算太紧时，GA 没有足够机会进化

当前 `population.py` 虽然已经支持 warm start、truck-only seed、OBL seed、B seed 和 balanced initialization，但如果时间预算很紧，GA 可能只来得及完成初始种群评估，没有时间进行足够的交叉、变异和选择。

这时 A 模式天然占优势：

```text
A 模式：卡车直送，通常最容易可行
B 模式：需要无人机在车上 + payload 可行 + 电量可行 + launch/recover 合法 + 同步等待可接受
C 模式：需要无人机在独立节点 + 回收点合法 + 电量可行
```

因此，在 `actual_generations = 0` 或 `actual_generations = 1` 的情况下，B/C 还没来得及通过变异和选择形成优势结构，最终结果很容易被 A 模式占据。

这不是简单的“B 不可行”，而是：

```text
B/C 评估更复杂、更难随机命中、更需要进化搜索；
但动态预算太短，导致搜索没展开。
```

### 2.6 卡车直送模式过多，空地协同潜力没有充分利用

如果明明可以使用 B 空地协同，但结果很多是卡车直送 A 模式，说明当前 GA 的动态搜索存在明显的模式利用不足问题。

可能原因包括：

```text
1. 初始种群中 A 个体更稳定、更容易可行。
2. B/C 候选虽然可行，但还没经过足够进化就被时间截断。
3. B/C 的等待惩罚、同步惩罚、回收绕行成本较高，导致局部评分不占优。
4. B seed 数量有限，不能覆盖足够多订单和 rendezvous 组合。
5. 没有保存上一轮 B/C-heavy 个体，导致每次重规划都重新寻找协同结构。
```

当前代码中虽然有 `make_combined_b_seed_individual()` 和 `make_single_b_seed_individual()`，但这些只是初始化层面的补救，并没有形成长期积累的 B/C 搜索经验。

### 2.7 每次重规划可能改变卡车后续路线，导致路线震荡

当前动态重规划是滚动重优化，理论上会根据当前订单池重新选择较优路径。但如果没有路线稳定性约束或局部连续性惩罚，每次新单到来后，卡车可能被新的全局方案牵引到另一个区域。

当前优化目标更关注：

```text
当前个体总成本最低
当前订单池整体可行
当前模式组合成本较低
```

但没有显式强调：

```text
卡车不要频繁改变方向
卡车优先清理当前区域附近订单
卡车不要因为一个远处订单立刻离开局部高密度区域
已经规划但未执行的短期路线尽量保持稳定
```

所以可能出现：

```text
卡车当前区域还有很多订单
但新一轮重规划后卡车离开当前区域
附近订单被延后
部分订单超时
卡车后面又绕回来，产生绕路
```

这属于典型的滚动优化路线震荡问题。

### 2.8 当前区域订单密度没有被显式建模

当前希望卡车/无人机在当前区域附近先完成一批订单，但 GA 的目标函数和候选评估更像是逐订单、逐候选地计算代价，而不是明确建模：

```text
当前区域订单密度
局部订单簇
卡车当前位置附近未完成订单
无人机从当前车载状态可覆盖的局部订单集合
区域内订单即将超时风险
```

因此，算法可能看到某个远处订单在局部评分上更优，或者某个 recover 点使当前个体总成本暂时更低，就让卡车离开当前区域。

这会导致两个后果：

```text
局部订单没有被连续服务；
卡车路径呈现跳跃式重规划，而不是区域内逐步清扫。
```

### 2.9 B 模式 launch/recover 决策可能把卡车牵引到不合适的回收点

B 模式评估中，卡车需要从当前位置到 launch 点，再到 recover 点；候选的最终 truck node 也会变成 recover node。

这意味着，如果 GA 选择了一个远离当前订单簇的 recover station，卡车就会被迫前往该回收点。即使当前区域还有很多订单，卡车也可能因为 B 模式回收约束被拉走。

隐含问题是：

```text
B 模式虽然实现了空地协同，
但如果 rendezvous 选择不好，
反而会诱导卡车绕路或离开当前服务区域。
```

这不是 B 模式本身的问题，而是 B 模式需要更强的 rendezvous 约束，例如：

```text
回收点不能明显偏离当前区域；
回收点应位于卡车未来局部路径上；
B 模式收益必须覆盖卡车绕行成本；
优先选择能服务当前订单簇的 launch/recover。
```

### 2.10 订单超时说明 deadline / lateness 在动态阶段优先级可能不够强

`PhysicalEvaluator` 中候选会计算 `lateness`，并在评分中加入 delay penalty。但如果动态运行中仍观察到订单超时，说明可能存在：

```text
1. deadline 惩罚权重不够强。
2. 即将超时订单没有被强制提前。
3. 当前区域订单没有按紧迫性聚类处理。
4. 重规划只看整体成本，牺牲了部分订单的时效性。
5. 卡车频繁被新方案牵引，导致原本快要服务的订单被推迟。
```

也就是说，超时不一定是路线不可行，而是动态目标函数没有足够强调：

```text
即将超时订单优先级
当前区域订单优先级
已接近服务完成的订单不要被反复推后
```

### 2.11 Runtime 层可以执行多段无人机任务，但频繁重规划会扰动任务队列

`runtime_adapter.py` 已经支持同一无人机多个 GA segment 排队执行，并且会把 GA plan 转换为仿真可执行的 waypoint 队列。

但重规划频繁时会出现风险：

```text
新 plan 应用后，尚未开始执行的无人机 route 可能被替换或清理；
正在执行的无人机会被跳过或锁定；
未飞但已经规划好的任务队列可能发生变化。
```

这会让系统表现为：

```text
计划不断变；
卡车路线不断变；
无人机任务队列不断重排；
局部订单迟迟没有稳定执行。
```

因此当前不仅是求解慢，还有一个计划稳定性不足的问题。

---

## 3. 根因总结：为什么慢，为什么容易退化为全 A

### 3.1 只复用 previous_best，历史搜索空间丢失

当前动态复用主要是：

```python
previous_best = copy.deepcopy(getattr(solver, "last_best_individual", None))
previous_decode = getattr(solver, "last_best_decode_result", None)
```

然后 `build_warm_start_population()` 围绕 `previous_best` 进行修补和 mutation。

这会丢掉上一轮 population 中大量有价值但不是 best 的结构，例如：

```text
低等待 B-heavy 个体
C-heavy 个体
B/C 多但总成本略高的个体
可行但有不同 rendezvous 结构的个体
延迟低但能耗略高的个体
```

动态订单一来，这些结构又要靠随机初始化或变异重新碰出来。

### 3.2 初始评估成本太高，generation loop 可能还没开始就超时

`solver.solve()` 当前顺序是：

```text
build_ga_context()
_prepare_distance_context()
_build_greedy_seed()
_build_b_precheck_and_seed_data()
_prepare_warm_start_seeds()
initialize_population()
_evaluate_population()
for generation in range(...):
    if timeout: break
```

因此预算检查主要发生在初始评估之后和 generation loop 内。动态预算很紧时会出现：

```text
initial population 已评估完成
time_budget_hit = True
actual_generations = 0
best = 初始种群中的某个个体
```

如果初始 best 偏 A，输出就会是全 A 或 A-heavy。

### 3.3 候选和距离重复计算严重

`physical_evaluator.py` 的 A/B/C 候选评估会反复计算：

```text
truck road distance
UAV straight distance
UAV energy
payload feasibility
launch/recover legality
truck wait / UAV wait
deadline lateness
station/recovery soft penalty
```

相邻两次动态重规划中，大量组合只是时间、位置轻微变化，但当前没有候选缓存和距离缓存，导致重复开销很高。

---

## 4. 目标架构

目标是把动态 GA 改成以下结构：

```text
RollingGAOptimizer
  |
  |-- Archive：保留历史优秀种群，而不是只保留 previous_best
  |
  |-- Fast Incumbent：动态触发后先快速评估少量历史种子，得到可执行方案
  |
  |-- Candidate Cache：缓存 A/B/C 候选评估
  |
  |-- Distance Cache：缓存 road distance / UAV distance / UAV energy
  |
  |-- Archive Warm Start Adapter：把历史个体适配到当前订单池和当前 gene_pool
  |
  |-- Anytime Slice：仿真空闲 tick 继续小步演化 archive
  |
  |-- Incremental Decode：中长期保存 prefix trace，只重评估变化后缀
```

推荐落地优先级：

```text
P0：保留并复用动态 archive
P1：增加 fast incumbent，避免 0 代全 A
P2：增加 distance cache
P3：增加 candidate cache
P4：增加 time-sliced anytime GA
P5：增加 incremental decode / prefix trace
```

---

## 5. 优化路线总原则：先稳住事件触发重规划，不直接改成持续滚动

当前 `GAMMCESolver.should_replan_unfinished()` 已经固定返回 `True`，新单到达后会进入 `dispatch_replan_current_state()`，本质上已经是“事件触发的滚动重优化”。现在的问题不是没有滚动，而是：

```text
1. 重规划同步执行，阻塞仿真循环。
2. 3 秒动态预算没有覆盖 precheck / warm start / 初始种群评估。
3. actual_generations 经常为 0，GA 没有真正进化。
4. previous_best-only 复用太弱，B/C 结构难以延续。
5. 每次可行 plan 都可能覆盖卡车后续路线，缺少稳定性门槛。
```

因此第一版不要直接引入后台持续滚动 GA。后台持续滚动需要额外解决线程安全、快照版本、计划过期、路线安全切换和执行锁等问题，改动面很大，而且可能把“偶发阻塞”变成“频繁抢占”。

推荐改造原则：

```text
先让每次事件触发重规划更轻、更稳、更会复用历史结果；
再考虑在仿真空闲 tick 做极小时间片的 archive 优化；
最后才考虑真正后台持续滚动优化器。
```

---

## 6. 第一阶段：让动态预算真正生效

### 6.1 目标

现有动态配置中 `time_budget_seconds = 3.0`，但 `solver.solve()` 的时间检查主要发生在 generation loop 内。当前顺序是：

```text
_prepare_distance_context()
_build_greedy_seed()
_build_b_precheck_and_seed_data()
_prepare_warm_start_seeds()
initialize_population()
_evaluate_population()
for generation:
    if timeout: break
```

因此动态预算经常在进入 generation loop 前已经耗尽。第一阶段的核心目标是把时间预算扩展到以下步骤：

```text
B/C precheck
warm start seed 评估
初始 population 构造
初始 population 评估
generation loop
fallback 评估
```

### 6.2 修改 solver.py：引入 solve deadline

在 `GAMMCESolver.solve()` 开始处，把当前 `started` 和 `_active_time_budget_seconds` 合成一个内部 deadline：

```text
self._active_deadline = started + time_budget_seconds
```

继续保留现有 `_timeout(started)`，但新增更直接的辅助方法：

```text
_remaining_budget_seconds() -> float | None
_has_dynamic_budget(min_remaining=0.0) -> bool
```

需要插入预算检查的位置：

```text
1. _build_b_precheck_and_seed_data() 每个 order 前后。
2. _prepare_warm_start_seeds() 每个 seed 前。
3. _evaluate_population() 每个 individual 前。
4. generation loop 内每个 child 评估前。
```

预算耗尽时不抛异常，返回当前已评估的最好结果；若没有任何可用评估结果，则走 fast incumbent / previous_plan fallback。

### 6.3 修改 B precheck：限制组合爆炸

当前 `_best_initial_mode_b()` 会遍历：

```text
order_id * truck_drone_ids * support_node_ids * support_node_ids
```

当站点和无人机较多时，这会吞掉动态预算。第一阶段建议新增动态场景专用限制：

```text
dynamic_b_precheck_max_orders = 8
dynamic_b_precheck_max_drones_per_order = 4
dynamic_b_precheck_max_launch_nodes = 4
dynamic_b_precheck_max_recover_nodes = 4
dynamic_b_precheck_budget_seconds = 0.8
```

节点筛选规则先保持简单，按与订单和卡车当前位置的距离排序：

```text
launch 候选：
  1. 卡车当前节点/附近站点
  2. 订单附近站点
  3. previous_best 中该订单使用过的 launch

recover 候选：
  1. 订单附近站点
  2. 卡车当前区域附近站点
  3. previous_best 中该订单使用过的 recover
```

如果预算不足，只做 A/C 或跳过 B 全组合，不再让 precheck 阻塞整次重规划。

### 6.4 修改初始种群评估：允许部分评估

当前 `_evaluate_population(population, state, context)` 会完整评估所有个体。动态模式下应改成：

```text
for individual in population:
    if timeout:
        mark remaining as big_m or leave unevaluated
        break
    evaluate individual
```

排序前只使用已经评估过的个体；未评估个体不能成为 best。这样至少可以保证：

```text
time_budget_seconds 不再被初始种群无限突破；
actual_generations = 0 时也能明确知道 initial_eval_count；
日志能区分“没进化”与“初始评估也没完成”。
```

### 6.5 第一阶段新增诊断字段

写入 `plan.summary`、`logs/ga_dynamic_replan.csv` 和 debug log：

```text
precheck_elapsed_seconds
b_precheck_eval_count
b_precheck_budget_hit
warm_start_eval_count
initial_population_eval_count
initial_population_budget_hit
actual_generations
time_budget_hit
```

---

## 7. 第二阶段：Fast Incumbent，避免 0 代进化直接退化为 A-heavy

### 7.1 目标

动态触发时先不赌完整 GA，而是先从少量历史种子中评估出一个可执行 incumbent：

```text
previous_best 修补种子
上一轮 last_population top seeds
B/C-heavy seeds
combined B seed
greedy seed
truck-only seed
```

即使后续 `solver.solve()` 被预算截断，也能用 fast incumbent 兜底。

### 7.2 最小实现不改 solver.solve() 签名

第一版可以在 `dynamic_rescheduler.reschedule_on_event()` 中完成：

```text
warm_starts = build_warm_start_population(...)
fast_plan = _fast_incumbent_plan(
    solver=solver,
    snapshot=snapshot,
    warm_starts=warm_starts,
    dynamic_config=dynamic_config,
    planning_orders=planning_orders,
)
ga_plan = solver.solve(...)
```

选择逻辑：

```text
1. ga_plan 不可行，fast_plan 可行：选 fast_plan。
2. ga_plan 可行，但 time_budget_hit=True 且 actual_generations=0：
   比较 ga_plan.cost_total 和 fast_plan.cost_total，选更优。
3. fast_plan 不可行：保持现有 warm_start_repaired -> greedy_insertion -> previous_plan fallback。
```

这一步不需要大改 `solver.solve()`，风险较低。

### 7.3 fast incumbent 预算

建议新增配置：

```text
fast_incumbent_enabled = True
fast_incumbent_eval_count = 8
fast_incumbent_budget_seconds = 0.3
```

注意这里的评估对象必须少。不要为了 fast incumbent 再完整评估 30 个个体，否则会把阻塞提前到 `reschedule_on_event()`。

### 7.4 可行性判断

不能只看 `plan.summary["feasible"] == total_orders`。应复用当前已有函数：

```text
_plan_is_feasible_for_orders(plan, planning_orders)
_unserved_order_ids(plan, planning_orders)
```

并额外记录：

```text
fast_incumbent_found
fast_incumbent_selected
fast_incumbent_A_count
fast_incumbent_B_count
fast_incumbent_C_count
```

---

## 8. 第三阶段：保存 last_population 和轻量 Archive

### 8.1 为什么先做轻量 Archive

当前只保存：

```text
last_best_individual
last_best_decode_result
```

这会丢掉上一轮中大量不是 best 但对动态重规划有价值的结构，例如 B-heavy、C-heavy、低等待、低 delay 个体。第一版不需要做复杂持久化 archive，只要先在 solver 生命周期内保留最近若干个体即可。

### 8.2 修改 solver.py

在 `GAMMCESolver.__init__()` 增加：

```text
self.last_population: list[Individual] = []
self.dynamic_archive: list[Individual] = []
```

在 `solve()` 中这些位置更新：

```text
1. 初始 population 部分评估后，保存已评估 top-K。
2. 每代生成后，保存当代 top-K。
3. solve 返回前，保存最终 top-K。
```

第一版建议不新增 `archive.py`，先在 `solver.py` 内部实现：

```text
_remember_dynamic_population(population, context, source)
_dynamic_archive_seeds()
```

稳定后再抽成 `backend/solver/ga_mmce/archive.py`。

### 8.3 Archive 保留策略

不能只按 fitness 保留，否则 A-heavy 个体会挤掉 B/C 结构。建议保留：

```text
fitness top N
B_count top N
C_count top N
B+C count top N
delay_cost 低 top N
waiting_cost 低 top N
最近若干个体
```

每轮最多保留：

```text
dynamic_archive_size = 60
dynamic_archive_update_top_k = 12
dynamic_archive_seed_count = 16
```

### 8.4 Archive 个体适配规则

历史个体进入当前动态规划前，必须适配：

```text
1. 删除 completed / locked 订单。
2. 删除不在当前 planning_orders 中的订单。
3. 保留仍存在订单的 sequence / assignment / rendezvous。
4. gene 不在当前 gene_pool 中时降级为 A。
5. rendezvous 节点不在当前 support_node_ids 中时重新生成。
6. 新订单按 deadline 或 nearest order 插入。
7. frozen_future_ids 必须继续调用 enforce_fixed_tail()。
8. validate_with_context() 失败则丢弃。
```

这些逻辑可以放在 `dynamic_rescheduler.py`，因为它已经持有：

```text
completed_ids
locked_ids
planning_orders
reoptimized_ids
frozen_future_ids
fixed_tail_gene_by_order
context.gene_pool
context.support_node_ids
```

### 8.5 与现有 warm start 合并

当前已有：

```text
build_warm_start_population(previous_best=...)
```

建议扩展为：

```text
previous_best_seeds = build_warm_start_population(...)
archive_seeds = build_archive_warm_starts(...)
warm_starts = merge_warm_starts(
    previous_best_seeds,
    archive_seeds,
    max_count=dynamic_warm_start_limit,
)
```

去重 key：

```text
(
  tuple(ind.sequence),
  tuple(ind.assignment),
  repr(ind.rendezvous),
)
```

---

## 9. 第四阶段：动态窗口选择加入局部性、紧迫性和路线稳定性

### 9.1 当前窗口选择问题

`_select_reoptimization_window()` 当前主要按上一轮 `previous_best.sequence` 中的 rank 和 deadline 排序。它能保留上一轮顺序，但没有显式考虑：

```text
订单与当前卡车位置距离
当前区域订单密度
订单是否即将超时
新单是否远离当前服务区域
短期路线是否已经接近执行
```

### 9.2 修改目标

`reopt_window_size = 8` 不应简单理解为“上一轮序列前 8 个”。它应该选择最值得在本轮重排的局部窗口：

```text
新单必须进入 reoptimized_ids；
即将超时订单优先进入；
当前卡车附近订单优先进入；
与当前局部订单簇相近的订单优先进入；
已经处在短期执行前缀中的订单尽量不被重排。
```

### 9.3 建议评分

为每个 pending order 计算动态窗口分数：

```text
score =
  w_prev_rank * normalized_previous_rank
  + w_deadline * urgency_score
  + w_local * distance_to_truck_or_local_cluster
  + w_density * local_density_bonus
  + w_stability * short_route_change_penalty
```

含义：

```text
urgency_score：deadline - event_time 越小越优先。
distance_to_truck_or_local_cluster：越近越优先。
local_density_bonus：周围订单越多越优先。
short_route_change_penalty：上一轮短期 route 前缀中的订单不轻易挪走。
```

第一版可以先实现简单版本：

```text
排序 key = (
  is_new_or_urgent,
  distance_to_current_truck_position,
  previous_rank,
  deadline - event_time,
  order_id,
)
```

### 9.4 frozen_future 不再无条件参与昂贵评估

当前 `planning_orders = reoptimized_ids + frozen_future_ids`，frozen future 仍会参与 decode。第一版先不改变这个结构，避免大改 decoder；但可以限制它们的扰动：

```text
1. fixed tail 继续强制保留。
2. frozen_future 只允许沿用 previous_best gene/rendezvous。
3. precheck 和 archive 适配只重点处理 reoptimized_ids。
4. 日志区分 reoptimized_eval_count 与 frozen_tail_eval_count。
```

中期再考虑真正的 “tail plan splice”，即 frozen future 不进入 GA 个体评估，只在最终 plan 阶段拼接。

---

## 10. 第五阶段：加入计划稳定性和接受门槛

### 10.1 为什么需要接受门槛

当前动态重规划只要返回可行 plan，`DecisionEngine._apply_plan(..., incremental=False)` 就会覆盖卡车路线。这样新 plan 即使只比旧 plan 略好，也可能导致：

```text
卡车方向频繁变化；
附近订单反复后移；
B recover 点把卡车拉到远处；
无人机未起飞队列被清理或替换；
```

因此动态重规划不能只看新 plan 是否可行，还要判断是否值得覆盖当前执行计划。

### 10.2 plan acceptance 规则

在 `dynamic_rescheduler.py` 中，`ga_plan / fast_plan / fallback_plan` 选出候选后，增加 `_accept_dynamic_plan()`：

```text
accept if:
  1. previous_plan 不存在；
  2. 新 plan 明显降低 cost_total；
  3. 新 plan 减少 unserved_order_ids；
  4. 新 plan 明显减少即将超时订单的 lateness；
  5. 新 plan 增加 B/C 协同且没有明显增加卡车绕路；
  6. 当前旧 plan 已经不可行或资源状态发生关键变化。

otherwise:
  保留 previous_plan 的短期前缀，只合入新单或走 fallback。
```

建议配置：

```text
dynamic_accept_min_improvement_ratio = 0.03
dynamic_accept_lateness_improvement_s = 30.0
dynamic_accept_max_extra_truck_distance_m = 500.0
```

### 10.3 稳定性惩罚

在 fitness 中加入动态稳定性 penalty，先放在 `decoder.py` 的 cost breakdown 或 `compute_fitness()` 前后的动态专用逻辑中：

```text
route_change_penalty：新 truck route 前几个节点与 previous_plan 不一致。
near_term_order_delay_penalty：上一轮即将执行的订单被推迟太多。
recover_deviation_penalty：B recover 点远离当前局部订单簇。
local_cluster_leave_penalty：当前区域仍有高密度订单时，卡车被拉远。
```

第一版可以不修改全局 `compute_fitness()`，而是在动态 plan 选择时计算稳定性调整分：

```text
adjusted_dynamic_score = cost_total + stability_penalty
```

这样对静态 GA 和现有解码逻辑影响更小。

### 10.4 B recover 局部约束

在 `_build_b_precheck_and_seed_data()` 和 random rendezvous 生成前做候选过滤：

```text
recover 点距离当前局部订单簇过远：降权或剔除；
recover 点导致 truck_dist_launch_to_recover 明显过大：降权或剔除；
B 模式节省的 UAV 服务收益小于 truck 绕行成本：不作为 B seed；
```

第一版先只影响 B seed 和 precheck，不禁止 GA 后续 mutation 产生这些 rendezvous，降低误杀可行解的风险。

---

## 11. 第六阶段：Distance Cache

### 11.1 当前实现注意点

`solver._prepare_distance_context()` 当前会清理 `greedy_helper._road_distance_memo`。这意味着跨轮动态重规划不能复用 road distance 结果。第一版可先改为：

```text
静态 full solve 可清理；
dynamic_replan 不清理或只按 bbox/scene_id 变化清理；
```

### 11.2 缓存挂载位置

优先挂在 solver 生命周期对象上：

```text
solver.greedy_helper._ga_distance_cache
```

`PhysicalEvaluator` 已持有 `greedy_helper`，可以通过它读写缓存。

### 11.3 缓存内容

先缓存低风险项：

```text
road distance: greedy._road_dist(pos_a, pos_b)
UAV straight distance: greedy._dist(pos_a, pos_b)
truck energy by distance
nearest support node lookup
```

第二步再缓存：

```text
UAV energy by drone/payload/segment
flight energy
```

### 11.4 key 设计

建议使用位置 bucket，避免浮点坐标无法命中：

```text
("road", scene_id, pos_bucket_a, pos_bucket_b)
("uav_dist", pos_bucket_a, pos_bucket_b)
("truck_energy", distance_bucket)
("uav_energy", drone_id_or_model, pos_bucket_a, pos_bucket_b, payload_bucket)
```

配置建议：

```text
distance_cache_enabled = True
distance_cache_size = 50000
distance_cache_position_bucket_m = 10.0
distance_cache_payload_bucket_kg = 0.1
```

---

## 12. 第七阶段：Candidate Cache

### 12.1 实施时机

Candidate cache 收益大，但风险也高，因为 A/B/C 可行性依赖 `_ga_time`、`_ga_position`、`_ga_energy`、host 状态和当前 config。建议在 Distance Cache 稳定后再做。

### 12.2 包装函数

在 `physical_evaluator.py` 中包装：

```text
evaluate_fixed_mode_a()
evaluate_fixed_mode_b()
evaluate_fixed_mode_c()
```

把原主体挪为：

```text
_evaluate_fixed_mode_a_uncached()
_evaluate_fixed_mode_b_uncached()
_evaluate_fixed_mode_c_uncached()
```

### 12.3 candidate key 必须包含状态签名

不能只用：

```text
(order_id, mode, drone_id, launch, recover)
```

必须包含：

```text
order_sig:
  payload_weight
  deadline bucket
  delivery_loc bucket

truck_sig:
  truck_id
  _ga_time bucket
  _ga_position bucket

drone_sig:
  drone_id
  _ga_time bucket
  _ga_position bucket
  _ga_energy bucket
  _ga_host_type
  _ga_host_node_id
  _ga_transport_truck_id
  _ga_waiting_station_id

config_sig:
  weight_energy
  weight_delay
  weight_waiting
  air_ground_mode_reward
  truck_wait_max_s
  soft_rendezvous_violation
```

### 12.4 复用策略

```text
1. precheck 和 seed 排序可以使用 candidate cache。
2. normal decode 可以使用严格 key cache。
3. 最终 best plan 建议做一次 strict re-evaluation，确认没有 stale cache。
4. 若发现 energy/host/runtime 状态不一致，清空当前动态轮的 candidate cache。
```

---

## 13. 第八阶段：Time-Sliced Archive 优化作为后续可选项

### 13.1 不作为第一版目标

真正持续滚动动态重规划不建议第一版做。原因：

```text
1. 当前阻塞主因是单次动态 solve 太重。
2. runtime 实体状态是可变对象，后台优化容易读到半更新状态。
3. anytime plan 如果自动应用，会和卡车/无人机执行队列冲突。
4. 没有 plan generation 和 safe switch point 时，旧 plan 可能覆盖新状态。
```

### 13.2 可接受的低风险版本

后续如果要做，只做单线程、小时间片、只更新 archive 的版本：

```text
每个仿真 tick：
  如果没有新单待调度；
  如果没有 dispatch 正在应用；
  如果 solver 是 ga_mmce；
  clone 当前 state；
  improve_archive_slice(snapshot, budget=0.03s ~ 0.05s)；
  只更新 dynamic_archive，不应用 plan。
```

### 13.3 新增方法

```text
GAMMCESolver.improve_archive_slice(state, time_budget_seconds=0.05)
```

职责：

```text
1. build dynamic context。
2. 从 archive 构造小种群。
3. 跑极少量 generation 或直到预算耗尽。
4. 只写 solver.dynamic_archive。
5. 不调用 DecisionEngine._apply_plan()。
```

---

## 14. 暂不建议第一版实施的内容

以下设计可以保留为远期方向，但不建议放入当前第一轮改造：

```text
1. 后台线程式持续滚动优化器。
2. 自动把 anytime plan 覆盖 runtime route。
3. Incremental Decode / Prefix Trace。
4. frozen_future 完全从 GA decode 中剥离并做 tail plan splice。
5. 大规模重写 population.py 的初始化框架。
```

原因：

```text
这些改动都涉及 decoder、runtime_adapter、DecisionEngine 和实体运行态的一致性；
在当前已经存在同步阻塞和路线震荡时，先做这些会扩大 bug 面；
收益也依赖前面预算守卫、archive、稳定性门槛先落地。
```

---

## 15. 配置新增建议

在 `backend/solver/ga_mmce/config.py` 中新增字段时，应保持默认保守，避免影响静态 GA。

```text
# Dynamic budget guard
dynamic_budget_guard_enabled = True
dynamic_b_precheck_budget_seconds = 0.8
dynamic_initial_eval_budget_seconds = 1.2
dynamic_warm_start_eval_budget_seconds = 0.4
dynamic_b_precheck_max_drones_per_order = 4
dynamic_b_precheck_max_launch_nodes = 4
dynamic_b_precheck_max_recover_nodes = 4

# Fast incumbent
fast_incumbent_enabled = True
fast_incumbent_eval_count = 8
fast_incumbent_budget_seconds = 0.3

# Lightweight archive
dynamic_archive_enabled = True
dynamic_archive_size = 60
dynamic_archive_seed_count = 16
dynamic_archive_update_top_k = 12
dynamic_warm_start_limit = 24

# Window and stability
dynamic_local_window_enabled = True
dynamic_accept_min_improvement_ratio = 0.03
dynamic_accept_lateness_improvement_s = 30.0
dynamic_accept_max_extra_truck_distance_m = 500.0
dynamic_stability_penalty_weight = 1.0
dynamic_recover_deviation_penalty_weight = 1.0

# Distance cache
distance_cache_enabled = True
distance_cache_size = 50000
distance_cache_position_bucket_m = 10.0
distance_cache_payload_bucket_kg = 0.1

# Candidate cache: later phase
candidate_cache_enabled = False
candidate_cache_size = 20000
candidate_cache_time_bucket_s = 10.0
candidate_cache_energy_bucket_wh = 10.0
final_strict_reevaluation = True

# Anytime archive slice: later phase
anytime_archive_slice_enabled = False
anytime_slice_seconds = 0.05
```

注意：第一版不要增大 `population_size`。当前问题不是种群太小，而是 30 个个体都可能评估不完。

---

## 16. 文件级修改清单

### 16.1 第一阶段：预算守卫

修改：

```text
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/config.py
backend/solver/ga_mmce/dynamic_rescheduler.py
```

重点：

```text
1. precheck / warm start / initial population 都检查 budget。
2. B precheck 限制候选数量。
3. summary 增加 initial_eval_count、precheck_budget_hit。
```

### 16.2 第二阶段：Fast Incumbent

修改：

```text
backend/solver/ga_mmce/dynamic_rescheduler.py
backend/solver/ga_mmce/config.py
```

重点：

```text
1. solver.solve() 前评估少量 warm starts。
2. ga_plan 0 代或不可行时可选择 fast_plan。
3. 不改 runtime_adapter。
```

### 16.3 第三阶段：轻量 Archive

修改：

```text
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/dynamic_rescheduler.py
backend/solver/ga_mmce/config.py
```

可选新增：

```text
backend/solver/ga_mmce/archive.py
```

建议先不新增文件，等逻辑稳定后再抽出。

### 16.4 第四阶段：窗口与稳定性

修改：

```text
backend/solver/ga_mmce/dynamic_rescheduler.py
backend/solver/ga_mmce/decoder.py 可选
backend/solver/ga_mmce/physical_evaluator.py 可选
```

重点：

```text
1. _select_reoptimization_window() 加局部性和紧迫性。
2. final_plan 选择前加 acceptance gate。
3. B precheck 的 launch/recover 候选按局部性筛选。
```

### 16.5 第五阶段：Distance Cache

修改：

```text
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/physical_evaluator.py
backend/solver/greedy_mmce_bi.py 或 greedy helper 相关距离函数
```

重点：

```text
1. dynamic_replan 不要每次清空 road distance memo。
2. 增加跨轮 distance cache。
3. 增加 hit/miss 诊断。
```

### 16.6 后续阶段

```text
Candidate Cache：physical_evaluator.py
Anytime Archive Slice：solver.py + sim_engine.py/decision_engine.py
Incremental Decode：decoder.py + solver.py，远期再做
```

---

## 17. 诊断字段建议

动态重规划必须先能被观测，否则很难判断优化是否有效。建议记录：

```text
event_time
pending_count
reoptimized_order_count
frozen_future_order_count
warm_start_count

precheck_elapsed_seconds
b_precheck_eval_count
b_precheck_budget_hit
warm_start_eval_count
initial_population_eval_count
initial_population_budget_hit

actual_generations
elapsed_seconds
time_budget_hit

fast_incumbent_enabled
fast_incumbent_found
fast_incumbent_selected
fast_incumbent_A_count
fast_incumbent_B_count
fast_incumbent_C_count

archive_enabled
archive_size_before
archive_size_after
archive_seed_count
archive_seed_accepted_count

plan_acceptance_decision
plan_acceptance_reason
stability_penalty
route_prefix_changed_count
recover_deviation_penalty

distance_cache_hit
distance_cache_miss
candidate_cache_hit
candidate_cache_miss

final_A_count
final_B_count
final_C_count
unserved_order_ids
```

写入位置：

```text
plan.summary
logs/ga_dynamic_replan.csv
backend/solver/ga_mmce_debug_log
```

---

## 18. 验证计划

### 18.1 脚本级回归

继续使用：

```text
python backend/scripts/test_ga_dynamic_rescheduler.py
```

新增断言方向：

```text
1. 动态重规划返回仍可行。
2. time_budget_seconds 不再被 initial population 大幅突破。
3. initial_population_eval_count <= population_size。
4. actual_generations = 0 时 fast_incumbent 参与选择。
5. warm_start / archive seed 不破坏 fixed_tail。
```

### 18.2 日志级验证

重点观察：

```text
elapsed_seconds 是否从 20s+ 降到接近动态预算。
actual_generations 是否从长期 0 变成偶尔能进入进化。
b_precheck_budget_hit 是否可控。
initial_population_eval_count 是否被预算限制。
final_B_count / final_C_count 是否不再长期接近 0。
plan_acceptance_reason 是否能解释路线为何被覆盖或保留。
```

### 18.3 行为级验证

在仿真中观察：

```text
1. 新单到达时前端冻结时间明显缩短。
2. 卡车不会因为单个远处订单频繁改变方向。
3. 当前区域附近订单更倾向于连续完成。
4. B recover 点不再频繁把卡车拉离当前订单簇。
5. 无人机 route queue 不被无意义清理。
```

### 18.4 回归风险

重点防：

```text
1. 预算中断后没有任何 evaluated individual，导致 best 为空。
2. archive 个体引用旧 gene_pool 或旧 rendezvous 节点。
3. fixed tail 被 archive 或 mutation 破坏。
4. acceptance gate 过严，导致新单长期不被接收。
5. B precheck 过滤过严，误杀本来可行的 B 协同。
6. distance cache 因 scene/bbox 变化复用错误路线距离。
7. candidate cache 用旧 host/energy 状态得出错误可行性。
```

---

## 19. 推荐实施顺序

### Step 1：预算守卫和 B precheck 限流

目标：

```text
动态重规划 elapsed_seconds 接近 time_budget_seconds；
不再出现 3 秒预算却阻塞 20 秒以上。
```

### Step 2：Fast Incumbent

目标：

```text
actual_generations = 0 时仍有历史可行方案兜底；
0 代不再天然等价于 A-heavy。
```

### Step 3：last_population / 轻量 Archive

目标：

```text
动态重规划从 previous_best-only 变成 previous_best + 历史 top-K + B/C-heavy seeds。
```

### Step 4：局部窗口和计划接受门槛

目标：

```text
减少路线震荡；
提升当前区域订单连续服务；
避免 B recover 把卡车拉到不合适区域。
```

### Step 5：Distance Cache

目标：

```text
减少重复距离和能耗计算；
提高动态预算内可评估个体数量。
```

### Step 6：Candidate Cache

目标：

```text
减少 A/B/C 候选重复评估；
但必须等状态签名和 strict reevaluation 机制稳定后再做。
```

### Step 7：Time-Sliced Archive Slice

目标：

```text
在无新单时小步优化 archive；
不直接应用 plan，不碰 runtime route。
```

---

## 20. 最小可落地版本

如果只做一版最小但有效的改造，建议范围是：

```text
1. solver.py：预算守卫，B precheck 限流，初始 population 部分评估。
2. dynamic_rescheduler.py：fast incumbent，final_plan 选择逻辑增强。
3. solver.py：保存 last_population，给 dynamic_rescheduler 提供 archive-like seeds。
4. dynamic_rescheduler.py：archive seeds 适配当前 planning_orders/gene_pool/fixed_tail。
5. config.py：增加预算、fast incumbent、轻量 archive 配置。
6. 日志：增加 precheck / initial eval / fast incumbent / archive 诊断。
```

这一版解决最核心问题：

```text
动态重规划不再长期突破预算；
0 代进化时仍有可解释的 incumbent；
历史 B/C 协同结构能跨动态轮次复用；
路线覆盖开始有接受门槛；
后续再做 cache 和 anytime 时有诊断基础。
```
