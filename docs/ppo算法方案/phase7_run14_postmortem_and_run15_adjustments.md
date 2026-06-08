# Phase 7 Run14 Postmortem And Run15 Adjustments 2026-05-08

## 本次目标

Run15 不再继续以 YAML 微调为主，而是先把 `mode C` 的问题拆清楚，再按单变量顺序修复。  
核心原则只有两条：

1. 先增加诊断，再修改行为
2. 一次只改一类结构问题，避免因果关系重新混乱

本轮不追求“一次性彻底修好 `mode C`”，而是追求：

- 明确失败发生在哪一段
- 用最小改动验证真正有效的结构修复

## Run14 结论摘要

基于当前 `run14` 的非 `benchmark` 结果，可归纳为：

- `low` 档基本不受影响，仍然完全不使用 `mode C`
- `medium` 档有小幅改善，`mode C` 已从“完全不用”变成“少量有效使用”
- `high` 档仍未稳定，`mode C` 使用增多，但 fallback / reservation timeout 仍存在
- `post_delivery_revalidation_fail_count = 0` 说明 split margin 已修复“送达后立即重验失败”这一类问题
- 但 reservation timeout 仍在，说明失败已迁移到执行后半段

因此，Run15 的重点不应再是继续调 `safe_margin`，而应是修正 `mode C` 的结构化决策链路。

## 当前代码层面的关键判断

### 1. `mode C` 评分语义确实不统一，但两个阶段不能强行写成同一个公式

当前存在两层排序：

- `recovery_pool_selector.py`
  只能看到 coarse plan 级别信息，使用 `proxy_rendezvous_margin`
- `candidate_builder.py`
  已经知道具体 UAV 状态，使用真实 `rendezvous_margin`

这意味着：

- 两层确实应该优化同一类语义目标
- 但不能要求它们用完全相同的 score 公式

更准确的目标应是：

- pool selector 使用 `proxy_mode_c_score`
- candidate builder 使用 `exact_mode_c_score`
- 两者语义对齐，但输入不同

### 2. `queue_time_est` / `swap_time` 在 mode C 路径上是 mode B 逻辑的错误复用

`_predicted_queue_time_est()` 计算的是"站点当前排队无人机数 × 换电服务时长"，`swap_time` 是站点的换电服务时长。两者反映的都是**充换电站的排队/服务时间**。

这对 mode B 是正确的：mode B 的 UAV 飞到站点后要排队充换电，queue 长短直接影响总耗时。

但 mode C 的 UAV 到达回收节点后直接进入 `WAITING_FOR_TRUCK`，等卡车来上车充电，**完全不经过站点队列，也不使用换电服务**。这两个量对 mode C 没有物理意义。

当前错误复用出现在以下五处：

- `recovery_pool_selector.py` 排序键第4位：`queue_time_est`
  → 影响哪些节点能进入 recovery pool
- `candidate_builder._build_mode_c_candidates` 排序键第1、2位：`queue_time_est > threshold` 和 `queue_time_est`
  → 影响 recovery candidates 的排列顺序，进而影响 `best_mode_c_*` 摘要
- `OrderFeatures.best_mode_c_queue_time_est`（最优mode C候选的排队时间摘要特征）
  → 暴露给 policy，policy 在用一个对 mode C 无意义的信号做决策
- `observation_tensorizer.py` recovery token 第10维：`predicted_queue_time_est_norm`
  → 直接喂给 recovery head，就算改掉排序和摘要，policy 仍在每次选 recovery slot 时消费这个无意义信号
- `observation_tensorizer.py` recovery token 第11维：`service_time_norm`（来自 `node_state.swap_time`）
  → 同上，换电服务时长对 mode C rendezvous 没有物理意义

另外，`_acquire_reservation()` 中的 reservation 过期窗计算（`env_adapter.py:1923`）：

```python
tau_res = reservation_alpha * truck_eta_to_recovery
        + reservation_gamma * float(host.estimate_wait_time(self._t_now))
```

`host.estimate_wait_time()` 返回的是站点排队等待时间，与 `queue_time_est` 是同一类量。这里用它来计算 mode C reservation 的过期时长，同属错误复用。这与 Run15C 里把 `gamma` 重新定义为"queue / service 风险权重"直接矛盾——如果这类量对 mode C 无意义，`gamma` 就不应该继续乘以站点排队时间。

**修复方向**：直接移除，不替换。`reservation_count` 已确认是误导信号（见下文 recovery token 分析），不应出现在排序键或 token 中。

### 3. reservation 问题的核心不是服务时长漂移，而是 commitment 语义缺失

当前 reservation timeout 检查使用的是运行时重建的 `coarse_plan.truck_eta_map`，而不是 dispatch 时刻的 commitment。  
所以真正需要讨论的边界是：

- timeout 应该对齐 dispatch 时承诺的 truck ETA
- 还是对齐运行时滚动更新后的 truck ETA

这比继续调 `alpha/gamma` 更关键。

### 4. 新增 observation 特征不是零成本改动

如果给 `OrderFeatures` 新增聚合特征：

- order token 维度会变化
- actor 输入维度会变化
- 旧 checkpoint 不能直接复用

当前训练入口本来就是从头初始化训练，所以这不是当前流程的阻塞项；  
但如果后续想做 warm-start / hot-start，就必须把这个代价明确写出来。

## Run15 执行顺序

Run15 不建议把所有结构改动一次性上线。建议拆成四个子阶段：

1. `Run15A`
   只加诊断指标，不改行为
2. `Run15B`
   只改评分语义
3. `Run15C`
   只改 reservation 语义
4. `Run15D`
   最后才考虑加新的 observation 聚合特征

下面的修改清单按这个顺序展开。

## Run15A：先上诊断，不改行为

### 目标

在不改变当前 policy 行为的前提下，把 `mode C` 的失败位置看清楚。

### 具体改动

在 episode / eval 诊断中新增以下字段：

- `mode_c_selected_filter_margin_sum`
- `mode_c_selected_execution_slack_sum`
- `mode_c_selected_reservation_count_sum`
- `mode_c_selected_truck_eta_remaining_sum`
- `mode_c_timeout_from_state`
  细分为：
  - `delivered`
  - `return_to_rendezvous`
  - `waiting_for_truck`
- `mode_c_fallback_from_state`
  同样按状态拆分

另外建议新增一组 dispatch-time commitment 诊断值：

- `mode_c_selected_planned_truck_eta_sum`
- `mode_c_selected_planned_uav_eta_sum`
- `mode_c_selected_planned_slack_sum`

### 预期收益

- 明确 reservation timeout 主要发生在哪个状态
- 明确 fallback 是“送达后立即进入”还是“返航 / 等车阶段触发”
- 为后续评分或 reservation 修复提供因果依据

### 说明

- `Run15A` 不允许改 reward
- 不允许改排序
- 不允许改 timeout 逻辑

这是纯诊断轮。

## Run15B：单独修评分语义

### 问题

当前存在三套“best mode C”语义：

- recovery pool 认为的 best
- recovery slot 排序后的 best
- order token 中 `best_mode_c_*` 摘要对应的 best

它们不一定一致。

### 具体改动

**第一步：从排序键和摘要中移除 mode C 路径上的 queue/swap 量**

- `recovery_pool_selector.py` 排序键：直接移除 `queue_time_est`，不替换
- `candidate_builder._build_mode_c_candidates` 排序键：移除 `queue_time_est > threshold` 和 `queue_time_est` 两项，不替换
- `OrderFeatures.best_mode_c_queue_time_est`：移除该字段，不替换

注意：mode B 路径（`_select_best_mode_b_host`）的 `queue_time_est` 不动，那里是正确的。

**第二步：从 recovery token 中移除无意义维度**

`RECOVERY_TOKEN_FIELDS` 当前共11维，需移除以下四维：

```python
"has_truck_eta",                   # 移除：恒为 True，对 policy 无区分度
"predicted_queue_time_est_norm",   # 移除：mode B 逻辑错误复用，mode C 不经过站点队列
"service_time_norm",               # 移除：mode B 逻辑错误复用，mode C 不使用换电服务
"reservation_count_norm",          # 移除：当前训练环境回收逻辑绕过了 truck.parking_slots 容量检查，
                                   #        多架 UAV 同时等在同一节点会被全部回收，不存在真实竞争，
                                   #        该信号会误导 policy 回避实际上没有约束的节点
```

移除后 recovery token 从11维降到7维，recovery head 的输入投影层维度变化，旧 checkpoint 不能直接复用。当前流程本来是从头训练，因此可以做。

**第三步：引入两个显式评分函数**

```text
proxy_mode_c_score
exact_mode_c_score
```

要求如下：

- `proxy_mode_c_score`
  只用于 `recovery_pool_selector.py`
- `exact_mode_c_score`
  只用于 `candidate_builder.py`
- `best_mode_c_*` 摘要必须来自 `argmax(exact_mode_c_score)`，而不是默认取 slot0

建议的语义目标统一为：

```python
# 归一化基准：upper_horizon_sec = 3600s
normalized_margin = rendezvous_margin / upper_horizon_sec      # 会合时间余量归一化
normalized_eta    = truck_eta_remaining / upper_horizon_sec    # 卡车到达剩余时间归一化

exact_mode_c_score = 1.0 * normalized_margin - 0.05 * normalized_eta
```

权重说明：
- `w_margin = 1.0`，`w_eta = 0.05`
- 归一化后两者量纲统一（均为 [0,1]），`w_eta = 0.05` 才是真正的次要权重
- 不归一化直接用原始秒数时，`w_eta = 0.1` 实际上会让 `truck_eta_remaining` 与 `rendezvous_margin` 同量级，不是"小权重"

`proxy_mode_c_score`（pool selector 用，不知道具体 UAV 状态）：

```python
proxy_mode_c_score = proxy_rendezvous_margin / upper_horizon_sec
```

pool selector 只做粗筛，`proxy_rendezvous_margin`（代理会合时间余量，用2D距离下界估算）已是最好的可用信号，不加 `w_eta` 项。


但要明确：

- pool selector 用 proxy 量（2D距离下界估算，不知道具体 UAV 状态）
- candidate builder 用真实量（已知 UAV 位置、电量、速度）

### 暂时不要做的事

- 不要在这一轮新增 observation 特征
- 不要同时改 reservation timeout 逻辑

### 成功标准

如果只改评分语义就有效，应看到：

- `mode_c_selected_count` 变化不一定大
- 但 `mode_c_success_rate` 提升
- `reservation_timeout_count` 与 `fallback_count` 至少有一项下降

## Run15C：单独修 reservation 语义

### 问题

**`expires_at`（reservation 过期的绝对时刻）算错了，不是 strict vs rolling 的选择问题。**

当前 `_acquire_reservation()` 的计算：

```python
truck_eta_to_recovery = t_arrive_truck - t_now   # 卡车到达回收节点还需要多少秒
tau_res = alpha * truck_eta_to_recovery + gamma * host.estimate_wait_time()
expires_at = t_now + tau_res                      # reservation 过期的绝对时刻
```

`tau_res` 的物理含义是”卡车到达时间的一个加权倍数”，但 UAV 完成 mode C 实际需要的时间是：

```
送达服务时间 + 从送达点飞到回收节点的时间
```

这两项都没有进入 `tau_res`。所以 `expires_at` 不是”UAV 执行完整个流程的 deadline”，而是”卡车到达时刻的粗略倍数”，两者没有直接的物理对应关系。

当前 timeout 检查有两个条件（`env_adapter.py:2013`）：

- **条件1**：`t_now >= expires_at`（固定过期时刻到了）
- **条件2**：`t_arrive_uav（无人机从当前位置预计到达回收节点的时刻） + 执行安全余量 > t_arrive_truck（卡车到达回收节点的当前预计时刻）`，每步用当前 coarse plan 重新估算

条件1 依赖一个算错的 `expires_at`，条件2 是动态的但完全不依赖 dispatch 时的承诺。两者都不是”基于 UAV 执行进度的 deadline”。

所谓 `strict_commitment` vs `rolling_commitment` 的讨论，本质上是在问”用条件1还是条件2”——但这是个伪问题，因为条件1 的 `expires_at` 本身就没有物理意义。真正需要做的是把 `expires_at` 改成一个有物理意义的量。

### 具体改动

**决策：采用 rolling_commitment，移除条件1**

条件2（每步用当前 truck ETA 动态判断）语义正确，直接测试”无人机现在还能不能赶上卡车”。条件1 依赖算错的 `expires_at`，移除即可。

实现：

1. 删除 `_reservation_timeout_cost` 中的条件1（`if t_now >= reservation.expires_at`）
2. `ReservationState` 保留 `issued_at`（dispatch 时刻，用于诊断），删除 `expires_at`
3. `_acquire_reservation` 中的 `tau_res` 计算整段删除

在 `DispatchCommit` 中新增以下字段，仅用于诊断，不参与 timeout 判断：

- `planned_truck_arrival_time`（dispatch 时刻卡车预计到达回收节点的绝对时刻）
- `planned_uav_arrival_time_lb`（dispatch 时刻估算的无人机预计到达回收节点的时刻下界 = 送达完成时刻 + 从送达点飞到回收节点的时间）
- `planned_execution_slack_sec`（dispatch 时刻的会合时间余量 = planned_truck_arrival_time - planned_uav_arrival_time_lb）

### 对 `alpha / beta / gamma` 的处理

选了 rolling_commitment 后，`expires_at` 不再参与 timeout 判断，整个 `tau_res` 公式失去作用。`alpha`、`beta`、`gamma` 三个参数**全部从 reservation 公式中删除**，不需要重绑定。

### 暂时不要做的事

- 不要同时改评分语义
- 不要同时加新 observation 特征

### 成功标准

如果只改 reservation 语义就有效，应看到：

- `reservation_timeout_count` 明显下降
- `post_delivery_revalidation_fail_count` 不反弹
- `fallback_count` 至少不恶化

## Run15D：最后再决定是否加 observation 聚合特征

### 问题

当前 mode head 看到的 `mode C` 信息只有单点摘要，表达能力不够。  
但新增特征会改变 observation 维度，意味着模型输入结构变化。

### 建议新增的聚合特征

- `mode_c_candidate_count`（当前订单的 mode C 可行候选节点数量）
- `second_best_mode_c_score`（第二优候选的 exact_mode_c_score，当前 token 完全没有第二候选信息）
- `best_minus_second_gap`（最优与第二优的分差，可推导但有独立语义价值：大 gap 表示最优节点明显优于其他，小 gap 表示多个选项相近）

以下字段从原方案中移除：

- `best_mode_c_score`：是 `best_mode_c_rendezvous_margin` 和 `best_mode_c_truck_eta_remaining` 的线性组合，冗余
- `best_mode_c_reservation_count`：与 recovery token 里已移除的 `reservation_count_norm` 是同类误导信号，不应移除后又在 order token 里重新引入
- `max_mode_c_rendezvous_margin`：Run15B 后 `best_mode_c_rendezvous_margin` 来自 `argmax(exact_mode_c_score)`，由于 `w_margin=1.0` 主导，两者几乎总是指向同一节点，近似冗余
- `min_mode_c_queue_time_est`：mode C 不经过站点队列，无物理意义

### 注意

这一步会带来：

- order token 维度变化
- actor 输入投影层维度变化
- 旧 checkpoint 无法直接复用

当前流程本来是从头训练，因此可以做；  
但不应在 `Run15B / Run15C` 之前做，否则无法判断到底是语义修复生效，还是模型输入变化生效。

### 成功标准

只有在前两步仍不能让 `mode C` 稳定时，才建议做这一轮。  
它应被视为“增强感知能力”的后手，而不是 Run15 的起手。

## Run15 暂时不要动的项

为了保持实验纪律，Run15 全阶段建议保持以下项不动：

- `reward.rendezvous_arrive_bonus`
- `reward.rendezvous_bonus`
- `reward.mode_c_attempt_bonus`
- `training.entropy_coef`
- `training.recovery_entropy_coef`
- `candidate.rendezvous_filter_margin_sec`
- `candidate.rendezvous_execution_margin_sec`
- `light_drone` 物理参数

原因：

- Run14 已经证明 split margin 至少修好了 post-delivery revalidation 这类硬失败
- 当前更像是评分语义、reservation 语义和观测不足的问题
- 若再叠加 reward / 物理参数改动，会把因果关系重新搅乱

## Run15 验证指标

本轮完成后，优先看以下指标，而不是先看总 reward：

1. `sum_mode_c_selected_count`
2. `sum_mode_c_success_count`
3. `mode_c_success_rate`
4. `sum_reservation_timeout_count`
5. `sum_fallback_count`
6. `sum_mode_c_post_delivery_revalidation_fail_reasons.rendezvous_time_feasible`
7. `sum_feasible_mode_c_recover_node_count_total`

辅助诊断指标：

8. `mode_c_timeout_from_state.*`
9. `mode_c_fallback_from_state.*`
10. `mode_c_selected_planned_slack_sum`

## 成功标准

Run15 的最低成功标准应按阶段定义：

### `Run15A`

- 新增指标稳定产出
- 不改变现有 reward / mode 选择行为

### `Run15B`

- `mode_c_success_rate` 高于 Run14
- `fallback_count` 或 `reservation_timeout_count` 至少下降一项
- 不出现新的大规模 `post_delivery_revalidation_fail`

### `Run15C`

- `reservation_timeout_count` 明显下降
- timeout 触发状态分布更集中、更可解释

### `Run15D`

- 只有在 B/C 都不足以解决问题时才做
- 做完后必须接受“从头训练、不能直接复用旧 checkpoint”的代价

## 最终建议

Run15 不建议作为“一次性大版本”上线。  
更专业的执行方式是：

1. `Run15A` 先补诊断
2. 根据诊断结果，二选一先做：
   - `Run15B` 评分语义
   - 或 `Run15C` reservation 语义
3. 如果 B/C 单独验证后仍不足，再考虑 `Run15D` 的 observation 扩展

这个顺序比“一次性同时动 4 类结构改动”更稳，也更容易解释结果。
