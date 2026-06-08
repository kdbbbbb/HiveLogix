# Phase 7 Mode C 可学性分析与后续方案

**分析对象**：`phase7_20260429_run04`  
**关联 run**：`phase7_20260428_run01`、`phase7_20260428_run02`、`phase7_20260429_run03`、`phase7_20260429_run04`  
**文档目标**：明确当前 `mode C` 为什么没有被策略真正学起来，并给出后续实验应如何分层推进，避免继续把“候选供给问题”“奖励目标问题”“critic 不稳问题”混在一起处理

---

## 一、结论先行

当前 `mode C` 的核心问题，不是：

- `run05` 的 early-stop 语义导致训练太早结束；
- 或者 `mode C` 已经学会，只是评估时偶然没有被触发。

更合理的判断是：

- 现阶段策略已经把 `mode C` 当作“通常不值得使用”的分支；
- 这不是单一超参数造成的，而是由候选供给收窄、合法性窗口脆弱、奖励缺少正向闭环、critic 对长链条价值估计不稳、评估目标不要求 `mode C` 五类因素共同造成的。

换句话说，`mode C` 当前不是“差一点就学会”，而是“在现有目标函数下，模型有充分理由退化成高质量 `mode B-only`”。

因此，后续工作不应直接理解为：

1. 想办法让 `mode C` 比例变高；
2. 或者直接提高探索强度。

而应先拆成两个问题：

1. 当前环境到底给了多少真实可用的 `mode C` 机会？
2. 当这些机会存在时，为什么策略仍持续偏向 `mode B`？

只有先把这两个问题拆开，后续改动才不会重演 `run01` 那种“`mode C` 比例高，但 fallback / reservation timeout 爆炸”的退化路径。

---

## 二、现象与证据

## 2.1 Run04 的评估结果不是“偶尔不用 C”，而是“系统性不用 C”

从 `phase7_20260429_run04` 的评估产物可以直接看到：

- `benchmark_report.json` 中 `mode_c_dispatch_ratio = 0.0`
- `stochastic_report.json` 中 low / medium / high 三档 `mode_c_dispatch_ratio` 也均为 `0.0`

这说明：

- 当前 best policy 在 benchmark 和 stochastic 泛化评估里已经稳定退化为 `mode B-only`
- `mode C` 不是“评估集运气不好没撞到”，而是整体策略选择上已经被放弃

## 2.2 四轮 run 的演化路径说明：高 C 比例并不等于高质量 C 能力

按训练期 episode 聚合统计，`mode C` 使用轨迹大致如下：

1. `run01`
   - `mode_c_ratio ≈ 54.06%`
   - `fallback = 112477`
   - `reservation_timeout = 112473`
   - `timeout = 530`

2. `run02`
   - `mode_c_ratio ≈ 13.31%`
   - `fallback = 12991`
   - `reservation_timeout = 12991`
   - `timeout = 57`

3. `run03`
   - `mode_c_ratio ≈ 1.76%`
   - `fallback = 4961`
   - `reservation_timeout = 4804`
   - `timeout = 2981`

4. `run04`
   - `mode_c_ratio ≈ 1.93%`
   - `fallback = 2236`
   - `reservation_timeout = 2158`
   - `timeout = 1366`

这条轨迹的含义非常明确：

- `run01` 的问题不是“C 不会用”，而是“C 被大量使用，但大量失败”
- `run02` 到 `run04` 的收益，主要来自持续压低不稳定的 `mode C`
- 当前更好的总体表现，并不是建立在更强的 `mode C rendezvous` 上，而是建立在更稳定的 `mode B` 上

因此，不能把“最近几轮表现更好”误解为“离学会 `mode C` 更近了”。

## 2.3 本地抽样结果表明：`mode C` 机会不是完全不存在

对当前 poisson 训练环境做了小规模本地抽样，使用 30 个 episode 统计决策点：

### 策略 A：`B-first`

- 总决策点：`2348`
- 存在 dispatch 机会的决策点：`445`
- 其中存在合法 `mode C` 的决策点：`306`
- 在所有决策点上的 `has_c_ratio ≈ 13.03%`
- 在有 dispatch 的决策点中，存在合法 `mode C` 的比例约为 `306 / 445 ≈ 68.8%`

### 策略 B：`random_dispatch`

- 总决策点：`2368`
- 存在 dispatch 机会的决策点：`455`
- 其中随机选到 `mode C` 的 dispatch 次数：`163`
- `chosen_c_ratio ≈ 35.82%`

这说明两个关键事实：

1. 当前环境里并不是“根本没有合法 `mode C`”
2. 但即使合法 `mode C` 经常存在，训练出来的策略仍倾向把它当作劣解

因此，问题已经从“有没有 C 候选”上升到了“为什么 C 候选没有学出长期价值”。

---

## 三、为什么当前实现会自然退化成 B-only

## 3.1 当前评估目标并不要求策略必须掌握 `mode C`

当前 best checkpoint selection 与 early-stop 主要围绕：

- `stochastic_high`
- `stochastic_medium`
- benchmark guardrail

而这些评估并没有单独提供一组：

- `mode C` 显著优于 `mode B` 的场景
- 或者“不掌握 `mode C` 就明显拿不到高分”的场景

这意味着：

- 只要 `mode B-only` 已经能在现有 poisson 分布里稳定拿到更高 reward
- PPO 就没有动力继续保留一个高风险、长链条、脆弱时序依赖的 `mode C`

从优化目标角度看，这不是 bug，而是策略对当前目标函数的合理响应。

## 3.2 `mode C` 的候选供给先被裁得很窄

当前 coarse plan 对每个 UAV 可配送订单都允许 `{B, C}`，但给 `mode C` 的回收候选池非常保守：

- `recovery_pool[order] = truck_backbone_route[:max_candidate_recovery_per_order]`
- 默认 `max_candidate_recovery_per_order = 4`

这意味着：

- `mode C` 首先只能看未来骨架中最靠前的少量节点
- 即使更后面的卡车节点在物理上更适合 rendezvous，也根本不会进入候选池

该逻辑位于：

- `backend/training/env_adapter.py`

语义上，这一步会把 `mode C` 的真实可行域先裁小一轮，然后才交给运行时合法性过滤。

## 3.3 `mode C` 的物理合法性窗口本身就脆弱

在当前实现里，`mode C` 合法性至少同时要求：

1. 送达后仍有剩余能量
2. 回收点仍在未来骨架中
3. `t_arrive_truck > t_deliver_finish`
4. `t_arrive_uav + rendezvous_eta_safe_margin_sec <= t_arrive_truck`
5. UAV 能带着当前剩余电量安全飞到回收点

而且当前默认：

- `rendezvous_eta_safe_margin_sec = 15`

这种约束本身没有问题，但它决定了：

- `mode C` 不是“只要能飞到就行”
- 而是“必须在一个很窄的时序窗口里，精确比卡车更早抵达，还要留安全边际”

一旦：

- 送达点离卡车未来节点略远
- 卡车 ETA 偏早
- UAV 剩余能量略低

`mode C` 就会立即失效。

## 3.4 `mode B` 的比较对象却天然更强

当前 `mode B` 的返程宿主选择是：

- 在所有可达宿主中搜索
- 按 `送达后时刻 + 飞行时间 + queue_time + service_time` 选最优宿主

这等价于：

- `mode B` 可以在整个返程宿主集合中做全局贪心
- `mode C` 却只能在少量 future backbone 节点上做受限 rendezvous

因此，哪怕不考虑学习，仅从“动作设计不对称性”看：

- `B` 也是一个搜索空间更大、成功条件更宽松的基线动作
- `C` 若想胜出，必须在长期收益上足够明显

## 3.5 当前 reward 没有给 `mode C` 成功闭环以明确正反馈

现有 reward 结构里主要包含：

- 送达奖励 `R_delivery_bonus = 100`
- `wait / idle / queue / overdue / fallback / reservation_timeout / hard_failure` 等惩罚

但没有一个显式项直接奖励：

- 成功完成 `mode C rendezvous`
- 通过 rendezvous 获得的后续重定位优势
- 因搭车返程而节省的未来代价

这会带来一个很现实的问题：

- `mode C` 的价值几乎全靠 critic 从长时间跨度回报里间接学出来
- 但 `mode C` 的失败代价却是即时且明确的

也就是说：

- `mode C` 的好处是延迟、稀疏、难归因的
- `mode C` 的坏处是立即、密集、好归因的

这是最不利于 PPO 学习稀疏长链条动作的结构之一。

## 3.6 critic 当前不稳，会直接打掉 `mode C` 这类长链条价值估计

前面的 run03 / run04 复盘已经说明：

- critic 的 value loss 有明显震荡
- 当前还没有 return normalization / PopArt / Huber value loss 这类尺度稳定器

而 `mode C` 恰恰依赖 critic 去学出：

- “这一步虽然没有立刻多拿 reward，但后续可用性更好”
- “这一步多等一段时间搭车回收，未来能减少 fallback / queue / energy pressure”

当 critic 自身对长时 credit assignment 都不稳时，最先被放弃的通常就是这类动作。

因此，当前不是简单的“探索不够”，而是：

- 即使探索到了 `mode C`
- 价值头也未必能稳定告诉 actor 什么时候它真的是更优动作

---

## 四、问题拆解：必须分清“供给端”和“学习端”

后续若继续只看 `mode_c_dispatch_ratio`，很容易把不同问题混为一谈。

更合理的拆法是：

## 4.1 供给端问题

即：

- 当前真实决策点中，合法 `mode C` 的机会有多少？
- 每次出现 `mode C` 时，通常有多少个回收节点可选？
- `mode C` 候选最常因什么原因在送达后复核失败？

如果这些值本来就很低，那么优先要做的是：

- 扩候选池
- 调整 backbone 候选策略
- 检查 ETA margin 是否过紧

## 4.2 学习端问题

即：

- 当合法 `mode C` 已经出现时，策略为何依旧持续偏好 `mode B`？
- 是 reward 没给出足够正反馈？
- 是 critic 无法估计长期价值？
- 还是训练分布里根本没有足够多“C 真优于 B”的样本？

如果供给端并不匮乏，而策略仍不用 C，那么优先要做的是：

- critic 稳定化
- 增强 `mode C` 相关评估目标
- 必要时引入 imitation / warm start / 条件探索

---

## 五、建议的解决方案分层

## 5.1 第一层：先加诊断指标，不急着改策略目标

这是优先级最高的一层。

当前缺少的不是“更多猜测”，而是“更高信息量的训练与评估日志”。

建议新增以下指标：

1. `dispatch_decision_count`
   - 当前 episode 中存在至少一个 dispatch 动作的决策点数

2. `dispatch_decision_with_legal_mode_c_count`
   - 当前 episode 中存在至少一个合法 `mode C` 动作的 dispatch 决策点数

3. `avg_feasible_mode_c_nodes_per_dispatch_decision`
   - 每个 dispatch 决策点平均存在多少个合法 `mode C` 回收点

4. `mode_c_selected_count`
   - 策略真正选择 `mode C` 的次数

5. `mode_c_post_delivery_revalidation_fail_count`
   - 已选 `mode C` 后，在送达后复核失败的次数

6. `mode_c_post_delivery_revalidation_fail_reasons`
   - 至少拆成：
   - `node_still_valid = false`
   - `rendezvous_time_feasible = false`
   - `energy_feasible = false`

7. `mode_c_success_count`
   - 成功完成 rendezvous recovery 的次数

这些指标的意义是：

- 先知道 `C` 机会是否存在
- 再知道 `C` 为什么失败
- 最后才知道是否值得改 reward / 探索

如果没有这些指标，后续任何“mode C 还是不行”的结论都不够干净。

## 5.2 第二层：加入专门的 `C-sensitive eval split`

当前 benchmark 与 stochastic 指标都允许 `B-only` 获得较高分数。

因此建议新增一组独立验证集：

- `stochastic_mode_c_sensitive`

这组验证集的目标不是替代主 benchmark，而是专门检查：

- 在 `mode C` 明显优于 `mode B` 的场景里，策略是否具备使用 `mode C` 的能力

该 split 建议满足：

1. 订单位置更靠近 truck future backbone corridor
2. `mode B` 最近返程宿主虽然可达，但未来重定位价值差
3. `mode C` 若成功，会显著改善后续派送机会或能量可用性

只有把这种 split 纳入：

- checkpoint selection 的次级参考
- 或 postmortem 的固定报告项

后续才能避免“主 reward 更高，但 `mode C` 能力彻底丢失”的盲区。

## 5.3 第三层：先稳 critic，再谈推动 `mode C`

在现阶段，我更倾向把 critic 稳定化排在“主动提升 `mode C` 使用率”之前。

建议顺序：

1. 若允许做完整训练语义改动
   - 优先做 `return normalization`
   - 或 `PopArt`

2. 若本轮只接受较小改动
   - 优先加入 `Huber value loss`

3. 同时进行保守参数调整
   - `ppo_learning_rate: 3e-4 -> 2e-4`
   - `value_loss_coef: 0.25 -> 0.10`

原因很直接：

- `mode C` 的好处主要是长期收益
- 长期收益首先依赖 critic 学出来
- critic 不稳时，直接提升探索通常只会先把失败路径重新放大

因此，若不先稳 critic，后续任何 `mode C` 鼓励措施都可能只是把 `run01` 的问题重新引回来。

## 5.4 第四层：增加 `mode C` 的真实供给，而不是粗暴“鼓励选 C”

这一层的目标不是让策略“多用 C”，而是让策略在真正该用时能看见更合理的 C 机会。

### 方案 A：放宽 recovery pool 的构造方式

当前是：

- 直接取 `truck_backbone_route[:K]`

更合理的候选方式是：

1. 先看更长的 future backbone 区间
2. 再按确定性 score 选 top-K

例如可按以下维度排序：

- `rendezvous_margin`
- `truck_eta`
- 节点类型
- queue_time_est
- 与订单 / UAV 当前区域的后续协同价值

这样做的好处是：

- 不会因为“骨架前 4 个点碰巧都不合适”而直接把更优的后续 rendezvous 节点裁掉

### 方案 B：适度放大 `max_candidate_recovery_per_order`

当前默认：

- `max_candidate_recovery_per_order = 4`

可先做保守实验：

- `4 -> 6`
- 若收益明确，再尝试 `8`

注意：

- 这一步应与诊断指标结合看
- 如果 `avg_feasible_mode_c_nodes` 本来就很低，单纯增大上限未必有效

### 方案 C：谨慎检查 ETA safe margin 是否过紧

当前：

- `rendezvous_eta_safe_margin_sec = 15`

可以做小步 A/B：

- `15 -> 10`

但不建议更激进地直接降到很低，因为这会增加：

- 时序脆弱性
- 复核失败概率
- train/eval 语义不一致的风险

因此这一步只能作为保守实验，不应作为第一优先级。

## 5.5 第五层：让训练分布中真实出现“C 优于 B”的样本

如果训练输入长期来自“`mode B-only` 已经足够高分”的 poisson 分布，那么 PPO 继续忽略 `mode C` 是自然结果。

因此建议引入一部分专门的 `C-friendly` 训练 episode，作为混合训练分布的一部分，而不是完全替换现有 poisson 主分布。

这类 episode 的设计原则可以是：

1. 订单靠近卡车未来将访问的站点走廊
2. 使用 `mode C` 后能显著提升后续重定位质量
3. `mode B` 虽然可行，但会把 UAV 拉回到对后续订单不利的位置
4. `mode C` 的成功不会过分依赖极端窄窗口

目标不是造一个“只有 C 能赢”的玩具场景，而是提高训练数据中：

- “C 真正有用”
- “而且这种有用性能被 critic 学出来”

的样本占比。

## 5.6 第六层：给 `mode C` 成功闭环更清晰的 reward 信号

这一步要非常谨慎，原则是：

- 奖励“成功的 C 效果”
- 不奖励“选择了 C 这个动作本身”

推荐方向：

1. 对成功完成 `mode C rendezvous recovery` 给一个小的正向 shaping
2. 或使用 potential-based shaping，奖励未来可用性提升
3. 或奖励“成功上车后一定时间窗口内的后续调度优势”

不推荐方向：

1. 直接给所有 `mode C` 动作固定 bonus
2. 只要选了 C 就加分

原因是后者很容易导致：

- 策略为了拿 bonus 滥用 C
- fallback / reservation timeout 被重新放大

也就是把系统拉回 `run01` 的坏轨道。

## 5.7 第七层：在必要时引入 imitation / conditional exploration

如果完成前面几层后，`mode C` 仍然学不上来，那么再考虑：

### 方案 A：behavior cloning warm start

用规则或启发式生成一批高质量 `mode C` 样本：

- 不要求全局最优
- 但要保证失败率可控

先做短阶段 imitation warm start，再接 PPO。

这能解决的问题是：

- PPO 从零开始很难覆盖到足够多高质量 `mode C` 轨迹

### 方案 B：只对“有合法 C 的 mode head”做条件探索增强

例如：

- 仅在合法 `mode C` 存在时提高局部 entropy pressure
- 或对 mode head 施加临时偏置后逐步退火

这样比全局提高 entropy 更安全，因为它不会把所有动作都一起扰乱。

### 方案 C：对有合法 `mode C` 的决策点做重加权采样

让 minibatch 中这类样本占比更高，以缓解：

- `mode C` 机会本来稀疏
- 有价值轨迹更稀疏

的问题。

---

## 六、明确不建议直接做的事

为了避免方向跑偏，下面几件事不建议作为第一反应：

## 6.1 不建议直接全局提高 entropy

原因：

- `run01` 已经证明高 `mode C` 使用率并不等于高质量 `mode C`
- 全局探索变强后，最先回来的通常是失败路径，而不是干净的长期价值学习

## 6.2 不建议直接给 `mode C` 动作硬加大 bonus

原因：

- 这会鼓励“选 C”而不是“成功完成 C”
- 极易导致滥用 `mode C`

## 6.3 不建议只盯着 `mode_c_dispatch_ratio`

原因：

- 比例变高并不代表质量变高
- `run01` 已经给出反例

真正该看的至少包括：

- `mode_c_success_count`
- `mode_c_post_delivery_revalidation_fail_count`
- `fallback`
- `reservation_timeout`
- 以及 `C-sensitive eval split`

## 6.4 不建议把 YAML 中 `curriculum` 字段当作现成解决方案

当前配置里虽然保留了：

- `station_queue_noise`
- `truck_delay_noise`
- `uav_failure_prob`
- `swap_time_noise`

但训练侧当前实现并未形成一套完整、已启用的 curriculum 闭环。

因此，单纯改这些字段并不能替代：

- 训练分布设计
- 指标补全
- critic 稳定化
- reward 与 eval 目标修正

---

## 七、推荐的实验执行顺序

为了控制回归风险，建议按以下顺序推进，而不是并行乱改。

## 7.1 第一阶段：只补诊断与评估，不主动鼓励 C

目标：

- 确认 `mode C` 当前到底是供给不足还是学习失败

建议动作：

1. 新增 `mode C` 相关 episode / eval 诊断指标
2. 新增 `C-sensitive eval split`
3. 保持现有 reward 主结构不变

完成标准：

- 能稳定回答“C 机会有多少”
- 能稳定回答“C 为什么失败”

## 7.2 第二阶段：只做 critic 稳定化

目标：

- 在不强推 `mode C` 的前提下，先恢复长期价值估计能力

建议动作：

1. `return normalization` 或 `PopArt`
2. 若改动预算有限，则先上 `Huber value loss`
3. 配合下调 `ppo_learning_rate`
4. 配合下调 `value_loss_coef`

完成标准：

- value loss 波动明显收敛
- stochastic high 不因 critic 抖动而持续退化

## 7.3 第三阶段：只调整 `mode C` 候选供给

目标：

- 不改奖励，仅扩大合理的 C 候选暴露

建议动作：

1. 放宽 recovery pool 生成方式
2. 小步提高 `max_candidate_recovery_per_order`
3. 必要时小步下调 `rendezvous_eta_safe_margin_sec`

完成标准：

- `dispatch_decision_with_legal_mode_c_count` 上升
- `mode_c_post_delivery_revalidation_fail_count` 没有同步爆炸

## 7.4 第四阶段：再考虑 reward shaping 或 imitation

目标：

- 在供给端与 critic 都改善后，再补 learning signal

建议动作：

1. 成功 rendezvous shaping
2. `C-friendly` 混合训练分布
3. behavior cloning warm start
4. conditional exploration

完成标准：

- `C-sensitive eval split` 确认策略开始真正使用高质量 `mode C`
- 且 benchmark / stochastic 主指标没有明显回归

---

## 八、建议的近期落地版本

如果只允许做一轮保守迭代，我建议优先落地一个“Mode C 诊断增强版”，目标不是立刻把 `mode C` 学起来，而是把问题看清。

推荐版本可包含：

1. 为 episode / eval 报告新增 `mode C` 供给与失败原因指标
2. 增加一个 `C-sensitive eval split`
3. 做一轮 critic 稳定化
4. 保守调整 recovery pool 暴露逻辑

暂不建议首轮就做：

1. 全局提高 entropy
2. 给 `mode C` 固定 bonus
3. 大幅放宽时序安全边际

因为这三类改动都更像“强行把 C 推出来”，而不是“让 C 在正确场景下自然学出来”。

---

## 九、最终判断

当前 `mode C` 学不起来，不应被简化理解为：

- “再多跑一点就好了”
- “early-stop 修完就好了”
- “把探索调大一点就好了”

更准确的判断是：

- 当前系统已经学会了一个高质量 `mode B-only` 解
- 而 `mode C` 在现有训练目标下既难学、又高风险、又没有足够独立评估压力

所以，后续正确路径不是“硬推 `mode C`”，而是：

1. 先补诊断
2. 再稳 critic
3. 再扩大合理候选供给
4. 最后才用 reward shaping / imitation / 条件探索去补 learning signal

只有按这个顺序推进，后续才有机会得到：

- `mode C` 比例不一定虚高
- 但在真正该用时能够稳定用出来
- 且不会重新带回大量 fallback / reservation timeout 的高质量策略

---

## 十、已落地实现记录（2026-04-29）

截至当前代码版本，本文前述方案中已有两层完成落地：

1. 第一层：先加诊断指标，不急着改策略目标
2. 第二层：加入专门的 `C-sensitive eval split`

下面记录本次实现的具体口径，避免后续文档与代码脱节。

## 10.1 第一层已落地内容：Mode C 诊断指标

当前 `TrainingEnvAdapter` 已增加以下 episode 级诊断指标：

1. `dispatch_decision_count`
   - 当前 episode 中，实际对外暴露过至少一个 dispatch 动作的决策点数

2. `dispatch_decision_with_legal_mode_c_count`
   - 上述 dispatch 决策点中，存在至少一个合法 `mode C` 动作的决策点数

3. `feasible_mode_c_recover_node_count_total`
   - 所有 dispatch 决策点上，合法 `mode C` 回收节点数量总和

4. `avg_feasible_mode_c_nodes_per_dispatch_decision`
   - `feasible_mode_c_recover_node_count_total / dispatch_decision_count`

5. `mode_c_selected_count`
   - 策略真实选择 `mode C` 的次数

6. `mode_c_success_count`
   - 成功完成 rendezvous recovery 的次数

7. `mode_c_post_delivery_revalidation_fail_count`
   - 已选择 `mode C` 后，在送达后复核失败的次数

8. `mode_c_post_delivery_revalidation_fail_reasons`
   - 当前已拆为三类布尔条件失败计数：
   - `energy_feasible`
   - `rendezvous_time_feasible`
   - `node_still_valid`

### 10.1.1 当前统计口径说明

本次实现明确采用以下口径：

1. 决策点统计只在 `reset()` / `step()` 对外返回的新 `decision_context` 上记录一次
   - 不在 `current_decision_context` 属性访问时累计
   - 这样可避免同一决策点被重复读取时重复计数

2. `dispatch_decision_count`
   - 只统计对外暴露的、且 `action_lookup` 中至少包含一个 `DispatchAction` 的决策点

3. `dispatch_decision_with_legal_mode_c_count`
   - 只要该决策点存在至少一个合法 `mode C` 动作，就记一次

4. `mode_c_post_delivery_revalidation_fail_reasons`
   - 当前实现不是互斥分类，而是按复核布尔条件分别累计失败
   - 因此一次复核失败可能同时增加多个 reason 计数

### 10.1.2 当前输出位置

这些诊断指标当前会出现在：

1. 单 episode snapshot
   - `env.build_episode_metrics_snapshot()`

2. 训练 / 评估 episode 记录
   - `episode_metrics.jsonl`
   - `_evaluate_policy_episode()` 返回值

3. 聚合评估报告
   - `_summarize_episode_records()` 会继续汇总为：
   - `sum_dispatch_decision_count`
   - `sum_dispatch_decision_with_legal_mode_c_count`
   - `sum_feasible_mode_c_recover_node_count_total`
   - `avg_feasible_mode_c_nodes_per_dispatch_decision`
   - `sum_mode_c_selected_count`
   - `sum_mode_c_success_count`
   - `sum_mode_c_post_delivery_revalidation_fail_count`
   - `sum_mode_c_post_delivery_revalidation_fail_reasons`

因此，从当前版本开始，postmortem 已经可以直接回答：

1. `mode C` 机会有多少
2. 机会出现时平均有几个合法回收点
3. 策略到底有没有选择 `mode C`
4. 选了之后更多是成功 rendezvous，还是在送达后复核失败

## 10.2 第二层已落地内容：`C-sensitive eval split`

当前评估链路已增加一个新的额外 split：

- `c_sensitive`

其定位是：

- 一个 benchmark-like 的 deterministic replay split
- 不替代原 benchmark
- 不替代 stochastic low / medium / high
- 只作为额外的 `mode C` 能力观察窗口

### 10.2.1 本次实现没有引入新的订单生成机制

为了保持改动收敛，本次没有新增新的 poisson 分布，也没有扩展 `OrderManager` 的 geo 采样逻辑。

当前 `c_sensitive` 的实现方式是：

1. 仍然基于 `orders.json.dynamic_orders`
2. 仍然走 benchmark / scheduled replay 链路
3. 但会先从全部 benchmark 动态单中筛出一个确定性子集
4. 再仅对该子集做一次独立 replay 评估

也就是说，它本质上是：

- “benchmark 动态单的语义子集”

而不是：

- “另一套新造的随机订单源”

### 10.2.2 当前筛选规则

当前 `c_sensitive` replay 子集的筛选规则不是依赖：

- `fulfillment_mode`

而是直接复用现有 `env_adapter.py` 的真实物理判定。

筛选步骤如下：

1. 构造一个 benchmark 模式、但不注入 dynamic replay 的分析环境
2. 选取一架 `depot-home` 且在 `t=0` 初始为 `idle` 的 UAV 作为分析基准
3. 对 `scene_ctx.dynamic_orders` 中的每一条动态单：
   - 把环境时刻移动到该订单的 `spawn_sim_s`
   - 把该 UAV 放回 depot，电量恢复到满电
   - 用当前 coarse backbone 口径构造 `coarse_plan`
   - 调用现有 `_iter_feasible_mode_c_recovery_nodes()`
   - 同时要求 `_has_mode_b_return_host()` 也为真
4. 若该订单满足：
   - 合法 `mode C` 回收点数 `>= c_sensitive_eval_min_legal_mode_c_recovery_nodes`
   - 且 `mode B` 返程宿主存在
   则该订单被选入 `c_sensitive` replay 子集

这一实现的好处是：

1. 完全复用当前 env 的真实 `mode C` 合法性语义
2. 不需要新增一套“猜测型”的标签口径
3. 不需要引入前文已明确不建议依赖的 `fulfillment_mode`

### 10.2.3 当前配置字段

当前配置文件 `backend/config/rh_alns_cmrappo.yaml` 已新增：

1. `c_sensitive_eval_enabled: true`
2. `c_sensitive_eval_seed: 20260426`
3. `c_sensitive_eval_min_legal_mode_c_recovery_nodes: 1`

其当前语义为：

1. `c_sensitive_eval_enabled`
   - 是否启用该额外 split

2. `c_sensitive_eval_seed`
   - 该 replay split 的固定随机种子

3. `c_sensitive_eval_min_legal_mode_c_recovery_nodes`
   - 一条 benchmark 动态单至少要满足多少个合法 `mode C` 回收点，才会被选入该 split

### 10.2.4 当前输出位置

当前 `c_sensitive` split 已被接入：

1. `_run_periodic_evaluation()`
2. `eval_metrics.jsonl`
3. 训练输出目录中的：
   - `c_sensitive_report.json`

并且：

- `eval_report["c_sensitive"]` 会保留完整 episode 与聚合指标
- `eval_metrics.jsonl` 中会保留剥离 episode 详情后的精简摘要

### 10.2.5 当前输出的附加元信息

当前 `c_sensitive` report 还会额外输出：

1. `c_sensitive_seed`
2. `selected_dynamic_order_count`
3. `selected_dynamic_order_ids`
4. `selected_dynamic_order_legal_mode_c_node_counts`
5. `c_sensitive_min_legal_mode_c_recovery_nodes`
6. `selection_drone_id`

因此，postmortem 时不仅能看到：

- 这个 split 上策略表现如何

还能看到：

- 这个 split 究竟是由哪些 benchmark 动态单构成的
- 每条动态单在筛选时有多少个合法 `mode C` 回收点

## 10.3 当前默认场景下的实际筛选结果

基于默认场景与当前实现，`c_sensitive` 当前会从 `orders.json.dynamic_orders` 中筛出以下 5 条动态单：

1. `DYN-BENCH-01`
2. `DYN-BENCH-02`
3. `DYN-BENCH-03`
4. `DYN-BENCH-04`
5. `DYN-BENCH-05`

其对应的合法 `mode C` 回收点数量分别为：

1. `DYN-BENCH-01 -> 4`
2. `DYN-BENCH-02 -> 4`
3. `DYN-BENCH-03 -> 3`
4. `DYN-BENCH-04 -> 2`
5. `DYN-BENCH-05 -> 1`

而：

1. `DYN-BENCH-06`
2. `DYN-BENCH-07`
3. `DYN-BENCH-08`
4. `DYN-BENCH-09`
5. `DYN-BENCH-10`

在当前默认口径下未被选入该 split，因为它们在分析时刻没有满足阈值要求的合法 `mode C` 回收点。

这说明：

- 当前 `C-sensitive` split 不是人为拍脑袋指定的订单列表
- 而是由现有环境语义和默认场景共同决定的一组 deterministic replay 子集

## 10.4 当前实现的边界

需要明确，本次实现故意保持保守，因此仍有以下边界：

1. `c_sensitive` 目前只是额外评估 split
   - 尚未接入 best checkpoint selection
   - 尚未接入 early-stop

2. `c_sensitive` 当前仍是单次 deterministic replay
   - 不是多 seed 随机评估

3. 当前筛选基准 UAV 只取一架 `depot-home idle` 无人机
   - 尚未扩展为多 UAV / 多初始态联合筛选

4. 当前筛选只要求：
   - 存在合法 `mode C`
   - 且 `mode B` 可行
   - 但还没有进一步要求“`mode C` 必须显著优于 `mode B`”

因此，这一版实现的定位仍然是：

- 先把 `mode C` 的独立观察窗口补出来

而不是：

- 已经完成了最终版 `C-sensitive` 泛化评估体系

## 10.5 本次实现的验证结果

本次修改后，已完成验证：

1. `python -m unittest backend.training.test_phase7_model_runtime`
   - 通过

2. `python -m unittest backend.training.test_env_adapter_phase5b`
   - 通过

3. 默认场景下离线探针验证
   - 成功筛出 5 条 `C-sensitive` 动态单
   - 与当前 env 物理判定结果一致

## 10.6 对后续阶段的意义

这两层实现完成后，当前代码已经能支持下一阶段更高信息量的 postmortem：

1. 不再只看 `mode_c_dispatch_ratio`
2. 可以先看 `mode C` 机会是否存在
3. 可以再看 `mode C` 是否被选择
4. 可以继续看 `mode C` 是成功 rendezvous 还是在送达后复核失败
5. 还可以单独看一组“对当前环境语义来说更偏向 `mode C` 的 replay 子集”上的表现

这意味着，后续若 `mode C` 仍学不起来，分析将能更明确地定位是：

1. 候选供给问题
2. critic 长期价值估计问题
3. reward 信号问题
4. 还是 checkpoint selection / 主评估目标仍不关心 `mode C` 的问题

## 10.7 第三层已落地内容：critic 稳定化第一步

截至当前版本，本文前述“5.3 第三层：先稳 critic，再谈推动 `mode C`”中，已经完成了一轮保守实现。

本轮没有直接上：

1. `return normalization`
2. `value normalization`
3. `PopArt`

原因是这些方案会同时牵涉：

- return target 的尺度语义
- bootstrap / GAE 的口径
- checkpoint 与训练过程中的统计状态管理

对当前代码结构而言，改动面更大，也更容易与前两层正在观察的 `mode C` 指标链路交叉污染。

因此，本轮优先落地的是文档中“小改动优先”的方案：

1. 引入可配置的 `Huber value loss`
2. 同时下调 `ppo_learning_rate`
3. 同时下调 `value_loss_coef`

### 10.7.1 当前 critic loss 已改为显式可配置

当前训练配置 `_TrainingConfig` 已增加两个字段：

1. `value_loss_type`
2. `value_huber_delta`

其中：

1. `value_loss_type`
   - 当前只允许：
   - `mse`
   - `huber`

2. `value_huber_delta`
   - 仅在 `value_loss_type = huber` 时生效
   - 当前要求必须为正数

这意味着，critic loss 已不再写死为：

- `0.5 * (V - R)^2`

而是通过统一 helper 显式切换。

### 10.7.2 当前已新增 `_compute_value_loss_masked()`

本次实现新增了统一的 critic loss helper：

- `_compute_value_loss_masked()`

当前语义如下：

1. `loss_type = mse`
   - 使用原有口径：
   - `0.5 * (V - R)^2`

2. `loss_type = huber`
   - 使用标准 Huber 形式
   - 在小残差区间保持二次项
   - 在大残差区间切换为线性增长

3. 最终仍统一通过 `valid_mask` 做 masked mean

这一步的主要目的不是改变 actor 目标，而是：

- 降低少量极端 TD-error 对 critic 梯度的放大效应

也就是说，本轮稳定化优先解决的是：

- critic 被少量坏样本拉偏过快的问题

### 10.7.3 当前 PPO 更新主链已改走该 helper

当前 `_ppo_update()` 中的 value loss 计算，已经从：

- 直接写死 MSE

改为：

- 调用 `_compute_value_loss_masked()`

因此，从当前版本开始：

1. critic loss 类型完全由训练配置驱动
2. 不需要再次改训练主循环代码就可以对比：
   - `mse`
   - `huber`

### 10.7.4 当前默认训练参数已切到更保守的 critic 稳态组合

本次同时修改了 `backend/config/rh_alns_cmrappo.yaml` 的默认训练参数：

1. `ppo_learning_rate: 0.0003 -> 0.0002`
2. `value_loss_coef: 0.25 -> 0.10`
3. `value_loss_type: huber`
4. `value_huber_delta: 1.0`

这些改动的合并意图是：

1. 先减小 critic 梯度更新步长
2. 先降低 critic loss 在总目标中的权重
3. 再用 Huber loss 抑制大误差样本的极端影响

也就是说，当前默认方案不是单独依赖某一个技巧，而是：

- “更小步长 + 更小权重 + 更稳健 loss”的保守组合

### 10.7.5 当前实现的边界

需要明确，这一轮 critic 稳定化仍然是第一步，不应被误读成“critic 问题已经彻底解决”。

当前仍然没有引入：

1. `return normalization`
2. `value normalization`
3. `PopArt`
4. 额外的 value target running stats

因此，这一版实现更准确的定位是：

- 先降低 critic 发散速度与极端波动风险

而不是：

- 已经完成 target scale 层面的最终稳态修复

换句话说，这一轮更像是：

- “让 critic 不要那么容易炸”

而不是：

- “让 critic 已经具备完整的尺度鲁棒性”

### 10.7.6 为什么这一步仍然重要

即使这一轮没有上 `return normalization / PopArt`，它依然对后续 `mode C` 学习很关键。

原因在于：

1. `mode C` 的好处本身偏长期收益
2. 长期收益首先依赖 critic 给出更平滑的价值估计
3. 若 critic 连局部大误差样本都压不住，那么后续无论是：
   - 扩大 `mode C` 候选供给
   - 增加 `C-sensitive eval`
   - 还是做 success shaping
   都会更容易出现“actor 刚探索到一点 C，critic 又把 trunk 拉坏”的问题

因此，这一轮 critic 稳定化虽然保守，但它是后续继续推进 `mode C` 的必要前置条件。

### 10.7.7 本次实现的验证结果

围绕这次 critic 稳定化实现，当前已完成验证：

1. `python -m unittest backend.training.test_phase7_model_runtime`
   - 通过

2. `python -m unittest backend.training.test_env_adapter_phase5b`
   - 通过

3. 新增单测验证
   - `_compute_value_loss_masked()` 同时支持 `mse` 与 `huber`
   - 在同一残差下，`huber` 数值符合预期
   - `ppo_update()` 主链在新配置字段存在时仍能正常运行

### 10.7.8 对后续阶段的意义

在第一层诊断指标与第二层 `C-sensitive eval split` 已落地的基础上，这一轮 critic 稳定化的意义在于：

1. 让后续 `mode C` 相关实验更不容易被 critic 失控噪声淹没
2. 让后续对 `mode C` 表现的判断更接近“策略问题”，而不是“value head 抖动问题”
3. 为后续若仍需进一步升级到：
   - `return normalization`
   - `PopArt`
   提供一个更平稳的中间基线

因此，截至当前版本，本文前三层中已有以下状态：

1. 第一层：诊断指标已落地
2. 第二层：`C-sensitive eval split` 已落地
3. 第三层：critic 稳定化已完成第一步保守实现

后续若仍观察到：

- `mode C` 机会存在
- `C-sensitive` 上仍学不出来
- 且 critic 虽较前稳定，但仍对长收益分支区分力不足

那么下一阶段才更适合继续推进：

1. `return normalization`
2. `PopArt`
3. 或更强的 success shaping / imitation 路线

## 10.8 第四层已落地内容：放宽 recovery pool 的构造方式

围绕前文“方案 A：放宽 recovery pool 的构造方式”，当前代码已经完成第一版保守落地。

这次改动的目标不是直接“奖励 mode C”，而是先解决一个更基础的问题：

- 不能因为 `truck_backbone_route[:K]` 的硬截断，把本来位于稍后位置、但更适合作为 rendezvous recovery 的节点过早裁掉

也就是说，本次修改解决的是：

- `mode C` 候选供给边界过窄

而不是：

- `mode C` 最终动作合法性判断

后者仍然由运行时 action mask 和送达后 revalidation 继续负责。

### 10.8.1 当前 recovery pool 已不再直接使用 `truck_backbone_route[:K]`

在本次实现之前，`env_adapter` 与 `PlannerBridge` 的 coarse plan 构造都采用同一个简单策略：

1. 取 future backbone 去重后的固定节点序列
2. 直接截取前 `max_candidate_recovery_per_order` 个节点

也就是：

- `truck_backbone_route[:K]`

这个策略的问题在于：

1. 它默认“越靠前的骨架节点越值得保留”
2. 但在 `mode C` 语义下，这个假设经常并不成立
3. 前几个节点可能：
   - 排队更差
   - 节点类型不优
   - 与订单后续区域协同价值更弱
   - 或仅仅因为几何位置不合适，导致真正更优的后续节点完全没机会进入候选池

因此，当前版本已经把该构造方式替换为：

1. 先扫描更长的 future backbone 前缀
2. 再按确定性 score 排序
3. 最后选出 top-K，作为 coarse 层的 `recovery_pool`

### 10.8.2 当前已新增共享选择器，避免两条链路行为漂移

本次没有分别在不同文件里各写一套“差不多”的排序逻辑，而是新增了统一的共享选择器：

- `backend/training/recovery_pool_selector.py`

当前：

1. `TrainingEnvAdapter._build_coarse_plan_view()`
2. `PlannerBridge._build_plan()`

都改为调用同一个：

- `select_recovery_pool_for_order(...)`

这样做的意义是：

1. 避免训练环境内联 coarse plan 和 planner bridge 各自维护一套逻辑
2. 避免未来出现：
   - 训练时是一个 recovery pool
   - 评估/重规划时又是另一套 recovery pool
3. 让后续如果继续调 recovery score，只需要改一个地方

### 10.8.3 当前确定性 score 只依赖现有真实字段

为了满足“代码上下文衔接正确，不凭空引入新状态”的原则，这次排序逻辑只使用了当前链路里已经真实存在的字段。

当前 score 依赖以下信息：

1. `proxy rendezvous margin`
   - 用 `truck_eta - delivery_to_node_fly_time_lower_bound` 作为代理项
   - 它不是最终运行时合法性判定，只是 coarse 层的排序信号

2. `truck_eta`
   - 越早被卡车访问的节点，在其他条件接近时优先

3. `node_type`
   - 当前显式区分 `station` 与 `depot`

4. `queue_time_est`
   - 使用节点当前 `available_slots / queue_length / swap_time` 推导的排队时间估计

5. `delivery_loc -> recovery_node` 的距离
   - 作为“与订单 / UAV 当前区域的后续协同价值”的保守代理项

最终排序是确定性的，不依赖随机性，不会给训练引入额外抖动源。

### 10.8.4 当前实现仍然严格保持 coarse/runtime 职责边界

需要强调的是，这次放宽 recovery pool 的构造方式，并没有改变系统原有的职责边界。

当前仍然成立的是：

1. coarse plan 只负责暴露：
   - “哪些 recovery 节点值得进入粗候选池”

2. candidate builder / env runtime 仍然负责：
   - 当前时刻 `mode C` 是否真的合法
   - 送达后原 recovery 节点是否还能通过 revalidation

所以，这次改动并不是把“真实合法性判定”前移到了 coarse 层，而只是把：

- coarse 层的供给边界从“前 K 个节点”升级成“长窗口扫描后的 top-K”

这点非常重要，因为如果 coarse 层就过早裁掉节点，后面的 actor 与 runtime 再强也没有机会学到那部分动作空间。

### 10.8.5 当前新增了新的配置字段

为了让该策略可调而不是写死，当前配置中已经新增：

- `candidate.recovery_pool_future_scan_limit`

默认值为：

- `8`

同时保留原有：

- `candidate.max_candidate_recovery_per_order`

当前语义是：

1. 先扫描 future backbone 的前 `recovery_pool_future_scan_limit` 个固定节点
2. 再按确定性 score 选出前 `max_candidate_recovery_per_order` 个节点

并且当前代码已显式校验：

- `recovery_pool_future_scan_limit >= max_candidate_recovery_per_order`

避免配置出现“扫描窗口反而比最终 top-K 还小”的无效状态。

### 10.8.6 当前实现的边界

本次实现虽然已经把 recovery pool 从“死板截断”提升到“可排序粗筛”，但它仍然是第一步，不应被误解为最终最优策略。

当前仍然没有做的事情包括：

1. 没有引入更复杂的多项归一化打分
2. 没有显式建模未来多个订单区域之间的联合协同收益
3. 没有把真实能量可行性提前搬进 coarse score
4. 没有在 coarse 层直接计算完整 rendezvous 成功概率

换句话说，当前实现的定位是：

- 用现有字段，先把“明显比前 K 截断更合理”的 deterministic top-K 粗筛跑起来

而不是：

- 已经完成最终版 recovery pool 最优排序器

### 10.8.7 本次实现的验证结果

围绕这次 recovery pool 构造改动，当前已完成验证：

1. `python -m unittest backend.training.test_env_adapter_phase5b`
   - 通过

2. `python -m unittest backend.training.test_phase6_integration`
   - 通过

3. `python -m unittest backend.training.test_phase7_model_runtime`
   - 通过

同时新增了两类针对性单测：

1. `env_adapter` 路径
   - 构造“前 4 个骨架站点都较差、后面某站点明显更优”的场景
   - 验证该后续节点可以进入 `recovery_pool`

2. `PlannerBridge` 路径
   - 用同类场景验证 planner bridge 的 coarse plan 输出与 env 内联 coarse plan 保持一致

这说明本次实现不仅修改了主代码路径，也已经验证了：

- 两条 coarse plan 生成链路在 recovery pool 选择上不会出现语义漂移

### 10.8.8 对后续阶段的意义

这一步的意义，不在于它会立刻让 `mode C` 学会，而在于它先消除了一个明显的供给侧瓶颈。

在当前版本下，它至少带来三点价值：

1. 后续节点不再因为“没排进骨架前 K”而完全失去被 actor 看见的机会
2. `C-sensitive eval split` 上观测到的 `mode C` 机会，更接近真实结构性机会，而不是粗候选池裁剪伪造成的机会缺失
3. 后续如果仍然观察到：
   - legal `mode C` 机会存在
   - critic 也较前稳定
   - 但 actor 仍持续忽略 `mode C`
   那么问题定位就能更聚焦到：
   - reward / credit assignment
   - exploration
   - imitation / shaping

因此，这一轮 recovery pool 改造的实际定位是：

- 先把 `mode C` 的粗候选供给边界修正到更合理的位置

这样后续关于 `mode C` 是否“真的学不会”的判断，才更有诊断价值。

## 10.9 `run05_recovery_pool_scan` 结果记录（2026-04-29）

在完成以下三项改动之后：

1. 第一层：`mode C` 诊断指标
2. 第二层：`C-sensitive eval split`
3. 第四层：放宽 recovery pool 的构造方式

当前已完成一轮新的训练验证：

- `phase7_20260429_run05_recovery_pool_scan`

本节记录这轮实验的客观结果，以及它对前文判断的影响。

### 10.9.1 本次训练基本信息

本轮训练输出目录为：

- `backend/weights/rh_alns_cmrappo/phase7_20260429_run05_recovery_pool_scan`

训练返回结果显示：

1. `global_step = 819436`
2. `updates = 200`
3. `stopped_early = True`
4. `stop_reason = early_stop:stochastic_high_no_improve=5,...`
5. `selected_policy_source = policy_best.pt`

这说明本轮不是因为训练报错中断，而是在当前 early-stop 规则下正常提前结束。

### 10.9.2 本轮最重要的结论

这次实验的最核心结论不是：

- `mode C` 已经学会

而是：

1. `recovery_pool` 放宽后，`mode C` 的候选供给明显改善了
2. critic 稳定化后的训练数值质量明显改善了
3. 但最终评估策略依然稳定退化为 `B-only`

也就是说，这一轮实验验证了：

- “供给不足”确实是问题的一部分

但同时也继续证明：

- 仅靠放宽候选供给，还不足以让 `mode C` 真正被学起来

### 10.9.3 recovery pool 放宽已明确生效

从本轮评估产物看，新的 recovery pool 逻辑已经真正把更多合法 `mode C` 机会暴露给了策略。

在 benchmark 上：

1. `dispatch_decision_count = 4`
2. `dispatch_decision_with_legal_mode_c_count = 4`
3. `feasible_mode_c_recover_node_count_total = 10`
4. `avg_feasible_mode_c_nodes_per_dispatch_decision = 2.5`

也就是说：

- 4 个 dispatch 决策点里，4 个都存在合法 `mode C`
- 且每个决策点平均有 2.5 个合法 recovery 节点

在 `C-sensitive` split 上结果完全相同：

1. `dispatch_decision_count = 4`
2. `dispatch_decision_with_legal_mode_c_count = 4`
3. `feasible_mode_c_recover_node_count_total = 10`
4. `avg_feasible_mode_c_nodes_per_dispatch_decision = 2.5`

在 stochastic 上，合法 `mode C` 供给也已经不再稀缺：

1. `low`
   - `sum_dispatch_decision_count = 21`
   - `sum_dispatch_decision_with_legal_mode_c_count = 19`
   - `sum_feasible_mode_c_recover_node_count_total = 56`
   - `avg_feasible_mode_c_nodes_per_dispatch_decision = 2.6667`

2. `medium`
   - `sum_dispatch_decision_count = 81`
   - `sum_dispatch_decision_with_legal_mode_c_count = 57`
   - `sum_feasible_mode_c_recover_node_count_total = 132`
   - `avg_feasible_mode_c_nodes_per_dispatch_decision = 1.6296`

3. `high`
   - `sum_dispatch_decision_count = 93`
   - `sum_dispatch_decision_with_legal_mode_c_count = 67`
   - `sum_feasible_mode_c_recover_node_count_total = 159`
   - `avg_feasible_mode_c_nodes_per_dispatch_decision = 1.7097`

这部分结果已经足以说明：

- `run05` 中，`mode C` 不是“没有机会”

而是：

- “机会已经出现，但策略仍然不愿意选”

### 10.9.4 最终评估策略仍然是稳定的 `B-only`

虽然合法 `mode C` 机会显著增加，但最终评估中 `mode C` 的使用率依然为零。

本轮：

1. benchmark
   - `mode_c_dispatch_ratio = 0.0`
   - `dispatch_mode_c_count = 0`

2. `C-sensitive`
   - `mode_c_dispatch_ratio = 0.0`
   - `dispatch_mode_c_count = 0`

3. stochastic `low / medium / high`
   - `mode_c_dispatch_ratio = 0.0`
   - `sum_mode_c_selected_count = 0`
   - `sum_mode_c_success_count = 0`

因此，这轮实验不能被解释为：

- 放宽 recovery pool 后，策略已经开始稳定利用 `mode C`

更准确的解释应当是：

- 放宽 recovery pool 后，策略已经看得见更多 `mode C`，但它在当前目标下依然更偏好 `B`

### 10.9.5 整体 reward 明显提升，但提升并不来自 `mode C`

和 `run04` 相比，本轮总体 reward 明显更好：

1. benchmark
   - `mean_total_reward: -79.4355 -> 220.5645`

2. stochastic low
   - `mean_total_reward: -126.8844 -> 81.1156`

3. stochastic medium
   - `mean_total_reward: 366.3487 -> 516.0450`

4. stochastic high
   - `mean_total_reward: 614.8322 -> 950.1573`

但因为最终评估里的：

- `mode_c_dispatch_ratio` 仍然全部为 `0.0`

所以这次 reward 提升不能归因为：

- `mode C` 被学会了

它更可能表示：

1. critic 更稳定后，`B-only` 策略质量本身提升了
2. 放宽 recovery pool 没有破坏整体主策略
3. 新训练配置在当前任务上比 `run04` 更稳

### 10.9.6 critic 稳定化第一步在本轮中是成功的

从训练指标看，这轮 critic 数值稳定性相比 `run04` 有非常明显的改善。

`run04`：

1. `value_loss_mean ≈ 2174.51`
2. `value_loss_max ≈ 61855.10`
3. 最后 5 次 `value_loss` 仍在 `1397 ~ 2915` 区间

`run05`：

1. `value_loss_mean ≈ 34.74`
2. `value_loss_max ≈ 97.53`
3. 最后 5 次 `value_loss` 大致在 `30 ~ 70` 区间

因此，这轮实验已经明确支持前文第三层的判断：

- critic 稳定化第一步是有效的

同时，本轮 early-stop 的触发原因也不是：

- value loss 再次失控

而是：

- `stochastic_high` 连续 5 次无提升

这意味着当前训练停止的主因，已经更接近“策略性能平台期”，而不是“critic 数值炸裂”。

### 10.9.7 训练期确实更常接触 `mode C`，但这仍然不等于学会

如果只看训练期 episode 指标，本轮策略比 `run04` 更常探索到 `mode C`。

以训练期统计看：

1. `run04`
   - `mode_c_ratio_over_dispatches ≈ 1.93%`

2. `run05`
   - `mode_c_ratio_over_dispatches ≈ 3.21%`

同时，`run05` 训练期累计：

1. `dispatch_decision_count = 165901`
2. `dispatch_decision_with_legal_mode_c_count = 116017`
3. `feasible_mode_c_recover_node_count_total = 267348`
4. `mode_c_selected_count = 5215`

这说明训练采样层面已经明显更容易“碰到并尝试 `mode C`”。

但是，成功质量非常弱：

1. `mode_c_success_count = 6`
2. `mode_c_post_delivery_revalidation_fail_count = 4`

也就是说：

- 训练中虽然比以前更常选到 `C`

但这些 `C` 绝大多数并没有形成足够稳定的成功收益链条，因此没有被固化为最终评估策略。

### 10.9.8 训练期副作用也上升了

和 `run04` 相比，训练期的 fallback / reservation timeout 强度也有所上升。

按 delivery 归一化后：

1. `fallback_per_delivery`
   - `run04 ≈ 1.997%`
   - `run05 ≈ 3.264%`

2. `reservation_timeout_per_delivery`
   - `run04 ≈ 1.928%`
   - `run05 ≈ 3.206%`

这和前面的观察是吻合的：

1. 候选供给变宽后，策略更容易探索 `mode C`
2. 但高质量 `mode C` 仍然不足
3. 因此训练中会多出一部分低质量 `C` 尝试及其副作用

所以，这轮实验虽然没有把系统重新拉回 `run01` 那种严重失控状态，但它也说明：

- 单纯扩大 `mode C` 机会，会带来更多探索噪声和失败成本

### 10.9.9 当前 `C-sensitive eval split` 还不够“尖锐”

本轮还暴露出另一个重要事实：

- 当前 `C-sensitive` split 虽然已经筛出了 legal-`C` 订单，但它还没有形成足够强的“必须用 `C` 才更优”的评估压力

当前 `C-sensitive` 选中了 5 个动态单：

1. `DYN-BENCH-01: 4`
2. `DYN-BENCH-02: 4`
3. `DYN-BENCH-03: 3`
4. `DYN-BENCH-04: 2`
5. `DYN-BENCH-05: 1`

这些数字表示：

- 各订单在筛选时刻存在的合法 `mode C` recovery 节点数量

但是最终 `C-sensitive` 的整体结果与 benchmark 几乎完全一致：

1. 总 reward 一样
2. `dispatch_mode_c_count = 0`
3. `mode_c_dispatch_ratio = 0.0`

这说明当前这版 `C-sensitive` 仍然更像：

- “存在 legal `C` 机会的 replay 子集”

而不是：

- “若不用 `C` 就明显吃亏的强区分验证集”

因此，第二层虽然已经落地，但它还需要继续强化，才能真正承担“验证 `mode C` 是否学会”的职责。

### 10.9.10 本轮结果对问题定位的影响

这轮 `run05` 非常有价值，因为它进一步压缩了问题空间。

截至当前版本，可以更有把握地说：

1. 问题已经不再主要是：
   - `mode C` 根本没有候选机会

2. 问题也不再主要是：
   - critic 完全不稳定，导致任何长收益分支都学不动

3. 当前更像是：
   - `mode C` 在当前 reward / value / exploration 组合下，仍然没有形成足够强的相对优势

因此，这轮实验把主问题更清楚地收敛到了：

1. reward shaping / success signal 不足
2. 长收益 credit assignment 仍然不够强
3. `C-sensitive` 验证集还不够能区分 `B-only` 与 `C-capable`
4. 必要时可能需要 imitation / warm start，而不是继续单纯扩候选池

### 10.9.11 当前最合理的下一步

基于这轮结果，后续最合理的顺序应当是：

1. 先强化 `C-sensitive eval split`
   - 让它真正变成“选 `C` 才明显更优”的验证集

2. 再考虑对 `mode C` 成功完成 rendezvous 引入小而明确的成功 shaping
   - 重点是奖励“成功完成 `C`”，而不是奖励“选了 `C`”

3. 如果仍然学不起来，再考虑：
   - imitation warm start
   - 或专门针对 legal-`C` 决策点的探索/采样重加权

不建议下一步继续单独做的事情是：

- 继续只扩大 recovery pool，而不改变评估压力和成功 credit

因为 `run05` 已经说明：

- 仅靠把更多 `mode C` 候选暴露出来，不足以让最终策略学会稳定使用 `mode C`
