# Phase7 Run03 训练复盘与 Run04 调整计划

**基准 run**：`phase7_20260429_run03`  
**对比 run**：`phase7_20260428_run02`  
**文档目标**：基于当前机制设置与参数设置，明确 `run03` 的收益来源、主要风险、以及 `run04` 的调整顺序

---

## 一、Run03 结果结论

### 1.1 总体判断

`run03` 明显优于 `run02`，应作为后续主线继续推进，不建议回滚到 `run02`。

原因很直接：

- benchmark 从 `-1602` 提升到 `-79.4`
- benchmark timeout 订单从 `6` 降到 `0`
- benchmark 结束原因从 `upper_horizon_reached` 变为 `all_orders_cleared`
- stochastic high reward 从 `-491` 提升到 `+588`
- stochastic high fallback 从 `14` 降到 `0`

因此，`run03` 的核心问题不是“策略坏了”，而是“critic 明显失稳，但 actor 已经学到了更好的行为”。

---

## 二、Run03 的主要现象

### 2.1 Benchmark 很早就稳定收敛

`run03` 在 `update=10` 时 benchmark 已达到：

- `mean_total_reward = -79.4355`
- `timeout_order_count = 0`
- `done_reason = all_orders_cleared`
- `episode_end_t_sec = 837.741998`

之后直到最终 `update=320`，benchmark 基本不再变化。

这说明：

- actor 在训练早期已经找到稳定可用的策略
- 后续大量训练并没有继续提升 benchmark
- 后期训练的主要变化来自 critic 与共享 trunk 的震荡，而不是行为策略继续变强

### 2.2 Run03 的提升主要来自更稳定的 Mode B

`run03` 在 benchmark 与 stochastic 中几乎都退化为：

- `mode_b_dispatch_ratio = 1.0`
- `mode_c_dispatch_ratio = 0.0`

这意味着本轮提升并不是来自更强的 `mode C rendezvous` 能力，而是来自更稳定的 `mode B` 清单策略：

- 更少 timeout
- 更少 fallback
- 更少 hard overdue

换句话说，`run03` 当前学到的是“保守但有效”的解。

### 2.3 Value loss 明显发散

`run03` 最后阶段 value loss 呈现剧烈震荡：

- `840`
- `5906`
- `1413`
- `4669`
- `848`
- `3399`

与此同时，最后几次 update 的 `return_mean` 仅约 `17~22`，但由 `value_loss = 0.5 * mse` 反推得到的 value RMSE 已达到约 `41~109`。

这说明：

- critic 误差量级已经明显大于 target 本身
- value head 没有形成稳定基线
- 当前训练后期存在“actor 够用，但 critic 在抖”的典型症状

---

## 三、为什么 Run03 会出现“策略变好但 critic 发散”

## 3.1 奖励结构本身是高方差 target

当前 reward 机制同时包含以下几类信号：

- 送达一次性正奖励：`R_delivery_bonus = 100`
- overdue 按时间持续累计惩罚
- fallback 按时间持续累计惩罚
- queue / wait / idle 按时间持续累计惩罚
- hard overdue 一次性惩罚：`600`
- hard failure 一次性惩罚：`1200`

这会造成 critic target 同时包含：

- 稀疏的大正奖励
- 稀疏的大负奖励
- 大量持续性小负奖励

对未归一化 value target 来说，这是典型高方差场景。

## 3.2 Run03 去掉了旧的 value 约束，但没有补上新的尺度稳定器

当前 PPO 更新中的 value loss 是直接 MSE：

```python
value_loss = 0.5 * mean((V - R)^2)
```

这比 `run02` 的旧 `vf_clip` 更合理，但它也意味着当前 critic 训练完全依赖：

- `value_loss_coef`
- `max_grad_norm`
- 共享 trunk 的整体稳定性

问题在于，当前实现里还没有：

- return normalization
- value normalization
- PopArt
- Huber critic loss

因此一旦少量样本出现大 TD-error，critic 很容易被直接拉偏。

## 3.3 当前 critic 是无界输出，且与 actor 共享一部分表达

当前模型里的 critic head 是普通 MLP 回归头，最终输出无范围约束。  
在这种结构下，如果 target 分布尺度抖动较大，而训练又持续到 actor 已经基本收敛之后，就容易出现：

- critic 输出随少量难样本大幅摆动
- 共享 trunk 被 critic 梯度反复拉动
- benchmark 不再提升，但 value loss 继续恶化

## 3.4 Run03 的最终 checkpoint 不是最佳 checkpoint

从 `eval_metrics.jsonl` 看：

- stochastic high 最优不是最终 `update=320`
- `update=140` 的 high reward 可达 `650.25`
- `update=160` 的 high timeout 最低仅 `2`
- 最终 `update=320` 的 high reward 反而回落到 `587.57`

因此当前还有一个工程问题：

- checkpoint 选择逻辑过于依赖“最后一个”
- 没有把“benchmark 早已饱和、stochastic 已达峰值”及时转成 early stop / best checkpoint 保留

---

## 四、Run04 调整原则

### 4.1 原则一：优先稳 critic，不回退 actor 已获得的收益

`run03` 的主要收益已经验证成立，因此 `run04` 的首要目标不是“重新探索更激进策略”，而是：

- 保住 `run03` 的 benchmark 清空能力
- 保住 stochastic high 的正收益
- 把 value loss 从失控状态压回稳定区间

### 4.2 原则二：先解决训练稳定性，再谈 Mode C 提升

当前 `mode C` 比例已经接近 `0`。  
这并不一定代表 `mode C` 机制本身错误，更可能代表：

- 当前 reward 结构下，纯 `mode B` 已足够拿到高分
- critic 尚未稳定，无法可靠区分 B/C 的长期价值

因此，`run04` 不应先通过提高 entropy 或强推 `mode C` 来增加探索，否则容易把：

- fallback
- timeout
- hard overdue

重新拉回来。

### 4.3 原则三：先解决“选模”问题，再解决“继续训练”问题

由于 `run03` 在早期就已经出现更好的 stochastic checkpoint，说明下一轮必须把：

- best checkpoint selection
- early stopping

纳入训练产物流程，否则即便训练中途已经出现更优模型，最终产出仍可能不是最佳版本。

---

## 五、Run04 调整计划

## 5.1 第一优先级：修 checkpoint 选择与 early stop

### 计划

为训练流程增加：

1. best checkpoint 选择逻辑
2. early stop 逻辑

### 原因

当前 `run03` 的 benchmark 在 `update=10` 之后已不再变化，继续训练的边际收益极低。  
而 stochastic high 在中途已经出现更优点位，说明：

- “继续跑到最后”不能保证结果更好
- 最终 checkpoint 可能只是“更晚”，不是“更优”

### 建议指标

best checkpoint 建议按以下优先级排序：

1. benchmark timeout 最少
2. benchmark 是否 `all_orders_cleared`
3. stochastic high timeout 最少
4. stochastic high reward 最高
5. stochastic medium reward 作为次级 tie-break

### early stop 建议

满足以下条件时可提前终止：

- benchmark 连续 `3` 次评估无提升
- stochastic high 连续 `5` 次评估无提升
- 且 value loss 没有稳定下降趋势

这一步优先级最高，因为它不改变策略学习目标，却能直接避免“好模型被后期震坏”。

---

## 5.2 第二优先级：为 critic 增加目标尺度稳定器

### 计划

在以下方案中优先选择一种落地：

1. return normalization
2. value normalization / PopArt
3. Huber value loss

推荐顺序：

- 第一选择：return normalization 或 PopArt
- 第二选择：Huber value loss

### 原因

当前问题的根本不是 actor 学不会，而是 critic 的 target 尺度波动太大。  
如果不处理 target scale，仅靠降低学习率或调小 `value_loss_coef`，通常只能“减轻发散”，不能真正解决 critic 训练不稳定。

### 具体判断

若本轮只允许做小改动，优先上 Huber loss。  
若允许做更完整的训练语义改动，优先做 return normalization 或 PopArt。

---

## 5.3 第三优先级：先做一轮纯参数稳态实验

如果 `run04` 先不改训练机制代码，只改参数，建议先试以下组合：

### 建议参数

- `ppo_learning_rate: 3e-4 -> 2e-4`
- `value_loss_coef: 0.25 -> 0.10`
- `max_grad_norm: 1.0 -> 0.5`
- `target_kl: 0.05 -> 0.03`
- `gae_lambda: 0.95 -> 0.92`

### 原因

#### 1. `ppo_learning_rate` 下调

共享 trunk 同时服务 actor 与 critic。  
当 critic 震荡很大时，过高学习率会加剧 trunk 被来回拖拽。

#### 2. `value_loss_coef` 下调

当前 critic 已经不是“学不动”，而是“学得太不稳”。  
此时继续维持较高 value 权重，容易让 critic 噪声反向干扰 actor。

#### 3. `max_grad_norm` 下调

当前大 value error 样本会产生很大的梯度。  
把全局梯度裁剪从 `1.0` 收紧到 `0.5`，可以降低单次暴冲的概率。

#### 4. `target_kl` 下调

actor 已经在早期找到可用策略，因此后续 update 不需要继续允许较大的策略位移。  
更严格的 KL 上限有利于后期稳态训练。

#### 5. `gae_lambda` 微降

在高方差 reward 下，略微降低 `gae_lambda` 可以减少长回报链上的方差传递。  
这通常会轻微增加 bias，但能改善 critic 目标的可学性。

---

## 5.4 第四优先级：单独优化 low / medium 的 reward 表现

### 计划

在 critic 稳住后，可考虑下调：

- `wait_idle_penalty_coef: 0.25 -> 0.15`

### 原因

`run03` 的 low 档经常是：

- 全部清空订单
- 没有 timeout
- 没有 fallback
- 但 reward 仍偏负

这类现象通常不是调度失败，而是 idle 成本过重。  
对于低强度场景，系统本来就更容易出现等待窗口，过重的 idle 惩罚会把“安全等待”也打成差策略。

### 注意

这一步应排在 critic 稳定之后。  
否则会把“reward 结构调整”和“critic 训练稳定性”两个变量混在一起，不利于判断。

---

## 5.5 第五优先级：暂缓主动提高 Mode C 占比

### 计划

`run04` 初期不主动通过以下方式强推 `mode C`：

- 提高 entropy
- 放宽 mode C mask 以鼓励更多 C
- 人为给 mode C 加额外偏置

### 原因

当前 `run03` 的主要优势来自：

- `mode B` 稳定
- timeout 少
- fallback 少

在 critic 还不稳时，过早鼓励 `mode C` 更可能带来：

- rendezvous 误判
- reservation timeout
- fallback 回升

如果后续在 critic 稳定后，high 档仍有 `2~3` 个 timeout 未消除，再考虑微调：

- `rendezvous_eta_safe_margin_sec`
- candidate recovery 筛选规则
- reservation timeout 相关惩罚与门槛

---

## 六、建议的 Run04 实验拆分

## 6.1 Run04A：训练流程稳态版

### 内容

- 增加 best checkpoint selection
- 增加 early stopping
- 其余策略与 reward 不变

### 目标

验证“仅通过正确选模，是否已经能显著优于当前最终 checkpoint”

---

## 6.2 Run04B：critic 稳定版

### 内容

在 Run04A 基础上增加以下之一：

- return normalization / PopArt
- 或 Huber value loss

并同时加入参数稳态组合：

- `lr = 2e-4`
- `value_loss_coef = 0.10`
- `max_grad_norm = 0.5`
- `target_kl = 0.03`
- `gae_lambda = 0.92`

### 目标

验证：

- value loss 是否明显回落
- stochastic high 是否保持正收益
- benchmark 是否仍稳定清空

---

## 6.3 Run04C：低强度 reward 优化版

### 内容

在 Run04B 基础上再调整：

- `wait_idle_penalty_coef = 0.15`

### 目标

验证：

- low 档 reward 是否由负转正或明显改善
- medium 档是否保持稳定
- benchmark / high 不应明显退化

---

## 七、Run04 的验收标准

建议把验收标准分成两层。

### 7.1 必须达成

- benchmark 保持 `all_orders_cleared`
- benchmark timeout 仍为 `0`
- stochastic high reward 保持显著正值
- stochastic high fallback 不回升
- value loss 不再出现 `1000~5000` 级别的持续震荡

### 7.2 争取达成

- stochastic high timeout 进一步从 `3` 降到 `2` 或更低
- low 档 reward 明显改善
- mode C 在不增加 fallback 的前提下恢复少量有效使用

---

## 八、最终建议

当前最重要的判断是：

- `run03` 已经证明现有 reward 与环境语义下，策略端可以学到明显更好的行为
- 下一轮工作的重点应转向“训练稳定性”和“最佳 checkpoint 保留”
- 不应因为 value loss 很差就回滚 actor 端已经获得的性能收益

因此，`run04` 的推荐执行顺序是：

1. 先补 best checkpoint 与 early stop
2. 再稳 critic target scale
3. 再做保守参数收紧
4. 最后再单独优化 low 档 reward 与 mode C 利用率

这条路径风险最小，也最符合当前 `run03` 已经暴露出的真实问题结构。
