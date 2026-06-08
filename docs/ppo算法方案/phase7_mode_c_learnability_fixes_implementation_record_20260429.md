# Phase 7 Mode C 可学性修复实施记录

**记录日期**：2026-04-29  
**关联分析文档**：`docs/ppo算法方案/phase7_mode_c_learnability_analysis_and_solution_plan_20260429.md`  
**记录目标**：沉淀本轮围绕 `mode C` 可学性问题完成的实现修改、关键语义确认与验证结果，避免后续 run 只看到训练结果而看不到环境与训练口径已经发生的变化

---

## 1. 本轮修改范围

本轮已落地 4 类修复：

1. 特征表示不对称修复
2. entropy 正则化不对称修复
3. 奖励信号不对称修复
4. rendezvous 成功状态覆盖语义确认

这些修改共同目标是：

- 让 `mode_head` 在 `B vs C` 选择时看到对称摘要信息
- 降低 `mode C` 因多一层 `recovery` 分支而承受的额外 entropy 惩罚
- 给 `mode C` 的 rendezvous 成功增加小幅正向探索信号
- 确认延迟奖励归因不会因为状态机窗口过窄而漏发 bonus

---

## 2. 第一层：特征表示不对称修复

### 2.1 问题

原实现中：

- `order_tokens` 内已经包含 `mode B` 的最优返程宿主摘要
- `mode C` 的信息只存在于 `recovery_tokens`
- `mode_head` 做 `B vs C` 选择时并不会直接看到 `mode C` 的最优摘要

结果是：

- `mode_head` 的输入对 `B` 分支更友好
- `C` 分支需要等到 `recovery_head` 才消费节点信息
- `B vs C` 决策天然不对称

### 2.2 已实现修改

在 `backend/training/contracts.py` 的 `OrderFeatures` 中新增 5 个字段：

- `has_mode_c_action`
- `best_mode_c_rendezvous_margin`
- `best_mode_c_queue_time_est`
- `best_mode_c_node_type`
- `best_mode_c_truck_eta_remaining`

在 `backend/training/candidate_builder.py` 中：

- 保留现有 `_build_mode_c_candidates()` 排序逻辑
- 新增 `_summarize_best_mode_c()`
- 直接从已排序 `recovery_candidates[0]` 提取最优 `mode C` 摘要

在 `backend/training/observation_tensorizer.py` 中：

- `ORDER_TOKEN_FIELDS` 从 13 维扩展到 18 维
- 将上述 5 个 `mode C` 摘要字段编码进 `order_tokens`

### 2.3 当前语义

`best_mode_c_truck_eta_remaining` 当前定义为：

```text
max(0, best_candidate.truck_eta - runtime_state.t_now)
```

即：

- 相对当前决策时刻的卡车剩余到达时间
- 与现有 `recovery_tokens` 的 remaining-time 语义保持一致

---

## 3. 第二层：Entropy 正则化不对称修复

### 3.1 问题

原实现里 `evaluate_actions()` 返回的 joint entropy 为：

```text
H(root)
+ P(dispatch) * H(order)
+ P(dispatch) * E[H(mode)]
+ P(dispatch) * P(mode=C) * E[H(recovery)]
```

训练侧再统一乘 `entropy_coef`。

这会导致：

- `mode C` 比 `mode B` 多一项 `recovery entropy`
- 早期训练中 `C` 分支梯度更嘈杂
- 策略存在通过压低 `P(mode=C)` 来降低 entropy 惩罚的动机

### 3.2 已实现修改

在 `backend/training/model.py` 的 `evaluate_actions()` 中新增参数：

- `recovery_entropy_coef: float = 1.0`

并将 recovery 项改为：

```text
entropy += p_dispatch * recovery_entropy_coef * expected_recovery_entropy
```

在 `backend/training/train_cmrappo.py` 中：

- `_TrainingConfig` 新增 `recovery_entropy_coef`
- `_ppo_update()` 显式把 `train_cfg.recovery_entropy_coef` 传入 `model.evaluate_actions()`

在 `backend/config/rh_alns_cmrappo.yaml` 中：

- 新增 `training.recovery_entropy_coef: 0.4`

### 3.3 当前语义

- 总体 PPO loss 仍然是 `- entropy_coef * entropy_loss`
- 但 `entropy_loss` 内部的 recovery 分支已经单独降权
- 旧配置若无该字段，读取时默认回退为 `1.0`

---

## 4. 第三层：奖励信号不对称修复

### 4.1 问题

`mode C` 的结构天然更难学：

- 成本大多是即时的：等待卡车、reservation timeout、fallback
- 收益是延迟的：成功 rendezvous 后才能体现后续接单价值
- 若只依赖长期 return，critic 需要跨更长时间窗口估计 `mode C` 价值

因此需要一个小幅 shaping reward，让策略更容易保留 `mode C` 探索。

### 4.2 已实现修改

在 `backend/config/rh_alns_cmrappo.yaml` 的 `reward:` 段新增：

- `rendezvous_bonus: 0.2`

在 `backend/training/contracts.py` 的 `RewardMeta` 中同步新增：

- `rendezvous_bonus`

在 `backend/training/train_cmrappo.py` 中：

- `_TrainingConfig` 新增 `rendezvous_bonus`
- `_load_training_config()` 从 `reward.rendezvous_bonus` 读取该值
- `_validate_training_config()` 要求其非负

训练侧新增两类 shaping 入口：

1. `_shape_post_action_reward_for_rendezvous()`
2. `_shape_pending_transition_reward_for_rendezvous()`

### 4.3 为什么不是只改 `post_action_reward`

如果 rendezvous 成功恰好发生在当前 action 的 `step_result` 内：

- bonus 会直接加到当前 `RolloutTransition.reward`

但如果 rendezvous 成功是在该无人机的后续等待过程中才发生：

- 当前 transition 会先作为 `pending_transition`
- 等到下一个决策点或 episode 终止时，才通过 `carry_in` 路径 finalize

因此本轮实现同时覆盖了：

1. 即时成功：`post_action_reward + rendezvous_bonus`
2. 延迟成功：finalize pending transition 时补发 `rendezvous_bonus`
3. episode 终止前仍未出队的 pending transition：terminal flush 时补发 `rendezvous_bonus`

### 4.4 防重复措施

在 `backend/training/rollout_buffer.py` 的 `RolloutTransition` 中新增：

- `rendezvous_bonus_applied: bool = False`

该字段用于保证：

- 同一个 `mode C` transition 只奖励一次
- 不会在 `post_action` 与 `pending finalize` 两个阶段重复加 bonus

### 4.5 当前语义边界

本轮 shaping 只作用于训练侧 transition reward，不改环境真值 reward：

- `RolloutTransition.reward` 使用 shaped reward
- `env.step_result.reward` 保持原始环境奖励
- episode metrics 仍然反映原始环境口径，而不是 shaped reward

这样做的目的，是避免污染环境语义与离线评估口径。

---

## 5. 第四层：rendezvous 成功状态覆盖语义确认

### 5.1 问题

延迟归因路径 `_shape_pending_transition_reward_for_rendezvous()` 需要在“下一次决策点到来时”判断旧的 pending transition 是否已经成功 rendezvous。

隐含风险是：

- 如果“会合成功状态”只持续极短一瞬间
- 而下一次决策点到来时无人机已经离开该状态
- 则 bonus 可能漏发

### 5.2 现有状态机确认结果

经核对 `backend/training/env_adapter.py`，当前 `mode C` 成功后的真实状态链路为：

```text
WAITING_FOR_TRUCK
-> CHARGING_ON_TRUCK
-> RIDING_WITH_TRUCK
-> truck_station_arrival 触发下一次决策
```

关键事实：

1. rendezvous 成功发生在 `_process_rendezvous_recovery()`
   - 成功后状态立即写为 `CHARGING_ON_TRUCK`

2. 车载回收充电结束后
   - 状态从 `CHARGING_ON_TRUCK -> RIDING_WITH_TRUCK`

3. 下一次真正可入队决策时
   - 车载路径只会在 `RIDING_WITH_TRUCK` 状态上通过 `truck_station_arrival` 触发

4. 当前实现中
   - 不存在 `mode C` 成功后在下一次决策点前直接落到 `IDLE` 的路径

### 5.3 当前结论

因此，当前 `_is_rendezvous_success_state()` 只匹配：

- `charging_on_truck`
- `riding_with_truck`

是充分的，不会漏掉从会合成功到下一次决策点之间的合法成功窗口。

需要注意的仅是未来演化风险：

- 如果后续状态机新增新的“已成功回收但尚未决策”的中间态
- 或允许车载 UAV 在下一次决策前直接转入 `IDLE`

则这里必须同步扩展状态集合。

---

## 6. 本轮涉及文件

核心代码改动位于：

- `backend/training/contracts.py`
- `backend/training/candidate_builder.py`
- `backend/training/observation_tensorizer.py`
- `backend/training/model.py`
- `backend/training/train_cmrappo.py`
- `backend/training/rollout_buffer.py`
- `backend/config/rh_alns_cmrappo.yaml`

测试补充位于：

- `backend/training/test_phase6_integration.py`
- `backend/training/test_phase7_snapshot_and_tensorizers.py`
- `backend/training/test_phase7_model_runtime.py`

---

## 7. 已完成验证

本轮已通过的关键验证包括：

```bash
python -m unittest backend.training.test_phase6_integration
python -m unittest backend.training.test_phase7_snapshot_and_tensorizers
python -m unittest backend.training.test_phase7_model_runtime
```

验证重点覆盖：

- `mode C` 最优摘要是否进入 `order_tokens`
- `order_tokens` 维度是否从 13 扩展到 18
- `recovery_entropy_coef` 是否真实作用于 entropy 公式
- PPO update 是否把 `recovery_entropy_coef` 正确传入 `evaluate_actions()`
- rendezvous bonus 是否能在即时成功路径生效
- rendezvous bonus 是否能在 pending finalize / terminal flush 的延迟成功路径生效
- rendezvous bonus 是否会被重复发放

---

## 8. 当前默认参数

本轮引入并默认启用的新增参数如下：

```yaml
reward:
  rendezvous_bonus: 0.2

training:
  recovery_entropy_coef: 0.4
```

当前建议解释：

- `rendezvous_bonus = 0.2`
  - 仅提供轻量正向探索信号
  - 不试图压过送达奖励或长期完成时间目标

- `recovery_entropy_coef = 0.4`
  - 降低 `mode C` 因 recovery 分支带来的额外 entropy 惩罚
  - 保留一定探索，但避免早期训练中过度偏向压低 `P(mode=C)`

---

## 9. 后续 run 解读注意事项

从本轮开始，后续训练结果的解读口径需要显式区分两件事：

1. 环境原始 reward / episode metrics 是否改善
2. 训练侧 shaping 后的策略学习信号是否更容易保留 `mode C`

这意味着：

- 即便后续 `mode C` 比例回升，也不能只看比例
- 还要同时看：
  - fallback
  - reservation timeout
  - rendezvous success
  - timeout
  - benchmark / stochastic 泛化表现

只有当 `mode C` 使用率、成功率和总体任务完成质量一起改善时，才能认为这轮修复真正有效。
