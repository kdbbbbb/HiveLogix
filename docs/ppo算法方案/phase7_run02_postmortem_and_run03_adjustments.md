# Phase7 Run02 训练复盘与 Run03 代码修复记录

**基准 run**：`phase7_20260428_run02`（1.5M steps，367 updates）  
**修复目标**：修复代码层面已识别的关键 bug，而非继续调参  
**修改文件**：`backend/training/env_adapter.py`、`backend/training/rollout_buffer.py`、`backend/training/train_cmrappo.py`、`backend/config/rh_alns_cmrappo.yaml`、`backend/training/test_phase7_model_runtime.py`

---

## 一、Run02 结果回顾

### 训练指标

| 指标 | Run02 实际 | Run02 目标 | 是否达成 |
|------|-----------|-----------|---------|
| benchmark reward | -1602 | >-200 | ❌ |
| benchmark pending orders | 4 | 0-1 | ❌ |
| stochastic medium reward | +373 | 正值 | ✅ |
| value loss（后期） | 1-12 | 稳定 | ✅ |
| 训练熵 | 仍频繁崩溃 | 0.05-0.15 | ❌ |

### 核心发现

Run02 的参数调整（entropy_coef 0.01→0.03、lambda_wait 0.10→0.25、R_delivery_bonus 60→100、vf_clip_coef 新增 0.20）对 stochastic 性能有改善，但 benchmark 性能完全未变。通过代码审查发现多处关键实现 bug，这些 bug 的影响远超参数调整的效果。

---

## 二、Bug 诊断

### Bug 1：训练每个 episode 使用相同随机种子

**位置**：`backend/training/env_adapter.py:391`（修复前）

**问题描述**：

`reset()` 每次都调用 `random.seed(self._order_source.seed)`，而 `order_source.seed` 固定为 `20260424`。这意味着整个训练过程中（约 30000 个 episode）每个 episode 都生成**完全相同**的 Poisson 订单序列。

Policy 实际上是在对单一固定场景做过拟合，而不是学习通用的派单策略。这直接解释了：
- stochastic eval（用不同 seed）表现尚可：policy 碰巧在这些 seed 上也能工作
- benchmark（固定场景）表现极差：policy 过拟合到了训练的那个固定 Poisson 序列，对 benchmark 的订单模式没有泛化能力

**根本原因**：每个 episode 的订单流完全相同，policy 记住了这个固定序列的最优响应，而不是学会了如何处理任意订单分布。

---

### Bug 5：奖励归因错误——单个无人机的决策承担所有无人机的 per-dt 成本

**位置**：`backend/training/env_adapter.py`，`_settle_per_dt_rewards` / `_advance_to_event` / `step()`

**问题描述**：

`step()` 返回的 reward 包含了从当前时刻到下一个决策点之间**所有无人机**的 overdue、queue、fallback、wait 惩罚，但这个 reward 被归因到**当前决策无人机**的 transition 上。

典型场景：无人机 A 选择 WAIT，时间推进 60s，期间无人机 B 超时产生 `-lambda_overdue * 60` 惩罚，这个惩罚被记在 A 的 WAIT 决策上。A 对 B 的超时没有控制权，但 PPO 会把这个负信号关联到 A 的 WAIT 动作，强化了"WAIT 是坏的"的错误信号。

**根本原因**：`_settle_per_dt_rewards` 把所有无人机的持续成本加总成一个全局 float 返回，`_advance_to_event` 把它直接累加进 reward，`step()` 把这个全局 reward 归属给当前决策无人机。

---

### Bug 2：vf_clip_coef=0.20 与未归一化 return 量级不匹配

**位置**：`backend/training/train_cmrappo.py:1007-1017`（修复前）

**问题描述**：

Value function clipping 的实现逻辑本身正确，但 `vf_clip_coef=0.20` 相对于实际 return 量级（~40-90）产生了**反效果**。

标准 PPO vf clipping 的计算方式：

```
v_clipped = old_value + clip(new_value - old_value, -eps, +eps)
value_loss = max((new_value - return)², (v_clipped - return)²)
```

`max()` 的语义是：取两者中**更大**的 loss。当 value 尚未收敛时（`old_value=0`，`return=40`，`new_value=20`）：

```
v_clipped = 0 + clip(20, -0.20, 0.20) = 0.20
vl_unclipped = (20 - 40)² = 400
vl_clipped   = (0.20 - 40)² = 1584
max(400, 1584) = 1584  ← clipped 赢
```

梯度方向是把 value 拉回 `old_value + 0.20 = 0.20`，而不是拉向 return 目标 40。Value 每次只能向 return 移动约 0.20 个单位，从 0 收敛到 87 需要约 435 次 minibatch 更新。

这完美解释了 Run02 的观测现象：value_loss 在 update 1-150 居高不下（最高 18230），update 150 之后才急剧下降——正是 value 以每步 0.20 的速度爬行到接近 return 目标所需的时间。

**根本原因**：vf clipping 的设计假设 return 已被归一化到 [-1, 1] 附近，此时 eps=0.20 是合理的更新步长。在未归一化的 return（量级 ~100）下，eps=0.20 使 clipping 始终激活，value 无法正常收敛。

---

### Bug 3：GAE 在多无人机混合 transition 序列上计算

**位置**：`backend/training/rollout_buffer.py`、`backend/training/train_cmrappo.py`

**问题描述**：

Run02 的原始实现中，rollout buffer 会按全局时间顺序存储所有无人机的 transition；在 12 架无人机交替决策的场景里，相邻 transition 往往来自不同 drone。原始 GAE 计算直接沿这个混合序列展开：

```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

这会出现：

- `s_t` 是无人机 A 的决策状态
- `s_{t+1}` 却可能是无人机 B 的决策状态
- A 的 advantage 被错误地用 B 的 value 做 bootstrap

后续即使再按 `(episode_id, actor_drone_id, recurrent_segment_id)` 做 recurrent sequence materialization，也已经来不及，因为 `advantages / returns` 在分组前就算错了。

进一步地，单纯把 GAE 挪到分组后还不够：若某个 drone 的序列在 rollout 尾部被截断，尾步 bootstrap 也必须使用**同一 drone / 同一 recurrent segment 的后继 value**，不能退化成“用本步自己的 `value_old`”或“沿混合序列取下一个 transition 的 value”。

**根本原因**：

- GAE 的时间相邻关系必须定义在**同一 agent 的有效决策序列**上，而不是多 agent 交错后的全局写入顺序
- recurrent PPO 的 sequence chunking 只是训练组织方式，不能反过来决定 advantage / return 的边界

---

### Bug 4：approx_kl 使用绝对值且在 `optimizer.step()` 后才 early stop

**位置**：`backend/training/train_cmrappo.py`，`_ppo_update`

**问题描述**：

原始实现中的 KL 近似是：

```python
latest_approx_kl = _masked_mean_tensor(
    (old_log_probs - new_log_probs).abs(),
    valid_mask_flat,
)
```

这有两个问题：

1. `abs()` 把所有样本都变成非负值，负向项（`new_log_prob > old_log_prob`）也会被当作 KL 增大处理，系统性高估策略偏移。
2. `latest_approx_kl` 是在 `optimizer.step()` 之后才检查，意味着超出 `target_kl` 的那次更新已经写入参数，`early stop` 只能阻止后续 minibatch，不能阻止本次过大的 step。

在这种实现下，很多本应继续训练的 minibatch 会被提前截断；而真正超限的更新又无法回滚，最终同时带来**样本利用率低**和**单步更新过猛**两个问题。

**根本原因**：

- PPO 常用的 `approx_kl` 是 `mean(old_log_prob - new_log_prob)` 的**有符号均值**，不是绝对值均值。
- `target_kl` 的语义是“在应用参数更新前判断本 minibatch 是否已偏离过大”，而不是“先更新，再决定后面是否停”。

---

## 三、修复内容

### 修复 1：每个 episode 使用不同随机种子

**文件**：`backend/training/env_adapter.py`

**改动 1**：在 `__init__` 中初始化 episode 计数器：

```python
# 修复前
self._t_now = 0.0
self._planned_route_stops: list[PlannedStop] = []

# 修复后
self._t_now = 0.0
self._reset_count = 0
self._planned_route_stops: list[PlannedStop] = []
```

**改动 2**：`reset()` 中对 POISSON 模式使用递增种子：

```python
# 修复前
def reset(self) -> EnvStepResult:
    random.seed(self._order_source.seed)

# 修复后
def reset(self) -> EnvStepResult:
    if self._order_source.mode == OrderSourceMode.POISSON:
        random.seed(self._order_source.seed + self._reset_count)
    else:
        random.seed(self._order_source.seed)
```

**改动 3**：`reset()` 末尾递增计数器：

```python
# 修复前
        return self._build_step_result(reward=0.0, info={"event": "reset"})

# 修复后
        self._reset_count += 1
        return self._build_step_result(reward=0.0, info={"event": "reset"})
```

**设计说明**：
- POISSON 模式（训练）：第 0 次 reset 用 `20260424`，第 1 次用 `20260425`，以此类推，每个 episode 看到不同的订单流
- BENCHMARK/HYBRID 模式（eval）：保持 `random.seed(base_seed)` 不变，eval 仍然是确定性的
- eval 环境每次都是新建的 `TrainingEnvAdapter` 实例，`_reset_count` 从 0 开始，不影响 benchmark eval 的确定性

---

### 修复 2：禁用 vf clipping，改用直接 MSE loss

**文件**：`backend/training/train_cmrappo.py`、`backend/config/rh_alns_cmrappo.yaml`

**改动 1**：`_ppo_update` 中移除 clipping，改为直接 MSE：

```python
# 修复前
v_clipped = old_values_flat + torch.clamp(
    values_flat - old_values_flat,
    -train_cfg.vf_clip_coef,
    train_cfg.vf_clip_coef,
)
vl_unclipped = (values_flat - returns) ** 2
vl_clipped = (v_clipped - returns) ** 2
value_loss = 0.5 * _masked_mean_tensor(
    torch.max(vl_unclipped, vl_clipped),
    valid_mask_flat,
)

# 修复后
value_loss = 0.5 * _masked_mean_tensor(
    (values_flat - returns) ** 2,
    valid_mask_flat,
)
```

**改动 2**：`_TrainingConfig` 删除 `vf_clip_coef` 字段：

```python
# 修复前
    clip_coef: float
    vf_clip_coef: float
    entropy_coef: float

# 修复后
    clip_coef: float
    entropy_coef: float
```

**改动 3**：`_load_training_config` 删除对应读取行：

```python
# 修复前
        clip_coef=float(training["clip_coef"]),
        vf_clip_coef=float(training.get("vf_clip_coef", 0.2)),
        entropy_coef=float(training["entropy_coef"]),

# 修复后
        clip_coef=float(training["clip_coef"]),
        entropy_coef=float(training["entropy_coef"]),
```

**改动 4**：yaml 注释掉 `vf_clip_coef`：

```yaml
# 修复前
  clip_coef: 0.20
  vf_clip_coef: 0.20
  entropy_coef: 0.03

# 修复后
  clip_coef: 0.20
  # vf_clip_coef 已禁用：max(vl_unclipped, vl_clipped) 在 return 未归一化时
  # 始终选 clipped（更大的 loss），梯度方向错误，导致 value 收敛极慢。
  # 直接使用 0.5*(V-R)^2，由 value_loss_coef 和 max_grad_norm 控制更新幅度。
  entropy_coef: 0.03
```

**设计说明**：
- 禁用 vf clipping 符合原始 PPO 论文（OpenAI 2017），clipping 是后来部分实现加入的可选项
- `value_loss_coef=0.25` 限制 value loss 对总梯度的贡献权重
- `max_grad_norm=1.0` 对所有参数的梯度做全局裁剪，防止单步更新过大
- 两者共同承担了原本 vf clipping 试图实现的"限制 value 更新幅度"职责，且不会产生反效果

---

### 修复 3：按 drone / recurrent segment 独立计算 GAE，并显式解析尾部 bootstrap

**文件**：`backend/training/rollout_buffer.py`、`backend/training/train_cmrappo.py`、`backend/training/test_phase7_model_runtime.py`、`backend/training/test_phase7_snapshot_and_tensorizers.py`

**改动 1**：`RolloutBatchView` 不再提前保存 `advantages / returns`，只保留原始 rollout 物料：

```python
@dataclass(frozen=True)
class RolloutBatchView:
    transitions: tuple[RolloutTransition, ...]
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
```

`RolloutBuffer.build_batch_view()` 现在只负责物化 `transitions / rewards / dones / values`，不再在混合序列上直接跑 GAE。

**改动 2**：在 `train_cmrappo.py` 中新增 rollout backlog 机制，只有当“当前 backlog 的每个 drone 尾部 bootstrap 都已可解析”时，才触发本次 PPO update。

核心思想是：

- rollout 期间把 transition 先暂存在 `rollout_backlog`
- 当 backlog 已达到 `rollout_steps` 且每条 per-drone 尾部都能解析 bootstrap 时，整体切成一个 update batch
- update 完成后清空 backlog，不把 lookahead transition 留到下一次 update，避免跨 update 的 off-policy 污染

**改动 3**：新增 `_resolve_rollout_prefix_bootstrap_values(...)`，显式为每个 `(episode_id, actor_drone_id, recurrent_segment_id)` 解析尾部 bootstrap：

优先级固定为：

1. 若同一 rollout backlog 后续已经出现了同一 drone / 同一 segment 的下一条 transition，则取其 `value_old`
2. 若同一 episode 已结束，或同一 drone 已进入新的 recurrent segment，则 bootstrap = 0
3. 若 backlog 尾部正好停在当前 env 的队首决策点，且该决策点属于同一 drone / 同一 segment，则现算 boundary value
4. 若该 drone 已发生 `airborne_energy_failure`，则 bootstrap = 0
5. 仅在训练总步数耗尽、无法继续收集样本时，允许对无法继续解析的尾部做 0-bootstrap 截断

**改动 4**：`_materialize_recurrent_sequences(...)` 现在先按
`(episode_id, actor_drone_id, recurrent_segment_id)` 分组，再在组内按 `local_decision_index` 排序，对每个组独立计算完整 GAE：

```python
drone_advantages, drone_returns = compute_gae(
    rewards=drone_rewards,
    dones=drone_dones,
    values=drone_values,
    last_value=tail_bootstrap_values[key],
    gamma=gamma,
    gae_lambda=gae_lambda,
)
```

算完完整 per-drone 序列后，才再按 `sequence_len` 做 chunking。也就是说：

- `done` / bootstrap 边界先服务于完整 per-drone trajectory
- recurrent sequence chunking 只是后续训练组织步骤

**改动 5**：补充回归测试，覆盖三类关键行为：

- 混合 rollout 中按 actor 分组后的 GAE 结果正确
- 同 actor 后继 transition 会被正确用作尾部 bootstrap
- backlog 边界 value 与 episode 结束归零语义都能正确处理

**设计说明**：

- 这次修复同时解决了两个层次的问题：
  - 混合序列上的跨 drone value 污染
  - per-drone 序列尾部在 rollout 截断处的 bootstrap 语义
- 由于 update batch 现在在“bootstrap 已可解析”后才提交，GAE 目标与 recurrent sequence 边界保持了一致的时序定义

---

### 修复 4：修正 `approx_kl` 语义并前移 early stopping

**文件**：`backend/training/train_cmrappo.py`、`backend/training/test_phase7_model_runtime.py`

**改动 1**：新增 `_compute_masked_approx_kl(...)` helper，改为有符号近似：

```python
def _compute_masked_approx_kl(
    *,
    old_log_probs,
    new_log_probs,
    valid_mask,
):
    return _masked_mean_tensor(old_log_probs - new_log_probs, valid_mask)
```

修复前：

```python
latest_approx_kl = _masked_mean_tensor(
    (old_log_probs - new_log_probs).abs(),
    valid_mask_flat,
)
```

修复后：

```python
approx_kl_t = _compute_masked_approx_kl(
    old_log_probs=old_log_probs,
    new_log_probs=new_log_probs,
    valid_mask=valid_mask_flat,
)
latest_approx_kl = float(approx_kl_t.detach().cpu().item())
```

**改动 2**：将 `target_kl` 检查前移到 `optimizer.step()` 之前：

修复前：

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
optimizer.step()

if latest_approx_kl > train_cfg.target_kl:
    stop_early = True
    break
```

修复后：

```python
if latest_approx_kl > train_cfg.target_kl:
    stop_early = True
    break

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
optimizer.step()
```

这保证了：一旦当前 minibatch 的策略偏移已经超过阈值，就**不执行**这次参数更新。

**改动 3**：补充两条针对性单测：

- `test_compute_masked_approx_kl_uses_signed_mean`
  - 验证 `approx_kl = mean(old_log_probs - new_log_probs)` 保留符号，不再错误使用 `abs()`
- `test_ppo_update_skips_optimizer_step_when_target_kl_exceeded`
  - 验证当 `approx_kl > target_kl` 时，`optimizer.step()` 不会被调用

**设计说明**：

- 这里保留的是 PPO 工程中常见的 `approx_kl` 监控口径，而不是更昂贵的精确 KL
- 由于 `valid_timestep_mask` 仍参与 `_masked_mean_tensor`，padding timestep 不会污染 KL 统计
- `early stop` 的职责是保护“当前 rollout 的重复利用强度”；它不会结束整个训练，只会提前结束当前 update 的后续 minibatch / epoch

---

### 修复 5：奖励归因强化版——按无人机分解 per-dt 成本

**文件**：`backend/training/env_adapter.py`

**设计原则**：

| 成本类型 | 归因对象 |
|---|---|
| T_overdue（已接单） | `order.assigned_vehicle_id`（owner drone） |
| T_overdue（未接单） | 仅进系统指标，不归因任何 drone |
| T_wait | 正在等待卡车的无人机自身 |
| T_idle（riding_with_truck 路径） | 显式 WAIT 的无人机自身 |
| T_idle（IDLE 路径） | 决策无人机自身（一次性结算） |
| T_queue | 排队的无人机自身 |
| T_fallback | fallback 中的无人机自身 |
| hard_overdue（已接单） | owner drone |
| hard_overdue（未接单） | 仅进系统指标 |
| reservation_timeout | 持有该 reservation 的无人机 |
| delivery_bonus | 完成送达的无人机 |
| hard_failure | 失败的无人机 |

**改动 1**：新增 `_agent_cost_accum` 累积器，在 `__init__` 和 `reset()` 中初始化/清零：

```python
# __init__ 中新增
self._agent_cost_accum: dict[DroneId, float] = {}

# reset() 中新增
self._agent_cost_accum.clear()
```

**改动 2**：`_settle_per_dt_rewards` 按无人机分别写入累积器，不再返回全局 per-dt reward 给调用方使用：

```python
# T_overdue：已接单归因给 owner，未接单仅进系统指标
for order in self._active_uav_orders():
    overdue_dt = max(0.0, t_next - max(t_prev, float(order.deadline)))
    owner = order.assigned_vehicle_id
    if owner and owner in self._drone_state:
        penalty = -self._cfg.lambda_overdue * overdue_dt
        self._agent_cost_accum[owner] = self._agent_cost_accum.get(owner, 0.0) + penalty
    # else: 未接单，仅累计 _episode_overdue_time_sec，不写 accum

# T_wait / T_queue / T_fallback / T_idle：各自归因给对应无人机
for drone_id, state in self._drone_state.items():
    if state == TrainingDroneState.WAITING_FOR_TRUCK:
        penalty = -self._cfg.lambda_wait * dt
        self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + penalty
    elif state == TrainingDroneState.QUEUEING_AT_HOST:
        penalty = -self._cfg.lambda_queue * dt
        self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + penalty
    elif state == TrainingDroneState.FALLBACK_RECOVERY:
        penalty = -self._cfg.lambda_miss * dt
        self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + penalty
    # ACTIVE_WAIT + riding_with_truck 路径同理
```

**改动 3**：`_advance_to_event` 中 delivery_bonus 和 hard_failure 也写入对应无人机的累积器：

```python
for drone_id, leg in delivery_ready:
    bonus = self._process_delivery_event(drone_id, leg)
    delivery_reward += bonus
    self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + bonus

for drone_id, _leg, _failure_time in hard_failure_ready:
    penalty = self._process_airborne_failure_event(drone_id)
    hard_failure_reward += penalty
    self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + penalty
```

`_advance_to_event` 的返回值保持 `_global_per_dt + event_reward`，供测试和 `_last_reward_breakdown` 观察，不影响 PPO 奖励路径。

**改动 4**：`_apply_hard_overdue_penalty` 在移除订单前先记录 owner，将惩罚写入对应无人机的累积器：

```python
for order_id in assigned_remove:
    owner = order_mgr.assigned_orders[order_id].assigned_vehicle_id  # 移除前先取
    penalty = self._force_remove_assigned_order(order_id, t_now)
    if owner and owner in self._drone_state:
        self._agent_cost_accum[owner] = self._agent_cost_accum.get(owner, 0.0) + penalty
# pending_remove（未接单）：不写 accum，仅进系统指标
```

**改动 5**：`_process_reservation_timeouts` 将 timeout 惩罚归因给持有 reservation 的无人机：

```python
timeout_penalty = -self._cfg.lambda_res_timeout * timeout_cost
self._agent_cost_accum[drone_id] = self._agent_cost_accum.get(drone_id, 0.0) + timeout_penalty
```

**改动 6**：`step()` 在 advance 前清零决策无人机的累积器，advance 结束后 pop 出其归因奖励：

```python
# advance 前清零
self._agent_cost_accum[deciding_drone_id] = 0.0

# T_idle（IDLE 路径）一次性写入
idle_penalty = -self._cfg.wait_idle_penalty_coef * delta_wait
self._agent_cost_accum[deciding_drone_id] = idle_penalty

# ... advance ...

# advance 后取出归因奖励
reward = self._agent_cost_accum.pop(deciding_drone_id, 0.0)
```

**设计说明**：
- 其他无人机在同一时间窗口内产生的成本留在各自的累积器中，等到它们自己的决策点时被取走
- `_advance_to_event` 的返回值（全局总奖励）仅用于 `_last_reward_breakdown` 记录和单元测试断言，不再进入 PPO reward 路径
- 未接单订单的 overdue 成本仅进系统指标（`_episode_overdue_time_sec`），不污染任何 drone 的 advantage 估计

---

## 四、预期改善

| 指标 | Run02 实际 | Run03 预期 |
|------|-----------|-----------|
| 训练订单多样性 | 单一固定序列 | 每 episode 不同序列 |
| value loss 收敛速度 | update 150 前居高不下 | 应在 update 20-30 内快速下降 |
| value loss 峰值 | 18230 | 应显著降低 |
| benchmark 泛化能力 | 过拟合单一序列 | 应有实质改善 |
| GAE bootstrap 语义 | 跨 drone 串线且尾部 bootstrap 错位 | 每个 drone / segment 独立闭合 |
| PPO 样本利用率 | 大量 update 被错误 early stop | 仅在真实 KL 超限时截断 |
| 单步参数更新风险 | 超限 minibatch 已先执行 step | 超阈值 minibatch 不再写入参数 |
| WAIT 动作偏差 | 被他人 overdue/fallback 错误惩罚 | 仅承担自身归因成本 |
| advantage 信噪比 | 跨 drone 成本污染 | 每 drone 独立归因，信号更干净 |

---

## 五、修复 6：熵计算不对称性（evaluate_actions）

### 问题描述

**位置**：`backend/training/model.py`，`evaluate_actions`

原始实现中，熵的计算是条件性的：

```python
entropy = root_dist.entropy()          # 所有步骤都有，最大 ln(2) ≈ 0.693

if dispatch_mask.any():
    entropy = entropy + order_dist.entropy() * dispatch_mask   # 仅 DISPATCH 步骤
    entropy = entropy + mode_dist.entropy() * dispatch_mask    # 仅 DISPATCH 步骤
    if recovery_dispatch_mask.any():
        entropy = entropy + recovery_dist.entropy() * recovery_dispatch_mask
```

这导致：
- **WAIT 步骤**：`entropy ≈ H(root)` ≤ 0.693 nats
- **DISPATCH 步骤**：`entropy = H(root) + H(order) + H(mode) + H(recovery)` ≈ 数 nats

当 policy 以 90% 概率选 WAIT 时，batch 平均熵被 0.693 主导，`entropy_coef=0.03` 的梯度信号极小，熵崩溃后无法被正则化项拉回。这是熵崩溃无法被修复的**结构性原因**，而非参数调整问题。

### 根本原因

层次策略的真实联合熵应为：

```
H(π) = H(root) + P(dispatch) * [H(order) + H(mode|order) + P(recovery_mode|order) * H(recovery|order)]
```

原实现用 `dispatch_mask`（0/1 的实际动作）代替了 `p_dispatch`（连续概率），导致 WAIT 步骤完全丢失了 dispatch 子策略的熵梯度。

### 修复内容

**文件**：`backend/training/model.py`，`evaluate_actions`（436-463 行）

将熵计算从"按实际动作条件化"改为"按 P(dispatch) 加权的无条件期望熵"：

```python
# 修复前：dispatch 子策略熵仅对 DISPATCH 步骤贡献
entropy = root_dist.entropy()
if dispatch_mask.any():
    entropy = entropy + order_dist.entropy() * dispatch_mask
    entropy = entropy + mode_dist.entropy() * dispatch_mask
    ...

# 修复后：所有步骤都贡献完整的联合熵，WAIT 步骤通过 p_dispatch 加权
p_dispatch = root_dist.probs[:, 1]
entropy = root_dist.entropy()
entropy = entropy + p_dispatch * order_dist.entropy()
entropy = entropy + p_dispatch * mode_dist.entropy()
p_recovery_mode = mode_dist.probs[:, 1]
entropy = entropy + p_dispatch * p_recovery_mode * recovery_dist.entropy()
```

同时移除了 `if dispatch_mask.any():` 和 `if recovery_dispatch_mask.any():` 两个条件分支——order/mode/recovery 分布对所有步骤都无条件计算。

**log_prob 计算保持不变**：WAIT 步骤不加 sub-action 的 log_prob，这是正确的（WAIT 动作没有子动作）。

### 效果

| 场景 | 修复前 entropy | 修复后 entropy |
|------|--------------|--------------|
| 90% WAIT batch | ≈ 0.693（被 WAIT 主导） | H(root) + 0.1 * H(dispatch_sub) |
| 50% WAIT batch | ≈ 0.5 * 0.693 + 0.5 * (0.693 + H_sub) | H(root) + 0.5 * H(dispatch_sub) |
| 全 DISPATCH batch | H(root) + H_sub | H(root) + 1.0 * H(dispatch_sub) |

修复后，即使 policy 以 90% 概率选 WAIT，`p_dispatch=0.1` 仍会把 order/mode/recovery 的熵梯度传回网络，推动 dispatch 分支保持探索，`entropy_coef` 正则化项得以真正生效。

---

## 六、待后续处理的已知问题

以下问题已在代码审查中识别，留待后续 run 处理：

1. **Reservation timeout 高发**：alpha=1.5、beta=1.2 的预约窗口估算过于保守，需要调整

2. **benchmark eval 3 个 episode 完全相同**：固定 seed + greedy 推理导致结果完全一致，`benchmark_eval_episodes=3` 无额外信息价值
