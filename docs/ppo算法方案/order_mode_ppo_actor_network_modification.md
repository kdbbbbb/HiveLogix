# Order-Mode PPO Actor 网络结构修改方案

## 0. 修改目标
（已完成）
当前目标不是把 PPO actor 改成复杂的 Transformer 或 GNN，而是先形成一个**干净、稳定、可解释的 order-mode PPO actor**。

修改后的 actor 只做两个核心决策：

```text
1. 是否派送：root ∈ {WAIT, DISPATCH}
2. 如果派送：选择 order，并选择 mode ∈ {B, C}
```

也就是说，修改后的策略网络不再决策 recovery pool，不再输出 recovery index，也不再把 recovery pool 明细作为 actor observation 输入。

最终机制应该是：

```text
PPO actor 决策：
  WAIT
  或 DispatchAction(order_id, mode)

环境 / 规则模块负责：
  如果 mode = C，则在送达后根据实时 truck backbone、能量、等待时间等规则选择具体 recovery node。
```

这使得网络机制、动作空间、论文表述保持一致：

```text
PPO 学习 order-mode 决策；
mode C 的具体回收点不是 PPO 的动作，而是环境执行层的后置规则选择。
```

---

## 1. 修改后的整体机制

修改后的数据流应当是：

```text
CandidateBuilder
  ├─ 基于 runtime_state、coarse_plan、当前无人机构造候选订单
  ├─ 对每个 order 判断 mode B 是否可行
  ├─ 对每个 order 判断 mode C 是否可行
  ├─ 不再把 recovery pool 明细作为 actor 输入
  └─ 将 mode C 可行性与风险压缩成 order-level summary

ObservationTensorizer
  ├─ uav_self_token
  ├─ order_tokens
  │    ├─ order 基础信息
  │    ├─ mode B 摘要
  │    └─ mode C 摘要
  ├─ infra_tokens
  ├─ history_tokens
  └─ padding_mask

SharedPPOActorCritic
  ├─ uav_proj
  ├─ order_proj
  ├─ infra_proj
  ├─ history_proj
  ├─ order_mean / order_max / order_logmeanexp
  ├─ base_context
  ├─ recurrent_core
  ├─ context
  ├─ root_head(context)
  ├─ order_head(context, order_embed)
  └─ mode_head(context, order_embed)

ActionMask
  ├─ root_branch_mask
  ├─ order_mask
  └─ mode_mask

ResolvedActionLookup
  └─ root / order / mode → WAIT 或 DispatchAction(order_id, mode)
```

修改后的机制可以概括为：

```text
候选层负责“合法性与摘要”；
张量层负责“固定 schema 与归一化”；
网络层负责“order-mode 决策”；
环境层负责“mode C 执行细节与后置 recovery node 选择”。
```

---

## 2. 为什么必须这样改

旧版 actor 的核心结构是：

```text
uav_embed
order_summary = masked_mean(order_embed)
infra_summary = mean(infra_embed)
history_summary = LSTM(history_embed)

base_context = context_proj(
    uav_embed
    + order_summary
    + infra_summary
    + history_summary
)

recurrent_context = recurrent_core(base_context)
context = base_context + recurrent_context

root_head(context)
order_head(context + each_order_embed)
mode_head(context + each_order_embed)
```

旧版最大问题是：

```text
order_summary 只使用 masked_mean(order_embed)
```

如果当前有 32 个候选订单，其中只有 1 个订单非常紧急、非常适合派送，那么 mean pooling 会把这个强信号稀释掉。

这会直接影响 root head：

```text
root_head 只能看到“平均订单状态”，看不到“是否存在一个非常值得派送的订单”。
```

所以修改后的核心思路是：

```text
不仅看订单平均状态，还要看订单集合中的强信号。
```

也就是将：

```text
order_summary = masked_mean(order_embed)
```

改成：

```text
order_mean = masked_mean(order_embed)
order_max = masked_max(order_embed)
order_lse = masked_logmeanexp(order_embed)
```

其中：

```text
order_mean：
  表示候选订单集合的整体状态。

order_max：
  表示候选订单集合中最强的局部信号。

order_logmeanexp：
  表示比 mean 更关注高值订单、但比 max 更平滑的聚合结果。
```

注意，推荐使用 `logmeanexp`，而不是裸 `logsumexp`。

原因是：

```text
logsumexp 会随候选订单数量增加而自然变大；
logmeanexp = logsumexp - log(valid_count)，可以削弱候选数量变化带来的尺度干扰。
```

---

## 3. 修改后的网络结构

修改后的 actor 网络应当变成：

```text
uav_embed = uav_proj(uav_self_token)
order_embed = order_proj(order_tokens)
infra_embed = infra_proj(infra_tokens)
history_embed = history_proj(history_tokens)

history_summary = history_encoder(history_embed)

order_valid_mask = ~order_padding_mask

order_mean = masked_mean(order_embed, order_valid_mask)
order_max = masked_max(order_embed, order_valid_mask)
order_lse = masked_logmeanexp(order_embed, order_valid_mask)

infra_summary = mean(infra_embed)

base_context = context_proj(
    concat(
        uav_embed,
        order_mean,
        order_max,
        order_lse,
        infra_summary,
        history_summary,
    )
)

recurrent_out = recurrent_core(base_context)
recurrent_context = recurrent_proj(recurrent_out)

context = base_context + recurrent_context

root_branch_logits = root_head(context)

order_logits = order_head(
    concat(context, each_order_embed)
)

mode_logits = mode_head(
    concat(context, each_order_embed)
)

value = critic_head(...)
```

这里第一版只增强全局 context，不建议马上改 order_head 和 mode_head。

也就是说：

```text
第一阶段只改 order aggregation；
不要同时引入 self-attention、cross-attention、dropout、GNN。
```

原因是当前训练还存在 loss 不收敛问题。此时应优先做可控修改，避免引入太多新变量。

---

## 4. model.py 修改方案

### 4.1 修改 `context_proj` 输入维度

旧版：

```python
self.context_proj = nn.Sequential(
    nn.Linear(d_model * 3 + lstm_hidden, ff_dim),
    nn.ReLU(),
    nn.Linear(ff_dim, d_model),
    nn.ReLU(),
)
```

旧版输入是：

```text
uav_embed        d_model
order_summary   d_model
infra_summary   d_model
history_summary lstm_hidden
```

新版输入是：

```text
uav_embed        d_model
order_mean       d_model
order_max        d_model
order_lse        d_model
infra_summary    d_model
history_summary  lstm_hidden
```

所以应改成：

```python
self.context_proj = nn.Sequential(
    nn.Linear(d_model * 5 + lstm_hidden, ff_dim),
    nn.ReLU(),
    nn.Linear(ff_dim, d_model),
    nn.ReLU(),
)
```

第一阶段不建议加入 dropout。

如果后续训练稳定，再考虑：

```python
nn.Dropout(p=0.05)
```

但不要在 loss 尚未稳定前加入。

---

### 4.2 增加 `_masked_max`

```python
def _masked_max(values: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
    """
    values:
        shape = [B, T, N, D]

    valid_mask:
        shape = [B, T, N]
        True 表示有效订单，False 表示 padding。

    return:
        shape = [B, T, D]
    """
    mask = valid_mask.unsqueeze(-1)
    has_valid = valid_mask.any(dim=dim, keepdim=False).unsqueeze(-1)

    masked_values = values.masked_fill(~mask, -1e9)
    pooled = masked_values.max(dim=dim).values

    return torch.where(has_valid, pooled, torch.zeros_like(pooled))
```

作用：

```text
从候选订单集合中提取最强订单信号；
避免紧急订单、强可行订单被 mean pooling 稀释。
```

---

### 4.3 增加 `_masked_logmeanexp`

```python
def _masked_logmeanexp(values: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
    """
    稳定版 logsumexp pooling。

    logmeanexp = logsumexp - log(valid_count)

    这样可以避免候选订单数量越多，pooled 特征天然越大。
    """
    mask = valid_mask.unsqueeze(-1)
    has_valid = valid_mask.any(dim=dim, keepdim=False).unsqueeze(-1)

    safe_values = values.masked_fill(~mask, -1e9)
    pooled = torch.logsumexp(safe_values, dim=dim)

    count = valid_mask.sum(dim=dim).clamp(min=1).to(dtype=values.dtype).unsqueeze(-1)
    pooled = pooled - torch.log(count)

    return torch.where(has_valid, pooled, torch.zeros_like(pooled))
```

作用：

```text
比 mean 更关注高价值订单；
比 max 更平滑；
比 logsumexp 更稳定，不直接受到候选订单数量影响。
```

---

### 4.4 修改 `forward_sequence()` 中的 order summary

旧版：

```python
order_summary = _masked_mean(order_embed, ~order_padding_mask, dim=2)
infra_summary = infra_embed.mean(dim=2)

base_context = self.context_proj(
    torch.cat(
        [uav_embed, order_summary, infra_summary, history_summary],
        dim=-1,
    )
)
```

新版：

```python
order_valid_mask = ~order_padding_mask

order_mean = _masked_mean(order_embed, order_valid_mask, dim=2)
order_max = _masked_max(order_embed, order_valid_mask, dim=2)
order_lse = _masked_logmeanexp(order_embed, order_valid_mask, dim=2)

infra_summary = infra_embed.mean(dim=2)

base_context = self.context_proj(
    torch.cat(
        [
            uav_embed,
            order_mean,
            order_max,
            order_lse,
            infra_summary,
            history_summary,
        ],
        dim=-1,
    )
)
```

后续结构第一阶段保持不变：

```python
recurrent_out, next_lstm_state = self.recurrent_core(base_context, lstm_state)
recurrent_context = self.recurrent_proj(recurrent_out)
context = base_context + recurrent_context

root_branch_logits = self.root_head(context)

context_per_order = context.unsqueeze(2).expand(-1, -1, order_embed.size(2), -1)

order_logits = self.order_head(
    torch.cat([context_per_order, order_embed], dim=-1)
).squeeze(-1)

mode_logits = self.mode_head(
    torch.cat([context_per_order, order_embed], dim=-1)
)
```

---

## 5. mode C 信息如何压缩进 order token

因为 actor 不再读取 recovery pool，所以 mode C 的关键信息必须变成 order-level summary。

也就是说，修改后的 actor 不应该看到：

```text
recovery_tokens
recovery_padding_mask
每个订单的 recovery node 明细
```

而应该看到：

```text
这个订单是否可以走 mode C？
mode C 候选数量多不多？
最佳 mode C 等待时间多长？
最佳 mode C 能量余量是否充足？
最佳 mode C 与卡车时间匹配是否稳定？
mode C 是否有 timeout 风险？
```

因此，`order_tokens` 中必须包含 mode C 摘要字段。

---

## 6. ORDER_TOKEN_FIELDS 新版设计

建议新版 `ORDER_TOKEN_FIELDS` 为：

```python
ORDER_TOKEN_FIELDS = (
    # order 基础信息
    "is_valid",
    "weight_norm",
    "deadline_slack_norm",
    "delivery_x_norm",
    "delivery_y_norm",
    "delivery_z_norm",
    "distance_to_order_norm",
    "order_pre_score_norm",
    "priority_band_norm",

    # mode B 摘要
    "has_mode_b_action",
    "best_mode_b_return_score_norm",
    "best_mode_b_host_type_code_norm",
    "best_mode_b_queue_time_est_norm",

    # mode C 摘要
    "has_mode_c_action",
    "mode_c_candidate_count_norm",
    "best_mode_c_rendezvous_margin_norm",
    "best_mode_c_wait_time_norm",
    "best_mode_c_uav_flight_time_norm",
    "best_mode_c_energy_margin_ratio",
    "best_mode_c_node_type_code_norm",
    "best_mode_c_truck_eta_remaining_norm",
    "best_mode_c_timeout_risk_norm",
)
```

这几个新增字段的含义如下。

### 6.1 `mode_c_candidate_count_norm`

表示当前订单可用的 mode C recovery 候选数量。

意义：

```text
候选越多，说明 mode C 越稳；
候选越少，说明 mode C 对 truck route 和 timing 更敏感。
```

归一化建议：

```python
mode_c_candidate_count_norm = clip01(
    mode_c_candidate_count / max_candidate_recovery_per_order
)
```

---

### 6.2 `best_mode_c_wait_time_norm`

表示最佳 mode C 候选下，UAV 到达 recovery node 后需要等待 truck 的时间。

意义：

```text
等待时间越长，mode C 越可能带来 T_wait 惩罚；
等待时间过短或为负，可能说明 truck 已经过站或 rendezvous 不稳定。
```

归一化建议：

```python
best_mode_c_wait_time_norm = norm_time_nonneg(best_mode_c_wait_time)
```

---

### 6.3 `best_mode_c_uav_flight_time_norm`

表示送达后 UAV 从客户点飞到最佳 recovery node 的时间。

意义：

```text
飞行时间越长，能耗越高，也越容易受到 truck 到达时间变化影响。
```

归一化建议：

```python
best_mode_c_uav_flight_time_norm = norm_time_nonneg(best_mode_c_uav_flight_time)
```

---

### 6.4 `best_mode_c_energy_margin_ratio`

表示 UAV 完成送达后，再飞到 recovery node 之后的能量余量比例。

建议定义为：

```text
energy_margin_ratio =
    (energy_after_delivery - energy_to_recover - safe_margin)
    / battery_max
```

意义：

```text
越接近 0，mode C 越危险；
越大，mode C 能量越安全。
```

归一化建议：

```python
best_mode_c_energy_margin_ratio = clip01(
    max(0.0, best_mode_c_energy_margin_ratio)
)
```

---

### 6.5 `best_mode_c_timeout_risk_norm`

表示 mode C rendezvous 超时风险。

可以简单定义为：

```python
timeout_risk = 1.0 - min(
    1.0,
    max(0.0, best_rendezvous_margin) / rendezvous_max_wait_sec,
)
```

含义：

```text
margin 越大，风险越小；
margin 越接近 0，风险越大；
margin 为负时，风险接近 1。
```

---

## 7. contracts.py 修改方案

### 7.1 修改 `OrderFeatures`

建议将 `OrderFeatures` 改成：

```python
@dataclass(frozen=True)
class OrderFeatures:
    order_id: str
    weight: float
    deadline: float
    remaining_time: float
    delivery_x: float
    delivery_y: float
    delivery_z: float
    distance_to_order: float
    order_pre_score: float
    priority_band: int

    # mode B 摘要
    has_mode_b_action: bool
    best_mode_b_return_score: float
    best_mode_b_host_type: str
    best_mode_b_queue_time_est: float

    # mode C 摘要
    has_mode_c_action: bool
    mode_c_candidate_count: int
    best_mode_c_rendezvous_margin: float
    best_mode_c_wait_time: float
    best_mode_c_uav_flight_time: float
    best_mode_c_energy_margin_ratio: float
    best_mode_c_node_type: str
    best_mode_c_truck_eta_remaining: float
    best_mode_c_timeout_risk: float

    is_valid: bool
```

### 7.2 修改 padding order feature

`padding_order_feature()` 中需要同步补默认值：

```python
mode_c_candidate_count=0,
best_mode_c_wait_time=0.0,
best_mode_c_uav_flight_time=0.0,
best_mode_c_energy_margin_ratio=0.0,
best_mode_c_timeout_risk=0.0,
```

### 7.3 修改 `ObservationBatch`

如果已经彻底删除 actor 侧 recovery pool，那么 `ObservationBatch` 应改成：

```python
@dataclass(frozen=True)
class ObservationBatch:
    """Actor 输入张量。所有 tensor 已 materialize。"""

    uav_self_token: Any
    order_tokens: Any
    infra_tokens: Any
    history_tokens: Any
    history_padding_mask: Any
    padding_mask: Any
```

不再包含：

```text
recovery_tokens
recovery_padding_mask
```

### 7.4 保持 `FactorizedActionMask` 不变

`FactorizedActionMask` 应继续保持：

```python
@dataclass(frozen=True)
class FactorizedActionMask:
    root_branch_mask: Any
    order_mask: Any
    mode_mask: Any
```

因为修改后的动作空间就是：

```text
root / order / mode
```

不需要 recovery mask。

### 7.5 保持 `ResolvedActionLookup` 不变

`ResolvedActionLookup.resolve()` 应继续只根据：

```text
root_branch_idx
order_idx
mode_idx
```

解析动作。

也就是说：

```text
root/order/mode → WAIT 或 DispatchAction(order_id, mode)
```

不需要 recovery_idx。

---

## 8. candidate_builder.py 修改方案

`CandidateBuilder` 的职责应从：

```text
生成 recovery pool 并交给 actor
```

改成：

```text
基于 recovery 可行性计算 mode C 摘要，并写入 OrderFeatures
```

### 8.1 建议新增 `_ModeCSummary`

```python
@dataclass(frozen=True)
class _ModeCSummary:
    candidate_count: int
    best_rendezvous_margin: float
    best_wait_time: float
    best_uav_flight_time: float
    best_energy_margin_ratio: float
    best_node_type: str
    best_truck_eta_remaining: float
    timeout_risk: float
```

### 8.2 mode C 摘要应从真实可行性计算得到

mode C summary 不应该凭字段名猜，也不应该手工造假。

应从以下真实链路得到：

```text
当前无人机位置
→ 订单 delivery_loc
→ 送达飞行时间与能耗
→ 送达后剩余电量
→ 可选 recovery node
→ 从 delivery_loc 到 recovery node 的飞行时间与能耗
→ truck 到达 recovery node 的时间
→ planned_wait
→ rendezvous_margin
→ timeout_risk
```

也就是：

```text
energy_after_delivery
energy_to_recover
safe_margin
planned_truck_arrival_time
planned_uav_arrival_time
rendezvous_execution_margin
```

这些值必须来自 `CandidateBuilder` 里真实的可达性与时间计算。

---

### 8.3 计算能量余量

建议在 mode C 候选构造中计算：

```python
recover_leg = self._estimate_uav_leg(
    drone_view=drone_view,
    from_pos=deliver_pos,
    to_pos=node_state.position,
    payload=0.0,
)

uav_flight_time = float(recover_leg.flight_time_sec)
energy_to_recover = float(recover_leg.energy_j)

safe_margin = float(self._safe_margin_j_by_drone.get(drone_view.drone_id, 0.0))

energy_margin_j = (
    float(energy_after_delivery)
    - energy_to_recover
    - safe_margin
)

energy_margin_ratio = energy_margin_j / max(float(drone_view.battery_max), _TIME_EPS)
```

然后用：

```python
if energy_margin_j < -_TIME_EPS:
    continue
```

过滤不可达 mode C 候选。

这样 mode C 的能量逻辑更清楚：

```text
送达后剩余电量
- 去 recovery node 的能耗
- 安全裕度
= mode C 能量余量
```

---

### 8.4 计算 timeout risk

建议：

```python
timeout_risk = 1.0 - min(
    1.0,
    max(0.0, best_rendezvous_margin)
    / max(self._cfg.rendezvous_max_wait_sec, _TIME_EPS),
)
```

含义：

```text
best_rendezvous_margin 越大，timeout_risk 越小；
best_rendezvous_margin 越小，timeout_risk 越大；
best_rendezvous_margin <= 0 时，timeout_risk 接近 1。
```

---

### 8.5 构造 OrderFeatures 时写入 mode C summary

构造 `OrderFeatures` 时，应由：

```python
has_mode_c_action=mode_c_summary is not None,
best_mode_c_rendezvous_margin=...,
best_mode_c_node_type=...,
best_mode_c_truck_eta_remaining=...,
```

升级为：

```python
has_mode_c_action=mode_c_summary is not None,

mode_c_candidate_count=(
    int(mode_c_summary.candidate_count)
    if mode_c_summary is not None
    else 0
),

best_mode_c_rendezvous_margin=(
    float(mode_c_summary.best_rendezvous_margin)
    if mode_c_summary is not None
    else 0.0
),

best_mode_c_wait_time=(
    float(mode_c_summary.best_wait_time)
    if mode_c_summary is not None
    else 0.0
),

best_mode_c_uav_flight_time=(
    float(mode_c_summary.best_uav_flight_time)
    if mode_c_summary is not None
    else 0.0
),

best_mode_c_energy_margin_ratio=(
    float(mode_c_summary.best_energy_margin_ratio)
    if mode_c_summary is not None
    else 0.0
),

best_mode_c_node_type=(
    str(mode_c_summary.best_node_type)
    if mode_c_summary is not None
    else ""
),

best_mode_c_truck_eta_remaining=(
    float(mode_c_summary.best_truck_eta_remaining)
    if mode_c_summary is not None
    else 0.0
),

best_mode_c_timeout_risk=(
    float(mode_c_summary.timeout_risk)
    if mode_c_summary is not None
    else 0.0
),
```

---

## 9. observation_tensorizer.py 修改方案

### 9.1 删除 actor 侧 recovery token

`ObservationTensorizer.build()` 不应再构造：

```python
recovery_tokens, recovery_padding_mask = self._build_recovery_tokens(...)
```

返回值应变成：

```python
return ObservationBatch(
    uav_self_token=uav_self_token,
    order_tokens=order_tokens,
    infra_tokens=infra_tokens,
    history_tokens=history_tokens,
    history_padding_mask=history_padding_mask,
    padding_mask=order_padding_mask,
)
```

同时可以删除：

```text
RECOVERY_TOKEN_FIELDS
_build_recovery_tokens()
recovery_padding_mask
```

如果环境执行层仍然需要 recovery pool，那也不应该通过 actor observation 传给模型。

---

### 9.2 修改 `_build_order_tokens()`

新增字段后，`_build_order_tokens()` 应同步填入：

```python
tokens[idx, :] = np.asarray(
    [
        # order 基础信息
        1.0,
        self._clip01(float(item.weight) / max(self._cfg.payload_norm_kg, _TIME_EPS)),
        self._norm_time_signed(float(item.remaining_time)),
        self._norm_x(float(item.delivery_x)),
        self._norm_y(float(item.delivery_y)),
        self._norm_z(float(item.delivery_z)),
        self._norm_distance(float(item.distance_to_order)),
        self._norm_time_nonneg(float(item.order_pre_score)),
        self._clip01(float(item.priority_band) / 2.0),

        # mode B 摘要
        self._bool(bool(item.has_mode_b_action)),
        self._norm_time_nonneg(float(item.best_mode_b_return_score)),
        self._code_norm(_HOST_TYPE_CODE, str(item.best_mode_b_host_type)),
        self._norm_time_nonneg(float(item.best_mode_b_queue_time_est)),

        # mode C 摘要
        self._bool(bool(item.has_mode_c_action)),
        self._clip01(
            float(item.mode_c_candidate_count)
            / max(self._cfg.max_candidate_recovery_per_order, 1)
        ),
        self._norm_time_signed(float(item.best_mode_c_rendezvous_margin)),
        self._norm_time_nonneg(float(item.best_mode_c_wait_time)),
        self._norm_time_nonneg(float(item.best_mode_c_uav_flight_time)),
        self._clip01(max(0.0, float(item.best_mode_c_energy_margin_ratio))),
        self._code_norm(_HOST_TYPE_CODE, str(item.best_mode_c_node_type)),
        self._norm_time_nonneg(float(item.best_mode_c_truck_eta_remaining)),
        self._clip01(float(item.best_mode_c_timeout_risk)),
    ],
    dtype=_FLOAT_DTYPE,
)
```

需要确保：

```text
ORDER_TOKEN_FIELDS 的字段数量
=
_build_order_tokens() 写入的特征数量
=
model 初始化时 order_feat_dim
```

---

## 10. train_cmrappo.py 修改方案

由于 `ObservationBatch` 删除了 recovery 字段，训练侧所有 stack / zero / sequence padding 都要同步删除 recovery 相关字段。

### 10.1 修改 observation batch stacking

旧版可能存在：

```python
recovery_tokens=np.stack([item.recovery_tokens for item in step_batches], axis=0)
recovery_padding_mask=np.stack([item.recovery_padding_mask for item in step_batches], axis=0)
```

应删除，改为：

```python
def _stack_observation_batches_from_steps(step_batches: Sequence[Any]) -> Any:
    cls = step_batches[0].__class__
    return cls(
        uav_self_token=np.stack(
            [item.uav_self_token for item in step_batches],
            axis=0,
        ),
        order_tokens=np.stack(
            [item.order_tokens for item in step_batches],
            axis=0,
        ),
        infra_tokens=np.stack(
            [item.infra_tokens for item in step_batches],
            axis=0,
        ),
        history_tokens=np.stack(
            [item.history_tokens for item in step_batches],
            axis=0,
        ),
        history_padding_mask=np.stack(
            [item.history_padding_mask for item in step_batches],
            axis=0,
        ),
        padding_mask=np.stack(
            [item.padding_mask for item in step_batches],
            axis=0,
        ),
    )
```

### 10.2 修改 zero observation

旧版如果构造 recovery zero tensor，应删除。

新版：

```python
def _zero_observation_batch_like(template: Any) -> Any:
    cls = template.__class__
    return cls(
        uav_self_token=np.zeros_like(template.uav_self_token),
        order_tokens=np.zeros_like(template.order_tokens),
        infra_tokens=np.zeros_like(template.infra_tokens),
        history_tokens=np.zeros_like(template.history_tokens),
        history_padding_mask=np.ones_like(
            template.history_padding_mask,
            dtype=np.bool_,
        ),
        padding_mask=np.ones_like(
            template.padding_mask,
            dtype=np.bool_,
        ),
    )
```

### 10.3 检查 sequence 相关函数

需要检查并同步修改：

```text
_stack_observation_batches_from_steps()
_stack_sequence_observation_rows()
_zero_observation_batch_like()
_materialize_sequences()
_build_sequence_minibatch()
```

原则是：

```text
凡是出现 recovery_tokens / recovery_padding_mask 的地方，都应该删除；
除非该字段只用于环境执行层 debug，而不是 actor observation。
```

---

## 11. policy_inference.py 修改方案

推理侧也要保持与训练侧一致。

需要检查：

```text
load_trained_policy()
run_policy_episode()
bootstrap_observation
model 初始化 order_feat_dim
ObservationTensorizer.build()
```

原则：

```text
训练时 actor 输入是什么，推理时 actor 输入必须完全一致。
```

如果 `ObservationBatch` 删除了 recovery 字段，推理侧不能再访问：

```text
observation.recovery_tokens
observation.recovery_padding_mask
```

否则训练能跑，推理会崩。

---

## 12. 配置文件修改方案

旧版配置类似：

```yaml
policy:
  encoder_type: attn_lstm_lite
  d_model: 128
  nhead: 8
  ff_dim: 256
  dropout: 0.10
  lstm_hidden: 128
  lstm_layers: 1
  hist_len: 6
```

但当前网络并没有真正使用 attention，也没有真正使用 nhead/dropout。

建议第一阶段改成：

```yaml
policy:
  encoder_type: pool_lstm_v2
  d_model: 128
  ff_dim: 256
  dropout: 0.0
  lstm_hidden: 128
  lstm_layers: 1
  hist_len: 6
```

如果短期内 `PolicyMeta` 或配置解析仍然要求 `nhead`，可以临时保留：

```yaml
nhead: 8  # legacy only, pool_lstm_v2 does not use it
```

但更干净的做法是：

```text
如果当前模型没有 attention，就不要让 PolicyMeta 强制检查 nhead。
```

推荐命名：

```text
pool_lstm_v2
```

不要继续叫：

```text
attn_lstm_lite
```

否则文档、配置和真实模型不一致。

---

## 13. 不建议现在直接上 attention

当前主要问题是：

```text
loss 大；
PPO 不稳定；
value loss 和 policy loss 都难以下降。
```

此时不建议马上改成：

```text
Transformer
MultiHeadAttention
GNN
order-infra cross-attention
```

因为这会同时引入：

```text
更多参数；
更多 mask 细节；
更复杂的 dropout；
更难定位的 shape bug；
更强的过拟合风险。
```

建议当前阶段只做：

```text
pool_lstm_v2:
  order_mean
  order_max
  order_logmeanexp
  LSTM
  root/order/mode heads
```

等该版本稳定后，再考虑：

```text
attn_lstm_v3:
  order self-attention
  order-infra cross-attention
  LSTM
  factorized heads
```

---

## 14. 修改后的机制总结

修改后的机制应明确为：

```text
1. Actor 只做 order-mode 决策。
2. Actor 不决策 recovery node。
3. Actor 不读取 recovery pool 明细。
4. mode C 的可行性、风险、等待时间、能量余量等信息全部压缩进 order token。
5. root head 不再只看 mean pooled order summary，而是同时看：
   - order_mean
   - order_max
   - order_logmeanexp
6. order_head 和 mode_head 第一阶段保持原结构，减少额外不稳定因素。
7. 环境执行层负责 mode C 送达后的具体 recovery node 选择。
8. ActionMask 仍然保持 root/order/mode 三段式。
9. ResolvedActionLookup 仍然只解析 root/order/mode。
10. 配置名称从 attn_lstm_lite 改为 pool_lstm_v2，避免名实不符。
```

最终 actor 可以表述为：

```text
基于候选订单集合的多池化上下文增强型 recurrent factorized PPO actor。
```

或者更正式地写成：

```text
A pooled-recurrent factorized PPO actor for order-mode dispatching.
```

它的核心不是直接选择 truck-rendezvous recovery node，而是：

```text
利用 order-level mode C feasibility summary 学习订单与配送模式选择。
```

---

## 15. 修改检查清单

### 15.1 model.py

- [ ] `context_proj` 输入维度由 `d_model * 3 + lstm_hidden` 改为 `d_model * 5 + lstm_hidden`
- [ ] 新增 `_masked_max`
- [ ] 新增 `_masked_logmeanexp`
- [ ] `forward_sequence()` 中新增 `order_mean / order_max / order_lse`
- [ ] `base_context` 拼接新版 order pooling
- [ ] 第一阶段不修改 `order_head`
- [ ] 第一阶段不修改 `mode_head`
- [ ] 第一阶段不引入 attention

### 15.2 contracts.py

- [ ] `OrderFeatures` 新增 mode C summary 字段
- [ ] `padding_order_feature()` 新增默认值
- [ ] `ObservationBatch` 删除 `recovery_tokens`
- [ ] `ObservationBatch` 删除 `recovery_padding_mask`
- [ ] `FactorizedActionMask` 保持不变
- [ ] `ResolvedActionLookup` 保持 root/order/mode 解析逻辑

### 15.3 candidate_builder.py

- [ ] 不再把 recovery pool 明细交给 actor
- [ ] 新增 `_ModeCSummary` 或等价结构
- [ ] 计算 mode C candidate count
- [ ] 计算 best wait time
- [ ] 计算 best UAV flight time
- [ ] 计算 energy margin ratio
- [ ] 计算 reservation count
- [ ] 计算 timeout risk
- [ ] 将 mode C summary 写入 `OrderFeatures`

### 15.4 observation_tensorizer.py

- [ ] 删除 `RECOVERY_TOKEN_FIELDS`
- [ ] 删除或停用 `_build_recovery_tokens()`
- [ ] `ObservationTensorizer.build()` 不再返回 recovery 字段
- [ ] `ORDER_TOKEN_FIELDS` 新增 mode C summary 字段
- [ ] `_build_order_tokens()` 写入新增字段
- [ ] 确保字段数量和 token shape 一致

### 15.5 train_cmrappo.py

- [ ] stack observation 时删除 recovery 字段
- [ ] zero observation 时删除 recovery 字段
- [ ] sequence padding 时删除 recovery 字段
- [ ] minibatch 构造时删除 recovery 字段
- [ ] 确保训练侧 actor 输入与 `ObservationBatch` 新结构一致

### 15.6 policy_inference.py

- [ ] 推理侧不再访问 recovery 字段
- [ ] bootstrap observation shape 与训练一致
- [ ] model 初始化的 `order_feat_dim` 自动读取新版 order token 维度

### 15.7 rh_alns_cmrappo.yaml

- [ ] `encoder_type` 改为 `pool_lstm_v2`
- [ ] 如果不用 attention，删除或标注 `nhead`
- [ ] `dropout` 第一阶段设为 `0.0`
- [ ] 保持 `d_model=128`
- [ ] 保持 `ff_dim=256`
- [ ] 保持 `lstm_hidden=128`
- [ ] 保持 `lstm_layers=1`

---

## 16. 建议的修改顺序

结合当前代码状态，推荐按以下顺序修改。

当前代码中已经完成的部分包括：

```text
1. Actor observation 已经没有 recovery_tokens / recovery_padding_mask。
2. FactorizedActionMask 已经是 root / order / mode。
3. ResolvedActionLookup 已经只用 root / order / mode 解析动作。
4. CandidateBuilder 生成的 mode C DispatchAction 当前不携带 recover_node_id。
5. mode C 送达后的 recovery node 选择已经在 env_adapter 执行层后置完成。
```

因此，当前不应再把重点放在“删除 actor recovery token”上，而应重点补齐：

```text
1. mode C order-level summary 是否足够完整；
2. order token schema 是否同步扩展；
3. actor 的 order aggregation 是否从 mean-only 升级为 mean / max / logmeanexp；
4. 配置命名是否与真实模型一致。
```

注意：

```text
recovery_pool 当前仍可以保留在 CoarsePlanView / CandidateBuilder 内部，
作为 mode C 可行性粗边界和候选来源。

它不应作为 actor observation 明细输入，
也不应变成 PPO 动作空间中的 recovery index。
```

推荐落地顺序如下：

```text
第 0 步：先做现状核验，不改行为
  目标：
    确认当前代码确实没有 actor recovery token / recovery mask / recovery index。

  需要核验：
    contracts.py
      ObservationBatch 不包含 recovery_tokens / recovery_padding_mask。
      FactorizedActionMask 只包含 root_branch_mask / order_mask / mode_mask。
      ResolvedActionLookup.resolve() 只解析 root_branch_idx / order_idx / mode_idx。

    candidate_builder.py
      mode C DispatchAction 不携带 recover_node_id。

    env_adapter.py
      mode C 送达后由执行层选择 recover node。

  这一步只是防止误改，不需要重构。

第一步：扩展 contracts.py 中的 OrderFeatures
  contracts.py
  只新增 mode C summary 字段，不需要修改 ObservationBatch。

  新增字段：
    mode_c_candidate_count
    best_mode_c_wait_time
    best_mode_c_uav_flight_time
    best_mode_c_energy_margin_ratio
    best_mode_c_timeout_risk

  同步修改：
    candidate_builder.py 里的 _padding_order_feature()

  不建议在这一步删除或重命名 recovery_pool。
  recovery_pool 仍是 planner / candidate 内部机制，不是 actor 输入。

第二步：补齐 CandidateBuilder 的 mode C summary
  candidate_builder.py
  将当前 _ModeCRecoveryCandidate 升级为 _ModeCSummary 或等价结构。

  需要基于真实数据流计算：
    当前无人机位置 / 触发点
    delivery leg 的时间与能耗
    energy_after_delivery
    recovery_pool 中每个候选节点
    delivery -> recovery node 的飞行时间与能耗
    truck ETA
    planned_wait
    rendezvous_margin
    energy_margin_ratio
    timeout_risk

  需要输出：
    candidate_count
    best_wait_time
    best_uav_flight_time
    best_energy_margin_ratio
    best_rendezvous_margin
    best_node_type
    best_truck_eta_remaining
    timeout_risk

  注意：
    recovery_pool 在这里仍然只是内部候选来源。
    不要把 recovery node 列表写入 ObservationBatch。
    不要让 DispatchAction(mode=C) 提前携带 recover_node_id。

第三步：同步 ObservationTensorizer 的 order token schema
  observation_tensorizer.py
  修改 ORDER_TOKEN_FIELDS 和 _build_order_tokens()。

  加入新增 mode C summary 字段：
    mode_c_candidate_count_norm
    best_mode_c_wait_time_norm
    best_mode_c_uav_flight_time_norm
    best_mode_c_energy_margin_ratio
    best_mode_c_timeout_risk_norm

  同时确认：
    ORDER_TOKEN_FIELDS 数量
    _build_order_tokens() 写入数量
    bootstrap observation 推导出的 order_feat_dim
  三者一致。

  当前代码已经没有 actor recovery token，
  所以这里只需要核验没有新增回去。

第四步：跑 candidate / tensorizer 层测试
  先跑较窄的测试，避免模型结构变化掩盖 token schema 问题。

  建议：
    backend/training/test_phase6_integration.py 中与 CandidateBuilder 相关的用例
    backend/training/test_phase7_snapshot_and_tensorizers.py

  重点检查：
    mode_mask 仍正确表达 B/C 可行性
    has_mode_c_action 与 mode_c_candidate_count 一致
    mode C action resolve 后 recover_node_id 仍为 None
    order_tokens shape 与字段数一致

第五步：检查训练/推理侧是否需要跟随调整
  train_cmrappo.py / policy_inference.py
  当前没有 recovery_tokens 字段引用时，不需要做删除性修改。

  需要确认：
    model 初始化 order_feat_dim 是否来自 bootstrap observation
    stack / zero / sequence minibatch 是否只依赖 ObservationBatch 当前字段
    policy_inference.py 与 train_cmrappo.py 使用同一套 tensorizer 输出

  如果这些逻辑已经按 ObservationBatch 自动 stack，
  这一步只做 smoke test，不做结构性改动。

第六步：修改 model.py 的 order aggregation
  model.py
  新增：
    _masked_max
    _masked_logmeanexp

  修改：
    context_proj 输入维度：
      d_model * 3 + lstm_hidden
      ->
      d_model * 5 + lstm_hidden

    forward_sequence():
      order_summary
      ->
      order_mean / order_max / order_lse

  保持不变：
    order_head
    mode_head
    recurrent_core
    critic_head

  不在这一步加入 attention / dropout / GNN。

第七步：修改配置命名
  rh_alns_cmrappo.yaml
  将 encoder_type 从 attn_lstm_lite 改为 pool_lstm_v2。

  dropout 第一阶段设为 0.0。

  如果 PolicyMeta / 配置解析仍要求 nhead：
    短期保留 nhead: 8，并标注 legacy only。

  如果同步清理 PolicyMeta：
    再移除 nhead 强制校验。

第八步：跑模型 shape smoke test
  确认：
    forward()
    forward_sequence()
    sample_action()
    evaluate_actions()
  全部正常。

  重点检查：
    root_branch_logits shape
    order_logits shape
    mode_logits shape
    value shape
    sequence batch 的 LSTM state shape
    WAIT 样本下 evaluate_actions 不错误消费 mode 分支

第九步：小步训练验证
  先跑短 episode / 小 batch。

  观察：
    policy loss / value loss 是否出现 NaN
    entropy 是否正常
    dispatch / wait 比例是否异常塌缩
    mode C action 数量是否与 candidate supply 大体一致
    mode C 送达后 recovery selection / fallback 指标是否正常

  通过后再开始正式 PPO 训练。
```

---

## 17. 最终机制的一句话描述

修改后的机制是：

```text
一个不显式决策 recovery node 的 order-mode PPO actor。
它通过 CandidateBuilder 将 mode C 的可行性、等待、能量、拥挤和超时风险压缩进 order token；
再通过 mean / max / logmeanexp 多池化增强订单集合表征；
最后由 recurrent factorized actor 输出 WAIT / order / mode 决策。
```

这比旧版更自洽：

```text
旧版：
  recovery pool 构造了，但 actor 没真正使用；
  root 只看订单均值，强订单信号容易被稀释；
  配置叫 attention，但模型没有 attention。

新版：
  actor 不再接收 recovery pool；
  mode C 信息被压缩到 order token；
  root 能同时看到订单均值、极值和平滑强信号；
  配置名称与真实网络结构一致。
```
