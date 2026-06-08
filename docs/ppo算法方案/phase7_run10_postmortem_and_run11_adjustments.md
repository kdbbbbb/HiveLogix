# Phase 7 Run10 Postmortem And Run11 Adjustments 2026-05-05

## 本次修改目标

Run11 分两类调整，且全部通过 YAML 生效，不改训练代码：

1. 修正 reward 与 early-stop 口径，避免 mode C 被等待惩罚和 value-loss 条件过早压死
2. 回标 `light_drone` 物理参数，使仿真机体质量、速度与电池规模回到更合理区间

## 实际配置修改

### 1. PPO 训练配置

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已更新为：

```yaml
reward:
  lambda_wait: 0.08
  rendezvous_arrive_bonus: 8.0
  rendezvous_bonus: 20.0
  mode_c_attempt_bonus: 5.0

training:
  value_huber_delta: 5.0
  ppo_learning_rate: 0.00008
  early_stop_value_loss_min_delta: 0.0
  early_stop_stochastic_high_patience: 15
```

对应变化如下：

- `lambda_wait: 0.25 -> 0.08`
- `rendezvous_arrive_bonus: 4.0 -> 8.0`
- `rendezvous_bonus: 12.0 -> 20.0`
- `mode_c_attempt_bonus: 3.0 -> 5.0`
- `value_huber_delta: 15.0 -> 5.0`
- `ppo_learning_rate: 0.00005 -> 0.00008`
- `early_stop_value_loss_min_delta: 0.01 -> 0.0`
- `early_stop_stochastic_high_patience: 10 -> 15`

### 2. 无人机物理参数

`[backend/config/drone_params.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/drone_params.yaml)` 的 `light_drone` 已更新为：

```yaml
light_drone:
  empty_weight: 6.5
  cruise_speed: 10.0
  k1: 35.0
  k2: 0.0059
  battery_capacity_j: 1080000
  payload_capacity: 2.0
  safe_margin_ratio: 0.10
```

对应变化如下：

- `empty_weight: 1.5 -> 6.5`
- `cruise_speed: 20.0 -> 10.0`
- `k1: 28.3 -> 35.0`
- `k2: 0.0059 -> 0.0059`（不变）
- `battery_capacity_j: 360000 -> 1080000`
- `payload_capacity: 2.0 -> 2.0`（不变）
- `safe_margin_ratio: 0.10 -> 0.10`（不变）

## 调整逻辑

### 1. 降低等待惩罚，恢复 mode C 可学性

- `lambda_wait` 从 `0.25` 降到 `0.08`，这是本轮最关键改动
- mode C 的固有弱点是会经历更多 waiting / rendezvous 相关时间段
- 如果等待惩罚过强，策略会在探索早期直接把 mode C 判成“天然吃亏”

这次先降低等待惩罚，再同步提高 rendezvous 与 attempt 奖励，目的是让 mode C 至少能重新进入可竞争区间。

### 2. 回补 rendezvous 激励，但不回到 Run09 的高方差尺度

Run11 下，mode C 成功时总正奖励为：

```text
R_delivery_bonus 100
+ rendezvous_arrive_bonus 8
+ rendezvous_bonus 20
+ mode_c_attempt_bonus 5
= 133
```

这比 Run10 的 `119` 更强，但仍明显低于 Run09 的超高塑形强度，因此是在“恢复动机”和“控制 return 方差”之间取中间值。

### 3. 修正 early stop 对 value loss 的误判

- `value_huber_delta` 调回 `5.0`，避免 Huber 区间过宽导致 value loss 曲线失真
- `early_stop_value_loss_min_delta` 改成 `0.0`，等价于禁用 value loss 改善幅度条件
- `early_stop_stochastic_high_patience` 提到 `15`，给 stochastic 指标更多收敛时间
- `ppo_learning_rate` 小幅回升到 `8e-5`，避免 Run10 步长过小导致整体更新太慢

目标是减少“训练还没真正展开，就被 early stop 规则提前判死”的情况。

### 4. 回标轻型无人机物理参数

本轮把 `light_drone` 从偏轻、偏快、偏小电池的抽象设定，调整到更接近商用多旋翼任务机的区间：

- 更重的 `empty_weight=6.5kg`
- 更保守的 `cruise_speed=10m/s`
- 更高的 `k1=35.0`，反映更重机体下的悬停主功耗
- 更大的 `battery_capacity_j=1080000`，即 `300Wh`

关于电池容量，本次明确采用 `300Wh` 而不是 `526Wh`：

- `526Wh` 对 `6.5kg` 机体并非绝对不可行，但会把总重进一步推高到约 `8.6kg`
- 相比当前项目里的轻型机定位，这个量级过于激进
- `300Wh` 是更保守、更合理的起点，便于先观察航程与 mode 选择结构是否进入可信区间

## 实施说明

- 本轮是纯 YAML 修改
- 未改 `backend/training/*.py`
- 未改 reward 塑形接线、early-stop 实现逻辑或能耗模型代码

因此 Run11 的行为变化应全部来自配置口径变化，而不是代码分支变化。
