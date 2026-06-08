# Phase 7 Run09 Postmortem And Run10 Adjustments 2026-05-05

## 本次修改目标

围绕 Run09 暴露出的高方差 return 与 critic 收敛困难问题，Run10 只做两类高优先级调整：

1. 大幅降低 rendezvous 成功奖励，削弱 mode C 成功路径对 return scale 的放大
2. 进一步压制 value loss，降低 critic 对整体 PPO 更新的扰动

本次不改奖励接线逻辑，只调整默认训练配置口径。

## 实际参数修改

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已更新为：

```yaml
reward:
  rendezvous_arrive_bonus: 4.0
  rendezvous_bonus: 12.0
  mode_c_attempt_bonus: 3.0

training:
  ppo_learning_rate: 0.00005
  value_loss_coef: 0.05
  value_huber_delta: 15.0
```

对应变化如下：

- `rendezvous_arrive_bonus: 15.0 -> 4.0`
- `rendezvous_bonus: 40.0 -> 12.0`
- `mode_c_attempt_bonus: 3.0 -> 3.0`（保持不变）
- `ppo_learning_rate: 0.0001 -> 0.00005`
- `value_loss_coef: 0.10 -> 0.05`
- `value_huber_delta: 5.0 -> 15.0`

## 调整逻辑

### 1. 降低 rendezvous 奖励

- `mode_c_attempt_bonus` 已经承担“先让策略愿意点 mode C”的探索驱动力
- 不再需要依赖过高的 rendezvous 奖励去强推成功样本
- 将 rendezvous 成功塑形从 `55` 压到 `16`，可以明显缩小 return 波动

Run10 下，mode C 成功时总正奖励仍然是：

```text
R_delivery_bonus 100
+ rendezvous_arrive_bonus 4
+ rendezvous_bonus 12
+ mode_c_attempt_bonus 3
= 119
```

这仍然显著高于 mode B 的 `100`，因此 mode C 的学习动机仍然存在。

### 2. 进一步压制 value loss

三个参数协同作用如下：

- `value_loss_coef` 下调到 `0.05`，直接降低 critic loss 对总梯度的影响权重
- `value_huber_delta` 放宽到 `15.0`，让大误差样本不过早进入更尖锐的惩罚区间
- `ppo_learning_rate` 下调到 `5e-5`，让 actor 和 critic 的共享更新步长都更保守

目标不是让 critic 更快拟合，而是先避免它继续主导训练波动。

## 代码与测试

- 奖励塑形接线不变，`mode_c_attempt_bonus` 仍只在选择 `mode C` 的当前 action 发放一次
- pending finalize / terminal flush 仍只负责补发 `rendezvous_arrive_bonus` 与 `rendezvous_bonus`
- `[backend/training/test_phase7_model_runtime.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/test_phase7_model_runtime.py)` 中新增的 attempt bonus 单测参数已同步到 Run10 数值，避免测试语义继续停留在 Run09

## 建议观察指标

- `value_loss`
- `explained_variance`
- `mean_return` 与 `return_std`
- `mode_c_usage_rate`
- `mode_c_success_rate`

如果 Run10 后 `mode_c_usage_rate` 没掉太多，但 `value_loss` 和 return 波动明显回落，这次收缩就是有效的。
