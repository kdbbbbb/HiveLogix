# Phase 7 Run11 Postmortem And Run12 Adjustments 2026-05-06

## 本次修改目标

Run11 暴露出三项明确且可直接修复的技术缺陷：

1. `mode C` 送达后重验大规模失败，且失败原因几乎全部集中在 `rendezvous_time_feasible`
2. `recovery_entropy_coef` 计划改为 `1.0`，但实际 YAML 仍停留在 `0.4`
3. `wait_idle_penalty_coef=0.25` 对 `10m/s` 轻型无人机过重，导致低密度场景 reward 被 idle 噪声淹没

本轮只修正配置口径，不改训练代码逻辑。

## 实际配置修改

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已更新为：

```yaml
candidate:
  rendezvous_eta_safe_margin_sec: 30

reward:
  wait_idle_penalty_coef: 0.10

training:
  recovery_entropy_coef: 1.0
```

对应变化如下：

- `candidate.rendezvous_eta_safe_margin_sec: 15 -> 30`
- `reward.wait_idle_penalty_coef: 0.25 -> 0.10`
- `training.recovery_entropy_coef: 0.4 -> 1.0`

## 调整逻辑

### 1. 修复 `rendezvous_time_feasible` 的系统性误判

Run11 的 `light_drone.cruise_speed` 已从 `20m/s` 降到 `10m/s`，但 `rendezvous_eta_safe_margin_sec` 仍保持 `15s`。  
这会造成一个稳定偏差：

- 派单时，候选集仍把一批“边界可行”的 mode C 节点放进动作空间
- 真正完成配送后，无人机返回 rendezvous 节点的剩余时间变长
- 卡车已经越过该节点，导致送达后重验大量触发 `rendezvous_time_feasible=false`

本轮直接把安全裕量按速度变化翻倍到 `30s`，目标是提前在候选集阶段裁掉这类伪可行节点，而不是等送达后再 fallback。

### 2. 恢复 recovery head 的正常熵正则

`mode C` 本来就是低频分支，而 recovery head 只在 `mode C` 被选中时才真正参与采样。  
如果还把 `recovery_entropy_coef` 压到 `0.4`，训练上几乎等于默认鼓励它塌到第一个候选节点。

因此 Run12 直接恢复到：

```yaml
training:
  recovery_entropy_coef: 1.0
```

目标不是让 recovery 头“更随机”，而是避免它在样本极少时过早退化成固定选项。

### 3. 降低显式 WAIT 的 idle 惩罚

`wait_idle_penalty_coef=0.25` 是在更轻、更快的机型背景下形成的权重。  
对当前 `10m/s` 的轻型无人机而言：

- 同样一次保守等待，物理耗时会更长
- 但这不等价于策略更差
- 在 `stochastic_low` 这类低密度场景里，过重的 idle 惩罚会把本来健康的轨迹 reward 直接压穿

本轮回调到 `0.10`，让低密度场景的 reward 重新主要由“是否清单、是否超时、是否 fallback”决定，而不是被 WAIT 动作长度主导。

## 预期效果

- `mode_c_post_delivery_revalidation_fail_reasons.rendezvous_time_feasible` 应明显下降
- `mode_c_success_count` 与 `mode_c_dispatch_ratio` 应有恢复空间
- recovery 候选选择不再长期固定为第一个节点
- `stochastic_low.mean_total_reward` 应显著抬升，不再停留在几十的量级

## 实施说明

- 本轮仅修改 YAML
- 未改 `backend/training/*.py`
- 未改 reward 塑形、候选生成或重验逻辑本身

因此 Run12 的行为变化应完全来自配置修正，而不是代码路径变化。
