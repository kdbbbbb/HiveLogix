# Phase 7 Run13 Postmortem And Run14 Adjustments 2026-05-07

## 本次修改目标

Run13 暴露出一个结构性问题：同一个 `rendezvous` 安全边际同时承担了两种不同职责：

1. 候选集过滤
2. 送达后重验 / reservation timeout 执行检查

这会导致调参方向相互冲突：

- margin 设大了，`mode C` 候选数被过度压缩
- margin 设小了，候选集看起来合法，但执行层面大量失败

Run14 的修复不是继续猜一个折中值，而是把它拆成两个独立参数。

## 实际修改

### 1. YAML 参数拆分

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已改为：

```yaml
candidate:
  rendezvous_filter_margin_sec: 30
  rendezvous_execution_margin_sec: 10
```

语义如下：

- `rendezvous_filter_margin_sec`
  候选过滤器使用，保持 `30s`
- `rendezvous_execution_margin_sec`
  timeout / revalidation 执行检查使用，设为 `10s`

### 2. 代码接线

已完成以下机械性修改：

- `[backend/training/candidate_builder.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/candidate_builder.py)`
  候选集生成改为只使用 `rendezvous_filter_margin_sec`
- `[backend/training/env_adapter.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py)`
  - mode C 候选合法性检查使用 `rendezvous_filter_margin_sec`
  - reservation timeout 检查使用 `rendezvous_execution_margin_sec`
  - 送达后 `revalidate_mode_c_recover_node()` 使用 `rendezvous_execution_margin_sec`
- `[backend/training/contracts.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/contracts.py)`
  `CandidateMeta` 改为记录两个独立 margin
- `[backend/training/train_cmrappo.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/train_cmrappo.py)`
  训练 meta / snapshot 导出同步改为写出两个 margin

同时保留了解析层回退兼容：

- 若旧配置仍只有 `rendezvous_eta_safe_margin_sec`
- 新的 filter / execution 字段会自动回退到旧值

## 调整逻辑

### 1. filter margin 应该保守

候选集的任务是防止“明显时间窗口过紧”的 node 进入动作空间。  
这一层宁可略保守，也不应该把大量边界点暴露给 policy。

因此 `rendezvous_filter_margin_sec` 保持在 `30s`。

### 2. execution margin 应该轻量

执行层的任务不是重新做一次激进筛选，而是在：

- reservation timeout
- 送达后重验

这些节点上判断“是否还有足够缓冲继续执行”。  
这里如果继续沿用 `30s`，会把许多本来仍能成功完成的 mode C 轨迹直接判死。

因此 `rendezvous_execution_margin_sec` 单独降到 `10s`。

## 预期效果

- `mode C` 候选数量不会像单一大 margin 那样被过度压缩
- 送达后 `rendezvous_time_feasible` 失败率应明显下降
- `fallback_count` 与 `reservation_timeout_count` 应随之下降
- `mode_c_success_count` 与 `mode_c_success_rate` 应同步改善

## 验证建议

优先观察：

- `sum_feasible_mode_c_recover_node_count_total`
- `sum_mode_c_post_delivery_revalidation_fail_reasons.rendezvous_time_feasible`
- `sum_reservation_timeout_count`
- `sum_fallback_count`
- `sum_mode_c_success_count`

如果这次拆分有效，理想现象应是：

- legal mode C 候选数不再被过度压缩
- post-delivery revalidation fail 显著下降
- 最终成功回收次数上升
