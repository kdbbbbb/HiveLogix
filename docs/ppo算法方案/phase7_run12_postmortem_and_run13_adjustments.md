# Phase 7 Run12 Postmortem And Run13 Adjustments 2026-05-06

## 本次修改目标

Run12 之后继续收紧 `mode C` 的候选集时间筛选口径，只做一项纯 YAML 修改：

1. 提高 `candidate.rendezvous_eta_safe_margin_sec`

本轮不改训练代码，不改 reward，不改 early-stop。

## 实际配置修改

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已更新为：

```yaml
candidate:
  rendezvous_eta_safe_margin_sec: 70
```

对应变化如下：

- `candidate.rendezvous_eta_safe_margin_sec: 30 -> 70`

## 调整逻辑

Run11 之后的轻型无人机巡航速度已降到 `10m/s`。在这个机型设定下，一次典型的 `mode C` 链路大致需要：

```text
送达飞行      50s
+ 客户点服务   30s
+ 返回 rendezvous 50s
= 执行主链路 130s
```

如果只给 `30s` 的安全边际，候选集阶段实际上只是在做“刚好不违约”的弱过滤：

- 候选节点只要 `margin >= 0` 就可能被保留
- 但执行中只要出现很小的额外延迟，就会在送达后重验或 reservation 阶段失败
- 结果就是看起来 legal 的 `mode C` 候选很多，真正能成功的比例却不高

本轮把 `safe margin` 提到 `70s`，目的不是增加 `mode C` 数量，而是提升候选质量：

- 在候选集生成阶段直接过滤掉时间窗口过紧的节点
- 接受 `mode C` 候选总数下降
- 换取更高的 `mode C` 成功率
- 同时减少 fallback 与 reservation timeout 惩罚

## 预期效果

- `sum_feasible_mode_c_recover_node_count_total` 会下降
- `mode_c_dispatch_ratio` 可能略降
- `mode_c_success_rate` 预期从约 `33%` 提升到 `60%~70%`
- `fallback_count` 与 `reservation_timeout_count` 应同步下降
- 净 `mode_c_success_count` 可能持平或更高

## 实施说明

- 本轮仅修改 YAML
- 未改 `backend/training/*.py`
- 未改 reward、候选排序规则或 revalidation 逻辑

因此 Run13 的变化应完全来自 `mode C` 候选时间窗口过滤更严格，而不是代码分支变化。
