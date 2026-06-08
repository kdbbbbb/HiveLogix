# Phase 7 Run16 Critic Adjustments 2026-05-09

## 四、Run16 具体修改建议

### 4.1 核心修复：Critic 参数（YAML 修改，高优先级）

Run16 优先只做一类修改：修正 critic 的训练尺度，使 value function 重新进入可学习区间。  
本轮不混入 reward、reservation 语义、mode C 排序或 observation 结构改动，避免再次打乱因果关系。

## 实际配置修改

已更新 `[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 中的 `training` 配置：

```yaml
training:
  ppo_learning_rate: 0.0001
  value_loss_coef: 0.10
  value_huber_delta: 25.0
```

对应变化如下：

- `value_huber_delta: 5.0 -> 25.0`
- `value_loss_coef: 0.05 -> 0.10`
- `ppo_learning_rate: 0.00008 -> 0.0001`

## 调整逻辑

### 1. 提高 `value_huber_delta`

当前判断是：`delta=5.0` 相对 `return_std≈40~50` 明显过小，导致绝大多数 critic 样本落在 Huber 的线性区。  
在线性区里，梯度幅值基本被截平，critic 很难根据误差大小自适应地区分“中等误差”和“大误差”样本。

将 `value_huber_delta` 提高到 `25.0` 的目的，是让更多样本重新进入二次区，使 critic 恢复更合理的误差敏感度与梯度分辨率。

### 2. 提高 `value_loss_coef`

当 `delta` 变大后，critic 对误差的响应结构会变得更合理，但在共享优化器和 `max_grad_norm=1.0` 的限制下，value 分支仍可能被 actor 更新噪声压制。  
因此将 `value_loss_coef` 从 `0.05` 提高到 `0.10`，给 critic 额外一点更新权重，但仍保持在较保守区间，避免回到早期高 value loss 污染 policy 的状态。

### 3. 小幅回升 `ppo_learning_rate`

更大的 `value_huber_delta` 会让一部分样本从“截平梯度”回到“随误差变化的二次梯度”，但整体梯度绝对值不一定比旧配置更大。  
因此把 `ppo_learning_rate` 从 `8e-5` 调到 `1e-4`，作为轻量补偿，避免 critic 更新重新变慢。

## 预期效果

本轮的主要观察目标不是总 reward，而是 critic 是否重新恢复可学习性。  
预期结果：

- `value_loss` 均值从约 `122` 下降到 `30~50`
- advantage 估计质量改善，波动收敛更快
- `mode_c` 的选择质量随 critic 改善而提升，即使 `mode_c_selected_count` 未必立刻明显增加

## 验证重点

建议优先观察以下指标，而不是先看单次 best reward：

1. `value_loss`
2. `explained_variance`（若当前日志已输出）
3. `advantage` 分布是否比 Run15 更稳定
4. `mode_c_success_rate`
5. `reservation_timeout_count`
6. `fallback_count`

## 实施边界

- 本轮是纯 YAML 调参
- 未修改 `backend/training/*.py` 训练逻辑
- 未新增或删除 observation 特征
- 未改 reward 塑形
- 未改 reservation timeout 语义

因此如果 Run16 行为发生变化，应主要归因于 critic 训练尺度修正，而不是其他结构变量。
