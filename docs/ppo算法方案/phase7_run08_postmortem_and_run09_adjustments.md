# Phase 7 Run08 Postmortem And Run09 Adjustments 2026-05-05

## 本次修改目标

围绕 Run09 的 `mode C` 探索不足、时效压力不足与 critic 波动问题，按方案完成以下五项调整：

1. 大幅提高 rendezvous 分段奖励
2. 新增 `mode_c_attempt_bonus`，只要选择 `mode C` 就给即时探索奖励
3. 提高 `training.entropy_coef`
4. 下调 `training.ppo_learning_rate` 并放宽 `training.value_huber_delta`
5. 收紧 poisson 订单 deadline 窗口，给 `mode B` 增加真实时效压力

本次同时检查配置解析、奖励塑形、meta 输出与测试覆盖，确保修改不是停留在 YAML 表面。

## 实际代码修改

### 1. YAML 参数更新

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已改为：

```yaml
reward:
  rendezvous_arrive_bonus: 15.0
  rendezvous_bonus: 40.0
  mode_c_attempt_bonus: 3.0

training:
  entropy_coef: 0.10
  ppo_learning_rate: 0.0001
  value_huber_delta: 5.0

scene:
  order_window_min_min: 15
  order_window_max_min: 40
```

其中：

- `mode C` 成功时总 shaping 奖励变为 `15 + 40 = 55`
- 若与原有 `R_delivery_bonus: 100` 叠加，则成功路径总正奖励显著高于 `mode B`
- 即使未成功 rendezvous，只要选了 `mode C`，也会先拿到 `3.0` 的探索奖励
- deadline 窗口从 `20~60` 分钟收紧为 `15~40` 分钟，先采用保守版本，避免直接压到大量超时崩盘

### 2. `mode_c_attempt_bonus` 接线

本轮新增字段已贯通以下位置：

- `[backend/training/contracts.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/contracts.py)`
  - `RewardMeta` 新增 `mode_c_attempt_bonus`
- `[backend/training/train_cmrappo.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/train_cmrappo.py)`
  - `_TrainingConfig` 新增 `mode_c_attempt_bonus`
  - `_load_training_config()` 从 `reward.mode_c_attempt_bonus` 读取，默认回退 `0.0`
  - `_validate_training_config()` 新增非负校验
  - 训练 `meta` 导出构造 `RewardMeta(...)` 时同步写出新字段

这保证了配置、训练运行时对象和产出元数据三处口径一致。

### 3. 奖励塑形逻辑调整

`_shape_post_action_reward_for_rendezvous()` 现已改为：

```text
post_action_reward
+ mode_c_attempt_bonus
+ rendezvous_arrive_bonus（到达 waiting / charging / riding）
+ rendezvous_bonus（成功 charging / riding）
```

具体语义：

- 非 `mode C` 动作直接返回原始 `post_action_reward`
- 命中 `mode C` 后，先无条件加 `mode_c_attempt_bonus`
- 若同一步已进入 `waiting_for_truck` / `charging_on_truck` / `riding_with_truck`，再加 `rendezvous_arrive_bonus`
- 若同一步已成功上车，再额外加 `rendezvous_bonus`

延迟补发链路 `_shape_pending_transition_reward_for_rendezvous()` 未加入 `mode_c_attempt_bonus`，因此：

- `attempt bonus` 只在“选中 mode C 的当前 action”发一次
- `arrive/success bonus` 仍可在 pending finalize / terminal flush 阶段补发
- 不会发生重复计数

### 4. 订单 deadline 收紧

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 的 `scene` 段已改为：

```yaml
scene:
  order_window_min_min: 15
  order_window_max_min: 40
```

本次没有直接采用更激进的 `12/35`，而是先用你建议的保守起步值 `15/40`。原因是：

- Run08 中 `mode B` 的 `on_time_rate` 过高，说明当前 deadline 太宽松，`mode B` 几乎总能按时完成
- 若 deadline 略收紧，`mode B` 在排队、返航、服务时间上的劣势才会开始显现
- `mode C` 的 rendezvous 路径由于少一次排队换电，才更有机会在时效指标上体现优势

这项修改只影响 poisson 订单的 deadline 生成口径，不需要新增代码接线；训练环境现有订单生成逻辑会直接消费这两个字段。

建议观察目标：

- 训练 / eval 的 `on_time_rate` 是否从接近 `1.0` 回落到 `0.85 ~ 0.90`
- 是否出现大面积 timeout 或 return 明显塌陷
- `mode C` 选择率与成功率是否同步上升，而不是仅仅整体服务质量下降

## 代码上下文衔接检查结果

已确认以下调用链有效：

- `reward.mode_c_attempt_bonus -> _load_training_config() -> train_cfg.mode_c_attempt_bonus -> _shape_post_action_reward_for_rendezvous()`
- `reward.rendezvous_arrive_bonus / reward.rendezvous_bonus`
  - 即时发生时走 post-action shaping
  - 延迟发生时走 pending finalize / terminal flush
- `training.entropy_coef -> PPO 总 loss 中的 entropy 正则项`
- `training.ppo_learning_rate -> Adam optimizer`
- `training.value_huber_delta -> value huber loss`
- `scene.order_window_min_min / order_window_max_min -> poisson 订单 deadline 生成逻辑`

结论：

- 新字段没有悬空
- 旧字段没有断链
- 非 `mode C` 动作不会误拿 `attempt bonus`
- `attempt bonus` 不会与 pending 的 rendezvous 奖励补发冲突
- deadline 收紧是纯配置变更，训练环境现有链路可直接生效

## 测试补充

已同步更新 `[backend/training/test_phase7_model_runtime.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/test_phase7_model_runtime.py)`：

- `_TrainingConfig` 测试构造补充 `mode_c_attempt_bonus`
- 新增单测覆盖：
  - `mode C` 被选择但未到达/未成功时，仍会获得 `attempt bonus`
  - 非 `mode C` 动作不会获得 `attempt bonus`

## 回归验证

已执行：

```bash
python -m unittest backend.training.test_phase7_model_runtime
```

本次 Run09 所需参数与代码接线已完成，且关键奖励路径与配置消费链路已核对通过。
