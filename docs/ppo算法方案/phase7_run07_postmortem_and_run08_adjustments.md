# Phase 7 Rendezvous Reward Split And Training Adjustments 2026-05-04

## 本次修改目标

按最新训练方案完成三项高优先级调整，并确保配置、代码调用链、测试与文档保持一致：

1. 将 `rendezvous_bonus` 拆分为“到达 rendezvous 点奖励”和“成功上车奖励”两段
2. 将 `training.entropy_coef` 从 `0.03` 提高到 `0.05`
3. 放宽 early stop，给 critic 更多收敛时间

## 实际代码修改

### 1. 奖励拆分

配置文件 `[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)` 已调整为：

```yaml
reward:
  rendezvous_arrive_bonus: 2.0
  rendezvous_bonus: 5.0
```

代码侧联动如下：

- `[backend/training/train_cmrappo.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/train_cmrappo.py)`
  - `_TrainingConfig` 新增 `rendezvous_arrive_bonus`
  - `_load_training_config()` 从 `reward.rendezvous_arrive_bonus` 读取配置
  - 新增 `_is_rendezvous_arrival_state()`
  - `_shape_post_action_reward_for_rendezvous()` 现在区分：
    - 到达 `waiting_for_truck`
    - 成功进入 `charging_on_truck` / `riding_with_truck`
  - `_shape_pending_transition_reward_for_rendezvous()` 现在可分别补发 arrive bonus / success bonus
  - `_finalize_pending_transition_for_next_decision()` 与 `_flush_terminal_pending_transitions()` 都已透传两段 bonus
- `[backend/training/rollout_buffer.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/rollout_buffer.py)`
  - `RolloutTransition` 将原来的单个 `rendezvous_bonus_applied` 拆为：
    - `rendezvous_arrive_bonus_applied`
    - `rendezvous_success_bonus_applied`

这样可以保证：

- 若同一 action 窗口内已经到达 rendezvous 点，会立即发放 `rendezvous_arrive_bonus`
- 若上车成功发生在后续等待阶段，则 pending finalize / terminal flush 仍能补发 `rendezvous_bonus`
- 若某次窗口内同时完成“到达 + 上车”，则会按方案发放两段奖励
- 两段奖励都不会重复发放

### 2. 熵系数调整

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)`：

```yaml
training:
  entropy_coef: 0.05
```

训练代码无需额外改动，因为 `train_cmrappo.py` 现有链路已经正确消费 `training.entropy_coef`。

### 3. Early stop 放宽

`[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)`：

```yaml
training:
  early_stop_stochastic_high_patience: 10
  early_stop_value_loss_min_delta: 0.01
```

训练主循环原有 `_should_stop_early()` 与配置校验链路无需结构性修改，当前仅更新参数值即可生效。

## 兼容性补充

- `[backend/training/contracts.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/contracts.py)` 的 `RewardMeta` 已新增 `rendezvous_arrive_bonus`
- `train_cmrappo.py` 构建 `meta.json` 时已写出新字段
- 训练配置校验已新增 `rendezvous_arrive_bonus >= 0` 约束

## 测试修改

已同步更新 `[backend/training/test_phase7_model_runtime.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/test_phase7_model_runtime.py)`：

- immediate shaping 下：
  - 成功上车会同时获得 arrive bonus 与 success bonus
  - 仅到达 `waiting_for_truck` 时只获得 arrive bonus
- pending shaping 下：
  - `waiting_for_truck` 只补发 arrive bonus
  - `riding_with_truck` / `charging_on_truck` 会补发两段奖励
- finalize 后会分别记录两个 `*_bonus_applied` 标记

## 结论

本次修改已经把“到达 rendezvous 点”和“成功上车”拆成两个独立、可追踪、不会重复发放的训练奖励事件；同时按方案提高了 `entropy_coef` 并放宽了 early stop 参数。

## Run08 实际落地修改（2026-05-04 追加）

本节记录本轮按 Run07 postmortem 继续推进的三项结构性调整。重点不是只改配置名义值，而是确保训练运行时真正消费到这些新值。

### 1. 卡车速度提升到 `16.5 m/s`

实际修改文件：

- `[backend/test_data/default_scene/entities.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/entities.json)`
  - `trucks[0].speed: 15 -> 16.5`
- `[backend/generate_test_scene.py](/Users/myx/Documents/GitHub/HiveLogix/backend/generate_test_scene.py)`
  - 默认测试场景生成参数同步改为 `16.5`

关键联动说明：

- Phase 7 训练环境在 reset 时通过 `scene_loader -> EntityManager` 直接读取 `entities.json` 中的 `truck.speed`
- 但 Phase 4 基础骨架路线不是运行时现算，而是从
  `[backend/test_data/default_scene/sumo/phase4_truck_route/truck_execution_route.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/sumo/phase4_truck_route/truck_execution_route.json)`
  等导出产物中读取
- 因此如果只改 `entities.json` 而不重导出 route，训练会出现“truck 实体速度已变，但基础 ETA 仍是旧值”的不一致

为消除这条隐性错误链路，已重新执行：

```bash
python backend/training/export_sumo_truck_route.py --config backend/config/rh_alns_cmrappo.yaml
```

重导出后关键 ETA 变为：

- `STA-TEST-02: 439.76545271030636`
- `STA-TEST-08: 864.0359416403659`
- `STA-TEST-09: 1067.0887683068943`
- `DEP-TEST-01: 1273.871568381955`

对应更新文件包括：

- `[truck_execution_route.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/sumo/phase4_truck_route/truck_execution_route.json)`
- `[truck_eta_map.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/sumo/phase4_truck_route/truck_eta_map.json)`
- `[route_drift_ref.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/sumo/phase4_truck_route/route_drift_ref.json)`
- `[validation_report.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/sumo/phase4_truck_route/validation_report.json)`
- 以及同目录下其余 Phase 4 导出产物

### 2. 换电时间提高到 `120s`

实际修改文件：

- `[backend/test_data/default_scene/entities.json](/Users/myx/Documents/GitHub/HiveLogix/backend/test_data/default_scene/entities.json)`
  - `depots[*].swap_time: 90 -> 120`
  - `stations[*].swap_time: 60 -> 120`
  - `trucks[*].swap_time: 90 -> 120`
- `[backend/generate_test_scene.py](/Users/myx/Documents/GitHub/HiveLogix/backend/generate_test_scene.py)`
  - 场景生成默认值同步改到 `120`

关键联动说明：

- `swap_time` 不是 PPO YAML 参数，而是 ChargingHost 实体静态属性
- 训练环境的 mode B 返程、排队与充换电服务时间，最终都读取 `Depot` / `SwapStation` / `Truck` 的 `swap_time`
- 因此这项改动必须落到默认场景实体，而不是只写在方案文档里

### 3. `patrol_stations_per_loop` 从 `3` 提高到 `5`

实际修改文件：

- `[backend/config/rh_alns_cmrappo.yaml](/Users/myx/Documents/GitHub/HiveLogix/backend/config/rh_alns_cmrappo.yaml)`
  - `planner.patrol_stations_per_loop: 3 -> 5`
  - 同时更新了相邻注释，明确这是 Run08 的显式巡站覆盖率调整

关键联动说明：

- 该参数由 `train_cmrappo -> TrainingEnvAdapter._load_env_yaml()` 读取
- 最终在 `[backend/training/env_adapter.py](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py)` 的 `_append_patrol_loop_if_needed()` 中生效
- 这里不需要额外改函数签名或新增字段，现有代码链已经完整消费 `planner.patrol_stations_per_loop`

### 4. 本轮无需再改训练主循环代码

这三项改动里，真正需要改“代码逻辑”的并不是 PPO 主循环，而是：

- 场景实体默认值
- 训练 YAML 参数值
- Phase 4 路线导出产物

现有调用链已经支持这些参数：

- `truck.speed`：被 route export 与 poisson 巡站追加逻辑消费
- `swap_time`：被 host service / queue 逻辑消费
- `patrol_stations_per_loop`：被 poisson 巡站生成逻辑消费

因此本轮重点是把真实生效源改对，并保持 route 产物与 scene 实体一致。

### 5. 回归验证

已执行并通过：

```bash
python -m unittest backend.training.test_env_adapter_phase5c
python -m unittest backend.training.test_phase7_model_runtime
```

结论：

- Run08 的三项调整已落到真实生效链路
- 不存在“只改注释/文档、训练实际不消费”的断链问题
- `truck.speed` 与 Phase 4 ETA 产物已重新对齐，避免了场景值与骨架路线值不一致
