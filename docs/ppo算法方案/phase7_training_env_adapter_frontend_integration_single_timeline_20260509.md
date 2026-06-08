# Phase 7 TrainingEnvAdapter Frontend Integration Single Timeline 2026-05-09

## 一、结论

前端接入 `CMRAPPO` 时，建议**不再让 `SimulationEngine` 和 `TrainingEnvAdapter` 并行维护两套仿真状态**。  
应改为：

1. `TrainingEnvAdapter` 作为**唯一仿真状态机**
2. 一个独立的在线运行协调器持有 `TrainingEnvAdapter`、策略模型、LSTM state、history buffer
3. 定时线程每 `100ms wall-clock` 从 `TrainingEnvAdapter` 读取当前状态并广播 `TICK`
4. 若 `TrainingEnvAdapter` 当前存在决策点，则在同一线程内完成策略推理并调用 `env.step()`

这样系统里只有一套时间轴：  
**前端看到的时间、策略决策使用的时间、事件推进使用的时间，全部来自 `TrainingEnvAdapter.t_now`。**

这比把 `SimulationEngine` 的物理步进和 `TrainingEnvAdapter` 的事件推进强行对齐更稳，也更容易解释和维护。

## 二、为什么要这样做

当前 `SimulationEngine` 的职责是：

- 自己维护 `current_time`
- 调 `EntityManager.tick_all()` 推动物理状态
- 调 `OrderManager.tick()` 注入订单
- 调 `broadcast_tick()` 广播前端遥测

可参考 [`backend/environment/state/sim_engine.py`](/Users/myx/Documents/GitHub/HiveLogix/backend/environment/state/sim_engine.py:183)。

而 `TrainingEnvAdapter` 已经具备另一套完整语义：

- 自己维护 `t_now`
- 自己维护无人机训练态、飞行段、delivery service、fallback、reservation、dispatch commitment
- 自己决定下一事件时刻并推进
- 自己暴露当前决策点
- 已提供轻量可视化快照接口 [`build_visualization_snapshot()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:913)

如果两套系统同时存在，就会出现以下结构问题：

1. 两套时间轴
2. 两套实体位置真值
3. 两套订单状态推进
4. 两套“谁来触发决策”的逻辑

你现在提出的方向本质上是在消掉这些重复语义，这个方向是对的。

## 三、目标架构

建议新增一个专门的在线策略编排层，例如：

```text
TrainingPolicyRuntimeCoordinator
```

职责如下：

1. 持有 `TrainingEnvAdapter`
2. 持有已加载的 PPO policy runtime
3. 持有在线推理所需的：
   - `last_seen_plan_version_by_drone`
   - `history_buffer`
   - `lstm_state_by_drone`
   - `recurrent_segment_id_by_drone`
4. 提供：
   - `start()`
   - `pause()`
   - `reset()`
   - `build_tick_payload()`
   - `build_full_snapshot()`
5. 在后台循环里：
   - 推进 wall-clock
   - 读取 `TrainingEnvAdapter` 当前插值状态
   - 有决策点时执行一次策略推理并 `env.step()`
   - 广播 `TICK`

### 3.1 组件职责切分

#### `TrainingEnvAdapter`

唯一状态真值来源，负责：

- 仿真时间 `t_now`
- 事件推进
- 位置插值
- reward 结算
- reservation / fallback / rendezvous 语义
- 决策触发
- 可视化快照构造

#### 在线运行协调器

负责：

- 管理 wall-clock 定时循环
- 管理策略模型和推理缓存
- 在决策点调用 actor
- 将 `TrainingEnvAdapter` 的状态转换成前端遥测格式

#### `SimulationEngine`

不再作为 PPO 在线模式的物理仿真引擎。  
它当前的 `_run_loop()` 结构可参考，但不应直接复用其内部的：

- `EntityManager.tick_all()`
- `OrderManager.tick()`
- `_try_auto_dispatch()`

这些逻辑在 PPO 在线模式下都应由 `TrainingEnvAdapter` 接管。

## 四、主循环建议

建议保留“`100ms wall-clock` 唤醒一次”的节奏，但将循环内容改成面向 `TrainingEnvAdapter` 的版本：

```text
while running:
    1. 记录 loop_start
    2. 读取当前 wall-clock 与 speed_ratio
    3. 读取当前 env.t_now，并检查是否存在 current_decision_context
    4. 若存在 current_decision_context：
         5.1 构造 CandidateOutput
         5.2 ObservationTensorizer.build(...)
         5.3 CriticBatchBuilder.build(...)
         5.4 model.forward(...)
         5.5 sample_action(deterministic=True/False)
         5.6 resolved_action_lookup.resolve(...)
         5.7 env.step(env_action)
         5.8 更新 history_buffer / lstm_state
       继续循环，直到当前没有新的决策点
    5. 从 TrainingEnvAdapter 读取当前状态快照
    6. 转成前端 TICK payload
    7. broadcast_tick(payload)
    8. 睡眠补偿到 100ms
```

这里要特别说明一件事：

- `TrainingEnvAdapter` 当前并**没有**提供“推进到某个外部指定 `target_sim_time` 就停下”的接口
- 它的推进是由 `env.step()` 触发的
- `env.step()` 内部会通过事件循环推进到“下一个决策点或 episode 结束”

因此，在线协调器在每个 wall-clock tick 里**不应**尝试主动把 env 推进到某个目标仿真时刻；  
它只需要：

1. 看当前有没有 `decision_context`
2. 有就做一次策略推理并 `env.step()`
3. `env.step()` 返回后，如果又暴露了新的决策点，就继续处理
4. 没有决策点时，直接读取当前快照并广播

### 4.1 关于单帧预算

本方案**暂不建议**引入“每帧最多处理 N 次决策 / 最多消耗 X ms 推理时间”的预算机制。

原因不是它一定做不到，而是它会引入新的语义复杂度：

1. 某一帧中若决策链被人为截断，`TrainingEnvAdapter` 内部 `_decision_queue` 仍残留未处理决策点
2. 但 wall-clock 已继续前进到下一帧
3. 下一帧继续消费这些旧决策点时，env 的 `t_now` 可能并未前进
4. 这会让“广播节奏”和“仿真真时间”之间形成不必要的漂移解释问题

在当前阶段，更稳妥的做法是：

- 先按“单帧内把当前连锁决策处理干净”来设计
- 只有在真实压测证明单帧推理明显超过 `100ms` 时，再引入预算和背压机制

换句话说，预算机制应是**性能优化后手**，不应作为第一版主语义的一部分。

## 五、时间轴如何统一

这是该方案最核心的部分。

### 5.1 唯一仿真时间

在线 PPO 模式下，唯一有效的仿真时间是：

```text
TrainingEnvAdapter._t_now
```

前端 `sim_time`、决策日志时间、位置插值时间，都从这里取。

### 5.2 wall-clock 只负责“驱动节奏”

wall-clock 不再直接决定实体如何运动，它只决定：

- 多久唤醒一次后台线程
- 多久广播一次前端可见状态

也就是说：

```text
wall-clock = 驱动器
env.t_now   = 业务真时间
```

### 5.3 插值状态由 TrainingEnvAdapter 负责，但快照读取链路要明确

`TrainingEnvAdapter` 已经在 [`_sync_in_transit_positions()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:2772) 中统一处理：

- 卡车位置插值
- 飞行中的无人机位置插值
- riding_with_truck / charging_on_truck 的位置跟随

但这里要注意一个实现细节：

- `_sync_in_transit_positions()` 是在 `env.step()` 内部推进事件时调用的
- `build_runtime_state_view()` **不会**主动调用 `_sync_in_transit_positions()`
- 因此在线协调器不应假设“任意读取 `RuntimeStateView` 都会顺便完成位置同步”

本方案建议明确冻结一条规则：

1. 前端可视化快照只通过统一的快照构造入口读取
2. 该入口必须保证返回的是“已经对齐到当前 `env.t_now` 的状态”
3. 不允许前端遥测拼装逻辑自己直接散读 `entity_mgr` / `RuntimeStateView` 后假设位置已经同步

如果后续需要支持“在不调用 `env.step()` 的情况下，显式重算一次 `env.t_now` 对应的插值位置”，应在 `TrainingEnvAdapter` 外围补一个**明确的公共快照包装方法**，而不是隐式依赖内部私有函数调用链。

## 六、前端遥测输出建议

现有 WebSocket 广播机制本身可以继续用：

- `telemetry.register_route()`
- `telemetry.set_snapshot_builder()`
- `broadcast_tick()`

见 [`backend/api/websockets/telemetry.py`](/Users/myx/Documents/GitHub/HiveLogix/backend/api/websockets/telemetry.py:43)。

要改的是**谁来构造 payload**。

### 6.1 FULL_SNAPSHOT

建议在线 PPO 模式下的 `FULL_SNAPSHOT` 改由在线运行协调器生成，而不是 `SimulationEngine.build_full_snapshot()`。

建议 payload 结构：

```json
{
  "type": "FULL_SNAPSHOT",
  "payload": {
    "sim_time": 123.4,
    "is_running": true,
    "speed_ratio": 1.0,
    "sim_start_wall_ms": 1710000000000,
    "entities": {
      "truck": [...],
      "drones": [...],
      "depots": [...],
      "stations": [...]
    },
    "orders": [...],
    "stats": {
      "active_policy": "rh_alns_cmrappo",
      "checkpoint": ".../policy.pt",
      "dispatch_count": 17,
      "mode_c_success_count": 3,
      "reservation_timeout_count": 1
    }
  }
}
```

### 6.2 TICK

建议 `TICK` 同样由协调器基于 `TrainingEnvAdapter.build_visualization_snapshot()` 生成，但**不能直接原样透传**。

当前 `build_visualization_snapshot()` 已包含：

- `t_now`
- `truck`
- `drones`
- `orders`
- `current_decision`
- `last_reward_breakdown`

见 [`env_adapter.py`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:913)。

但是它有两个重要限制：

1. 坐标是 UTM 风格的 `x / y / z`，不是前端当前 TICK 体系期望的 `lng / lat / altitude`
2. 它没有包含 `stations / depots` 的实时队列与槽位状态

因此协调器必须显式做两类转换，而不是只改字段名：

### 6.2.1 坐标系转换

需要把以下对象从 UTM 转成 WGS84：

- truck
- drones
- orders（若前端当前订单展示链路统一使用经纬度）
- stations / depots

不能只做：

```text
x -> lng
y -> lat
```

必须做真实的坐标系转换。

### 6.2.2 站点/仓库实时状态补充

前端若要显示充换电站与仓库的实时状态，协调器还必须从 `RuntimeStateView.node_states` 额外补以下字段：

- `queue_length`
- `available_slots`
- `parking_slots`
- `swap_time`
- `node_type`

也就是说，`build_visualization_snapshot()` 只适合作为：

- 当前 truck / drones / orders / decision 的轻量来源

而不是完整 TICK payload 的唯一数据源。

更准确的说法应是：

- `build_visualization_snapshot()` 提供轻量基础视图
- `build_runtime_state_view()` / `node_states` 提供站点运行态补充信息
- 协调器负责把两者合并，并统一转成前端所需的 WGS84 遥测格式

建议新增一层转换，把它映射到前端现有习惯的字段：

```json
{
  "type": "TICK",
  "payload": {
    "sim_time": 123.4,
    "entities": {
      "trucks": [...],
      "drones": [...],
      "depots": [...],
      "stations": [...]
    },
    "orders": [...],
    "stats": {
      "current_decision": {
        "drone_id": "DRN-01",
        "trigger_type": "truck_station_arrival",
        "trigger_station_id": "ST-03"
      },
      "last_reward_breakdown": {...},
      "mode_c_success_count": 3,
      "fallback_count": 2
    }
  }
}
```

这样前端现有 WebSocket 接收模式几乎不用变，主要是 payload 生产者变了。

## 七、策略决策接口建议

你前一轮已经确认：PPO 不应被硬塞成传统 `/dispatch` 批式求解器。  
在这个“单状态机”方案下，这一点更明确。

建议新增独立控制接口，而不是复用 `/api/sim/dispatch`：

### 7.1 建议接口

#### `POST /api/sim/policy/activate`

作用：

- 创建 `TrainingEnvAdapter`
- 加载 checkpoint
- 创建在线运行协调器
- 注册 WebSocket 快照构造函数
- 启动后台线程

请求体建议包含：

```json
{
  "policy_name": "rh_alns_cmrappo",
  "policy_path": "backend/weights/.../policy.pt",
  "config_path": "backend/config/rh_alns_cmrappo.yaml",
  "deterministic": true,
  "speed_ratio": 1.0,
  "scene_id": "default_test_4x4km"
}
```

#### `POST /api/sim/policy/pause`

作用：

- 暂停在线运行协调器循环

#### `POST /api/sim/policy/resume`

作用：

- 恢复在线运行协调器循环

#### `POST /api/sim/policy/reset`

作用：

- 销毁当前 env/runtime
- 重新 `env.reset()`
- 清空 history / LSTM state

#### `GET /api/sim/policy/state`

作用：

- 返回当前 checkpoint、是否运行、当前 `sim_time`、最近一次决策摘要

### 7.2 为什么不建议继续走 `/dispatch`

`/dispatch` 当前语义是：

- 收集当前 `pending_orders`
- 一次性求解
- 产出 `DispatchPlan`
- 更新订单与实体状态

这与 PPO 的语义不一致。  
PPO 在线模式更像“持续接管”，不是“点击一次算一批”。

## 八、前端展示建议

在这个单状态机方案下，前端应从“看计划图”为主，转为“看运行状态 + 决策流”为主。

### 8.1 继续保留

- 地图实体实时位置
- 订单状态展示
- 最近一次分配/飞行路径的辅助 overlay

### 8.2 新增重点面板

#### 当前策略状态

- `active_policy`
- `checkpoint`
- `deterministic`
- `speed_ratio`
- `sim_time`

#### 最近决策流

- `t_decision`
- `drone_id`
- `trigger_type`
- `selected_order_id`
- `selected_mode`
- `recover_node_id`

#### 当前候选解释

可直接暴露自 `CandidateOutput` 摘要：

- `order_pre_score`
- `priority_band`
- `best_mode_b_return_score`
- `best_mode_c_rendezvous_margin`
- `best_mode_c_truck_eta_remaining`

#### 运行指标

- `dispatch_count`
- `mode_c_success_count`
- `reservation_timeout_count`
- `fallback_count`
- `hard_failure_count`

## 九、与现有代码的对应关系

### 9.1 可直接复用

- WebSocket 注册与广播机制
  - [`telemetry.py`](/Users/myx/Documents/GitHub/HiveLogix/backend/api/websockets/telemetry.py:49)
- `TrainingEnvAdapter` 的状态推进逻辑
  - [`_advance_to_event()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:1497)
- `TrainingEnvAdapter` 的可视化快照
  - [`build_visualization_snapshot()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:913)
- 模型加载与推理
  - [`load_trained_policy()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/policy_inference.py:49)
  - [`run_policy_episode()`](/Users/myx/Documents/GitHub/HiveLogix/backend/training/policy_inference.py:132) 的局部逻辑

### 9.2 只参考结构，不直接复用

- `SimulationEngine._run_loop()`
  - 只参考其 `100ms` 节拍和睡眠补偿结构
  - 不复用其 `EntityManager.tick_all()` / `OrderManager.tick()` / `_try_auto_dispatch()`

### 9.3 不再作为 PPO 在线模式主入口

- `DispatchDecisionEngine`
- `DispatchSolver`
- `/api/sim/dispatch`

它们仍保留给 `greedy / market / incremental route rebuild` 这类批式求解器。

## 十、推荐落地顺序

### 第一步

新增在线运行协调器，但先不接前端：

1. `activate()`
2. `pause()/resume()`
3. `build_tick_payload()`
4. `build_full_snapshot()`

先在后端单测 / 本地脚本中验证：

- 单线程循环稳定
- 决策点能自动被消费
- `env.t_now` 单调递增
- 无双时间轴

### 第二步

把 WebSocket 快照构造函数切到协调器：

- `set_snapshot_builder(coordinator.build_full_snapshot)`
- `broadcast_tick(coordinator.build_tick_payload())`

### 第三步

前端新增“策略接管模式”：

- 激活 / 暂停 / 复位按钮
- 当前 checkpoint 显示
- 最近决策流面板

### 第四步

如果还需要兼容原 `DispatchCenter` 的“路线展示”体验，再补一层：

- 最近一次决策的 UAV 路径 overlay
- 当前 coarse backbone 的 truck overlay

但这应该是附加展示，不应再把 `DispatchPlan` 当主语义中心。

## 十一、最终建议

本轮推荐明确冻结以下原则：

1. **在线 PPO 模式下，`TrainingEnvAdapter` 是唯一状态机**
2. **在线 PPO 模式下，`TrainingEnvAdapter.t_now` 是唯一仿真时间**
3. **`SimulationEngine` 仅可提供循环节拍参考，不再驱动实体物理状态**
4. **前端主通道继续用 WebSocket，但 payload 生产者改为 PPO 在线运行协调器**
5. **`/dispatch` 继续服务传统 solver，PPO 单独走 `policy activate/pause/reset/state` 接口**

如果按这套边界落地，前后端的职责会比“`SimulationEngine + DecisionEngine + TrainingEnvAdapter` 三套语义并存”清晰得多，后续也更容易排查时间轴、位置同步和决策触发问题。
