# 离线训练链服务时长与站点停留语义修改方案

## 0. 文档目的

本文档用于补充 `offline_training_and_validation_implementation_plan.md` 当前实现链路中缺失的时间语义，目标是在 **不讨论 `greedy` / `market` 算法实现** 的前提下，为 RH-ALNS + CMRAPPO 离线训练、离线验证、SUMO 离线复核补齐以下 4 类业务时长：
从`backend/config/drone_params.yaml`中读取下面的数值：
- `truck_service_time_order_s`
- `drone_service_time_order_s`
- `truck_drone_launch_time_s`
- `truck_drone_recover_time_s`

本次修改的核心不是“简单把若干秒数相加”，而是：

1. 这些时长必须进入 **事件驱动仿真语义**
2. 这些时长必须进入 **决策 mask / 可行性判断 / ETA 预测**
3. 这些时长必须进入 **卡车骨干路线各站点预计到达时刻**
4. 这些时长必须进入 **训练、验证、SUMO 导出的一致口径**

---

## 1. 术语冻结

### 1.1 本文只使用 ETA，不再混用 EPA

本次语义冻结后，本文统一使用：

- `Truck Station ETA`
  - 含义：**卡车预计到达某个固定节点（station / depot）的时刻**
  - 即 arrival time
- `Truck Station Departure Time`
  - 含义：卡车在该固定节点完成本次停留后离开的时刻
  - 即 departure time

后续代码中：

- 现有 `truck_eta_map` 若保留，必须明确只表示 **到达时刻 arrival**
- 不允许再把 `truck_eta_map` 混用成“到达后可离开时刻”
- 若运行时需要 departure 语义，应新增明确字段，例如：
  - `truck_departure_map`
  - 或 `truck_hold_until_map`

### 1.2 本次冻结的并行语义

#### 1.2.1 卡车客户点服务停留

- 卡车到达 customer 后需停留 `truck_service_time_order_s`
- 该停留时长只阻塞 **卡车自身后续移动**
- 其他无人机继续按各自状态并行运行，不因卡车在 customer 停留而全局暂停

#### 1.2.2 无人机客户点服务停留

- 无人机到达 customer 后需停留 `drone_service_time_order_s`
- 该停留时长只阻塞 **该架无人机自身后续返程**
- 其他无人机和卡车继续并行运行

#### 1.2.3 站点固定停留

- 对于卡车骨干路线中的 **候选充换电站 station**，卡车每次到站都固定停留 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
- 该固定停留与该站是否实际发生放飞或回收无关
- 目的不是模拟精细装卸流程，而是让 **预测的卡车站点到达序列与实际执行偏差更小**

本文对该固定停留窗口的冻结解释为：

- `station.departure_time = station.arrival_time + max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
- 若同站同时存在放飞和回收，并行实现，即放飞 / 回收在语义上视为共享同一个站点停留窗口

#### 1.2.4 无人机放飞时刻

冻结为：

```text
t_launch = t_truck_arrive_station + truck_drone_launch_time_s
```

即：

- 决策触发点仍可发生在卡车 **到站时刻**

#### 1.2.5 无人机可被卡车回收的判定时刻

冻结为：

```text
若 UAV 到达回收站时刻 <= 卡车到达该站时刻
则视为本次 visit 可被该卡车回收
否则视为错过该次回收
```

注意：

- 这里比较的是 **卡车到站 arrival time**
- 不是比较 station departure time
- 即 UAV 若在卡车到站后才到达该站，即使卡车还在这段 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 固定停留窗口内，也不视为被本次卡车回收

---

## 2. 必须冻结的新时间公式

## 2.1 卡车 stop 时序

### 2.1.1 customer

```text
t_depart_customer = t_arrive_customer + truck_service_time_order_s
```

### 2.1.2 station

```text
t_depart_station = t_arrive_station + max(truck_drone_launch_time_s, truck_drone_recover_time_s)
```

这里的站点停留窗口在离线训练链中语义被冻结为：

- 候选 station 的统一停留时长
- 同时承担“放飞/回收窗口”的时间缓冲作用

### 2.1.3 depot

本次不强制给 depot 新增统一固定停留时长，保持现有语义，除非后续明确要求 depot 也进入同一站点窗口模型。

---

## 2.2 无人机派单完成时刻

### 2.2.1 mode C 从 depot 出发

```text
t_deliver_arrive = t_start + fly(depot -> customer)
t_deliver_finish = t_deliver_arrive + drone_service_time_order_s
```

后续返程 / 回收可行性必须从 `t_deliver_finish` 开始计算，不能从 `t_deliver_arrive` 直接起算。

### 2.2.2 riding_with_truck / truck_station_arrival 触发后的放飞

```text
t_launch = t_truck_arrive_station + truck_drone_launch_time_s
t_deliver_arrive = t_launch + fly(station -> customer)
t_deliver_finish = t_deliver_arrive + drone_service_time_order_s
```

即：

- 决策时刻与真实起飞时刻分离
- 到 customer 后还要再加 `drone_service_time_order_s` 服务停留

---

## 2.3 mode C 回收可行性

### 2.3.1 mask / 候选生成期

对任意回收节点 `r`：

```text
t_uav_arrive_r = t_deliver_finish + fly(customer -> r)
可行当且仅当：
t_uav_arrive_r + safe_margin <= t_truck_arrive_r
```

其中：

- `t_truck_arrive_r` 是卡车到达回收站的 ETA arrival
- 不使用 `t_truck_departure_r`
- `safe_margin` 若继续保留现有 `rendezvous_eta_safe_margin_sec`，则仍比较到 arrival 一侧

### 2.3.2 运行时复核期

送达后对 mode C 原选节点复核时，也必须使用同一口径：

```text
t_now_after_delivery = t_deliver_finish
t_uav_arrive_r = t_now_after_delivery + fly(current_pos -> r)
要求：
t_uav_arrive_r + safe_margin <= truck_arrive_r
```

不能再以“卡车 departure 前赶到即可”作为合法条件。

---

## 2.4 订单完成时刻

训练链中 `actual_deliver_time` 必须统一解释为：

- 卡车订单：**设备到达 customer 点的时刻**
- 无人机订单：**设备到达 customer 点的时刻**

因此：

- on-time 判定基于 **到达 customer 点的时刻**
- 若未来显式统计 `T_complete`，也应基于 **到达 customer 点的时刻**

需要特别区分：

- **订单完成时刻**
  - 仍按设备到达 customer 点计算
- **设备后续可用时刻**
  - 仍需额外加上服务停留

例如无人机：

```text
order_complete_time = t_deliver_arrive
device_available_for_return = t_deliver_finish
```

即：

- 订单在 `t_deliver_arrive` 即视为完成
- 但无人机仍需在 customer 点停留 `drone_service_time_order_s`，之后才能开始返程/回收相关动作

---

## 3. 训练主链必须修改的模块

以下为 **离线训练/验证主链 mandatory 修改项**。

---

## 3.1 `backend/training/export_sumo_truck_route.py`

### 必改原因

当前导出的 `truck_execution_route.json` 中：

- customer 默认 `departure = arrival`
- station 默认 `departure = arrival`

这与本次冻结语义冲突，且会导致：

- 卡车骨干路线 ETA 低估
- env adapter 导入后 station/customer 停留全丢失
- SUMO 复核与训练语义不一致

### 必改内容

1. 从共享配置读取：
   - `truck_service_time_order_s`
   - `truck_drone_launch_time_s`
   - `truck_drone_recover_time_s`
2. customer stop 写入：

```text
departure = arrival + truck_service_time_order_s
```

3. 候选 station stop 写入：

```text
departure = arrival + max(truck_drone_launch_time_s, truck_drone_recover_time_s)
```

4. `truck_eta_map` 仍输出 arrival 语义
5. 如后续契约层显式区分 arrival/departure，则导出阶段同步输出 departure map

### 额外必改点

`env_adapter.py` 中 poisson patrol loop 追加的 station / depot stop 当前也是 `departure = arrival`，该逻辑必须同步修正，不能只修 Phase4 静态产物。

---

## 3.2 `backend/solver/decision_engine.py`

本文件属于共享执行链，但本次方案中它不是“可选增强”，而是需要明确到函数级别的改造点。原因是：

- 当前卡车路线的 `arrival/departure` 重定时逻辑已经集中在这里
- 如果这里不改，customer `truck_service_time_order_s` 与 candidate station `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 不会稳定进入共享卡车路线时序

### 3.2.1 必改函数

- [_recalculate_truck_route_timing_for_b_wait()](/Users/myx/Documents/GitHub/HiveLogix/backend/solver/decision_engine.py:798)

### 3.2.2 必改内容

1. 在 `__init__` 中新增加载：
   - `self.TRUCK_SERVICE_TIME_ORDER = runtime_cfg.truck_service_time_order_s`
   - 当前只加载了：
     - `TRUCK_DRONE_LAUNCH_TIME`
     - `TRUCK_DRONE_RECOVER_TIME`
2. `base_services` 初始化时，对 `customer` 节点显式加入：

```text
base_service(customer) = truck_service_time_order_s
```

不能继续让 customer 的基础停留时间默认为 0。

3. `station` 节点不能再只在“有 alloc”时才出现停留。
   - 当前逻辑偏向“只有发生实际放飞/回收时，station 才有 op_hold”
   - 本次冻结后应改为：

```text
base_service(station) = max(truck_drone_launch_time_s, truck_drone_recover_time_s)
```

即：

- 候选 station 到站默认停 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
- 即使该站本轮没有实际 alloc，也不能视为 `arrival = departure`

4. 对“有 alloc 的 station”和“纯路过 station”，卡车停站时长不做区分：
   - 两者统一固定停留 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
   - 即 `station.departure_time = station.arrival_time + max(truck_drone_launch_time_s, truck_drone_recover_time_s)`

5. 二者的区别只体现在无人机事件语义上，而不体现在卡车 `departure_time` 上：
   - 回收：在卡车到站时刻判定，凡 `uav_arrive_station <= truck_arrive_station` 的无人机可被本次 visit 回收
   - 放飞：真实放飞时刻固定为 `truck_arrive_station + truck_drone_launch_time_s`
   - 同站同时存在放飞与回收时，二者按并行口径处理，不将卡车停留累加为 `truck_drone_launch_time_s + truck_drone_recover_time_s`


### 3.2.3 本文件在当前方案中的定位

它的作用不是决定 mode C mask，而是确保共享卡车路线的时序真值本身正确。mask 仍然以训练主链中的 `truck arrival ETA` 为准。

---

## 3.3 `backend/training/env_adapter.py`

本文件是本次改动的核心。

### 3.4.1 必须新增 / 改造的执行语义

#### A. 卡车必须真正尊重 stop 停留窗口

当前训练环境虽然导入了 `arrival_time / departure_time`，但 truck 物理位置推进并未严格执行站点停留窗口。

必须冻结为：

- 卡车在 `[arrival_time, departure_time)` 内固定停在该 stop
- customer 停 `truck_service_time_order_s`
- station 停 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
- 后续节点 arrival 必须以前一 stop 的 departure 为起点推进

即：

```text
next_arrival = prev_departure + travel_time
```

而不是隐式按连续几何路径滑过去。

#### B. 无人机必须有显式 delivery service 阶段

当前实现中虽然订单在 UAV 到达 customer 时可以完成，但设备自身仍必须进入显式的 customer service 阶段。

因此应改成：

1. `FLYING_TO_DELIVER` 到达 customer
2. 立即写入：
   - `actual_deliver_time = arrival_time`
   - 订单转 `COMPLETED`
   - 可在该时刻结算 delivery bonus
3. 进入新状态，例如：
   - `DELIVERY_SERVICE`
   - 或等价命名
4. 持续 `drone_service_time_order_s`
5. 服务结束后再进入：
   - return
   - rendezvous
   - fallback

也就是说：

- 订单完成口径仍是“到达 customer”
- 服务停留只阻塞该 UAV 的后续动作，不推迟订单完成时间

#### C. riding_with_truck 决策的真实起飞时刻必须后移 `truck_drone_launch_time_s`

当前 station arrival 决策时，很多内部计算直接把 `runtime_state.t_now` 当成飞行起点时刻。

必须改为：

```text
t_launch_effective = trigger_station_arrival_time + truck_drone_launch_time_s
```

后续一切预测都从该时刻起算。

#### D. mode C 回收判定必须以 truck arrival 为准

在 station arrival 事件上，回收逻辑必须冻结为：

- 在卡车到站这一刻检查
- 所有 `uav_arrive_station <= truck_arrive_station` 的待回收 UAV 被本次 visit 回收
- `uav_arrive_station > truck_arrive_station` 的 UAV 视为错过本次回收

不能因为卡车还处于 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 停留窗口，就放宽成“到 departure 前都可回收”。

### 3.4.2 需要修改的具体子逻辑

#### `_apply_dispatch_action`

当前直接调度飞行段，未建模 `truck_drone_launch_time_s`。

修改后：

- 若是 `truck_station_arrival` / `riding_with_truck` 触发下的派送动作
  - 先记录 `effective_launch_time = t_now + truck_drone_launch_time_s`
  - 无人机在此之前保持随车 / station 上待起飞
  - 飞行 leg 的 `start_time` 不能再等于当前时刻
- 若是 depot-home mode C
  - 无 truck launch delay
  - 但后续 `drone_service_time_order_s` 仍必须计入

#### `_schedule_flight_leg`

当前默认 `start_time = self._t_now`。

需支持：

- 允许显式传入 `start_time`
- arrival time 按 `start_time + flight_time`
- 这样 riding_with_truck 的真实 delivery ETA 才能正确后移 `truck_drone_launch_time_s`

#### `_process_delivery_event`

当前在到达 customer 时立即：

- 写 `actual_deliver_time`
- 订单转 `COMPLETED`
- 直接衔接返程

必须改成：

1. 到达 customer
2. 立即：
   - 写 `actual_deliver_time = arrival_time`
   - 订单转 `COMPLETED`
   - 结算 delivery bonus
3. 切入 `DELIVERY_SERVICE`
4. 生成 `service_end_time = arrival + drone_service_time_order_s`
5. 在 service end 事件上再：
   - 执行 mode B/mode C 后续转移

不能改成“等 service end 才完成订单”。

#### `_next_event_time`

必须纳入：

- 无人机 delivery service completion
- truck stop departure 相关冻结边界

否则事件驱动主循环无法在正确时刻恢复无人机或卡车。

#### `_sync_in_transit_positions`

必须保证：

- 卡车在 stop window 内位置固定
- 不能仅由 `truck.get_location(t)` 的连续几何插值决定

#### `_estimate_delivery_arrival_time`

当前只算飞行到达 customer。

必须区分：

- `delivery_arrival_time`
- `delivery_finish_time`

对 mask / feasibility / snapshot 要使用后者的地方，必须全部切换。

#### `_estimate_eta_to_available_for_snapshot`

当前很多状态只返回飞行 arrival 或 truck charge 完成。

必须把以下时间计入：

- 待起飞无人机的 `truck_drone_launch_time_s`
- delivery service `drone_service_time_order_s`
- 必要时卡车 customer/service 对可用时刻的影响

### 3.4.3 `WAIT` 相关语义

`WAIT` 本身不新增业务语义，但以下时刻必须受新时间模型影响：

1. riding_with_truck 在当前站点若选择 WAIT
   - 下一次可触发决策的全局时刻，取决于后续站点的 **arrival**
   - 而这些 arrival 已经必须包含前序站点 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` / customer `truck_service_time_order_s`
2. idle WAIT 的“下一个全局决策事件”
   - 若最近会发生 host charge complete / truck station arrival，事件时刻也必须建立在新时序之上

---

### 3.3.4 poisson patrol loop 必须显式修正

以下点必须在 [_append_patrol_loop_if_needed()](/Users/myx/Documents/GitHub/HiveLogix/backend/training/env_adapter.py:2738) 中写明确：

1. station stop 不能再写成：

```text
departure_time = arrival_time
```

2. 必须改为：

```text
departure_time = arrival_time + max(truck_drone_launch_time_s, truck_drone_recover_time_s)
```

3. `t_cursor` 在每个 station 后也必须推进：

```text
t_cursor += max(truck_drone_launch_time_s, truck_drone_recover_time_s)
```

否则虽然 stop 上写了 departure，但后续节点 arrival 仍然会整体偏早。

---

## 3.4 `backend/training/candidate_builder.py`

本文件是本次 **决策 mask 修正** 的核心。

### 3.5.1 必须修正 launch 时刻

当前 `build()` 内对 riding_with_truck 只修正了 launch 位置，没有修正 launch 时间。

因此必须新增：

- `effective_launch_time`
- 对于 `truck_station_arrival` 触发：

```text
effective_launch_time = runtime_state.t_now + truck_drone_launch_time_s
```

### 3.5.2 必须修正 delivery 预测

当前：

```text
t_deliver = now + fly(launch_pos -> customer)
```

修改后：

```text
t_deliver_arrive = effective_launch_time + fly(launch_pos -> customer)
t_deliver_finish = t_deliver_arrive + drone_service_time_order_s
```

之后：

- mode B host 选择时的后续 return 预测，起点应为 `t_deliver_finish`
- mode C recovery feasibility，起点也必须为 `t_deliver_finish`

### 3.5.3 mode C 合法性必须比较 truck arrival

恢复节点 `r` 的合法性应改为：

```text
t_uav_arrive_r = t_deliver_finish + fly(customer -> r)
合法 iff t_uav_arrive_r + margin <= truck_eta_map[r]
```

即：

- `truck_eta_map[r]` 明确表示 truck arrival
- 不与 departure 比较

### 3.5.4 order / recovery 特征中的时间语义

当前特征里虽然没有直接暴露 `t_deliver_finish` 字段，但以下派生量必须间接受其影响：

- `best_mode_b_return_score`
- `predicted_queue_time_est`
- `rendezvous_margin`

尤其：

- `rendezvous_margin` 必须改成基于 `t_deliver_finish`
- 不能继续用“纯飞到 customer 的到达时刻”

---

## 3.5 `backend/environment/state/entity_manager.py`

本文件也不应只留在“共享执行链建议补齐”层面，本次至少有两处需要明确写入。

### 3.5.1 `__init__` 参数加载要补全

当前 [EntityManager.__init__()](/Users/myx/Documents/GitHub/HiveLogix/backend/environment/state/entity_manager.py:70) 只加载了：

- `DRONE_SERVICE_TIME_ORDER`

本次应补充加载：

- `TRUCK_SERVICE_TIME_ORDER`
- `TRUCK_DRONE_LAUNCH_TIME`
- `TRUCK_DRONE_RECOVER_TIME`

其中：

- `TRUCK_SERVICE_TIME_ORDER` 直接关系到 customer stop 时序
- `TRUCK_DRONE_LAUNCH_TIME` / `TRUCK_DRONE_RECOVER_TIME` 虽然当前不一定都在 `EntityManager` 内直接参与计算，但作为共享执行层常量应一并就位，避免再次出现训练链与执行链口径分裂

### 3.5.2 `_get_truck_wait_stop()` 需要把 `station` 纳入冻结集合

当前 [_get_truck_wait_stop()](/Users/myx/Documents/GitHub/HiveLogix/backend/environment/state/entity_manager.py:576) 只对：

- `customer`
- `recovery`

生效。

本次应改为：

- `customer`
- `recovery`
- `station`

原因是：

- station stop 已被冻结为 `arrival + max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 的 departure
- 如果这里不冻结，卡车物理位置仍会在 station window 内继续推进，导致“时刻表停了、物理位置没停”

### 3.5.3 `_handle_truck_stop_event()` 不需要新增 station 事件逻辑

这里要特别说明：

- `station` 节点当前主要负责 drone recovery
- 本次不要求在该函数内新增额外 station 业务动作
- station 停留本身由 `departure_time` 与 `_get_truck_wait_stop()` 的冻结机制控制即可

---

## 3.6 `backend/training/test_env_adapter_phase5a.py`
## 3.7 `backend/training/test_env_adapter_phase5b.py`
## 3.8 `backend/training/test_env_adapter_phase5c.py`
## 3.9 `backend/training/test_phase6_integration.py`

这些测试都必须补改。

### 必须新增的断言方向

1. customer stop 的 departure 正确为 `arrival + truck_service_time_order_s`
2. station stop 的 departure 正确为 `arrival + max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
3. riding_with_truck 派送时：
   - launch 发生在 `station arrival + truck_drone_launch_time_s`
   - 不是 station arrival 即刻
4. UAV 到达 customer 后：
   - 订单应立即完成
   - 但 UAV 必须等待 `drone_service_time_order_s` service end 后才能进入返程/回收
5. mode C recovery mask：
   - 以 truck arrival 判断能否赶上
   - `arrival 后、departure 前` 到站的 UAV 不应被视为可回收
6. `actual_deliver_time`：
   - 记录 customer arrival time
   - 不是 service finish time
7. snapshot / candidate / action lookup：
   - 与新的 launch/service 口径一致

---

## 4. 对决策 mask 的具体影响

本节是本次修改最容易遗漏、但必须落实的部分。

## 4.1 影响来源

mask 不能再只看：

- 当前位置
- 飞行距离
- truck 骨干 arrival

还必须引入：

1. 卡车 customer 服务停留 `truck_service_time_order_s`
   - 影响后续所有 station 的 truck arrival ETA
2. 每个候选 station 的固定停留 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
   - 影响后续 station 的 truck arrival ETA
   - 影响当前站点的真实 launch time
3. UAV customer 服务停留 `drone_service_time_order_s`
   - 影响 mode B / mode C 后续 return feasibility

## 4.2 riding_with_truck 触发下的 mask

在 `truck_station_arrival` 触发点：

- 决策时刻是 `t_arrive_station`
- 但 UAV 真正能起飞的最早时刻是 `t_arrive_station + truck_drone_launch_time_s`

因此：

- 所有 mode B / mode C 候选的 delivery ETA 必须从 `+truck_drone_launch_time_s` 起算
- 不能再默认从 `t_now` 起飞

## 4.3 mode C 候选回收点的 mask

任意回收点 `r`：

```text
t_uav_arrive_r =
  t_launch
  + fly(launch -> customer)
  + drone_service_time_order_s
  + fly(customer -> r)
```

合法 iff：

```text
t_uav_arrive_r + rendezvous_eta_safe_margin_sec <= truck_arrive_r
```

不是：

```text
<= truck_depart_r
```

## 4.4 mode B host 选择

mode B 在送达后选择 return host 时，虽然不需要追 truck，但必须使用：

```text
t_return_start = t_deliver_finish
```

不能继续默认：

```text
t_return_start = t_deliver_arrive
```

否则：

- mode B host score 偏乐观
- 电量和可用时刻预测偏乐观

---

## 5. 对卡车骨干路线 ETA 的影响

## 5.1 必须进入骨干 ETA 的时间

卡车到达每个 station 的 ETA 必须显式包含：

1. 前序道路 travel time
2. 前序 customer 的 `truck_service_time_order_s` 停留
3. 前序候选 station 的 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 停留

即后续站点的 truck arrival 不是纯路程累计，而是：

```text
arrival(next) = departure(prev) + travel(prev -> next)
```

## 5.2 这项修改直接影响

1. coarse plan 的 `truck_eta_map`
2. candidate builder 的 mode C feasibility
3. WAIT 动作下下一次决策事件的实际时刻
4. SUMO 离线复核路径时间轴

---

## 6. 当前最小闭环中可降级的点

以下两项不是当前语义下的硬前置，不必在这轮最小闭环里强制落地。

## 6.1 `backend/training/contracts.py`

当前不要求在 [CoarsePlanView](/Users/myx/Documents/GitHub/HiveLogix/backend/training/contracts.py:112) 中新增：

- `truck_departure_map`
- 或 `truck_hold_until_map`

原因是本次已冻结：

- mode C 回收判定比较的是 **truck arrival**
- 不是 **truck departure**

因此当前最小闭环里：

- `truck_eta_map` 保持 arrival-only 即可
- 足以支撑：
  - mode C mask
  - recovery candidate feasibility
  - 相关时间预测

## 6.2 `backend/training/planner_bridge.py`

当前也不要求 `PlannerBridge` 同步把 departure 上升为训练公共契约。

原因不是 departure 不重要，而是：

- departure 目前可以先只存在于：
  - `BackboneVisit`
  - `PlannedStop`
  - `truck_execution_route.json`
  - 运行时内部 stop/freeze 逻辑
- 不必立即暴露给训练侧所有消费方

### 6.2.1 这意味着什么

当前最小闭环中可以接受：

1. `truck_eta_map` 继续只表达 arrival
2. truck stop freeze / departure 推进仅在运行时内部使用 departure
3. 不新增公共 `departure` 契约字段

### 6.2.2 什么时候再升级为公共契约

只有当后续出现以下需求时，才建议把它们升级为硬需求：

1. 决策层显式使用 truck departure 做合法性判断
2. observation / snapshot 需要公开同时暴露 arrival 与 departure
3. 想避免 departure 只存在于 env 内部，而希望训练侧所有模块统一消费

---

## 7. 与共享执行链保持一致的建议修改

本节不属于“离线训练主链最小闭环”的 mandatory 范围，但若后续要保证训练 / 在线仿真 / 前端可视化口径一致，建议同步修改。

## 7.1 `backend/environment/state/entity_manager.py`

当前该文件已经对 UAV customer service 有部分实现，但还不够：

1. truck wait stop 只冻结 `customer / recovery`
   - 应把 `station` 也纳入冻结窗口
2. 卡车到站回收逻辑需明确按 truck arrival 判定
3. truck customer completion 时刻应对齐 `arrival + truck_service_time_order_s`

## 7.2 `backend/solver/decision_engine.py`

当前该文件已部分接入：

- `truck_drone_launch_time_s`
- `truck_drone_recover_time_s`
- `DRONE_SERVICE_TIME_ORDER`

但仍建议补齐：

1. 基础卡车路线中，候选 station 默认 service time = `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
2. 不是只在“有实际回收/放飞”时才给 station 增加停留
3. 明确区分：
   - truck station arrival ETA
   - station departure time

## 7.3 `backend/api/routes/simulation_bp.py`

若前端仍要展示 mode B / mode C 重建路线，则路由序列化中：

- 任何与追车可行性相关的比较，都应对齐本次 arrival-only 回收口径

---

## 8. 推荐实施顺序

建议按以下顺序落地，避免中间状态语义混乱。

1. 先改 `export_sumo_truck_route.py`
   - 先把 Phase4 路线 stop 的 arrival/departure 资产改正确
2. 再改 `decision_engine.py`
   - 先把共享卡车路线时序中的 customer `truck_service_time_order_s` 与 station `max(truck_drone_launch_time_s, truck_drone_recover_time_s)` 基础停留补齐
3. 再改 `env_adapter.py`
   - 先修 truck freeze
   - 再修 patrol loop 的 station departure / `t_cursor`
   - 再修 UAV delivery service
   - 再修 launch `truck_drone_launch_time_s`
   - 最后修 recovery arrival-only 判定
4. 再改 `candidate_builder.py`
   - 把 mask 与候选时间口径切到新语义
5. 再改 `entity_manager.py`
   - 补全参数读取
   - 把 station 纳入 truck freeze 集合
6. 最后统一修测试
   - 先 phase5a/b/c
   - 再 phase6 integration

---

## 9. 本次修改后应满足的验收条件

## 9.1 训练环境语义验收

1. 卡车在 customer 真实停留 `truck_service_time_order_s`
2. 卡车在候选 station 真实停留 `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
3. 无人机 customer 到达后真实停留 `drone_service_time_order_s`
4. riding_with_truck 触发下无人机实际起飞时刻为 `station arrival + truck_drone_launch_time_s`
5. mode C 合法性按 truck arrival 判定，而不是按 departure 判定
6. `actual_deliver_time` 统一等于设备到达 customer 点的时刻

## 9.2 决策侧验收

1. 所有 mask / ETA 预测均纳入：
   - truck customer `truck_service_time_order_s`
   - station `max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
   - UAV customer `drone_service_time_order_s`
2. `truck_eta_map` 明确只表示 truck 到达各站点的 ETA
3. arrival / departure 不再混用

## 9.3 SUMO 复核验收

1. `truck_execution_route.json` 中 customer stop 存在 `arrival != departure`
2. `truck_execution_route.json` 中 station stop 存在 `departure = arrival + max(truck_drone_launch_time_s, truck_drone_recover_time_s)`
3. 训练环境与 SUMO 导出的卡车时间轴一致

---

## 10. 本次冻结的最终口径

最后再次强调本次最关键的 4 条冻结口径：

1. **卡车预计到达各站点的时间** 指的是 `arrival time`，本文统一称 `ETA`
2. **无人机被放飞时刻** 为 `卡车到站时间 + truck_drone_launch_time_s`
3. **无人机能否被卡车回收** 以 `UAV 是否在卡车到站时刻及之前到达站点` 判定
4. **customer / station 服务停留** 必须进入物理仿真、mask、候选时间预测、设备后续可用时刻和骨干路线 ETA
