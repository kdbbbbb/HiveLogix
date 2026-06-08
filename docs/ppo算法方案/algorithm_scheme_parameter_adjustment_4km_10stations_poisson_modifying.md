# RH-ALNS + Shared PPO-Lite 新调度算法接入实施方案（4km × 4km，10 站点，泊松动态订单）

## 0. 文档定位与本次修正原则

本方案用于正式替换原有文档中“RH-ALNS 高频主导 + PPO 辅助微调”的 Phase 5 策略层叙述，并将其重构为一种更接近论文《Distributed Task Allocation Algorithm for Heterogeneous UAVs Based on Reinforcement Learning》的轻量化实现路径。

本次修正的核心结论只有一条：

- 事件驱动环境与物理语义保持不变；
- RH-ALNS 不再承担高频逐单决策器角色，而降为低频粗规划器；
- 高频局部决策改由共享参数的 PPO 策略承担；
- 论文中的 CCT 机制不再用于“凑技能队伍”，而映射为本项目中的 `rendezvous reservation timeout` 机制；
- 推理阶段默认采用 `greedy` 动作选择，以吞吐、稳定性和可解释性优先。

为降低工程切换成本，本方案继续沿用以下兼容性约定：

1. solver 标识短期内仍可继续使用 `rh_alns_cmrappo`
2. 配置文件路径短期内仍建议保留为 `backend/config/rh_alns_cmrappo.yaml`
3. 权重目录短期内仍建议保留为 `backend/weights/rh_alns_cmrappo/<version>/`

但需要明确：

- 上述名称仅是工程兼容层；
- 其内部算法语义已改写为 `RH-ALNS + Shared PPO-Lite`；
- 文档中的所有模块、参数、奖励、触发条件和上线要求，都必须按新架构理解，不再按“ALNS 高频主导”理解。

本次修正遵循 6 条原则：

1. 保留事件驱动环境，因为 `mode C`、`ETA`、能量约束、`queue`、`miss`、`hard failure` 的物理语义不能丢。
2. 共享 PPO 策略必须直接嵌入现有环境语义，而不是另起一套抽象仿真。
3. RH-ALNS 只负责低频、粗粒度、滚动式的上层规划，不再为每个局部决策实时求解。
4. 所有候选动作都必须经过显式 `action_mask` 约束，任何物理非法动作不得进入策略分布。
5. 文档中的参数名必须优先映射到当前项目真实字段；当前项目没有入口的参数必须明确标注为“新增配置”。
6. 本方案不是“在旧方案后追加一个补丁章节”，而是直接重写每个环节的职责边界。

本轮二次确认后，以下 4 个机制选择已正式冻结：

1. `mode A` 不进入 shared PPO，继续由 `RH-ALNS / truck-side coarse planner` 管理
2. reservation 采用“非排他软预留”
3. 第一版采用严格 gating，local policy 不得越过 coarse plan 授权边界
4. 训练阶段固定采用 `centralized_train_only critic`，优先保证训练稳定性

同时补充冻结一条强约束：

5. **禁止支持“无人机等待卡车”**：UAV 在 `station/depot` 回收节点若先于卡车到达，不允许进入等待状态，必须立即触发 fallback 或重决策

---

## 1. 现有代码与场景基线（继续复用）

### 1.1 `default_scene` 仍是首轮验证主输入

当前项目已经具备一套适合作为 P0 基线的固定资产：

- `backend/test_data/default_scene/scene_config.json`
- `backend/test_data/default_scene/entities.json`
- `backend/test_data/default_scene/orders.json`
- `backend/test_data/default_scene/osm_network.geojson`
- `backend/test_data/default_scene/osm_network.xml`
- `backend/test_data/default_scene/buildings.geojson`

其关键事实如下：

1. `scene_id = default_test_4x4km`
2. 场景尺度由 `scene_config.json.bounds` 给出，本质上仍应视为约 `4km × 4km`
3. 实体组成：
   - `1` 个 depot
   - `10` 个 station
   - `1` 辆 truck
   - `12` 架 drone
   - 其中 `9` 架 `LightDrone`、`3` 架 `HeavyDrone`
4. 订单组成：
   - `10` 个静态订单
   - `10` 个预定义动态订单
5. 动态订单 `spawn_sim_s` 分布在 `120s ~ 1740s`

因此，本方案不应再从抽象“4km 地图、10 站点”出发设计训练输入，而应直接把 `default_scene` 作为：

- 固定 benchmark 回放集
- 第一个离线验证集
- SUMO 复核集

### 1.2 现有前后端链路继续保留

当前项目已有可复用的调度链路：

1. `POST /api/sim/init`
2. `OrderManager`
3. `DispatchDecisionEngine`
4. `DispatchSolver` 协议
5. `SimulationEngine`

前端已有可复用链路：

1. `systemStore.initSim(...)`
2. `systemStore.dispatch(...)`
3. `DispatchCenter`

结论：

- 本次调整不是推翻平台结构；
- 而是在既有平台上重构“策略层”与“重规划层”的职责分工。

### 1.3 基于当前项目真实资产的参数标定依据

为避免参数停留在抽象推荐值，本方案后续采用的关键默认值，统一以当前仓库内真实资产为标定依据：

1. 地图尺度
   - `scene_config.json.bounds` 对应的实测宽高约为 `4011.5m × 4011.5m`

2. 站点空间分布
   - 站点最近邻间距最小/均值/最大约为 `720m / 949m / 1479m`
   - depot 到各 station 的距离最小/均值/最大约为 `1018m / 2207m / 3279m`

3. 订单空间分布
   - depot 到订单的一跳距离最小/均值/最大约为 `613m / 2201m / 3406m`
   - 订单到最近 station 的距离最小/均值/最大约为 `203m / 500m / 869m`

4. 动态订单强度
   - `default_scene/orders.json.dynamic_orders` 的 `10` 个动态单分布在 `120s ~ 1740s`
   - 其 benchmark 强度约为 `0.33 ~ 0.37 单/分钟`
   - 因而本文把 `arrival_rate ≈ 0.35~0.40/min` 作为首轮默认动态强度，是有项目内数据支撑的，不是凭空设定

5. 真实运载能力约束
   - 全部 `20` 个 benchmark 订单中：
     - 仅 `7` 个订单可由 `LightDrone` 承运
     - `16` 个订单可由 `HeavyDrone` 承运
     - 有 `4` 个订单超过 `HeavyDrone.payload_capacity=10kg`
   - 这说明 `mode A` 在当前项目里不是边缘模式，而是刚性必要模式，不能并入 UAV local policy

6. 真实速度与续航约束
   - 当前共享配置中：
     - `truck.speed = 15 m/s`
     - `light_drone.cruise_speed = 20 m/s`
     - `heavy_drone.cruise_speed = 15 m/s`
   - 轻型机满载理论航程约 `31km`，重型机满载理论航程约 `32km`
   - 这意味着当前 4km 场景下，电量约束更多体现为“安全余量 + 回收链路合法性”，而不是“平飞距离绝对不够”

7. 站点切换时间量级
   - 按当前真实速度估算，切换到最近邻 station 的平均时间约为：
     - `LightDrone ≈ 47s`
     - `HeavyDrone ≈ 63s`
   - `HeavyDrone` 的最大最近邻切换时间约 `99s`
   - 因此，只要某站的预测等待时间达到 `240s` 量级，继续死等通常就不再划算

本节结论会直接约束后文的 `support_radius_km`、`station_wait_threshold_sec`、`coarse_replan_interval_sec`、`mode A` 边界和 reward 标定。

---

## 2. 新架构总览：Phase 5 轻量替代为 RH-ALNS + Shared PPO-Lite

### 2.1 总体分层

新方案分为 4 层：

1. 事件驱动环境层
   - 保持当前物理语义
   - 负责 `mode C`、`ETA`、能量、排队、错过卡车、硬失败、终端清场

2. 低频粗规划层：RH-ALNS
   - 输出卡车未来骨架路线、订单优先级桶、局部支持半径、可用回收节点池和粗粒度资源预算
   - 只在少数时机触发，不参与逐步局部动作

3. 高频局部策略层：Shared PPO-Lite
   - 对每个异步触发决策的无人机，基于共享参数策略在 `mask` 后候选集中选择动作
   - 决策内容包括：
     - 派哪个订单
     - 选 `mode B` 或 `mode C`
     - 选哪个回收节点

4. 局部承诺与回滚层：Reservation Timeout
   - 由论文中的 CCT 机制映射而来
   - 负责 future station/depot 的局部承诺、超时回滚、软惩罚和 fallback

### 2.2 与论文方法的对应关系

论文中的核心设计与本项目的映射关系如下：

| 论文设计 | 论文中作用 | 本项目映射 |
| --- | --- | --- |
| 去中心化异步决策 | 无需全局调度中心 | 每架无人机在事件触发点独立调用共享策略 |
| shared PPO policy | 多 UAV 共享参数、分布式决策 | 所有 drone 使用同一 actor；训练时固定使用 `centralized_train_only critic` |
| LSTM + attention | 捕捉时序相关性与协同依赖 | 作为轻量骨干网络编码订单、truck/station ETA、queue、energy 与历史 |
| CCT | 防止过度分散导致死锁 | 改造为 rendezvous reservation timeout，防止未来回收点局部承诺长期占位 |
| greedy inference | 推理阶段优先稳定性与效率 | 在线默认 `greedy`，仅训练阶段采样 |

### 2.3 本次取代的旧设计

旧设计中默认 RH-ALNS 频繁重规划并主导大量局部动作。本方案正式取消以下假设：

1. 取消“每次小状态波动都交给 ALNS 重新算一轮”的假设
2. 取消“PPO 只做次级评分或残差修正”的假设
3. 取消“CCT 只对协同执行任务凑队有意义”的假设

取而代之：

1. RH-ALNS 只负责慢变量和上层骨架
2. Shared PPO-Lite 负责快变量和逐步局部动作
3. Reservation timeout 负责局部承诺、释放和软惩罚

---

## 3. 环境语义冻结：事件驱动环境必须保留

### 3.1 保留事件驱动推进的原因

本项目的关键约束不是抽象图搜索，而是具有明确物理含义的时空过程：

- `mode C` 的返程只能回到卡车未来会经过的合法节点
- `t_arrive_truck`（卡车绝对到达时刻）与 `t_arrive_uav`（无人机绝对到达时刻）的先后关系直接决定 catch / miss
- 电量是否足够决定动作合法性与硬失败
- 站点 `parking_slots` 与 FIFO 队列决定等待和排队时间
- fallback 不是“重新打分”，而是真实的后续飞行与恢复过程

因此，离线训练、离线验证和在线推理都必须建立在事件驱动或准事件驱动环境之上。不能把这些过程简化为粗时间步近似，否则会产生：

- 伪 catch / 伪 miss
- 不稳定 `action_mask`
- 奖励与物理事实脱节
- 训练策略在线上失真

### 3.2 模式 B / 模式 C 的正式动作语义

在本方案中，局部策略动作统一写成：

```text
a = (order_i, mode, recover_node_j)
```

其中：

1. `order_i`
   - 本轮要服务的订单

2. `mode`
   - `mode B`：送达后回 depot / station 的固定安全回收逻辑
   - `mode C`：送达后尝试在卡车未来将经过的合法节点完成 rendezvous recovery

3. `recover_node_j`
   - 仅允许为 `depot` 或 `station`
   - 不允许把“道路中间运动中的卡车本体”当作可回收节点

这里必须强调：

- “返回卡车”在本方案中的唯一合法解释，是返回卡车未来路径上的合法交接节点；
- 绝不允许写成“飞向道路中运动中的 truck 实体”。
- 本版本明确禁止“无人机到点后等待卡车”；若 UAV 先到，必须立即 fallback 或重决策。

**时序量统一记号（全文强制）**

- `fly_time(A → B)`（飞行耗时）：duration（持续时间），单位秒
- `t_arrive_uav(A → B)`（无人机到达 `B` 的绝对仿真时刻）：timestamp（绝对时刻）。
  若该飞行段从当前时刻起飞，则 `t_arrive_uav(A → B) = t_now + fly_time(A → B)`；
  若该飞行段从送达时刻起飞，则 `t_arrive_uav(deliver → r_j) = t_deliver + fly_time(deliver → r_j)`
- `t_arrive_truck(r_j)`（卡车到达回收节点 `r_j` 的绝对仿真时刻）：timestamp（绝对时刻），
  固定定义为 `truck_eta_map[r_j]`

禁止在公式里继续混用“无人机 ETA / 卡车 ETA”这类历史写法。所有 mode C 时序判断、
reservation timeout 和 `T_timeout_cost` 必须只使用上述三个量。

### 3.3 模式 C 合法性判定

对任一返程候选 `r_j`，只有同时满足以下条件才能进入 `action_mask`：

1. 节点合法：
   - `node_type(r_j) ∈ {station, depot}`
   - `t_arrive_truck(r_j) > t_deliver`

2. 时序合法：
   - `t_arrive_uav(deliver -> r_j) + delta_safe <= t_arrive_truck(r_j)`

补充本版本约束：

- 不支持 UAV 到点后等待卡车，因此不允许“明显提前到达后原地等待”的方案进入执行；
- 当事件推进到 UAV 先到且卡车未到时，该回收动作立即失效并触发 fallback/replan。

3. 能量合法：
   - `E_need(deliver -> r_j) + E_safe <= E_rem`

4. 节点容量与站点状态未被硬性否决：
   - 若站点可等待或可排队，则允许进入候选集
   - 若节点处于不可达、禁用或明确关闭状态，则必须剔除

其中：

- `delta_safe` 是回收时序安全余度
- `E_safe` 是返程电量安全余量

### 3.4 回收优先级与保底规则

订单送达后，返程目标按以下顺序筛选：

1. 首选合法的 future rendezvous 节点
2. 若不存在，则尝试返回 depot
3. 若 depot 不可达，则飞往最近可达 station
4. 若不存在任何合法安全节点，则判定硬失败

这套规则既服务于动作构造，也服务于 fallback 恢复逻辑。

### 3.5 软失败与硬失败的正式定义

#### 3.5.1 软失败：miss 后仍可自救

触发条件：

1. 订单已经成功送达
2. 无人机原本承诺在某个 future rendezvous 节点回收
3. 实际推进后发生以下任一情况：
   - `t_arrive_uav(current_loc -> r_j) > t_arrive_truck(r_j)`
   - 卡车已离开 `r_j`
   - UAV 先到 `r_j` 且卡车未到（本版本不允许等待）
   - 节点状态变化导致等待已无意义
4. 仍存在其他合法安全节点可飞往

状态转移：

- 无人机进入 `fallback_recovery`
- 重新选择 depot 或最近可达 station
- 记录 soft miss

软惩罚定义：

```text
soft_miss_penalty = lambda_miss * T_fallback
```

其中 `T_fallback` 为 miss 确认后飞往下一合法节点所用时间。

#### 3.5.2 硬失败：无合法安全节点可达

触发条件：

1. 无人机当前不在 `station` / `truck` / `depot`
2. 剩余电量不足以飞往任何合法安全节点
3. 或已出现空中能源失效

硬失败记为：

- `airborne_energy_failure`

后果：

1. 状态层面：无人机退出当前 episode 的可用集合
2. 目标层面：记为大惩罚项

### 3.6 必须维护的状态变量

环境层至少显式维护以下状态：

1. 无人机状态
   - `idle`
   - `flying_to_deliver`
   - `delivered`
   - `return_to_rendezvous`
   - `return_to_depot`
   - `return_to_station`
   - `queueing_at_host`
   - `charging_or_swap`
   - `fallback_recovery`
   - `recovered`
   - `airborne_energy_failure`

2. 站点状态
   - `parking_slots`
   - 当前占用数
   - FIFO 队列
   - 每架无人机入队时间
   - 预测服务完成时间

3. 卡车状态
   - 当前节点
   - 未来骨架路线
   - 各 future station/depot 的 `ETA`
   - route drift 指标

4. 局部承诺状态
   - 当前预留的 `recover_node`
   - reservation 发起时刻
   - reservation 过期时刻
   - reservation 历史统计

### 3.7 必须显式实现的关键事件

1. 订单送达
   - 订单记为 `delivered`
   - 触发返程决策

2. 局部动作选择
   - 构造候选订单、模式与回收节点
   - 应用 `action_mask`
   - 策略输出局部动作

3. reservation 建立
   - 为 future station/depot 记录局部承诺
   - 启动 timeout 计时器

4. 卡车到站
   - 若无人机已在场，则回收成功
   - 若无人机未到，则继续推进并可能触发 miss

5. 无人机到站
   - 若卡车在场则回收
   - 若卡车未到，则不允许等待，立即进入 `fallback_recovery` 或触发重决策
   - 若需要换电，则进入 FIFO 队列

6. reservation timeout
   - 当前局部承诺失效
   - 触发局部回滚与 soft penalty

7. fallback recovery
   - 重选 depot 或最近安全 station
   - 累计 `T_fallback`

8. hard failure
   - 无人机进入 `airborne_energy_failure`
   - 记录大惩罚并移出可用集合

### 3.8 FIFO、工位容量与终端清场

FIFO、工位容量和清场要求不再只是参数，而必须作为正式规则：

1. `parking_slots` 决定并行服务上限
2. 超出容量必须进入 FIFO 队列
3. `swap_time` / `charge_time` 结束后释放工位并唤醒队首
4. 排队时间按“入队时刻到开始服务时刻”精确累计
5. episode 结束时必须清点：
   - 已完成订单
   - 超时订单
   - 仍在返程中的无人机
   - 仍在排队中的无人机
   - 仍在等待卡车的无人机
   - 已发生硬失败的无人机

---

## 4. 论文式 CCT 的正式映射：Reservation Timeout 机制

### 4.1 设计目的

论文中的 CCT 机制用于避免多智能体过度分散地占坑，最终陷入协同死锁。映射到本项目后，主要问题不再是“异构技能队伍凑不齐”，而是：

1. 无人机对 future station/depot 作出局部承诺后长期占位
2. 卡车骨架路线轻微漂移后，旧承诺仍未释放
3. 多架无人机把局部决策押注在同一未来节点，造成等待堆积和 miss 连锁

因此，本方案将 CCT 改写为 `rendezvous reservation timeout`：

- 每个局部承诺都带有过期时间；
- 过期后自动回滚，不依赖中心化协调；
- 回滚附带软惩罚，但不把订单状态回滚；
- 局部超时可级联打散错误承诺链，避免长时间占用未来节点。

### 4.2 机制定义

当无人机在时间 `t` 选择动作 `(order_i, mode C, recover_node_j)` 后，立即进入 `pre-reservation` 阶段，并启动本地计时器：

```text
tau_res = alpha * fly_time(deliver -> r_j)
        + beta  * t_hist(node_type_j, mode C)
        + gamma * q_est(r_j)
```

其中：

- `fly_time(deliver -> r_j)`：送达后飞往目标回收节点的飞行耗时（duration）
- `t_hist(...)`：同类回收节点的历史平均等待/汇合时间
- `q_est(r_j)`：节点当前或预测的排队压力项
- `alpha, beta, gamma`：经验系数

若在区间 `[t, t + tau_res)` 内，该局部承诺仍然满足：

1. `t_arrive_uav(deliver -> r_j) + delta_safe <= t_arrive_truck(r_j)`
2. 能量仍然可达
3. 节点未被关闭
4. route drift 未超过失效阈值
5. UAV 未出现“先到回收节点且卡车未到”的禁用等待场景

则 reservation 持续有效。

若任一条件失效或 timeout 到期，则：

1. 取消 reservation
2. 进入 `fallback_recovery` 或重新决策
3. 施加局部软惩罚

### 4.3 奖励处理

reservation timeout 的局部惩罚记为：

```text
r_timeout = - lambda_res_timeout * T_timeout_cost
```

其中 `T_timeout_cost` 可由以下量组成：

- 已等待时间
- reservation 释放后的追加飞行时间
- 节点排队占用造成的局部损失

训练初期可采用较大的 timeout 惩罚，随后衰减到较小值，以提升早期收敛稳定性。这一做法与论文中 CCT penalty 的课程式衰减一致，但语义上已替换为 rendezvous reservation。

### 4.4 工程约束

该机制必须满足：

1. 完全依赖局部可观测信息和共享环境状态，不引入新的中心调度器
2. 不回滚已送达订单
3. 不允许 reservation 持久化占用未来节点
4. 必须支持 timeout 后自动 fallback

### 4.5 Reservation Protocol 正式定义

为避免与现有 `station queue` 语义冲突，本方案将 reservation 正式定义为“软预留、非排他、容量感知”的局部承诺协议，而不是物理工位锁。

其正式语义如下：

1. reservation 不是物理工位占用
   - 不直接占用 `parking_slots`
   - 不直接占用 FIFO 队列位置
   - 只表示“该 UAV 当前计划在未来某节点完成回收/汇合”

2. reservation 默认非排他
   - 多架 UAV 可对同一 `station/depot` 同时建立 reservation
   - 环境维护 `reservation_count(node)` 作为拥挤信号
   - 若预测等待时间或节点压力过高，则通过 `action_mask` 与惩罚项抑制，而不是硬拒绝

3. reservation 具有容量感知
   - `reservation_count(node)`、`parking_slots(node)`、`predicted_queue_time(node)` 共同决定候选动作是否保留
   - 若 `predicted_queue_time(node) > station_wait_threshold_sec`，则该节点必须从局部候选集中剔除

4. reservation 只对 `mode C` 生效
   - `mode B` 走固定安全回收逻辑，不建立 future rendezvous reservation
   - `mode A` 不进入 reservation protocol

这样定义的原因是：

- 与当前项目已存在的 FIFO/工位容量语义兼容；
- 不把未来承诺误写成即时工位锁；
- 不会与论文中“局部超时回滚”思想冲突；
- 也不会凭空引入新的中心化资源占用机制。

### 4.6 Timeout、Route Drift 与 Queue 的精确定义

#### 4.6.1 Timeout 的判定时机

reservation timeout 在以下任一时机进行检查：

1. 每次局部决策事件触发时
2. 每次 truck 到达或离开节点时
3. 每次 UAV 到达节点、进入等待或结束充换电时
4. 每次 station queue 长度或预测等待时间发生离散变化时
5. 当前仿真时刻首次满足 `t_now >= t_res_expire` 时

即：

```text
timeout(res) =
    (t_now >= t_res_expire)
 OR (t_arrive_uav(current_loc -> r_j) + delta_safe > t_arrive_truck(r_j))
 OR (uav_arrived_before_truck == true)  // 本版本：禁止无人机等待卡车
 OR (energy_feasible == false)
 OR (route_drift_invalid == true)
 OR (node_available == false)
```

只要上述任一条件成立，reservation 即失效。

#### 4.6.2 Route Drift 失效条件

定义 coarse planner 在版本 `v` 下为每个 reservation 节点记录：

- `eta_ref_v(node)`
- `route_index_ref_v(node)`

若更新后满足以下任一条件，则 `route_drift_invalid = true`：

1. 节点已不在 truck 未来骨架路径中
2. `|eta_new(node) - eta_ref_v(node)| > max(60s, 0.15 * tau_res)`
3. 节点在骨架路径中的相对顺序发生逆转或被跳过

上述阈值与论文不冲突，因为论文只要求“局部 timeout + rollback”，并未限定本项目的 drift 判据；这里是对本项目事件驱动环境的必要工程化补充。

#### 4.6.3 与 Station Queue 的关系

reservation 与 queue 的关系正式定义为：

1. reservation 不等于入队
2. UAV 只有在实际到达 station 且需要等待服务时，才进入 FIFO 队列
3. reservation 只影响：
   - 候选动作合法性
   - 预测等待时间
   - 拥堵特征
   - timeout 风险
4. queue 只影响：
   - 实际等待时间
   - `T_queue`
   - 站点可服务开始时刻

也就是说：

- reservation 是“未来意图”；
- queue 是“现实排队状态”。

---

## 5. 决策层重构：RH-ALNS 低频化，Shared PPO 高频化

### 5.1 RH-ALNS 的新职责：低频粗规划器

RH-ALNS 在新方案中的职责缩减为：

1. 生成 truck 的未来骨架路线
2. 给订单池做粗粒度优先级排序与窗口分桶
3. 生成 station/depot 支持半径与可行 recovery 池
4. 给未来一段时间内的回收节点分配粗粒度负载预算
5. 提供 route drift 参考基线

它不再负责：

1. 每次局部起降都重新做高成本优化
2. 每次 miss / queue 小波动都即时重算全局
3. 对单个局部动作直接给出最终决策

### 5.2 RH-ALNS 的触发条件

RH-ALNS 只在以下时机调用：

1. 固定较长的 `replan interval`
2. 新订单累积达到阈值
3. truck route 与上次骨架相比出现显著失真
4. `fallback` / `hard failure` 在短时间窗口内过多

推荐触发规则：

```text
Trigger_ALNS =
    periodic(interval >= T_replan)
 OR backlog_new_orders >= N_new
 OR route_drift_ratio >= rho_drift
 OR fallback_count(window) >= N_fb
 OR hard_failure_count(window) >= N_hf
```

### 5.3 4km × 4km、10 站点场景下的触发建议值

首轮联调建议值如下：

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| `planner.coarse_replan_interval_sec` | `420` | 低频滚动重规划，不再做高频重算 |
| `planner.coarse_new_order_trigger` | `3` | 累积到 3 个新单再触发一次粗规划 |
| `planner.route_drift_trigger_ratio` | `0.18` | truck 未来 ETA 或节点序列偏移超过阈值时重规划 |
| `planner.fallback_burst_trigger_count` | `2` | 300s 窗内出现 2 次 fallback 触发一次上层修正 |
| `planner.fallback_burst_window_sec` | `300` | fallback 观测窗口 |
| `planner.hard_failure_trigger_count` | `1` | 任一 hard failure 均可触发上层修正 |
| `planner.upper_horizon_sec` | `3600` | 1 小时滚动窗 |

这里的关键不是某个数值本身，而是职责边界：

- RH-ALNS 只处理慢变量；
- 快变量交给 shared PPO。

### 5.3.1 ALNS ↔ PPO 正式接口契约

为避免 coarse planner 与 local policy 各自实现一套语义，定义只读的 `CoarsePlanView` 作为正式边界对象。

#### A. Planner 输出结构

```text
CoarsePlanView {
  plan_version: int
  issued_at: float
  valid_until: float
  truck_backbone_route: [node_id]
  truck_eta_map: {node_id -> t_arrive_truck_sec}
  authorized_orders: [order_id]
  order_priority_band: {order_id -> int}
  order_pre_score: {order_id -> float}
  order_mode_mask: {order_id -> {A,B,C}}
  recovery_pool: {order_id -> [node_id]}
  node_load_budget: {node_id -> int}
  route_drift_ref: {node_id -> (eta_ref, route_index_ref)}
}
```

说明：

1. `authorized_orders`
   - 当前 coarse planner 允许局部策略处理的订单集合

2. `order_mode_mask`
   - coarse planner 允许的模式上界
   - local policy 只能在其子集中选择

3. `recovery_pool`
   - 对每个订单给出 coarse planner 允许的回收节点池

4. `node_load_budget`
   - 节点的粗粒度预算，不是硬容量锁

#### B. Policy 输入结构

local policy 的输入由两部分组成：

1. `LocalEventObs`
   - 当前决策 UAV 的局部状态
   - 当前订单、truck、station、queue、energy、reservation 与历史信息

2. `CoarsePlanView`
   - 作为只读条件输入

#### C. 权限边界

第一版中，local policy 的权限严格限定为：

1. 只能在 `authorized_orders` 内选订单
2. 只能在 `order_mode_mask[order]` 内选模式
3. 只能在 `recovery_pool[order]` 内选回收节点
4. 不能改写 `truck_backbone_route`
5. 不能直接增删 coarse planner 的授权集合

唯一例外是安全兜底：

- 当所有 coarse-plan 授权动作都非法，但 depot / nearest safe station 仍可达时，允许进入 `safety fallback action`

这不与论文冲突，因为论文的 decentralized policy 也是在环境约束下做局部选择；这里只是把本项目的 coarse plan 进一步形式化为只读边界。第一版不开放常规越权能力。

#### D. Coarse Plan 更新后的失效规则

当 `plan_version` 从 `v` 更新到 `v+1` 时：

1. 若 reservation 节点不再出现在 `truck_backbone_route` 中，则立即失效
2. 若订单不再属于 `authorized_orders`，则该订单相关未执行局部动作立即失效
3. 若 recovery 节点不再属于 `recovery_pool[order]`，则相关 `mode C` 动作立即失效
4. 已经起飞且订单已送达的 UAV 不回滚订单状态，只重算返程
5. LSTM hidden state 不做全局硬重置，但观测中必须注入 `plan_version_delta`

第 5 条是为了避免与事件连续性冲突；它也不违背论文中的时序建模思路。

### 5.4 Shared PPO-Lite 的新职责：高频局部决策器

共享 PPO 策略在每次局部事件触发时工作，例如：

1. 某无人机完成上一单
2. 某无人机在 depot / station 充换电结束
3. 某局部 reservation 被取消
4. 某候选订单进入可服务窗口
5. 某 station 队列状态显著变化

该策略必须在 `action_mask` 过滤后的候选集中选择：

1. `order_i`
2. `mode ∈ {B, C}`
3. `recover_node_j`

### 5.4.1 动作空间正式定义

本方案正式采用 `factorized action space`，不再保留 `flat` 作为并列默认选项。

动作顺序固定为：

1. 先选 `order_i`
2. 再选 `mode_i`
3. 再选 `recover_node_j`

即：

```text
a_t = (a_order, a_mode, a_recover)
```

采用 factorized 的原因是：

1. 更接近论文里“先对任务做 masked selection”的 actor 形式
2. 更适合当前 `order × mode × recovery` 的组合结构
3. 避免在 10 站点场景下把扁平动作空间无意义膨胀

#### A. 各阶段动作域

1. 订单阶段
   - 候选大小至多为 `max_candidate_orders`

2. 模式阶段
   - 固定为 `{WAIT, B, C}`
   - 但经 `mode_mask(order)` 过滤后，可能只剩其中部分动作

3. 回收节点阶段
   - 当模式为 `B` 或 `C` 时才激活
   - 候选大小至多为 `max_candidate_recovery_per_order`

#### B. `WAIT` 是否必有

`WAIT` 为必备动作，且必须始终存在于模式阶段动作域中。

其语义不是“放弃订单”，而是：

1. 当前不启动新服务动作
2. 保持当前 UAV 局部状态不变
3. 等待下一个环境事件或 coarse plan 更新

若当前没有任何物理合法服务动作，`WAIT` 不能被 mask 掉。

#### C. `mode A` 是否进入 PPO

`mode A` 已正式冻结为不进入 shared PPO 的局部动作空间。

其职责边界是：

1. `mode A` 仍由 RH-ALNS / truck-side coarse planner 管理
2. local policy 只负责 UAV 相关局部动作
3. 若订单被 coarse planner 判定必须走 `mode A`，则该订单不进入 UAV policy 的 `authorized_orders`

这一点与论文不冲突，因为论文中的 agent 本身就是 UAV；将 truck-side mode A 排除在 UAV policy 外是合理的业务适配。

补充说明：

- 当前代码中的 `B_WAIT`、`B_DYNAMIC` 不再作为 policy 一级模式；
- 它们在新方案中应被视为执行层子状态或执行标签：
  - `B_WAIT`：`mode B` 被选中后，在执行层进入等待
  - `B_DYNAMIC`：执行中因事件驱动重选 recovery 后形成的执行标签
- policy 一级模式仍只保留 `{WAIT, B, C}`

#### D. `max_candidate_actions` 的正式含义

虽然策略采用 factorized action space，但仍保留：

- `candidate.max_candidate_actions`

其定义改为：

```text
N_valid_triplets =
  sum_i [ |valid_modes(order_i)| * valid_recovery_count(order_i, mode) ]
```

要求：

```text
N_valid_triplets <= max_candidate_actions
```

若超过上限，则按以下顺序截断：

1. 先按 `order_pre_score` 截断低优先级订单
2. 再对每个订单截断较差的 recovery 候选
3. 不允许截掉 `WAIT`

因此 `max_candidate_actions` 在新方案中是“组合动作预算约束”，而不是 flat logits 的直接维度。

### 5.5 Shared PPO-Lite 的观测设计

参考论文的任务编码、UAV 编码、cross-attention 与 LSTM 设计，本项目中的最小观测应包含：

1. 订单特征
   - 坐标
   - 重量
   - deadline / slack
   - 当前是否已分配
   - 由 coarse planner 输出的优先级

2. 无人机特征
   - 位置
   - 当前状态
   - 剩余电量
   - 剩余服务/返程/等待时间
   - 当前 reservation 状态

3. truck 与站点特征
   - future route 节点序列
   - future station/depot `ETA`
   - station `parking_slots`
   - 当前队列长度
   - 预测等待时间

4. 事件历史特征
   - 最近 `L` 步局部动作
   - 最近 `L` 步 miss / fallback / timeout
   - 最近 `L` 步 route drift 信号

5. 掩码特征
   - 订单是否可服务
   - 回收节点是否可达
   - 模式是否合法

### 5.5.1 Observation Tensor Spec

为保证训练、离线回放和在线推理输入一致，观测张量正式定义为 5 类 token。

#### A. Token 组成

1. `uav_self_token`
   - 当前决策 UAV 的单个 token
   - 包含位置、剩余电量、当前状态、剩余时间、reservation 状态、plan_version_delta

2. `order_tokens`
   - 最多 `max_candidate_orders` 个
   - 每个 token 对应一个候选订单

3. `recovery_tokens`
   - 对当前所选订单的回收节点候选做条件编码
   - 每次最多 `max_candidate_recovery_per_order` 个

4. `infra_tokens`
   - truck 当前骨架摘要 + station/depot 摘要 token
   - 包含 queue、ETA、parking_slots、node_load_budget 等

5. `history_tokens`
   - 最近 `hist_len` 个局部决策事件的摘要

#### B. Padding / Masking 规则

1. `padding_mask`
   - 只用于 attention 层屏蔽无效 token

2. `action_mask`
   - 只用于 actor head 屏蔽物理非法动作

3. `padding_mask` 与 `action_mask` 不允许混用
   - 前者是张量对齐问题
   - 后者是业务合法性问题

#### C. `hist_len=6` 的正式含义

`hist_len=6` 指最近 6 个“局部决策事件步”，不是 6 个固定仿真秒步，也不是 6 个全局环境事件。

局部决策事件包括：

1. 完成上一单后重新决策
2. 充换电完成后重新决策
3. reservation 失效后重新决策
4. 候选集因 coarse plan 更新而重新决策

这样定义与论文中的 LSTM temporal window 更一致，因为论文建模的是 decision sequence，不是固定时间片。

### 5.6 网络结构建议

Shared PPO-Lite 的建议结构如下：

1. Task encoder
   - 对候选订单集合编码

2. UAV / station / truck encoder
   - 对局部载体与基础设施上下文编码

3. Cross-attention
   - 融合“当前决策无人机”和“候选订单/回收节点”关系

4. LSTM temporal unit
   - 建模最近 `L` 步局部决策历史

5. Actor head
   - 输出 masked action logits

6. Critic head
   - 训练时进行价值估计

### 5.6.1 Critic 形式

本方案正式固定：

- `critic_mode = centralized_train_only`

其含义是：

1. Actor 只接收当前 UAV 的局部观测与只读 coarse-plan 条件
2. Critic 在训练时可以额外读取：
   - 全局订单池摘要
   - 全局 UAV 可用性摘要
   - 全局 station queue 摘要
   - 当前 coarse-plan 全局摘要
3. 在线推理时不需要 centralized critic

这不会与论文的改进冲突。论文本身强调 shared policy 和全局上下文编码；这里的 centralized critic 只是训练期稳定化手段，不改变推理时的 decentralized actor。本方案第一版优先训练稳定性，因此不再把 decentralized critic 作为并列默认选项。

这是论文结构的轻量化映射：

- 保留 attention + LSTM 的时序与协同建模思想；
- 不必机械复制其完整多 UAV 协同任务定义；
- 但必须保留 shared policy、mask、history 和 PPO clipped objective 的核心。

### 5.7 推理阶段默认 greedy

在线推理阶段默认使用：

- `greedy`

理由如下：

1. 与论文对 inference 的结论一致
2. 当前业务更重视吞吐、稳定性、可解释性与复现实验一致性
3. 在线调度不应默认引入额外随机性

若需要保留采样型策略，只能作为离线评测或对照实验选项，不应作为线上默认。

---

## 6. 目标函数、奖励与验证口径统一

### 6.1 主目标函数

本方案继续采用单一主目标：

```text
J =
  T_complete
  + lambda_wait * T_wait
  + lambda_queue * T_queue
  + lambda_miss * T_fallback
  + lambda_res_timeout * T_res_timeout
  + Lambda_hard * N_hard_fail
```

其中：

- `T_complete`：总完成时间或总完成时长汇总
- `T_wait`：兼容保留字段；本版本禁用“无人机等待卡车”，该项应为 `0`（不再计入互等语义）
- `T_queue`：补能宿主排队总时间（`station / depot`）
- `T_fallback`：miss 后飞往下一合法节点的总时间
- `T_res_timeout`：reservation timeout 造成的总局部损失
- `N_hard_fail`：硬失败次数

设计要求：

1. 小惩罚项必须属于主目标，而不是游离代理奖励
2. 大惩罚项必须明显高一个量级
3. 离线验证与线上监控必须直接回到该主目标及分解项

### 6.2 PPO 训练奖励的口径

PPO 训练中的 reward 可以采用 episode 末端稀疏主奖励加局部事件辅助惩罚的形式：

1. 终局主奖励
   - `-J`

2. 过程型辅助惩罚
   - reservation timeout penalty
   - soft miss penalty
   - queue accumulation penalty
   - hard failure immediate penalty

但必须遵守一个原则：

- 这些过程惩罚只能是主目标的分解实现，不允许另造一套与业务目标无关的代理 reward。

### 6.2.1 Reward Scale Calibration

当前主目标中的所有时间项统一使用“秒”作为基础量纲，整个目标函数的单位解释为：

- `effective seconds`

也就是说：

1. `T_complete`
2. `T_wait`
3. `T_queue`
4. `T_fallback`
5. `T_res_timeout`

都直接按秒累计。

因此：

- `lambda_wait`
- `lambda_queue`
- `lambda_miss`
- `lambda_res_timeout`

都是“秒到有效秒”的无量纲转换系数。

#### A. 当前推荐数量级的依据

首轮推荐口径如下：

1. `lambda_wait = 0.10`
   - 1 秒等待折算为 0.1 有效秒

2. `lambda_queue = 0.10`
   - 排队与普通等待同量级起步

3. `lambda_miss = 0.15`
   - miss 比普通等待更坏，但仍然属于可恢复扰动

4. `lambda_res_timeout = 0.10`
   - timeout 是局部承诺失效，应与等待同阶

5. `Lambda_hard`
   - 必须直接按“秒级等价损失”定义，而不是小量纲系数

#### B. Hard Failure 为什么必须“足够大”

`N_hard_fail` 是计数项，不是时间项，因此不能再用 `10.0` 这种与秒项同层相加的小系数。

本方案将其重定义为：

```text
Lambda_hard = hard_failure_penalty_sec
```

首轮建议值：

```text
hard_failure_penalty_sec = 1200
```

理由：

1. 4km 图上一次长距离 fallback + 等待 + 重规划的可恢复损失通常在数百秒量级
2. hard failure 必须显著大于一次最坏可恢复扰动
3. 取 1200s 作为起点，等价于“发生 1 次硬失败，至少比一次严重但可恢复的 miss/fallback 链更糟”

这不会与论文冲突。论文讨论的是 sparse reward 稳定训练，并未给出本项目里 `hard failure` 这种业务事件的量纲；这里是本项目必须补上的标定。

#### C. Penalty Sweep 流程

首轮 penalty sweep 建议按以下顺序进行：

1. 固定 PPO 超参数与 coarse planner
2. 扫 `lambda_queue ∈ {0.05, 0.10, 0.20}`
3. 扫 `lambda_miss ∈ {0.10, 0.15, 0.25}`
4. 扫 `lambda_res_timeout ∈ {0.05, 0.10, 0.20}`
5. 扫 `hard_failure_penalty_sec ∈ {900, 1200, 1800}`

评价准则：

1. 不允许通过降低 `hard_failure` 换取表面更短的 `T_complete`
2. 若 `hard_failure` 未显著下降，则较大 penalty 无意义
3. 若 `queue/miss/timeout` 明显下降且 `T_complete` 轻微上升，可接受

### 6.3 训练与验证必须共同输出的指标

1. 完成订单数
2. 总完成时间
3. 无人机等待卡车时间
4. 补能宿主排队时间
5. miss 次数
6. fallback 次数
7. reservation timeout 次数
8. hard failure 次数
9. 模式 B / C 采用比例
10. greedy 推理下的吞吐与稳定性指标

---

## 7. 参数体系重构：结合当前项目真实入口重新整理

### 7.1 旧概念与当前真实字段的映射

| 抽象概念 | 当前项目真实字段或入口 | 当前状态 | 结论 |
| --- | --- | --- | --- |
| 泊松到达率 | `order_gen_config.arrival_rate` | 后端真实生效 | 继续使用 |
| 时间窗 | `order_gen_config.window_min/window_max` | 后端真实生效 | 继续使用 |
| 重量范围 | `order_gen_config.weight_min/weight_max` | 后端真实生效 | 继续使用 |
| 订单总量上限 | `order_gen_config.max_orders` | 后端真实生效 | 继续使用 |
| 突发订单 | `burst_enabled/burst_multiplier` | 后端部分生效 | 可继续使用，但不能写成完整 burst 窗口模型 |
| 地图尺度 | `bbox` / `scene_id` | 运行时真实来源 | 地图宽高由 `bbox` 派生，不新增独立 API 字段 |
| 卡车速度 | 卡车实体配置 `speed` | 已有 | 不新增全局字段 |
| 无人机速度 | `light_drone.cruise_speed` / `heavy_drone.cruise_speed` | 已有 | 不新增全局字段 |
| 共享能耗与服务时长 | `solver_energy.*` | 已有 | 继续复用运行时基础参数 |
| `geo_mode/priority_*` | 前端静态订单生成器 | 后端泊松流未真实消费 | 只能写成前端侧字段，不能宣称后端已支持 |

### 7.2 继续保留的共享运行时参数

`backend/config/drone_params.yaml` 中以下字段应继续作为共享运行时参数：

- `truck_energy_kwh_per_km`
- `uav_energy_model`
- `uav_alpha_wh_per_kg_km`
- `allow_moving_truck_launch`
- `truck_service_time_order_s`
- `drone_service_time_order_s`
- `truck_drone_launch_time_s`
- `truck_drone_recover_time_s`

理由：

- 这些字段描述物理运行时与业务服务时长；
- 不应被写死在单个模型权重里；
- 它们是 shared PPO 与事件驱动环境共同依赖的基础事实。

### 7.3 必须迁入算法专属配置的参数

以下参数不应继续与共享 `solver_energy` 混用，而应迁入算法专属配置：

1. 低频粗规划参数
2. 候选集参数
3. reservation timeout 参数
4. PPO 训练参数
5. 主目标中的惩罚权重
6. inference 策略参数

建议统一放入：

- `backend/config/rh_alns_cmrappo.yaml`

### 7.4 推荐的配置结构

```yaml
scene:
  derive_map_size_from_bbox: true
  default_support_radius_km: 1.2

planner:
  coarse_replan_interval_sec: 420
  coarse_new_order_trigger: 3
  route_drift_trigger_ratio: 0.18
  fallback_burst_trigger_count: 2
  fallback_burst_window_sec: 300
  hard_failure_trigger_count: 1
  upper_horizon_sec: 3600
  incremental_dispatch_min_interval_s: 2.0
  incremental_dispatch_debounce_s: 0.0
  incremental_dispatch_max_wait_s: 5.0

candidate:
  max_candidate_orders: 32
  max_candidate_recovery_per_order: 4
  max_candidate_actions: 128
  station_wait_threshold_sec: 240
  rendezvous_eta_safe_margin_sec: 15
  energy_safe_margin_ratio: 0.10

action_space:
  type: factorized
  enable_wait_action: true
  include_mode_a_in_policy: false

reservation:
  enable: true
  exclusive: false
  alpha: 1.5
  beta: 1.2
  gamma: 0.3
  history_window: 64
  drift_eta_abs_threshold_sec: 60
  drift_eta_rel_threshold_ratio: 0.15
  timeout_penalty_init: 0.50
  timeout_penalty_final: 0.10
  timeout_penalty_decay: 0.995

policy:
  shared_policy: true
  encoder_type: attn_lstm_lite
  d_model: 128
  nhead: 8
  ff_dim: 256
  lstm_hidden: 128
  lstm_layers: 1
  hist_len: 6
  dropout: 0.1
  inference_mode: greedy
  critic_mode: centralized_train_only

reward:
  completion_time_coef: 1.0
  wait_penalty_coef: 0.10
  queue_penalty_coef: 0.10
  miss_fallback_time_penalty_coef: 0.15
  reservation_timeout_penalty_coef: 0.10
  hard_failure_penalty_sec: 1200

training:
  ppo_learning_rate: 0.0003
  rollout_steps: 4096
  batch_size: 512
  ppo_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  entropy_coef: 0.01
  value_loss_coef: 0.5
  max_grad_norm: 1.0

curriculum:
  station_queue_noise: 0.0
  truck_delay_noise: 0.0
  uav_failure_prob: 0.0
  swap_time_noise: 0.0
```

### 7.5 4km × 4km、10 站点场景的推荐解释

这些推荐值的逻辑如下：

1. `coarse_replan_interval_sec = 420`
   - 让 RH-ALNS 明确退到低频层
   - 按 benchmark 动态强度 `0.33~0.37/min` 估算，每 `420s` 平均新增约 `2.3~2.6` 单，恰好落在“每次粗规划处理约 2~3 个新单”的合理区间

2. `coarse_new_order_trigger = 3`
   - 与上面的 `420s` 周期配套
   - 不会因 1 个新单就频繁触发 RH-ALNS，也不会积压到 4~5 单才重规划

3. `station_wait_threshold_sec = 240`
   - 基于当前真实场景，最近邻站点切换平均只需 `47~63s`
   - `240s` 约等于 `4` 个 `swap_time=60s` 服务周期，也约为一次平均换站成本的 `4~5` 倍
   - 继续采用旧的 `480s` 会过度容忍单站死等，与 10 站点密集布局不匹配

4. `max_candidate_actions = 128`
   - 在严格 gating、`mode A` 排除、factorized action 的前提下，`128` 足够覆盖首轮组合动作预算
   - 对当前 benchmark 规模是稳健上限，不会过早因动作截断损失可行解

5. `nhead = 8`
   - 回归论文已验证的稳定默认值
   - `d_model=128` 配 `8` 头时每头维度为 `16`，是标准而稳定的设置
   - 在当前 token 数量级下，没有必要为了压缩模型而先退到 `4` 头

6. `hist_len = 6`
   - 与论文中的时间窗口设置保持一致数量级

7. `clip_coef = 0.2`
   - 与论文 PPO 设置一致，稳定且工程上常用

8. `hard_failure_penalty_sec = 1200`
   - 当前 4km 场景下，一次严重但可恢复的 miss/fallback/queue 链通常在数百秒量级
   - 取 `1200s` 能保证 hard failure 明显劣于任意单次可恢复扰动

9. `inference_mode = greedy`
   - 明确线上默认策略，不再模糊

### 7.6 当前仅前端静态订单生成器生效的字段

以下字段目前仍只能视为前端静态订单生成器字段：

- `geo_mode`
- `cluster_radius_km`
- `priority_urgent`
- `priority_normal`
- `priority_low`
- `burst_duration_s`

因此必须写明：

1. 它们不能被表述为“后端泊松流已真实支持”
2. 若训练环境强依赖这些分布特征，则要么扩展后端 `OrderManager`，要么在训练阶段使用固定订单输入

---

## 8. 训练与上线双阶段落地方案

### 8.1 阶段 A：离线训练

建议新增或重构以下文件：

1. `backend/training/train_cmrappo.py`
   - 语义上改为训练 shared PPO-lite；短期可保留文件名以降低改动面

2. `backend/training/env_adapter.py`
   - 把当前事件驱动环境映射成 PPO 训练环境

3. `backend/config/rh_alns_cmrappo.yaml`
   - 承载 planner / candidate / reservation / policy / reward / training 参数

4. `backend/weights/rh_alns_cmrappo/<version>/policy.pt`
   - 保存共享策略权重

5. `backend/weights/rh_alns_cmrappo/<version>/meta.json`
   - 保存训练元数据、锁定字段和运行时固定快照

### 8.2 离线训练的推荐输入顺序

1. 第一层：固定场景资产
   - `scene_config.json`
   - `entities.json`
   - `osm_network.xml/geojson`

2. 第二层：固定 benchmark 订单
   - `orders.json.static_orders`
   - `orders.json.dynamic_orders`

3. 第三层：随机泊松流扩展
   - 在固定 benchmark 跑稳后，再启用 `arrival_rate > 0`

### 8.3 SUMO 离线复核链路

建议继续保留并强化离线可视化链路：

1. 使用现有 4km 路网
2. 用当前设施分布算法生成 depot / station 坐标
3. 写入 SUMO additional 文件
4. 回放粗规划结果与局部动作结果
5. 先验证场景与动作语义，再进入前后端联调

### 8.4 阶段 B：在线推理接入

建议在线接入顺序如下：

1. 实现 `RhAlnsCmrappoSolver`
2. 在 `backend/solver/factory.py` 注册 `rh_alns_cmrappo`
3. 扩展前端 solver 枚举
4. 前端进入模型模式时读取 `meta.json`
5. 后端在 `/api/sim/init` 与 `/api/sim/dispatch` 进行一致性校验
6. 复用现有 `DispatchPlan` 绘图链路

---

## 9. 与当前工程实现的主要冲突点及正式修正

### 9.1 solver 构造参数模式

现状：

- `create_solver(name, entity_mgr)` 只传 `entity_mgr`

修正结论：

1. 第一阶段不强制改工厂签名
2. `RhAlnsCmrappoSolver` 内部自行读取配置与权重
3. 如需 `order_manager` 视图，参考现有 solver 的后绑定方式

### 9.2 前端 solver 枚举固定

现状：

- 前端只允许 `greedy / greedy_mmce / greedy_mmce_bi / market`

修正结论：

- 必须新增 `rh_alns_cmrappo`

### 9.3 `SimulationEngine` 增量调度节流硬编码

现状：

- `_incremental_dispatch_min_interval_s = 2.0`
- `_incremental_dispatch_debounce_s = 0.0`
- `_incremental_dispatch_max_wait_s = 5.0`

修正结论：

1. 这 3 个值迁入算法配置文件
2. 它们只控制局部决策触发节流，不等价于 ALNS 粗规划周期
3. 不允许再把它们误解为“高频 ALNS 的重规划参数”

### 9.4 后端泊松流未消费 `geo_mode/priority_*`

修正结论：

1. 文档必须明确此事实
2. 首轮训练不应依赖这些后端尚未支持的特征
3. 若后续研究需要，再补 `OrderManager`

### 9.5 地图尺度事实来源统一为 `bbox`

修正结论：

1. 运行时统一以 `bbox` 为准
2. 训练元数据可记录派生宽高
3. 不新增 `map_width_m/map_height_m` 作为在线请求强输入

### 9.6 参数锁定必须做成前后端一体

`meta.json` 需要承担 3 个职责：

1. 记录训练快照
2. 给前端提供禁改/限幅规则
3. 给后端提供强校验规则

不能只做前端 UI 禁用。

### 9.7 当前最大的风险已不是“如何接入 PPO”，而是职责边界混淆

必须明确：

1. RH-ALNS 不再是高频决策器
2. Shared PPO-Lite 才是局部动作选择器
3. Reservation timeout 是 CCT 的业务语义映射
4. 训练、验证和在线都必须共享同一事件驱动语义

---

## 10. 实施优先级清单

### 10.1 P0（必须完成）

1. 冻结第 3 节事件驱动语义与第 4 节 reservation timeout 机制
2. 新增或重构 `backend/config/rh_alns_cmrappo.yaml`
3. 把 RH-ALNS 的角色改为低频 coarse planner
4. 在训练环境中接入 shared PPO-lite 的 masked local action
5. 导出 `policy.pt + meta.json`
6. 直接复用 `default_scene` 做首轮 benchmark
7. 输出 miss / fallback / reservation timeout / hard failure / queue / wait 指标
8. 默认使用 greedy inference 做离线回放

### 10.2 P1（强烈建议）

1. 实现 `RhAlnsCmrappoSolver`
2. 注册 solver 并打通 `DispatchPlan`
3. 后端支持按 `meta.json` 强校验参数
4. 前端扩展 solver 枚举并支持模型模式参数锁定
5. 在 benchmark 稳定后，再接入随机泊松训练

### 10.3 P2（增强）

1. 扩展后端 `OrderManager` 对 `geo_mode/priority_*` 的支持
2. 引入课程学习噪声
3. 做 greedy 与采样 inference 的离线对比
4. 调整 route drift 检测与 fallback 触发的自适应阈值

---

## 11. 验收标准（Definition of Done）

### 11.1 后端验收

1. `solver=rh_alns_cmrappo` 请求返回 200
2. `DispatchPlan` 字段完整
3. 低频 RH-ALNS 与高频局部策略都能独立触发
4. `rh_alns_cmrappo.yaml` 配置变更能真实影响 solver 行为
5. 参数与 `meta.json` 不一致时接口会拒绝请求

### 11.2 策略层验收

1. 局部决策不再高频调用 ALNS
2. Shared PPO-Lite 只在 masked action set 中选动作
3. 非法 `mode C` 回收动作不会进入策略分布
4. reservation timeout 会触发自动回滚与软惩罚
5. greedy inference 能稳定跑完 benchmark 回放

### 11.3 训练验收

1. 训练脚本可运行并保存权重
2. 至少保存一组可复现实验元数据
3. 训练和验证报告能解释：
   - miss 次数
   - fallback 时间
   - reservation timeout 次数
   - 补能宿主排队时间
   - 无人机等待卡车时间（应为 0）
   - hard failure 次数

### 11.4 前端验收

1. 可在 UI 中选择 `rh_alns_cmrappo`
2. 调度后能显示卡车路径和无人机路径
3. 进入模型模式后，锁定字段不可随意修改

---

## 12. 本次修正后的最终结论

1. 本方案已不再采用“RH-ALNS 高频主导 + PPO 辅助微调”的策略层结构，而是正式切换为“RH-ALNS 低频粗规划 + Shared PPO-Lite 高频局部决策”。
2. 论文中的去中心化 shared PPO、attention + LSTM、CCT 和 greedy inference 已被完整映射到当前项目，但映射后的业务语义是：
   - shared PPO 负责局部派单、模式选择与回收节点选择；
   - CCT 改造为 rendezvous reservation timeout；
   - greedy 作为线上默认推理方式；
   - 事件驱动环境与物理语义保持不变。
3. `mode C`、`ETA`、能量、`queue`、`miss`、`hard failure` 的物理含义必须继续由环境显式模拟，不能被简化为纯表格打分。
4. `backend/test_data/default_scene` 仍应作为首轮训练和验证的主输入资产。
5. 参数体系需要明确分层：
   - 基础运行时物理参数继续放在共享配置；
   - planner / reservation / policy / reward / training 参数迁入 `backend/config/rh_alns_cmrappo.yaml`；
   - 训练快照与上线锁定由 `meta.json` 统一承接。
6. 本方案最重要的不是新增更多参数，而是冻结职责边界：
   - ALNS 管慢变量；
   - PPO 管快变量；
   - timeout 管局部承诺回滚；
   - 事件驱动环境管物理真实性。

---

## 13. 下一步建议

推荐按以下顺序推进：

1. 先在代码设计层冻结第 3 节和第 4 节的环境语义与 timeout 机制
2. 再把 `rh_alns_cmrappo.yaml` 改成本文给出的新结构
3. 基于 `default_scene` 跑第一轮固定 benchmark 回放
4. 默认使用 greedy inference 观察吞吐、miss、queue、fallback 与 hard failure
5. 确认 shared PPO-Lite 的局部动作与 RH-ALNS 的低频骨架分工稳定后，再扩展到泊松随机训练

这样能先保证“问题定义正确”，再保证“训练稳定”，最后再做“平台接入与参数锁定”。
