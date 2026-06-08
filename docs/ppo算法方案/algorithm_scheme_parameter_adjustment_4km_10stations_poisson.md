# 参数调整版（面向 4km × 4km 地图、10 个充换电站、泊松动态订单）

## 0. 适用场景与调整原则

本版参数专门针对以下场景做调整：

- 地图大小：**4 km × 4 km**
- 充换电站数量：**10 个**
- 动态订单：按**泊松过程**生成
- 总体算法框架保持不变：**上层 RH-ALNS + 下层 CMRAPPO**

这意味着你原先参数里有几项需要重点改：

1. **地图尺度从抽象/归一化小图，变成 16 km² 实地图**
2. **站点密度较高**，所以站点支撑半径、等待阈值、回收节点候选数都要改
3. **动态订单是泊松到达**，所以滚动重规划周期、触发阈值、训练课程和 PPO 采样长度要更稳一些
4. 神经网络主干结构建议**先不大改**，继续保留参考论文里已经验证过较稳的 **d_model=128、8 头注意力、LSTM 时间窗 L=6** 作为骨干默认值；论文中 PPO 的基线超参数也可以继续作为出发点，但在你的随机到达场景下建议把学习率略降、rollout 略增。

---

## 1. 基础尺度统一：强烈建议全部改成“米-秒”制

### 1.1 坐标与距离单位

不要再用 `map_size: 1.0` 这种归一化写法，直接统一成真实尺度：

```yaml
env:
  map_width_m: 4000
  map_height_m: 4000
```

### 1.2 距离计算建议

- 无人机：欧氏距离
- 卡车：若没有真实路网，则用  
  $$
  d^{road}_{ab}=1.25\cdot d^{euclid}_{ab}
  $$

### 1.3 速度建议

若你目前没有实测参数，建议先用下面这一版：

```yaml
truck_speed_mps: 8.3        # 约 30 km/h
uav_speed_mps: [14.0, 16.0, 18.0]
```

这组参数的直觉是：

- 4 km × 4 km 地图的对角线约为  
  $$
  \sqrt{4^2+4^2}=5.66\text{ km}
  $$
- 若无人机速度约 16 m/s，则跨越全图对角线飞行时间约 354 s（约 5.9 分钟）
- 若卡车有效速度约 8.3 m/s，再乘以绕行影响，则跨图时间通常显著长于无人机，这与“卡车作为骨架、无人机做局部快响应”是匹配的

---

## 2. 基于 10 个站点密度，重调站点相关参数

## 2.1 站点支撑半径 $R_s$

原方案中给的 $R_s=8$ km 在你的地图里已经明显过大，因为整张图边长都只有 4 km。

你的场景里：

- 总面积：
  $$
  A=16 \text{ km}^2
  $$
- 站点数：
  $$
  |\mathcal S|=10
  $$
- 平均每站覆盖面积：
  $$
  A_s = 16/10 = 1.6 \text{ km}^2
  $$

若把每个站的服务区域近似成圆，则等效半径约为：

$$
r_{eq}=\sqrt{\frac{1.6}{\pi}}\approx 0.71 \text{ km}
$$

为了保留一定重叠与调度灵活性，建议把支撑半径设为：

```yaml
support_radius_km: 1.2
```

即：

$$
R_s=1.2 \text{ km}
$$

### 建议区间
- 保守：1.0 km
- 推荐：**1.2 km**
- 偏宽松：1.5 km

---

## 2.2 回收站点候选数

由于你有 10 个站点，原先“每个订单最多考虑 3 个回收站点”略偏紧。

建议改成：

```yaml
max_candidate_recovery_per_order: 4
```

若你显卡算力足够，也可以试：

```yaml
max_candidate_recovery_per_order: 5
```

但我更推荐先用 **4**，原因是：

- 3 个有时会漏掉“稍远但几乎不排队”的优质站点
- 5 个以上又会明显增加候选动作数，拖慢 PPO

---

## 2.3 站点等待阈值

由于 10 个站点已经比较密，系统应当倾向于“换站”，而不是长时间死等某一个站点。

因此把原先较宽的等待阈值收紧：

```yaml
station_wait_threshold_sec: 480
```

即：

$$
w_s^{pred,max}=480\text{ s}
$$

### 建议区间
- 高动态、高密度站点：420 s
- 默认推荐：**480 s**
- 若你发现频繁换站导致空驶过高：放宽到 600 s

---

## 2.4 站点排队惩罚权重

因为现在站点多，**排队不应再被“过度容忍”**，所以建议略提高 queue 相关惩罚：

```yaml
reward:
  lambda_queue: 0.20
```

比原来更大一些，鼓励策略主动避开拥堵站。

---

## 3. 动态订单为泊松到达时，重调滚动规划参数

## 3.1 泊松生成方式

若动态订单按齐次泊松过程生成，则：

$$
N(t+\Delta)-N(t)\sim \text{Poisson}(\lambda \Delta)
$$

相邻订单到达间隔满足：

$$
\Delta t \sim \text{Exp}(\lambda)
$$

其中 $\lambda$ 建议统一成 **单/分钟**。

代码里建议这样写：

```python
delta_t_min = np.random.exponential(scale=1.0 / lam)
delta_t_sec = 60.0 * delta_t_min
```

---

## 3.2 上层重规划周期：改成“随 $\lambda$ 自适应”

原先固定 600 s 在你的场景里偏慢，特别是订单泊松到达时，系统会频繁出现“骨架已过时”。

建议把上层重规划周期设成：

$$
\Delta_H^\star = \operatorname{clip}\left(\frac{2.5}{\lambda}, 3, 8\right)\text{ min}
$$

换成秒就是：

$$
\text{upper\_replan\_interval\_sec}
=
60\cdot
\operatorname{clip}\left(\frac{2.5}{\lambda}, 3, 8\right)
$$

这里的含义是：  
**希望每个上层重规划周期内，平均有 2～3 个新动态订单出现。**

### 如果你还没定 $\lambda$，先用默认值

我建议先按“中等动态强度”调：

$$
\lambda=0.4 \text{ 单/分钟}
$$

则：

$$
\Delta_H^\star=\frac{2.5}{0.4}=6.25\text{ min}
$$

你可以直接取：

```yaml
upper_replan_interval_sec: 360
```

### 推荐分档
- 低强度：$\lambda \le 0.2$ 单/分钟 → 480 s
- 中强度：$0.2<\lambda\le 0.5$ → **360 s**
- 高强度：$\lambda>0.5$ → 180~300 s

---

## 3.3 新订单触发阈值 $N_{new}$

原先 `N_new >= 4` 在泊松动态下偏大，会让上层反应滞后。

建议改成：

```yaml
upper_replan_new_order_trigger: 2
```

如果你想做自适应，可用：

$$
N_{new}^{trigger}
=
\operatorname{clip}
\left(
\left\lceil 0.8\lambda \Delta_H^\star \right\rceil,
2,4
\right)
$$

若采用默认 $\lambda=0.4$、$\Delta_H^\star=6.25$ min，则期望新单数约 2.5，触发阈值取 **2** 最合适。

---

## 3.4 上层规划时域 $H_{roll}$

由于地图不大、站点较多、动态订单持续到达，原先 5400 s（90 分钟）略长，会导致上层规划过于“看远”。

建议改成：

```yaml
upper_horizon_sec: 3600
```

即：

$$
H_{roll}=3600\text{ s}
$$

### 推荐区间
- 高动态：2400~3000 s
- 默认推荐：**3600 s**
- 若动态订单极少：可放宽到 4200 s

---

## 3.5 ALNS 迭代数与破坏比例

因为你会更频繁重规划，所以每次上层求解不必太重。

建议把：

- `alns_iters: 400` 改为 **250**
- `destroy_ratio: 0.15` 改为 **0.12**

推荐：

```yaml
upper:
  alns_iters: 250
  destroy_ratio: 0.12
  sa_temp0: 4.0
  sa_cooling: 0.996
  operator_update_every: 40
```

---

## 4. 下层候选动作参数：针对 10 站点与泊松新单做调整

## 4.1 候选订单数

在泊松动态下，未完成订单池会波动。原先 32 基本够用，但你现在站点多、回收方案多，候选集需要稍微宽一点。

推荐：

```yaml
max_candidate_orders: 36
```

### 建议区间
- 训练早期：24~32
- 正式训练：**36**
- 若动态单非常密、GPU 允许：40

---

## 4.2 候选动作总数上限

由于回收站点候选从 3 提高到 4，建议把动作上限从 128 放宽到：

```yaml
max_candidate_actions: 160
```

若显存紧张，就继续用 128；  
若你发现“动作经常被截断”，就升到 160。

---

## 4.3 订单预筛选分数权重

在泊松到达环境下，订单新旧会不断变化，因此建议更强调“剩余 slack”与“是否逾期”。

建议：

$$
Score_i^{pre}=
1.2 \cdot \frac{\omega_i}{1+\text{slack}_i^+}
+0.4 \cdot \frac{1}{1+d_{near}(i)}
+1.8 \cdot \mathbf 1\{i\text{已逾期}\}
$$

即把原先推荐值调整为：

```yaml
pre_score:
  zeta_1: 1.2
  zeta_2: 0.4
  zeta_3: 1.8
```

---

## 5. 奖励函数参数：针对“连续动态到达 + 多站点”重调

在你的场景里，我建议：

- 更强调超时惩罚
- 略增强站点拥堵惩罚
- 略减空驶惩罚
- 增强终局未完成惩罚

推荐改成：

```yaml
reward:
  lambda_late: 3.0
  lambda_wait: 0.25
  lambda_queue: 0.20
  lambda_empty: 0.15
  complete_bonus: 1.5
  rollback_penalty: 0.8
  terminal_unserved_penalty: 8.0
```

### 调整逻辑

#### 1）`lambda_late` 提高到 3.0
泊松动态订单意味着“未来还会继续来单”，如果 lateness 权重偏小，策略容易一直拖延旧订单。

#### 2）`lambda_wait` 略降到 0.25
因为有 10 个站点，纯等待本身未必是核心问题，真正要打击的是“在拥堵站上长等”。

#### 3）`lambda_queue` 提高到 0.20
因为站点多，所以更应该鼓励智能绕开拥堵。

#### 4）`lambda_empty` 降到 0.15
你现在站点密度较高，适当空驶去更优站点往往是值得的，不应罚得太重。

#### 5）`terminal_unserved_penalty` 提高到 8.0
动态订单环境下，终局还有未服务订单要给更强约束。

---

## 6. CCT / SR-CCT 参数重调

参考论文中的 CCT 机制是用于防止多 UAV 协作死锁，文中经验值为 $\alpha=1.5,\beta=1.2$，并采用从较大到较小的 timeout penalty 衰减思路。你的问题虽然不是原论文那种协作凑组任务，但“预承诺—等待—回滚”的思想仍然适用。

但在你的 4km × 4km、10 站点场景里，飞行距离更短、替代站点更多，所以 timeout 不应过长。

建议把 SR-CCT 改成：

```yaml
cct:
  alpha: 1.25
  beta: 1.00
  gamma: 0.60
  init_penalty: 0.40
  final_penalty: 0.10
```

即：

$$
\tau_{cct}(a)=
1.25\cdot \hat\tau^{fly}(a)
+1.00\cdot \hat w^{queue}(r)
+0.60\cdot \max(0,\eta_T(r)-t_r^{arr}(a))
$$

### 为什么这么调

- `alpha` 降低：因为图不大，飞行时间短，没必要给过长预承诺时间
- `beta` 降低：站点多，排队长时更应该尽快切站
- `gamma` 降低：卡车未来 ETA 仍要考虑，但不应该压倒无人机本地决策

---

## 7. PPO 参数：建议“更稳一点”

参考论文给出的 PPO 基线是：  
学习率 $3\times10^{-4}$、$\gamma=0.99$、GAE $\lambda=0.95$、clip $0.2$、buffer 2048、4 轮更新、batch 64；网络方面采用 $d=128$、8 头注意力、LSTM 时间窗 $L=6$。这些都可以作为你当前实现的出发点。

但因为你的环境里有：

- 泊松到达
- 上层滚动规划变化
- 站点排队随机性
- 更复杂的动作 mask

所以我建议 PPO 稍微调得稳一点：

```yaml
model:
  d_model: 128
  nhead: 8
  ff_dim: 256
  lstm_hidden: 128
  lstm_layers: 1
  hist_len: 6
  dropout: 0.10

ppo:
  lr: 2.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.20
  value_coef: 0.5
  entropy_coef: 0.015
  rollout_steps: 4096
  update_epochs: 4
  minibatch_size: 128
  max_grad_norm: 1.0
  normalize_advantage: true
  target_kl: 0.015
```

## 7.1 为什么这样改

### 学习率：从 `3e-4` 降到 `2e-4`
动态订单随机性更大，降低学习率通常会更稳。

### `rollout_steps`：从 `2048` 提到 `4096`
泊松到达会让单次 rollout 里的状态分布波动更大，多采一些轨迹更稳。

### `minibatch_size`：从 `64` 提到 `128`
与更长 rollout 匹配，减少梯度噪声。

### `entropy_coef`：提高到 `0.015`
训练早期需要鼓励探索更多站点回收与模式切换策略。

---

## 8. 训练课程：要显式加入泊松强度课程

如果你直接从高强度泊松到达开始训，PPO 很容易不稳定。建议分三阶段：

### Stage A：低动态
```yaml
curriculum_stage_a:
  lambda_poisson_per_min: [0.10, 0.20]
  orders_static: [8, 12]
  num_uav: [2, 3]
  num_station: 10
```

### Stage B：中动态
```yaml
curriculum_stage_b:
  lambda_poisson_per_min: [0.20, 0.50]
  orders_static: [12, 18]
  num_uav: [3, 5]
  num_station: 10
  station_queue_noise: true
  truck_delay_noise: true
```

### Stage C：高动态 + 干扰
```yaml
curriculum_stage_c:
  lambda_poisson_per_min: [0.50, 0.80]
  orders_static: [15, 22]
  num_uav: [4, 6]
  num_station: 10
  station_queue_noise: true
  truck_delay_noise: true
  uav_failure_prob: [0.00, 0.05]
  swap_time_noise: true
```

---

## 9. 直接可覆盖的最终参数模板（推荐版）

下面这份 YAML 你可以直接作为当前场景的**推荐版默认参数**：

```yaml
env:
  map_width_m: 4000
  map_height_m: 4000
  road_detour_factor: 1.25

  truck_speed_mps: 8.3
  load_time_sec: 60
  drop_time_sec: 30
  recovery_time_sec: 30

  upper_replan_interval_sec: 360
  upper_horizon_sec: 3600
  upper_replan_new_order_trigger: 2

  support_radius_km: 1.2

  max_candidate_orders: 36
  max_candidate_recovery_per_order: 4
  max_candidate_actions: 160
  station_wait_threshold_sec: 480

uav_defaults:
  cap_kg: [2.0, 2.5, 3.0]
  e_max: [100.0, 110.0, 120.0]
  e_safe_ratio: 0.15
  speed_mps: [14.0, 16.0, 18.0]
  alpha: [1.00, 1.08, 1.15]
  beta: [0.10, 0.12, 0.14]

upper:
  alns_iters: 250
  destroy_ratio: 0.12
  sa_temp0: 4.0
  sa_cooling: 0.996
  operator_update_every: 40

pre_score:
  zeta_1: 1.2
  zeta_2: 0.4
  zeta_3: 1.8

reward:
  lambda_late: 3.0
  lambda_wait: 0.25
  lambda_queue: 0.20
  lambda_empty: 0.15
  complete_bonus: 1.5
  rollback_penalty: 0.8
  terminal_unserved_penalty: 8.0

model:
  d_model: 128
  nhead: 8
  ff_dim: 256
  lstm_hidden: 128
  lstm_layers: 1
  hist_len: 6
  dropout: 0.10

ppo:
  lr: 2.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.20
  value_coef: 0.5
  entropy_coef: 0.015
  rollout_steps: 4096
  update_epochs: 4
  minibatch_size: 128
  max_grad_norm: 1.0
  normalize_advantage: true
  target_kl: 0.015

cct:
  alpha: 1.25
  beta: 1.00
  gamma: 0.60
  init_penalty: 0.40
  final_penalty: 0.10

dynamic_orders:
  process: poisson
  lambda_per_min_default: 0.4
```

---

## 10. 如果你马上就要开始实验，我建议你先这样跑

### 第一组默认实验
- 地图：4km × 4km
- 站点：10
- 动态订单强度：$\lambda=0.4$ 单/分钟
- 上层重规划周期：360 s
- 候选动作上限：160
- 支撑半径：1.2 km
- PPO 学习率：2e-4

### 若出现以下现象，对应这样调：

#### 现象 1：站点排队严重、无人机经常长等
改：
```yaml
station_wait_threshold_sec: 420
reward.lambda_queue: 0.25
```

#### 现象 2：空驶太多、总飞行里程偏大
改：
```yaml
reward.lambda_empty: 0.20
max_candidate_recovery_per_order: 3
```

#### 现象 3：训练不稳、回报振荡大
改：
```yaml
ppo.lr: 1.0e-4
ppo.entropy_coef: 0.010
ppo.rollout_steps: 4096
```

#### 现象 4：高动态下经常来不及重规划
改：
```yaml
upper_replan_interval_sec: 240
upper_replan_new_order_trigger: 2
upper.alns_iters: 180
```

---

## 11. 最后一个关键建议

你现在最缺的其实不是“再复杂一点的网络”，而是**先确定泊松强度 $\lambda$**。  
因为下面这些参数几乎都和 $\lambda$ 强相关：

- `upper_replan_interval_sec`
- `upper_replan_new_order_trigger`
- `max_candidate_orders`
- `reward.lambda_late`
- `ppo.rollout_steps`

所以如果你下一步能给我一个更明确的动态订单强度，比如：

- $\lambda = 0.2$ 单/分钟
- $\lambda = 0.5$ 单/分钟
- 或者“每小时平均多少个动态订单”

我可以直接继续给你出一版**固定数值版**参数，而不是现在这种“推荐值 + 自适应公式版”。
