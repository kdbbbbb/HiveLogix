# 一辆无人车 + 多辆无人机动态协同配送问题建模与 GA 调度框架

## 0. 总体说明

整体思路是：**GA 只负责宏观“订单顺序 + 载具/模式分配”**，真正的起飞点、回收点、充换电站排队、等待、能量校验，由 **Decoder + Repair Simulator** 在解码阶段自动生成。

这适合“一辆 UGV + 多 UAV”的动态协同配送场景，尤其适合存在动态订单接入、无人机能量约束、充换电站约束、卡车-无人机同步约束的系统。

---

# 1. 问题定义

## 1.1 禱号描述与代码对应

为了与实体设计文档保持一致，此处简述数学禱号与代码属性的对应关系：

| 数学禱号 | 代码属性 | 说明 |
| :--- | :--- | :--- |
| $i$ / $u$ / $s$ | `order_id` / `drone_id` / `station_id` | 订单、无人机、站点齐次旨序 |
| $delivery\_loc$ / $create\_time$ / $deadline$ | `delivery_loc` / `create_time` / `deadline` | 订单属性 |
| $payload\_weight$ | `payload_weight` | 订单重量 |
| $speed$ | `speed` | 卡车行驶速度 |
| $cruise\_speed$ | `cruise_speed` | 无人机巡航速度 |
| $max\_inventory$ | `max_inventory` | 卡车最大载量 |
| $payload\_capacity$ | `payload_capacity` | 无人机最大载重 |
| $battery\_max$ | `battery_max` | 无人机满电电量 |
| $battery\_current$ | `battery_current` | 无人机当前电量 |
| $swap\_time$ | `swap_time` | 换电耗时 |
| $parking\_slots$ | `parking_slots` | 站点、卡车、仓库的值班数 |
| $T_{wait}^{max}$ | 10s | 卡车最大等待时间 |
| $q_i$ (下标) | `order[i].payload_weight` | 订单i的重量（下标形式） |
| $C_u$ (下标) | `drone[u].payload_capacity` | 无人机u的最大载重（下标形式） |
| $d_i$ (下标) | `order[i].deadline` | 订单i的截止时间（下标形式） |
| $\delta_i$ (下标) | `truck_service_time_order_s` / `drone_service_time_order_s` | 全局服务时间参数 |

**说明：** 约束条件中使用的带下标的数学符号（如 $q_i$、$C_u$、$d_i$ 等）遵循标准数学记号，通过对应关系明确其含义。

---

## 1.2 集合

设：

$$
O = O^{static} \cup O^{dynamic}
$$

为所有已释放、待调度订单集合。

$$
S = \{s_1,s_2,\dots,s_{|S|}\}
$$

为静态充换电站集合。

$$
D = \{1,2,\dots,M\}
$$

为无人机集合。

$$
0
$$

表示仓库 Depot。

$$
H = \{0\} \cup S
$$

表示合法的无人机起降、换电、回收节点集合。注意：**客户点不属于合法回收节点**。

卡车可访问节点集合：

$$
V_T = \{0\} \cup O \cup S
$$

无人机可访问节点集合：

$$
V_D = \{0\} \cup O \cup S
$$

---

## 1.2 参数

订单 $$i \in O$$：

$$
Order_i = \langle delivery\_loc, payload\_weight, create\_time, deadline \rangle
$$

其中：

- $$payload\_weight$$：订单重量或需求量；
- $$create\_time$$：订单释放时间；
- $$deadline$$：软截止时间。

卡车参数：

$$
speed
$$

为卡车行驶速度；

$$
max\_inventory
$$

为卡车库存容量或当前可用库存；

$$
dist^T_{ij}
$$

为卡车从节点 $$i$$ 到节点 $$j$$ 的道路距离；

$$
t^T_{ij} = \frac{dist^T_{ij}}{speed}
$$

为卡车行驶时间。

无人机参数：

$$
cruise\_speed
$$

为无人机 $$u$$ 的巡航速度；

$$
payload\_capacity
$$

为无人机 $$u$$ 最大载重；

$$
battery\_max
$$

为无人机 $$u$$ 满电电量；

$$
battery\_current
$$

为无人机当前剩余电量；

$$
dist^D_{ij}
$$

为无人机从节点 $$i$$ 到节点 $$j$$ 的飞行距离；

$$
t^D_{uij} = \frac{dist^D_{ij}}{cruise\_speed}
$$

为无人机 $$u$$ 的飞行时间。

无人机能耗函数设为：

$$
e_u(i,j,w) = \alpha_u dist^D_{ij} + \beta_u w dist^D_{ij}
$$

其中 $$w$$ 为载重。若暂时不想引入复杂气动模型，可先用线性能耗模型：

$$
e_u(i,j,w) = \eta_u(w) dist^D_{ij}
$$

服务时间参数：

$$
service\_time\_s
$$

为客户点卸货服务时间；

$$
swap\_time
$$

为充换电站或卡车的换电耗时（秒）；

$$
T_{wait}^{max}
$$

为卡车等待无人机的最大允许时间。

根据问题设定，统一取值：

$$
T_{wait}^{max}=10s
$$

充换电站参数：

$$
parking\_slots
$$

为站点 $$s$$ 同时可服务的无人机最大数量（工位数）。

大惩罚常数：

$$
M_{big}
$$

用于软约束惩罚。

---

# 2. 决策变量

## 2.1 订单服务模式变量

对每个订单 $$i$$，定义：

$$
x_i^A =
\begin{cases}
1, & \text{订单 } i \text{ 由卡车直递，模式 A}\\
0, & \text{否则}
\end{cases}
$$

$$
x_{u h i l}^{B} =
\begin{cases}
1, & \text{无人机 } u \text{ 从卡车所在合法节点 } h \text{ 起飞，服务 } i \text{ 后在 } l \text{ 回收}\\
0, & \text{否则}
\end{cases}
$$

其中：

$$
u \in D,\quad h,l \in H,\quad i \in O
$$

$$
x_{u 0 i l}^{C} =
\begin{cases}
1, & \text{无人机 } u \text{ 从仓库出发，服务 } i \text{ 后降落于 } l\\
0, & \text{否则}
\end{cases}
$$

这里 $$l$$ 可以是仓库，也可以是充换电站。

为了简化，也可以统一记为：

$$
z_{u h i l} =
\begin{cases}
1, & \text{无人机 } u \text{ 执行 sortie } h \to i \to l\\
0, & \text{否则}
\end{cases}
$$

其中：

- 若 $$h,l \in H$$，且 $$h,l$$ 在卡车路径上，则为模式 B；
- 若 $$h=0$$，且无人机从仓库直接出发，则为模式 C；
- 若服务客户后需要经多个充换电站空载中继返回，则由解码器扩展为模式 E 的衍生航段。

---

## 2.2 卡车路径变量

$$
y_{ij} =
\begin{cases}
1, & \text{卡车从节点 } i \text{ 行驶到节点 } j\\
0, & \text{否则}
\end{cases}
$$

其中：

$$
i,j \in V_T,\quad i \neq j
$$

卡车是否访问节点 $$i$$：

$$
v_i =
\begin{cases}
1, & \text{卡车访问节点 } i\\
0, & \text{否则}
\end{cases}
$$

---

## 2.3 时间变量

卡车到达和离开节点 $$i$$ 的时间：

$$
A_i^T,\quad D_i^T
$$

无人机 $$u$$ 执行 sortie $$h \to i \to l$$ 的起飞时间：

$$
T_{u h i l}^{takeoff}
$$

无人机到达客户 $$i$$ 的时间：

$$
A_{u h i}^{D}
$$

无人机完成订单 $$i$$ 的时间：

$$
C_i
$$

无人机到达回收点 $$l$$ 的时间：

$$
A_{u i l}^{D}
$$

订单迟到时间：

$$
L_i = \max(0, C_i - d_i)
$$

---

## 2.4 能量变量

无人机起飞时电量：

$$
E_{u h i l}^{start}
$$

无人机到达客户后剩余电量：

$$
E_{u h i}^{after}
$$

无人机到达落脚点后剩余电量：

$$
E_{u h i l}^{land}
$$

---

# 3. 目标函数

建议目标函数采用多目标加权形式：

$$
\min F =
\omega_1 \sum_{i \in O} C_i
+
\omega_2 \sum_{i \in O} L_i
+
\omega_3 E^{total}
+
\omega_4 W^{total}
+
\omega_5 P^{infeasible}
$$

其中：

$$
E^{total}
=
E_T^{total}
+
\sum_{u \in D} E_u^{total}
$$

卡车能耗可简化为：

$$
E_T^{total}
=
\sum_{i \in V_T}
\sum_{j \in V_T}
c_{truck\_energy} \cdot dist^T_{ij} \cdot y_{ij}
$$

其中 $$c_{truck\_energy}$$ 为卡车单位距离能耗（如 kWh/km）。

无人机能耗为：

$$
E_u^{total}
=
\sum_{h \in H}
\sum_{i \in O}
\sum_{l \in H}
z_{u h i l}
\left[
e_u(h,i,payload\_weight) + e_u(i,l,0)
\right]
$$

其中 $$e_u(\cdot,\cdot,\cdot)$$ 为无人机的多旋翼功率模型或线性能耗模型。

迟到惩罚：

$$
L_i \ge C_i - d_i
$$

$$
L_i \ge 0
$$

等待惩罚可定义为：

$$
W^{total}
=
\sum_{u \in D}
\sum_{h \in H}
\sum_{i \in O}
\sum_{l \in H}
W_{u h i l}
$$

其中 $$W_{u h i l}$$ 表示无人机和卡车在回收点的等待或错过惩罚。

---

# 4. 约束条件

## 4.1 每个订单必须且只能服务一次

$$
x_i^A
+
\sum_{u \in D}
\sum_{h \in H}
\sum_{l \in H}
z_{u h i l}
=
1,
\quad \forall i \in O
$$

这条约束体现：

- 静态订单和动态订单都必须送达；
- 不能漏送；
- 不能重复配送。

---

## 4.2 卡车闭环路径约束

卡车从仓库出发：

$$
\sum_{j \in V_T, j \neq 0} y_{0j} = 1
$$

卡车最终回到仓库：

$$
\sum_{i \in V_T, i \neq 0} y_{i0} = 1
$$

对中间访问节点流量守恒：

$$
\sum_{j \in V_T, j \neq i} y_{ij}
=
v_i,
\quad \forall i \in V_T \setminus \{0\}
$$

$$
\sum_{j \in V_T, j \neq i} y_{ji}
=
v_i,
\quad \forall i \in V_T \setminus \{0\}
$$

若订单 $$i$$ 由卡车直递，则卡车必须访问客户点 $$i$$：

$$
v_i \ge x_i^A,
\quad \forall i \in O
$$

若某充换电站 $$s$$ 被用作无人机起飞或回收点，则卡车必须访问该站：

$$
v_s
\ge
z_{u s i l},
\quad \forall u,i,l,s
$$

$$
v_s
\ge
z_{u h i s},
\quad \forall u,i,h,s
$$

---

## 4.3 消除卡车子回路约束

可使用 MTZ 约束。令 $$p_i$$ 为卡车访问顺序变量：

$$
p_i - p_j + |V_T| y_{ij}
\le |V_T| - 1,
\quad \forall i,j \in V_T \setminus \{0\}, i \neq j
$$

---

## 4.4 卡车时间递推约束

若卡车从 $$i$$ 到 $$j$$，则：

$$
A_j^T
\ge
D_i^T + t_{ij}^T
-
M_{big}(1-y_{ij})
$$

卡车在客户点直递时需要服务时间：

$$
D_i^T
\ge
A_i^T + \delta_i x_i^A,
\quad \forall i \in O
$$

卡车在充换电站停留时间可写为：

$$
D_s^T
\ge
A_s^T + \tau_s^{truck},
\quad \forall s \in S
$$

其中 $$\tau_s^{truck}$$ 可包括停车、无人机回收、换电、装卸等时间。

---

## 4.5 无人机载重约束

若无人机 $$u$$ 服务订单 $$i$$，必须满足：

$$
q_i z_{u h i l}
\le
C_u,
\quad \forall u,h,i,l
$$

若超载，则该 sortie 不可行，解码器应将其切换为模式 A，或给予大惩罚。

---

## 4.6 无人机一次 sortie 只能服务一个订单

每个 sortie 的形式固定为：

$$
h \to i \to l
$$

而不是：

$$
h \to i \to j \to l
$$

所以无人机 sortie 变量已经天然满足单订单配送。若用路径变量表达，则需增加：

$$
\sum_{i \in O} z_{u h i l} \le 1,
\quad \forall u,h,l,\text{ 每次起飞任务}
$$

在 GA 解码中，推荐直接把无人机任务列表解码为多个单订单 sortie。

---

## 4.7 模式 B 起飞同步约束

若无人机 $$u$$ 从卡车所在节点 $$h$$ 起飞服务订单 $$i$$，则卡车必须已经到达 $$h$$：

$$
T_{u h i l}^{takeoff}
\ge
A_h^T
-
M_{big}(1-z_{u h i l})
$$

卡车必须等无人机完成起飞准备后才能离开：

$$
D_h^T
\ge
T_{u h i l}^{takeoff} + \tau_{launch}
-
M_{big}(1-z_{u h i l})
$$

其中 $$h \in H$$。

---

## 4.8 模式 B 回收同步约束

无人机到达回收节点 $$l$$ 的时间：

$$
A_{u i l}^{D}
=
T_{u h i l}^{takeoff}
+
\tau_{launch}
+
t_{u h i}^{D}
+
\delta_i
+
t_{u i l}^{D}
$$

无人机必须在卡车离开回收点 $$l$$ 之前到达：

$$
A_{u i l}^{D}
\le
D_l^T
+
M_{big}(1-z_{u h i l})
$$

如果采用硬约束，则必须满足上式。

如果采用软约束，可引入错过变量：

$$
Miss_{u h i l}
\ge
A_{u i l}^{D} - D_l^T
$$

$$
Miss_{u h i l} \ge 0
$$

并将其加入惩罚项：

$$
P^{miss}
=
\lambda_{miss}
\sum_{u,h,i,l} Miss_{u h i l}
$$

---

## 4.9 卡车等待无人机上限约束

若卡车先到回收点 $$l$$，无人机后到，则卡车等待时间为：

$$
W_{u h i l}^{truck}
=
\max(0, A_{u i l}^{D} - A_l^T)
$$

硬约束形式：

$$
W_{u h i l}^{truck}
\le
T_{wait}^{max}
+
M_{big}(1-z_{u h i l})
$$

软约束形式：

$$
P_{wait}
=
\lambda_{wait}
\sum_{u,h,i,l}
\max(0, W_{u h i l}^{truck} - T_{wait}^{max})
$$

按照设定，卡车最多等待 10 秒；超过后卡车继续走，无人机自动转入最近可达站点或仓库等待，并给予大惩罚。

---

## 4.10 无人机能量守恒约束

无人机从 $$h$$ 起飞，携货飞到客户 $$i$$：

$$
E_{u h i}^{after}
=
E_{u h i l}^{start}
-
e_u(h,i,q_i)
$$

服务完客户后空载飞到回收点 $$l$$：

$$
E_{u h i l}^{land}
=
E_{u h i}^{after}
-
e_u(i,l,0)
$$

飞行全过程必须高于安全电量：

$$
E_{u h i}^{after}
\ge
E_u^{safe}
-
M_{big}(1-z_{u h i l})
$$

$$
E_{u h i l}^{land}
\ge
E_u^{safe}
-
M_{big}(1-z_{u h i l})
$$

若无人机从仓库、卡车或换电站起飞且已经完成换电，则：

$$
E_{u h i l}^{start}
=
E_u^{max}
$$

若动态重调度时无人机已经在空中，则其起始电量应取当前剩余电量：

$$
E_{u h i l}^{start}
=
E_u^{current}(t)
$$

---

## 4.11 前瞻能量校验约束

无人机在接单前必须保证：

$$
e_u(h,i,q_i) + e_u(i,l,0)
\le
E_{u}^{start} - E_u^{safe}
$$

即：

$$
E_u^{start}
-
e_u(h,i,q_i)
-
e_u(i,l,0)
\ge
E_u^{safe}
$$

这条约束是无人机“不坠机”的核心红线。

对于模式 C，即仓库无人机直递：

$$
h = 0
$$

则必须一次性验证：

$$
E_u^{max}
-
e_u(0,i,q_i)
-
e_u(i,l,0)
\ge
E_u^{safe}
$$

注意：根据规则，无人机携货飞行期间不能中途降落于换电站补能，因此模式 C 不能写成：

$$
0 \to s \to i \to l
$$

只有完成卸货之后，才允许：

$$
0 \to i \to s_1 \to s_2 \to \cdots \to l
$$

也就是说，换电站中继只允许发生在**卸货之后的空载阶段**，不能发生在携货阶段。

---
## 4.12 充换电站 / 卡车换电假设与电量重置约束

本算法中对无人机换电过程作如下简化假设：

1. 无人机到达充换电站后，若需要换电，则立即换上满电电池；
2. 无人机到达卡车并需要继续执行任务时，也假设可立即换上满电电池；
3. 换电过程不产生额外时间消耗；
4. 不考虑充换电站排队等待时间；
5. 不考虑充换电站容量限制对调度结果的影响。

因此，无人机到达充换电站或卡车后，若发生换电，则电量直接恢复为最大电量：

$$
E_u^{after\_swap} = E_u^{max}
$$

换电完成时间等于到达时间：

$$
T_{u,s}^{ready}
=
T_{u,s}^{arrive}
$$

其中，$$s$$ 可以表示充换电站，也可以表示卡车所在的协同节点。

因此，原先的排队等待时间：

$$
Q_{u,s}
$$

以及换电服务时间：

$$
\tau_{swap}
$$

在本文模型中均取为 0：

$$
Q_{u,s} = 0
$$

$$
\tau_{swap} = 0
$$

即：

$$
T_{u,s}^{ready}
=
T_{u,s}^{arrive}
+
Q_{u,s}
+
\tau_{swap}
=
T_{u,s}^{arrive}
$$

由于不考虑充换电站容量限制，因此不再设置如下累计资源约束：

$$
\sum_{u \in D}
\sum_{k}
\mathbf{1}
\left(
T_{u,s,k}^{swap\_start}
\le t <
T_{u,s,k}^{swap\_end}
\right)
\le
parking\_slots,
\quad \forall s \in S,\forall t
$$

工程实现时，Decoder 中也不需要维护充换电站服务队列，例如不再需要：

```python
station.next_available_slots = [0.0] * station.parking_slots

Decoder 只需要在无人机到达充换电站或卡车后，根据后续任务需求判断是否换电。若换电，则直接将无人机电量重置为最大电量：



该假设使得算法重点集中在任务序列、配送模式、载具分配以及卡车-无人机协同路径的解码与优化上，而不额外引入充换电站排队和容量调度问题。

4.13 货物库存充足假设

本算法中对货物供给过程作如下简化假设：

仓库货物始终充足；
卡车上货物始终充足；
无人机从仓库起飞时，始终能够取得其配送任务所需货物；
无人机从卡车起飞时，始终能够从卡车上取得其配送任务所需货物；
卡车执行直递任务时，始终能够从自身库存中取得所需货物；
因此，不考虑卡车动态库存约束、补货任务和库存不足导致的任务修复。

因此，原先卡车库存状态：

I
T
	​

(t)

不再作为限制调度可行性的约束变量。

对于任意由卡车或无人机服务的订单 i，均假设其货物需求量 q
i
	​

 可以被满足：

q
i
	​

≤I
T
	​

(t)

恒成立。

等价地，可以认为：

I
T
	​

(t)=+∞

因此不再需要显式计算：

I
T
	​

(t)=I
T
	​

(0)−
i∈O
A
	​

(t)
∑
	​

q
i
	​

−
i∈O
B
	​

(t)
∑
	​

q
i
	​

+R
T
	​

(t)

也不再需要检查库存非负约束：

I
T
	​

(t)≥0

在 Decoder 解码过程中，不需要因为卡车库存不足而触发如下修复操作：

将订单切换为模式 C，由仓库无人机直递；
插入仓库补货任务或等待补货任务，形成模式 D；
若均不可行，则给个体大惩罚。

工程实现时，Decoder 只需要根据任务分配结果判断订单由卡车配送、仓库无人机配送，还是卡车-无人机协同配送，而不需要维护卡车库存变量。例如：

# 不再检查 truck.inventory 是否足够
# 默认卡车和仓库均可提供订单所需货物

if task.mode == "truck_direct":
    decode_truck_delivery(task)

elif task.mode == "drone_from_warehouse":
    decode_warehouse_drone_delivery(task)

elif task.mode == "truck_drone_collaboration":
    decode_truck_drone_delivery(task)

该假设可以避免库存补货过程对调度模型的干扰，使算法重点关注卡车路径、无人机路径、协同起降节点选择以及任务序列优化。

---

## 4.14 动态订单释放时间约束

动态订单 $$i$$ 在释放时间 $$create\_time_i$$ 之前不能被服务：

$$
C_i \ge create\_time_i + \delta_i
$$

更严格地说，起飞或卡车出发服务该订单的时间也不能早于释放时间：

$$
T_{u h i l}^{takeoff}
\ge
r_i
-
M_{big}(1-z_{u h i l})
$$

$$
A_i^T
\ge
r_i
-
M_{big}(1-x_i^A)
$$

---

# 5. 推荐的双层染色体编码

原来的“任务序列 + 切分点”可以工作，但对动态订单和多模式修复不够灵活。建议改成：

## 染色体 1：任务序列

$$
\pi = [o_3,o_1,o_5,o_2,o_4,o_6]
$$

表示订单的全局调度优先级。

## 染色体 2：载具/模式分配基因

$$
\gamma = [g_3,g_1,g_5,g_2,g_4,g_6]
$$

其中：

$$
g_i \in \{A, B_1, B_2,\dots,B_M,C_1,C_2,\dots,C_M\}
$$

含义为：

- $$A$$：卡车直递；
- $$B_u$$：由卡车搭载无人机 $$u$$ 执行模式 B；
- $$C_u$$：由仓库无人机 $$u$$ 执行模式 C。

例如：

$$
\pi = [o_3,o_1,o_5,o_2,o_4,o_6]
$$

$$
\gamma = [A,B_1,B_2,A,C_1,B_1]
$$

解码结果为：

- $$o_3$$：卡车直递；
- $$o_1$$：无人机 1 从卡车起飞服务；
- $$o_5$$：无人机 2 从卡车起飞服务；
- $$o_2$$：卡车直递；
- $$o_4$$：无人机 1 从仓库直递；
- $$o_6$$：无人机 1 从卡车起飞服务。

如果必须保留“切分点 $$\rho$$”形式，也可以：

$$
\rho = [2,5]
$$

但建议切分点只用于初始化或启发式种子；正式 GA 迭代最好用 $$\gamma$$ 这种逐订单载具分配方式，否则某个无人机段内连续多个订单的时空同步会非常难修复。

---

# 6. GA 解码逻辑

GA 个体本身不直接保存充换电站、起飞点、回收点。它只保存：

```python
Individual:
    sequence: List[order_id]      # 染色体1：任务序列 π
    assignment: List[gene]        # 染色体2：模式/载具分配 γ
```

真正的物理可行性由 Decoder 生成：

```python
DecodedPlan:
    truck_route
    drone_sorties
    station_queues
    completion_times
    energy_records
    penalties
    objective
```

---

# 7. 核心 Python 代码框架

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import math
import copy


BIG_M = 10**9


@dataclass
class Order:
    oid: int
    loc: Tuple[float, float]
    demand: float
    release_time: float
    deadline: float
    status: str = "unserved"  # unserved, assigned, in_service, completed


@dataclass
class DroneSpec:
    uid: int
    speed: float
    payload_cap: float
    e_max: float
    e_safe: float
    alpha: float
    beta: float


@dataclass
class TruckState:
    loc: Tuple[float, float]
    time: float
    speed: float
    inventory: float
    route: List = field(default_factory=list)


@dataclass
class DroneState:
    uid: int
    loc: Tuple[float, float]
    time: float
    energy: float
    status: str  # on_truck, at_depot, at_station, flying, unavailable
    host: Optional[int] = None  # depot/station/truck id


@dataclass
class StationState:
    sid: int
    loc: Tuple[float, float]
    cap: int
    swap_time: float
    # 每个工位的最早可用时间
    slot_available_times: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.slot_available_times:
            self.slot_available_times = [0.0 for _ in range(self.cap)]


@dataclass
class SystemState:
    current_time: float
    depot_loc: Tuple[float, float]
    truck: TruckState
    drones: Dict[int, DroneState]
    drone_specs: Dict[int, DroneSpec]
    stations: Dict[int, StationState]
    orders: Dict[int, Order]


@dataclass
class Individual:
    sequence: List[int]
    assignment: List[str]   # e.g. ["A", "B_1", "C_2"]
    fitness: float = BIG_M
    decoded_plan: Optional[dict] = None
```

---

## 7.1 距离与能耗函数

```python
def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def truck_travel_time(a, b, truck_speed: float) -> float:
    # 实际项目中这里应替换成 road-network shortest path
    return euclidean(a, b) / truck_speed


def drone_travel_time(a, b, drone_speed: float) -> float:
    return euclidean(a, b) / drone_speed


def drone_energy(spec: DroneSpec, a, b, payload: float) -> float:
    dist = euclidean(a, b)
    return spec.alpha * dist + spec.beta * payload * dist
```

---

# 8. 无人机 sortie 前瞻能量校验

```python
def check_drone_sortie_energy(
    spec: DroneSpec,
    start_energy: float,
    start_loc: Tuple[float, float],
    customer_loc: Tuple[float, float],
    landing_loc: Tuple[float, float],
    payload: float,
) -> bool:
    """
    检查无人机是否能完成：
        起点 -> 客户，携货
        客户 -> 落脚点，空载

    注意：携货阶段不允许中途去充换电站。
    """
    e1 = drone_energy(spec, start_loc, customer_loc, payload)
    e2 = drone_energy(spec, customer_loc, landing_loc, 0.0)

    remaining_after_customer = start_energy - e1
    remaining_after_landing = remaining_after_customer - e2

    return (
        remaining_after_customer >= spec.e_safe and
        remaining_after_landing >= spec.e_safe
    )
```

---

# 9. 选择回收点 / 落脚点

模式 B 中，回收点必须是仓库或充换电站，不能是客户点。

```python
def candidate_landing_nodes(state: SystemState):
    """
    合法落脚点：仓库 + 所有充换电站。
    """
    nodes = [("depot", 0, state.depot_loc)]
    for sid, s in state.stations.items():
        nodes.append(("station", sid, s.loc))
    return nodes


def select_best_landing_node(
    state: SystemState,
    spec: DroneSpec,
    drone_state: DroneState,
    order: Order,
    mode: str,
) -> Optional[Tuple[str, int, Tuple[float, float]]]:
    """
    在所有合法落脚点中选择一个可行且代价最小的点。
    代价可以综合：
        - 飞行距离
        - 与卡车预计到达时间的匹配程度
        - 换电站排队时间
    """
    best = None
    best_cost = BIG_M

    for node_type, node_id, loc in candidate_landing_nodes(state):
        feasible = check_drone_sortie_energy(
            spec=spec,
            start_energy=drone_state.energy,
            start_loc=drone_state.loc,
            customer_loc=order.loc,
            landing_loc=loc,
            payload=order.demand,
        )

        if not feasible:
            continue

        flight_cost = (
            euclidean(drone_state.loc, order.loc) +
            euclidean(order.loc, loc)
        )

        # 简化：先只按飞行距离选。实际可加入卡车到达该站的时间差。
        cost = flight_cost

        if cost < best_cost:
            best_cost = cost
            best = (node_type, node_id, loc)

    return best
```

---

# 10. 充换电站排队模拟

```python
def station_swap(
    station: StationState,
    arrive_time: float
) -> Tuple[float, float, float]:
    """
    返回：
        swap_start_time
        swap_finish_time
        queue_wait_time
    """
    # 找最早可用工位
    idx = min(range(station.cap), key=lambda k: station.slot_available_times[k])
    slot_ready = station.slot_available_times[idx]

    swap_start = max(arrive_time, slot_ready)
    queue_wait = swap_start - arrive_time
    swap_finish = swap_start + station.swap_time

    station.slot_available_times[idx] = swap_finish

    return swap_start, swap_finish, queue_wait
```

---

# 11. 解码器：从染色体到物理计划

这是 GA 的核心。一个个体是否好，不是看染色体本身，而是看解码后是否满足物理规则。

```python
def decode_individual(
    individual: Individual,
    init_state: SystemState,
    weights: Dict[str, float],
    service_time: float,
    truck_wait_max: float = 10.0,
) -> Tuple[float, dict]:
    """
    输入：
        individual.sequence: 订单顺序
        individual.assignment: 每个订单对应 A / B_u / C_u

    输出：
        fitness, decoded_plan
    """

    state = copy.deepcopy(init_state)

    truck = state.truck
    drones = state.drones
    specs = state.drone_specs

    completion_times = {}
    drone_sorties = []
    truck_route = []
    penalties = {
        "delay": 0.0,
        "energy": 0.0,
        "infeasible": 0.0,
        "waiting": 0.0,
        "inventory": 0.0,
    }

    total_drone_energy = 0.0
    total_truck_energy = 0.0

    for oid, gene in zip(individual.sequence, individual.assignment):
        order = state.orders[oid]

        if order.status == "completed":
            continue

        # 订单释放时间约束
        earliest_service_time = max(state.current_time, order.release_time)

        if gene == "A":
            # 模式 A：卡车直递
            if truck.inventory < order.demand:
                # 库存不足：触发模式 D 或给惩罚
                penalties["inventory"] += BIG_M * 0.01

                # 简化修复：切换为仓库无人机 C
                repaired = try_repair_by_depot_drone(
                    state=state,
                    order=order,
                    service_time=service_time,
                    penalties=penalties,
                    drone_sorties=drone_sorties,
                    completion_times=completion_times,
                )
                if not repaired:
                    penalties["infeasible"] += BIG_M
                continue

            travel = truck_travel_time(truck.loc, order.loc, truck.speed)
            arrive = max(truck.time + travel, earliest_service_time)
            finish = arrive + service_time

            total_truck_energy += euclidean(truck.loc, order.loc)
            truck.loc = order.loc
            truck.time = finish
            truck.inventory -= order.demand

            truck_route.append({
                "type": "truck_direct",
                "order": oid,
                "arrive": arrive,
                "finish": finish,
                "loc": order.loc,
            })

            completion_times[oid] = finish
            order.status = "completed"

        elif gene.startswith("B_"):
            # 模式 B：卡车搭载无人机执行
            uid = int(gene.split("_")[1])
            drone = drones[uid]
            spec = specs[uid]

            if order.demand > spec.payload_cap:
                # 无人机超载，回退模式 A
                penalties["infeasible"] += 10000.0
                individual.assignment[individual.sequence.index(oid)] = "A"
                continue

            # 无人机必须在卡车上，或与卡车位于同一合法节点
            if drone.status not in ["on_truck", "at_station", "at_depot"]:
                penalties["infeasible"] += BIG_M
                continue

            # 起飞点：当前卡车位置
            launch_loc = truck.loc
            launch_time = max(truck.time, drone.time, earliest_service_time)

            # 为该订单选择合法回收点
            drone.loc = launch_loc
            drone.energy = spec.e_max  # 默认在卡车上已换满电

            landing = select_best_landing_node(
                state=state,
                spec=spec,
                drone_state=drone,
                order=order,
                mode="B",
            )

            if landing is None:
                # 修复失败，回退为卡车直递或给大惩罚
                penalties["infeasible"] += BIG_M
                continue

            landing_type, landing_id, landing_loc = landing

            # 飞行时间
            t1 = drone_travel_time(launch_loc, order.loc, spec.speed)
            t2 = drone_travel_time(order.loc, landing_loc, spec.speed)

            e1 = drone_energy(spec, launch_loc, order.loc, order.demand)
            e2 = drone_energy(spec, order.loc, landing_loc, 0.0)

            arrive_customer = launch_time + t1
            finish_customer = arrive_customer + service_time
            arrive_landing = finish_customer + t2

            # 能耗更新
            drone_energy_left = spec.e_max - e1 - e2

            if drone_energy_left < spec.e_safe:
                penalties["energy"] += BIG_M
                continue

            # 如果降落在换电站，模拟换电排队
            queue_wait = 0.0
            ready_time = arrive_landing

            if landing_type == "station":
                station = state.stations[landing_id]
                swap_start, swap_finish, queue_wait = station_swap(
                    station, arrive_landing
                )
                ready_time = swap_finish
                drone_energy_left = spec.e_max

            elif landing_type == "depot":
                ready_time = arrive_landing
                drone_energy_left = spec.e_max

            # 卡车需要去回收点
            truck_to_landing = truck_travel_time(truck.loc, landing_loc, truck.speed)
            truck_arrive_landing = truck.time + truck_to_landing

            # 如果卡车先到，可等无人机，但最多 truck_wait_max
            truck_wait = max(0.0, ready_time - truck_arrive_landing)

            if truck_wait > truck_wait_max:
                penalties["waiting"] += 10000.0 * (truck_wait - truck_wait_max)
                # 软修复：无人机留在站点/仓库等待，卡车不强制等
                truck_depart_landing = truck_arrive_landing + truck_wait_max
            else:
                truck_depart_landing = max(truck_arrive_landing, ready_time)

            # 更新状态
            truck.loc = landing_loc
            truck.time = truck_depart_landing
            truck.inventory -= order.demand

            drone.loc = landing_loc
            drone.time = ready_time
            drone.energy = drone_energy_left
            drone.status = "on_truck" if truck_wait <= truck_wait_max else "at_station"

            total_drone_energy += e1 + e2
            total_truck_energy += euclidean(launch_loc, landing_loc)

            completion_times[oid] = finish_customer
            order.status = "completed"

            drone_sorties.append({
                "mode": "B",
                "drone": uid,
                "order": oid,
                "launch_loc": launch_loc,
                "landing": landing,
                "takeoff": launch_time,
                "arrive_customer": arrive_customer,
                "finish_customer": finish_customer,
                "arrive_landing": arrive_landing,
                "ready_time": ready_time,
                "queue_wait": queue_wait,
                "truck_wait": truck_wait,
                "energy_used": e1 + e2,
            })

        elif gene.startswith("C_"):
            # 模式 C：仓库无人机直递
            uid = int(gene.split("_")[1])
            drone = drones[uid]
            spec = specs[uid]

            if order.demand > spec.payload_cap:
                penalties["infeasible"] += BIG_M
                continue

            # 仓库无人机从 depot 出发
            start_loc = state.depot_loc
            start_time = max(state.current_time, order.release_time, drone.time)
            start_energy = spec.e_max

            # 选择合法落脚点
            temp_drone_state = copy.deepcopy(drone)
            temp_drone_state.loc = start_loc
            temp_drone_state.energy = start_energy

            landing = select_best_landing_node(
                state=state,
                spec=spec,
                drone_state=temp_drone_state,
                order=order,
                mode="C",
            )

            if landing is None:
                penalties["energy"] += BIG_M
                continue

            landing_type, landing_id, landing_loc = landing

            t1 = drone_travel_time(start_loc, order.loc, spec.speed)
            t2 = drone_travel_time(order.loc, landing_loc, spec.speed)

            e1 = drone_energy(spec, start_loc, order.loc, order.demand)
            e2 = drone_energy(spec, order.loc, landing_loc, 0.0)

            arrive_customer = start_time + t1
            finish_customer = arrive_customer + service_time
            arrive_landing = finish_customer + t2

            energy_left = start_energy - e1 - e2
            if energy_left < spec.e_safe:
                penalties["energy"] += BIG_M
                continue

            queue_wait = 0.0
            ready_time = arrive_landing

            if landing_type == "station":
                station = state.stations[landing_id]
                swap_start, swap_finish, queue_wait = station_swap(
                    station, arrive_landing
                )
                ready_time = swap_finish
                energy_left = spec.e_max

            elif landing_type == "depot":
                energy_left = spec.e_max

            drone.loc = landing_loc
            drone.time = ready_time
            drone.energy = energy_left
            drone.status = "at_station" if landing_type == "station" else "at_depot"

            total_drone_energy += e1 + e2
            completion_times[oid] = finish_customer
            order.status = "completed"

            drone_sorties.append({
                "mode": "C",
                "drone": uid,
                "order": oid,
                "launch_loc": start_loc,
                "landing": landing,
                "takeoff": start_time,
                "finish_customer": finish_customer,
                "arrive_landing": arrive_landing,
                "ready_time": ready_time,
                "queue_wait": queue_wait,
                "energy_used": e1 + e2,
            })

        else:
            penalties["infeasible"] += BIG_M

    # 所有无人机最终回仓校验
    final_return_penalty = enforce_final_return_to_depot(
        state=state,
        penalties=penalties
    )

    # 卡车最终回仓
    back_time = truck_travel_time(truck.loc, state.depot_loc, truck.speed)
    truck.time += back_time
    total_truck_energy += euclidean(truck.loc, state.depot_loc)
    truck.loc = state.depot_loc

    # 迟到惩罚
    delay_penalty = 0.0
    sum_completion = 0.0

    for oid, ctime in completion_times.items():
        order = state.orders[oid]
        lateness = max(0.0, ctime - order.deadline)
        delay_penalty += lateness
        sum_completion += ctime

    objective = (
        weights["completion"] * sum_completion +
        weights["delay"] * delay_penalty +
        weights["drone_energy"] * total_drone_energy +
        weights["truck_energy"] * total_truck_energy +
        weights["waiting"] * penalties["waiting"] +
        weights["infeasible"] * penalties["infeasible"] +
        weights["inventory"] * penalties["inventory"] +
        weights["energy_violation"] * penalties["energy"]
    )

    decoded_plan = {
        "truck_route": truck_route,
        "drone_sorties": drone_sorties,
        "completion_times": completion_times,
        "penalties": penalties,
        "total_drone_energy": total_drone_energy,
        "total_truck_energy": total_truck_energy,
        "final_truck_time": truck.time,
        "objective": objective,
    }

    return objective, decoded_plan
```

---

# 12. 库存不足时的修复逻辑

```python
def try_repair_by_depot_drone(
    state: SystemState,
    order: Order,
    service_time: float,
    penalties: dict,
    drone_sorties: list,
    completion_times: dict,
) -> bool:
    """
    当卡车库存不足时，优先尝试用仓库无人机模式 C 修复。
    若所有无人机都不可行，则返回 False。
    """
    best_candidate = None
    best_finish_time = BIG_M

    for uid, drone in state.drones.items():
        spec = state.drone_specs[uid]

        if order.demand > spec.payload_cap:
            continue

        start_loc = state.depot_loc
        start_energy = spec.e_max

        temp_drone = copy.deepcopy(drone)
        temp_drone.loc = start_loc
        temp_drone.energy = start_energy

        landing = select_best_landing_node(
            state=state,
            spec=spec,
            drone_state=temp_drone,
            order=order,
            mode="C",
        )

        if landing is None:
            continue

        _, _, landing_loc = landing

        if not check_drone_sortie_energy(
            spec=spec,
            start_energy=start_energy,
            start_loc=start_loc,
            customer_loc=order.loc,
            landing_loc=landing_loc,
            payload=order.demand,
        ):
            continue

        t1 = drone_travel_time(start_loc, order.loc, spec.speed)
        finish = max(state.current_time, order.release_time, drone.time) + t1 + service_time

        if finish < best_finish_time:
            best_finish_time = finish
            best_candidate = (uid, landing)

    if best_candidate is None:
        return False

    uid, landing = best_candidate
    drone = state.drones[uid]
    spec = state.drone_specs[uid]
    _, _, landing_loc = landing

    start_time = max(state.current_time, order.release_time, drone.time)
    t1 = drone_travel_time(state.depot_loc, order.loc, spec.speed)
    t2 = drone_travel_time(order.loc, landing_loc, spec.speed)

    e1 = drone_energy(spec, state.depot_loc, order.loc, order.demand)
    e2 = drone_energy(spec, order.loc, landing_loc, 0.0)

    arrive_customer = start_time + t1
    finish_customer = arrive_customer + service_time
    arrive_landing = finish_customer + t2

    drone.loc = landing_loc
    drone.time = arrive_landing
    drone.energy = spec.e_max
    drone.status = "at_station"

    completion_times[order.oid] = finish_customer
    order.status = "completed"

    drone_sorties.append({
        "mode": "C_REPAIR",
        "drone": uid,
        "order": order.oid,
        "takeoff": start_time,
        "finish_customer": finish_customer,
        "landing": landing,
        "energy_used": e1 + e2,
    })

    return True
```

---

# 13. 最终回仓约束修复

无人机最终必须回到仓库。若电量不足以直飞仓库，应通过充换电站中继回仓。

```python
def enforce_final_return_to_depot(
    state: SystemState,
    penalties: dict,
) -> float:
    """
    所有无人机最终回到 depot。
    若不能直飞，则尝试通过充换电站中继。
    """
    total_penalty = 0.0

    for uid, drone in state.drones.items():
        spec = state.drone_specs[uid]

        if euclidean(drone.loc, state.depot_loc) < 1e-6:
            continue

        # 先尝试直飞回仓
        e_need = drone_energy(spec, drone.loc, state.depot_loc, 0.0)

        if drone.energy - e_need >= spec.e_safe:
            t = drone_travel_time(drone.loc, state.depot_loc, spec.speed)
            drone.time += t
            drone.loc = state.depot_loc
            drone.energy -= e_need
            drone.status = "at_depot"
            continue

        # 尝试经由一个充换电站回仓
        feasible_station = None
        best_cost = BIG_M

        for sid, station in state.stations.items():
            e_to_s = drone_energy(spec, drone.loc, station.loc, 0.0)
            e_s_to_depot = drone_energy(spec, station.loc, state.depot_loc, 0.0)

            can_reach_station = drone.energy - e_to_s >= spec.e_safe
            can_station_to_depot = spec.e_max - e_s_to_depot >= spec.e_safe

            if can_reach_station and can_station_to_depot:
                cost = euclidean(drone.loc, station.loc) + euclidean(station.loc, state.depot_loc)
                if cost < best_cost:
                    best_cost = cost
                    feasible_station = station

        if feasible_station is None:
            penalties["energy"] += BIG_M
            total_penalty += BIG_M
            continue

        # 执行中继回仓
        t_to_s = drone_travel_time(drone.loc, feasible_station.loc, spec.speed)
        arrive_s = drone.time + t_to_s

        _, swap_finish, _ = station_swap(feasible_station, arrive_s)

        t_to_depot = drone_travel_time(feasible_station.loc, state.depot_loc, spec.speed)

        drone.time = swap_finish + t_to_depot
        drone.loc = state.depot_loc
        drone.energy = spec.e_max - drone_energy(spec, feasible_station.loc, state.depot_loc, 0.0)
        drone.status = "at_depot"

    return total_penalty
```

---

# 14. GA 主流程

## 14.1 初始化种群

```python
def initialize_population(
    orders: List[int],
    drone_ids: List[int],
    pop_size: int,
) -> List[Individual]:
    population = []

    possible_genes = ["A"]
    for uid in drone_ids:
        possible_genes.append(f"B_{uid}")
        possible_genes.append(f"C_{uid}")

    # 1. 随机个体
    for _ in range(int(pop_size * 0.7)):
        seq = orders[:]
        random.shuffle(seq)
        assignment = [random.choice(possible_genes) for _ in seq]
        population.append(Individual(seq, assignment))

    # 2. 纯卡车种子
    seq = orders[:]
    assignment = ["A" for _ in seq]
    population.append(Individual(seq, assignment))

    # 3. 距离优先种子应在外部根据坐标排序，这里略写
    seq = orders[:]
    assignment = []
    for k, oid in enumerate(seq):
        if k % 3 == 0:
            assignment.append("A")
        else:
            assignment.append(f"B_{random.choice(drone_ids)}")
    population.append(Individual(seq, assignment))

    # 4. 对立学习 OBL 种子：反转序列 + 反向资源分配
    base_seq = orders[:]
    random.shuffle(base_seq)
    opposite_seq = list(reversed(base_seq))
    opposite_assignment = [random.choice(possible_genes) for _ in opposite_seq]
    population.append(Individual(opposite_seq, opposite_assignment))

    # 补齐
    while len(population) < pop_size:
        seq = orders[:]
        random.shuffle(seq)
        assignment = [random.choice(possible_genes) for _ in seq]
        population.append(Individual(seq, assignment))

    return population
```

---

## 14.2 OX 顺序交叉

```python
def order_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    n = len(parent1.sequence)
    a, b = sorted(random.sample(range(n), 2))

    def make_child(p1, p2):
        child_seq = [None] * n
        child_seq[a:b+1] = p1.sequence[a:b+1]

        fill = [x for x in p2.sequence if x not in child_seq]
        idx = 0
        for k in range(n):
            if child_seq[k] is None:
                child_seq[k] = fill[idx]
                idx += 1

        # assignment 按订单 id 对齐，而不是按位置直接复制
        gene_map_1 = {oid: gene for oid, gene in zip(p1.sequence, p1.assignment)}
        gene_map_2 = {oid: gene for oid, gene in zip(p2.sequence, p2.assignment)}

        child_assignment = []
        for oid in child_seq:
            if random.random() < 0.5:
                child_assignment.append(gene_map_1[oid])
            else:
                child_assignment.append(gene_map_2[oid])

        return Individual(child_seq, child_assignment)

    return make_child(parent1, parent2), make_child(parent2, parent1)
```

---

## 14.3 变异

```python
def mutate(
    ind: Individual,
    drone_ids: List[int],
    p_swap: float = 0.2,
    p_assign: float = 0.2,
):
    n = len(ind.sequence)

    # 任务序列交换变异
    if random.random() < p_swap and n >= 2:
        i, j = random.sample(range(n), 2)
        ind.sequence[i], ind.sequence[j] = ind.sequence[j], ind.sequence[i]
        ind.assignment[i], ind.assignment[j] = ind.assignment[j], ind.assignment[i]

    # 载具/模式变异
    possible_genes = ["A"]
    for uid in drone_ids:
        possible_genes.append(f"B_{uid}")
        possible_genes.append(f"C_{uid}")

    for k in range(n):
        if random.random() < p_assign:
            ind.assignment[k] = random.choice(possible_genes)
```

---

## 14.4 选择与进化

```python
def tournament_select(population: List[Individual], k: int = 3) -> Individual:
    candidates = random.sample(population, k)
    candidates.sort(key=lambda x: x.fitness)
    return copy.deepcopy(candidates[0])


def run_ga(
    init_state: SystemState,
    active_order_ids: List[int],
    pop_size: int,
    generations: int,
    weights: Dict[str, float],
    service_time: float,
    warm_start: Optional[List[Individual]] = None,
) -> Individual:

    drone_ids = list(init_state.drones.keys())

    population = initialize_population(
        orders=active_order_ids,
        drone_ids=drone_ids,
        pop_size=pop_size,
    )

    if warm_start:
        population[:len(warm_start)] = warm_start[:pop_size]

    # 初始评估
    for ind in population:
        ind.fitness, ind.decoded_plan = decode_individual(
            individual=ind,
            init_state=init_state,
            weights=weights,
            service_time=service_time,
        )

    for gen in range(generations):
        population.sort(key=lambda x: x.fitness)
        new_population = []

        # 精英保留
        elite_num = max(1, int(0.05 * pop_size))
        new_population.extend(copy.deepcopy(population[:elite_num]))

        while len(new_population) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)

            c1, c2 = order_crossover(p1, p2)

            mutate(c1, drone_ids)
            mutate(c2, drone_ids)

            for child in [c1, c2]:
                child.fitness, child.decoded_plan = decode_individual(
                    individual=child,
                    init_state=init_state,
                    weights=weights,
                    service_time=service_time,
                )
                new_population.append(child)

                if len(new_population) >= pop_size:
                    break

        population = new_population

    population.sort(key=lambda x: x.fitness)
    return population[0]
```

---

# 15. 动态订单事件触发式重调度

核心思想：

当动态订单在 $$t$$ 时刻到达时，不是重新从仓库开始规划，而是：

1. 冻结已经完成的任务；
2. 锁定正在执行的物理动作；
3. 读取卡车和无人机当前状态；
4. 将未完成订单与新订单合并；
5. 以上一轮优秀计划的剩余部分作为 warm start；
6. 从当前状态重新运行 GA。

---

## 15.1 动态重调度数学描述

在时间 $$t$$，已完成订单集合：

$$
O^{done}(t)
$$

正在执行且不可中断订单集合：

$$
O^{lock}(t)
$$

未完成订单集合：

$$
O^{remain}(t)
=
O \setminus
\left(
O^{done}(t) \cup O^{lock}(t)
\right)
$$

新到达动态订单集合：

$$
O^{new}(t)
$$

新的待优化集合为：

$$
O^{replan}(t)
=
O^{remain}(t)
\cup
O^{new}(t)
$$

新的初始系统状态为：

$$
State(t)
=
\{
loc_T(t), I_T(t), time_T(t),
loc_u(t), E_u(t), status_u(t)
\}_{u \in D}
$$

新一轮 GA 求解：

$$
Plan^*(t)
=
GA\left(
O^{replan}(t),
State(t)
\right)
$$

最终执行计划为：

$$
Plan(t)
=
Plan^{locked}(t)
\oplus
Plan^*(t)
$$

其中 $$\oplus$$ 表示将已锁定动作的后续状态与新计划拼接。

---

## 15.2 动态重调度核心代码

```python
@dataclass
class RunningAction:
    entity_type: str          # "truck" or "drone"
    entity_id: int
    action_type: str          # "drive", "fly", "service", "swap"
    order_id: Optional[int]
    start_time: float
    finish_time: float
    start_loc: Tuple[float, float]
    end_loc: Tuple[float, float]
    energy_after: Optional[float] = None
    inventory_after: Optional[float] = None


@dataclass
class ExecutionSnapshot:
    time: float
    completed_orders: List[int]
    locked_orders: List[int]
    running_actions: List[RunningAction]
    truck_state: TruckState
    drone_states: Dict[int, DroneState]
    station_states: Dict[int, StationState]
```

---

## 15.3 推进仿真到事件时刻

```python
def advance_to_event_time(
    current_state: SystemState,
    current_plan: dict,
    event_time: float,
) -> ExecutionSnapshot:
    """
    将系统推进到动态订单到达时刻 event_time。

    这里做三件事：
        1. 找出已经完成的订单；
        2. 找出正在执行、不可打断的动作；
        3. 计算卡车、无人机、站点在 event_time 的物理状态。
    """

    completed_orders = []
    locked_orders = []
    running_actions = []

    # 实际项目中应从仿真器/实体状态机读取。
    # 这里示意性从 decoded_plan 中解析。
    for sortie in current_plan.get("drone_sorties", []):
        oid = sortie["order"]
        finish_customer = sortie["finish_customer"]
        takeoff = sortie["takeoff"]
        arrive_landing = sortie["arrive_landing"]

        if finish_customer <= event_time:
            completed_orders.append(oid)

        elif takeoff <= event_time < arrive_landing:
            # 无人机正在执行这个订单，视为锁定不可中断
            locked_orders.append(oid)
            running_actions.append(
                RunningAction(
                    entity_type="drone",
                    entity_id=sortie["drone"],
                    action_type="fly",
                    order_id=oid,
                    start_time=takeoff,
                    finish_time=arrive_landing,
                    start_loc=sortie["launch_loc"],
                    end_loc=sortie["landing"][2],
                )
            )

    for step in current_plan.get("truck_route", []):
        oid = step.get("order")
        finish = step.get("finish")
        arrive = step.get("arrive")

        if oid is not None:
            if finish <= event_time:
                completed_orders.append(oid)
            elif arrive <= event_time < finish:
                locked_orders.append(oid)

    # 此处简化：直接复制当前状态。
    # 工程中应根据 running action 插值计算 loc、energy、inventory。
    snapshot = ExecutionSnapshot(
        time=event_time,
        completed_orders=list(set(completed_orders)),
        locked_orders=list(set(locked_orders)),
        running_actions=running_actions,
        truck_state=copy.deepcopy(current_state.truck),
        drone_states=copy.deepcopy(current_state.drones),
        station_states=copy.deepcopy(current_state.stations),
    )

    return snapshot
```

---

## 15.4 锁定正在执行动作

现实物理系统中，不建议让无人机在空中突然改任务。因此动态重调度应采用“非抢占式重调度”：

```python
def apply_locked_actions(snapshot: ExecutionSnapshot):
    """
    对正在执行的动作进行冻结。

    策略：
        - 已送达订单：从新一轮优化集合中删除；
        - 正在配送订单：视为 locked，不参与新 GA；
        - 对应实体直到 locked action 结束前不可用；
        - locked action 结束后的状态作为实体下一次可用状态。
    """

    truck_state = snapshot.truck_state
    drone_states = snapshot.drone_states

    for action in snapshot.running_actions:
        if action.entity_type == "drone":
            drone = drone_states[action.entity_id]

            # 无人机在当前飞行动作结束前不可用
            drone.status = "unavailable"
            drone.time = action.finish_time
            drone.loc = action.end_loc

            if action.energy_after is not None:
                drone.energy = action.energy_after

        elif action.entity_type == "truck":
            truck_state.time = action.finish_time
            truck_state.loc = action.end_loc

            if action.inventory_after is not None:
                truck_state.inventory = action.inventory_after

    return truck_state, drone_states
```

---

## 15.5 提取上一轮计划剩余部分作为 warm start

这是动态 GA 收敛速度的关键。

```python
def build_warm_start_from_previous_plan(
    previous_best: Individual,
    completed_orders: List[int],
    locked_orders: List[int],
    new_order_ids: List[int],
    pop_size: int,
) -> List[Individual]:
    """
    用上一轮优秀个体的未执行部分生成 warm start。
    """

    excluded = set(completed_orders) | set(locked_orders)

    old_pairs = [
        (oid, gene)
        for oid, gene in zip(previous_best.sequence, previous_best.assignment)
        if oid not in excluded
    ]

    remaining_seq = [p[0] for p in old_pairs]
    remaining_assignment = [p[1] for p in old_pairs]

    # 新订单插入：可随机插入，也可按最近邻/截止期插入
    for oid in new_order_ids:
        pos = random.randint(0, len(remaining_seq))
        remaining_seq.insert(pos, oid)

        # 新订单初始模式可给 C 或 B，动态订单常用 C 作为启发式
        remaining_assignment.insert(pos, "C_1")

    warm = [Individual(remaining_seq, remaining_assignment)]

    # 生成一些扰动版本
    while len(warm) < max(1, int(0.2 * pop_size)):
        ind = copy.deepcopy(warm[0])
        mutate(ind, drone_ids=[1, 2, 3], p_swap=0.5, p_assign=0.3)
        warm.append(ind)

    return warm
```

---

## 15.6 动态订单接入主函数

```python
def handle_dynamic_orders_event(
    current_state: SystemState,
    current_plan: dict,
    previous_best_individual: Individual,
    new_orders: List[Order],
    event_time: float,
    weights: Dict[str, float],
    service_time: float,
    pop_size: int = 80,
    generations: int = 100,
) -> Individual:
    """
    动态订单到达时触发重调度。
    """

    # Step 1: 推进系统到事件时刻
    snapshot = advance_to_event_time(
        current_state=current_state,
        current_plan=current_plan,
        event_time=event_time,
    )

    # Step 2: 冻结正在执行的动作
    new_truck_state, new_drone_states = apply_locked_actions(snapshot)

    # Step 3: 更新系统状态
    replan_state = copy.deepcopy(current_state)
    replan_state.current_time = event_time
    replan_state.truck = new_truck_state
    replan_state.drones = new_drone_states
    replan_state.stations = snapshot.station_states

    # Step 4: 加入新订单
    for order in new_orders:
        replan_state.orders[order.oid] = order

    completed = set(snapshot.completed_orders)
    locked = set(snapshot.locked_orders)

    # Step 5: 构建新的待优化订单集合
    active_order_ids = []
    for oid, order in replan_state.orders.items():
        if oid in completed:
            continue
        if oid in locked:
            continue
        if order.status == "completed":
            continue
        if order.release_time <= event_time:
            active_order_ids.append(oid)

    new_order_ids = [o.oid for o in new_orders]

    # Step 6: warm start
    warm_start = build_warm_start_from_previous_plan(
        previous_best=previous_best_individual,
        completed_orders=list(completed),
        locked_orders=list(locked),
        new_order_ids=new_order_ids,
        pop_size=pop_size,
    )

    # Step 7: 从当前状态重新运行 GA
    best = run_ga(
        init_state=replan_state,
        active_order_ids=active_order_ids,
        pop_size=pop_size,
        generations=generations,
        weights=weights,
        service_time=service_time,
        warm_start=warm_start,
    )

    return best
```

---

# 16. 推荐的工程结构

结合已有文件结构，建议这样接入：

```text
backend/
  solver/
    ga_dynamic_mmce.py          # 新增：动态 GA 主求解器
    ga_chromosome.py            # 个体、交叉、变异
    ga_decoder.py               # 解码器：物理约束、能量、换电站、回收点
    ga_repair.py                # 修复策略：模式切换、换电站中继、库存不足
    dynamic_rescheduler.py      # 动态订单事件触发重调度
    greedy_mmce.py              # 保留：作为启发式种子生成器
    decision_engine.py          # 调用 GA 或 greedy
```

现有的 `greedy_mmce.py` 可以继续保留，作为 GA 的：

1. 种子解生成器；
2. 局部修复器；
3. 对照 baseline。

---

# 17. 关键实现建议

## 建议 1：GA 不直接优化连续起飞点

第一阶段 GA 只决定：

$$
\text{谁送？按什么顺序送？}
$$

也就是：

$$
(\pi, \gamma)
$$

第二阶段 Decoder 再决定：

$$
\text{从哪里起飞？在哪里回收？是否插入换电站？}
$$

这样可以避免 GA 染色体过长、约束过硬、可行解太少。

---

## 建议 2：模式 E 不作为原始基因，而作为修复模式

不要让染色体直接编码：

$$
E
$$

而是在解码时触发：

```python
if customer_to_recovery_energy_not_enough:
    try_insert_station_after_delivery()
```

也就是说：

$$
B_u \to B_u + E
$$

或者：

$$
C_u \to C_u + E
$$

但注意：根据规则，换电站不能插入在携货段中，因此只允许：

$$
h \to customer \to s_1 \to s_2 \to l
$$

不允许：

$$
h \to s_1 \to customer \to l
$$

---

## 建议 3：动态重调度必须非抢占

正在执行的动作不要打断：

- 卡车正在行驶：可选择继续到当前目标节点；
- 无人机正在飞：必须完成当前 sortie；
- 无人机正在卸货：必须完成服务；
- 无人机正在换电：完成换电后再参与重调度。

这样模型更接近真实无人系统。

---

## 建议 4：适应度中保留大惩罚，但不要直接丢弃个体

推荐：

$$
Fitness =
\omega_1 \sum C_i
+
\omega_2 \sum L_i
+
\omega_3 E
+
\omega_4 W
+
\omega_5 P
$$

其中 $$P$$ 包括：

- 能量不足；
- 无法找到合法落脚点；
- 无法最终回仓；
- 卡车等待超时；
- 充换电站严重拥堵；
- 库存不足且修复失败。

这样 GA 可以自然淘汰坏解，但不会因为可行解稀少而过早崩溃。

---

# 18. 最终算法流程总结

```text
输入：
    初始静态订单 O_static
    卡车状态
    无人机状态
    充换电站状态
    参数配置

初始化：
    构造 π + γ 双层染色体种群
    注入 greedy seed / truck-only seed / OBL seed

循环：
    1. 解码个体
        - 卡车路径生成
        - 无人机 sortie 生成
        - 起飞同步
        - 回收同步
        - 电量前瞻校验
        - 充换电站排队
        - 库存修复
        - 最终回仓修复

    2. 计算适应度
        - 完成时间
        - 迟到惩罚
        - 能耗
        - 等待惩罚
        - 不可行惩罚

    3. 选择
    4. OX / PMX 交叉
    5. 载具分配变异
    6. 精英保留

执行：
    输出当前最优计划的前若干动作

动态事件：
    新订单到达 t
        - 推进系统到 t
        - 冻结已完成和正在执行动作
        - 读取卡车当前位置、库存
        - 读取无人机位置、电量、状态
        - 合并未完成订单和新订单
        - 用上一轮计划剩余部分 warm start
        - 重新运行 GA
        - 输出新计划
```

---

# 19. 推荐算法名称

建议命名为：

$$
\textbf{D-GA-MMCE}
$$

即：

**Dynamic Genetic Algorithm for Multi-Modal Collaborative Execution**

核心特点：

1. **双层编码**：订单序列 $$\pi$$ + 载具/模式分配 $$\gamma$$；
2. **事件触发式重调度**：动态订单到达即冻结状态并重启 GA；
3. **Decoder 物理修复**：能量、换电站、回收点、等待、库存全部在解码阶段处理；
4. **滚动执行**：每次只执行当前计划前几步，等待新事件；
5. **软硬结合约束**：坠机、电量红线、载重为硬约束；迟到、等待、错过回收可作为软惩罚。
