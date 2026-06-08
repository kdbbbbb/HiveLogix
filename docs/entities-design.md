### 0. 基础共享类与空间语义 (Section 0)

#### 0.1 空间与状态原语

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

@dataclass
class Position3D:
    x: float  # UTM X（米），WGS84 转换前使用 UTM 坐标系
    y: float  # UTM Y（米）
    z: float  # 高度（米），相对海平面，用于禁飞区判断与三维路径规划

class SourceType(Enum):
    """取货源/归属宿主的类型。替代裸字符串，避免大小写不一致引发的逻辑错误。"""
    DEPOT = "DEPOT"   # 固定仓库节点
    TRUCK = "TRUCK"   # 移动卡车节点

class WaypointAction(Enum):
    """无人机到达某航路点后应执行的动作语义，使 route_plan 不再是纯坐标序列。"""
    PICKUP       = "PICKUP"        # 从源载具取货，触发 assign_order()
    DELIVER      = "DELIVER"       # 向客户投递货物，触发 release_order() 与订单状态更新
    SWAP_BATTERY = "SWAP_BATTERY"  # 在充换电宿主（站/仓/车）执行换电，触发 arrive()
    RENDEZVOUS   = "RENDEZVOUS"    # 与移动卡车在动态汇合点会合（坐标由调度器实时插值给出）
    DOCK_TRUCK   = "DOCK_TRUCK"    # 降落并停靠至卡车起降平台（Mode B 回收）
    DOCK_DEPOT   = "DOCK_DEPOT"    # 降落并停靠至仓库（Mode C 返仓）

@dataclass
class RouteWaypoint:
    loc: Position3D               # 航路点的三维空间坐标
    action: WaypointAction        # 到达该点后执行的动作
    target_entity_id: Optional[str] = None  # 关联的实体ID（如充换电站ID、卡车ID、订单ID），供动作执行时查找对象

class TaskStatus(Enum):
    """订单的生命周期状态机。"""
    PENDING    = "PENDING"     # 已生成，尚未分配给任何载具
    ASSIGNED   = "ASSIGNED"    # 已分配给载具，等待取货
    PICKED_UP  = "PICKED_UP"   # 无人机/卡车已完成取货，货物脱离源载具
    DELIVERING = "DELIVERING"  # 货物在途，飞往或行驶至客户点
    COMPLETED  = "COMPLETED"   # 货物已成功送达客户
    TIMEOUT    = "TIMEOUT"     # 超过软时间窗截止时间，产生惩罚成本，仍需继续履约
    REJECTED   = "REJECTED"    # 订单被系统拒绝（保留状态，当前场景下 100% 履约不应触发）

class DroneStatus(Enum):
    """无人机的物理状态机，驱动仿真引擎中的行为分支。"""
    IDLE               = "IDLE"               # 空闲停靠（在仓库或卡车上），可被调度
    FLYING_TO_PICKUP   = "FLYING_TO_PICKUP"   # 飞往取货点
    FLYING_TO_DELIVER  = "FLYING_TO_DELIVER"  # 飞往客户投递点
    FLYING_TO_STATION  = "FLYING_TO_STATION"  # 飞往充换电站补能
    FLYING_TO_TRUCK    = "FLYING_TO_TRUCK"    # 追赶移动卡车（Mode B 回收 / Mode D 补货）
    RETURNING_TO_DEPOT = "RETURNING_TO_DEPOT" # 完成任务后返回仓库（Mode C）
    QUEUING            = "QUEUING"            # 已抵达充换电宿主，在队列中等待空闲槽位
    CHARGING           = "CHARGING"           # 占用槽位，正在执行换电操作
    LOADING            = "LOADING"            # 在载具上装载货物，短暂不可调度
    UNLOADING          = "UNLOADING"          # 在载具上卸载/交接货物，短暂不可调度
    DEAD               = "DEAD"               # 异常终止（电量耗尽/坠毁），需人工介入

class TruckStatus(Enum):
    """卡车的运行状态机。"""
    IDLE            = "IDLE"            # 静止待命，未执行任何任务
    DRIVING         = "DRIVING"         # 沿规划路网行驶
    WAITING         = "WAITING"         # 停车等待无人机飞回汇合（Mode B），不推进路网
    LOADING_DRONE   = "LOADING_DRONE"   # 正在回收无人机并执行换电，短暂阻塞行驶
    UNLOADING_DRONE = "UNLOADING_DRONE" # 正在释放无人机起飞，短暂阻塞行驶
```

---

#### 0.2 充换电宿主基类 `ChargingHost`

充换电站、仓库、卡车均继承此抽象基类。调度器通过统一接口查询所有补能节点，无需感知具体类型。

```python
class ChargingHost(ABC):
    """所有可对无人机执行充换电服务的宿主基类。"""

    # ── 子类须在 __init__ 中初始化的共享属性 ─────────────────
    swap_time: float                    # 单次换电耗时（秒），由具体宿主的硬件参数决定
    parking_slots: int                  # 并发服务槽位数 K，即同一时刻最多可服务的无人机数量
    wait_queue: list[str]               # 等待换电的无人机 ID 队列，按 FIFO 顺序调度
    serving_drones: dict[str, float]    # 正在占用槽位换电的无人机及其预计完成时间戳 {drone_id: finish_time}

    # ── 共享方法（提供默认实现，子类按需重写）──────────────────

    def arrive(self, drone_id: str, current_time: float) -> None:
        """
        无人机请求进入充换电服务。
        检查当前占用槽位数是否小于 parking_slots：
          - 有空位：立即开始服务，记录预计完成时间为 current_time + swap_time；
          - 无空位：加入 wait_queue 末尾等待。
        Truck 子类需重写此方法，在调用 super().arrive() 前额外检查起降平台物理占位。
        """
        ...

    def depart(self, drone_id: str, current_time: float) -> None:
        """
        换电完成，释放该无人机占用的槽位。
        若 wait_queue 非空，立即拉取队首无人机进入服务，
        为其记录新的预计完成时间为 current_time + swap_time。
        """
        ...

    def tick_update(self, current_time: float) -> list[str]:
        """
        仿真心跳调用（每个时间步执行一次）。
        遍历 serving_drones，找出预计完成时间 ≤ current_time 的无人机，
        对每架调用 depart() 释放槽位并触发后续队列调度。
        返回本轮完成换电的无人机 ID 列表，供仿真引擎更新这些无人机的状态。
        """
        ...

    def estimate_wait_time(self, current_time: float) -> float:
        """
        调度器在派单前调用，预估一架无人机此刻飞来后需等待多久才能开始换电。
        若当前有空闲槽位则返回 0；
        否则取 serving_drones 中最早完成时间距 current_time 的差值作为首个可用槽的等待时长，
        再叠加 wait_queue 中排在前面的无人机所需的累计换电时间。
        """
        ...

    # ── 抽象方法（子类必须实现）──────────────────────────────

    @abstractmethod
    def get_location(self, current_time: float) -> Position3D:
        """
        返回宿主在给定时刻的三维坐标（UTM）。
        固定节点（SwapStation / Depot）直接返回存储坐标，忽略时间参数；
        移动节点（Truck）需按 route_nodes 序列对 current_time 做线性插值。
        调度器选择最优补能宿主时统一调用此接口计算飞行距离与时间成本。
        """
        ...

    @abstractmethod
    def to_telemetry_dict(self) -> dict:
        """
        将宿主当前状态序列化为字典，坐标转换为 WGS84 格式，
        供 WebSocket 帧推送至前端地图渲染（位置标记、队列长度、槽位占用可视化等）。
        """
        ...
```

**继承体系一览：**

```
ChargingHost (ABC)
│  swap_time / parking_slots / wait_queue / serving_drones
│  arrive() / depart() / tick_update() / estimate_wait_time()
│  get_location() [abstract] / to_telemetry_dict() [abstract]
│
├── SwapStation(ChargingHost)   固定节点，直接复用基类换电逻辑
├── Depot(ChargingHost)         固定节点，仓库充能 + 资源管理
└── Truck(ChargingHost)         动态节点，重写 arrive() 加入起降平台占位逻辑
```

---

### 1. 订单模型 (Order)

| 属性字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `order_id` | `str` | 全局唯一ID |
| `create_time` | `float` | 仿真时间轴上的生成时间 |
| `deadline` | `float` | 期望最晚送达时间（软时间窗） |
| `delivery_loc` | `Position3D` | 客户收货静态三维坐标 |
| `pickup_source_id` | `str` | 货物所在载具/节点ID（如 `depot_1`, `truck_2`） |
| `source_type` | `SourceType` | ✅ 取货源类型枚举（`DEPOT` 或 `TRUCK`），类型安全，避免裸字符串歧义 |
| `payload_weight` | `float` | 货物重量（kg） |
| `status` | `TaskStatus` | 订单当前状态 |
| `assigned_vehicle_id` | `str` | 接单的载具ID（多跳 Mode E 中记录**发起派单**的实体） |
| `assigned_mode` | `str` | 记录选用的履约模式 (Mode A~E) |
| `penalty_rate` | `float` | 单位时间超时罚金权重 |
| `actual_deliver_time` | `float` | 实际送达时间 |

**核心类函数 (Methods):**
* `update_status(new_status)`: 状态机推进。
* `get_delay_penalty(current_time: float) -> float`: $\max(0, current\_time - deadline) \times penalty\_rate$。
* `get_rl_state_vector(ref_loc: Position3D, current_time: float) -> list`: [RL接口] 传入参考系坐标，返回归一化后的 `[相对dx, 相对dy, 剩余时间, 重量]` 向量。

---

### 2. 异构无人机模型 (Drone)

| 属性字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `drone_id` | `str` | 全局唯一ID |
| `drone_type` | `str` | `LIGHT` 或 `HEAVY` |
| `home_id` | `str` | 归属母体ID（仓库或卡车），返航基准 |
| `home_type` | `SourceType` | ✅ 归属类型（`DEPOT`/`TRUCK`），区分静态返仓与动态追车两种返航逻辑 |
| `current_loc` | `Position3D` | 当前实时坐标 |
| `status` | `DroneStatus` | 当前物理状态 |
| `battery_max` | `float` | 最大电池容量 (Joules) |
| `battery_current` | `float` | 当前剩余电量 |
| `payload_capacity` | `float` | 最大载重限制 (kg) |
| `current_payload` | `float` | 当前挂载重量 (kg) |
| `carrying_order_id` | `Optional[str]` | ✅ 当前挂载的订单ID（空载为 `None`） |
| `cruise_speed` | `float` | 巡航速度 (m/s) |
| `empty_weight` | `float` | 机身自重 (kg) |
| `k_1` | `float` | ✅ 诱导功率系数（由子类固化，影响悬停/爬升功耗） |
| `k_2` | `float` | ✅ 废阻功率系数（由子类固化，影响高速巡航功耗） |
| `route_plan` | `list[RouteWaypoint]` | 完整航路点队列（**只读，不弹出**） |
| `current_waypoint_index` | `int` | ✅ 当前执行到的航路点下标；`move_step` 推进指针，列表本身保持完整以支持回放和诊断 |
| `cumulative_distance` | `float` | 累计飞行距离（维护/寿命统计） |

**核心类函数 (Methods):**
* `move_step(dt: float)`: 沿 `route_plan[current_waypoint_index]` 移动。到达后触发 action，`current_waypoint_index += 1`（不修改列表本身）。
* `calculate_power(payload: float, v: float) -> float`: 多旋翼功率模型：$P = k_1 \cdot (m_{empty} + payload)^{3/2} + k_2 \cdot v^3$（诱导功率 + 废阻功率）。
* `consume_energy(dt: float)`: `battery_current -= calculate_power(...) * dt`。
* `can_reach(target: Position3D, payload: float, safe_margin: float) -> bool`: ✅ 判断当前电量能否飞到目标点并保留安全余量，**调度器派单前必须调用**。
* `get_remaining_range() -> float`: 快速预估当前电量支持的最大飞行距离（空载）。
* `assign_order(order_id: str)` / `release_order()`: 订单绑定/解除，同步更新 `carrying_order_id` 和 `current_payload`。
* `to_telemetry_dict() -> dict`: 序列化并转换为 WGS84 坐标格式。

**异构子类实现：**
* **LightDrone(Drone):** $cruise\_speed=15\ \text{m/s},\ payload\_capacity=2\ \text{kg},\ empty\_weight=1.5\ \text{kg},\ k_1=?,\ k_2=?$
* **HeavyDrone(Drone):** $cruise\_speed=10\ \text{m/s},\ payload\_capacity=10\ \text{kg},\ empty\_weight=5\ \text{kg},\ k_1=?,\ k_2=?$

> $k_1$、$k_2$ 由空气动力学标定实验或文献给出，在 `config/` 中配置，子类 `__init__` 中读取赋值。

---

### 3. 充换电站 (SwapStation) — 继承自 `ChargingHost`

固定节点，直接复用基类全部充换电逻辑，自身仅扩展身份与位置信息。

| 属性字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `station_id` | `str` | 全局唯一ID |
| `location` | `Position3D` | 电站固定位置 |
| *(继承)* `swap_time` | `float` | 每次换电固定耗时（秒） |
| *(继承)* `parking_slots` | `int` | 最大并发服务槽位 $K$ |
| *(继承)* `wait_queue` | `list[str]` | 等待换电的无人机队列 |
| *(继承)* `serving_drones` | `dict[str, float]` | 正在换电的无人机 `{drone_id: finish_time}` |

**实现抽象方法：**
* `get_location(current_time: float) -> Position3D`: 直接返回 `self.location`（固定节点，忽略时间参数）。
* `to_telemetry_dict() -> dict`: ✅ 序列化位置（WGS84）、槽位占用数、队列长度供前端渲染。

**继承自基类（无需重写）：**
`arrive()` / `depart()` / `tick_update()` / `estimate_wait_time()`

---

### 4. 移动基站：卡车 (Truck) — 继承自 `ChargingHost`

卡车重写 `arrive()` 以加入起降平台的物理占位逻辑。`parking_slots` 复用语义等同于旧版 `max_drone_slots`。

| 属性字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `truck_id` | `str` | 全局唯一ID |
| `status` | `TruckStatus` | 卡车当前状态（含 `WAITING` 等待无人机汇合） |
| `current_loc` | `Position3D` | 实时坐标 |
| `speed` | `float` | 地面行驶速度 |
| `route_nodes` | `list[str]` | SUMO 路网节点序列 |
| `inventory` | `dict[str, Order]` | 车厢订单包裹 `{order_id: Order}` |
| `max_inventory` | `int` | 车厢最大装载包裹数量上限 |
| `docked_drones` | `list[str]` | 当前停靠在车顶的无人机 |
| *(继承)* `swap_time` | `float` | 车载换电耗时（秒） |
| *(继承)* `parking_slots` | `int` | 起降平台并发数（即原 `max_drone_slots`） |
| *(继承)* `wait_queue` | `list[str]` | 等待降落/换电的无人机队列 |
| *(继承)* `serving_drones` | `dict[str, float]` | 正在换电的无人机 |

**重写/扩展方法：**
* `arrive(drone_id: str, current_time: float)` *(override)*: 先检查 `docked_drones` 平台是否有空位；有位则将无人机加入 `docked_drones` 并调用 `super().arrive()` 开始换电；否则加入 `wait_queue`。
* `get_location(current_time: float) -> Position3D` *(implement)*: 按路网节点序列做线性插值，返回当前估计坐标。
* `to_telemetry_dict() -> dict` *(implement)*: ✅ 序列化 WGS84 位置、状态、库存快照供前端渲染。

**专属方法：**
* `move_step(dt: float)`: 沿地面路网推演位置。
* `launch_drone(drone_id: str, order_id: str)`: 从 `docked_drones` 移除无人机，`status → UNLOADING_DRONE`，短暂阻塞后恢复 `DRIVING`。
* `recover_drone(drone_id: str, current_time: float)`: 收回无人机，调用 `arrive()` 进入换电队列，`status → LOADING_DRONE`。
* `deliver_order(order_id: str, current_time: float)`: ✅ Mode A 直递：从 `inventory` 移除订单，更新订单状态为 `COMPLETED`，记录 `actual_deliver_time`。
* `get_future_trajectory(time_window: float) -> list[Position3D]`: 预测未来轨迹，为无人机规划 `RENDEZVOUS` 汇合点。

---

### 5. 全局起点：仓库 (Depot) — 继承自 `ChargingHost`

`parking_slots` 对应旧版 `recharge_slots`，充换电并发逻辑全部由基类接管。

| 属性字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `depot_id` | `str` | 全局唯一ID |
| `location` | `Position3D` | 仓库固定位置 |
| `drone_fleet` | `list[str]` | 归属本仓的无人机全集 |
| `truck_fleet` | `list[str]` | 归属本仓的卡车全集 |
| `idle_drones` | `list[str]` | 当前在库可调度的空闲无人机 |
| `capacity` | `int` | 仓库订单吞吐量上限 |
| `pending_orders` | `list[Order]` | 待分配/装载的订单池 |
| *(继承)* `swap_time` | `float` | 仓库换电耗时（秒） |
| *(继承)* `parking_slots` | `int` | 仓库充/换电并行位数（即原 `recharge_slots`） |
| *(继承)* `wait_queue` | `list[str]` | 等待充换电的无人机队列 |
| *(继承)* `serving_drones` | `dict[str, float]` | 正在换电的无人机 |

**实现抽象方法：**
* `get_location(current_time: float) -> Position3D`: 直接返回 `self.location`（固定节点）。
* `to_telemetry_dict() -> dict`: ✅ 输出静态 WGS84 坐标、库存快照及充电槽位状态。

**专属方法：**
* `dispatch_drone(drone_id: str)`: 从 `idle_drones` 移除，标记无人机出发。
* `receive_drone(drone_id: str, current_time: float)`: 接收无人机，调用 `arrive()` 进入换电队列；换电完成后由 `tick_update()` 回调将其加回 `idle_drones`。
* `load_truck(truck_id: str, orders: list[str])`: 为卡车装车，受 `Truck.max_inventory` 约束，转移 `pending_orders` 中对应的订单。