# Phase 3: 后端状态管理与前后端协同架构设计方案（修订版 v4）

> **修订说明（v4.9）**：在 v4.8 基础上纳入第十三轮代码核查发现的 2 项文档内部矛盾：
> 1. **⚪ `EntityManager.get_telemetry()` 描述遗漏 `_metadata` 合并**：Section 2.1 将 `get_telemetry()` 描述为"仅返回高频变化字段...调用 `to_dynamic_state()`"，未提及 `_metadata` 合并。但 `DepotConfig.name`、`StationConfig.name`、`TruckConfig.name` 和 `TruckConfig.home_depot_id` 均为 TypeScript **非 Optional** 必填字段，`setRuntimeAll()` 全量替换后若缺失则立即变为 `undefined`（仓库/充换电站/卡车名称字段在首 TICK 后全部显示为空）。这些字段不在任何实体的 `to_dynamic_state()` 输出中（存于 `_metadata`），必须在 `get_telemetry()` 中同 `to_dynamic_state()` 结果合并（`{**entity.to_dynamic_state(), **_metadata[id]}`）——与 `get_static_snapshot()` 中的 `{**to_telemetry_dict(), **_metadata[id]}` 模式完全对称。**修正**：在 Section 2.1 `get_telemetry()` 描述中补充 `_metadata` 合并说明。
> 2. **⚪ `EntityManager.load_from_config()` 引用 `coord_transformer.wgs84_to_utm` 无法直接导入**：Section 2.1 描述中写 "WGS84 → UTM 使用已有 `coord_transformer.wgs84_to_utm(lon, lat)`"，但 `backend/core/entities/coord_transformer.py` **不存在**（正确路径为 `backend/environment/scene/coord_transformer.py`）；`primitives.py` 文档注释也明确指出"WGS84↔UTM 转换由 `utils.coord_utils` 负责"。Section 2.2 已正确使用 `from utils.coord_utils import wgs84_to_utm`。若实现者按 Section 2.1 字面尝试 `from core.entities.coord_transformer import wgs84_to_utm`，会得到 `ImportError`。**修正**：将 Section 2.1 涉及坐标转换的描述改为显式 `from utils.coord_utils import wgs84_to_utm`，与 Section 2.2 保持一致。
>
> **修订说明（v4.8）**：在 v4.7 基础上纳入第十二轮代码核查发现的 1 项 `to_dynamic_state()` 遗漏字段问题：
> 1. **⚪ `truck.to_dynamic_state()` 缺少 `available_slots` 和 `docked_drones`**：两者均是动态字段——`available_slots`（= `parking_slots - len(serving_drones)`，随无人机换电进出变化）由 `ChargingHost.available_slots` 属性提供（内部已加锁，可直接调用）；`docked_drones: list[str]` 记录停靠车顶的无人机 ID，随无人机起降变化。`to_telemetry_dict()` 已包含两者，TypeScript `Truck.available_slots?: number` / `Truck.docked_drones?: string[]` 均已声明为 Optional，FULL_SNAPSHOT 示例亦包含这两个字段。但当前 `to_dynamic_state()` 规格未列出，导致首次 TICK 全量替换后 `rtTrucks[i].available_slots` 和 `rtTrucks[i].docked_drones` 变为 `undefined`，任何读取这两个字段的 UI 组件（车辆详情面板、地图标记悬停信息）将在联调时静默失效。**修正**：在 `truck.to_dynamic_state()` 动态字段区补充两者，并同步更新 TICK 示例与 Section 7.1.2、7.4 汇总表。
>
> **修订说明（v4.7）**：在 v4.6 基础上纳入第十一轮代码核查发现的 2 项 `to_dynamic_state()` 遗漏字段问题：
> 1. **⚪ `swap_station.to_dynamic_state()` 缺少 `serving_drone_ids`**：`charging_host.py` 基类的 `serving_drones: dict[str, float]` 字段记录正在换电的无人机，`swap_station.to_telemetry_dict()` 已序列化为 `serving_drone_ids`，TypeScript `Station.serving_drone_ids?: string[]` 已声明，FULL_SNAPSHOT 示例亦包含此字段。但 Section 7.1.2 的 `to_dynamic_state()` 规格漏掉了它。由于 `setRuntimeAll()` 做全量数组替换，每次 TICK 后 `rtStations[i].serving_drone_ids` 变为 `undefined`，充换电站"正在服务无人机"UI 数据在首帧后永久丢失。**修正**：在 `swap_station.to_dynamic_state()` 末尾补充 `"serving_drone_ids": list(self.serving_drones.keys())`（需在 `with self._lock:` 内读取），并同步更新 TICK 示例与 Section 7.1.2。
> 2. **⚪ `drone.to_dynamic_state()` 缺少 `cumulative_distance_m` / `remaining_range_m`**：`drone.to_telemetry_dict()` 已序列化这两个字段（均含 `_m` 后缀），TypeScript `Drone.cumulative_distance_m?: number` / `Drone.remaining_range_m?: number` 均已声明，FULL_SNAPSHOT 示例也包含它们。两者均为高频动态字段（无人机每步飞行累计里程递增、电量消耗导致剩余航程下降）。但 Section 7.1.2 的 `to_dynamic_state()` 规格漏掉了两者，TICK 全量替换后 `rtDrones[i].cumulative_distance_m` / `rtDrones[i].remaining_range_m` 变为 `undefined`，导致战情大屏"累计里程"/"剩余航程"指标在首次 TICK 后全部归零消失。**修正**：在 `drone.to_dynamic_state()` 末尾补充两个字段，并同步更新 TICK 示例与 Section 7.1.2。
>
> **修订说明（v4.6）**：在 v4.5 基础上纳入第十轮代码核查发现的 1 项**文档内部矛盾**：
> 1. **🔴 Section 4 TICK `entities` 示例与 Section 7.1.2 `to_dynamic_state()` 规格冲突**：Section 4 的 TICK 示例中，每类实体只列了 2–4 个精简字段（早期草稿残留）：`trucks` 仅有 `{truck_id, lng, lat, status}`；`depots` 仅有 `{depot_id, pending_count, idle_drone_count}`；`stations` 仅有 `{station_id, available_slots, queue_length}`；`drones` 仅有 `{drone_id, lng, lat, battery_ratio, status}`。但 `entityStore.setRuntimeAll()` 做**全量数组替换**（非字段合并），TICK 帧实体对象必须包含所有在 TypeScript `***Config` 中声明为非 Optional 的字段（如 `TruckConfig.name/home_depot_id/speed/max_inventory/swap_time/parking_slots`，`DepotConfig.lng/lat/altitude/capacity/swap_time/parking_slots/name` 等），否则第一个 TICK 就使这些字段变为 `undefined`。Section 7.1.2 的 `to_dynamic_state()` 规格是正确的权威规格，Section 4 示例系过时草稿。**修正**：将 Section 4 TICK 实体示例全部更新为与 Section 7.1.2 对齐的完整字段对象（已更新）。
>
> **修订说明（v4.5）**：在 v4.4 基础上纳入第八轮代码核查发现的 1 项新矛盾：
> 1. **⚪ `get_status_summary()` 返回键名与 TICK JSON / `SimStats` 不一致**：Section 2.2 描述 `get_status_summary()` 返回 `{pending, assigned, completed, timeout}`（无 `orders_` 前缀），但 Section 4 TICK 帧 JSON 示例和 Section 7.6 `SimStats` TypeScript 接口均使用 `orders_pending/orders_assigned/orders_completed/orders_timeout`（含前缀）。开发者按 Section 2.2 实现后端后，前端读取 `stats.orders_pending` 将得到 `undefined`。**修正**：将 Section 2.2 的描述改为 `{orders_pending, orders_assigned, orders_completed, orders_timeout}`，与 TICK JSON 和 SimStats 接口对齐。
>
> **修订说明（v4.4）**：在 v4.3 基础上纳入第七轮代码核查发现的 2 项新矛盾：
> 1. **🔴 `depot.to_dynamic_state()` 缺少 `lng/lat/altitude`**：`entity.ts` 中 `setRuntimeAll` 对 `depots` 做全量数组替换（`rtDepots.value = data.depots`）。`DepotConfig`（`Depot` 的父类型）中 `lng`、`lat`、`altitude` 均为**非 Optional 必填字段**。若 TICK 帧的 `to_dynamic_state()` 不含坐标，第一个 TICK 后 `rtDepots[i].lng === undefined`，地图仓库标记立刻失位/消失。**修正**：在 `depot.to_dynamic_state()` 中增加 `lon, lat = self.location.to_wgs84()` 调用，补充 `"lng"`, `"lat"`, `"altitude": self.location.z` 三个字段。
> 2. **🔴 `station.to_dynamic_state()` 缺少 `lng/lat/altitude`**：与矛盾 1 完全对称。`StationConfig.lng/lat/altitude` 也是非 Optional 必填字段，TICK 全量替换后充换电站标记同样失位。**修正**：在 `station.to_dynamic_state()` 中补充同样的三个坐标字段。
>
> **修订说明（v4.3）**：在 v4.2 基础上纳入第六轮代码核查发现的 3 项新矛盾：
> 1. **🔴 TICK 替换后 rtTrucks/rtDepots/rtStations 丢失 Config 必填静态字段**：v4.2 只为 Drone 补了 `drone_type/home_id/home_type`，但 `depot.to_dynamic_state()` 不含 `capacity/swap_time/parking_slots`，`station.to_dynamic_state()` 不含 `swap_time/parking_slots`，`truck.to_dynamic_state()` 不含 `speed/max_inventory/swap_time/parking_slots`。首帧 FULL_SNAPSHOT 后第一个 TICK 就把 `rtDepots/rtTrucks` 替换为缺字段对象，组件读取 `rtDepots[i].capacity` 等必填字段得到 `undefined`。**修正**：三个 `to_dynamic_state()` 各自补充对应的静态 Config 字段。
> 2. **🔴 `OrderManager.update_timeouts()` 调用路径违反状态机**：`primitives.py` 中 `valid_transitions()` 规定 `PENDING` 只能转移到 `{ASSIGNED, REJECTED}`，不允许 `PENDING → TIMEOUT`。文档要求对超时 pending 订单调用 `update_status(TIMEOUT)` 会直接抛出 `ValueError`。**修正**：在 `primitives.py` 的 `valid_transitions()` 中为 `PENDING` 追加 `cls.TIMEOUT`。
> 3. **⚪ `SimStats` 字段名与 TICK JSON 不一致**：Section 4 JSON 统一使用 `orders_pending/orders_assigned/orders_completed/orders_timeout`，但 Section 7.6 的 `SimStats` 接口用不带前缀的 `pending/assigned/completed/timeout`，前端读取 `stats.pending` 拿到 `undefined`。**修正**：`SimStats` 加 `orders_` 前缀。
>
> **修订说明（v4.2）**：在 v4.1 基础上纳入第五轮代码核查发现的 2 项新矛盾（均为严重运行时错误）：
> 1. **🔴 TICK 全量替换丢失静态字段**：`setRuntimeAll` 对每个数组做全量替换（`rtDrones.value = data.drones`）。TICK 帧若只推 `to_dynamic_state()` 的动态字段（无 `drone_type / home_id / home_type`），首帧 TICK 后 `rtDrones` 就丢失这些字段，地图图标类型（LightDrone vs HeavyDrone）等渲染信息全部丢失。**修正**：将 `drone_type / home_id / home_type` 纳入 `to_dynamic_state()` 输出；Truck 同理补充 `home_depot_id / name`。
> 2. **🔴 `orderStore.stats` 字段不存在**：`order.ts` 的 Pinia store 只有 `generatorConfig / generatedOrders / generatorRunning` 等字段，无 `stats`。TICK 处理器 `orderStore.stats = payload.stats` 是对 Pinia store Proxy 的非声明式赋值，在 Vue 3 strict 模式下会抛出异常，否则是 non-reactive 的死赋值。**修正**：在 `orderStore` 中声明 `const stats = ref<SimStats|null>(null)` 并暴露。
>
> **修订说明（v4.1）**：在 v4 基础上纳入第四轮代码核查发现的 6 项新矛盾（含 3 个严重运行时错误）：
> 1. **entityStore 运行时命名**：运行时快照在 entityStore 中名称为 `rtDrones/rtDepots/...`（数组），不是 `drones[id]` 字典；且已有 `setRuntimeAll()` 方法，FULL_SNAPSHOT/TICK 处理器均应调用它。
> 2. **systemStore 缺少 `simStartWallMs`**：该字段不存在，需在 `system.ts` 中添加。
> 3. **`Order.to_telemetry_dict()` AttributeError**：`source_type` 改为 Optional 后，`self.source_type.value` 在 None 时崩溃，需要 None 守卫。
> 4. **TICK 热更新代码修正**：`rtDrones` 是数组，不支持 `[id]` 访问；统一改为 `setRuntimeAll()` 全量替换。
> 5. **`BASE_DIR` 未显式注入 sys.path**：`wgs84_to_utm` 导入依赖 Python 隐式 CWD，需显式 `sys.path.insert(0, BASE_DIR)`。
> 6. **前端 `Order` 接口 `pickup_source_id` 需改 Optional**：后端改为可返回 null，前端类型需同步。
>
> **修订说明（v4）**：本版本在 v3 基础上，纳入第三轮代码核查发现的 7 项新矛盾：
> 1. **delivery_loc & bbox**：订单配送目的地与仓库无关，但坐标生成需要地图边界。放弃依赖 scene_service 内存缓存（服务重启失效），改为在 `/api/sim/init` 请求体中随实体清单一起传入 `bbox` 字段，OrderManager 缓存使用。
> 2. **pickup_source_id 改为 Optional**：取货源由调度算法决定，不在订单生成时指定，需修改 `Order.__init__` 使 `pickup_source_id` / `source_type` 变为 `Optional`。
> 3. **Truck 归属无人机初始化顺序**：第二步内部拆成「先建 Truck，再建 Drone」两阶段，`home_type=="TRUCK"` 的无人机 `init_loc` 取自已建好的 Truck；关联注册改为 `docked_drones.append()`（Truck 无 `register_drone` 方法）。
> 4. **ROUTES_DIR 补入 sys.path**：`simulation_bp.py` 在 `api/routes/` 下，需追加 `ROUTES_DIR` 注入。
> 5. **SimEngine 广播机制**：`flask-sock` 每连接独占线程，后台线程不能直接调用 ws.send()；新增线程安全连接注册表（set + Lock）与 `broadcast_tick()` 函数。
> 6. **WsClient 调用顺序 + FULL_SNAPSHOT 幂等性**：`.on()` 必须在 `.connect()` 之前调用；FULL_SNAPSHOT 处理器必须用 ID 覆盖而非 push，防重连重复实体。
> 7. **simTime 从未更新**：TICK 处理器中补充 `systemStore.simTime = payload.sim_time`。
>
> **v3 修订说明**：将代码深度核查发现的 7 项矛盾全部纳入设计：新增 Sidecar 元数据模式解决实体字段缺失；引入 `sim_start_wall_ms` 锚点与 `time_domain` 字段解决时间单位冲突；明确各实体分别实现 `to_dynamic_state()` 实现瘦 TICK 帧；修正 CORS 配置；以最小侵入的 `sys.path.insert` 方案解决新模块导入问题。
>
> **v2 修订说明**：在 v1 基础上，将评审意见中经代码核查确认的问题（双时钟、状态恢复、Tick 解耦）正式纳入设计，对 v1 原始方案中描述不准确的部分进行修正，并补充了 Z 轴坐标映射规则。

> **设计基准与现状**：
> 经过对当前代码库的全面梳理，发现系统具有"强前端、强核心实体、弱中间桥梁"的特点。
> 1. **前端自持**：`entityStore` (载具/设施) 数据仅保存在浏览器 `localStorage`；`orderStore` (订单推演) 所有生成逻辑（包括 `_mainTimerId`、`_burstExtraId` 等三个独立时钟）完全依赖物理 wall-clock，`systemStore.speedRatio` 从未被任何 Store 读取；WebSocket 工具类 `WsClient` 已完整封装但无任何业务调用。
> 2. **核心实体完善**：后端 `core/entities` 目录下有极高完成度的领域模型（完整状态机、排队模型、电量模型、`to_telemetry_dict` 等），`coord_transformer.py` 已封装 WGS84↔UTM 双向转换工具。
> 3. **管理与 API 层脱节**：后端没有全局容器（Manager）去持有并驱动这些实体；`backend/environment/state/` 目录为空；`backend/api/websockets/` 目录为空。

---

## 1. 核心架构与数据流流向

我们贯彻 **后端作为 Single Source of Truth（唯一真实数据源）** 的原则。整体数据管道如下：

```
前端（配置器 + 展示器）
   │
   │  POST /api/sim/init  ── 推送实体清单 + 订单生成策略参数
   ▼
EntityManager  →  实例化 Depot / SwapStation / Drone / Truck (core.entities)
OrderManager   →  按仿真时钟自驱生成 Order，维护 pending/assigned/completed 池
SimulationEngine  → 物理步进（physics_tick），固定 10fps 向前端广播
   │
   │  WebSocket ws://.../api/ws/telemetry
   │    ├ 建连首帧：FULL_SNAPSHOT（包含完整静态元数据 + 当前运行态）
   │    └ 持续推帧：TICK（仅推动态字段，100ms 固定间隔）
   ▼
前端 systemStore（WsClient 消费）→ entityStore 运行时字段更新 → 地图 / 大屏渲染
```

> **与 v1 的关键差异**：
> - 订单生成不再由前端定时器控制，改由后端 `OrderManager` 按仿真时钟自驱。
> - WebSocket 广播频率与物理计算频率解耦，固定 10fps。
> - 新增 `GET /api/sim/state` 和 WebSocket 首帧 `FULL_SNAPSHOT` 应对状态恢复场景。

---

## 2. Manager 层设计 (`backend/environment/state/`)

在后端增加长期驻留内存的单例管理器层。

### 2.1 `EntityManager` (`entity_manager.py`)

**作用**：全局实体容器，维护当前所有的载具与基础设施实例。

**存储结构**：
```python
depots:   dict[str, Depot]
stations: dict[str, SwapStation]
drones:   dict[str, Drone]
trucks:   dict[str, Truck]
# Sidecar 元数据：存储 core.entities 构造函数不接受的 UI 字段（name / home_depot_id 等）
_metadata: dict[str, dict]   # {entity_id: {"name": ..., "type": "DEPOT"|"TRUCK"|..., "home_depot_id": ...}}
```

**核心方法**：

- `load_from_config(config_json)`:
  接收前端 `/api/sim/init` 传来的实体清单，实例化对应 `core.entities` 对象。**需严格按以下顺序执行**：

  **第一步：实例化基础设施（Depot / SwapStation）**，Z 轴映射规则：
  - `Depot`：`Position3D(x, y, z=config.altitude)`，`altitude` 字段由前端配置传入（默认 0）。
  - `SwapStation`：同上，使用 `config.altitude`。
  - WGS84 → UTM：使用 `from utils.coord_utils import wgs84_to_utm`（`backend/utils/coord_utils.py` 包装层；`core/entities/coord_transformer.py` 不存在，`environment/scene/coord_transformer.py` 须通过此包装层访问；**与 Section 2.2 `OrderManager` 保持一致**，`sys.path.insert(0, BASE_DIR)` 使其可导入，详见 Section 7.2.2）。Z 轴不需投影变换，直接赋值：`Position3D(x=x, y=y, z=config["altitude"])`。

  **第二步：实例化移动载具（分两阶段，Truck 必须先于 Drone）**：

  - **阶段 2a — 先建 Truck**：`init_loc = Position3D(x=depots[home_depot_id].location.x, y=..., z=0)`（取对应 Depot 的 UTM xy，高度为 0）。
  - **阶段 2b — 再建 Drone**（Truck 已可在上步查找）：
    - `home_type == "DEPOT"`：`init_loc = depots[home_id].location`
    - `home_type == "TRUCK"`：`init_loc = trucks[home_id].current_loc`（取已建好的 Truck 当前位置）

  > **为什么必须拆两阶段**：`home_type == "TRUCK"` 的无人机需要从 Truck 读取初始坐标。若 Drone 与 Truck 混序创建，会出现 KeyError。

  **第三步：关联注册（必须在实体实例化后执行，否则 `idle_drone_count` / `drone_fleet_count` 永远为 0）**：
  - 每个 `home_type == "DEPOT"` 的无人机：调用 `depots[home_id].register_drone(drone_id, is_idle=True)`。
  - 每个 `home_type == "TRUCK"` 的无人机：直接 `trucks[home_id].docked_drones.append(drone_id)`（Truck 没有 `register_drone` 方法，只有 `docked_drones: list[str]`）。
  - 每个 Truck（按 `home_depot_id`）：调用 `depots[home_depot_id].register_truck(truck_id)`。

  **第四步：填充 Sidecar 元数据**（补充 `core.entities` 构造函数不接受的 UI 字段）：
  - Depot / Station：`_metadata[id] = {"name": ..., "type": "DEPOT"/"STATION"}`。
  - Truck：`_metadata[id] = {"name": ..., "type": "TRUCK", "home_depot_id": ...}`。
  - Drone：`_metadata[id] = {"type": "DRONE"}`（`home_id` / `home_type` 已在 Drone 构造函数中存储，无需 sidecar）。

- `get_static_snapshot() -> dict`:
  返回所有实体的**完整元数据**（包括静态配置字段 `name`、`drone_type`、`home_id` 等），用于 WebSocket 首帧 `FULL_SNAPSHOT` 和 `GET /api/sim/state`。
  实现：对每个实体调用 `to_telemetry_dict()`，再与 `_metadata[entity_id]` 合并（`{**telemetry, **_metadata[id]}`），使 `name`、`home_depot_id` 等字段出现在输出中。

- `get_telemetry() -> dict`:
  遍历所有实体调用 `to_dynamic_state()`（**而非** `to_telemetry_dict()`），将结果与 `_metadata[entity_id]` 合并后返回，用于 100ms 广播 `TICK` 帧。
  **（v4.9 修正）必须合并 `_metadata`**，与 `get_static_snapshot()` 完全对称：`{**entity.to_dynamic_state(), **_metadata[entity_id]}`。`DepotConfig.name`、`StationConfig.name`、`TruckConfig.name`、`TruckConfig.home_depot_id` 均为 TypeScript 非 Optional 字段，不在任何实体的 `to_dynamic_state()` 输出中；若不合并，首 TICK 后这些字段在 `rtDepots/rtStations/rtTrucks` 中全部变为 `undefined`，UI 实体名称字段全部消失。
  各实体类分别实现 `to_dynamic_state()`（异构字段不适合基类统一），详见第 7 节「代码修改清单」。

- `tick_all(current_time: float, dt: float)`:
  驱动所有实体的物理步进：站点的 `tick_update()`、无人机与卡车的位移/电量更新。

### 2.2 `OrderManager` (`order_manager.py`)

**作用**：按仿真时钟自驱生成订单，维护全局订单池，向调度引擎暴露可用任务。

> **v1 修正**：v1 中 `inject_orders` 语义是"接收前端推入的订单"，与双时钟问题冲突。
> 现改为：订单由后端完全自驱生成，前端只传入生成策略参数（`OrderGeneratorConfig`）。

**存储结构**：
```python
pending_orders:   dict[str, Order]   # 待分配
assigned_orders:  dict[str, Order]   # 配送中
completed_orders: list[Order]        # 完成/超时（仅保留末尾 N 条用于统计）

# 生成策略（来自 /api/sim/init 中的 order_gen_config 字段）
_gen_config: dict          # arrival_rate, geo_mode, cluster_radius_km, burst_*, max_orders 等
_next_order_time: float    # 下一次生成订单的仿真时间点

# 地图边界（来自 /api/sim/init 中的 bbox 字段，用于生成配送目的地坐标）
# 注意：不从 scene_service 查询，避免服务重启后内存缓存失效
_bbox: dict   # {"min_lng": ..., "min_lat": ..., "max_lng": ..., "max_lat": ...}
```

**核心方法**：

- `configure(gen_config, bbox)`:
  接收前端传入的策略参数（与前端 `types/index.ts` 的 `OrderGeneratorConfig` 字段完全对齐，含 Phase 2 扩展的 6 个字段：`max_orders`、`geo_mode`、`cluster_radius_km`、`burst_enabled`、`burst_multiplier`、`burst_duration_s`），同时存储 `bbox`。

  **`window_min` / `window_max` 单位转换**：前端传入的是分钟，后端 Order 使用秒。创建订单时：
  ```python
  window_s = random.uniform(gen_config["window_min"] * 60, gen_config["window_max"] * 60)
  deadline = current_time + window_s
  ```

- `tick(current_time: float, entity_mgr: EntityManager)`:
  每个物理步进周期调用一次。依据 `_gen_config.arrival_rate` 与 `current_time` 比较 `_next_order_time`，决定是否生成新订单；若开启 burst，在 burst 窗口内提升生成频率。

  **配送目的地生成**：使用 `_bbox` 在地图范围内均匀随机采样经纬度，调用 `wgs84_to_utm` 转换为 `Position3D`：
  ```python
  from utils.coord_utils import wgs84_to_utm
  lon = random.uniform(_bbox["min_lng"], _bbox["max_lng"])
  lat = random.uniform(_bbox["min_lat"], _bbox["max_lat"])
  x, y = wgs84_to_utm(lon, lat)
  delivery_loc = Position3D(x=x, y=y, z=0.0)
  ```

  **pickup_source_id 不在此处指定**：订单创建时 `pickup_source_id=None, source_type=None`，由后续调度算法决定从哪个仓库或卡车出货（需配合修改 `Order.__init__` 使两字段变为 `Optional`，详见第 7 节）。

- `update_timeouts(current_time: float)`:
  将超出 `deadline` 的 `pending_orders` 标记为 `TIMEOUT`。

- `get_status_summary() -> dict`:
  返回四项计数，用于大屏统计展示。**v4.5 修正**：键名必须使用 `orders_` 前缀，与 TICK 帧 JSON 和前端 `SimStats` 接口对齐：
  ```python
  return {
      "orders_pending":   len(self.pending_orders),
      "orders_assigned":  len(self.assigned_orders),
      "orders_completed": sum(1 for o in self.completed_orders if o.status == TaskStatus.COMPLETED),
      "orders_timeout":   sum(1 for o in self.completed_orders if o.status == TaskStatus.TIMEOUT),
  }
  ```

### 2.3 `SimulationEngine` (`sim_engine.py`)

**作用**：统筹仿真时钟、物理步进与广播节奏。

> **v1 修正**：v1 描述"每秒执行一次"模糊了物理步进频率与广播频率的关系，导致两者绑死。现将二者完全解耦。

**状态变量**：
```python
current_time: float    # 仿真累计时间（秒）
is_running:   bool
speed_ratio:  float    # 仿真加速倍率（对应前端 systemStore.speedRatio）

_physics_interval_s: float = 0.1   # 后台线程真实睡眠间隔（100ms）
# 每次唤醒后推进的仿真时长 = 0.1 × speed_ratio
```

**主循环（后台线程，真实 100ms 间隔）**：
```
每 100ms（wall-clock 真实时间）唤醒：
  1. 推进仿真时间：current_time += 0.1 × speed_ratio
  2. EntityManager.tick_all(current_time, dt = 0.1 × speed_ratio)
  3. OrderManager.tick(current_time, entity_mgr)
  4. OrderManager.update_timeouts(current_time)
  5. 广播（固定每次唤醒都广播，即固定 10fps）：
       telemetry = EntityManager.get_telemetry()
       summary   = OrderManager.get_status_summary()
       ws_broadcast(TICK 帧)
```

> `speed_ratio` 只影响每次唤醒推进的仿真时长，不影响广播帧率，彻底解耦物理精度与网络带宽。

---

## 3. 标准 API 层接入方案 (`backend/api/routes/simulation_bp.py`)

| 接口 | 方法 | 功能描述 | 前端调用时机 |
|:---|:---:|:---|:---|
| `/api/sim/init` | `POST` | 接收实体清单（depots/stations/trucks/drones）+ 场景 scene_id + 订单生成参数（`order_gen_config`）。后端重置 `EntityManager`、`OrderManager`，准备就绪后返回实体汇总。 | 仿真配置页点击【▶ 启动】时 |
| `/api/sim/control` | `POST` | 控制仿真启停与速率：`{"action": "start"|"pause"|"reset", "speed": 1.0}`。对 `SimEngine.speed_ratio` 生效。 | 前端调整仿真倍速或暂停时 |
| `/api/sim/state` | `GET` | 返回后端当前完整快照：仿真是否在跑、`current_time`、所有实体静态元数据 + 运行时状态、订单统计。**专为 F5 刷新 / 重新连接场景设计。** | 前端页面挂载时探针调用 |
| `/api/sim/orders` | `GET` | 返回最近 N 条订单详情（`?limit=100`），供 `OrderTask` 页面列表展示。 | 订单管理页面挂载或分页时 |

> **v1 变更说明**：
> - 删除了 `POST /api/sim/orders`（订单注入改为后端自驱，不再由前端推入）。
> - `POST /api/sim/init` 的请求体新增 `order_gen_config` 字段，完整对应前端 `OrderGeneratorConfig` 所有字段（含 Phase 2 扩展的 6 个字段）。
> - 新增 `GET /api/sim/state` 供重连恢复使用。
> - 新增 `GET /api/sim/orders` 供订单列表查询。
>
> **v4 新增**：`POST /api/sim/init` 请求体新增 `bbox` 字段（前端从已加载地图中读取地理边界，随实体清单一起上传），OrderManager 缓存后用于生成配送目的地坐标，不依赖 scene_service 内存缓存。

### `/api/sim/init` 请求体结构

```json
{
  "scene_id": "xxxx",
  "bbox": {
    "min_lng": 121.0, "min_lat": 31.0,
    "max_lng": 121.5, "max_lat": 31.5
  },
  "entities": {
    "depots": [
      { "depot_id": "D-1", "name": "仓库-01", "lng": 121.1, "lat": 31.2,
        "altitude": 0, "capacity": 500, "swap_time": 120, "parking_slots": 4 }
    ],
    "stations": [
      { "station_id": "S-1", "name": "换电站-01", "lng": 121.2, "lat": 31.3,
        "altitude": 0, "swap_time": 90, "parking_slots": 2 }
    ],
    "trucks": [
      { "truck_id": "T-1", "name": "货车-01", "speed": 10, "max_inventory": 20,
        "swap_time": 60, "parking_slots": 2, "home_depot_id": "D-1" }
    ],
    "drones": [
      { "drone_id": "UAV-1", "drone_type": "LightDrone", "home_id": "D-1", "home_type": "DEPOT" }
    ]
  },
  "order_gen_config": {
    "arrival_rate": 4, "weight_min": 0.5, "weight_max": 5.0,
    "window_min": 20, "window_max": 60,
    "priority_urgent": 20, "priority_normal": 60, "priority_low": 20,
    "max_orders": null, "geo_mode": "uniform", "cluster_radius_km": 1.5,
    "burst_enabled": false, "burst_multiplier": 3, "burst_duration_s": 60
  }
}
```

---

## 4. WebSocket 遥测流设计 (`backend/api/websockets/telemetry.py`)

**技术选型**：引入 `flask-sock`（原生 WebSocket，无需 SocketIO 的 polling 降级开销）。

**广播频率**：固定每 100ms 一帧（10fps），与 `SimEngine` 主循环同步，**不随 `speed_ratio` 变化**。

**线程安全广播架构**：`flask-sock` 为每个 WebSocket 连接分配独立线程；SimEngine 在后台线程中运行。二者之间**不能直接调用**，必须通过连接注册表中转。实现方式：

```python
# backend/api/websockets/telemetry.py
import threading, json, re
from flask import request

_connections: set  = set()
_conn_lock         = threading.Lock()
_ALLOWED_ORIGIN    = re.compile(r"^http://(localhost|127\.0\.0\.1):517\d$")

@sock.route("/api/ws/telemetry")
def telemetry_ws(ws):
    origin = request.headers.get("Origin", "")
    if not _ALLOWED_ORIGIN.match(origin):
        ws.close(code=1008)   # Policy Violation
        return
    with _conn_lock:
        _connections.add(ws)
    try:
        ws.send(json.dumps(build_full_snapshot()))  # 建连后立即推送首帧
        while True:
            ws.receive()   # 阻塞保持连接；客户端断开时此处抛出异常
    finally:
        with _conn_lock:
            _connections.discard(ws)

def broadcast_tick(payload: dict):
    """由 SimEngine 后台线程调用；线程安全。"""
    dead = set()
    with _conn_lock:
        snapshot = set(_connections)  # 复制一份，减少持锁时间
    for ws in snapshot:
        try:
            ws.send(json.dumps({"type": "TICK", "payload": payload}))
        except Exception:
            dead.add(ws)
    if dead:
        with _conn_lock:
            _connections -= dead
```

`SimEngine` 主循环中将第 5 步改为调用 `broadcast_tick()`，而非直接访问 WebSocket 对象。

### 帧类型 1：`FULL_SNAPSHOT`（建连后立即推送一次）

包含所有实体**完整静态元数据 + 当前运行时状态**，确保前端在 F5 刷新后无需读 localStorage 即可还原完整 UI。

```json
{
  "type": "FULL_SNAPSHOT",
  "payload": {
    "sim_time": 105.0,
    "is_running": true,
    "speed_ratio": 2.0,
    "sim_start_wall_ms": 1711872000000,
    "entities": {
      "depots": [{
        "depot_id": "D-1", "name": "仓库-01", "lng": 121.1, "lat": 31.2,
        "altitude": 0, "capacity": 500, "swap_time": 120, "parking_slots": 4,
        "pending_count": 8, "idle_drone_count": 3, "available_slots": 2, "queue_length": 1
      }],
      "stations": [{
        "station_id": "S-1", "name": "换电站-01", "lng": 121.2, "lat": 31.3,
        "altitude": 0, "swap_time": 90, "parking_slots": 2,
        "available_slots": 1, "queue_length": 0, "serving_drone_ids": ["UAV-3"]
      }],
      "trucks": [{
        "truck_id": "T-1", "name": "货车-01", "speed": 10, "max_inventory": 20,
        "swap_time": 60, "parking_slots": 2, "home_depot_id": "D-1",
        "status": "DRIVING", "lng": 121.15, "lat": 31.25,
        "inventory_count": 5, "docked_drones": [], "available_slots": 2
      }],
      "drones": [{
        "drone_id": "UAV-1", "drone_type": "LightDrone", "home_id": "D-1", "home_type": "DEPOT",
        "status": "FLYING_TO_DELIVER", "lng": 121.13, "lat": 31.22,
        "battery_ratio": 0.73, "carrying_order_id": "ORD-A1B2-1234",
        "cumulative_distance_m": 4200, "remaining_range_m": 18000
      }]
    },
    "stats": { "orders_pending": 45, "orders_assigned": 12, "orders_completed": 87, "orders_timeout": 3 }
  }
}
```

> **v3 字段名对齐说明**：
> - `swap_time` 字段：上方 JSON 示例已展示目标状态（无 `_s` 后缀）。当前后端三个实体类的 `to_telemetry_dict()` 返回的是 `swap_time_s`，需将其改为 `swap_time`（详见第 7 节）。
> - `sim_start_wall_ms`：后端接收到启动指令时的真实服务器时间戳（ms），作为前端展示层换算俯真时间的参考锚点。公式：`DisplayTime = new Date(sim_start_wall_ms + order.deadline_s * 1000)`。
> - `charging_drone_ids`：Depot 的 `to_telemetry_dict()` 实际返回此字段（代替文档示例中未列出的该字段），与充换电站 `serving_drone_ids` 语义不同。

### 帧类型 2：`TICK`（每 100ms 推送）

只推会变化的运行时字段，不重复发送静态元数据，控制帧体积。

> **v4.6 修正**：此前 TICK 示例中每类实体只列出 2–4 个字段（早期草稿保留），与 Section 7.1.2 中 `to_dynamic_state()` 的完整规格不符。由于 `entityStore.setRuntimeAll()` 做**全量数组替换**（非字段合并），TICK 帧中每个实体对象必须包含所有在 TypeScript `***Config` 接口中声明为非 Optional 的字段，否则第一次 TICK 后这些字段变为 `undefined`。

```json
{
  "type": "TICK",
  "payload": {
    "sim_time": 105.1,
    "entities": {
      "drones": [{
        "drone_id": "UAV-1",
        "drone_type": "LightDrone", "home_id": "D-1", "home_type": "DEPOT",
        "lng": 121.132, "lat": 31.221, "altitude": 80.0,
        "status": "FLYING_TO_DELIVER",
        "battery_ratio": 0.72, "carrying_order_id": "ORD-A1B2-1234",
        "cumulative_distance_m": 4350.5, "remaining_range_m": 17649.5
      }],
      "trucks": [{
        "truck_id": "T-1",
        "name": "货车-01", "home_depot_id": "D-1",
        "speed": 10, "max_inventory": 20, "swap_time": 60, "parking_slots": 2,
        "lng": 121.151, "lat": 31.251,
        "status": "DRIVING", "inventory_count": 5,
        "available_slots": 2, "docked_drones": []
      }],
      "stations": [{
        "station_id": "S-1",
        "name": "换电站-01",
        "lng": 121.2, "lat": 31.3, "altitude": 0,
        "swap_time": 90, "parking_slots": 2,
        "available_slots": 1, "queue_length": 1, "serving_drone_ids": ["UAV-3"]
      }],
      "depots": [{
        "depot_id": "D-1",
        "name": "仓库-01",
        "lng": 121.1, "lat": 31.2, "altitude": 0,
        "capacity": 500, "swap_time": 120, "parking_slots": 4,
        "pending_count": 9, "idle_drone_count": 2, "available_slots": 2, "queue_length": 1
      }]
    },
    "stats": { "orders_pending": 46, "orders_assigned": 12, "orders_completed": 87, "orders_timeout": 3 }
  }
}
```

---

## 5. 前端架构适配与退位修改

### 5.1 `systemStore.ts` 的改造

- 页面挂载时调用 `GET /api/sim/state` 探测后端状态：
  - 若后端正在运行（`is_running: true`）：以 `FULL_SNAPSHOT` 中的数据直接初始化 `entityStore` 运行时字段，跳过 `/api/sim/init`，直接进入运行态。
  - 若后端未运行：保持现有本地配置态，等待用户点击启动。
- `start()` 改为调用 `POST /api/sim/control {"action": "start"}`，后端启动 `SimEngine` 后本地 `running` 置 `true`。
- **WsClient 初始化顺序**（`services/websocket.ts` 的 `WsClient` 要求处理器必须在 `connect()` 之前通过 `.on()` 注册）：
  ```typescript
  const ws = new WsClient('ws://localhost:8000/api/ws/telemetry')
  // 必须先注册处理器，再调用 connect()
  ws.on('FULL_SNAPSHOT', handleFullSnapshot)
  ws.on('TICK', handleTick)
  ws.connect()  // 内置 3s 自动重连，断线后后端会重推 FULL_SNAPSHOT
  ```
- **FULL_SNAPSHOT 处理器**（使用 `entityStore.setRuntimeAll()` 整体替换运行时快照——数组不支持 ID 字典访问；3s 自动重连重推时整体覆盖自然幂等）：
  ```typescript
  function handleFullSnapshot(payload) {
    // entityStore 运行时快照名为 rtDrones/rtDepots/... (Drone[])，不是字典
    // 使用现有 setRuntimeAll() 方法整体赋值
    entityStore.setRuntimeAll({
      depots:   payload.entities.depots,
      stations: payload.entities.stations,
      trucks:   payload.entities.trucks,
      drones:   payload.entities.drones,
    })
    // systemStore 需新增 simStartWallMs 字段（见 7.5 修改清单）
    systemStore.simStartWallMs = payload.sim_start_wall_ms
    systemStore.simTime        = payload.sim_time
  }
  ```
- **TICK 处理器补充 simTime 更新**（`simTime = ref(0)` 在当前实现中从未被修改）；TICK 的增量热更新也直接用 `setRuntimeAll()` 替换整个数组，避免在高频 100ms 场景中对数组做 `findIndex` O(n) 查找：
  ```typescript
  function handleTick(payload) {
    systemStore.simTime = payload.sim_time  // 必须更新，否则仿真时钟显示永远为 0
    // setRuntimeAll 仅替换有值的字段，payload.entities 各数组直接传入
    // 注意：to_dynamic_state() 已包含 drone_type/home_id/home_type（v4.2修正），
    //       全量替换后不会丢失静态渲染字段
    entityStore.setRuntimeAll(payload.entities)
    // v4.2修正：orderStore.stats 字段需在 order.ts 中声明后才可赋值
    orderStore.stats = payload.stats
  }
  ```

### 5.2 `orderStore.ts` 的改造

> **v1 修正**：前端不再主动生成并推送订单给后端。

- `generateOnce / startGenerator / stopGenerator` **退化为纯 UI 演示模拟**，保持现有逻辑不变，供前端独立运行时体验交互。
- **正式联调模式**下（`systemStore.running === true` 且后端已连接）：
  - `orderStore` 不再主动生成订单，直接从 WebSocket `TICK` 帧的 `stats` 读取统计展示。
  - 订单详情列表通过 `GET /api/sim/orders?limit=100` 查询。
  - 前端 `OrderGeneratorConfig` 全部参数在点击【▶ 启动】时一次性随 `/api/sim/init` 的 `order_gen_config` 字段传给后端，后端完全接管生成节奏与时序。
  - **时间展示适配**：后端 `Order.create_time` / `deadline` 均为俯真秒（`float`，如 `42.5`），而 demo 模式下前端生成的订单使用 `Date.now()` 毫秒时间戳。为区分两种模式，`GET /api/sim/orders` 返回的订单数据中包含 `time_domain: "sim_s"` 标记，前端展示层按标进行渲染：
    - `time_domain === "sim_s"`：使用 `FULL_SNAPSHOT` 中的 `sim_start_wall_ms` 作为锚点，公式为 `new Date(sim_start_wall_ms + deadline * 1000)`。
    - `time_domain === "wall_ms"`（或未定义）：demo 模式，直接 `new Date(deadline)` 格式化。

### 5.3 地图渲染 `UnifiedMapView.vue` 升级

- 从 `systemStore` 订阅 `TICK` 推送下来的 `entities.drones` / `entities.trucks`。
- 每帧调用 `marker.setLatLng([lat, lng])` 更新 Leaflet Marker 位置。
- 无人机图标根据 `status` 字段动态变色（`FLYING_TO_DELIVER` → 蓝色，`QUEUING` → 黄色，`DEAD` → 红色等）。
- 无需前端插值平滑——10fps 在城市尺度地图精度下足够流畅。

---

## 总结

经修订后的系统完整数据管道：

```
用户在「基础设施」「载具管理」「仿真配置」页面完成配置
    ↓
POST /api/sim/init（实体清单 + 订单生成策略 order_gen_config）
    ↓
EntityManager 实例化 core.entities 实体（含 Z 轴映射）
    +
OrderManager 接收策略参数，准备按仿真时钟自驱生成
    ↓
SimEngine 后台线程启动（物理步进 = 100ms × speed_ratio 仿真时长，广播固定 10fps）
    ↓
WebSocket 首帧 FULL_SNAPSHOT → 前端完整初始化
持续 TICK 帧 → entityStore 热更新 → UnifiedMapView 无人机/卡车动态渲染
    + OrderTask 页面 GET /api/sim/orders 查询订单详情
```

---

## 6. 外部评审意见核查（基于实际代码的客观评估）

> 注：以下评估基于对 `frontend/src/stores/order.ts`、`stores/system.ts`、`utils/orderGen.ts`、`types/index.ts`，以及 `backend/core/entities/primitives.py`、`drone.py`、`depot.py`、`swap_station.py`、`order.py`、`coord_transformer.py` 的逐行阅读，以下结论均以代码现实为准。

### 建议 1：「双时钟问题」—— ✅ 问题真实存在，方案 A 正确，方案 B 不推荐

**代码事实**：`orderStore.ts` 中 `_mainTimerId` 和 `_burstExtraId` 均使用物理时钟 `setInterval`，`systemStore.speedRatio` 完全没有被 `orderStore` 读取。已纳入 v2 修订：选方案 A，前端 `orderStore` 退化为 UI 演示，后端 `OrderManager` 完全接管订单生成。

### 建议 2：「状态恢复盲区」—— ✅ 问题真实，已有部分基础（scene_blueprint 已有 GET 接口）

**代码事实**：`scene_blueprint.py` 已有 `GET /api/scene/<scene_id>` 但只覆盖路网恢复。已纳入 v2 修订：新增 `GET /api/sim/state` + `FULL_SNAPSHOT` 首帧，且明确 `FULL_SNAPSHOT` 必须包含静态元数据（`drone_type`、`home_id`、`name` 等 `DroneConfig` 的全部字段）。

### 建议 3：「Tick 频率与渲染风暴」—— ✅ 问题真实，解耦方案正确

**代码事实**：后端无任何仿真引擎实现，`systemStore.speedRatio` 为死值 `ref(1)`，但问题在引入 SimEngine 后必然出现。已纳入 v2 修订：物理步进与广播频率均固定 100ms，`speed_ratio` 只影响每步推进的仿真时长，不影响广播帧率。前端不做插值。

### 建议 4：「Z 轴丢失」—— ⚠️ 部分正确，严重程度被高估

**代码事实**：`DepotConfig.altitude` 和 `StationConfig.altitude` 在 `types/index.ts` 中已存在，无需新增前端字段。已在 v2 中补充 `EntityManager.load_from_config` 的 Z 轴映射规则，无需其他改动。

| # | 建议 | 评估 | v2 处理 |
|:--|:--|:--|:--|
| 1 | 双时钟问题 | ✅ 真实 | 采用方案 A，OrderManager 后端自驱 |
| 2 | 状态恢复盲区 | ✅ 真实 | 新增 /api/sim/state + FULL_SNAPSHOT 首帧 |
| 3 | Tick 频率解耦 | ✅ 真实 | 固定 100ms 循环，speed_ratio 仅影响仿真时长 |
| 4 | Z 轴丢失 | ⚠️ 被高估 | 补充映射规则，前端 altitude 已存在无需改动 |
---

## 7. v3 新增：代码修改清单（需在实现阶段执行）

以下列出与当前设计方案存在矛盾、**必须在实现阶段修改的现有代码文件**，按优先级排序。

---

### 7.1 `backend/core/entities/` — 字段名统一与轻量序列化方法

#### 7.1.1 `swap_time_s` → `swap_time` 字段名修正

**影响文件**：`depot.py`、`swap_station.py`、`truck.py` 各自的 `to_telemetry_dict()` 方法

**问题**：三个文件均将充换电耗时序列化为 `"swap_time_s"`（含后缀），而前端 `types/index.ts` 定义与设计文档接口示例均使用 `"swap_time"`（无后缀）。前端重连恢复时 `entityStore.swap_time` 将始终为 `undefined`。

**修改方式**：将三个文件 `to_telemetry_dict()` 返回字典中的键名改为 `"swap_time"`：

```python
# depot.py / swap_station.py / truck.py  to_telemetry_dict() 中
# 修改前：
"swap_time_s": self.swap_time,
# 修改后：
"swap_time":   self.swap_time,
```

#### 7.1.2 各实体类新增 `to_dynamic_state()` 轻量序列化方法

**影响文件**：`drone.py`、`truck.py`、`depot.py`、`swap_station.py`

**问题**：当前所有实体只有 `to_telemetry_dict()` 全量序列化方法。每 100ms 的 TICK 帧若调用全量方法，会将 `battery_max_j`、`max_inventory`、`parking_slots` 等静态字段重复发送，浪费带宽。

**说明**：各类分别实现（异构字段不适合基类统一）。**坐标必须通过 `to_wgs84()` 转换**，不能直接暴露 `current_loc.x/y`（那是 UTM 米制坐标，非经纬度）：

```python
# drone.py — 新增方法
# 注意：必须包含 drone_type / home_id / home_type，否则 TICK 全量替换 rtDrones 后
# 地图渲染丢失图标类型（v4.2 修正）
# v4.7 修正：补充 cumulative_distance_m / remaining_range_m，
#   两者均为高频变化字段，TICK 替换后若缺失则前端"累计里程"/"剩余航程"归零
def to_dynamic_state(self) -> dict:
    lon, lat = self.current_loc.to_wgs84()
    return {
        "drone_id":               self.drone_id,
        "drone_type":             self.__class__.__name__,  # 必须保留：地图图标类型
        "home_id":                self.home_id,             # 必须保留：归属关系
        "home_type":              self.home_type.value,     # 必须保留：归属类型
        "lng":                    lon,
        "lat":                    lat,
        "altitude":               self.current_loc.z,
        "status":                 self.status.value,
        "battery_ratio":          round(self.battery_ratio, 4),
        "carrying_order_id":      self.carrying_order_id,
        "cumulative_distance_m":  round(self.cumulative_distance, 2),  # 必须保留（v4.7）
        "remaining_range_m":      round(self.get_remaining_range(), 2),  # 必须保留（v4.7）
    }

# truck.py — 新增方法
# v4.3 修正：补充 TruckConfig 必填字段（speed/max_inventory/swap_time/parking_slots），
# 避免 setRuntimeAll 全量替换后这些字段在 rtTrucks 中变为 undefined
# v4.8 修正：补充 available_slots / docked_drones——两者均为动态字段；
#   available_slots 由 ChargingHost 属性提供（内部已加锁）；
#   FULL_SNAPSHOT 示例包含两者，TICK 若缺失则首帧替换后变 undefined
def to_dynamic_state(self) -> dict:
    lon, lat = self.current_loc.to_wgs84()
    return {
        "truck_id":        self.truck_id,
        # ── 静态 TruckConfig 必填字段（不能被 TICK 替换丢失）────────────────────
        "speed":           self.speed,
        "max_inventory":   self.max_inventory,
        "swap_time":       self.swap_time,
        "parking_slots":   self.parking_slots,
        # name / home_depot_id 由 EntityManager.get_telemetry() 从 _metadata 合并：
        #   {**truck.to_dynamic_state(),
        #    "name": self._metadata[truck_id]["name"],
        #    "home_depot_id": self._metadata[truck_id]["home_depot_id"]}
        # ── 动态字段 ──────────────────────────────────────────────────────────
        "lng":             lon,
        "lat":             lat,
        "status":          self.status.value,
        "inventory_count": len(self.inventory),
        "available_slots": self.available_slots,       # 必须保留（v4.8），ChargingHost.available_slots 内部已加锁
        "docked_drones":   list(self.docked_drones),   # 必须保留（v4.8），防御性拷贝
    }

# depot.py — 新增方法
# v4.3 修正：补充 DepotConfig 必填字段（capacity/swap_time/parking_slots）
# v4.4 修正：补充 lng/lat/altitude 坐标字段——setRuntimeAll 做全量数组替换，
#           DepotConfig.lng/lat/altitude 不是 Optional，TICK 后若缺失则地图标记失位
def to_dynamic_state(self) -> dict:
    lon, lat = self.location.to_wgs84()   # 位置固定，可安全重复调用
    return {
        "depot_id":         self.depot_id,
        # ── 静态 DepotConfig 必填字段（不能被 TICK 替换丢失）──────────────────
        "lng":              lon,           # 必须包含：地图标记定位（v4.4）
        "lat":              lat,           # 必须包含：地图标记定位（v4.4）
        "altitude":         self.location.z,
        "capacity":         self.capacity,
        "swap_time":        self.swap_time,
        "parking_slots":    self.parking_slots,
        # name 由 EntityManager.get_telemetry() 从 _metadata 合并
        # ── 动态字段 ──────────────────────────────────────────────────────────
        "pending_count":    self.pending_count,
        "idle_drone_count": self.idle_drone_count,
        "available_slots":  self.available_slots,
        "queue_length":     self.queue_length,
    }

# swap_station.py — 新增方法
# v4.3 修正：补充 StationConfig 必填字段（swap_time/parking_slots）
# v4.4 修正：补充 lng/lat/altitude——StationConfig.lng/lat/altitude 不是 Optional（同 Depot）
# v4.7 修正：补充 serving_drone_ids——TypeScript Station.serving_drone_ids?: string[] 已声明，
#   FULL_SNAPSHOT 示例包含此字段，omit 后 TICK 全量替换使"正在换电无人机"数据永久丢失
def to_dynamic_state(self) -> dict:
    lon, lat = self.location.to_wgs84()   # 位置固定，可安全重复调用
    with self._lock:                       # 必须加锁——serving_drones 由 tick_update 线程修改
        serving_ids = list(self.serving_drones.keys())
    return {
        "station_id":        self.station_id,
        # ── 静态 StationConfig 必填字段 ───────────────────────────────────────
        "lng":               lon,           # 必须包含：地图标记定位（v4.4）
        "lat":               lat,           # 必须包含：地图标记定位（v4.4）
        "altitude":          self.location.z,
        "swap_time":         self.swap_time,
        "parking_slots":     self.parking_slots,
        # name 由 EntityManager.get_telemetry() 从 _metadata 合并
        # ── 动态字段 ──────────────────────────────────────────────────────────
        "available_slots":   self.available_slots,
        "queue_length":      self.queue_length,
        "serving_drone_ids": serving_ids,   # 必须包含（v4.7）：TICK 替换后否则变 undefined
    }
```

---

### 7.2 `backend/app.py` — CORS 修正与新模块路径注入

#### 7.2.1 CORS 配置修正

**问题一**：当前仅允许 `http://localhost:5173`，未覆盖 `127.0.0.1` 和 Vite 端口自增场景（5174、5175 等）。

**问题二**：`flask-cors` 的 CORS 配置对 WebSocket 握手的 `Origin` 头**无效**，WebSocket 请求需在 `flask-sock` 处理函数中单独校验（Phase 3 主要流量正是 WebSocket）。

```python
# 修改前：
CORS(app, resources={r"/api/*": {
    "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
}})

# 修改后（在文件顶部 import re，然后）：
import re
CORS(app, resources={r"/api/*": {
    "origins": re.compile(r"^http://(localhost|127\.0\.0\.1):517\d$"),
}})
```

WebSocket 握手的 Origin 校验需在 `backend/api/websockets/telemetry.py` 的连接入口处实现：

```python
# backend/api/websockets/telemetry.py
import re
from flask import request

_ALLOWED_ORIGIN = re.compile(r"^http://(localhost|127\.0\.0\.1):517\d$")

@sock.route("/api/ws/telemetry")
def telemetry_ws(ws):
    origin = request.headers.get("Origin", "")
    if not _ALLOWED_ORIGIN.match(origin):
        ws.close(code=1008)   # Policy Violation
        return
    # ... 正常处理逻辑
```

#### 7.2.2 新模块路径注入（最小侵入方案）

**问题**：`environment/state/` 目录未在 `sys.path` 中，`simulation_bp.py` 无法导入 `EntityManager` 等。

**修改方式**：沿用现有 `GEO_DIR` / `SCENE_DIR` 注入模式，仅新增一行：

```python
# backend/app.py — 在现有 GEO_DIR / SCENE_DIR 两行之后追加：
STATE_DIR  = os.path.join(BASE_DIR, "environment", "state")
ROUTES_DIR = os.path.join(BASE_DIR, "api", "routes")
sys.path.insert(0, BASE_DIR)    # 使 order_manager.py 中 from utils.coord_utils import ... 可用
sys.path.insert(0, STATE_DIR)
sys.path.insert(0, ROUTES_DIR)
```

> **不采用全包式导入重构**：全包重构需同步修改 `environment/geo/` 目录内所有平坦导入，会破坏 `environment/geo/app.py` 的独立运行能力，改动范围远超收益。全包重构作为独立技术债任务另行处理。

---

### 7.3 `frontend/src/types/index.ts` — Order 时间语义标记

**问题**：前端 demo 模式下 `Order.create_time` / `deadline` 是 `Date.now()` 毫秒时间戳；后端联调模式是仿真秒（float）。同一字段两种语义，展示层无法区分，会产生 `new Date(42.5)` → 1970-01-01 的渲染错误。

**修改方式**：在 `Order` 接口中新增可选 `time_domain` 字段：

```typescript
// frontend/src/types/index.ts  Order 接口末尾追加：
export interface Order {
  // ... 现有字段（保持不变）...

  /**
   * 时间字段的语义域（决定 create_time / deadline 的单位）
   *  'wall_ms' : 前端 demo 模式，值为 Date.now() 毫秒时间戳
   *  'sim_s'   : 后端联调模式，值为仿真累计秒（float）
   *  未定义时默认按 'wall_ms' 处理，确保向后兼容
   */
  time_domain?: 'wall_ms' | 'sim_s'
}
```

展示层（`OrderTask` 页面及任何格式化 `deadline` 的组件）统一封装以下辅助函数（建议放 `utils/formatOrderTime.ts`）：

```typescript
export function formatDeadline(order: Order, simStartWallMs: number): string {
  if (!order.time_domain || order.time_domain === 'wall_ms') {
    // demo 模式：直接格式化毫秒时间戳
    return new Date(order.deadline).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  // sim_s 模式：用 FULL_SNAPSHOT 中的 sim_start_wall_ms 作为参考锚点
  return new Date(simStartWallMs + order.deadline * 1000)
    .toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}
```

> **暂缓其他 `types/index.ts` 更新**：`battery_current_j`、`current_waypoint_idx` 等 WebSocket 额外字段不触发 TypeScript 结构类型错误（接口是子类型），等 Phase 3 地图渲染组件确认消费后再按需增量添加，避免维护超前于实现的类型定义。

---

---

### 7.3（续）`backend/core/entities/order.py` — pickup_source_id 改为 Optional

**问题**：当前 `Order.__init__` 要求 `pickup_source_id: str` 和 `source_type: SourceType` 为必填参数，但实际取货源由调度算法决定，不应在订单生成时指定。

**修改方式**：

```python
# order.py — __init__ 签名改为 Optional
def __init__(
    self,
    *,
    order_id: str,
    create_time: float,
    deadline: float,
    delivery_loc: Position3D,
    pickup_source_id: Optional[str] = None,    # ← 改为 Optional，默认 None
    source_type: Optional[SourceType] = None,  # ← 改为 Optional，默认 None
    payload_weight: float,
    penalty_rate: float = 1.0,
    assigned_vehicle_id: Optional[str] = None,
    assigned_mode: Optional[str] = None,
) -> None:
    # 注意：移除原来对 pickup_source_id 非空的校验（若有）
    ...
    self.pickup_source_id: Optional[str] = pickup_source_id
    self.source_type: Optional[SourceType] = source_type
```

同时修改 `to_telemetry_dict()` 中的 `source_type` 序列化，防止 `None.value` AttributeError：

```python
# order.py — to_telemetry_dict() 中对应行改为：
"source_type":      self.source_type.value if self.source_type else None,
"pickup_source_id": self.pickup_source_id,  # 可为 None，JSON 序列化为 null
```

调度算法选定仓库/卡车后，直接赋值：
```python
order.pickup_source_id = "D-1"
order.source_type      = SourceType.DEPOT
```

---

### 7.5 `frontend/src/stores/system.ts` 和 `types/index.ts` — 新增 simStartWallMs

**`system.ts` 缺少 `simStartWallMs` 字段**（当前只有 `running/simTime/speedRatio`）：

```typescript
// system.ts — 新增字段
const simStartWallMs = ref(0)   // FULL_SNAPSHOT 中的服务器启动时间戳（ms）

return { running, simTime, speedRatio, simStartWallMs, start, pause, reset }
```

**前端 `Order` 接口需同步将 `pickup_source_id` 和 `source_type` 改为 Optional**（后端改为可返回 null，若前端仍为必填，展示时会显示 "null" 字符串）：

```typescript
// types/index.ts — Order 接口中改为：
pickup_source_id?:  string | null   // 调度算法尚未分配时为 null
source_type?:       HomeType | null
```

---

### 7.6 `frontend/src/stores/order.ts` — 新增 stats 字段（v4.2）

**问题**：TICK 处理器中有 `orderStore.stats = payload.stats`，但 `order.ts` 只有 `generatorConfig / generatedOrders / generatorRunning` 等字段，完全没有 `stats`。对 Pinia composition store Proxy 进行未声明属性的赋值，在 Vue 3 `strict` 模式下会抛出运行时错误，在非严格模式下是 non-reactive 的死赋值（UI 不会响应更新）。

**修改方式**：在 `order.ts` defineStore 内部新增并暴露 `stats`：

```typescript
// order.ts — 在 defineStore 内新增类型与 ref

/** 后端仿真实时统计（来自 TICK 帧，联调模式下使用）
 * v4.3 修正：字段名与 Section 4 TICK/FULL_SNAPSHOT JSON 统一，使用 orders_ 前缀。
 * 后端 OrderManager.get_status_summary() 返回的 key 必须与此一致。
 */
export interface SimStats {
  orders_pending:   number
  orders_assigned:  number
  orders_completed: number
  orders_timeout:   number
}

// 在 useOrderStore 函数体内添加：
const stats = ref<SimStats | null>(null)

// 返回值中补充：
return {
  // ... 现有字段 ...
  stats,
}
```

影响地方：大屏统计组件（如 Dashboard / MainMonitor）改为读取 `orderStore.stats?.orders_pending` 等字段替代本地计算值。

---

### 7.7 `backend/core/entities/primitives.py` — TaskStatus 状态机修正（v4.3）

**问题**：`valid_transitions()` 中 `PENDING` 状态的合法目标只有 `{ASSIGNED, REJECTED}`，不包含 `TIMEOUT`。`OrderManager.update_timeouts()` 需要将逾期的 pending 订单标记为 `TIMEOUT`（超期未被调度），若调用 `order.update_status(TaskStatus.TIMEOUT)` 则直接抛出 `ValueError`。

**修改方式**：在 `valid_transitions()` 中为 `PENDING` 追加 `cls.TIMEOUT`：

```python
# primitives.py — TaskStatus.valid_transitions() 修改
# 修改前：
cls.PENDING:    {cls.ASSIGNED, cls.REJECTED},
# 修改后：
cls.PENDING:    {cls.ASSIGNED, cls.REJECTED, cls.TIMEOUT},
```

**语义说明**：`PENDING → TIMEOUT` 表示"订单在等待调度期间截止时间已过，产生超期惩罚，系统仍可继续尝试分配"。与 `PENDING → REJECTED`（系统主动拒绝）语义不同，不应合并。`update_timeouts()` 实现中只对 `_status == PENDING` 且 `deadline < current_time` 的订单执行此转移；在途订单（`ASSIGNED/PICKED_UP/DELIVERING`）逾期走 `DELIVERING → TIMEOUT` 路径（原始状态机已定义）。

---

### 7.4 修改清单汇总

| 文件 | 修改类型 | 关联矛盾 |
|:---|:---|:---|
| `core/entities/order.py` | `pickup_source_id`/`source_type` 改为 Optional；`to_telemetry_dict()` 加 None 守卫 | v4 新增；v4.1 矛盾3 |
| `core/entities/depot.py` | `swap_time_s`→`swap_time`；新增 `to_dynamic_state()`（含 `lng/lat/altitude` + `capacity/swap_time/parking_slots`） | v3 矛盾 3、4；v4.3 矛盾1；v4.4 矛盾1 |
| `core/entities/swap_station.py` | `swap_time_s`→`swap_time`；新增 `to_dynamic_state()`（含 `lng/lat/altitude` + `swap_time/parking_slots` + `serving_drone_ids`） | v3 矛盾 3、4；v4.3 矛盾1；v4.4 矛盾2；v4.7 矛盾1 |
| `core/entities/truck.py` | `swap_time_s`→`swap_time`；新增 `to_dynamic_state()`（含全部 TruckConfig 静态字段 + `available_slots/docked_drones`） | v3 矛盾 3、4；v4.2/v4.3 矛盾1；v4.8 矛盾1 |
| `core/entities/drone.py` | 新增 `to_dynamic_state()`（含 `drone_type/home_id/home_type` + `cumulative_distance_m/remaining_range_m`） | v3 矛盾 4；v4.2 矛盾1；v4.7 矛盾2 |
| `core/entities/primitives.py` | `TaskStatus.valid_transitions()` 中 PENDING 增加 `cls.TIMEOUT` | v4.3 矛盾2 |
| `environment/state/entity_manager.py` | `get_telemetry()` 须合并 `_metadata`（与 `get_static_snapshot()` 对称）；坐标转换使用 `from utils.coord_utils import wgs84_to_utm` | v4.9 矛盾1；v4.9 矛盾2 |
| `backend/app.py` | CORS 正则化；追加 `BASE_DIR`+`STATE_DIR`+`ROUTES_DIR` sys.path | v3 矛盾 5、6；v4 新增；v4.1 矛盾5 |
| `api/websockets/telemetry.py` | 连接注册表 + `broadcast_tick()` + Origin 校验（新建文件） | v4 新增 |
| `frontend/src/types/index.ts` | `Order` 接口新增 `time_domain?`；`pickup_source_id`/`source_type` 改 Optional | v3 矛盾 2；v4.1 矛盾6 |
| `frontend/src/stores/system.ts` | 新增 `simStartWallMs`；WsClient 调用顺序修正；TICK 处理器 | v4 新增；v4.1 矛盾2 |
| `frontend/src/stores/order.ts` | 新增 `SimStats`（`orders_` 前缀）+ `stats = ref<SimStats\|null>(null)` 并暴露 | v4.2 矛盾2；v4.3 矛盾3 |