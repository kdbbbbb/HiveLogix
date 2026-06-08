# 完整数据链：前端配置 → 发送到后端

> 基于代码直接阅读，覆盖文件：
> `SimulationBox/index.vue` · `stores/system.ts` · `stores/entity.ts` · `stores/order.ts` ·
> `stores/scene.ts` · `simulation_bp.py` · `entity_manager.py` · `order_manager.py` · `sim_engine.py`

---

## 第一段：用户修改配置 → 暂存到 localStorage

所有增删改操作最终都落在 `frontend/src/stores/entity.ts` 的 `_persist()` 函数，
写入浏览器 `localStorage` key `hl-entity-config-v1`。

### 仓库（Depot）
```
页面：frontend/src/views/Infrastructure/index.vue
  新增 → openAddDepot()（第244行）→ 表单提交 → entityStore.addDepot(form)（第256行）
  编辑 → openEditDepot()        → 表单提交 → entityStore.updateDepot(form)（第257行）
  删除 → deleteDepot(id)（第261行） → entityStore.removeDepot(id)

Store：frontend/src/stores/entity.ts
  addDepot(d)    → depots.value.push(d)     → _persist()
  updateDepot(d) → depots.value[i] = d      → _persist()
  removeDepot(id)→ depots.value.filter(...) → _persist()

_persist()（第65行）→ localStorage.setItem('hl-entity-config-v1', JSON.stringify({depots, stations, trucks, drones}))
```

### 充换电站（SwapStation）
```
页面：frontend/src/views/Infrastructure/index.vue
  新增 → openAddStation()（第283行）→ entityStore.addStation(form)（第295行）
  编辑 →                            → entityStore.updateStation(form)（第296行）
  删除 → deleteStation(id)（第300行）→ entityStore.removeStation(id)

Store：entity.ts → addStation / updateStation / removeStation → _persist()
```

### 卡车（Truck）
```
页面：frontend/src/views/FleetManagement/index.vue
  新增 → openAddTruck()（第246行）→ entityStore.addTruck(form)（第258行）
  编辑 →                          → entityStore.updateTruck(form)（第259行）
  删除 → deleteTruck(id)（第263行）→ entityStore.removeTruck(id)

Store：entity.ts → addTruck / updateTruck / removeTruck → _persist()
```

### 无人机（Drone）
```
页面：frontend/src/views/FleetManagement/index.vue
  新增 → openAddDrone()（第281行）→ entityStore.addDrone(form)（第293行）
  编辑 →                          → entityStore.updateDrone(form)（第294行）
  删除 → deleteDrone(id)（第298行）→ entityStore.removeDrone(id)

Store：entity.ts → addDrone / updateDrone / removeDrone → _persist()
```

### 订单生成参数（OrderGeneratorConfig）
```
页面：frontend/src/views/SimulationBox/index.vue → 订单生成器 Tab
  各参数输入 → orderStore.updateConfig(patch)

Store：frontend/src/stores/order.ts
  updateConfig() → generatorConfig.value = {..., ...patch}
                 → localStorage.setItem('hl-order-gen-config-v1', JSON.stringify(config))
```

### 地图 bbox（GeoTool）
```
页面：frontend/src/views/GeoTool/index.vue
  框选地图 → handleExportToDispatch()（第220行）
           → sceneStore.prepareScene({sel_bounds, threshold, height_column})
           → POST /api/scene/prepare   ← 这是唯一实时到后端（geo模块）的操作
           → 返回 scene context（含 road_network.bounds）
           → entityStore.redistributeByBounds(bounds)（第228行）
              → 仓库/充换电站 lng/lat 在内存+localStorage 中重新均匀分配
              → _persist()

注：bbox 此刻只存在于 sceneStore.context（内存）和 localStorage，
    后端 EntityManager 尚未感知。
```

---

## 第二段：点击「🚀 初始化并发送到后端」→ 触发链

### 按钮位置
```
页面：frontend/src/views/SimulationBox/index.vue
  仿真控制台 Tab → <button @click="doInit">（第360行）
```

### doInit()（SimulationBox/index.vue 第481行）
```javascript
doInit() {
  bounds = sceneStore.context?.road_network.bounds   // 从内存取 bbox（若选过地图）
  result = await systemStore.initSim({ bbox: bounds, sceneId: scene_id })
  // 成功后在日志面板显示：✅ 初始化成功 — 仓库×N · 站点×N ...
}
```

### systemStore.initSim()（stores/system.ts 第98行）
```javascript
initSim({ bbox, sceneId }) {
  // 读取 entityStore 内存中的配置（已从 localStorage 加载）
  entities = {
    depots:   entityStore.depots,    // DepotConfig[]
    stations: entityStore.stations,  // StationConfig[]
    trucks:   entityStore.trucks,    // TruckConfig[]
    drones:   entityStore.drones,    // DroneConfig[]
  }
  // 读取 orderStore 的生成参数
  order_gen_config = orderStore.generatorConfig

  // bbox 若无场景则用默认上海范围
  bbox = bbox ?? { min_lng: 121.40, min_lat: 31.10, max_lng: 121.60, max_lat: 31.30 }

  // 发出 HTTP 请求
  return http.post('/api/sim/init', { scene_id, bbox, entities, order_gen_config })
}
```

---

## 第三段：后端接收 → 重建所有对象

### simulation_bp.py → sim_init()（第54行）
```python
def sim_init():
    body = request.get_json()
    entities   = body["entities"]       # 前端传来的4类实体列表
    bbox       = body["bbox"]           # 地图边界
    gen_config = body["order_gen_config"]

    # 1. 重置旧引擎
    _sim_engine.reset()                  # sim_engine.py → reset()，停线程、清时钟

    # 2. 重建三个单例
    _entity_mgr = EntityManager()
    _order_mgr  = OrderManager()
    _sim_engine = SimulationEngine()

    # 3. 分别初始化
    _entity_mgr.load_from_config(body)   # ← 见下方
    _order_mgr.configure(gen_config, bbox)
    _sim_engine.attach(_entity_mgr, _order_mgr)
    set_snapshot_builder(_sim_engine.build_full_snapshot)  # 绑定 WS 快照函数

    return { "status": "initialized", "summary": {depots:N, stations:N, ...} }
```

### entity_manager.py → load_from_config()（第75行）
```python
def load_from_config(config_json):
    entities = config_json["entities"]

    # 步骤1：Depot → wgs84_to_utm(lng, lat) → Position3D → Depot(depot_id, location, swap_time, parking_slots, capacity)
    # 步骤2a：Truck → 取归属 Depot 坐标作为初始位置 → Truck(truck_id, speed, max_inventory, ...)
    # 步骤2b：Drone → 按 home_type 取 Depot/Truck 坐标 → LightDrone 或 HeavyDrone(drone_id, home_id, home_type, init_loc)
    # 步骤3：关联注册
    #   → depot.register_drone(drone_id)      # 无人机归仓库
    #   → truck.docked_drones.append(drone_id) # 无人机归卡车
    #   → depot.register_truck(truck_id)       # 卡车归仓库
    # 步骤4：_metadata sidecar
    #   → { depot_id: {name, type:"DEPOT"}, truck_id: {name, home_depot_id, type:"TRUCK"}, ... }
    #   → TICK 帧输出时与 to_dynamic_state() 合并，确保 name 等字段不丢
```

### order_manager.py → configure()（第67行）
```python
def configure(gen_config, bbox):
    self._gen_config      = gen_config   # 存储生成策略（arrival_rate, weight_min/max, ...）
    self._bbox            = bbox         # 存储地图边界（订单配送目的地在此范围内随机采样）
    self._next_order_time = 0.0
    self._order_seq       = 0
    # 清空所有订单池
```

### sim_engine.py → attach()（第64行）
```python
def attach(entity_mgr, order_mgr):
    self._entity_mgr = entity_mgr   # 挂载实体管理器
    self._order_mgr  = order_mgr    # 挂载订单管理器
    # 此时引擎就绪，等待 start() 调用
```

---

## 第四段：点击「▶ 启动」→ 仿真运行 → 数据回流前端

```
按钮 @click="systemStore.start()"（SimulationBox/index.vue 第364行）
  → POST /api/sim/control { action: "start" }
  → sim_engine.start()（sim_engine.py 第83行）
     → 创建后台线程 _run_loop()（daemon=True）

_run_loop() 每 100ms：
  1. current_time += 0.1 × speed_ratio
  2. entity_mgr.tick_all(current_time, dt)        # 仓库/换电站/卡车队列步进
  3. order_mgr.tick(current_time, entity_mgr)     # 泊松过程生成新订单
     order_mgr.update_timeouts(current_time)      # 超时订单移入 completed
  4. payload = _build_tick_payload()
       → entity_mgr.get_telemetry()               # 每实体调用 to_dynamic_state() + _metadata 合并
       → order_mgr.get_status_summary()           # {orders_pending, orders_assigned, ...}
     broadcast_tick(payload)                      # telemetry.py → 向所有 WS 连接发送

前端 WsClient（websocket.ts）接收：
  type="FULL_SNAPSHOT" → systemStore.handleFullSnapshot()
    → entityStore.setRuntimeAll({depots, stations, trucks, drones})
       → rtDepots / rtStations / rtTrucks / rtDrones 全量替换
    → simTime / running / speedRatio / simStartWallMs 更新

  type="TICK" → systemStore.handleTick()
    → simTime.value = payload.sim_time
    → entityStore.setRuntimeAll(payload.entities)
    → orderStore.stats = payload.stats
```

---

## 汇总

| 阶段 | 数据位置 | 触发方式 |
|---|---|---|
| 用户增删改实体 | `localStorage['hl-entity-config-v1']` + Pinia 内存 | 各配置页面操作 |
| 用户修改订单参数 | `localStorage['hl-order-gen-config-v1']` + Pinia 内存 | 订单生成器 Tab 输入 |
| 用户选地图 | `sceneStore.context`内存（bbox） + 后端 geo 模块 | GeoTool 框选→发布 |
| **发送到后端** | 后端 `EntityManager` / `OrderManager` / `SimulationEngine` 内存 | **手动点「🚀 初始化」** |
| 仿真运行时回流 | 前端 `rtDepots/rtTrucks/rtDrones`（只读） | WebSocket TICK 自动推送 |
