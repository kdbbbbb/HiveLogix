# 前端实体 CRUD 与订单生成器实现文档

> 完成日期：2026-03-29  
> 涉及模块：`frontend/src/types/`、`frontend/src/stores/`、`frontend/src/views/`

---

## 一、设计原则

### 实体分类

| 分类 | 实体 | 管理方式 |
|------|------|----------|
| 基础设施（用户手动配置） | 仓库（Depot）、充换电站（Station） | 前端表单 CRUD，localStorage 持久化 |
| 载具（用户手动配置） | 卡车（Truck）、无人机（Drone） | 前端表单 CRUD，localStorage 持久化 |
| 订单 | Order | **算法动态生成**，用户可调整生成参数 |
| 运行时遥测 | Depot / Station / Truck / Drone 扩展字段 | WebSocket 推送，只读，不持久化 |

### 两层类型架构

```
XxxConfig   ← 用户可编辑的静态配置（localStorage）
Xxx         ← 继承 XxxConfig，追加运行时遥测字段（WebSocket）
```

---

## 二、修改文件清单

### 1. `frontend/src/types/index.ts`

新增/重写的类型定义：

| 类型 | 说明 |
|------|------|
| `DepotConfig` | 仓库静态配置：depot_id、name、lng/lat/altitude、capacity、swap_time、parking_slots |
| `StationConfig` | 充换电站静态配置：station_id、name、lng/lat/altitude、swap_time、parking_slots |
| `TruckConfig` | 卡车静态配置：truck_id、name、speed、max_inventory、swap_time、parking_slots、home_depot_id |
| `DroneConfig` | 无人机静态配置：drone_id、drone_type（LightDrone/HeavyDrone）、home_type（DEPOT/TRUCK）、home_id |
| `Depot/Station/Truck/Drone` | 扩展对应 Config，含运行时遥测字段（状态、电量、队列等） |
| `Order` | 订单完整字段，新增 `priority`、`warehouse_name`、`deadline_iso`、`priority_label`、`fulfillment_mode` 展示辅助字段 |
| `OrderGeneratorConfig` | 生成器参数：arrival_rate、weight_min/max、window_min/max、priority_urgent/normal/low |
| 枚举 | `TaskStatus`、`DroneStatusType`、`TruckStatusType`、`DroneType`、`HomeType`、`FulfillmentMode` |

---

### 2. `frontend/src/stores/entity.ts`

**完整重写**，从 20 行空壳扩展为 ~165 行完整 CRUD 仓库。

**功能：**
- `depots / stations / trucks / drones`：静态配置（`ref<XxxConfig[]>`），localStorage 键 `hl-entity-config-v1`
- `rtDepots / rtStations / rtTrucks / rtDrones`：运行时快照，由 `setRuntimeAll(data)` 写入
- CRUD 操作：`addXxx(item)` / `updateXxx(item)` / `removeXxx(id)`，每次操作后自动调用 `_persist()`
- `loadConfig()`：从 localStorage 恢复，在 `App.vue` onMounted 时调用
- 导出工具函数 `genEntityId(prefix)`：生成格式为 `DEP-4X2AB-3721` 的唯一 ID
- 计算属性：`droneCount`、`truckCount`、`lightCount`（轻型数）、`heavyCount`（重型数）、`depotOptions`（供 select 绑定）、`truckOptions`

---

### 3. `frontend/src/stores/order.ts`

**完整重写**，从 25 行空壳扩展为 ~200 行完整订单生成器仓库。

**功能：**

| 功能 | 说明 |
|------|------|
| `generatorConfig` | 响应式，localStorage 持久化（键 `hl-order-gen-config-v1`） |
| `generatedOrders` | 生成的订单列表（最多 500 条），localStorage 持久化（键 `hl-orders-v1`） |
| `startGenerator()` | 按 `arrival_rate`（单/分钟）设置定时器，自动生成订单 |
| `stopGenerator()` | 清除定时器 |
| `generateOnce()` | 单次生成一笔订单 |
| `clearOrders()` | 清空订单并删除 localStorage 记录 |
| `updateConfig(patch)` | 更新参数并持久化 |
| `restartIfRunning()` | 参数变更后若生成器运行中则重启，保证频率即时生效 |
| 仓库池 `warehousePool` | 优先使用 entityStore 中用户配置的 Depot；若无，回退到 3 个默认上海仓库 |
| 计算属性 | `pendingOrders`、`activeOrders`、`finishedOrders`、`totalCount`、`recentOrders`（最新 100 条倒序） |

**订单生成算法：**
1. 从仓库池随机选取来源仓库
2. 按 `weight_min ~ weight_max` 均匀随机生成货物重量
3. 按 `priority_urgent / priority_normal / priority_low` 权重随机选取优先级
4. 按 `window_min ~ window_max` 随机生成时间窗，计算 deadline
5. 在仓库坐标 5km 范围内随机生成配送目的地（角度随机偏移）
6. 根据重量和优先级设定 `fulfillment_mode`

---

### 4. `frontend/src/views/Infrastructure/index.vue`

**完整重写**。

**功能：**
- 仓库（Depot）面板：数据行展示 name、ID、坐标、容量、泊位数、换电耗时；支持新增/编辑/删除
- 充换电站（Station）面板：数据行展示对应字段；支持新增/编辑/删除
- 弹窗：`<Teleport to="body">` + `position: fixed` 遮罩，点击遮罩关闭
- 表单校验：名称必填、经纬度范围（lng: -180~180，lat: -90~90）、泊位/换电时间正整数，错误内联
- 新增时自动用 `genEntityId` 生成 ID，编辑时保留原 ID

---

### 5. `frontend/src/views/FleetManagement/index.vue`

**完整重写**。

**功能：**
- 顶部统计胶囊：卡车总数、无人机总数、轻型架数、重型架数（来自 entityStore 计算属性）
- 卡车（Truck）面板：数据行展示 name、ID、速度、载货量、归属仓库；CRUD
- 无人机（Drone）面板：数据行展示 ID、机型 badge（轻型蓝/重型橙）、归属类型、归属 ID；CRUD
- 无人机表单联动：切换 `home_type`（DEPOT/TRUCK）后，`home_id` 下拉选项自动切换为 `depotOptions` 或 `truckOptions`
- 机型信息提示框：选择机型后展示额定载荷、速度、电池参数

---

### 6. `frontend/src/views/SimulationBox/index.vue`

**仅修改 Tab "订单生成器"**（原为静态占位，现为完整功能页）。

**Tab badge 变化：**
- 生成器停止时：`READY`（绿色）
- 生成器运行时：`LIVE`（绿色 + 脉冲动画）

**控制卡片：**

| 控件 | 说明 |
|------|------|
| `arrival_rate` 滑条 | 范围 0.5~30 单/分钟，拖动后若生成器运行则立即重启应用新频率 |
| 重量范围 | min/max 双输入框（kg） |
| 时间窗范围 | min/max 双输入框（分钟） |
| 优先级权重 | 紧急/普通/低 三个带颜色指示点的数字输入框，总和不强制归一化，内部加权随机采样 |
| 启动 / 停止 / 单次生成 / 清空 | 操作按钮组 |

**状态栏：** 运行指示灯（脉冲动画）+ 实时订单统计

**仓库来源标签：** 显示当前池中仓库名称 tag；若使用后备池则显示提示

**实时订单表：** 显示最新 100 条（倒序），含优先级 badge（紧急红/普通蓝/低灰）、状态 badge（待分配/已分配/配送中/已完成/已失败）

---

### 7. `frontend/src/views/OrderTask/index.vue`

**接入真实数据**，修改 script 和 template。

- 状态过滤 select（全部/待分配/已分配/配送中/已完成/已失败）
- 履单模式过滤 select（全部/直送/卡车中继/多跳中继）
- 过滤结果计数展示
- 真实数据行渲染：优先级 badge、状态 badge、坐标、截止时间、履单模式
- **导出 CSV** 功能：将当前过滤结果生成 Blob 文件下载（含 9 列）

---

### 8. `frontend/src/App.vue`

新增 `<script setup>`，在 `onMounted` 中调用 `entityStore.loadConfig()`，确保页面加载时从 localStorage 恢复用户配置的仓库/卡车/无人机数据。

---

## 三、数据流向

```
用户在 Infrastructure / FleetManagement 页面进行 CRUD
        │
        ▼
useEntityStore（Pinia）← localStorage 持久化
        │
        ├──►  useOrderStore.warehousePool（仓库源）
        │
        └──►  FleetManagement 无人机表单 home_id 下拉
                    （depotOptions / truckOptions 计算属性）

用户在 SimulationBox "订单生成器" Tab 调整参数 + 启动
        │
        ▼
useOrderStore 定时生成 Order → generatedOrders
        │
        └──►  OrderTask "订单与任务管理" 页面展示 + 过滤 + 导出
```

---

## 四、localStorage 键说明

| 键名 | 存储内容 |
|------|----------|
| `hl-entity-config-v1` | `{ depots, stations, trucks, drones }` 静态配置 JSON |
| `hl-order-gen-config-v1` | `OrderGeneratorConfig` 生成器参数 JSON |
| `hl-orders-v1` | `Order[]` 最近 500 条已生成订单 JSON |

---

## 五、类型检查

运行 `npx vue-tsc --noEmit` 后，我们修改的所有文件（stores/entity.ts、stores/order.ts、types/index.ts、Infrastructure/index.vue、FleetManagement/index.vue、SimulationBox/index.vue、OrderTask/index.vue）**零类型错误**。

剩余的类型错误均为已有的 pre-existing 问题（`GeoTool/MapView.vue` 中 Leaflet Draw 类型缺失、`services/http.ts` 中 `ImportMeta.env` 类型声明缺失），与本次修改无关。
