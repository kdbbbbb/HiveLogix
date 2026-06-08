# Phase 2 实体与订单系统增强规划

> 文档日期：2026-03-30（v3，架构重设计版）  
> 基于当前代码状态（`stores/entity.ts` v1、`stores/order.ts` v1）制定

---

## 零、架构全景（修改前现状）

```
frontend/src/
├── types/index.ts            统一类型定义（Config 层 + 运行时层 + 消息层）
├── services/
│   ├── http.ts               封装 fetch，统一 baseURL / 错误处理
│   └── websocket.ts          WsClient 长连接封装（消息分发）
├── stores/
│   ├── entity.ts             实体 CRUD + localStorage 持久化 + 运行时快照
│   ├── order.ts              订单生成器 + 持久化
│   └── scene.ts              场景上下文（bbox / 路网）桥接 GeoTool ↔ Dispatch
├── data/                     【空目录，待建】静态种子数据
├── utils/                    【空目录，待建】纯计算工具函数
└── views/
    ├── GeoTool/              地图选取 → prepareScene → router.push('/dispatch')
    ├── SimulationBox/        仿真配置 tabs（含 GeoToolView + 订单生成器）
    ├── Infrastructure/       仓库 / 充换电站 CRUD
    ├── FleetManagement/      卡车 / 无人机 CRUD
    └── OrderTask/            订单列表 + 筛选 + CSV 导出

backend/
├── environment/scene/
│   ├── scene_blueprint.py    POST /api/scene/prepare → 返回 SceneContext
│   └── scene_service.py      路网打包 + UTM 计算 + MD5 幂等缓存
└── environment/geo/
    └── geo_blueprint.py      /api/geo/query  /api/geo/export  /api/geo/status
```

**现有架构关键特征**：
- 位置坐标计算（UTM、路网 bbox）在**后端**完成，前端只消费 `SceneContext`
- `services/http.ts` 是唯一 HTTP 封装出口，所有 API 调用应经此层
- `stores/scene.ts` 是场景数据的唯一前端存储，`road_network.bounds` 已含完整 bbox
- `types/index.ts` 是类型定义的唯一出口，不得在 store/view 内重复声明接口
- `data/` 与 `utils/` 目录已存在但为空，设计上预留给静态数据与纯函数

---

## 一、现状梳理

### 已完成（Phase 1）

| 模块 | 文件 | 状态 |
|------|------|------|
| 实体类型定义 | `src/types/index.ts` | ✅ 完整，含 Config / 运行时两层 |
| 实体 Pinia Store | `src/stores/entity.ts` | ✅ CRUD + localStorage 持久化 |
| 订单生成 Store | `src/stores/order.ts` | ✅ 参数化生成器 + 持久化 |
| 基础设施页面 | `views/Infrastructure/index.vue` | ✅ 仓库 + 充换电站 CRUD 弹窗 |
| 载具管理页面 | `views/FleetManagement/index.vue` | ✅ 卡车 + 无人机 CRUD 弹窗 |
| 仿真控制页 - 订单 tab | `views/SimulationBox/index.vue` | ✅ 生成器控制 + 参数调节 |
| 订单任务页 | `views/OrderTask/index.vue` | ✅ 列表展示 + 筛选 + CSV 导出 |
| 场景 Store | `stores/scene.ts` | ✅ `sceneContext.road_network.bounds` 含 bbox |
| GeoTool 跳转流程 | `views/GeoTool/index.vue` | ✅ `handleExportToDispatch` → `prepareScene` → `router.push('/dispatch')` |

### 当前核心缺口

1. **首次打开页面实体为空**：`loadConfig()` 读不到 localStorage 时不加载任何默认值
2. **仓库/充换电站无位置自动化**：手动填经纬度，用户不知道坐标；切换仿真地图后坐标不随之更新
3. **地图与实体位置解耦**：仓库/充换电站坐标应绑定仿真地图 bbox，每次重新选区域后应自动重分布
4. **订单超参不完整**：缺少总量上限、地理分布模式、突发模式等控制
5. **默认规模不足**：默认配置应能支撑 ≥10 架无人机、≥50 订单的完整仿真场景

---

## 二、关键背景约束

| 项目 | 要求 |
|------|------|
| 目标仿真区域 | ≈ 4000 m × 4000 m（用户在 GeoTool 内框选） |
| 最小无人机数量 | ≥ 10 架 |
| 充换电站数量 | ≥ 10 个 |
| 持续处理订单 | ≥ 50 个随机动态订单 |
| 实体命名 | 使用通用编号（仓库-01、换电站-01），**不使用任何真实地名** |
| 位置来源 | 仓库与充换电站坐标**必须来自地图 bbox**，无地图时不自动分配坐标 |
| 计算原则 | 坐标计算（bbox、分布）在**前端轻量工具函数**中完成；若后续需路网对齐，迁移至后端 API |

---

## 三、架构分层设计

本次新增涉及 **4 个职责层**，每层分工明确，互不越界：

```
┌───────────────────────────────────────────────────────┐
│  Layer 4  Views（展示 + 用户交互，不含业务逻辑）        │
│  Infrastructure / FleetManagement / SimulationBox       │
└──────────────────────────┬────────────────────────────┘
                           │ 调用 Store actions
┌──────────────────────────▼────────────────────────────┐
│  Layer 3  Stores（状态管理 + 持久化 + 跨视图协调）      │
│  entity.ts  /  order.ts  /  scene.ts                   │
└──────────────────────────┬────────────────────────────┘
                           │ 使用工具函数 / 读取种子数据
┌──────────────────────────▼────────────────────────────┐
│  Layer 2  Utils & Data（纯函数 + 静态数据，无副作用）   │
│  utils/geoLayout.ts  /  utils/orderGen.ts              │
│  data/defaultEntities.ts                               │
└──────────────────────────┬────────────────────────────┘
                           │ 通过 services/http.ts 调用
┌──────────────────────────▼────────────────────────────┐
│  Layer 1  Services（HTTP / WS 封装，无状态）            │
│  services/http.ts  /  services/websocket.ts            │
└───────────────────────────────────────────────────────┘
```

**分层约束**：
- View 只能调用 Store 的 action，不直接操作 localStorage、不调用 http
- Store 不做 DOM 操作，不直接 import View 组件
- Utils 是纯函数（输入 → 输出），不 import Store / View，便于独立测试
- Data 是只读静态对象，不包含任何响应式状态或函数逻辑

---

## 四、文件职责清单（新增 / 修改）

### 新增文件

#### `src/data/defaultEntities.ts` — 静态种子数据

**职责**：唯一存储默认实体参数，不含任何逻辑。  
**规模**（满足约束）：

```
2 个仓库     capacity: 500  parking_slots: 4  swap_time: 90  lng/lat: 0（待布局）
10 个换电站  parking_slots: 2  swap_time: 60  lng/lat: 0（待布局）
3 辆卡车     speed: 15/12  max_inventory: 30/40  parking_slots: 3
12 架无人机  LightDrone × 9（卡车各1 + 仓库各3）  HeavyDrone × 3（仓库各1~2）
```

命名格式：`仓库-01`、`换电站-01`、`卡车-01`、无 ID（由 `genEntityId` 生成）。

**可扩展性**：后续可在此文件增加多套 `preset`（小场景/大场景/压测场景），Store 按名加载。

---

#### `src/utils/geoLayout.ts` — 坐标布局纯函数

**职责**：给定数量 N 和 bbox，输出 N 个均匀分布的 WGS84 坐标，不含任何状态。

```typescript
export interface Bbox {
  min_lng: number; max_lng: number
  min_lat: number; max_lat: number
}

// 唯一导出函数：均匀栅格 + ±15% 随机扰动
export function gridLayout(n: number, bbox: Bbox): Array<{ lng: number; lat: number }>
```

算法：`rows = ceil(sqrt(n))`，`cols = ceil(n/rows)`，格子中心 + rand(±15% 格宽/高)，越界截断。

**可扩展性**：函数签名包含可选 `options` 参数（`jitter`、`seed`），后续可添加 k-means 模式而不破坏调用方。

---

#### `src/utils/orderGen.ts` — 订单生成纯函数 **（从 store 中拆出）**

**职责**：将 `order.ts` 中的 `generateOrder()` 逻辑抽离为纯函数，Store 只做调度。

```typescript
export function buildOrder(
  config: OrderGeneratorConfig,
  warehousePool: WarehouseEntry[],
  sceneBbox?: Bbox          // geo_mode 需要
): Order
```

**为何拆出**：当前 `order.ts` 函数体（帮助函数 `offsetCoord`、`pickWeighted`、`priorityLabel` 等）已嵌在 Store 文件内，随超参扩展会持续膨胀。拆到 `utils/orderGen.ts` 后，Store 只负责持久化和定时器调度，纯逻辑可单独测试。

---

### 修改文件

#### `src/types/index.ts` — 类型扩展

新增到 `OrderGeneratorConfig` 接口（6 个字段）：

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `max_orders` | `number \| null` | 总量上限，null = 无限 | `null` |
| `geo_mode` | `'uniform' \| 'clustered' \| 'depot_nearby'` | 收货点分布模式 | `'uniform'` |
| `cluster_radius_km` | `number` | 热点半径（km） | `1.5` |
| `burst_enabled` | `boolean` | 突发密集模式 | `false` |
| `burst_multiplier` | `number` | 突发期倍率 | `3` |
| `burst_duration_s` | `number` | 每次突发持续（秒） | `60` |

新增辅助类型（供 Utils 使用，不重复定义）：

```typescript
export interface WarehouseEntry { id: string; name: string; lng: number; lat: number }
export type GeoMode = 'uniform' | 'clustered' | 'depot_nearby'
```

---

#### `src/stores/entity.ts` — 默认注入 + 地图联动

**新增内容**（不破坏现有接口）：

1. **`loadConfig()` 扩展**：读取 localStorage → 为空时调用 `injectDefaults(defaultEntities)`
2. **`redistributeByBounds(bounds: SceneBounds)`**：新增 action，接收 bbox，调用 `gridLayout` 分别为 depots / stations 写入 `lng/lat`，最后 `_persist()`
3. **`hasUnlocated` computed**：`depots` 或 `stations` 中存在 `lng === 0` 时为 true，供 View 显示提示

**保持不变**：所有 CRUD action、`setRuntimeAll`、`depotOptions`、`truckOptions`，签名零改动。

---

#### `src/stores/order.ts` — 生成逻辑瘦身 + 超参扩展

**修改内容**：

1. 将 `generateOrder()` 内部逻辑迁出至 `utils/orderGen.ts` 的 `buildOrder()`，Store 内仅保留调用：
   ```typescript
   function generateOnce() {
     if (maxReached()) return               // max_orders 检查
     const order = buildOrder(generatorConfig.value, warehousePool.value, sceneBbox.value)
     addGeneratedOrder(order)
   }
   ```
2. **`burst` 定时器**：在 `startGenerator()` 内，若 `burst_enabled` 为 true，用独立 `setBurstTimer()` 交替切换到达率，与主定时器并行，`stopGenerator()` 时一并清理
3. **`sceneBbox` computed**：从 `useSceneStore` 读取 `context?.road_network?.bounds ?? null`，传给 `buildOrder` 以支持 `geo_mode='uniform'`/`'clustered'` 的 bbox 约束
4. 新增 `DEFAULT_CONFIG` 字段补全（对应 6 个新 type 字段）

**保持不变**：`startGenerator`/`stopGenerator`/`clearOrders`/`addOrder`/`completeOrder` 等公开接口签名。

---

#### `src/views/GeoTool/index.vue` — 触发重布局

**修改内容**：仅在 `handleExportToDispatch()` 函数内，`prepareScene()` 成功后、`router.push` 之前，插入一行调用：

```typescript
if (!sceneStore.error && sceneStore.context) {
  entityStore.redistributeByBounds(sceneStore.context.road_network.bounds)
}
router.push('/dispatch')
```

**不改动**：函数其余逻辑、模板、其他事件处理函数。

---

#### `src/views/Infrastructure/index.vue` — 未定位提示 + 资源假设标注

**新增 UI 元素**（模板层，不动 script 业务逻辑）：

1. 面板顶部：`v-if="entityStore.hasUnlocated"` 的提示横幅
   ```
   📍 仓库/换电站坐标待分配 · 请前往「仿真与配置」→「地图选取」选取仿真范围
   ```
2. 列表行坐标列：`lng === 0` 时显示 `⚠️ 待定位` 黄色徽章，否则显示坐标数字
3. 面板底部说明框（现有 `.infra-note`）：补充资源假设说明

---

#### `src/views/SimulationBox/index.vue` — 订单超参扩展

**新增 UI 元素**（订单 tab 内，折叠展示）：

```
▼ 收货点分布
  地理模式  ○ bbox 均匀  ○ 热点聚集  ○ 仓库覆盖圈
  热点半径（km）[仅非 uniform 时可见，slider 0.5~5]

▼ 突发控制
  □ 启用突发模式
  突发倍率  [slider 1~10，默认 3]
  突发持续  [input 30~300 秒]

▼ 总量控制
  □ 设定上限
  最大订单数  [input，不勾选时置灰]
```

v-model 绑定 `orderStore.generatorConfig` 对应字段，通过现有 `updateField()` 写入。

---

## 五、完整文件变更清单

| 文件 | 操作 | 修改范围 | 影响现有代码 |
|------|------|----------|-------------|
| `src/types/index.ts` | 修改 | `OrderGeneratorConfig` +6 字段；新增 `WarehouseEntry`、`GeoMode` | 零破坏（仅追加） |
| `src/data/defaultEntities.ts` | **新建** | 静态种子数据 | 无 |
| `src/utils/geoLayout.ts` | **新建** | `gridLayout()` 纯函数 | 无 |
| `src/utils/orderGen.ts` | **新建** | `buildOrder()` 纯函数（从 order.ts 迁出） | 修改 order.ts 调用处 |
| `src/stores/entity.ts` | 修改 | `loadConfig` 扩展；新增 `redistributeByBounds`、`hasUnlocated` | 零破坏（仅追加） |
| `src/stores/order.ts` | 修改 | `generateOnce` 改调 `buildOrder`；`burst` 定时器；新增 `sceneBbox` | 公开接口不变 |
| `src/views/GeoTool/index.vue` | 修改 | `handleExportToDispatch` 内插入 1 行调用 | 逻辑极小改动 |
| `src/views/Infrastructure/index.vue` | 修改 | 提示横幅 + 坐标列状态 + 说明框 | 仅模板追加 |
| `src/views/SimulationBox/index.vue` | 修改 | 订单 tab 新增 3 个折叠参数组 | 仅模板追加 |

---

## 六、数据流（修订版）

```
首次加载
  App.vue onMounted
    └─ entityStore.loadConfig()
         ├─ localStorage 有数据 → 直接恢复
         └─ localStorage 为空  → injectDefaults(defaultEntities)
              → 2 仓库 + 10 换电站 + 3 卡车 + 12 无人机（lng=0 占位）
              → _persist()

地图选取 → 实体坐标重布局
  GeoTool：用户框选 ≈ 4000m × 4000m
    └─ handleExportToDispatch()
         ├─ sceneStore.prepareScene()          // 后端: 路网 + bbox
         ├─ entityStore.redistributeByBounds(  // 前端: gridLayout 分配坐标
         │       sceneStore.context.road_network.bounds)
         │    ├─ geoLayout(2 depots, bbox)  → 写 lng/lat
         │    ├─ geoLayout(10 stations, bbox) → 写 lng/lat
         │    └─ _persist()
         └─ router.push('/dispatch')

订单生成
  orderStore.generateOnce()
    └─ buildOrder(config, warehousePool, sceneBbox)   // utils/orderGen.ts
         ├─ geo_mode='uniform'       → bbox 内均匀随机
         ├─ geo_mode='clustered'     → bbox 内热点 + 正态
         └─ geo_mode='depot_nearby'  → 仓库坐标 + radius 偏移
    └─ max_orders 检查 → addGeneratedOrder

突发模式（order.ts 内部）
  startGenerator()
    ├─ 主定时器：arrival_rate 间隔
    └─ burst_enabled → setBurstTimer()
           每隔 burst_duration_s 切换 × burst_multiplier 倍率

重新选图
  handleExportToDispatch() 再次触发
    └─ redistributeByBounds(新 bbox) 覆盖旧坐标 → _persist()
```

---

## 七、实施顺序

```
Step 1  src/types/index.ts             追加 6 字段 + 2 辅助类型（独立，无依赖）
Step 2  src/data/defaultEntities.ts    静态种子数据（独立，无依赖）
Step 3  src/utils/geoLayout.ts         gridLayout 纯函数（独立，无依赖）
Step 4  src/utils/orderGen.ts          buildOrder 纯函数（依赖 Step 1）
Step 5  src/stores/entity.ts           接入 Step 2、3（injectDefaults + redistributeByBounds）
Step 6  src/stores/order.ts            接入 Step 4（buildOrder + burst + max_orders）
Step 7  src/views/GeoTool/index.vue    插入 redistributeByBounds 调用（依赖 Step 5）
Step 8  src/views/Infrastructure/      未定位提示 + 资源假设标注（依赖 Step 5）
Step 9  src/views/SimulationBox/       订单超参折叠 UI（依赖 Step 6）
Step 10 全流程验证
          清空 localStorage → 刷新 → 确认 12 架无人机 + 10 换电站
          → GeoTool 框选 → 前往指挥中心 → Infrastructure 确认坐标更新
          → 订单生成 50+ 笔，geo_mode 切换验证
          → 重新选图 → 坐标再次更新
```

---

## 八、扩展预留说明

| 扩展场景 | 预留点 |
|----------|--------|
| 后端路网吸附（Phase 3）| `redistributeByBounds` 改为调用后端 `/api/scene/layout`，接口签名不变 |
| 多套默认场景预设 | `defaultEntities.ts` 按 `preset` 键导出多套，Store 按参数加载 |
| 地图点击选点（Phase 3）| `gridLayout` 产出坐标已可被单条 `updateDepot` 覆盖，无需额外改造 |
| 自定义订单分布函数 | `buildOrder` 接受可选 `geoStrategy` 函数参数，Strategy 模式 |
| 订单超参同步后端 | `orderStore.updateConfig` 改为同时 POST `/api/order/config`，流程不变 |
| 卡车/无人机自动布局 | 在 `redistributeByBounds` 内按仓库坐标分配卡车 `home_depot_id`，已留接口位置 |

---

## 九、约束与边界说明

- **不引入路网吸附**：`gridLayout` 不依赖 OSM，Phase 3 再做
- **不实现地图点击选点**：坐标由自动布局写入，用户可通过列表表单微调
- **重新选图必然覆盖坐标**：保证实体始终在当前仿真地图范围内
- **无地图时跳过布局**：`lng=0 lat=0` 为未定位标识，View 显示提示
- **resources 永远充足**：`capacity`/`parking_slots` 字段仅供调度算法参考
- **突发模式纯前端**：`burst` 由前端 interval 倍率控制，不涉及后端协议
- **utils 函数零副作用**：`geoLayout.ts` / `orderGen.ts` 可直接用于单元测试
