# 前端页面重排方案（细化版）

> 严格基于现有代码，所有文件名、函数名、对象名均经核实。
> 状态：方案阶段，未动代码。

---

## 一、核心问题重新定位

### 1.1 前端订单生成器的真实身份

`orderStore.startGenerator()` / `stopGenerator()` 是一个纯客户端的 `setInterval` 循环，
它调用 `buildOrder(generatorConfig, warehousePool, sceneBbox)` 生成 `Order` 对象，
写入 `generatedOrders`（Pinia ref）→ `localStorage['hl-orders-v1']`。
**这些订单从来不传给后端。**

真正驱动仿真的订单生成在后端 `order_manager.py` 的 `tick(current_time, entity_mgr)`，
它由 `sim_engine.py` 的 `_run_loop()` 每 100ms 自动调用一次，
生命周期完全由 `systemStore.start()` → `POST /api/sim/control {action:"start"}` 控制。

### 1.2 原方案的错误

上一版方案将 `startGenerator()` / `stopGenerator()` 保留在 `/orders` 页面，
理由是"用户可以预先验证参数"。这个逻辑存在两个问题：

1. **产生虚假感知**：给用户一个"订单生成器已启动"的运行中状态，
   而这些订单对仿真完全无效，会造成"我明明启动了订单生成，为什么后端没有订单？"的困惑。
2. **两套控制并存**：`/orders` 里 `▶ 启动` / `⏹ 停止`，加上 `/dispatch` 里 `▶ 启动仿真` / `⏸ 暂停`，
   用户需要管两个状态机，而这两个状态机之间没有任何关联。

### 1.3 修正后的认知

| 概念 | 实际含义 | 对用户呈现 |
|---|---|---|
| 前端 `orderStore.startGenerator()` | 纯参数预览工具，生成样例订单供肉眼检查 | 不再有启停按钮，只有"预览生成一笔" |
| 后端 `order_manager.tick()` | 正式订单生成，在 `_run_loop()` 中自动执行 | 由仿真启停控制，结果通过 WebSocket TICK 回显 |
| `orderStore.stats` | 后端推送的四项计数，`SimStats {orders_pending, orders_assigned, orders_completed, orders_timeout}` | 在 `/dispatch` 和 `/orders` 显示实时统计 |

---

## 二、改动目标（最终导航结构）

```
改动前                          改动后
/dashboard  📊 调度性能大屏     → 删除导航入口，内容并入 /dispatch
/dispatch   🗺️ 实时指挥中心    → 扩充：仿真控制栏 + KPI + 地图 + 实体详情
/orders     📦 订单与任务管理   → 扩充：订单参数配置（无启停）+ 后端实时统计 + 表格
/fleet      🛸 载具管理调度     → 不动
/infra      🏗️ 基础设施配置    → 不动
/simulation ⚙️ 仿真与配置       → 缩减为纯地图环境构建，更名「环境构建」
```

**用户操作流程（改动后）**:
```
① 环境构建  →  ② 基础设施配置  →  ③ 载具管理配置  →  ④ 订单参数配置
                                                              ↓
                             ← 点「🚀 初始化」发送全量配置到后端
                                                              ↓
              ⑤ 实时指挥中心：点「▶ 启动仿真」→ 地图/KPI/统计同屏实时呈现
```

---

## 三、`/orders` — 订单与任务管理（`OrderTask/index.vue`）

### 3.1 页面职责重定义

**只做两件事：**
1. **配置订单生成参数**（参数保存在 `orderStore.generatorConfig`，`doInit()` 时传给后端）
2. **查看订单状态**（仿真运行中实时显示 `orderStore.stats`；预览样例可用"生成预览"按钮）

**不再做的事：**
- ~~`startGenerator()` / `stopGenerator()`~~ — 从 UI 完全移除
- ~~`generatorRunning` 状态指示灯~~ — 不再展示

### 3.2 新增内容来源

从 `SimulationBox/index.vue` 的 `orders` Tab（第 31–265 行）迁入以下内容：

| 代码块 | 原位置 | 迁移内容 | 备注 |
|---|---|---|---|
| `<div class="orders-control-card">` 参数表单 | 第 31–265 行 | 全部参数面板（到达率/重量/时间窗/优先级/地理模式/突发控制/总量控制） | 去掉内部 `▶ 启动` 和 `⏹ 停止` 按钮；保留 `单次生成` 按钮，改文案为 `📋 预览一笔` |
| `<div class="warehouse-info">` | 第 267–276 行 | 仓库池来源提示 | 原样迁入 |

**删除不迁移**：`SimulationBox/index.vue` `orders` Tab 内的 `<div class="orders-list-card">`（第 279–317 行，简版订单列表），因 `/orders` 已有功能更完整的表格。

### 3.3 新增后端统计面板（新增组件区块）

位置：参数配置面板下方，表格上方。
数据来源：`orderStore.stats`（类型 `SimStats | null`，由 `systemStore.handleTick()` 通过 WebSocket TICK 帧更新）。

```
┌─────────────────────────────────────────────┐
│  📡 后端订单实况（仿真运行中实时刷新）        │
│  待分配 N  ·  配送中 N  ·  已完成 N  ·  超时 N │
│  v-if="orderStore.stats !== null"           │
│  v-else: "仿真未运行，以下为参数预览数据"     │
└─────────────────────────────────────────────┘
```

字段映射：`orderStore.stats.orders_pending` / `orders_assigned` / `orders_completed` / `orders_timeout`

### 3.4 需要新增到 `<script setup>` 的内容

`OrderTask/index.vue` 现有脚本已导入 `useOrderStore()`，还需补充：

```ts
import { useEntityStore } from '@/stores/entity'
import { useSystemStore  } from '@/stores/system'    // 用于显示 systemStore.running 驱动 stats 提示
import { useSceneStore   } from '@/stores/scene'

const entityStore = useEntityStore()
const systemStore = useSystemStore()
const sceneStore  = useSceneStore()

// 折叠面板（从 SimulationBox 迁入）
const showDistSection  = ref(false)
const showBurstSection = ref(false)
const showLimitSection = ref(false)
```

以下 5 个函数从 `SimulationBox/index.vue`（第 438–468 行）迁入，
**无任何外部依赖，直接复制可用**：

- `updateRate(e: Event)` — 调用 `orderStore.updateConfig({ arrival_rate })` + `orderStore.restartIfRunning()`
- `updateField(key, e: Event)` — 调用 `orderStore.updateConfig({ [key]: v })`
- `clearOrders()` — 调用 `orderStore.clearOrders()`
- `onBurstEnabledChange(e: Event)` — 调用 `orderStore.updateConfig({ burst_enabled })`
- `onMaxOrdersToggle(e: Event)` — 调用 `orderStore.updateConfig({ max_orders })`

> 注意：`orderStore.restartIfRunning()` 在上述函数中仍然保留调用，
> 但由于我们不再在 UI 中暴露启停按钮，`generatorRunning` 始终为 `false`，
> `restartIfRunning()` 只是无操作（短路返回）。不需要删除这个调用，行为无副作用。

### 3.5 页面布局（从上到下）

```
┌─────────────────────────────────────────────────────┐
│  PageShell 头部（现有）                              │
├─────────────────────────────────────────────────────┤
│  ⚙️ 订单生成参数配置面板（从 SimBox 迁入）           │
│    到达率 / 重量 / 时间窗 / 优先级分布               │
│    [📋 预览一笔]  [🗑️ 清空预览]                     │
│    收货点分布（折叠）/ 突发控制（折叠）/ 总量（折叠） │
├─────────────────────────────────────────────────────┤
│  📦 仓库池来源提示（从 SimBox 迁入）                 │
├─────────────────────────────────────────────────────┤
│  📡 后端订单实况（新增区块，来自 orderStore.stats）  │
│    仿真未运行时：灰色提示文字                        │
│    仿真运行中：4 项数字实时刷新                      │
├─────────────────────────────────────────────────────┤
│  过滤工具栏（现有）                                  │
│  订单表格（现有，来自 orderStore.recentOrders）      │
│  派送任务拆解说明（现有）                            │
└─────────────────────────────────────────────────────┘
```

---

## 四、`/dispatch` — 实时指挥中心（`DispatchCenter/index.vue`）

### 4.1 页面职责

**这里是唯一的仿真控制入口 + 可视化结果中心。**
包含：配置验证 → 初始化发送 → 仿真启停 → 地图可视化 → KPI 统计，一屏完成。

### 4.2 新增顶部控制栏（从 SimulationBox 迁入）

从 `SimulationBox/index.vue` 的 `solver` Tab 迁入以下区块：

| 代码块 | 原位置 | 内容 |
|---|---|---|
| `<div class="sc-status">` | 第 312–319 行 | 仿真状态指示（运行中/已初始化/未初始化） |
| `<div class="sc-entity-row">` | 第 321–336 行 | 实体数量摘要（仓库/电站/卡车/无人机各 N 个） |
| `<div class="sc-bbox-hint">` | 第 338–348 行 | 场景 bbox 提示 |
| `<div class="sc-action-row">` | 第 350–362 行 | `[🚀 初始化]` `[▶ 启动]` `[⏸ 暂停]` `[🔄 重置]` |
| `<div class="sc-speed-row">` | 第 364–373 行 | 仿真倍率 ×0.5 / ×1 / ×2 / ×5 / ×10 |
| `<div class="sc-log">` | 第 375–392 行 | 后端响应日志 |

### 4.3 initDone 状态问题的处理

`SimulationBox/index.vue` 中 `initDone` 是一个纯本地 `ref(false)`，
迁移到 `DispatchCenter` 后导航离开再返回会被重置。

**修复方式（不改 Store）**：

```ts
// DispatchCenter/index.vue <script setup>
const initDone = computed(() =>
  systemStore.running || systemStore.simTime > 0
)
```

依据：`systemStore.running`（`ref(false)`）和 `systemStore.simTime`（`ref(0)`）
在应用生命周期内持久，路由切换不会重置（Pinia store 跨路由保持）。
`simTime > 0` 表示曾经初始化并启动过。

### 4.4 主体三栏布局

```
┌──────────────────────────────────────────────────────────────┐
│  顶部控制栏（仿真状态 + 实体摘要 + bbox + 按钮行 + 倍率 + 日志）│
└──────────────────────────────────────────────────────────────┘
┌──────────────────┬────────────────────────┬──────────────────┐
│  左侧面板 240px  │  中央地图（flex: 1）    │  右侧面板 280px  │
│                  │                        │                  │
│  KPI 卡片组      │  <UnifiedMapView>      │  实体详情        │
│  （4 张）        │  （现有组件）          │  （现有）        │
│                  │                        │                  │
│  订单实况统计    │                        │  履约模式构成图  │
│  （stats 4项）   │                        │  （从 Dashboard  │
│                  │                        │   迁入）         │
│  告警中心        │                        │                  │
│  （现有空壳）    │                        │  能量效率图      │
│                  │                        │  （从 Dashboard  │
│  快速决策        │                        │   迁入）         │
│  （现有空壳）    │                        │                  │
│                  │                        │  成本趋势图      │
│                  │                        │  （从 Dashboard  │
│                  │                        │   迁入）         │
└──────────────────┴────────────────────────┴──────────────────┘
```

### 4.5 左侧面板：KPI 卡片组（从 Dashboard 迁入）

来源：`Dashboard/index.vue` 第 12–22 行，`<div class="kpi-row">` 内的 4 个 `kpi-card`。
数据：`kpiList` 数组（现为静态占位 `—`，仿真接入后换成实时数据）。
迁移时将 `kpiList` 常量从 `Dashboard/index.vue` 的 `<script setup>` 搬入 `DispatchCenter/index.vue`。

### 4.6 左侧面板：订单实况统计

数据来源：`orderStore.stats`（`SimStats | null`）。
字段：`orders_pending` / `orders_assigned` / `orders_completed` / `orders_timeout`。
显示逻辑：`v-if="orderStore.stats"` 时显示 4 项数字；否则显示原有空壳文本。
实现方式：改造现有 `<SectionCard title="动态订单队列" icon="📋">` 的内容区。

### 4.7 右侧面板：图表区（从 Dashboard 迁入）

来源：`Dashboard/index.vue` 第 24–91 行，`<div class="chart-grid">` 中的 3 个 `chart-card`。

| 图表 | 来源行 | 说明 |
|---|---|---|
| 履约模式构成 | 第 26–48 行 | `modes` 数组 + `mode-bars` 条形图 |
| 能量与中继效率 | 第 50–68 行 | 热力格占位图 + `heatColor()` 函数 |
| 时效与惩罚成本趋势 | 第 70–91 行 | SVG 模拟折线图 |

迁移时将 `modes` 常量和 `heatColor()` 函数从 `Dashboard/index.vue` 搬入 `DispatchCenter/index.vue`。

### 4.8 新增到 `<script setup>` 的内容

`DispatchCenter/index.vue` 现有脚本只有 `useSceneStore()`，需补充：

```ts
import { computed, ref } from 'vue'
import { useSystemStore } from '@/stores/system'
import { useEntityStore } from '@/stores/entity'
import { useOrderStore  } from '@/stores/order'

const systemStore = useSystemStore()
const entityStore = useEntityStore()
const orderStore  = useOrderStore()

// initDone 从 store 派生，路由切换不丢失
const initDone    = computed(() => systemStore.running || systemStore.simTime > 0)
const initLoading = ref(false)

interface CtrlLog { type: 'info' | 'success' | 'error' | 'warn'; ts: string; msg: string }
const ctrlLogs = ref<CtrlLog[]>([])

// 从 SimulationBox 迁入的函数（第 479–533 行），无需修改：
function _log(type: CtrlLog['type'], msg: string) { ... }
async function doInit() { ... }       // 调用 systemStore.initSim({bbox, sceneId})
async function doReset() { ... }      // 调用 systemStore.reset()
async function doSetSpeed(ratio) { ... } // 调用 systemStore.setSpeed(ratio)

// 从 Dashboard 迁入的数据
const kpiList = [ ... ]    // 4 张 KPI 卡片（目前静态）
const modes   = [ ... ]    // 5 种履约模式
function heatColor(i: number) { ... }
```

---

## 五、`/simulation` — 环境构建（`SimulationBox/index.vue`）

### 5.1 改动内容

删除 `orders` Tab 和 `solver` Tab，**只保留 `geo` Tab**（`<GeoToolView />`）。

具体删除点：
- `tabs` computed 数组中的 `orders` 和 `solver` 两项（第 431–436 行）
- `v-else-if="activeTab === 'orders'"` 整个 DIV（第 27–319 行）
- `v-else-if="activeTab === 'solver'"` 整个 DIV（第 321–392 行）
- Tab 导航栏（`<div class="sim-box__tabs">`）本身——只剩一个 Tab 时不需要 Tab 切换 UI
- `<script setup>` 中删除：`useOrderStore()`、`useSystemStore()`、`activeTab`、
  `showDistSection`、`showBurstSection`、`showLimitSection`、`initLoading`、`initDone`、
  `ctrlLogs`、所有 `update*/clear*/onBurst*/onMaxOrders*/doInit/doReset/doSetSpeed/_log` 函数
- 同步删除对应的大量 `<style scoped>` 中的 `.orders-*`、`.oc-*`、`.sc-*`、`.param-*` 等类

改动后文件只需保留：
- `import GeoToolView from '@/views/GeoTool/index.vue'`
- `<GeoToolView />` 作为根模版
- 少量布局 CSS（或完全依赖 GeoTool 自带）

> 备选：将 `/simulation` 路由直接指向 `GeoTool/index.vue`，删除 `SimulationBox/index.vue` 整个文件。
> 前提：确认 `GeoTool/index.vue` 是否有自己的 `PageShell` 包裹（需检查）。

---

## 六、导航栏与路由（`NavSidebar/index.vue` + `router/index.ts`）

### `NavSidebar/index.vue` — `navItems` 数组（第 43–50 行）

```ts
// 改动前
{ path: '/dashboard',  icon: '📊', label: '调度性能大屏' },     // ← 删除

// 改动后
{ path: '/dispatch',   icon: '🗺️', label: '实时指挥中心' },    // ← 保留
{ path: '/orders',     icon: '📦', label: '订单与任务管理' },   // ← 保留
{ path: '/fleet',      icon: '🛸', label: '载具管理调度' },     // ← 不动
{ path: '/infra',      icon: '🏗️', label: '基础设施配置' },    // ← 不动
{ path: '/simulation', icon: '🗺️', label: '环境构建' },         // ← label 改名，icon 可换
```

`badge: 'LIVE'` 的逻辑：当前 `/simulation` 是硬编码的静态 `'LIVE'`，
改动后可以绑定到 `systemStore.running`，由 `DispatchCenter` 的 LIVE 状态决定 `/dispatch` 的 badge。
但 `NavSidebar` 目前不导入 store，实现 badge 响应式需要新增 store 导入，
可作为后续优化，本次不强制处理。

### `router/index.ts` 改动

```ts
// 原第 11 行
{ path: '/analytics', redirect: '/dashboard' },

// 改为
{ path: '/analytics', redirect: '/dispatch' },

// 新增（原 /dashboard 书签不失效）
{ path: '/dashboard', redirect: '/dispatch' },
```

根路径重定向（第 8 行）`{ path: '/', redirect: '/dashboard' }` 改为 `redirect: '/dispatch'`。

---

## 七、`Dashboard/index.vue` 的处置

内容全部被迁入 `DispatchCenter`，此文件可选处置：
- **保留文件，路由 redirect**（推荐）：`/dashboard` 路由改为 `redirect: '/dispatch'`，文件不删
- **删除文件**：需同时删除路由中 `import('@/views/Dashboard/index.vue')` 的懒加载

推荐保留文件，防止意外路径报错。

---

## 八、改动文件清单与工作量

| 文件 | 操作 | 工作量 |
|---|---|---|
| `views/OrderTask/index.vue` | 顶部插入参数配置面板（约 230 行模板 + 3 个 ref + 5 个函数）；新增 stats 统计区块 | 中 |
| `views/DispatchCenter/index.vue` | 顶部新增控制栏；左侧 KPI + stats；右侧 3 个图表；script 补充 4 个 store + 函数 | 大 |
| `views/SimulationBox/index.vue` | 删除 orders/solver Tab，保留 geo Tab，清理 script | 中（主要是删除） |
| `components/NavSidebar/index.vue` | `navItems` 删除 `/dashboard`，修改 `/simulation` label | 极小 |
| `router/index.ts` | 新增 `/dashboard` redirect，修改 `/analytics` redirect，修改根路径 redirect | 极小 |
| `views/Dashboard/index.vue` | 不删除，路由 redirect 处理 | 无 |
