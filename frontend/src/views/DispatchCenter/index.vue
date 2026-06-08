<template>
  <PageShell
    icon="🗺️"
    title="实时指挥中心"
    desc="仿真控制 · 多维 GIS 可视化 · 实时调度干预 · KPI 统计"
    :badge="systemStore.running || systemStore.trainingRunning ? 'LIVE' : 'READY'"
    :badge-type="systemStore.running || systemStore.trainingRunning ? 'live' : 'info'"
  >
    <div class="dispatch-body">
      <!-- ── 顶部仿真控制栏（迁自 SimulationBox solver tab）── -->
      <div class="dispatch-ctrl">
        <!-- 实时状态 -->
        <div class="sc-status" :class="systemStore.running || systemStore.trainingRunning ? 'sc-status--on' : 'sc-status--off'">
          <span class="sc-dot" :class="systemStore.running || systemStore.trainingRunning ? 'sc-dot--live' : 'sc-dot--idle'"></span>
          <span>{{ systemStore.trainingRunning ? 'PPO 训练直播中' : (systemStore.running ? '仿真运行中' : (initDone ? '已初始化，等待启动' : '未初始化')) }}</span>
          <span class="sc-sep">|</span>
          <span>仿真时钟：<strong>{{ systemStore.simTime.toFixed(2) }} s</strong></span>
          <span class="sc-sep">|</span>
          <span>倍率：× {{ systemStore.speedRatio }}</span>
          <span class="sc-sep">|</span>
          <span>运行模式：<strong>{{ systemStore.trainingActive ? 'PPO 训练可视化' : (systemStore.policyActive ? 'PPO 在线推理' : 'Classic 仿真') }}</strong></span>
        </div>

        <!-- 实体摘要 -->
        <div class="sc-entity-row">
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.depots.length }">
            🏭 仓库 <strong>{{ entityStore.depots.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.stations.length }">
            ⚡ 充换电站 <strong>{{ entityStore.stations.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.trucks.length }">
            🚛 卡车 <strong>{{ entityStore.trucks.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.drones.length }">
            🚁 无人机 <strong>{{ entityStore.drones.length }}</strong>
          </div>
        </div>

        <!-- 控制操作行 -->
        <div class="sc-ctrl-row">
          <!-- 操作按钮 + 速率选择 -->
          <div class="sc-actions-area">
            <!-- bbox 提示 -->
            <div class="sc-bbox-hint">
              <span v-if="sceneStore.context">
                📐 场景 bbox：{{ sceneStore.context.road_network.bounds.min_lng.toFixed(4) }},
                {{ sceneStore.context.road_network.bounds.min_lat.toFixed(4) }}
                → {{ sceneStore.context.road_network.bounds.max_lng.toFixed(4) }},
                {{ sceneStore.context.road_network.bounds.max_lat.toFixed(4) }}
              </span>
              <span v-else class="sc-bbox-hint--warn">⚠️ 未加载仿真场景，将使用默认 bbox（上海）</span>
            </div>

            <!-- 统一控制行 -->
            <div class="sc-action-row">
              <!-- 算法选择 -->
              <button class="sc-btn sc-btn--dispatch sc-btn--algo"
                :class="{ 'sc-btn--dispatch-active': dispatchSolver === 'greedy_mmce_bi' }"
                :disabled="!initDone || dispatchLoading || systemStore.policyActive || systemStore.trainingRunning || systemStore.running"
                @click="dispatchSolver = 'greedy_mmce_bi'">
                贪心插入式算法
              </button>
              <button class="sc-btn sc-btn--dispatch sc-btn--algo"
                :class="{ 'sc-btn--dispatch-active': dispatchSolver === 'ga_mmce' }"
                :disabled="!initDone || dispatchLoading || systemStore.running"
                @click="dispatchSolver = 'ga_mmce'">
                遗传算法
              </button>
              <button class="sc-btn sc-btn--policy sc-btn--algo"
                :class="{ 'sc-btn--policy-active': systemStore.policyActive }"
                :disabled="!initDone || dispatchLoading || policyLoading || systemStore.trainingRunning || (!systemStore.policyActive && !ppoReadyForActivation) || (systemStore.policyActive && systemStore.running)"
                @click="doPpoPolicyButton">
                <span>{{ ppoPolicyButtonText }}</span>
                <span v-if="policyLoading" class="sc-btn-spinner" aria-hidden="true"></span>
              </button>
              
              <!-- GA 缓存复用 -->
              <label v-if="dispatchSolver === 'ga_mmce'" class="dispatch-cache-toggle">
                <input
                  v-model="reuseGaStaticPlan"
                  type="checkbox"
                  :disabled="dispatchLoading || systemStore.running"
                >
                <span>复用上次静态结果</span>
              </label>
              
              <!-- 信息统计 -->
              <span class="dispatch-solver-tag">
                当前算法：{{ dispatchSolverLabel(dispatchSolver) }}
              </span>
              <span v-if="lastDispatchResult" class="dispatch-quick-stat">
                ✓ {{ lastDispatchResult.plan.feasible }}/{{ lastDispatchResult.plan.total_orders }} 可行
              </span>
              <span v-if="systemStore.policyActive" class="dispatch-quick-stat dispatch-quick-stat--policy">
                PPO 模式下调度已禁用
              </span>
              <span v-if="systemStore.trainingRunning" class="dispatch-quick-stat dispatch-quick-stat--policy">
                PPO 训练直播中调度已禁用
              </span>

              <button class="sc-btn sc-btn--init" :disabled="initLoading || systemStore.trainingRunning" @click="doInit">
                {{ initLoading ? '⏳ 初始化中...' : '🚀 初始化并发送到后端' }}
              </button>
              <button class="sc-btn sc-btn--start"
                :disabled="!initDone || systemStore.running || systemStore.trainingRunning || dispatchLoading"
                @click="doStartWithDispatch">{{ dispatchLoading ? '⏳ 调度中...' : '▶ 启动' }}</button>
              <button class="sc-btn sc-btn--pause"
                :disabled="!systemStore.running"
                @click="systemStore.pause()">⏸ 暂停</button>
              <button class="sc-btn sc-btn--reset" :disabled="systemStore.trainingRunning" @click="doReset">🔄 重置</button>
          
              <!-- 预设保存 -->
              <button class="sc-btn sc-btn--export"
                :disabled="savingPreset || systemStore.running"
                @click="doSavePreset"
                title="将当前调整的实体配置和任务点保存到预设文件">
                {{ savingPreset ? '⏳ 保存中...' : '💾 保存预设' }}
              </button>
            </div>

            <div class="sc-speed-row">
              <span class="sc-speed-label">仿真倍率</span>
              <div class="sc-speed-btns">
                <button v-for="s in [0.5, 1, 2, 5, 10]" :key="s"
                  class="sc-spd" :class="{ 'sc-spd--active': systemStore.speedRatio === s }"
                  @click="doSetSpeed(s)">× {{ s }}</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── 三栏主体 ── -->
      <div class="dispatch-main">
        <!-- 左侧面板 -->
        <aside class="dispatch-aside">
          <div class="dispatch-aside-scroll">
            <!-- KPI 卡片组（迁自 Dashboard） -->
            <div class="kpi-grid">
              <div v-for="k in kpiList" :key="k.label" class="kpi-card">
                <div class="kpi-card__icon">{{ k.icon }}</div>
                <div class="kpi-card__value">{{ k.value }}</div>
                <div class="kpi-card__label">{{ k.label }}</div>
                <div class="kpi-card__trend" :class="k.up ? 'trend--up' : 'trend--down'">
                  {{ k.up ? '↑' : '↓' }} {{ k.change }}
                </div>
              </div>
            </div>

            <!-- 动态订单队列（实况统计） -->
            <SectionCard title="动态订单队列" icon="📋" class="dispatch-section dispatch-section--orders">
              <div class="order-panel-scroll">
                <template v-if="orderStore.stats">
                  <div class="order-stats-grid">
                    <div class="order-stat">
                      <span class="order-stat__num">{{ orderStore.stats.orders_pending }}</span>
                      <span class="order-stat__lbl">待分配</span>
                    </div>
                    <div class="order-stat">
                      <span class="order-stat__num order-stat__num--active">{{ orderStore.stats.orders_assigned }}</span>
                      <span class="order-stat__lbl">配送中</span>
                    </div>
                    <div class="order-stat">
                      <span class="order-stat__num order-stat__num--success">{{ orderStore.stats.orders_completed }}</span>
                      <span class="order-stat__lbl">已完成</span>
                    </div>
                    <div class="order-stat">
                      <span class="order-stat__num order-stat__num--danger">{{ orderStore.stats.orders_timeout }}</span>
                      <span class="order-stat__lbl">超时</span>
                    </div>
                  </div>
                </template>
                <div v-if="dynamicOrderQueue.length" class="dynamic-order-list">
                  <div
                    v-for="order in dynamicOrderQueue"
                    :key="order.order_id"
                    class="dynamic-order-row"
                    :class="dynamicOrderStatusClass(order.status)"
                  >
                    <div class="dynamic-order-row__main">
                      <span class="dynamic-order-row__id">{{ order.order_id }}</span>
                      <span class="dynamic-order-row__badge">{{ dynamicOrderStatusText(order.status) }}</span>
                    </div>
                    <div class="dynamic-order-row__meta">
                      <span>注入 {{ formatSimClock(order.spawn_sim_s) }}</span>
                      <span>截止 {{ formatSimClock(order.deadline_sim_s) }}</span>
                    </div>
                    <div class="dynamic-order-row__meta">
                      <span>{{ formatWeight(order.payload_weight) }}</span>
                      <span>{{ order.priority_label || order.priority || '普通' }}</span>
                    </div>
                  </div>
                </div>
                <div v-else class="empty-list">
                  <span>⏳ 等待仿真数据接入</span>
                </div>
              </div>
            </SectionCard>

            <SectionCard title="告警中心" icon="🔔" class="dispatch-section dispatch-section--alerts">
              <div v-if="deadlineAlerts.length" class="alert-list">
                <div
                  v-for="alert in deadlineAlerts"
                  :key="alert.order_id"
                  class="alert-row"
                  :class="alert.severity === 'overdue' ? 'alert-row--danger' : 'alert-row--warn'"
                >
                  <div class="alert-row__main">
                    <span class="alert-row__id">{{ alert.order_id }}</span>
                    <span class="alert-row__badge">{{ alert.sourceLabel }}</span>
                  </div>
                  <div class="alert-row__meta">
                    <span>{{ alert.remainingLabel }}</span>
                    <span>截止 {{ alert.deadlineLabel }}</span>
                  </div>
                </div>
              </div>
              <div v-else class="empty-list">
                <div class="alert-badge alert-badge--ok">✅ 无活跃告警</div>
              </div>
            </SectionCard>
          </div>
        </aside>

        <!-- 中央地图区 -->
        <div class="dispatch-map">
          <UnifiedMapView v-if="sceneStore.context" ref="mapRef" />
          <div v-else class="map-placeholder">
            <div class="map-placeholder__icon">🌐</div>
            <p class="map-placeholder__title">尚未加载仿真场景</p>
            <p class="map-placeholder__desc">
              请前往「环境构建」完成建筑分析后<br />
              点击「<strong>导出到指挥中心地图</strong>」以建立仿真底图。
            </p>
            <router-link to="/simulation" class="go-btn">前往环境构建 →</router-link>
          </div>
        </div>

        <!-- 右侧动态流程面板 -->
        <aside class="dispatch-detail">
          <DynamicFlowPanel ref="flowPanelRef" />
        </aside>
      </div>
    </div>
  </PageShell>
</template>

<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import PageShell     from '@/components/PageShell/index.vue'
import SectionCard   from './components/SectionCard.vue'
import UnifiedMapView from './components/UnifiedMapView.vue'
import DynamicFlowPanel from './components/DynamicFlowPanel.vue'
import { useSceneStore }  from '@/stores/scene'
import { useSystemStore } from '@/stores/system'
import { useEntityStore } from '@/stores/entity'
import { useOrderStore }  from '@/stores/order'
import type { DispatchPlan, DispatchTruckRoute, DispatchDroneRoute, Order } from '@/types'

const sceneStore  = useSceneStore()
const systemStore = useSystemStore()
const entityStore = useEntityStore()
const orderStore  = useOrderStore()

const mapRef = ref<InstanceType<typeof UnifiedMapView> | null>(null)
const flowPanelRef = ref<InstanceType<typeof DynamicFlowPanel> | null>(null)

// initDone 从 store 派生，路由切换不丢失 (改用 initialized 标志)
const initDone    = computed(() => systemStore.initialized)
const initLoading = ref(false)
type PolicyOrderSourceMode = 'benchmark' | 'poisson' | 'hybrid'
const PPO_SCENE_ID = 'default_test_4x4km'
const policyLoading = ref(false)
const policyPath = ref('config/policy_best.pt')
const policyConfigPath = ref('config/rh_alns_cmrappo.yaml')
const policySceneId = ref(PPO_SCENE_ID)
const policyDeterministic = ref(true)
const policyUseCurrentInitPayload = ref(true)
const policyOrderSourceMode = ref<PolicyOrderSourceMode>('benchmark')
const currentSceneId = computed(() => String(sceneStore.context?.scene_id ?? '').trim())
const ppoCurrentSceneCompatible = computed(() => currentSceneId.value === PPO_SCENE_ID)
const ppoInitSceneCompatible = computed(() => String(systemStore.initializedSceneId ?? '').trim() === PPO_SCENE_ID)
const ppoReadyForActivation = computed(() =>
  systemStore.initialized &&
  ppoCurrentSceneCompatible.value &&
  ppoInitSceneCompatible.value
)
const ppoCompatibilityMessage = computed(() => {
  if (!sceneStore.context) {
    return `请先加载默认预设场景 ${PPO_SCENE_ID}。`
  }
  if (!ppoCurrentSceneCompatible.value) {
    return `当前页面场景为 ${currentSceneId.value}，现有 PPO 权重仅支持 ${PPO_SCENE_ID}。`
  }
  if (!systemStore.initialized) {
    return `请先使用 ${PPO_SCENE_ID} 完成一次初始化。`
  }
  if (!ppoInitSceneCompatible.value) {
    return `最近一次初始化场景为 ${systemStore.initializedSceneId || 'unknown'}，请切回 ${PPO_SCENE_ID} 后重新初始化。`
  }
  return `当前页面与最近一次初始化均匹配 ${PPO_SCENE_ID}，可以激活 PPO。`
})
const ppoPolicyButtonText = computed(() => {
  if (policyLoading.value) return 'PPO 算法'
  if (systemStore.policyActive) return systemStore.running ? 'PPO 在线运行中' : '▶ 启动 PPO 在线策略'
  return 'PPO 算法'
})

// 调度相关状态
const dispatchLoading = ref(false)
const lastDispatchResult = ref<any>(null)
const dispatchPlan = ref<DispatchPlan | null>(null)
const totalEnergyCostWh = ref(0)
// 本地 UI 游标：避免把同一条 PPO 决策事件重复写入 DynamicFlowPanel。
const lastRenderedDecisionEventSeq = ref(0)
type DispatchSolverName = 'greedy_mmce_bi' | 'ga_mmce'
const dispatchSolver = ref<DispatchSolverName>('greedy_mmce_bi')
const reuseGaStaticPlan = ref(false)

function dispatchSolverLabel(solver: DispatchSolverName): string {
  if (solver === 'greedy_mmce_bi') return '贪心（增量）'
  return '遗传算法'
}

// 预设保存状态
const savingPreset = ref(false)

interface CtrlLog { type: 'info' | 'success' | 'error' | 'warn'; ts: string; msg: string }
const ctrlLogs = ref<CtrlLog[]>([])

// ── 迁自 Dashboard ──────────────────────────────────────────────────
function toSimSeconds(value: number | undefined, timeDomain?: 'wall_ms' | 'sim_s') {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return timeDomain === 'wall_ms' ? value / 1000 : value
}

const DEADLINE_ALERT_WINDOW_SEC = 10 * 60
type DynamicOrderUiStatus = Order['status'] | 'WAITING' | string

interface DynamicOrderQueueRow {
  order_id: string
  spawn_sim_s: number
  deadline_sim_s: number
  payload_weight: number
  priority?: string
  priority_label?: string
  status: DynamicOrderUiStatus
}

interface DeadlineAlertRow {
  order_id: string
  sourceLabel: string
  deadline_sim_s: number
  deadlineLabel: string
  remainingSec: number
  remainingLabel: string
  severity: 'warn' | 'overdue'
}

function formatSimClock(value: number | undefined | null) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return '--'
  const sign = seconds < 0 ? '-' : ''
  const abs = Math.abs(seconds)
  const h = Math.floor(abs / 3600)
  const m = Math.floor((abs % 3600) / 60)
  const s = Math.floor(abs % 60)
  if (h > 0) return `${sign}${h}时${String(m).padStart(2, '0')}分${String(s).padStart(2, '0')}秒`
  return `${sign}${m}分${String(s).padStart(2, '0')}秒`
}

function formatWeight(value: number | undefined | null) {
  const weight = Number(value)
  return Number.isFinite(weight) ? `${weight.toFixed(2)} kg` : '-- kg'
}

function scheduledDynamicDeadline(order: any) {
  const absoluteDeadline = Number(order?.deadline_sim_s)
  if (Number.isFinite(absoluteDeadline)) return absoluteDeadline
  const spawn = Number(order?.spawn_sim_s ?? 0)
  const offset = Number(order?.deadline_offset_s ?? 900)
  return spawn + offset
}

function dynamicOrderStatusText(status: DynamicOrderUiStatus) {
  const labels: Record<string, string> = {
    WAITING: '待注入',
    PENDING: '待分配',
    ASSIGNED: '已分配',
    PICKED_UP: '已取货',
    DELIVERING: '配送中',
    IN_TRANSIT: '配送中',
    COMPLETED: '已完成',
    TIMEOUT: '已超时',
    REJECTED: '已拒绝',
  }
  return labels[String(status)] || String(status)
}

function dynamicOrderStatusClass(status: DynamicOrderUiStatus) {
  if (status === 'COMPLETED') return 'dynamic-order-row--done'
  if (status === 'TIMEOUT' || status === 'REJECTED') return 'dynamic-order-row--danger'
  if (status === 'WAITING') return 'dynamic-order-row--waiting'
  return 'dynamic-order-row--active'
}

const dynamicOrderIds = computed(() =>
  new Set(orderStore.scheduledDynamicOrders.map(order => String(order.order_id)))
)

const dynamicOrderQueue = computed<DynamicOrderQueueRow[]>(() => {
  const runtimeOrdersById = new Map(
    orderStore.generatedOrders.map(order => [String(order.order_id), order])
  )
  return orderStore.scheduledDynamicOrders
    .map(order => {
      const runtimeOrder = runtimeOrdersById.get(String(order.order_id))
      return {
        order_id: String(order.order_id),
        spawn_sim_s: Number(order.spawn_sim_s ?? runtimeOrder?.create_time ?? 0),
        deadline_sim_s: toSimSeconds(runtimeOrder?.deadline, runtimeOrder?.time_domain)
          ?? scheduledDynamicDeadline(order),
        payload_weight: Number(runtimeOrder?.payload_weight ?? order.payload_weight ?? 0),
        priority: String(runtimeOrder?.priority ?? order.priority ?? ''),
        priority_label: String(runtimeOrder?.priority_label ?? order.priority_label ?? ''),
        status: runtimeOrder?.status ?? 'WAITING',
      }
    })
    .sort((a, b) => a.spawn_sim_s - b.spawn_sim_s)
})

function runtimeOrderRemaining(order: Order) {
  if (order.time_domain === 'wall_ms') {
    return (Number(order.deadline) - Date.now()) / 1000
  }
  return Number(order.deadline) - Number(systemStore.simTime)
}

function runtimeOrderDeadlineLabel(order: Order) {
  if (order.time_domain === 'wall_ms') {
    const deadlineMs = Number(order.deadline)
    if (!Number.isFinite(deadlineMs)) return '--'
    return new Date(deadlineMs).toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }
  return formatSimClock(Number(order.deadline))
}

function remainingLabel(remainingSec: number) {
  if (remainingSec < 0) return `已超时 ${formatSimClock(Math.abs(remainingSec))}`
  return `剩余 ${formatSimClock(remainingSec)}`
}

const deadlineAlerts = computed<DeadlineAlertRow[]>(() => {
  const rows: DeadlineAlertRow[] = []
  const seen = new Set<string>()
  const closedStatuses = new Set(['COMPLETED', 'REJECTED'])

  for (const order of orderStore.generatedOrders) {
    const status = String(order.status)
    if (closedStatuses.has(status)) continue
    const remainingSec = runtimeOrderRemaining(order)
    if (!Number.isFinite(remainingSec)) continue
    if (remainingSec > DEADLINE_ALERT_WINDOW_SEC && status !== 'TIMEOUT') continue
    seen.add(String(order.order_id))
    rows.push({
      order_id: String(order.order_id),
      sourceLabel: dynamicOrderIds.value.has(String(order.order_id)) ? '动态订单' : '静态订单',
      deadline_sim_s: toSimSeconds(order.deadline, order.time_domain) ?? 0,
      deadlineLabel: runtimeOrderDeadlineLabel(order),
      remainingSec,
      remainingLabel: remainingLabel(remainingSec),
      severity: remainingSec < 0 || status === 'TIMEOUT' ? 'overdue' : 'warn',
    })
  }

  for (const order of dynamicOrderQueue.value) {
    if (seen.has(order.order_id) || order.status === 'COMPLETED' || order.status === 'REJECTED') continue
    const remainingSec = order.deadline_sim_s - Number(systemStore.simTime)
    if (remainingSec > DEADLINE_ALERT_WINDOW_SEC && order.status !== 'TIMEOUT') continue
    rows.push({
      order_id: order.order_id,
      sourceLabel: '动态订单',
      deadline_sim_s: order.deadline_sim_s,
      deadlineLabel: formatSimClock(order.deadline_sim_s),
      remainingSec,
      remainingLabel: remainingLabel(remainingSec),
      severity: remainingSec < 0 || order.status === 'TIMEOUT' ? 'overdue' : 'warn',
    })
  }

  return rows
    .sort((a, b) => a.remainingSec - b.remainingSec)
    .slice(0, 8)
})

const kpiList = computed(() => {
  const runtimeEnergyWh = Number(orderStore.stats?.total_energy_cost_wh)
  const energyWhDisplay = Number.isFinite(runtimeEnergyWh) && runtimeEnergyWh >= 0
    ? runtimeEnergyWh
    : totalEnergyCostWh.value

  const orders = orderStore.generatedOrders
  const totalOrders = orders.length
  const completedOrders = orders.filter(o => o.status === 'COMPLETED')
  const completionRate = totalOrders > 0 ? (completedOrders.length / totalOrders) * 100 : 0

  const completedWithTiming = completedOrders.filter(o =>
    typeof o.deadline === 'number' && typeof o.actual_deliver_time === 'number'
  )

  let onTimeCount = 0
  const delaysInMinutes: number[] = []
  for (const order of completedWithTiming) {
    const deadlineSec = toSimSeconds(order.deadline, order.time_domain)
    const deliverSec = toSimSeconds(order.actual_deliver_time, order.time_domain)
    if (deadlineSec === null || deliverSec === null) continue
    if (deliverSec <= deadlineSec) onTimeCount += 1
    delaysInMinutes.push(Math.max(0, deliverSec - deadlineSec) / 60)
  }

  const onTimeRate = completedWithTiming.length > 0
    ? (onTimeCount / completedWithTiming.length) * 100
    : 0

  const avgDelayMin = delaysInMinutes.length > 0
    ? delaysInMinutes.reduce((a, b) => a + b, 0) / delaysInMinutes.length
    : 0

  return [
    {
      icon: '✅',
      label: '综合任务完成率',
      value: `${completionRate.toFixed(1)}%`,
      change: `${completedOrders.length}/${totalOrders || 0}`,
      up: true,
    },
    {
      icon: '⏱️',
      label: '准时送达率',
      value: `${onTimeRate.toFixed(1)}%`,
      change: `${onTimeCount}/${completedWithTiming.length || 0}`,
      up: true,
    },
    {
      icon: '📉',
      label: '平均订单延迟',
      value: `${avgDelayMin.toFixed(2)} min`,
      change: `${completedWithTiming.length || 0} 单`,
      up: false,
    },
    {
      icon: '⚡',
      label: '总体能耗成本',
      value: `${energyWhDisplay.toFixed(2)} Wh`,
      change: '累计',
      up: false,
    },
  ]
})

const modes = [
  { label: '模式A · 卡车直送',   pct: 0, color: '#2563eb' },
  { label: '模式B · 卡车+无人机', pct: 0, color: '#7c3aed' },
  { label: '模式C · 仓库直飞',   pct: 0, color: '#0891b2' },
  { label: '模式D · 空投补货',   pct: 0, color: '#d97706' },
  { label: '模式E · 多跳中继',   pct: 0, color: '#dc2626' },
]

function heatColor(i: number) {
  const hue = 200 + i * 8
  return `hsl(${hue}, 70%, 55%)`
}

// ── 仿真控制函数（迁自 SimulationBox solver tab）─────────────────────
function _log(type: CtrlLog['type'], msg: string) {
  ctrlLogs.value.unshift({ type, ts: new Date().toLocaleTimeString('zh-CN'), msg })
}

async function doInit() {
  initLoading.value = true
  totalEnergyCostWh.value = 0
  _log('info', '正在发送初始化请求到后端...')
  try {
    const bounds = sceneStore.context?.road_network.bounds

    // ⚠️ 修复 v5.0：移除这里的 redistributeByBounds 调用
    // 原因：GeoTool 在选择场景时已经调用过一次，这里再调用会导致每次初始化时
    // 生成不同的随机坐标，造成充电站"瞬移"效果（用户看到坐标从一个位置跳到另一个位置）
    // 
    // 正确的流程应该是：
    //   1. 使用 GeoTool 选择场景 → 自动调用 redistributeByBounds 生成初始坐标
    //   2. 在 DispatchCenter 点击初始化 → 使用已有坐标，不重新生成
    //   3. 若需要重新生成坐标，明确点击"重新分配坐标"按钮（待实现）
    
    if (!bounds) {
      _log('warn', '⚠️ 未加载场景，请先在「地图工具」中选择场景')
      initLoading.value = false
      return
    }

    // 📊 保存发送前的坐标快照（用于诊断和对比）
    const stationSnapshotBefore = entityStore.stations.map(s => ({ 
      id: s.station_id, 
      lng: s.lng, 
      lat: s.lat 
    }))
    const depotSnapshotBefore = entityStore.depots.map(d => ({
      id: d.depot_id,
      lng: d.lng,
      lat: d.lat
    }))
    
    _log('info',
      `📐 使用现有坐标 — 仓库 ${entityStore.depots.length} 个` +
      `· 充换电站 ${entityStore.stations.length} 个`
    )
    console.log('[doInit] 发送前坐标快照：', { depots: depotSnapshotBefore, stations: stationSnapshotBefore })

    const result = await systemStore.initSim({
      bbox:    bounds,
      sceneId: sceneStore.context?.scene_id,
    })
    await systemStore.fetchPolicyState().catch(() => {})
    const s = result.summary
    _log('success', `✅ 初始化成功 — 仓库×${s.depots} · 站点×${s.stations} · 卡车×${s.trucks} · 无人机×${s.drones}`)
    
    // 初始化后等待一小段时间确保 WebSocket FULL_SNAPSHOT 被推送
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // 🔍 收到后端 WebSocket 后，检查坐标是否发生变化
    const stationSnapshotAfter = entityStore.stations.map(s => ({ 
      id: s.station_id, 
      lng: s.lng, 
      lat: s.lat 
    }))
    const depotSnapshotAfter = entityStore.depots.map(d => ({
      id: d.depot_id,
      lng: d.lng,
      lat: d.lat
    }))
    console.log('[doInit] 收到 WebSocket 后坐标快照：', { depots: depotSnapshotAfter, stations: stationSnapshotAfter })
    
    // 检查坐标是否发生了变化
    const stationChanged = stationSnapshotAfter.some((s, i) => {
      const before = stationSnapshotBefore[i]
      return !before || Math.abs(s.lng - before.lng) > 0.00001 || Math.abs(s.lat - before.lat) > 0.00001
    })
    const depotChanged = depotSnapshotAfter.some((d, i) => {
      const before = depotSnapshotBefore[i]
      return !before || Math.abs(d.lng - before.lng) > 0.00001 || Math.abs(d.lat - before.lat) > 0.00001
    })
    
    if (stationChanged || depotChanged) {
      _log('warn', '⚠️ 坐标发生变化（可能由WGS84↔UTM转换精度差异导致）')
      console.warn('[doInit] 坐标变化详情：', { 
        stations: { before: stationSnapshotBefore, after: stationSnapshotAfter },
        depots: { before: depotSnapshotBefore, after: depotSnapshotAfter }
      })
    } else {
      _log('info', '✓ 坐标验证通过，发送和接收的坐标一致')
    }
    
    if (DEBUG_VEHICLE_UPDATES) {
      console.log('[doInit] 初始化完成，当前卡车数:', entityStore.rtTrucks.length)
      _log('info', `📍 前端已接收 ${entityStore.rtTrucks.length} 辆卡车`)
    }
  } catch (e: any) {
    systemStore.initialized = false  // 初始化失败时重置标志
    _log('error', `❌ 初始化失败：${e.message}`)
  } finally {
    initLoading.value = false
  }
}

async function doReset() {
  await systemStore.reset().catch(() => {})
  lastRenderedDecisionEventSeq.value = 0
  totalEnergyCostWh.value = 0
  _log('warn', '🔄 仿真已重置，请重新初始化')
}

async function doSetSpeed(ratio: number) {
  await systemStore.setSpeed(ratio)
  _log('info', `⚡ 速率已设为 ×${ratio}`)
}

async function doActivatePolicy() {
  if (!ppoReadyForActivation.value) {
    _log('warn', `⚠️ ${ppoCompatibilityMessage.value}`)
    return
  }
  policyLoading.value = true
  lastRenderedDecisionEventSeq.value = 0
  _log('info', `🤖 正在激活 PPO 在线策略：${policyPath.value}`)
  try {
    await systemStore.activatePolicy({
      policy_name: 'rh_alns_cmrappo',
      policy_path: policyPath.value.trim(),
      config_path: policyConfigPath.value.trim(),
      scene_id: policySceneId.value.trim(),
      deterministic: policyDeterministic.value,
      speed_ratio: Number(systemStore.speedRatio) || 1,
      order_source_mode: policyOrderSourceMode.value,
      use_current_init_payload: policyUseCurrentInitPayload.value,
    })
    await systemStore.fetchPolicyState()
    _log('success', `✅ PPO 已激活：${systemStore.policyName || 'rh_alns_cmrappo'}，点击「启动」即可执行权重推理`)
  } catch (e: any) {
    _log('error', `❌ PPO 激活失败：${e.message}`)
  } finally {
    policyLoading.value = false
  }
}

async function doPpoPolicyButton() {
  if (!systemStore.policyActive) {
    await doActivatePolicy()
    return
  }
  if (systemStore.running) return
  try {
    await systemStore.start()
    _log('success', '▶ PPO 在线推理已启动')
  } catch (e: any) {
    _log('error', `❌ PPO 启动失败：${e.message}`)
  }
}

/**
 * 启动前自动执行调度，然后启动仿真
 */
async function doStartWithDispatch() {
  if (systemStore.running || systemStore.trainingRunning || dispatchLoading.value) return
  if (!initDone.value) return
  
  // 如果是 PPO 模式，直接启动（不需要调度）
  if (systemStore.policyActive) {
    try {
      await systemStore.start()
      _log('success', '▶ PPO 在线推理已启动')
    } catch (e: any) {
      _log('error', `❌ PPO 启动失败：${e.message}`)
    }
    return
  }
  
  // Classic 模式：先调度，再启动
  _log('info', `🎯 正在执行${dispatchSolverLabel(dispatchSolver.value)}调度算法...`)
  try {
    // 执行调度
    await doDispatch()
    
    // 调度成功后，自动启动仿真
    if (lastDispatchResult.value && lastDispatchResult.value.status === 'ok') {
      _log('info', '▶ 调度完成，正在启动仿真...')
      await systemStore.start()
      _log('success', '✅ 仿真已启动')
    }
  } catch (e: any) {
    _log('error', `❌ 启动过程出错：${e.message}`)
  }
}

async function doDispatch() {
  dispatchLoading.value = true
  _log('info', `🎯 正在执行${dispatchSolverLabel(dispatchSolver.value)}调度算法...`)
  try {
    // 检查是否有加载场景
    if (!sceneStore.context) {
      _log('error', '❌ 尚未加载场景，请先完成环境构建')
      dispatchLoading.value = false
      return
    }

    const bbox = mapRef.value?.getCurrentBounds?.()
    if (DEBUG_VEHICLE_UPDATES) {
      console.log('[doDispatch] bbox object:', bbox)
      console.log('[doDispatch] bbox keys:', bbox ? Object.keys(bbox) : 'null')
    }
    _log('info', `📍 地图边界: ${bbox ? `(${bbox.minx.toFixed(3)}, ${bbox.miny.toFixed(3)}) - (${bbox.maxx.toFixed(3)}, ${bbox.maxy.toFixed(3)})` : 'null'}`)
    
    if (!bbox || typeof bbox.minx !== 'number') {
      _log('error', '❌ 无法获取地图边界，请检查地图是否正常加载')
      dispatchLoading.value = false
      return
    }
    
    if (DEBUG_VEHICLE_UPDATES) {
      console.log('[doDispatch] 将调用 dispatch，bbox:', bbox)
    }
    const result = await systemStore.dispatch(
      bbox,
      dispatchSolver.value,
      { reuseStaticGaPlan: dispatchSolver.value === 'ga_mmce' && reuseGaStaticPlan.value },
    )
    if (DEBUG_VEHICLE_UPDATES) {
      console.log('[doDispatch] dispatch 返回:', result)
    }
    lastDispatchResult.value = result

    if (result.status === 'ok' && result.plan) {
      const plan = result.plan as DispatchPlan
      const backendSolver = String((plan as any)?.solver || result.runtime_metrics?.active_solver || '')
      if (backendSolver && backendSolver !== dispatchSolver.value) {
        _log('warn', `⚠️ 算法不一致：前端选择 ${dispatchSolver.value}，后端生效 ${backendSolver}`)
      }

      const modeStr = Object.entries(plan.modes || {})
        .map(([k, v]) => `${k}:${v}`)
        .join(' ')

      _log('success',
        `✅ 调度完成 — ${plan.feasible}/${plan.total_orders} 可行 ` +
        `(${modeStr ? '模式分布: ' + modeStr : '暂无分配'}) · ` +
        `待派 ${result.pending_count} | 派送中 ${result.assigned_count}`
      )
      if (plan.static_cache_reused) {
        _log('info', '♻️ 已复用上一次 GA 静态调度结果')
      }

      dispatchPlan.value = plan

      const runtimeEnergyWh = Number(result.runtime_metrics?.total_energy_cost_wh)
      if (Number.isFinite(runtimeEnergyWh) && runtimeEnergyWh >= 0) {
        totalEnergyCostWh.value = runtimeEnergyWh
      }

      mapRef.value?.clearDispatchRoutes?.()
      mapRef.value?.drawRuntimePaths?.({ trucks: [], drones: [] })
      
      // 调试：打印无人机路线信息便于诊断显示问题
      if (plan.drone_routes && plan.drone_routes.length > 0) {
        console.group('🚁 无人机路线详情')
        for (const route of plan.drone_routes) {
          console.log(`📍 订单 ${route.order_id}:`, {
            drone_id: route.drone_id,
            mode: route.mode,
            recovery_station_id: route.recovery_station_id,
            path: route.path,
          })
        }
        console.groupEnd()
      }

      if (dispatchSolver.value === 'ga_mmce') {
        mapRef.value?.drawCurrentDroneRoutes?.(plan, Number(result.timestamp ?? 0))
      } else {
        mapRef.value?.drawDispatchRoutes?.(plan)
      }
      
      // 清除之前的动态实体标记，为仿真做准备
      mapRef.value?.clearDynamicEntities?.()
      
      flowPanelRef.value?.addEvent?.('🎯', `调度完成 ${plan.feasible}/${plan.total_orders} 订单可行`)
    } else {
      _log('error', `❌ 调度失败：${result.error || '未知错误'}`)
    }
  } catch (e: any) {
    _log('error', `❌ 调度出错：${e.message}`)
  } finally {
    dispatchLoading.value = false
  }
}

/**
 * 保存调整后的预设场景到后端磁盘文件
 */
async function doSavePreset() {
  savingPreset.value = true
  _log('info', '💾 正在保存预设场景...')
  try {
    // 构建要保存的数据
    const payload = {
      entities: {
        depots: entityStore.depots,
        stations: entityStore.stations,
        trucks: entityStore.trucks,
        drones: entityStore.drones,
      },
      orders: {
        static_orders: orderStore.generatedOrders,
        dynamic_orders: orderStore.scheduledDynamicOrders,
      },
    }

    const response = await fetch('/api/sim/preset/entities/default_test_4x4km', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (response.ok) {
      const result = await response.json()
      _log('success', `✅ 预设场景已保存: ${result.message}`)
      flowPanelRef.value?.addEvent?.('💾', '调整的实体配置和任务点已保存到预设文件')
    } else {
      const error = await response.json()
      _log('error', `❌ 保存失败: ${error.error}`)
    }
  } catch (e: any) {
    _log('error', `❌ 保存出错：${e.message}`)
  } finally {
    savingPreset.value = false
  }
}

// ── WebSocket 实时位置更新 ────────────────────────────────────────────

// ── 调试开关 ────────────────────────────────────────────────────────
const DEBUG_VEHICLE_UPDATES = false

function setupRealtimeUpdates() {
  // 首先初始化 WebSocket 连接
  systemStore.initWs()
  
  // 监听 TICK 事件，实时更新卡车、无人机和订单状态
  let tickCount = 0
  systemStore.onTick(() => {
    tickCount++
    if (DEBUG_VEHICLE_UPDATES && tickCount <= 3) {
      console.log(`[setupRealtimeUpdates] TICK 回调被触发 (${tickCount})，当前卡车: ${entityStore.rtTrucks.length}`)
      if (entityStore.rtTrucks.length > 0) {
        const truck = entityStore.rtTrucks[0]
        console.log(`  - 第一辆卡车: ${truck.truck_id}, 坐标: (${truck.lng}, ${truck.lat})`)
      }
    }
    
    // 更新卡车位置（使用运行时数据 rtTrucks）
    for (const truck of entityStore.rtTrucks) {
      if (truck.lng && truck.lat) {
        if (DEBUG_VEHICLE_UPDATES) {
          console.log(`[setupRealtimeUpdates] 更新卡车 ${truck.truck_id} 到 (${truck.lng.toFixed(5)}, ${truck.lat.toFixed(5)})`)
        }
        mapRef.value?.updateTruck?.(truck.truck_id, truck.lng, truck.lat, truck.status)
      } else if (DEBUG_VEHICLE_UPDATES && tickCount <= 1) {
        console.warn(`[setupRealtimeUpdates] 卡车 ${truck.truck_id} 坐标无效: lng=${truck.lng}, lat=${truck.lat}`)
      }
    }
    
    // 更新无人机位置（使用运行时数据 rtDrones）
    for (const drone of entityStore.rtDrones) {
      if (drone.lng && drone.lat) {
        mapRef.value?.updateDrone?.(drone.drone_id, drone.lng, drone.lat, drone.status)
      }
    }

    const runtimePathsForMap = dispatchSolver.value === 'ga_mmce'
      ? { trucks: [], drones: systemStore.runtimePaths.drones ?? [] }
      : systemStore.runtimePaths
    const hasRuntimePaths = dispatchSolver.value === 'ga_mmce'
      ? (runtimePathsForMap.drones?.length ?? 0) > 0
      : ((runtimePathsForMap.trucks?.length ?? 0) > 0 || (runtimePathsForMap.drones?.length ?? 0) > 0)
    if (hasRuntimePaths || systemStore.running || systemStore.policyActive) {
      mapRef.value?.drawRuntimePaths?.(runtimePathsForMap)
    }

    for (const event of systemStore.decisionEvents) {
      if (event.event_seq <= lastRenderedDecisionEventSeq.value) continue
      const mode = event.selected_mode ? ` ${event.selected_mode}` : ''
      const order = event.selected_order_id ? ` / ${event.selected_order_id}` : ''
      const recover = event.selected_recover_node
        ? ` / 回收点 ${event.selected_recover_node}`
        : event.recovery_selection_stage === 'pending_post_delivery_selection'
          ? ' / 回收点送达后选择'
          : ''
      const latency = event.inference_latency_ms != null ? ` · ${event.inference_latency_ms}ms` : ''
      if (event.status === 'DECISION_PENDING') {
        flowPanelRef.value?.addEvent?.('⏳', `PPO 决策待处理 #${event.decision_id} · ${event.drone_id}`)
      } else if (event.status === 'DECISION_APPLIED') {
        flowPanelRef.value?.addEvent?.('🤖', `PPO 决策已应用 #${event.decision_id}${mode}${order}${recover}${latency}`)
      } else if (event.status === 'EXECUTION_HARD_FAILED') {
        flowPanelRef.value?.addEvent?.('⚠️', `PPO 执行硬失败 #${event.decision_id} · ${event.failure_type || 'unknown'}`)
      }
      lastRenderedDecisionEventSeq.value = Math.max(lastRenderedDecisionEventSeq.value, event.event_seq)
    }
    
    // 重绘订单标记（使用最新的订单状态）
    mapRef.value?.drawOrders?.()
  })
}

onMounted(() => {
  setupRealtimeUpdates()
  systemStore.fetchPolicyState().catch(() => {})
  
  // 如果已加载预设场景，自动加载预设的实体配置
  if (sceneStore.context?.scene_id) {
    loadPresetEntitiesIfAvailable()
  }
})

/**
 * 尝试从后端加载预设场景的实体配置（仓库、充电站、无人机）
 */
async function loadPresetEntitiesIfAvailable() {
  try {
    // 判断是否是预设场景
    const roadSource = (sceneStore.context?.meta as any)?.road_source
    const presetId = roadSource === 'osm_preset' 
      ? 'default_test_4x4km' 
      : null
    
    if (!presetId) return
    
    const response = await fetch(`/api/sim/preset/entities/${presetId}`)
    if (!response.ok) return
    
    const data = await response.json()
    if (!data || !data.depots) return
    
    // 使用预设的实体配置替换前端配置
    if (data.depots) entityStore.depots = data.depots
    if (data.stations) entityStore.stations = data.stations
    if (data.trucks) entityStore.trucks = data.trucks
    if (data.drones) entityStore.drones = data.drones
    
    // 加载预设的静态任务点（补全所有必需字段）
    if (data.orders && Array.isArray(data.orders)) {
      const now = Date.now()
      orderStore.generatedOrders = data.orders.map((o: any) => {
        const rowSim = o.time_domain === 'sim_s'
        const priority = o.priority || 'NORMAL'
        const priorityLabels: Record<string, string> = {
          'URGENT': '紧急',
          'NORMAL': '普通',
          'LOW': '低优先'
        }
        const dl = o.deadline ?? (rowSim ? 600 : now + 600000)
        const fmtSimDeadline = (sec: number) => {
          const m = Math.floor(sec / 60)
          const s = Math.floor(sec % 60)
          return `仿真+${m}分${String(s).padStart(2, '0')}秒`
        }
        return {
          order_id: o.order_id,
          create_time: o.create_time ?? 0,
          deadline: dl,
          delivery_lng: o.delivery_lng,
          delivery_lat: o.delivery_lat,
          delivery_z: o.delivery_z ?? 0,
          payload_weight: o.payload_weight,
          priority: priority,
          status: 'PENDING',
          source_type: 'DEPOT',
          pickup_source_id: null,
          fulfillment_mode: o.fulfillment_mode ?? (o.payload_weight > 3.5 ? 'DRONE_TRUCK_DEPOT' : 'DRONE_DIRECT'),
          warehouse_name: o.warehouse_name || data.depots[0]?.name || '仓库-中心',
          deadline_iso: rowSim
            ? fmtSimDeadline(Number(dl))
            : new Date(dl).toLocaleTimeString('zh-CN', {
                hour: '2-digit',
                minute: '2-digit',
              }),
          priority_label: o.priority_label || priorityLabels[priority] || '普通',
          time_domain: rowSim ? 'sim_s' : 'wall_ms',
        } as Order
      })
    }
    orderStore.scheduledDynamicOrders = Array.isArray(data.dynamic_orders)
      ? data.dynamic_orders
      : []
  } catch (e) {
    console.warn('[DispatchCenter] 加载预设实体失败:', e)
  }
}

onBeforeUnmount(() => {
  mapRef.value?.clearDynamicEntities?.()
})
</script>

<style scoped>
/* ── 主体：flex 列布局 ─────────────────────────────────────────────── */
.dispatch-body {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: var(--hl-space-md);
  overflow: hidden;
}

/* ── 顶部控制栏 ──────────────────────────────────────────────────── */
.dispatch-ctrl {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-sm);
}

/* 状态栏 */
.sc-status {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 10px 14px; border-radius: var(--hl-card-radius);
  font-size: 12.5px;
  border: 1px solid var(--hl-border);
}
.sc-status--on  { background: #d1fae5; border-color: #6ee7b7; color: #065f46; }
.sc-status--off { background: var(--hl-card-bg); color: var(--hl-text-secondary); }
.sc-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sc-dot--live { background: #10b981; animation: sc-pulse 1.5s infinite; }
.sc-dot--idle { background: var(--hl-text-muted); }
.sc-sep { color: var(--hl-border); padding: 0 2px; }

@keyframes sc-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}

/* 控制操作行：左实体摘要，右按钮+速率 */
.sc-ctrl-row {
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-sm);
}

/* 实体摘要 */
.sc-entity-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
.sc-ent {
  padding: 8px 12px;
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  font-size: 13px; color: var(--hl-text-secondary);
}
.sc-ent strong { color: var(--hl-primary); font-size: 15px; margin-left: 4px; }
.sc-ent--empty strong { color: var(--hl-danger); }

/* 操作区：bbox + 按钮行 + 速率 */
.sc-actions-area {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

/* bbox 提示 */
.sc-bbox-hint {
  font-size: 11.5px; color: var(--hl-text-muted);
  padding: 5px 10px;
  background: var(--hl-content-bg);
  border-radius: var(--hl-border-radius);
}
.sc-bbox-hint--warn { color: #d97706; }

/* 操作按钮行 */
.sc-action-row { display: flex; gap: 8px; flex-wrap: wrap; }
.sc-btn {
  height: 34px; padding: 0 16px; border-radius: var(--hl-border-radius);
  font-size: 13px; font-weight: 500; cursor: pointer;
  transition: background var(--hl-transition), opacity var(--hl-transition);
  border: none;
}
.sc-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.sc-btn--init  { background: var(--hl-primary); color: #fff; flex: 1; min-width: 160px; }
.sc-btn--init:hover:not(:disabled)  { background: #1d4ed8; }
.sc-btn--start { background: var(--hl-success); color: #fff; }
.sc-btn--start:hover:not(:disabled) { background: #15803d; }
.sc-btn--pause { background: var(--hl-warning); color: #fff; }
.sc-btn--pause:hover:not(:disabled) { background: #b45309; }
.sc-btn--reset { background: none; border: 1px solid var(--hl-border); color: var(--hl-text-secondary); }
.sc-btn--reset:hover { background: var(--hl-content-bg); color: var(--hl-danger); border-color: var(--hl-danger); }
.sc-btn--dispatch { background: var(--hl-info); color: #fff; min-width: 140px; }
.sc-btn--dispatch:hover:not(:disabled) { background: #0284c7; }
.sc-btn--dispatch-active {
  background: #0369a1;
  box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.18);
}
.sc-btn--algo {
  min-width: 178px;
  width: 178px;
}
.sc-btn--dispatch-run {
  flex: 1;
  min-width: 180px;
}
.sc-btn--export { background: #7c3aed; color: #fff; }
.sc-btn--export:hover:not(:disabled) { background: #6d28d9; }

.dispatch-solver-tag {
  display: inline-flex;
  align-items: center;
  padding: 0 10px;
  height: 34px;
  border-radius: var(--hl-border-radius);
  background: var(--hl-content-bg);
  color: var(--hl-text-secondary);
  font-size: 12px;
  white-space: nowrap;
}

.dispatch-cache-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  background: var(--hl-content-bg);
  color: var(--hl-text-secondary);
  font-size: 12px;
  white-space: nowrap;
}

.dispatch-cache-toggle input {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: var(--hl-info);
}

/* 调度统计快览 */
.dispatch-quick-stat {
  display: inline-flex;
  align-items: center;
  padding: 0 8px;
  font-size: 12px;
  color: var(--hl-success);
  font-weight: 500;
}
.dispatch-quick-stat--policy { color: #b45309; }

.sc-btn--policy {
  background: #0f766e;
  color: #fff;
  min-width: 178px;
}

.sc-btn--policy:hover:not(:disabled) {
  background: #0d9488;
}

.sc-btn--policy-active {
  background: #115e59;
  box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.22);
  transform: translateY(1px);
}

.sc-btn-spinner {
  display: inline-flex;
  width: 12px;
  height: 12px;
  margin-left: 6px;
  border: 2px solid rgba(255, 255, 255, 0.42);
  border-top-color: #fff;
  border-radius: 50%;
  animation: sc-spin 0.8s linear infinite;
}

@keyframes sc-spin {
  to { transform: rotate(360deg); }
}

/* 速率选择 */
.sc-speed-row { display: flex; align-items: center; gap: 12px; }
.sc-speed-label { font-size: 12px; font-weight: 500; color: var(--hl-text-secondary); white-space: nowrap; }
.sc-speed-btns  { display: flex; gap: 6px; flex-wrap: wrap; }
.sc-spd {
  height: 28px; min-width: 44px; padding: 0 10px;
  border: 1px solid var(--hl-border); border-radius: 6px;
  background: var(--hl-card-bg); font-size: 12px;
  color: var(--hl-text-secondary); cursor: pointer;
  transition: all var(--hl-transition);
}
.sc-spd:hover      { border-color: var(--hl-primary); color: var(--hl-primary); }
.sc-spd--active    { background: var(--hl-primary); color: #fff; border-color: var(--hl-primary); font-weight: 600; }

/* ── 三栏主体 ─────────────────────────────────────────────────────── */
.dispatch-main {
  display: flex;
  flex: 1;
  min-height: 0;
  gap: var(--hl-space-md);
  overflow: hidden;
}

/* 左侧面板 */
.dispatch-aside {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.dispatch-aside-scroll {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-md);
  overflow-y: auto;
  padding-right: 4px;
}

.dispatch-section {
  flex-shrink: 0;
}

.dispatch-section--orders :deep(.section-card__body) {
  padding: 0;
}

.order-panel-scroll {
  max-height: clamp(260px, 36vh, 420px);
  overflow-y: auto;
  padding: 12px 14px;
  min-height: 0;
}

/* KPI 网格 */
.kpi-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--hl-space-sm);
}

.kpi-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  padding: 10px 12px;
  box-shadow: var(--hl-card-shadow);
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.kpi-card__icon  { font-size: 16px; }
.kpi-card__value { font-size: 20px; font-weight: 700; color: var(--hl-text); line-height: 1.2; }
.kpi-card__label { font-size: 10.5px; color: var(--hl-text-muted); }
.kpi-card__trend { font-size: 10px; font-weight: 600; margin-top: 2px; }
.trend--up   { color: var(--hl-success); }
.trend--down { color: var(--hl-danger); }

/* 订单统计格 */
.order-stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.order-stat {
  display: flex; flex-direction: column; align-items: center;
  padding: 8px 4px;
  background: var(--hl-content-bg);
  border-radius: var(--hl-border-radius);
}
.order-stat__num   { font-size: 20px; font-weight: 700; color: var(--hl-text); }
.order-stat__num--active  { color: #3b82f6; }
.order-stat__num--success { color: var(--hl-success); }
.order-stat__num--danger  { color: var(--hl-danger); }
.order-stat__lbl   { font-size: 11px; color: var(--hl-text-muted); }

.dynamic-order-list,
.alert-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
  padding-right: 4px;
}

.dynamic-order-list {
  max-height: none;
  overflow: visible;
}

.alert-list {
  max-height: clamp(140px, 22vh, 240px);
  overflow-y: auto;
}

.dynamic-order-row,
.alert-row {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 9px 10px;
  border: 1px solid var(--hl-border);
  border-radius: 8px;
  background: var(--hl-content-bg);
}

.dynamic-order-row__main,
.alert-row__main,
.dynamic-order-row__meta,
.alert-row__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.dynamic-order-row__id,
.alert-row__id {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--hl-text);
}

.dynamic-order-row__badge,
.alert-row__badge {
  flex-shrink: 0;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 10.5px;
  font-weight: 700;
  background: #e2e8f0;
  color: #475569;
}

.dynamic-order-row__meta,
.alert-row__meta {
  font-size: 10.5px;
  color: var(--hl-text-muted);
}

.dynamic-order-row--waiting .dynamic-order-row__badge {
  background: #f1f5f9;
  color: #64748b;
}

.dynamic-order-row--active .dynamic-order-row__badge {
  background: #dbeafe;
  color: #1d4ed8;
}

.dynamic-order-row--done .dynamic-order-row__badge {
  background: #dcfce7;
  color: #166534;
}

.dynamic-order-row--danger,
.alert-row--danger {
  border-color: #fecaca;
  background: #fef2f2;
}

.dynamic-order-row--danger .dynamic-order-row__badge,
.alert-row--danger .alert-row__badge {
  background: #fee2e2;
  color: #b91c1c;
}

.alert-row--warn {
  border-color: #fed7aa;
  background: #fff7ed;
}

.alert-row--warn .alert-row__badge {
  background: #ffedd5;
  color: #c2410c;
}

/* 地图区 */
.dispatch-map {
  flex: 1;
  min-width: 0;
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.map-placeholder {
  text-align: center;
  padding: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}
.map-placeholder__icon  { font-size: 48px; }
.map-placeholder__title { font-size: 16px; font-weight: 600; color: var(--hl-text); }
.map-placeholder__desc  { font-size: 12.5px; color: var(--hl-text-muted); line-height: 1.7; }

.go-btn {
  display: inline-block;
  margin-top: 6px;
  padding: 8px 20px;
  background: var(--hl-primary, #1565c0);
  color: #fff;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
  transition: opacity 0.2s;
}
.go-btn:hover { opacity: 0.85; }

/* 右侧详情面板 */
.dispatch-detail {
  width: 280px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-md);
  overflow-y: auto;
}

@media (max-width: 1080px) {
  .sc-entity-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .sc-entity-row {
    grid-template-columns: 1fr;
  }
}

.detail-hint {
  font-size: 11.5px; color: var(--hl-text-muted); margin-bottom: 10px;
}
.detail-tabs { display: flex; gap: 4px; margin-bottom: 12px; }
.detail-tab {
  flex: 1; height: 28px;
  border: 1px solid var(--hl-border);
  border-radius: 4px; background: none;
  font-size: 12px; cursor: pointer;
  color: var(--hl-text-secondary);
  transition: all var(--hl-transition);
}
.detail-tab--active {
  background: var(--hl-primary-alpha);
  border-color: var(--hl-primary);
  color: var(--hl-primary);
  font-weight: 600;
}

.empty-list { font-size: 12px; color: var(--hl-text-muted); padding: 12px 0; text-align: center; }
.alert-badge--ok { font-size: 12px; color: var(--hl-success); }

/* ── 图表卡片（迁自 Dashboard）─────────────────────────────────────── */
.chart-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
}
.chart-card__header {
  padding: 10px 14px 8px;
  border-bottom: 1px solid var(--hl-border);
  display: flex; align-items: baseline; gap: 8px;
}
.chart-card__title { font-size: 12.5px; font-weight: 600; color: var(--hl-text); }
.chart-card__sub   { font-size: 10.5px; color: var(--hl-text-muted); }
.chart-card__body  { padding: 12px 14px; }
.chart-card__body--placeholder {
  display: flex; align-items: center; justify-content: center;
}

/* 模式条形 */
.mode-bars { display: flex; flex-direction: column; gap: 8px; }
.mode-bar  { display: flex; align-items: center; gap: 6px; }
.mode-bar__label { font-size: 11px; color: var(--hl-text-secondary); width: 130px; flex-shrink: 0; }
.mode-bar__track {
  flex: 1; height: 5px;
  background: var(--hl-border); border-radius: 3px; overflow: hidden;
}
.mode-bar__fill  { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.mode-bar__pct   { font-size: 10.5px; color: var(--hl-text-muted); width: 28px; text-align: right; }

/* 占位可视化 */
.placeholder-visual { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 8px; }
.placeholder-visual__grid {
  display: grid; grid-template-columns: repeat(10, 1fr);
  gap: 2px; width: 100%;
}
.placeholder-visual__cell { aspect-ratio: 1; border-radius: 2px; }
.placeholder-visual__hint { font-size: 10.5px; color: var(--hl-text-muted); text-align: center; }
.mock-chart { width: 100%; height: 70px; }
</style>
