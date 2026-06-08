import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { Order, OrderGeneratorConfig, ScheduledDynamicOrder, TaskStatus, WarehouseEntry } from '@/types'
import { http } from '@/services/http'

/** 后端仿真运行时订单统计（由 TICK 帧推送） */
export interface SimStats {
  orders_pending:   number
  orders_assigned:  number
  orders_completed: number
  orders_timeout:   number
  total_energy_cost_wh?: number
  total_dispatch_cost?: number
  dispatch_count?: number
}
import { useEntityStore } from './entity'
import { useSceneStore } from './scene'
import { buildOrder } from '@/utils/orderGen'

// ── localStorage Keys ────────────────────────────────────────────
/** v2：默认关闭泊松随机流，与预设静态+动态订单基准一致；扩展随机流时可把 arrival_rate 调回 >0 */
const LS_CONFIG_KEY     = 'hl-order-gen-config-v2'
const LS_ORDERS_KEY     = 'hl-orders-v1'
const MAX_STORED_ORDERS = 500

// ── 默认仓库后备池（entityStore 为空时兼容层，不含真实地名） ─────────────
const DEFAULT_WAREHOUSES: WarehouseEntry[] = [
  { id: 'depot-fallback-1', name: '仓库后备-01', lng: 0, lat: 0 },
  { id: 'depot-fallback-2', name: '仓库后备-02', lng: 0, lat: 0 },
]

const DEFAULT_CONFIG: OrderGeneratorConfig = {
  /** 0 = 仅 orders.json 静态 + 动态注入，便于算法对比；后续压力测试改为 4 等 */
  arrival_rate:      0,
  weight_min:        0.5,
  weight_max:        5.0,
  window_min:        20,
  window_max:        60,
  priority_urgent:   20,
  priority_normal:   60,
  priority_low:      20,
  // 10 静态 + 10 动态 + 余量；开启随机流时可再调高
  max_orders:        40,
  geo_mode:          'uniform',
  cluster_radius_km: 1.5,
  burst_enabled:     false,
  burst_multiplier:  3,
  burst_duration_s:  60,
}

// ── 工具函数 ─────────────────────────────────────────────────────
function loadConfig(): OrderGeneratorConfig {
  try {
    const raw = localStorage.getItem(LS_CONFIG_KEY)
    if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) }
  } catch { /* ignore */ }
  return { ...DEFAULT_CONFIG }
}

function loadOrders(): Order[] {
  try {
    const raw = localStorage.getItem(LS_ORDERS_KEY)
    if (raw) return JSON.parse(raw) as Order[]
  } catch { /* ignore */ }
  return []
}

export const useOrderStore = defineStore('order', () => {
  const entityStore = useEntityStore()
  const sceneStore  = useSceneStore()

  // ── 响应式状态 ────────────────────────────────────────────────
  const generatorConfig  = ref<OrderGeneratorConfig>(loadConfig())
  const generatedOrders  = ref<Order[]>(loadOrders())
  /** 与 orders.json dynamic_orders 一致，随 initSim 发给后端 */
  const scheduledDynamicOrders = ref<ScheduledDynamicOrder[]>([])
  const generatorRunning = ref(false)

  // ── 后端联调统计（由 TICK 帧 stats 字段更新）──────────────────────────
  const stats = ref<SimStats | null>(null)

  // ── 后端订单列表（仿真运行时从 GET /api/sim/orders 拉取）──────────────
  const backendOrders = ref<any[]>([])

  // 定时器句柄：主速率 / burst 切换 / burst 额外订单
  const _mainTimerId   = ref<ReturnType<typeof setInterval> | null>(null)
  const _burstToggleId = ref<ReturnType<typeof setInterval> | null>(null)
  const _burstExtraId  = ref<ReturnType<typeof setInterval> | null>(null)

  // ── 持久化 ────────────────────────────────────────────────────
  function _persistConfig() {
    localStorage.setItem(LS_CONFIG_KEY, JSON.stringify(generatorConfig.value))
  }
  function _persistOrders() {
    const recent = generatedOrders.value.slice(-MAX_STORED_ORDERS)
    generatedOrders.value = recent
    localStorage.setItem(LS_ORDERS_KEY, JSON.stringify(recent))
  }

  function updateConfig(patch: Partial<OrderGeneratorConfig>) {
    generatorConfig.value = { ...generatorConfig.value, ...patch }
    _persistConfig()
  }

  // ── 仓库池（优先使用用户配置的仓库） ─────────────────────────
  const warehousePool = computed<WarehouseEntry[]>(() => {
    if (entityStore.depots.length > 0) {
      return entityStore.depots.map(d => ({
        id:   d.depot_id,
        name: d.name,
        lng:  d.lng,
        lat:  d.lat,
      }))
    }
    return DEFAULT_WAREHOUSES
  })

  /** 当前仿真场景 bbox（无地图时为 null） */
  const sceneBbox = computed(() => sceneStore.context?.road_network?.bounds ?? null)

  // ── 核心生成逻辑 ──────────────────────────────────────────────

  function addGeneratedOrder(order: Order) {
    generatedOrders.value.push(order)
    _persistOrders()
  }

  function generateOnce() {
    // 总量上限检查
    const maxOrders = generatorConfig.value.max_orders ?? null
    if (maxOrders !== null && generatedOrders.value.length >= maxOrders) {
      stopGenerator()
      return
    }
    const order = buildOrder(generatorConfig.value, warehousePool.value, sceneBbox.value)
    addGeneratedOrder(order)
  }

  /**
   * 一次性批量生成 N 个订单（不启动定时器，静态生成）。
   * 用于快速测试或生成初始订单集合。
   * 生成完后自动设置 max_orders 为当前订单数，防止动态生成新的。
   */
  function generateBatch(count: number) {
    const maxOrders = generatorConfig.value.max_orders ?? null
    let generated = 0
    for (let i = 0; i < count; i++) {
      if (maxOrders !== null && generatedOrders.value.length >= maxOrders) {
        break  // 达到上限，停止
      }
      const order = buildOrder(generatorConfig.value, warehousePool.value, sceneBbox.value)
      addGeneratedOrder(order)
      generated++
    }
    
    // 生成完后，自动设置 max_orders 为当前订单数（防止继续生成新的）
    if (generated > 0) {
      generatorConfig.value.max_orders = generatedOrders.value.length
      _persistConfig()
    }
    
    return generated
  }

  /**
   * 启动突发周期：每隔 burst_duration_s 秒交替开启/关闭一个额外定时器，
   * 额外定时器以 (burst_multiplier - 1) 倍速度生成订单。
   */
  function _startBurstCycle(cfg: OrderGeneratorConfig) {
    const durationMs = (cfg.burst_duration_s ?? 60) * 1000
    let burstOn      = false

    _burstToggleId.value = setInterval(() => {
      burstOn = !burstOn
      if (burstOn) {
        const rate      = Math.max(0.5, Math.min(cfg.arrival_rate, 60))
        const extraRate = rate * ((cfg.burst_multiplier ?? 3) - 1)
        if (extraRate > 0) {
          const extraMs = (60 / extraRate) * 1000
          _burstExtraId.value = setInterval(() => generateOnce(), extraMs)
        }
      } else {
        if (_burstExtraId.value !== null) {
          clearInterval(_burstExtraId.value)
          _burstExtraId.value = null
        }
      }
    }, durationMs)
  }

  function startGenerator() {
    if (generatorRunning.value) return
    generatorRunning.value = true
    const cfg        = generatorConfig.value
    const rate       = Math.max(0.5, Math.min(cfg.arrival_rate, 60))
    const intervalMs = (60 / rate) * 1000
    _mainTimerId.value = setInterval(() => generateOnce(), intervalMs)
    if (cfg.burst_enabled) {
      _startBurstCycle(cfg)
    }
  }

  function stopGenerator() {
    if (_mainTimerId.value !== null) {
      clearInterval(_mainTimerId.value)
      _mainTimerId.value = null
    }
    if (_burstToggleId.value !== null) {
      clearInterval(_burstToggleId.value)
      _burstToggleId.value = null
    }
    if (_burstExtraId.value !== null) {
      clearInterval(_burstExtraId.value)
      _burstExtraId.value = null
    }
    generatorRunning.value = false
  }

  function restartIfRunning() {
    if (generatorRunning.value) {
      stopGenerator()
      startGenerator()
    }
  }

  function clearOrders() {
    generatedOrders.value = []
    scheduledDynamicOrders.value = []
    localStorage.removeItem(LS_ORDERS_KEY)
  }

  /** 从后端拉取最近 N 条仿真订单（仿真运行时调用，替换 backendOrders 列表） */
  async function fetchBackendOrders(limit = 100) {
    try {
      const res = await http.get<{ total: number; orders: any[] }>(
        `/api/sim/orders?limit=${limit}`
      )
      backendOrders.value = res.orders ?? []
    } catch {
      // 后端未启动或仿真未初始化时静默忽略
    }
  }

  // ── 标准 CRUD（兼容旧接口） ───────────────────────────────────
  function addOrder(order: Order) { addGeneratedOrder(order) }

  function completeOrder(id: string) {
    const o = generatedOrders.value.find(x => x.order_id === id)
    if (o) o.status = 'COMPLETED' as TaskStatus
    _persistOrders()
  }

  // ── 计算属性 ──────────────────────────────────────────────────
  const pendingOrders  = computed(() => generatedOrders.value.filter(o => o.status === 'PENDING'))
  const activeOrders   = computed(() => generatedOrders.value.filter(o => ['ASSIGNED', 'IN_TRANSIT'].includes(o.status)))
  const finishedOrders = computed(() => generatedOrders.value.filter(o => ['COMPLETED', 'FAILED'].includes(o.status)))
  const totalCount     = computed(() => generatedOrders.value.length)
  const recentOrders   = computed(() => [...generatedOrders.value].reverse().slice(0, 100))

  return {
    generatorConfig,
    generatedOrders,
    scheduledDynamicOrders,
    generatorRunning,
    stats,
    backendOrders,
    warehousePool,
    // 操作
    updateConfig,
    generateOnce,
    generateBatch,
    startGenerator,
    stopGenerator,
    restartIfRunning,
    clearOrders,
    fetchBackendOrders,
    // 兼容旧接口
    addOrder,
    completeOrder,
    // 计算属性
    pendingOrders,
    activeOrders,
    finishedOrders,
    totalCount,
    recentOrders,
  }
})

