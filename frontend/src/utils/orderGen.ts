/**
 * orderGen — 订单生成纯函数
 *
 * 约束：
 *   - 纯函数，不 import 任何 Store / Service / Vue 响应式对象
 *   - 所有依赖通过参数传入，可直接用于单元测试
 *   - 从 stores/order.ts 的 generateOrder() 迁移而来，并扩展 geo_mode 支持
 */

import type { Order, OrderGeneratorConfig, TaskStatus, WarehouseEntry } from '@/types'
import type { Bbox } from './geoLayout'

// ── 内部工具函数 ─────────────────────────────────────────────────

function genOrderId(): string {
  const ts = Date.now().toString(36).toUpperCase().slice(-6)
  const r  = Math.floor(Math.random() * 9000 + 1000)
  return `ORD-${ts}-${r}`
}

function pickWeighted(choices: Array<{ value: string; weight: number }>): string {
  const total = choices.reduce((s, c) => s + c.weight, 0)
  let rand    = Math.random() * total
  for (const c of choices) {
    rand -= c.weight
    if (rand <= 0) return c.value
  }
  return choices[choices.length - 1].value
}

/**
 * 在给定坐标附近随机偏移（角度 + 距离均均匀随机）
 * 采用简化平面近似（适用于 < 50 km 范围）
 */
function offsetCoord(
  lng: number,
  lat: number,
  radiusKm: number,
): { lng: number; lat: number } {
  const angle = Math.random() * Math.PI * 2
  const dist  = Math.random() * radiusKm
  const dLat  = (dist * Math.cos(angle)) / 111
  const dLng  = (dist * Math.sin(angle)) / (111 * Math.cos((lat * Math.PI) / 180))
  return { lng: lng + dLng, lat: lat + dLat }
}

/** 在 bbox 内生成均匀随机点 */
function randomInBbox(bbox: Bbox): { lng: number; lat: number } {
  return {
    lng: bbox.min_lng + Math.random() * (bbox.max_lng - bbox.min_lng),
    lat: bbox.min_lat + Math.random() * (bbox.max_lat - bbox.min_lat),
  }
}

/**
 * Box-Muller 变换产生标准正态随机数（N(0,1)）
 * 对 u1 做下界保护，防止 log(0)
 */
function gaussianRand(): number {
  const u1 = Math.max(Math.random(), 1e-10)
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/**
 * 在 bbox 中随机选取一个热点，然后在热点附近正态分布，
 * 超出 bbox 的点截断到边界。
 */
function clusteredPoint(
  bbox: Bbox,
  radiusKm: number,
): { lng: number; lat: number } {
  const hotspot = randomInBbox(bbox)
  // σ ≈ radiusKm / 3：使约 99.7% 的点落在 radiusKm 范围内
  const sigma = radiusKm / 3
  const dLat  = gaussianRand() * sigma / 111
  const dLng  = gaussianRand() * sigma / (111 * Math.cos((hotspot.lat * Math.PI) / 180))
  return {
    lng: Math.max(bbox.min_lng, Math.min(bbox.max_lng, hotspot.lng + dLng)),
    lat: Math.max(bbox.min_lat, Math.min(bbox.max_lat, hotspot.lat + dLat)),
  }
}

function priorityLabel(p: string): string {
  return p === 'URGENT' ? '紧急' : p === 'NORMAL' ? '普通' : '低优先'
}

// ── 核心导出 ─────────────────────────────────────────────────────

/**
 * 生成一个订单。纯函数，不含任何副作用。
 *
 * @param config        订单生成器参数（来自 generatorConfig.value）
 * @param warehousePool 可用仓库列表（来自 warehousePool.value，至少 1 个）
 * @param sceneBbox     当前仿真地图 bbox（来自 sceneStore.context.road_network.bounds，可为 null）
 */
export function buildOrder(
  config: OrderGeneratorConfig,
  warehousePool: WarehouseEntry[],
  sceneBbox?: Bbox | null,
): Order {
  const now      = Date.now()
  const geoMode  = config.geo_mode ?? 'uniform'
  const radiusKm = config.cluster_radius_km ?? 5

  // 随机选仓库
  const wh = warehousePool[Math.floor(Math.random() * warehousePool.length)]

  // 货物重量（保留 2 位小数）
  const payload = parseFloat(
    (config.weight_min + Math.random() * (config.weight_max - config.weight_min)).toFixed(2),
  )

  // 优先级（加权随机）
  const priority = pickWeighted([
    { value: 'URGENT', weight: config.priority_urgent },
    { value: 'NORMAL', weight: config.priority_normal },
    { value: 'LOW',    weight: config.priority_low },
  ])

  // 时间窗口（分钟 → ms）
  const windowMs =
    (config.window_min + Math.random() * (config.window_max - config.window_min)) * 60_000

  // 目的地坐标——根据 geo_mode 选择策略
  let dest: { lng: number; lat: number }

  if (geoMode === 'depot_nearby') {
    // 围绕仓库坐标的圆形覆盖圈
    dest = offsetCoord(wh.lng, wh.lat, radiusKm)
  } else if (geoMode === 'clustered' && sceneBbox) {
    // bbox 内随机热点 + 正态分布
    dest = clusteredPoint(sceneBbox, radiusKm)
  } else if (geoMode === 'uniform' && sceneBbox) {
    // bbox 内均匀随机
    dest = randomInBbox(sceneBbox)
  } else {
    // 地图未选定时兜底：仓库坐标周边 5 km
    dest = offsetCoord(wh.lng, wh.lat, 5)
  }

  const deadline = now + windowMs

  const order: Order = {
    order_id:         genOrderId(),
    status:           'PENDING' as TaskStatus,
    source_type:      'DEPOT',
    pickup_source_id: wh.id,
    payload_weight:   payload,
    create_time:      now,
    deadline,
    delivery_lng:     parseFloat(dest.lng.toFixed(6)),
    delivery_lat:     parseFloat(dest.lat.toFixed(6)),
    delivery_z:       0,
    fulfillment_mode: payload > 3.5 ? 'DRONE_TRUCK_DEPOT' : 'DRONE_DIRECT',
    priority,
    warehouse_name:   wh.name,
    deadline_iso:     new Date(deadline).toLocaleTimeString('zh-CN', {
      hour:   '2-digit',
      minute: '2-digit',
    }),
    priority_label: priorityLabel(priority),
  }

  return order
}
