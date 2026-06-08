<template>
  <div class="unified-map-wrap">
    <!-- Leaflet 地图容器 -->
    <div ref="mapEl" class="unified-map"></div>

    <!-- 加载中遮罩 -->
    <div v-if="layerLoading" class="umap-loading">
      <div class="umap-spinner"></div>
      <span>{{ layerLoadingText }}</span>
    </div>

    <!-- 图例 -->
    <div class="umap-legend">
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#1e40af">🏭</span>仓库
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#d97706">⚡</span>充换电站
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#6d28d9">🕐</span>待分配订单
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#0369a1">📍</span>卡车配送订单
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#0284c7">🚀</span>无人机配送订单
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__pin" style="background:#16a34a">✅</span>已完成订单
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__box" style="background:#ff4444;opacity:.65"></span>禁飞区建筑
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__box" style="background:#33bb55;opacity:.4"></span>可飞区建筑
      </div>
      <div class="umap-legend__item">
        <span class="umap-legend__line" style="background:#e6a817"></span>道路网络
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import { useSceneStore, type SceneContext } from '@/stores/scene'
import { useEntityStore } from '@/stores/entity'
import { useOrderStore } from '@/stores/order'
import { useSystemStore } from '@/stores/system'

declare const L: typeof import('leaflet')

// ── Store & refs ──────────────────────────────────────────────────

const sceneStore  = useSceneStore()
const entityStore = useEntityStore()
const orderStore  = useOrderStore()
const systemStore = useSystemStore()

const mapEl         = ref<HTMLDivElement | null>(null)
const layerLoading  = ref(false)
const layerLoadingText = ref('正在加载地图数据…')

let map: ReturnType<typeof L.map> | null = null
let buildingLayer:  L.GeoJSON      | null = null
let roadLayer:      L.GeoJSON      | null = null
let depotGroup:     L.LayerGroup   | null = null
let stationGroup:   L.LayerGroup   | null = null
let orderGroup:     L.LayerGroup   | null = null
let routeGroup:     L.LayerGroup   | null = null
let routeMarkerGroup: L.LayerGroup | null = null
let runtimePathGroup: L.LayerGroup | null = null
let truckMarkers:   Map<string, L.Marker> = new Map()
let droneMarkers:   Map<string, L.Marker> = new Map()
let truckGroup:     L.LayerGroup   | null = null
let droneGroup:     L.LayerGroup   | null = null
const activeDronePathCache = new Map<string, [number, number][]>()


// ── 地图初始化 ────────────────────────────────────────────────────

function initMap() {
  if (!mapEl.value) return

  // L.canvas() 渲染器：强制，不可更改（支持 3000+ 多边形高频刷新）
  const renderer = L.canvas({ padding: 0.5 })

  map = L.map(mapEl.value, {
    center:   [31.23, 121.47],
    zoom:     13,
    renderer,
    zoomControl: true,
  })

  // 无人机航线专用图层：放在设施 marker 之上，避免被仓库/站点图标遮挡
  const droneRoutePane = map.createPane('droneRoutePane')
  droneRoutePane.style.zIndex = '680'
  droneRoutePane.style.pointerEvents = 'none'

  const droneMarkerPane = map.createPane('droneMarkerPane')
  droneMarkerPane.style.zIndex = '690'
  droneMarkerPane.style.pointerEvents = 'none'

  L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { attribution: '© CARTO · © OpenStreetMap 贡献者', maxZoom: 19 }
  ).addTo(map)
}

// ── 数据加载 ───────────────────────────────────────────────────────

async function loadLayers(scene: SceneContext) {
  if (!map) return

  layerLoading.value     = true
  layerLoadingText.value = '正在渲染场景数据…'

  // ── 建筑图层：优先读取 sceneStore 缓存，避免重复 API 调用 ──────────────────
  if (buildingLayer) { map.removeLayer(buildingLayer); buildingLayer = null }

  let buildData: GeoJSON.FeatureCollection | null =
    sceneStore.buildingsGeoJSON as GeoJSON.FeatureCollection | null

  if (!buildData) {
    // 未命中缓存（页面刷新后 / 首次直接访问）：从后端加载并写入缓存
    layerLoadingText.value = '正在加载建筑数据（首次加载）…'
    try {
      const r = await fetch('/api/geo/query', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          scene_id:      scene.scene_id,  // 添加 scene_id，使后端可以识别预设场景并用缓存
          minx:          scene.sel_bounds.minx,
          miny:          scene.sel_bounds.miny,
          maxx:          scene.sel_bounds.maxx,
          maxy:          scene.sel_bounds.maxy,
          threshold:     scene.threshold,
          height_column: scene.height_column ?? null,
          max:           30000,
        }),
      })
      const d = await r.json()
      if (!d.error) {
        buildData = d as GeoJSON.FeatureCollection
        // 写入缓存供下次路由切换复用
        sceneStore.setBuildingsGeoJSON(buildData)
      }
    } catch {
      // 静默处理：geo 后端未启动时地图无建筑层，但不崩溃
    }
  }

  if (buildData) {
    buildingLayer = L.geoJSON(buildData, {
      style: (feature: GeoJSON.Feature | undefined) => {
        const nf = feature?.properties?.nf
        return {
          color:       nf ? '#cc0000' : '#008800',
          weight:      0.6,
          fillColor:   nf ? '#ff4444' : '#33bb55',
          fillOpacity: nf ? 0.65 : 0.35,
        }
      },
      // onEachFeature 已移除：Canvas 渲染器下 N 万个独立 popup 绑定严重消耗内存
    }).addTo(map)

    // 事件委托：图层级单次 click，动态创建 Popup，替代逐个 Feature 绑定
    buildingLayer.on('click', (e) => {
      const evt = e as unknown as L.LeafletMouseEvent & {
        layer?: L.Layer & { feature?: GeoJSON.Feature }
      }
      const p = evt.layer?.feature?.properties
      if (!p || !map) return
      L.popup({ maxWidth: 180 })
        .setLatLng(evt.latlng)
        .setContent(
          `<div style="font-size:1rem;font-weight:700">${p.h} m</div>` +
          `<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600;` +
          `background:${p.nf ? '#ae2012' : '#2d6a4f'};color:#fff">` +
          `${p.nf ? '🔴 禁飞区' : '🟢 可飞区'}</span>`
        )
        .openOn(map)
    })
  }

  // ── 道路图层：直接从 scene.road_network.edges 构造，无需 Overpass API ─────
  // edges.shape 已是 [lng, lat][] 序列，与 GeoJSON LineString 坐标格式完全一致
  if (roadLayer) { map.removeLayer(roadLayer); roadLayer = null }

  if (scene.road_network.edges.length > 0) {
    const roadFeatures: GeoJSON.Feature[] = scene.road_network.edges.map(e => ({
      type:       'Feature'    as const,
      geometry:   { type: 'LineString' as const, coordinates: e.shape },
      properties: {},
    }))
    roadLayer = L.geoJSON(
      { type: 'FeatureCollection', features: roadFeatures } as GeoJSON.FeatureCollection,
      { style: () => ({ color: '#e6a817', weight: 2, opacity: 0.85 }) }
    ).addTo(map)
    roadLayer.bringToBack()
  }

  // ── 缩放到路网范围 ─────────────────────────────────────────────────────────
  const rb = scene.road_network?.bounds
  if (rb) {
    map.fitBounds([[rb.min_lat, rb.min_lng], [rb.max_lat, rb.max_lng]], { padding: [20, 20] })
  } else {
    map.fitBounds([
      [scene.sel_bounds.miny, scene.sel_bounds.minx],
      [scene.sel_bounds.maxy, scene.sel_bounds.maxx],
    ], { padding: [20, 20] })
  }

  // 场景加载完成后立即绘制设施标注（确保显示在建筑/道路图层之上）
  drawFacilities()

  layerLoading.value = false
}

// ── 设施标注（仓库 + 充换电站）────────────────────────────────────

function _depotIcon(name: string) {
  return L.divIcon({
    className: '',
    html: `<div class="fac-pin fac-pin--depot" title="${name}">🏭</div>`,
    iconSize:   [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -18],
  })
}

function _stationIcon(name: string) {
  return L.divIcon({
    className: '',
    html: `<div class="fac-pin fac-pin--station" title="${name}">⚡</div>`,
    iconSize:   [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -15],
  })
}

const ORDER_STATUS_META: Record<string, { icon: string; cls: string; label: string }> = {
  PENDING:    { icon: '🕐', cls: 'order-pin--pending',    label: '待分配' },
  ASSIGNED:   { icon: '📍', cls: 'order-pin--assigned',   label: '已分配' },
  PICKED_UP:  { icon: '🚁', cls: 'order-pin--pickedup',   label: '取货中' },
  DELIVERING: { icon: '🚀', cls: 'order-pin--delivering', label: '配送中' },
  COMPLETED:  { icon: '✅', cls: 'order-pin--completed',  label: '已完成' },
  TIMEOUT:    { icon: '⚠️', cls: 'order-pin--timeout',    label: '已超时' },
  REJECTED:   { icon: '❌', cls: 'order-pin--rejected',   label: '已拒绝' },
}

const DRONE_ASSIGNED_ORDER_META = {
  icon: '🚀',
  cls: 'order-pin--drone-assigned',
  label: '无人机配送订单',
}

function _isDroneDeliveryMode(mode?: unknown): boolean {
  const normalized = String(mode ?? '').trim().toUpperCase()
  if (!normalized || normalized === '—') return false
  return normalized === 'B'
    || normalized === 'B_WAIT'
    || normalized === 'B_DYNAMIC'
    || normalized === 'C'
    || normalized.startsWith('B_')
    || normalized.startsWith('C_')
    || normalized.startsWith('DRONE')
}

function _orderDisplayMeta(status: string, deliveryMode?: unknown) {
  const meta = ORDER_STATUS_META[status] ?? ORDER_STATUS_META['PENDING']
  if (status === 'ASSIGNED' && _isDroneDeliveryMode(deliveryMode)) {
    return DRONE_ASSIGNED_ORDER_META
  }
  return meta
}

function _orderIcon(status: string, priority?: string, deliveryMode?: unknown) {
  const meta = _orderDisplayMeta(status, deliveryMode)
  const pClass = priority === 'URGENT' ? ' order-pin--urgent' : ''
  return L.divIcon({
    className: '',
    html: `<div class="order-pin ${meta.cls}${pClass}" title="${meta.label}">${meta.icon}</div>`,
    iconSize:   [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -14],
  })
}

function drawFacilities() {
  if (!map) return

  // 确保 LayerGroup 存在（创建一次后复用）
  if (!depotGroup)   { depotGroup   = L.layerGroup().addTo(map) }
  if (!stationGroup) { stationGroup = L.layerGroup().addTo(map) }

  depotGroup.clearLayers()
  stationGroup.clearLayers()

  for (const depot of entityStore.depots) {
    // 跳过坐标未就绪（未选地图时默认 0,0）
    if (depot.lng === 0 && depot.lat === 0) continue
    const marker = L.marker([depot.lat, depot.lng], { icon: _depotIcon(depot.name), draggable: true })
      .bindPopup(
        `<div class="fac-popup">
          <div class="fac-popup__title">🏭 ${depot.name}</div>
          <div class="fac-popup__row"><span>ID</span><span>${depot.depot_id}</span></div>
          <div class="fac-popup__row"><span>容量</span><span>${depot.capacity} 件</span></div>
          <div class="fac-popup__row"><span>充换电位</span><span>${depot.parking_slots} 个</span></div>
          <div class="fac-popup__row"><span>坐标</span><span>${depot.lng.toFixed(5)}, ${depot.lat.toFixed(5)}</span></div>
        </div>`,
        { maxWidth: 220, className: 'fac-popup-wrap' }
      )
      .on('dragend', (e: any) => {
        const newPos = e.target.getLatLng()
        entityStore.updateDepot({
          ...depot,
          lat: parseFloat(newPos.lat.toFixed(6)),
          lng: parseFloat(newPos.lng.toFixed(6)),
        })
      })
      .addTo(depotGroup)
  }

  for (const sta of entityStore.stations) {
    if (sta.lng === 0 && sta.lat === 0) continue
    const marker = L.marker([sta.lat, sta.lng], { icon: _stationIcon(sta.name), draggable: true })
      .bindPopup(
        `<div class="fac-popup">
          <div class="fac-popup__title">⚡ ${sta.name}</div>
          <div class="fac-popup__row"><span>ID</span><span>${sta.station_id}</span></div>
          <div class="fac-popup__row"><span>并发槽位</span><span>${sta.parking_slots} 个</span></div>
          <div class="fac-popup__row"><span>换电耗时</span><span>${sta.swap_time} s</span></div>
          <div class="fac-popup__row"><span>坐标</span><span>${sta.lng.toFixed(5)}, ${sta.lat.toFixed(5)}</span></div>
        </div>`,
        { maxWidth: 220, className: 'fac-popup-wrap' }
      )
      .on('dragend', (e: any) => {
        const newPos = e.target.getLatLng()
        entityStore.updateStation({
          ...sta,
          lat: parseFloat(newPos.lat.toFixed(6)),
          lng: parseFloat(newPos.lng.toFixed(6)),
        })
      })
      .addTo(stationGroup)
  }
}

// ── 订单任务点标注 ────────────────────────────────────────────────

function drawOrders() {
  if (!map) return
  if (!orderGroup) { orderGroup = L.layerGroup().addTo(map) }
  orderGroup.clearLayers()

  const orders = orderStore.generatedOrders
  for (const order of orders) {
    const lng = order.delivery_lng
    const lat = order.delivery_lat
    if (!lng || !lat || (lng === 0 && lat === 0)) continue

    const status   = order.status ?? 'PENDING'
    const meta     = ORDER_STATUS_META[status] ?? ORDER_STATUS_META['PENDING']
    const priority = order.priority ?? order.priorityLabel ?? ''
    const pLabel   = priority === 'URGENT' ? '🔴 紧急'
                   : priority === 'LOW'    ? '🟢 低优先级'
                   : '🟡 普通'

    const deadlineStr = order.deadline_iso
      ?? (order.deadlineText ?? (order.deadline ? new Date(order.deadline).toLocaleString('zh-CN') : '—'))
    const createStr   = order.create_time
      ? (order.time_domain === 'sim_s'
          ? `${order.create_time.toFixed(1)} s`
          : new Date(order.create_time).toLocaleString('zh-CN')
        )
      : '—'
    const modeStr = order.assigned_mode ?? order.fulfillment_mode ?? '—'
    const vehicleStr = order.assigned_vehicle_id ?? '—'
    const displayMeta = _orderDisplayMeta(status, modeStr)

    const marker = L.marker([lat, lng], { icon: _orderIcon(status, priority, modeStr), draggable: true })
      .bindPopup(
        `<div class="fac-popup">
          <div class="fac-popup__title order-popup__title">${displayMeta.icon} 订单 ${order.order_id.slice(-6)}</div>
          <div class="fac-popup__row"><span>状态</span><span>${meta.label}</span></div>
          <div class="fac-popup__row"><span>优先级</span><span>${pLabel}</span></div>
          <div class="fac-popup__row"><span>重量</span><span>${order.payload_weight.toFixed(2)} kg</span></div>
          <div class="fac-popup__row"><span>截止时间</span><span>${deadlineStr}</span></div>
          <div class="fac-popup__row"><span>创建时间</span><span>${createStr}</span></div>
          <div class="fac-popup__row"><span>履约模式</span><span>${modeStr}</span></div>
          <div class="fac-popup__row"><span>分配载具</span><span>${vehicleStr}</span></div>
          <div class="fac-popup__row"><span>坐标</span><span>${lng.toFixed(5)}, ${lat.toFixed(5)}</span></div>
        </div>`,
        { maxWidth: 240, className: 'fac-popup-wrap' }
      )
      .on('dragend', (e: any) => {
        const newPos = e.target.getLatLng()
        const updatedOrder = {
          ...order,
          delivery_lat: parseFloat(newPos.lat.toFixed(6)),
          delivery_lng: parseFloat(newPos.lng.toFixed(6)),
        }
        const orderIdx = orderStore.generatedOrders.findIndex(o => o.order_id === order.order_id)
        if (orderIdx !== -1) {
          orderStore.generatedOrders[orderIdx] = updatedOrder
        }
      })
      .addTo(orderGroup)
  }
}

function clearRouteOverlays() {
  if (routeGroup) {
    routeGroup.clearLayers()
    routeGroup.remove()
    routeGroup = null
  }
  if (routeMarkerGroup) {
    routeMarkerGroup.clearLayers()
    routeMarkerGroup.remove()
    routeMarkerGroup = null
  }
}

function drawDispatchRoutes(plan: { truck_routes?: Record<string, any>; drone_routes?: any[] }) {
  if (!map) return
  clearRouteOverlays()

  routeGroup = L.layerGroup().addTo(map)
  routeMarkerGroup = L.layerGroup().addTo(map)

  const addLabelMarker = (lat: number, lng: number, label: string, color: string) => {
    L.circleMarker([lat, lng], {
      radius: 5,
      color,
      fillColor: color,
      fillOpacity: 0.95,
      weight: 2,
    })
      .bindTooltip(label, { permanent: false, direction: 'top', offset: [0, -8] })
      .addTo(routeMarkerGroup as L.LayerGroup)
  }

  if (plan.truck_routes) {
    for (const route of Object.values(plan.truck_routes)) {
      const coords = Array.isArray(route.geometry) && route.geometry.length > 1
        ? route.geometry.map((p: any) => [p.lat, p.lng] as [number, number])
        : route.nodes.map((node: any) => [node.lat, node.lng] as [number, number])
      if (coords.length > 1) {
        L.polyline(coords, {
          color: '#2563eb',
          weight: 4,
          opacity: 0.65,
        }).addTo(routeGroup as L.LayerGroup)
      }
      if (route.nodes.length > 0) {
        const first = route.nodes[0]
        const last = route.nodes[route.nodes.length - 1]
        addLabelMarker(first.lat, first.lng, `TRK ${route.truck_id}`, '#2563eb')
        addLabelMarker(last.lat, last.lng, `END ${route.truck_id}`, '#2563eb')
      }
    }
  }

  if (plan.drone_routes) {
    for (const flight of plan.drone_routes) {
      const coords = flight.path.map(([lng, lat]: [number, number]) => [lat, lng] as [number, number])
      if (coords.length > 1) {
        L.polyline(coords, {
          color: '#7c3aed',
          weight: 3,
          opacity: 0.9,
          dashArray: '8,6',
        }).addTo(routeGroup as L.LayerGroup)
      }
    }
  }
}

function drawCurrentDroneRoutes(plan: { drone_routes?: any[] }, currentTime = 0) {
  if (!map) return
  if (runtimePathGroup) {
    runtimePathGroup.clearLayers()
    runtimePathGroup.remove()
    runtimePathGroup = null
  }

  runtimePathGroup = L.layerGroup().addTo(map)

  const routes = Array.isArray(plan.drone_routes) ? plan.drone_routes : []
  const currentRoutes = routes.filter((flight: any) => {
    const launchTime = Number(flight?.launch_time ?? 0)
    const mode = String(flight?.mode ?? '')
    return mode === 'C' || mode === 'B' || launchTime <= currentTime + 1e-6
  })

  const printable = currentRoutes.length > 0 ? currentRoutes : routes.slice(0, 1)
  for (const flight of printable) {
    const rawPath = Array.isArray(flight?.path) ? flight.path : []
    const coords: [number, number][] = rawPath
      .filter((point: any) => Array.isArray(point) && point.length >= 2)
      .map(([lng, lat]: [number, number]) => [Number(lat), Number(lng)] as [number, number])
      .filter((point: [number, number]) => Number.isFinite(point[0]) && Number.isFinite(point[1]))
    if (coords.length <= 1) continue
    const line = L.polyline(coords, {
      color: '#7c3aed',
      weight: 3,
      opacity: 0.9,
      dashArray: '8,6',
      pane: 'droneRoutePane',
    }).addTo(runtimePathGroup as L.LayerGroup)

    const label = [
      `UAV ${flight.drone_id ?? ''}`,
      flight.order_id ? `订单 ${flight.order_id}` : '',
      flight.mode ? `模式 ${flight.mode}` : '',
      coords.length > 2 ? `避障点 ${coords.length}` : '直达',
    ].filter(Boolean).join(' · ')
    line.bindTooltip(label, {
      sticky: true,
      direction: 'top',
      className: 'runtime-path-tooltip',
    })
  }
}

function drawRuntimePaths(paths: { trucks?: any[]; drones?: any[] }) {
  if (!map) return
  if (runtimePathGroup) {
    runtimePathGroup.clearLayers()
    runtimePathGroup.remove()
    runtimePathGroup = null
  }

  runtimePathGroup = L.layerGroup().addTo(map)

  const activeDroneKeys = new Set(
    (paths.drones ?? [])
      .filter((entry: any) => (entry?.status ?? 'active') === 'active')
      .map((entry: any) => runtimeDronePathKey(entry))
      .filter(Boolean)
  )
  for (const key of activeDronePathCache.keys()) {
    if (!activeDroneKeys.has(key)) activeDronePathCache.delete(key)
  }

  const drawPath = (entry: any, color: string, dashed = false, cacheFullDronePath = false) => {
    const rawPath = Array.isArray(entry?.path) ? entry.path : []
    let coords: [number, number][] = rawPath
      .filter((point: any) => Array.isArray(point) && point.length >= 2)
      .map(([lng, lat]: [number, number]) => [Number(lat), Number(lng)] as [number, number])
      .filter((point: [number, number]) => Number.isFinite(point[0]) && Number.isFinite(point[1]))
    if (cacheFullDronePath) {
      const key = runtimeDronePathKey(entry)
      const cached = key ? activeDronePathCache.get(key) : undefined
      if (key && (!cached || pathLength(coords) > pathLength(cached))) {
        activeDronePathCache.set(key, coords.map(([lat, lng]) => [lat, lng] as [number, number]))
      }
      coords = (key ? activeDronePathCache.get(key) : undefined) ?? coords
    }
    if (coords.length <= 1) return
    const line = L.polyline(coords, {
      color,
      weight: dashed ? 3 : 4,
      opacity: dashed ? 0.88 : 0.72,
      dashArray: dashed ? '8,6' : undefined,
      pane: dashed ? 'droneRoutePane' : undefined,
    }).addTo(runtimePathGroup as L.LayerGroup)
    const title = buildRuntimePathLabel(entry, dashed)
    if (title) {
      line.bindTooltip(title, {
        sticky: true,
        direction: 'top',
        className: 'runtime-path-tooltip',
      })
    }
  }

  for (const truckPath of paths.trucks ?? []) {
    if ((truckPath?.status ?? 'active') === 'active') {
      drawPath(truckPath, '#2563eb', false)
    }
  }
  for (const dronePath of paths.drones ?? []) {
    if ((dronePath?.status ?? 'active') === 'active') {
      drawPath(dronePath, '#7c3aed', true, true)
    }
  }
}

function runtimeDronePathKey(entry: any): string {
  const entityId = String(entry?.entity_id ?? '').trim()
  if (!entityId) return ''
  return [
    entityId,
    String(entry?.route_version ?? ''),
    String(entry?.segment_id ?? ''),
    String(entry?.start_time ?? ''),
    String(entry?.end_time ?? ''),
  ].join('|')
}

function pathLength(coords: [number, number][]): number {
  let total = 0
  for (let i = 1; i < coords.length; i += 1) {
    const [lat1, lng1] = coords[i - 1]
    const [lat2, lng2] = coords[i]
    total += Math.hypot(lat2 - lat1, lng2 - lng1)
  }
  return total
}

function buildRuntimePathLabel(entry: any, isDronePath: boolean): string {
  const entity = entry?.entity_id ? String(entry.entity_id) : ''
  const mode = entry?.motion_mode ? String(entry.motion_mode) : ''
  if (isDronePath) {
    const order = entry?.order_id ? ` / 订单 ${entry.order_id}` : ''
    const target = entry?.target_node_id ? ` / 目标 ${entry.target_node_id}` : ''
    return `${entity} ${entry?.segment_id || 'flight'}${order}${target}`
  }
  const from = entry?.from_node_id ? String(entry.from_node_id) : ''
  const to = entry?.to_node_id ? String(entry.to_node_id) : ''
  return `${entity} 卡车路网 ${from} → ${to}${mode ? ` / ${mode}` : ''}`
}

// ── 生命周期 ──────────────────────────────────────────────────────

onMounted(() => {
  initMap()
  if (sceneStore.context) {
    loadLayers(sceneStore.context)
  } else {
    // 无场景时也尝试绘制设施（坐标为 0 的会被跳过）
    drawFacilities()
  }
  drawOrders()
})

onBeforeUnmount(() => {
  depotGroup?.remove()
  stationGroup?.remove()
  orderGroup?.remove()
  runtimePathGroup?.remove()
  map?.remove()
  map = null
})

// ── 监听 sceneStore 变更（跨路由导航时自动刷新地图层）───────────────

watch(() => sceneStore.context, (ctx) => {
  if (ctx && map) loadLayers(ctx)
})

// ── 监听实体变更（redistributeByBounds 或初始化后重绘设施标注）───────

watch(
  () => [entityStore.depots, entityStore.stations],
  () => { if (map) drawFacilities() },
  { deep: true }
)

// ── 监听订单变更（生成/清空后重绘订单标注）────────────────────────

watch(
  () => orderStore.generatedOrders,
  () => { if (map) drawOrders() },
  { deep: true }
)

// ── P2 接口（供 DispatchCenter 父组件调用）────────────────────────────

function setFacilities(_geojson: unknown) { drawFacilities() }

function _truckIcon(name: string, status: string) {
  const statusColor: Record<string, string> = {
    'DRIVING': '#2563eb',   // 蓝色-行驶
    'IDLE': '#7c3aed',      // 紫色-空闲
    'CHARGING': '#dc2626',  // 红色-充电
  }
  const color = statusColor[status] || '#64748b'
  return L.divIcon({
    className: '',
    html: `<div class="dynamic-pin dynamic-pin--truck" style="background:${color}" title="${name}">🚛</div>`,
    iconSize:   [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -18],
  })
}

function _droneIcon(name: string, status: string) {
  const statusColor: Record<string, string> = {
    'FLYING': '#7c3aed',      // 紫色-飞行
    'IDLE': '#94a3b8',        // 灰色-空闲
    'CHARGING': '#f59e0b',    // 橙色-充电
    'LANDING': '#0ea5e9',     // 天蓝-降落
  }
  const color = statusColor[status] || '#64748b'
  return L.divIcon({
    className: '',
    html: `<div class="dynamic-pin dynamic-pin--drone" style="background:${color}" title="${name}">🚁</div>`,
    iconSize:   [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -16],
  })
}

function updateTruck(truckId: string, lng: number, lat: number, status: string = 'IDLE') {
  if (!map) return
  
  if (!truckGroup) {
    truckGroup = L.layerGroup().addTo(map)
  }

  // 从静态配置中找名字
  const truckConfig = entityStore.trucks.find(t => t.truck_id === truckId)
  const name = truckConfig?.name || truckId
  
  // 从运行时数据获取装载信息
  const rtTruck = entityStore.rtTrucks.find(t => t.truck_id === truckId)
  const dockedCount = rtTruck?.docked_drones?.length || 0
  const dockedStr = dockedCount > 0 ? rtTruck!.docked_drones!.join(', ') : '无'

  if (truckMarkers.has(truckId)) {
    const marker = truckMarkers.get(truckId)!
    marker.setLatLng([lat, lng])
    marker.setIcon(_truckIcon(name, status))
    marker.getPopup()?.setContent(
      `<div class="fac-popup">
        <div class="fac-popup__title">🚛 ${name}</div>
        <div class="fac-popup__row"><span>状态</span><span>${status}</span></div>
        <div class="fac-popup__row"><span>速度</span><span>${truckConfig?.speed.toFixed(1) || '—'} m/s</span></div>
        <div class="fac-popup__row"><span>载机数量</span><span>${dockedCount} 架</span></div>
        ${dockedCount > 0 ? `<div class="fac-popup__row"><span>载机ID</span><span style="font-size: 10px; max-width: 140px; word-break: break-all;">${dockedStr}</span></div>` : ''}
        <div class="fac-popup__row"><span>坐标</span><span>${lng.toFixed(5)}, ${lat.toFixed(5)}</span></div>
      </div>`
    )
  } else {
    const marker = L.marker([lat, lng], { icon: _truckIcon(name, status) })
      .bindPopup(
        `<div class="fac-popup">
          <div class="fac-popup__title">🚛 ${name}</div>
          <div class="fac-popup__row"><span>状态</span><span>${status}</span></div>
          <div class="fac-popup__row"><span>速度</span><span>${truckConfig?.speed.toFixed(1) || '—'} m/s</span></div>
          <div class="fac-popup__row"><span>载机数量</span><span>${dockedCount} 架</span></div>
          ${dockedCount > 0 ? `<div class="fac-popup__row"><span>载机ID</span><span style="font-size: 10px; max-width: 140px; word-break: break-all;">${dockedStr}</span></div>` : ''}
          <div class="fac-popup__row"><span>坐标</span><span>${lng.toFixed(5)}, ${lat.toFixed(5)}</span></div>
        </div>`,
        { maxWidth: 220, className: 'fac-popup-wrap' }
      )
      .addTo(truckGroup)
    truckMarkers.set(truckId, marker)
  }
}

function updateDrone(droneId: string, lng: number, lat: number, status: string = 'IDLE') {
  if (!map) return

  if (!droneGroup) {
    droneGroup = L.layerGroup().addTo(map)
  }

  const drone = entityStore.drones.find(d => d.drone_id === droneId)
  const rtDrone = entityStore.rtDrones.find(d => d.drone_id === droneId)
  const name = (rtDrone as any)?.name || (drone as any)?.name || drone?.drone_id || droneId
  const batteryText = rtDrone?.battery_ratio !== undefined
    ? `${(rtDrone.battery_ratio * 100).toFixed(0)}%`
    : '—'
  const popupContent = `<div class="fac-popup">
    <div class="fac-popup__title">🚁 ${name}</div>
    <div class="fac-popup__row"><span>状态</span><span>${status}</span></div>
    <div class="fac-popup__row"><span>电量</span><span>${batteryText}</span></div>
    <div class="fac-popup__row"><span>坐标</span><span>${lng.toFixed(5)}, ${lat.toFixed(5)}</span></div>
  </div>`
  
  // 检查无人机是否在卡车上
  const isOnTruck = entityStore.rtTrucks.some(t => t.docked_drones?.includes(droneId))
  const displayOpacity = isOnTruck ? 0 : 1

  if (droneMarkers.has(droneId)) {
    const marker = droneMarkers.get(droneId)!
    marker.setLatLng([lat, lng])
    marker.setIcon(_droneIcon(name, status))
    marker.setOpacity(displayOpacity)
    marker.getPopup()?.setContent(popupContent)
    const el = marker.getElement()
    if (el) {
      el.style.pointerEvents = isOnTruck ? 'none' : 'auto'
    }
  } else {
    const marker = L.marker([lat, lng], { icon: _droneIcon(name, status), opacity: displayOpacity })
      .bindPopup(
        popupContent,
        { maxWidth: 200, className: 'fac-popup-wrap' }
      )
      .addTo(droneGroup)
    droneMarkers.set(droneId, marker)
    
    // Initial pointer events setup
    requestAnimationFrame(() => {
      const el = marker.getElement()
      if (el) el.style.pointerEvents = isOnTruck ? 'none' : 'auto'
    })
  }
}

function addOrder(_id: string, _lng: number, _lat: number) { /* P2: 订单标记 */ }

function clearDynamic() {
  truckGroup?.clearLayers()
  droneGroup?.clearLayers()
  truckMarkers.clear()
  droneMarkers.clear()
}

function clearDynamicEntities() {
  if (truckGroup) {
    map?.removeLayer(truckGroup)
    truckGroup = null
  }
  if (droneGroup) {
    map?.removeLayer(droneGroup)
    droneGroup = null
  }
  truckMarkers.clear()
  droneMarkers.clear()
}

function getCurrentBounds() {
  console.log('[UnifiedMapView.getCurrentBounds] map:', map)
  if (!map) {
    console.warn('[UnifiedMapView.getCurrentBounds] map 未初始化')
    return null
  }
  const bounds = map.getBounds()
  console.log('[UnifiedMapView.getCurrentBounds] bounds:', bounds)
  return {
    minx: bounds.getWest(),
    miny: bounds.getSouth(),
    maxx: bounds.getEast(),
    maxy: bounds.getNorth(),
  }
}

defineExpose({ setFacilities, updateTruck, updateDrone, addOrder, clearDynamic, clearDispatchRoutes: clearRouteOverlays, drawDispatchRoutes, drawCurrentDroneRoutes, drawRuntimePaths, drawOrders, clearDynamicEntities, getCurrentBounds })
</script>

<style scoped>
.unified-map-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 400px;
  border-radius: inherit;
  overflow: hidden;
}

.unified-map {
  width: 100%;
  height: 100%;
}

/* ── 加载遮罩 ── */
.umap-loading {
  position: absolute;
  inset: 0;
  background: rgba(240, 244, 248, 0.92);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  z-index: 9999;
  font-size: 0.9rem;
  color: #333;
}

.umap-spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #b0c4de;
  border-top-color: #1565c0;
  border-radius: 50%;
  animation: umap-spin 0.9s linear infinite;
}

@keyframes umap-spin { to { transform: rotate(360deg); } }

/* ── 图例 ── */
.umap-legend {
  position: absolute;
  bottom: 28px;
  right: 10px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid #b0c4de;
  border-radius: 8px;
  padding: 10px 14px;
  z-index: 800;
  font-size: 0.78rem;
  color: #333;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.umap-legend__item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.umap-legend__box {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
  border: 1px solid #ccc;
  flex-shrink: 0;
}

.umap-legend__pin {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  font-size: 0.7rem;
  flex-shrink: 0;
  border: 1.5px solid rgba(255, 255, 255, 0.9);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
}

.umap-legend__line {
  display: inline-block;
  width: 20px;
  height: 3px;
  border-radius: 2px;
  flex-shrink: 0;
}

/* ── 设施标注 Pin（DivIcon，使用 :deep 穿透 Leaflet DOM）── */
:deep(.fac-pin) {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
  border: 2.5px solid rgba(255, 255, 255, 0.95);
  cursor: pointer;
  transition: transform 0.15s;
  user-select: none;
}

:deep(.fac-pin:hover) { transform: scale(1.18); }

:deep(.fac-pin--depot) {
  width: 34px;
  height: 34px;
  font-size: 1.15rem;
  background: #1e40af;
}

:deep(.fac-pin--station) {
  width: 28px;
  height: 28px;
  font-size: 0.95rem;
  background: #d97706;
}

/* ── 订单任务点 Pin ── */
:deep(.order-pin) {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  font-size: 0.78rem;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
  border: 2px solid rgba(255, 255, 255, 0.9);
  cursor: pointer;
  transition: transform 0.15s;
  user-select: none;
}

:deep(.order-pin:hover) { transform: scale(1.25); }

:deep(.order-pin--pending)    { background: #6d28d9; }
:deep(.order-pin--assigned)   { background: #0369a1; }
:deep(.order-pin--drone-assigned) { background: #0284c7; }
:deep(.order-pin--pickedup)   { background: #0891b2; }
:deep(.order-pin--delivering) { background: #0284c7; }
:deep(.order-pin--completed)  { background: #16a34a; }
:deep(.order-pin--timeout)    { background: #b91c1c; }
:deep(.order-pin--rejected)   { background: #6b7280; }

/* 紧急订单：橙色环形脉冲 */
:deep(.order-pin--urgent) {
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.45), 0 2px 6px rgba(0,0,0,0.3);
}

/* 订单弹窗标题背景 */
:deep(.order-popup__title) {
  background: #ede9fe;
  border-bottom-color: #c4b5fd;
}

/* ── 设施弹窗内容 ── */
:deep(.fac-popup-wrap .leaflet-popup-content-wrapper) {
  border-radius: 10px;
  padding: 0;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
}

:deep(.fac-popup-wrap .leaflet-popup-content) {
  margin: 0;
}

:deep(.fac-popup) {
  min-width: 180px;
  font-size: 0.82rem;
  color: #1a1a2e;
}

:deep(.fac-popup__title) {
  padding: 8px 12px;
  font-weight: 700;
  font-size: 0.9rem;
  background: #f0f4ff;
  border-bottom: 1px solid #dde6f5;
}

:deep(.fac-popup__row) {
  display: flex;
  justify-content: space-between;
  padding: 4px 12px;
  border-bottom: 1px solid #f0f0f0;
  gap: 12px;
}

:deep(.fac-popup__row:last-child) { border-bottom: none; }
:deep(.fac-popup__row span:first-child) { color: #666; }
:deep(.fac-popup__row span:last-child) { font-weight: 600; color: #1a1a2e; }

:deep(.runtime-path-tooltip) {
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  color: #0f172a;
  font-size: 0.75rem;
  font-weight: 600;
}

/* ── 动态实体标记（卡车和无人机）── */
:deep(.dynamic-pin) {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 0 2px rgba(255, 255, 255, 0.95);
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  user-select: none;
  font-weight: 600;
  animation: pulse 2s infinite;
}

:deep(.dynamic-pin:hover) {
  transform: scale(1.25);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 0 3px rgba(255, 255, 255, 1);
}

:deep(.dynamic-pin--truck) {
  font-size: 1.1rem;
}

:deep(.dynamic-pin--drone) {
  font-size: 1rem;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}
</style>
