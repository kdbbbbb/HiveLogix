<template>
  <!-- 加载遮罩 -->
  <LoadingMask
    :progress="loadProgress"
    :sub-text="loadSubText"
    :hidden="appReady"
  />

  <!-- 应用主体 -->
  <div class="geo-app">
    <!-- 顶部栏 -->
    <TopBar :status="statusInfo" />

    <!-- 主体 -->
    <div id="main">
      <!-- 侧边栏 -->
      <SideBar
        :server-state="serverState"
        :sel-bounds="selBounds"
        :query-result="queryResult"
        :app-ready="appReady"
        @query="handleQuery"
        @export="handleExport"
        @clear="handleClear"
        @draw="handleDraw"
        @quick-mode="handleQuickMode"
        @height-change="currentThreshold = $event"
        @fmt-change="currentFmt = $event"
        @hcol-change="currentHCol = $event"
        @grid-spacing-change="gridSpacing = $event"
        @export-to-dispatch="handleExportToDispatch"
        :dispatch-loading="sceneStore.loading"
        :dispatch-error="sceneStore.error"
      />

      <!-- 地图区域 -->
      <MapView
        ref="mapRef"
        :overlay-text="overlayText"
        @selection="handleSelection"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import LoadingMask from './components/LoadingMask.vue'
import TopBar      from './components/TopBar.vue'
import SideBar     from './components/SideBar.vue'
import MapView     from './components/MapView.vue'
import { useSceneStore } from '@/stores/scene'
import { useEntityStore } from '@/stores/entity'
import './geo_map.css'

const sceneStore  = useSceneStore()
const entityStore = useEntityStore()
const router      = useRouter()

// ── 类型 ──────────────────────────────────────────────────────────

interface SelBounds { minx: number; miny: number; maxx: number; maxy: number }

interface ServerState {
  loaded:          boolean
  loading:         boolean
  progress:        number
  total:           number
  height_column:   string
  numeric_columns: string[]
  height_stats:    { max: number; mean: number } | null
  bounds:          { minx: number; miny: number; maxx: number; maxy: number; center_lon: number; center_lat: number } | null
  error:           string | null
}

interface QueryStats {
  total: number; shown: number; no_fly: number; fly: number; truncated: boolean
}

// ── 响应式状态 ──────────────────────────────────────────────────────

const appReady        = ref(false)
const loadProgress    = ref(0)
const loadSubText     = ref('请稍候，文件较大（~240 MB）约需 30–60 秒')
const serverState     = ref<ServerState | null>(null)
const selBounds       = ref<SelBounds | null>(null)
const queryResult     = ref<QueryStats | null>(null)
const overlayText     = ref('请先选取研究范围')
const currentThreshold = ref(120)
const currentFmt      = ref('sumo_zip_osm')
const currentHCol     = ref<string | null>(null)
const gridSpacing     = ref(200)
// 保存最近一次 /api/geo/query 的建筑 GeoJSON，供「导出到指挥中心」时递传给 sceneStore
// 避免 UnifiedMapView 挂载时重复请求。类型宽松为 any，因为 fetch 返回本就是 untyped JSON
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const lastBuildingsGeoJSON = ref<any>(null)

const statusInfo = ref({ dot: '', text: '连接中…' })

const mapRef = ref<InstanceType<typeof MapView> | null>(null)

// ── 轮询后端状态 ──────────────────────────────────────────────────

let pollTimer: ReturnType<typeof setInterval> | null = null

async function pollStatus() {
  try {
    const res  = await fetch('/api/geo/status')
    const s: ServerState = await res.json()
    serverState.value  = s
    loadProgress.value = s.progress

    if (s.error) {
      loadSubText.value  = '❌ 加载失败：' + s.error
      statusInfo.value   = { dot: '', text: '加载失败' }
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
      return
    }

    if (s.loaded) {
      appReady.value    = true
      statusInfo.value  = { dot: 'ok', text: `就绪 · ${s.total.toLocaleString()} 栋建筑` }
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
      mapRef.value?.onDataReady(s)
    } else if (s.loading) {
      statusInfo.value  = { dot: 'spin', text: `加载中 ${s.progress}%…` }
      loadSubText.value = `已处理 ${s.progress}%，请稍候`
    }
  } catch {
    statusInfo.value = { dot: '', text: '连接失败' }
  }
}

// ── 事件处理 ──────────────────────────────────────────────────────

function handleSelection(bounds: SelBounds) {
  selBounds.value = bounds
}

async function handleQuery() {
  if (!selBounds.value) return
  overlayText.value = '正在分析建筑数据…'
  try {
    const res = await fetch('/api/geo/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        ...selBounds.value,
        threshold:     currentThreshold.value,
        height_column: currentHCol.value,
        max:           30000,
      }),
    })
    const geojson = await res.json()
    if (geojson.error) throw new Error(geojson.error)

    queryResult.value = geojson.stats
    // 缓存建筑 GeoJSON，供后续「导出到指挥中心」直接使用
    lastBuildingsGeoJSON.value = geojson
    mapRef.value?.renderBuildings(geojson)
    mapRef.value?.fetchRoads(selBounds.value)

    const s = geojson.stats as QueryStats
    const pct = s.total > 0 ? ((s.no_fly / s.total) * 100).toFixed(1) : 0
    overlayText.value =
      `共 ${s.total.toLocaleString()} 栋建筑 ｜ 禁飞区 ${s.no_fly.toLocaleString()} (${pct}%) · 可飞区 ${s.fly.toLocaleString()}`
  } catch (e: unknown) {
    overlayText.value = '❌ 查询失败：' + (e instanceof Error ? e.message : String(e))
  }
}

function handleClear() {
  selBounds.value   = null
  queryResult.value = null
  overlayText.value = '请先选取研究范围'
  mapRef.value?.clearSelection()
}

function handleDraw() {
  // 触发 Leaflet 画矩形工具（由 MapView 暴露）
  mapRef.value?.startDraw()
}

function handleQuickMode(on: boolean, qw: number, qh: number) {
  mapRef.value?.setQuickMode(on, qw, qh)
  if (on) {
    overlayText.value = '点击地图设置中心点'
  } else if (!selBounds.value) {
    overlayText.value = '请先选取研究范围'
  }
}

async function handleExport() {
  if (!selBounds.value) return
  try {
    const res = await fetch('/api/geo/export', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        ...selBounds.value,
        threshold:     currentThreshold.value,
        height_column: currentHCol.value,
        format:        currentFmt.value,
        grid_spacing:  gridSpacing.value,
      }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.error ?? res.statusText)
    }
    const blob  = await res.blob()
    const cd    = res.headers.get('Content-Disposition') ?? ''
    const match = cd.match(/filename="?([^"]+)"?/)
    const fname = match ? match[1] : 'export_file'
    const url   = URL.createObjectURL(blob)
    const a     = document.createElement('a')
    a.href = url; a.download = fname; a.click()
    URL.revokeObjectURL(url)
  } catch (e: unknown) {
    alert('导出失败：' + (e instanceof Error ? e.message : String(e)))
  }
}

async function handleExportToDispatch() {
  if (!selBounds.value) return
  // 1. 请求后端构建场景（会内部清空旧缓存）
  await sceneStore.prepareScene({
    sel_bounds:    selBounds.value,
    threshold:     currentThreshold.value,
    height_column: currentHCol.value ?? null,
  })
  if (!sceneStore.error && sceneStore.context) {
    // 2. 将本次查询的建筑 GeoJSON 写入缓存——
    //    必须在 prepareScene 成功后设置，因为 prepareScene 会先清空旧缓存
    sceneStore.setBuildingsGeoJSON(lastBuildingsGeoJSON.value)
    // 3. 根据新地图 bbox 重新均匀分配仓库和充换电站坐标
    entityStore.redistributeByBounds(sceneStore.context.road_network.bounds)
    router.push('/dispatch')
  }
}

// ── 生命周期 ──────────────────────────────────────────────────────

onMounted(() => {
  pollTimer = setInterval(pollStatus, 1500)
  pollStatus()
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
