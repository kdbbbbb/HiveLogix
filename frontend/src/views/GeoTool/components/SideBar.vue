<template>
  <div id="sidebar">

    <!-- ── 数据集信息 ── -->
    <div class="card-dark" id="info-card">
      <h6>📊 数据集信息</h6>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="n">{{ serverState?.total?.toLocaleString() ?? '—' }}</div>
          <div class="l">总建筑数</div>
        </div>
        <div class="stat-item">
          <div class="n" style="font-size:.9rem">{{ serverState?.height_column ?? '—' }}</div>
          <div class="l">高度字段</div>
        </div>
        <div class="stat-item">
          <div class="n val-red">{{ serverState?.height_stats?.max ?? '—' }}</div>
          <div class="l">最高 (m)</div>
        </div>
        <div class="stat-item">
          <div class="n val-grn">{{ serverState?.height_stats?.mean ?? '—' }}</div>
          <div class="l">均值 (m)</div>
        </div>
      </div>
    </div>

    <!-- ── 选区工具 ── -->
    <div class="card-dark">
      <h6>🗺 选取研究范围</h6>
      <div class="d-flex gap-2 mb-2">
        <button class="btn-outline-light-sm"
                :class="{ active: activeMode === 'draw', inactive: activeMode === 'quick' }"
                @click="handleDraw">✏ 手动框选</button>
        <button class="btn-outline-light-sm"
                :class="{ active: activeMode === 'quick', inactive: activeMode === 'draw' }"
                @click="toggleQuick">📍 中心点选区</button>
      </div>
      <!-- 快速选区参数 -->
      <div v-show="quickMode" style="margin-top:8px;">
        <span class="label-sm">点击地图设置中心，范围（米）：</span>
        <div class="d-flex gap-2">
          <div style="flex:1">
            <span class="label-sm">宽 (E-W)</span>
            <input type="number" v-model.number="qw" min="100" max="20000" step="100"
                   class="form-control-dark" style="width:100%">
          </div>
          <div style="flex:1">
            <span class="label-sm">高 (N-S)</span>
            <input type="number" v-model.number="qh" min="100" max="20000" step="100"
                   class="form-control-dark" style="width:100%">
          </div>
        </div>
      </div>
      <!-- 当前选区信息 -->
      <div v-if="selBounds" style="margin-top:8px;">
        <div style="font-size:.75rem; color:#aaa; margin-bottom:4px;">
          已选区域：<span style="color:#7ec8e3;font-weight:600">{{ selSizeText }}</span>
        </div>
        <table class="coord-table">
          <tbody>
            <tr><td>↙ 西南角 (SW)</td><td>{{ coords.sw }}</td></tr>
            <tr><td>↗ 东北角 (NE)</td><td>{{ coords.ne }}</td></tr>
            <tr><td>↙ 西北角 (NW)</td><td>{{ coords.nw }}</td></tr>
            <tr><td>↗ 东南角 (SE)</td><td>{{ coords.se }}</td></tr>
            <tr><td>⊙ 中心点</td><td>{{ coords.ctr }}</td></tr>
          </tbody>
        </table>
        <span class="copy-btn" @click="copyCoords">{{ copyLabel }}</span>
      </div>
      <button v-if="selBounds" class="btn-accent mt-2"
              style="opacity:.7; font-weight:400; font-size:.78rem; padding:5px;"
              @click="$emit('clear')">✕ 清除选区</button>
    </div>

    <!-- ── 高度阈值设置 ── -->
    <div class="card-dark">
      <h6>⚡ 高度阈值设置</h6>
      <span class="label-sm">无人机飞行高度 AGL (m) — 超过此值为禁飞区</span>
      <div class="d-flex align-items-center gap-2" style="margin-top:6px;">
        <input type="range" v-model.number="threshold" min="10" max="500" step="5" style="flex:1">
        <input type="number" v-model.number="threshold" min="0" max="9999"
               class="form-control-dark" style="width:70px; text-align:center;">
        <span style="font-size:.8rem; color:#aaa;">m</span>
      </div>
      <div class="range-labels"><span>10m</span><span>500m</span></div>
      <div style="margin-top:8px; font-size:.75rem; color:#aaa; line-height:1.5">
        常见场景：城区低空 <b style="color:#f8961e">50m</b> ·
        一般巡航 <b style="color:#f8961e">120m</b> ·
        高空飞行 <b style="color:#f8961e">300m</b>
      </div>
      <hr style="border-color:#1a4480; margin:10px 0">
      <span class="label-sm">高度字段（自动检测，可手动切换）</span>
      <select v-model="hCol" class="form-select-dark mt-1" style="width:100%">
        <option v-if="!serverState?.numeric_columns?.length" value="">— 加载中 —</option>
        <option v-for="c in serverState?.numeric_columns" :key="c" :value="c">{{ c }}</option>
      </select>
    </div>

    <!-- ── 查询按钮 ── -->
    <button class="btn-accent" :disabled="!selBounds || !appReady" @click="$emit('query')">
      🔍 分析选区建筑
    </button>

    <!-- ── 统计结果 ── -->
    <div class="card-dark" v-if="queryResult">
      <h6>📈 查询结果</h6>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="n">{{ queryResult.total.toLocaleString() }}</div>
          <div class="l">选区建筑</div>
        </div>
        <div class="stat-item">
          <div class="n">{{ queryResult.shown.toLocaleString() }}</div>
          <div class="l">已显示</div>
        </div>
        <div class="stat-item">
          <div class="n val-red">{{ queryResult.no_fly.toLocaleString() }}</div>
          <div class="l">🔴 禁飞区</div>
        </div>
        <div class="stat-item">
          <div class="n val-grn">{{ queryResult.fly.toLocaleString() }}</div>
          <div class="l">🟢 可飞区</div>
        </div>
      </div>
      <div v-if="queryResult.truncated"
           style="margin-top:6px; font-size:.72rem; color:#f8961e;">
        ⚠ 建筑数量超限，仅显示部分结果。导出时将包含所有数据。
      </div>
    </div>

    <!-- ── 导出 ── -->
    <div class="card-dark" v-if="queryResult">
      <h6>💾 导出数据</h6>
      <span class="label-sm">选择格式</span>
      <div class="d-flex gap-1 mt-1 mb-2 flex-wrap">
        <button v-for="f in fmtList" :key="f.value"
                class="btn-outline-light-sm"
                :class="{ active: currentFmt === f.value }"
                @click="selectFmt(f.value)">
          {{ f.label }}
        </button>
      </div>
      <!-- 网格间距 -->
      <div v-if="currentFmt === 'sumo_zip'" style="margin-bottom:8px;">
        <span class="label-sm">路网网格间距（米）</span>
        <div class="d-flex align-items-center gap-2">
          <input type="number" v-model.number="localGridSpacing" min="50" max="1000" step="50"
                 class="form-control-dark" style="width:90px; text-align:center;">
          <span style="font-size:.75rem; color:#888;">m 选择范围 50–1000</span>
        </div>
      </div>
      <div style="font-size:.72rem; color:#888; margin-bottom:10px; line-height:1.5"
           v-html="fmtDesc"></div>
      <button class="btn-accent" @click="$emit('export')">⬇ 下载导出文件</button>
    </div>

    <!-- ── 导出到指挥中心地图 ── -->
    <div class="card-dispatch" v-if="queryResult">
      <div class="card-dispatch__header">
        <span class="card-dispatch__icon">🚀</span>
        <div>
          <div class="card-dispatch__title">导出到指挥中心地图</div>
          <div class="card-dispatch__sub">对齐路网与禁飞区，建立仿真底图</div>
        </div>
      </div>
      <!-- 错误提示 -->
      <div v-if="dispatchError" class="card-dispatch__error">
        ❌ {{ dispatchError }}
      </div>
      <button
        class="btn-dispatch"
        :disabled="dispatchLoading"
        @click="$emit('export-to-dispatch')"
      >
        <span v-if="dispatchLoading" class="btn-dispatch__spinner"></span>
        <span v-else>🗺️ 前往实时指挥中心</span>
      </button>
    </div>

    <!-- ── 使用说明 ── -->
    <div class="card-dark" style="font-size:.72rem; color:#888; line-height:1.6">
      <h6>📖 使用步骤</h6>
      1. 等待数据加载完成<br>
      2. 用"手动框选"或"中心点选区"圈定研究范围<br>
      3. 调整飞行高度阈值<br>
      4. 点击"分析选区建筑"<br>
      5. 查看红色禁飞区 / 绿色可飞区<br>
      6. 选择格式并下载导出文件<br><br>
      <span style="color:#7ec8e3">SUMO 加载示例：</span><br>
      <code style="color:#f8961e; font-size:.7rem">
        sumo-gui -n grid.net.xml<br>
        &nbsp;&nbsp;--additional-files no_fly_zones.add.xml
      </code>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

// ── Props & Emits ──────────────────────────────────────────────────

interface ServerState {
  total:           number
  height_column:   string
  numeric_columns: string[]
  height_stats:    { max: number; mean: number } | null
}
interface SelBounds { minx: number; miny: number; maxx: number; maxy: number }
interface QueryStats { total: number; shown: number; no_fly: number; fly: number; truncated: boolean }

const props = defineProps<{
  serverState:    ServerState | null
  selBounds:      SelBounds | null
  queryResult:    QueryStats | null
  appReady:       boolean
  dispatchLoading?: boolean
  dispatchError?:   string | null
}>()

const emit = defineEmits<{
  (e: 'query'): void
  (e: 'export'): void
  (e: 'clear'): void
  (e: 'draw'): void
  (e: 'quick-mode', on: boolean, qw: number, qh: number): void
  (e: 'export-to-dispatch'): void
  (e: 'height-change', v: number): void
  (e: 'fmt-change', v: string): void
  (e: 'hcol-change', v: string | null): void
  (e: 'grid-spacing-change', v: number): void
}>()

// ── 本地状态 ──────────────────────────────────────────────────────

const quickMode         = ref(false)
const activeMode        = ref<'draw' | 'quick' | null>(null)
const qw                = ref(4000)
const qh                = ref(4000)
const threshold         = ref(120)
const hCol              = ref<string>('')
const currentFmt        = ref('sumo_zip_osm')
const localGridSpacing  = ref(200)
const copyLabel         = ref('📋 复制四角坐标')

// ── 向父组件同步变化 ──────────────────────────────────────────────

watch(threshold,        v => emit('height-change', v))
watch(currentFmt,       v => emit('fmt-change', v))
watch(hCol,             v => emit('hcol-change', v || null))
watch(localGridSpacing, v => emit('grid-spacing-change', v))

// 绘制完成后（selBounds 变化）重置手动框选状态
watch(() => props.selBounds, (val) => {
  if (val && activeMode.value === 'draw') {
    activeMode.value = null
  }
})

// ── 快速模式切换（通知父/MapView） ───────────────────────────────

function handleDraw() {
  activeMode.value = 'draw'
  quickMode.value  = false
  emit('quick-mode', false, qw.value, qh.value)
  emit('draw')
}

function toggleQuick() {
  if (activeMode.value === 'quick') {
    activeMode.value = null
    quickMode.value  = false
  } else {
    activeMode.value = 'quick'
    quickMode.value  = true
  }
  emit('quick-mode', quickMode.value, qw.value, qh.value)
}

// ── 坐标计算（用于展示） ─────────────────────────────────────────

const LAT_M = 1 / 111000

function fmtCoord(lat: number, lng: number) {
  return `${lat.toFixed(6)}°N,  ${lng.toFixed(6)}°E`
}

const coords = computed(() => {
  const b = props.selBounds
  if (!b) return { sw: '—', ne: '—', nw: '—', se: '—', ctr: '—' }
  return {
    sw:  fmtCoord(b.miny, b.minx),
    ne:  fmtCoord(b.maxy, b.maxx),
    nw:  fmtCoord(b.maxy, b.minx),
    se:  fmtCoord(b.miny, b.maxx),
    ctr: fmtCoord((b.miny + b.maxy) / 2, (b.minx + b.maxx) / 2),
  }
})

const selSizeText = computed(() => {
  const b = props.selBounds
  if (!b) return ''
  const cLat = (b.miny + b.maxy) / 2
  const wM   = Math.round((b.maxx - b.minx) * 111000 * Math.cos(cLat * Math.PI / 180))
  const hM   = Math.round((b.maxy - b.miny) * 111000)
  return `${(wM / 1000).toFixed(2)} km × ${(hM / 1000).toFixed(2)} km`
})

// ── 复制坐标 ──────────────────────────────────────────────────────

function copyCoords() {
  const b = props.selBounds
  if (!b) return
  const cLat = (b.miny + b.maxy) / 2
  const cLon = (b.minx + b.maxx) / 2
  const text =
    `西南角 (SW): ${b.miny.toFixed(6)}°N, ${b.minx.toFixed(6)}°E\n` +
    `东北角 (NE): ${b.maxy.toFixed(6)}°N, ${b.maxx.toFixed(6)}°E\n` +
    `西北角 (NW): ${b.maxy.toFixed(6)}°N, ${b.minx.toFixed(6)}°E\n` +
    `东南角 (SE): ${b.miny.toFixed(6)}°N, ${b.maxx.toFixed(6)}°E\n` +
    `中心点 (CTR): ${cLat.toFixed(6)}°N, ${cLon.toFixed(6)}°E\n---\n` +
    `minLon=${b.minx.toFixed(6)}  maxLon=${b.maxx.toFixed(6)}\n` +
    `minLat=${b.miny.toFixed(6)}  maxLat=${b.maxy.toFixed(6)}`
  navigator.clipboard.writeText(text).then(() => {
    copyLabel.value = '✓ 已复制'
    setTimeout(() => { copyLabel.value = '📋 复制四角坐标' }, 2000)
  })
}

// ── 格式列表 ──────────────────────────────────────────────────────

const fmtList = [
  { value: 'sumo_zip_osm', label: 'SUMO 真实路网 (.zip) 🌟' },
  { value: 'sumo_zip',     label: 'SUMO 网格路网 (.zip)' },
  { value: 'sumo_poly',    label: '仅 .add.xml' },
  { value: 'geojson',      label: 'GeoJSON' },
  { value: 'csv',          label: 'CSV' },
]

const fmtDescs: Record<string, string> = {
  sumo_zip_osm: '下载选区内真实 OSM 道路，包含 <b style="color:#7ec8e3">roads.net.xml</b>（真实道路）+<b style="color:#7ec8e3">no_fly_zones.add.xml</b>，解压后直接运行。需联网（Overpass API）。',
  sumo_zip:     '纯 Python 自动生成简化<b style="color:#7ec8e3">网格路网</b>+禁飞区叠加层，无需联网。',
  sumo_poly:    'SUMO polygon additional file，需配合已有 net.xml 使用。',
  geojson:      'GeoJSON 格式，包含建筑高度与禁飞标记，适用于 QGIS / 网页地图。',
  csv:          'CSV 表格，含每栋建筑质心坐标、高度、是否禁飞，便于数据分析。',
}

const fmtDesc = computed(() => fmtDescs[currentFmt.value] ?? '')

function selectFmt(v: string) {
  currentFmt.value = v
}
</script>
