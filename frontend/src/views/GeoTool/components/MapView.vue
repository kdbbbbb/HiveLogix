<template>
  <div id="map-wrap">
    <div id="map" ref="mapEl"></div>
    <div id="map-overlay">{{ overlayText }}</div>
    <div id="legend">
      <div style="font-size:.72rem; color:#1565c0; font-weight:600; margin-bottom:6px;">图例</div>
      <div class="legend-item">
        <div class="legend-box" style="background:rgba(255,0,0,.65)"></div>禁飞区（高于阈值）
      </div>
      <div class="legend-item">
        <div class="legend-box" style="background:rgba(0,180,0,.45)"></div>可飞区（低于阈值）
      </div>
      <div class="legend-item">
        <div class="legend-box" style="background:#e6a817; height:6px; border-radius:3px; border:none; margin-top:5px;"></div>OSM 道路
      </div>
      <div class="legend-item">
        <div class="legend-box" style="background:rgba(255,165,0,.4)"></div>选区边界
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

// ── 类型声明（Leaflet 通过 CDN 全局加载） ────────────────────────
declare const L: typeof import('leaflet')

interface SelBounds { minx: number; miny: number; maxx: number; maxy: number }

interface DataReadyState {
  bounds: { minx: number; miny: number; maxx: number; maxy: number; center_lon: number; center_lat: number }
  total:  number
  height_column: string
  height_stats:  { max: number; mean: number } | null
}

// ── Props & Emits ─────────────────────────────────────────────────

const props = defineProps<{ overlayText: string }>()
const emit  = defineEmits<{ (e: 'selection', bounds: SelBounds): void }>()

// ── Leaflet 实例 ──────────────────────────────────────────────────

const mapEl       = ref<HTMLDivElement | null>(null)
const LAT_M       = 1 / 111000
const colorNoFly  = '#ff4444'
const colorFly    = '#33bb55'

let map:           ReturnType<typeof L.map> | null         = null
let drawControl:   L.Control.Draw | null                   = null
let drawnItems:    L.FeatureGroup | null                   = null
let buildingLayer: L.GeoJSON | null                        = null
let roadLayer:     L.GeoJSON | null                        = null
let selRect:       L.Rectangle | null                      = null
let quickMode     = false
let qw            = 4000
let qh            = 4000

// ── 初始化地图 ────────────────────────────────────────────────────

function initMap(centerLon = 121.47, centerLat = 31.23) {
  if (!mapEl.value || map) return
  map = L.map(mapEl.value, { center: [centerLat, centerLon], zoom: 13, zoomControl: true, preferCanvas: true })

  // ── Leaflet.draw 汉化 ──────────────────────────────────────────
  L.drawLocal.draw.toolbar.buttons.rectangle        = '框选矩形区域'
  L.drawLocal.draw.toolbar.actions.title            = '取消绘制'
  L.drawLocal.draw.toolbar.actions.text             = '取消'
  L.drawLocal.draw.toolbar.finish.title             = '完成绘制'
  L.drawLocal.draw.toolbar.finish.text              = '完成'
  L.drawLocal.draw.toolbar.undo.title               = '撤销上一个点'
  L.drawLocal.draw.toolbar.undo.text                = '撤销'
  L.drawLocal.draw.handlers.rectangle.tooltip.start = '点击并拖拽以框选研究范围'
  L.drawLocal.edit.toolbar.actions.save.title       = '保存修改'
  L.drawLocal.edit.toolbar.actions.save.text        = '保存'
  L.drawLocal.edit.toolbar.actions.cancel.title     = '取消修改，放弃更改'
  L.drawLocal.edit.toolbar.actions.cancel.text      = '取消'
  L.drawLocal.edit.toolbar.actions.clearAll.title   = '清除所有图层'
  L.drawLocal.edit.toolbar.actions.clearAll.text    = '全部清除'
  L.drawLocal.edit.toolbar.buttons.edit             = '编辑选区'
  L.drawLocal.edit.toolbar.buttons.editDisabled     = '无可编辑的图层'
  L.drawLocal.edit.toolbar.buttons.remove           = '删除图层'
  L.drawLocal.edit.toolbar.buttons.removeDisabled   = '无可删除的图层'
  L.drawLocal.edit.handlers.edit.tooltip.text       = '拖动控制点以编辑要素'
  L.drawLocal.edit.handlers.edit.tooltip.subtext    = '点击"取消"放弃更改'
  L.drawLocal.edit.handlers.remove.tooltip.text     = '点击要素以删除'

  const positron = L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { attribution: '© CARTO · © OpenStreetMap 贡献者', maxZoom: 19 }
  )
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap 贡献者', maxZoom: 19
  })
  positron.addTo(map)
  L.control.layers({ '浅色底图 CartoDB': positron, 'OpenStreetMap': osm }, {}, { collapsed: true }).addTo(map)

  drawnItems = new L.FeatureGroup()
  map.addLayer(drawnItems)

  drawControl = new L.Control.Draw({
    draw: {
      rectangle: { shapeOptions: { color: '#f8961e', weight: 2, fillOpacity: 0.08 } },
      polygon: false, polyline: false, circle: false,
      circlemarker: false, marker: false,
    },
    edit: { featureGroup: drawnItems, remove: false },
  })
  map.addControl(drawControl)

  map.on(L.Draw.Event.CREATED, (e: L.DrawEvents.Created) => {
    setSelection((e.layer as L.Rectangle).getBounds())
  })

  map.on('click', (e: L.LeafletMouseEvent) => {
    if (!quickMode) return
    const lat   = e.latlng.lat
    const lon   = e.latlng.lng
    const dLat  = (qh / 2) * LAT_M
    const dLon  = (qw / 2) * LAT_M / Math.cos(lat * Math.PI / 180)
    setSelection(L.latLngBounds([lat - dLat, lon - dLon], [lat + dLat, lon + dLon]))
  })
}

// ── 设置选区 ──────────────────────────────────────────────────────

function setSelection(lBounds: L.LatLngBounds) {
  if (!map) return
  if (selRect) map.removeLayer(selRect)

  selRect = L.rectangle(lBounds, {
    color: '#f8961e', weight: 2, fillColor: '#f8961e', fillOpacity: 0.06,
  }).addTo(map)

  const sw = lBounds.getSouthWest()
  const ne = lBounds.getNorthEast()

  emit('selection', { minx: sw.lng, miny: sw.lat, maxx: ne.lng, maxy: ne.lat })

  if (quickMode) setQuickMode(false)
}

// ── 快速模式 (由父组件调用) ───────────────────────────────────────

function setQuickMode(on: boolean, newQw?: number, newQh?: number) {
  quickMode = on
  if (newQw !== undefined) qw = newQw
  if (newQh !== undefined) qh = newQh
  if (!map) return
  map.getContainer().style.cursor = on ? 'crosshair' : ''
}

// ── 渲染建筑层（由父组件调用）────────────────────────────────────

function renderBuildings(geojson: GeoJSON.FeatureCollection) {
  if (!map) return
  if (buildingLayer) map.removeLayer(buildingLayer)

  buildingLayer = L.geoJSON(geojson, {
    style: (feature) => {
      const nf = feature?.properties?.nf
      return {
        color:       nf ? '#cc0000' : '#008800',
        weight:      0.6,
        fillColor:   nf ? colorNoFly : colorFly,
        fillOpacity: nf ? 0.70 : 0.35,
      }
    },
    // onEachFeature 已移除：Canvas 模式下 3 万个独立监听器会占用大量内存并連带 mouseover 卡顿
  }).addTo(map)

  // 事件委托：图层级单次 click，动态生成 Popup，替代逐个 Feature 绑定
  buildingLayer.on('click', (e) => {
    const evt = e as unknown as L.LeafletMouseEvent & {
      layer?: L.Layer & { feature?: GeoJSON.Feature }
    }
    const p = evt.layer?.feature?.properties
    if (!p || !map) return
    L.popup({ maxWidth: 180 })
      .setLatLng(evt.latlng)
      .setContent(
        `<div class="popup-h">${p.h} m</div>` +
        `<span class="popup-tag ${p.nf ? 'nf' : 'ok'}">` +
        `${p.nf ? '🔴 禁飞区' : '🟢 可飞区'}</span>`
      )
      .openOn(map)
  })

  if (selRect) map.fitBounds(selRect.getBounds(), { padding: [20, 20] })
}

// ── 道路叠加层（由父组件调用）────────────────────────────────────

async function fetchRoads(bounds: SelBounds) {
  if (!map) return
  if (roadLayer) { map.removeLayer(roadLayer); roadLayer = null }
  try {
    const res = await fetch('/api/geo/roads', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(bounds),
    })
    const gj = await res.json()
    if (gj.error) { console.warn('[roads]', gj.error); return }

    const widths: Record<string, number> = {
      motorway:5, trunk:4, primary:3.5, secondary:3,
      tertiary:2.5, residential:2, service:1.5, unclassified:2,
      motorway_link:2.5, trunk_link:2, primary_link:2,
      secondary_link:1.5, tertiary_link:1.5, living_street:1.5, road:2,
    }
    roadLayer = L.geoJSON(gj, {
      style: (f) => ({ color: '#e6a817', weight: widths[f?.properties?.highway] ?? 2, opacity: 0.9 }),
      onEachFeature: (f, layer) => {
        const n = f.properties?.name
        if (n) layer.bindTooltip(
          `${n} <span style="color:#888">(${f.properties?.highway})</span>`,
          { sticky: true, opacity: 0.92 }
        )
      },
    }).addTo(map)
    roadLayer.bringToBack()
  } catch (e) {
    console.warn('[roads] 下载失败:', e)
  }
}

// ── 清除（由父组件调用）──────────────────────────────────────────

function clearSelection() {
  if (!map) return
  if (selRect)       { map.removeLayer(selRect);       selRect       = null }
  if (buildingLayer) { map.removeLayer(buildingLayer); buildingLayer = null }
  if (roadLayer)     { map.removeLayer(roadLayer);     roadLayer     = null }
}

// ── 数据就绪（由父组件调用）──────────────────────────────────────

function onDataReady(s: DataReadyState) {
  const b = s.bounds
  if (!map) {
    initMap(b.center_lon, b.center_lat)
  }
  const dataLatLngBounds = L.latLngBounds([b.miny, b.minx], [b.maxy, b.maxx])
  map!.fitBounds(dataLatLngBounds, { padding: [30, 30] })

  L.rectangle(dataLatLngBounds, {
    color: '#7ec8e3', weight: 1, dashArray: '4,6', fill: false, interactive: false,
  }).addTo(map!).bindTooltip(
    `数据集范围<br>SW: ${b.miny.toFixed(4)}°N, ${b.minx.toFixed(4)}°E<br>` +
    `NE: ${b.maxy.toFixed(4)}°N, ${b.maxx.toFixed(4)}°E`,
    { sticky: true }
  )
}

// ── 暴露给父组件 ──────────────────────────────────────────────────

function startDraw() {
  if (!map || !drawControl) return
  new L.Draw.Rectangle(map, (drawControl.options as L.Control.DrawConstructorOptions).draw?.rectangle ?? {}).enable()
}

defineExpose({ renderBuildings, fetchRoads, clearSelection, onDataReady, setQuickMode, startDraw })

// ── 挂载后初始化地图（默认上海） ──────────────────────────────────

onMounted(() => {
  initMap(121.47, 31.23)
})
</script>
