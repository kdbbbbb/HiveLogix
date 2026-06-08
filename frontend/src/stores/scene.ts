/**
 * sceneStore — 仿真场景上下文状态
 *
 * 职责：
 *   作为 SimulationBox/GeoTool 与 DispatchCenter 之间的唯一数据桥梁。
 *   两端视图不直接 import 对方，通过本 store 传递 SceneContext。
 *
 * 数据流：
 *   GeoTool 完成分析后 → prepareScene() → sceneStore.context 写入
 *   router.push('/dispatch') → DispatchCenter 读取 sceneStore.context
 *   DispatchCenter UnifiedMapView 挂载时并行请求建筑/路网数据
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import { http } from '@/services/http'
import type { FeatureCollection } from 'geojson'

// ── 类型定义（与后端 SceneContext 协议对齐）─────────────────────────────────

/** UTM 原点偏移量，用于 physics engine 还原 SUMO 坐标 */
export interface NetOffset {
  /** UTM 东向原点（米），对应选区左下角经度 */
  ox: number
  /** UTM 北向原点（米），对应选区左下角纬度 */
  oy: number
}

/** 场景元数据（轻量级，随 /api/scene/prepare 响应返回） */
export interface SceneMeta {
  road_source:  'osm' | 'grid'
  road_nodes:   number
  road_edges:   number
  created_at:   string
  utm_zone:     number
  utm_band:     string
  net_offset:   NetOffset
}

/** 路网节点（WGS84 原始经纬度） */
export interface RoadNode {
  id:  string
  lng: number
  lat: number
}

/** 路网边（折线 shape，WGS84 经纬度序列） */
export interface RoadEdge {
  shape: [number, number][]   // [lng, lat] 序列
}

/** 路网地理范围 */
export interface SceneBounds {
  min_lng: number
  min_lat: number
  max_lng: number
  max_lat: number
}

/** 完整场景上下文（buildings GeoJSON 独立缓存于 sceneStore.buildingsGeoJSON） */
export interface SceneContext {
  scene_id:     string
  sel_bounds:   { minx: number; miny: number; maxx: number; maxy: number }
  threshold:    number
  height_column: string | null
  meta:         SceneMeta
  road_network: {
    nodes:  RoadNode[]
    edges:  RoadEdge[]
    bounds: SceneBounds
  }
}

// ── Store 定义 ───────────────────────────────────────────────────────────────

export const useSceneStore = defineStore('scene', () => {
  /** 当前已加载的场景上下文，null 表示尚未导出 */
  const context = ref<SceneContext | null>(null)

  /**
   * 对应 context 的建筑 GeoJSON 缓存。
   * 由 GeoTool 查询后写入，UnifiedMapView 优先读取以跳过重复 API 调用。
   * scene 切换时（prepareScene / clear）自动清空，防止展示陈旧数据。
   */
  const buildingsGeoJSON = ref<FeatureCollection | null>(null)

  /** /api/scene/prepare 请求进行中 */
  const loading = ref(false)

  /** 最近一次错误信息，成功后清空 */
  const error   = ref<string | null>(null)

  /**
   * 调用后端 /api/scene/prepare，打包场景上下文到 store。
   * 支持幂等：相同参数请求后端直接返回缓存结果。
   * loading 为 true 时阻止重复触发。
   */
  async function prepareScene(payload: {
    sel_bounds:     { minx: number; miny: number; maxx: number; maxy: number }
    threshold:      number
    height_column:  string | null
  }): Promise<void> {
    if (loading.value) return
    loading.value = true
    error.value   = null
    // 清空上一场景的建筑缓存，防止新场景渲染陈旧数据
    buildingsGeoJSON.value = null

    try {
      const data = await http.post<SceneContext>('/api/scene/prepare', {
        ...payload.sel_bounds,
        threshold:     payload.threshold,
        height_column: payload.height_column,
      })
      context.value = data
      // 持久化 scene_id，供页面刷新后 restoreScene() 使用
      localStorage.setItem('hl-scene-id', data.scene_id)
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  /** 清除场景（用于重新选区或退出仿真） */
  function clear(): void {
    context.value        = null
    buildingsGeoJSON.value = null
    error.value          = null
    localStorage.removeItem('hl-scene-id')
  }

  /**
   * 写入建筑 GeoJSON 缓存。
   * 由 GeoTool 在用户点击「导出到指挥中心地图」前调用，
   * 保证 UnifiedMapView 挂载时直接使用而无需重新请求后端。
   */
  function setBuildingsGeoJSON(geojson: FeatureCollection | null): void {
    buildingsGeoJSON.value = geojson
  }

  /**
   * 从后端按 scene_id 恢复已有场景（页面刷新后调用）。
   * 若 scene_id 不存在或已失效（服务重启）则静默忽略。
   */
  async function restoreScene(sceneId: string): Promise<void> {
    if (!sceneId || context.value?.scene_id === sceneId) return
    try {
      const data = await http.get<SceneContext>(`/api/scene/${encodeURIComponent(sceneId)}`)
      context.value = data
    } catch {
      // 404 或网络错误：保持当前状态
    }
  }

  /**
   * 加载预设场景（从后端的磁盘缓存）。
   * 用于快速加载预生成的测试场景，无需重新下载 OSM 或查询建筑。
   */
  async function loadPresetScene(presetId: string): Promise<void> {
    if (loading.value) return
    loading.value = true
    error.value   = null
    buildingsGeoJSON.value = null

    try {
      const data = await http.get<SceneContext>(`/api/scene/preset/${encodeURIComponent(presetId)}`)
      context.value = data
      localStorage.setItem('hl-scene-id', data.scene_id)
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { context, loading, error, buildingsGeoJSON, setBuildingsGeoJSON, prepareScene, restoreScene, loadPresetScene, clear }

})
