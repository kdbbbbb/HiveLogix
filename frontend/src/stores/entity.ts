import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type {
  DepotConfig, StationConfig, TruckConfig, DroneConfig,
  Depot, Station, Truck, Drone,
} from '@/types'
import type { SceneBounds } from '@/stores/scene'
import { DEFAULT_DEPOTS, DEFAULT_STATIONS, DEFAULT_TRUCKS, DEFAULT_DRONES } from '@/data/defaultEntities'
import { gridLayoutSplit } from '@/utils/geoLayout'

// ── localStorage 持久化 ───────────────────────────────────────────
const LS_KEY = 'hl-entity-config-v2'

function loadFromStorage(): {
  depots: DepotConfig[]
  stations: StationConfig[]
  trucks: TruckConfig[]
  drones: DroneConfig[]
} | null {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function saveToStorage(data: object) {
  localStorage.setItem(LS_KEY, JSON.stringify(data))
}

// ── ID 生成工具 ───────────────────────────────────────────────────
export function genEntityId(prefix: string): string {
  const ts = Date.now().toString(36).toUpperCase().slice(-5)
  const r  = Math.floor(Math.random() * 9000 + 1000)
  return `${prefix}-${ts}-${r}`
}

/**
 * 实体状态仓库
 *
 * 两类数据：
 *   1. 静态配置（XxxConfig）：用户手动填写，localStorage 持久化，前端 CRUD
 *   2. 运行时快照（Xxx extends XxxConfig）：WebSocket 推送，只读，不持久化
 */
export const useEntityStore = defineStore('entity', () => {

  // ── 静态配置（持久化） ────────────────────────────────────────────
  const depots   = ref<DepotConfig[]>([])
  const stations = ref<StationConfig[]>([])
  const trucks   = ref<TruckConfig[]>([])
  const drones   = ref<DroneConfig[]>([])

  // ── 运行时快照（WebSocket 推送，不持久化）─────────────────────────
  const rtDepots   = ref<Depot[]>([])
  const rtStations = ref<Station[]>([])
  const rtTrucks   = ref<Truck[]>([])
  const rtDrones   = ref<Drone[]>([])

  // ── 持久化 ────────────────────────────────────────────────────────
  function _persist() {
    saveToStorage({
      depots:   depots.value,
      stations: stations.value,
      trucks:   trucks.value,
      drones:   drones.value,
    })
  }

  function loadConfig() {
    const saved = loadFromStorage()
    if (!saved) {
      // 首次加载（localStorage 为空）：注入默认实体（坐标占位 0，等待地图布局）
      _injectDefaults()
      return
    }
    if (Array.isArray(saved.depots))   depots.value   = saved.depots
    if (Array.isArray(saved.stations)) stations.value = saved.stations
    if (Array.isArray(saved.trucks))   trucks.value   = saved.trucks
    if (Array.isArray(saved.drones))   drones.value   = saved.drones
  }

  /** 注入默认实体（浅拷贝，防止直接修改种子数据） */
  function _injectDefaults() {
    depots.value   = DEFAULT_DEPOTS.map(d => ({ ...d }))
    stations.value = DEFAULT_STATIONS.map(s => ({ ...s }))
    trucks.value   = DEFAULT_TRUCKS.map(t => ({ ...t }))
    drones.value   = DEFAULT_DRONES.map(d => ({ ...d }))
    _persist()
  }

  /**
   * 根据地图 bbox 重新均匀分布仓库和充换电站坐标。
   * 每次选取新地图后由 GeoTool/index.vue 调用，覆盖旧坐标并持久化。
   *
   * @param bounds 当前仿真地图边界（来自 sceneStore.context.road_network.bounds）
   */
  function redistributeByBounds(bounds: SceneBounds) {
    // 整体均匀分布：仓库与充换电站共用同一均匀网格
    // 仓库占据 2D 子网格映射位置（自身也均匀），充换电站填充剩余格点
    const { depots: depotCoords, stations: stationCoords } =
      gridLayoutSplit(depots.value.length, stations.value.length, bounds)

    depots.value = depots.value.map((d, i) => ({
      ...d,
      lng: parseFloat(depotCoords[i].lng.toFixed(6)),
      lat: parseFloat(depotCoords[i].lat.toFixed(6)),
    }))

    stations.value = stations.value.map((s, i) => ({
      ...s,
      lng: parseFloat(stationCoords[i].lng.toFixed(6)),
      lat: parseFloat(stationCoords[i].lat.toFixed(6)),
    }))

    _persist()
  }

  // ── Depot CRUD ────────────────────────────────────────────────────
  function addDepot(d: DepotConfig)    { depots.value.push(d); _persist() }
  function updateDepot(d: DepotConfig) {
    const i = depots.value.findIndex(x => x.depot_id === d.depot_id)
    if (i !== -1) { depots.value[i] = d; _persist() }
  }
  function removeDepot(id: string) {
    depots.value = depots.value.filter(x => x.depot_id !== id)
    _persist()
  }

  // ── Station CRUD ──────────────────────────────────────────────────
  function addStation(s: StationConfig)    { stations.value.push(s); _persist() }
  function updateStation(s: StationConfig) {
    const i = stations.value.findIndex(x => x.station_id === s.station_id)
    if (i !== -1) { stations.value[i] = s; _persist() }
  }
  function removeStation(id: string) {
    stations.value = stations.value.filter(x => x.station_id !== id)
    _persist()
  }

  // ── Truck CRUD ────────────────────────────────────────────────────
  function addTruck(t: TruckConfig)    { trucks.value.push(t); _persist() }
  function updateTruck(t: TruckConfig) {
    const i = trucks.value.findIndex(x => x.truck_id === t.truck_id)
    if (i !== -1) { trucks.value[i] = t; _persist() }
  }
  function removeTruck(id: string) {
    // 归属该卡车的无人机 home_type 置为 orphan 保护（仅警告，不自动删除）
    trucks.value = trucks.value.filter(x => x.truck_id !== id)
    _persist()
  }

  // ── Drone CRUD ────────────────────────────────────────────────────
  function addDrone(d: DroneConfig)    { drones.value.push(d); _persist() }
  function updateDrone(d: DroneConfig) {
    const i = drones.value.findIndex(x => x.drone_id === d.drone_id)
    if (i !== -1) { drones.value[i] = d; _persist() }
  }
  function removeDrone(id: string) {
    drones.value = drones.value.filter(x => x.drone_id !== id)
    _persist()
  }

  // ── 运行时快照更新（WebSocket 回调） ─────────────────────────────
  function setRuntimeAll(data: {
    depots?:   Depot[]
    stations?: Station[]
    trucks?:   Truck[]
    drones?:   Drone[]
  }) {
    if (data.depots)   rtDepots.value   = data.depots
    if (data.stations) rtStations.value = data.stations
    if (data.trucks)   rtTrucks.value   = data.trucks
    if (data.drones)   rtDrones.value   = data.drones
  }

  // ── 统计计算 ──────────────────────────────────────────────────────
  const droneCount    = computed(() => drones.value.length)
  const truckCount    = computed(() => trucks.value.length)
  const lightCount    = computed(() => drones.value.filter(d => d.drone_type === 'LightDrone').length)
  const heavyCount    = computed(() => drones.value.filter(d => d.drone_type === 'HeavyDrone').length)
  const depotOptions  = computed(() =>
    depots.value.map(d => ({ label: `${d.name} (${d.depot_id})`, value: d.depot_id }))
  )
  const truckOptions  = computed(() =>
    trucks.value.map(t => ({ label: `${t.name} (${t.truck_id})`, value: t.truck_id }))
  )
  /** 是否存在坐标仍为 (0, 0) 的仓库或充换电站（地图尚未选取） */
  const hasUnlocated  = computed(() =>
    depots.value.some(d => d.lng === 0 && d.lat === 0) ||
    stations.value.some(s => s.lng === 0 && s.lat === 0),
  )

  return {
    // 静态配置
    depots, stations, trucks, drones,
    // 运行时快照
    rtDepots, rtStations, rtTrucks, rtDrones,
    // 操作
    loadConfig,
    redistributeByBounds,
    addDepot, updateDepot, removeDepot,
    addStation, updateStation, removeStation,
    addTruck, updateTruck, removeTruck,
    addDrone, updateDrone, removeDrone,
    setRuntimeAll,
    // 计算属性
    droneCount, truckCount, lightCount, heavyCount,
    depotOptions, truckOptions,
    hasUnlocated,
  }
})
