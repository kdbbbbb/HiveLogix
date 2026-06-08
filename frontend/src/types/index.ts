/**
 * HiveLogix 核心类型定义
 * 与后端 core/entities 对齐，区分"用户静态配置"与"运行时遥测快照"两个层次：
 *   - XxxConfig  : 用户在前端手动创建/编辑的字段，存入 localStorage
 *   - Xxx        : 继承 Config，追加 WebSocket 推送的运行时字段（只读）
 */

// ══════════════════════════════════════════════════════════════════
// 基础枚举 & 联合类型
// ══════════════════════════════════════════════════════════════════

export type FulfillmentMode = 'A' | 'B' | 'C' | 'D' | 'E'
export type DroneType       = 'LightDrone' | 'HeavyDrone'
export type HomeType        = 'DEPOT' | 'TRUCK'

export type TaskStatus =
  | 'PENDING' | 'ASSIGNED' | 'PICKED_UP'
  | 'DELIVERING' | 'COMPLETED' | 'TIMEOUT' | 'REJECTED'

export type DroneStatusType =
  | 'IDLE' | 'FLYING_TO_PICKUP' | 'FLYING_TO_DELIVER'
  | 'FLYING_TO_STATION' | 'FLYING_TO_TRUCK'
  | 'RETURNING_TO_DEPOT' | 'QUEUING' | 'CHARGING'
  | 'LOADING' | 'UNLOADING' | 'DEAD'

export type TruckStatusType =
  | 'IDLE' | 'DRIVING' | 'WAITING'
  | 'LOADING_DRONE' | 'UNLOADING_DRONE'


// ══════════════════════════════════════════════════════════════════
// 用户手动配置实体（静态配置，localStorage 持久化）
// ══════════════════════════════════════════════════════════════════

/** 仓库配置（用户手动新建 · Infrastructure 页面） */
export interface DepotConfig {
  depot_id:      string   // 自动生成，格式 D-xxxxxx
  name:          string   // 仓库名称
  lng:           number   // WGS84 经度
  lat:           number   // WGS84 纬度
  altitude:      number   // 高度（m），默认 0
  capacity:      number   // 订单池上限（件），默认 500
  swap_time:     number   // 充换电耗时（秒）
  parking_slots: number   // 充换电并行位数
}

/** 充换电站配置（用户手动新建 · Infrastructure 页面） */
export interface StationConfig {
  station_id:    string
  name:          string
  lng:           number
  lat:           number
  altitude:      number
  swap_time:     number   // 每次换电耗时（秒）
  parking_slots: number   // 并发槽位数 K
}

/** 卡车配置（用户手动新建 · FleetManagement 页面） */
export interface TruckConfig {
  truck_id:      string
  name:          string
  speed:         number   // 地面行驶速度（m/s），默认 10
  max_inventory: number   // 车厢最大包裹数，默认 20
  swap_time:     number   // 车载无人机换电耗时（秒）
  parking_slots: number   // 起降平台并发数 K
  home_depot_id: string   // 归属仓库 ID
}

/** 无人机配置（用户手动新建 · FleetManagement 页面） */
export interface DroneConfig {
  drone_id:   string
  drone_type: DroneType   // LightDrone(2kg/15m/s) | HeavyDrone(10kg/10m/s)
  home_id:    string      // 归属母体 ID（depot_id 或 truck_id）
  home_type:  HomeType    // DEPOT | TRUCK
}


// ══════════════════════════════════════════════════════════════════
// 运行时遥测快照（WebSocket 推送，只读）
// ══════════════════════════════════════════════════════════════════

export interface Depot extends DepotConfig {
  pending_count?:     number
  idle_drone_count?:  number
  available_slots?:   number
  queue_length?:      number
}

export interface Station extends StationConfig {
  available_slots?:   number
  queue_length?:      number
  serving_drone_ids?: string[]
}

export interface Truck extends TruckConfig {
  status?:          TruckStatusType
  lng?:             number
  lat?:             number
  inventory_count?: number
  docked_drones?:   string[]
  available_slots?: number
}

export interface Drone extends DroneConfig {
  status?:                DroneStatusType
  lng?:                   number
  lat?:                   number
  battery_ratio?:         number
  carrying_order_id?:     string
  cumulative_distance_m?: number
  remaining_range_m?:     number
  reservation_node_id?:    string | null
  dispatch_chain?:         DispatchChain | null
  dispatch_order_id?:      string | null
  dispatch_mode?:          string | null
  selected_recover_node_id?: string | null
  recovery_stage?:         string | null
  planned_truck_arrival_time?: number | null
  planned_uav_arrival_time_lb?: number | null
  planned_execution_slack_sec?: number | null
}


// ══════════════════════════════════════════════════════════════════
// 订单（算法生成・运行时只读）
// ══════════════════════════════════════════════════════════════════

export interface Order {
  order_id:            string
  status:              TaskStatus
  /** 取货源类型（demo 模式由生成器填充；后端联调时由调度算法填充，生成时为 null） */
  source_type?:        HomeType | null
  /** 取货源 ID（同上） */
  pickup_source_id?:   string | null
  payload_weight:      number   // kg
  create_time:         number   // ms timestamp（demo）或仿真秒（sim_s，见 time_domain）
  deadline:            number   // ms timestamp（demo）或仿真秒（sim_s，见 time_domain）
  actual_deliver_time?: number
  assigned_vehicle_id?: string
  assigned_mode?:       FulfillmentMode
  fulfillment_mode?:    string   // 订单生成器填写的履单模式
  delivery_lng:         number
  delivery_lat:         number
  delivery_z:           number
  // 前端展示辅助字段（生成器填写，snake_case）
  priority?:            string   // URGENT / NORMAL / LOW
  warehouse_name?:      string   // 来源仓库名称
  deadline_iso?:        string   // 格式化截止时间
  priority_label?:      string   // 中文优先级标签
  // 前端展示辅助字段（旧版 camelCase，兼容保留）
  releaseTimeText?:      string
  deadlineText?:         string
  timeWindowText?:       string
  deliveryLocationText?: string
  warehouseName?:        string
  taskDescription?:      string
  priorityLabel?:        string
  priorityLevel?:        1 | 2 | 3
  orderType?:            string
  payloadVolume?:        number
  orderValue?:           number
  /**
   * 时间字段语义域（决定 create_time / deadline 的单位）
   *   'wall_ms' : 前端 demo 模式，值为 Date.now() 毫秒时间戳（默认）
   *   'sim_s'   : 后端联调模式，值为仿真累计秒（float）
   */
  time_domain?: 'wall_ms' | 'sim_s'
}

/**
 * 后端在仿真时刻 spawn_sim_s 注入的预定义动态订单（与 orders.json 对齐，保证复现）
 * 使用 deadline_offset_s 或绝对 deadline_sim_s 之一。
 */
export interface ScheduledDynamicOrder {
  order_id: string
  spawn_sim_s: number
  /** 从 spawn 起算的截止偏移（秒） */
  deadline_offset_s?: number
  /** 仿真绝对截止时间（秒），若设置则忽略 deadline_offset_s */
  deadline_sim_s?: number
  delivery_lng: number
  delivery_lat: number
  delivery_z?: number
  payload_weight: number
  source_type?: string | null
  pickup_source_id?: string | null
  fulfillment_mode?: string
  priority?: string
  priority_label?: string
}

/** 订单生成器参数（用户可在仿真面板调整） */
export interface OrderGeneratorConfig {
  arrival_rate:    number   // 单/分钟
  weight_min:      number   // kg
  weight_max:      number   // kg
  window_min:      number   // 时间窗最小宽度（分钟）
  window_max:      number   // 时间窗最大宽度（分钟）
  priority_urgent: number   // 紧急优先级概率（%）
  priority_normal: number   // 普通优先级概率（%）
  priority_low:    number   // 低优先级概率（%）
  // Phase 2 扩展字段（可选：向后兼容 localStorage 旧版）
  max_orders?:        number | null   // 总量上限，null = 无限
  geo_mode?:          GeoMode        // 收货点地理分布模式
  cluster_radius_km?: number         // 热点 / 仓库覆盖半径（km）
  burst_enabled?:     boolean        // 是否启用突发密集模式
  burst_multiplier?:  number         // 突发期到达率倍率
  burst_duration_s?:  number         // 每次突发持续时间（秒）
}

/** 收货点地理分布模式 */
export type GeoMode = 'uniform' | 'clustered' | 'depot_nearby'

/** 仓库条目（供订单生成器 warehousePool 使用） */
export interface WarehouseEntry {
  id:   string
  name: string
  lng:  number
  lat:  number
}


// ══════════════════════════════════════════════════════════════════
// WebSocket 推送消息
// ══════════════════════════════════════════════════════════════════

export interface DispatchRouteNode {
  node_id: string
  node_type: string
  lng: number
  lat: number
  arrival_time: number
  departure_time: number
  order_id: string
}

export interface DispatchTruckRoute {
  truck_id: string
  nodes: DispatchRouteNode[]
  total_distance: number
  charging_stop_ids: string[]
  geometry?: { lng: number; lat: number }[]
}

export interface DispatchDroneRoute {
  drone_id: string
  order_id: string
  mode: FulfillmentMode
  launch_node_id?: string
  launch_node_type?: string
  launch_time?: number
  recovery_station_id?: string  // 新增：回收点 ID（充电站或仓库）
  path: [number, number][]
}

export interface RuntimePathEntry {
  entity_id: string
  route_version?: number
  segment_id?: string | number
  status?: string
  motion_mode?: string
  start_time?: number
  end_time?: number
  from_node_id?: string
  to_node_id?: string
  target_node_id?: string | null
  target_node_type?: string | null
  order_id?: string | null
  distance_m?: number
  path: [number, number][]
}

export interface RuntimePaths {
  trucks: RuntimePathEntry[]
  drones: RuntimePathEntry[]
}

export interface DispatchChain {
  drone_id: string
  order_id: string
  mode: string
  trigger_station_id?: string | null
  selected_recover_node_id?: string | null
  reservation_node_id?: string | null
  recovery_stage?: string
  planned_truck_arrival_time?: number | null
  planned_uav_arrival_time_lb?: number | null
  planned_execution_slack_sec?: number | null
  training_state?: string | null
  carrying_order_id?: string | null
  active_leg_kind?: string | null
  active_leg_target_node_id?: string | null
  active_leg_target_node_type?: string | null
  snapshot_time?: number
}

export interface DecisionEvent {
  decision_id: number
  sim_time: number
  wall_time_ms: number
  event_seq: number
  drone_id: string
  trigger_type: string
  trigger_station_id?: string | null
  candidate_summary?: Record<string, unknown>
  selected_action?: Record<string, unknown>
  selected_order_id?: string | null
  selected_mode?: string | null
  selected_recover_node?: string | null
  recovery_selection_stage?: string | null
  inference_latency_ms?: number | null
  status: 'DECISION_PENDING' | 'DECISION_APPLIED' | 'EXECUTION_HARD_FAILED'
  actor_drone_final_state?: string | null
  failure_type?: string | null
}

export interface DispatchPlan {
  total_orders: number
  feasible: number
  modes: Record<string, number>
  cost_total: number
  cost_breakdown?: {
    dist: number
    energy: number
    penalty: number
  }
  static_cache_reused?: boolean
  static_cache_path?: string
  allocations: unknown[]
  truck_routes?: Record<string, DispatchTruckRoute>
  drone_routes?: DispatchDroneRoute[]
}

export interface WsMessage<T = unknown> {
  type: string
  payload: T
}
