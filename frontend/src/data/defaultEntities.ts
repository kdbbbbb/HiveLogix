/**
 * defaultEntities — 系统默认实体种子数据
 *
 * 约束：
 *   - 纯静态数据，不含任何逻辑、响应式状态或函数
 *   - lng / lat 均为 0：坐标由地图选取后 redistributeByBounds() 自动填充
 *   - 命名使用通用编号，不绑定任何真实地名或城市
 *   - 规模：1 仓库 + 10 充换电站 + 1 卡车 + 12 无人机
 */

import type { DepotConfig, StationConfig, TruckConfig, DroneConfig } from '@/types'

// ── 默认仓库（1 个） ─────────────────────────────────────────────

export const DEFAULT_DEPOTS: DepotConfig[] = [
  {
    depot_id:      'DEP-DEFAULT-01',
    name:          '仓库-01',
    lng:           0,
    lat:           0,
    altitude:      0,
    capacity:      500,
    swap_time:     90,
    parking_slots: 4,
  },
]

// ── 默认充换电站（10 个） ────────────────────────────────────────

export const DEFAULT_STATIONS: StationConfig[] = Array.from({ length: 10 }, (_, i) => ({
  station_id:    `STA-DEFAULT-${String(i + 1).padStart(2, '0')}`,
  name:          `换电站-${String(i + 1).padStart(2, '0')}`,
  lng:           0,
  lat:           0,
  altitude:      0,
  swap_time:     60,
  parking_slots: 2,
}))

// ── 默认卡车（1 辆） ─────────────────────────────────────────────

export const DEFAULT_TRUCKS: TruckConfig[] = [
  {
    truck_id:      'TRK-DEFAULT-01',
    name:          '卡车-01',
    speed:         15,
    max_inventory: 30,
    swap_time:     90,
    parking_slots: 3,
    home_depot_id: 'DEP-DEFAULT-01',
  },
]

// ── 默认无人机（12 架：LightDrone × 9 + HeavyDrone × 3） ────────
//
//   卡车搭载：1 架 LightDrone
//   仓库-01：  LightDrone × 3 + HeavyDrone × 2

export const DEFAULT_DRONES: DroneConfig[] = [
  // 卡车搭载
  { drone_id: 'UAV-DEFAULT-01', drone_type: 'LightDrone', home_id: 'TRK-DEFAULT-01', home_type: 'TRUCK' },
  // 仓库-01 LightDrone
  { drone_id: 'UAV-DEFAULT-02', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-03', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-04', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-05', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-06', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-07', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-08', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-09', drone_type: 'LightDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  // 仓库-01 HeavyDrone
  { drone_id: 'UAV-DEFAULT-10', drone_type: 'HeavyDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-11', drone_type: 'HeavyDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
  { drone_id: 'UAV-DEFAULT-12', drone_type: 'HeavyDrone', home_id: 'DEP-DEFAULT-01', home_type: 'DEPOT' },
]
