<template>
  <div class="dynamic-flow">
    <!-- 卡车实时状态板 -->
    <div class="flow-section">
      <div class="flow-title">📦 卡车配送实时追踪</div>
      <div v-if="trucks.length === 0" class="flow-empty">暂无卡车数据</div>
      <div v-for="truck in trucks" :key="truck.truck_id" class="truck-card">
        <div class="truck-header">
          <span class="truck-icon">🚛</span>
          <span class="truck-name">{{ truck.name || truck.truck_id }}</span>
          <span class="truck-status" :class="statusClass(truck.status)">
            {{ getStatusLabel(truck.status) }}
          </span>
        </div>
        <div class="truck-details">
          <div class="detail-row">
            <span class="label">位置</span>
            <span class="value">{{ formatCoord(truck.lng) }}, {{ formatCoord(truck.lat) }}</span>
          </div>
          <div class="detail-row">
            <span class="label">货物</span>
            <span class="value">{{ truck.inventory_count ?? 0 }} / {{ truck.max_inventory ?? 0 }} 件</span>
          </div>
          <div class="detail-row">
            <span class="label">停靠无人机</span>
            <span class="value">{{ truck.docked_drones?.length ?? 0 }} 架</span>
          </div>
          <div v-if="(truck.docked_drones?.length ?? 0) > 0" class="drone-list">
            <span v-for="did in truck.docked_drones" :key="did" class="drone-tag">🚁 {{ did }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 无人机状态板 -->
    <div class="flow-section">
      <div class="flow-title">🚁 无人机配送实时追踪</div>
      <div v-if="drones.length === 0" class="flow-empty">暂无无人机数据</div>
      <div v-for="drone in drones" :key="drone.drone_id" class="drone-card">
        <div class="drone-header">
          <span class="drone-icon">🚁</span>
          <span class="drone-name">{{ drone.name || drone.drone_id }}</span>
          <span class="drone-status" :class="statusClass(drone.status)">
            {{ getStatusLabel(drone.status) }}
          </span>
        </div>
        <div class="drone-details">
          <div class="detail-row">
            <span class="label">位置</span>
            <span class="value">{{ formatCoord(drone.lng) }}, {{ formatCoord(drone.lat) }}</span>
          </div>
          <div class="detail-row">
            <span class="label">电量</span>
            <div class="battery-bar">
              <div class="battery-fill" :style="{ width: drone.batteryPercent + '%' }"></div>
              <span class="battery-text">{{ drone.batteryPercent.toFixed(0) }}%</span>
            </div>
          </div>
          <div class="detail-row">
            <span class="label">速度</span>
            <span class="value">{{ (drone.cruise_speed ?? 0).toFixed(1) }} m/s</span>
          </div>
          <div v-if="drone.payload_capacity" class="detail-row">
            <span class="label">可载重</span>
            <span class="value">{{ drone.payload_capacity }} kg</span>
          </div>
          <div v-if="drone.dispatch_chain" class="detail-row">
            <span class="label">决策链路</span>
            <span class="value">{{ formatChainBrief(drone.dispatch_chain) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- PPO 决策链路 -->
    <div class="flow-section">
      <div class="flow-title">🔗 PPO 决策链路</div>
      <div v-if="dispatchChains.length === 0" class="flow-empty">暂无执行中决策链路</div>
      <div v-for="chain in dispatchChains" :key="`${chain.drone_id}-${chain.order_id}`" class="chain-card">
        <div class="chain-header">
          <span class="chain-main">{{ chain.drone_id }} · {{ chain.mode }} · {{ chain.order_id }}</span>
          <span class="chain-stage" :class="`stage-${chain.recovery_stage || 'none'}`">
            {{ getRecoveryStageLabel(chain.recovery_stage) }}
          </span>
        </div>
        <div class="chain-steps">
          <span>决策接单</span>
          <span class="chain-arrow">→</span>
          <span>{{ getTrainingStateLabel(chain.training_state) }}</span>
          <span v-if="chain.mode === 'C'" class="chain-arrow">→</span>
          <span v-if="chain.mode === 'C'">{{ chain.selected_recover_node_id || '送达后选择回收站' }}</span>
        </div>
        <div class="chain-meta">
          <span v-if="chain.reservation_node_id">reservation: {{ chain.reservation_node_id }}</span>
          <span v-if="chain.active_leg_kind">flight: {{ chain.active_leg_kind }}</span>
          <span v-if="chain.planned_execution_slack_sec != null">slack: {{ chain.planned_execution_slack_sec.toFixed(1) }}s</span>
        </div>
      </div>
    </div>

    <!-- 实时事件日志 -->
    <div class="flow-section">
      <div class="flow-title">📜 事件日志</div>
      <div class="event-log">
        <div v-if="events.length === 0" class="flow-empty">等待事件...</div>
        <div v-for="(evt, i) in events" :key="i" class="event-item">
          <span class="event-time">{{ evt.time.toFixed(2) }}s</span>
          <span class="event-icon">{{ evt.icon }}</span>
          <span class="event-msg">{{ evt.msg }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useEntityStore } from '@/stores/entity'
import { useSystemStore } from '@/stores/system'

const entityStore = useEntityStore()
const systemStore = useSystemStore()

const events = ref<Array<{ time: number; icon: string; msg: string }>>([])

const trucks = computed(() => {
  const runtimeTrucks = entityStore.rtTrucks.length > 0 ? entityStore.rtTrucks : entityStore.trucks
  return runtimeTrucks.map((truck: any) => {
    const config = entityStore.trucks.find(t => t.truck_id === truck.truck_id)
    return {
      ...config,
      ...truck,
      status: truck.status ?? 'IDLE',
      inventory_count: truck.inventory_count ?? 0,
      max_inventory: truck.max_inventory ?? config?.max_inventory ?? 0,
      docked_drones: truck.docked_drones ?? [],
    }
  })
})

const drones = computed(() => {
  const runtimeDrones = entityStore.rtDrones.length > 0 ? entityStore.rtDrones : entityStore.drones
  return runtimeDrones.map((drone: any) => {
    const config = entityStore.drones.find(d => d.drone_id === drone.drone_id)
    return {
      ...config,
      ...drone,
      status: drone.status ?? 'IDLE',
      batteryPercent: getBatteryPercent(drone),
    }
  })
})

const dispatchChains = computed(() => systemStore.dispatchChains)

const statusLabelMap: Record<string, string> = {
  'IDLE': '空闲',
  'DRIVING': '行驶中',
  'CHARGING': '充电中',
  'RECOVERY': '回收中',
  'ASSIGNED': '已分配',
  'FLYING': '飞行中',
  'LANDING': '落地中',
  'CHARGING_WAIT': '等待充电',
}

function getStatusLabel(status: string): string {
  return statusLabelMap[status] || status
}

const recoveryStageLabelMap: Record<string, string> = {
  not_applicable: '无需回收站',
  pending_post_delivery_selection: '待选回收站',
  reservation_active: '已预约',
  rendezvous_selected: '已选回收站',
}

const trainingStateLabelMap: Record<string, string> = {
  idle: '空闲',
  flying_to_deliver: '配送飞行',
  delivery_service: '客户点服务',
  delivered: '已送达',
  return_to_rendezvous: '飞往汇合点',
  waiting_for_truck: '等待卡车',
  return_to_station: '返站',
  return_to_depot: '返仓',
  fallback_recovery: '兜底回收',
  charging_or_swap: '充换电',
  charging_on_truck: '车载充电',
  riding_with_truck: '随车',
}

function getRecoveryStageLabel(stage: string | undefined): string {
  return recoveryStageLabelMap[stage || ''] || stage || '—'
}

function getTrainingStateLabel(state: string | null | undefined): string {
  return trainingStateLabelMap[state || ''] || state || '执行中'
}

function formatChainBrief(chain: any): string {
  const mode = chain.mode ? `模式${chain.mode}` : '模式—'
  const order = chain.order_id || '无订单'
  const recover = chain.selected_recover_node_id || chain.reservation_node_id || '回收站待定'
  return `${mode} / ${order} / ${recover}`
}

function statusClass(status: string | undefined): string {
  return `status-${(status ?? 'IDLE').toLowerCase()}`
}

function formatCoord(value: number | undefined): string {
  return typeof value === 'number' ? value.toFixed(5) : '—'
}

function getBatteryPercent(drone: any): number {
  if (typeof drone.battery_ratio === 'number') {
    return clampPercent(drone.battery_ratio * 100)
  }
  if (typeof drone.battery_percent === 'number') {
    return clampPercent(drone.battery_percent)
  }
  return 0
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value))
}

// 添加事件日志
function addEvent(icon: string, msg: string) {
  const e = { time: systemStore.simTime, icon, msg }
  events.value.unshift(e)
  if (events.value.length > 20) events.value.pop()
}

defineExpose({ addEvent })
</script>

<style scoped>
.dynamic-flow {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 12px;
  height: 100%;
  overflow-y: auto;
  font-size: 0.85rem;
}

.flow-section {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
}

.flow-title {
  font-weight: 600;
  color: #334155;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 2px solid #cbd5e1;
}

.flow-empty {
  text-align: center;
  color: #94a3b8;
  padding: 16px;
  font-style: italic;
}

/* 卡车卡片 */
.truck-card {
  background: white;
  border: 1px solid #dbeafe;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}

.truck-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-weight: 600;
}

.truck-icon {
  font-size: 1.2rem;
}

.truck-name {
  flex: 1;
  color: #1e40af;
}

.truck-status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  background: #dbeafe;
  color: #1e40af;
}

.status-driving {
  background: #d1fae5 !important;
  color: #059669 !important;
}

.status-idle {
  background: #f3e8ff !important;
  color: #7c3aed !important;
}

.status-charging {
  background: #fed7aa !important;
  color: #dc2626 !important;
}

.truck-details {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 0.8rem;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.label {
  font-weight: 600;
  color: #475569;
  min-width: 60px;
}

.value {
  color: #0f172a;
  font-family: 'Monaco', 'Courier New', monospace;
}

.drone-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 5px;
}

.drone-tag {
  display: inline-block;
  padding: 2px 6px;
  background: #e0e7ff;
  border-radius: 3px;
  color: #4338ca;
  font-size: 0.75rem;
}

/* 无人机卡片 */
.drone-card {
  background: white;
  border: 1px solid #e9d5ff;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}

.drone-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-weight: 600;
}

.drone-icon {
  font-size: 1.2rem;
}

.drone-name {
  flex: 1;
  color: #7c3aed;
}

.drone-status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  background: #e9d5ff;
  color: #7c3aed;
}

.drone-details {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.battery-bar {
  width: 100px;
  height: 16px;
  border: 1px solid #cbd5e1;
  border-radius: 2px;
  background: #f1f5f9;
  overflow: hidden;
  position: relative;
}

.battery-fill {
  height: 100%;
  background: linear-gradient(90deg, #7c3aed, #2563eb);
  transition: width 0.2s;
}

.battery-text {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 600;
  color: #0f172a;
}

.chain-card {
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}

.chain-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.chain-main {
  flex: 1;
  min-width: 0;
  color: #0f172a;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.chain-stage {
  padding: 2px 6px;
  border-radius: 4px;
  background: #e2e8f0;
  color: #334155;
  font-size: 0.72rem;
  font-weight: 700;
  white-space: nowrap;
}

.stage-reservation_active {
  background: #dcfce7;
  color: #166534;
}

.stage-pending_post_delivery_selection {
  background: #fef3c7;
  color: #92400e;
}

.stage-rendezvous_selected {
  background: #dbeafe;
  color: #1d4ed8;
}

.chain-steps,
.chain-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  color: #475569;
  font-size: 0.78rem;
  line-height: 1.35;
}

.chain-arrow {
  color: #94a3b8;
}

.chain-meta {
  margin-top: 6px;
  color: #64748b;
  font-family: 'Monaco', 'Courier New', monospace;
}

/* 事件日志 */
.event-log {
  max-height: 250px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.event-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-bottom: 1px solid #e2e8f0;
  font-size: 0.8rem;
}

.event-time {
  color: #94a3b8;
  font-weight: 600;
  min-width: 50px;
  font-family: 'Monaco', monospace;
}

.event-icon {
  font-size: 1rem;
  min-width: 20px;
}

.event-msg {
  flex: 1;
  color: #334155;
}
</style>
