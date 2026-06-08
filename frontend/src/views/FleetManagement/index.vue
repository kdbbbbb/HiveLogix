<template>
  <PageShell
    icon="🛸"
    title="载具管理调度"
    desc="卡车序列 · 无人机队列 · 实时状态 · 分配策略"
  >
    <!-- 统计摘要 -->
    <div class="fleet-summary">
      <div class="stat-pill">
        <span class="stat-pill__icon">🚚</span>
        <span class="stat-pill__value">{{ entityStore.truckCount }}</span>
        <span class="stat-pill__label">卡车总数</span>
      </div>
      <div class="stat-pill">
        <span class="stat-pill__icon">🛸</span>
        <span class="stat-pill__value">{{ entityStore.droneCount }}</span>
        <span class="stat-pill__label">无人机总数</span>
      </div>
      <div class="stat-pill">
        <span class="stat-pill__icon">⚡</span>
        <span class="stat-pill__value">{{ entityStore.lightCount }}</span>
        <span class="stat-pill__label">轻型无人机</span>
      </div>
      <div class="stat-pill">
        <span class="stat-pill__icon">🚁</span>
        <span class="stat-pill__value">{{ entityStore.heavyCount }}</span>
        <span class="stat-pill__label">重型无人机</span>
      </div>
    </div>

    <!-- 两列：卡车 + 无人机 -->
    <div class="fleet-grid">
      <!-- 卡车面板 -->
      <div class="fleet-panel">
        <div class="fleet-panel__header">
          <div class="fleet-panel__title-row">
            <span class="fleet-panel__title">🚚 卡车序列</span>
            <span class="fleet-panel__count">{{ entityStore.trucks.length }} 辆</span>
          </div>
          <button class="btn-add" @click="openAddTruck">+ 新增卡车</button>
        </div>

        <div class="fleet-table-head">
          <span class="tc-name">名称</span>
          <span class="tc-id">ID</span>
          <span class="tc-num">速度(m/s)</span>
          <span class="tc-num">载货容量</span>
          <span class="tc-depot">归属仓库</span>
          <span class="tc-ops">操作</span>
        </div>

        <template v-if="entityStore.trucks.length">
          <div v-for="t in entityStore.trucks" :key="t.truck_id" class="fleet-row">
            <span class="tc-name fleet-row__primary">{{ t.name }}</span>
            <span class="tc-id fleet-row__id">{{ t.truck_id }}</span>
            <span class="tc-num">{{ t.speed }}</span>
            <span class="tc-num">{{ t.max_inventory }}</span>
            <span class="tc-depot fleet-row__muted">{{ t.home_depot_id || '—' }}</span>
            <span class="tc-ops fleet-row__actions">
              <button class="btn-icon" title="编辑" @click="openEditTruck(t)">✏️</button>
              <button class="btn-icon btn-icon--del" title="删除" @click="deleteTruck(t.truck_id)">🗑️</button>
            </span>
          </div>
        </template>
        <div v-else class="fleet-empty">🚚 暂无卡车 · 点击右上角新增</div>
      </div>

      <!-- 无人机面板 -->
      <div class="fleet-panel">
        <div class="fleet-panel__header">
          <div class="fleet-panel__title-row">
            <span class="fleet-panel__title">🛸 无人机序列</span>
            <span class="fleet-panel__count">{{ entityStore.drones.length }} 架</span>
          </div>
          <button class="btn-add" @click="openAddDrone">+ 新增无人机</button>
        </div>

        <div class="fleet-table-head">
          <span class="dc-id">ID</span>
          <span class="dc-type">机型</span>
          <span class="dc-home-type">归属类型</span>
          <span class="dc-home">归属ID</span>
          <span class="dc-ops">操作</span>
        </div>

        <template v-if="entityStore.drones.length">
          <div v-for="d in entityStore.drones" :key="d.drone_id" class="fleet-row">
            <span class="dc-id fleet-row__id">{{ d.drone_id }}</span>
            <span class="dc-type">
              <span class="type-badge" :class="d.drone_type === 'LightDrone' ? 'type-badge--light' : 'type-badge--heavy'">
                {{ d.drone_type === 'LightDrone' ? '轻型' : '重型' }}
              </span>
            </span>
            <span class="dc-home-type fleet-row__muted">{{ d.home_type }}</span>
            <span class="dc-home fleet-row__muted">{{ d.home_id }}</span>
            <span class="dc-ops fleet-row__actions">
              <button class="btn-icon" title="编辑" @click="openEditDrone(d)">✏️</button>
              <button class="btn-icon btn-icon--del" title="删除" @click="deleteDrone(d.drone_id)">🗑️</button>
            </span>
          </div>
        </template>
        <div v-else class="fleet-empty">🛸 暂无无人机 · 点击右上角新增</div>
      </div>
    </div>

    <!-- 卡车弹窗 -->
    <Teleport to="body">
      <div v-if="truckModal.open" class="modal-overlay" @click.self="closeTruckModal">
        <div class="modal-card">
          <div class="modal-header">
            <span class="modal-title">{{ truckModal.mode === 'add' ? '新增卡车' : '编辑卡车' }}</span>
            <button class="modal-close" @click="closeTruckModal">✕</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label>名称 *</label>
              <input v-model="truckModal.form.name" placeholder="如：TR-01 沪A" maxlength="32" />
              <span v-if="truckErrors.name" class="form-error">{{ truckErrors.name }}</span>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>巡航速度 (m/s) *</label>
                <input v-model.number="truckModal.form.speed" type="number" min="1" step="0.5" placeholder="15" />
                <span v-if="truckErrors.speed" class="form-error">{{ truckErrors.speed }}</span>
              </div>
              <div class="form-group">
                <label>最大载货量（件）*</label>
                <input v-model.number="truckModal.form.max_inventory" type="number" min="1" placeholder="30" />
                <span v-if="truckErrors.max_inventory" class="form-error">{{ truckErrors.max_inventory }}</span>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>无人机泊位数 *</label>
                <input v-model.number="truckModal.form.parking_slots" type="number" min="1" max="8" placeholder="2" />
                <span v-if="truckErrors.parking_slots" class="form-error">{{ truckErrors.parking_slots }}</span>
              </div>
              <div class="form-group">
                <label>换电耗时 τ_swap (s) *</label>
                <input v-model.number="truckModal.form.swap_time" type="number" min="1" placeholder="90" />
                <span v-if="truckErrors.swap_time" class="form-error">{{ truckErrors.swap_time }}</span>
              </div>
            </div>
            <div class="form-group">
              <label>归属仓库（出发地）</label>
              <select v-model="truckModal.form.home_depot_id">
                <option value="">— 未绑定 —</option>
                <option v-for="o in entityStore.depotOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
              </select>
            </div>
            <div class="form-id-hint">ID：<code>{{ truckModal.form.truck_id }}</code></div>
          </div>
          <div class="modal-footer">
            <button class="btn-cancel" @click="closeTruckModal">取消</button>
            <button class="btn-save" @click="saveTruck">保存</button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- 无人机弹窗 -->
    <Teleport to="body">
      <div v-if="droneModal.open" class="modal-overlay" @click.self="closeDroneModal">
        <div class="modal-card">
          <div class="modal-header">
            <span class="modal-title">{{ droneModal.mode === 'add' ? '新增无人机' : '编辑无人机' }}</span>
            <button class="modal-close" @click="closeDroneModal">✕</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label>机型 *</label>
              <select v-model="droneModal.form.drone_type">
                <option value="LightDrone">LightDrone（轻型，100 Wh）</option>
                <option value="HeavyDrone">HeavyDrone（重型，800 Wh）</option>
              </select>
            </div>
            <div class="form-group">
              <label>归属类型 *</label>
              <select v-model="droneModal.form.home_type" @change="droneModal.form.home_id = ''">
                <option value="DEPOT">DEPOT（仓库）</option>
                <option value="TRUCK">TRUCK（卡车）</option>
              </select>
            </div>
            <div class="form-group">
              <label>归属目标 *</label>
              <select v-model="droneModal.form.home_id">
                <option value="">— 请选择 —</option>
                <template v-if="droneModal.form.home_type === 'DEPOT'">
                  <option v-for="o in entityStore.depotOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
                </template>
                <template v-else>
                  <option v-for="o in entityStore.truckOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
                </template>
              </select>
              <span v-if="droneErrors.home_id" class="form-error">{{ droneErrors.home_id }}</span>
            </div>
            <div class="form-id-hint">ID：<code>{{ droneModal.form.drone_id }}</code></div>
            <div class="form-info-box">
              <template v-if="droneModal.form.drone_type === 'LightDrone'">
                轻型：最大载荷 1.5 kg · 巡航速度 15 m/s · 电池 100 Wh
              </template>
              <template v-else>
                重型：最大载荷 20 kg · 巡航速度 10 m/s · 电池 800 Wh
              </template>
            </div>
          </div>
          <div class="modal-footer">
            <button class="btn-cancel" @click="closeDroneModal">取消</button>
            <button class="btn-save" @click="saveDrone">保存</button>
          </div>
        </div>
      </div>
    </Teleport>
  </PageShell>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import PageShell from '@/components/PageShell/index.vue'
import { useEntityStore, genEntityId } from '@/stores/entity'
import type { TruckConfig, DroneConfig } from '@/types'

const entityStore = useEntityStore()

// ── Truck Modal ────────────────────────────────────────────────────
const mkTruck = (): TruckConfig => ({
  truck_id: '', name: '', speed: 15, max_inventory: 30,
  swap_time: 90, parking_slots: 2, home_depot_id: '',
})

const truckModal  = reactive({ open: false, mode: 'add' as 'add' | 'edit', form: mkTruck() })
const truckErrors = ref<Record<string, string>>({})

function validateTruck(): boolean {
  const e: Record<string, string> = {}
  const f = truckModal.form
  if (!f.name.trim())       e.name          = '请输入名称'
  if (f.speed        <= 0)  e.speed         = '速度必须 > 0'
  if (f.max_inventory < 1)  e.max_inventory = '至少 1 件'
  if (f.parking_slots < 1)  e.parking_slots = '至少 1 个泊位'
  if (f.swap_time    <= 0)  e.swap_time     = '必须大于 0 秒'
  truckErrors.value = e
  return Object.keys(e).length === 0
}

function openAddTruck() {
  Object.assign(truckModal, { open: true, mode: 'add', form: { ...mkTruck(), truck_id: genEntityId('TRK') } })
  truckErrors.value = {}
}
function openEditTruck(item: TruckConfig) {
  Object.assign(truckModal, { open: true, mode: 'edit', form: { ...item } })
  truckErrors.value = {}
}
function closeTruckModal() { truckModal.open = false }
function saveTruck() {
  if (!validateTruck()) return
  truckModal.mode === 'add'
    ? entityStore.addTruck({ ...truckModal.form })
    : entityStore.updateTruck({ ...truckModal.form })
  closeTruckModal()
}
function deleteTruck(id: string) {
  if (window.confirm('确认删除该卡车？')) entityStore.removeTruck(id)
}

// ── Drone Modal ────────────────────────────────────────────────────
const mkDrone = (): DroneConfig => ({
  drone_id: '', drone_type: 'LightDrone', home_type: 'DEPOT', home_id: '',
})

const droneModal  = reactive({ open: false, mode: 'add' as 'add' | 'edit', form: mkDrone() })
const droneErrors = ref<Record<string, string>>({})

function validateDrone(): boolean {
  const e: Record<string, string> = {}
  if (!droneModal.form.home_id) e.home_id = '请选择归属目标'
  droneErrors.value = e
  return Object.keys(e).length === 0
}

function openAddDrone() {
  Object.assign(droneModal, { open: true, mode: 'add', form: { ...mkDrone(), drone_id: genEntityId('UAV') } })
  droneErrors.value = {}
}
function openEditDrone(item: DroneConfig) {
  Object.assign(droneModal, { open: true, mode: 'edit', form: { ...item } })
  droneErrors.value = {}
}
function closeDroneModal() { droneModal.open = false }
function saveDrone() {
  if (!validateDrone()) return
  droneModal.mode === 'add'
    ? entityStore.addDrone({ ...droneModal.form })
    : entityStore.updateDrone({ ...droneModal.form })
  closeDroneModal()
}
function deleteDrone(id: string) {
  if (window.confirm('确认删除该无人机？')) entityStore.removeDrone(id)
}
</script>

<style scoped>
.fleet-summary {
  display: flex;
  gap: var(--hl-space-md);
  flex-wrap: wrap;
  margin-bottom: var(--hl-space-lg);
}
.stat-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: 999px;
  padding: 8px 18px;
  box-shadow: var(--hl-shadow-sm);
}
.stat-pill__icon  { font-size: 16px; }
.stat-pill__value { font-size: 18px; font-weight: 700; color: var(--hl-text); }
.stat-pill__label { font-size: 12px; color: var(--hl-text-muted); }

.fleet-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--hl-space-md);
}

.fleet-panel {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
  min-width: 0;
}
.fleet-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
  gap: 8px;
}
.fleet-panel__title-row { display: flex; align-items: center; gap: 8px; }
.fleet-panel__title     { font-size: 13px; font-weight: 600; color: var(--hl-text); }
.fleet-panel__count {
  font-size: 11px;
  background: var(--hl-primary-alpha);
  color: var(--hl-primary);
  padding: 1px 7px;
  border-radius: 99px;
}

.fleet-table-head {
  display: flex;
  align-items: center;
  padding: 7px 16px;
  gap: 8px;
  border-bottom: 1px solid var(--hl-border);
  font-size: 11px;
  font-weight: 600;
  color: var(--hl-text-muted);
  text-transform: uppercase;
}

.tc-name  { flex: 1.2; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tc-id    { flex: 1;   min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tc-num   { width: 68px; text-align: right; flex-shrink: 0; }
.tc-depot { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tc-ops   { width: 60px; text-align: right; flex-shrink: 0; }

.dc-id        { flex: 1.2; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dc-type      { width: 60px; flex-shrink: 0; }
.dc-home-type { width: 70px; flex-shrink: 0; }
.dc-home      { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dc-ops       { width: 60px; text-align: right; flex-shrink: 0; }

.fleet-row {
  display: flex;
  align-items: center;
  padding: 9px 16px;
  gap: 8px;
  border-bottom: 1px solid var(--hl-border);
  font-size: 12.5px;
  color: var(--hl-text-secondary);
  transition: background var(--hl-transition);
}
.fleet-row:last-child { border-bottom: none; }
.fleet-row:hover      { background: var(--hl-primary-light); }

.fleet-row__primary { font-weight: 500; color: var(--hl-text); }
.fleet-row__id      { font-family: monospace; font-size: 11px; color: var(--hl-text-muted); }
.fleet-row__muted   { color: var(--hl-text-muted); font-size: 12px; }
.fleet-row__actions { display: flex; align-items: center; justify-content: flex-end; gap: 4px; }

.fleet-empty {
  padding: 32px 16px;
  text-align: center;
  font-size: 12px;
  color: var(--hl-text-muted);
}

.type-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 4px;
}
.type-badge--light { background: #dbeafe; color: #1e40af; }
.type-badge--heavy { background: #fef3c7; color: #92400e; }

.btn-add {
  height: 28px;
  padding: 0 12px;
  border: 1px solid var(--hl-primary);
  border-radius: var(--hl-border-radius);
  background: none;
  color: var(--hl-primary);
  font-size: 12px;
  cursor: pointer;
  transition: background var(--hl-transition), color var(--hl-transition);
}
.btn-add:hover { background: var(--hl-primary); color: #fff; }

.btn-icon {
  width: 26px;
  height: 26px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: none;
  cursor: pointer;
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background var(--hl-transition), border-color var(--hl-transition);
}
.btn-icon:hover      { background: var(--hl-primary-alpha); border-color: var(--hl-primary); }
.btn-icon--del:hover { background: var(--hl-danger-light);  border-color: var(--hl-danger); }

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(15, 23, 42, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-shadow-lg);
  width: 460px;
  max-width: calc(100vw - 32px);
  max-height: 90vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--hl-border);
  flex-shrink: 0;
}
.modal-title { font-size: 14px; font-weight: 600; color: var(--hl-text); }
.modal-close {
  width: 28px;
  height: 28px;
  border: none;
  background: none;
  font-size: 14px;
  cursor: pointer;
  color: var(--hl-text-muted);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background var(--hl-transition);
}
.modal-close:hover { background: var(--hl-content-bg); color: var(--hl-text); }

.modal-body   { padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.modal-footer {
  padding: 14px 20px;
  border-top: 1px solid var(--hl-border);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  flex-shrink: 0;
}

.form-group { display: flex; flex-direction: column; gap: 4px; }
.form-group label { font-size: 12px; font-weight: 500; color: var(--hl-text-secondary); }
.form-group input,
.form-group select {
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  background: var(--hl-card-bg);
  color: var(--hl-text);
  font-size: 13px;
  width: 100%;
  outline: none;
  transition: border-color var(--hl-transition);
}
.form-group input:focus,
.form-group select:focus { border-color: var(--hl-primary); }

.form-row   { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.form-error { font-size: 11px; color: var(--hl-danger); }

.form-id-hint {
  font-size: 11.5px;
  color: var(--hl-text-muted);
  background: var(--hl-content-bg);
  border-radius: 6px;
  padding: 7px 10px;
}
.form-id-hint code { font-family: monospace; color: var(--hl-primary); }

.form-info-box {
  font-size: 12px;
  color: var(--hl-text-muted);
  background: var(--hl-primary-light);
  border-radius: 6px;
  padding: 8px 12px;
  border: 1px solid #bfdbfe;
}

.btn-cancel {
  height: 32px;
  padding: 0 16px;
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  background: none;
  color: var(--hl-text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: background var(--hl-transition);
}
.btn-cancel:hover { background: var(--hl-content-bg); }

.btn-save {
  height: 32px;
  padding: 0 20px;
  border: none;
  border-radius: var(--hl-border-radius);
  background: var(--hl-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background var(--hl-transition);
}
.btn-save:hover { background: var(--hl-primary-dark); }
</style>
