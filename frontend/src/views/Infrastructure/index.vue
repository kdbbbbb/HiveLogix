<template>
  <PageShell
    icon="🏗️"
    title="基础设施配置"
    desc="仓库网点 · 充换电站网络 · 站点部署与参数配置"
  >
    <div class="infra-grid">
      <!-- 坐标自动分配提示横幅 -->
      <div class="infra-unlocated-banner infra-unlocated-banner--info">
        📐 坐标由系统自动分配 &nbsp;·&nbsp;
        在「实时指挥中心」点击「初始化并发送到后端」时，仓库将自动分配到地图中心区，充换电站均匀铺满全图，无需手动填写经纬度
      </div>
      <!-- 仓库面板 -->
      <div class="infra-panel">
        <div class="infra-panel__header">
          <div class="infra-panel__title-row">
            <span class="infra-panel__title">🏭 仓库网点</span>
            <span class="infra-panel__count">{{ entityStore.depots.length }} 个</span>
          </div>
          <button class="btn-add" @click="openAddDepot">+ 新增仓库</button>
        </div>

        <div class="infra-table-head">
          <span class="col-name">名称</span>
          <span class="col-id">ID</span>
          <span class="col-coord">坐标</span>
          <span class="col-num">容量</span>
          <span class="col-num">泊位</span>
          <span class="col-num">τ_swap (s)</span>
          <span class="col-ops">操作</span>
        </div>

        <template v-if="entityStore.depots.length">
          <div v-for="d in entityStore.depots" :key="d.depot_id" class="infra-row">
            <span class="col-name infra-row__primary">{{ d.name }}</span>
            <span class="col-id infra-row__id">{{ d.depot_id }}</span>
            <span class="col-coord infra-row__muted">
              <span v-if="d.lng === 0 && d.lat === 0" class="badge-auto">📐 初始化时自动分配</span>
              <template v-else>{{ d.lng.toFixed(4) }}, {{ d.lat.toFixed(4) }}</template>
            </span>
            <span class="col-num">{{ d.capacity }}</span>
            <span class="col-num">{{ d.parking_slots }}</span>
            <span class="col-num">{{ d.swap_time }}</span>
            <span class="col-ops infra-row__actions">
              <button class="btn-icon" title="编辑" @click="openEditDepot(d)">✏️</button>
              <button class="btn-icon btn-icon--del" title="删除" @click="deleteDepot(d.depot_id)">🗑️</button>
            </span>
          </div>
        </template>
        <div v-else class="infra-empty">🏭 暂无仓库 · 点击右上角新增</div>
      </div>

      <!-- 充换电站面板 -->
      <div class="infra-panel">
        <div class="infra-panel__header">
          <div class="infra-panel__title-row">
            <span class="infra-panel__title">⚡ 充换电站</span>
            <span class="infra-panel__count">{{ entityStore.stations.length }} 个</span>
          </div>
          <button class="btn-add" @click="openAddStation">+ 新增站点</button>
        </div>

        <div class="infra-table-head">
          <span class="col-name">名称</span>
          <span class="col-id">ID</span>
          <span class="col-coord">坐标</span>
          <span class="col-num">泊位</span>
          <span class="col-num">τ_swap (s)</span>
          <span class="col-ops">操作</span>
        </div>

        <template v-if="entityStore.stations.length">
          <div v-for="s in entityStore.stations" :key="s.station_id" class="infra-row">
            <span class="col-name infra-row__primary">{{ s.name }}</span>
            <span class="col-id infra-row__id">{{ s.station_id }}</span>
            <span class="col-coord infra-row__muted">
              <span v-if="s.lng === 0 && s.lat === 0" class="badge-auto">📐 初始化时自动分配</span>
              <template v-else>{{ s.lng.toFixed(4) }}, {{ s.lat.toFixed(4) }}</template>
            </span>
            <span class="col-num">{{ s.parking_slots }}</span>
            <span class="col-num">{{ s.swap_time }}</span>
            <span class="col-ops infra-row__actions">
              <button class="btn-icon" title="编辑" @click="openEditStation(s)">✏️</button>
              <button class="btn-icon btn-icon--del" title="删除" @click="deleteStation(s.station_id)">🗑️</button>
            </span>
          </div>
        </template>
        <div v-else class="infra-empty">⚡ 暂无站点 · 点击右上角新增</div>
      </div>
    </div>

    <!-- 说明框 -->
    <div class="infra-note">
      <strong>补能模式参数说明：</strong>
      充换电站（Station）与仓库（Depot）均支持换电服务，<code>swap_time</code> 为单次换电耗时（秒），
      <code>parking_slots</code> 为同时容纳无人机数量。仓库 <code>capacity</code> 表示最大库存包裹件数。
      <br /><strong>仿真假设：</strong>货物供应和换电资源永远充足，<code>capacity</code> 与 <code>parking_slots</code> 仅为调度算法提供约束参考，前端不做消耗计算。
      <br /><strong>坐标分配：</strong>无需手动填写经纬度。每次在「实时指挥中心」点击「初始化并发送到后端」时，系统自动根据仿真场景 bbox 均匀分配坐标——仓库分布在地图中心区域，充换电站均匀铺满全图。
    </div>

    <!-- 仓库弹窗 -->
    <Teleport to="body">
      <div v-if="depotModal.open" class="modal-overlay" @click.self="closeDepotModal">
        <div class="modal-card">
          <div class="modal-header">
            <span class="modal-title">{{ depotModal.mode === 'add' ? '新增仓库' : '编辑仓库' }}</span>
            <button class="modal-close" @click="closeDepotModal">✕</button>
          </div>
          <div class="modal-body">
            <div class="form-hint-auto">📐 经纬度由系统自动分配（初始化时均匀布局到地图中心区域）</div>
            <div class="form-group">
              <label>名称 *</label>
              <input v-model="depotModal.form.name" placeholder="如：浦东枢纽仓" maxlength="32" />
              <span v-if="depotErrors.name" class="form-error">{{ depotErrors.name }}</span>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>海拔高度 (m)</label>
                <input v-model.number="depotModal.form.altitude" type="number" step="1" placeholder="0" />
              </div>
              <div class="form-group">
                <label>库存容量（件）*</label>
                <input v-model.number="depotModal.form.capacity" type="number" min="1" placeholder="200" />
                <span v-if="depotErrors.capacity" class="form-error">{{ depotErrors.capacity }}</span>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>无人机泊位数 *</label>
                <input v-model.number="depotModal.form.parking_slots" type="number" min="1" max="20" placeholder="4" />
                <span v-if="depotErrors.parking_slots" class="form-error">{{ depotErrors.parking_slots }}</span>
              </div>
              <div class="form-group">
                <label>换电耗时 τ_swap (s) *</label>
                <input v-model.number="depotModal.form.swap_time" type="number" min="1" placeholder="90" />
                <span v-if="depotErrors.swap_time" class="form-error">{{ depotErrors.swap_time }}</span>
              </div>
            </div>
            <div class="form-id-hint">ID：<code>{{ depotModal.form.depot_id }}</code></div>
          </div>
          <div class="modal-footer">
            <button class="btn-cancel" @click="closeDepotModal">取消</button>
            <button class="btn-save" @click="saveDepot">保存</button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- 充换电站弹窗 -->
    <Teleport to="body">
      <div v-if="stationModal.open" class="modal-overlay" @click.self="closeStationModal">
        <div class="modal-card">
          <div class="modal-header">
            <span class="modal-title">{{ stationModal.mode === 'add' ? '新增充换电站' : '编辑充换电站' }}</span>
            <button class="modal-close" @click="closeStationModal">✕</button>
          </div>
          <div class="modal-body">
            <div class="form-hint-auto">📐 经纬度由系统自动分配（初始化时均匀铺满全图）</div>
            <div class="form-group">
              <label>名称 *</label>
              <input v-model="stationModal.form.name" placeholder="如：虹桥补能站-1" maxlength="32" />
              <span v-if="stationErrors.name" class="form-error">{{ stationErrors.name }}</span>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>海拔高度 (m)</label>
                <input v-model.number="stationModal.form.altitude" type="number" step="1" placeholder="0" />
              </div>
              <div class="form-group">
                <label>无人机泊位数 *</label>
                <input v-model.number="stationModal.form.parking_slots" type="number" min="1" max="20" placeholder="2" />
                <span v-if="stationErrors.parking_slots" class="form-error">{{ stationErrors.parking_slots }}</span>
              </div>
            </div>
            <div class="form-group">
              <label>换电耗时 τ_swap (s) *</label>
              <input v-model.number="stationModal.form.swap_time" type="number" min="1" placeholder="60" />
              <span v-if="stationErrors.swap_time" class="form-error">{{ stationErrors.swap_time }}</span>
            </div>
            <div class="form-id-hint">ID：<code>{{ stationModal.form.station_id }}</code></div>
          </div>
          <div class="modal-footer">
            <button class="btn-cancel" @click="closeStationModal">取消</button>
            <button class="btn-save" @click="saveStation">保存</button>
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
import type { DepotConfig, StationConfig } from '@/types'

const entityStore = useEntityStore()

// ── Depot Modal ─────────────────────────────────────────────────────
const mkDepot = (): DepotConfig => ({
  depot_id: '', name: '', lng: 0, lat: 0, altitude: 0,
  capacity: 200, parking_slots: 4, swap_time: 90,
})

const depotModal  = reactive({ open: false, mode: 'add' as 'add' | 'edit', form: mkDepot() })
const depotErrors = ref<Record<string, string>>({})

function validateDepot(): boolean {
  const e: Record<string, string> = {}
  const f = depotModal.form
  if (!f.name.trim())       e.name          = '请输入名称'
  if (f.capacity      < 1)  e.capacity      = '容量至少 1 件'
  if (f.parking_slots < 1)  e.parking_slots = '至少 1 个泊位'
  if (f.swap_time     <= 0) e.swap_time     = '必须 > 0 秒'
  // lng/lat 由系统自动分配，不做校验
  depotErrors.value = e
  return Object.keys(e).length === 0
}

function openAddDepot() {
  Object.assign(depotModal, { open: true, mode: 'add', form: { ...mkDepot(), depot_id: genEntityId('DEP') } })
  depotErrors.value = {}
}
function openEditDepot(item: DepotConfig) {
  Object.assign(depotModal, { open: true, mode: 'edit', form: { ...item } })
  depotErrors.value = {}
}
function closeDepotModal() { depotModal.open = false }
function saveDepot() {
  if (!validateDepot()) return
  depotModal.mode === 'add'
    ? entityStore.addDepot({ ...depotModal.form })
    : entityStore.updateDepot({ ...depotModal.form })
  closeDepotModal()
}
function deleteDepot(id: string) {
  if (window.confirm('确认删除该仓库？')) entityStore.removeDepot(id)
}

// ── Station Modal ───────────────────────────────────────────────────
const mkStation = (): StationConfig => ({
  station_id: '', name: '', lng: 0, lat: 0, altitude: 0,
  parking_slots: 2, swap_time: 60,
})

const stationModal  = reactive({ open: false, mode: 'add' as 'add' | 'edit', form: mkStation() })
const stationErrors = ref<Record<string, string>>({})

function validateStation(): boolean {
  const e: Record<string, string> = {}
  const f = stationModal.form
  if (!f.name.trim())       e.name          = '请输入名称'
  if (f.parking_slots < 1)  e.parking_slots = '至少 1 个泊位'
  if (f.swap_time     <= 0) e.swap_time     = '必须 > 0 秒'
  // lng/lat 由系统自动分配，不做校验
  stationErrors.value = e
  return Object.keys(e).length === 0
}

function openAddStation() {
  Object.assign(stationModal, { open: true, mode: 'add', form: { ...mkStation(), station_id: genEntityId('STA') } })
  stationErrors.value = {}
}
function openEditStation(item: StationConfig) {
  Object.assign(stationModal, { open: true, mode: 'edit', form: { ...item } })
  stationErrors.value = {}
}
function closeStationModal() { stationModal.open = false }
function saveStation() {
  if (!validateStation()) return
  stationModal.mode === 'add'
    ? entityStore.addStation({ ...stationModal.form })
    : entityStore.updateStation({ ...stationModal.form })
  closeStationModal()
}
function deleteStation(id: string) {
  if (window.confirm('确认删除该充换电站？')) entityStore.removeStation(id)
}
</script>

<style scoped>
.infra-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--hl-space-md);
  margin-bottom: var(--hl-space-md);
}

.infra-panel {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
  min-width: 0;
}
.infra-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
  gap: 8px;
}
.infra-panel__title-row { display: flex; align-items: center; gap: 8px; }
.infra-panel__title     { font-size: 13px; font-weight: 600; color: var(--hl-text); }
.infra-panel__count {
  font-size: 11px;
  background: var(--hl-primary-alpha);
  color: var(--hl-primary);
  padding: 1px 7px;
  border-radius: 99px;
}

.infra-table-head {
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

.col-name  { flex: 1.2; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.col-id    { flex: 1;   min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.col-coord { flex: 1.6; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.col-num   { width: 64px; text-align: right; flex-shrink: 0; }
.col-ops   { width: 60px; text-align: right; flex-shrink: 0; }

.infra-row {
  display: flex;
  align-items: center;
  padding: 9px 16px;
  gap: 8px;
  border-bottom: 1px solid var(--hl-border);
  font-size: 12.5px;
  color: var(--hl-text-secondary);
  transition: background var(--hl-transition);
}
.infra-row:last-child { border-bottom: none; }
.infra-row:hover      { background: var(--hl-primary-light); }

.infra-row__primary { font-weight: 500; color: var(--hl-text); }
.infra-row__id      { font-family: monospace; font-size: 11px; color: var(--hl-text-muted); }
.infra-row__muted   { color: var(--hl-text-muted); font-size: 12px; }
.infra-row__actions { display: flex; align-items: center; justify-content: flex-end; gap: 4px; }

.infra-empty {
  padding: 32px 16px;
  text-align: center;
  font-size: 12px;
  color: var(--hl-text-muted);
}

.infra-note {
  font-size: 12px;
  color: var(--hl-text-muted);
  background: var(--hl-content-bg);
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-card-radius);
  padding: 12px 16px;
  line-height: 1.7;
}
.infra-note code {
  font-family: monospace;
  font-size: 11px;
  background: var(--hl-primary-alpha);
  color: var(--hl-primary);
  padding: 0 4px;
  border-radius: 3px;
}

.infra-unlocated-banner {
  grid-column: 1 / -1;
  border-radius: var(--hl-card-radius);
  padding: 10px 14px;
  font-size: 12.5px;
  margin-bottom: 0;
  line-height: 1.6;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1e40af;
}

.infra-unlocated-banner--info {
  background: #eff6ff;
  border-color: #bfdbfe;
  color: #1e40af;
}

.badge-unlocated {
  display: inline-block;
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
  border-radius: 4px;
  padding: 0 5px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}

.badge-auto {
  display: inline-block;
  background: #eff6ff;
  color: #1e40af;
  border: 1px solid #bfdbfe;
  border-radius: 4px;
  padding: 0 5px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}

/* 表单顶部自动分配提示 */
.form-hint-auto {
  padding: 7px 11px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  color: #1e40af;
  font-size: 12px;
  margin-bottom: 12px;
}

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
  width: 480px;
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
.form-group input {
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
.form-group input:focus { border-color: var(--hl-primary); }

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
