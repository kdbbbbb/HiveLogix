<template>
  <PageShell
    icon="📦"
    title="订单与任务管理"
    desc="订单全量表 · 派送任务拆解 · 履约状态追踪"
    badge="开发中"
    badge-type="dev"
  >
    <!-- ⚙️ 订单生成参数配置面板（迁自 SimulationBox） -->
    <div class="orders-config-card">
      <div class="oc-header">
        <span class="oc-title">⚙️ 订单生成参数配置</span>
        <div class="oc-controls">
          <button class="btn-gen-once" @click="orderStore.generateOnce()">📋 预览一笔</button>
          <button class="btn-gen-batch" @click="generateBatch(10)">📦 快速生成10笔</button>
          <button class="btn-clear" @click="clearOrders">🗑️ 清空预览</button>
        </div>
      </div>

      <div class="oc-params">
        <!-- 到达率 -->
        <div class="param-item">
          <label>到达频率（单/分钟）</label>
          <div class="param-input-row">
            <input
              type="range" min="0.5" max="30" step="0.5"
              :value="orderStore.generatorConfig.arrival_rate"
              @input="updateRate($event)"
            />
            <span class="param-value">{{ orderStore.generatorConfig.arrival_rate }}</span>
          </div>
        </div>

        <!-- 重量范围 -->
        <div class="param-group">
          <label>货物重量范围 (kg)</label>
          <div class="param-twin">
            <div class="param-item-sm">
              <span class="twin-label">最小</span>
              <input
                type="number" min="0.1" max="19.9" step="0.1"
                :value="orderStore.generatorConfig.weight_min"
                @change="updateField('weight_min', $event)"
              />
            </div>
            <span class="twin-sep">~</span>
            <div class="param-item-sm">
              <span class="twin-label">最大</span>
              <input
                type="number" min="0.2" max="20" step="0.1"
                :value="orderStore.generatorConfig.weight_max"
                @change="updateField('weight_max', $event)"
              />
            </div>
          </div>
        </div>

        <!-- 时间窗 -->
        <div class="param-group">
          <label>配送时间窗范围（分钟）</label>
          <div class="param-twin">
            <div class="param-item-sm">
              <span class="twin-label">最小</span>
              <input
                type="number" min="5" max="119" step="1"
                :value="orderStore.generatorConfig.window_min"
                @change="updateField('window_min', $event)"
              />
            </div>
            <span class="twin-sep">~</span>
            <div class="param-item-sm">
              <span class="twin-label">最大</span>
              <input
                type="number" min="6" max="120" step="1"
                :value="orderStore.generatorConfig.window_max"
                @change="updateField('window_max', $event)"
              />
            </div>
          </div>
        </div>

        <!-- 优先级权重 -->
        <div class="param-group priority-group">
          <label>优先级权重分布（紧急 / 普通 / 低）</label>
          <div class="priority-inputs">
            <div class="priority-item">
              <span class="priority-dot priority-dot--urgent"></span>
              <input
                type="number" min="0" max="100" step="1"
                :value="orderStore.generatorConfig.priority_urgent"
                @change="updateField('priority_urgent', $event)"
              />
            </div>
            <span class="twin-sep">/</span>
            <div class="priority-item">
              <span class="priority-dot priority-dot--normal"></span>
              <input
                type="number" min="0" max="100" step="1"
                :value="orderStore.generatorConfig.priority_normal"
                @change="updateField('priority_normal', $event)"
              />
            </div>
            <span class="twin-sep">/</span>
            <div class="priority-item">
              <span class="priority-dot priority-dot--low"></span>
              <input
                type="number" min="0" max="100" step="1"
                :value="orderStore.generatorConfig.priority_low"
                @change="updateField('priority_low', $event)"
              />
            </div>
          </div>
        </div>

        <!-- 收货点分布（折叠） -->
        <div class="param-section">
          <button class="param-section__toggle" @click="showDistSection = !showDistSection">
            <span>📍 收货点分布</span>
            <span class="toggle-arrow">{{ showDistSection ? '▲' : '▼' }}</span>
          </button>
          <div v-show="showDistSection" class="param-section__body">
            <div class="param-group">
              <label>地理模式</label>
              <div class="geo-mode-options">
                <label class="radio-option">
                  <input type="radio" name="geoMode" value="uniform"
                    :checked="(orderStore.generatorConfig.geo_mode ?? 'uniform') === 'uniform'"
                    @change="orderStore.updateConfig({ geo_mode: 'uniform' })" />
                  <span>bbox 均匀</span>
                </label>
                <label class="radio-option">
                  <input type="radio" name="geoMode" value="clustered"
                    :checked="(orderStore.generatorConfig.geo_mode ?? 'uniform') === 'clustered'"
                    @change="orderStore.updateConfig({ geo_mode: 'clustered' })" />
                  <span>热点聚集</span>
                </label>
                <label class="radio-option">
                  <input type="radio" name="geoMode" value="depot_nearby"
                    :checked="(orderStore.generatorConfig.geo_mode ?? 'uniform') === 'depot_nearby'"
                    @change="orderStore.updateConfig({ geo_mode: 'depot_nearby' })" />
                  <span>仓库覆盖圈</span>
                </label>
              </div>
            </div>
            <div v-if="(orderStore.generatorConfig.geo_mode ?? 'uniform') !== 'uniform'" class="param-item">
              <label>覆盖半径（km）</label>
              <div class="param-input-row">
                <input type="range" min="0.5" max="5" step="0.5"
                  :value="orderStore.generatorConfig.cluster_radius_km ?? 1.5"
                  @input="updateField('cluster_radius_km', $event)" />
                <span class="param-value">{{ orderStore.generatorConfig.cluster_radius_km ?? 1.5 }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 突发控制（折叠） -->
        <div class="param-section">
          <button class="param-section__toggle" @click="showBurstSection = !showBurstSection">
            <span>⚡ 突发控制</span>
            <span class="toggle-arrow">{{ showBurstSection ? '▲' : '▼' }}</span>
          </button>
          <div v-show="showBurstSection" class="param-section__body">
            <div class="param-item">
              <label class="checkbox-label">
                <input type="checkbox"
                  :checked="orderStore.generatorConfig.burst_enabled ?? false"
                  @change="onBurstEnabledChange" />
                启用突发模式
              </label>
            </div>
            <template v-if="orderStore.generatorConfig.burst_enabled">
              <div class="param-item">
                <label>突发倍率</label>
                <div class="param-input-row">
                  <input type="range" min="2" max="10" step="1"
                    :value="orderStore.generatorConfig.burst_multiplier ?? 3"
                    @input="updateField('burst_multiplier', $event)" />
                  <span class="param-value">× {{ orderStore.generatorConfig.burst_multiplier ?? 3 }}</span>
                </div>
              </div>
              <div class="param-group">
                <label>突发持续时间（秒）</label>
                <div class="param-item-sm">
                  <input type="number" min="30" max="300" step="10"
                    :value="orderStore.generatorConfig.burst_duration_s ?? 60"
                    @change="updateField('burst_duration_s', $event)" />
                </div>
              </div>
            </template>
          </div>
        </div>

        <!-- 总量控制（折叠） -->
        <div class="param-section">
          <button class="param-section__toggle" @click="showLimitSection = !showLimitSection">
            <span>🔢 总量控制</span>
            <span class="toggle-arrow">{{ showLimitSection ? '▲' : '▼' }}</span>
          </button>
          <div v-show="showLimitSection" class="param-section__body">
            <div class="param-item">
              <label class="checkbox-label">
                <input type="checkbox"
                  :checked="(orderStore.generatorConfig.max_orders ?? null) !== null"
                  @change="onMaxOrdersToggle" />
                设定订单总量上限
              </label>
            </div>
            <div v-if="(orderStore.generatorConfig.max_orders ?? null) !== null" class="param-group">
              <label>最大订单数</label>
              <div class="param-item-sm">
                <input type="number" min="1" max="9999" step="1"
                  :value="orderStore.generatorConfig.max_orders ?? 100"
                  @change="updateField('max_orders', $event)" />
              </div>
            </div>
            <div class="param-hint">
              当前已生成 {{ orderStore.totalCount }} 笔
              <template v-if="(orderStore.generatorConfig.max_orders ?? null) !== null">
                &nbsp;/&nbsp;上限 {{ orderStore.generatorConfig.max_orders }} 笔
              </template>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 📦 仓库池来源提示 -->
    <div v-if="orderStore.warehousePool.length" class="warehouse-info">
      <span class="wi-label">📦 仓库池：</span>
      <span
        v-for="w in orderStore.warehousePool"
        :key="w.id"
        class="wi-tag"
      >{{ w.name }}</span>
      <span v-if="!entityStore.depots.length" class="wi-default">（默认后备池，请在基础设施页面添加仓库）</span>
    </div>

    <!-- 📡 后端订单实况 -->
    <div class="stats-panel" :class="{ 'stats-panel--active': orderStore.stats !== null }">
      <div class="stats-panel__header">
        <span class="stats-dot" :class="systemStore.running ? 'stats-dot--live' : ''"></span>
        📡 后端订单实况
        <span v-if="systemStore.running" class="stats-hint">（仿真运行中实时刷新）</span>
        <span v-else class="stats-hint">（仿真未运行，以下为参数预览数据）</span>
      </div>
      <template v-if="orderStore.stats !== null">
        <div class="stats-row">
          <div class="stat-item">
            <span class="stat-item__value">{{ orderStore.stats.orders_pending }}</span>
            <span class="stat-item__label">待分配</span>
          </div>
          <div class="stat-item">
            <span class="stat-item__value stat-item__value--active">{{ orderStore.stats.orders_assigned }}</span>
            <span class="stat-item__label">配送中</span>
          </div>
          <div class="stat-item">
            <span class="stat-item__value stat-item__value--success">{{ orderStore.stats.orders_completed }}</span>
            <span class="stat-item__label">已完成</span>
          </div>
          <div class="stat-item">
            <span class="stat-item__value stat-item__value--danger">{{ orderStore.stats.orders_timeout }}</span>
            <span class="stat-item__label">超时</span>
          </div>
        </div>
      </template>
      <div v-else class="stats-empty">
        暂无后端数据 · 请在「实时指挥中心」初始化并启动仿真后查看
      </div>
    </div>

    <!-- 工具栏 -->
    <div class="toolbar">
      <div class="toolbar__filters">
        <select class="filter-select" v-model="statusFilter">
          <option value="all">全部状态</option>
          <option value="PENDING">待分配</option>
          <option value="ASSIGNED">已分配</option>
          <option value="IN_TRANSIT">履帄中</option>
          <option value="COMPLETED">已完成</option>
          <option value="FAILED">已失败</option>
        </select>
        <select class="filter-select" v-model="modeFilter">
          <option value="all">全部模式</option>
          <option value="DRONE_DIRECT">直送 (A/B)</option>
          <option value="DRONE_TRUCK_DEPOT">卡车中继 (C/D)</option>
          <option value="MULTI_HOP">多跳中继 (E)</option>
        </select>
        <span class="filter-hint">共 {{ filteredOrders.length }} 笔</span>
      </div>
      <div class="toolbar__actions">
        <button class="btn-outline" @click="exportCsv">导出 CSV</button>
      </div>
    </div>

    <!-- 订单表格 -->
    <div class="table-card">
      <div class="table-head">
        <span class="tc tc--id">订单ID</span>
        <span class="tc tc--pri">优先级</span>
        <span class="tc tc--wh">来源仓库</span>
        <span class="tc tc--num">重量(kg)</span>
        <span class="tc tc--coord">投递坐标</span>
        <span class="tc tc--ddl">截止时间</span>
        <span class="tc tc--status">状态</span>
        <span class="tc tc--mode">履帄模式</span>
      </div>

      <template v-if="filteredOrders.length">
        <div v-for="o in filteredOrders" :key="o.order_id" class="table-row">
          <span class="tc tc--id  td-id"  >{{ o.order_id }}</span>
          <span class="tc tc--pri">
            <span class="pri-badge" :class="`pri-badge--${(o.priority||'normal').toLowerCase()}`">
              {{ o.priority_label || o.priority }}
            </span>
          </span>
          <span class="tc tc--wh  td-muted">{{ o.warehouse_name || o.pickup_source_id }}</span>
          <span class="tc tc--num td-num" >{{ o.payload_weight }}</span>
          <span class="tc tc--coord td-muted">{{ o.delivery_lng?.toFixed(4) }}, {{ o.delivery_lat?.toFixed(4) }}</span>
          <span class="tc tc--ddl td-muted">{{ o.deadline_iso || formatTime(o.deadline) }}</span>
          <span class="tc tc--status">
            <span class="status-badge" :class="`status-badge--${o.status.toLowerCase()}`">
              {{ statusLabel(o.status) }}
            </span>
          </span>
          <span class="tc tc--mode td-muted">{{ o.fulfillment_mode || '—' }}</span>
        </div>
      </template>

      <div v-else class="table-empty">
        <div class="table-empty__icon">📭</div>
        <p class="table-empty__text">暂无订单数据</p>
        <p class="table-empty__hint">请在「实时指挥中心」初始化并启动仿真，或点击上方「📋 预览一笔」生成样例订单</p>
      </div>
    </div>

    <!-- 任务拆解说明 -->
    <div class="feature-hint">
      <span class="feature-hint__icon">🚀</span>
      <div>
        <p class="feature-hint__title">派送任务拆解视图（树形表格）</p>
        <p class="feature-hint__desc">
          选中订单后，展开查看无人机 / 卡车完整流转节点：起飞 → 投递 → 中继补能（模式E）→ 汇合
        </p>
      </div>
    </div>
  </PageShell>
</template>

<script setup lang="ts">
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue'
import PageShell from '@/components/PageShell/index.vue'
import { useOrderStore }  from '@/stores/order'
import { useEntityStore } from '@/stores/entity'
import { useSystemStore } from '@/stores/system'
import type { OrderGeneratorConfig } from '@/types'

const orderStore  = useOrderStore()
const entityStore = useEntityStore()
const systemStore = useSystemStore()

const statusFilter = ref('all')
const modeFilter   = ref('all')

// 折叠面板开关（迁自 SimulationBox）
const showDistSection  = ref(false)
const showBurstSection = ref(false)
const showLimitSection = ref(false)

// ── 后端订单轮询（仿真运行时每 5s 自动拉取一次）──────────────────────────
let _pollTimer: ReturnType<typeof setInterval> | null = null

function _startPoll() {
  orderStore.fetchBackendOrders()
  if (!_pollTimer) {
    _pollTimer = setInterval(() => orderStore.fetchBackendOrders(), 5000)
  }
}

function _stopPoll() {
  if (_pollTimer !== null) {
    clearInterval(_pollTimer)
    _pollTimer = null
  }
}

watch(() => systemStore.running, (running) => {
  if (running) _startPoll()
  else          _stopPoll()
}, { immediate: false })

onMounted(() => {
  if (systemStore.running) _startPoll()
})

onBeforeUnmount(() => _stopPoll())

// ── 订单列表：仿真运行时优先展示后端实时订单，否则显示本地预览订单 ──────
const filteredOrders = computed(() => {
  const source = (systemStore.running && orderStore.backendOrders.length > 0)
    ? orderStore.backendOrders.map(o => ({
        ...o,
        // 字段映射：后端 assigned_mode → fulfillment_mode
        fulfillment_mode: o.assigned_mode ?? o.fulfillment_mode,
        // sim_s 截止时间转为可读标签
        deadline_iso: o.time_domain === 'sim_s'
          ? `第 ${Math.round(o.deadline)} 秒`
          : undefined,
      }))
    : orderStore.recentOrders

  return source.filter((o: any) => {
    const statusOk = statusFilter.value === 'all' || o.status === statusFilter.value
    const modeOk   = modeFilter.value   === 'all'
      || o.fulfillment_mode === modeFilter.value
      || o.assigned_mode    === modeFilter.value
    return statusOk && modeOk
  })
})

function statusLabel(s: string): string {
  const map: Record<string, string> = {
    PENDING: '待分配', ASSIGNED: '已分配', IN_TRANSIT: '配送中',
    COMPLETED: '已完成', FAILED: '已失败',
  }
  return map[s] ?? s
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

// ── 参数配置函数（迁自 SimulationBox）─────────────────────────────────
function updateRate(e: Event) {
  const v = parseFloat((e.target as HTMLInputElement).value)
  orderStore.updateConfig({ arrival_rate: v })
  orderStore.restartIfRunning()
}

function updateField(key: keyof OrderGeneratorConfig, e: Event) {
  const v = parseFloat((e.target as HTMLInputElement).value)
  if (!isNaN(v)) {
    orderStore.updateConfig({ [key]: v } as Partial<OrderGeneratorConfig>)
  }
}

function clearOrders() {
  if (window.confirm('确认清空所有已生成订单？')) orderStore.clearOrders()
}

function generateBatch(count: number) {
  const generated = orderStore.generateBatch(count)
  const msg = `已生成 ${generated} 笔订单`
  alert(msg)
}

function onBurstEnabledChange(e: Event) {
  orderStore.updateConfig({ burst_enabled: (e.target as HTMLInputElement).checked })
}

function onMaxOrdersToggle(e: Event) {
  orderStore.updateConfig({ max_orders: (e.target as HTMLInputElement).checked ? 100 : null })
}

function exportCsv() {
  if (!filteredOrders.value.length) return
  const header = '订单ID,优先级,来源仓库,重量(kg),投递经度,投递纬度,截止时间,状态,履单模式'
  const rows = filteredOrders.value.map(o =>
    [
      o.order_id, o.priority || '', o.warehouse_name || o.pickup_source_id,
      o.payload_weight, o.delivery_lng, o.delivery_lat,
      o.deadline_iso || formatTime(o.deadline),
      statusLabel(o.status), o.fulfillment_mode || '',
    ].join(',')
  )
  const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `orders_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
/* ── 工具栏 ──────────────────────────────────────────────────────── */
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--hl-space-md);
  gap: var(--hl-space-md);
}

.toolbar__filters { display: flex; gap: var(--hl-space-sm); }
.toolbar__actions  { display: flex; gap: var(--hl-space-sm); }

.filter-select {
  height: 32px;
  padding: 0 10px;
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  background: var(--hl-card-bg);
  font-size: 13px;
  color: var(--hl-text-secondary);
  cursor: not-allowed;
  opacity: 0.7;
}

.btn-outline {
  height: 32px;
  padding: 0 14px;
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  background: none;
  font-size: 13px;
  color: var(--hl-text-secondary);
  cursor: pointer;
  transition: background var(--hl-transition);
}
.btn-outline:hover { background: var(--hl-content-bg); }

/* ── 过滤提示 ────────────────────────────────────────────────────── */
.filter-hint {
  font-size: 12px; color: var(--hl-text-muted);
  padding: 0 4px; white-space: nowrap;
}

/* ── 过滤 select 可用状态 ────────────────────────────────────────── */
.filter-select { cursor: pointer; opacity: 1; }

/* ── 表格列宽定义 (共享 header/row) ──────────────────────────────── */
.tc { font-size: 11.5px; flex-shrink: 0; }
.tc--id     { flex: 1.5; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tc--pri    { width: 56px; }
.tc--wh     { flex: 1;   min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tc--num    { width: 72px; text-align: right; }
.tc--coord  { width: 136px; }
.tc--ddl    { width: 68px; text-align: right; }
.tc--status { width: 68px; }
.tc--mode   { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.td-id      { font-family: monospace; color: var(--hl-text-muted); font-size: 11px; }
.td-muted   { color: var(--hl-text-muted); }
.td-num     { font-variant-numeric: tabular-nums; }

/* ── 表格行 ──────────────────────────────────────────────────────── */
.table-row {
  display: flex; align-items: center;
  padding: 0 16px; height: 38px; gap: 12px;
  border-bottom: 1px solid var(--hl-border);
  color: var(--hl-text-secondary);
  transition: background var(--hl-transition);
}
.table-row:last-child { border-bottom: none; }
.table-row:hover      { background: var(--hl-primary-light); }

/* ── 优先级 badge ────────────────────────────────────────────────── */
.pri-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 5px; border-radius: 4px;
}
.pri-badge--urgent { background: #fee2e2; color: #991b1b; }
.pri-badge--normal { background: #dbeafe; color: #1e40af; }
.pri-badge--low    { background: #f1f5f9; color: #64748b; }

/* ── 状态 badge ──────────────────────────────────────────────────── */
.status-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 6px; border-radius: 4px;
}
.status-badge--pending    { background: var(--hl-warning-light); color: #92400e; }
.status-badge--assigned   { background: #dbeafe; color: #1e40af; }
.status-badge--in_transit { background: var(--hl-success-light); color: #14532d; }
.status-badge--completed  { background: #d1fae5; color: var(--hl-success); }
.status-badge--failed     { background: var(--hl-danger-light);  color: var(--hl-danger); }

/* ── 表格 ────────────────────────────────────────────────────────── */
.table-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
  margin-bottom: var(--hl-space-md);
}

.table-head {
  display: flex;
  align-items: center;
  padding: 0 16px;
  height: 40px;
  background: var(--hl-content-bg);
  border-bottom: 1px solid var(--hl-border);
  gap: 12px;
  font-weight: 600;
  color: var(--hl-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.2px;
}

.table-empty {
  padding: 48px 24px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.table-empty__icon { font-size: 36px; }
.table-empty__text { font-size: 14px; color: var(--hl-text-secondary); font-weight: 500; }
.table-empty__hint { font-size: 12px; color: var(--hl-text-muted); }

/* ── 功能说明条 ──────────────────────────────────────────────────── */
.feature-hint {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  background: var(--hl-primary-light);
  border: 1px solid #bfdbfe;
  border-radius: var(--hl-card-radius);
  padding: 14px 18px;
}

.feature-hint__icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.feature-hint__title { font-size: 13px; font-weight: 600; color: var(--hl-primary-dark); }
.feature-hint__desc  { font-size: 12px; color: var(--hl-text-secondary); margin-top: 4px; line-height: 1.6; }

/* ── 订单参数配置卡片 ───────────────────────────────────────────── */
.orders-config-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
  margin-bottom: var(--hl-space-md);
}

.oc-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
}
.oc-title { font-size: 13px; font-weight: 600; color: var(--hl-text); }
.oc-controls { display: flex; gap: 8px; }

.btn-gen-once, .btn-clear {
  height: 28px; padding: 0 13px; border-radius: var(--hl-border-radius);
  font-size: 12px; font-weight: 500; cursor: pointer;
  transition: background var(--hl-transition), color var(--hl-transition);
}
.btn-gen-once {
  border: 1px solid var(--hl-border); background: none;
  color: var(--hl-text-secondary);
}
.btn-gen-once:hover { background: var(--hl-content-bg); color: var(--hl-text); }
.btn-gen-batch {
  height: 28px; padding: 0 13px; border-radius: var(--hl-border-radius);
  font-size: 12px; font-weight: 500; cursor: pointer;
  border: 1px solid var(--hl-primary); background: var(--hl-primary);
  color: #fff;
  transition: background var(--hl-transition);
}
.btn-gen-batch:hover { background: #1d4ed8; }
.btn-clear {
  border: 1px solid var(--hl-border); background: none;
  color: var(--hl-danger);
}
.btn-clear:hover { background: var(--hl-danger-light); border-color: var(--hl-danger); }

/* ── 参数区 ──────────────────────────────────────────────── */
.oc-params {
  display: flex; flex-direction: column; gap: 14px;
  padding: 16px;
}
.param-item { display: flex; flex-direction: column; gap: 6px; }
.param-item label,
.param-group > label {
  font-size: 12px; font-weight: 500; color: var(--hl-text-secondary);
}
.param-input-row { display: flex; align-items: center; gap: 12px; }
.param-input-row input[type="range"] { flex: 1; accent-color: var(--hl-primary); }
.param-value {
  width: 36px; text-align: right;
  font-size: 13px; font-weight: 600; color: var(--hl-primary);
}
.param-group { display: flex; flex-direction: column; gap: 8px; }
.param-twin  { display: flex; align-items: center; gap: 8px; }
.twin-sep    { font-size: 14px; color: var(--hl-text-muted); flex-shrink: 0; }
.param-item-sm { display: flex; align-items: center; gap: 4px; flex: 1; }
.twin-label { font-size: 11px; color: var(--hl-text-muted); white-space: nowrap; }
.param-item-sm input {
  flex: 1; height: 30px; padding: 0 8px;
  border: 1px solid var(--hl-border); border-radius: 6px;
  background: var(--hl-card-bg); color: var(--hl-text); font-size: 13px;
  outline: none;
}
.param-item-sm input:focus { border-color: var(--hl-primary); }

.priority-inputs { display: flex; align-items: center; gap: 8px; }
.priority-item   { display: flex; align-items: center; gap: 5px; flex: 1; }
.priority-dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.priority-dot--urgent { background: #ef4444; }
.priority-dot--normal { background: #3b82f6; }
.priority-dot--low    { background: #94a3b8; }
.priority-item input {
  flex: 1; height: 30px; padding: 0 8px;
  border: 1px solid var(--hl-border); border-radius: 6px;
  background: var(--hl-card-bg); color: var(--hl-text); font-size: 13px;
  outline: none;
}
.priority-item input:focus { border-color: var(--hl-primary); }

/* ── 折叠参数组 ───────────────────────────── */
.param-section {
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  overflow: hidden;
}
.param-section__toggle {
  width: 100%;
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 10px;
  background: var(--hl-content-bg);
  border: none; cursor: pointer;
  font-size: 12px; font-weight: 500;
  color: var(--hl-text-secondary); text-align: left;
  transition: background var(--hl-transition), color var(--hl-transition);
}
.param-section__toggle:hover { background: var(--hl-primary-light); color: var(--hl-primary); }
.toggle-arrow { font-size: 10px; color: var(--hl-text-muted); }
.param-section__body {
  padding: 10px 12px;
  display: flex; flex-direction: column; gap: 10px;
  border-top: 1px solid var(--hl-border);
}
.geo-mode-options { display: flex; gap: 14px; flex-wrap: wrap; }
.radio-option {
  display: flex; align-items: center; gap: 5px;
  cursor: pointer; font-size: 12px; color: var(--hl-text-secondary);
}
.radio-option input { cursor: pointer; accent-color: var(--hl-primary); }
.checkbox-label {
  display: flex; align-items: center; gap: 6px;
  cursor: pointer; font-size: 12px;
  font-weight: 500; color: var(--hl-text-secondary);
}
.checkbox-label input { cursor: pointer; accent-color: var(--hl-primary); }
.param-hint { font-size: 11px; color: var(--hl-text-muted); line-height: 1.5; }

/* ── 仓库池来源提示 ───────────────────────────────────────── */
.warehouse-info {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  font-size: 12px; color: var(--hl-text-muted);
  margin-bottom: var(--hl-space-md);
}
.wi-label { font-weight: 500; color: var(--hl-text-secondary); }
.wi-tag {
  background: var(--hl-primary-alpha); color: var(--hl-primary);
  border-radius: 4px; padding: 1px 7px; font-size: 11.5px;
}
.wi-default { font-size: 11.5px; color: var(--hl-warning); }

/* ── 后端订单实况面板 ──────────────────────────────────── */
.stats-panel {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  padding: 14px 16px;
  margin-bottom: var(--hl-space-md);
}
.stats-panel--active { border-color: var(--hl-primary); }
.stats-panel__header {
  display: flex; align-items: center; gap: 6px;
  font-size: 12.5px; font-weight: 600;
  color: var(--hl-text); margin-bottom: 12px;
}
.stats-dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
  background: var(--hl-text-muted);
}
.stats-dot--live { background: var(--hl-success); animation: stats-pulse 1.5s infinite; }
@keyframes stats-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}
.stats-hint { font-size: 11px; color: var(--hl-text-muted); font-weight: 400; }
.stats-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: var(--hl-space-sm);
}
.stat-item {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 10px 8px;
  background: var(--hl-content-bg);
  border-radius: var(--hl-border-radius);
}
.stat-item__value {
  font-size: 22px; font-weight: 700;
  color: var(--hl-text); line-height: 1;
}
.stat-item__value--active  { color: #3b82f6; }
.stat-item__value--success { color: var(--hl-success); }
.stat-item__value--danger  { color: var(--hl-danger); }
.stat-item__label { font-size: 11px; color: var(--hl-text-muted); }
.stats-empty { font-size: 12px; color: var(--hl-text-muted); text-align: center; padding: 8px 0; }
</style>
