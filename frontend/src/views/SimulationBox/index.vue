<template>
  <div class="sim-box">
    <!-- Tab 导航栏 -->
    <div class="sim-box__tabs" role="tablist" aria-label="仿真配置选项卡">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="sim-box__tab"
        :class="{ 'sim-box__tab--active': activeTab === tab.id }"
        :aria-selected="activeTab === tab.id"
        role="tab"
        @click="activeTab = tab.id"
      >
        <span class="sim-tab__icon">{{ tab.icon }}</span>
        <span class="sim-tab__label">{{ tab.label }}</span>
        <span v-if="tab.badge" class="sim-tab__badge" :class="`sim-tab__badge--${tab.badgeType}`">
          {{ tab.badge }}
        </span>
      </button>
    </div>

    <!-- Tab 内容区 -->
    <div class="sim-box__content" role="tabpanel">
      <!-- Tab 0：地图选取与环境构建（GeoTool） -->
      <GeoToolView v-if="activeTab === 'geo'" />

      <!-- Tab 1：动态订单生成器 -->
      <div v-else-if="activeTab === 'orders'" class="orders-tab">

        <!-- 控制卡片 -->
        <div class="orders-control-card">
          <div class="oc-header">
            <span class="oc-title">⚙️ 生成器参数</span>
            <div class="oc-controls">
              <button class="btn-gen-once" @click="orderStore.generateOnce()">单次生成</button>
              <button
                v-if="!orderStore.generatorRunning"
                class="btn-start"
                @click="orderStore.startGenerator()"
              >▶ 启动</button>
              <button
                v-else
                class="btn-stop"
                @click="orderStore.stopGenerator()"
              >⏹ 停止</button>
              <button class="btn-clear" @click="clearOrders">清空</button>
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

            <!-- 收货点分布 -->
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

            <!-- 突发控制 -->
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
                      <span class="param-value">× {{ orderStore.generatorConfig.burst_multiplier ?? 3 }}</span>
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

            <!-- 总量控制 -->
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
            </div><!-- /param-section: 总量控制 -->
          </div><!-- /oc-params -->

          <div class="oc-status" :class="orderStore.generatorRunning ? 'oc-status--running' : ''">
            <span v-if="orderStore.generatorRunning" class="status-dot status-dot--live"></span>
            <span v-else class="status-dot status-dot--idle"></span>
            {{ orderStore.generatorRunning ? '生成器运行中' : '生成器已停止' }}
            &nbsp;·&nbsp;
            共 {{ orderStore.totalCount }} 笔订单
            &nbsp;·&nbsp;
            待处理 {{ orderStore.pendingOrders.length }} 笔
          </div>
        </div>

        <!-- 仓库来源说明 -->
        <div v-if="orderStore.warehousePool.length" class="warehouse-info">
          <span class="wi-label">📦 仓库池：</span>
          <span
            v-for="w in orderStore.warehousePool"
            :key="w.id"
            class="wi-tag"
          >{{ w.name }}</span>
          <span v-if="!entityStore.depots.length" class="wi-default">（默认后备池，请在基础设施页面添加仓库）</span>
        </div>

        <!-- 订单列表 -->
        <div class="orders-list-card">
          <div class="ol-head">
            <div class="ol-col-id">订单 ID</div>
            <div class="ol-col-wh">仓库</div>
            <div class="ol-col-num">重量 (kg)</div>
            <div class="ol-col-pri">优先级</div>
            <div class="ol-col-ddl">截止时间</div>
            <div class="ol-col-status">状态</div>
          </div>

          <div v-if="!orderStore.recentOrders.length" class="ol-empty">
            暂无订单 · 点击「启动」或「单次生成」开始
          </div>

          <div
            v-for="o in orderStore.recentOrders"
            :key="o.order_id"
            class="ol-row"
          >
            <div class="ol-col-id ol-id">{{ o.order_id }}</div>
            <div class="ol-col-wh ol-muted">{{ o.warehouse_name || o.pickup_source_id }}</div>
            <div class="ol-col-num">{{ o.payload_weight }}</div>
            <div class="ol-col-pri">
              <span class="pri-badge" :class="`pri-badge--${(o.priority || 'normal').toLowerCase()}`">
                {{ o.priority_label || o.priority }}
              </span>
            </div>
            <div class="ol-col-ddl ol-muted">{{ o.deadline_iso || formatTime(o.deadline) }}</div>
            <div class="ol-col-status">
              <span class="status-badge" :class="`status-badge--${o.status.toLowerCase()}`">
                {{ statusLabel(o.status) }}
              </span>
            </div>
          </div>
        </div>

      </div>

      <!-- Tab 2：仿真后端控制台 -->
      <div v-else-if="activeTab === 'solver'" class="sim-ctrl-tab">

        <!-- 实时状态栏 -->
        <div class="sc-status" :class="systemStore.running ? 'sc-status--on' : 'sc-status--off'">
          <span class="sc-dot" :class="systemStore.running ? 'sc-dot--live' : 'sc-dot--idle'"></span>
          <span>{{ systemStore.running ? '仿真运行中' : (initDone ? '已初始化，等待启动' : '未初始化') }}</span>
          <span class="sc-sep">|</span>
          <span>仿真时钟：<strong>{{ systemStore.simTime.toFixed(2) }} s</strong></span>
          <span class="sc-sep">|</span>
          <span>倍率：× {{ systemStore.speedRatio }}</span>
          <span class="sc-sep">|</span>
          <span>运行模式：<strong>{{ systemStore.policyActive ? 'PPO 在线推理' : 'Classic 仿真' }}</strong></span>
        </div>

        <!-- 实体摘要 -->
        <div class="sc-entity-row">
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.depots.length }">
            🏭 仓库 <strong>{{ entityStore.depots.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.stations.length }">
            ⚡ 充换电站 <strong>{{ entityStore.stations.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.trucks.length }">
            🚛 卡车 <strong>{{ entityStore.trucks.length }}</strong>
          </div>
          <div class="sc-ent" :class="{ 'sc-ent--empty': !entityStore.drones.length }">
            🚁 无人机 <strong>{{ entityStore.drones.length }}</strong>
          </div>
        </div>

        <!-- bbox 提示 -->
        <div class="sc-bbox-hint">
          <span v-if="sceneStore.context">
            📐 使用场景 bbox：{{ sceneStore.context.road_network.bounds.min_lng.toFixed(4) }},
            {{ sceneStore.context.road_network.bounds.min_lat.toFixed(4) }}
            → {{ sceneStore.context.road_network.bounds.max_lng.toFixed(4) }},
            {{ sceneStore.context.road_network.bounds.max_lat.toFixed(4) }}
          </span>
          <span v-else class="sc-bbox-hint--warn">⚠️ 未加载仿真场景，将使用默认 bbox（上海）</span>
        </div>

        <!-- 操作按钮行 -->
        <div class="sc-action-row">
          <button class="sc-btn sc-btn--init" :disabled="initLoading" @click="doInit">
            {{ initLoading ? '⏳ 初始化中...' : '🚀 初始化并发送到后端' }}
          </button>
          <button class="sc-btn sc-btn--start"
            :disabled="!initDone || systemStore.running"
            @click="systemStore.start()">▶ 启动</button>
          <button class="sc-btn sc-btn--pause"
            :disabled="!systemStore.running"
            @click="systemStore.pause()">⏸ 暂停</button>
          <button class="sc-btn sc-btn--reset" @click="doReset">🔄 重置</button>
        </div>

        <div class="sc-policy-card">
          <div class="sc-policy-head">
            <span class="sc-policy-title">🤖 PPO 在线推理</span>
            <span class="sc-policy-badge" :class="systemStore.policyActive ? 'sc-policy-badge--on' : 'sc-policy-badge--off'">
              {{ systemStore.policyActive ? '已激活' : '未激活' }}
            </span>
          </div>
          <div class="sc-policy-grid">
            <label class="sc-policy-field">
              <span>策略权重路径</span>
              <input v-model="policyPath" type="text" placeholder="weights/.../policy_best.pt" />
            </label>
            <label class="sc-policy-field">
              <span>配置文件路径</span>
              <input v-model="policyConfigPath" type="text" placeholder="config/rh_alns_cmrappo.yaml" />
            </label>
            <label class="sc-policy-field">
              <span>场景 ID</span>
              <input v-model="policySceneId" type="text" readonly />
            </label>
            <label class="sc-policy-field">
              <span>订单来源</span>
              <select v-model="policyOrderSourceMode">
                <option value="benchmark">benchmark</option>
                <option value="poisson">poisson</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>
          </div>
          <div class="sc-policy-options">
            <label class="sc-policy-check">
              <input v-model="policyDeterministic" type="checkbox" />
              <span>确定性推理</span>
            </label>
            <label class="sc-policy-check">
              <input v-model="policyUseCurrentInitPayload" type="checkbox" />
              <span>复用当前初始化实体与订单</span>
            </label>
            <span class="sc-policy-runtime">runtime: {{ systemStore.policyRuntimeType }}</span>
            <span class="sc-policy-runtime">init_scene: {{ systemStore.initializedSceneId || '未初始化' }}</span>
          </div>
          <div class="sc-policy-hint" :class="ppoReadyForActivation ? 'sc-policy-hint--ok' : 'sc-policy-hint--warn'">
            {{ ppoCompatibilityMessage }}
          </div>
          <div class="sc-action-row">
            <button class="sc-btn sc-btn--policy"
              :disabled="!ppoReadyForActivation || policyLoading || systemStore.policyActive"
              @click="doActivatePolicy">
              {{ policyLoading ? '⏳ 激活中...' : '🤖 切换到 PPO 在线策略' }}
            </button>
            <button class="sc-btn sc-btn--policy-off"
              :disabled="policyLoading || !systemStore.policyActive"
              @click="doDeactivatePolicy">
              ↩ 切回 Classic 仿真
            </button>
            <span class="sc-policy-summary">
              {{ systemStore.policyActive ? `当前策略：${systemStore.policyName || 'PPO Policy'}` : '当前未使用策略运行时' }}
            </span>
          </div>
          <div v-if="systemStore.policyCheckpoint" class="sc-policy-checkpoint">
            checkpoint：{{ systemStore.policyCheckpoint }}
          </div>
        </div>

        <!-- 速率选择 -->
        <div class="sc-speed-row">
          <span class="sc-speed-label">仿真倍率</span>
          <div class="sc-speed-btns">
            <button v-for="s in [0.5, 1, 2, 5, 10]" :key="s"
              class="sc-spd" :class="{ 'sc-spd--active': systemStore.speedRatio === s }"
              @click="doSetSpeed(s)">× {{ s }}</button>
          </div>
        </div>

        <!-- 后端响应日志 -->
        <div class="sc-log">
          <div class="sc-log__head">
            📋 后端响应日志
            <button class="sc-log__clear" @click="ctrlLogs = []">清空</button>
          </div>
          <div v-if="!ctrlLogs.length" class="sc-log__empty">
            点击「初始化」后，后端响应与控制结果将显示在这里
          </div>
          <div v-for="(l, i) in ctrlLogs" :key="i"
            class="sc-log__row" :class="`sc-log__row--${l.type}`">
            <span class="sc-log__ts">{{ l.ts }}</span>
            <span class="sc-log__msg">{{ l.msg }}</span>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import GeoToolView from '@/views/GeoTool/index.vue'
import ConfigItem  from './components/ConfigItem.vue'
import { useOrderStore } from '@/stores/order'
import { useEntityStore } from '@/stores/entity'
import { useSystemStore } from '@/stores/system'
import { useSceneStore }  from '@/stores/scene'
import type { OrderGeneratorConfig } from '@/types'

const orderStore  = useOrderStore()
const entityStore = useEntityStore()
const systemStore = useSystemStore()
const sceneStore  = useSceneStore()

const activeTab = ref<'geo' | 'orders' | 'solver'>('geo')

// 折叠面板开关
const showDistSection  = ref(false)
const showBurstSection = ref(false)
const showLimitSection = ref(false)

// ── 仿真控制台状态 ──────────────────────────────────────────────────
const initLoading = ref(false)
const initDone    = ref(false)
type PolicyOrderSourceMode = 'benchmark' | 'poisson' | 'hybrid'
const PPO_SCENE_ID = 'default_test_4x4km'
const policyLoading = ref(false)
const policyPath = ref('config/policy_best.pt')
const policyConfigPath = ref('config/rh_alns_cmrappo.yaml')
const policySceneId = ref(PPO_SCENE_ID)
const policyDeterministic = ref(true)
const policyUseCurrentInitPayload = ref(true)
const policyOrderSourceMode = ref<PolicyOrderSourceMode>('benchmark')
const currentSceneId = computed(() => String(sceneStore.context?.scene_id ?? '').trim())
const ppoCurrentSceneCompatible = computed(() => currentSceneId.value === PPO_SCENE_ID)
const ppoInitSceneCompatible = computed(() => String(systemStore.initializedSceneId ?? '').trim() === PPO_SCENE_ID)
const ppoReadyForActivation = computed(() =>
  systemStore.initialized &&
  ppoCurrentSceneCompatible.value &&
  ppoInitSceneCompatible.value
)
const ppoCompatibilityMessage = computed(() => {
  if (!sceneStore.context) {
    return `请先加载默认预设场景 ${PPO_SCENE_ID}。`
  }
  if (!ppoCurrentSceneCompatible.value) {
    return `当前页面场景为 ${currentSceneId.value}，现有 PPO 权重仅支持 ${PPO_SCENE_ID}。`
  }
  if (!systemStore.initialized) {
    return `请先使用 ${PPO_SCENE_ID} 完成一次初始化。`
  }
  if (!ppoInitSceneCompatible.value) {
    return `最近一次初始化场景为 ${systemStore.initializedSceneId || 'unknown'}，请切回 ${PPO_SCENE_ID} 后重新初始化。`
  }
  return `当前页面与最近一次初始化均匹配 ${PPO_SCENE_ID}，可以激活 PPO。`
})
interface CtrlLog { type: 'info' | 'success' | 'error' | 'warn'; ts: string; msg: string }
const ctrlLogs    = ref<CtrlLog[]>([])

const tabs = computed(() => [
  { id: 'geo'    as const, icon: '🗺️', label: '地图选取与环境构建', badge: 'LIVE', badgeType: 'live' },
  { id: 'orders' as const, icon: '📦', label: '动态订单生成器',
    badge: orderStore.generatorRunning ? 'LIVE' : 'READY', badgeType: orderStore.generatorRunning ? 'live' : 'ready' },
  { id: 'solver' as const, icon: '🧠', label: '仿真控制台',
    badge: systemStore.running ? 'LIVE' : (initDone.value ? 'READY' : 'DEV'),
    badgeType: systemStore.running ? 'live' : (initDone.value ? 'ready' : 'dev') },
])

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

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function statusLabel(s: string): string {
  const map: Record<string, string> = {
    PENDING: '待分配', ASSIGNED: '已分配', IN_TRANSIT: '配送中',
    COMPLETED: '已完成', FAILED: '已失败',
  }
  return map[s] ?? s
}

// ── 仿真控制台操作函数 ─────────────────────────────────────────────
function onBurstEnabledChange(e: Event) {
  orderStore.updateConfig({ burst_enabled: (e.target as HTMLInputElement).checked })
}
function onMaxOrdersToggle(e: Event) {
  orderStore.updateConfig({ max_orders: (e.target as HTMLInputElement).checked ? 100 : null })
}

function _log(type: CtrlLog['type'], msg: string) {
  ctrlLogs.value.unshift({ type, ts: new Date().toLocaleTimeString('zh-CN'), msg })
}

async function doInit() {
  initLoading.value = true
  _log('info', '正在发送初始化请求到后端...')
  try {
    const bounds = sceneStore.context?.road_network.bounds
    const result = await systemStore.initSim({
      bbox:    bounds,
      sceneId: sceneStore.context?.scene_id,
    })
    await systemStore.fetchPolicyState().catch(() => {})
    initDone.value = true
    const s = result.summary
    _log('success', `✅ 初始化成功 — 仓库×${s.depots} · 站点×${s.stations} · 卡车×${s.trucks} · 无人机×${s.drones}`)
  } catch (e: any) {
    _log('error', `❌ 初始化失败：${e.message}`)
    initDone.value = false
  } finally {
    initLoading.value = false
  }
}

async function doReset() {
  await systemStore.reset().catch(() => {})
  initDone.value = false
  _log('warn', '🔄 仿真已重置，请重新初始化')
}

async function doSetSpeed(ratio: number) {
  await systemStore.setSpeed(ratio)
  _log('info', `⚡ 速率已设为 ×${ratio}`)
}

async function doActivatePolicy() {
  if (!ppoReadyForActivation.value) {
    _log('warn', `⚠️ ${ppoCompatibilityMessage.value}`)
    return
  }
  policyLoading.value = true
  _log('info', `🤖 正在激活 PPO 在线策略：${policyPath.value}`)
  try {
    await systemStore.activatePolicy({
      policy_name: 'rh_alns_cmrappo',
      policy_path: policyPath.value.trim(),
      config_path: policyConfigPath.value.trim(),
      scene_id: policySceneId.value.trim(),
      deterministic: policyDeterministic.value,
      speed_ratio: Number(systemStore.speedRatio) || 1,
      order_source_mode: policyOrderSourceMode.value,
      use_current_init_payload: policyUseCurrentInitPayload.value,
    })
    await systemStore.fetchPolicyState()
    _log('success', `✅ PPO 已激活：${systemStore.policyName || 'rh_alns_cmrappo'}，点击「启动」即可执行权重推理`)
  } catch (e: any) {
    _log('error', `❌ PPO 激活失败：${e.message}`)
  } finally {
    policyLoading.value = false
  }
}

async function doDeactivatePolicy() {
  policyLoading.value = true
  _log('info', '↩ 正在切回 Classic 仿真运行时...')
  try {
    await systemStore.deactivatePolicy()
    await systemStore.fetchPolicyState()
    _log('success', '✅ 已切回 Classic 仿真运行时')
  } catch (e: any) {
    _log('error', `❌ 切回 Classic 失败：${e.message}`)
  } finally {
    policyLoading.value = false
  }
}

onMounted(() => {
  systemStore.fetchPolicyState().catch(() => {})
})
</script>

<style scoped>
/* ── 整体容器 ─────────────────────────────────────────────────────── */
.sim-box {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Tab 导航栏 ──────────────────────────────────────────────────── */
.sim-box__tabs {
  display: flex;
  align-items: stretch;
  gap: 0;
  border-bottom: 1px solid var(--hl-border);
  background: var(--hl-card-bg);
  flex-shrink: 0;
  padding: 0 var(--hl-page-padding);
}

.sim-box__tab {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 0 18px;
  height: 44px;
  border: none;
  border-bottom: 2px solid transparent;
  background: none;
  font-size: 13.5px;
  font-weight: 500;
  color: var(--hl-text-muted);
  cursor: pointer;
  transition: color var(--hl-transition), border-color var(--hl-transition);
  white-space: nowrap;
}

.sim-box__tab:hover {
  color: var(--hl-text);
}

.sim-box__tab--active {
  color: var(--hl-primary);
  border-bottom-color: var(--hl-primary);
  font-weight: 600;
}

.sim-tab__icon  { font-size: 15px; line-height: 1; }
.sim-tab__label { }

.sim-tab__badge {
  font-size: 9px;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 3px;
  letter-spacing: 0.3px;
}

.sim-tab__badge--live { background: #dcfce7; color: #166534; }
.sim-tab__badge--dev  { background: #fef3c7; color: #92400e; }

/* ── Tab 内容区 ──────────────────────────────────────────────────── */
.sim-box__content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── 占位内容 ─────────────────────────────────────────────────────── */
.sim-placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 48px var(--hl-page-padding);
  overflow-y: auto;
}

.sim-placeholder__icon  { font-size: 52px; line-height: 1; }
.sim-placeholder__title { font-size: 18px; font-weight: 700; color: var(--hl-text); }
.sim-placeholder__desc  {
  font-size: 13px;
  color: var(--hl-text-muted);
  text-align: center;
  line-height: 1.8;
}

.config-grid {
  display: flex;
  gap: var(--hl-space-md);
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 8px;
}

/* ═══════════════════════════════════════════════════
   Orders Tab
   ═══════════════════════════════════════════════════ */
.orders-tab {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-md);
  padding: var(--hl-space-md) var(--hl-page-padding);
  overflow-y: auto;
}

/* ── 控制卡片 ──────────────────────────────────────────────────── */
.orders-control-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
}

.oc-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
}
.oc-title { font-size: 13px; font-weight: 600; color: var(--hl-text); }
.oc-controls { display: flex; gap: 8px; }

.btn-gen-once, .btn-start, .btn-stop, .btn-clear {
  height: 28px; padding: 0 13px; border-radius: var(--hl-border-radius);
  font-size: 12px; font-weight: 500; cursor: pointer;
  transition: background var(--hl-transition), color var(--hl-transition);
}
.btn-gen-once {
  border: 1px solid var(--hl-border); background: none;
  color: var(--hl-text-secondary);
}
.btn-gen-once:hover { background: var(--hl-content-bg); color: var(--hl-text); }

.btn-start {
  border: none; background: var(--hl-success); color: #fff;
}
.btn-start:hover { background: #15803d; }

.btn-stop {
  border: none; background: var(--hl-warning); color: #fff;
}
.btn-stop:hover { background: #b45309; }

.btn-clear {
  border: 1px solid var(--hl-border); background: none;
  color: var(--hl-danger);
}
.btn-clear:hover { background: var(--hl-danger-light); border-color: var(--hl-danger); }

/* ── 参数区 ─────────────────────────────────────────────────────── */
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

/* ── 状态栏 ─────────────────────────────────────────────────────── */
.oc-status {
  display: flex; align-items: center; gap: 6px;
  padding: 10px 16px; border-top: 1px solid var(--hl-border);
  font-size: 12px; color: var(--hl-text-muted);
  background: var(--hl-content-bg);
}
.oc-status--running { color: var(--hl-success); }

.status-dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.status-dot--live { background: var(--hl-success); animation: pulse 1.5s infinite; }
.status-dot--idle { background: var(--hl-text-muted); }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}

/* ── 仓库来源 ───────────────────────────────────────────────────── */
.warehouse-info {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  font-size: 12px; color: var(--hl-text-muted);
}
.wi-label { font-weight: 500; color: var(--hl-text-secondary); }
.wi-tag {
  background: var(--hl-primary-alpha); color: var(--hl-primary);
  border-radius: 4px; padding: 1px 7px; font-size: 11.5px;
}
.wi-default { font-size: 11.5px; color: var(--hl-warning); }

/* ── 订单列表 ───────────────────────────────────────────────────── */
.orders-list-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
  flex: 1;
}

.ol-head {
  display: flex; align-items: center;
  padding: 7px 16px; gap: 8px;
  border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
  font-size: 11px; font-weight: 600;
  color: var(--hl-text-muted); text-transform: uppercase;
  position: sticky; top: 0; z-index: 1;
}
.ol-row {
  display: flex; align-items: center;
  padding: 8px 16px; gap: 8px;
  border-bottom: 1px solid var(--hl-border);
  font-size: 12.5px; color: var(--hl-text-secondary);
  transition: background var(--hl-transition);
}
.ol-row:last-child { border-bottom: none; }
.ol-row:hover      { background: var(--hl-primary-light); }

.ol-col-id     { flex: 1.5; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ol-col-wh     { flex: 1;   min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ol-col-num    { width: 72px; text-align: right; flex-shrink: 0; }
.ol-col-pri    { width: 56px; flex-shrink: 0; }
.ol-col-ddl    { width: 72px; text-align: right; flex-shrink: 0; }
.ol-col-status { width: 68px; text-align: right; flex-shrink: 0; }

.ol-id   { font-family: monospace; font-size: 11.5px; color: var(--hl-text-muted); }
.ol-muted{ color: var(--hl-text-muted); }
.ol-empty { padding: 28px 16px; text-align: center; font-size: 12px; color: var(--hl-text-muted); }

.pri-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 5px; border-radius: 4px;
}
.pri-badge--urgent { background: #fee2e2; color: #991b1b; }
.pri-badge--normal { background: #dbeafe; color: #1e40af; }
.pri-badge--low    { background: #f1f5f9; color: #64748b; }

.status-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 6px; border-radius: 4px;
}
.status-badge--pending    { background: var(--hl-warning-light); color: #92400e; }
.status-badge--assigned   { background: #dbeafe; color: #1e40af; }
.status-badge--in_transit { background: var(--hl-success-light); color: #14532d; }
.status-badge--completed  { background: #d1fae5; color: var(--hl-success); }
.status-badge--failed     { background: var(--hl-danger-light);  color: var(--hl-danger); }

/* 新增 ready badge type */
.sim-tab__badge--ready { background: #d1fae5; color: #065f46; }

/* ── 折叠参数组 ──────────────────────────────────────── */
.param-section {
  border: 1px solid var(--hl-border);
  border-radius: var(--hl-border-radius);
  overflow: hidden;
}
.param-section__toggle {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  background: var(--hl-content-bg);
  border: none;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  color: var(--hl-text-secondary);
  text-align: left;
  transition: background var(--hl-transition), color var(--hl-transition);
}
.param-section__toggle:hover { background: var(--hl-primary-light); color: var(--hl-primary); }
.toggle-arrow { font-size: 10px; color: var(--hl-text-muted); }
.param-section__body {
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
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

.param-hint {
  font-size: 11px; color: var(--hl-text-muted); line-height: 1.5;
}

/* ══════════════════════════════════════════════════
   仿真控制台 Tab（solver）
   ══════════════════════════════════════════════════ */
.sim-ctrl-tab {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--hl-space-md);
  padding: var(--hl-space-md) var(--hl-page-padding);
  overflow-y: auto;
}

/* 状态栏 */
.sc-status {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 10px 14px; border-radius: var(--hl-card-radius);
  font-size: 12.5px;
  border: 1px solid var(--hl-border);
}
.sc-status--on  { background: #d1fae5; border-color: #6ee7b7; color: #065f46; }
.sc-status--off { background: var(--hl-card-bg); color: var(--hl-text-secondary); }
.sc-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sc-dot--live { background: #10b981; animation: pulse 1.5s infinite; }
.sc-dot--idle { background: var(--hl-text-muted); }
.sc-sep { color: var(--hl-border); padding: 0 2px; }

/* 实体摘要行 */
.sc-entity-row {
  display: flex; gap: 10px; flex-wrap: wrap;
}
.sc-ent {
  flex: 1; min-width: 120px;
  padding: 10px 14px;
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  font-size: 13px; color: var(--hl-text-secondary);
}
.sc-ent strong { color: var(--hl-primary); font-size: 16px; margin-left: 4px; }
.sc-ent--empty strong { color: var(--hl-danger); }

/* bbox 提示 */
.sc-bbox-hint {
  font-size: 11.5px; color: var(--hl-text-muted);
  padding: 6px 12px;
  background: var(--hl-content-bg);
  border-radius: var(--hl-border-radius);
}
.sc-bbox-hint--warn { color: #d97706; }

/* 操作按钮行 */
.sc-action-row {
  display: flex; gap: 8px; flex-wrap: wrap;
}
.sc-btn {
  height: 34px; padding: 0 16px; border-radius: var(--hl-border-radius);
  font-size: 13px; font-weight: 500; cursor: pointer;
  transition: background var(--hl-transition), opacity var(--hl-transition);
  border: none;
}
.sc-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.sc-btn--init  { background: var(--hl-primary); color: #fff; flex: 1; min-width: 180px; }
.sc-btn--init:hover:not(:disabled)  { background: #1d4ed8; }
.sc-btn--start { background: var(--hl-success); color: #fff; }
.sc-btn--start:hover:not(:disabled) { background: #15803d; }
.sc-btn--pause { background: var(--hl-warning); color: #fff; }
.sc-btn--pause:hover:not(:disabled) { background: #b45309; }
.sc-btn--reset { background: none; border: 1px solid var(--hl-border); color: var(--hl-text-secondary); }
.sc-btn--reset:hover { background: var(--hl-content-bg); color: var(--hl-danger); border-color: var(--hl-danger); }

.sc-policy-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  background: linear-gradient(135deg, #eff6ff, #f8fafc);
  border: 1px solid #bfdbfe;
  border-radius: var(--hl-card-radius);
}

.sc-policy-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.sc-policy-title {
  font-size: 13px;
  font-weight: 700;
  color: #1e3a8a;
}

.sc-policy-badge {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.sc-policy-badge--on {
  background: #dcfce7;
  color: #166534;
}

.sc-policy-badge--off {
  background: #e2e8f0;
  color: #475569;
}

.sc-policy-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(220px, 1fr));
  gap: 10px;
}

.sc-policy-field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 0;
}

.sc-policy-field span {
  font-size: 12px;
  font-weight: 600;
  color: var(--hl-text-secondary);
}

.sc-policy-field input,
.sc-policy-field select {
  width: 100%;
  height: 34px;
  padding: 0 10px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #fff;
  color: var(--hl-text);
  font-size: 12px;
  outline: none;
}

.sc-policy-field input:focus,
.sc-policy-field select:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.sc-policy-options {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.sc-policy-check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--hl-text-secondary);
  cursor: pointer;
}

.sc-policy-check input {
  accent-color: #2563eb;
}

.sc-policy-runtime {
  font-size: 11.5px;
  color: #475569;
}

.sc-policy-hint {
  font-size: 12px;
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: 8px;
}

.sc-policy-hint--ok {
  background: #ecfdf5;
  color: #166534;
}

.sc-policy-hint--warn {
  background: #fff7ed;
  color: #9a3412;
}

.sc-btn--policy {
  background: #2563eb;
  color: #fff;
  min-width: 180px;
}

.sc-btn--policy:hover:not(:disabled) {
  background: #1d4ed8;
}

.sc-btn--policy-off {
  background: #fff;
  color: #334155;
  border: 1px solid #cbd5e1;
}

.sc-btn--policy-off:hover:not(:disabled) {
  background: #f8fafc;
}

.sc-policy-summary {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  font-size: 12px;
  color: #1e293b;
}

.sc-policy-checkpoint {
  font-size: 11.5px;
  color: #475569;
  word-break: break-all;
}

/* 速率选择 */
.sc-speed-row {
  display: flex; align-items: center; gap: 12px;
}
.sc-speed-label { font-size: 12px; font-weight: 500; color: var(--hl-text-secondary); white-space: nowrap; }
.sc-speed-btns  { display: flex; gap: 6px; flex-wrap: wrap; }
.sc-spd {
  height: 28px; min-width: 44px; padding: 0 10px;
  border: 1px solid var(--hl-border); border-radius: 6px;
  background: var(--hl-card-bg); font-size: 12px;
  color: var(--hl-text-secondary); cursor: pointer;
  transition: all var(--hl-transition);
}
.sc-spd:hover      { border-color: var(--hl-primary); color: var(--hl-primary); }
.sc-spd--active    { background: var(--hl-primary); color: #fff; border-color: var(--hl-primary); font-weight: 600; }

/* 后端响应日志 */
.sc-log {
  flex: 1; min-height: 180px;
  background: #0f172a;
  border-radius: var(--hl-card-radius);
  overflow: hidden;
  display: flex; flex-direction: column;
}
.sc-log__head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px;
  font-size: 12px; font-weight: 500;
  color: #94a3b8;
  border-bottom: 1px solid #1e293b;
}
.sc-log__clear {
  font-size: 11px; color: #64748b; background: none; border: none;
  cursor: pointer; padding: 0 4px;
}
.sc-log__clear:hover { color: #94a3b8; }
.sc-log__empty {
  flex: 1; display: flex; align-items: center; justify-content: center;
  font-size: 12px; color: #475569;
}
.sc-log__row {
  display: flex; gap: 10px; align-items: baseline;
  padding: 4px 14px;
  font-size: 12px; font-family: 'Courier New', monospace;
  border-bottom: 1px solid #1e293b;
}
.sc-log__ts  { color: #475569; white-space: nowrap; flex-shrink: 0; }
.sc-log__msg { color: #e2e8f0; word-break: break-all; }
.sc-log__row--success .sc-log__msg { color: #4ade80; }
.sc-log__row--error   .sc-log__msg { color: #f87171; }
.sc-log__row--warn    .sc-log__msg { color: #fbbf24; }
.sc-log__row--info    .sc-log__msg { color: #94a3b8; }

@media (max-width: 1080px) {
  .sc-policy-grid {
    grid-template-columns: 1fr;
  }
}
</style>
