<template>
  <PageShell
    icon="📊"
    title="调度性能大屏"
    desc="全局 KPI 统计 · 履约模式分布 · 能量效率 · 成本趋势"
    badge="开发中"
    badge-type="dev"
  >
    <!-- KPI 横排 -->
    <div class="kpi-row">
      <div v-for="k in kpiList" :key="k.label" class="kpi-card">
        <div class="kpi-card__icon">{{ k.icon }}</div>
        <div class="kpi-card__value">{{ k.value }}</div>
        <div class="kpi-card__label">{{ k.label }}</div>
        <div class="kpi-card__trend" :class="k.up ? 'trend--up' : 'trend--down'">
          {{ k.up ? '↑' : '↓' }} {{ k.change }}
        </div>
      </div>
    </div>

    <!-- 图表占位网格 -->
    <div class="chart-grid">
      <!-- 履约模式饼图 -->
      <div class="chart-card">
        <div class="chart-card__header">
          <span class="chart-card__title">履约模式构成</span>
          <span class="chart-card__sub">模式 A–E 触发频次与占比</span>
        </div>
        <div class="chart-card__body">
          <div class="mode-bars">
            <div v-for="m in modes" :key="m.label" class="mode-bar">
              <span class="mode-bar__label">{{ m.label }}</span>
              <div class="mode-bar__track">
                <div class="mode-bar__fill" :style="{width: m.pct + '%', background: m.color}" />
              </div>
              <span class="mode-bar__pct">{{ m.pct }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 能量监控 -->
      <div class="chart-card">
        <div class="chart-card__header">
          <span class="chart-card__title">能量与中继效率</span>
          <span class="chart-card__sub">充换电站排队拥堵 · 平均换电滞留时间</span>
        </div>
        <div class="chart-card__body chart-card__body--placeholder">
          <div class="placeholder-visual">
            <div class="placeholder-visual__grid">
              <div v-for="i in 20" :key="i"
                class="placeholder-visual__cell"
                :style="{opacity: Math.random() * 0.6 + 0.2, background: heatColor(i)}"
              />
            </div>
            <p class="placeholder-visual__hint">ECharts 热力图 · 待接入实时数据</p>
          </div>
        </div>
      </div>

      <!-- 成本趋势折线 -->
      <div class="chart-card chart-card--wide">
        <div class="chart-card__header">
          <span class="chart-card__title">时效与惩罚成本趋势</span>
          <span class="chart-card__sub">订单涌入量 vs 累计超时惩罚成本（双折线图）</span>
        </div>
        <div class="chart-card__body chart-card__body--placeholder">
          <div class="placeholder-visual">
            <!-- 模拟折线 SVG -->
            <svg viewBox="0 0 400 80" class="mock-chart" preserveAspectRatio="none">
              <polyline points="0,60 40,50 80,55 120,30 160,35 200,20 240,25 280,15 320,22 360,10 400,18"
                fill="none" stroke="var(--hl-primary)" stroke-width="2" />
              <polyline points="0,70 40,65 80,70 120,60 160,62 200,55 240,58 280,50 320,52 360,45 400,42"
                fill="none" stroke="var(--hl-danger)" stroke-width="2" stroke-dasharray="4 2" />
            </svg>
            <p class="placeholder-visual__hint">ECharts 时序折线图 · 待接入仿真数据流</p>
          </div>
        </div>
      </div>
    </div>
  </PageShell>
</template>

<script setup lang="ts">
import PageShell from '@/components/PageShell/index.vue'

const kpiList = [
  { icon: '✅', label: '综合任务完成率', value: '—',     change: '—',    up: true  },
  { icon: '⏱️', label: '准时送达率',     value: '—',     change: '—',    up: true  },
  { icon: '📉', label: '平均订单延迟',   value: '— min', change: '—',    up: false },
  { icon: '⚡', label: '总体能耗成本',   value: '— Wh',  change: '—',    up: false },
]

const modes = [
  { label: '模式A · 卡车直送',          pct: 0, color: '#2563eb' },
  { label: '模式B · 卡车+无人机',        pct: 0, color: '#7c3aed' },
  { label: '模式C · 仓库直飞',           pct: 0, color: '#0891b2' },
  { label: '模式D · 空投补货',           pct: 0, color: '#d97706' },
  { label: '模式E · 多跳中继',           pct: 0, color: '#dc2626' },
]

function heatColor(i: number) {
  const hue = 200 + i * 8
  return `hsl(${hue}, 70%, 55%)`
}
</script>

<style scoped>
/* ── KPI 横排 ─────────────────────────────────────────────────────── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--hl-space-md);
  margin-bottom: var(--hl-space-lg);
}

.kpi-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  padding: var(--hl-space-md) var(--hl-space-lg);
  box-shadow: var(--hl-card-shadow);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.kpi-card__icon { font-size: 20px; }

.kpi-card__value {
  font-size: 26px;
  font-weight: 700;
  color: var(--hl-text);
  line-height: 1.2;
}

.kpi-card__label {
  font-size: 12px;
  color: var(--hl-text-muted);
}

.kpi-card__trend {
  font-size: 11px;
  font-weight: 600;
  margin-top: 2px;
}
.trend--up   { color: var(--hl-success); }
.trend--down { color: var(--hl-danger);  }

/* ── 图表网格 ─────────────────────────────────────────────────────── */
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--hl-space-md);
}

.chart-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
}

.chart-card--wide {
  grid-column: 1 / -1;
}

.chart-card__header {
  padding: 14px 18px 10px;
  border-bottom: 1px solid var(--hl-border);
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.chart-card__title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--hl-text);
}

.chart-card__sub {
  font-size: 11px;
  color: var(--hl-text-muted);
}

.chart-card__body {
  padding: 18px;
  min-height: 200px;
}

.chart-card__body--placeholder {
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── 模式条形 ─────────────────────────────────────────────────────── */
.mode-bars { display: flex; flex-direction: column; gap: 10px; }

.mode-bar {
  display: flex;
  align-items: center;
  gap: 8px;
}

.mode-bar__label { font-size: 12px; color: var(--hl-text-secondary); width: 145px; flex-shrink: 0; }

.mode-bar__track {
  flex: 1;
  height: 6px;
  background: var(--hl-border);
  border-radius: 3px;
  overflow: hidden;
}

.mode-bar__fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.4s ease;
}

.mode-bar__pct { font-size: 11px; color: var(--hl-text-muted); width: 30px; text-align: right; }

/* ── 占位可视化 ───────────────────────────────────────────────────── */
.placeholder-visual { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 10px; }

.placeholder-visual__grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
  width: 100%;
  max-width: 300px;
}

.placeholder-visual__cell {
  aspect-ratio: 1;
  border-radius: 3px;
}

.placeholder-visual__hint {
  font-size: 11px;
  color: var(--hl-text-muted);
  text-align: center;
}

.mock-chart {
  width: 100%;
  height: 80px;
}
</style>
