<!-- 通用页面骨架：提供统一的页头 + 内容区布局 -->
<template>
  <div class="page-shell">
    <!-- 页头 -->
    <header class="page-shell__header">
      <div class="page-shell__header-left">
        <span class="page-shell__icon" aria-hidden="true">{{ icon }}</span>
        <div>
          <h1 class="page-shell__title">{{ title }}</h1>
          <p v-if="desc" class="page-shell__desc">{{ desc }}</p>
        </div>
      </div>
      <div class="page-shell__header-right">
        <slot name="actions" />
        <span v-if="badge" class="page-shell__badge" :class="`page-shell__badge--${badgeType}`">
          {{ badge }}
        </span>
      </div>
    </header>

    <!-- 内容区 -->
    <div class="page-shell__body">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  icon:       string
  title:      string
  desc?:      string
  badge?:     string
  badgeType?: 'dev' | 'live' | 'info'
}>(), {
  badgeType: 'dev',
})
</script>

<style scoped>
.page-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* ── 页头 ────────────────────────────────────────────────────────── */
.page-shell__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--hl-page-padding);
  height: 64px;
  flex-shrink: 0;
  background: var(--hl-card-bg);
  border-bottom: 1px solid var(--hl-border);
}

.page-shell__header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-shell__header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.page-shell__icon {
  font-size: 24px;
  line-height: 1;
}

.page-shell__title {
  font-size: 17px;
  font-weight: 700;
  color: var(--hl-text);
  letter-spacing: -0.2px;
}

.page-shell__desc {
  font-size: 12px;
  color: var(--hl-text-muted);
  margin-top: 1px;
}

/* Badge */
.page-shell__badge {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.5px;
  padding: 2px 8px;
  border-radius: 4px;
}

.page-shell__badge--dev  { background: #fef3c7; color: #92400e; }
.page-shell__badge--live { background: #dcfce7; color: #166534; }
.page-shell__badge--info { background: #dbeafe; color: #1e40af; }

/* ── 内容区 ──────────────────────────────────────────────────────── */
.page-shell__body {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: var(--hl-page-padding);
}
</style>
