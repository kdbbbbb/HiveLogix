<!-- DispatchCenter 内部可复用卡片组件 -->
<template>
  <div class="section-card">
    <div class="section-card__header">
      <span class="section-card__icon" aria-hidden="true">{{ icon }}</span>
      <span class="section-card__title">{{ title }}</span>
    </div>
    <div class="section-card__body">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  icon:  string
  title: string
}>()
</script>

<style scoped>
.section-card {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  overflow: hidden;
}

.section-card__header {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 11px 14px;
  border-bottom: 1px solid var(--hl-border);
  background: var(--hl-content-bg);
}

.section-card__icon { font-size: 14px; }

.section-card__title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--hl-text);
}

.section-card__body {
  padding: 12px 14px;
}
</style>
