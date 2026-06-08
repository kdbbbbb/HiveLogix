<!-- SimulationBox 内部可复用配置项卡片 -->
<template>
  <div class="config-item">
    <span class="config-item__icon" aria-hidden="true">{{ icon }}</span>
    <p class="config-item__title">{{ title }}</p>
    <p class="config-item__desc">{{ desc }}</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  icon:  string
  title: string
  desc:  string
}>()
</script>

<style scoped>
.config-item {
  background: var(--hl-card-bg);
  border: 1px solid var(--hl-card-border);
  border-radius: var(--hl-card-radius);
  box-shadow: var(--hl-card-shadow);
  padding: 18px 20px;
  width: 200px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  text-align: center;
  align-items: center;
}

.config-item__icon  { font-size: 28px; }
.config-item__title { font-size: 13px; font-weight: 600; color: var(--hl-text); }
.config-item__desc  { font-size: 11.5px; color: var(--hl-text-muted); line-height: 1.5; }
</style>
