<template>
  <div id="topbar">
    <div>
      <div class="title">✈ UAV 禁飞区地图</div>
      <div class="subtitle">上海建筑高度数据 2020 · WGS1984 · shp</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
      <div id="status-dot" :class="status.dot"></div>
      <div id="status-text">{{ status.text }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  status: { dot: string; text: string }
}>()
</script>
