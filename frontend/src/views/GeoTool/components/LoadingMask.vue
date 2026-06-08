<template>
  <div id="loading-mask" :class="{ hidden }">
    <div class="spinner"></div>
    <div class="loading-text">正在加载上海建筑高度数据…</div>
    <div style="width:260px;">
      <div class="progress-dark">
        <div class="progress-bar-accent" :style="{ width: progress + '%' }"></div>
      </div>
    </div>
    <div class="loading-sub">{{ subText }}</div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  progress: number
  subText:  string
  hidden:   boolean
}>()
</script>
