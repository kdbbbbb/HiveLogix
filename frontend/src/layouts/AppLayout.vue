<template>
  <div class="hl-layout">
    <NavSidebar />
    <main class="hl-layout__content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import NavSidebar from '@/components/NavSidebar/index.vue'
</script>

<style scoped>
.hl-layout {
  display: flex;
  height: 100vh;
  width: 100%;
  overflow: hidden;
}

/* 内容区：撑满剩余空间，子页面按需决定是否滚动 */
.hl-layout__content {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: var(--hl-content-bg);
}
</style>
