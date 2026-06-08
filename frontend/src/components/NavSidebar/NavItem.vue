<template>
  <router-link :to="path" class="nav-item" active-class="nav-item--active">
    <span class="nav-item__icon" aria-hidden="true">{{ icon }}</span>
    <span class="nav-item__label">{{ label }}</span>
    <span v-if="badge" class="nav-item__badge">{{ badge }}</span>
  </router-link>
</template>

<script setup lang="ts">
defineProps<{
  path:   string
  icon:   string
  label:  string
  badge?: string
}>()
</script>

<style scoped>
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  height: var(--hl-sidebar-item-h);
  padding: 0 12px;
  border-radius: var(--hl-sidebar-item-radius);
  text-decoration: none;
  color: var(--hl-nav-text);
  font-size: 13.5px;
  font-weight: 500;
  transition: background var(--hl-transition), color var(--hl-transition);
  white-space: nowrap;
  overflow: hidden;
  cursor: pointer;
}

.nav-item:hover {
  background: var(--hl-nav-hover-bg);
  color: #e2e8f0;
}

.nav-item--active {
  background: var(--hl-nav-active-bg);
  color: var(--hl-nav-text-active);
}

/* 图标 */
.nav-item__icon {
  font-size: 17px;
  width: 22px;
  text-align: center;
  flex-shrink: 0;
  line-height: 1;
}

/* 文字标签 */
.nav-item__label {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Badge 标记 */
.nav-item__badge {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.4px;
  padding: 1px 5px;
  border-radius: 3px;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.12);
  color: #94a3b8;
}

.nav-item--active .nav-item__badge {
  background: rgba(255, 255, 255, 0.22);
  color: #bfdbfe;
}
</style>
