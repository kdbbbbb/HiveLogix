<template>
  <aside class="nav-sidebar" role="navigation" aria-label="主导航">
    <!-- Logo 区域 -->
    <div class="nav-sidebar__logo">
      <span class="nav-sidebar__logo-icon" aria-hidden="true">🚁</span>
      <div class="nav-sidebar__logo-text">
        <span class="nav-sidebar__logo-name">HiveLogix</span>
        <span class="nav-sidebar__logo-sub">物流调度平台</span>
      </div>
    </div>

    <!-- 导航菜单 -->
    <nav class="nav-sidebar__nav">
      <NavItem
        v-for="item in navItems"
        :key="item.path"
        :icon="item.icon"
        :label="item.label"
        :path="item.path"
        :badge="item.badge"
      />
    </nav>

    <!-- 底部版本信息 -->
    <div class="nav-sidebar__footer">
      <span class="nav-sidebar__version">HiveLogix · v0.1.0-dev</span>
    </div>
  </aside>
</template>

<script setup lang="ts">
import NavItem from './NavItem.vue'

interface NavItemDef {
  path:   string
  icon:   string
  label:  string
  badge?: string
}

const navItems: NavItemDef[] = [
  { path: '/dispatch',   icon: '🗺️', label: '实时指挥中心' },
  { path: '/orders',     icon: '📦', label: '订单与任务管理' },
  { path: '/fleet',      icon: '🛸', label: '载具管理调度' },
  { path: '/infra',      icon: '🏗️', label: '基础设施配置' },
  { path: '/simulation', icon: '🌍', label: '环境构建' },
]
</script>

<style scoped>
/* ── 侧边栏容器 ──────────────────────────────────────────────────── */
.nav-sidebar {
  width: var(--hl-sidebar-w);
  background: var(--hl-sidebar-bg);
  border-right: 1px solid var(--hl-sidebar-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow: hidden;
  user-select: none;
}

/* ── Logo ────────────────────────────────────────────────────────── */
.nav-sidebar__logo {
  height: var(--hl-sidebar-logo-h);
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 11px;
  border-bottom: 1px solid var(--hl-sidebar-border);
  flex-shrink: 0;
}

.nav-sidebar__logo-icon {
  font-size: 26px;
  line-height: 1;
  flex-shrink: 0;
}

.nav-sidebar__logo-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 1px;
}

.nav-sidebar__logo-name {
  font-size: 15px;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 0.4px;
  white-space: nowrap;
}

.nav-sidebar__logo-sub {
  font-size: 10px;
  color: var(--hl-nav-text);
  white-space: nowrap;
  letter-spacing: 0.2px;
}

/* ── 导航区 ──────────────────────────────────────────────────────── */
.nav-sidebar__nav {
  flex: 1;
  padding: var(--hl-sidebar-padding);
  display: flex;
  flex-direction: column;
  gap: var(--hl-sidebar-item-gap);
  overflow-y: auto;
  overflow-x: hidden;
  /* 细滚动条 */
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
}

/* ── 底部版本 ─────────────────────────────────────────────────────── */
.nav-sidebar__footer {
  padding: 12px 18px;
  border-top: 1px solid var(--hl-sidebar-border);
  flex-shrink: 0;
}

.nav-sidebar__version {
  font-size: 10px;
  color: var(--hl-nav-icon);
  letter-spacing: 0.2px;
}
</style>
