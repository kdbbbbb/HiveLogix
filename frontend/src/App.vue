<template>
  <router-view />
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useEntityStore } from '@/stores/entity'
import { useSystemStore } from '@/stores/system'
import { useSceneStore }  from '@/stores/scene'

const entityStore = useEntityStore()
const systemStore = useSystemStore()
const sceneStore  = useSceneStore()

onMounted(() => {
  entityStore.loadConfig()
  systemStore.initWs()       // 建立 WebSocket 遥测连接
  systemStore.probeBackend() // 探测后端是否已在运行（恢复快照）

  // 优先加载预设测试场景（确保总是有有效的场景）
  // 如果加载成功，后续恢复逻辑会通过 DispatchCenter 路由检测处理
  sceneStore.loadPresetScene('default_test_4x4km').catch(() => {
    // 预设场景加载失败时，尝试从上次会话恢复
    const cachedSceneId = localStorage.getItem('hl-scene-id')
    if (cachedSceneId) {
      sceneStore.restoreScene(cachedSceneId)
    }
  })
})
</script>
