import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

const DEV_BACKEND_HOST = '127.0.0.1'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },
  server: {
    port: 5173,
    proxy: {
      // WebSocket 遥测端点（必须在 /api 通用规则之前，更具体的路径优先匹配）
      '/api/ws': {
        // Node 解析 localhost 时会优先尝试 ::1；后端当前开发服务并未稳定监听 IPv6。
        target: `ws://${DEV_BACKEND_HOST}:8000`,
        ws: true,
        changeOrigin: true,
      },
      // 开发时将所有 /api/* 请求代理到 Flask 主入口（backend/app.py）
      // 如需独立调试 Geo 模块，可临时改为 5000
      '/api': {
        target: `http://${DEV_BACKEND_HOST}:8000`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
