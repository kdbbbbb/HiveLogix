# 前后端连接机制与新功能开发指南

> 本文档说明 HiveLogix 前后端的连接方式、两种 Blueprint 模式的区别，以及新增功能的完整操作步骤。

---

## 一、整体连接架构

```
浏览器 (localhost:5173)
    │
    │  fetch('/api/geo/status')   ← 前端发出的请求（相对路径，不含端口）
    ▼
Vite 开发服务器 (localhost:5173)
    │
    │  代理规则：/api/* → localhost:8000  (vite.config.ts)
    ▼
Flask 主入口 backend/app.py (localhost:8000)
    │
    │  Blueprint 路由匹配：/api/geo/* → geo_bp
    │                       /api/fleet/* → fleet_bp（未来）
    │                       /api/dispatch/* → dispatch_bp（未来）
    ▼
对应 Blueprint 文件（geo_blueprint.py / api/routes/fleet.py / ...）
    │
    │  执行业务逻辑，调用领域服务
    ▼
返回 JSON 响应，原路返回给浏览器
```

三个关键节点：

| 层 | 文件 | 职责 |
|---|---|---|
| **代理层** | `frontend/vite.config.ts` | 开发时将 `/api/*` 转发到 `:8000`，消除跨域 |
| **主入口** | `backend/app.py` | 注册所有 Blueprint、配置 CORS、提供 `/api/health` |
| **业务路由** | 各 Blueprint 文件 | 具体端点实现（见第三节） |

> **前端只有一个。** Vue 应用始终连接同一套 API，后端提供两种启动模式（`:8000` 集成 / `:5000` 单模块调试），但 API 路径和行为完全一致。

---

## 二、以 GeoTool "获取状态" 为例——全链路追踪

### 第 1 步：前端发出请求

`frontend/src/views/GeoTool/index.vue`，`pollStatus` 函数：

```typescript
const res = await fetch('/api/geo/status')
const s: ServerState = await res.json()
```

前端只写相对路径 `/api/geo/status`，不关心端口。

---

### 第 2 步：Vite 代理拦截并转发

`frontend/vite.config.ts`：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',   // 转发目标
      changeOrigin: true,
    },
  },
},
```

Vite 将请求改写为 `http://localhost:8000/api/geo/status` 发出，浏览器看不到真实端口，也不会触发跨域错误。

> **注意**：代理只在 `npm run dev`（开发模式）下生效。生产构建后需由 Nginx 反向代理接管（见第五节）。

---

### 第 3 步：Flask 主入口收到请求

`backend/app.py`：

```python
sys.path.insert(0, GEO_DIR)          # 让 geo/ 目录下的模块可以平坦式 import
from geo_blueprint import geo_bp
app.register_blueprint(geo_bp, url_prefix="/api/geo")
```

Flask 匹配到 `/api/geo/status`，派发给 `geo_bp` 处理。

---

### 第 4 步：Blueprint 处理并返回

`backend/environment/geo/geo_blueprint.py`：

```python
@geo_bp.route("/status")
def api_status():
    return jsonify(get_state())
```

`get_state()` 返回 Shapefile 的加载进度、建筑数量等信息，序列化为 JSON 后原路返回。

---

### 第 5 步：前端消费响应

```typescript
serverState.value  = s
loadProgress.value = s.progress
if (s.loaded) {
  appReady.value = true
}
```

前端根据 `loaded` 字段决定是否隐藏加载遮罩，整个闭环完成。

---

## 三、两种 Blueprint 模式

本项目的 Blueprint 分两类，放置位置不同，适用场景也不同。

### 类型 A：领域自包含型（以 geo 为例）

```
backend/environment/geo/
├── geo_blueprint.py      ← Blueprint 与领域代码同目录
├── data_loader.py
├── building_service.py
├── osm_service.py
└── exporters/
```

**特征**：
- Blueprint 深度依赖模块内部的多个服务文件
- 模块具备独立启动调试能力（`geo/app.py` → `:5000`）
- `sys.path` 注入使内部可以使用平坦式 import，Blueprint 必须待在同目录才能工作

**适用于**：有独立生命周期、可单独部署的子系统（地理服务、仿真引擎等）

---

### 类型 B：薄适配型（新功能的标准做法）

```
backend/api/routes/
├── __init__.py
├── fleet.py              ← 只做 HTTP↔服务调用的翻译，几乎无业务逻辑
├── orders.py
└── dispatch.py
```

**特征**：
- 文件极薄，只负责接收 HTTP 请求 → 调用 `core/` 或 `solver/` 中的领域服务 → 返回 JSON
- 不包含业务逻辑，不需要独立运行
- import 路径相对于 `backend/`，无需 `sys.path` 注入

**适用于**：调用共享领域服务的常规 REST 接口

---

### 决策规则

```
新功能的 Blueprint 放哪里？

├── 模块有大量内部依赖 AND 需要独立调试
│   └── 放在领域模块目录内（类型 A）
│       示例：backend/environment/新模块/新模块_blueprint.py
│
└── 逻辑在 core/ 或 solver/ 中，Blueprint 只是薄翻译层
    └── 放在 backend/api/routes/（类型 B）
        示例：backend/api/routes/fleet.py
```

---

## 四、如何新增一个功能（完整步骤）

以 **"获取所有卡车列表"** 为例，前端 `/fleet` 页面，后端返回卡车 JSON 数组。
此功能逻辑将来在 `core/` 中实现，Blueprint 作薄适配，走**类型 B**。

---

### Step 1：在后端创建 Blueprint 文件

新建 `backend/api/routes/fleet.py`：

```python
from flask import Blueprint, jsonify

fleet_bp = Blueprint("fleet", __name__)

# 模拟数据——后期替换为调用 core/entities/ 中的领域服务
_trucks = [
    {"id": "T001", "status": "idle",   "location": [121.47, 31.23]},
    {"id": "T002", "status": "active", "location": [121.50, 31.20]},
]

@fleet_bp.route("/trucks", methods=["GET"])
def get_trucks():
    return jsonify({"trucks": _trucks})
```

---

### Step 2：在主入口注册 Blueprint

`backend/app.py`，在现有注册区域添加：

```python
# 顶部导入区
from api.routes.fleet import fleet_bp

# Blueprint 注册区
app.register_blueprint(fleet_bp, url_prefix="/api/fleet")
```

注册后，端点完整路径为：`GET /api/fleet/trucks`

---

### Step 3：定义前端 TypeScript 类型

`frontend/src/types/index.ts` 已有 `Truck` 接口定义，确认字段与后端返回一致即可。如需新类型，在此文件追加：

```typescript
export interface Truck {
  id:       string
  status:   'idle' | 'active' | 'offline'
  location: [number, number]   // [lng, lat]
}
```

---

### Step 4：在前端页面调用 API

`frontend/src/views/FleetManagement/index.vue`，使用封装好的 `http` 工具（推荐）或裸 `fetch`（流式/进度场景）：

```typescript
import { ref, onMounted } from 'vue'
import { http } from '@/services/http'
import type { Truck } from '@/types'

const trucks = ref<Truck[]>([])

onMounted(async () => {
  const data = await http.get<{ trucks: Truck[] }>('/api/fleet/trucks')
  trucks.value = data.trucks
})
```

`http.get` / `http.post` 封装在 `frontend/src/services/http.ts`，自动携带 `Content-Type` 并在非 2xx 时抛出错误。

---

### Step 5：在模板中渲染数据

```html
<template>
  <ul>
    <li v-for="truck in trucks" :key="truck.id">
      {{ truck.id }} — {{ truck.status }}
    </li>
  </ul>
</template>
```

---

### 验证全链路

```bash
# 终端 1：启动后端
cd backend && python app.py

# 终端 2：启动前端
cd frontend && npm run dev

# 终端 3：直接测试后端端点（跳过前端）
curl http://localhost:8000/api/fleet/trucks

# 浏览器：访问前端页面
# http://localhost:5173/fleet
```

---

## 五、通信方式选择

### REST（HTTP）— 大多数场景

| 场景 | 推荐写法 |
|---|---|
| 普通 GET 查询 | `http.get<T>('/api/...')` |
| 带请求体的 POST | `http.post<T>('/api/...', body)` |
| 下载文件 / 流式响应 | 裸 `fetch`（GeoTool 导出即用此方式） |

### WebSocket — 仿真实时推送

`frontend/src/services/websocket.ts` 已封装 `WsClient`，支持按消息类型订阅和自动重连（3 秒）：

```typescript
import { WsClient } from '@/services/websocket'

const ws = new WsClient('ws://localhost:8000/ws/simulation')
ws.connect()

// 订阅特定消息类型
ws.on<{ trucks: Truck[] }>('entity_snapshot', (payload) => {
  trucks.value = payload.trucks
})

// 关闭连接
onUnmounted(() => ws.disconnect())
```

后端对应需要使用 `flask-sock` 或 `flask-socketio` 提供 WebSocket 端点（待实现）。

---

## 六、geo 模块的独立调试模式

geo 是目前项目中唯一具备独立运行能力的模块。当只需要调试 geo 功能时，可以不启动主后端：

```bash
cd backend/environment/geo
python app.py        # → http://localhost:5000
                     #   API 路径：/api/status、/api/query、/api/roads、/api/export
```

此时若想用前端连接，临时修改 `frontend/vite.config.ts`：

```typescript
target: 'http://localhost:5000',   // 临时改为 5000，调试完改回 8000
```

> **注意**：独立模式的路径前缀是 `/api/...`，集成模式是 `/api/geo/...`。临时切换时前端的 fetch 路径也需同步调整，调试完毕后记得还原。

---

## 七、生产部署

开发模式的代理由 Vite 完成；生产部署（`npm run build`）后静态文件由 Nginx 托管，API 由 Nginx 反向代理转发：

```nginx
# 前端静态文件（frontend/dist）
location / {
    root /var/www/hivelogix/dist;
    try_files $uri $uri/ /index.html;
}

# API 请求转发到 Flask
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

同源部署后 Flask 的 CORS 配置可以关闭。

---

## 八、新增功能检查清单

```
后端
  ☐ 确定 Blueprint 类型：领域自包含（类型 A）or 薄适配（类型 B）
  ☐ 类型 A：在领域模块目录创建 <模块名>_blueprint.py，在 backend/app.py 注入 sys.path 并注册
  ☐ 类型 B：在 backend/api/routes/ 创建 <模块名>.py，在 backend/app.py 直接 import 并注册
  ☐ 注册路径：url_prefix="/api/<模块名>"
  ☐ 用 curl 直接验证端点返回正确数据

前端
  ☐ 在 src/types/index.ts 定义响应数据类型
  ☐ 使用 http.get / http.post 调用（流式场景用裸 fetch）
  ☐ 请求路径以 /api/<模块名>/ 开头（Vite 自动代理，无需写端口）
  ☐ 在 src/router/index.ts 确认页面路由已注册
  ☐ 浏览器 Network 面板确认请求状态 200
```
