# HiveLogix — 卡车-无人机-充换电站 多模态协同调度系统

基于离散事件仿真的城市物流"最后一公里"协同配送平台，支持卡车直递、卡-空协同、仓-空直递、动态补货、多跳中继五种履约模式，实现 100% 动态订单履约。

---

## 项目架构

```
HiveLogix/
├── backend/                                    # 后端：算法引擎与仿真系统
│   ├── app.py                                  # 主入口（端口 8000，注册所有 Blueprint）
│   │
│   ├── api/                                    # HTTP / WebSocket 接口层
│   │   ├── __init__.py
│   │   ├── routes/                             # 新功能 Blueprint 路由文件存放处
│   │   │   └── __init__.py
│   │   └── websockets/                         # WebSocket 消息/会话处理
│   │
│   ├── core/                                   # 核心领域模型
│   │   ├── __init__.py
│   │   ├── entities/                           # 全部实体类（已实现）
│   │   │   ├── __init__.py                     # 统一公共 API 出口（from core.entities import *）
│   │   │   ├── primitives.py                   # Position3D / 全状态机枚举 / RouteWaypoint（零依赖）
│   │   │   ├── charging_host.py                # ChargingHost 抽象基类（M/D/K 换电队列模型）
│   │   │   ├── order.py                        # Order（状态机校验 / 软时间窗惩罚 / RL 向量接口）
│   │   │   ├── drone.py                        # Drone / LightDrone / HeavyDrone（多旋翼功耗模型）
│   │   │   ├── swap_station.py                 # SwapStation（固定换电站，复用基类队列逻辑）
│   │   │   ├── truck.py                        # Truck（路网插值 / 起降平台占位 / 轨迹预测）
│   │   │   └── depot.py                        # Depot（换电回调钩子 / 订单池 / 装车管理）
│   │   └── models/                             # 订单模型扩展（静态批量订单，待实现）
│   │
│   ├── environment/                            # 仿真环境与地理服务
│   │   ├── geo/
│   │   │   ├── app.py                          # 独立运行入口（端口 5000，挂载 geo_bp）
│   │   │   ├── geo_blueprint.py                # 所有 Geo API 路由（Blueprint 实例 geo_bp）
│   │   │   ├── data_loader.py                  # Shapefile 异步加载 + 状态管理
│   │   │   ├── building_service.py             # 空间裁剪 + 禁飞区阈值分类
│   │   │   ├── osm_service.py                  # Overpass API + OSM→GeoJSON 转换
│   │   │   ├── preset_scenes.py                # 预设测试场景生成器
│   │   │   ├── requirements.txt                # Python 依赖（flask, geopandas 等）
│   │   │   ├── exporters/                      # Geo / SUMO / CSV 导出工具
│   │   │   ├── static/                         # Flask 静态资源（旧版模板保留）
│   │   │   └── shanghai_map/                   # 上海建筑高度 Shapefile 数据
│   │   ├── path_planning/                      # 路径规划服务
│   │   │   ├── grid_astar.py                    # 网格 A* 路径搜索实现
│   │   │   ├── planner.py                       # 路径规划器（调度调用接口）
│   │   │   └── __pycache__/                     # Python 缓存
│   │   ├── scene/                              # 场景构建与动态订单生成
│   │   └── state/                              # 全局状态管理 + 时间轴控制
│   │
│   ├── solver/                                 # 调度求解层
│   │   ├── decision_engine.py                  # 编排层：查货→查距/电→选模式→派单
│   │   ├── market_based_solver.py              # 基于市场的调度求解
│   │   ├── greedy_baseline.py                  # 贪心基线算法
│   │   ├── greedy_mmce.py                      # MMCE 贪心调度
│   │   ├── greedy_mmce_bi.py                   # 双向 MMCE 贪心调度
│   │   ├── ga_mmce/                            # 遗传算法 / 进化调度模块
│   │   ├── factory.py                          # 求解器工厂
│   │   ├── interfaces.py                       # 求解器接口定义
│   │   └── kpi_metrics.py                      # 调度 KPI 统计
│   │
│   ├── storage/                                # 数据持久化（调度日志/仿真快照/回放）
│   ├── config/                                 # 物理参数 + 算法超参 YAML
│   │   ├── __init__.py
│   │   ├── loader.py                           # YAML 加载器（frozen dataclass + lru_cache）
│   │   ├── drone_params.yaml                   # LightDrone / HeavyDrone 气动参数（k1/k2/电池容量等）
│   │   └──  policy_best.pt                     # 训练模型权重
│   │  
│   └── utils/                                  # 通用工具
│       ├── __init__.py
│       └── coord_utils.py                      # UTM ↔ WGS84 坐标转换包装层（隔离 pyproj 依赖）
│   ├── scripts/                                # 辅助脚本
│   ├── test_data/                              # 测试场景数据
│   ├── training/                               # 训练与模型数据
│   ├── download_test_map.py                    # 地图下载测试脚本
│   ├── generate_test_scene.py                  # 生成测试场景脚本
│   └── test_simulation_policy_routes.py        # 策略路由测试
│
├── frontend/                                   # 前端：调度监控大屏（Vue 3 + TypeScript + Vite）
│   ├── index.html
│   ├── vite.config.ts                          # 开发代理：/api/* → :8000
│   ├── tsconfig.json
│   ├── package.json
│   └── src/
│       ├── main.ts                             # Vue 应用挂载（导入全局设计令牌）
│       ├── App.vue                             # 根组件（<router-view />）
│       │
│       ├── styles/
│       │   └── tokens.css                      # 全局 CSS 设计令牌（颜色/间距/阴影）
│       │
│       ├── layouts/
│       │   └── AppLayout.vue                   # 主布局骨架（左侧导航栏 + 内容区）
│       │
│       ├── components/                         # 全局可复用组件
│       │   ├── NavSidebar/
│       │   │   ├── index.vue                   # 深色侧边导航栏（Logo + 菜单 + 版本号）
│       │   │   └── NavItem.vue                 # 单个导航项（激活态/悬停态/Badge）
│       │   └── PageShell/
│       │       └── index.vue                   # 通用页面骨架（统一页头 + 内容区）
│       │
│       ├── router/
│       │   └── index.ts                        # 路由：六大模块，均挂载于 AppLayout
│       │
│       ├── stores/                             # Pinia 全局状态
│       │   ├── system.ts                       # 仿真运行状态（时刻/加速比/开关）
│       │   ├── entity.ts                       # 实体快照（仓库/卡车/站点/无人机）
│       │   └── order.ts                        # 订单生命周期（待派/在途/完成）
│       │
│       ├── services/
│       │   ├── http.ts                         # 统一 fetch 封装（baseURL + 错误处理）
│       │   └── websocket.ts                    # WebSocket 长连接 + 自动重连（预留）
│       │
│       ├── types/
│       │   └── index.ts                        # 全系统 TS 类型（与后端模型对齐）
│       │
│       └── views/
│           ├── Dashboard/
│           │   └── index.vue                   # 调度性能大屏（KPI / 模式分布 / 成本趋势）
│           │
│           ├── DispatchCenter/
│           │   ├── index.vue                   # 实时指挥中心（三栏：状态面板/地图/实体抽屉）
│           │   └── components/
│           │       └── SectionCard.vue         # 可复用侧边卡片组件
│           │
│           ├── OrderTask/
│           │   └── index.vue                   # 订单与任务管理（订单表 + 派送拆解视图）
│           │
│           ├── FleetManagement/
│           │   └── index.vue                   # 载具管理调度（卡车序列 + 无人机序列）
│           │
│           ├── Infrastructure/
│           │   └── index.vue                   # 基础设施配置（充换电站 + 仓库网点）
│           │
│           ├── SimulationBox/                  # 仿真与动态场景（Tab 容器）
│           │   ├── index.vue                   # Tab 1：嵌入 GeoTool（LIVE）
│           │   │                               # Tab 2：动态订单生成器）
│           │   │                               # Tab 3：调度求解器参数
│           │   └── components/
│           │       └── ConfigItem.vue          # 配置项说明卡片
│           │
│           └── GeoTool/                        # UAV 禁飞区地图工具（嵌入 SimulationBox）
│               ├── index.vue                   # 主视图：状态编排 + API 调用
│               ├── geo_map.css                 # 页面样式
│               └── components/
│                   ├── LoadingMask.vue          # 数据加载进度遮罩
│                   ├── TopBar.vue               # 顶部状态栏
│                   ├── SideBar.vue              # 控制面板（选区/阈值/统计/导出）
│                   └── MapView.vue              # Leaflet 地图（建筑/道路/选区图层）
│
├── docs/                                       # 项目设计文档与需求说明
│   ├── frontend-backend-guide.md
│   ├── frontend-entity-crud.md
│   ├── frontend-redesign-plan.md
│   ├── frontend-ui-planning.md
│   ├── geo-tool.md
│   ├── phase2-entity-order-enhancement.md
│   ├── phase3-backend-manager-api-design.md
│   ├── decision_engine_responsibility_and_sequences.md
│   ├── dispatch-map-integration.md
│   ├── entities-design.md
│   ├── data-sync-reality-check.md
│   ├── ppo算法方案/
│   └── 遗传算法设计方案/
├── reference-orderGenerate/                     # 参考方案与额外资料
└── README.md
```

---

## 快速启动

### 前提：安装依赖

```bash
# Python 环境（conda）
conda create -n hivelogix python=3.11
conda activate hivelogix
pip install -r backend/environment/geo/requirements.txt
pip install flask-cors

# 前端
cd frontend
npm install
```

### 启动主后端（集成模式，推荐）

```bash
conda activate hivelogix
cd backend
python app.py          # → http://localhost:8000
                       #   Geo API: /api/geo/status
                       #   健康检查: /api/health
```

### 启动前端

```bash
cd frontend
npm run dev            # → http://localhost:5173
```

前端开发服务器自动将 `/api/*` 代理到 `:8000`。

### 独立调试 Geo 模块（可选）

```bash
conda activate hivelogix
cd backend/environment/geo
python app.py          # → http://localhost:5000
                       #   路径保持旧版 /api/* 不变，兼容直连调试
```

---

## 前端导航架构

| 路径 | 页面 | 状态 |
|------|------|------|
| `/dashboard`  | 📊 调度性能大屏  | ✅ 骨架已建，图表待接入 |
| `/dispatch`   | 🗺️ 实时指挥中心  | ✅ 三栏布局已建，地图待接入 |
| `/orders`     | 📦 订单与任务管理 | ✅ 表格骨架已建，数据待接入 |
| `/fleet`      | 🛸 载具管理调度   | ✅ 双列表骨架已建，数据待接入 |
| `/infra`      | 🏗️ 基础设施配置  | ✅ 骨架已建，CRUD 待实现 |
| `/simulation` | ⚙️ 仿真与配置    | ✅ GeoTool 已嵌入，其余 Tab 待开发 |

---

## 后端 API 路由

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET  /api/health`         | GET  | 服务健康探针 |
| `GET  /api/geo/status`     | GET  | Shapefile 加载状态与进度 |
| `POST /api/geo/query`      | POST | 查询选区建筑数据，返回 GeoJSON |
| `POST /api/geo/roads`      | POST | 下载选区 OSM 道路数据 |
| `POST /api/geo/export`     | POST | 导出 SUMO / GeoJSON / CSV 文件 |

> 新增功能模块：在 `backend/api/routes/` 下创建 Blueprint 文件，然后在 `backend/app.py` 中注册。

---

## 核心实体

| 实体 | 说明 |
|------|------|
| **仓库 (Depot)**         | 固定位置，库存无限，无人机起降与维护中心 |
| **卡车 (Truck)**         | 路网移动，移动基站+微仓，搭载 K 架无人机，容量 $C_{truck}$ |
| **充换电站 (Station)**   | 离散分布固定节点，换电 $\tau_{swap}$ / 充电 $T_{charge} = \alpha \cdot \Delta E$ |
| **无人机 (Drone)**       | 最大载重 $C_{drone}$，电池容量 $E_{max}$，能耗随距离/载重动态变化 |

## 五种履约模式

| 模式 | 名称 | 路径 |
|------|------|------|
| A | 卡车直递         | 卡车 → 客户 |
| B | 卡-空短途协同    | 卡车 → 无人机起飞 → 客户 → 无人机回收至卡车下一节点 |
| C | 仓-空直递        | 仓库 → 无人机 → 客户 → 仓库 |
| D | 空-地动态补货    | 仓库无人机携货飞向卡车 → 卡车后续派送 |
| E | 多跳中继         | 卡车/仓库 → 客户 → 充换电站补能 → 追赶卡车/返回仓库 |

## 核心约束

- **库存不透支**：卡车/无人机容量实时追踪
- **时空同步**：无人机与卡车汇合必须满足位置+时间一致
- **能量守恒**：任意飞行段终点 $E_{arrive} \ge 0$，节点补能状态严格更新
- **时间惩罚**：充换电站滞留时间计入全局时间轴，影响后续汇合可行性
- **100% 履约**：所有动态订单不可拒绝，软时间窗超时产生惩罚成本
