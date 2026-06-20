# HiveLogix

HiveLogix 是一个面向城市物流场景的卡车、无人机、仓库与充换电站协同调度仿真项目。项目采用前后端分离架构：

- 后端：Flask，负责地图/场景接口、仿真运行时、调度求解、PPO 策略运行与训练接口。
- 前端：Vue 3 + TypeScript + Vite，负责地图选区、实体配置、订单生成、仿真控制和运行状态展示。
- 数据与配置：预设场景、订单、实体和算法参数主要放在 `backend/test_data/` 与 `backend/config/`。

## 目录结构

```text
.
├── backend/                 # Flask 后端、仿真、求解器、训练代码
│   ├── app.py               # 主后端入口，默认端口 8000
│   ├── api/                 # REST / WebSocket 接口
│   ├── config/              # 无人机、能耗、PPO/策略等配置
│   ├── environment/         # 地图、场景、状态与路径规划模块
│   ├── solver/              # 调度求解器
│   ├── test_data/           # 预设场景与订单数据
│   └── training/            # PPO 训练与在线策略运行
├── frontend/                # Vue 前端
├── docs/                    # 设计文档与阶段方案
├── requirements.txt         # 后端 Python 依赖
└── README.md
```

## 环境要求

- Python 3.11
- Node.js 18 或更高版本
- npm

建议使用虚拟环境或 conda 环境运行后端。

## 安装依赖

### 后端

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 conda：

```bash
conda create -n hivelogix python=3.11
conda activate hivelogix
pip install -r requirements.txt
```

### 前端

```bash
cd frontend
npm install
```

## 启动项目

### 1. 启动后端

在项目根目录执行：

```bash
source .venv/bin/activate
cd backend
python app.py
```

默认地址：

- 后端服务：`http://localhost:8000`
- 健康检查：`http://localhost:8000/api/health`
- Geo 状态：`http://localhost:8000/api/geo/status`

后端启动时会注册以下主要接口前缀：

- `/api/geo/*`：地图、建筑、道路、导出接口
- `/api/scene/*`：场景打包与预设场景读取
- `/api/sim/*`：仿真初始化、控制、调度、策略与训练接口
- `/api/ws/telemetry`：仿真遥测 WebSocket

### 2. 启动前端

另开一个终端：

```bash
cd frontend
npm run dev
```

默认地址：

```text
http://localhost:5173
```

开发环境下，Vite 会把前端的 `/api/*` 请求代理到 `http://127.0.0.1:8000`，WebSocket 也会代理到同一个后端。

## 常用命令

前端：

```bash
cd frontend
npm run dev          # 本地开发
npm run build        # 类型检查并构建
npm run type-check   # 仅运行 TypeScript 检查
```

后端：

```bash
cd backend
python app.py
```

如果只需要单独调试 Geo 模块，也可以运行：

```bash
cd backend/environment/geo
python app.py
```

Geo 独立入口默认使用 `5000` 端口；正常联调推荐使用 `backend/app.py` 的集成入口。

## 配置修改

### 前端接口地址

前端默认使用相对路径请求 API，并通过 Vite 代理转发到后端。代理配置在：

```text
frontend/vite.config.ts
```

默认后端地址：

```ts
const DEV_BACKEND_HOST = '127.0.0.1'
```

如果需要直连其他后端，可以在 `frontend/.env.local` 中设置：

```bash
VITE_API_BASE=http://127.0.0.1:8000
VITE_WS_BASE=ws://127.0.0.1:8000
```

### 后端端口

主后端端口在 `backend/app.py` 的入口处设置：

```python
port = 8000
```

修改端口后，需要同步调整 `frontend/vite.config.ts` 或前端环境变量。

### 无人机与能耗参数

无人机物理参数、载重、电池、安全余量和求解器能耗参数位于：

```text
backend/config/drone_params.yaml
```

这些参数由 `backend/config/loader.py` 读取并缓存，主要影响无人机实体、能耗估算和求解器评分。

### PPO / 策略配置

默认 PPO 与在线策略配置位于：

```text
backend/config/rh_alns_cmrappo.yaml
```

前端中的策略路径默认值为：

```text
config/policy_best.pt
config/rh_alns_cmrappo.yaml
```

相关配置通常包括场景数据路径、订单源、候选集、动作空间、奖励和训练超参数。

### 场景与订单数据

预设场景数据位于：

```text
backend/test_data/default_scene/
backend/test_data/scene-2/
```

常见文件包括：

- `scene_config.json`：场景边界和基础信息
- `entities.json`：仓库、充换电站、卡车、无人机等实体
- `orders.json`：静态订单与动态订单
- `osm_network.geojson` / `osm_network.xml`：道路网络
- `buildings.geojson`、`no_fly_zones.geojson`：建筑与禁飞区数据

如果修改了预设场景或订单，需要确认对应配置文件中的 `scene_bundle_dir`、`entities_file`、`orders_file` 等路径仍然一致。

## 运行流程建议

1. 启动后端 `backend/app.py`。
2. 启动前端 `npm run dev`。
3. 在前端选择或加载场景。
4. 配置实体与订单。
5. 初始化仿真。
6. 启动仿真、执行调度，或激活 PPO 策略。

更多实现细节和阶段性方案可以查看 `docs/` 目录。
