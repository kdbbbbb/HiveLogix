# HiveLogix 仿真场景导出至实时指挥中心 — 技术方案设计文档

> 版本：v1.1（含评审修订）
> 日期：2026-03-28
> 范围：SimulationBox / GeoTool → DispatchCenter 地图联通，以及后续动态实体演示的底层基础建设

---

## 一、现状梳理与问题定义

### 1.1 已有能力（不重复造轮子）

| 已有能力 | 代码位置 | 说明 |
|---|---|---|
| 建筑禁飞/可飞区分析 | `backend/environment/geo/building_service.py` | 输出 GeoJSON，坐标系 WGS84 (EPSG:4326) |
| SUMO 路网生成 | `exporters/sumo_net_osm.py` | 输出 `net.xml`，内含 `netOffset` 和 `projParameter=+zone=51` |
| 禁飞区叠加层导出 | `exporters/sumo_poly.py` | 输出 `no_fly_zones.add.xml` |
| 建筑查询接口 | `geo_blueprint.py` → `POST /api/geo/query` | 已有 30,000 条上限保护，运行稳定 |
| 道路查询接口 | `geo_blueprint.py` → `POST /api/geo/roads` | 返回 WGS84 GeoJSON LineString，复用 Overpass OSM 数据 |
| 选区边界 | `GeoTool/index.vue` `selBounds: {minx,miny,maxx,maxy}` | WGS84 经纬度 |
| 跨视图状态 | `stores/entity.ts` `stores/system.ts` | Pinia，已投入使用 |
| WebSocket 封装 | `services/websocket.ts` `WsClient` | 消息格式 `{type, payload}` |

### 1.2 核心问题

**GeoTool（仿真配置子页）与 DispatchCenter（实时指挥中心）之间缺少数据通道**：
- 用户在 GeoTool 完成建筑分析后，分析结果无法传递到指挥中心地图；
- 两套数据（建筑 GeoJSON / SUMO 路网）坐标系不同，需要统一；
- 后续仿真动态实体（无人机、卡车）需要一个已对齐的统一坐标地图作为演示底座。

### 1.3 坐标系不统一的根因

```
建筑 GeoJSON          → WGS84 (EPSG:4326，经纬度)
SUMO net.xml          → UTM Zone 51N (EPSG:32651，米制)，原点 = 选区左下角
Leaflet 地图          → WGS84 (EPSG:4326)
后续物理引擎运算       → 必须是 UTM（米制，欧氏距离、速度计算）
```

---

## 二、架构决策总则

以下四条铁律指导所有详细设计：

| 编号 | 铁律 | 原则 |
|---|---|---|
| A1 | **后端计算用 UTM，对外接口用 WGS84** | 物理引擎（速度、能耗、避障）全部在米制坐标系内运算；WebSocket/REST 对外推送前统一转换为 WGS84 |
| A2 | **`/api/scene/prepare` 只返回轻量级 meta** | 重载荷 GeoJSON 通过现有 `/api/geo/query` 和 `/api/geo/roads` 端点独立异步加载，不在 prepare 响应中打包 |
| A3 | **WGS84 为系统唯一对外坐标协议** | `types/index.ts` 中所有实体坐标字段统一为 `{ lat: number; lon: number }`，禁止在 store 或 API 层暴露 UTM 或像素坐标 |
| A4 | **`UnifiedMapView` 必须使用 Canvas 渲染模式** | Leaflet 默认 SVG 在 3000+ 多边形 + 高频动态刷新下性能不可接受，强制 `L.canvas()` |

---

## 三、后端新增模块：`scene` 服务

### 3.1 目录结构

`geo` 模块只负责地理数据分析与导出，单一职责不变。`scene` 作为独立模块承接"场景打包"职责。

```
backend/
└── environment/
    └── scene/                         ← 新增
        ├── __init__.py
        ├── coord_transformer.py       ← 纯工具函数：WGS84 ↔ UTM 互转（无副作用）
        ├── scene_service.py           ← 核心业务：场景上下文打包、幂等缓存
        └── scene_blueprint.py         ← Flask Blueprint：/api/scene/*
```

---

### 3.2 `coord_transformer.py`

纯函数，无状态，供 `scene_service` 和未来 physics engine 共用。

```python
from pyproj import Transformer

_tr_to_utm  = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
_tr_to_wgs  = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

def wgs84_to_utm(lon: float, lat: float) -> tuple[float, float]:
    """返回 (x_m, y_m) UTM Zone 51N"""
    return _tr_to_utm.transform(lon, lat)

def utm_to_wgs84(x_m: float, y_m: float) -> tuple[float, float]:
    """返回 (lon, lat) WGS84"""
    return _tr_to_wgs.transform(x_m, y_m)
```

> **注意**：`Transformer` 对象在模块级创建一次，避免每次调用重复初始化的开销。高频调用（仿真帧推送）时直接 `import coord_transformer` 即可，单次转换为微秒级运算，无性能问题。

---

### 3.3 `scene_service.py`

核心职责：提取路网轓点表（WGS84）、打包 SceneContext、幂等缓存。

```python
import hashlib, json, uuid
from typing import Optional
from osm_service   import download_osm, osm_to_geojson
from exporters.sumo_net_osm import osm_to_sumo_net
from exporters.sumo_net_grid import export_grid_net
from coord_transformer import utm_to_wgs84

# 内存缓存：params_hash → SceneContext
_cache: dict[str, dict] = {}

def _params_hash(sel_bounds: dict, threshold: float) -> str:
    key = json.dumps({**sel_bounds, "thr": threshold}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]

def prepare_scene(sel_bounds: dict, threshold: float, height_column: Optional[str]) -> dict:
    """
    打包场景上下文（轻量级 meta）。
    1. 尝试下载 OSM 路网 → 失败时降级为网格路网
    2. 解析路网节点，保留 WGS84 经纬度
    3. 记录 netOffset 和 utm_zone 供后续 physics engine 使用
    4. 相同参数直接返回缓存（幂等）
    """
    h = _params_hash(sel_bounds, threshold)
    if h in _cache:
        return _cache[h]

    minx, miny = sel_bounds["minx"], sel_bounds["miny"]
    maxx, maxy = sel_bounds["maxx"], sel_bounds["maxy"]

    # ── 路网来源 ──────────────────────────────────────────────────
    road_source = "osm"
    osm_xml = None
    try:
        osm_xml = download_osm(minx, miny, maxx, maxy)
    except Exception:
        road_source = "grid"

    # ── 从 osm_xml 提取节点 WGS84 表 ──────────────────────────────
    # osm_to_sumo_net 内部已有 osm_nodes (nid→lon,lat)，复用其解析逻辑
    road_nodes_wgs84 = []
    road_edges_wgs84 = []
    node_count = 0
    edge_count = 0

    if road_source == "osm" and osm_xml:
        # 解析 OSM 节点（直接从 XML 读，不经 UTM，精度最高）
        import xml.etree.ElementTree as ET
        root = ET.fromstring(osm_xml)
        osm_nodes = {}
        for nd in root.iter("node"):
            nid = nd.get("id")
            osm_nodes[nid] = (float(nd.get("lon", 0)), float(nd.get("lat", 0)))
        from osm_service import HW_SPEED
        for way in root.iter("way"):
            tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
            if tags.get("highway", "") not in HW_SPEED:
                continue
            coords = [[osm_nodes[r.get("ref")][0], osm_nodes[r.get("ref")][1]]
                      for r in way.iter("nd") if r.get("ref") in osm_nodes]
            if len(coords) >= 2:
                road_edges_wgs84.append({"shape": coords})
                edge_count += 1
        road_nodes_wgs84 = [{"id": k, "lng": v[0], "lat": v[1]}
                             for k, v in osm_nodes.items()]
        node_count = len(road_nodes_wgs84)

    # ── 计算 netOffset（与 sumo_net_osm.py 一致） ─────────────────
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    ox, oy = tr.transform(minx, miny)

    scene_id = str(uuid.uuid4())
    ctx = {
        "scene_id": scene_id,
        "sel_bounds": sel_bounds,
        "threshold": threshold,
        "height_column": height_column,
        "meta": {
            "road_source":  road_source,
            "road_nodes":   node_count,
            "road_edges":   edge_count,
            "created_at":   __import__("datetime").datetime.utcnow().isoformat() + "Z",
            # ── physics engine 必须字段 ────────────────────────────
            "utm_zone":     51,
            "utm_band":     "N",
            "net_offset":   {"ox": round(ox, 2), "oy": round(oy, 2)},
            # net.xml 中节点坐标 = (UTM_x - ox, UTM_y - oy)，还原公式：
            # utm_x = sumo_x + ox;  utm_y = sumo_y + oy
        },
        "road_network": {
            "nodes": road_nodes_wgs84,
            "edges": road_edges_wgs84,
            "bounds": {
                "min_lng": minx, "min_lat": miny,
                "max_lng": maxx, "max_lat": maxy,
            },
        },
    }
    _cache[h] = ctx
    return ctx
```

**关键设计说明**：

- 路网节点直接从 OSM XML 读取 WGS84 原始经纬度，**不经过 UTM 再反投影**，消除投影精度损失
- `net_offset`（即 `ox, oy`）携带在 `meta` 中，后续 physics engine 用于 SUMO 坐标 → UTM 的还原：`utm_x = sumo_node_x + ox`
- 幂等缓存基于 `(sel_bounds + threshold)` 的 MD5，同一参数多次调用直接返回，前端重复点击无副作用

---

### 3.4 `scene_blueprint.py`

```python
from flask import Blueprint, request, jsonify
from scene_service import prepare_scene

scene_bp = Blueprint("scene", __name__)

@scene_bp.route("/prepare", methods=["POST"])
def api_prepare():
    body = request.get_json(force=True)
    required = ("minx", "miny", "maxx", "maxy")
    if not all(k in body for k in required):
        return jsonify({"error": "缺少必要参数: minx/miny/maxx/maxy"}), 400
    try:
        sel_bounds = {k: float(body[k]) for k in required}
        ctx = prepare_scene(
            sel_bounds,
            threshold=float(body.get("threshold", 120)),
            height_column=body.get("height_column"),
        )
        return jsonify(ctx)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@scene_bp.route("/<scene_id>", methods=["GET"])
def api_get_scene(scene_id: str):
    from scene_service import _cache
    for ctx in _cache.values():
        if ctx["scene_id"] == scene_id:
            return jsonify(ctx)
    return jsonify({"error": "scene_id 不存在或已失效"}), 404
```

---

### 3.5 注册 Blueprint（`backend/app.py`）

```python
# 在现有 geo_bp 注册之后追加：
from environment.scene.scene_blueprint import scene_bp
app.register_blueprint(scene_bp, url_prefix="/api/scene")
```

---

## 四、接口契约（前后端唯一约定）

### `POST /api/scene/prepare`

**请求体**
```json
{
  "minx": 120.895,  "miny": 31.073,
  "maxx": 120.904,  "maxy": 31.086,
  "threshold": 120,
  "height_column": "Height"
}
```

**响应体（SceneContext，轻量级 meta）**
```json
{
  "scene_id":   "a3f8c2d1-...",
  "sel_bounds": { "minx": 120.895, "miny": 31.073, "maxx": 120.904, "maxy": 31.086 },
  "threshold":  120,
  "meta": {
    "road_source":  "osm",
    "road_nodes":   412,
    "road_edges":   589,
    "created_at":   "2026-03-28T12:00:00Z",
    "utm_zone":     51,
    "utm_band":     "N",
    "net_offset":   { "ox": 327642.99, "oy": 3443801.12 }
  },
  "road_network": {
    "nodes": [{ "id": "123456", "lng": 120.897, "lat": 31.075 }],
    "edges": [{ "shape": [[120.897, 31.075], [120.898, 31.076]] }],
    "bounds": { "min_lng": 120.895, "min_lat": 31.073, "max_lng": 120.904, "max_lat": 31.086 }
  }
}
```

**说明**：
- 响应体**不包含** buildings GeoJSON，体积始终 < 500KB
- 建筑数据通过现有 `POST /api/geo/query` 独立加载
- 道路叠加层通过现有 `POST /api/geo/roads` 独立加载

### `POST /api/geo/query`（现有，UnifiedMapView 复用）

入参与现有 GeoTool 完全一致，`UnifiedMapView` 直接调用。

### `POST /api/geo/roads`（现有，UnifiedMapView 复用）

入参与现有 GeoTool 完全一致，`UnifiedMapView` 直接调用。

---

## 五、后续 Physics Engine 坐标约定（预留规范）

> 当前 `backend/environment/simulation/` 目录为空，此节为预留约定，physics engine 实现时必须遵守。

| 场景 | 坐标系 | 原因 |
|---|---|---|
| 实体内部状态（无人机/卡车位置、速度） | UTM Zone 51N，米制 | 欧氏距离、能耗公式、速度积分全部依赖欧氏空间 |
| 碰撞检测、路径规划、避障 | UTM Zone 51N，米制 | 同上 |
| WebSocket 帧推送给前端 | **WGS84**，在发送前最后一刻转换 | 调用 `coord_transformer.utm_to_wgs84(x + ox, y + oy)` |
| REST API 响应体 | **WGS84** | 同上 |

**还原公式**（已在 `SceneContext.meta.net_offset` 中携带）：

$$x_{utm} = x_{sumo} + o_x \qquad y_{utm} = y_{sumo} + o_y$$

$$\text{lon, lat} = \text{utm\_to\_wgs84}(x_{utm},\ y_{utm})$$

### 5.1 WebSocket 帧格式规范（TICK_SYNC）

Physics engine **不得**逐个实体推送坐标消息。每次仿真帧（推荐间隔 250ms，与 `useMapTicker` 节流周期对齐）推送一条合并帧：

```json
{
  "type": "TICK_SYNC",
  "payload": {
    "sim_time": 10.5,
    "drones": [["d1", 120.897, 31.075, 45.0, 0.82], ["d2", ...]],
    "trucks": [["t1", 120.901, 31.079], ["t2", ...]]
  }
}
```

字段说明：
- `sim_time`：仿真逻辑时间（秒，从 0 开始的相对时间），**必须携带**
- `drones` 数组元素：`[id, lng, lat, altitude_m, battery_ratio]`，采用数组而非字典以减少 JSON 体积约 60%
- `trucks` 数组元素：`[id, lng, lat]`

**前端收到 TICK_SYNC 后的处理链（强制约定）**：
```typescript
// DispatchCenter 中 WsClient 监听
ws.on('TICK_SYNC', (payload) => {
  systemStore.simTime = payload.sim_time   // ← 必须：驱动仿真时钟 UI
  payload.drones.forEach(([id, lng, lat]) => mapRef.value?.updateDrone(id, [lng, lat]))
  payload.trucks.forEach(([id, lng, lat]) => mapRef.value?.updateTruck(id, [lng, lat]))
})
```

> **`systemStore.simTime` 说明**：`stores/system.ts` 中已定义 `simTime: ref(0)`，TICK_SYNC 是唯一有权写入该值的来源。前端所有需要感知仿真时间的组件（订单超时倒计时、告警面板）统一 `watch(systemStore.simTime)`，不得各自维护本地时间。

---

## 六、前端新增/修改

### 6.1 新增 Store：`stores/scene.ts`

**职责**：作为 `SimulationBox/GeoTool` 与 `DispatchCenter` 之间的唯一数据桥梁，两端视图不直接 import 对方。

```typescript
// stores/scene.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface NetOffset { ox: number; oy: number }

export interface SceneMeta {
  road_source:  'osm' | 'grid'
  road_nodes:   number
  road_edges:   number
  created_at:   string
  utm_zone:     number
  utm_band:     string
  net_offset:   NetOffset
}

export interface RoadNode { id: string; lng: number; lat: number }
export interface RoadEdge { shape: [number, number][] }

export interface SceneBounds {
  min_lng: number; min_lat: number; max_lng: number; max_lat: number
}

export interface SceneContext {
  scene_id:     string
  sel_bounds:   { minx: number; miny: number; maxx: number; maxy: number }
  threshold:    number
  meta:         SceneMeta
  road_network: { nodes: RoadNode[]; edges: RoadEdge[]; bounds: SceneBounds }
}

export const useSceneStore = defineStore('scene', () => {
  const context   = ref<SceneContext | null>(null)
  const loading   = ref(false)
  const error     = ref<string | null>(null)

  async function prepareScene(payload: {
    sel_bounds:    { minx: number; miny: number; maxx: number; maxy: number }
    threshold:     number
    height_column: string | null
  }) {
    if (loading.value) return
    loading.value = true
    error.value   = null
    try {
      const res = await fetch('/api/scene/prepare', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          ...payload.sel_bounds,
          threshold:     payload.threshold,
          height_column: payload.height_column,
        }),
      })
      if (!res.ok) {
        const e = await res.json()
        throw new Error(e.error ?? res.statusText)
      }
      context.value = await res.json()
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  function clear() { context.value = null; error.value = null }

  return { context, loading, error, prepareScene, clear }
})
```

---

### 6.2 GeoTool `SideBar.vue` 新增导出按钮

在现有 `queryResult` 卡片之后、`导出数据` 卡片之后新增独立操作卡：

```
[🔍 分析选区建筑]              ← 已有
────────────────────────────
[统计结果面板]                  ← 已有（queryResult 存在时显示）
[💾 导出数据]                  ← 已有
────────────────────────────
[🚀 导出到指挥中心地图]         ← 新增（queryResult 存在时显示，样式突出区分）
```

**交互逻辑**：
1. 用户点击按钮 → 调用 `sceneStore.prepareScene(selBounds, threshold, height_column)`
2. 按钮显示 Loading 状态，`disabled` 防止重复触发（幂等异常仍保留 UX 保障）
3. `sceneStore.loading` 结束后：
   - 成功 → 调用 `router.push('/dispatch')`，`DispatchCenter` 自动读取 Store
   - 失败 → 在按钮下方内联显示 `sceneStore.error`，不弹窗（符合现有 UI 风格）

**新增 emit**（`SideBar.vue` → `GeoTool/index.vue`）：
```typescript
(e: 'export-to-dispatch'): void
```

**`GeoTool/index.vue` 处理函数**：
```typescript
import { useSceneStore } from '@/stores/scene'
import { useRouter }     from 'vue-router'

const sceneStore = useSceneStore()
const router     = useRouter()

async function handleExportToDispatch() {
  if (!selBounds.value) return
  await sceneStore.prepareScene({
    sel_bounds:    selBounds.value,
    threshold:     currentThreshold.value,
    height_column: currentHCol.value,
  })
  if (!sceneStore.error) {
    router.push('/dispatch')
  }
}
```

---

### 6.3 新增组件：`DispatchCenter/components/UnifiedMapView.vue`

**职责**：封装所有地图与坐标逻辑。`DispatchCenter/index.vue` 只负责布局，不包含任何坐标计算代码。

#### 图层架构

```
UnifiedMapView
└── L.Map（renderer: L.canvas()  ← 强制指定，Canvas 模式）
    ├── baseLayer        CartoDB DarkMatter（底图，与 GeoTool 保持一致）
    ├── buildingLayer    L.GeoJSON（WGS84，POST /api/geo/query 异步加载）
    │                    颜色：红 = 禁飞（nf: true）/ 绿 = 可飞（nf: false）
    ├── roadLayer        L.GeoJSON（WGS84 LineString，POST /api/geo/roads 异步加载）
    │                    颜色：#e6a817（与 GeoTool 图例一致）
    ├── facilityLayer    L.LayerGroup（仓库/充换电站图标）  ← 预留接口
    ├── truckLayer       L.LayerGroup（卡车轨迹 + 实时位置）← 预留接口
    ├── droneLayer       L.LayerGroup（无人机实时位置 + 轨迹）← 预留接口
    ├── orderLayer       L.LayerGroup（订单热力点）         ← 预留接口
    └── layerControl     L.Control.Layers（图层显隐开关面板）
```

#### 强制 Canvas 渲染（初始化关键代码）

```typescript
// UnifiedMapView.vue  <script setup>
const renderer = L.canvas({ padding: 0.5 })
map = L.map(mapEl.value, {
  renderer,
  center:   [sceneBounds.center_lat, sceneBounds.center_lng],
  zoom:     14,
})

// 所有 GeoJSON 层必须传入 renderer 选项
buildingLayer = L.geoJSON(null, {
  renderer,
  style: f => ({ color: f.properties.nf ? '#ff4444' : '#33bb55', ... }),
}).addTo(map)
```

#### 视口自适应

```typescript
// sceneContext 就绪后自动缩放到仿真区域
const b = scene.road_network.bounds
map.fitBounds([
  [b.min_lat, b.min_lng],
  [b.max_lat, b.max_lng],
], { padding: [20, 20] })
```

#### 异步双流加载序列（非阻塞）

```typescript
async function loadLayers(scene: SceneContext) {
  const bounds = scene.sel_bounds

  // 两个请求并行发出，互不阻塞
  const [buildRes, roadRes] = await Promise.allSettled([
    fetch('/api/geo/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...bounds, threshold: scene.threshold, max: 30000,
      }),
    }).then(r => r.json()),

    fetch('/api/geo/roads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bounds),
    }).then(r => r.json()),
  ])

  if (buildRes.status === 'fulfilled') buildingLayer.addData(buildRes.value)
  if (roadRes.status  === 'fulfilled') roadLayer.addData(roadRes.value)
}
```

> **为什么复用现有端点而非 `/api/scene/prepare` 一次性返回？**
> 1. 现有端点已稳定运行，无需引入新代码路径
> 2. 两个请求并行，总延迟 = max(建筑查询, 路网查询)，而非两者之和
> 3. 避免 prepare 响应体体积膨胀（buildings GeoJSON 单独可达数 MB）
> 4. 不引入静态文件落盘、目录管理、文件清理等额外复杂度

#### 预留动态实体接口（P2 阶段使用）

```typescript
// 外部通过 expose 调用，内部维护图层，不暴露地图实例
defineExpose({
  // P2：仓库/电站（静态，一次性调用）
  setFacilities(depots: Depot[], stations: Station[]): void,
  // P2：卡车实时位置更新（WebSocket 每帧调用）
  updateTruck(id: string, lngLat: [number, number]): void,
  // P2：无人机实时位置更新（WebSocket 每帧调用）
  updateDrone(id: string, lngLat: [number, number]): void,
  // P2：追加订单热力点
  addOrder(order: Order): void,
  // P2：清除全部动态图层
  clearDynamic(): void,
})
```

---

### 6.4 `DispatchCenter/index.vue` 修改

将现有 `dispatch-map` 占位块替换为 `UnifiedMapView`：

```html
<!-- 中央地图区 -->
<div class="dispatch-map">
  <UnifiedMapView v-if="sceneStore.context" ref="mapRef" />
  <div v-else class="map-placeholder">
    <!-- 保留原有占位内容，引导用户先在仿真配置中选取区域 -->
    <div class="map-placeholder__icon">🌐</div>
    <p class="map-placeholder__title">尚未加载仿真场景</p>
    <p class="map-placeholder__desc">
      请前往「仿真与动态场景」→「地图选取与环境构建」，<br />
      完成建筑分析后点击「导出到指挥中心地图」
    </p>
  </div>
</div>
```

```typescript
import { useSceneStore } from '@/stores/scene'
import UnifiedMapView    from './components/UnifiedMapView.vue'

const sceneStore = useSceneStore()
const mapRef     = ref<InstanceType<typeof UnifiedMapView> | null>(null)
```

---

## 七、数据流全景（修订版）

```
SimulationBox / GeoTool
│
│  用户框选 → 分析建筑（POST /api/geo/query）→ queryResult 就绪
│                             ↓
│              [🚀 导出到指挥中心地图] 按钮
│                             ↓
│              POST /api/scene/prepare
│              ← SceneContext（轻量级 meta，< 500KB）
│                             ↓
│              sceneStore.context = SceneContext
│                             ↓
│              router.push('/dispatch')
│
DispatchCenter
│
│  UnifiedMapView 挂载（watch sceneStore.context）
│        ↓
│  并行发起两个请求（复用现有端点）：
│  ├── POST /api/geo/query → buildings GeoJSON → buildingLayer
│  └── POST /api/geo/roads → roads GeoJSON    → roadLayer
│        ↓
│  map.fitBounds(scene.road_network.bounds)
│        ↓
│  ✅ 统一坐标地图就绪（建筑禁飞层 + 路网层对齐）
│
│  （P2）WebSocket 帧推送（每 250ms 一帧）：
│  后端 physics engine 内部坐标为 UTM
│        ↓
│  按帧合并所有实体，坐标转 WGS84，组装 TICK_SYNC 帧
│        ↓
│  WsMessage { type: "TICK_SYNC", payload: {
│    sim_time,
│    drones: [[id, lng, lat, alt, battery], ...],
│    trucks: [[id, lng, lat], ...]
│  }}
│        ↓
│  前端收到后：
│  ├── systemStore.simTime = sim_time    （驱动仿真时钟 UI）
│  ├── mapRef.updateDrone(id, [lng, lat]) × N
│  └── mapRef.updateTruck(id, [lng, lat]) × M
```

---

## 八、补全原方案遗漏项（完整版）

| 编号 | 遗漏点 | 已纳入方案的解法 |
|---|---|---|
| L1 | SUMO netOffset 未传给前端/算法层 | `SceneContext.meta.net_offset` 强制携带 `{ox, oy}`，还原公式在文档中明确 |
| L2 | SceneContext 响应体体积过大 | `prepare` 只返回 meta，可视化数据通过现有 geo 端点并行加载 |
| L3 | Leaflet SVG 渲染性能 | `UnifiedMapView` 强制 `L.canvas()`，所有 GeoJSON 层携带 `renderer` 选项 |
| L4 | 重复点击导致多次后端调用 | 前端 `loading` 状态禁用按钮，后端参数哈希幂等缓存 |
| L5 | 无网络时 OSM 下载失败 | `scene_service` 自动降级网格路网，`meta.road_source` 标注来源 |
| L6 | 场景生命周期/版本化 | `scene_id` (UUID) + 内存缓存，`GET /api/scene/{id}` 支持会话恢复 |
| L7 | 后续动态实体坐标规范无约束 | 铁律 A1/A3 明确：physics 内部 UTM，对外 WGS84；`types/index.ts` 已定义 `LatLon` |
| L8 | 两视图直接耦合 | `sceneStore` 作为唯一桥梁，`GeoTool` 不 import `DispatchCenter` |
| L9 | 仿真时钟与地图刷新无节流 | 预留 `useMapTicker(mapRef, simTime)` Composable，P2 实现时节流 250ms 刷新 |
| L11 | WebSocket 逐实体推送导致消息条数爆炸 | TICK_SYNC 合并帧：每帧1条消息，数组格式去除 key 冗余，体积压缩约 60% |
| L12 | `systemStore.simTime` 与 WebSocket 帧未连线 | TICK_SYNC 帧强制携带 `sim_time`，前端收帧后唯一写入路径为 `systemStore.simTime = payload.sim_time` |
| L10 | 动态图层接口无约定 | `UnifiedMapView` 通过 `defineExpose` 提供 `setFacilities / updateTruck / updateDrone / addOrder / clearDynamic` 接口 |

---

## 九、开发阶段规划

| 阶段 | 任务 | 文件范围 |
|---|---|---|
| **P0** | `coord_transformer.py` | `backend/environment/scene/` |
| **P0** | `scene_service.py`（含幂等缓存） | `backend/environment/scene/` |
| **P0** | `scene_blueprint.py` + `app.py` 注册 | `backend/` |
| **P0** | `stores/scene.ts` | `frontend/src/stores/` |
| **P0** | `SideBar.vue` 新增导出按钮 + loading 状态 | `GeoTool/components/SideBar.vue` |
| **P0** | `GeoTool/index.vue` 新增 `handleExportToDispatch` | `GeoTool/index.vue` |
| **P1** | `UnifiedMapView.vue`（底图 + Canvas + buildingLayer + roadLayer） | `DispatchCenter/components/` |
| **P1** | `DispatchCenter/index.vue` 接入 `UnifiedMapView` | `DispatchCenter/index.vue` |
| **P1** | 视口自适应 + 图层控制面板 + meta 浮窗 | `UnifiedMapView.vue` |
| **P2** | `defineExpose` 动态实体接口实现 | `UnifiedMapView.vue` |
| **P2** | `useMapTicker` Composable | `frontend/src/composables/useMapTicker.ts` |
| **P2** | Physics engine 使用 `coord_transformer` 推送 WGS84 帧 | `backend/environment/simulation/` |

---

## 十、高内聚低耦合关键决策一览

| 决策 | 理由 |
|---|---|
| `scene` 后端模块独立于 `geo` 模块 | `geo` 只知道查建筑/生路网，`scene` 只负责打包，单一职责 |
| `sceneStore` 作为两视图唯一通信桥梁 | 路由视图解耦，`GeoTool` 与 `DispatchCenter` 零直接依赖 |
| `UnifiedMapView` 封装全部坐标与渲染逻辑 | `DispatchCenter` 组件零坐标代码，地图引擎可独立替换 |
| 复用现有 geo 端点而非 prepare 打包 | 不引入新代码路径，并行加载性能更优，避免文件落盘复杂度 |
| WGS84 为对外唯一坐标协议 | `types/index.ts` 已有 `LatLon` 约束，store 层统一，算法层内部自由 |
| Canvas 渲染模式强制指定 | 3000+ 静态建筑 + 高频动态实体，SVG 无法承受；Canvas 一行代码解决 |
| `net_offset` 随 SceneContext 携带 | physics engine 无需重新解析 `net.xml` 即可完成坐标还原 |
| 幂等缓存基于参数哈希 | 前端无需幂等逻辑，后端自然去重，分布式扩展时替换 Redis 即可 |
