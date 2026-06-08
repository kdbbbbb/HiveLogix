# UAV 禁飞区地图生成工具

基于上海建筑高度矢量数据，自动识别无人机禁飞区并支持导出 SUMO 仿真文件。

---

## 项目背景

### 数据来源

| 项目 | 说明 |
|---|---|
| 数据名称 | 上海市建筑轮廓与建筑高度矢量数据 |
| 数据时间 | 2020 年 |
| 发布时间 | 2024 年 5 月 30 日 |
| 坐标系 | WGS1984（EPSG:4326） |
| 格式 | Shapefile（.shp） |
| 数据量 | ~236 MB（.shp 主文件） |

### 数据说明

该数据包含建筑轮廓矢量与建筑高度信息，字段如下：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `id` | 数值 | 建筑唯一编号 |
| `Height` | 浮点 | 建筑高度（米） |
| `geometry` | Polygon | 建筑轮廓（WGS84 经纬度） |

建筑高度由研究者通过集成多源遥感特征（SAR、光学、地形、社会经济图像）和 XGBoost 机器学习回归方法估算得到，训练数据来自 ONEGEO Map、微软建筑物足迹、百度地图等。

建筑轮廓基础数据来自 Qian Shi 等学者发布的东亚五国建筑矢量数据，通过 Google Earth 2020–2022 年 0.5m 分辨率影像提取。

**数据覆盖范围（经纬度）：**
```
西南角：31.0733°N, 120.8953°E
东北角：31.0862°N, 120.9035°E
```

---

## 项目功能

### 核心逻辑

设定一个**飞行高度阈值**（地面高度 + 无人机常规飞行高度 AGL），将建筑物分为两类：

- 🔴 **禁飞区**：建筑高度 ≥ 阈值（超高建筑，无人机需绕行或抬升）
- 🟢 **可飞区**：建筑高度 < 阈值（可在该高度安全飞越）

### 可视化界面

基于 Flask + Leaflet.js 的 Web 地图工具，在浏览器中交互操作：

| 功能 | 说明 |
|---|---|
| 数据加载 | 后台线程异步加载，前端实时显示进度条 |
| 手动框选 | 在地图上拖拽绘制矩形选区 |
| 中心点选区 | 输入尺寸（默认 4×4 km），点击地图自动生成选框 |
| 高度阈值调节 | 滑块 + 数字输入，范围 10–500m，实时联动 |
| 四角坐标显示 | 框选后显示西南/东北/西北/东南/中心点的精确经纬度（6位小数） |
| 一键复制坐标 | 复制选区四角坐标及 minLon/maxLon/minLat/maxLat |
| 建筑渲染 | 红色=禁飞，绿色=可飞，鼠标悬停显示具体高度 |
| 统计面板 | 选区总建筑数、禁飞区数量、可飞区数量及占比 |

### 导出格式

| 格式 | 文件 | 说明 |
|---|---|---|
| **SUMO 真实路网包**（推荐） | `uav_no_fly_sumo_osm.zip` | 下载选区内真实 OSM 道路生成路网，解压即用 |
| SUMO 网格路网包 | `uav_no_fly_sumo.zip` | 自动生成简化网格路网，无需联网 |
| SUMO 仅叠加层 | `no_fly_zones.add.xml` | 配合已有路网使用 |
| GeoJSON | `buildings.geojson` | 适用于 QGIS、网页地图 |
| CSV | `buildings.csv` | 含质心坐标、高度、禁飞标记，便于数据分析 |

---

## 文件结构

```
D:\Judy_Tongji\UAV\map\
│
├── app.py                      # Flask 后端主程序
├── requirements.txt            # Python 依赖列表
├── README.md                   # 本文档
│
├── templates\
│   └── index.html              # 前端地图界面
│
├── output\                     # 导出文件存放目录（自动创建）
│   ├── uav_no_fly_sumo.zip     # SUMO 完整包（导出后生成）
│   ├── no_fly_zones.add.xml    # SUMO 禁飞区叠加层
│   ├── grid.net.xml            # SUMO 网格路网
│   ├── buildings.geojson       # GeoJSON 导出
│   └── buildings.csv           # CSV 导出
│
└── shanghai_map\               # 原始数据（只读，勿修改）
    ├── shanghai.shp            # 建筑轮廓与高度主文件（236 MB）
    ├── shanghai.dbf            # 属性数据库（73 MB）
    ├── shanghai.shx            # 空间索引（12 MB）
    ├── shanghai.prj            # 坐标系定义
    └── shanghai.cpg            # 字符编码
```

---

## 环境配置与运行

### 依赖安装

```bash
pip install flask geopandas pandas numpy shapely pyproj
```

> 推荐使用 conda 安装 geopandas，避免依赖冲突：
> ```bash
> conda install geopandas flask pandas numpy
> ```

### 启动服务

```bash
cd D:\Judy_Tongji\UAV\map
python app.py
```

启动后在浏览器打开：**http://localhost:5000**

> 首次加载约需 30–60 秒（读取 236MB shapefile），页面进度条实时显示，加载完成后自动跳转地图。

---

## SUMO 仿真使用方法

### 导出流程

1. 在地图中框选目标区域（建议 ≤ 5×5 km，避免文件过大）
2. 设置飞行高度阈值（例如城区低空 50m，常规巡航 120m）
3. 点击「分析选区建筑」，确认红/绿渲染结果
4. 选择格式「**SUMO 真实路网 (.zip)**」（默认，需联网）
5. 点击「下载导出文件」，得到 `uav_no_fly_sumo_osm.zip`

> 真实路网导出需向 Overpass API 联网下载 OSM 数据；
> 如无网络，可选「**SUMO 网格路网**」离线导出。

### SUMO 加载

解压 zip 后，在 SUMO 安装目录执行：

```bash
# 真实路网包解压后
sumo-gui -n roads.net.xml --additional-files no_fly_zones.add.xml

# 网格路网包解压后
sumo-gui -n grid.net.xml --additional-files no_fly_zones.add.xml
```

在 sumo-gui 中：
- **红色多边形**（`type="no_fly_zone"`）= 建筑高度 ≥ 阈值，无人机禁飞区
- **绿色多边形**（`type="fly_zone"`）= 建筑高度 < 阈值，可飞区

### 导出文件坐标系说明

| 文件 | 坐标系 | 原点 |
|---|---|---|
| `roads.net.xml` | EPSG:32651 (UTM Zone 51N)，单位：米 | 选区左下角 |
| `grid.net.xml` | EPSG:32651 (UTM Zone 51N)，单位：米 | 选区左下角 |
| `no_fly_zones.add.xml` | EPSG:32651 (UTM Zone 51N)，单位：米 | 选区左下角（与路网一致） |

两个文件坐标系完全对齐，可直接叠加使用。

---

## 技术栈

| 层次 | 技术 |
|---|---|
| 后端 | Python 3, Flask |
| 地理空间处理 | GeoPandas, Shapely, PyProj |
| 前端地图 | Leaflet.js 1.9, Leaflet.draw |
| UI 框架 | Bootstrap 5 |
| 底图 | CartoDB Dark Matter / OpenStreetMap |
| SUMO 路网生成 | 纯 Python 实现（无需安装 SUMO 即可生成路网文件） |

---

## 已知限制

- 数据覆盖范围为上海局部区域（约 1km × 1.5km），非全市数据
- 建筑高度为模型估算值，非测量值，存在一定误差
- 框选区域过大（> 10万栋建筑）时，前端仅渲染部分结果；导出文件包含全量数据
- 真实路网导出需联网，离线环境请使用「网格路网」选项
- 生成的 `roads.net.xml` 仅包含 OSM 长途道路类型，小巷、人行沙滑等可能被过滤
