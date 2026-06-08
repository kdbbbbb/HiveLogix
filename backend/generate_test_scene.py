#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为测试场景生成禁飞区建筑数据和默认实体配置

此脚本：
1. 从 Shapefile 查询该区域的建筑物数据
2. 导出为 GeoJSON 格式（禁飞区）
3. 创建默认的仓库、充电站、任务点、无人机配置
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment", "geo"))
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_state, get_gdf, load_shapefile_async
from building_service import query_buildings

# 等待 Shapefile 加载完成
print("[INFO] 等待 Shapefile 加载...")
load_shapefile_async()
import time
for i in range(60):
    state = get_state()
    if state["loaded"]:
        print(f"[OK] Shapefile 加载完成 ({i+1}s)")
        break
    time.sleep(1)
else:
    print("❌ Shapefile 加载超时")
    sys.exit(1)

# 测试区域坐标
MIN_LAT = 31.023747
MAX_LAT = 31.059783
MIN_LON = 121.195602
MAX_LON = 121.237661

CENTER_LAT = 31.041765
CENTER_LON = 121.216631

CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_data", "default_scene")
os.makedirs(CACHE_DIR, exist_ok=True)

print("\n[INFO] 查询该区域的建筑物禁飞区...")
try:
    # 高度阈值 80m
    threshold = 80
    state = get_state()
    height_column = state.get("height_column", "height")

    # data_loader 会将建筑数据统一转换到 EPSG:4326，因此这里直接使用经纬度。
    # query_buildings 参数顺序为 (minx=min_lng, miny=min_lat, maxx=max_lng, maxy=max_lat)。
    
    result = query_buildings(
        get_gdf(),
        MIN_LON, MIN_LAT, MAX_LON, MAX_LAT,
        threshold=threshold,
        h_col=height_column,
        max_feat=30000,
    )
    
    print(f"[OK] 查询完成，找到 {result['stats']['no_fly']} 个禁飞区（高度>80m）")
    
    # 保存禁飞区 GeoJSON
    buildings_path = os.path.join(CACHE_DIR, "buildings.geojson")
    with open(buildings_path, "w", encoding="utf-8") as f:
        json.dump(result["features"], f)
    print(f"[OK] 禁飞区建筑已保存：{buildings_path}")

except Exception as e:
    print(f"❌ 查询失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建默认实体配置
print("\n[INFO] 生成默认实体配置...")

entities_config = {
    "depots": [
        {
            "depot_id": "DEP-TEST-01",
            "name": "仓库-中心",
            "lng": CENTER_LON,
            "lat": CENTER_LAT,
            "altitude": 0,
            "capacity": 500,
            "swap_time": 120,
            "parking_slots": 4,
        }
    ],
    "stations": [
        {
            "station_id": f"STA-TEST-{i:02d}",
            "name": f"充电站-{i:02d}",
            "lng": CENTER_LON + (i % 5) * 0.004,  # 在中心周围均匀分布
            "lat": CENTER_LAT + (i // 5) * 0.004,
            "altitude": 0,
            "swap_time": 120,
            "parking_slots": 2,
        }
        for i in range(1, 11)  # 10 个充电站
    ],
    "trucks": [
        {
            "truck_id": "TRK-TEST-01",
            "name": "卡车-01",
            "speed": 13.0,
            "max_inventory": 30,
            "swap_time": 120,
            "parking_slots": 3,
            "home_depot_id": "DEP-TEST-01",
        }
    ],
    "drones": [
        # 8 架 LightDrone，其中 4 架初始搭载在卡车上
        *[
            {
                "drone_id": f"UAV-TEST-{i:02d}",
                "drone_type": "LightDrone",
                "home_id": "TRK-TEST-01" if i <= 4 else "DEP-TEST-01",
                "home_type": "TRUCK" if i <= 4 else "DEPOT",
            }
            for i in range(1, 9)
        ],
        # 4 架 HeavyDrone，其中 2 架初始搭载在卡车上
        {
            "drone_id": "UAV-TEST-09",
            "drone_type": "HeavyDrone",
            "home_id": "DEP-TEST-01",
            "home_type": "DEPOT",
        },
        {
            "drone_id": "UAV-TEST-10",
            "drone_type": "HeavyDrone",
            "home_id": "TRK-TEST-01",
            "home_type": "TRUCK",
        },
        {
            "drone_id": "UAV-TEST-11",
            "drone_type": "HeavyDrone",
            "home_id": "TRK-TEST-01",
            "home_type": "TRUCK",
        },
        {
            "drone_id": "UAV-TEST-12",
            "drone_type": "HeavyDrone",
            "home_id": "DEP-TEST-01",
            "home_type": "DEPOT",
        },
    ],
}

entities_path = os.path.join(CACHE_DIR, "entities.json")
with open(entities_path, "w", encoding="utf-8") as f:
    json.dump(entities_config, f, indent=2, ensure_ascii=False)
print(f"[OK] 实体配置已保存：{entities_path}")

# 创建默认任务点配置（采用随机+聚集分布，类似于 orderGen.ts）
print("\n[INFO] 生成默认任务点...")

import random
import math

def gaussian_rand():
    """Box-Muller 变换产生标准正态随机数 N(0,1)"""
    u1 = max(random.random(), 1e-10)
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def clustered_point(min_lng, max_lng, min_lat, max_lat, radius_km=1.5):
    """
    在 bbox 中随机选取一个热点，然后在热点附近正态分布。
    超出 bbox 的点截断到边界。
    """
    # 随机选热点
    hotspot_lng = min_lng + random.random() * (max_lng - min_lng)
    hotspot_lat = min_lat + random.random() * (max_lat - min_lat)
    
    # σ ≈ radius_km / 3：使约 99.7% 的点落在 radius_km 范围内
    sigma = radius_km / 3
    d_lat = gaussian_rand() * sigma / 111
    d_lng = gaussian_rand() * sigma / (111 * math.cos(math.radians(hotspot_lat)))
    
    return {
        "lng": max(min_lng, min(max_lng, hotspot_lng + d_lng)),
        "lat": max(min_lat, min(max_lat, hotspot_lat + d_lat)),
    }

orders_config = {
    "static_orders": [
        {
            "order_id": f"ORD-TEST-{i:03d}",
            "create_time": 0,
            "deadline": 600 + i * 20,  # 每个订单截止时间错开 20 秒
            # 采用聚集分布：在多个热点周围正态分布
            **clustered_point(MIN_LON, MAX_LON, MIN_LAT, MAX_LAT, radius_km=1.5),
            "delivery_z": 0,
            # 随机重量分布
            "payload_weight": round(random.uniform(0.5, 3.5), 2),
            # 优先级随机分配（80% 普通，15% 紧急，5% 低）
            "priority": random.choices(
                ["NORMAL", "URGENT", "LOW"],
                weights=[80, 15, 5],
                k=1
            )[0],
        }
        for i in range(1, 11)  # 10 个任务点
    ]
}

orders_path = os.path.join(CACHE_DIR, "orders.json")
with open(orders_path, "w", encoding="utf-8") as f:
    json.dump(orders_config, f, indent=2, ensure_ascii=False)
print(f"[OK] 任务点配置已保存：{orders_path}")

# 更新场景配置元数据
print("\n[INFO] 更新场景配置元数据...")
scene_config_path = os.path.join(CACHE_DIR, "scene_config.json")
with open(scene_config_path, "r", encoding="utf-8") as f:
    scene_config = json.load(f)

scene_config.update({
    "buildings_file": "buildings.geojson",
    "entities_file": "entities.json",
    "orders_file": "orders.json",
    "height_threshold": 80,
})

with open(scene_config_path, "w", encoding="utf-8") as f:
    json.dump(scene_config, f, indent=2, ensure_ascii=False)
print(f"[OK] 场景配置已更新")

print("\n✅ 测试场景的所有数据已生成！")
print(f"   数据位置：{CACHE_DIR}")
print(f"   文件列表：")
print(f"     - scene_config.json     (场景配置)")
print(f"     - osm_network.geojson   (道路网络)")
print(f"     - buildings.geojson     (禁飞区建筑)")
print(f"     - entities.json         (仓库、充电站、无人机)")
print(f"     - orders.json           (默认任务点)")
