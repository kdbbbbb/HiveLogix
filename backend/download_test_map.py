#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载约4×4km的OSM道路网络数据用于测试场景

位置：上海浦东区域
坐标：minLat=31.023747  maxLat=31.059783  minLon=121.195602  maxLon=121.237661
中心点：31.041765°N, 121.216631°E
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment", "geo"))

from osm_service import download_osm, osm_to_geojson, build_road_graph

# 测试区域坐标
MIN_LAT = 31.023747
MAX_LAT = 31.059783
MIN_LON = 121.195602
MAX_LON = 121.237661

CENTER_LAT = 31.041765
CENTER_LON = 121.216631

# 创建数据目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_data", "default_scene")
os.makedirs(CACHE_DIR, exist_ok=True)

print("[INFO] 开始下载 OSM 道路网络数据...")
print(f"[INFO] 区域范围：")
print(f"  西南角 (SW): {MIN_LAT}°N, {MIN_LON}°E")
print(f"  东北角 (NE): {MAX_LAT}°N, {MAX_LON}°E")
print(f"  中心点 (CTR): {CENTER_LAT}°N, {CENTER_LON}°E")

try:
    # 下载 OSM 数据（注意：Overpass API 的顺序是 (miny, minx, maxy, maxx)）
    osm_xml = download_osm(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)
    print(f"[OK] OSM 数据下载完成，大小：{len(osm_xml)} 字节")

    # 保存原始 XML
    osm_xml_path = os.path.join(CACHE_DIR, "osm_network.xml")
    with open(osm_xml_path, "w", encoding="utf-8") as f:
        f.write(osm_xml)
    print(f"[OK] 原始 OSM XML 已保存：{osm_xml_path}")

    # 转换为 GeoJSON
    geojson = osm_to_geojson(osm_xml)
    geojson_path = os.path.join(CACHE_DIR, "osm_network.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"[OK] OSM GeoJSON 已保存：{geojson_path}")
    print(f"[INFO] 路网特征数量：{len(geojson['features'])}")

    # 保存场景配置
    scene_config = {
        "scene_id": "default_test_4x4km",
        "name": "测试场景 - 4×4km (上海浦东)",
        "description": "预加载的测试场景，包含真实 OSM 路网和建筑禁飞区",
        "region": "shanghai_pudong",
        "bounds": {
            "min_lng": MIN_LON,
            "max_lng": MAX_LON,
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
        },
        "center": {
            "lng": CENTER_LON,
            "lat": CENTER_LAT,
        },
        "road_network": {
            "source": "osm",
            "file": "osm_network.geojson",
            "bounds": {
                "min_lng": MIN_LON,
                "max_lng": MAX_LON,
                "min_lat": MIN_LAT,
                "max_lat": MAX_LAT,
            },
        },
        "sel_bounds": {
            "minx": 3821000,
            "miny": 534300,
            "maxx": 3821400,
            "maxy": 534600,
        },
        "threshold": 80,
        "height_column": "height",
    }
    scene_config_path = os.path.join(CACHE_DIR, "scene_config.json")
    with open(scene_config_path, "w", encoding="utf-8") as f:
        json.dump(scene_config, f, indent=2, ensure_ascii=False)
    print(f"[OK] 场景配置已保存：{scene_config_path}")

    print("\n✅ 测试场景数据下载完成！")
    print(f"   数据保存在：{CACHE_DIR}")

except Exception as e:
    print(f"❌ 下载失败：{e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
