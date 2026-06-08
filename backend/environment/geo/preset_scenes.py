#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预设场景缓存加载服务

支持从磁盘加载预生成的场景数据，避免每次都调用 Overpass API 和 Shapefile 查询
"""

import json
import os
from pathlib import Path


CACHE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "default_scene")
DEFAULT_PRESET_ORDERS_FILE = "orders_ppo_training_intensity.json"

# 预设场景列表
PRESET_SCENES = {
    "default_test_4x4km": {
        "name": "测试场景 - 4×4km (上海浦东)",
        "description": "预加载的测试场景，包含真实 OSM 路网和建筑禁飞区",
        "bounds": {
            "min_lng": 121.195602,
            "max_lng": 121.237661,
            "min_lat": 31.023747,
            "max_lat": 31.059783,
        },
    }
}


def get_preset_scene(scene_id: str) -> dict | None:
    """获取预设场景的完整数据（包含场景配置、路网、建筑、实体、任务点）"""
    if scene_id not in PRESET_SCENES:
        return None
    
    scene_dir = os.path.join(CACHE_BASE_DIR, scene_id if scene_id != "default_test_4x4km" else ".")
    
    result = {
        "scene_id": scene_id,
        "name": PRESET_SCENES[scene_id]["name"],
        "description": PRESET_SCENES[scene_id]["description"],
        "bounds": PRESET_SCENES[scene_id]["bounds"],
    }
    
    # 加载 scene_config.json
    config_path = os.path.join(CACHE_BASE_DIR, "scene_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                result["config"] = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载场景配置失败: {e}")
    
    # 加载 osm_network.geojson
    osm_path = os.path.join(CACHE_BASE_DIR, "osm_network.geojson")
    if os.path.exists(osm_path):
        try:
            with open(osm_path, "r", encoding="utf-8") as f:
                result["osm_network"] = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载 OSM 数据失败: {e}")
    
    # 加载 buildings.geojson
    buildings_path = os.path.join(CACHE_BASE_DIR, "buildings.geojson")
    if os.path.exists(buildings_path):
        try:
            with open(buildings_path, "r", encoding="utf-8") as f:
                result["buildings"] = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载建筑数据失败: {e}")
    
    # 加载 entities.json
    entities_path = os.path.join(CACHE_BASE_DIR, "entities.json")
    if os.path.exists(entities_path):
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                result["entities"] = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载实体配置失败: {e}")
    
    # 加载默认前端预设订单；静态订单与动态订单结构均与 orders.json 对齐。
    orders_path = os.path.join(CACHE_BASE_DIR, DEFAULT_PRESET_ORDERS_FILE)
    if os.path.exists(orders_path):
        try:
            with open(orders_path, "r", encoding="utf-8") as f:
                result["orders"] = json.load(f)
        except Exception as e:
            print(f"[WARN] 加载任务点失败: {e}")
    
    return result


def get_buildings_from_cache(scene_id: str) -> dict | None:
    """仅获取建筑禁飞区 GeoJSON（用于 /api/geo/query）"""
    if scene_id not in PRESET_SCENES:
        return None
    
    buildings_path = os.path.join(CACHE_BASE_DIR, "buildings.geojson")
    if os.path.exists(buildings_path):
        try:
            with open(buildings_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # 兼容两种缓存格式：
            # 1) 直接 features 数组（历史格式）
            # 2) 完整 FeatureCollection（标准 GeoJSON）
            if isinstance(raw, dict) and raw.get("type") == "FeatureCollection":
                features = raw.get("features", [])
            elif isinstance(raw, list):
                features = raw
            else:
                print("[WARN] buildings.geojson 格式异常，期望 FeatureCollection 或 Feature 数组")
                return None

            # 转换为 /api/geo/query 的返回格式
            return {
                "type": "FeatureCollection",
                "features": features,
                "stats": {
                    "total": len(features),
                    "shown": len(features),
                    "no_fly": len([f for f in features if f.get("properties", {}).get("nf", False)]),
                    "fly": len([f for f in features if not f.get("properties", {}).get("nf", False)]),
                    "truncated": False,
                },
            }
        except Exception as e:
            print(f"[WARN] 加载建筑缓存失败: {e}")
    
    return None


def save_preset_entities(scene_id: str, entities: dict, orders: dict) -> bool:
    """
    保存调整后的预设场景实体和任务点到磁盘
    
    参数:
      scene_id : 预设场景 ID
      entities : 包含 depots, stations, trucks, drones 的字典
      orders   : 包含 static_orders 的字典
    
    返回: 成功返回 True，失败返回 False
    """
    if scene_id not in PRESET_SCENES:
        return False
    
    try:
        # 保存 entities.json
        entities_path = os.path.join(CACHE_BASE_DIR, "entities.json")
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] 实体配置已保存: {entities_path}")
        
        # 保存当前前端默认预设订单
        orders_path = os.path.join(CACHE_BASE_DIR, DEFAULT_PRESET_ORDERS_FILE)
        with open(orders_path, "w", encoding="utf-8") as f:
            json.dump(orders, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] 任务点配置已保存: {orders_path}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 保存预设场景失败: {e}")
        return False


def load_osm_from_cache(scene_id: str) -> tuple[str | None, dict | None]:
    """
    从预设场景缓存加载 OSM 网络数据。
    支持预定义的场景ID（如 'default_test_4x4km'）和动态UUID（如 'xxxxxxxx-xxxx-...'）。
    
    返回: (osm_xml_str, osm_geojson_dict) 或 (None, None)
    """
    # 确定场景缓存目录
    if scene_id in PRESET_SCENES:
        # 预定义场景：使用 CACHE_BASE_DIR
        cache_dir = CACHE_BASE_DIR
    else:
        # 动态UUID：尝试在 test_data/{scene_id} 目录中查找
        # 支持 test_data/{scene_id}/ 或 test_data/default_scene/ 等灵活路径
        alt_cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", scene_id)
        if os.path.exists(alt_cache_dir):
            cache_dir = alt_cache_dir
        else:
            # 也尝试用 default_scene 作为后备
            default_cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "default_scene")
            if os.path.exists(default_cache_dir):
                cache_dir = default_cache_dir
                print(f"[INFO] 未找到场景 '{scene_id}'，使用默认场景缓存")
            else:
                return None, None
    
    try:
        # 优先尝试加载 GeoJSON（更好用）
        osm_geojson_path = os.path.join(cache_dir, "osm_network.geojson")
        osm_geojson = None
        
        if os.path.exists(osm_geojson_path):
            with open(osm_geojson_path, "r", encoding="utf-8") as f:
                osm_geojson = json.load(f)
            print(f"[LOAD] 从缓存加载 OSM GeoJSON: {len(osm_geojson.get('features', []))} 个特征")
        
        # 也尝试加载原始 XML（备选）
        osm_xml = None
        osm_xml_path = os.path.join(cache_dir, "osm_network.xml")
        if os.path.exists(osm_xml_path):
            with open(osm_xml_path, "r", encoding="utf-8") as f:
                osm_xml = f.read()
            print(f"[LOAD] 从缓存加载 OSM XML: {len(osm_xml)} 字符")
        
        return osm_xml, osm_geojson
    except Exception as e:
        print(f"[ERROR] 加载 OSM 缓存失败: {e}")
        return None, None
