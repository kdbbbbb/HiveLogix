#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment.scene — 仿真场景打包模块

职责：
  - 将 geo 模块产出的路网信息与坐标元数据打包为统一的 SceneContext
  - 向前端提供轻量级 meta 接口（不包含大体积 GeoJSON）
  - 维护场景缓存（支持幂等调用）

对外暴露：
  scene_blueprint.scene_bp  — Flask Blueprint，挂载到 /api/scene
"""
