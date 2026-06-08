#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景打包 Flask Blueprint

路由：
  POST /api/scene/prepare    — 打包场景上下文，返回轻量级 SceneContext meta
  GET  /api/scene/<scene_id> — 按 scene_id 取回已有场景（会话恢复用）
"""

import traceback

from flask import Blueprint, jsonify, request

from scene_service import get_scene_by_id, prepare_scene, load_preset_scene

scene_bp = Blueprint("scene", __name__)


@scene_bp.route("/prepare", methods=["POST"])
def api_prepare():
    """
    打包仿真场景上下文。

    请求体 (JSON):
      minx, miny, maxx, maxy : float  — 选区 WGS84 经纬度边界（必填）
      threshold               : float  — 飞行高度阈值，单位米（默认 120）
      height_column           : str    — Shapefile 高度字段名（可选）

    响应体 (JSON):
      SceneContext 对象，包含 scene_id、sel_bounds、meta、road_network
      不包含 buildings GeoJSON（由前端另行请求 /api/geo/query 加载）
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "请求体不能为空，需要 JSON 格式"}), 400

    required_keys = ("minx", "miny", "maxx", "maxy")
    missing = [k for k in required_keys if k not in body]
    if missing:
        return jsonify({"error": f"缺少必要参数: {', '.join(missing)}"}), 400

    try:
        sel_bounds = {k: float(body[k]) for k in required_keys}
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"边界参数格式错误: {exc}"}), 400

    # 基本合法性校验
    if sel_bounds["minx"] >= sel_bounds["maxx"] or sel_bounds["miny"] >= sel_bounds["maxy"]:
        return jsonify({"error": "选区边界无效：min 值须小于 max 值"}), 400

    threshold    = float(body.get("threshold", 120))
    height_col   = body.get("height_column") or None

    try:
        ctx = prepare_scene(sel_bounds, threshold, height_col)
        return jsonify(ctx)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"场景构建失败: {str(exc)}"}), 500


@scene_bp.route("/<scene_id>", methods=["GET"])
def api_get_scene(scene_id: str):
    """
    按 scene_id 检索已缓存的场景（用于页面刷新后的会话恢复）。

    路径参数:
      scene_id : UUID 字符串

    响应:
      200 — SceneContext
      404 — scene_id 不存在或已失效（服务重启后缓存清空）
    """
    ctx = get_scene_by_id(scene_id)
    if ctx is None:
        return jsonify({"error": "scene_id 不存在或服务重启后已失效"}), 404
    return jsonify(ctx)


@scene_bp.route("/preset/<preset_id>", methods=["GET"])
def api_get_preset(preset_id: str):
    """
    加载预设场景（从磁盘缓存预生成的场景）。
    
    路径参数:
      preset_id : 预设场景 ID，如 'default_test_4x4km'
    
    响应:
      200 — SceneContext
      404 — 预设场景不存在或加载失败
    """
    ctx = load_preset_scene(preset_id)
    if ctx is None:
        return jsonify({"error": f"预设场景 '{preset_id}' 不存在或加载失败"}), 404
    return jsonify(ctx)
