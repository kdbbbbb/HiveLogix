#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — WebSocket 遥测广播模块

职责：
  - 维护线程安全的 WebSocket 连接注册表 (_connections)
  - 建连后立即推送 FULL_SNAPSHOT（通过 build_full_snapshot 获取）
  - 提供 broadcast_tick() 供 SimEngine 后台线程调用
  - 对 Origin 进行白名单校验（仅允许本地 Vite dev server）

使用方式（在 simulation_bp.py 中）：
  from flask_sock import Sock
  from api.websockets.telemetry import register_route, broadcast_tick

  sock = Sock()
  register_route(sock)
"""

from __future__ import annotations

import json
import logging
import re
import threading

from flask import request

logger = logging.getLogger(__name__)

# ── 连接注册表（线程安全）────────────────────────────────────────────────────
_connections: set = set()
_conn_lock         = threading.Lock()

# ── Origin 白名单（本地开发：localhost/127.0.0.1 任意端口，或空 Origin）
_ALLOWED_ORIGIN = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")

# ── 外部注入：由 simulation_bp 在初始化后赋值 ─────────────────────────────────
# 类型：Callable[[], dict]，返回完整快照字典（含 type="FULL_SNAPSHOT" 包装）
_snapshot_builder = None  # type: ignore[assignment]


def set_snapshot_builder(builder) -> None:  # type: ignore[no-untyped-def]
    """注册快照构造函数，由 simulation_bp 在应用启动时调用。"""
    global _snapshot_builder
    _snapshot_builder = builder


def register_route(sock) -> None:  # type: ignore[no-untyped-def]
    """
    在给定的 flask-sock Sock 实例上注册 /api/ws/telemetry 端点。

    Args:
        sock: flask_sock.Sock 实例（已通过 sock.init_app(app) 绑定 Flask app）
    """

    @sock.route("/api/ws/telemetry")
    def telemetry_ws(ws):  # type: ignore[no-untyped-def]
        origin = request.headers.get("Origin", "")
        # 允许本地连接或空Origin（代理转发）
        if origin and not _ALLOWED_ORIGIN.match(origin):
            logger.warning("[telemetry_ws] 拒绝非法 Origin: %s", origin)
            ws.close(code=1008)  # Policy Violation
            return

        with _conn_lock:
            _connections.add(ws)
        logger.info("[telemetry_ws] 客户端接入（Origin: %s），当前连接数: %d", origin or "(empty)", len(_connections))

        try:
            # 建连后立即推送当前完整快照
            if _snapshot_builder is not None:
                try:
                    snapshot = _snapshot_builder()
                    snapshot_json = json.dumps(snapshot)
                    logger.info(f"[telemetry_ws] 推送 FULL_SNAPSHOT，大小: {len(snapshot_json)} bytes")
                    ws.send(snapshot_json)
                except Exception as exc:
                    logger.exception("[telemetry_ws] 推送 FULL_SNAPSHOT 失败: %s", exc)
            else:
                logger.warning("[telemetry_ws] _snapshot_builder 未设置，跳过 FULL_SNAPSHOT 推送")

            # 阻塞保持连接；客户端断开时 ws.receive() 会抛出异常
            while True:
                ws.receive()
        except Exception:
            pass  # 连接断开属正常情况，静默处理
        finally:
            with _conn_lock:
                _connections.discard(ws)
            logger.info("[telemetry_ws] 客户端断开，当前连接数: %d", len(_connections))


def broadcast_tick(payload: dict) -> None:
    """
    向所有已连接的客户端广播一帧 TICK 数据。

    由 SimEngine 后台线程调用，线程安全。
    发送失败的连接会被从注册表中移除。

    Args:
        payload: TICK 帧内容（不含 type 包装），由 SimEngine 构造
    """
    message = json.dumps({"type": "TICK", "payload": payload})
    dead: set = set()

    with _conn_lock:
        snapshot = set(_connections)  # 复制一份，减少持锁时间
        conn_count = len(snapshot)

    if conn_count == 0:
        logger.debug("[broadcast_tick] 无连接，跳过广播")
        return

    for ws in snapshot:
        try:
            ws.send(message)
        except Exception as e:
            logger.debug(f"[broadcast_tick] 发送失败: {e}")
            dead.add(ws)

    if dead:
        with _conn_lock:
            _connections.difference_update(dead)
        logger.debug("[broadcast_tick] 清理断开连接 %d 个", len(dead))
