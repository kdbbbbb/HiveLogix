#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 主 Flask 应用入口

架构说明：
  - 各功能模块以 Blueprint 形式注册，挂载在独立 URL 前缀下
  - Geo 模块：  /api/geo/*
  - 未来扩展：  /api/dispatch/*, /api/orders/*, /api/fleet/*, ...

启动方式：
  cd backend
  python app.py          → http://localhost:8000

独立调试 Geo 模块：
  cd backend/environment/geo
  python app.py          → http://localhost:5000
"""

import os
import sys
import re
import logging

# ── 控制台日志（显示所有模块的 INFO/DEBUG 输出，方便调试）───────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 降低 Werkzeug 路由日志噪声（只保留 WARNING 以上）
logging.getLogger("werkzeug").setLevel(logging.WARNING)
# 地图加载阶段会触发 geopandas/fiona/GDAL 大量逐要素 DEBUG 日志；单独压低第三方噪声。
logging.getLogger("fiona").setLevel(logging.WARNING)
logging.getLogger("fiona.collection").setLevel(logging.WARNING)
logging.getLogger("fiona.env").setLevel(logging.WARNING)
logging.getLogger("fiona.ogrext").setLevel(logging.WARNING)
logging.getLogger("geopandas").setLevel(logging.WARNING)

from flask import Flask, jsonify
from flask_cors import CORS

# ── 模块路径注入（保持子模块内部的平坦式 import 不变）─────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GEO_DIR    = os.path.join(BASE_DIR, "environment", "geo")
SCENE_DIR  = os.path.join(BASE_DIR, "environment", "scene")
STATE_DIR  = os.path.join(BASE_DIR, "environment", "state")
ROUTES_DIR = os.path.join(BASE_DIR, "api", "routes")
sys.path.insert(0, BASE_DIR)      # 使 utils.coord_utils 等顶层包可导入
sys.path.insert(0, GEO_DIR)
sys.path.insert(0, SCENE_DIR)
sys.path.insert(0, STATE_DIR)
sys.path.insert(0, ROUTES_DIR)

# ── Blueprint 导入（在 sys.path 注入之后）─────────────────────────────────────
from geo_blueprint     import geo_bp                # noqa: E402
from data_loader       import load_shapefile_async  # noqa: E402
from scene_blueprint   import scene_bp              # noqa: E402
from simulation_bp     import sim_bp, sock          # noqa: E402

# ── 应用初始化 ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
sock.init_app(app)   # flask-sock 初始化（WebSocket 支持）

CORS(app, resources={r"/api/*": {
    "origins": re.compile(r"^http://(localhost|127\.0\.0\.1):517\d$"),
}})

# ── Blueprint 注册 ─────────────────────────────────────────────────────────────
app.register_blueprint(geo_bp,   url_prefix="/api/geo")
app.register_blueprint(scene_bp, url_prefix="/api/scene")
app.register_blueprint(sim_bp,   url_prefix="/api/sim")


# ── 通用端点 ───────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health_check():
    """服务健康探针。"""
    return jsonify({"status": "ok", "service": "HiveLogix"})


# ── 入口 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = 8000
    load_shapefile_async()
    print(f"[INFO] HiveLogix 主服务启动: http://localhost:{port}")
    print(f"[INFO] Geo API:   http://localhost:{port}/api/geo/status")
    print(f"[INFO] 健康检查:  http://localhost:{port}/api/health")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
