#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动 Phase 4 的 SUMO GUI 验证。

默认加载：
  backend/test_data/default_scene/sumo/phase4_truck_route/truck_route.sumocfg

用法：
  python backend/scripts/run_phase4_sumo_gui.py
  python backend/scripts/run_phase4_sumo_gui.py --regenerate
  python backend/scripts/run_phase4_sumo_gui.py --sumo-gui-bin /path/to/sumo-gui
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.export_sumo_truck_route import (
    export_phase4_truck_route,
    format_execution_route_id_sequence,
)
from training.sumo_net_normalizer import normalize_net_with_netconvert, resolve_netconvert


DEFAULT_SUMOCFG = (
    REPO_ROOT
    / "backend"
    / "test_data"
    / "default_scene"
    / "sumo"
    / "phase4_truck_route"
    / "truck_route.sumocfg"
)


DEFAULT_GUI_SETTINGS_NAME = "phase4_gui.view.xml"
DEFAULT_DEBUG_TRACE_NAME = "phase4_debug_trace.json"


def _resolve_sumo_gui(binary_override: str | None) -> str:
    if binary_override:
        return binary_override
    found = shutil.which("sumo-gui")
    if found:
        return found
    raise FileNotFoundError(
        "未找到 `sumo-gui`。请先安装 SUMO，或使用 "
        "`--sumo-gui-bin /path/to/sumo-gui` 指定可执行文件。"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 Phase 4 SUMO GUI 验证")
    parser.add_argument(
        "--sumocfg",
        type=str,
        default=str(DEFAULT_SUMOCFG),
        help="要加载的 .sumocfg 路径",
    )
    parser.add_argument(
        "--sumo-gui-bin",
        type=str,
        default="",
        help="sumo-gui 可执行文件路径；为空时自动从 PATH 查找",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="只加载场景，不自动开始仿真",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=250.0,
        help="GUI 自动运行时每步延迟毫秒数；越大越慢",
    )
    parser.add_argument(
        "--no-debug-print",
        action="store_true",
        help="不在启动前打印 Phase 4 调试摘要",
    )
    args = parser.parse_args(argv)

    result = export_phase4_truck_route()

    sumocfg = Path(args.sumocfg).resolve()
    if not sumocfg.is_file():
        raise FileNotFoundError(
            f"SUMO 配置文件不存在: {sumocfg}\n"
            "请先运行 `python backend/training/export_sumo_truck_route.py`。"
        )

    sumo_gui_bin = _resolve_sumo_gui(args.sumo_gui_bin or None)
    normalize_net_with_netconvert(
        sumocfg=sumocfg,
        netconvert_bin=resolve_netconvert(sumo_gui_bin),
    )
    if not args.no_debug_print:
        _print_debug_trace_summary(sumocfg.parent / DEFAULT_DEBUG_TRACE_NAME)
        print(f"truck_execution_route: {format_execution_route_id_sequence(result.execution_route)}")
    env = os.environ.copy()
    env.setdefault("QT_X11_NO_MITSHM", "1")
    command = [sumo_gui_bin, "-c", str(sumocfg), "--disable-textures"]
    gui_settings = sumocfg.parent / DEFAULT_GUI_SETTINGS_NAME
    if gui_settings.is_file():
        command.extend(["--gui-settings-file", str(gui_settings)])
    if not args.no_start:
        command.extend(["--start", "--delay", str(args.delay_ms)])
    subprocess.run(
        command,
        check=True,
        cwd=str(sumocfg.parent),
        env=env,
    )
    return 0


def _print_debug_trace_summary(debug_trace_path: Path) -> None:
    if not debug_trace_path.is_file():
        return
    payload = json.loads(debug_trace_path.read_text(encoding="utf-8"))
    print("[Phase4 Debug]")
    print(f"  truck_id: {payload.get('truck_id')}")
    print(f"  visited_order_ids: {payload.get('visited_order_ids', [])}")
    print(f"  visited_station_ids: {payload.get('visited_station_ids', [])}")
    print(f"  visited_fixed_node_ids: {payload.get('visited_fixed_node_ids', [])}")
    print(f"  inserted_fixed_nodes: {payload.get('inserted_fixed_nodes', [])}")
    print(f"  debug_json: {debug_trace_path}")


if __name__ == "__main__":
    raise SystemExit(main())
