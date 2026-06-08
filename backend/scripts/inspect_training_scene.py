#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 场景装载自测脚本。

运行方式：
  python backend/scripts/inspect_training_scene.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.scene_loader import load_default_scene


def main() -> int:
    context = load_default_scene()
    summary = context.build_summary()

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    expected = {
        "depots": 1,
        "stations": 10,
        "trucks": 1,
        "drones": 12,
        "static_orders": 10,
        "dynamic_orders": 10,
    }
    for key, expected_value in expected.items():
        actual_value = summary[key]
        if actual_value != expected_value:
            raise AssertionError(
                f"{key} 验收失败: expected={expected_value}, actual={actual_value}"
            )

    print("Phase 2 scene_loader self-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
