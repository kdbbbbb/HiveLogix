#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run train_cmrappo periodic evaluation for a saved CMRAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.export_sumo_truck_route import export_phase4_truck_route
from training.policy_inference import load_trained_policy
from training.scene_loader import DEFAULT_CONFIG_PATH
from training.train_cmrappo import (
    _load_eval_config,
    _load_training_config,
    _run_periodic_evaluation,
    _strip_episode_details,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_phase4_truck_route(config_path=args.config_path)
    train_cfg = _load_training_config(args.config_path)
    eval_cfg = _load_eval_config(args.config_path)
    if args.benchmark_seed is not None:
        eval_cfg = replace(eval_cfg, benchmark_seed=int(args.benchmark_seed))
    if args.benchmark_only:
        eval_cfg = replace(
            eval_cfg,
            stochastic_arrival_rates=(),
            c_sensitive_enabled=False,
        )

    runtime = load_trained_policy(
        policy_path=args.policy_path,
        config_path=args.config_path,
        device=args.device,
    )
    checkpoint_step = _optional_int(runtime.checkpoint_meta.get("global_step")) or 0
    checkpoint_update = _optional_int(runtime.checkpoint_meta.get("update")) or 0

    report = _run_periodic_evaluation(
        scene_ctx=runtime.scene_ctx,
        config_path=args.config_path,
        model=runtime.model,
        tensorizer=runtime.tensorizer,
        critic_builder=runtime.critic_builder,
        critic_schema=runtime.critic_schema,
        policy_cfg=runtime.policy_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        device=runtime.device,
        update_idx=checkpoint_update,
        global_step=checkpoint_step,
    )
    report["checkpoint_eval_meta"] = {
        "policy_path": str(runtime.policy_path),
        "policy_checkpoint_meta": runtime.checkpoint_meta,
        "config_path": str(args.config_path),
        "benchmark_only": bool(args.benchmark_only),
    }

    _write_json(args.output_dir / "periodic_eval_report.json", report)
    _write_json(args.output_dir / "benchmark_report.json", report["benchmark"])
    benchmark_compact = _strip_episode_details(report["benchmark"])
    _write_json(args.output_dir / "benchmark_report_compact.json", benchmark_compact)

    if "c_sensitive" in report:
        _write_json(args.output_dir / "c_sensitive_report.json", report["c_sensitive"])
        _write_json(
            args.output_dir / "c_sensitive_report_compact.json",
            _strip_episode_details(report["c_sensitive"]),
        )

    if not args.benchmark_only:
        _write_json(args.output_dir / "stochastic_report.json", report["stochastic_report"])
        _write_json(
            args.output_dir / "stochastic_report_compact.json",
            {
                **report["stochastic_report"],
                "profiles": {
                    name: _strip_episode_details(payload)
                    for name, payload in report["stochastic"].items()
                },
            },
        )

    print(json.dumps(benchmark_compact, ensure_ascii=False, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a policy .pt checkpoint and call train_cmrappo._run_periodic_evaluation.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="要评估的 policy .pt checkpoint 路径。",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="训练/评估配置 YAML 路径，必须与 checkpoint 的 observation schema 匹配。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "backend" / "runs" / "policy_periodic_eval",
        help="输出目录；默认写入 backend/runs/policy_periodic_eval。",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=None,
        help="覆盖 data.benchmark_eval_seed；默认使用配置文件中的 benchmark seed。",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="只跑周期性评估里的 benchmark 分支，跳过 stochastic 和 c-sensitive。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备：auto/cpu/mps/cuda 等。",
    )
    raw = parser.parse_args(argv)
    raw.config_path = _resolve_path(raw.config_path)
    raw.policy_path = _resolve_path(raw.policy_path)
    raw.output_dir = _resolve_path(raw.output_dir)
    if not raw.policy_path.is_file():
        raise FileNotFoundError(f"policy checkpoint 不存在: {raw.policy_path}")
    return raw


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
