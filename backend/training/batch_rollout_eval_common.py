#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for batch rollout baseline evaluation scripts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.actions import DispatchAction, EnvAction
from training.batch_matching_teacher import build_batch_matching_teacher_labels
from training.env_adapter import TrainingEnvAdapter
from training.export_sumo_truck_route import export_phase4_truck_route
from training.order_source_adapter import OrderSourceConfig, OrderSourceMode, build_order_source
from training.scene_loader import DEFAULT_CONFIG_PATH, load_default_scene
from training.train_cmrappo import _summarize_episode_records


BatchPolicyFn = Callable[
    [
        tuple[Any, ...],
        Mapping[str, Any],
        random.Random,
    ],
    Mapping[str, EnvAction],
]


@dataclass(frozen=True)
class EvalArgs:
    config_path: Path
    output_dir: Path
    order_source_mode: OrderSourceMode
    seeds: tuple[int, ...]
    poisson_arrival_rate: float | None
    max_decision_batches: int
    reward_scale: float
    random_seed: int


def build_arg_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="训练配置 YAML 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "backend" / "runs" / "batch_rollout_smoke",
        help="输出目录；默认写入 backend/runs/batch_rollout_smoke。",
    )
    parser.add_argument(
        "--order-source-mode",
        choices=[OrderSourceMode.POISSON.value, OrderSourceMode.BENCHMARK.value],
        default=OrderSourceMode.POISSON.value,
        help="评估订单源模式。",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="逗号分隔 seed 列表，例如 20260424,20260425。",
    )
    parser.add_argument(
        "--seeds-from-metrics",
        type=Path,
        default=None,
        help="从已有 episode_metrics.jsonl 读取 order_source_seed。",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="未提供 seeds 时使用的起始 seed。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="提供 seeds 时用于截断，0 表示全部；默认 3，用于小规模对比。",
    )
    parser.add_argument(
        "--poisson-arrival-rate",
        type=float,
        default=None,
        help="poisson 模式下覆盖订单到达率；默认读取配置。",
    )
    parser.add_argument(
        "--max-decision-batches",
        type=int,
        default=32,
        help="每个 episode 最多执行多少个同刻 decision batch；默认 32，用于小规模对比；0 表示直到 done。",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="汇总 total_reward 时使用的 reward scale。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260520,
        help="随机 baseline 的动作随机种子；teacher 脚本仅用于可复现实验元数据。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test 模式：若未显式设置，则只跑 1 个 episode、每个最多 8 个 decision batch。",
    )
    return parser


def parse_common_args(argv: Sequence[str] | None, *, description: str) -> EvalArgs:
    parser = build_arg_parser(description=description)
    raw = parser.parse_args(argv)
    config_path = _resolve_path(raw.config_path)
    output_dir = _resolve_path(raw.output_dir)
    episodes = int(raw.episodes)
    max_decision_batches = int(raw.max_decision_batches)
    if bool(raw.quick):
        if episodes == 3:
            episodes = 1
        if max_decision_batches == 32:
            max_decision_batches = 8
    if episodes < 0:
        raise ValueError("--episodes 不能为负数")
    if max_decision_batches < 0:
        raise ValueError("--max-decision-batches 不能为负数")
    if float(raw.reward_scale) <= 0.0:
        raise ValueError("--reward-scale 必须为正数")
    seeds = _resolve_seeds(
        config_path=config_path,
        explicit_seeds=str(raw.seeds),
        seeds_from_metrics=raw.seeds_from_metrics,
        seed_base=raw.seed_base,
        episodes=episodes,
    )
    return EvalArgs(
        config_path=config_path,
        output_dir=output_dir,
        order_source_mode=OrderSourceMode(str(raw.order_source_mode)),
        seeds=seeds,
        poisson_arrival_rate=(
            None if raw.poisson_arrival_rate is None else float(raw.poisson_arrival_rate)
        ),
        max_decision_batches=max_decision_batches,
        reward_scale=float(raw.reward_scale),
        random_seed=int(raw.random_seed),
    )


def choose_teacher_actions(
    decision_batch: tuple[Any, ...],
    candidate_outputs_by_drone: Mapping[str, Any],
    _rng: random.Random,
) -> Mapping[str, EnvAction]:
    teacher_result = build_batch_matching_teacher_labels(
        decision_contexts=tuple(decision_batch),
        candidate_outputs_by_drone=candidate_outputs_by_drone,
    )
    return dict(teacher_result.actions_by_drone)


def choose_random_legal_actions(
    decision_batch: tuple[Any, ...],
    _candidate_outputs_by_drone: Mapping[str, Any],
    rng: random.Random,
) -> Mapping[str, EnvAction]:
    selected_orders: set[str] = set()
    actions_by_drone: dict[str, EnvAction] = {}
    for context in decision_batch:
        legal_actions: list[EnvAction] = []
        for action in context.action_lookup:
            if isinstance(action, DispatchAction) and str(action.order_id) in selected_orders:
                continue
            legal_actions.append(action)
        if not legal_actions:
            raise RuntimeError(f"UAV {context.deciding_drone_id} 没有可选合法动作")
        selected = rng.choice(legal_actions)
        actions_by_drone[str(context.deciding_drone_id)] = selected
        if isinstance(selected, DispatchAction):
            selected_orders.add(str(selected.order_id))
    return actions_by_drone


def choose_random_mode_b_actions(
    decision_batch: tuple[Any, ...],
    _candidate_outputs_by_drone: Mapping[str, Any],
    rng: random.Random,
) -> Mapping[str, EnvAction]:
    selected_orders: set[str] = set()
    actions_by_drone: dict[str, EnvAction] = {}
    for context in decision_batch:
        mode_b_actions: list[DispatchAction] = []
        wait_actions: list[EnvAction] = []
        for action in context.action_lookup:
            if isinstance(action, DispatchAction):
                if str(action.mode) != "B":
                    continue
                if str(action.order_id) in selected_orders:
                    continue
                mode_b_actions.append(action)
            else:
                wait_actions.append(action)
        if mode_b_actions:
            selected: EnvAction = rng.choice(mode_b_actions)
            selected_orders.add(str(selected.order_id))
        elif wait_actions:
            selected = wait_actions[0]
        else:
            raise RuntimeError(f"UAV {context.deciding_drone_id} 没有可选 Mode B 或 WAIT 动作")
        actions_by_drone[str(context.deciding_drone_id)] = selected
    return actions_by_drone


def run_batch_rollout_eval(
    *,
    args: EvalArgs,
    strategy_name: str,
    choose_actions: BatchPolicyFn,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_phase4_truck_route(config_path=args.config_path)
    scene_ctx = load_default_scene(args.config_path)
    rng = random.Random(args.random_seed)

    episode_records: list[dict[str, Any]] = []
    episode_metrics_path = args.output_dir / f"{strategy_name}_episode_metrics.jsonl"
    with episode_metrics_path.open("w", encoding="utf-8") as fh:
        for episode_id, seed in enumerate(args.seeds):
            order_source = _build_eval_order_source(
                scene_ctx=scene_ctx,
                config_path=args.config_path,
                mode=args.order_source_mode,
                seed=seed,
                poisson_arrival_rate=args.poisson_arrival_rate,
            )
            record = _run_one_episode(
                scene_ctx=scene_ctx,
                config_path=args.config_path,
                order_source=order_source,
                episode_id=episode_id,
                phase=strategy_name,
                choose_actions=choose_actions,
                rng=rng,
                max_decision_batches=args.max_decision_batches,
                reward_scale=args.reward_scale,
            )
            episode_records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = _summarize_episode_records(
        split=strategy_name,
        order_source_mode=args.order_source_mode.value,
        episodes=episode_records,
        update_idx=0,
        global_step=0,
        generated_at=generated_at,
        extra={
            "strategy": strategy_name,
            "config_path": str(args.config_path),
            "seed_count": len(args.seeds),
            "seeds": list(args.seeds),
            "poisson_arrival_rate": args.poisson_arrival_rate,
            "max_decision_batches": args.max_decision_batches,
            "reward_scale": args.reward_scale,
            "random_seed": args.random_seed,
            "episode_metrics_path": str(episode_metrics_path),
        },
    )
    _add_script_batch_metrics(summary, episode_records)
    summary_path = args.output_dir / f"{strategy_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact_summary = {k: v for k, v in summary.items() if k != "episodes"}
    compact_path = args.output_dir / f"{strategy_name}_summary_compact.json"
    compact_path.write_text(
        json.dumps(compact_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    keys = [
        "strategy",
        "episode_count",
        "mean_total_reward",
        "mean_episode_length_decision_batches",
        "mean_delivery_count",
        "mean_on_time_rate",
        "mean_avg_order_delay_min",
        "weighted_avg_order_delay_min",
        "mean_tardiness_sec",
        "max_tardiness_sec",
        "mean_completion_rate",
        "mean_required_on_time_rate",
        "sum_required_primary_order_count",
        "sum_late_delivery_count",
        "sum_timeout_order_count",
        "sum_unserved_primary_order_count",
        "sum_fallback_count",
        "sum_hard_failure_count",
        "sum_mode_c_selected_count",
        "sum_mode_c_success_count",
        "sum_episode_uav_energy_reward_penalty",
        "sum_episode_uav_energy_ratio_sum",
        "sum_episode_uav_energy_penalty_events",
        "script_selected_C_given_legal_C",
        "episode_metrics_path",
    ]
    payload = {key: summary.get(key) for key in keys if key in summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_one_episode(
    *,
    scene_ctx: Any,
    config_path: Path,
    order_source: OrderSourceConfig,
    episode_id: int,
    phase: str,
    choose_actions: BatchPolicyFn,
    rng: random.Random,
    max_decision_batches: int,
    reward_scale: float,
) -> dict[str, Any]:
    env = TrainingEnvAdapter(
        scene_ctx=scene_ctx,
        order_source=order_source,
        config_path=config_path,
    )
    result = env.reset()
    total_reward = 0.0
    decision_count = 0
    decision_batch_count = 0
    script_legal_c_count = 0
    script_selected_c_given_legal_c_count = 0

    while not result.done:
        if max_decision_batches > 0 and decision_batch_count >= max_decision_batches:
            break
        decision_batch = env.peek_current_decision_batch()
        if not decision_batch:
            if env.is_done():
                break
            raise RuntimeError("batch rollout eval 遇到非终止且无 decision batch 的状态")

        candidate_outputs_by_drone = {
            str(context.deciding_drone_id): env.build_candidate_output(
                context,
                last_seen_plan_version=int(context.coarse_plan.plan_version),
            )
            for context in decision_batch
        }
        actions_by_drone = dict(
            choose_actions(decision_batch, candidate_outputs_by_drone, rng)
        )
        for context in decision_batch:
            drone_id = str(context.deciding_drone_id)
            candidate_out = candidate_outputs_by_drone[drone_id]
            action = actions_by_drone[drone_id]
            has_legal_c = _candidate_has_legal_mode_c(candidate_out)
            if has_legal_c:
                script_legal_c_count += 1
                if isinstance(action, DispatchAction) and str(action.mode) == "C":
                    script_selected_c_given_legal_c_count += 1

        result = env.apply_decision_batch(actions_by_drone)
        total_reward += float(result.reward) * float(reward_scale)
        decision_count += len(decision_batch)
        decision_batch_count += 1

    if env.is_done():
        terminal_reward = sum(float(v) for v in env.consume_terminal_agent_costs().values())
        total_reward += terminal_reward * float(reward_scale)

    snapshot = env.build_episode_metrics_snapshot()
    payload: dict[str, Any] = {
        "phase": phase,
        "episode_id": int(episode_id),
        "order_source_mode": str(order_source.mode.value),
        "order_source_seed": int(env.current_episode_order_source_seed()),
        "total_reward": float(total_reward),
        "episode_length_decisions": int(decision_count),
        "episode_length_decision_batches": int(decision_batch_count),
        "global_step_start": None,
        "global_step_end": None,
        "update_start": None,
        "update_end": None,
    }
    payload.update(snapshot)
    payload["script_legal_C_count"] = int(script_legal_c_count)
    payload["script_selected_C_given_legal_C_count"] = int(
        script_selected_c_given_legal_c_count
    )
    payload["script_selected_C_given_legal_C"] = (
        float(script_selected_c_given_legal_c_count) / float(script_legal_c_count)
        if script_legal_c_count
        else 0.0
    )
    return payload


def _build_eval_order_source(
    *,
    scene_ctx: Any,
    config_path: Path,
    mode: OrderSourceMode,
    seed: int,
    poisson_arrival_rate: float | None,
) -> OrderSourceConfig:
    overrides: dict[str, Any] = {}
    if mode == OrderSourceMode.POISSON and poisson_arrival_rate is not None:
        overrides["poisson_arrival_rate"] = float(poisson_arrival_rate)
    return build_order_source(
        scene_ctx,
        mode=mode,
        seed=int(seed),
        overrides=overrides,
        config_path=config_path,
    )


def _candidate_has_legal_mode_c(candidate_out: Any) -> bool:
    return any(
        isinstance(action, DispatchAction) and str(action.mode) == "C"
        for action in candidate_out.resolved_action_lookup.dispatch_actions.values()
    )


def _add_script_batch_metrics(
    summary: dict[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    legal_c = int(sum(int(item.get("script_legal_C_count", 0)) for item in episodes))
    selected_c = int(
        sum(int(item.get("script_selected_C_given_legal_C_count", 0)) for item in episodes)
    )
    decision_batches = int(
        sum(int(item.get("episode_length_decision_batches", 0)) for item in episodes)
    )
    summary["script_sum_legal_C_count"] = legal_c
    summary["script_sum_selected_C_given_legal_C_count"] = selected_c
    summary["script_selected_C_given_legal_C"] = (
        float(selected_c) / float(legal_c) if legal_c else 0.0
    )
    summary["script_sum_decision_batches"] = decision_batches


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _resolve_seeds(
    *,
    config_path: Path,
    explicit_seeds: str,
    seeds_from_metrics: Path | None,
    seed_base: int | None,
    episodes: int,
) -> tuple[int, ...]:
    if explicit_seeds.strip():
        seeds = tuple(
            int(item.strip()) for item in explicit_seeds.split(",") if item.strip()
        )
        if episodes > 0:
            seeds = tuple(seeds[:episodes])
    elif seeds_from_metrics is not None:
        seeds = _read_seeds_from_metrics(_resolve_path(seeds_from_metrics))
        if episodes > 0:
            seeds = tuple(seeds[:episodes])
    else:
        base = seed_base if seed_base is not None else _default_seed_base(config_path)
        count = episodes if episodes > 0 else 1
        seeds = tuple(int(base) + idx for idx in range(count))
    if not seeds:
        raise ValueError("seed 列表不能为空")
    return tuple(seeds)


def _read_seeds_from_metrics(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"episode metrics 文件不存在: {path}")
    seeds: list[int] = []
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        seed = int(payload["order_source_seed"])
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return tuple(seeds)


def _default_seed_base(config_path: Path) -> int:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("缺少 PyYAML，无法读取训练配置") from exc
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    data = raw.get("data", {}) if isinstance(raw, Mapping) else {}
    return int(data.get("stochastic_eval_seed_base", data.get("benchmark_eval_seed", 20260425)))
