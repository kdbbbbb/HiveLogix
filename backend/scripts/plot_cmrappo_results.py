#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Phase 7 CMRAPPO training/evaluation outputs into static PNG charts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "缺少 matplotlib，请先安装：\n"
            "  pip install matplotlib\n"
            "如果你在虚拟环境里训练，请在同一个环境里安装后再运行。"
        ) from exc
    return plt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - malformed input
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON") from exc
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    result: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        denom = min(idx + 1, window)
        result.append(running_sum / denom)
    return result


def _extract_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    series: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            series.append(math.nan)
        else:
            series.append(float(value))
    return series


def _plot_training(train_rows: list[dict[str, Any]], out_path: Path, plt: Any) -> None:
    steps = _extract_series(train_rows, "global_step")
    reward = _extract_series(train_rows, "reward_mean")
    returns = _extract_series(train_rows, "return_mean")
    value_loss = _extract_series(train_rows, "value_loss")
    entropy = _extract_series(train_rows, "entropy")
    approx_kl = _extract_series(train_rows, "approx_kl")

    window = max(5, min(25, len(train_rows) // 12 or 1))
    reward_ma = _moving_average(reward, window=window)
    returns_ma = _moving_average(returns, window=window)
    value_loss_ma = _moving_average(value_loss, window=window)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].plot(steps, reward, alpha=0.25, color="#4C78A8", label="reward_mean")
    axes[0].plot(steps, reward_ma, color="#4C78A8", linewidth=2, label=f"MA({window})")
    axes[0].plot(steps, returns_ma, color="#F58518", linewidth=2, label=f"return_mean MA({window})")
    axes[0].set_title("Training Reward / Return")
    axes[0].set_xlabel("global_step")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, value_loss, alpha=0.2, color="#E45756", label="value_loss")
    axes[1].plot(steps, value_loss_ma, color="#E45756", linewidth=2, label=f"MA({window})")
    axes[1].set_title("Value Loss")
    axes[1].set_xlabel("global_step")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(steps, entropy, color="#72B7B2", linewidth=2, label="entropy")
    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("global_step")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    axes[3].plot(steps, approx_kl, color="#54A24B", linewidth=2, label="approx_kl")
    axes[3].set_title("Approx KL")
    axes[3].set_xlabel("global_step")
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    fig.suptitle("CMRAPPO Training Overview", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _flatten_eval_rows(eval_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    flattened = {
        "benchmark": [],
        "low": [],
        "medium": [],
        "high": [],
    }
    for row in eval_rows:
        step = float(row["global_step"])
        benchmark = row.get("benchmark") or {}
        stochastic = row.get("stochastic") or {}
        flattened["benchmark"].append(
            {
                "global_step": step,
                "mean_total_reward": float(benchmark.get("mean_total_reward", math.nan)),
                "mean_delivery_count": float(benchmark.get("mean_delivery_count", math.nan)),
                "mean_on_time_rate": float(benchmark.get("mean_on_time_rate", math.nan)),
                "sum_timeout_order_count": float(benchmark.get("sum_timeout_order_count", math.nan)),
                "sum_fallback_count": float(benchmark.get("sum_fallback_count", math.nan)),
            }
        )
        for band in ("low", "medium", "high"):
            payload = stochastic.get(band) or {}
            flattened[band].append(
                {
                    "global_step": step,
                    "mean_total_reward": float(payload.get("mean_total_reward", math.nan)),
                    "mean_delivery_count": float(payload.get("mean_delivery_count", math.nan)),
                    "mean_on_time_rate": float(payload.get("mean_on_time_rate", math.nan)),
                    "sum_timeout_order_count": float(payload.get("sum_timeout_order_count", math.nan)),
                    "sum_fallback_count": float(payload.get("sum_fallback_count", math.nan)),
                }
            )
    return flattened


def _plot_eval_curves(eval_rows: list[dict[str, Any]], out_path: Path, plt: Any) -> None:
    flat = _flatten_eval_rows(eval_rows)
    palette = {
        "benchmark": "#4C78A8",
        "low": "#54A24B",
        "medium": "#F58518",
        "high": "#E45756",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    metrics = [
        ("mean_total_reward", "Eval Reward"),
        ("mean_delivery_count", "Mean Delivery Count"),
        ("sum_timeout_order_count", "Timeout Orders"),
        ("sum_fallback_count", "Fallback Count"),
    ]

    for axis, (metric_key, title) in zip(axes, metrics):
        for band, rows in flat.items():
            x = [row["global_step"] for row in rows]
            y = [row[metric_key] for row in rows]
            axis.plot(x, y, marker="o", linewidth=2, label=band, color=palette[band])
        axis.set_title(title)
        axis.set_xlabel("global_step")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle("CMRAPPO Evaluation Curves", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_final_snapshot(
    benchmark_report: dict[str, Any],
    stochastic_report: dict[str, Any],
    out_path: Path,
    plt: Any,
) -> None:
    profiles = stochastic_report.get("profiles") or {}
    labels = ["benchmark", "low", "medium", "high"]

    def _payload(label: str) -> dict[str, Any]:
        if label == "benchmark":
            return benchmark_report
        return profiles.get(label) or {}

    rewards = [float(_payload(label).get("mean_total_reward", math.nan)) for label in labels]
    deliveries = [float(_payload(label).get("mean_delivery_count", math.nan)) for label in labels]
    timeouts = [float(_payload(label).get("sum_timeout_order_count", math.nan)) for label in labels]
    fallbacks = [float(_payload(label).get("sum_fallback_count", math.nan)) for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].bar(labels, rewards, color=["#4C78A8", "#54A24B", "#F58518", "#E45756"])
    axes[0].set_title("Final Mean Reward")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, deliveries, color=["#4C78A8", "#54A24B", "#F58518", "#E45756"])
    axes[1].set_title("Final Mean Deliveries")
    axes[1].grid(axis="y", alpha=0.25)

    width = 0.35
    x = list(range(len(labels)))
    axes[2].bar([idx - width / 2 for idx in x], timeouts, width=width, label="timeouts", color="#E45756")
    axes[2].bar([idx + width / 2 for idx in x], fallbacks, width=width, label="fallbacks", color="#72B7B2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Final Timeouts / Fallbacks")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.suptitle("CMRAPPO Final Snapshot", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    benchmark_report: dict[str, Any],
    stochastic_report: dict[str, Any],
    out_path: Path,
) -> None:
    last_train = train_rows[-1]
    last_eval = eval_rows[-1]
    profiles = stochastic_report.get("profiles") or {}

    lines = [
        "# CMRAPPO Result Summary",
        "",
        "## Final Training Step",
        f"- update: {last_train.get('update')}",
        f"- global_step: {last_train.get('global_step')}",
        f"- reward_mean: {last_train.get('reward_mean'):.6f}",
        f"- return_mean: {last_train.get('return_mean'):.6f}",
        f"- value_loss: {last_train.get('value_loss'):.6f}",
        f"- entropy: {last_train.get('entropy'):.6f}",
        f"- approx_kl: {last_train.get('approx_kl'):.6f}",
        "",
        "## Final Eval Checkpoint",
        f"- update: {last_eval.get('update')}",
        f"- global_step: {last_eval.get('global_step')}",
        "",
        "## Benchmark",
        f"- mean_total_reward: {benchmark_report.get('mean_total_reward'):.6f}",
        f"- mean_delivery_count: {benchmark_report.get('mean_delivery_count'):.6f}",
        f"- sum_timeout_order_count: {benchmark_report.get('sum_timeout_order_count')}",
        f"- sum_fallback_count: {benchmark_report.get('sum_fallback_count')}",
        f"- mode_b_dispatch_ratio: {benchmark_report.get('mode_b_dispatch_ratio'):.6f}",
        f"- mode_c_dispatch_ratio: {benchmark_report.get('mode_c_dispatch_ratio'):.6f}",
        "",
        "## Stochastic Profiles",
    ]
    for band in ("low", "medium", "high"):
        payload = profiles.get(band) or {}
        lines.extend(
            [
                f"### {band}",
                f"- mean_total_reward: {payload.get('mean_total_reward'):.6f}",
                f"- mean_delivery_count: {payload.get('mean_delivery_count'):.6f}",
                f"- sum_timeout_order_count: {payload.get('sum_timeout_order_count')}",
                f"- sum_fallback_count: {payload.get('sum_fallback_count')}",
                f"- mode_b_dispatch_ratio: {payload.get('mode_b_dispatch_ratio'):.6f}",
                f"- mode_c_dispatch_ratio: {payload.get('mode_c_dispatch_ratio'):.6f}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CMRAPPO training outputs.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="训练输出目录，例如 backend/weights/rh_alns_cmrappo/phase7_20260428_run01",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="图表输出目录，默认写入 <run-dir>/plots",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = run_dir / "train_metrics.jsonl"
    eval_path = run_dir / "eval_metrics.jsonl"
    benchmark_path = run_dir / "benchmark_report.json"
    stochastic_path = run_dir / "stochastic_report.json"

    required_paths = [train_path, eval_path, benchmark_path, stochastic_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit("缺少输入文件：\n  " + "\n  ".join(missing))

    train_rows = _load_jsonl(train_path)
    eval_rows = _load_jsonl(eval_path)
    benchmark_report = _load_json(benchmark_path)
    stochastic_report = _load_json(stochastic_path)

    if not train_rows:
        raise SystemExit(f"{train_path} 为空，无法绘图")
    if not eval_rows:
        raise SystemExit(f"{eval_path} 为空，无法绘图")

    plt = _require_matplotlib()
    _plot_training(train_rows, out_dir / "training_overview.png", plt)
    _plot_eval_curves(eval_rows, out_dir / "evaluation_curves.png", plt)
    _plot_final_snapshot(benchmark_report, stochastic_report, out_dir / "final_snapshot.png", plt)
    _write_summary(
        train_rows=train_rows,
        eval_rows=eval_rows,
        benchmark_report=benchmark_report,
        stochastic_report=stochastic_report,
        out_path=out_dir / "summary.md",
    )

    print(f"图表已生成到: {out_dir}")
    print(f"- {out_dir / 'training_overview.png'}")
    print(f"- {out_dir / 'evaluation_curves.png'}")
    print(f"- {out_dir / 'final_snapshot.png'}")
    print(f"- {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
