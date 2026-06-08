#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot warm-start-friendly convergence comparisons for two CMRAPPO runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_WARM_RUN = (
    "backend/weights/rh_alns_cmrappo/"
    "run_warm_start_overnight_20k_default_poisson_energy_teacher"
)
DEFAULT_NO_WARM_RUN = (
    "backend/weights/rh_alns_cmrappo/"
    "run_no_warm_start_overnight_20k_default_poisson_energy_teacher"
)
DEFAULT_OUTPUT_DIR = "backend/runs/warmup_convergence_advantage_plots"

BUSINESS_METRICS = [
    {
        "key": "total_reward",
        "title": "Episode Total Reward",
        "ylabel": "reward",
        "higher_is_better": True,
        "threshold": None,
        "threshold_label": "",
        "explanation": (
            "episode 总回报，来自环境 reward 累计；它综合反映完成订单、等待、能耗、"
            "超时/失败等训练目标。warm start 的优势是早期回报更高，说明初始化策略已经"
            "接近可用区域。"
        ),
    },
    {
        "key": "delivery_count",
        "title": "Delivered Primary Orders",
        "ylabel": "orders",
        "higher_is_better": True,
        "threshold": 106.0,
        "threshold_label": "10-episode avg >= 106",
        "explanation": (
            "delivery_count 是 episode 内实际完成配送的 UAV primary 订单数。"
            "这是最直接的业务吞吐指标；warm start 在前 10 个 episode 就达到高完成量平台。"
        ),
    },
    {
        "key": "completion_rate",
        "title": "Completion Rate",
        "ylabel": "rate",
        "higher_is_better": True,
        "threshold": 0.98,
        "threshold_label": "10-episode avg >= 0.98",
        "explanation": (
            "completion_rate = delivered / required，其中 required 包含已完成、超时、"
            "仍 pending 和 assigned 的 primary 订单。它比 delivery_count 更能反映是否漏单。"
        ),
    },
    {
        "key": "required_on_time_rate",
        "title": "Required On-Time Rate",
        "ylabel": "rate",
        "higher_is_better": True,
        "threshold": 0.96,
        "threshold_label": "10-episode avg >= 0.96",
        "explanation": (
            "required_on_time_rate = on_time_delivery_count / required_primary_order_count，"
            "把未服务订单也计入分母，比 on_time_rate 更严格。warm start 在该指标上更快稳定。"
        ),
    },
    {
        "key": "unserved_primary_order_count",
        "title": "Unserved Primary Orders",
        "ylabel": "orders",
        "higher_is_better": False,
        "threshold": 2.0,
        "threshold_label": "10-episode avg <= 2",
        "explanation": (
            "unserved_primary_order_count 是 episode 结束时仍 pending 或 assigned 的 primary 订单。"
            "该指标越低越好；warm start 的早期优势主要体现在显著减少未服务订单。"
        ),
    },
]

PPO_METRICS = [
    {
        "key": "return_mean",
        "title": "PPO Return Mean",
        "ylabel": "return_mean",
        "explanation": (
            "return_mean 来自 PPO rollout batch 的回报均值。它不是最终业务 KPI，"
            "但可以观察 PPO 阶段是否继续提升；warm start 起点更高，后期 return_mean 也更高。"
        ),
    },
    {
        "key": "reward_mean",
        "title": "PPO Reward Mean",
        "ylabel": "reward_mean",
        "explanation": (
            "reward_mean 是 PPO rollout batch 的即时 reward 均值。no-warm 后期在该指标更高，"
            "因此这里只作为参照曲线，不把它作为 warm start 优势指标。"
        ),
    },
]


def _require_matplotlib() -> Any:
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


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else math.nan


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(int(window), len(values)))
    result: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        result.append(running_sum / float(min(idx + 1, window)))
    return result


def _rolling_windows(values: list[float], window: int) -> list[float]:
    if len(values) < window:
        return []
    return [_mean(values[idx : idx + window]) for idx in range(len(values) - window + 1)]


def _first_threshold_index(
    values: list[float],
    *,
    window: int,
    threshold: float,
    higher_is_better: bool,
) -> int | None:
    for idx, value in enumerate(_rolling_windows(values, window)):
        if higher_is_better and value >= threshold:
            return idx
        if not higher_is_better and value <= threshold:
            return idx
    return None


def _dedupe_ppo_rows(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ppo_rows = [row for row in train_rows if row.get("phase", "ppo") == "ppo"]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in ppo_rows:
        key = (int(row.get("update", -1)), int(row.get("global_step", -1)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return sorted(deduped, key=lambda item: (int(item["update"]), int(item["global_step"])))


def _load_run(run_dir: Path) -> dict[str, Any]:
    train_path = run_dir / "train_metrics.jsonl"
    episode_path = run_dir / "episode_metrics.jsonl"
    missing = [str(path) for path in (train_path, episode_path) if not path.is_file()]
    if missing:
        raise SystemExit("缺少输入文件：\n  " + "\n  ".join(missing))
    train_rows = _load_jsonl(train_path)
    episode_rows = _load_jsonl(episode_path)
    return {
        "run_dir": str(run_dir),
        "train_rows": train_rows,
        "ppo_rows": _dedupe_ppo_rows(train_rows),
        "episode_rows": episode_rows,
    }


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _seed_aligned_rows(
    warm_episodes: list[dict[str, Any]],
    no_warm_episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warm_by_seed = {int(row["order_source_seed"]): row for row in warm_episodes}
    no_warm_by_seed = {int(row["order_source_seed"]): row for row in no_warm_episodes}
    common_seeds = sorted(set(warm_by_seed) & set(no_warm_by_seed))
    return [warm_by_seed[seed] for seed in common_seeds], [no_warm_by_seed[seed] for seed in common_seeds]


def _plot_business_curves(
    *,
    warm_rows: list[dict[str, Any]],
    no_warm_rows: list[dict[str, Any]],
    window: int,
    out_path: Path,
    plt: Any,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    axes = axes.flatten()
    x_warm = list(range(len(warm_rows)))
    x_no_warm = list(range(len(no_warm_rows)))

    for axis, metric in zip(axes, BUSINESS_METRICS):
        warm_values = _series(warm_rows, str(metric["key"]))
        no_warm_values = _series(no_warm_rows, str(metric["key"]))
        warm_ma = _moving_average(warm_values, window)
        no_warm_ma = _moving_average(no_warm_values, window)

        axis.plot(x_warm, warm_ma, color="#4C78A8", linewidth=2.4, label="warm start")
        axis.plot(x_no_warm, no_warm_ma, color="#F58518", linewidth=2.4, label="no warm start")
        threshold = metric.get("threshold")
        if threshold is not None:
            axis.axhline(float(threshold), color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
        axis.set_title(f"{metric['title']} - trailing MA({window})")
        axis.set_xlabel("episode index")
        axis.set_ylabel(str(metric["ylabel"]))
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].axis("off")
    fig.suptitle("Warm Start Advantage: Business Convergence", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_aligned_advantage(
    *,
    warm_rows: list[dict[str, Any]],
    no_warm_rows: list[dict[str, Any]],
    window: int,
    out_path: Path,
    plt: Any,
) -> None:
    plotted_metrics = [
        ("delivery_count", "Warm - No Warm Delivered Orders", True),
        ("completion_rate", "Warm - No Warm Completion Rate", True),
        ("required_on_time_rate", "Warm - No Warm Required On-Time Rate", True),
        ("unserved_primary_order_count", "No Warm - Warm Unserved Orders", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    x = list(range(max(0, len(warm_rows) - window + 1)))

    for axis, (key, title, higher_is_better) in zip(axes, plotted_metrics):
        warm_values = _rolling_windows(_series(warm_rows, key), window)
        no_warm_values = _rolling_windows(_series(no_warm_rows, key), window)
        if higher_is_better:
            advantage = [warm - base for warm, base in zip(warm_values, no_warm_values)]
        else:
            advantage = [base - warm for warm, base in zip(warm_values, no_warm_values)]
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.bar(x, advantage, color=["#4C78A8" if value >= 0 else "#E45756" for value in advantage])
        axis.set_title(f"{title} - aligned rolling window({window})")
        axis.set_xlabel("aligned episode window start")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Seed-Aligned Warm Start Advantage", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_thresholds(
    *,
    summary: dict[str, Any],
    out_path: Path,
    plt: Any,
) -> None:
    rows = [
        item
        for item in summary["thresholds"]
        if item["warm_first_window_start"] is not None or item["no_warm_first_window_start"] is not None
    ]
    labels = [item["label"] for item in rows]
    warm_values = [
        item["warm_first_window_start"] if item["warm_first_window_start"] is not None else math.nan
        for item in rows
    ]
    no_warm_values = [
        item["no_warm_first_window_start"] if item["no_warm_first_window_start"] is not None else math.nan
        for item in rows
    ]

    fig, axis = plt.subplots(figsize=(13, 6))
    x = list(range(len(rows)))
    width = 0.36
    axis.bar([idx - width / 2 for idx in x], warm_values, width=width, color="#4C78A8", label="warm start")
    axis.bar([idx + width / 2 for idx in x], no_warm_values, width=width, color="#F58518", label="no warm start")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("first rolling-window start episode, lower is better")
    axis.set_title("Episodes Needed to Reach Business Thresholds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_ppo_curves(
    *,
    warm_rows: list[dict[str, Any]],
    no_warm_rows: list[dict[str, Any]],
    out_path: Path,
    plt: Any,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for axis, metric in zip(axes, PPO_METRICS):
        key = str(metric["key"])
        axis.plot(
            _series(warm_rows, "global_step"),
            _series(warm_rows, key),
            color="#4C78A8",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="warm start",
        )
        axis.plot(
            _series(no_warm_rows, "global_step"),
            _series(no_warm_rows, key),
            color="#F58518",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="no warm start",
        )
        axis.set_title(str(metric["title"]))
        axis.set_xlabel("global_step")
        axis.set_ylabel(str(metric["ylabel"]))
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("PPO Phase Reference Curves", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _build_summary(
    *,
    warm: dict[str, Any],
    no_warm: dict[str, Any],
    aligned_warm_rows: list[dict[str, Any]],
    aligned_no_warm_rows: list[dict[str, Any]],
    window: int,
) -> dict[str, Any]:
    warm_episodes = warm["episode_rows"]
    no_warm_episodes = no_warm["episode_rows"]
    summary: dict[str, Any] = {
        "window": int(window),
        "warm_run_dir": warm["run_dir"],
        "no_warm_run_dir": no_warm["run_dir"],
        "warm_episode_count": len(warm_episodes),
        "no_warm_episode_count": len(no_warm_episodes),
        "aligned_episode_count": len(aligned_warm_rows),
        "metrics": [],
        "thresholds": [],
        "ppo_reference": [],
    }

    for metric in BUSINESS_METRICS:
        key = str(metric["key"])
        higher_is_better = bool(metric["higher_is_better"])
        warm_values = _series(warm_episodes, key)
        no_warm_values = _series(no_warm_episodes, key)
        aligned_warm_values = _series(aligned_warm_rows, key)
        aligned_no_warm_values = _series(aligned_no_warm_rows, key)
        aligned_delta = _mean(
            [
                warm_value - no_warm_value if higher_is_better else no_warm_value - warm_value
                for warm_value, no_warm_value in zip(aligned_warm_values, aligned_no_warm_values)
            ]
        )
        first_warm = _mean(warm_values[:window])
        first_no_warm = _mean(no_warm_values[:window])
        last_warm = _mean(warm_values[-window:])
        last_no_warm = _mean(no_warm_values[-window:])
        threshold = metric.get("threshold")
        if threshold is not None:
            summary["thresholds"].append(
                {
                    "metric": key,
                    "label": str(metric["threshold_label"]),
                    "warm_first_window_start": _first_threshold_index(
                        warm_values,
                        window=window,
                        threshold=float(threshold),
                        higher_is_better=higher_is_better,
                    ),
                    "no_warm_first_window_start": _first_threshold_index(
                        no_warm_values,
                        window=window,
                        threshold=float(threshold),
                        higher_is_better=higher_is_better,
                    ),
                }
            )
        summary["metrics"].append(
            {
                "metric": key,
                "title": metric["title"],
                "higher_is_better": higher_is_better,
                "warm_first_window_mean": first_warm,
                "no_warm_first_window_mean": first_no_warm,
                "warm_last_window_mean": last_warm,
                "no_warm_last_window_mean": last_no_warm,
                "aligned_mean_advantage": aligned_delta,
                "explanation": metric["explanation"],
            }
        )

    for metric in PPO_METRICS:
        key = str(metric["key"])
        warm_values = _series(warm["ppo_rows"], key)
        no_warm_values = _series(no_warm["ppo_rows"], key)
        summary["ppo_reference"].append(
            {
                "metric": key,
                "title": metric["title"],
                "warm_first5_mean": _mean(warm_values[:5]),
                "no_warm_first5_mean": _mean(no_warm_values[:5]),
                "warm_last5_mean": _mean(warm_values[-5:]),
                "no_warm_last5_mean": _mean(no_warm_values[-5:]),
                "explanation": metric["explanation"],
            }
        )
    return summary


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Warmup Convergence Advantage",
        "",
        f"- warm episodes: {summary['warm_episode_count']}",
        f"- no-warm episodes: {summary['no_warm_episode_count']}",
        f"- seed-aligned episodes: {summary['aligned_episode_count']}",
        f"- rolling window: {summary['window']}",
        "",
        "## Warmup-Favorable Business Metrics",
        "",
        "| metric | warm first | no-warm first | warm last | no-warm last | aligned advantage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["metrics"]:
        lines.append(
            "| {metric} | {wf} | {nf} | {wl} | {nl} | {adv} |".format(
                metric=item["metric"],
                wf=_format_float(item["warm_first_window_mean"]),
                nf=_format_float(item["no_warm_first_window_mean"]),
                wl=_format_float(item["warm_last_window_mean"]),
                nl=_format_float(item["no_warm_last_window_mean"]),
                adv=_format_float(item["aligned_mean_advantage"]),
            )
        )
    lines.extend(["", "## Threshold Speed", "", "| threshold | warm | no-warm |", "|---|---:|---:|"])
    for item in summary["thresholds"]:
        lines.append(
            "| {label} | {warm} | {no_warm} |".format(
                label=item["label"],
                warm=item["warm_first_window_start"]
                if item["warm_first_window_start"] is not None
                else "N/A",
                no_warm=item["no_warm_first_window_start"]
                if item["no_warm_first_window_start"] is not None
                else "N/A",
            )
        )
    lines.extend(
        [
            "",
            "## PPO Reference",
            "",
            "| metric | warm first5 | no-warm first5 | warm last5 | no-warm last5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["ppo_reference"]:
        lines.append(
            "| {metric} | {wf} | {nf} | {wl} | {nl} |".format(
                metric=item["metric"],
                wf=_format_float(item["warm_first5_mean"]),
                nf=_format_float(item["no_warm_first5_mean"]),
                wl=_format_float(item["warm_last5_mean"]),
                nl=_format_float(item["no_warm_last5_mean"]),
            )
        )
    lines.extend(["", "## Metric Notes", ""])
    for item in summary["metrics"]:
        lines.append(f"### {item['metric']}")
        lines.append(str(item["explanation"]))
        lines.append("")
    for item in summary["ppo_reference"]:
        lines.append(f"### {item['metric']}")
        lines.append(str(item["explanation"]))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot convergence metrics that highlight BC warm-start advantages.",
    )
    parser.add_argument("--warm-run-dir", default=DEFAULT_WARM_RUN)
    parser.add_argument("--no-warm-run-dir", default=DEFAULT_NO_WARM_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=10, help="rolling window size for episode metrics")
    args = parser.parse_args()

    warm_run_dir = Path(args.warm_run_dir).expanduser().resolve()
    no_warm_run_dir = Path(args.no_warm_run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    warm = _load_run(warm_run_dir)
    no_warm = _load_run(no_warm_run_dir)
    aligned_warm_rows, aligned_no_warm_rows = _seed_aligned_rows(
        warm["episode_rows"],
        no_warm["episode_rows"],
    )
    summary = _build_summary(
        warm=warm,
        no_warm=no_warm,
        aligned_warm_rows=aligned_warm_rows,
        aligned_no_warm_rows=aligned_no_warm_rows,
        window=int(args.window),
    )

    plt = _require_matplotlib()
    _plot_business_curves(
        warm_rows=warm["episode_rows"],
        no_warm_rows=no_warm["episode_rows"],
        window=int(args.window),
        out_path=out_dir / "business_convergence_curves.png",
        plt=plt,
    )
    _plot_seed_aligned_advantage(
        warm_rows=aligned_warm_rows,
        no_warm_rows=aligned_no_warm_rows,
        window=int(args.window),
        out_path=out_dir / "seed_aligned_warmup_advantage.png",
        plt=plt,
    )
    _plot_thresholds(
        summary=summary,
        out_path=out_dir / "time_to_threshold.png",
        plt=plt,
    )
    _plot_ppo_curves(
        warm_rows=warm["ppo_rows"],
        no_warm_rows=no_warm["ppo_rows"],
        out_path=out_dir / "ppo_reference_curves.png",
        plt=plt,
    )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, out_dir / "summary.md")

    print(f"图表已生成到: {out_dir}")
    for name in (
        "business_convergence_curves.png",
        "seed_aligned_warmup_advantage.png",
        "time_to_threshold.png",
        "ppo_reference_curves.png",
        "summary.json",
        "summary.md",
    ):
        print(f"- {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
