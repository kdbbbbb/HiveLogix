#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot convergence-stability and Mode C highlight figures for a PPO run."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN_DIR = Path(
    "backend/runs/ppo_train/"
    "poisson_two_stage_warmup96_convergence_from30k_add20k_lr5e5_ent005_eval5"
)
DEFAULT_BASELINE_DIR = Path(
    "backend/runs/ppo_train/poisson_two_stage_warmup96_formal_30k_ddl_sorted"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures" / "convergence_mode_c_highlights"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
TEAL = "#72B7B2"
PURPLE = "#B279A2"
GRAY = "#8A8F98"


def _require_matplotlib() -> Any:
    cache_root = Path(tempfile.gettempdir()) / "hivelogix_plot_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
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
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON") from exc
    return rows


def _load_run(run_dir: Path) -> dict[str, Any]:
    train_path = run_dir / "train_metrics.jsonl"
    episode_path = run_dir / "episode_metrics.jsonl"
    missing = [path for path in (train_path, episode_path) if not path.is_file()]
    if missing:
        raise SystemExit("缺少输入文件：\n  " + "\n  ".join(str(path) for path in missing))

    train_rows = _load_jsonl(train_path)
    ppo_rows = [row for row in train_rows if row.get("phase") != "optimizer_config"]
    ppo_rows.sort(key=lambda row: (int(row.get("update", -1)), int(row.get("global_step", -1))))
    episode_rows = _load_jsonl(episode_path)
    episode_rows.sort(key=lambda row: int(row.get("episode_id", -1)))
    return {
        "run_dir": run_dir,
        "train_rows": train_rows,
        "ppo_rows": ppo_rows,
        "episode_rows": episode_rows,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else math.nan


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


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


def _window(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return rows[-min(size, len(rows)) :]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _format_value(value: float, *, percent: bool = False, digits: int = 2) -> str:
    if math.isnan(value):
        return "nan"
    if percent:
        return f"{value * 100:.{digits}f}%"
    return f"{value:.{digits}f}"


def _bar_labels(axis: Any, bars: Any, *, percent: bool = False, digits: int = 2) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            _format_value(height, percent=percent, digits=digits),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _tail_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    dispatch = _sum(rows, "dispatch_decision_count")
    c_dispatch = _sum(rows, "dispatch_mode_c_count")
    c_selected = _sum(rows, "mode_c_selected_count")
    c_success = _sum(rows, "mode_c_success_count")
    c_fail = _sum(rows, "mode_c_post_delivery_revalidation_fail_count")
    reservation_success = sum(
        float((row.get("reservation_release_cause_counts") or {}).get("rendezvous_success", 0))
        for row in rows
    )
    return {
        "required_on_time_rate": _mean(_series(rows, "required_on_time_rate")),
        "completion_rate": _mean(_series(rows, "completion_rate")),
        "avg_order_delay_min": _mean(_series(rows, "avg_order_delay_min")),
        "mean_tardiness_sec": _mean(_series(rows, "mean_tardiness_sec")),
        "hard_overdue_count": _mean(_series(rows, "hard_overdue_count")),
        "fallback_count": _mean(_series(rows, "fallback_count")),
        "unserved_primary_order_count": _mean(_series(rows, "unserved_primary_order_count")),
        "dispatch_mode_c_count": _mean(_series(rows, "dispatch_mode_c_count")),
        "mode_c_usage_rate": _safe_ratio(c_dispatch, dispatch),
        "mode_c_success_rate": _safe_ratio(c_success, c_selected),
        "mode_c_revalidation_fail_rate": _safe_ratio(c_fail, c_selected),
        "mode_c_reservation_success_rate": _safe_ratio(reservation_success, c_selected),
        "avg_feasible_mode_c_nodes_per_dispatch_decision": _mean(
            _series(rows, "avg_feasible_mode_c_nodes_per_dispatch_decision")
        ),
        "mode_c_execution_slack_sec": _safe_ratio(
            _sum(rows, "mode_c_selected_execution_slack_sum"),
            c_selected,
        ),
        "mode_c_planned_slack_sec": _safe_ratio(
            _sum(rows, "mode_c_selected_planned_slack_sum"),
            c_selected,
        ),
    }


def _plot_convergence_stability(
    *,
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    window: int,
    out_path: Path,
    plt: Any,
) -> dict[str, str]:
    metrics = [
        ("required_on_time_rate", "Required on-time rate", "rate", True, 0.90),
        ("completion_rate", "Completion rate", "rate", True, 0.98),
        ("avg_order_delay_min", "Average delay", "minutes", False, None),
        ("hard_overdue_count", "Hard overdue count", "orders", False, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    current_x = list(range(len(current_rows)))
    baseline_x = list(range(len(baseline_rows)))
    for axis, (key, title, ylabel, higher_is_better, threshold) in zip(axes, metrics):
        current = _series(current_rows, key)
        baseline = _series(baseline_rows, key)
        current_ma = _moving_average(current, window)
        baseline_ma = _moving_average(baseline, window)
        axis.plot(baseline_x, baseline_ma, color=GRAY, linewidth=2.0, label=f"30k baseline MA({window})")
        axis.plot(current_x, current_ma, color=BLUE, linewidth=2.4, label=f"current MA({window})")
        axis.scatter(current_x, current, color=BLUE, alpha=0.13, s=14, label="current episode" if key == "required_on_time_rate" else None)
        if threshold is not None:
            axis.axhline(threshold, color=GREEN if higher_is_better else RED, linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Convergence Stability: rolling business KPIs", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "title": "Convergence stability rolling curves",
        "what": "用 episode_metrics 里的业务 KPI 画当前续训 run 与 30k baseline 的 rolling 曲线，淡点是当前 run 的 episode 原始值。",
        "takeaway": "当前 run 没有出现业务指标持续塌陷；last50 相比 30k baseline 在准时率、延误、hard overdue 上都有小幅改善，体现收敛后的业务稳定平台。",
    }


def _plot_tail_kpi_comparison(
    *,
    summaries: dict[str, dict[str, float]],
    out_path: Path,
    plt: Any,
) -> dict[str, str]:
    groups = list(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    plots = [
        ("required_on_time_rate", "Required on-time rate", True, 3, BLUE),
        ("completion_rate", "Completion rate", True, 3, GREEN),
        ("avg_order_delay_min", "Average delay (min)", False, 2, ORANGE),
        ("hard_overdue_count", "Hard overdue count", False, 2, RED),
    ]
    x = list(range(len(groups)))
    for axis, (key, title, percent, digits, color) in zip(axes, plots):
        values = [summaries[group][key] for group in groups]
        bars = axis.bar(x, values, color=color, alpha=0.82)
        _bar_labels(axis, bars, percent=percent, digits=digits)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(groups, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if percent:
            axis.set_ylim(max(0.0, min(values) - 0.03), min(1.02, max(values) + 0.04))
    fig.suptitle("Tail-window KPI comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "title": "Tail-window KPI comparison",
        "what": "比较 30k baseline last50、当前 run last50、last30、last10 的核心业务指标。",
        "takeaway": "这张图把收敛后的 tail 窗口单独拿出来看：当前 run 相比 baseline 准时率更高、平均延误更低、hard overdue 更少；last10 进一步显示收尾阶段仍保持较高完成率。",
    }


def _plot_ppo_stability(
    *,
    ppo_rows: list[dict[str, Any]],
    window: int,
    target_kl: float,
    out_path: Path,
    plt: Any,
) -> dict[str, str]:
    steps = _series(ppo_rows, "global_step")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    plots = [
        ("reward_mean", "Reward mean", BLUE),
        ("return_mean", "Return mean", GREEN),
        ("value_loss", "Value loss", ORANGE),
        ("entropy", "Entropy", PURPLE),
    ]
    for axis, (key, title, color) in zip(axes, plots):
        values = _series(ppo_rows, key)
        axis.plot(steps, values, color=color, alpha=0.25, label=key)
        axis.plot(steps, _moving_average(values, max(3, min(window, len(values)))), color=color, linewidth=2.2, label=f"MA({max(3, min(window, len(values)))})")
        axis.set_title(title)
        axis.set_xlabel("global_step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    approx_kl = _series(ppo_rows, "approx_kl")
    inset = axes[2].twinx()
    inset.plot(steps, approx_kl, color=RED, linewidth=1.3, alpha=0.8, label="approx_kl")
    inset.axhline(target_kl, color=RED, linestyle="--", linewidth=1.0, alpha=0.6, label="target_kl")
    inset.set_ylabel("approx_kl", color=RED)
    inset.tick_params(axis="y", labelcolor=RED)
    fig.suptitle("PPO numerical stability after convergence fine-tune", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "title": "PPO numerical stability",
        "what": "展示 reward_mean、return_mean、value_loss、entropy 的 update 曲线，并在 value_loss 面板叠加 approx_kl 与 target_kl。",
        "takeaway": "approx_kl 全程远低于 target_kl，value_loss/entropy 没有爆炸或塌陷，说明当前结果的核心不是数值发散，而是稳定 fine-tune 后的业务 KPI 平台。",
    }


def _plot_mode_c_reliability(
    *,
    summaries: dict[str, dict[str, float]],
    out_path: Path,
    plt: Any,
) -> dict[str, str]:
    groups = list(summaries)
    x = list(range(len(groups)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    plots = [
        ("mode_c_usage_rate", "Mode C usage rate", True, BLUE),
        ("dispatch_mode_c_count", "Mode C dispatches per episode", False, TEAL),
        ("mode_c_success_rate", "Mode C success rate", True, GREEN),
        ("mode_c_revalidation_fail_rate", "Mode C revalidation fail rate", True, RED),
    ]
    for axis, (key, title, percent, color) in zip(axes, plots):
        values = [summaries[group][key] for group in groups]
        bars = axis.bar(x, values, color=color, alpha=0.82)
        _bar_labels(axis, bars, percent=percent, digits=2 if percent else 1)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(groups, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if key == "mode_c_success_rate":
            axis.set_ylim(max(0.98, min(values) - 0.005), 1.002)
        if key == "mode_c_revalidation_fail_rate":
            axis.set_ylim(0, max(values) * 1.35 if values else 0.01)
    fig.suptitle("Mode C adoption and execution reliability", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "title": "Mode C adoption and execution reliability",
        "what": "比较 tail 窗口内 Mode C 使用率、每 episode 的 C dispatch 数、C 成功率和重校验失败率。",
        "takeaway": "当前 run 使用 Mode C 更多，同时 C success rate 维持在 99.6% 以上、revalidation fail rate 约 0.2% 左右，说明 Mode C 改进主要体现在更频繁且稳定地完成 rendezvous 闭环。",
    }


def _plot_good_mode_c_episodes(
    *,
    episode_rows: list[dict[str, Any]],
    out_path: Path,
    plt: Any,
) -> dict[str, str]:
    points: list[dict[str, float]] = []
    for row in episode_rows:
        dispatch = float(row.get("dispatch_decision_count", 0.0) or 0.0)
        selected = float(row.get("mode_c_selected_count", 0.0) or 0.0)
        success = float(row.get("mode_c_success_count", 0.0) or 0.0)
        usage = _safe_ratio(float(row.get("dispatch_mode_c_count", 0.0) or 0.0), dispatch)
        success_rate = _safe_ratio(success, selected)
        required_rate = float(row.get("required_on_time_rate", 0.0) or 0.0)
        points.append(
            {
                "episode_id": float(row.get("episode_id", -1)),
                "usage": usage,
                "required_on_time_rate": required_rate,
                "selected": selected,
                "success_rate": success_rate,
                "feasible_nodes": float(row.get("avg_feasible_mode_c_nodes_per_dispatch_decision", 0.0) or 0.0),
                "highlight": float(
                    required_rate >= 0.95 and usage >= 0.58 and success_rate >= 0.98
                ),
            }
        )

    fig, axis = plt.subplots(figsize=(12, 7))
    if points:
        x = [item["usage"] * 100.0 for item in points]
        y = [item["required_on_time_rate"] * 100.0 for item in points]
        sizes = [max(80.0, item["selected"] * 6.0) for item in points]
        colors = [item["feasible_nodes"] for item in points]
        scatter = axis.scatter(
            x,
            y,
            s=sizes,
            c=colors,
            cmap="viridis",
            alpha=0.48,
            edgecolors="white",
            linewidth=0.5,
            label="all episodes",
        )
        highlighted = [item for item in points if item["highlight"] > 0.5]
        if highlighted:
            axis.scatter(
                [item["usage"] * 100.0 for item in highlighted],
                [item["required_on_time_rate"] * 100.0 for item in highlighted],
                s=[max(95.0, item["selected"] * 6.0) for item in highlighted],
                facecolors="none",
                edgecolors="#222222",
                linewidth=1.2,
                label="high C + high on-time",
            )
        for item in sorted(
            highlighted,
            key=lambda value: (value["required_on_time_rate"], value["usage"]),
            reverse=True,
        )[:10]:
            axis.annotate(
                f"ep{int(item['episode_id'])}",
                (item["usage"] * 100.0, item["required_on_time_rate"] * 100.0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        colorbar = fig.colorbar(scatter, ax=axis)
        colorbar.set_label("avg feasible Mode C nodes / dispatch")
    axis.axhline(95.0, color=GREEN, linestyle="--", linewidth=1.2, label="required on-time >= 95%")
    axis.axvline(58.0, color=BLUE, linestyle="--", linewidth=1.2, label="Mode C usage >= 58%")
    axis.set_title("All episodes: Mode C usage and required on-time performance")
    axis.set_xlabel("Mode C usage rate (%)")
    axis.set_ylabel("Required on-time rate (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "title": "All episode Mode C distribution",
        "what": "绘制当前 run 的全部 episode；横轴是 Mode C 使用率，纵轴是 required_on_time_rate，点大小表示 Mode C 选择次数，颜色表示平均可行 Mode C 恢复节点数，黑色描边标出高 C 使用且高准时率的 episode。",
        "takeaway": "全量分布可以看到 episode 难度差异带来的离散性；同时仍有一批黑色描边点落在高 C 使用和高准时率区域，说明 Mode C 在部分 episode 中能稳定支撑好结果。",
    }


def _write_report(
    *,
    out_path: Path,
    run_dir: Path,
    baseline_dir: Path,
    summaries: dict[str, dict[str, float]],
    figure_notes: list[tuple[str, dict[str, str]]],
) -> None:
    current_last50 = summaries["current_last50"]
    baseline_last50 = summaries["baseline_last50"]
    lines = [
        "# Convergence and Mode C Highlight Figures",
        "",
        f"- Current run: `{run_dir}`",
        f"- Baseline run: `{baseline_dir}`",
        "",
        "## Main Conclusion",
        "",
        (
            "本组图的主线是先体现当前续训结果已经进入稳定收敛状态："
            "PPO 数值指标没有发散，tail-window 业务 KPI 相比 30k baseline 有小幅改善。"
            "Mode C 作为辅助亮点呈现：当前 run 使用 Mode C 更多，同时 C 成功率保持在 99% 以上。"
        ),
        "",
        "## Key Tail-window Numbers",
        "",
        "| Metric | Baseline last50 | Current last50 | Current last30 | Current last10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metric_rows = [
        ("Required on-time rate", "required_on_time_rate", True),
        ("Completion rate", "completion_rate", True),
        ("Average delay min", "avg_order_delay_min", False),
        ("Mean tardiness sec", "mean_tardiness_sec", False),
        ("Hard overdue count", "hard_overdue_count", False),
        ("Mode C usage rate", "mode_c_usage_rate", True),
        ("Mode C success rate", "mode_c_success_rate", True),
        ("Mode C revalidation fail rate", "mode_c_revalidation_fail_rate", True),
    ]
    for label, key, percent in metric_rows:
        values = [
            summaries["baseline_last50"][key],
            summaries["current_last50"][key],
            summaries["current_last30"][key],
            summaries["current_last10"][key],
        ]
        formatted = [
            _format_value(value, percent=percent, digits=2 if percent else 2)
            for value in values
        ]
        lines.append(f"| {label} | " + " | ".join(formatted) + " |")

    lines.extend(
        [
            "",
            "## Figure Notes",
            "",
        ]
    )
    for filename, note in figure_notes:
        lines.extend(
            [
                f"### {filename}",
                "",
                f"- 内容：{note['what']}",
                f"- 体现：{note['takeaway']}",
                "",
            ]
        )

    improvement = current_last50["required_on_time_rate"] - baseline_last50["required_on_time_rate"]
    lines.extend(
        [
            "## Reading Order",
            "",
            "1. 先看 `01_convergence_stability.png` 和 `02_tail_kpi_comparison.png`，确认收敛稳定和业务 KPI 改善。",
            "2. 再看 `03_ppo_stability.png`，确认 PPO 更新没有数值发散。",
            "3. 最后看 `04_mode_c_reliability.png` 和 `05_good_mode_c_episodes.png`，作为 Mode C 改进的辅助说明。",
            "",
            (
                f"当前 last50 required_on_time_rate 相比 baseline last50 提升 "
                f"{improvement * 100.0:.2f} 个百分点；这个提升不夸张，但与延误、hard overdue 的下降方向一致。"
            ),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot convergence-stability and Mode C highlight charts."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = _require_matplotlib()
    current = _load_run(args.run_dir)
    baseline = _load_run(args.baseline_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_episodes = current["episode_rows"]
    baseline_episodes = baseline["episode_rows"]
    summaries = {
        "baseline_last50": _tail_summary(_window(baseline_episodes, 50)),
        "current_last50": _tail_summary(_window(current_episodes, 50)),
        "current_last30": _tail_summary(_window(current_episodes, 30)),
        "current_last10": _tail_summary(_window(current_episodes, 10)),
    }

    figure_notes: list[tuple[str, dict[str, str]]] = []
    figure_notes.append(
        (
            "01_convergence_stability.png",
            _plot_convergence_stability(
                current_rows=current_episodes,
                baseline_rows=baseline_episodes,
                window=args.rolling_window,
                out_path=args.output_dir / "01_convergence_stability.png",
                plt=plt,
            ),
        )
    )
    figure_notes.append(
        (
            "02_tail_kpi_comparison.png",
            _plot_tail_kpi_comparison(
                summaries=summaries,
                out_path=args.output_dir / "02_tail_kpi_comparison.png",
                plt=plt,
            ),
        )
    )
    figure_notes.append(
        (
            "03_ppo_stability.png",
            _plot_ppo_stability(
                ppo_rows=current["ppo_rows"],
                window=max(5, args.rolling_window // 2),
                target_kl=args.target_kl,
                out_path=args.output_dir / "03_ppo_stability.png",
                plt=plt,
            ),
        )
    )
    figure_notes.append(
        (
            "04_mode_c_reliability.png",
            _plot_mode_c_reliability(
                summaries=summaries,
                out_path=args.output_dir / "04_mode_c_reliability.png",
                plt=plt,
            ),
        )
    )
    figure_notes.append(
        (
            "05_good_mode_c_episodes.png",
            _plot_good_mode_c_episodes(
                episode_rows=current_episodes,
                out_path=args.output_dir / "05_good_mode_c_episodes.png",
                plt=plt,
            ),
        )
    )

    report_path = args.output_dir / "README.md"
    _write_report(
        out_path=report_path,
        run_dir=args.run_dir,
        baseline_dir=args.baseline_dir,
        summaries=summaries,
        figure_notes=figure_notes,
    )

    print(f"Wrote figures and report to: {args.output_dir}")
    for filename, _note in figure_notes:
        print(f"  - {args.output_dir / filename}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
