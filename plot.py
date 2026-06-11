import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager


def set_chinese_font():
    """自动寻找可用中文字体，避免中文乱码。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    available_fonts = {f.name for f in fontManager.ttflist}

    for font in candidates:
        if font in available_fonts:
            mpl.rcParams["font.sans-serif"] = [font]
            mpl.rcParams["axes.unicode_minus"] = False
            return

    print("警告：未找到常见中文字体，若中文乱码，请手动安装 SimHei / Microsoft YaHei / Noto Sans CJK SC。")


set_chinese_font()


# =========================
# 1. 数据区：低 / 中 / 高
# =========================

configs = ["低", "中", "高"]

data = {
    "贪心+插入启发式": {
        "综合任务完成率": [100.0, 100.0, 100.0],
        "平均订单延迟": [7.62, 18.38, 44.58],
        "准时送达率": [75.5, 61.0, 43.9],
        # 原始单位 Wh，转为 kWh
        "总体能耗成本": [50513.99 / 1000, 62112.73 / 1000, 89878.60 / 1000],
    },
    "遗传算法": {
        "综合任务完成率": [100.0, 100.0, 100.0],
        "平均订单延迟": [0.08, 0.32, 3.41],
        "准时送达率": [91.2, 81.3, 48.5],
        # 原始单位 Wh，转为 kWh
        "总体能耗成本": [95083.27 / 1000, 101584.41 / 1000, 103747.34 / 1000],
    },
    "PPO算法": {
        "综合任务完成率": [100.0, 100.0, 100.0],
        "平均订单延迟": [0.0000, 0.0000, 0.8244],
        "准时送达率": [100.0, 100.0, 92.06],
        "总体能耗成本": [17.7915, 17.5612, 17.7283],
    },
}


# =========================
# 2. 绘图配置
# =========================

algorithms = list(data.keys())

metrics = [
    ("综合任务完成率", "任务完成率 / %", "图1：任务完成率对比"),
    ("准时送达率", "准时送达率 / %", "图2：订单准时率对比"),
    ("平均订单延迟", "平均延迟 / min", "图3：平均延迟（分钟）对比"),
    ("总体能耗成本", "能耗成本 / kWh", "图4：综合调度成本对比"),
]

colors = ["#083b5c", "#4f789d", "#94b3cf"]

x = np.arange(len(configs))
bar_width = 0.23


# =========================
# 3. 绘制 2×2 子图
# =========================

fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
axes = axes.flatten()

for ax, (metric, ylabel, title) in zip(axes, metrics):
    for i, algorithm in enumerate(algorithms):
        values = data[algorithm][metric]
        offset = (i - 1) * bar_width

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=algorithm,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )

        # 数值标注
        for bar in bars:
            height = bar.get_height()

            if metric in ["综合任务完成率", "准时送达率"]:
                label = f"{height:.1f}"
            elif metric == "平均订单延迟":
                label = f"{height:.2f}"
            else:
                label = f"{height:.1f}"

            ax.annotate(
                label,
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

    # 百分比图固定到 0-110，观感更统一；能耗图使用非均匀轴，
    # 避免 PPO 的低能耗柱被 100 kWh 量级的柱子压扁。
    if metric in ["综合任务完成率", "准时送达率"]:
        ax.set_ylim(0, 110)
    elif metric == "总体能耗成本":
        ax.set_yscale("symlog", linthresh=25, linscale=1.1)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 10, 20, 50, 75, 100, 125])
        ax.set_yticklabels(["0", "10", "20", "50", "75", "100", "125"])
    else:
        ymax = max(max(data[alg][metric]) for alg in algorithms)
        ax.set_ylim(0, ymax * 1.18)


# 总图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.94),
    ncol=3,
    frameon=False,
    fontsize=11,
)

fig.suptitle("不同算法在不同订单流强度下的指标对比", fontsize=18, fontweight="bold", y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.90])

# 保存图片
plt.savefig("algorithm_metric_compare_2x2.png", dpi=300, bbox_inches="tight")
plt.savefig("algorithm_metric_compare_2x2.pdf", bbox_inches="tight")

plt.show()
