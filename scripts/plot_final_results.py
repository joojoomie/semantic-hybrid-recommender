"""Create publication-style figures for the final benchmark results.

The figures summarize the fixed multi-positive Movies_and_TV benchmark using
reported mean/std values across seeds. The script does not load data or rerun
models; it only plots the final metrics and writes PNG/PDF files under
results/figures/.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path("results/figures")
PNG_DPI = 300

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#666666",
}

METRICS = [
    {
        "label": "Two-Tower",
        "Recall@10_mean": 0.301456,
        "Recall@10_std": 0.000722,
        "NDCG@10_mean": 0.230036,
        "NDCG@10_std": 0.000088,
        "HeadRecall@10_mean": 0.634455,
        "HeadRecall@10_std": 0.007500,
        "TailRecall@10_mean": 0.005183,
        "TailRecall@10_std": 0.000941,
        "TailNDCG@10_mean": 0.001984,
        "TailNDCG@10_std": 0.000459,
        "TailExposure@10_mean": 0.030966,
        "TailExposure@10_std": 0.000318,
        "AvgTrainPopularity@10_mean": 14.141478,
        "AvgTrainPopularity@10_std": 0.034439,
    },
    {
        "label": "Popularity",
        "Recall@10_mean": 0.320576,
        "Recall@10_std": 0.000818,
        "NDCG@10_mean": 0.245931,
        "NDCG@10_std": 0.001131,
        "HeadRecall@10_mean": 0.681512,
        "HeadRecall@10_std": 0.002244,
        "TailRecall@10_mean": 0.000202,
        "TailRecall@10_std": 0.000171,
        "TailNDCG@10_mean": 0.000071,
        "TailNDCG@10_std": 0.000054,
        "TailExposure@10_mean": 0.000388,
        "TailExposure@10_std": 0.000231,
        "AvgTrainPopularity@10_mean": 14.691437,
        "AvgTrainPopularity@10_std": 0.023772,
    },
    {
        "label": "Mini-DLRM",
        "Recall@10_mean": 0.370926,
        "Recall@10_std": 0.005052,
        "NDCG@10_mean": 0.277062,
        "NDCG@10_std": 0.000287,
        "HeadRecall@10_mean": 0.604134,
        "HeadRecall@10_std": 0.004211,
        "TailRecall@10_mean": 0.152085,
        "TailRecall@10_std": 0.002824,
        "TailNDCG@10_mean": 0.080227,
        "TailNDCG@10_std": 0.002023,
        "TailExposure@10_mean": 0.426628,
        "TailExposure@10_std": 0.011186,
        "AvgTrainPopularity@10_mean": 8.582466,
        "AvgTrainPopularity@10_std": 0.112005,
    },
    {
        "label": "Hybrid",
        "Recall@10_mean": 0.434306,
        "Recall@10_std": 0.006158,
        "NDCG@10_mean": 0.341041,
        "NDCG@10_std": 0.005775,
        "HeadRecall@10_mean": 0.718288,
        "HeadRecall@10_std": 0.002428,
        "TailRecall@10_mean": 0.175722,
        "TailRecall@10_std": 0.016229,
        "TailNDCG@10_mean": 0.089091,
        "TailNDCG@10_std": 0.009816,
        "TailExposure@10_mean": 0.366850,
        "TailExposure@10_std": 0.003637,
        "AvgTrainPopularity@10_mean": 9.187212,
        "AvgTrainPopularity@10_std": 0.035983,
    },
    {
        "label": "Hybrid+Boost",
        "Recall@10_mean": 0.434340,
        "Recall@10_std": 0.006784,
        "NDCG@10_mean": 0.340624,
        "NDCG@10_std": 0.006036,
        "HeadRecall@10_mean": 0.715289,
        "HeadRecall@10_std": 0.002121,
        "TailRecall@10_mean": 0.178243,
        "TailRecall@10_std": 0.017227,
        "TailNDCG@10_mean": 0.090643,
        "TailNDCG@10_std": 0.010354,
        "TailExposure@10_mean": 0.372658,
        "TailExposure@10_std": 0.003883,
        "AvgTrainPopularity@10_mean": 9.122362,
        "AvgTrainPopularity@10_std": 0.032173,
    },
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [FIGURE_DIR / f"{stem}.png", FIGURE_DIR / f"{stem}.pdf"]
    fig.tight_layout()
    fig.savefig(outputs[0], dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def grouped_bar_chart(
    *,
    metrics: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    output_stem: str,
) -> list[Path]:
    labels = [row["label"] for row in METRICS]
    x = np.arange(len(labels))
    width = 0.34 if len(metrics) == 2 else 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for idx, (mean_key, std_key, display_name) in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        means = np.array([float(row[mean_key]) for row in METRICS])
        stds = np.array([float(row[std_key]) for row in METRICS])
        color = [COLORS["blue"], COLORS["orange"], COLORS["green"]][idx]
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            label=display_name,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=len(metrics))
    return save_figure(fig, output_stem)


def plot_overall_metrics() -> list[Path]:
    return grouped_bar_chart(
        metrics=[
            ("Recall@10_mean", "Recall@10_std", "Recall@10"),
            ("NDCG@10_mean", "NDCG@10_std", "NDCG@10"),
        ],
        title="Overall ranking quality",
        ylabel="Metric value",
        ylim=(0.0, 0.50),
        output_stem="final_overall_metrics",
    )


def plot_tail_metrics() -> list[Path]:
    return grouped_bar_chart(
        metrics=[
            ("TailRecall@10_mean", "TailRecall@10_std", "TailRecall@10"),
            ("TailNDCG@10_mean", "TailNDCG@10_std", "TailNDCG@10"),
        ],
        title="Tail-item relevance",
        ylabel="Tail metric value",
        ylim=(0.0, 0.23),
        output_stem="final_tail_metrics",
    )


def plot_head_tail_recall() -> list[Path]:
    return grouped_bar_chart(
        metrics=[
            ("HeadRecall@10_mean", "HeadRecall@10_std", "HeadRecall@10"),
            ("TailRecall@10_mean", "TailRecall@10_std", "TailRecall@10"),
        ],
        title="Head vs. tail recall",
        ylabel="Recall@10",
        ylim=(0.0, 0.80),
        output_stem="final_head_tail_recall",
    )


def plot_exposure_tradeoff() -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    x = np.array([float(row["AvgTrainPopularity@10_mean"]) for row in METRICS])
    y = np.array([float(row["TailExposure@10_mean"]) for row in METRICS])
    xerr = np.array([float(row["AvgTrainPopularity@10_std"]) for row in METRICS])
    yerr = np.array([float(row["TailExposure@10_std"]) for row in METRICS])
    labels = [row["label"] for row in METRICS]
    colors = [
        COLORS["gray"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["blue"],
        COLORS["purple"],
    ]

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="#555555",
        elinewidth=0.8,
        capsize=3,
        zorder=1,
    )
    ax.scatter(x, y, s=60, c=colors, edgecolor="black", linewidth=0.5, zorder=2)

    offsets = {
        "Two-Tower": (-42, 8),
        "Popularity": (-56, -8),
        "Mini-DLRM": (8, 4),
        "Hybrid": (8, -12),
        "Hybrid+Boost": (8, 8),
    }
    for xi, yi, label in zip(x, y, labels, strict=True):
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=offsets[label],
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6},
        )

    ax.set_title("Exposure-popularity tradeoff")
    ax.set_xlabel("Avg. train popularity of top-10 items")
    ax.set_ylabel("TailExposure@10")
    ax.set_xlim(max(x) + 0.4, min(x) - 0.6)
    ax.set_ylim(-0.02, max(y) + 0.07)
    ax.grid(axis="y")
    ax.text(
        0.02,
        0.02,
        "Lower x = less popularity-biased",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "final_exposure_tradeoff")


def main() -> None:
    configure_style()
    outputs: list[Path] = []
    outputs.extend(plot_overall_metrics())
    outputs.extend(plot_tail_metrics())
    outputs.extend(plot_exposure_tradeoff())
    outputs.extend(plot_head_tail_recall())

    print("Saved figures:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
