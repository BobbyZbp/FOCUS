"""Figure 1: footprint heterogeneity vs offline success across pretraining
checkpoints, replicated on two AntMaze environments.

Left panel  = medium-diverse  (8 ckpts, 550K--900K)
Right panel = large-diverse   (8 ckpts, 550K--900K)

Both axes are deliberately scaled identically so the reader can directly
compare $\\mathrm{CV}(\\rho)$ and offline success across environments.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_MED = "data/seed0_checkpoint_sweep.csv"
CSV_LRG = "data/seed0_checkpoint_sweep_large.csv"
OUT_PDF = "figures/checkpoint_audit_figure.pdf"
OUT_PNG = "figures/checkpoint_audit_figure.png"

CV_COLOR = "#2E5A88"
SUCC_COLOR = "#D67B1E"

# Each panel scales independently to its own data range; this keeps the
# success curve from being squashed to the bottom of the panel on the
# harder environment.


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("step").copy()
    df["step_k"] = df["step"] / 1000.0
    return df


def style_rcparams():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "text.color": "black",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "mathtext.default": "bf",
        }
    )


def draw_panel(
    ax,
    df: pd.DataFrame,
    title: str,
    show_legend: bool,
    show_ylabel_left: bool,
    show_ylabel_right: bool,
):
    ax2 = ax.twinx()

    l1 = ax.plot(
        df["step_k"],
        df["rho_cv"],
        marker="o",
        linewidth=2,
        markersize=6,
        color=CV_COLOR,
        label=r"CV($\rho$)",
    )
    l2 = ax2.plot(
        df["step_k"],
        df["success"],
        marker="s",
        linewidth=2,
        markersize=5,
        alpha=0.85,
        color=SUCC_COLOR,
        label="offline success",
    )

    # Threshold reference lines for CV (dashed: strong heterogeneity 0.15;
    # dotted: no-signal 0.05).
    ax.axhline(0.15, color=CV_COLOR, linestyle="--", linewidth=1, alpha=0.45)
    ax.axhline(0.05, color=CV_COLOR, linestyle=":", linewidth=1, alpha=0.45)

    ax.set_xlabel("Offline checkpoint step (K)")
    if show_ylabel_left:
        ax.set_ylabel(r"Footprint heterogeneity CV($\rho$)", color=CV_COLOR)
    if show_ylabel_right:
        ax2.set_ylabel("Offline success", color=SUCC_COLOR)
    ax.tick_params(axis="y", labelcolor=CV_COLOR)
    ax2.tick_params(axis="y", labelcolor=SUCC_COLOR)
    ax.set_title(title, fontsize=11, pad=10)

    # Independent per-panel scaling with small padding so threshold lines
    # and markers don't touch the axis edges.
    cv_pad = 0.05
    succ_pad = 0.04
    ax.set_ylim(
        max(0.0, df["rho_cv"].min() - cv_pad),
        df["rho_cv"].max() + cv_pad,
    )
    ax2.set_ylim(
        max(0.0, df["success"].min() - succ_pad),
        df["success"].max() + succ_pad,
    )
    ax.set_xlim(df["step_k"].min() - 20, df["step_k"].max() + 20)

    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    plt.setp(ax2.get_yticklabels(), fontweight="bold")

    if show_legend:
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        leg = ax.legend(
            lines,
            labels,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(0.18, 0.98),
            ncol=1,
            prop={"weight": "bold", "size": 9.5},
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0.3,
        )
        plt.setp(leg.get_texts(), fontweight="bold")

    return ax2


def main():
    style_rcparams()
    df_med = load(CSV_MED)
    df_lrg = load(CSV_LRG)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), constrained_layout=True)

    draw_panel(
        axes[0],
        df_med,
        "antmaze-medium-diverse-v2",
        show_legend=True,
        show_ylabel_left=True,
        show_ylabel_right=True,
    )
    draw_panel(
        axes[1],
        df_lrg,
        "antmaze-large-diverse-v2",
        show_legend=False,
        show_ylabel_left=True,
        show_ylabel_right=True,
    )

    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
