"""Per-head conservatism footprint heatmap across two AntMaze pretraining envs.

Reads per-checkpoint CFS-D diagnostic CSVs from
``results/cfs/seed0_sweep/`` (medium) and
``results/cfs/large_seed0_sweep/`` (large), restricted to the aligned
550K--900K window (50K interval, 8 checkpoints per env).

Produces a vertically stacked heatmap (two panels with a shared x-axis and
a single colorbar) sized for ICML two-column \\textwidth placement.
"""
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MED_DIR = Path("results/cfs/seed0_sweep")
LRG_DIR = Path("results/cfs/large_seed0_sweep")
OUT_PDF = Path("figures/rho_heatmap_figure.pdf")
OUT_PNG = Path("figures/rho_heatmap_figure.png")

# Aligned analysis window across both envs (50K interval).
STEP_MIN = 550_000
STEP_MAX = 900_000


def load_sweep(dir_path: Path, step_min: int = STEP_MIN, step_max: int = STEP_MAX):
    files = sorted(
        glob.glob(str(dir_path / "cfs_stats_step*.csv")),
        key=lambda x: int(re.search(r"step(\d+)", x).group(1)),
    )
    out = []
    for f in files:
        step = int(re.search(r"step(\d+)", f).group(1))
        if not (step_min <= step <= step_max):
            continue
        df = pd.read_csv(f).sort_values("head_id")
        out.append((step, df["rho_h"].values.astype(float)))
    return out


def main():
    medium = load_sweep(MED_DIR)
    large = load_sweep(LRG_DIR)
    assert medium and large, "Sweep CSVs missing; check paths."
    assert len(medium) == len(
        large
    ), f"Aligned-window mismatch: medium={len(medium)} vs large={len(large)}"

    n_heads = len(medium[0][1])
    steps = [s for s, _ in medium]
    mat_med = np.stack([r for _, r in medium], axis=1)  # (10, 8)
    mat_lrg = np.stack([r for _, r in large], axis=1)

    # Common color scale across panels so cells are visually comparable.
    vmin = float(min(mat_med.min(), mat_lrg.min()))
    vmax = float(max(mat_med.max(), mat_lrg.max()))

    # Match paper figure typography (STIXGeneral serif, bold).
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
        }
    )

    # ICML \textwidth ~= 6.75"; height tuned so each cell is roughly square.
    fig = plt.figure(figsize=(6.8, 5.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.025],
        height_ratios=[1.0, 1.0],
        hspace=0.32,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[:, 1])

    def draw_panel(ax, mat, title):
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"$h_{{{i}}}$" for i in range(n_heads)], fontsize=8)
        ax.set_ylabel("Critic head", fontsize=9)
        ax.set_title(title, fontsize=9.5, pad=4)
        # White dot on max-rho cell of every column.
        for j in range(mat.shape[1]):
            i = int(np.argmax(mat[:, j]))
            ax.plot(j, i, marker="o", markersize=3.0, color="white", markeredgewidth=0)
        return im

    im_top = draw_panel(ax_top, mat_med, "antmaze-medium-diverse-v2")
    im_bot = draw_panel(ax_bot, mat_lrg, "antmaze-large-diverse-v2")

    # Shared x-axis labels on bottom panel only.
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bot.set_xticks(range(len(steps)))
    ax_bot.set_xticklabels([f"{s // 1000}" for s in steps], fontsize=8)
    ax_bot.set_xlabel("Pretrain checkpoint step (K)", fontsize=9.5)

    # Single shared colorbar on right.
    cbar = fig.colorbar(im_bot, cax=cax)
    cbar.set_label(
        r"Per-head conservatism footprint $\rho_h$",
        fontsize=9,
        labelpad=4,
    )
    cbar.ax.tick_params(labelsize=8)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")

    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(
        f"matrix shape per panel = (10, {len(steps)})  "
        f"rho range = [{vmin:.3f}, {vmax:.3f}]"
    )


if __name__ == "__main__":
    main()
