#!/usr/bin/env python
"""
Generate the 4 paper figures from wandb runs.

Run AFTER all 4 cells finish:
    python analysis/make_figures.py

Or override run names:
    python analysis/make_figures.py \\
        --wsrl_full wsrl_full_medium_redq_seed0 \\
        --btccq_full btccq_full_medium_redq_seed0 \\
        --wsrl_reduced wsrl_reduced_medium_redq_seed0 \\
        --btccq_reduced btccq_reduced_medium_redq_seed0

Outputs to analysis/figures/.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
except ImportError:
    print("wandb not installed. pip install wandb")
    sys.exit(1)


# Default run name substrings (override with CLI args)
DEFAULT_RUNS = {
    "wsrl_full": "wsrl_full_medium_redq_seed0",
    "btccq_full": "btccq_full_medium_redq_seed0",
    "wsrl_reduced": "wsrl_reduced_medium_redq_seed0",
    "btccq_reduced": "btccq_reduced_medium_redq_seed0",
}

COLORS = {
    "wsrl_full": "#1f77b4",  # blue
    "btccq_full": "#ff7f0e",  # orange
    "wsrl_reduced": "#2ca02c",  # green
    "btccq_reduced": "#d62728",  # red
}

LABELS = {
    "wsrl_full": "WSRL (full warmup, K=5000)",
    "btccq_full": "BT-CCQ (full warmup, K=5000)",
    "wsrl_reduced": "WSRL (reduced warmup, K=1250)",
    "btccq_reduced": "BT-CCQ (reduced warmup, K=1250)",
}

LINESTYLES = {
    "wsrl_full": "-",
    "btccq_full": "-",
    "wsrl_reduced": "--",
    "btccq_reduced": "--",
}


# -----------------------------------------------------------------------------
# wandb data fetch
# -----------------------------------------------------------------------------


def fetch_history(
    name_pattern: str, entity: str = "bz292-cornell-university", project: str = "wsrl"
):
    """Fetch full history of the most recently created run matching name_pattern."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    matched = [r for r in runs if name_pattern in r.name]
    if not matched:
        return None
    matched.sort(key=lambda r: r.created_at, reverse=True)
    run = matched[0]
    print(f"  Found {run.name} (state={run.state}, created={run.created_at})")
    return run.history(samples=10000, pandas=True)


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _safe_col(df, col):
    """Return (steps, values) for a column or (None, None) if missing/empty."""
    if df is None or col not in df.columns:
        return None, None
    s = df[col].dropna()
    if len(s) == 0:
        return None, None
    return s.index.values, s.values


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def figure_learning_curves(histories, out_dir, online_start_step=None):
    """Figure 1: success_rate vs step (4 cells overlay)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    plotted = False
    for key in ["wsrl_full", "btccq_full", "wsrl_reduced", "btccq_reduced"]:
        df = histories.get(key)
        x, y = _safe_col(df, "evaluation/success_rate")
        if x is None:
            print(f"  WARN: {key} missing evaluation/success_rate")
            continue
        ax.plot(
            x,
            y,
            label=LABELS[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            marker="o",
            markersize=4,
        )
        plotted = True

    if online_start_step is not None and plotted:
        ax.axvline(
            online_start_step,
            color="grey",
            linestyle=":",
            alpha=0.6,
            label="online start",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate")
    ax.set_title("Online fine-tuning performance (antmaze-medium-diverse-v2)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig1_learning_curves.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


def figure_q_collapse(histories, out_dir, online_start_step=None):
    """Figure 2: q_eval_max trajectory on fixed offline batch (4 cells overlay)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    plotted = False
    for key in ["wsrl_full", "btccq_full", "wsrl_reduced", "btccq_reduced"]:
        df = histories.get(key)
        x, y = _safe_col(df, "q_diag/q_diag/q_eval_max")
        if x is None:
            print(f"  WARN: {key} missing q_diag/q_diag/q_eval_max")
            continue
        ax.plot(x, y, label=LABELS[key], color=COLORS[key], linestyle=LINESTYLES[key])
        plotted = True

    if online_start_step is not None and plotted:
        ax.axvline(
            online_start_step,
            color="grey",
            linestyle=":",
            alpha=0.6,
            label="online start",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\max_{(s,a)} Q(s,a)$ on fixed offline batch")
    ax.set_title("Q-collapse diagnostic")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig2_q_collapse.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


def figure_gate_trajectory(histories, out_dir):
    """Figure 3: gate_frac and gate_mean over training (only BT-CCQ cells)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for key in ["btccq_full", "btccq_reduced"]:
        df = histories.get(key)
        if df is None:
            print(f"  WARN: {key} run missing")
            continue
        for ax_idx, metric in enumerate(["gate_frac", "gate_mean"]):
            col = f"training/critic/btccq/{metric}"
            x, y = _safe_col(df, col)
            if x is None:
                continue
            axes[ax_idx].plot(
                x,
                y,
                label=LABELS[key],
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                alpha=0.85,
            )

    axes[0].set_ylabel("Gate firing fraction")
    axes[0].set_title("BT-CCQ gate firing rate over training")
    axes[1].set_ylabel("Gate mean weight")
    axes[1].set_title("BT-CCQ gate mean weight over training")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig3_gate_trajectory.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


def figure_calibration(histories, out_dir):
    """Figure 4: calibration diagnostic bar chart (only BT-CCQ cells)."""
    metric_cols = [
        ("q_hat", "q_hat (final)"),
        ("q_tail", "q_tail"),
        ("q_scale", "q_scale"),
        ("calib_positive_frac", "positive frac"),
        ("calib_zero_frac", "zero frac"),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cell_keys = ["btccq_full", "btccq_reduced"]
    n_metrics = len(metric_cols)
    bar_w = 0.35
    idx = np.arange(n_metrics)
    for i, cell in enumerate(cell_keys):
        df = histories.get(cell)
        if df is None:
            print(f"  WARN: {cell} run missing")
            continue
        vals = []
        for col_short, _ in metric_cols:
            col = f"btccq_calibration/btccq/{col_short}"
            if col in df.columns:
                v = df[col].dropna()
                vals.append(float(v.iloc[0]) if len(v) > 0 else 0.0)
            else:
                vals.append(0.0)
        offset = (i - 0.5) * bar_w
        ax.bar(idx + offset, vals, bar_w, label=LABELS[cell], color=COLORS[cell])
        for j, v in enumerate(vals):
            ax.text(
                idx[j] + offset, v, f"{v:.2g}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(idx)
    ax.set_xticklabels([m[1] for m in metric_cols], rotation=15, ha="right")
    ax.set_title("BT-CCQ calibration diagnostic (at offline→online transition)")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    out = os.path.join(out_dir, "fig4_calibration.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="analysis/figures")
    p.add_argument("--entity", default="bz292-cornell-university")
    p.add_argument("--project", default="wsrl")
    p.add_argument("--wsrl_full", default=DEFAULT_RUNS["wsrl_full"])
    p.add_argument("--btccq_full", default=DEFAULT_RUNS["btccq_full"])
    p.add_argument("--wsrl_reduced", default=DEFAULT_RUNS["wsrl_reduced"])
    p.add_argument("--btccq_reduced", default=DEFAULT_RUNS["btccq_reduced"])
    p.add_argument(
        "--online_start",
        type=int,
        default=1_000_000,
        help="Step at which online fine-tuning starts (for vertical line on figs 1,2).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_style()

    print("Fetching wandb runs...")
    histories = {
        "wsrl_full": fetch_history(args.wsrl_full, args.entity, args.project),
        "btccq_full": fetch_history(args.btccq_full, args.entity, args.project),
        "wsrl_reduced": fetch_history(args.wsrl_reduced, args.entity, args.project),
        "btccq_reduced": fetch_history(args.btccq_reduced, args.entity, args.project),
    }
    for key, df in histories.items():
        status = f"n_rows={len(df)}" if df is not None else "MISSING (skip in plots)"
        print(f"  {key}: {status}")

    print("\nGenerating figures...")
    figure_learning_curves(histories, args.out_dir, online_start_step=args.online_start)
    figure_q_collapse(histories, args.out_dir, online_start_step=args.online_start)
    figure_gate_trajectory(histories, args.out_dir)
    figure_calibration(histories, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/")
    print("\nNext: run again after all 4 cells finish to refresh:")
    print(f"  python {sys.argv[0]}")


if __name__ == "__main__":
    main()
