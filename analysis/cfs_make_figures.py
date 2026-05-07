"""Make CFS-D diagnostic figures from a CFS stats CSV."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "results/cfs/cfs_stats.csv", "Input CFS stats CSV.")
flags.DEFINE_string(
    "output_dir", "results/cfs/figures", "Output directory for figures."
)


def _savefig(name: str):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    path = os.path.join(FLAGS.output_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(path)


def _annotate_selected(df, y_col):
    if "selected" not in df.columns:
        return
    for _, row in df[df["selected"] == 1].iterrows():
        plt.text(row["head_id"], row[y_col], "*", ha="center", va="bottom")


def main(_):
    df = pd.read_csv(FLAGS.input)
    df = df.sort_values("head_id")

    plt.figure()
    plt.bar(df["head_id"], df["rho_h"])
    _annotate_selected(df, "rho_h")
    plt.xlabel("critic head id")
    plt.ylabel("rho_h")
    plt.title("CFS conservatism footprint")
    _savefig("cfs_rho_bar.png")

    plt.figure()
    plt.bar(df["head_id"], df["min_win_rate_d_h"])
    _annotate_selected(df, "min_win_rate_d_h")
    plt.axhline(1.0 / len(df), linestyle="--", linewidth=1)
    plt.xlabel("critic head id")
    plt.ylabel("min-win rate d_h")
    plt.title("REDQ target dominance")
    _savefig("cfs_dominance_bar.png")

    plt.figure()
    plt.bar(df["head_id"], df["eta_h"])
    _annotate_selected(df, "eta_h")
    plt.xlabel("critic head id")
    plt.ylabel("eta_h")
    plt.title("Target-dominant conservatism influence")
    _savefig("cfs_eta_bar.png")

    plt.figure()
    plt.scatter(df["rho_h"], df["min_win_rate_d_h"])
    for _, row in df.iterrows():
        plt.text(row["rho_h"], row["min_win_rate_d_h"], str(int(row["head_id"])))
    plt.xlabel("rho_h")
    plt.ylabel("min-win rate d_h")
    plt.title("Footprint vs target dominance")
    _savefig("cfs_rho_vs_dominance.png")

    plt.figure()
    width = 0.25
    x = df["head_id"].to_numpy()
    plt.bar(x - width, df["pess_gap_p_h"], width=width, label="p_h")
    plt.bar(x, df["bellman_err_e_h"], width=width, label="e_h")
    plt.bar(x + width, df["rho_h"], width=width, label="rho_h")
    plt.xlabel("critic head id")
    plt.ylabel("score")
    plt.title("CFS footprint decomposition")
    plt.legend()
    _savefig("cfs_decomposition.png")


if __name__ == "__main__":
    app.run(main)
