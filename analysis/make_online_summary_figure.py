from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== change this path if needed =====
CSV_PATH = "data/medium_seed0_postburnin_metrics.csv"
OUT_PDF = "figures/online_summary_figure.pdf"
OUT_PNG = "figures/online_summary_figure.png"

df = pd.read_csv(CSV_PATH)

order = [550000, 600000, 850000, 900000]
label_map = {
    550000: "550K",
    600000: "600K",
    850000: "850K",
    900000: "900K",
}
methods = ["WSRL", "FOCUS-Low-5", "FOCUS-High-5"]

# pretty labels (and consistent colors across panels)
method_label = {
    "WSRL": "WSRL",
    "FOCUS-Low-5": "FOCUS-Low-5",
    "FOCUS-High-5": "FOCUS-High-5",
}
method_color = {
    "WSRL": "#1f77b4",
    "FOCUS-Low-5": "#ff7f0e",
    "FOCUS-High-5": "#2ca02c",
}

df = df.copy()
df["ckpt_label"] = df["step"].map(label_map)

# Larger figure + room at top for legend so it doesn't overlap titles
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

x = np.arange(len(order))
width = 0.25

# -------------------------
# left: AUC after 20K
# -------------------------
ax = axes[0]
for i, m in enumerate(methods):
    sub = df[df["method"] == m].set_index("step").loc[order]
    ax.bar(
        x + (i - 1) * width,
        sub["auc_after_20k"].values,
        width=width,
        label=method_label[m],
        color=method_color[m],
        edgecolor="white",
        linewidth=0.6,
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel("AUC after 20K")
ax.set_ylim(0.75, 1.00)
ax.set_title("(a) Post-burn-in online return", fontsize=11, pad=8)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

# -------------------------
# right: fraction >= 0.9
# -------------------------
ax = axes[1]
for i, m in enumerate(methods):
    sub = df[df["method"] == m].set_index("step").loc[order]
    ax.bar(
        x + (i - 1) * width,
        sub["frac_ge_0p9_after_20k"].values,
        width=width,
        label=method_label[m],
        color=method_color[m],
        edgecolor="white",
        linewidth=0.6,
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel(r"Frac. evals $\geq 0.9$ after 20K")
ax.set_ylim(0.0, 1.05)
ax.set_title("(b) High-success occupancy", fontsize=11, pad=8)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

# Single legend ABOVE both panels — well clear of titles.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.00),
)

# Manual top margin so the figure-level legend has its own band
# (don't use constrained_layout — it conflicts with fig.legend()
#  positioned outside the axes).
plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13, wspace=0.28)

Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
