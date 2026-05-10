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
# Saturated, paper-safe colors (look solid both on screen and in printed PDFs)
method_color = {
    "WSRL": "#2E5A88",  # deep blue
    "FOCUS-Low-5": "#D67B1E",  # warm orange
    "FOCUS-High-5": "#2A7A2A",  # deep green
}

df = df.copy()
df["ckpt_label"] = df["step"].map(label_map)

# Match the surrounding paper body (Times-like serif, ~9pt) so the figure
# does not look faded next to body text. Also slightly heavier axes.
plt.rcParams.update(
    {
        # STIXGeneral is bundled with matplotlib and matches Times-style serif
        # used by ICML/NeurIPS LaTeX papers — heavier than DejaVu Serif.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Solid black axes for paper-print contrast
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "text.color": "black",
        # Make every piece of text bold so the figure does not look light next
        # to bold-feeling LaTeX body text.
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
        edgecolor="none",
        linewidth=0,
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel("AUC after 20K")
ax.set_ylim(0.75, 1.00)
ax.set_title("(a) Post-burn-in online return", fontsize=11, pad=8)
ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")

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
        edgecolor="none",
        linewidth=0,
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel(r"Frac. evals $\geq 0.9$ after 20K")
ax.set_ylim(0.0, 1.05)
ax.set_title("(b) High-success occupancy", fontsize=11, pad=8)
ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")

# Single legend ABOVE both panels — well clear of titles.
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.00),
    prop={"weight": "bold", "size": 10},
)
plt.setp(leg.get_texts(), fontweight="bold")

# Manual top margin so the figure-level legend has its own band
# (don't use constrained_layout — it conflicts with fig.legend()
#  positioned outside the axes).
plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13, wspace=0.28)

Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
