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

# pretty labels
method_label = {
    "WSRL": "WSRL",
    "FOCUS-Low-5": "FOCUS-Low-5",
    "FOCUS-High-5": "FOCUS-High-5",
}

df = df.copy()
df["ckpt_label"] = df["step"].map(label_map)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)

x = np.arange(len(order))
width = 0.23

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
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel("AUC after 20K")
ax.set_ylim(0.75, 1.00)
ax.set_title("(a) Post-burn-in online return", fontsize=11)

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
    )

ax.set_xticks(x)
ax.set_xticklabels([label_map[s] for s in order])
ax.set_xlabel("Checkpoint")
ax.set_ylabel("Frac. evals $\\geq 0.9$ after 20K")
ax.set_ylim(0.0, 1.0)
ax.set_title("(b) High-success occupancy", fontsize=11)

# single legend for whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05),
)

Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
