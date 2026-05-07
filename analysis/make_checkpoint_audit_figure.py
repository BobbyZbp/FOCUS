from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ===== change this path if needed =====
CSV_PATH = "data/seed0_checkpoint_sweep.csv"
OUT_PDF = "figures/checkpoint_audit_figure.pdf"
OUT_PNG = "figures/checkpoint_audit_figure.png"

df = pd.read_csv(CSV_PATH)

# make sure steps are sorted
df = df.sort_values("step").copy()
df["step_k"] = df["step"] / 1000.0

# bigger figure, more breathing room (slightly wider so right-side threshold
# labels in panel (a) have room outside the axes)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), constrained_layout=True)

# -------------------------
# (a) CV(rho) + offline success
# -------------------------
ax = axes[0]
ax2 = ax.twinx()

l1 = ax.plot(
    df["step_k"],
    df["rho_cv"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="#1f77b4",
    label=r"CV($\rho$)",
)
l2 = ax2.plot(
    df["step_k"],
    df["success"],
    marker="s",
    linewidth=2,
    markersize=5,
    alpha=0.85,
    color="#ff7f0e",
    label="offline success",
)

xs = df["step_k"]

# threshold reference lines (no text labels — explained in caption)
ax.axhline(0.15, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.45)
ax.axhline(0.05, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.45)

ax.set_xlabel("Offline checkpoint step (K)")
ax.set_ylabel(r"Footprint heterogeneity CV($\rho$)", color="#1f77b4")
ax2.set_ylabel("Offline success", color="#ff7f0e")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax.set_title("(a) Footprint heterogeneity across checkpoints", fontsize=11, pad=10)

# combined legend at top-center, inside but with headroom
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(
    lines,
    labels,
    loc="upper center",
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
)

# extra y headroom so the legend doesn't overlap the 550K spike
ax.set_ylim(0.0, 0.80)
ax.set_xlim(xs.min() - 20, xs.max() + 20)

# -------------------------
# (b) max-footprint head index
# -------------------------
ax = axes[1]
ax.plot(
    df["step_k"],
    df["max_head"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="#2ca02c",
)

for _, row in df.iterrows():
    ax.annotate(
        rf"$\rho$={row['max_rho']:.1f}",
        (row["step_k"], row["max_head"]),
        textcoords="offset points",
        xytext=(0, 11),
        ha="center",
        fontsize=8.5,
        color="#444",
        clip_on=False,
    )

ax.set_xlabel("Offline checkpoint step (K)")
ax.set_ylabel("Highest-footprint head index")
ax.set_yticks(sorted(df["max_head"].unique()))

# extra room at top so annotations don't get clipped
ymin, ymax = df["max_head"].min(), df["max_head"].max()
ax.set_ylim(ymin - 0.7, ymax + 1.2)

# extra x padding both sides
ax.set_xlim(xs.min() - 20, xs.max() + 30)

ax.set_title(r"(b) Highest-footprint head; labels show max $\rho$", fontsize=11, pad=10)

# save
Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
