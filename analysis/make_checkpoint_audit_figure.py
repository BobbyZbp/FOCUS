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

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)

# -------------------------
# (a) CV(rho) + offline success
# -------------------------
ax = axes[0]
ax2 = ax.twinx()

l1 = ax.plot(df["step_k"], df["rho_cv"], marker="o", linewidth=2, label="CV($\\rho$)")
l2 = ax2.plot(
    df["step_k"],
    df["success"],
    marker="s",
    linewidth=2,
    alpha=0.8,
    label="offline success",
)

ax.axhline(0.15, linestyle="--", linewidth=1)
ax.axhline(0.05, linestyle=":", linewidth=1)

ax.set_xlabel("Offline checkpoint step (K)")
ax.set_ylabel("Footprint heterogeneity CV($\\rho$)")
ax2.set_ylabel("Offline success")
ax.set_title("(a) Footprint heterogeneity across checkpoints", fontsize=11)

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper right", frameon=False, fontsize=9)

# -------------------------
# (b) max-footprint head index
# -------------------------
ax = axes[1]
ax.plot(df["step_k"], df["max_head"], marker="o", linewidth=2)

for _, row in df.iterrows():
    ax.annotate(
        f"{row['max_rho']:.1f}",
        (row["step_k"], row["max_head"]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=8,
    )

ax.set_xlabel("Offline checkpoint step (K)")
ax.set_ylabel("Highest-footprint head index")
ax.set_yticks(sorted(df["max_head"].unique()))
ax.set_title("(b) Highest-footprint head; labels show max $\\rho$", fontsize=11)

# save
Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
