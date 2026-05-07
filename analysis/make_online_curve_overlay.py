import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="Online success curves")
parser.add_argument("--out", type=str, default="figures/online_curve_overlay.pdf")
parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CSV"))
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)

for label, csv_path in args.run:
    df = pd.read_csv(csv_path)

    # change these if your csv uses different names
    if "online_step" in df.columns:
        x = df["online_step"]
    else:
        x = df.iloc[:, 0]

    if "eval_success" in df.columns:
        y = df["eval_success"]
    else:
        y = df.iloc[:, 1]

    ax.plot(x, y, marker="o", linewidth=1.6, markersize=3, label=label)

ax.set_xlabel("Online step")
ax.set_ylabel("Success rate")
ax.set_title(args.title, fontsize=11)
ax.legend(frameon=False)
Path("figures").mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
print(f"saved: {args.out}")
