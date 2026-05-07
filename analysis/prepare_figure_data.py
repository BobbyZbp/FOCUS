"""
Build the two CSVs that the figure scripts expect:
  data/seed0_checkpoint_sweep.csv         (for make_checkpoint_audit_figure.py)
  data/medium_seed0_postburnin_metrics.csv (for make_online_summary_figure.py)

Reads from existing files in results/cfs/.
"""

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "cfs"
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Pretrain success per medium ckpt step (from continue_runA wandb history;
# manually transcribed once — does not depend on wandb being reachable).
PRETRAIN_SUCCESS = {
    550000: 0.65,
    600000: 0.45,
    650000: 0.40,
    700000: 0.50,
    750000: 0.65,
    800000: 0.60,
    850000: 0.65,
    900000: 0.55,
}

# ============================================================
# 1. seed0_checkpoint_sweep.csv  (medium 8 ckpts diagnostic)
# ============================================================
sweep_dir = RES / "seed0_sweep"
rows = []
for f in sorted(sweep_dir.glob("cfs_stats_step*.csv")):
    step = int(re.search(r"step(\d+)", f.name).group(1))
    df = pd.read_csv(f)
    if df.empty:
        continue
    rows.append(
        {
            "step": step,
            "rho_cv": round(float(df["rho_cv"].iloc[0]), 3),
            "rho_mean": round(float(df["rho_mean"].iloc[0]), 3),
            "max_rho": round(float(df["rho_h"].max()), 3),
            "max_head": int(df.loc[df["rho_h"].idxmax(), "head_id"]),
            "success": PRETRAIN_SUCCESS.get(step, float("nan")),
        }
    )
audit_df = pd.DataFrame(rows).sort_values("step")
audit_csv = DATA / "seed0_checkpoint_sweep.csv"
audit_df.to_csv(audit_csv, index=False)
print(f"saved: {audit_csv}  ({len(audit_df)} rows)")
print(audit_df.to_string(index=False))

# ============================================================
# 2. medium_seed0_postburnin_metrics.csv
# ============================================================
src = RES / "medium_seed0_metrics.csv"
m = pd.read_csv(src)

# Map our naming → user's expected naming
METHOD_MAP = {
    "WSRL": "WSRL",
    "FOCUS-Low": "FOCUS-Low-5",
    "FOCUS-High": "FOCUS-High-5",
}
out = (
    pd.DataFrame(
        {
            "step": m["ckpt"].str.replace("k", "").astype(int) * 1000,
            "method": m["Method"].map(METHOD_MAP),
            "auc_after_20k": m["AUC_after_20k"],
            "frac_ge_0p9_after_20k": m["Frac_ge_0.9_after_20k"],
            "std_after_20k": m["Std_after_20k"],
            "final": m["Final"],
        }
    )
    .sort_values(["step", "method"])
    .reset_index(drop=True)
)

post_csv = DATA / "medium_seed0_postburnin_metrics.csv"
out.to_csv(post_csv, index=False)
print(f"\nsaved: {post_csv}  ({len(out)} rows)")
print(out.to_string(index=False))
