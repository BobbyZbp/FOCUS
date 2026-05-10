"""Quantitative analysis of the rho_h heatmap data across two AntMaze envs.

Reads per-checkpoint CFS-D diagnostics from results/cfs/seed0_sweep/ and
results/cfs/large_seed0_sweep/ and prints the numerical claims used in
Appendix A of the paper (transient outlier identity, CV range, max-rho
spikes, head-identity transitions, environment comparison).
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

MED_DIR = Path("results/cfs/seed0_sweep")
LRG_DIR = Path("results/cfs/large_seed0_sweep")

# Aligned analysis window across both envs (50K interval).
STEP_MIN = 550_000
STEP_MAX = 900_000


def load_sweep(dir_path: Path, step_min: int = STEP_MIN, step_max: int = STEP_MAX):
    files = sorted(
        glob.glob(str(dir_path / "cfs_stats_step*.csv")),
        key=lambda x: int(re.search(r"step(\d+)", x).group(1)),
    )
    rows = []
    for f in files:
        step = int(re.search(r"step(\d+)", f).group(1))
        if not (step_min <= step <= step_max):
            continue
        df = pd.read_csv(f).sort_values("head_id").reset_index(drop=True)
        rho = df["rho_h"].values.astype(float)
        rows.append(
            {
                "step": step,
                "rho_per_head": rho,
                "rho_mean": float(df["rho_mean"].iloc[0]),
                "rho_std": float(df["rho_std"].iloc[0]),
                "rho_cv": float(df["rho_cv"].iloc[0]),
                "rho_d_corr": float(df["rho_d_corr"].iloc[0]),
                "max_rho": float(rho.max()),
                "max_head": int(rho.argmax()),
                "min_rho": float(rho.min()),
            }
        )
    return pd.DataFrame(rows)


def transitions(seq):
    """Number of times consecutive entries differ."""
    return int((np.diff(np.array(seq)) != 0).sum())


def summarize(name: str, df: pd.DataFrame):
    print(f"\n{'='*70}\n  {name}  ({len(df)} ckpts)\n{'='*70}")
    print(
        df[["step", "rho_mean", "rho_std", "rho_cv", "max_rho", "max_head"]].to_string(
            index=False
        )
    )
    print(f"\nCV(rho) range: [{df['rho_cv'].min():.3f}, {df['rho_cv'].max():.3f}]")
    print(f"  - mean : {df['rho_cv'].mean():.3f}")
    print(f"  - median: {df['rho_cv'].median():.3f}")
    print(
        f"  - frac > 0.15 (strong heterogeneity): "
        f"{(df['rho_cv'] > 0.15).mean():.0%}"
    )
    print(
        f"  - frac < 0.05 (no signal):            "
        f"{(df['rho_cv'] < 0.05).mean():.0%}"
    )

    print(
        f"\nrho range: max(max_rho)={df['max_rho'].max():.2f} at step "
        f"{int(df.loc[df['max_rho'].idxmax(), 'step'])}, "
        f"head h{int(df.loc[df['max_rho'].idxmax(), 'max_head'])}"
    )

    seq = df.sort_values("step")["max_head"].tolist()
    n_trans = transitions(seq)
    n_pairs = len(seq) - 1
    print(f"\nMax-head identity sequence: {seq}")
    print(
        f"  - {n_trans} transitions across {n_pairs} consecutive ckpt pairs "
        f"({n_trans/n_pairs:.0%} switch rate)"
    )
    print(f"  - distinct heads ever 'max': {sorted(set(seq))} ({len(set(seq))} of 10)")

    # Outlier intensity per ckpt: max_rho / mean_rho ratio
    df["outlier_ratio"] = df["max_rho"] / df["rho_mean"]
    extreme = df[df["outlier_ratio"] > 2.0]
    print(f"\nExtreme outliers (max_rho > 2*mean_rho): {len(extreme)} ckpts")
    if len(extreme):
        print(
            extreme[
                ["step", "rho_mean", "max_rho", "outlier_ratio", "max_head"]
            ].to_string(index=False)
        )


def cross_env(med: pd.DataFrame, lrg: pd.DataFrame):
    print(f"\n{'='*70}\n  CROSS-ENVIRONMENT COMPARISON\n{'='*70}")

    rows = []
    for name, df in [
        (f"medium-diverse ({len(med)} ckpts)", med),
        (f"large-diverse ({len(lrg)} ckpts)", lrg),
    ]:
        rows.append(
            {
                "env": name,
                "n_ckpts": len(df),
                "CV_min": round(df["rho_cv"].min(), 3),
                "CV_max": round(df["rho_cv"].max(), 3),
                "CV_mean": round(df["rho_cv"].mean(), 3),
                "max(rho_h)": round(df["max_rho"].max(), 2),
                "%_high_CV(>0.15)": f"{(df['rho_cv'] > 0.15).mean():.0%}",
                "%_low_CV(<0.05)": f"{(df['rho_cv'] < 0.05).mean():.0%}",
                "max_head_transitions": transitions(
                    df.sort_values("step")["max_head"].tolist()
                ),
                "distinct_max_heads": len(set(df["max_head"])),
            }
        )
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))


def appendix_paragraph_numbers(med: pd.DataFrame, lrg: pd.DataFrame):
    """Print the exact numerical claims used in the Appendix A paragraph."""
    print(f"\n{'='*70}\n  KEY NUMBERS FOR APPENDIX A LATEX\n{'='*70}")
    lrg_sorted = lrg.sort_values("step")
    seq = lrg_sorted["max_head"].tolist()
    n_trans_lrg = transitions(seq)
    n_pairs_lrg = len(seq) - 1
    cv_min_step = int(lrg.loc[lrg["rho_cv"].idxmin(), "step"])
    cv_max_step = int(lrg.loc[lrg["rho_cv"].idxmax(), "step"])
    max_rho_step = int(lrg.loc[lrg["max_rho"].idxmax(), "step"])
    max_rho_head = int(lrg.loc[lrg["max_rho"].idxmax(), "max_head"])

    print(
        f"  large CV(rho) range: {lrg['rho_cv'].min():.2f} (step {cv_min_step}) "
        f"to {lrg['rho_cv'].max():.2f} (step {cv_max_step})"
    )
    print(
        f"  large max single-head footprint: {lrg['max_rho'].max():.2f} "
        f"at step {max_rho_step}, head h{max_rho_head}"
    )
    print(
        f"  900K rho_max = {float(lrg.loc[lrg['step']==900000,'max_rho'].iloc[0]):.2f} "
        f"(head h{int(lrg.loc[lrg['step']==900000,'max_head'].iloc[0])})"
    )
    print(f"  large max-head transitions: {n_trans_lrg} across {n_pairs_lrg} pairs")
    print(
        f"  ALL heads ever appear as max in large: "
        f"{sorted(set(seq))} ({len(set(seq))} of 10)"
    )

    # Medium for cross-check
    print(
        f"\n  medium CV(rho) range: {med['rho_cv'].min():.2f} to {med['rho_cv'].max():.2f}"
    )
    print(
        f"  medium max-head transitions: "
        f"{transitions(med.sort_values('step')['max_head'].tolist())}"
    )


def write_summary_csv(df: pd.DataFrame, out_path: Path):
    cols = [
        "step",
        "rho_mean",
        "rho_std",
        "rho_cv",
        "max_rho",
        "max_head",
        "rho_d_corr",
    ]
    out = df[cols].copy()
    for c in ["rho_mean", "rho_std", "rho_cv", "max_rho", "rho_d_corr"]:
        out[c] = out[c].round(3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    print(f"Aligned window: [{STEP_MIN}, {STEP_MAX}] (inclusive)")
    med = load_sweep(MED_DIR)
    lrg = load_sweep(LRG_DIR)
    summarize("antmaze-medium-diverse-v2", med)
    summarize("antmaze-large-diverse-v2", lrg)
    cross_env(med, lrg)
    appendix_paragraph_numbers(med, lrg)

    # Persist usable-window summaries next to the heatmap data.
    print(f"\n{'='*70}\n  WRITING USABLE-WINDOW SUMMARY CSVS\n{'='*70}")
    write_summary_csv(med, MED_DIR / "_summary_550_900.csv")
    write_summary_csv(lrg, LRG_DIR / "_summary_550_900.csv")
