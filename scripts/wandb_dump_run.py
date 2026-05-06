#!/usr/bin/env python
"""
Pull a wandb run's full metric history to CSV + print key summaries.

Usage:
    python scripts/wandb_dump_run.py <run_name_substring>

Example:
    python scripts/wandb_dump_run.py reduced_btccq_seed0

Defaults to wandb entity/project: bz292-cornell-university/wsrl
"""

import argparse
import os
import sys

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "name_filter",
        help="Substring to match in run name (e.g. reduced_btccq_seed0)",
    )
    p.add_argument("--entity", default="bz292-cornell-university")
    p.add_argument("--project", default="wsrl")
    p.add_argument("--out_dir", default="/workspace/results/wandb_dumps")
    p.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Max number of matching runs to export",
    )
    args = p.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    matched = [r for r in runs if args.name_filter in r.name]
    if not matched:
        print(f"No run matched '{args.name_filter}'")
        sys.exit(1)

    matched.sort(key=lambda r: r.created_at, reverse=True)
    matched = matched[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    for run in matched:
        print(f"\n{'='*70}")
        print(f"Run: {run.name}")
        print(f"ID:  {run.id}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        print(f"URL: {run.url}")
        print(f"{'='*70}")

        # Full history -> CSV
        hist = run.history(samples=10000, pandas=True)
        out_csv = os.path.join(args.out_dir, f"{run.name}.csv")
        hist.to_csv(out_csv, index=False)
        print(f"Wrote {len(hist)} rows to {out_csv}")

        # Print available columns (so you know what's logged)
        print("\nAvailable metrics:")
        for c in sorted(hist.columns):
            if c.startswith("_"):
                continue
            non_null = hist[c].dropna()
            if len(non_null) == 0:
                continue
            print(
                f"  {c:55s}  n={len(non_null):4d}  "
                f"first={non_null.iloc[0]:.4g}  last={non_null.iloc[-1]:.4g}"
            )

        # Highlight key metrics for BT-CCQ
        print("\n----- BT-CCQ key trajectory -----")
        key_cols = [
            "training/btccq/gate_frac",
            "training/btccq/gate_mean",
            "training/btccq/delta_mean",
            "training/btccq/q_hat",
            "training/critic_loss",
            "training/predicted_qs",
            "training/target_qs",
            "training/q_diag/q_eval_mean",
            "training/q_diag/q_eval_max",
            "evaluation/success_rate",
            "evaluation/average_return",
            "evaluation/average_normalized_return",
        ]
        for col in key_cols:
            if col in hist.columns:
                vals = hist[col].dropna()
                if len(vals) == 0:
                    continue
                # Show first, p25, mid, p75, last
                n = len(vals)
                snapshot = [
                    ("step0", vals.iloc[0]),
                    ("p25", vals.iloc[n // 4]),
                    ("mid", vals.iloc[n // 2]),
                    ("p75", vals.iloc[3 * n // 4]),
                    ("last", vals.iloc[-1]),
                ]
                line = "  ".join(f"{l}={v:.3g}" for l, v in snapshot)
                print(f"  {col:48s} {line}")
            else:
                print(f"  {col:48s} (NOT LOGGED)")


if __name__ == "__main__":
    main()
