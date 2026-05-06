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
import csv
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

        # Full history -> CSV (pandas optional)
        try:
            import pandas as pd  # noqa: F401

            hist_df = run.history(samples=10000, pandas=True)
            use_pandas = hasattr(hist_df, "to_csv")
        except ImportError:
            use_pandas = False
            hist_df = run.history(samples=10000, pandas=False)

        if use_pandas:
            out_csv = os.path.join(args.out_dir, f"{run.name}.csv")
            hist_df.to_csv(out_csv, index=False)
            print(f"Wrote {len(hist_df)} rows to {out_csv} (via pandas)")

            cols = sorted(hist_df.columns)

            def col_values(c):
                return [v for v in hist_df[c].tolist() if v is not None]

        else:
            # hist_df is a list of dicts
            rows = list(hist_df)
            print(f"Got {len(rows)} rows (no pandas, will write raw CSV)")
            # collect all keys
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())
            cols = sorted(all_keys)

            out_csv = os.path.join(args.out_dir, f"{run.name}.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in cols})
            print(f"Wrote {len(rows)} rows to {out_csv} (raw CSV)")

            def col_values(c):
                return [r.get(c) for r in rows if r.get(c) is not None]

        # Print available columns (so you know what's logged)
        print("\nAvailable metrics (non-empty):")
        for c in cols:
            if c.startswith("_"):
                continue
            vals = col_values(c)
            if not vals:
                continue
            try:
                print(
                    f"  {c:55s}  n={len(vals):4d}  "
                    f"first={float(vals[0]):.4g}  last={float(vals[-1]):.4g}"
                )
            except (TypeError, ValueError):
                # non-numeric (e.g. media), skip number formatting
                print(f"  {c:55s}  n={len(vals):4d}  (non-numeric)")

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
            vals = col_values(col)
            if not vals:
                print(f"  {col:48s} (NOT LOGGED)")
                continue
            try:
                vals_f = [float(v) for v in vals]
            except (TypeError, ValueError):
                print(f"  {col:48s} (non-numeric)")
                continue
            n = len(vals_f)
            snapshot = [
                ("step0", vals_f[0]),
                ("p25", vals_f[n // 4]),
                ("mid", vals_f[n // 2]),
                ("p75", vals_f[3 * n // 4]),
                ("last", vals_f[-1]),
            ]
            line = "  ".join(f"{l}={v:.3g}" for l, v in snapshot)
            print(f"  {col:48s} {line}")


if __name__ == "__main__":
    main()
