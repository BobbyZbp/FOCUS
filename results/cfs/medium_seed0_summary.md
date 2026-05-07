# Medium-Diverse Online Metrics — seed 0 (K=5, Final)

Setup: antmaze-medium-diverse-v2, 200K online steps, eval every 5K, n_eval=20, REDQ-10 subsample 2.
Burn-in = 20K, threshold = 0.9, k_stable = 3.

## Per-ckpt context

| Ckpt | Ckpt_Success | CV(ρ) |
|------|--------------|-------|
| 550K | 0.65 | 0.635 |
| 600K | 0.45 | 0.050 |
| 850K | 0.65 | 0.334 |
| 900K | 0.55 | 0.038 |

See `medium_seed0_metrics.csv` for full table.

## Δ vs WSRL highlights

- **850K** (CV=0.334) ⭐ CLEAN: FOCUS-High Frac>=0.9 drops 38pp, Std doubles. FOCUS-Low matches WSRL.
- **550K** (CV=0.635) REVERSED: FOCUS-High slightly best. Likely early-training "useful pessimism" effect.
- **600K** (CV=0.050) Mostly null. WSRL slightly best on AUC, FOCUS-High best on Frac>=0.9.
- **900K** (CV=0.038) Null as predicted. All variants within ±0.03 AUC.

## Pending

- Trim-1 ablation (cfs_top_k=9): 850K + 550K running on Pod 2
- Large pretrain done (1M, 20 ckpts), CFS diagnostic done (`large_seed0_sweep/_summary.csv`)
