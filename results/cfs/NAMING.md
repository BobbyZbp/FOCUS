# WandB Run Naming Conventions (CFS Project)

All runs go to `bz292-cornell-university/wsrl` regardless of `--project` flag.
Filter by `run.name` regex, not `group`.

## Patterns
- `wsrl_<X>k_baseline_reduced_seed<S>_*`  — WSRL baseline at <X>K ckpt (standard)
- `cfs_<X>k_low_eta_reduced_seed<S>_*`    — FOCUS-Low (K=5) at <X>K ckpt
- `cfs_<X>k_high_eta_reduced_seed<S>_*`   — FOCUS-High (K=5) at <X>K ckpt
- `cfs_<X>k_trim1_reduced_seed<S>_*`      — FOCUS Trim-1 (cfs_top_k=9) at <X>K ckpt
- `wsrl_reduced_<X>k_seed<S>_*`           — 850K baseline (variant 1)
- `wsrl_<X>k_reduced_seed<S>_*`           — 850K baseline (variant 2)
- `wsrl_reduced_seed<S>_*`                — 850K baseline (legacy, no ckpt prefix)
- `cfs_low_eta_reduced_seed<S>_*`         — 850K low_eta (legacy)
- `cfs_high_eta_reduced_seed<S>_*`        — 850K high_eta (legacy)

## Pretrain
- `calql_redq_medium_500k_seed<S>_*`        — Medium pretrain stage 1 (0→500K)
- `calql_redq_continue_runA_*`              — Medium pretrain stage 2 (500K→900K)
- `calql_redq_pretrain_large_seed<S>_*`     — Large pretrain (0→1M)

## When fetching: dedupe by (ckpt, method) keeping max n_evals.
