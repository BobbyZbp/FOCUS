#!/usr/bin/env bash
# Sequential reduced-warmup ablation runner.
set -euo pipefail

bash experiments/scripts/antmaze/cfs_baseline_wsrl_reduced_seed0.sh "$@"
bash experiments/scripts/antmaze/cfs_low_eta_reduced_seed0.sh "$@"
bash experiments/scripts/antmaze/cfs_low_rho_reduced_seed0.sh "$@"
bash experiments/scripts/antmaze/cfs_high_eta_reduced_seed0.sh "$@"
bash experiments/scripts/antmaze/cfs_random_topk_reduced_seed0.sh "$@"
