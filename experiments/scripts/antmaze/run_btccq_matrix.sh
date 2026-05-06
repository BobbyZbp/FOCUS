#!/usr/bin/env bash
# Run the 4-cell BT-CCQ experiment matrix from a single CalQL checkpoint.
#
# Cells: {reduced_wsrl, reduced_btccq, full_wsrl, full_btccq}
#
# Usage:
#   bash experiments/scripts/antmaze/run_btccq_matrix.sh /path/to/checkpoint_1000000 [seed] [env]
#
# Recommended: run cells one at a time with --debug first; only batch with this script
# after smoke_btccq_from_ckpt.sh has passed.

set -e

if [ -z "$1" ]; then
  echo "Usage: bash $(basename $0) /path/to/checkpoint_1000000 [seed=0] [env=antmaze-large-diverse-v2]"
  exit 1
fi

RESUME_PATH=$1
SEED=${2:-0}
ENV_NAME=${3:-antmaze-large-diverse-v2}
SAVE_ROOT="/workspace/results/btccq_matrix_${ENV_NAME//[-_]/}_seed${SEED}"

mkdir -p ${SAVE_ROOT}

echo "===== BT-CCQ matrix: 4 cells ====="
echo "Env:        ${ENV_NAME}"
echo "Seed:       ${SEED}"
echo "Checkpoint: ${RESUME_PATH}"
echo "Save root:  ${SAVE_ROOT}"
echo ""

# 1. Reduced-warmup WSRL (control: short warmup baseline)
echo ">>> [1/4] Reduced-warmup WSRL"
bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
  --env ${ENV_NAME} \
  --resume_path ${RESUME_PATH} \
  --seed ${SEED} \
  --save_dir ${SAVE_ROOT}/reduced_wsrl \
  --exp_name reduced_wsrl_seed${SEED}

# 2. Reduced-warmup BT-CCQ (test: short warmup + gate)
echo ">>> [2/4] Reduced-warmup BT-CCQ"
bash experiments/scripts/antmaze/launch_btccq_reduced_warmup.sh \
  --env ${ENV_NAME} \
  --resume_path ${RESUME_PATH} \
  --seed ${SEED} \
  --save_dir ${SAVE_ROOT}/reduced_btccq \
  --exp_name reduced_btccq_seed${SEED}

# 3. Full-warmup WSRL (control: standard WSRL baseline)
echo ">>> [3/4] Full-warmup WSRL"
bash experiments/scripts/antmaze/launch_wsrl_full_warmup.sh \
  --env ${ENV_NAME} \
  --resume_path ${RESUME_PATH} \
  --seed ${SEED} \
  --save_dir ${SAVE_ROOT}/full_wsrl \
  --exp_name full_wsrl_seed${SEED}

# 4. Full-warmup BT-CCQ (test: standard warmup + gate)
echo ">>> [4/4] Full-warmup BT-CCQ"
bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
  --env ${ENV_NAME} \
  --resume_path ${RESUME_PATH} \
  --seed ${SEED} \
  --save_dir ${SAVE_ROOT}/full_btccq \
  --exp_name full_btccq_seed${SEED}

echo ""
echo "===== Matrix complete ====="
echo "Results: ${SAVE_ROOT}"
