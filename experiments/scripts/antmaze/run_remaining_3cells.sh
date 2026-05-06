#!/usr/bin/env bash
# Run the 3 remaining cells after reduced_btccq (which already finished):
#   full_btccq, reduced_wsrl, full_wsrl
#
# Order rationale:
#   - full_btccq first: it's the second-most-important BT-CCQ data point
#   - reduced_wsrl: matched control for the existing reduced_btccq run
#   - full_wsrl: matched control for full_btccq
#
# Usage:
#   bash experiments/scripts/antmaze/run_remaining_3cells.sh /path/to/checkpoint [seed=0]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash $(basename $0) /path/to/checkpoint [seed=0]"
  exit 1
fi

CKPT="$1"
SEED="${2:-0}"
ENV_NAME="antmaze-large-diverse-v2"
SAVE_ROOT="/workspace/results/btccq_ld_seed${SEED}"
LOG_ROOT="/workspace/logs/btccq_ld_seed${SEED}"
mkdir -p "${SAVE_ROOT}" "${LOG_ROOT}"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

EVAL_ARGS=(
  --eval_interval 40000
  --n_eval_trajs 10
  --save_interval 100000
  --log_interval 5000
)

echo "============================================================"
echo "Running 3 remaining cells (full_btccq, reduced_wsrl, full_wsrl)"
echo "Checkpoint: ${CKPT}"
echo "Seed:       ${SEED}"
echo "Start time: $(date)"
python -c "import jax; print('JAX devices:', jax.devices())"
echo "============================================================"

run_job() {
  local NAME="$1"; shift
  echo ""
  echo "============================================================"
  echo ">>> Starting ${NAME} at $(date)"
  echo "============================================================"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv || true
  "$@" 2>&1 | tee "${LOG_ROOT}/${NAME}.log"
  echo ">>> Finished ${NAME} at $(date)"
  sleep 20
}

# 1. full BT-CCQ (test cell, full warmup)
run_job "full_btccq_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
    --env "${ENV_NAME}" --resume_path "${CKPT}" --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_btccq" \
    --exp_name "full_btccq_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 2. reduced WSRL (control for the reduced_btccq we already have)
run_job "reduced_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
    --env "${ENV_NAME}" --resume_path "${CKPT}" --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/reduced_wsrl" \
    --exp_name "reduced_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 3. full WSRL (control for full_btccq)
run_job "full_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_full_warmup.sh \
    --env "${ENV_NAME}" --resume_path "${CKPT}" --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_wsrl" \
    --exp_name "full_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

echo ""
echo "============================================================"
echo "All 3 remaining cells finished at $(date)"
echo "Combined with reduced_btccq, you now have the full 4-cell matrix."
echo "Logs:        ${LOG_ROOT}"
echo "Checkpoints: ${SAVE_ROOT}"
echo "============================================================"
