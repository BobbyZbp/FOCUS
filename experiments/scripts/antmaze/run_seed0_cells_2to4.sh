#!/usr/bin/env bash
# Resume run_seed0_sequential after cell 1 (reduced_btccq) succeeded but
# cell 2 (reduced_wsrl) crashed due to the antmaze_wsrl LayerNorm bug.
# Only re-runs cells 2-4 to avoid wasting GPU on the cell that already
# completed.
#
# Usage:
#   bash experiments/scripts/antmaze/run_seed0_cells_2to4.sh /path/to/checkpoint [seed=0]

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

echo "============================================================"
echo "BT-CCQ resume runner (cells 2-4 only)"
echo "Cell 1 (reduced_btccq) already completed successfully."
echo "Checkpoint: ${CKPT}"
echo "Seed:       ${SEED}"
echo "Start time: $(date)"
python -c "import jax; print('JAX devices:', jax.devices())"
echo "============================================================"

EVAL_ARGS=(
  --eval_interval 40000
  --n_eval_trajs 10
  --save_interval 100000
  --log_interval 5000
)

run_job () {
  local NAME="$1"
  shift
  echo ""
  echo "============================================================"
  echo ">>> Starting ${NAME} at $(date)"
  echo "============================================================"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv || true
  "$@" 2>&1 | tee "${LOG_ROOT}/${NAME}.log"
  echo "============================================================"
  echo ">>> Finished ${NAME} at $(date)"
  echo "============================================================"
  sleep 20
}

# 2. Reduced-warmup WSRL
run_job "reduced_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/reduced_wsrl" \
    --exp_name "reduced_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 3. Full-warmup BT-CCQ
run_job "full_btccq_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_btccq" \
    --exp_name "full_btccq_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 4. Full-warmup WSRL
run_job "full_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_full_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_wsrl" \
    --exp_name "full_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

echo ""
echo "============================================================"
echo "Cells 2-4 finished at $(date)"
echo "Logs:        ${LOG_ROOT}"
echo "Checkpoints: ${SAVE_ROOT}"
echo "============================================================"
