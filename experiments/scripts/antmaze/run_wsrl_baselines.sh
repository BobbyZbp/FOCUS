#!/usr/bin/env bash
# Run only the two WSRL baseline cells (reduced + full) after BT-CCQ is done.
# Both BT-CCQ cells (reduced + full) already completed in earlier runs.
#
# Usage:
#   bash experiments/scripts/antmaze/run_wsrl_baselines.sh /path/to/checkpoint [seed=0]

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
echo "Running 2 WSRL baseline cells (reduced + full)"
echo "BT-CCQ cells already complete in this seed."
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

# 1. reduced WSRL
run_job "reduced_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
    --env "${ENV_NAME}" --resume_path "${CKPT}" --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/reduced_wsrl" \
    --exp_name "reduced_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 2. full WSRL
run_job "full_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_full_warmup.sh \
    --env "${ENV_NAME}" --resume_path "${CKPT}" --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_wsrl" \
    --exp_name "full_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

echo ""
echo "============================================================"
echo "All WSRL baselines finished at $(date)"
echo "Combined with BT-CCQ runs, full 4-cell matrix is complete."
echo "Logs:        ${LOG_ROOT}"
echo "============================================================"
