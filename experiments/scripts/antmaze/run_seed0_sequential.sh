#!/usr/bin/env bash
# Sequentially run all 4 BT-CCQ matrix cells for a given seed and checkpoint,
# with explicit logging and GPU snapshots between runs.
#
# Designed to be safe to run inside tmux overnight on a single GPU:
#   - set -e: any failed step stops the rest (don't silently skip)
#   - tee: full stdout/stderr captured per cell
#   - sleep 20: lets CUDA/XLA cooldown between processes (prevents OOM)
#
# Usage:
#   bash experiments/scripts/antmaze/run_seed0_sequential.sh /path/to/checkpoint [seed=0]

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
echo "BT-CCQ sequential runner"
echo "Checkpoint: ${CKPT}"
echo "Seed:       ${SEED}"
echo "Env:        ${ENV_NAME}"
echo "Save root:  ${SAVE_ROOT}"
echo "Log root:   ${LOG_ROOT}"
echo "Python:     $(which python)"
echo "Start time: $(date)"
python -c "import jax; print('JAX devices:', jax.devices())"
echo "============================================================"

run_job () {
  local NAME="$1"
  shift

  echo ""
  echo "============================================================"
  echo ">>> Starting ${NAME}"
  echo "    Time: $(date)"
  echo "============================================================"

  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv || true

  "$@" 2>&1 | tee "${LOG_ROOT}/${NAME}.log"

  echo "============================================================"
  echo ">>> Finished ${NAME} at $(date)"
  echo "============================================================"

  sleep 20
}

# Order rationale: BT-CCQ before WSRL — fail-fast on the new code.
# BT-CCQ is the agent we wrote; if it has a bug (e.g. q_hat calibration
# blows up on the real checkpoint), we want to know in 10 min, not 30.
# WSRL is upstream / well-tested baseline; it almost never breaks.

# Eval is expensive on antmaze-large because untrained policies get stuck
# in the maze and run all 1000 steps per episode without reaching goal.
# 5 trajs every 50k steps = 4 evals total per cell = ~5 min eval / cell.
EVAL_ARGS=(
  --eval_interval 50000
  --n_eval_trajs 5
  --save_interval 100000
  --log_interval 10000
)

# 1. Reduced-warmup BT-CCQ  (test: short warmup + gate)
run_job "reduced_btccq_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_btccq_reduced_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/reduced_btccq" \
    --exp_name "reduced_btccq_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 2. Reduced-warmup WSRL  (control: short warmup baseline)
run_job "reduced_wsrl_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/reduced_wsrl" \
    --exp_name "reduced_wsrl_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 3. Full-warmup BT-CCQ  (test: standard warmup + gate)
run_job "full_btccq_seed${SEED}" \
  bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
    --env "${ENV_NAME}" \
    --resume_path "${CKPT}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_ROOT}/full_btccq" \
    --exp_name "full_btccq_seed${SEED}" \
    "${EVAL_ARGS[@]}"

# 4. Full-warmup WSRL  (control: standard WSRL baseline)
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
echo "All 4 jobs finished at $(date)"
echo "Logs:        ${LOG_ROOT}"
echo "Checkpoints: ${SAVE_ROOT}"
echo "============================================================"
