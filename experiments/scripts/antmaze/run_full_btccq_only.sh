#!/usr/bin/env bash
# Run only full_btccq_seed0 from a CalQL pretrain checkpoint.
#
# Rationale: WSRL paper provides full_wsrl baseline numbers, so we can
# compare full_btccq vs reported full_wsrl without re-running it ourselves.
# This saves ~30 min GPU and lets us focus on the BT-CCQ test cell.
#
# Usage:
#   bash experiments/scripts/antmaze/run_full_btccq_only.sh /path/to/checkpoint [seed=0]

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
echo "BT-CCQ full-warmup runner (single cell)"
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

NAME="full_btccq_seed${SEED}"

echo ""
echo "============================================================"
echo ">>> Starting ${NAME} at $(date)"
echo "============================================================"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv || true

bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
  --env "${ENV_NAME}" \
  --resume_path "${CKPT}" \
  --seed "${SEED}" \
  --save_dir "${SAVE_ROOT}/full_btccq" \
  --exp_name "${NAME}" \
  "${EVAL_ARGS[@]}" 2>&1 | tee "${LOG_ROOT}/${NAME}.log"

echo "============================================================"
echo ">>> Finished ${NAME} at $(date)"
echo "============================================================"
echo "Log:        ${LOG_ROOT}/${NAME}.log"
echo "Checkpoint: ${SAVE_ROOT}/full_btccq"
