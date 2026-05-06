#!/usr/bin/env bash
# Quick BT-CCQ smoke test from an existing CalQL checkpoint.
# Runs ~1000 online steps to verify q_hat / gate / loss are not NaN.
#
# Usage:
#   bash experiments/scripts/antmaze/smoke_btccq_from_ckpt.sh /path/to/checkpoint_1000000

set -e

if [ -z "$1" ]; then
  echo "Usage: bash $(basename $0) /path/to/checkpoint_1000000 [env_name]"
  exit 1
fi

RESUME_PATH=$1
ENV_NAME=${2:-antmaze-large-diverse-v2}

export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

echo "===== BT-CCQ smoke test ====="
echo "Checkpoint: ${RESUME_PATH}"
echo "Env:        ${ENV_NAME}"
echo ""

python finetune.py \
  --agent btccq \
  --config experiments/configs/train_config.py:antmaze_btccq \
  --env ${ENV_NAME} \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 1_000_000 \
  --num_online_steps 3000 \
  --resume_path ${RESUME_PATH} \
  --warmup_steps 100 \
  --utd 1 \
  --batch_size 256 \
  --btccq_alpha 0.1 \
  --btccq_w_out 0.2 \
  --btccq_calib_ratio 0.05 \
  --save_dir /workspace/checkpoints/smoke_btccq_online \
  --eval_interval 3000 \
  --log_interval 500 \
  --save_interval 3000 \
  --seed 0 \
  --debug

echo ""
echo "===== Smoke test complete ====="
echo "Look for in the log above:"
echo "  btccq/q_hat        : finite, not NaN, not exactly 0"
echo "  btccq/gate_frac    : in [0.01, 0.6]"
echo "  btccq/gate_mean    : in [0.6, 1.0]"
echo "  btccq/delta_mean   : finite"
echo "  critic_loss        : finite, not NaN"
