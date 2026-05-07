#!/usr/bin/env bash
# Random top-k control under reduced warmup.
set -euo pipefail

: "${CKPT:?Set CKPT=/path/to/pretrained/checkpoint}"

ENV_NAME="${ENV_NAME:-antmaze-large-diverse-v2}"
CONFIG="${CONFIG:-experiments/configs/train_config.py:antmaze_cql}"
SEED="${SEED:-0}"
TOP_K="${TOP_K:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1250}"
NUM_ONLINE_STEPS="${NUM_ONLINE_STEPS:-200000}"
STATS_OUT="${STATS_OUT:-results/cfs/cfs_random_topk_${ENV_NAME}_seed${SEED}.csv}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
mkdir -p "$(dirname "$STATS_OUT")"

python finetune.py \
  --agent calql \
  --config "${CONFIG}" \
  --env "${ENV_NAME}" \
  --resume_path "${CKPT}" \
  --use_redq \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 0 \
  --num_online_steps "${NUM_ONLINE_STEPS}" \
  --online_use_cql_loss=False \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps "${WARMUP_STEPS}" \
  --seed "${SEED}" \
  --use_cfs \
  --cfs_mode random_topk \
  --cfs_top_k "${TOP_K}" \
  --cfs_stats_output "${STATS_OUT}" \
  --project cfs-d \
  --group "${ENV_NAME}" \
  --exp_name cfs_random_topk_reduced_seed${SEED} \
  "$@"
