#!/usr/bin/env bash
# CFS-D checkpoint diagnostic for AntMaze REDQ checkpoints.
set -euo pipefail

: "${CKPT:?Set CKPT=/path/to/pretrained/checkpoint}"

ENV_NAME="${ENV_NAME:-antmaze-large-diverse-v2}"
CONFIG="${CONFIG:-experiments/configs/train_config.py:antmaze_cql}"
AGENT="${AGENT:-calql}"
SEED="${SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
TOP_K="${TOP_K:-5}"
OUT="${OUT:-results/cfs/cfs_stats_${ENV_NAME}_seed${SEED}.csv}"

mkdir -p "$(dirname "$OUT")"

python analysis/cfs_compute_stats.py \
  --agent "${AGENT}" \
  --config "${CONFIG}" \
  --env "${ENV_NAME}" \
  --checkpoint "${CKPT}" \
  --use_redq \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --seed "${SEED}" \
  --num_samples "${NUM_SAMPLES}" \
  --cfs_top_k "${TOP_K}" \
  --output "${OUT}" \
  "$@"
