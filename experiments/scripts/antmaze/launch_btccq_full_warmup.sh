#!/usr/bin/env bash
# BT-CCQ v2 (WSRL-aligned) on antmaze-large-diverse-v2 — full warmup (5000 steps).
#
# This is the WSRL-aligned variant:
#   - inherits SACAgent (no CQL penalty / no cql_max_target_backup / no
#     n-action sampling)
#   - restores from CalQL pretrain checkpoint (passed via --resume_path)
#   - num_offline_steps=0 -> goes straight into online fine-tuning
#   - critic loss = mean(gate * (Q - y_live)^2), nothing else
#
# Usage (from a CalQL checkpoint):
#   bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
#     --resume_path /path/to/checkpoint_1000000

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python finetune.py \
  --agent btccq \
  --config experiments/configs/train_config.py:antmaze_btccq_v2 \
  --env antmaze-large-diverse-v2 \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 0 \
  --num_online_steps 200_000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --btccq_alpha 0.1 \
  --btccq_w_out 0.2 \
  --btccq_calib_ratio 0.1 \
  --project bt-ccq \
  --group antmaze_large_diverse \
  --exp_name btccq_full_warmup \
  $@
