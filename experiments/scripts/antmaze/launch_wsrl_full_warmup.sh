#!/usr/bin/env bash
# WSRL baseline on antmaze-large-diverse-v2 — full warmup (5000 steps)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env antmaze-large-diverse-v2 \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 1_000_000 \
  --num_online_steps 200_000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --project bt-ccq \
  --group antmaze_large_diverse \
  --exp_name wsrl_full_warmup \
  $@
