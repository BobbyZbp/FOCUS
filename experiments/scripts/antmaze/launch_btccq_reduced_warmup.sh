#!/usr/bin/env bash
# BT-CCQ on antmaze-large-diverse-v2 — reduced warmup (1250 steps = 25% of full)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python finetune.py \
  --agent btccq \
  --config experiments/configs/train_config.py:antmaze_btccq \
  --env antmaze-large-diverse-v2 \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 1_000_000 \
  --num_online_steps 200_000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 1250 \
  --btccq_alpha 0.1 \
  --btccq_w_out 0.2 \
  --btccq_calib_ratio 0.1 \
  --project bt-ccq \
  --group antmaze_large_diverse \
  --exp_name btccq_reduced_warmup \
  $@
