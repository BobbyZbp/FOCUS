#!/usr/bin/env bash
# BT-CCQ Path B (WSRL-aligned) on antmaze-large-diverse-v2.
#
# Setup matches upstream WSRL exactly:
#   --agent btccq          (inherits SACAgent, uses SAC critic loss + BT gate
#                           after the offline -> online switch)
#   --config antmaze_wsrl  (10-Q ensemble, critic_subsample_size=2, LayerNorm
#                           on actor + critic, max_target_backup=True,
#                           4-hidden-layer critic, 2-hidden-layer policy,
#                           uniform std parameterization)
#   --num_offline_steps 1_000_000   (SAC offline pretraining as in WSRL repo's
#                                    launch_wsrl_finetune.sh; no CalQL restore)
#   --num_online_steps 200_000
#   --warmup_steps 5000     (full WSRL warmup)
#   --utd 4 --batch_size 1024
#   --reward_scale 10 --reward_bias -5
#
# BT-CCQ specifics:
#   --btccq_alpha 0.1       (positive-tail quantile level)
#   --btccq_w_out 0.5       (gate floor; raised from 0.2 to be conservative
#                            -- keep critic loss at >= 50% weight even on
#                            high-delta updates, so the gate cannot starve
#                            critic learning)
#   --btccq_calib_ratio 0.1
#
# BT-CCQ is wrapped from the SAC agent at the offline -> online transition
# (finetune.py training loop). Until then the agent trains as plain SAC,
# which gives the calibration step a meaningful "frozen offline reference"
# (the post-pretrain Q-function).
#
# No --resume_path: this run does its own SAC offline pretraining, exactly
# matching the upstream WSRL launch script's behavior.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

python finetune.py \
  --agent btccq \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env antmaze-large-diverse-v2 \
  --reward_scale 10.0 \
  --reward_bias -5.0 \
  --num_offline_steps 1_000_000 \
  --num_online_steps 200_000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --btccq_alpha 0.1 \
  --btccq_w_out 0.5 \
  --btccq_calib_ratio 0.1 \
  --project bt-ccq \
  --group antmaze_large_diverse \
  --exp_name btccq_full_warmup \
  $@
