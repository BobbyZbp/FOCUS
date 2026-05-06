# RunPod Setup — BT-CCQ Experiments

## 1. System deps (run as root)

```bash
apt-get update && apt-get install -y \
  libosmesa6-dev patchelf libgl1-mesa-glx libglfw3 libglew-dev \
  wget git
```

## 2. MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar -xzf /tmp/mujoco.tar.gz -C ~/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Add both exports to `~/.bashrc`.

## 3. Conda environment

```bash
conda create -n btccq python=3.9 -y
conda activate btccq
```

## 4. JAX with CUDA

```bash
pip install "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify:
```bash
python -c "import jax; print(jax.devices())"
# must show GPU, not CPU
```

## 5. WSRL Python stack

```bash
pip install \
  flax==0.7.5 chex==0.1.82 optax==0.1.5 distrax==0.1.2 \
  orbax-checkpoint==0.3.5 scipy==1.11.2 \
  ml_collections overrides wandb tqdm absl-py einops \
  mujoco-py==2.1.2.14
```

## 6. D4RL

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

## 7. Clone and verify

```bash
git clone git@github.com:BobbyZbp/ST-CCQ.git wsrl
cd wsrl

python - <<'PY'
import sys; sys.path.insert(0, ".")
from wsrl.agents import agents, BTCCQAgent
print("agents:", list(agents.keys()))
print("BTCCQAgent OK:", BTCCQAgent)
PY

python - <<'PY'
import d4rl, gym
env = gym.make("antmaze-medium-diverse-v2")
print("env OK:", env.observation_space.shape, env.action_space.shape)
ds = d4rl.qlearning_dataset(env)
print("dataset N:", ds["observations"].shape[0])
PY
```

Both must pass before running anything.

## 8. CalQL offline pretraining

```bash
python finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env antmaze-medium-diverse-v2 \
  --reward_scale 10.0 --reward_bias -5.0 \
  --num_offline_steps 1_000_000 --num_online_steps 0 \
  --batch_size 256 --utd 1 \
  --seed 0 \
  --save_dir ~/checkpoints/calql_antmaze_medium \
  --debug
```

Checkpoint saves every `--save_interval` steps (default 100k) under `~/checkpoints/`.
Find it with: `find ~/checkpoints -type d | sort | tail -5`

## 9. Smoke test — reduced BT-CCQ (seed 0, 1000 steps)

```bash
CKPT=~/checkpoints/calql_antmaze_medium/<run_dir>/checkpoint_1000000

python finetune.py \
  --agent btccq \
  --config experiments/configs/train_config.py:antmaze_btccq \
  --env antmaze-medium-diverse-v2 \
  --reward_scale 10.0 --reward_bias -5.0 \
  --num_offline_steps 1_000_000 --num_online_steps 1000 \
  --resume_path $CKPT \
  --warmup_steps 100 --utd 1 --batch_size 256 \
  --btccq_alpha 0.1 --btccq_w_out 0.2 --btccq_calib_ratio 0.1 \
  --log_interval 100 --eval_interval 1000 --n_eval_trajs 2 \
  --seed 0 --debug
```

**Expected output:**
```
[BT-CCQ calibration] {'q_hat': <finite>, 'btccq/calib_e_mean': <finite>, ...}
btccq/gate_frac  (early: 0.1–0.6)
btccq/gate_mean  (early: 0.6–0.95)
btccq/delta_mean (finite)
critic_loss      (finite)
```

**If `q_hat = 0`:** set `--btccq_alpha 0.05` or `--btccq_w_out 0.3`
**If `gate_frac = 0` always:** set `--btccq_alpha 0.2`
**If `gate_frac > 0.8` always:** set `--btccq_w_out 0.4` or `--btccq_alpha 0.05`

## 10. Full experiment matrix (after smoke test passes)

4 conditions × 3 seeds = 12 jobs

```bash
for SEED in 0 1 2; do
  # Full warmup WSRL
  bash experiments/scripts/antmaze/launch_wsrl_full_warmup.sh \
    --resume_path $CKPT --seed $SEED \
    --exp_name am_md_full_wsrl_s${SEED}

  # Full warmup BT-CCQ
  bash experiments/scripts/antmaze/launch_btccq_full_warmup.sh \
    --resume_path $CKPT --seed $SEED \
    --exp_name am_md_full_btccq_s${SEED}

  # Reduced warmup WSRL
  bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
    --resume_path $CKPT --seed $SEED \
    --exp_name am_md_red_wsrl_s${SEED}

  # Reduced warmup BT-CCQ
  bash experiments/scripts/antmaze/launch_btccq_reduced_warmup.sh \
    --resume_path $CKPT --seed $SEED \
    --exp_name am_md_red_btccq_s${SEED}
done
```

Run with `tmux` or `screen` — each job takes ~2–4h on 1 GPU.

## 11. Key metrics to track (via wandb)

| Metric | Good sign for BT-CCQ |
|---|---|
| `btccq/gate_frac` early | > 0 (gate is firing) |
| `btccq/gate_frac` later | decreasing (less OOD as agent improves) |
| `btccq/gate_mean` | < 1.0 early, → 1.0 later |
| `btccq/delta_mean` | finite, decreasing |
| `predicted_qs` (reduced BT-CCQ vs WSRL) | smaller gap = less Q collapse |
| `evaluation/average_return` | BT-CCQ ≥ WSRL (main claim) |
| `evaluation/success_rate` | BT-CCQ ≥ WSRL (antmaze sparse) |
