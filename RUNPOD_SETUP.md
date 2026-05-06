# RunPod Setup — BT-CCQ Experiments (verified)

This guide replicates the working environment used for BT-CCQ experiments.
Tested on RunPod with NVIDIA RTX 4090, Ubuntu 22.04, CUDA 12.4.

**Total setup time: ~15 minutes.** Most of it is `pip install` waiting.

---

## What you need before starting

- A RunPod pod with **NVIDIA GPU** (RTX 4090 or better) and **CUDA 12.x driver**
- SSH access to the pod
- ~30GB free disk on `/root` (the local overlay, NOT `/workspace` which is network-mounted)

Verify the pod has GPU:
```bash
nvidia-smi
```
Must show your GPU. If not, you got a CPU pod — terminate and re-create.

---

## Step 0. SSH in and create a tmux session

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_rsa
tmux new -s install
```

Everything below runs inside this tmux. If SSH drops, reconnect and `tmux attach -t install`.

---

## Step 1. System packages (~2 min)

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  python3.10 python3.10-venv python3.10-dev \
  build-essential gcc g++ \
  libosmesa6-dev patchelf libgl1-mesa-glx libglfw3 libglew-dev \
  wget git tmux unzip curl
```

---

## Step 2. MuJoCo 2.1.0 (~30 s)

```bash
rm -rf /root/.mujoco/mujoco210 /tmp/mujoco.tar.gz
mkdir -p /root/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar --no-same-owner -xzf /tmp/mujoco.tar.gz -C /root/.mujoco

# Verify
ls /root/.mujoco/mujoco210/bin/libmujoco210.so
```

**Note:** the `tar: Cannot change ownership` warnings are harmless — the files extract fine. The `--no-same-owner` flag suppresses them.

---

## Step 3. Python 3.10 venv on local disk (~10 s)

**Critical:** create the venv in `/root` (local overlay), NOT `/workspace` (network-mounted, can disappear).

```bash
rm -rf /root/btccq_env
python3.10 -m venv /root/btccq_env
source /root/btccq_env/bin/activate
which python   # must be /root/btccq_env/bin/python
python --version  # must be 3.10.x

python -m pip install --upgrade pip setuptools wheel
```

---

## Step 4. JAX with CUDA (~3 min)

```bash
pip install --no-cache-dir --timeout 600 \
  "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

This pulls ~3 GB of NVIDIA wheels (cublas, cudnn, etc). If it stalls for >5 min, Ctrl-C and retry.

---

## Step 5. Fix cuDNN version mismatch (~30 s)

`jaxlib==0.4.20+cuda12.cudnn89` requires **cuDNN 8.x**, but the latest `nvidia-cudnn-cu12` is 9.x. Force-downgrade:

```bash
pip install --no-cache-dir --force-reinstall "nvidia-cudnn-cu12<9"
```

---

## Step 6. WSRL stack (~3 min)

```bash
pip install --no-cache-dir --timeout 600 \
  "numpy==1.26.4" "scipy==1.11.2" \
  "flax==0.7.5" "chex==0.1.82" "optax==0.1.5" "distrax==0.1.2" \
  "orbax-checkpoint==0.3.5" \
  "ml_collections" "overrides" "wandb" "tqdm" "absl-py" "einops" \
  "Cython<3" "mujoco-py==2.1.2.14" "mujoco==3.1.6"
```

---

## Step 7. Gym + D4RL (~2 min)

```bash
pip install --no-cache-dir "gym==0.23.1"
pip install --no-cache-dir git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

---

## Step 8. Persist environment variables in venv activate

This makes JAX find cuDNN and MuJoCo automatically every time you `source activate`.
**Without this, JAX will silently fall back to CPU on every new shell.**

```bash
cat >> /root/btccq_env/bin/activate <<'EOF'

# BT-CCQ / JAX CUDA libs
export LD_LIBRARY_PATH=$(find /root/btccq_env -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH

# MuJoCo (for d4rl antmaze)
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Suppress noisy d4rl import warnings (Flow/CARLA/Bullet not installed)
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF

source /root/btccq_env/bin/activate
```

---

## Step 9. Verify everything (3 checks)

### 9a. JAX sees GPU
```bash
python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```
**Must show:**
```
backend: gpu
devices: [cuda(id=0)]
```
If it shows `CpuDevice`, something is broken — go back to Step 5.

### 9b. D4RL AntMaze loads
First-time mujoco_py import will Cython-compile (~30 s of compiler output), this is normal.

```bash
python - <<'PY'
import d4rl, gym
env = gym.make("antmaze-large-diverse-v2")
print("env OK:", env.observation_space.shape, env.action_space.shape)
ds = d4rl.qlearning_dataset(env)
print("dataset OK:", ds["observations"].shape)
PY
```
**Must show:**
```
env OK: (29,) (8,)
dataset OK: (~999000, 29)
```

### 9c. BT-CCQ agent registers
```bash
cd /workspace
git clone https://github.com/BobbyZbp/ST-CCQ.git wsrl 2>/dev/null || (cd wsrl && git pull)
cd /workspace/wsrl

python - <<'PY'
import sys; sys.path.insert(0, ".")
from wsrl.agents import agents, BTCCQAgent
print("agents:", list(agents.keys()))
print("BTCCQAgent:", BTCCQAgent)
PY
```
**Must show:**
```
agents: ['bc', 'btccq', 'iql', 'cql', 'calql', 'sac']
BTCCQAgent: <class 'wsrl.agents.btccq.BTCCQAgent'>
```

---

## Step 10. Smoke test — 1000-step CalQL (~2 min)

Confirms the full training pipeline works end-to-end before you commit to a 5-hour run.

```bash
bash experiments/scripts/antmaze/launch_calql_finetune.sh \
  --env antmaze-large-diverse-v2 \
  --num_offline_steps 1000 \
  --num_online_steps 0 \
  --save_dir /workspace/checkpoints/smoke_test \
  --debug
```

Expected end-of-output:
```
1000/1000 [01:30<00:00, 11.01it/s]
Saved checkpoint to /workspace/checkpoints/smoke_test/.../checkpoint_1000
```

If you see this, you're ready to launch real training.

---

## Step 11. Real training — open a new tmux per job

Don't run training inside the install tmux. Open a dedicated session per job so you can monitor independently.

```bash
# Detach install session: Ctrl-b, d
tmux new -s calql_large_play

source /root/btccq_env/bin/activate
cd /workspace/wsrl

bash experiments/scripts/antmaze/launch_calql_finetune.sh \
  --env antmaze-large-play-v2 \
  --num_offline_steps 1_000_000 \
  --num_online_steps 0 \
  --save_dir /workspace/checkpoints/calql_large_play_seed0 \
  --save_interval 200000 \
  --eval_interval 1000000 \
  --log_interval 10000 \
  --seed 0 \
  --debug

# After tqdm starts: Ctrl-b, d to detach
```

Monitor from a separate window:
```bash
tmux attach -t calql_large_play   # see live progress
nvidia-smi                        # see GPU usage (should be ~5-15 GB)
ls /workspace/checkpoints/calql_large_play_seed0/wsrl/*/   # checkpoints landing
```

---

## After CalQL pretrain finishes — the experiment matrix

For each environment (`antmaze-large-diverse-v2`, `antmaze-large-play-v2`):
For each seed (0, 1, 2):
  - Reduced-warmup WSRL  (control)
  - Reduced-warmup BT-CCQ (test)
  - Full-warmup WSRL     (control)
  - Full-warmup BT-CCQ   (test)

Each finetune ~2h on a 4090.

```bash
# Find the checkpoint path:
CKPT=$(ls -d /workspace/checkpoints/calql_large_diverse_seed0/wsrl/*/checkpoint_1000000)
echo $CKPT

# Reduced WSRL
bash experiments/scripts/antmaze/launch_wsrl_reduced_warmup.sh \
  --env antmaze-large-diverse-v2 \
  --resume_path $CKPT --seed 0 --debug

# Reduced BT-CCQ
bash experiments/scripts/antmaze/launch_btccq_reduced_warmup.sh \
  --env antmaze-large-diverse-v2 \
  --resume_path $CKPT --seed 0 --debug
```

---

## Troubleshooting

### `jax.devices()` shows CpuDevice
1. Check `nvidia-smi` works inside the venv
2. Check cuDNN version: `pip show nvidia-cudnn-cu12` — must be 8.x not 9.x
3. Check LD_LIBRARY_PATH: `echo $LD_LIBRARY_PATH` must contain `nvidia/cudnn/lib`
4. Re-run Step 5 + Step 8

### `EGL_NOT_INITIALIZED` / `eglQueryString` errors
The launch script is using EGL instead of OSMesa. RunPod doesn't have EGL.
```bash
sed -i 's/PYOPENGL_PLATFORM=egl/PYOPENGL_PLATFORM=osmesa/' experiments/scripts/antmaze/launch_*.sh
sed -i 's/MUJOCO_GL=egl/MUJOCO_GL=osmesa/' experiments/scripts/antmaze/launch_*.sh
```

### `ModuleNotFoundError: d4rl` on Python 3.11
You're not in the venv. Run `source /root/btccq_env/bin/activate`. Verify `which python` is `/root/btccq_env/bin/python`.

### `pip install d4rl` fails with "requires Python <3.11"
Same — you're using system Python 3.11. Activate the venv first.

### tmux session disappears after SSH reconnect
Use `tmux ls` to list. If gone, the pod restarted — `tmux new -s NAME` and re-launch.

### Training is very slow (<30 it/s)
- Check GPU utilization: `nvidia-smi` should show >50%
- Confirm `--utd 1` (default), not high UTD
- Reduce `eval_interval` calls — eval blocks GPU

### Disk full
`/workspace` has tons of space (network mount). `/root` is ~100 GB local.
Move checkpoints: `mv /workspace/checkpoints /workspace/old_checkpoints`

---

## File locations recap

| Path | What | Persistence |
|---|---|---|
| `/root/btccq_env/` | Python venv | Local overlay, persists across SSH reconnect |
| `/root/.mujoco/mujoco210/` | MuJoCo install | Local overlay |
| `/workspace/wsrl/` | Code (git clone) | Network mount, persists across pod restarts |
| `/workspace/checkpoints/` | Saved checkpoints | Network mount, persists across pod restarts |
| `/tmp/` | pip caches, build artifacts | Wiped on pod restart |

**If pod restarts:** venv at `/root/btccq_env` survives. Just re-`source` it.
**If pod is destroyed:** everything except `/workspace/*` is gone — re-run Steps 1-8.

---

## One-shot setup script (for next time)

Once Steps 1-8 are validated, paste this entire block to a new pod for fully automated setup:

```bash
#!/usr/bin/env bash
set -e

# Step 1: system
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev \
  build-essential gcc g++ libosmesa6-dev patchelf libgl1-mesa-glx \
  libglfw3 libglew-dev wget git tmux unzip curl

# Step 2: MuJoCo
rm -rf /root/.mujoco/mujoco210 /tmp/mujoco.tar.gz
mkdir -p /root/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar --no-same-owner -xzf /tmp/mujoco.tar.gz -C /root/.mujoco

# Step 3: venv
rm -rf /root/btccq_env
python3.10 -m venv /root/btccq_env
source /root/btccq_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Step 4: JAX CUDA
pip install --no-cache-dir --timeout 600 \
  "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Step 5: fix cuDNN
pip install --no-cache-dir --force-reinstall "nvidia-cudnn-cu12<9"

# Step 6: WSRL stack
pip install --no-cache-dir --timeout 600 \
  "numpy==1.26.4" "scipy==1.11.2" \
  "flax==0.7.5" "chex==0.1.82" "optax==0.1.5" "distrax==0.1.2" \
  "orbax-checkpoint==0.3.5" \
  "ml_collections" "overrides" "wandb" "tqdm" "absl-py" "einops" \
  "Cython<3" "mujoco-py==2.1.2.14" "mujoco==3.1.6"

# Step 7: gym + d4rl
pip install --no-cache-dir "gym==0.23.1"
pip install --no-cache-dir git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# Step 8: persist env vars
cat >> /root/btccq_env/bin/activate <<'EOF'
export LD_LIBRARY_PATH=$(find /root/btccq_env -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
source /root/btccq_env/bin/activate

# Step 9: verify
python -c "import jax; print('GPU:', jax.devices())"

# Step 10: clone repo
cd /workspace
git clone https://github.com/BobbyZbp/ST-CCQ.git wsrl 2>/dev/null || (cd wsrl && git pull)
cd /workspace/wsrl
python -c "from wsrl.agents import BTCCQAgent; print('BTCCQ OK:', BTCCQAgent)"

echo "===== Setup complete ====="
echo "Next: tmux new -s calql && bash experiments/scripts/antmaze/launch_calql_finetune.sh ..."
```

Save as `setup_runpod.sh`, then on a new pod:
```bash
curl -sL https://raw.githubusercontent.com/BobbyZbp/ST-CCQ/main/setup_runpod.sh | bash
```
