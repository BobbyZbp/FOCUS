#!/usr/bin/env bash
# One-shot RunPod setup for BT-CCQ experiments.
# Tested on Ubuntu 22.04 + CUDA 12.4 + RTX 4090.
# Usage:
#   curl -sL https://raw.githubusercontent.com/BobbyZbp/ST-CCQ/main/setup_runpod.sh | bash
#   OR
#   bash setup_runpod.sh

set -e
echo "===== BT-CCQ RunPod setup starting ====="

# ---------- Step 1: system packages ----------
echo "[1/9] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.10 python3.10-venv python3.10-dev \
  build-essential gcc g++ libosmesa6-dev patchelf libgl1-mesa-glx \
  libglfw3 libglew-dev wget git tmux unzip curl

# ---------- Step 2: MuJoCo 2.1.0 ----------
echo "[2/9] Installing MuJoCo 2.1.0..."
rm -rf /root/.mujoco/mujoco210 /tmp/mujoco.tar.gz
mkdir -p /root/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar --no-same-owner -xzf /tmp/mujoco.tar.gz -C /root/.mujoco

# ---------- Step 3: Python 3.10 venv on local disk ----------
echo "[3/9] Creating Python 3.10 venv at /root/btccq_env..."
rm -rf /root/btccq_env
python3.10 -m venv /root/btccq_env
source /root/btccq_env/bin/activate
python -m pip install --upgrade pip setuptools wheel -q

# ---------- Step 4: JAX with CUDA ----------
echo "[4/9] Installing JAX with CUDA 12 (~3GB of NVIDIA wheels)..."
pip install --no-cache-dir --timeout 600 -q \
  "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------- Step 5: fix cuDNN version ----------
echo "[5/9] Downgrading nvidia-cudnn-cu12 to 8.x for jaxlib compatibility..."
pip install --no-cache-dir --force-reinstall -q "nvidia-cudnn-cu12<9"

# ---------- Step 6: WSRL stack ----------
echo "[6/9] Installing WSRL Python stack..."
pip install --no-cache-dir --timeout 600 -q \
  "numpy==1.26.4" "scipy==1.11.2" \
  "flax==0.7.5" "chex==0.1.82" "optax==0.1.5" "distrax==0.1.2" \
  "orbax-checkpoint==0.3.5" \
  "ml_collections" "overrides" "wandb" "tqdm" "absl-py" "einops" \
  "Cython<3" "mujoco-py==2.1.2.14" "mujoco==3.1.6"

# ---------- Step 7: gym + d4rl ----------
echo "[7/9] Installing gym 0.23.1 + d4rl..."
pip install --no-cache-dir -q "gym==0.23.1"
pip install --no-cache-dir -q git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# ---------- Step 8: persist env vars in venv activate ----------
echo "[8/9] Persisting environment variables in venv activate..."
cat >> /root/btccq_env/bin/activate <<'EOF'

# BT-CCQ / JAX CUDA libs
export LD_LIBRARY_PATH=$(find /root/btccq_env -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH

# MuJoCo (for d4rl antmaze)
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Suppress noisy d4rl import warnings
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
source /root/btccq_env/bin/activate

# ---------- Step 9: clone repo and verify ----------
echo "[9/9] Cloning repo and running 3 verification checks..."
mkdir -p /workspace
cd /workspace
git clone https://github.com/BobbyZbp/ST-CCQ.git wsrl 2>/dev/null || (cd wsrl && git pull)
cd /workspace/wsrl

echo ""
echo "----- Check 1: JAX GPU -----"
python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu", "FAIL: JAX did not see GPU"
print("PASS")
PY

echo ""
echo "----- Check 2: D4RL AntMaze (first run will Cython-compile mujoco_py, ~30s) -----"
python - <<'PY'
import d4rl, gym
env = gym.make("antmaze-large-diverse-v2")
print("env shapes:", env.observation_space.shape, env.action_space.shape)
ds = d4rl.qlearning_dataset(env)
print("dataset N:", ds["observations"].shape[0])
assert ds["observations"].shape[0] > 100000, "FAIL: dataset too small"
print("PASS")
PY

echo ""
echo "----- Check 3: BT-CCQ agent registered -----"
python - <<'PY'
import sys; sys.path.insert(0, ".")
from wsrl.agents import agents, BTCCQAgent
print("agents:", list(agents.keys()))
print("BTCCQAgent:", BTCCQAgent)
assert "btccq" in agents, "FAIL: btccq not in registry"
print("PASS")
PY

echo ""
echo "===== Setup complete ====="
echo ""
echo "Next steps:"
echo "  source /root/btccq_env/bin/activate"
echo "  cd /workspace/wsrl"
echo "  tmux new -s calql"
echo "  bash experiments/scripts/antmaze/launch_calql_finetune.sh \\"
echo "    --env antmaze-large-play-v2 \\"
echo "    --num_offline_steps 1_000_000 --num_online_steps 0 \\"
echo "    --save_dir /workspace/checkpoints/calql_large_play_seed0 \\"
echo "    --save_interval 200000 --eval_interval 1000000 --log_interval 10000 \\"
echo "    --seed 0 --debug"
echo ""
echo "Then Ctrl-b d to detach. Training will run in background."
