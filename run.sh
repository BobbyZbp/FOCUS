#!/usr/bin/env bash
# Wrapper that sets the correct Python env + MuJoCo paths, then runs finetune.py.
# Usage: ./run.sh [finetune.py args...]
export MUJOCO_PY_MUJOCO_PATH=/home/bobby/TranQil/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/bobby/TranQil/.mujoco/mujoco210/bin:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON=/home/bobby/TranQil/.micromamba/root/envs/tranqil-qt/bin/python3.9

exec $PYTHON "$@"
