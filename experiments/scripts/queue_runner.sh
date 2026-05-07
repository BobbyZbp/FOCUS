#!/bin/bash
GPU_ID=$1
TASK_FILE=$2
cd /workspace/wsrl
source /root/btccq_env/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=$GPU_ID
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=osmesa
LOG_DIR=/workspace/logs/queue_gpu${GPU_ID}
mkdir -p $LOG_DIR
echo "[$(date)] [GPU $GPU_ID] Queue starting"
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$(echo "$line" | tr -d '[:space:]')" ] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    TASK_NAME=$(echo "$line" | grep -oP '(?<=--exp_name )\S+' || echo "task_$(date +%s)")
    LOG=$LOG_DIR/${TASK_NAME}.log
    echo "[$(date)] [GPU $GPU_ID] >>> Starting [$TASK_NAME]"
    eval "$line" 2>&1 | tee "$LOG"
    echo "[$(date)] [GPU $GPU_ID] <<< DONE [$TASK_NAME] (exit ${PIPESTATUS[0]})"
    sleep 5
done < "$TASK_FILE"
echo "[$(date)] [GPU $GPU_ID] All tasks complete."
