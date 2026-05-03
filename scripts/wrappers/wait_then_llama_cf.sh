#!/bin/bash
# Watch the Llama PO 200 python PID; once it exits, launch Llama CF on missing-50.
# CF eval runs on GPU 2 (the GPU PO frees up). Polls every 60s.
# Logs: /tmp/llama_cf_missing50.log + this script's own log /tmp/wait_then_llama_cf.log

set -u
WATCH_PID=159325         # Llama PO v2 200 python PID
WRAPPER=/root/autodl-tmp/ProactiveLLM/scripts/wrappers/llama_cf_missing50.sh
GPU=2
EVAL_LOG=/tmp/llama_cf_missing50.log
SELF_LOG=/tmp/wait_then_llama_cf.log

exec >> "$SELF_LOG" 2>&1
echo "==== started $(date) — watching PID $WATCH_PID ===="

# Wait for the PO python process to exit
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date)] PID $WATCH_PID exited."

# Sanity: confirm GPU 2 is idle (no python eating its memory)
sleep 10
GPU2_USE=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "[$(date)] GPU $GPU memory.used = ${GPU2_USE} MiB"
if [ "${GPU2_USE:-99999}" -gt 2000 ]; then
  echo "[$(date)] WARN: GPU $GPU still has ${GPU2_USE} MiB used — waiting 60s more"
  sleep 60
fi

# Launch Llama CF missing-50 on GPU 2
echo "[$(date)] launching: CUDA_VISIBLE_DEVICES=$GPU bash $WRAPPER"
CUDA_VISIBLE_DEVICES=$GPU bash "$WRAPPER" > "$EVAL_LOG" 2>&1 &
NEW_PID=$!
echo "[$(date)] launched, wrapper bash PID=$NEW_PID, log=$EVAL_LOG"

# Wait for it to finish, then log final state
wait "$NEW_PID"
RC=$?
echo "[$(date)] wrapper exited rc=$RC"
echo "==== done $(date) ===="
