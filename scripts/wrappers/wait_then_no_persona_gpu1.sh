#!/bin/bash
# Wait for current Llama Few-shot eval (PID 864655) to finish, then launch
# no_persona ablation rerun on GPU 1 with pre-fix prompt.
set -u
cd /root/autodl-tmp/ProactiveLLM

WATCH_PID=864655
LOG=logs/wait_then_no_persona_gpu1.log
mkdir -p logs

echo "[$(date)] watching PID $WATCH_PID (Llama Few-shot)" | tee -a $LOG
while kill -0 $WATCH_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] PID $WATCH_PID finished, sleeping 30s before launching no_persona on GPU 1" | tee -a $LOG
sleep 30
echo "[$(date)] launching no_persona ablation rerun (pre-fix) on GPU 1" | tee -a $LOG
bash scripts/wrappers/ablation_rerun_prefix.sh 1 no_persona 2>&1 | tee -a $LOG
echo "[$(date)] no_persona rerun done" | tee -a $LOG
