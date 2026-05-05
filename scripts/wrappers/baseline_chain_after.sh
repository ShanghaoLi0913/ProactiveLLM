#!/bin/bash
# Wait for currently-running PIDs to finish, then launch chain of baseline evals.
# Allocation:
#   GPU 1 (Few-shot Llama PID 864655) → Few-shot Qwen
#   GPU 2 (Ablation no_uncertainty PID 595947) → Random Llama → Random Qwen
set -u

cd /root/autodl-tmp/ProactiveLLM
mkdir -p logs

wait_pid() {
    local pid=$1
    [ -z "$pid" ] && return
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "  PID $pid done at $(date)"
}

# GPU 1 chain: Few-shot Qwen (after Few-shot Llama done)
(
    wait_pid 864655
    sleep 30
    echo "[$(date)] GPU 1: starting Few-shot Qwen"
    bash scripts/wrappers/few_shot_persona_eval.sh 1 qwen
) > logs/baseline_chain_gpu1.log 2>&1 &
echo "GPU 1 chain launcher PID: $!"

# GPU 2 chain: Random Llama → Random Qwen (after ablation done)
(
    wait_pid 595947
    sleep 30
    echo "[$(date)] GPU 2: starting Random Llama"
    bash scripts/wrappers/random_policy_eval.sh 2 llama
    echo "[$(date)] GPU 2: starting Random Qwen"
    bash scripts/wrappers/random_policy_eval.sh 2 qwen
) > logs/baseline_chain_gpu2.log 2>&1 &
echo "GPU 2 chain launcher PID: $!"

echo
echo "Chains queued. Will fire when current GPU 1 / GPU 2 jobs finish."
