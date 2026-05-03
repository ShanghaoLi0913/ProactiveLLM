#!/bin/bash
# Launch 3 GPU chains for bf16 rerun.
# Distributes 10 methods across GPUs to balance load:
#   GPU 0: heavy DPO + light methods (TactfulLLM Qwen + Direct + Base)
#   GPU 1: heavy DPO + light (TactfulLLM Llama + CF Llama + CF Qwen)
#   GPU 2: PO methods + remaining lights (PO both, Direct/Base remaining)
#
# Estimated load (bf16 on 4090, 200 state, 5 candidates):
#   DPO multi-turn (Novice 8-turn): ~3-4h
#   PO multi-turn:                  ~1.5h
#   Base prompt-only multi-turn:    ~1.5h  (may multi-turn for some persona)
#   CF (forced 2-turn):             ~1h
#   Direct (single turn):           ~30min

set -u

# Wait for current GPU jobs to finish
echo "Waiting for current jobs to free GPUs..."

# GPU 2: Llama Busy decoupled (~10min remaining)
PID_DEC=$(pgrep -f "llama_smart_execute_decoupled" | head -1)
echo "  GPU 2 decoupled PID: $PID_DEC"

# GPU 0: Qwen Busy smart (~1h remaining)
PID_QSMART=$(pgrep -f "Qwen.*Busy-Developer" | head -1)
echo "  GPU 0 Qwen smart PID: $PID_QSMART"

# GPU 1: CollabLLM bf16 (~3h remaining)
PID_COLLAB=$(pgrep -f "collabllm" | head -1)
echo "  GPU 1 CollabLLM PID: $PID_COLLAB"

# Wait helper
wait_pid() {
    local pid=$1
    [ -z "$pid" ] && return
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "  PID $pid done at $(date)"
}

# Launch GPU 2 chain first (decoupled finishes soonest ~10min)
(
    wait_pid "$PID_DEC"
    sleep 30  # GPU cleanup
    bash /tmp/bf16_rerun_chain_gpu.sh 2 "llama_direct,llama_base,qwen_direct,qwen_base,qwen_po"
) > /tmp/launch_gpu2.log 2>&1 &
echo "GPU 2 launcher PID: $!"

# Launch GPU 0 chain after Qwen Busy smart finishes (~1h)
(
    wait_pid "$PID_QSMART"
    sleep 30
    bash /tmp/bf16_rerun_chain_gpu.sh 0 "qwen_dpo,llama_po"
) > /tmp/launch_gpu0.log 2>&1 &
echo "GPU 0 launcher PID: $!"

# Launch GPU 1 chain after CollabLLM finishes (~3h)
(
    wait_pid "$PID_COLLAB"
    sleep 30
    bash /tmp/bf16_rerun_chain_gpu.sh 1 "llama_dpo,llama_cf,qwen_cf"
) > /tmp/launch_gpu1.log 2>&1 &
echo "GPU 1 launcher PID: $!"

echo
echo "All 3 chain launchers started. Will execute as GPUs free up."
echo "Monitor: tail -f /tmp/bf16_chain_gpu{0,1,2}.log"
