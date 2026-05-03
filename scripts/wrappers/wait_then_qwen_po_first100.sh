#!/bin/bash
# Wait for Qwen PO remaining-100 (chain stage 1) python process to exit,
# then start Qwen PO first-100 v2 on GPU 1 (BEFORE Llama Base in the chain).
#
# Implementation: poll /tmp/gpu1_chain.log for "STEP 2/2" line which the chain
# wrapper prints when Qwen PO rem-100 finishes. Then kill the chain wrapper's
# next-stage Llama Base, run Qwen PO first-100 v2, then restart Llama Base.

# Actually simpler: wait for the existing Qwen PO rem-100 python PID to exit,
# then immediately preempt the chain by running Qwen PO first-100 v2 + Llama Base
# manually in sequence on GPU 1.

set -e
cd /root/autodl-tmp/ProactiveLLM

# Find the Qwen PO rem-100 python process PID
QWEN_PO_PID=$(ps -eo pid,cmd | grep "evaluate_multi_turn.*prompt_only.*remaining100" | grep -v grep | awk '{print $1}' | head -1)

if [ -z "$QWEN_PO_PID" ]; then
  echo "[$(date)] No Qwen PO rem-100 python process found — assuming it already finished."
else
  echo "[$(date)] Waiting for Qwen PO rem-100 PID=$QWEN_PO_PID to finish..."
  while kill -0 "$QWEN_PO_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date)] Qwen PO rem-100 finished."
fi

# Now the chain script will try to start Llama Base. Wait briefly then check.
sleep 30

# Find and kill any Llama Base process that the chain just started on GPU 1
LLAMA_BASE_PID=$(ps -eo pid,cmd | grep "evaluate_multi_turn.*llama.*base.*200" | grep -v grep | awk '{print $1}' | head -1)
if [ -n "$LLAMA_BASE_PID" ]; then
  echo "[$(date)] Found chain's Llama Base PID=$LLAMA_BASE_PID — killing it (will restart after Qwen PO first-100 v2)"
  kill "$LLAMA_BASE_PID" 2>/dev/null || true
  # Also kill the chain wrapper bash
  CHAIN_BASH=$(ps -eo pid,cmd | grep "gpu1_chain_qwenpo_llamabase" | grep -v grep | awk '{print $1}' | head -1)
  [ -n "$CHAIN_BASH" ] && kill "$CHAIN_BASH" 2>/dev/null || true
  sleep 5
fi

# Now run our re-sequenced chain: Qwen PO first-100 v2 → Llama Base 200
echo "[$(date)] Starting Qwen PO first-100 v2 on GPU $CUDA_VISIBLE_DEVICES"
bash /tmp/qwen_po_first100_v2.sh

echo
echo "[$(date)] Starting Llama Base N=200 v2 on GPU $CUDA_VISIBLE_DEVICES"
bash /tmp/llama_base_200.sh

echo
echo "[$(date)] === GPU 1 re-sequenced chain done ==="
