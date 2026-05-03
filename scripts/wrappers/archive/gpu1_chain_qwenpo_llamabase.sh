#!/bin/bash
# GPU 1 chain: Qwen PO rem-100 (~5h) → Llama Base N=200 v2 (~10h)
set -e

cd /root/autodl-tmp/ProactiveLLM
echo "[$(date)] === GPU 1 chain start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ==="
echo

echo "[$(date)] STEP 1/2: Qwen PO remaining-100"
bash scripts/wrappers/qwen_po_remaining100_ft.sh

echo
echo "[$(date)] STEP 2/2: Llama Base N=200 v2"
bash /tmp/llama_base_200.sh

echo
echo "[$(date)] === GPU 1 chain done ==="
