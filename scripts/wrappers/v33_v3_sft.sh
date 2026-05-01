#!/bin/bash
# v33 v3: stronger SFT (epochs=3, LR=5e-5, alpha=32) to push Novice clarify rate up.
set -e

cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KEEP_PREFIX=1
export LORA_ALPHA=32
export LORA_R=64

OUTPUT=models/v33_v3_sft

echo "[$(date)] v33 v3 SFT (stronger)"
echo "  vs v33 v2: epochs 2→3, LR 2e-5→5e-5, alpha 16→32"
echo "  output:    $OUTPUT"
echo

python policy/train_sft_v33.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --output $OUTPUT \
  --epochs 3 \
  --lr 5e-5

echo
echo "[$(date)] v33 v3 done."
