#!/bin/bash
# Qwen v33 v3 DPO refinement (v2: epochs=1 to avoid over-fit collapse)
set -e
cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KEEP_PREFIX=1
export LORA_ALPHA=32
export LORA_R=64
export INIT_ADAPTER=models/v33_v3_qwen_sft

OUTPUT=models/v33_v3_qwen_dpo_v2

echo "[$(date)] v33 v3 Qwen DPO refinement v2 (epochs=1 to avoid over-fit)"
echo "  init_adapter: $INIT_ADAPTER"
echo "  output: $OUTPUT"
echo

python policy/train_dpo.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --output $OUTPUT \
  --epochs 1 --beta 0.1

echo "[$(date)] Qwen DPO v2 done."
