#!/bin/bash
# v33 v3 Qwen SFT (复用 Llama 同样 hparam)
set -e

cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KEEP_PREFIX=1
export LORA_ALPHA=32
export LORA_R=64

OUTPUT=models/v33_v3_qwen_sft

echo "[$(date)] v33 v3 Qwen SFT (apply Llama-validated recipe)"
echo "  base:        Qwen/Qwen2.5-7B-Instruct"
echo "  data:        prefs_v29_100states.jsonl"
echo "  hparam:      KEEP_PREFIX=1, alpha=32, r=64, epochs=3, LR=5e-5"
echo "  output:      $OUTPUT"
echo

python policy/train_sft_v33.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --output $OUTPUT \
  --epochs 3 --lr 5e-5

echo
echo "[$(date)] Qwen SFT done."
