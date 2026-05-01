#!/bin/bash
# v33 Stage 2: DPO refinement on top of v33_v3_sft.
# Uses INIT_ADAPTER to continue training the SFT LoRA.
set -e

cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KEEP_PREFIX=1
export LORA_ALPHA=32
export LORA_R=64
export INIT_ADAPTER=models/v33_v3_sft   # ← key: continue from SFT

OUTPUT=models/v33_v3_dpo

echo "[$(date)] v33 v3 Stage 2: DPO refinement on top of SFT"
echo "  init_adapter: $INIT_ADAPTER"
echo "  KEEP_PREFIX:  $KEEP_PREFIX"
echo "  LoRA r/alpha: 64/32 (matches SFT)"
echo "  beta:         0.1 (standard DPO)"
echo "  epochs:       3"
echo "  output:       $OUTPUT"
echo

python policy/train_dpo.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --output $OUTPUT \
  --epochs 3 \
  --beta 0.1

echo
echo "[$(date)] v33 v3 DPO refinement done."
