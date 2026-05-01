#!/bin/bash
# v33 v3 Qwen 5-state SFT eval
set -e
cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2

echo "[$(date)] v33 v3 Qwen SFT 5-state eval"
python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_qwen_sft \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 5 \
  --output outputs/eval_v33_v3_qwen_sft_5.json \
  --max_turns 7
echo "[$(date)] Done."
