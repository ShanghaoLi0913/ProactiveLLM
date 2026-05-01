#!/bin/bash
# Qwen Clarify-first remaining-100 (state 101-200) with v2 + fixed template.
# Run on GPU 1 (second stage of GPU1 chain, after PO finishes).
# Output: outputs/eval_v29_qwen_clarify_first_remaining100_ft.json
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_qwen_clarify_first_remaining100_ft.json

echo "[$(date)] Qwen Clarify-first remaining-100 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --always_clarify 1 \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200_remaining100.jsonl \
  --output "$OUT" \
  --max_turns 2 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
