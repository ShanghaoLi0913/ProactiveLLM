#!/bin/bash
# Qwen Direct-execution remaining-100 (state 101-200) with v2 + fixed Execute template.
# Caller sets CUDA_VISIBLE_DEVICES (intended: GPU 0).
# Output: outputs/eval_v29_qwen_direct_execution_remaining100_ft.json
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_qwen_direct_execution_remaining100_ft.json

echo "[$(date)] Qwen Direct remaining-100 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --direct_execution \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200_remaining100.jsonl \
  --output "$OUT" \
  --max_turns 1 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
