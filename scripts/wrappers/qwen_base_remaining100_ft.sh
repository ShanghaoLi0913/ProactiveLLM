#!/bin/bash
# Qwen Base remaining-100 (state 101-200) with v2 classifier + fixed Execute template.
# Run on GPU 0. Output: outputs/eval_v29_qwen_base_remaining100_ft.json
# Resume-aware: re-run after interruption picks up from last completed state.
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_qwen_base_remaining100_ft.json

echo "[$(date)] Qwen Base remaining-100 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200_remaining100.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
