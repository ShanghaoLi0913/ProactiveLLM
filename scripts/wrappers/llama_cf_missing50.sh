#!/bin/bash
# Llama Clarify-first on the 50 states in eval_200 missing from eval_150extra.
# After completion, merge with eval_v29_clarify_first_150extra.json → CF on full eval_200.
# Designed to launch on GPU 2 (or whichever GPU is free) after Llama PO 200 finishes.
# Output: outputs/eval_v29_clarify_first_missing50.json
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_clarify_first_missing50.json
SEED=data/seeds/test_states_v29_eval_200_missing_from_150extra.jsonl

echo "[$(date)] Llama Clarify-first missing-50 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  seed:   $SEED ($(wc -l < $SEED) states)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --always_clarify 1 \
  --base_model "meta-llama/Llama-3.1-8B-Instruct" \
  --test_states "$SEED" \
  --output "$OUT" \
  --max_turns 2 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
