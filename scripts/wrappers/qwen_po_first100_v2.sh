#!/bin/bash
# Re-run Qwen Prompt-only on the FIRST 100 states of eval_200 with v2 classifier.
# Replaces the v1-era first-100 (Apr 26) so the merged N=200 is classifier-consistent.
# ETA ~5h compute.
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_qwen_prompt_only_first100_v2.json

echo "[$(date)] Qwen PO first-100 v2 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --prompt_only \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200_first100.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
