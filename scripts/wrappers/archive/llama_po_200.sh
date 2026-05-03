#!/bin/bash
# Llama Prompt-only N=200 eval (v2 classifier).
# ETA ~12h compute.
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_llama_prompt_only_200_v2.json

echo "[$(date)] Llama Prompt-only N=200 v2 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --prompt_only \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
