#!/bin/bash
# CollabLLM Llama-3.1-8B-Instruct released checkpoint, 5-state sanity.
# Goal: confirm v2 classifier sees a mix of Clarify/Execute (NOT degenerate all-1-turn).
# Output: outputs/eval_collabllm_llama_sanity5.json
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=0          # may need to download collabllm/* first run
export TRANSFORMERS_OFFLINE=0
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_collabllm_llama_sanity5.json

echo "[$(date)] CollabLLM Llama sanity-5 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --prompt_only \
  --base_model "collabllm/CollabLLM-code-Llama-3.1-8B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 5 \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
