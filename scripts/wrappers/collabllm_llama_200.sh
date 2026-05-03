#!/bin/bash
# CollabLLM Llama-3.1-8B-Instruct released checkpoint, full 200-state eval.
# Run AFTER sanity passes (verifies v2 classifier sees Clarify/Execute mix).
# Output: outputs/eval_collabllm_llama_200.json
# ETA: ~5h on single 4090 (200 states × 3 personas)
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1          # checkpoint cached after sanity run
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_collabllm_llama_200.json

echo "[$(date)] CollabLLM Llama 200 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --no_lora --prompt_only \
  --base_model "collabllm/CollabLLM-code-Llama-3.1-8B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
