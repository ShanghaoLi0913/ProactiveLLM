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

SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--collabllm--CollabLLM-code-Llama-3.1-8B-Instruct/snapshots/4de9e9b4061a520b40c34245b0b0f20a35c4197a

# CollabLLM is a LoRA adapter; use --model_dir to load it on top of Llama base.
# --prompt_only triggers the persona-aware natural-language action selector
# (instead of the DPO state-rendering selector, which CollabLLM doesn't recognize).
python eval/evaluate_multi_turn_persona.py \
  --model_dir "$SNAPSHOT" \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --prompt_only \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
