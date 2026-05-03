#!/bin/bash
set -e
cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v29_qwen_ideal_disclosed_v2_200.json
echo "[$(date)] Qwen ideal_disclosed v2 200 (GPU=$CUDA_VISIBLE_DEVICES)"

python eval/evaluate_multi_turn_persona.py \
  --no_lora --ideal_disclosed \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --output "$OUT" \
  --max_turns 1 \
  --llm_model gpt-4o-mini --pass_at_k 1 5
echo "[$(date)] Done: $OUT"
