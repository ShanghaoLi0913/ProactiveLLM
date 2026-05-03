#!/bin/bash
# Llama v33 SFT+DPO N=200 eval (GPU 0).
# Uses LoRA adapter at models/v33_v3_dpo on top of Llama-3.1-8B-Instruct.
# ETA ~25h compute (multi-turn, Novice ~7-turn).
set -e

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1

OUT=outputs/eval_v33_v3_llama_dpo_200.json

echo "[$(date)] Llama v33 SFT+DPO N=200 (GPU=$CUDA_VISIBLE_DEVICES)"
echo "  output: $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_dpo \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --output "$OUT" \
  --max_turns 7 \
  --llm_model gpt-4o-mini --pass_at_k 1 5

echo
echo "[$(date)] Done: $OUT"
