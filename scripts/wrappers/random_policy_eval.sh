#!/bin/bash
# Random Policy baseline: 50/50 Clarify/Execute at each turn (seeded for reproducibility).
# Establishes lower bound — confirms TactfulLLM's gain isn't trivial.
# Usage: bash random_policy_eval.sh <gpu> <backbone>
#   backbone: llama | qwen
set -u
GPU=$1
BACKBONE=$2

cd /root/autodl-tmp/ProactiveLLM
mkdir -p logs

case $BACKBONE in
    llama) BASE="meta-llama/Llama-3.1-8B-Instruct" ;;
    qwen)  BASE="Qwen/Qwen2.5-7B-Instruct" ;;
    *) echo "Usage: $0 <gpu> <llama|qwen>"; exit 1 ;;
esac

OUT=outputs/eval_v29_random_policy_${BACKBONE}_200.json
LOG=logs/random_policy_${BACKBONE}.log

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v1
export PYTHONUNBUFFERED=1

echo "[$(date)] Random Policy baseline ($BACKBONE) on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU python eval/evaluate_multi_turn_persona.py \
    --random_policy \
    --base_model "$BASE" \
    --persona_filter "Novice-Learner,Experienced-Engineer,Busy-Developer" \
    --test_states data/seeds/test_states_v29_eval_200.jsonl \
    --output "$OUT" \
    --max_turns 7 \
    --llm_model gpt-4o-mini --pass_at_k 1 5 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done $BACKBONE → $OUT"
