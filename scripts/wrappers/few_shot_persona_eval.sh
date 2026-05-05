#!/bin/bash
# Few-shot Persona Prompt baseline evaluation.
# Usage: bash few_shot_persona_eval.sh <gpu> <backbone>
#   backbone: llama | qwen
# Output: outputs/eval_v29_few_shot_persona_<backbone>_200.json
# Era: 8-bit + v1 (paper consistent)

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

OUT=outputs/eval_v29_few_shot_persona_${BACKBONE}_200.json
LOG=logs/few_shot_persona_${BACKBONE}.log

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v1
export PYTHONUNBUFFERED=1
export FEW_SHOT_PERSONA=1   # ⭐ activate few-shot examples in select_action_prompt_only

echo "[$(date)] Few-shot persona prompt baseline ($BACKBONE) on GPU $GPU"
echo "  base_model: $BASE"
echo "  output: $OUT"

CUDA_VISIBLE_DEVICES=$GPU python eval/evaluate_multi_turn_persona.py \
    --no_lora --prompt_only \
    --base_model "$BASE" \
    --persona_filter "Novice-Learner,Experienced-Engineer,Busy-Developer" \
    --test_states data/seeds/test_states_v29_eval_200.jsonl \
    --output "$OUT" \
    --max_turns 7 \
    --llm_model gpt-4o-mini --pass_at_k 1 5 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done $BACKBONE → $OUT"
