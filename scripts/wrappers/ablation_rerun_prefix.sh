#!/bin/bash
# Re-run ablations on the "missing" state subsets with the PRE-FIX prompt
# template (PROMPT_DIR=prompts_prefix), so that legacy 50test+50extra (Apr 16
# pre-fix) and the new "missing" runs are evaluated under identical pipelines.
# Output goes to *_missing_prefix.json and is merged with legacy → clean N=200
# matched to paper TactfulLLM (also pre-fix).
#
# Usage: bash scripts/wrappers/ablation_rerun_prefix.sh <gpu_id> [mode]
#   mode: "both" (default), "no_persona", or "no_uncertainty"

set -u
cd /root/autodl-tmp/ProactiveLLM

GPU=${1:-0}
MODE=${2:-both}

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v1            # paper consistent
export PROMPT_DIR=prompts_prefix        # KEY: pre-fix prompt template
export PYTHONUNBUFFERED=1

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
LOG=logs/ablation_rerun_prefix_gpu${GPU}.log
mkdir -p logs

run_ablation() {
    local mode=$1
    local model_dir=models/v29_ablation_${mode}
    local seeds=data/seeds/test_states_${mode}_missing.jsonl
    local out=outputs/eval_v29_ablation_${mode}_missing_prefix.json

    echo "[$(date)] starting $mode (pre-fix prompt) on GPU $GPU → $out" | tee -a $LOG
    ABLATION_MODE=$mode CUDA_VISIBLE_DEVICES=$GPU python eval/evaluate_multi_turn_persona.py \
        --model_dir $model_dir \
        --base_model $LLAMA \
        --test_states $seeds \
        --persona_filter "Novice-Learner,Experienced-Engineer,Busy-Developer" \
        --output $out \
        --max_turns 7 \
        --llm_model gpt-4o-mini --pass_at_k 1 5 \
        2>&1 | tee -a $LOG
    echo "[$(date)] done $mode" | tee -a $LOG
}

echo "==== ablation rerun (pre-fix prompt) GPU $GPU mode=$MODE started $(date) ====" | tee -a $LOG
case "$MODE" in
    no_persona)     run_ablation no_persona ;;
    no_uncertainty) run_ablation no_uncertainty ;;
    both)
        run_ablation no_uncertainty   # longest first
        run_ablation no_persona
        ;;
    *) echo "Unknown mode: $MODE"; exit 1 ;;
esac
echo "==== ablation rerun complete $(date) ====" | tee -a $LOG
