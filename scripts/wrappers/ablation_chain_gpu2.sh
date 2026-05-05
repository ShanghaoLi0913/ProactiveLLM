#!/bin/bash
# Ablation chain on GPU 2:
#   1. no_persona (100 missing states, ~2-3h, single-turn collapse)
#   2. no_uncertainty (113 missing states, ~7-9h, multi-turn preserved)
# Total ETA: ~10-12h
# Each ablation runs paper-consistent: 8-bit + v1 classifier on v29 ablation models.

set -u
cd /root/autodl-tmp/ProactiveLLM

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v1   # paper consistent
export PYTHONUNBUFFERED=1
# NO DISABLE_8BIT → 8-bit (paper consistent)

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
LOG=logs/ablation_chain_gpu2.log
mkdir -p logs

run_ablation() {
    local mode=$1       # no_persona | no_uncertainty
    local model_dir=models/v29_ablation_${mode}
    local seeds=data/seeds/test_states_${mode}_missing.jsonl
    local out=outputs/eval_v29_ablation_${mode}_missing.json

    echo "[$(date)] starting ablation $mode → $out" | tee -a $LOG
    ABLATION_MODE=$mode CUDA_VISIBLE_DEVICES=2 python eval/evaluate_multi_turn_persona.py \
        --model_dir $model_dir \
        --base_model $LLAMA \
        --test_states $seeds \
        --output $out \
        --max_turns 7 \
        --llm_model gpt-4o-mini --pass_at_k 1 5 \
        2>&1 | tee -a $LOG
    echo "[$(date)] done ablation $mode" | tee -a $LOG
}

echo "==== ablation chain GPU 2 started $(date) ====" | tee -a $LOG
run_ablation no_persona
run_ablation no_uncertainty
echo "==== ablation chain GPU 2 complete $(date) ====" | tee -a $LOG
