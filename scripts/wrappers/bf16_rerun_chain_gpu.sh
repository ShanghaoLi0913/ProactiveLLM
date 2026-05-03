#!/bin/bash
# bf16 rerun chain — runs sequence of methods on assigned GPU.
# Usage: bash /tmp/bf16_rerun_chain_gpu.sh <GPU_ID> <method1,method2,...>
# Method names: llama_base, llama_direct, llama_cf, llama_po, llama_dpo
#               qwen_base, qwen_direct, qwen_cf, qwen_po, qwen_dpo

set -u
GPU=$1
METHODS=$2
LOG=/tmp/bf16_chain_gpu${GPU}.log

cd /root/autodl-tmp/ProactiveLLM
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2
export PYTHONUNBUFFERED=1
export DISABLE_8BIT=1   # ⭐ bf16

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
LLAMA_DPO="models/v33_v3_dpo"
QWEN_DPO="models/v33_v3_qwen_dpo_v2"
SEEDS=data/seeds/test_states_v29_eval_200.jsonl

run_method() {
    local m=$1
    local out
    case $m in
        llama_base)   ARGS="--no_lora --prompt_only --base_model $LLAMA --max_turns 7"; out=outputs/eval_v29_llama_base_200_v2_bf16.json ;;
        llama_direct) ARGS="--no_lora --direct_execution --base_model $LLAMA --max_turns 1"; out=outputs/eval_v29_direct_execution_200_bf16.json ;;
        llama_cf)     ARGS="--no_lora --always_clarify 1 --base_model $LLAMA --max_turns 2"; out=outputs/eval_v29_clarify_first_200_bf16.json ;;
        llama_po)     ARGS="--no_lora --prompt_only --base_model $LLAMA --max_turns 7"; out=outputs/eval_v29_llama_prompt_only_200_v2_bf16.json ;;
        llama_dpo)    ARGS="--model_dir $LLAMA_DPO --base_model $LLAMA --max_turns 7"; out=outputs/eval_v33_v3_llama_dpo_200_bf16.json ;;
        qwen_base)    ARGS="--no_lora --prompt_only --base_model $QWEN --max_turns 7"; out=outputs/eval_v29_qwen_base_200_bf16.json ;;
        qwen_direct)  ARGS="--no_lora --direct_execution --base_model $QWEN --max_turns 1"; out=outputs/eval_v29_qwen_direct_execution_200_bf16.json ;;
        qwen_cf)      ARGS="--no_lora --always_clarify 1 --base_model $QWEN --max_turns 2"; out=outputs/eval_v29_qwen_clarify_first_200_bf16.json ;;
        qwen_po)      ARGS="--no_lora --prompt_only --base_model $QWEN --max_turns 7"; out=outputs/eval_v29_qwen_prompt_only_200_bf16.json ;;
        qwen_dpo)     ARGS="--model_dir $QWEN_DPO --base_model $QWEN --max_turns 7"; out=outputs/eval_v33_v3_qwen_dpo_v2_200_bf16.json ;;
        *) echo "Unknown method: $m"; return 1 ;;
    esac

    echo "[$(date)] [GPU $GPU] starting $m → $out"
    CUDA_VISIBLE_DEVICES=$GPU python eval/evaluate_multi_turn_persona.py \
        $ARGS --test_states $SEEDS --output $out \
        --llm_model gpt-4o-mini --pass_at_k 1 5
    echo "[$(date)] [GPU $GPU] done $m"
    echo
}

echo "==== bf16 chain GPU $GPU started $(date) — methods: $METHODS ====" | tee -a $LOG

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
for m in "${METHOD_LIST[@]}"; do
    run_method "$m" 2>&1 | tee -a $LOG
done

echo "==== bf16 chain GPU $GPU complete $(date) ====" | tee -a $LOG
