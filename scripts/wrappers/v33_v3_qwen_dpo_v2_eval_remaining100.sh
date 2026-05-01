#!/bin/bash
# Qwen v33 SFT+DPO eval on remaining 100 states (101-200 of canonical 200-state set)
# Same model + fixed template + no-truncation eval. Combines with first 100 for N=200.
set -e
cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2

OUT=outputs/eval_v33_v3_qwen_dpo_v2_remaining100_ft.json

echo "[$(date)] Qwen v33 SFT+DPO eval on REMAINING 100 states"
echo "  test_states: data/seeds/test_states_v29_eval_200_remaining100.jsonl"
echo "  output:      $OUT"
echo "  fixed template + no truncation (verified)"
echo

python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_qwen_dpo_v2 \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200_remaining100.jsonl \
  --output "$OUT" \
  --max_turns 7

echo
echo "[$(date)] Done. Output: $OUT"
echo "Combine with first-100 file for N=200 paper-canonical eval."
