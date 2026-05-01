#!/bin/bash
# Qwen v33 SFT+DPO 100-state eval, RERUN with:
#   1. Fixed coding_execute.txt template (forces imports)
#   2. No-truncation evaluate_multi_turn_persona.py (full code saved)
# Apr 30 2026, post-PO-baseline rerun.
set -e
cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2

OUT=outputs/eval_v33_v3_qwen_dpo_v2_100_ft.json

echo "[$(date)] Qwen v33 SFT+DPO 100-state eval (fixed template + full code save)"
echo "  output:  $OUT"
echo

python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_qwen_dpo_v2 \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 100 \
  --output "$OUT" \
  --max_turns 7

echo
echo "[$(date)] Done. Output: $OUT"
