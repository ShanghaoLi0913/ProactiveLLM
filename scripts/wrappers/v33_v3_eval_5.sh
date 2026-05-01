#!/bin/bash
# v33 v3 SFT model eval, 5-state quick directional test.
# Uses v2 classifier (works correctly with explicit Clarify\n / Execute\n prefix).
set -e

cd /root/autodl-tmp/ProactiveLLM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLASSIFIER_VERSION=v2

echo "[$(date)] v33 v3 SFT quick eval (5-state)"
echo "  model_dir: models/v33_v3_sft"
echo "  classifier: v2"
echo "  max_samples: 5"
echo "  output: outputs/eval_v33_v3_sft_5.json"
echo

python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_sft \
  --base_model "meta-llama/Llama-3.1-8B-Instruct" \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 5 \
  --output outputs/eval_v33_v3_sft_5.json \
  --max_turns 7

echo
echo "[$(date)] Done."
