#!/bin/bash
# V20评估脚本

set -e

echo "=========================================="
echo "V20 Persona-Aware Rebalance Evaluation"
echo "=========================================="
echo ""

MODEL_DIR="checkpoints/v20_persona_aware_rebalance"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_STATES="data/seeds/test_states_v19.jsonl"
OUTPUT="eval_results/v20_persona_aware_rebalance_eval.json"
MAX_SAMPLES=30
MAX_TURNS=5

echo "评估配置:"
echo "  Model: $MODEL_DIR"
echo "  Base Model: $BASE_MODEL"
echo "  Test States: $TEST_STATES"
echo "  Output: $OUTPUT"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Max Turns: $MAX_TURNS"
echo ""

# 设置环境变量
export HF_HOME=${HF_HOME:-"/root/autodl-tmp/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/root/autodl-tmp/hf_cache"}

# 运行评估
python3 eval/evaluate_multi_turn_persona.py \
  --model_dir "$MODEL_DIR" \
  --base_model "$BASE_MODEL" \
  --test_states "$TEST_STATES" \
  --max_samples "$MAX_SAMPLES" \
  --max_turns "$MAX_TURNS" \
  --output "$OUTPUT" \
  --seed 42 \
  2>&1 | tee /tmp/eval_v20.log

echo ""
echo "✅ 评估完成！"
echo "结果保存在: $OUTPUT"
