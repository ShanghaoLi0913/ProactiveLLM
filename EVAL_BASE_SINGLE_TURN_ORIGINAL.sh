#!/bin/bash

# Base模型单轮评估 - 使用Original Query

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_STATES="data/seeds/bigcodebench_masked_states_clean_test.jsonl"
OUTPUT="eval_results/base_model_single_turn_original.json"
MAX_SAMPLES=20
SEED=42

echo "=========================================="
echo "Base模型单轮代码生成评估 (Original Query)"
echo "=========================================="

echo ""
echo "📋 评估配置:"
echo "  - Base模型: $BASE_MODEL"
echo "  - 测试数据: $TEST_STATES"
echo "  - 输出文件: $OUTPUT"
echo "  - 最大样本数: $MAX_SAMPLES"
echo "  - ✅ 直接生成代码（不经过Clarify/Execute决策）"
echo "  - ✅ 使用original_instruct_prompt"

echo ""
echo "🚀 开始评估..."

python3 eval/evaluate_base_model_single_turn.py \
    --base_model "$BASE_MODEL" \
    --test_states "$TEST_STATES" \
    --output "$OUTPUT" \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --use_original_query

echo ""
echo "✅ 评估完成！"
echo "📊 结果保存在: $OUTPUT"
