#!/bin/bash
# Base模型单轮代码生成评估

set -e

echo "=========================================="
echo "Base模型单轮代码生成评估"
echo "=========================================="
echo ""

# 配置
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_STATES="data/seeds/bigcodebench_masked_states_clean_test.jsonl"
OUTPUT="eval_results/base_model_single_turn.json"
MAX_SAMPLES=20
SEED=42

echo "📋 评估配置:"
echo "  - Base模型: $BASE_MODEL"
echo "  - 测试数据: $TEST_STATES"
echo "  - 输出文件: $OUTPUT"
echo "  - 最大样本数: $MAX_SAMPLES"
echo "  - ✅ 直接生成代码（不经过Clarify/Execute决策）"
echo "  - ✅ 使用masked query"
echo ""

# 检查文件
if [ ! -f "$TEST_STATES" ]; then
    echo "❌ 测试数据文件不存在: $TEST_STATES"
    exit 1
fi

# 创建输出目录
mkdir -p eval_results

# 运行评估
echo "🚀 开始评估..."
python eval/evaluate_base_model_single_turn.py \
    --base_model "$BASE_MODEL" \
    --test_states "$TEST_STATES" \
    --max_samples "$MAX_SAMPLES" \
    --output "$OUTPUT" \
    --seed "$SEED"

echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT"

