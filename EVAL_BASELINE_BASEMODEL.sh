#!/bin/bash
# 评估未训练的base model（baseline）

set -e

echo "=========================================="
echo "评估未训练的Base Model (Baseline)"
echo "=========================================="
echo ""

# 配置
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_STATES="data/seeds/bigcodebench_masked_states_clean_test.jsonl"
OUTPUT="eval_results/baseline_basemodel_masked.json"
MAX_SAMPLES=20
MAX_TURNS=3
SEED=42

# 使用一个不存在的model_dir，让代码fallback到base model
MODEL_DIR="/tmp/nonexistent_model_dir_for_baseline"

echo "📋 评估配置:"
echo "  - Base模型: $BASE_MODEL (未训练)"
echo "  - Model Dir: $MODEL_DIR (不存在，会fallback到base model)"
echo "  - 测试数据: $TEST_STATES"
echo "  - 输出文件: $OUTPUT"
echo "  - 最大样本数: $MAX_SAMPLES"
echo "  - 最大轮次: $MAX_TURNS"
echo "  - ✅ 使用 masked query"
echo "  - ⚠️  不使用训练后的模型（baseline）"
echo ""

# 检查文件
if [ ! -f "$TEST_STATES" ]; then
    echo "❌ 测试数据文件不存在: $TEST_STATES"
    exit 1
fi

# 创建输出目录
mkdir -p eval_results

# 运行评估（传入不存在的model_dir，会fallback到base model）
echo "🚀 开始评估（baseline base model）..."
python eval/evaluate_multi_turn_persona.py \
    --model_dir "$MODEL_DIR" \
    --base_model "$BASE_MODEL" \
    --test_states "$TEST_STATES" \
    --max_samples "$MAX_SAMPLES" \
    --max_turns "$MAX_TURNS" \
    --output "$OUTPUT" \
    --seed "$SEED"

echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT"

