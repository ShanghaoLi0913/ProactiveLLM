#!/bin/bash
# 评估初步训练的模型

set -e

echo "=========================================="
echo "评估初步训练的DPO模型"
echo "=========================================="
echo ""

# 配置
MODEL_DIR="checkpoints/dpo_colm_preliminary_195pairs"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_STATES="data/seeds/bigcodebench_masked_states_clean_test.jsonl"
OUTPUT="eval_results/preliminary_eval_195pairs.json"
MAX_SAMPLES=20
MAX_TURNS=3
SEED=42

echo "📋 评估配置:"
echo "  - 模型目录: $MODEL_DIR"
echo "  - Base模型: $BASE_MODEL"
echo "  - 测试数据: $TEST_STATES"
echo "  - 输出文件: $OUTPUT"
echo "  - 最大样本数: $MAX_SAMPLES"
echo "  - 最大轮次: $MAX_TURNS"
echo ""

# 检查文件
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 模型目录不存在: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$TEST_STATES" ]; then
    echo "❌ 测试数据文件不存在: $TEST_STATES"
    exit 1
fi

# 创建输出目录
mkdir -p eval_results

# 运行评估
echo "🚀 开始评估..."
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

