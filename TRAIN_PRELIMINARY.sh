#!/bin/bash
# 初步训练脚本 - 使用195个pairs验证模型是否能work

set -e

echo "=========================================="
echo "初步DPO训练 - 验证模型可行性"
echo "=========================================="
echo ""

# 配置
DATA_FILE="data/dpo/traj_colm_3turn_persona_150states_20260227_122315_train_prefs.jsonl"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="checkpoints/dpo_colm_preliminary_195pairs"
EPOCHS=3
LR=5e-5
BETA=0.1

echo "📋 训练配置:"
echo "  - 数据文件: $DATA_FILE"
echo "  - Base模型: $BASE_MODEL"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LR"
echo "  - DPO beta: $BETA"
echo ""

# 检查数据文件
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据文件不存在: $DATA_FILE"
    exit 1
fi

PAIRS=$(wc -l < "$DATA_FILE")
echo "✅ 找到 $PAIRS 个preference pairs"
echo ""

# 开始训练
echo "🚀 开始训练..."
python policy/train_dpo.py \
    --data "$DATA_FILE" \
    --model "$BASE_MODEL" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --beta "$BETA"

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo "模型保存在: $OUTPUT_DIR"
echo ""
echo "下一步: 使用 eval/evaluate_multi_turn_persona.py 评估模型"

