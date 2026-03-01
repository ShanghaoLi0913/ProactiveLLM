#!/bin/bash
# V20训练：使用Persona-Aware Rebalance的preference pairs

set -e

echo "=========================================="
echo "V20 Persona-Aware Rebalance Training"
echo "=========================================="
echo ""

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_DATA="data/dpo/prefs_colm_3turn_persona_150states_persona_aware_rebalance_final_train.jsonl"
OUTPUT_DIR="checkpoints/v20_persona_aware_rebalance"
EPOCHS=3
LR=5e-5
BETA=0.1

echo "训练配置:"
echo "  Model: $MODEL_NAME"
echo "  Data: $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  LR: $LR"
echo "  Beta: $BETA"
echo ""

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

# 设置环境变量
export HF_HOME=${HF_HOME:-"/root/autodl-tmp/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/root/autodl-tmp/hf_cache"}

# 运行训练
python3 policy/train_dpo.py \
  --data "$TRAIN_DATA" \
  --model "$MODEL_NAME" \
  --output "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --beta "$BETA"

echo ""
echo "✅ 训练完成！"
