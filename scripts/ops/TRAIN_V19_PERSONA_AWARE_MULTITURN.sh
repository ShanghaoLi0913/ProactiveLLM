#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Train V19 Model: Persona-Aware Multi-turn DPO
# 使用最新的ideal preference pairs，支持多轮学习和persona-aware行为

set -e

echo "=========================================="
echo "Train V19: Persona-Aware Multi-turn DPO"
echo "=========================================="
echo ""

# 配置
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_DATA="data/dpo/prefs_colm_3turn_persona_150states_ideal_20260224_055853_train.jsonl"
OUTPUT_DIR="checkpoints/v19_persona_aware_multiturn"
EPOCHS=3
LR=5e-5
BETA=0.1

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

echo "📊 训练配置:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Training Data: $TRAIN_DATA"
echo "  - Output Dir: $OUTPUT_DIR"
echo "  - Epochs: $EPOCHS"
echo "  - Learning Rate: $LR"
echo "  - DPO Beta: $BETA"
echo ""

# 检查数据
echo "📊 检查训练数据..."
python3 << EOF
import json
from pathlib import Path

with open("$TRAIN_DATA", 'r') as f:
    prefs = [json.loads(line) for line in f]

print(f"  总preference pairs: {len(prefs)}条")

# 检查persona分布
from collections import Counter
personas = Counter(p.get('persona', {}).get('name', 'Unknown') for p in prefs)
print(f"  Persona分布:")
for persona, count in personas.items():
    print(f"    {persona}: {count}条 ({count*100/len(prefs):.1f}%)")

# 检查Turn 1+数据
turn_1_plus = sum(1 for p in prefs if p.get('dialogue_turn', 0) > 0)
print(f"  Turn 1+数据: {turn_1_plus}条 ({turn_1_plus*100/len(prefs):.1f}%)")

# 检查prev_action
has_prev_action = sum(1 for p in prefs if 'prev_action' in p)
print(f"  包含prev_action: {has_prev_action}条 ({has_prev_action*100/len(prefs):.1f}%)")
EOF

echo ""
echo "🚀 开始训练..."
echo ""

# 设置环境变量
# HF_TOKEN should be set as environment variable, not hardcoded
# export HF_TOKEN=${HF_TOKEN:-"your_token_here"}  # Remove hardcoded token for security
# 使用本地模型缓存路径，避免重新下载
export HF_HOME=${HF_HOME:-"/root/autodl-tmp/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/root/autodl-tmp/hf_cache"}

# 运行训练
python3 policy/train_dpo.py \
  --data "$TRAIN_DATA" \
  --model "$BASE_MODEL" \
  --output "$OUTPUT_DIR" \
  --epochs $EPOCHS \
  --lr $LR \
  --beta $BETA \
  2>&1 | tee "${OUTPUT_DIR}_training.log"

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "模型保存位置: $OUTPUT_DIR"
echo "训练日志: ${OUTPUT_DIR}_training.log"
echo ""
echo "下一步:"
echo "  1. 评估模型: python3 eval/evaluate_multi_turn_persona.py --model_dir $OUTPUT_DIR"
echo "  2. 验证persona区分度和多轮行为"
echo ""
