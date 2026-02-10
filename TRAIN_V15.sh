#!/bin/bash
# V15 训练脚本
# 修复：添加chat template以提供清晰的prompt/response边界

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    开始训练 V15                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 关键改进: 使用Llama-3.1-Instruct的chat template"
echo "   - 添加<|start_header_id|>assistant<|end_header_id|>标记"
echo "   - 模型将知道从哪里开始生成response"
echo ""
echo "📊 数据特点:"
echo "   • Trajectory-level reward (Clarify获得后续成功的credit)"
echo "   • Persona-aware preferences (每个persona独立比较)"
echo "   • Balanced distribution (6.9%-15.5% Clarify)"
echo "   • 143 train prefs, 38 test prefs"
echo ""
echo "🔧 训练配置:"
echo "   • Base model: Llama-3.1-8B-Instruct"
echo "   • Method: DPO + QLoRA"
echo "   • Epochs: 3"
echo "   • Learning rate: 5e-5"
echo "   • Beta: 0.1"
echo ""

cd /root/ProactiveLLM

# 检查数据文件
if [ ! -f "data/dpo/prefs_100states_balanced_train.jsonl" ]; then
    echo "❌ 错误: 找不到训练数据文件"
    echo "   data/dpo/prefs_100states_balanced_train.jsonl"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""
echo "🚀 开始训练..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python policy/train_dpo.py \
  --data data/dpo/prefs_100states_balanced_train.jsonl \
  --model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --output outputs/v15_with_chat_template \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 训练完成！"
echo ""
echo "📂 模型保存位置: outputs/v15_with_chat_template"
echo ""
echo "🎯 下一步: 评估V15性能"
echo "   bash EVAL_V15.sh"
echo ""
