#!/bin/bash
# V16 训练脚本
# 关键改进：在assistant message前添加action标签 ("Execute\n..." 或 "Clarify\n...")

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    开始训练 V16                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 关键改进: Assistant message添加action前缀"
echo "   训练格式:"
echo "   Execute"
echo "   \`\`\`python"
echo "   def task_func():"
echo "       ..."
echo "   \`\`\`"
echo ""
echo "   或"
echo ""
echo "   Clarify"
echo "   Could you clarify..."
echo ""
echo "📊 数据特点:"
echo "   • 相同的trajectory-level reward和persona-aware"
echo "   • 新增：Action前缀让模型学会先决策再生成"
echo "   • 143 train prefs, 38 test prefs"
echo ""
echo "🔧 训练配置:"
echo "   • Base model: Llama-3.1-8B-Instruct"
echo "   • Method: DPO + QLoRA"
echo "   • Chat template: ✅ (V15已添加)"
echo "   • Epochs: 3"
echo "   • Learning rate: 5e-5"
echo "   • Beta: 0.1"
echo ""

cd /root/ProactiveLLM

# 检查数据文件
if [ ! -f "data/dpo/prefs_100states_v16_train.jsonl" ]; then
    echo "❌ 错误: 找不到训练数据文件"
    echo "   data/dpo/prefs_100states_v16_train.jsonl"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""
echo "🚀 开始训练..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python policy/train_dpo.py \
  --data data/dpo/prefs_100states_v16_train.jsonl \
  --model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --output outputs/v16_with_action_prefix \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 训练完成！"
echo ""
echo "📂 模型保存位置: outputs/v16_with_action_prefix"
echo ""
echo "🎯 下一步: 评估V16性能"
echo "   bash EVAL_V16.sh"
echo ""
