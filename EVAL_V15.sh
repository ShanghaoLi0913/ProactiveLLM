#!/bin/bash
# V15 评估脚本

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    评估 V15 性能                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 评估内容:"
echo "   • Action Accuracy (按persona)"
echo "   • Task Success Rate (代码是否通过测试)"
echo "   • Persona差异分析"
echo ""
echo "🎯 预期改进 (相比V14):"
echo "   V14: Action 0%, Task 0%  →  V15: Action >50%, Task >30%"
echo ""

cd /root/ProactiveLLM

# 检查模型文件
if [ ! -d "outputs/v15_with_chat_template" ]; then
    echo "❌ 错误: 找不到V15模型"
    echo "   请先运行: bash TRAIN_V15.sh"
    exit 1
fi

# 检查测试数据
if [ ! -f "data/dpo/prefs_100states_balanced_test.jsonl" ]; then
    echo "❌ 错误: 找不到测试数据文件"
    exit 1
fi

echo "✅ 文件检查通过"
echo ""
echo "🚀 开始评估..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python eval/evaluate_v13_persona.py \
  --model_dir outputs/v15_with_chat_template \
  --base_model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --test_data data/dpo/prefs_100states_balanced_test.jsonl \
  --output outputs/eval_results/v15_eval.json

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 评估完成！"
echo ""
echo "📊 结果保存在: outputs/eval_results/v15_eval.json"
echo ""
echo "🎯 查看详细结果:"
echo "   cat outputs/eval_results/v15_eval.json | python -m json.tool"
echo ""
