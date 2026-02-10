#!/bin/bash
# V16 评估脚本

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    评估 V16 性能                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 评估内容:"
echo "   • Action Accuracy (模型是否先输出Execute/Clarify)"
echo "   • Task Success Rate (代码是否通过测试)"
echo "   • Persona差异分析"
echo ""
echo "🎯 预期改进 (相比V15):"
echo "   V15: Action 0%, Task 10.5%"
echo "   V16: Action >80%, Task >30% (预期)"
echo ""
echo "💡 评估逻辑:"
echo "   模型生成: Execute\\n\`\`\`python..."
echo "   提取action: 第一行是否为Execute/Clarify"
echo ""

cd /root/ProactiveLLM

# 检查模型文件
if [ ! -d "outputs/v16_with_action_prefix" ]; then
    echo "❌ 错误: 找不到V16模型"
    echo "   请先运行: bash TRAIN_V16.sh"
    exit 1
fi

# 检查测试数据
if [ ! -f "data/dpo/prefs_100states_v16_test.jsonl" ]; then
    echo "❌ 错误: 找不到测试数据文件"
    exit 1
fi

echo "✅ 文件检查通过"
echo ""
echo "🚀 开始评估..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python eval/evaluate_v13_persona.py \
  --model_dir outputs/v16_with_action_prefix \
  --base_model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --test_data data/dpo/prefs_100states_v16_test.jsonl \
  --output outputs/eval_results/v16_eval.json

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 评估完成！"
echo ""
echo "📊 结果保存在: outputs/eval_results/v16_eval.json"
echo ""
echo "🎯 查看详细结果:"
echo "   cat outputs/eval_results/v16_eval.json | python -m json.tool"
echo ""
