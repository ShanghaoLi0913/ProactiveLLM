#!/bin/bash
# V14 训练脚本
# Trajectory-level + Persona-aware + Balanced

echo "🚀 开始训练 V14..."
echo ""
echo "数据特点:"
echo "  • Trajectory-level reward"
echo "  • Persona-aware preferences"
echo "  • Balanced distribution (6.9%-15.5% Clarify)"
echo "  • 143 train prefs, 38 test prefs"
echo ""

cd /root/ProactiveLLM

python policy/train_dpo.py \
  --data data/dpo/prefs_100states_balanced_train.jsonl \
  --model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --output outputs/v14_final \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1

echo ""
echo "✅ 训练完成！"
echo "模型保存在: outputs/v14_final"
echo ""
echo "下一步: 评估模型"
echo "  python eval/evaluate_v13_persona.py \\"
echo "    --model_dir outputs/v14_final \\"
echo "    --prefs_path data/dpo/prefs_100states_balanced_test.jsonl \\"
echo "    --output outputs/eval_results/v14_eval.json"
