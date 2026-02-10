#!/bin/bash
# 评估V14模型

echo "📊 开始评估 V14..."
echo ""

cd /root/ProactiveLLM

python eval/evaluate_v13_persona.py \
  --model_dir outputs/v14_final \
  --base_model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --test_data data/dpo/prefs_100states_balanced_test.jsonl \
  --output outputs/eval_results/v14_eval.json

echo ""
echo "✅ 评估完成！"
echo "结果保存在: outputs/eval_results/v14_eval.json"
