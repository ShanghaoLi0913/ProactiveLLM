#!/bin/bash
# 继续生成数据 - 从BigCodeBench/357继续，跳过已完成的部分

set -e

echo "=========================================="
echo "继续生成数据 (Resume Mode)"
echo "=========================================="
echo ""
echo "【当前状态】"
echo "  ✅ 已完成: 129 states (完全完成)"
echo "  ⚠️  未完全完成: 1 state (BigCodeBench/357)"
echo "  ⚠️  完全未处理: 20 states"
echo ""
echo "【使用 --resume 参数跳过已完成的部分】"
echo ""

# 参数配置
DATASET_PATH="data/seeds/bigcodebench_masked_states.jsonl"
N_STATES=150
MAX_TURNS=3
LLM_MODEL="gpt-4o-mini"
SEED=42

# 使用已存在的输出文件（resume模式）
OUTPUT_FILE="logs/traj_colm_3turn_persona_150states_20260225_115940_20260225_115942.jsonl"

echo "[继续生成轨迹数据]"
echo "  - States: $N_STATES"
echo "  - Personas: All (3)"
echo "  - Samples per (state, persona): 4 (2 Execute + 2 Clarify)"
echo "  - Max turns: $MAX_TURNS"
echo "  - Resume: 跳过已完成的(state, persona, action)组合"
echo "  - 输出文件: $OUTPUT_FILE"
echo ""

python scripts/generate_trajectories.py \
  --mode dataset \
  --dataset_path "$DATASET_PATH" \
  --domain coding \
  --n_states "$N_STATES" \
  --all_personas \
  --n_samples 4 \
  --sampling_strategy heuristic \
  --max_turns "$MAX_TURNS" \
  --llm_model "$LLM_MODEL" \
  --out "$OUTPUT_FILE" \
  --seed "$SEED" \
  --temperature 0.7 \
  --top_p 0.9 \
  --progress_every 10 \
  --resume

echo ""
echo "✓ 数据生成完成！"
echo ""
