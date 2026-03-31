#!/usr/bin/bash
# 多轮 persona 评估（Step 2，较慢）：Base vs SFT vs DPO
# 建议先跑单轮 TSR：eval/run_task_success_rate_first.sh
#
# 需：OPENAI_API_KEY（user 模拟默认 gpt-4o-mini）
#
# 用法：
#   chmod +x eval/compare_base_sft_dpo_multi_turn.sh
#   ./eval/compare_base_sft_dpo_multi_turn.sh
#
# 环境变量（可选）：
#   BASE_MODEL  DPO_DIR  SFT_DIR  TEST_STATES  MAX_SAMPLES  MAX_TURNS  SEED  OUT_DIR

set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DPO_DIR="${DPO_DIR:-checkpoints/medium_scale_5epochs_20260304_021950}"
SFT_DIR="${SFT_DIR:-outputs/proactive_llm_colm_150states_sft}"

TEST_STATES="${TEST_STATES:-data/dpo/test_states_clean_for_eval.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-40}"
MAX_TURNS="${MAX_TURNS:-5}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-eval_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Multi-turn eval: Base vs SFT vs DPO"
echo "=========================================="
echo "BASE_MODEL=$BASE_MODEL"
echo "DPO_DIR=$DPO_DIR"
echo "SFT_DIR=$SFT_DIR"
echo "TEST_STATES=$TEST_STATES"
echo "MAX_SAMPLES=$MAX_SAMPLES MAX_TURNS=$MAX_TURNS SEED=$SEED"
echo "OUT_DIR=$OUT_DIR"
echo ""

run_eval() {
  local tag="$1"
  shift
  local out="${OUT_DIR}/multi_turn_${tag}_${TS}.json"
  echo ">>> [${tag}] -> ${out}"
  python eval/evaluate_multi_turn_persona.py "$@" --output "$out"
  echo ""
}

# 1) Base（无 LoRA）
run_eval "base" \
  --no_lora \
  --base_model "$BASE_MODEL" \
  --test_states "$TEST_STATES" \
  --max_samples "$MAX_SAMPLES" \
  --max_turns "$MAX_TURNS" \
  --seed "$SEED"

# 2) DPO
if [[ ! -d "$DPO_DIR" ]]; then
  echo "⚠️  跳过 DPO：目录不存在: $DPO_DIR"
else
  run_eval "dpo" \
    --model_dir "$DPO_DIR" \
    --base_model "$BASE_MODEL" \
    --test_states "$TEST_STATES" \
    --max_samples "$MAX_SAMPLES" \
    --max_turns "$MAX_TURNS" \
    --seed "$SEED"
fi

# 3) SFT（LoRA 或完整保存目录）
if [[ ! -d "$SFT_DIR" ]]; then
  echo "⚠️  跳过 SFT：目录不存在: $SFT_DIR"
  echo "    训练示例: python policy/train_sft.py --data ... --model $BASE_MODEL --output $SFT_DIR --use_preference_pairs"
else
  run_eval "sft" \
    --model_dir "$SFT_DIR" \
    --base_model "$BASE_MODEL" \
    --test_states "$TEST_STATES" \
    --max_samples "$MAX_SAMPLES" \
    --max_turns "$MAX_TURNS" \
    --seed "$SEED"
fi

echo "✅ 完成。结果 JSON：${OUT_DIR}/multi_turn_*_${TS}.json"
