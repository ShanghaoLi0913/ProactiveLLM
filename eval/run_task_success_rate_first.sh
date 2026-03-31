#!/usr/bin/bash
# Step 1：先跑单轮 Task Success Rate（快于多轮 persona 评估）
# 使用 eval/evaluate_dpo_model.py：每条样本一次 action + 一次代码生成 + 测例打分
#
# 多轮交互、Pass@K、OpenAI user 等请用 Step 2：eval/compare_base_sft_dpo_multi_turn.sh
# 或手动跑 eval/evaluate_multi_turn_persona.py
#
# 用法：
#   ./eval/run_task_success_rate_first.sh
#
# 环境变量（可选）：
#   BASE_MODEL  DPO_DIR  SFT_DIR  PREFS  MAX_SAMPLES  OUT_DIR

set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DPO_DIR="${DPO_DIR:-checkpoints/medium_scale_5epochs_20260304_021950}"
SFT_DIR="${SFT_DIR:-outputs/proactive_llm_colm_150states_sft}"
# 与 n4 轨迹同分布的 test prefs（可按需改）
PREFS="${PREFS:-data/dpo/prefs_colm_n4_150states_test.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
OUT_DIR="${OUT_DIR:-eval_results}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Step 1: Task Success Rate (single-turn eval)"
echo "=========================================="
echo "BASE_MODEL=$BASE_MODEL"
echo "PREFS=$PREFS"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "code_samples=1 (最快；需要 best-of 可改脚本)"
echo ""

run_tsr() {
  local tag="$1"
  local mdir="$2"
  local out="${OUT_DIR}/tsr_${tag}_${TS}.json"
  echo ">>> [${tag}] model_dir=$mdir -> $out"
  python eval/evaluate_dpo_model.py \
    --model_dir "$mdir" \
    --base_model "$BASE_MODEL" \
    --prefs "$PREFS" \
    --max_samples "$MAX_SAMPLES" \
    --code_samples 1 \
    --seed 42 \
    --output "$out"
  echo ""
}

if [[ ! -f "$PREFS" ]]; then
  echo "❌ PREFS 不存在: $PREFS"
  exit 1
fi

if [[ -d "$DPO_DIR" ]]; then
  run_tsr "dpo" "$DPO_DIR"
else
  echo "⚠️  跳过 DPO：$DPO_DIR 不存在"
fi

if [[ -d "$SFT_DIR" ]]; then
  run_tsr "sft" "$SFT_DIR"
else
  echo "⚠️  跳过 SFT：$SFT_DIR 不存在"
fi

echo "=========================================="
echo "Step 1 完成。查看 summary.task_success_rate 等字段。"
echo ""
echo "Step 2（更慢，多轮 + user API 等）示例："
echo "  ./eval/compare_base_sft_dpo_multi_turn.sh"
echo "  # 或减小规模：MAX_SAMPLES=10 MAX_TURNS=3 ./eval/compare_base_sft_dpo_multi_turn.sh"
echo "=========================================="
