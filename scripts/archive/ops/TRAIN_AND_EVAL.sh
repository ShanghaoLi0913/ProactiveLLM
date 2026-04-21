#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 训练和评估脚本 - 使用生成的150 states数据
# 数据文件: data/logs/traj_colm_n4_150states_20260317_100056.jsonl

set -e

# 修复OpenMP环境变量问题
export OMP_NUM_THREADS=1

echo "=========================================="
echo "ProactiveLLM 训练和评估流程"
echo "=========================================="
echo ""

# 配置
TRAJ_FILE="data/logs/traj_colm_n4_150states_20260317_100056.jsonl"
PREFS_FILE="data/dpo/prefs_colm_n4_150states.jsonl"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  # 可以根据需要修改
OUTPUT_DIR="outputs/proactive_llm_colm_150states"
EVAL_OUTPUT="outputs/eval_results_colm_150states.json"

# 检查数据文件
if [ ! -f "$TRAJ_FILE" ]; then
    echo "❌ 错误: 轨迹数据文件不存在: $TRAJ_FILE"
    exit 1
fi

echo "✅ 数据文件检查通过: $TRAJ_FILE"
echo ""

# Step 1: 计算奖励并生成preference pairs
echo "[Step 1/3] 计算奖励并生成preference pairs..."
echo "  输入: $TRAJ_FILE"
echo "  输出: $PREFS_FILE"
echo ""

python reward/compute_rewards.py \
    --trajectories "$TRAJ_FILE" \
    --out "$PREFS_FILE" \
    --w_task 1.0 \
    --w_interrupt 0.3 \
    --use_trajectory_level

if [ $? -ne 0 ]; then
    echo "❌ Step 1 失败"
    exit 1
fi

echo ""
echo "✅ Step 1 完成: $PREFS_FILE"
echo ""

# Step 2: 训练DPO模型
echo "[Step 2/3] 训练DPO模型..."
echo "  模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE"
echo "  输出: $OUTPUT_DIR"
echo ""

python policy/train_dpo.py \
    --model "$MODEL_NAME" \
    --data "$PREFS_FILE" \
    --output "$OUTPUT_DIR" \
    --epochs 3 \
    --lr 5e-5 \
    --beta 0.1

if [ $? -ne 0 ]; then
    echo "❌ Step 2 失败"
    exit 1
fi

echo ""
echo "✅ Step 2 完成: $OUTPUT_DIR"
echo ""

# Step 3: 评估模型
echo "[Step 3/3] 评估模型..."
echo "  模型: $OUTPUT_DIR"
echo "  基础模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE"
echo "  输出: $EVAL_OUTPUT"
echo ""

python eval/evaluate_dpo_model.py \
    --model_dir "$OUTPUT_DIR" \
    --base_model "$MODEL_NAME" \
    --prefs "$PREFS_FILE" \
    --output "$EVAL_OUTPUT"

if [ $? -ne 0 ]; then
    echo "❌ Step 3 失败"
    exit 1
fi

echo ""
echo "✅ Step 3 完成: $EVAL_OUTPUT"
echo ""

echo "=========================================="
echo "✅ 训练和评估完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - Preference pairs: $PREFS_FILE"
echo "  - 训练模型: $OUTPUT_DIR"
echo "  - 评估结果: $EVAL_OUTPUT"
echo ""
