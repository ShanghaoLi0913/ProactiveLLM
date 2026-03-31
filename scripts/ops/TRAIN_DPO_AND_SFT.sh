#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 训练脚本 - DPO和SFT两个版本
# 数据文件: data/logs/traj_colm_n4_150states_20260317_100056.jsonl

set -e

# 修复OpenMP环境变量问题
export OMP_NUM_THREADS=1

# 设置离线模式（使用本地缓存的模型）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 取消代理设置（如果代理不可用）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "=========================================="
echo "ProactiveLLM 训练流程 (DPO + SFT)"
echo "=========================================="
echo ""

# 配置
TRAJ_FILE="data/logs/traj_colm_n4_150states_20260317_100056.jsonl"
PREFS_FILE="data/dpo/prefs_colm_n4_150states_train.jsonl"  # 使用训练集
PREFS_FILE_FULL="data/dpo/prefs_colm_n4_150states.jsonl"  # 完整数据用于评估
# 使用本地模型路径（如果网络不可用）
MODEL_SNAPSHOT=$(find ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots -maxdepth 1 -type d 2>/dev/null | head -1)
if [ -n "$MODEL_SNAPSHOT" ] && [ -f "$MODEL_SNAPSHOT/config.json" ]; then
    MODEL_NAME="$MODEL_SNAPSHOT"
    echo "✅ 使用本地模型路径: $MODEL_NAME"
else
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  # 回退到标准名称
    echo "⚠️  使用标准模型名称: $MODEL_NAME"
fi

# DPO模型配置
DPO_OUTPUT_DIR="outputs/proactive_llm_colm_150states_dpo"
DPO_EVAL_OUTPUT="outputs/eval_results_colm_150states_dpo.json"

# SFT模型配置
SFT_OUTPUT_DIR="outputs/proactive_llm_colm_150states_sft"
SFT_EVAL_OUTPUT="outputs/eval_results_colm_150states_sft.json"

# 检查数据文件
if [ ! -f "$TRAJ_FILE" ]; then
    echo "❌ 错误: 轨迹数据文件不存在: $TRAJ_FILE"
    exit 1
fi

echo "✅ 数据文件检查通过: $TRAJ_FILE"
echo ""

# Step 1: 计算奖励并生成preference pairs
echo "[Step 1/5] 计算奖励并生成preference pairs..."
echo "  输入: $TRAJ_FILE"
echo "  输出: $PREFS_FILE"
echo ""

if [ ! -f "$PREFS_FILE" ]; then
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
else
    echo "✅ Preference pairs已存在，跳过生成: $PREFS_FILE"
fi
echo ""

# Step 2: 训练DPO模型
echo "[Step 2/5] 训练DPO模型..."
echo "  模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE"
echo "  输出: $DPO_OUTPUT_DIR"
echo ""

python policy/train_dpo.py \
    --model "$MODEL_NAME" \
    --data "$PREFS_FILE" \
    --output "$DPO_OUTPUT_DIR" \
    --epochs 3 \
    --lr 5e-5 \
    --beta 0.1

if [ $? -ne 0 ]; then
    echo "❌ Step 2 失败"
    exit 1
fi

echo ""
echo "✅ Step 2 完成: $DPO_OUTPUT_DIR"
echo ""

# Step 3: 训练SFT模型
echo "[Step 3/5] 训练SFT模型..."
echo "  模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE (使用chosen actions)"
echo "  输出: $SFT_OUTPUT_DIR"
echo ""

python policy/train_sft.py \
    --model "$MODEL_NAME" \
    --data "$PREFS_FILE" \
    --output "$SFT_OUTPUT_DIR" \
    --epochs 3 \
    --lr 5e-5 \
    --use_preference_pairs

if [ $? -ne 0 ]; then
    echo "❌ Step 3 失败"
    exit 1
fi

echo ""
echo "✅ Step 3 完成: $SFT_OUTPUT_DIR"
echo ""

# Step 4: 评估DPO模型
echo "[Step 4/5] 评估DPO模型..."
echo "  模型: $DPO_OUTPUT_DIR"
echo "  基础模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE"
echo "  输出: $DPO_EVAL_OUTPUT"
echo ""

python eval/evaluate_dpo_model.py \
    --model_dir "$DPO_OUTPUT_DIR" \
    --base_model "$MODEL_NAME" \
    --prefs "$PREFS_FILE_FULL" \
    --output "$DPO_EVAL_OUTPUT"

if [ $? -ne 0 ]; then
    echo "❌ Step 4 失败"
    exit 1
fi

echo ""
echo "✅ Step 4 完成: $DPO_EVAL_OUTPUT"
echo ""

# Step 5: 评估SFT模型
echo "[Step 5/5] 评估SFT模型..."
echo "  模型: $SFT_OUTPUT_DIR"
echo "  基础模型: $MODEL_NAME"
echo "  数据: $PREFS_FILE"
echo "  输出: $SFT_EVAL_OUTPUT"
echo ""

python eval/evaluate_dpo_model.py \
    --model_dir "$SFT_OUTPUT_DIR" \
    --base_model "$MODEL_NAME" \
    --prefs "$PREFS_FILE_FULL" \
    --output "$SFT_EVAL_OUTPUT"

if [ $? -ne 0 ]; then
    echo "❌ Step 5 失败"
    exit 1
fi

echo ""
echo "✅ Step 5 完成: $SFT_EVAL_OUTPUT"
echo ""

echo "=========================================="
echo "✅ 训练和评估完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - Preference pairs: $PREFS_FILE"
echo "  - DPO训练模型: $DPO_OUTPUT_DIR"
echo "  - SFT训练模型: $SFT_OUTPUT_DIR"
echo "  - DPO评估结果: $DPO_EVAL_OUTPUT"
echo "  - SFT评估结果: $SFT_EVAL_OUTPUT"
echo ""
