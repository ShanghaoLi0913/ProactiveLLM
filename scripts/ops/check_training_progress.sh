#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 监控训练进度

LOG_FILE="data/logs/train_dpo_sft.log"

echo "【训练进度监控】"
echo "=" | head -c 70 && echo ""

# 检查进程是否运行
if pgrep -f "train_dpo.py\|train_sft.py" > /dev/null; then
    echo "✅ 训练进程正在运行"
    PID=$(pgrep -f "train_dpo.py\|train_sft.py" | head -1)
    echo "   PID: $PID"
    
    # 检查GPU使用
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "📊 GPU使用情况:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "   GPU使用率: %s%% | 显存: %s/%s MB\n", $1, $2, $3}'
    fi
else
    echo "❌ 训练进程未运行"
fi

echo ""

# 检查日志文件
if [ -f "$LOG_FILE" ]; then
    echo "📊 最新日志输出（最后30行）："
    echo "-" | head -c 70 && echo ""
    tail -30 "$LOG_FILE"
    echo ""
    
    # 检查是否有错误
    if grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5 | grep -v "^$" > /dev/null; then
        echo "⚠️  发现错误："
        grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5
        echo ""
    fi
    
    # 检查训练进度
    if grep -i "epoch\|step\|loss\|progress" "$LOG_FILE" | tail -5 | grep -v "^$" > /dev/null; then
        echo "📈 训练进度："
        grep -i "epoch\|step\|loss" "$LOG_FILE" | tail -5
        echo ""
    fi
else
    echo "⚠️  日志文件不存在: $LOG_FILE"
fi

# 检查输出目录
echo "📁 输出目录:"
if [ -d "outputs/proactive_llm_colm_150states_dpo" ]; then
    echo "  ✅ DPO模型目录存在"
    ls -lh outputs/proactive_llm_colm_150states_dpo/*.json 2>/dev/null | head -3 || echo "     (模型文件尚未保存)"
else
    echo "  ⏳ DPO模型目录尚未创建"
fi

if [ -d "outputs/proactive_llm_colm_150states_sft" ]; then
    echo "  ✅ SFT模型目录存在"
    ls -lh outputs/proactive_llm_colm_150states_sft/*.json 2>/dev/null | head -3 || echo "     (模型文件尚未保存)"
else
    echo "  ⏳ SFT模型目录尚未创建"
fi

echo ""
echo "💡 实时查看进度："
echo "   tail -f $LOG_FILE"
