#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 监控preference pairs生成进度

LOG_FILE="data/logs/compute_rewards_150states_final.log"
OUTPUT_FILE="data/dpo/prefs_colm_n4_150states.jsonl"

echo "【Preference Pairs生成进度监控】"
echo "=" | head -c 70 && echo ""

# 检查进程是否运行
if pgrep -f "compute_rewards.py.*150states" > /dev/null; then
    echo "✅ 进程正在运行"
    PID=$(pgrep -f "compute_rewards.py.*150states" | head -1)
    echo "   PID: $PID"
    
    # 检查CPU使用率
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null | tr -d ' ')
    if [ ! -z "$CPU" ]; then
        echo "   CPU使用率: ${CPU}%"
    fi
else
    echo "❌ 进程未运行"
fi

echo ""

# 检查日志文件
if [ -f "$LOG_FILE" ]; then
    echo "📊 最新日志输出（最后20行）："
    echo "-" | head -c 70 && echo ""
    tail -20 "$LOG_FILE"
    echo ""
    
    # 检查是否有错误
    if grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5 | grep -v "^$" > /dev/null; then
        echo "⚠️  发现错误："
        grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5
        echo ""
    fi
    
    # 检查进度信息
    if grep -i "progress\|computed rewards\|generated.*pairs" "$LOG_FILE" | tail -3 | grep -v "^$" > /dev/null; then
        echo "📈 进度信息："
        grep -i "progress\|computed rewards\|generated.*pairs" "$LOG_FILE" | tail -3
        echo ""
    fi
else
    echo "⚠️  日志文件不存在: $LOG_FILE"
fi

# 检查输出文件
if [ -f "$OUTPUT_FILE" ]; then
    PAIR_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1)
    echo "📁 输出文件: $OUTPUT_FILE"
    echo "   已生成pairs: $PAIR_COUNT"
    echo "   文件大小: $FILE_SIZE"
else
    echo "📁 输出文件尚未创建: $OUTPUT_FILE"
fi

echo ""
echo "💡 实时查看进度："
echo "   tail -f $LOG_FILE"
