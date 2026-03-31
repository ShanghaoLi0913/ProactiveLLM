#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 检查preference pairs生成进度

echo "=========================================="
echo "Preference Pairs 生成进度检查"
echo "=========================================="
echo ""

# 1. 检查进程状态
echo "【进程状态】"
echo "────────────────────────────────────────────────────────────────────────────────"
PID=$(pgrep -f compute_rewards.py | head -1)
if [ -n "$PID" ]; then
    CPU_USAGE=$(ps -p $PID -o %cpu | tail -1 | xargs)
    MEM_USAGE=$(ps -p $PID -o %mem | tail -1 | xargs)
    echo "✅ 进程正在运行"
    echo "   PID: $PID"
    echo "   CPU: ${CPU_USAGE}%"
    echo "   内存: ${MEM_USAGE}%"
else
    echo "❌ 进程未运行"
fi
echo ""

# 2. 检查输出文件
echo "【输出文件统计】"
echo "────────────────────────────────────────────────────────────────────────────────"
OUTPUT_FILE="data/dpo/prefs_colm_n4_150states.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
    SIZE=$(du -b "$OUTPUT_FILE" | awk '{print $1}')
    EXPECTED=1806  # 预计的pairs数量（trajectory-level模式）
    
    if [ "$LINES" -gt 0 ]; then
        PROGRESS_PERCENT=$(echo "scale=1; $LINES * 100 / $EXPECTED" | bc 2>/dev/null || echo "计算中")
        REMAINING=$((EXPECTED - LINES))
        
        echo "✅ 输出文件存在"
        echo "   文件: $OUTPUT_FILE"
        echo "   大小: $(numfmt --to=iec-i --suffix=B --format="%.1f" $SIZE 2>/dev/null || echo "${SIZE}B")"
        echo "   已生成pairs: $LINES"
        echo "   预计总数: $EXPECTED"
        echo "   进度: ${PROGRESS_PERCENT}%"
        echo "   剩余: ${REMAINING} pairs"
        
        # 显示最后几行示例
        if [ "$LINES" -gt 0 ]; then
            echo ""
            echo "   最后生成的pair示例:"
            tail -1 "$OUTPUT_FILE" | python3 -m json.tool 2>/dev/null | head -10 || tail -1 "$OUTPUT_FILE" | head -c 200
        fi
    else
        echo "⏳ 文件存在但为空"
    fi
else
    echo "⏳ 输出文件尚未生成"
fi
echo ""

# 3. 检查日志文件
echo "【日志文件】"
echo "────────────────────────────────────────────────────────────────────────────────"
LOG_FILE="data/logs/compute_rewards_150states.log"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -b "$LOG_FILE" | awk '{print $1}')
    echo "✅ 日志文件存在，大小: $(numfmt --to=iec-i --suffix=B --format="%.1f" $LOG_SIZE 2>/dev/null || echo "${LOG_SIZE}B")"
    echo ""
    echo "   最后15行日志:"
    tail -15 "$LOG_FILE"
else
    echo "❌ 日志文件不存在"
fi
echo ""

# 4. 错误检查
echo "【错误检查】"
echo "────────────────────────────────────────────────────────────────────────────────"
if [ -f "$LOG_FILE" ]; then
    if grep -q "Error\|Exception\|Traceback" "$LOG_FILE"; then
        echo "❌ 发现错误或异常！"
        echo ""
        echo "   错误摘要:"
        grep -i "error\|exception" "$LOG_FILE" | tail -5
    else
        echo "✅ 未发现错误"
    fi
else
    echo "⚠️  无法检查错误（日志文件不存在）"
fi
echo ""

echo "=========================================="
echo "💡 提示:"
echo "  - 实时监控: tail -f $LOG_FILE"
echo "  - 查看输出: tail -f $OUTPUT_FILE"
echo "  - 停止任务: pkill -f compute_rewards.py"
echo "=========================================="
