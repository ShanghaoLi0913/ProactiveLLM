#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 检查轨迹生成进度

echo "=" | awk '{printf "%.0s=", $1}' && echo "="
echo "📊 轨迹生成进度检查"
echo "=" | awk '{printf "%.0s=", $1}' && echo "="
echo ""

# 检查进程状态
echo "【进程状态】"
echo "────────────────────────────────────────────────────────────────────────────────"
if pgrep -f "generate_trajectories.py" > /dev/null; then
    PID=$(pgrep -f "generate_trajectories.py" | head -1)
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')
    echo "✅ 进程正在运行"
    echo "   PID: $PID"
    echo "   CPU: ${CPU}%"
    echo "   内存: ${MEM}%"
else
    echo "❌ 进程未运行"
fi
echo ""

# 检查输出文件
echo "【输出文件统计】"
echo "────────────────────────────────────────────────────────────────────────────────"
# 检查两个可能的输出路径
OUTPUT_FILES=$(ls logs/traj_colm_n4_150states_*.jsonl data/logs/traj_colm_n4_150states_*.jsonl 2>/dev/null | sort -u)
if [ -n "$OUTPUT_FILES" ]; then
    TOTAL_LINES=0
    TOTAL_SIZE=0
    FILE_COUNT=0
    
    for file in $OUTPUT_FILES; do
        if [ -f "$file" ]; then
            LINES=$(wc -l < "$file" 2>/dev/null || echo "0")
            SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            TOTAL_LINES=$((TOTAL_LINES + LINES))
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
            FILE_COUNT=$((FILE_COUNT + 1))
        fi
    done
    
    echo "✅ 已生成 $FILE_COUNT 个输出文件"
    echo "   总trajectories: $TOTAL_LINES"
    echo "   总大小: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "${TOTAL_SIZE} bytes")"
    
    # 计算进度
    EXPECTED=1800
    if [ $TOTAL_LINES -gt 0 ]; then
        PROGRESS=$(echo "scale=2; $TOTAL_LINES * 100 / $EXPECTED" | bc 2>/dev/null || echo "0")
        REMAINING=$((EXPECTED - TOTAL_LINES))
        echo "   进度: ${PROGRESS}% ($TOTAL_LINES / $EXPECTED)"
        echo "   剩余: $REMAINING trajectories"
    fi
    
    # 显示最新文件
    echo ""
    echo "   最新文件:"
    ls -lht logs/traj_colm_n4_150states_*.jsonl data/logs/traj_colm_n4_150states_*.jsonl 2>/dev/null | head -3 | awk '{print "     " $9 " (" $5 ")"}'
else
    echo "⏳ 输出文件尚未生成"
fi
echo ""

# 检查日志文件
echo "【日志文件】"
echo "────────────────────────────────────────────────────────────────────────────────"
LOG_FILE="data/logs/generation_150states_console.log"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$LOG_SIZE" -gt 0 ]; then
        echo "✅ 日志文件存在，大小: $(numfmt --to=iec-i --suffix=B $LOG_SIZE 2>/dev/null || echo "${LOG_SIZE} bytes")"
        echo ""
        echo "   最后10行日志:"
        tail -10 "$LOG_FILE" | sed 's/^/     /'
    else
        echo "⏳ 日志文件为空（可能还在缓冲）"
    fi
else
    echo "⚠️  日志文件不存在"
fi
echo ""

# 检查是否有错误
echo "【错误检查】"
echo "────────────────────────────────────────────────────────────────────────────────"
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    ERROR_COUNT=$(grep -i "error\|exception\|traceback" "$LOG_FILE" | wc -l | tr -d ' ')
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  发现 $ERROR_COUNT 个错误/异常"
        echo "   最近的错误:"
        grep -i "error\|exception" "$LOG_FILE" | tail -3 | sed 's/^/     /'
    else
        echo "✅ 未发现错误"
    fi
else
    echo "⏳ 无法检查错误（日志文件为空）"
fi
echo ""

# 总结
echo "=" | awk '{printf "%.0s=", $1}' && echo "="
echo "💡 提示:"
echo "  - 实时监控: tail -f $LOG_FILE"
echo "  - 查看输出: ls -lh logs/traj_colm_n4_150states_*.jsonl data/logs/traj_colm_n4_150states_*.jsonl"
echo "  - 停止任务: pkill -f generate_trajectories.py"
echo "=" | awk '{printf "%.0s=", $1}' && echo "="
