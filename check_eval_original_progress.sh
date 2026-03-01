#!/bin/bash
# 检查original query评估进度

echo "📊 Original Query评估进度"
echo "========================"

PID_FILE="eval_original.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 评估进程运行中 (PID: $PID)"
    else
        echo "✅ 评估进程已完成"
    fi
fi

if [ -f eval_original.log ]; then
    echo ""
    echo "📝 最新日志（最后15行）:"
    tail -15 eval_original.log | sed 's/^/   /'
    
    if grep -q "✅.*完成\|评估完成\|saved" eval_original.log; then
        echo ""
        echo "✅ 评估已完成！"
    fi
fi

# 检查输出文件
OUTPUT="eval_results/preliminary_eval_195pairs_original.json"
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "📁 输出文件:"
    echo "   $OUTPUT ($SIZE)"
fi

