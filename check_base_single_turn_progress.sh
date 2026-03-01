#!/bin/bash
# 检查base模型单轮评估进度

echo "📊 Base模型单轮评估进度"
echo "========================"

PID_FILE="eval_base_single_turn.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 评估进程运行中 (PID: $PID)"
    else
        echo "✅ 评估进程已完成"
    fi
fi

if [ -f eval_base_single_turn.log ]; then
    echo ""
    echo "📝 最新日志（最后20行）:"
    tail -20 eval_base_single_turn.log | sed 's/^/   /'
    
    if grep -q "✅.*完成\|评估完成\|saved" eval_base_single_turn.log; then
        echo ""
        echo "✅ 评估已完成！"
    fi
fi

# 检查输出文件
OUTPUT="eval_results/base_model_single_turn.json"
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "📁 输出文件:"
    echo "   $OUTPUT ($SIZE)"
    
    # 快速查看结果
    echo ""
    echo "📊 快速结果预览:"
    python3 << 'PYEOF'
import json
try:
    with open("eval_results/base_model_single_turn.json") as f:
        results = json.load(f)
    
    summary = results.get("summary", {})
    print(f"   Task Success Rate: {summary.get('task_success_rate', 0):.1f}%")
    print(f"   ({summary.get('task_success_count', 0)}/{summary.get('evaluated_samples', 0)})")
    print(f"   平均Task Score: {summary.get('avg_task_score', 0):.3f}")
except Exception as e:
    print(f"   ⚠️  结果文件还未完成或格式错误: {e}")
PYEOF
fi

