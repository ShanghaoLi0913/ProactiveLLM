#!/bin/bash
# 检查baseline评估进度

echo "📊 Baseline评估进度"
echo "========================"

PID_FILE="eval_baseline.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 评估进程运行中 (PID: $PID)"
    else
        echo "✅ 评估进程已完成"
    fi
fi

if [ -f eval_baseline.log ]; then
    echo ""
    echo "📝 最新日志（最后20行）:"
    tail -20 eval_baseline.log | sed 's/^/   /'
    
    if grep -q "✅.*完成\|评估完成\|saved" eval_baseline.log; then
        echo ""
        echo "✅ 评估已完成！"
    fi
fi

# 检查输出文件
OUTPUT="eval_results/baseline_basemodel_masked.json"
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
    with open("eval_results/baseline_basemodel_masked.json") as f:
        results = json.load(f)
    
    print("   Task Success Rate:")
    for persona, stats in results["summary"].items():
        rate = stats.get("task_success_rate", 0)
        count = stats.get("task_success_count", 0)
        total = stats.get("task_evaluated_count", 0)
        print(f"     {persona}: {rate:.1f}% ({count}/{total})")
    
    # 总体
    total_success = sum(m["task_success_count"] for m in results["summary"].values())
    total_eval = sum(m["task_evaluated_count"] for m in results["summary"].values())
    if total_eval > 0:
        total_rate = total_success / total_eval * 100
        print(f"     总体: {total_rate:.1f}% ({total_success}/{total_eval})")
except Exception as e:
    print(f"   ⚠️  结果文件还未完成或格式错误: {e}")
PYEOF
fi

