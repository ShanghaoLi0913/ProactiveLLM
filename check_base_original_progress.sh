#!/bin/bash

echo "📊 Base模型单轮评估进度 (Original Query)"
echo "================================"

# 检查进程
if [ -f eval_base_single_turn_original.pid ]; then
    PID=$(cat eval_base_single_turn_original.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 评估进程运行中 (PID: $PID)"
    else
        echo "⚠️  评估进程已停止"
    fi
else
    # 尝试查找实际运行的进程
    ACTUAL_PID=$(ps aux | grep "evaluate_base_model_single_turn.*original" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$ACTUAL_PID" ]; then
        echo "✅ 评估进程运行中 (PID: $ACTUAL_PID)"
    else
        echo "⚠️  未找到评估进程"
    fi
fi

echo ""
echo "📝 最新日志（最后20行）:"
tail -20 eval_base_single_turn_original.log 2>/dev/null | strings | grep -v "Loading weights" | grep -v "Materializing" | grep -v "bitsandbytes" | tail -10 || echo "   (日志文件为空或无法读取)"

echo ""
echo "📄 结果文件:"
if [ -f eval_results/base_model_single_turn_original.json ]; then
    SIZE=$(ls -lh eval_results/base_model_single_turn_original.json | awk '{print $5}')
    echo "   ✅ 已生成 (大小: $SIZE)"
    
    # 快速查看结果
    python3 << 'PYEOF'
import json
try:
    with open("eval_results/base_model_single_turn_original.json") as f:
        d = json.load(f)
    s = d.get("summary", {})
    print(f"   📊 Task Success Rate: {s.get('task_success_rate', 0):.1f}%")
    print(f"   📈 已评估: {s.get('evaluated_samples', 0)}/{s.get('total_samples', 0)}")
except:
    print("   (文件可能还在写入中)")
PYEOF
else
    echo "   ⏳ 还未生成"
fi
