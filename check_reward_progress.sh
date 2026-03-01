#!/bin/bash
# 检查reward计算进度

echo "📊 Reward计算进度检查"
echo "===================="

# 检查进程
PID_FILE="reward_compute.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 进程运行中 (PID: $PID)"
        # 显示CPU和内存使用
        ps -p "$PID" -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{print "   CPU: "$2"%  MEM: "$3"%  运行时间: "$4}'
    else
        echo "✅ 进程已完成"
    fi
else
    echo "❌ PID文件不存在"
fi

# 检查日志
if [ -f reward_compute.log ]; then
    echo ""
    echo "📝 最新日志（最后15行）:"
    tail -15 reward_compute.log | sed 's/^/   /'
    
    # 检查是否完成
    if grep -q "✅.*preference pairs" reward_compute.log || grep -q "Saved.*prefs" reward_compute.log; then
        echo ""
        echo "✅ Reward计算已完成！"
    fi
else
    echo ""
    echo "⏳ 日志文件尚未创建"
fi

# 检查生成的文件
echo ""
echo "📁 生成的文件:"
PREFS_FILE="data/dpo/traj_colm_3turn_persona_150states_20260227_122315_prefs.jsonl"
if [ -f "$PREFS_FILE" ]; then
    LINES=$(wc -l < "$PREFS_FILE")
    SIZE=$(du -h "$PREFS_FILE" | cut -f1)
    echo "   ✅ Preference文件: $(basename $PREFS_FILE)"
    echo "      Pairs数: $LINES"
    echo "      大小: $SIZE"
else
    echo "   ⏳ Preference文件: 尚未生成"
fi

