#!/bin/bash
echo "📊 放宽版Reward计算进度"
echo "======================"

PID_FILE="reward_compute_relaxed.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 进程运行中 (PID: $PID)"
    else
        echo "✅ 进程已完成"
    fi
fi

if [ -f reward_compute_relaxed.log ]; then
    echo ""
    echo "📝 最新日志（最后20行）:"
    tail -20 reward_compute_relaxed.log | sed 's/^/   /'
    
    if grep -q "✅.*preference pairs\|Saved.*prefs" reward_compute_relaxed.log; then
        echo ""
        echo "✅ 计算已完成！"
        
        # 统计pairs数
        PREFS_FILE="data/dpo/traj_colm_3turn_persona_150states_20260227_122315_prefs_relaxed.jsonl"
        if [ -f "$PREFS_FILE" ]; then
            LINES=$(wc -l < "$PREFS_FILE")
            echo "   Pairs数: $LINES"
        fi
    fi
fi

