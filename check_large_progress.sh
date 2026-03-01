#!/bin/bash
# 检查150 states数据生成进度

echo "📊 150 States数据生成进度检查"
echo "================================"

# 检查进程
if ps aux | grep -q "[G]ENERATE_COLM_DATA_150STATES.sh\|[g]enerate_trajectories.py.*150"; then
    echo "✅ 进程运行中"
    ps aux | grep -E "[G]ENERATE_COLM_DATA_150STATES.sh|[g]enerate_trajectories.py.*150" | awk '{print "   CPU: "$3"%  MEM: "$4"%"}'
else
    echo "❌ 进程未运行（可能已完成或未启动）"
fi

# 检查日志
if [ -f large_generation.log ]; then
    echo ""
    echo "📝 最新日志（最后10行）:"
    tail -10 large_generation.log | sed 's/^/   /'
    
    # 检查是否完成
    if grep -q "✓ 150 states数据生成完成" large_generation.log; then
        echo ""
        echo "✅ 数据生成已完成！"
    fi
else
    echo ""
    echo "⏳ 日志文件尚未创建"
fi

# 检查生成的文件
echo ""
echo "📁 生成的文件:"
TRAJ_FILE=$(ls -t data/logs/traj_colm_3turn_persona_150states_*.jsonl 2>/dev/null | head -1)
if [ -n "$TRAJ_FILE" ] && [ -s "$TRAJ_FILE" ]; then
    LINES=$(wc -l < "$TRAJ_FILE")
    SIZE=$(du -h "$TRAJ_FILE" | cut -f1)
    echo "   ✅ 轨迹文件: $(basename $TRAJ_FILE)"
    echo "      行数: $LINES"
    echo "      大小: $SIZE"
    # 估算进度（150 states × 3 personas × 4 samples = 1800 trajectories）
    # 每个trajectory平均约1.7轮，所以总行数约3060
    ESTIMATED_TOTAL=3060
    if [ $LINES -lt $ESTIMATED_TOTAL ]; then
        PROGRESS=$((LINES * 100 / ESTIMATED_TOTAL))
        echo "      估算进度: ~${PROGRESS}%"
    fi
else
    echo "   ⏳ 轨迹文件: 尚未生成"
fi

PREFS_FILE=$(ls -t data/dpo/traj_colm_3turn_persona_150states_*_prefs.jsonl 2>/dev/null | head -1)
if [ -n "$PREFS_FILE" ] && [ -s "$PREFS_FILE" ]; then
    LINES=$(wc -l < "$PREFS_FILE")
    SIZE=$(du -h "$PREFS_FILE" | cut -f1)
    echo "   ✅ Preference文件: $(basename $PREFS_FILE)"
    echo "      Pairs数: $LINES"
    echo "      大小: $SIZE"
else
    echo "   ⏳ Preference文件: 尚未生成"
fi

echo ""
