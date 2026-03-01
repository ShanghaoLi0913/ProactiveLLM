#!/bin/bash
# 快速查看进度

echo "📊 数据生成进度快速查看"
echo "================================"

# 检查进程
if ps aux | grep -q "[g]enerate_trajectories.py"; then
    echo "✅ 进程运行中"
    ps aux | grep "[g]enerate_trajectories.py" | awk '{print "   CPU: "$3"%  MEM: "$4"%"}'
else
    echo "❌ 进程未运行"
fi

# 检查文件
TRAJ_FILE=$(ls -t data/logs/medium_traj_*.jsonl 2>/dev/null | head -1)
if [ -n "$TRAJ_FILE" ] && [ -s "$TRAJ_FILE" ]; then
    LINES=$(wc -l < "$TRAJ_FILE")
    SIZE=$(du -h "$TRAJ_FILE" | cut -f1)
    echo ""
    echo "📁 轨迹文件: $(basename $TRAJ_FILE)"
    echo "   行数: $LINES"
    echo "   大小: $SIZE"
    echo "   估算进度: ~$((LINES * 100 / 480))% (240 trajectories × ~2 turns)"
else
    echo ""
    echo "⏳ 轨迹文件: 尚未生成或为空"
fi

# 检查最新日志
if [ -f medium_generation.log ]; then
    echo ""
    echo "📝 最新日志:"
    tail -3 medium_generation.log | grep -v "^$" | sed 's/^/   /'
fi

