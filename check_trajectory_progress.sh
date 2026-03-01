#!/bin/bash
# 查看轨迹生成进度

echo "📊 轨迹生成进度检查"
echo "===================="

# 1. 检查是否有正在运行的轨迹生成进程
echo ""
echo "1. 运行中的进程:"
if ps aux | grep -q "[g]enerate_trajectories.py"; then
    echo "   ✅ 轨迹生成进程运行中"
    ps aux | grep "[g]enerate_trajectories.py" | grep -v grep | awk '{print "   PID: "$2"  CPU: "$3"%  MEM: "$4"%  CMD: "$11" "$12" "$13" "$14}'
else
    echo "   ❌ 没有运行中的轨迹生成进程"
fi

# 2. 检查最新的轨迹文件
echo ""
echo "2. 最新的轨迹文件:"
LATEST_TRAJ=$(ls -t data/logs/traj_*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST_TRAJ" ]; then
    LINES=$(wc -l < "$LATEST_TRAJ" 2>/dev/null || echo "0")
    SIZE=$(du -h "$LATEST_TRAJ" 2>/dev/null | cut -f1)
    MTIME=$(stat -c %y "$LATEST_TRAJ" 2>/dev/null | cut -d'.' -f1)
    echo "   ✅ 文件: $(basename $LATEST_TRAJ)"
    echo "      行数: $LINES"
    echo "      大小: $SIZE"
    echo "      修改时间: $MTIME"
    
    # 估算进度（如果是150 states）
    if echo "$LATEST_TRAJ" | grep -q "150states"; then
        # 150 states × 3 personas × 4 samples = 1800 trajectories
        # 每个trajectory平均约1.74 turns，所以总行数约3132
        ESTIMATED_TOTAL=3132
        if [ "$LINES" -gt 0 ] && [ "$LINES" -lt "$ESTIMATED_TOTAL" ]; then
            PROGRESS=$((LINES * 100 / ESTIMATED_TOTAL))
            REMAINING=$((ESTIMATED_TOTAL - LINES))
            echo "      估算进度: ~${PROGRESS}% (剩余约${REMAINING}行)"
        elif [ "$LINES" -ge "$ESTIMATED_TOTAL" ]; then
            echo "      ✅ 已完成（行数已达到或超过预期）"
        fi
    fi
else
    echo "   ⏳ 尚未生成轨迹文件"
fi

# 3. 检查日志文件
echo ""
echo "3. 日志文件:"
if [ -f "large_generation.log" ]; then
    echo "   ✅ large_generation.log 存在"
    echo "   最新日志（最后10行）:"
    tail -10 large_generation.log | sed 's/^/      /'
    
    if grep -q "✓.*数据生成完成\|完成时间" large_generation.log; then
        echo ""
        echo "   ✅ 数据生成已完成！"
    fi
else
    echo "   ⏳ large_generation.log 不存在"
fi

# 4. 检查是否有其他相关日志
echo ""
echo "4. 其他相关日志:"
for log in reward_compute.log reward_compute_relaxed.log; do
    if [ -f "$log" ]; then
        SIZE=$(du -h "$log" 2>/dev/null | cut -f1)
        LINES=$(wc -l < "$log" 2>/dev/null || echo "0")
        echo "   ✅ $log ($SIZE, $LINES 行)"
        if [ "$LINES" -gt 0 ]; then
            echo "      最后一行: $(tail -1 "$log" | cut -c1-80)..."
        fi
    fi
done

# 5. 统计已生成的trajectories
echo ""
echo "5. 已生成的trajectories统计:"
if [ -n "$LATEST_TRAJ" ] && [ -f "$LATEST_TRAJ" ]; then
    python3 << PYEOF
import json
from collections import defaultdict

try:
    with open("$LATEST_TRAJ") as f:
        turns = [json.loads(line) for line in f if line.strip()]
    
    traj_groups = defaultdict(list)
    for turn in turns:
        traj_id = turn.get("trajectory_id", "unknown")
        traj_groups[traj_id].append(turn)
    
    persona_count = defaultdict(int)
    action_count = defaultdict(int)
    for turn in turns:
        persona = turn.get("persona", {})
        if isinstance(persona, dict):
            persona_name = persona.get("name", "unknown")
        else:
            persona_name = str(persona)
        persona_count[persona_name] += 1
        action_count[turn.get("action", "unknown")] += 1
    
    print(f"   总turns: {len(turns)}")
    print(f"   总trajectories: {len(traj_groups)}")
    print(f"   Persona分布:")
    for p, count in sorted(persona_count.items()):
        print(f"     {p}: {count} turns")
    print(f"   Action分布:")
    for a, count in sorted(action_count.items()):
        print(f"     {a}: {count} turns")
except Exception as e:
    print(f"   错误: {e}")
PYEOF
fi

echo ""
