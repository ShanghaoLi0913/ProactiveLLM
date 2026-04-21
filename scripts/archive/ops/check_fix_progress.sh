#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 检查修复脚本的进度

TRAJ_FILE="data/logs/traj_colm_3turn_persona_150states_20260212_052612_20260212_052615.jsonl"
FIXED_FILE="data/logs/traj_colm_3turn_persona_150states_20260212_052612_20260212_052615_fixed.jsonl"

echo "=========================================="
echo "🔍 修复进度检查"
echo "=========================================="
echo ""

# 检查进程
echo "【进程检查】"
if pgrep -f "fix_trajectories_without_execute" > /dev/null; then
    echo "  ✅ 修复脚本正在运行"
    ps aux | grep fix_trajectories | grep -v grep | head -1
else
    echo "  ⚠️  修复脚本未运行"
fi
echo ""

# 检查文件
echo "【文件检查】"
if [ -f "$TRAJ_FILE" ]; then
    ORIGINAL_SIZE=$(wc -l < "$TRAJ_FILE")
    echo "  原始文件: $TRAJ_FILE"
    echo "  行数: $ORIGINAL_SIZE"
else
    echo "  ⚠️  原始文件不存在: $TRAJ_FILE"
fi

if [ -f "$FIXED_FILE" ]; then
    FIXED_SIZE=$(wc -l < "$FIXED_FILE")
    echo "  修复文件: $FIXED_FILE"
    echo "  行数: $FIXED_SIZE"
    
    if [ -f "$TRAJ_FILE" ]; then
        PROGRESS=$(echo "scale=1; $FIXED_SIZE * 100 / $ORIGINAL_SIZE" | bc)
        echo "  进度: ${PROGRESS}%"
    fi
else
    echo "  ⏳ 修复文件尚未生成"
fi
echo ""

# 使用Python进行详细统计
python3 << 'PYEOF'
import json
from pathlib import Path
from collections import defaultdict

traj_file = Path("data/logs/traj_colm_3turn_persona_150states_20260212_052612_20260212_052615.jsonl")
fixed_file = Path("data/logs/traj_colm_3turn_persona_150states_20260212_052612_20260212_052615_fixed.jsonl")

print("【详细统计】")

# 原始数据
if traj_file.exists():
    with open(traj_file, 'r') as f:
        all_trajs = [json.loads(line) for line in f if line.strip()]
    
    traj_groups = defaultdict(list)
    for traj in all_trajs:
        traj_id = traj.get("trajectory_id", "unknown")
        traj_groups[traj_id].append(traj)
    
    needs_fix = sum(1 for traj_id, turns in traj_groups.items() 
                   if not any(t.get("action") == "Execute" for t in turns))
    
    print(f"  原始conversations: {len(traj_groups)}")
    print(f"  需要修复: {needs_fix}")

# 修复后数据
if fixed_file.exists():
    with open(fixed_file, 'r') as f:
        fixed_trajs = [json.loads(line) for line in f if line.strip()]
    
    fixed_groups = defaultdict(list)
    for traj in fixed_trajs:
        traj_id = traj.get("trajectory_id", "unknown")
        fixed_groups[traj_id].append(traj)
    
    execute_count = sum(1 for traj_id, turns in fixed_groups.items() 
                       if any(t.get("action") == "Execute" for t in turns))
    fixed_turns = sum(1 for traj in fixed_trajs if traj.get("fixed", False))
    
    print(f"  修复后conversations: {len(fixed_groups)}")
    print(f"  有Execute: {execute_count}/{len(fixed_groups)} ({execute_count/len(fixed_groups)*100:.1f}%)")
    print(f"  修复的turns: {fixed_turns}")
    
    if traj_file.exists():
        total_needs_fix = sum(1 for traj_id, turns in traj_groups.items() 
                                  if not any(t.get("action") == "Execute" for t in turns))
        if total_needs_fix > 0:
            progress = (len(fixed_groups) - (len(traj_groups) - total_needs_fix)) / total_needs_fix * 100
            print(f"  修复进度: {progress:.1f}% ({fixed_turns}/{total_needs_fix})")
else:
    print("  ⏳ 修复文件尚未生成")

print("")
PYEOF

echo "=========================================="
echo "💡 提示: 运行 'watch -n 5 ./check_fix_progress.sh' 可以每5秒自动刷新"
echo "=========================================="
