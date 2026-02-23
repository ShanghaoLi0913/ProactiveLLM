#!/bin/bash
# 检查验证数据的修复效果

echo "=" * 80
echo "🔍 验证修复效果检查"
echo "=" * 80

# 找到最新的验证轨迹文件
LATEST_TRAJ=$(find data/logs -name "traj_colm_3turn_persona_20states_*.jsonl" -type f 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_TRAJ" ]; then
    echo "⚠️  未找到验证轨迹文件"
    exit 1
fi

echo ""
echo "📂 检查文件: $LATEST_TRAJ"
echo ""

python3 << EOF
import json
from pathlib import Path
from collections import Counter, defaultdict

traj_file = Path("$LATEST_TRAJ")

if not traj_file.exists():
    print("⚠️  轨迹文件不存在")
    exit(1)

with open(traj_file, 'r') as f:
    trajs = [json.loads(line) for line in f]

print("=" * 80)
print("🔍 修复效果验证")
print("=" * 80)

# 按trajectory_id分组
traj_groups = defaultdict(list)
for t in trajs:
    traj_id = t.get("trajectory_id", "unknown")
    traj_groups[traj_id].append(t)

print(f"\n【基本信息】")
print(f"  总trajectories: {len(trajs)}")
print(f"  唯一conversations: {len(traj_groups)}")
print(f"  平均轮次: {len(trajs) / len(traj_groups):.2f}" if traj_groups else "N/A")

# 检查1: Execute之后是否还有Execute
print(f"\n【检查1: Execute之后是否还有Execute】")
print("-" * 80)
execute_after_execute = []
for traj_id, turns in traj_groups.items():
    actions = [t.get("action", "Unknown") for t in turns]
    for i in range(len(actions) - 1):
        if actions[i] == "Execute" and actions[i+1] == "Execute":
            execute_after_execute.append((traj_id, actions))

if execute_after_execute:
    print(f"  ❌ 发现 {len(execute_after_execute)} 个Execute-Execute序列:")
    for tid, actions in execute_after_execute[:5]:
        print(f"    {tid}: {actions}")
else:
    print(f"  ✅ 没有Execute-Execute序列（修复成功！）")

# 检查2: 多轮对话中的persona差异
print(f"\n【检查2: 多轮对话中的persona差异】")
print("-" * 80)

persona_turn_stats = defaultdict(lambda: {"conversations": [], "total_turns": 0})

for traj_id, turns in traj_groups.items():
    persona_name = turns[0].get("persona", {}).get("name", "Unknown")
    turn_count = len(turns)
    first_action = turns[0].get("action", "Unknown")
    persona_turn_stats[persona_name]["conversations"].append({
        "turns": turn_count,
        "first_action": first_action,
    })
    persona_turn_stats[persona_name]["total_turns"] += turn_count

for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
    if persona not in persona_turn_stats:
        continue
    
    stats = persona_turn_stats[persona]
    convs = stats["conversations"]
    total_turns = stats["total_turns"]
    avg_turns = total_turns / len(convs) if convs else 0
    
    # 多轮对话统计
    multi_turn = sum(1 for c in convs if c["turns"] >= 2)
    three_turn = sum(1 for c in convs if c["turns"] >= 3)
    
    print(f"\n{persona}:")
    print(f"  Conversations: {len(convs)}")
    print(f"  平均轮次: {avg_turns:.2f}")
    print(f"  2轮+: {multi_turn} ({multi_turn/len(convs)*100:.1f}%)")
    print(f"  3轮+: {three_turn} ({three_turn/len(convs)*100:.1f}%)")

# 检查3: 多轮对话的action序列
print(f"\n【检查3: 多轮对话的action序列】")
print("-" * 80)

multi_turn_patterns = []
for traj_id, turns in traj_groups.items():
    if len(turns) >= 2:
        actions = tuple(t.get("action", "Unknown") for t in turns)
        persona = turns[0].get("persona", {}).get("name", "Unknown")
        multi_turn_patterns.append((persona, actions))

if multi_turn_patterns:
    print(f"  多轮对话模式（前10个）:")
    for persona, pattern in multi_turn_patterns[:10]:
        print(f"    {persona}: {pattern}")
    
    # 统计模式
    pattern_counts = Counter(multi_turn_patterns)
    print(f"\n  模式分布:")
    for (persona, pattern), count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {persona}: {pattern} - {count}次")
else:
    print("  ⚠️  没有多轮对话")

# 检查4: Persona差异是否明显
print(f"\n【检查4: Persona差异是否明显】")
print("-" * 80)

avg_turns_by_persona = {}
for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
    if persona in persona_turn_stats:
        stats = persona_turn_stats[persona]
        convs = stats["conversations"]
        total_turns = stats["total_turns"]
        avg_turns = total_turns / len(convs) if convs else 0
        avg_turns_by_persona[persona] = avg_turns

if len(avg_turns_by_persona) == 3:
    busy_avg = avg_turns_by_persona.get("Busy-Developer", 0)
    exp_avg = avg_turns_by_persona.get("Experienced-Engineer", 0)
    novice_avg = avg_turns_by_persona.get("Novice-Learner", 0)
    
    print(f"  平均轮次:")
    print(f"    Busy-Developer: {busy_avg:.2f}")
    print(f"    Experienced-Engineer: {exp_avg:.2f}")
    print(f"    Novice-Learner: {novice_avg:.2f}")
    
    if busy_avg < exp_avg < novice_avg:
        print(f"  ✅ Persona差异明显（Busy < Experienced < Novice）")
    else:
        print(f"  ⚠️  Persona差异不明显")

print("\n" + "=" * 80)
EOF
