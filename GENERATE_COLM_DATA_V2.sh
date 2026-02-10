#!/bin/bash
# COLM 2026数据生成 - 体现Persona轮次差异
# 基于改进的select_mainline_action_from_persona逻辑

set -e

echo "=========================================="
echo "COLM 2026 Data Generation (2-turn)"
echo "=========================================="
echo ""
echo "设计要点:"
echo "  - max_turns=2 (上限，不是固定轮次)"
echo "  - Persona差异体现在平均轮次:"
echo "    * Busy-Developer: ~1.15轮 (很少Clarify)"
echo "    * Experienced-Engineer: ~1.40轮 (适度Clarify)"
echo "    * Novice-Learner: ~1.70轮 (经常Clarify)"
echo ""
echo "----------------------------------------"
echo ""

# 参数配置
DATASET_PATH="data/states/bigcode_100states_train.jsonl"
N_STATES=100  # 先用100测试，确认效果后再扩展到500
MAX_TURNS=2
LLM_MODEL="gpt-4o-mini"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_PREFIX="traj_colm_2turn_${N_STATES}states_${TIMESTAMP}"
TRAJ_FILE="data/logs/${OUTPUT_PREFIX}.jsonl"
PREFS_FILE="data/dpo/prefs_${OUTPUT_PREFIX}.jsonl"

# Step 1: 生成轨迹数据
echo "[Step 1/4] 生成2轮轨迹数据..."
echo "  - States: $N_STATES"
echo "  - Personas: All (3)"
echo "  - Samples per (state, persona): 2"
echo "  - Sampling: heuristic (Force Execute + Force Clarify)"
echo "  - Max turns: $MAX_TURNS"
echo ""

python scripts/generate_trajectories.py \
  --mode dataset \
  --dataset_path "$DATASET_PATH" \
  --domain coding \
  --n_states "$N_STATES" \
  --all_personas \
  --n_samples 2 \
  --sampling_strategy heuristic \
  --max_turns "$MAX_TURNS" \
  --llm_model "$LLM_MODEL" \
  --out "logs/${OUTPUT_PREFIX}.jsonl" \
  --seed "$SEED" \
  --temperature 0.7 \
  --top_p 0.9 \
  --progress_every 10

echo ""
echo "✓ 轨迹数据已生成: $TRAJ_FILE"
echo ""

# Step 2: 分析轨迹质量（验证persona差异）
echo "[Step 2/4] 分析轨迹质量（验证persona差异）..."
echo ""

python << EOF
import json
from collections import Counter, defaultdict

# 读取轨迹数据
with open("$TRAJ_FILE") as f:
    trajs = [json.loads(line) for line in f]

print("=" * 70)
print("轨迹数据质量报告 - Persona差异分析")
print("=" * 70)

# 按trajectory_id分组
traj_groups = defaultdict(list)
for t in trajs:
    traj_id = t.get("trajectory_id", "unknown")
    traj_groups[traj_id].append(t)

print(f"\n总体统计:")
print(f"  - 总trajectory数: {len(traj_groups)}")
print(f"  - 总turn数: {len(trajs)}")
print(f"  - 平均轮次: {len(trajs) / len(traj_groups):.2f}")

# 按persona统计
persona_stats = defaultdict(list)
for traj_id, turns in traj_groups.items():
    persona_name = turns[0]["persona"]["name"]
    traj_length = len(turns)
    first_action = turns[0]["action"]
    completed = any(t.get("task_completed", False) for t in turns)
    
    persona_stats[persona_name].append({
        "length": traj_length,
        "first_action": first_action,
        "completed": completed,
    })

print(f"\n{'='*70}")
print("【按Persona统计 - 关键差异指标】")
print(f"{'='*70}")

for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
    if persona not in persona_stats:
        print(f"\n⚠️  {persona}: 没有数据")
        continue
    
    stats = persona_stats[persona]
    avg_length = sum(s["length"] for s in stats) / len(stats)
    
    # First turn action分布
    first_actions = [s["first_action"] for s in stats]
    action_counts = Counter(first_actions)
    clarify_pct = action_counts.get("Clarify", 0) / len(stats) * 100
    execute_pct = action_counts.get("Execute", 0) / len(stats) * 100
    
    # 完成率
    completion_rate = sum(s["completed"] for s in stats) / len(stats) * 100
    
    # 轮次分布
    length_counts = Counter(s["length"] for s in stats)
    one_turn_pct = length_counts.get(1, 0) / len(stats) * 100
    two_turn_pct = length_counts.get(2, 0) / len(stats) * 100
    
    print(f"\n{persona}:")
    print(f"  轨迹数: {len(stats)}")
    print(f"  平均轮次: {avg_length:.2f}")
    print(f"  第1轮action分布:")
    print(f"    - Clarify: {clarify_pct:.1f}%")
    print(f"    - Execute: {execute_pct:.1f}%")
    print(f"  轨迹长度分布:")
    print(f"    - 1轮: {one_turn_pct:.1f}%")
    print(f"    - 2轮: {two_turn_pct:.1f}%")
    print(f"  任务完成率: {completion_rate:.1f}%")

# 期望值检查
print(f"\n{'='*70}")
print("【期望值检查】")
print(f"{'='*70}")
print(f"\n基于Clarify阈值设计，期望:")
print(f"  - Busy-Developer: 平均~1.15轮, Clarify@T0 ~15%")
print(f"  - Experienced-Engineer: 平均~1.40轮, Clarify@T0 ~40%")
print(f"  - Novice-Learner: 平均~1.70轮, Clarify@T0 ~70%")
print(f"\n如果实际值接近期望，说明逻辑正确✓")
print(f"如果差异很大，需要检查task_uncertainty分布或调整阈值")
print(f"{'='*70}\n")

EOF

echo ""
echo "✓ 轨迹质量分析完成"
echo ""

# Step 3: 计算Trajectory-level Rewards
echo "[Step 3/4] 计算Trajectory-level Rewards..."
echo ""

python reward/compute_rewards.py \
  --traj_path "$TRAJ_FILE" \
  --out "dpo/${OUTPUT_PREFIX}_prefs.jsonl" \
  --w_task 1.0 \
  --w_interrupt 0.5 \
  --reward_mode trajectory \
  --verbose

echo ""
echo "✓ Preference pairs已生成: $PREFS_FILE"
echo ""

# Step 4: 分析Preference数据质量
echo "[Step 4/4] 分析Preference数据质量..."
echo ""

python << EOF
import json

with open("$PREFS_FILE") as f:
    prefs = [json.loads(line) for line in f]

print("=" * 70)
print("Preference数据质量报告")
print("=" * 70)
print(f"总preference pairs: {len(prefs)}")

# 统计chosen/rejected action分布
from collections import Counter
chosen_actions = [p["chosen_action"] for p in prefs]
rejected_actions = [p["rejected_action"] for p in prefs]

print(f"\nChosen action分布:")
for action, count in Counter(chosen_actions).items():
    print(f"  {action}: {count} ({count/len(prefs)*100:.1f}%)")

print(f"\nRejected action分布:")
for action, count in Counter(rejected_actions).items():
    print(f"  {action}: {count} ({count/len(prefs)*100:.1f}%)")

# Reward margin分析
margins = [p["chosen_reward"] - p["rejected_reward"] for p in prefs]
avg_margin = sum(margins) / len(margins)
print(f"\nReward margin:")
print(f"  平均: {avg_margin:.3f}")
print(f"  最小: {min(margins):.3f}")
print(f"  最大: {max(margins):.3f}")

print(f"{'='*70}\n")
EOF

echo ""
echo "=========================================="
echo "✓ 数据生成完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  1. 轨迹数据: $TRAJ_FILE"
echo "  2. Preference pairs: $PREFS_FILE"
echo ""
echo "下一步:"
echo "  1. 检查上面的【按Persona统计】是否符合预期"
echo "  2. 如果平均轮次差异明显（Busy<Exp<Novice），说明成功✓"
echo "  3. 如果满意，扩展到500 states: 修改N_STATES=500"
echo "  4. 然后训练V17模型: bash TRAIN_COLM_V17.sh"
echo ""
