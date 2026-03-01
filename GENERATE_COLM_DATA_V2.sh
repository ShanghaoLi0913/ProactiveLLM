#!/bin/bash
# COLM 2026数据生成 - 体现Persona轮次差异
# 基于改进的select_mainline_action_from_persona逻辑

set -e

echo "=========================================="
echo "COLM 2026 Data Generation (3-turn + Persona-Aware)"
echo "=========================================="
echo ""
echo "设计要点:"
echo "  - max_turns=3 (上限，不是固定轮次)"
echo "  - Persona信息显式传递 (User Profile)"
echo "  - Persona差异体现在平均轮次:"
echo "    * Busy-Developer: ~1.16轮 (很少Clarify)"
echo "    * Experienced-Engineer: ~1.52轮 (适度Clarify)"
echo "    * Novice-Learner: ~1.98轮 (经常Clarify)"
echo ""
echo "----------------------------------------"
echo ""

# 参数配置
# ⚠️ 修复：使用有测试用例的数据源（bigcodebench_masked_states.jsonl）
# 原数据源 train_100states_coding.jsonl 没有测试用例，导致task success rate无法计算
DATASET_PATH="data/seeds/bigcodebench_masked_states.jsonl"
N_STATES=150  # ⭐ 150 states用于正式训练（120 train + 30 test，推荐）
TEST_RATIO=0.2  # ⭐ 20%作为测试集
MAX_TURNS=3  # ⭐ 3轮设计
LLM_MODEL="gpt-4o-mini"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_PREFIX="traj_colm_3turn_persona_${N_STATES}states_${TIMESTAMP}"
TRAJ_FILE="data/logs/${OUTPUT_PREFIX}.jsonl"
PREFS_FILE="data/dpo/${OUTPUT_PREFIX}_prefs.jsonl"
PREFS_TRAIN="data/dpo/${OUTPUT_PREFIX}_train_prefs.jsonl"
PREFS_TEST="data/dpo/${OUTPUT_PREFIX}_test_prefs.jsonl"

# Step 1: 生成轨迹数据
echo "[Step 1/4] 生成多轮轨迹数据..."
echo "  - States: $N_STATES"
echo "  - Personas: All (3)"
echo "  - Samples per (state, persona): 4 (2 Execute + 2 Clarify)"
echo "  - Sampling: heuristic (2×Execute + 2×Clarify)"
echo "  - Max turns: $MAX_TURNS"
echo "  - 目的: 确保有多个trajectories进入Turn 1+，可以生成Turn 1+的pairs"
echo ""

python scripts/generate_trajectories.py \
  --mode dataset \
  --dataset_path "$DATASET_PATH" \
  --domain coding \
  --n_states "$N_STATES" \
  --all_personas \
  --n_samples 4 \
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
  --trajectories "$TRAJ_FILE" \
  --out "data/dpo/${OUTPUT_PREFIX}_prefs.jsonl" \
  --w_task 1.0 \
  --w_interrupt 0.15 \
  --use_trajectory_level \
  --target_execute_ratio 0.7 \
  --rebalance_seed 42

echo ""
echo "✓ Preference pairs已生成: $PREFS_FILE"
echo ""

# Step 4: 分析Preference数据质量
echo "[Step 4/4] 分析Preference数据质量..."
echo ""

python << EOF
import json
from pathlib import Path

prefs_file = Path("$PREFS_FILE")
if not prefs_file.exists():
    print(f"⚠️  Preference文件不存在: {prefs_file}")
    exit(1)

with open(prefs_file) as f:
    prefs = [json.loads(line) for line in f]

print("=" * 70)
print("Preference数据质量报告")
print("=" * 70)
print(f"总preference pairs: {len(prefs)}")

# 检查测试用例
has_tests = sum(1 for p in prefs if p.get("state", {}).get("convcodeworld_tests"))
print(f"\n✅ 测试用例检查:")
print(f"  有测试用例的prefs: {has_tests}/{len(prefs)} ({has_tests/len(prefs)*100:.1f}%)")
if has_tests == len(prefs):
    print("  ✅ 所有prefs都包含测试用例！")
else:
    print(f"  ⚠️  有 {len(prefs) - has_tests} 个prefs缺少测试用例")

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

# Persona分布
persona_counts = Counter(p.get("persona", {}).get("name", "Unknown") for p in prefs)
print(f"\nPersona分布:")
for persona, count in persona_counts.items():
    print(f"  {persona}: {count} ({count/len(prefs)*100:.1f}%)")

print(f"{'='*70}\n")
EOF

# Step 5: 分割训练集和测试集
echo "[Step 5/5] 分割训练集和测试集..."
echo ""

python << EOF
import json
from pathlib import Path
from collections import defaultdict

prefs_file = Path("$PREFS_FILE")
if not prefs_file.exists():
    print(f"⚠️  Preference文件不存在: {prefs_file}")
    exit(1)

with open(prefs_file) as f:
    prefs = [json.loads(line) for line in f]

print("=" * 70)
print("数据分割 - Train/Test Split")
print("=" * 70)
print(f"总preference pairs: {len(prefs)}")

# 按state_id分组，确保同一个state的所有prefs在同一集合中
state_groups = defaultdict(list)
for p in prefs:
    state_id = p.get("state", {}).get("id", "unknown")
    state_groups[state_id].append(p)

print(f"唯一state数: {len(state_groups)}")

# 按state_id分割（确保同一个state不会同时出现在train和test中）
state_ids = sorted(state_groups.keys())
n_test_states = max(1, int(len(state_ids) * $TEST_RATIO))
test_state_ids = set(state_ids[:n_test_states])
train_state_ids = set(state_ids[n_test_states:])

print(f"\n分割配置:")
print(f"  测试集states: {len(test_state_ids)} ({len(test_state_ids)/len(state_ids)*100:.1f}%)")
print(f"  训练集states: {len(train_state_ids)} ({len(train_state_ids)/len(state_ids)*100:.1f}%)")

# 分割prefs
train_prefs = []
test_prefs = []

for p in prefs:
    state_id = p.get("state", {}).get("id", "unknown")
    if state_id in test_state_ids:
        test_prefs.append(p)
    else:
        train_prefs.append(p)

print(f"\n分割结果:")
print(f"  训练集prefs: {len(train_prefs)} ({len(train_prefs)/len(prefs)*100:.1f}%)")
print(f"  测试集prefs: {len(test_prefs)} ({len(test_prefs)/len(prefs)*100:.1f}%)")

# 保存训练集
train_file = Path("$PREFS_TRAIN")
train_file.parent.mkdir(parents=True, exist_ok=True)
with open(train_file, 'w', encoding='utf-8') as f:
    for p in train_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"\n✅ 训练集已保存: {train_file}")

# 保存测试集
test_file = Path("$PREFS_TEST")
test_file.parent.mkdir(parents=True, exist_ok=True)
with open(test_file, 'w', encoding='utf-8') as f:
    for p in test_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"✅ 测试集已保存: {test_file}")

# 验证数据质量
print(f"\n数据质量验证:")
train_has_tests = sum(1 for p in train_prefs if p.get("state", {}).get("convcodeworld_tests"))
test_has_tests = sum(1 for p in test_prefs if p.get("state", {}).get("convcodeworld_tests"))
print(f"  训练集测试用例: {train_has_tests}/{len(train_prefs)} ({train_has_tests/len(train_prefs)*100:.1f}%)")
print(f"  测试集测试用例: {test_has_tests}/{len(test_prefs)} ({test_has_tests/len(test_prefs)*100:.1f}%)")

# Persona分布
from collections import Counter
train_personas = Counter(p.get("persona", {}).get("name", "Unknown") for p in train_prefs)
test_personas = Counter(p.get("persona", {}).get("name", "Unknown") for p in test_prefs)

print(f"\n训练集Persona分布:")
for persona, count in train_personas.items():
    print(f"  {persona}: {count} ({count/len(train_prefs)*100:.1f}%)")

print(f"\n测试集Persona分布:")
for persona, count in test_personas.items():
    print(f"  {persona}: {count} ({count/len(test_prefs)*100:.1f}%)")

print(f"\n{'='*70}")
print("✅ 数据分割完成！")
print(f"{'='*70}\n")
EOF

echo ""
echo "=========================================="
echo "✓ 数据生成完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  1. 轨迹数据: $TRAJ_FILE"
echo "  2. 完整Preference pairs: $PREFS_FILE"
echo "  3. 训练集: $PREFS_TRAIN"
echo "  4. 测试集: $PREFS_TEST"
echo ""
echo "数据统计:"
echo "  - 总states: $N_STATES"
echo "  - 训练集: ~$((N_STATES * (1 - TEST_RATIO))) states"
echo "  - 测试集: ~$((N_STATES * TEST_RATIO)) states"
echo ""
echo "下一步:"
echo "  1. 检查上面的【按Persona统计】是否符合预期"
echo "  2. 检查数据分割是否合理（训练集足够，测试集有代表性）"
echo "  3. 如果满意，使用训练集训练模型: $PREFS_TRAIN"
echo "  4. 使用测试集评估模型: $PREFS_TEST"
echo ""
