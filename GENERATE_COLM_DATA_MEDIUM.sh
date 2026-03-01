#!/bin/bash
# 中等规模数据生成测试

set -e

echo "=========================================="
echo "中等规模数据生成测试"
echo "=========================================="
echo ""

# 参数配置（中等规模）
DATASET_PATH="data/seeds/bigcodebench_masked_states.jsonl"
N_STATES=20  # 中等规模：20个states
TEST_RATIO=0.2
MAX_TURNS=3
LLM_MODEL="gpt-4o-mini"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_PREFIX="medium_traj_${N_STATES}states_${TIMESTAMP}"
TRAJ_FILE="data/logs/${OUTPUT_PREFIX}.jsonl"
PREFS_FILE="data/dpo/${OUTPUT_PREFIX}_prefs.jsonl"
PREFS_TRAIN="data/dpo/${OUTPUT_PREFIX}_train_prefs.jsonl"
PREFS_TEST="data/dpo/${OUTPUT_PREFIX}_test_prefs.jsonl"

# Step 1: 生成轨迹数据
echo "[Step 1/5] 生成多轮轨迹数据..."
echo "  - States: $N_STATES"
echo "  - Personas: All (3)"
echo "  - Samples per (state, persona): 4 (2 Execute + 2 Clarify)"
echo "  - Max turns: $MAX_TURNS"
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
  --progress_every 5

echo ""
echo "✓ 轨迹数据已生成: $TRAJ_FILE"
echo ""

# Step 2: 轨迹质量分析
echo "[Step 2/5] 轨迹质量分析..."
python << PYEOF
import json
from collections import Counter, defaultdict
from pathlib import Path

traj_file = Path("$TRAJ_FILE")
if not traj_file.exists():
    print(f"❌ 文件不存在: {traj_file}")
    exit(1)

with open(traj_file) as f:
    trajs = [json.loads(line) for line in f]

print("=" * 70)
print("轨迹数据质量报告")
print("=" * 70)
print(f"总trajectory turns: {len(trajs)}")

# 按trajectory_id分组
traj_groups = defaultdict(list)
for t in trajs:
    traj_id = t.get("trajectory_id", "unknown")
    traj_groups[traj_id].append(t)

print(f"总trajectories: {len(traj_groups)}")
print(f"平均轮次: {len(trajs) / len(traj_groups):.2f}")

# 按persona统计
persona_stats = defaultdict(lambda: {"trajs": [], "total_turns": 0, "execute_turns": 0, "clarify_turns": 0})

for traj_id, turns in traj_groups.items():
    persona_name = turns[0]["persona"]["name"]
    traj_length = len(turns)
    execute_count = sum(1 for t in turns if t.get("action") == "Execute")
    clarify_count = sum(1 for t in turns if t.get("action") == "Clarify")
    
    persona_stats[persona_name]["trajs"].append(traj_length)
    persona_stats[persona_name]["total_turns"] += traj_length
    persona_stats[persona_name]["execute_turns"] += execute_count
    persona_stats[persona_name]["clarify_turns"] += clarify_count

print(f"\n{'='*70}")
print("按Persona统计")
print(f"{'='*70}")

for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
    if persona not in persona_stats:
        continue
    
    stats = persona_stats[persona]
    n_trajs = len(stats["trajs"])
    avg_length = sum(stats["trajs"]) / n_trajs if n_trajs > 0 else 0
    
    print(f"\n{persona}:")
    print(f"  - 轨迹数: {n_trajs}")
    print(f"  - 平均轮次: {avg_length:.2f}")
    print(f"  - Execute turns: {stats['execute_turns']}")
    print(f"  - Clarify turns: {stats['clarify_turns']}")

# Action分布
actions = [t.get("action") for t in trajs]
action_counts = Counter(actions)
print(f"\n{'='*70}")
print("Action分布")
print(f"{'='*70}")
for action, count in action_counts.items():
    print(f"  {action}: {count} ({count/len(trajs)*100:.1f}%)")

# 检查是否有Execute
has_execute = sum(1 for t in trajs if t.get("action") == "Execute")
traj_with_execute = sum(1 for tid, turns in traj_groups.items() 
                        if any(t.get("action") == "Execute" for t in turns))
print(f"\n✅ Execute turns: {has_execute}")
print(f"✅ Trajectories with Execute: {traj_with_execute}/{len(traj_groups)}")

# 检查original_instruct_prompt
has_original = sum(1 for t in trajs if t.get("state", {}).get("original_instruct_prompt"))
print(f"✅ 有original_instruct_prompt: {has_original}/{len(trajs)} ({has_original/len(trajs)*100:.1f}%)")

print(f"{'='*70}\n")
PYEOF

# Step 3: 计算Rewards
echo "[Step 3/5] 计算Trajectory-level Rewards..."
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

# Step 4: Preference质量分析
echo "[Step 4/5] Preference质量分析..."
python << PYEOF
import json
from collections import Counter, defaultdict

prefs_file = Path("$PREFS_FILE")
if not prefs_file.exists():
    print(f"❌ 文件不存在: {prefs_file}")
    exit(1)

with open(prefs_file) as f:
    prefs = [json.loads(line) for line in f]

print("=" * 70)
print("Preference数据质量报告")
print("=" * 70)
print(f"总preference pairs: {len(prefs)}")

# 按persona统计
persona_prefs = defaultdict(list)
for pref in prefs:
    persona_name = pref.get("persona", {}).get("name", "Unknown")
    persona_prefs[persona_name].append(pref)

print(f"\n{'='*70}")
print("按Persona统计")
print(f"{'='*70}")

for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
    if persona not in persona_prefs:
        continue
    
    prefs_list = persona_prefs[persona]
    n_pairs = len(prefs_list)
    
    chosen_clarify = sum(1 for p in prefs_list if p["chosen_action"] == "Clarify")
    chosen_execute = sum(1 for p in prefs_list if p["chosen_action"] == "Execute")
    
    avg_margin = sum(p["chosen_reward"] - p["rejected_reward"] for p in prefs_list) / n_pairs
    avg_uncertainty = sum(p["state"].get("task_uncertainty", 0) for p in prefs_list) / n_pairs
    
    print(f"\n{persona}:")
    print(f"  - Pairs数: {n_pairs}")
    print(f"  - Chosen Clarify: {chosen_clarify} ({chosen_clarify/n_pairs*100:.1f}%)")
    print(f"  - Chosen Execute: {chosen_execute} ({chosen_execute/n_pairs*100:.1f}%)")
    print(f"  - 平均reward margin: {avg_margin:.3f}")
    print(f"  - 平均task_uncertainty: {avg_uncertainty:.2f}")

# Action分布
chosen_actions = [p["chosen_action"] for p in prefs]
rejected_actions = [p["rejected_action"] for p in prefs]

print(f"\n{'='*70}")
print("Action分布")
print(f"{'='*70}")
print(f"Chosen action分布:")
for action, count in Counter(chosen_actions).items():
    print(f"  {action}: {count} ({count/len(prefs)*100:.1f}%)")

print(f"\nRejected action分布:")
for action, count in Counter(rejected_actions).items():
    print(f"  {action}: {count} ({count/len(prefs)*100:.1f}%)")

# Reward margin
margins = [p["chosen_reward"] - p["rejected_reward"] for p in prefs]
avg_margin = sum(margins) / len(margins) if margins else 0
print(f"\nReward margin:")
print(f"  平均: {avg_margin:.3f}")
print(f"  最小: {min(margins):.3f}")
print(f"  最大: {max(margins):.3f}")

# 数据质量检查
has_tests = sum(1 for p in prefs if p.get("state", {}).get("convcodeworld_tests") or p.get("state", {}).get("test"))
has_original = sum(1 for p in prefs if p.get("state", {}).get("original_instruct_prompt"))
print(f"\n✅ 数据质量:")
print(f"  - 有测试用例: {has_tests}/{len(prefs)} ({has_tests/len(prefs)*100:.1f}%)")
print(f"  - 有original_instruct_prompt: {has_original}/{len(prefs)} ({has_original/len(prefs)*100:.1f}%)")

print(f"{'='*70}\n")
PYEOF

# Step 5: 分割训练集和测试集
echo "[Step 5/5] 分割训练集和测试集..."
python << PYEOF
import json
from pathlib import Path
from collections import defaultdict

prefs_file = Path("$PREFS_FILE")
if not prefs_file.exists():
    print(f"❌ 文件不存在: {prefs_file}")
    exit(1)

with open(prefs_file) as f:
    prefs = [json.loads(line) for line in f]

# 按state_id分组
state_groups = defaultdict(list)
for p in prefs:
    state_id = p.get("state", {}).get("id", "unknown")
    state_groups[state_id].append(p)

state_ids = sorted(state_groups.keys())
n_test_states = max(1, int(len(state_ids) * $TEST_RATIO))
test_state_ids = set(state_ids[:n_test_states])
train_state_ids = set(state_ids[n_test_states:])

train_prefs = [p for p in prefs if p.get("state", {}).get("id") in train_state_ids]
test_prefs = [p for p in prefs if p.get("state", {}).get("id") in test_state_ids]

# 保存
train_file = Path("$PREFS_TRAIN")
train_file.parent.mkdir(parents=True, exist_ok=True)
with open(train_file, 'w', encoding='utf-8') as f:
    for p in train_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

test_file = Path("$PREFS_TEST")
test_file.parent.mkdir(parents=True, exist_ok=True)
with open(test_file, 'w', encoding='utf-8') as f:
    for p in test_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"✅ 训练集: {len(train_prefs)} pairs ({len(train_state_ids)} states)")
print(f"✅ 测试集: {len(test_prefs)} pairs ({len(test_state_ids)} states)")
PYEOF

echo ""
echo "=========================================="
echo "✓ 中等规模数据生成完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  1. 轨迹数据: $TRAJ_FILE"
echo "  2. Preference pairs: $PREFS_FILE"
echo "  3. 训练集: $PREFS_TRAIN"
echo "  4. 测试集: $PREFS_TEST"
echo ""
