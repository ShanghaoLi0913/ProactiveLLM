#!/bin/bash
# 小规模数据生成测试

set -e

echo "=========================================="
echo "小规模数据生成测试"
echo "=========================================="
echo ""

# 参数配置（小规模）
DATASET_PATH="data/seeds/bigcodebench_masked_states.jsonl"
N_STATES=5  # 只生成5个states
TEST_RATIO=0.2
MAX_TURNS=3
LLM_MODEL="gpt-4o-mini"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_PREFIX="test_traj_${N_STATES}states_${TIMESTAMP}"
TRAJ_FILE="data/logs/${OUTPUT_PREFIX}.jsonl"
PREFS_FILE="data/dpo/${OUTPUT_PREFIX}_prefs.jsonl"
PREFS_TRAIN="data/dpo/${OUTPUT_PREFIX}_train_prefs.jsonl"
PREFS_TEST="data/dpo/${OUTPUT_PREFIX}_test_prefs.jsonl"

# Step 1: 生成轨迹数据
echo "[Step 1/4] 生成多轮轨迹数据..."
echo "  - States: $N_STATES"
echo "  - Personas: All (3)"
echo "  - Samples per (state, persona): 2 (1 Execute + 1 Clarify)"
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
  --progress_every 1

echo ""
echo "✓ 轨迹数据已生成: $TRAJ_FILE"
echo ""

# Step 2: 快速分析
echo "[Step 2/4] 快速分析..."
python << PYEOF
import json
from collections import Counter, defaultdict

with open("$TRAJ_FILE") as f:
    trajs = [json.loads(line) for line in f]

print("=" * 70)
print("轨迹数据统计")
print("=" * 70)
print(f"总trajectory turns: {len(trajs)}")

# 按trajectory_id分组
traj_groups = defaultdict(list)
for t in trajs:
    traj_id = t.get("trajectory_id", "unknown")
    traj_groups[traj_id].append(t)

print(f"总trajectories: {len(traj_groups)}")
print(f"平均轮次: {len(trajs) / len(traj_groups):.2f}")

# Action分布
actions = [t.get("action") for t in trajs]
action_counts = Counter(actions)
print(f"\nAction分布:")
for action, count in action_counts.items():
    print(f"  {action}: {count}")

# Persona分布
personas = [t.get("persona", {}).get("name", "Unknown") for t in trajs]
persona_counts = Counter(personas)
print(f"\nPersona分布:")
for persona, count in persona_counts.items():
    print(f"  {persona}: {count}")

# 检查是否有Execute
has_execute = sum(1 for t in trajs if t.get("action") == "Execute")
print(f"\n✅ Execute turns: {has_execute}")
print(f"✅ Clarify turns: {len(trajs) - has_execute}")

# 检查补Execute的情况
traj_with_execute = sum(1 for tid, turns in traj_groups.items() 
                        if any(t.get("action") == "Execute" for t in turns))
print(f"✅ Trajectories with Execute: {traj_with_execute}/{len(traj_groups)}")

print(f"{'='*70}\n")
PYEOF

# Step 3: 计算Rewards
echo "[Step 3/4] 计算Trajectory-level Rewards..."
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

# Step 4: 快速分析Preference
echo "[Step 4/4] 分析Preference数据..."
python << PYEOF
import json
from collections import Counter

with open("$PREFS_FILE") as f:
    prefs = [json.loads(line) for line in f]

print("=" * 70)
print("Preference数据统计")
print("=" * 70)
print(f"总preference pairs: {len(prefs)}")

# Action分布
chosen_actions = [p["chosen_action"] for p in prefs]
rejected_actions = [p["rejected_action"] for p in prefs]

print(f"\nChosen action分布:")
for action, count in Counter(chosen_actions).items():
    print(f"  {action}: {count}")

print(f"\nRejected action分布:")
for action, count in Counter(rejected_actions).items():
    print(f"  {action}: {count}")

# Reward margin
margins = [p["chosen_reward"] - p["rejected_reward"] for p in prefs]
avg_margin = sum(margins) / len(margins) if margins else 0
print(f"\nReward margin: {avg_margin:.3f}")

print(f"{'='*70}\n")
PYEOF

echo ""
echo "=========================================="
echo "✓ 小规模数据生成完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  1. 轨迹数据: $TRAJ_FILE"
echo "  2. Preference pairs: $PREFS_FILE"
echo ""
