#!/bin/bash
# 150 states数据生成 - 后台运行版本
# 使用nohup确保SSH断开后继续运行

set -e

echo "=========================================="
echo "COLM 2026 Data Generation (150 States)"
echo "=========================================="
echo ""
echo "⚠️  此脚本将在后台运行，即使SSH断开也会继续"
echo "📝 日志文件: large_generation.log"
echo ""

# 参数配置
DATASET_PATH="data/seeds/bigcodebench_masked_states.jsonl"
N_STATES=150
TEST_RATIO=0.2
MAX_TURNS=3
LLM_MODEL="gpt-4o-mini"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_PREFIX="traj_colm_3turn_persona_${N_STATES}states_${TIMESTAMP}"
TRAJ_FILE="data/logs/${OUTPUT_PREFIX}.jsonl"
PREFS_FILE="data/dpo/${OUTPUT_PREFIX}_prefs.jsonl"
PREFS_TRAIN="data/dpo/${OUTPUT_PREFIX}_train_prefs.jsonl"
PREFS_TEST="data/dpo/${OUTPUT_PREFIX}_test_prefs.jsonl"

LOG_FILE="large_generation.log"

# 记录开始时间
echo "==========================================" >> "$LOG_FILE"
echo "数据生成开始: $(date)" >> "$LOG_FILE"
echo "States: $N_STATES" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Step 1: 生成轨迹数据
echo "[Step 1/5] 生成多轮轨迹数据..." | tee -a "$LOG_FILE"
echo "  - States: $N_STATES" | tee -a "$LOG_FILE"
echo "  - Personas: All (3)" | tee -a "$LOG_FILE"
echo "  - Samples per (state, persona): 4" | tee -a "$LOG_FILE"
echo "  - Max turns: $MAX_TURNS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

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
  --progress_every 10 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✓ 轨迹数据已生成: $TRAJ_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2: 计算Rewards
echo "[Step 2/5] 计算Trajectory-level Rewards..." | tee -a "$LOG_FILE"
python reward/compute_rewards.py \
  --trajectories "$TRAJ_FILE" \
  --out "$PREFS_FILE" \
  --w_task 1.0 \
  --w_interrupt 0.15 \
  --use_trajectory_level \
  --target_execute_ratio 0.7 \
  --rebalance_seed 42 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✓ Preference pairs已生成: $PREFS_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 3: 分割训练集和测试集
echo "[Step 3/5] 分割训练集和测试集..." | tee -a "$LOG_FILE"
python << PYEOF | tee -a "$LOG_FILE"
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

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "✓ 150 states数据生成完成！" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "生成的文件:" | tee -a "$LOG_FILE"
echo "  1. 轨迹数据: $TRAJ_FILE" | tee -a "$LOG_FILE"
echo "  2. Preference pairs: $PREFS_FILE" | tee -a "$LOG_FILE"
echo "  3. 训练集: $PREFS_TRAIN" | tee -a "$LOG_FILE"
echo "  4. 测试集: $PREFS_TEST" | tee -a "$LOG_FILE"
