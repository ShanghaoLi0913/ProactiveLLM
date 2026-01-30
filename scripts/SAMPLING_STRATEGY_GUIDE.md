# 采样策略使用指南

## 概述

现在支持多样化的采样策略，确保生成的数据既有"因为直接做而成功/失败"的例子，也有"因为提问而成功/断连"的例子。

## 新增功能

### 1. 多Persona支持 (`--all_personas`)

每个Task在所有Persona维度下都生成轨迹：

```bash
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 5 \
  --out logs/traj_all_personas.jsonl \
  --llm_model gpt-4o-mini \
  --max_turns 5 \
  --all_personas
```

**结果**: 每个state × 3个personas = 总共15个对话

### 2. 强制起始动作 (`--force_first_action`)

手动控制Assistant的起始动作：

```bash
# 强制首轮必须 Execute（盲猜）
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 5 --out logs/traj_force_execute.jsonl \
  --llm_model gpt-4o-mini --force_first_action Execute

# 强制首轮必须 Clarify（提问）
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 5 --out logs/traj_force_clarify.jsonl \
  --llm_model gpt-4o-mini --force_first_action Clarify
```

### 3. Heuristic采样策略 (`--sampling_strategy heuristic` + `--n_samples`)

自动为每个(state, persona)组合生成多样本：

```bash
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 5 --out logs/traj_heuristic.jsonl \
  --llm_model gpt-4o-mini \
  --sampling_strategy heuristic \
  --n_samples 3
```

**Heuristic策略的逻辑**：
- `--n_samples 1`: 生成1个样本，强制 `Execute`（盲猜）
- `--n_samples 2`: 生成2个样本：强制 `Execute` + 强制 `Clarify`
- `--n_samples 3+`: 生成3+个样本：强制 `Execute` + 强制 `Clarify` + Free (auto-select)

### 4. Free采样策略 (`--sampling_strategy free`)

随机选择起始动作或auto-select：

```bash
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 5 --out logs/traj_free.jsonl \
  --llm_model gpt-4o-mini \
  --sampling_strategy free \
  --n_samples 4
```

**Free策略的逻辑**：
- 50%概率随机选择 `Execute` 或 `Clarify`
- 50%概率auto-select（基于persona和state）

### 5. 组合使用（推荐）

为了获得最全面的数据，推荐组合使用：

```bash
# 每个state × 所有personas × 多样本（heuristic策略）
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 10 --out logs/traj_comprehensive.jsonl \
  --llm_model gpt-4o-mini \
  --max_turns 5 \
  --all_personas \
  --sampling_strategy heuristic \
  --n_samples 3
```

**结果**:
- 10 states × 3 personas × 3 samples = 90个对话
- 每个对话有：1个Execute样本 + 1个Clarify样本 + 1个Free样本
- 确保既有"直接做"的例子，也有"提问"的例子

## 输出统计

生成完成后会显示：
- 总轨迹数（所有turns）
- 总对话数
- 平均每个对话的轮次
- 完成任务数
- **首轮动作分布**（Execute/Clarify的数量）

## 示例输出

```
============================================================
Generation Summary
============================================================
Wrote 120 trajectory turns to logs/traj_comprehensive.jsonl
  - Total conversations: 90
  - Average turns per conversation: 1.33
  - Completed conversations: 85/90
  - First action distribution:
      Execute: 60
      Clarify: 30
============================================================
```

## 注意事项

1. **成本控制**: `--all_personas` 和 `--n_samples` 会大幅增加API调用次数
   - 建议先用小规模测试（`--n_states 2 --n_samples 2`）
   
2. **数据多样性**: 
   - Heuristic策略确保每个(state, persona)都有Execute和Clarify的例子
   - Free策略增加随机性，可能发现意外的成功/失败模式

3. **与旧代码兼容**: 
   - 不指定新参数时，行为与之前完全一致（单个persona，auto-select）
   - `--persona_idx` 在 `--all_personas` 时会被忽略

## 使用建议

### 快速测试（小规模）
```bash
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 2 --out logs/test.jsonl \
  --llm_model gpt-4o-mini \
  --all_personas --n_samples 2
```

### 生产数据生成（大规模）
```bash
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 100 --out logs/traj_production.jsonl \
  --llm_model gpt-4o-mini \
  --all_personas \
  --sampling_strategy heuristic \
  --n_samples 3 \
  --max_turns 5
```

这样生成的数据集将包含：
- ✅ 所有Persona维度下的轨迹
- ✅ "直接做而成功/失败"的例子
- ✅ "提问而成功/断连"的例子
- ✅ 模型可以学会不同Persona下的边界

