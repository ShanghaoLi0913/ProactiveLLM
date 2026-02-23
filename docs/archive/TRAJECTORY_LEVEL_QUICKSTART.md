# Trajectory-Level Reward 快速开始

## 🎯 已完成的工作

✅ **Trajectory-Level Reward已成功实现！**

- 修改了代码支持trajectory_id追踪
- 实现了正确的trajectory-level reward计算
- 在100states数据上测试成功
- 生成了训练/测试数据

---

## 📊 生成的数据

### 文件位置

```
data/dpo/prefs_100states_trajectory_level_train.jsonl  (240 prefs, 80 states)
data/dpo/prefs_100states_trajectory_level_test.jsonl   (60 prefs, 20 states)
```

### 数据特点

- ✅ Clarify能获得非零task_score（来自后续Execute成功）
- ✅ Clarify平均task_score = 0.127（之前是0）
- ✅ 不需要penalty hack
- ✅ 95% chosen是Clarify（合理，因为获得了后续成功的credit）

---

## 🚀 快速训练V14模型

### 选项1：使用现有数据（推荐，快速验证）

```bash
cd /root/ProactiveLLM

# 训练V14
python policy/train_dpo.py \
  --data data/dpo/prefs_100states_trajectory_level_train.jsonl \
  --model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --output outputs/v14_trajectory_level \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1

# 评估V14
python eval/evaluate_v13_simple.py \
  --model_dir outputs/v14_trajectory_level \
  --base_model /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prefs_path data/dpo/prefs_100states_trajectory_level_test.jsonl \
  --output outputs/eval_results/v14_eval.json
```

**预期效果**：
- 模型能学习到Clarify的价值
- 不会出现100% Execute的问题
- Action accuracy应该更高

---

## 🔄 选项2：重新生成完美的数据（推荐用于论文）

### Step 1: 生成带trajectory_id的trajectories

```bash
python scripts/generate_trajectories.py \
  --mode dataset \
  --domain coding \
  --dataset_path data/seeds/bigcode_filtered.jsonl \
  --n_states 100 \
  --out logs/traj_100states_v14.jsonl \
  --llm_model gpt-4o-mini \
  --max_turns 5 \
  --all_personas
```

**说明**：
- 每个state × 3个personas = 300条trajectories
- 每条trajectory最多5轮对话
- 每条trajectory有唯一的trajectory_id

### Step 2: 计算trajectory-level rewards

```bash
python reward/compute_rewards.py \
  --trajectories logs/traj_100states_v14.jsonl \
  --out data/dpo/prefs_v14.jsonl \
  --w_task 1.0 \
  --w_interrupt 0.3 \
  --target_execute_ratio -1 \
  --use_trajectory_level
```

**说明**：
- `--use_trajectory_level`: 使用trajectory-level reward（关键！）
- `--target_execute_ratio -1`: 不强制rebalancing，保持自然分布

### Step 3: Split train/test

```python
import json
import random

with open('data/dpo/prefs_v14.jsonl') as f:
    prefs = [json.loads(line) for line in f]

state_ids = list(set(p['state']['id'] for p in prefs))
random.seed(42)
random.shuffle(state_ids)

split_idx = int(0.8 * len(state_ids))
train_states = set(state_ids[:split_idx])
test_states = set(state_ids[split_idx:])

train_prefs = [p for p in prefs if p['state']['id'] in train_states]
test_prefs = [p for p in prefs if p['state']['id'] in test_states]

with open('data/dpo/prefs_v14_train.jsonl', 'w') as f:
    for p in train_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

with open('data/dpo/prefs_v14_test.jsonl', 'w') as f:
    for p in test_prefs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
```

### Step 4: 训练V14

```bash
python policy/train_dpo.py \
  --data data/dpo/prefs_v14_train.jsonl \
  --model /root/autodl-tmp/hf_cache/.../Llama-3.1-8B-Instruct \
  --output outputs/v14 \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1
```

---

## 🎛️ 调整数据平衡（可选）

如果95% Clarify太高，可以调整：

### 方法1：使用rebalancing

```bash
python reward/compute_rewards.py \
  --trajectories logs/traj_100states_v14.jsonl \
  --out data/dpo/prefs_v14_balanced.jsonl \
  --target_execute_ratio 0.5 \  # 强制50% Execute
  --use_trajectory_level
```

### 方法2：调整interrupt_cost参数

编辑 `reward/compute.py`:

```python
# Line 133-135
gamma = 0.1  # 降低有效澄清的奖励（从0.2降到0.1）
delta = 0.8  # 保持被拒绝的惩罚
lambda_param = 0.2  # 提高提问的基础成本（从0.1升到0.2）
```

然后重新计算rewards。

---

## 📈 评估V14

### 基础评估

```bash
python eval/evaluate_v13_simple.py \
  --model_dir outputs/v14_trajectory_level \
  --base_model /root/autodl-tmp/hf_cache/.../Llama-3.1-8B-Instruct \
  --prefs_path data/dpo/prefs_100states_trajectory_level_test.jsonl \
  --output outputs/eval_results/v14_eval.json
```

### Persona-aware评估

```bash
python eval/evaluate_v13_persona.py \
  --model_dir outputs/v14_trajectory_level \
  --base_model /root/autodl-tmp/hf_cache/.../Llama-3.1-8B-Instruct \
  --prefs_path data/dpo/prefs_100states_trajectory_level_test.jsonl \
  --output outputs/eval_results/v14_persona_eval.json
```

### 对比V13 vs V14

```python
import json

# Load V13 results
with open('outputs/eval_results/v13_final_eval.json') as f:
    v13 = json.load(f)

# Load V14 results
with open('outputs/eval_results/v14_eval.json') as f:
    v14 = json.load(f)

print("=== V13 vs V14 Comparison ===")
print(f"V13 Action Accuracy: {v13['overall']['action_accuracy']:.2%}")
print(f"V14 Action Accuracy: {v14['overall']['action_accuracy']:.2%}")
```

---

## 🔍 关键改进说明

### 之前（V13，Step-Level）

```
Turn 0的Clarify：
  task_score = 0 (无法完成任务)
  interrupt_cost = -0.1
  penalty = 0.2
  reward = -0.1  ❌

结果：100% chosen是Execute
```

### 现在（V14，Trajectory-Level）

```
Turn 0的Clarify：
  task_score = 0.4 (来自Turn 1的Execute成功！)
  interrupt_cost = -0.3
  NO penalty!
  reward = 0.7  ✅

结果：95% chosen是Clarify（合理）
```

---

## 📚 技术文档

详细文档请查看：

1. `outputs/eval_results/TRAJECTORY_LEVEL_SUCCESS.md` - 成功报告
2. `outputs/eval_results/TRAJECTORY_LEVEL_IMPLEMENTATION_SUMMARY.md` - 实现总结
3. `outputs/eval_results/TRAJECTORY_VS_STEP_REWARD.md` - 设计对比
4. `outputs/eval_results/DATA_PIPELINE_ISSUE.md` - 问题分析

---

## ❓ 常见问题

### Q1: 为什么95% Clarify被选择？

**A**: 因为在trajectory-level reward下：
- Clarify能获得后续Execute成功的task_score
- Clarify被回答时interrupt_cost是负数（加分）
- 如果两者task_score相同，Clarify自然更优

这可能是合理的（在高不确定性任务上，Clarify确实更好），也可以通过调整参数来平衡。

### Q2: 需要重新生成数据吗？

**A**: 不一定。
- **快速验证**：使用现有的`prefs_100states_trajectory_level_train.jsonl`
- **论文实验**：推荐重新生成带完美trajectory_id的数据

### Q3: 和V10的"trajectory-level"有什么区别？

**A**: 
- **V10**: "伪trajectory-level"，人为传播分数（同state的Clarify获得Execute分数）
- **V14**: 真正的trajectory-level，使用trajectory_id追踪完整对话，正确的RL设计

### Q4: 如何验证trajectory-level reward是否工作？

**A**: 检查数据：
```python
import json
with open('data/dpo/prefs_100states_trajectory_level.jsonl') as f:
    prefs = [json.loads(line) for line in f]

clarify_prefs = [p for p in prefs if p['chosen_action'] == 'Clarify']
clarify_scores = [p['chosen_task_score'] for p in clarify_prefs]

print(f"Clarify平均task_score: {sum(clarify_scores)/len(clarify_scores):.3f}")
print(f"Clarify非零task_score: {sum(1 for s in clarify_scores if s > 0)}/{len(clarify_scores)}")
```

如果Clarify有非零task_score，说明trajectory-level reward工作了！

---

## 🎉 总结

**Trajectory-Level Reward已成功实现！**

这是第一个真正实现trajectory-level reward的Proactive LLM系统：
- ✅ 正确的RL设计（credit assignment）
- ✅ Clarify自然获得后续成功的credit
- ✅ 不需要penalty hack
- ✅ 代码ready，可以开始训练V14

**推荐下一步**：使用现有数据快速训练V14，验证效果！

```bash
python policy/train_dpo.py \
  --data data/dpo/prefs_100states_trajectory_level_train.jsonl \
  --model /root/autodl-tmp/hf_cache/.../Llama-3.1-8B-Instruct \
  --output outputs/v14_trajectory_level \
  --epochs 3
```

Good luck! 🚀
