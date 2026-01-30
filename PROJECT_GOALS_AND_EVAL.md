# 项目目标和实验设计

## 🎯 项目核心目标

**ProactiveLLM - Context-Aware Proactivity Calibration**

**核心问题**：
- LLM助手何时应该**主动提问（Clarify）** vs **直接执行（Execute）**？
- 问太多 → 用户不耐烦（interrupt cost高）
- 问太少 → 任务失败率高（task success低）

**目标**：
训练一个Policy模型，能根据上下文自动决策：
- **输入**：State（task_uncertainty, dialogue_turn, prev_reject等）
- **输出**：Action（Clarify 或 Execute）
- **优化**：平衡Task Success和Interrupt Cost（Pareto最优）

## 📊 实验对比设计

### 1. 训练后的Policy模型（DPO-trained Policy）

**方法**：
- 使用DPO训练的Policy模型
- 根据State自动选择Action（Clarify/Execute）
- 训练数据：从multi-turn trajectories生成的preference pairs

**评估指标**：
- **Task Success Rate**：任务成功率（代码通过测试的比例）
- **Interrupt Cost**：中断成本（提问次数、用户拒绝等）
- **Total Reward**：`R = R_task - C_interrupt`（平衡两项）

### 2. 可能的Baseline（需要确认）

根据项目代码结构，可能的对比baseline包括：

**Baseline A：Always Execute（总是执行）**
- 每个turn都选择Execute
- 不提问，直接生成代码
- **预期**：Interrupt Cost低，但Task Success可能也低（因为任务不清晰时直接执行容易失败）

**Baseline B：Always Clarify（总是提问）**
- 每个turn都选择Clarify
- 一直问问题，直到max_turns
- **预期**：Task Success可能高（因为问清楚了），但Interrupt Cost高（用户不耐烦）

**Baseline C：基于规则的策略（Rule-based）**
- 例如：`if task_uncertainty < threshold: Clarify else: Execute`
- 简单的启发式规则
- **预期**：可能比随机好，但不如学习到的Policy

**Baseline D：Base Model（未训练的模型）**
- 使用基础模型（如Llama-3.1-8B-Instruct）但不经过DPO训练
- **预期**：性能较差，因为模型没有学习到决策策略

### 3. 评估流程

**步骤1：训练Policy模型**
```bash
python policy/train_dpo.py \
  --data data/dpo/prefs_*.jsonl \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir outputs/policy_model
```

**步骤2：评估Policy模型**
```bash
python eval/evaluate_dpo_model.py \
  --model_dir outputs/policy_model \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --prefs_path data/dpo/prefs_*.jsonl \
  --output outputs/eval_results.json
```

**步骤3：评估Baseline（需要实现）**
- 在相同测试集上运行baseline策略
- 计算相同的指标（Task Success, Interrupt Cost, Total Reward）

**步骤4：对比分析**
- 绘制Pareto曲线（Task Success vs Interrupt Cost）
- 对比不同方法的性能
- 分析Policy模型是否在两者之间找到更好的平衡

## 📈 评估指标详解

### Task Success (R_task)
- **范围**：0.0 - 1.0
- **计算**：基于代码是否通过测试
- **目标**：越高越好

### Interrupt Cost (C_interrupt)
- **范围**：0.0 - 正无穷（理论上）
- **计算**：`C = δ × b × r + λ × b - γ × b × a`
  - `b`: 是否提问
  - `r`: 用户是否拒绝
  - `a`: 用户是否回答
- **目标**：越低越好

### Total Reward
- **公式**：`R = w_task × R_task - w_interrupt × C_interrupt`
- **目标**：越高越好（平衡两项）

## 🔍 关键实验问题

1. **Policy模型是否能学习到合适的决策时机？**
   - 在高不确定性时选择Clarify
   - 在低不确定性时选择Execute
   - 在用户拒绝后不再提问

2. **Policy模型是否优于简单baseline？**
   - 相比Always Execute：是否能在保持低Interrupt Cost的同时提高Task Success？
   - 相比Always Clarify：是否能在保持高Task Success的同时降低Interrupt Cost？

3. **Multi-turn学习是否有效？**
   - 模型是否能学习到对话历史的影响？
   - 是否能在不同dialogue_turn做出合适的决策？

## ⚠️ 当前状态

**已实现**：
- ✅ DPO训练流程
- ✅ Policy模型评估流程
- ✅ Reward计算（Task Success + Interrupt Cost）

**待实现/待确认**：
- ⚠️ Baseline实现（Always Execute, Always Clarify等）
- ⚠️ Baseline评估流程
- ⚠️ 对比实验脚本
- ⚠️ Pareto曲线绘制（plot_pareto.py目前是dummy）

## 💡 建议

1. **实现Baseline评估**：
   - Always Execute：固定选择Execute
   - Always Clarify：固定选择Clarify
   - Rule-based：基于task_uncertainty阈值

2. **统一评估流程**：
   - 在相同测试集上评估所有方法
   - 使用相同的指标和计算方法

3. **可视化对比**：
   - Pareto曲线（Task Success vs Interrupt Cost）
   - 不同方法的性能对比表


