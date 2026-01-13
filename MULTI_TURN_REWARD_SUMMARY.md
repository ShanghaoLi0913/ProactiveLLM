# Multi-Turn模式下的Reward计算总结

## 📊 当前实现方式

### 1. Trajectory生成（Multi-Turn模式）

- **每个state生成一个完整的多轮对话**
- **每个turn是一个独立的trajectory**（写入JSONL文件）
- 每个trajectory格式：
  ```json
  {
    "state": {...},              // 当前turn的state（dialogue_turn会递增）
    "action": "Clarify" | "Execute",
    "assistant_msg": "...",
    "user_reaction": {...},
    "turn": 1, 2, 3, ...         // turn编号
    "task_completed": true/false // 最后一个turn可能标记为completed
  }
  ```

### 2. Reward计算流程

#### 2.1 加载和分组
- 所有trajectories被加载到内存
- **按照 `(state_id, decision_point)` 分组**
  - `state_id`: 初始state的id（所有turns的state_id相同）
  - `decision_point`: 默认为0（所有turns的decision_point相同）
- **结果**: 同一个conversation的所有turns会被分到同一个group

#### 2.2 每个Trajectory的Reward计算

对于每个trajectory（每个turn），使用**single-interaction模式**计算reward：

```python
# 公式: R = w_task × R_task - w_interrupt × C_interrupt

# 1. R_task (Task Success Score)
task_score = compute_task_score(state, domain, assistant_output=assistant_msg)
# - 如果有代码且通过测试: 1.0
# - 如果有代码但无测试: 0.5
# - 如果只有问题（无代码）: 0.0

# 2. C_interrupt (Interrupt Cost)
interrupt_cost = compute_interrupt_cost_v2(meta, n_questions, assistant_msg)
# 公式: C = δ × b × r + λ × b - γ × b × a
# 其中:
#   b = 1 if n_questions > 0 else 0  # 是否提问
#   a = answered_clarification (0/1)  # 用户是否回答
#   r = reject_signal (0/1)           # 用户是否拒绝
#   δ = 0.7  # 拒绝惩罚
#   λ = 0.0  # 提问成本（当前设为0）
#   γ = 0.3  # 有效澄清奖励

# 3. Total Reward
total_reward = w_task × task_score - w_interrupt × interrupt_cost
# w_task = 1.0
# w_interrupt = 0.15
```

#### 2.3 C_interrupt的具体计算

**如果当前turn有提问 (b=1)**:
- **有效澄清** (a=1, r=0): `C = n_questions × (0 - 0.3) = -0.3 × n_questions` （负值=奖励）
- **被拒绝** (r=1): `C = n_questions × (0.7 + 0) = 0.7 × n_questions` （惩罚）
- **未回答** (a=0, r=0): `C = n_questions × 0 = 0` （无成本）

**如果当前turn无提问 (b=0)**:
- `C = 0`

#### 2.4 Preference Pairs生成

在每个group内（即同一个conversation的所有turns）：
1. 按照 `total_reward` 降序排序
2. 选择reward最高的trajectory作为 `chosen`
3. 选择reward最低的trajectory作为 `rejected`
4. 生成preference pair: `(chosen, rejected)`

## 🔍 关键特点

### ✅ 优点
1. **每个turn独立计算reward**：可以比较同一个conversation中不同turns的效果
2. **支持turn-level learning**：可以学习"在什么时候应该Clarify，什么时候应该Execute"
3. **简单直接**：每个turn都是single-interaction，计算逻辑清晰

### ⚠️ 注意事项
1. **R_task是sparse的**：只有最后一个包含代码的turn才有R_task > 0
2. **Grouping逻辑**：同一个conversation的所有turns会被group在一起，然后选best/worst turn
3. **不是conversation-level reward**：当前实现是turn-level的，不是整个conversation的累积reward

## 📝 示例

假设一个3-turn conversation：

**Turn 1** (Clarify, 提问2个):
- R_task = 0.0 (无代码)
- C_interrupt = -0.3 × 2 = -0.6 (有效澄清，负值=奖励)
- R = 1.0 × 0.0 - 0.15 × (-0.6) = **0.09**

**Turn 2** (Clarify, 提问1个):
- R_task = 0.0 (无代码)
- C_interrupt = -0.3 × 1 = -0.3 (有效澄清)
- R = 1.0 × 0.0 - 0.15 × (-0.3) = **0.045**

**Turn 3** (Execute, 提供代码):
- R_task = 1.0 (代码通过测试)
- C_interrupt = 0.0 (无提问)
- R = 1.0 × 1.0 - 0.15 × 0.0 = **1.0**

**Preference Pair**:
- Chosen: Turn 3 (R=1.0)
- Rejected: Turn 2 (R=0.045)

## 🤔 潜在问题

1. **是否应该累积整个conversation的reward？**
   - 当前：每个turn单独计算
   - 可能更合理：整个conversation的累积reward（R_task只在最后计算，C_interrupt累积所有turns）

2. **Grouping是否合理？**
   - 当前：同一个conversation的所有turns被group在一起
   - 可能更合理：每个turn单独group（如果state_id不同）或使用conversation_id

3. **R_task应该只在最后一个turn计算吗？**
   - 当前：每个turn都计算，但只有包含代码的turn才有R_task > 0
   - 这符合sparse reward的设计


