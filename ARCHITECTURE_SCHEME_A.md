# 方案A：分离架构设计

## 🏗️ 架构设计

### 核心思想
将**action预测**和**code generation**完全分离，避免DPO训练污染code generation能力。

### 架构流程

```
┌─────────────┐
│    State    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   Policy Model      │  ← DPO训练，只预测action
│  (Action Predictor) │
└──────┬──────────────┘
       │
       ▼
   Action Token
  (LOW/MID/HIGH)
       │
       ▼
┌─────────────────────┐
│  Template Selector  │  ← 根据action选择template
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Code Generation     │  ← 独立模型，不受DPO影响
│ (Base/Code Model)   │    可以是base model或专门的code model
└──────┬──────────────┘
       │
       ▼
   Clean Code
```

## 📋 实现步骤

### 阶段1: 修改训练（只训练action预测）

**修改点**：
1. `train_dpo.py`: 恢复使用action token训练（不是完整回复）
2. 训练目标：`state → action_token (LOW/MID/HIGH)`
3. 不训练code generation部分

### 阶段2: 修改推理（分离架构）

**修改点**：
1. `policy/infer.py`: 实现分离的inference流程
2. 步骤：
   - State → Policy Model → Action
   - Action → Template Selector → Template
   - State + Template → Code Generation Model → Code

### 阶段3: 可选优化

- Code Generation可以使用专门的code model（如CodeLlama）
- 或者使用base model但不受DPO影响

## 🔧 代码修改计划

### 1. 恢复action token训练

- `train_dpo.py`: 使用action token而不是完整回复
- 添加LOW/MID/HIGH特殊token
- DPO训练只优化action预测

### 2. 实现分离推理

- `policy/infer.py`: 
  - `select_action()`: 使用policy模型预测action
  - `generate_code()`: 使用独立模型生成代码
  - `execute_action()`: 整合两个步骤

### 3. 更新评估

- `eval/evaluate_dpo_model.py`: 
  - 使用分离架构进行评估
  - 先预测action，再生成代码
  - 评估action准确率和代码质量

## ✅ 优势

1. **Code generation不受污染**：DPO训练不影响代码生成能力
2. **Action选择独立优化**：可以专注于学习何时问问题
3. **灵活可扩展**：可以独立替换code generation模型
4. **工业标准**：符合实际应用的最佳实践

## 📝 实现细节

### 训练数据格式

```python
{
    "prompt": render_state(state),  # 纯state信息
    "chosen": "LOW",  # action token
    "rejected": "HIGH"  # action token
}
```

### 推理流程

```python
# 1. 预测action
action = policy_model.predict(state)  # LOW/MID/HIGH

# 2. 选择template
template = get_template(action, domain)  # coding_low.txt等

# 3. 生成代码
code = code_model.generate(state['query'], template)
```

