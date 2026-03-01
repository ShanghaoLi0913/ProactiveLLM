# 训练数据质量分析与解决方案

## 问题核心

**当前问题：**
- 训练数据生成时：GPT用masked query生成代码 → 代码质量低 → task_score低 → 训练数据质量差
- 评估时：模型用masked query生成代码 → task_score也不高
- 这导致模型学不到高质量的代码生成能力

**核心矛盾：**
- **目标**：让模型学会评估task_uncertainty和用户对主动性的偏好
- **现实**：masked query导致代码质量低 → 训练数据质量差 → 模型学不好

---

## 方案对比

### 方案1：提升样本代码质量（推荐）

**设计：**
- **Action选择**：用masked query（保持task_uncertainty的真实性）
- **代码生成（Execute）**：用original query（让GPT生成高质量代码）

**优点：**
- ✅ 训练数据质量高（GPT用完整信息生成高质量代码）
- ✅ 模型能学到高质量的代码生成能力（知识蒸馏）
- ✅ 训练时task_score高 → preference pairs质量好
- ✅ 保持了不确定性判断的真实性（action选择仍用masked query）

**缺点：**
- ⚠️  训练时GPT看到完整query，但模型评估时看到masked query
- ⚠️  可能造成训练-测试不一致（但这是知识蒸馏的常见做法）

**关键点：**
- Action选择用masked query → 模型学会判断task_uncertainty
- 代码生成用original query → GPT生成高质量代码 → 模型学到高质量代码
- 这样模型既学会了判断不确定性，又学到了高质量代码

---

### 方案2：Execute时给完整query（不推荐）

**设计：**
- 模型Execute时，自动补充完整query

**优点：**
- ✅ 代码生成质量高
- ✅ 评估时task_score高

**缺点：**
- ❌ **破坏学习目标**：模型学不会在信息不足时判断是否Clarify
- ❌ 模型会倾向于直接Execute（因为总能拿到完整信息）
- ❌ 无法测试模型对task_uncertainty的判断能力
- ❌ 违背了项目的核心目标

---

## 推荐方案：方案1的改进版

### 1. 训练数据生成

**设计：**
- **Action选择**：用masked query（保持task_uncertainty的真实性）
- **代码生成（Execute）**：用original query（让GPT生成高质量代码）

**实现：**
```python
# 在 generate_trajectories.py 中
def llm_output(state: Dict, action_prompt: str, model: str, ...):
    # 如果是Execute action，使用original query
    if action == "Execute" and state.get("original_instruct_prompt"):
        user = f"[Task]\n{state['original_instruct_prompt']}"
    else:
        user = f"[Task]\n{state['query']}"  # masked query for Clarify
    
    return chat_complete(system, user, model=model, ...)
```

**效果：**
- 训练数据既有高质量代码，又保持了不确定性判断的真实性
- GPT用完整信息生成代码 → task_score高 → preference pairs质量好

---

### 2. 评估时

**设计：**
- **Action选择**：用masked query（测试模型对不确定性的判断）
- **代码生成**：用训练好的模型（测试知识蒸馏效果）

**可选：**
- 同时评估masked和original query版本，对比分析
- 如果模型学到了GPT的能力，即使masked query也能生成较好代码

---

### 3. 关键设计原则

**训练阶段：**
- GPT用original query生成代码 → 高质量训练数据
- Action选择用masked query → 保持不确定性判断的真实性

**评估阶段：**
- 模型用masked query生成代码 → 测试真实场景
- 如果模型学到了GPT的能力，即使masked query也能生成较好代码

**知识蒸馏流程：**
```
训练数据生成（Teacher）:
  GPT (original query) → 高质量代码 → Preference Pairs (chosen)
         ↓
DPO训练（Student Learning）:
  Llama → 学习GPT的代码生成能力 + 学习判断task_uncertainty
         ↓
评估（Student Testing）:
  训练好的Llama (masked query) → 测试是否学到了GPT的能力
```

---

## 实现建议

### 修改点1：`scripts/generate_trajectories.py`

在`llm_output()`函数中，Execute action使用original query：

```python
def llm_output(
    state: Dict,
    action_prompt: str,
    model: str,
    action: str = None,  # 新增参数
    ...
) -> str:
    from llm.provider import chat_complete
    system = action_prompt
    
    # 如果是Execute action且有original query，使用original query
    if action == "Execute" and state.get("original_instruct_prompt"):
        user = f"[Task]\n{state['original_instruct_prompt']}"
    else:
        # Clarify action或没有original query时，使用masked query
        user = f"[Task]\n{state['query']}"
    
    return chat_complete(system, user, model=model, ...)
```

### 修改点2：调用时传递action参数

在`generate_multi_turn_conversation()`中：

```python
# Generate assistant message
if local_generator is not None:
    # 对于Execute，使用original query
    if action == "Execute" and current_state.get("original_instruct_prompt"):
        task_query = current_state['original_instruct_prompt']
    else:
        task_query = current_state['query']
    assistant_msg = local_generator.chat_complete(action_prompt, f"[Task]\n{task_query}")
elif llm_model:
    assistant_msg = llm_output(current_state, action_prompt, llm_model, 
                              action=action,  # 传递action参数
                              temperature=temperature, top_p=top_p)
```

---

## 预期效果

### 训练数据质量提升

**之前：**
- GPT用masked query生成代码
- task_score: ~0.3-0.5（部分测试通过）
- 训练数据质量：中等

**之后：**
- GPT用original query生成代码
- task_score: ~0.8-1.0（大部分或全部测试通过）
- 训练数据质量：高

### 模型能力提升

**之前：**
- 模型学到的是"用masked query生成代码"的能力
- 代码质量受限于masked query的信息不足

**之后：**
- 模型学到的是"GPT用完整信息生成高质量代码"的能力
- 即使评估时用masked query，模型也能利用学到的能力生成较好代码

### 评估结果预期

**Masked Query评估：**
- 如果模型学到了GPT的能力，即使masked query也能生成较好代码
- Task Success Rate应该比之前高（因为训练数据质量提升了）

**Original Query评估：**
- 应该接近或达到GPT的水平（知识蒸馏成功）

---

## 总结

**推荐方案：方案1（改进版）**

1. **训练数据生成**：Execute action使用original query生成代码
2. **保持学习目标**：Action选择仍用masked query，保持不确定性判断的真实性
3. **知识蒸馏**：让模型学到GPT的代码生成能力，同时学会判断task_uncertainty

**关键原则：**
- Action选择：用masked query（保持不确定性判断的真实性）
- 代码生成：用original query（让GPT生成高质量代码）
- 评估时：用masked query（测试真实场景和知识蒸馏效果）

这样既解决了训练数据质量低的问题，又保持了项目的核心学习目标。
