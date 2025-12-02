# ⚠️ 关键细节：确保训练正确性

## 两个必须遵守的细节

### ✅ 1. Prompt中不能包含action_prompt或模板内容

**问题**：如果prompt中包含"你现在要做LOW/MID/HIGH行为"这样的指令，模型会被锁死，无法根据state自由判断。

**解决方案**：
- ✅ 创建了统一的 `policy/render_state.py` 模块
- ✅ `render_state()` 函数只包含纯state信息：
  ```
  [Domain] coding
  [Turn] 1
  [User Clarity] 0.3
  [Prev Reject] 1
  [Task] write a python script...
  ```
- ✅ **不包含**任何action指令、模板或行为描述
- ✅ 让模型自由判断：要问几个问题、怎么问、是否该问

**验证方法**：
```python
from policy.render_state import render_state
state = {...}
prompt = render_state(state)
# 检查：prompt中不应该包含 "LOW", "MID", "HIGH", "action", "template" 等词
assert "LOW" not in prompt
assert "MID" not in prompt
assert "HIGH" not in prompt
assert "action" not in prompt.lower()
```

### ✅ 2. 训练和推理必须使用完全相同的render_state

**问题**：如果训练和推理的prompt格式不一致，模型性能会大幅下降。

**解决方案**：
- ✅ 创建了统一的 `policy/render_state.py` 模块
- ✅ `train_dpo.py` 和 `evaluate_dpo_model.py` 都导入并使用相同的函数：
  ```python
  from policy.render_state import render_state
  ```
- ✅ 确保训练prompt = 推理prompt（完全相同）

**验证方法**：
```python
# 在训练和评估脚本中，确保使用相同的函数
from policy.render_state import render_state

# 训练时
train_prompt = render_state(state)

# 评估时
eval_prompt = render_state(state)

# 必须完全相同
assert train_prompt == eval_prompt
```

## 当前实现状态

### ✅ 已实现

1. **统一的render_state函数** (`policy/render_state.py`)
   - 只包含纯state信息
   - 格式清晰：`[Domain]`, `[Turn]`, `[User Clarity]`, `[Prev Reject]`, `[Task]`
   - 不包含任何action指令

2. **训练脚本** (`policy/train_dpo.py`)
   - ✅ 导入并使用统一的 `render_state()`
   - ✅ 使用完整回复进行训练（不是action token）

3. **评估脚本** (`eval/evaluate_dpo_model.py`)
   - ✅ 导入并使用统一的 `render_state()`
   - ✅ 与训练使用完全相同的prompt格式

### 📝 使用示例

```python
from policy.render_state import render_state

state = {
    "domain": "coding",
    "dialogue_turn": 1,
    "query_clarity": 0.3,
    "prev_reject": 1,
    "query": "write a python script that scrapes data"
}

prompt = render_state(state)
# 输出：
# [Domain] coding
# [Turn] 1
# [User Clarity] 0.30
# [Prev Reject] 1
# [Task]
# write a python script that scrapes data
```

## 检查清单

在训练前，请确认：

- [ ] `render_state()` 函数不包含任何 "LOW", "MID", "HIGH" 关键词
- [ ] `render_state()` 函数不包含任何 action_prompt 或模板内容
- [ ] `train_dpo.py` 和 `evaluate_dpo_model.py` 使用相同的 `render_state()` 函数
- [ ] 训练和评估的prompt格式完全一致

## 如果违反这些规则会怎样？

1. **如果prompt包含action指令**：
   - 模型会被强制按照指令行为，无法根据state自由判断
   - state信息（clarity, prev_reject等）会被忽略
   - 模型无法学习真正的策略

2. **如果训练和推理prompt不一致**：
   - 模型在训练时学习一种格式，推理时看到另一种格式
   - 性能会大幅下降
   - 可能出现分布外问题

