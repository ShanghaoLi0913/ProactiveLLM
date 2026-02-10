# V15 最终总结：意外失败

**日期**: 2026-02-09  
**状态**: ❌ 训练成功，但评估失败

---

## 🎯 V15目标

修复V14的致命问题：**添加Llama-3.1-Instruct的chat template**，让模型知道prompt/response边界。

## ✅ 完成的工作

### 1. 代码修改
- ✅ `policy/train_dpo.py` - 添加了`apply_chat_template`
- ✅ `eval/evaluate_v13_persona.py` - 评估时使用chat template
- ✅ 清理了磁盘空间（释放33G）
- ✅ V15训练成功（4分10秒）

### 2. 训练指标（看起来不错）
```
Epoch 1: Loss=0.600, Accuracy=37.5%, Margins=-0.010
Epoch 3: Loss=0.424, Accuracy=77.5%, Margins=+0.520
```
- Loss ↓ 29%
- Accuracy ↑ 107%
- Margins: 负转正

---

## ❌ 评估结果（灾难性）

### 整体性能
| 指标 | V14 | V15 | 变化 |
|------|-----|-----|------|
| **Action Accuracy** | 0% | 0% | 无改善 |
| **Task Success Rate** | 0% | 10.5% | +10.5% |

### 按Persona
| Persona | Action Acc | Task SR |
|---------|-----------|---------|
| Busy-Developer | 0% | 0% |
| Experienced-Engineer | 0% | 13.3% |
| Novice-Learner | 0% | 11.1% |

---

## 🔍 问题分析

### 1. 模型生成的是什么？

#### 预期生成（训练数据格式）：
```
Clarify

或

Execute
```python
def task_func():
    ...
```
```

#### 实际生成（V15模型）：
```
```python
import csv
import collections
import ope
```
（50个tokens后截断）

### 2. 根本问题

**模型学会了生成代码，但没有学会决策**

可能原因：

#### 问题1：DPO训练数据格式错误
当前格式：
```python
chosen = ex["chosen_assistant_msg"]      # 完整response（代码/问题）
rejected = ex["rejected_assistant_msg"]  # 完整response
```

**这训练模型去生成完整response，而不是决策action！**

应该是什么？
```python
chosen = "Execute"  # 或 "Clarify"
rejected = "Clarify"  # 或 "Execute"
```

#### 问题2：数据理解错误
我们一直认为模型应该生成完整response（代码或问题），但这不是一个**action decision**任务，这是一个**code generation**任务！

真正的proactivity calibration应该是：
1. **模型预测**: "Execute" 或 "Clarify"
2. **后续处理**: 根据预测调用code generator或question generator

#### 问题3：评估脚本的假设
`extract_action_from_response()`期望从response中提取"Clarify"或"Execute"，但模型根本没有生成这些词！

---

## 🤔 架构问题

### 当前架构（错误）
```
State → DPO Model → Full Response (code/question)
                      ↓
                  提取action（失败！）
```

### 应该的架构（正确）
```
State → DPO Model → Action ("Clarify"/"Execute")
                      ↓
          ┌───────────┴───────────┐
          ↓                        ↓
   Code Generator          Question Generator
```

---

## 💡 解决方案

### 方案1：重新设计training data format ⭐️ 推荐
**目标**: 训练一个纯action decision model

```python
# preference data format
{
    "state": {...},
    "persona": {...},
    "chosen": "Execute",        # ← 只有action标签！
    "rejected": "Clarify",
    "chosen_reward": 0.85,
    "rejected_reward": 0.12
}
```

**优点**:
- 模型只学习action decision
- 评估清晰（action accuracy）
- 符合proactivity calibration的定义

**缺点**:
- 需要重新生成所有preference data
- 需要修改`train_dpo.py`和`compute_rewards.py`

### 方案2：Two-Stage Architecture
**阶段1**: Action Decision Model（本项目）  
**阶段2**: Code/Question Generator（已有base model）

### 方案3：修改评估脚本（临时方案）
让评估脚本从生成的代码中**推断**action：
- 如果生成了代码 → "Execute"
- 如果生成了问题 → "Clarify"

**问题**: 这不是真正的action decision，而是事后推断

---

## 📊 数据分析

### 训练数据现状
```python
# 当前格式 (v14/v15)
{
    "chosen_assistant_msg": "```python\ndef task_func():\n    ...\n```",  # ← 完整代码
    "rejected_assistant_msg": "Could you clarify..."  # ← 完整问题
}
```

**问题**: 
1. DPO学习生成**完整response**，而不是**决策**
2. `chosen`和`rejected`长度差异巨大（代码 vs 简短问题）
3. 模型需要同时学会：决策 + 代码生成 + 问题生成（太复杂）

### 正确的数据格式（v16建议）
```python
{
    "state": {...},
    "chosen_action": "Execute",  # ← 简单的action标签
    "rejected_action": "Clarify",
    "chosen_reward": 0.85,
    "rejected_reward": 0.12
}
```

**优点**:
1. DPO只学习**action decision**
2. `chosen`和`rejected`长度一致
3. 任务清晰、可评估

---

## 🎯 V16 计划

### 核心修改

#### 1. 修改`reward/compute_rewards.py`
```python
# 当前 (错误)
prefs.append({
    "chosen_assistant_msg": hi["assistant_msg"],  # 完整response
    "rejected_assistant_msg": lo["assistant_msg"]
})

# V16 (正确)
prefs.append({
    "chosen_action": hi["action"],  # "Execute" or "Clarify"
    "rejected_action": lo["action"],
    "chosen_reward": hi["total_reward"],
    "rejected_reward": lo["total_reward"]
})
```

#### 2. 修改`policy/train_dpo.py`
```python
# 当前 (错误)
dataset["chosen"].append(ex["chosen_assistant_msg"])  # 完整response
dataset["rejected"].append(ex["rejected_assistant_msg"])

# V16 (正确)
dataset["chosen"].append(ex["chosen_action"])  # "Execute"
dataset["rejected"].append(ex["rejected_action"])  # "Clarify"
```

#### 3. 修改`eval/evaluate_v13_persona.py`
```python
# 当前 (错误 - 从代码中推断action)
predicted_action = extract_action_from_response(response)

# V16 (正确 - 直接获取action)
predicted_action = response.strip()  # "Execute" or "Clarify"
```

---

## 📉 为什么训练指标好，但评估差？

### 训练时（DPO loss）
DPO loss衡量的是：**模型是否偏好chosen response而不是rejected response**

V15模型学会了：
- ✅ 生成类似代码的token序列
- ✅ chosen有更高的概率
- ❌ 但不是生成正确的action decision

### 评估时（Action Accuracy）
我们期望：
- ✅ 模型生成"Execute"或"Clarify"
- ❌ 实际：模型生成代码片段

**不匹配！**

---

## 🔄 V14 → V15 → V16 演进

| 版本 | 主要改进 | 结果 | 根本问题 |
|------|---------|------|---------|
| **V14** | Trajectory-level + Persona-aware | Action 0%, Task 0% | 缺少chat template |
| **V15** | 添加chat template | Action 0%, Task 10.5% | 训练目标错误（生成response而不是决策） |
| **V16** | Action-only DPO | ? | TBD |

---

## 🎓 经验教训

### 1. 训练指标 ≠ 任务目标
即使loss下降、accuracy上升，也不代表模型在做正确的事情。

### 2. 数据格式决定模型行为
如果训练数据是`chosen_assistant_msg`（完整response），模型就学习生成response，而不是decision。

### 3. 评估的重要性
如果只看训练loss，我们永远不会发现这个问题。

### 4. 架构设计 > 技术细节
Chat template、QLoRA、hyperparameters都是次要的。最重要的是：**模型到底在学什么？**

---

## ✅ V15总结

| 方面 | 状态 | 备注 |
|------|------|------|
| **代码修改** | ✅ | 添加了chat template |
| **磁盘清理** | ✅ | 释放33G |
| **训练** | ✅ | Loss下降，accuracy上升 |
| **评估** | ❌ | Action 0%, Task 10.5% |
| **根本问题** | 🔍 | 训练目标错误（response而不是action） |
| **下一步** | 📋 | V16: Action-only DPO |

---

## 🚀 接下来做什么？

### 选项1: 修复并训练V16 ⭐️ 推荐
**时间**: ~1-2小时（代码修改 + 数据重生成 + 训练）

**步骤**:
1. 修改`compute_rewards.py`（输出action而不是full response）
2. 修改`train_dpo.py`（训练action decision）
3. 修改`evaluate_v13_persona.py`（评估action accuracy）
4. 重新生成preference data
5. 训练V16
6. 评估

### 选项2: 分析当前模型
检查V15到底学到了什么（生成更长的response看看）

### 选项3: 暂停并总结
文档化所有问题，重新设计整个pipeline

---

## 📖 相关文档

- `V14_EXECUTIVE_SUMMARY.md` - V14问题和V15计划
- `V14_vs_V15_COMPARISON.md` - V14和V15的代码对比
- `V15_CHANGES_AND_FIX.md` - V15的修改细节
- `DISK_CLEANUP_PLAN.md` - 磁盘清理计划

---

**结论**: V15添加了chat template，但训练了一个**response generator**而不是**action decision model**。需要V16重新设计数据格式，训练一个纯粹的action decision model。
