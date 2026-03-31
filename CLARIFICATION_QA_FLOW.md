# 澄清问答流程详解

**文档生成时间**: 2026-03-17  
**相关文件**: 
- `scripts/generate_trajectories.py` - 轨迹生成
- `simulator/simulate.py` - 用户模拟器
- `simulator/disclosure.py` - Disclosure规则处理

---

## 📊 完整问答流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Assistant生成澄清问题                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │ 检查disclosure_rule是否存在      │
        └─────────────────────────────────┘
                  ↓                    ↓
           有disclosure_rule      无disclosure_rule
                  ↓                    ↓
        ┌──────────────────┐   ┌──────────────┐
        │ 增强prompt        │   │ 使用基础prompt│
        │ - 添加masked信息  │   │              │
        │ - 引导问关键问题  │   │              │
        └──────────────────┘   └──────────────┘
                  ↓                    ↓
        ┌─────────────────────────────────┐
        │ LLM生成澄清问题                  │
        │ (基于增强/基础prompt)            │
        └─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: User Simulator生成回答                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │ 决定是否回答 (基于patience)      │
        │ P(answer) = patience × (0.85)^Turn│
        └─────────────────────────────────┘
                  ↓                    ↓
              回答 (P)             拒绝 (1-P)
                  ↓                    ↓
        ┌──────────────────┐   ┌──────────────┐
        │ 检查是否给模糊回复 │   │ 生成拒绝消息  │
        │ (第一次clarify时)  │   │              │
        └──────────────────┘   └──────────────┘
                  ↓                    ↓
        模糊回复?              正常生成
         (65%概率)                  ↓
                  ↓         ┌──────────────────┐
        "I want a general │ 生成base_answer    │
         solution..."      │ (LLM生成，基于     │
                          │  expertise)        │
                          └──────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │ 应用disclosure_rule?    │
                    └─────────────────────────┘
                              ↓        ↓
                        有disclosure  无disclosure
                              ↓        ↓
                    ┌──────────────────┐  ┌──────────┐
                    │ 从disclosure_rule│  │ 使用     │
                    │ 提取相关信息     │  │ base_answer│
                    │ - 匹配问题关键词  │  │          │
                    │ - 渐进式披露     │  │          │
                    └──────────────────┘  └──────────┘
                              ↓
                    ┌──────────────────┐
                    │ 整合到回答       │
                    │ base_answer +    │
                    │ disclosure_info  │
                    └──────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 更新State                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │ 追加对话历史到query              │
        │ query = query +                  │
        │   "\n\n[Assistant]: question"   │
        │   "\n[User]: answer"             │
        └─────────────────────────────────┘
```

---

## 1️⃣ Assistant生成澄清问题

### 流程

**位置**: `scripts/generate_trajectories.py` (第581-610行)

**步骤**:

1. **检查disclosure_rule**
   ```python
   if action == "Clarify":
       disclosure_rule = current_state.get("disclosure_rule")
   ```

2. **如果有disclosure_rule，增强prompt**
   ```python
   if disclosure_rule:
       masked_fields = disclosure_rule.get("masked_fields", {})
       
       # 提取关键信息
       guidance_parts = []
       if masked_fields.get("input_constraints"):
           guidance_parts.append("- Input constraints or default values")
       if masked_fields.get("output_format"):
           guidance_parts.append("- Output format or return type")
       if masked_fields.get("edge_cases"):
           guidance_parts.append("- Edge cases to handle")
       if masked_fields.get("validation_rules"):
           guidance_parts.append("- Validation rules or error handling")
       
       # 增强prompt
       action_prompt = f"""{action_prompt}
       
IMPORTANT: The task description may be missing some information. 
Consider asking about:
{guidance_text}

Generate 1-2 specific questions that would help clarify these missing aspects."""
   ```

3. **LLM生成问题**
   ```python
   assistant_msg = llm_output(current_state, action_prompt, llm_model, ...)
   ```

### 特点

- ✅ **有disclosure_rule**: prompt被增强，引导模型问关键信息
- ✅ **无disclosure_rule**: 使用基础prompt，模型自由生成
- ✅ **灵活性**: 问题仍由模型生成，更自然

---

## 2️⃣ User Simulator生成回答

### 流程

**位置**: `simulator/simulate.py` - `react()` 函数

### Step 2.1: 决定是否回答

**公式**: `P(answer) = patience × (0.85)^Turn`

**影响因素**:
- **Base patience**: 根据persona
  - Novice-Learner: 0.9 (high patience)
  - Experienced-Engineer: 0.7 (mid patience)
  - Busy-Developer: 0.3 (low patience)
- **Patience decay**: 每轮衰减15% (`0.85^Turn`)
- **特殊调整**:
  - Busy-Developer: 强制降低到0.1
  - Experienced-Engineer遇到明显问题: 降低50%
  - 多轮问题后: 额外降低15%

### Step 2.2: 如果回答，生成回答内容

#### 情况A: 模糊回复（第一次clarify，65%概率）

**条件**:
- `dialogue_turn == 0` (第一次clarify)
- `disclosure_rule`存在
- `edge_cases`数量 > 1
- 随机概率 < 0.65

**回复**:
```python
user_reply = "I want a general solution that works. Just do it the standard way."
answer_clarity = 0.2  # 低清晰度
```

**目的**: 迫使Assistant再次Clarify，实现多轮对话

#### 情况B: 正常生成回答

**步骤**:

1. **生成base_answer** (LLM生成)
   ```python
   base_answer = generate_specific_answer_llm(
       assistant_msg,  # Assistant的问题
       user_msg,        # 原始任务
       domain,
       llm_model,
       expertise        # 影响回答清晰度
   )
   ```
   
   **Expertise影响**:
   - `low`: 可能提供模糊或不完整的回答
   - `mid`: 提供清晰、具体的回答
   - `high`: 提供非常清晰、详细、专业的回答

2. **应用disclosure_rule** (如果有)
   ```python
   if disclosure_rule:
       user_reply = generate_answer_with_disclosure(
           assistant_question,
           user_query,
           disclosure_rule,
           expertise,
           base_answer,
           dialogue_turn
       )
   else:
       user_reply = base_answer
   ```

### Step 2.3: Disclosure信息提取

**位置**: `simulator/disclosure.py` - `get_disclosure_info()`

**流程**:

1. **匹配问题关键词**
   ```python
   question_lower = assistant_question.lower()
   
   # 输入约束相关
   if "input" in question_lower or "empty" in question_lower:
       # 提取input_constraints信息
   
   # 输出格式相关
   if "output" in question_lower or "return" in question_lower:
       # 提取output_format信息
   
   # 边界情况相关
   if "edge case" in question_lower:
       # 提取edge_cases信息
   ```

2. **渐进式披露机制**
   
   根据expertise和dialogue_turn决定每次披露多少信息：
   
   | Persona | Turn 0-1 | Turn 2 | Turn 3+ |
   |---------|----------|--------|---------|
   | Novice-Learner | 1个 | 1个 | 1个 |
   | Busy-Developer | 1个 | 1个 | 2个 |
   | Experienced-Engineer | 1个 | 2个 | 2个 |

3. **根据expertise调整信息表达**
   ```python
   if expertise == "high":
       disclosure_text = f"Edge case: {edge_case}"  # 详细
   elif expertise == "low":
       disclosure_text = "可能需要处理一些特殊情况。"  # 模糊
   else:
       disclosure_text = f"Should handle: {edge_case}"  # 中等
   ```

4. **整合到回答**
   ```python
   if disclosure_info:
       return f"{base_answer} {disclosure_info}"
   else:
       return base_answer
   ```

### Step 2.4: 如果拒绝

**概率**: `P(reject) = 1 - P(answer)`

**回复示例**:
- Experienced-Engineer (明显问题): "This information is already in the prompt. Please proceed with the implementation."
- 其他: "Just proceed with the implementation."

---

## 3️⃣ 更新State

**位置**: `scripts/generate_trajectories.py` - `update_state_for_next_turn()`

**更新内容**:

1. **更新dialogue_turn**: `dialogue_turn++`

2. **更新query** (追加对话历史)
   ```python
   if answered_clarification > 0:
       new_state["query"] = f"{current_state['query']}\n\n[Assistant]: {assistant_msg}\n[User]: {user_reply}"
   else:
       new_state["query"] = f"{current_state['query']}\n\n[Assistant]: {assistant_msg}"
   ```

3. **更新task_uncertainty** (如果用户回答了)
   ```python
   if answered_clarification > 0:
       # U_{t+1} = U_t (1 - 0.5 · answer_clarity)
       new_task_uncertainty = current_uncertainty * (1 - 0.5 * answer_clarity)
   ```

---

## 📋 完整示例

### 示例: 多轮澄清对话

**初始状态**:
```json
{
  "query": "Calculates the average... (masked query)",
  "dialogue_turn": 0,
  "disclosure_rule": {
    "masked_fields": {
      "edge_cases": ["empty input", "negative numbers", "single element"],
      "output_format": ["should output with: float"]
    }
  }
}
```

**Turn 1: Clarify**

1. **Assistant生成问题** (基于增强prompt):
   ```
   "To clarify your request, could you specify:
   1. What edge cases should I handle?
   2. What format should the output be?"
   ```

2. **User Simulator生成回答**:
   - 决定回答: `P(answer) = 0.9 × 0.85^0 = 0.9` (Novice-Learner)
   - 第一次clarify，65%概率给模糊回复:
     ```
     "I want a general solution that works. Just do it the standard way."
     ```
   - 或者正常回答 + disclosure:
     ```
     base_answer = "Please handle empty lists and negative numbers."
     disclosure_info = "Should handle: empty input"  # 只披露1个信息点
     user_reply = "Please handle empty lists and negative numbers. Should handle: empty input"
     ```

3. **更新State**:
   ```json
   {
     "query": "Calculates the average...\n\n[Assistant]: To clarify...\n[User]: Please handle empty lists...",
     "dialogue_turn": 1,
     "task_uncertainty": 0.45  # 降低（因为用户回答了）
   }
   ```

**Turn 2: Clarify (如果需要)**

1. **Assistant生成问题** (基于新的query，包含对话历史):
   ```
   "Got it! Just to confirm, should I also handle single-element lists?"
   ```

2. **User Simulator生成回答**:
   - 决定回答: `P(answer) = 0.9 × 0.85^1 = 0.765`
   - 正常回答 + disclosure:
     ```
     base_answer = "Yes, please handle single-element lists."
     disclosure_info = "Should handle: single element"  # 再披露1个信息点
     user_reply = "Yes, please handle single-element lists. Should handle: single element"
     ```

**Turn 3: Execute**

- **Execute时的query**:
  ```
  "Calculates the average... (masked query)
  
  [Assistant]: To clarify your request...
  [User]: Please handle empty lists... Should handle: empty input
  
  [Assistant]: Got it! Just to confirm...
  [User]: Yes, please handle single-element lists. Should handle: single element"
  ```

- **生成3个版本的代码**:
  1. `masked_with_clarification`: 基于上述完整query
  2. `full_query`: 基于original_instruct_prompt
  3. `masked_only`: 基于初始masked query

---

## 🔍 关键机制

### 1. 渐进式披露 (Progressive Disclosure)

**目的**: 让对话持续多轮，而不是一次性披露所有信息

**机制**:
- 每次只披露1-2个信息点
- 根据expertise和dialogue_turn调整披露数量
- 确保需要多轮才能获得完整信息

### 2. 模糊回复策略

**目的**: 增加多轮对话的概率

**机制**:
- 第一次clarify时，65%概率给模糊回复
- 迫使Assistant再次Clarify
- 实现更真实的对话模式

### 3. Patience衰减

**目的**: 模拟用户耐心随轮次递减

**机制**:
- `P(answer) = patience × (0.85)^Turn`
- 不同persona有不同的base patience
- 多轮后用户更可能拒绝

### 4. 信息匹配

**目的**: 确保disclosure信息与问题相关

**机制**:
- 根据问题关键词匹配disclosure_rule中的信息
- 如果匹配到，提取相关信息
- 如果没有匹配，按顺序提取

---

## 📝 总结

### 用户回答生成流程

1. **决定是否回答** (基于patience)
2. **如果回答**:
   - 可能给模糊回复 (第一次clarify，65%概率)
   - 否则生成base_answer (LLM生成，基于expertise)
   - 应用disclosure_rule (如果有):
     - 匹配问题关键词
     - 提取相关信息 (渐进式披露)
     - 整合到回答中
3. **如果拒绝**: 生成拒绝消息

### 关键特点

- ✅ **基于disclosure_rule**: 用户回答会包含被mask的信息
- ✅ **渐进式披露**: 每次只披露部分信息，需要多轮对话
- ✅ **Persona差异**: 不同persona有不同的披露速度和清晰度
- ✅ **自然对话**: base_answer由LLM生成，disclosure信息自然整合

---

*文档生成时间: 2026-03-17*
