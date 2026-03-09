# 问答机制完整总结

## 一、整体流程

```
Assistant Clarify → User Simulator → 回答/拒绝 → Disclosure整合 → Reconstruction → Execute
```

## 二、Assistant生成Clarify问题

### 2.1 问题生成方式

**文件**：`scripts/generate_trajectories.py`

**方式**：
- 使用LLM（如`gpt-4o-mini`）根据`action_prompt`生成澄清问题
- `action_prompt`来自模板：`prompts/coding_clarify.txt`
- 输入：`[Task]\n{state['query']}`（masked query）

**特点**：
- Assistant不知道被mask的信息
- 只能基于masked query提问
- 问题质量取决于LLM能力

## 三、User Simulator回答机制

### 3.1 回答/拒绝决策

**文件**：`simulator/simulate.py:react()`

**决策公式**：
```
P(answer) = patience × (0.85)^Turn
P(reject) = 1 - P(answer)
```

**Persona差异**：
- **Novice-Learner** (patience=0.98): 几乎总是回答
- **Busy-Developer** (patience=0.2): 经常拒绝
- **Experienced-Engineer** (patience=0.75): 中等概率回答

**特殊逻辑**：
1. **Busy-Developer**: `effective_patience = min(patience, 0.1)`（强制降低）
2. **Experienced-Engineer**: 如果问题太明显，`effective_patience *= 0.5`
3. **多轮后**: 如果`total_questions_asked >= 2`，`effective_patience *= 0.85`

### 3.2 回答类型决策

如果用户决定回答（`random.random() < effective_patience`），需要决定是**模糊回答**还是**正常回答**。

#### 3.2.1 模糊回答概率（Turn 0 First Clarify）

**配置**（`VAGUE_REPLY_PROB_MAP`）：
```python
{
    "Busy-Developer": 0.5,           # 50% - 时间压力，更容易敷衍
    "Experienced-Engineer": 0.1,    # 10% - 偶尔不耐烦
    "Novice-Learner": 0.25,         # 25% - 表达能力有限
}
```

**触发条件**：
- `dialogue_turn == 0`（第一次clarify）
- `disclosure_rule`存在
- `random.random() < vague_prob`
- `len(edge_cases) > 1`（信息需要分步披露）

#### 3.2.2 模糊回答内容（改进后）

**策略**：`vague + partial disclosure`（不短路信息流）

**生成流程**：
1. 基础模糊回复：`"I want a general solution that works. Just do it the standard way."`
2. 尝试从`disclosure_rule`提取一个信息点
3. 如果成功：`"{vague_base} Also, {partial_disclosure.lower()}."`
4. 如果失败：`"{vague_base} Also, please make sure it handles empty input."`

**示例**：
```
"I want a general solution that works. Just do it the standard way. Also, please make sure it handles empty input."
```

**好处**：
- 保留真实用户行为（确实会说"随便做个通用的"）
- 但不短路信息流，确保disclosure信息总是被整合
- 避免reviewer说"模拟器太理想化"

#### 3.2.3 正常回答生成

**方式1：LLM生成**（`generate_specific_answer_llm()`）
- 使用LLM根据assistant的问题生成回答
- Expertise影响回答清晰度：
  - `low`: "You are a beginner. Your answer may be somewhat vague or incomplete."
  - `mid`: "Provide a clear and specific answer."
  - `high`: "You are an expert. Provide a very clear, detailed, and professional answer."

**方式2：Dummy生成**（测试模式）
- `low`: "可能是这样的吧，我也不太确定。"
- `mid`: "需要处理空字符串。"
- `high`: "需要处理空字符串的情况，使用递归实现，时间复杂度O(n)。"

## 四、Disclosure信息整合

### 4.1 Disclosure机制

**文件**：`simulator/disclosure.py`

**原理**：
- 只有当assistant ASK clarification时，用户（模拟器）才会补充被mask的信息
- 从`disclosure_rule`中提取相关的被mask信息
- 根据assistant的问题类别匹配信息

### 4.2 渐进式披露机制

**披露步长**（基于expertise和dialogue_turn）：
- **Novice-Learner** (expertise=low): 每次固定1个信息点
- **Busy-Developer** (expertise=mid): 
  - Turn 0-2: 每次1个
  - Turn 3+: 每次2个
- **Experienced-Engineer** (expertise=high):
  - Turn 0-1: 每次1个
  - Turn 2: 每次2个
  - Turn 3+: 每次2个

**累计信息点**（Turn 0-2）：
- Novice: 3个
- Busy: 3个
- Experienced: 4个

### 4.3 信息匹配逻辑

**根据assistant问题关键词匹配**：
1. **Input相关** (`input`, `empty`, `negative`, `zero`, `null`等):
   - 提取`edge_cases`和`hints`
   - 根据expertise格式化：
     - `high`: `"Edge case: {ec}"`
     - `low`: `"可能需要处理一些特殊情况。"`
     - `mid`: `"Should handle: {ec}"`

2. **Output相关** (`output`, `return`, `format`, `type`等):
   - 提取`output_format`的`specification`
   - 根据expertise格式化

3. **Validation相关** (`error`, `exception`, `validate`等):
   - 提取`validation_rules`

4. **Fallback**:
   - 如果没有匹配，按顺序从`masked_fields`中提取

### 4.4 信息整合到用户回答

**正常回答**：
```python
base_answer = generate_specific_answer_llm(...)
if disclosure_rule:
    user_reply = generate_answer_with_disclosure(
        assistant_msg, user_msg, disclosure_rule, 
        persona.expertise, base_answer, dialogue_turn
    )
```

**模糊回答**：
```python
vague_base = "I want a general solution that works. Just do it the standard way."
partial_disclosure = get_disclosure_info(...)
user_reply = f"{vague_base} Also, {partial_disclosure.lower()}."
```

## 五、Reconstruction机制

### 5.1 提取用户回答

**文件**：`eval/reconstruct_state.py`

**函数**：`extract_user_answers_from_query()`

**逻辑**：
- 从`query`中提取所有`[User]:`之后的内容
- 移除`[Assistant]:`的问题部分
- 返回用户回答列表（按时间顺序）

### 5.2 Canonicalization层（改进）

**函数**：`canonicalize_text()`

**目的**：解决"关键词匹配对表达敏感"的问题

**映射规则**：
- `empty list / empty input / empty string → EMPTY_INPUT`
- `output should be Counter / return a Counter → OUTPUT_COUNTER`
- `O(n) / linear time / time complexity is linear → TIME_LINEAR`

**实现**：
```python
def canonicalize_text(text: str) -> str:
    # 检查每个canonical category
    for category, mapping in CANONICAL_MAP.items():
        for pattern in mapping["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                canonical_tokens.append(mapping["canonical"])
    return f"{text} {' '.join(canonical_tokens)}"
```

### 5.3 结构化Spec提取

**函数**：`parse_user_answer_to_structured_spec()`

**流程**：
1. 使用canonicalization层处理同义表达
2. 先检查canonical tokens（更可靠）
3. 也保留原有的pattern匹配（作为fallback）

**提取字段**：
- `edge_cases`: EMPTY_INPUT, NULL_VALUE, SINGLE_ELEMENT等
- `output_format`: OUTPUT_COUNTER, OUTPUT_DICT, OUTPUT_LIST等
- `constraints`: TIME_LINEAR, TIME_QUADRATIC等
- `input_constraints`: default value, range, type等

### 5.4 生成Clarified Requirements

**函数**：`reconstruct_state_for_execute()`

**输出格式**：
```
Edge cases:
- empty input
- single element

Output format:
- should output with Counter

Constraints:
- time complexity O(n)
```

## 六、Execute阶段使用Reconstruction

### 6.1 Prompt构建

**文件**：`scripts/generate_trajectories.py:llm_output()`

**逻辑**：
```python
if action == "Execute":
    reconstructed = reconstruct_state_for_execute(state)
    original_query = reconstructed["original_query"]
    clarified_requirements = reconstructed["clarified_requirements"]
    
    if clarified_requirements:
        user = f"[Task]\n{original_query}\n\n[Clarified Requirements]\n{clarified_requirements}\n\n[Instruction]\nWrite the implementation.\nDo not ask further questions."
    else:
        user = f"[Task]\n{original_query}"
```

### 6.2 三版本代码生成

**assistant_code**（实际执行）：
- 使用：`masked query + clarified requirements`
- 代表真实场景

**teacher_code**（理想参考）：
- 使用：`original_instruct_prompt`（full query）
- 代表理想场景

**masked_code**（基线对比）：
- 使用：`masked query`（无clarification）
- 代表无clarification基线

## 七、关键设计点

### 7.1 Persona差异体现

1. **回答概率**：Patience影响
2. **回答清晰度**：Expertise影响（`answer_clarity = EXPERTISE_MAP[expertise]`）
3. **模糊回复概率**：Persona-specific配置
4. **Disclosure步长**：Expertise影响

### 7.2 信息流完整性

1. **模糊回复也包含disclosure**：不短路信息流
2. **Canonicalization层**：提高reconstruction成功率
3. **三版本代码生成**：允许对比分析

### 7.3 真实性与研究价值平衡

1. **保留模糊回复**：真实用户确实会说"随便做个通用的"
2. **但包含部分信息**：不短路信息流
3. **可配置概率**：比硬编码更有研究价值

## 八、统计口径

### 8.1 模糊回复统计

**统计口径1**：在Turn0的first-clarify情况下，模糊回复比例
- 公式：`vague_in_turn0 / turn0_first_clarify`

**统计口径2**：在所有user replies中，模糊回复比例
- 公式：`vague_in_all / all_user_replies`

**按persona统计**：
- Busy-Developer: 50%概率
- Experienced-Engineer: 10%概率
- Novice-Learner: 25%概率

### 8.2 Reconstruction统计

**成功率**：`reconstruction_has_content / with_clarification_count`

**覆盖率**：
- `edge_cases_only`: X个
- `output_format_only`: Y个
- `constraints_only`: Z个
- `multiple`: W个

## 九、改进前后对比

### 改进前
- 模糊回复：65%概率（硬编码），完全跳过disclosure
- Reconstruction：纯关键词匹配，对表达敏感
- 成功率：40.9%

### 改进后
- 模糊回复：Persona-specific概率（0.1-0.5），包含部分disclosure
- Reconstruction：Canonicalization层 + 关键词匹配
- 预期成功率：60-80%

## 十、论文价值

1. **可配置的模糊回复概率**：
   - "We model unhelpful user feedback with a configurable probability, reflecting real-world ambiguity."
   - 不同persona有不同的模糊回复概率，更真实

2. **模糊回复+部分disclosure**：
   - 保留真实用户行为，但不短路信息流
   - 避免reviewer说"模拟器太理想化"

3. **Canonicalization层**：
   - 解决了"关键词匹配对表达敏感"的问题
   - 不需要引入额外模型，成本低
