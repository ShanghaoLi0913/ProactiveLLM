# 信息披露机制总结

## 一、每个Query的Masked信息数量

### 1.1 Masked信息结构

每个query的masked信息存储在`disclosure_rule`中，包含两部分：

**1. `masked_fields`**：被mask的字段列表
- `input_constraints`: 输入约束（如"default length is 100"）
- `output_format`: 输出格式（如"should output with Counter"）
- `edge_cases`: Edge cases列表（如["empty input", "negative numbers"]）
- `validation_rules`: 验证规则

**2. `disclosure_info`**：用于reveal的结构化信息
- `input_constraints`: 
  - `edge_cases`: Edge cases列表（通常4-6个）
  - `hints`: 提示信息
- `output_format`:
  - `specification`: 输出格式规格
- `validation_rules`:
  - `rules`: 验证规则列表

### 1.2 典型数量（基于30个states的统计）

**每个query的masked信息点数量**：
- **Edge cases**: 平均1.97条（范围: 1-6条）
- **Output format**: 平均0.97条（范围: 0-1条，约97%的任务有）
- **Input constraints**: 平均0.27条（范围: 0-4条）
- **Validation rules**: 平均0.23条（范围: 0-1条）

**总计**：每个query平均有**3.43条**masked信息点（范围: 2-8条）

**典型分布**：
- 最少：2条（只有edge_cases）
- 最多：8条（edge_cases + output_format + input_constraints + validation_rules）
- 最常见：3-5条（edge_cases + output_format）

## 二、信息披露机制

### 2.1 核心原理

**设计原则**：
- 只有当assistant ASK clarification时，用户（模拟器）才会补充被mask的信息
- 根据assistant的问题类别，智能匹配相关的masked信息
- 根据用户的expertise，控制每次reveal的信息点数量

### 2.2 渐进式披露机制（简化版）

**配置（固定披露步长）**：

| Persona | Expertise | 每次披露信息点数量 | 理由 |
|---------|-----------|------------------|------|
| **Novice-Learner** | low | **1个**（固定） | expertise低，表达能力有限 |
| **Busy-Developer** | mid | **1个**（固定） | 时间压力大，每次只reveal少量信息 |
| **Experienced-Engineer** | high | **3个**（固定） | expertise高，能一次性reveal更多信息 |

**代码实现**（`simulator/disclosure.py:55-62`）：
```python
if expertise == "low":  # Novice-Learner
    max_points = 1
elif expertise == "mid":  # Busy-Developer
    max_points = 1
else:  # expertise == "high" (Experienced-Engineer)
    max_points = 3
```

### 2.3 累计信息点对比

**Turn 0-2累计信息点**：
- **Novice-Learner**: 1 + 1 + 1 = **3个**
- **Busy-Developer**: 1 + 1 + 1 = **3个**
- **Experienced-Engineer**: 3 + 3 + 3 = **9个**

**设计优势**：
- ✅ 更简单：不需要根据turn动态调整
- ✅ 更符合persona特征：expertise高 → reveal更多
- ✅ 更易理解：固定步长，逻辑清晰

### 2.4 信息匹配机制

**根据Assistant问题关键词匹配**：

| 问题类别 | 关键词 | 提取的信息 |
|---------|--------|-----------|
| **Input相关** | `["input", "empty", "negative", "zero", "null", "constraint", "range", "value", "default"]` | `edge_cases`, `hints`, `input_constraints` |
| **Output相关** | `["output", "return", "format", "type", "result", "should return"]` | `output_format["specification"]` |
| **Validation相关** | `["error", "exception", "validate", "check", "raise"]` | `validation_rules["rules"]` |
| **Fallback** | 无匹配 | 按顺序从`masked_fields`提取 |

**匹配示例**：
```
Assistant问："What edge cases should I handle?"
→ 匹配到input_keywords中的"edge"
→ 提取disclosure_info["input_constraints"]["edge_cases"]
→ 根据expertise格式化输出
```

### 2.5 信息格式化（根据Expertise）

**不同expertise的格式化方式**：

| Expertise | 格式化方式 | 示例 |
|-----------|-----------|------|
| **low** (Novice) | 模糊表达 | "可能需要处理一些特殊情况。" |
| **mid** (Busy) | 简洁明确 | "Should handle: empty input" |
| **high** (Experienced) | 详细专业 | "Edge case: empty input. Edge case: negative numbers. Edge case: zero value" |

### 2.6 整合方式

#### 2.6.1 正常回答整合

**流程**：
```python
# 1. 生成基础回答（LLM或dummy）
base_answer = generate_specific_answer_llm(...)

# 2. 获取disclosure信息
disclosure_info = get_disclosure_info(
    assistant_question, disclosure_rule, expertise, dialogue_turn
)

# 3. 整合
user_reply = f"{base_answer} {disclosure_info}"
```

**示例**：
```
基础回答: "Yes, I want to handle edge cases."
Disclosure信息: "Should handle: empty input"
最终回答: "Yes, I want to handle edge cases. Should handle: empty input"
```

#### 2.6.2 模糊回答整合

**流程**：
```python
# 1. 基础模糊回复
vague_base = "I want a general solution that works. Just do it the standard way."

# 2. 提取部分disclosure信息（第一个信息点）
partial_disclosure = get_disclosure_info(...)

# 3. 整合
user_reply = f"{vague_base} Also, {partial_disclosure.lower()}."
```

**示例**：
```
基础模糊回复: "I want a general solution that works. Just do it the standard way."
Disclosure信息: "Should handle: empty input"
最终回答: "I want a general solution that works. Just do it the standard way. Also, should handle: empty input."
```

**关键设计**：
- ✅ 模糊回答也包含disclosure信息（不短路信息流）
- ✅ 只提取第一个信息点（简化处理）
- ✅ 如果无法提取，使用fallback（至少提到一个edge case）

## 三、完整流程示例

### 3.1 场景：Multi-turn对话

**初始状态**：
- `disclosure_rule`包含：`edge_cases = ["empty input", "negative numbers", "zero value", "single element", "large inputs"]`（5个）
- Persona: Experienced-Engineer (expertise=high, max_points=3)

**Turn 0**：
```
Assistant: "What edge cases should I handle?"

匹配过程：
1. 关键词匹配：找到"edge" → 匹配input_keywords
2. 提取edge_cases
3. max_points = 3 (expertise=high)
4. 格式化：expertise=high → "Edge case: empty input. Edge case: negative numbers. Edge case: zero value"

用户回答（正常）：
"Yes, please handle edge cases. Edge case: empty input. Edge case: negative numbers. Edge case: zero value"
```

**Turn 1**：
```
Assistant: "Any other constraints?"

匹配过程：
1. 关键词匹配：找到"constraint" → 匹配input_keywords
2. 提取edge_cases（继续）
3. max_points = 3
4. 格式化：expertise=high → "Edge case: single element. Edge case: large inputs. [其他信息]"

用户回答（正常）：
"Also consider single elements and large inputs. Edge case: single element. Edge case: large inputs. [其他信息]"
```

**累计reveal**：
- Turn 0: 3个信息点
- Turn 1: 3个信息点
- **总计**: 6个信息点（已reveal所有5个edge_cases + 其他信息）

### 3.2 场景：模糊回答整合

**Turn 0**（模糊回答概率触发，Busy-Developer）：
```
Assistant: "What edge cases should I handle?"

匹配过程：
1. 关键词匹配：找到"edge" → 匹配input_keywords
2. 提取edge_cases
3. max_points = 1 (expertise=mid)
4. 格式化：expertise=mid → "Should handle: empty input"

用户回答（模糊）：
"I want a general solution that works. Just do it the standard way. Also, should handle: empty input."
```

## 四、关键设计优势

### 4.1 渐进式披露
- ✅ 不同expertise用户，信息reveal速度不同（更真实）
- ✅ Experienced-Engineer能更快reveal所有信息（3倍速度）
- ✅ 固定步长，逻辑简单清晰

### 4.2 信息匹配
- ✅ 智能匹配，只reveal相关问题相关的信息（提高效率）
- ✅ 支持多种问题类别（input/output/validation）
- ✅ Fallback机制，确保总是有信息reveal

### 4.3 整合方式
- ✅ 正常回答和模糊回答都包含disclosure信息（不短路信息流）
- ✅ 信息量根据expertise调整（更真实）
- ✅ 格式自然，与基础回答融合（提高reconstruction成功率）

## 五、统计口径

### 5.1 Masked信息数量（基于30个states的统计）
- **每个query平均**: 3.43条masked信息点（范围: 2-8条）
- **Edge cases**: 平均1.97条（范围: 1-6条）
- **Output format**: 平均0.97条（范围: 0-1条，约97%的任务有）
- **Input constraints**: 平均0.27条（范围: 0-4条）
- **Validation rules**: 平均0.23条（范围: 0-1条）

### 5.2 披露速度对比
- **Novice-Learner**: 每次1个 → 需要3-4轮才能reveal所有信息（平均3.43条）
- **Busy-Developer**: 每次1个 → 需要3-4轮才能reveal所有信息（平均3.43条）
- **Experienced-Engineer**: 每次3个 → 需要1-2轮就能reveal所有信息（平均3.43条）

**实际场景**：
- 如果query有8条masked信息：
  - Novice/Busy: 需要8轮
  - Experienced: 需要3轮（3+3+2）
- 如果query有3条masked信息：
  - Novice/Busy: 需要3轮
  - Experienced: 需要1轮（3）
