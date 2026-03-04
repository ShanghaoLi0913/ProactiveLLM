# State Reconstruction 实现总结

## 改进目标

将Clarify得到的用户回答转换为结构化的Spec，用于Execute阶段，避免直接拼接对话历史导致的问题。

## 实现内容

### 1. 新增模块：`eval/reconstruct_state.py`

核心函数：
- `extract_user_answers_from_query()`: 从query中提取所有用户回答
- `extract_original_query()`: 提取原始query（去掉对话历史）
- `parse_user_answer_to_structured_spec()`: 将用户回答解析为结构化字段
- `merge_structured_specs()`: 合并多轮Clarify的spec
- `reconstruct_state_for_execute()`: 主函数，重构state用于Execute

### 2. 修改：`eval/evaluate_multi_turn_persona.py`

在 `generate_assistant_message()` 中：
- Execute时调用 `reconstruct_state_for_execute()` 重构state
- 使用结构化的 `[Clarified Requirements]` 格式
- 去掉Assistant的问题，只保留用户回答的结构化信息

### 3. 修改：`prompts/coding_execute.txt`

添加说明：
```
Note: If [Clarified Requirements] section is provided, incorporate those requirements into your implementation.
```

## 改进前后对比

### 改进前（有问题）

```
[Task]
Convert elements in 'T1' to integers...

[Assistant]: To better understand your requirements, I have a few clarifying questions:
1. What is the data type of `T1`?
2. Do you want to convert the elements in `T1` to integers?
...

[User]: 需要处理空字符串的情况，使用递归实现，时间复杂度O(n)。 Output specification: should output with: Counter: ...
```

**问题**：
1. 包含Assistant的问题（信息重复/污染）
2. 模型可能继续进入对话模式
3. 结构不清晰，难以提取关键信息

### 改进后（结构化）

```
[Task]
Convert elements in 'T1' to integers...

[Clarified Requirements]
Edge cases:
- empty input

Output format:
- should output with: Counter: A Counter object representing the count of each number appearing in the list of generated random integers

Constraints:
- time complexity O(n)
- use recursion

[Instruction]
Write the implementation.
Do not ask further questions.
```

**改进**：
1. ✅ 只保留原始query（去掉对话历史）
2. ✅ 提取用户回答并结构化（Edge cases, Output format, Constraints）
3. ✅ 去掉Assistant的问题（避免信息重复）
4. ✅ 清晰的格式，便于模型理解

## 结构化字段映射

从用户回答中提取的信息映射到以下字段：

| 字段 | 提取规则 | 示例 |
|------|---------|------|
| **Edge cases** | 关键词匹配：empty, null, single, large, negative, zero, duplicate | "需要处理空字符串的情况" → `empty input` |
| **Output format** | 匹配 "should output with:", "output format:", "return ..." | "Output specification: should output with: Counter: ..." → `should output with: Counter: ...` |
| **Constraints** | 匹配 "time complexity", "space complexity", "O(...)", "recursion", "iteration" | "时间复杂度O(n)" → `time complexity O(n)` |
| **Input constraints** | 匹配 "default", "range", "type" | "default is 100" → `default value: 100` |

## 使用方式

### 在评估中自动使用

评估脚本会自动调用 `reconstruct_state_for_execute()`，无需手动调用。

### 测试

运行测试脚本查看效果：
```bash
python3 test_state_reconstruction.py
```

## 潜在优化

1. **改进正则表达式**：当前"O(n)"的提取可能只得到"n"，需要改进
2. **支持更多字段**：可以扩展支持更多类型的约束（如validation_rules）
3. **与disclosure_rule对齐**：可以验证提取的信息是否与disclosure_rule中的masked_fields对应

## 下一步

1. ✅ 实现基本功能
2. ⏳ 优化正则表达式，提高提取准确性
3. ⏳ 在评估中验证效果（是否提高了task_score）
4. ⏳ 对比改进前后的代码生成质量
