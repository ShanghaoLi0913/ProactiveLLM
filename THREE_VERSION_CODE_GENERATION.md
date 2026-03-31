# 三版本代码生成功能说明

**实现时间**: 2026-03-17  
**修改文件**: `scripts/generate_trajectories.py`

---

## 📋 功能概述

在轨迹生成时，当action为Execute时，现在会生成**3个版本的代码**用于对比和评估：

1. **Version 1: masked query + 用户澄清得到的信息** (当前版本)
   - 使用: `current_state['query']` (已包含对话历史)
   - 字段名: `code_versions["masked_with_clarification"]`

2. **Version 2: full query生成的代码**
   - 使用: `current_state.get('original_instruct_prompt', '')`
   - 字段名: `code_versions["full_query"]`
   - 如果`original_instruct_prompt`不存在，则为`None`

3. **Version 3: masked query本身生成的代码**
   - 使用: `initial_state['query']` (初始masked query，无澄清信息)
   - 字段名: `code_versions["masked_only"]`
   - 如果没有发生clarify，则与Version 1相同

---

## 🔧 实现细节

### 代码位置

修改位置: `scripts/generate_trajectories.py` (第647-687行)

### 实现逻辑

```python
if action == "Execute":
    # Generate 3 versions of code for comparison
    code_versions = {}
    
    # Version 1: masked query + 用户澄清得到的信息 (current version)
    code_versions["masked_with_clarification"] = assistant_msg
    
    # Version 2: full query生成的代码
    original_query = current_state.get("original_instruct_prompt", "")
    if original_query:
        # 使用full query生成代码
        code_versions["full_query"] = generated_code
    else:
        code_versions["full_query"] = None
    
    # Version 3: masked query本身生成的代码
    initial_masked_query = initial_state.get("query", "")
    if initial_masked_query != current_state.get("query", ""):
        # 使用初始masked query生成代码
        code_versions["masked_only"] = generated_code
    else:
        # 如果没有clarify，与Version 1相同
        code_versions["masked_only"] = assistant_msg
    
    # Add code_versions to trajectory
    traj["code_versions"] = code_versions
```

### 数据格式

在trajectory JSON中，Execute action的trajectory会包含`code_versions`字段：

```json
{
  "trajectory_id": "...",
  "state": {...},
  "action": "Execute",
  "assistant_msg": "...",  // Version 1的代码（保持向后兼容）
  "code_versions": {
    "masked_with_clarification": "...",  // Version 1: masked + clarification
    "full_query": "...",                  // Version 2: full query
    "masked_only": "..."                  // Version 3: masked only
  },
  "task_completed": true,
  ...
}
```

---

## 📊 使用场景

### 1. 对比分析

可以对比3个版本的代码质量：
- **masked_with_clarification**: 实际对话中使用的代码
- **full_query**: 如果有完整信息，代码质量如何
- **masked_only**: 如果只有masked信息，代码质量如何

### 2. 评估澄清的价值

通过对比`masked_with_clarification`和`masked_only`，可以评估：
- 澄清信息对代码质量的提升
- 不同澄清策略的效果

### 3. 评估信息完整性的影响

通过对比`full_query`和其他版本，可以评估：
- 完整信息vs部分信息的代码质量差异
- masked策略的有效性

---

## 🔍 数据示例

### 示例1: 有clarify的对话

```json
{
  "action": "Execute",
  "assistant_msg": "```python\ndef task_func(...):\n    # 基于masked query + clarification的代码\n```",
  "code_versions": {
    "masked_with_clarification": "```python\ndef task_func(...):\n    # 基于masked query + clarification的代码\n```",
    "full_query": "```python\ndef task_func(...):\n    # 基于full query的代码\n```",
    "masked_only": "```python\ndef task_func(...):\n    # 基于masked query only的代码\n```"
  }
}
```

### 示例2: 没有clarify的对话（第一轮Execute）

```json
{
  "action": "Execute",
  "assistant_msg": "```python\ndef task_func(...):\n    # 代码\n```",
  "code_versions": {
    "masked_with_clarification": "```python\ndef task_func(...):\n    # 代码\n```",
    "full_query": "```python\ndef task_func(...):\n    # 基于full query的代码\n```",
    "masked_only": "```python\ndef task_func(...):\n    # 代码\n```"  // 与masked_with_clarification相同
  }
}
```

---

## ⚙️ 配置说明

### 代码生成参数

所有3个版本使用相同的生成参数：
- `temperature`: 默认0.7
- `top_p`: 默认0.9
- `llm_model`: 与主对话使用相同的模型

### 性能考虑

- 每个Execute会额外生成2个版本的代码（如果`original_instruct_prompt`存在）
- 会增加API调用次数（如果使用OpenAI API）
- 建议在需要对比分析时启用此功能

---

## 📝 注意事项

1. **向后兼容**: `assistant_msg`字段仍然存在，保持向后兼容
2. **可选字段**: `code_versions["full_query"]`可能为`None`（如果`original_instruct_prompt`不存在）
3. **相同版本**: 如果没有发生clarify，`masked_only`和`masked_with_clarification`相同
4. **API成本**: 使用OpenAI API时，每个Execute会增加2次API调用

---

## 🔄 后续优化建议

1. **缓存机制**: 对于相同的query，可以缓存生成的代码
2. **批量生成**: 可以批量生成多个版本的代码以提高效率
3. **选择性生成**: 可以添加参数控制是否生成3个版本
4. **评估指标**: 可以自动计算3个版本的代码质量对比

---

*文档生成时间: 2026-03-17*
