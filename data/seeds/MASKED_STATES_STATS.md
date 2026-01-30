# BigCodeBench Masked States 数据统计

本文档描述了 `bigcodebench_masked_states.jsonl` 文件中 masked fields 的统计信息。

## 数据概览

- **总任务数**: 470

## Masked Fields 结构

每条数据的 `disclosure_rule.masked_fields` 都包含以下 4 个字段：
- `input_constraints`: 被 mask 的输入约束信息（列表）
- `output_format`: 被 mask 的输出格式信息（列表）
- `edge_cases`: 从测试用例中提取的边界情况（列表）
- `validation_rules`: 被 mask 的验证规则（列表）

**注意**: 这些字段总是存在，但可能是空列表 `[]`。

## 各字段填充情况统计

| 字段 | 有内容的数量 | 占比 |
|------|------------|------|
| `edge_cases` | 470 / 470 | **100.0%** |
| `output_format` | 456 / 470 | **97.0%** |
| `validation_rules` | 130 / 470 | **27.7%** |
| `input_constraints` | 60 / 470 | **12.8%** |

### 说明

- **`edge_cases` (100%)**: 所有任务都有边界情况，这些信息从测试用例中自动提取
- **`output_format` (97%)**: 绝大多数任务都有输出格式说明被 mask
- **`validation_rules` (27.7%)**: 约四分之一的任务包含验证规则（如异常处理要求）
- **`input_constraints` (12.8%)**: 少数任务有输入约束被 mask（如默认值、输入范围等）

## 字段组合统计

最常见的字段组合（缩写：IC=input_constraints, OF=output_format, EC=edge_cases, VR=validation_rules）：

| 组合 | 数量 | 占比 | 说明 |
|------|------|------|------|
| **OF+EC** | 311 / 470 | **66.2%** | 最常见：只有输出格式和边界情况 |
| **OF+EC+VR** | 99 / 470 | **21.1%** | 包含验证规则 |
| **IC+OF+EC+VR** | 25 / 470 | **5.3%** | 所有字段都有内容 |
| **IC+OF+EC** | 21 / 470 | **4.5%** | 包含输入约束但无验证规则 |
| **IC+EC** | 8 / 470 | **1.7%** | 只有输入约束和边界情况 |
| **IC+EC+VR** | 6 / 470 | **1.3%** | 有输入约束和验证规则但无输出格式 |

## 字段数量分布

| 恰好有 N 个字段有内容 | 数量 | 占比 |
|---------------------|------|------|
| 2 个字段 | 319 / 470 | **67.9%** |
| 3 个字段 | 126 / 470 | **26.8%** |
| 4 个字段 | 25 / 470 | **5.3%** |

**说明**: 所有任务至少都有 2 个字段有内容（`edge_cases` 和 `output_format`），因为 `edge_cases` 是 100% 填充的。

## Mask 策略

Mask 操作由 `scripts/mask_task_details.py` 执行，使用 `mask_level="moderate"`（默认）：

1. **输入约束 (input_constraints)**: 
   - 移除包含 "default", "handle empty/negative/zero" 等描述的文本
   - 只有 12.8% 的任务匹配到这些模式

2. **输出格式 (output_format)**:
   - 移除 "should output with: ..." 后的详细说明
   - 97% 的任务都有输出格式说明被 mask

3. **边界情况 (edge_cases)**:
   - 从测试用例中自动提取（通过关键词匹配：empty, negative, zero, single, duplicate 等）
   - 100% 的任务都能提取到边界情况

4. **验证规则 (validation_rules)**:
   - 提取 "raise ... if/when ..." 等异常处理要求
   - 27.7% 的任务包含验证规则

## 使用说明

`disclosure_rule` 中的信息用于：
- **`masked_fields`**: 记录被 mask 的具体文本内容，用于追溯和分析
- **`disclosure_info`**: 结构化信息，供 `simulator/simulate.py` 在回答澄清问题时使用

注意：`disclosure_rule` **对 Policy 模型不可见**，只在轨迹生成阶段的 simulator 中使用。





