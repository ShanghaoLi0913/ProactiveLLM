# BigCodeBench Masked States 数据统计

本文档描述了 `bigcodebench_masked_states.jsonl` 文件中 masked fields 的统计信息。

> **v29（2026-04-10）**：masking 全部重写。固定结构锚点替代 regex，零断句残留。新增 validation_rules / note_that masking，移除不准确的 edge_cases。总任务数从 470 → 1140（全量）。

---

## BigCodeBench instruct_prompt 格式

每个 task 的 `instruct_prompt` 固定结构如下：

```
[函数描述]                                               ← 100%，不 mask（核心功能说明）
Note that: [特殊说明]                                    ← 18.9%，已 mask ✅
The function should raise the exception for: [异常]      ← 26.0%，已 mask ✅
The function should output with:
    [返回值类型和描述]                                    ← 100%，已 mask ✅
You should write self-contained code starting with:
```[import + 函数签名]```                                ← 100%，不 mask（任务框架）
```

### 各部分 mask 决策

| 部分 | mask？ | 频率 | 理由 |
|------|--------|------|------|
| 函数描述 | ❌ | 100% | 核心功能说明，mask 则模型不知做什么 |
| `Note that:` | ✅ | 18.9% | 含命名规则/阈值等，测试会 assert 具体值 |
| `raise exception for:` | ✅ | 26.0% | 异常处理要求，测试会 assert 异常类型 |
| `output with:` | ✅ | 100% | 返回值类型，测试严格 assert |
| 代码模板（import + 函数签名） | ❌ | 100% | 模型写代码的必要框架，mask 则任务无法完成 |

---

## Masking 示例

**原始 instruct_prompt：**
```
Zips all files in the specified directory and returns the path to the created zip file.
Note that: The zip name is always 'files.zip'
The function should raise the exception for: FileNotFoundError: if the directory does not exist
The function should output with:
    str: The path to the generated zip file. Returns None if no files found.
You should write self-contained code starting with:
```
import os, glob, zipfile
def task_func(directory):
```
```

**模型实际看到的（mask 后）：**
```
Zips all files in the specified directory and returns the path to the created zip file.
You should write self-contained code starting with:
```
import os, glob, zipfile
def task_func(directory):
```
```

三个字段全部删除，模型需要通过 Clarify 逐步恢复。

---

## 数据概览

- **总任务数**: 1140（BigCodeBench Full Set v0.1.4）
- **断句残留**: 0/1140
- **output_format masked**: 1140/1140（100%）
- **validation_rules masked**: 296/1140（26.0%）
- **note_that masked**: 216/1140（18.9%）
- **disclosure_info 内容完整**: 1140/1140（100%，与 masked_fields 完全一致）

## 各字段填充情况

| 字段 | 有内容的数量 | 占比 |
|------|------------|------|
| `output_format` | 1140 / 1140 | **100.0%** |
| `validation_rules` | 296 / 1140 | **26.0%** |
| `note_that` | 216 / 1140 | **18.9%** |
| `edge_cases` | — | 已移除（从 test 代码关键词提取，不准确） |

## 字段组合统计

| 组合 | 数量 | 占比 |
|------|------|------|
| **OF 仅输出格式** | ~648 | ~57% |
| **OF + VR** | ~236 | ~21% |
| **OF + NT** | ~156 | ~14% |
| **OF + VR + NT** | ~60 | ~5% |

（OF = output_format，VR = validation_rules，NT = note_that）

## Output Format 内容复杂度

| 复杂度 | 标准 | 数量 | 占比 |
|--------|------|------|------|
| 简单 | < 50 chars（单类型，易推断） | 138 | **12.1%** |
| 中等 | 50–150 chars | 682 | **59.8%** |
| 复杂 | > 150 chars（tuple/dict 结构，严格 assert） | 320 | **28.1%** |

长度：min=4, median=98, mean=119, max=561 chars

## Task Uncertainty 分布

- min=0.50, mean=0.79, max=1.00
- 1138/1140（99.8%）> 0.5，足够触发轨迹生成的 Clarify 探索

## Masking 实现

全部基于固定结构锚点，不使用 regex：

| 字段 | 锚点 start | 锚点 end |
|------|-----------|---------|
| `output_format` | `"The function should output with:\n"` | `"You should write self-contained code"` |
| `validation_rules` | `"The function should raise the exception for:"` | 下一个 section 锚点 |
| `note_that` | `"Note that:"` | 下一个 section 锚点 |

## Disclosure 机制

`disclosure_rule` 对 Policy 模型不可见，仅在轨迹生成阶段的 simulator 使用：

| 字段 | 触发关键词（示例） | 优先级 |
|------|----------------|--------|
| `output_format` | output, return, format, type, result, what | 1（最高） |
| `validation_rules` | error, exception, raise, handle, fail | 2 |
| `note_that` | note, special, specific, always, fixed, name | 3 |

- Novice (low)：每轮最多披露 1 条
- Busy (mid)：每轮最多披露 3 条
- Experienced (high)：每轮最多披露 6 条

## 与 v28 对比

| 指标 | v28 (470 tasks) | v29 (1140 tasks) |
|------|----------------|-----------------|
| 断句残留 | 470/470 (100%) | **0/1140 (0%)** |
| output_format masked | 456/470 (97%) | **1140/1140 (100%)** |
| validation_rules masked | regex，不稳定 | **296/1140，固定锚点** |
| note_that masked | ❌ | **216/1140，固定锚点** |
| disclosure_info 完整 | 0/470 (0%) | **1140/1140 (100%)** |
| Base Llama unmasked pass@1 | ~30% | ~30%（模型相同） |
| Base Llama masked pass@1 | 0% | 待测 |
