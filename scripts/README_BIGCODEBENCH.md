# BigCodeBench Dataset Processing Pipeline

本文档描述如何使用三步流程处理BigCodeBench数据集，创建适合ProactiveLLM训练的masked任务。

## 三步流程概述

### Step 1: 筛选适合做澄清的题目

不是所有BigCodeBench题目都适合。需要筛选那些"天然可以被写得不完整"的题目。

**特征：**
- 输入约束未写清（是否有空输入、负数、重复等）
- 输出格式是否严格
- edge case行为是否可多种合理解释

**使用方法：**

```bash
# 规则筛选（快速，无需API）
python scripts/filter_ambiguity_friendly_tasks.py \
  --input data/external/BigCodeBench/v0.1.4.jsonl \
  --output data/external/BigCodeBench/ambiguous_tasks.jsonl \
  --method rule

# LLM筛选（更准确，需要API key）
export OPENAI_API_KEY=sk-...
python scripts/filter_ambiguity_friendly_tasks.py \
  --input data/external/BigCodeBench/v0.1.4.jsonl \
  --output data/external/BigCodeBench/ambiguous_tasks.jsonl \
  --method llm \
  --llm_model gpt-4o-mini
```

### Step 2: Mask关键细节（制造不确定性）

对选中的任务，有意识地删掉/mask一些信息。

**Mask的信息：**
- 输入范围/约束（空输入、负数、重复等）
- 特殊值处理规则
- 返回值细节
- 是否需要排序、过滤、去重等

**结果：**
- 用户最初看到的task = 不完整specification
- ground truth和hidden tests不变
- 只是assistant在初始状态看不到完整信息

**使用方法：**

```bash
python scripts/mask_task_details.py \
  --input data/external/BigCodeBench/ambiguous_tasks.jsonl \
  --output data/external/BigCodeBench/masked_tasks.jsonl \
  --mask_level moderate
```

**Mask级别：**
- `light`: 轻度mask，只移除部分细节
- `moderate`: 中度mask（推荐），mask输入约束和edge cases
- `heavy`: 重度mask，mask大部分约束和格式要求

### Step 3: 定义Disclosure Rule（什么时候补信息）

只有当assistant ASK clarification时，用户（模拟器）才会补充被mask的信息。

**规则：**
- `assistant = EXECUTE` → 用户不会主动给缺失信息
- `assistant = ASK` → user simulator从被mask的字段中选相关项，给出补充说明

**转换为State格式：**

```bash
python scripts/convert_masked_to_states.py \
  --input data/external/BigCodeBench/masked_tasks.jsonl \
  --output data/seeds/bigcodebench_masked_states.jsonl \
  --domain coding
```

## 完整流程示例

```bash
# Step 1: 筛选
python scripts/filter_ambiguity_friendly_tasks.py \
  --input data/external/BigCodeBench/v0.1.4.jsonl \
  --output data/external/BigCodeBench/ambiguous_tasks.jsonl \
  --method rule \
  --limit 100  # 测试时限制数量

# Step 2: Mask
python scripts/mask_task_details.py \
  --input data/external/BigCodeBench/ambiguous_tasks.jsonl \
  --output data/external/BigCodeBench/masked_tasks.jsonl \
  --mask_level moderate

# Step 3: 转换为State
python scripts/convert_masked_to_states.py \
  --input data/external/BigCodeBench/masked_tasks.jsonl \
  --output data/seeds/bigcodebench_masked_states.jsonl \
  --domain coding

# 然后使用转换后的states生成轨迹
export OPENAI_API_KEY=sk-...
python scripts/generate_trajectories.py \
  --mode dataset \
  --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 100 \
  --out logs/traj_bigcodebench.jsonl \
  --llm_model gpt-4o-mini
```

## 技术细节

### Disclosure Rule结构

每个masked任务包含`disclosure_rule`字段：

```json
{
  "masked_fields": {
    "input_constraints": [...],
    "output_format": [...],
    "edge_cases": [...],
    "validation_rules": [...]
  },
  "disclosure_info": {
    "input_constraints": {
      "edge_cases": ["empty input", "negative numbers"],
      "hints": ["Should handle empty inputs", ...]
    },
    "output_format": {
      "specification": "..."
    },
    "validation_rules": {
      "rules": [...]
    }
  }
}
```

### Simulator集成

`simulator/react()`函数已更新，支持`disclosure_rule`参数：

```python
reaction = react(
    user_msg=state["query"],
    assistant_msg=assistant_msg,
    persona=persona,
    llm_model=llm_model,
    disclosure_rule=state.get("disclosure_rule")  # 可选
)
```

当assistant问澄清问题时，simulator会从`disclosure_rule`中提取相关信息并补充到回答中。

## 文件说明

- `scripts/filter_ambiguity_friendly_tasks.py`: Step 1脚本
- `scripts/mask_task_details.py`: Step 2脚本
- `scripts/convert_masked_to_states.py`: Step 3脚本
- `simulator/disclosure.py`: Disclosure规则模块
- `simulator/simulate.py`: 已更新支持disclosure_rule


