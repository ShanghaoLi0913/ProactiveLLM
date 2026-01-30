# ProactiveLLM 项目流程总览

## 🎯 项目核心目标

训练一个**Policy模型**，让AI助手能够根据上下文自动决定：
- **Clarify（问问题）**: 当任务不明确时，先问问题获取更多信息
- **Execute（直接执行）**: 当任务明确或用户不耐烦时，直接提供解决方案

**核心问题**: 如何让模型学会"校准"（Calibration）——在不同情况下选择最合适的策略？

---

## 📊 完整流程概览

```
1. 数据生成 (generate_trajectories.py)
   └─> 生成多轮对话数据
   
2. 奖励计算 (compute_rewards.py)
   └─> 计算每个turn的reward，生成DPO偏好对
   
3. 模型训练 (train_dpo.py)
   └─> 使用DPO训练Policy模型
   
4. 模型评估 (evaluate_dpo_model.py)
   └─> 评估模型性能
```

---

## 1️⃣ 数据生成 (Step 1: Generate Trajectories)

### 目标
生成多轮对话数据，记录每个turn的状态、动作、assistant消息、user反应。

### 输入
- **初始States**: 来自 `data/seeds/bigcodebench_masked_states.jsonl`

#### 初始State的完整结构

每个初始State是一个JSON对象，包含以下字段：

**1. 核心字段（必需）**:
- `id` (string): 任务唯一标识符，例如 `"BigCodeBench/0"`
- `domain` (string): 任务领域，`"coding"` 或 `"planning"`
- `query` (string): **Masked版本的任务描述**（部分信息被隐藏）
  - 这是Policy模型和Assistant最初看到的内容
  - 例如：缺少输入约束、输出格式细节、边界情况等信息
- `dialogue_turn` (int): 对话轮次，初始为 `0`
- `prev_reject` (int): 是否被拒绝，初始为 `0`
- `task_uncertainty` (float): 任务不确定性分数，范围 `[0.0, 1.0]`
  - 基于masked后的`query`自动计算
  - masked越多，不确定性越高

**2. 评估相关字段（用于计算reward和测试）**:
- `original_instruct_prompt` (string): **原始完整任务描述**（未被mask）
  - **用途**：作为ground truth参考，用于评估分析
  - **注意**：评估时Policy模型**仍然只看到masked版本的`query`**（和训练时一致）
  - `original_instruct_prompt`用于：
    - 人工评估时参考（对比masked版本和完整版本，分析模型理解程度）
    - 调试分析（看masked版本缺失了什么信息）
    - 作为"正确答案"的参考标准
  - Policy模型不可见（保证训练和评估的一致性）
- `canonical_solution` (string): 标准答案代码
- `test` (string): 完整的测试用例代码（Python unittest格式）
  - **基于完整版本设计**：测试用例包含了所有边界情况和完整要求
  - 用于验证生成的代码是否正确
- `entry_point` (string): 函数入口点，例如 `"task_func"`

**3. Simulator相关字段（用于模拟用户回答）**:
- `disclosure_rule` (dict): **被mask的信息**，包含：
  - `masked_fields`: 被隐藏的字段（输入约束、输出格式、边界情况等）
  - `disclosure_info`: 隐藏的详细信息
  - 当Assistant问问题时，Simulator根据这个规则提供相应的信息
  - **Policy模型不可见**（避免"作弊"）

#### 初始State的生成流程

```
1. 原始任务（BigCodeBench数据集）
   ↓ mask_task_details.py
2. Masked任务（部分信息被隐藏）
   ↓ convert_masked_to_states.py
3. 初始State（包含disclosure_rule）
```

**关键点**:
- `query` = masked版本（不完整，**训练和评估时Policy模型都只看到这个**）
- `original_instruct_prompt` = 完整版本（作为ground truth参考，**Policy模型不可见**）
- `disclosure_rule` = 被隐藏的信息（用于Simulator模拟用户回答）
- `test` = 测试用例（基于完整版本设计，用于验证代码正确性）

**为什么需要`original_instruct_prompt`？**
- 不是给Policy模型看的（评估时模型仍然只看到masked版本，保证公平）
- 而是作为**ground truth参考标准**：
  - 测试用例基于完整版本设计（包含所有边界情况）
  - 人工评估时可以对比分析（看模型是否理解了完整任务）
  - 调试时参考（看masked版本缺失了什么关键信息）

#### 初始State示例

以下是一个真实的初始State（来自BigCodeBench/0）：

**Masked版本（query字段）** - Policy和Assistant看到的：
```
"Calculates the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the differences. Args: - numbers (list): A list of numbers. \nThe function \nYou should write self-contained code starting with:\n```\nimport itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n```"
```

**完整版本（original_instruct_prompt字段）** - 评估时使用的：
```
"Calculates the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the differences. Args: - numbers (list): A list of numbers. Default is numbers from 1 to 10.\nThe function should output with:\n    float: The average of the sums of absolute differences for each shuffled permutation of the list.\nYou should write self-contained code starting with:\n```\nimport itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n```"
```

**对比差异**（被mask的信息）：
- ❌ **缺失**: "Default is numbers from 1 to 10." （输入约束）
- ❌ **缺失**: "should output with: float: The average..." （输出格式说明）

**disclosure_rule（被隐藏的信息）** - Simulator用这个来回答：
```json
{
  "masked_fields": {
    "input_constraints": ["Default is numbers from 1 to 10."],
    "output_format": ["should output with:\n    float: The average of the sums..."],
    "edge_cases": ["empty input", "negative numbers", "zero value", "single element", "identical elements", "large inputs"]
  },
  "disclosure_info": {
    "input_constraints": {
      "edge_cases": ["empty input", "negative numbers", ...],
      "hints": ["Should handle empty inputs", "Should handle negative numbers", ...]
    }
  }
}
```

**含义**:
- Policy模型看到的是**不完整**的任务描述（query），不知道默认值是1到10，也不知道输出格式要求
- 当Assistant选择Clarify时，Simulator会根据`disclosure_rule`提供这些被隐藏的信息（模拟用户回答）
- 评估时，使用完整版本（original_instruct_prompt）来验证代码是否正确

### 过程

#### 1.1 对每个初始State生成完整对话

```
初始State (dialogue_turn=0)
  ↓
Turn 1: Policy选择动作 → Assistant生成消息 → Simulator模拟用户反应
  ↓
更新State (dialogue_turn=1, query累积)
  ↓
Turn 2: Policy选择动作 → Assistant生成消息 → Simulator模拟用户反应
  ↓
...
直到任务完成或达到max_turns
```

#### 1.2 关键组件

**Overview**:
数据生成阶段：输入是「初始 states」，输出是「trajectories（轨迹）」
Persona在这一阶段的作用：Persona作为System Prompt（系统提示词）的方式，强行给LLM注入一套灵魂。

**数据生成阶段的三个组成部分**:

1. **Assistant LLM**: 负责回答（生成澄清问题或代码）
2. **User LLM**: 负责提问（根据Assistant的问题生成用户回答）
3. **Program**: 负责控制问答流程
   - 控制Assistant LLM什么时候答、大概答什么、什么时候结束
   - Program和User LLM共同组成User Simulator（用户模拟器）

**Persona（用户画像）的设计理念**:

这个项目的目的是能让LLM综合考虑**任务成功率**和**用户偏好**：

- **任务成功率**：主要由模型本身的能力（这个没法控制）和任务模糊度（state里有`task_uncertainty`字段）决定
- **用户偏好**：直接反应用户回答澄清问题的可能性，主要由：
  - 用户本身对于任务不确定性的认知程度（`expertise`决定）
  - 耐心水平（`Patience`决定）

**三个Persona类型**:

```python
# 配置文件中保留数值，方便随时微调
USER_PROFILES = {
    "Novice-Learner": {
        "expertise": 0.3,  # 影响提问后的反馈质量（Noise）
        "patience": 0.9,   
        "label": "Beginner with plenty of patience"
    },
    "Busy-Developer": {
        "expertise": 0.6,
        "patience": 0.2,   # 极其容易流失
        "label": "Professional in a hurry"
    },
    "Experienced-Engineer": {
        "expertise": 0.9,
        "patience": 0.4,
        "label": "Expert seeking efficiency"
    }
}
```

**1. Novice-Learner（小白但有耐心）**:
- 高`task_uncertainty` + 低`expertise`
- 能接受多轮Clarify，不太容易reject
- LLM的合理策略：多问、慢慢来

**2. Busy-Developer（忙碌的熟手）**:
- 对不确定性有判断，但没时间讨论
- Clarify成本高，容易reject
- 更偏好"直接给我一个能跑的方案"
- LLM的合理策略：少问甚至直接Execute

**3. Experienced-Engineer（专家但没空）**:
- 能识别哪些uncertainty是致命的
- 接受Clarify，但只接受高质量、一次到位
- 不接受"新手式追问"
- LLM的合理策略：问1次关键问题

**目标**:
> **让LLM在任务信息缺失的情况下，根据用户对不确定性的容忍度和耐心，决定是否、何时、以及问多少澄清问题。**

在这套Persona下的策略对应关系：

| Persona              | task_uncertainty 高时 | LLM的合理策略       |
| -------------------- | --------------------- | -------------------- |
| Novice-Learner       | 容忍度高              | 多问、慢慢来         |
| Busy-Developer       | 容忍度低              | 少问甚至直接Execute |
| Experienced-Engineer | 选择性容忍            | 问1次关键问题        |

**Program的逻辑（Simulator的核心机制）**:

**维度A：Patience决定"存活概率"（生存判定）**

这是为了模拟用户流失。

- **当前实现**：`P(answer) = patience`（每次turn独立判断，无衰减）
- **设计意图**（待实现）：更复杂的衰减公式
  - **公式**：$P_{survive} = Patience \times (0.7)^{Turn-1}$
  - **执行**：
    - 第一轮提问：如果是Busy-Dev (0.2)，存活率 20%
    - 第二轮提问：存活率降至 14%
    - 一旦随机数触发失败，Program立即返回 `[SYSTEM_SIGNAL: USER_QUIT]`，不调用User LLM

**维度B：Expertise决定"披露步长"（内容抽取）**

这是为了模拟用户表达能力。

- **当前实现**：基于关键词匹配的披露，expertise影响回答详细程度
- **设计意图**（待实现）：逐步披露机制
  - **逻辑**：每次Clarify，Program决定从`knowledge_pool`中新释放多少个点
  - **计算步长 ($K$)**：$K = \lceil Expertise \times 3 \rceil$
    - Novice-Learner (low, 0.3): K = ⌈0.9⌉ = 1 个信息点
    - Busy-Developer (mid, 0.6): K = ⌈1.8⌉ = 2 个信息点
    - Experienced-Engineer (high, 0.9): K = ⌈2.7⌉ = 3 个信息点
  - **披露逻辑**：
    - 第一轮：披露前 $K$ 个点
    - 第二轮：披露前 $2K$ 个点
    - **注意**：如果Assistant问的问题很泛，Program就按顺序给；如果Assistant问得准，Program优先匹配对应类别的字段

**数据生成阶段，又没有训练好的Policy，LLM怎么做出反应？**

当前使用**Persona规则**（Heuristic Policy）来选择动作：

- 根据Persona和State选择动作:
  - 低耐心 → Execute
  - 高耐心 + 高不确定性 → Clarify
  - 之前被拒绝 → Execute
  - 对话轮次太多 → Execute

**采样策略建议**（用于确保数据多样性）:

为了确保能采到"高分"和"低分"的轨迹，建议在生成时手动控制Assistant的起始动作：

1. **采样1（Heuristic）**：强制模型首轮必须`Execute`（盲猜）
2. **采样2（Heuristic）**：强制模型首轮必须`Clarify`（提问）
3. **采样3 & 4（Free）**：让模型根据Temperature自由发挥

这样就能确保：既有"因为直接做而成功/失败"的例子，也有"因为提问而成功/断连"的例子。对比这些例子，模型才能学会不同Persona下的边界在哪里。

**每个Task必须在所有Persona维度下都生成轨迹**:

为了获得全面的训练数据，建议对每个初始State，使用所有Persona（或至少3-4个不同的Persona）生成轨迹。这样可以确保模型学到不同用户类型下的最优策略。

#### 1.3 输出格式

每个turn生成一个Trajectory记录（保存到 `data/logs/traj_*.jsonl`）:

```json
{
  "state": {
    "id": "task_001",                    // 任务ID（所有turns相同）
    "domain": "coding",                  // 领域
    "dialogue_turn": 0,                  // 当前turn编号（从0开始）
    "query": "...",                      // 累积的对话内容
    "prev_reject": 0,                    // 是否被拒绝
    "task_uncertainty": 0.8,             // 任务不确定性
    // 以下字段可选（如果从数据集加载则存在）:
    "test": "...",                       // 测试用例（基于完整版本设计）
    "canonical_solution": "...",         // 标准答案
    "entry_point": "task_func",          // 函数入口
    "original_instruct_prompt": "...",   // 原始任务描述（完整版本，Policy不可见）
    "disclosure_rule": {...}             // 信息披露规则（Policy不可见，用于Simulator）
  },
  "action": "Clarify" | "Execute",      // 选择的动作
  "action_prompt": "...",                // 动作的prompt模板（用于生成assistant_msg）
  "assistant_msg": "...",                // Assistant生成的消息（代码或澄清问题）
  "persona": {                           // 使用的用户画像
    "name": "Novice-Learner",
    "domain": "coding",
    "expertise": "low",
    "patience": "high"
  },
  "user_reaction": {                     // 用户反应
    "user_reply": "...",                 // 用户的文本回复
    "meta": {
      "answered_clarification": 1,       // 是否回答了澄清问题 (0/1)
      "reject_signal": 0,                // 是否拒绝 (0/1)
      "answer_clarity": 0.3,             // 回答清晰度 [0.0, 1.0]
      "satisfaction": 0.85,              // 用户满意度 [0.0, 1.0]
      "silence": 0,                      // 是否沉默 (0/1)
      "off_topic_flag": 0                // 是否离题 (0/1)
    }
  },
  "turn": 1,                             // turn编号（从1开始，与dialogue_turn不同）
  "is_mainline": true,                   // multi-turn模式下都是true
  "is_terminal": false,                  // 是否轨迹终点（如果为true，这是最后一条轨迹）
  "task_completed": false,               // 可选：是否完成任务（如果完成任务则为true）
  "user_stopped": false                  // 可选：用户是否停止（如果用户拒绝则为true）
}
```

**关键点**:
- **State.query是累积的**: 每个turn的`query`包含前面所有turns的对话历史
- **State.id不变**: 同一个任务的所有turns共享相同的`id`
- **dialogue_turn与turn的区别**:
  - `dialogue_turn`: State中的字段，从0开始（0, 1, 2, ...）
  - `turn`: Trajectory中的字段，从1开始（1, 2, 3, ...）
  - 例如：Turn 1对应`dialogue_turn=0`的State，Turn 2对应`dialogue_turn=1`的State
- **is_terminal字段的作用**:
  - 明确标识轨迹的终点，便于在DPO训练中快速定位
  - 当`is_terminal=true`时，表示这是该对话的最后一条轨迹
  - 触发`is_terminal=true`的情况：
    1. `task_completed=true`（正常完成）
    2. `user_stopped=true`（用户拒绝，非正常结束）
    3. 达到`max_turns`（超时结束）
- **task_completed和user_stopped的互斥性**:
  - 当`user_stopped=true`时，`task_completed`必须为`false`（用户中断 = 任务失败）
  - 当`task_completed=true`时，不会再有下一个turn（对话成功结束）
  - **重要**：如果用户拒绝（`reject_signal=1`），对话立即停止，不会生成代码，`task_completed=false`
- **prev_reject的传递逻辑**:
  - 如果Turn N的`user_reaction.meta.reject_signal=1`，那么下一个turn的`state.prev_reject=1`
  - 注意：在实际代码中，如果`reject_signal=1`，对话会立即停止（`user_stopped=true`），不会有下一个turn
  - 但保留这个逻辑是为了完整性，确保如果策略改变（允许reject后继续），State能正确传递用户态度
- **State字段的完整性**:
  - 核心字段（必需）: `id`, `domain`, `query`, `dialogue_turn`, `prev_reject`, `task_uncertainty`
  - 评估相关字段（可选，从数据集加载时存在）: `test`, `canonical_solution`, `entry_point`, `original_instruct_prompt`
  - Simulator相关字段（可选，从数据集加载时存在）: `disclosure_rule`
- **disclosure_rule不传给Policy**: Policy只能看到masked信息（通过`render_state()`），不能"作弊"
- **action_prompt**: 包含用于生成assistant_msg的完整prompt模板
- **persona**: 完整的用户画像信息，用于理解用户反应

---

## 2️⃣ 奖励计算 (Step 2: Compute Rewards)

### 目标
计算每个trajectory的reward，生成DPO训练所需的preference pairs。

### 输入
- **Trajectories**: 来自 `data/logs/traj_*.jsonl`

### 过程

#### 2.1 分组

**按 `(state_id, dialogue_turn)` 分组**:
- 关键: 只在**同一个dialogue_turn**内比较
- 原因: 避免"特征偏移"（Feature Drift）
  - Turn 1和Turn 3的`query`长度不同
  - 跨turn比较会让模型学到"把上下文拉长"而不是"做出正确决策"

```python
groups = {}
for trajectory in trajectories:
    state_id = trajectory["state"]["id"]
    dialogue_turn = trajectory["state"]["dialogue_turn"]
    key = (state_id, dialogue_turn)
    groups[key].append(trajectory)
```

**重要说明**: 要在同一个`dialogue_turn`内生成preference pairs，需要在该turn下有**多个轨迹**（分叉路径）。如果每个turn只有一条轨迹，则无法生成preference pairs。

#### 2.2 计算Reward

**公式**:
```
R = w_task × R_task - w_interrupt × C_interrupt
```

其中:
- `w_task = 1.0` (任务成功权重)
- `w_interrupt = 0.15` (中断成本权重)

**R_task (任务成功分数)**:
- 有代码且通过测试: `R_task = 1.0`
- 有代码但无测试: `R_task = 0.5`
- 只有问题（无代码）: `R_task = 0.0`

**C_interrupt (中断成本)**:
```
C = δ × b × r + λ × b - γ × b × a
```

参数:
- `b = 1` 如果当前turn有提问，否则 `0`
- `a = answered_clarification` (用户是否回答，0/1)
- `r = reject_signal` (用户是否拒绝，0/1)
- `δ = 0.7` (拒绝惩罚)
- `λ = 0.0` (提问成本，当前设为0)
- `γ = 0.3` (有效澄清奖励)

计算结果:
- **有效澄清** (a=1, r=0): `C = -0.3 × n_questions` (负值=奖励)
- **被拒绝** (r=1): `C = 0.7 × n_questions` (惩罚)
- **未回答** (a=0, r=0): `C = 0`
- **无提问** (b=0): `C = 0`

#### 2.3 生成Preference Pairs

对每个group（同一个`(state_id, dialogue_turn)`）:
1. 计算所有trajectories的reward
2. 按reward降序排序
3. 选择reward最高的作为`chosen`
4. 选择reward最低的作为`rejected`
5. 生成preference pair

**输出**: `data/dpo/prefs_*.jsonl`

```json
{
  "state": {...},
  "chosen_action": "Execute",
  "rejected_action": "Clarify",
  "chosen_text": "Execute",
  "rejected_text": "Clarify",
  "rewards": {
    "chosen": 1.0,
    "rejected": 0.09
  },
  "task_scores": {...},
  "interrupt_costs": {...}
}
```

---

## 3️⃣ 模型训练 (Step 3: Train Policy)

### 目标
使用DPO训练Policy模型，学习何时选择Clarify/Execute。

### 输入
- **Preference Pairs**: 来自 `data/dpo/prefs_*.jsonl`
- **Base Model**: 例如 `meta-llama/Llama-3.1-8B-Instruct`

### 过程

#### 3.1 Policy模型的作用

**Policy模型只做一件事**: 根据State预测动作（Clarify/Execute）

**输入**: State的文本表示（通过`render_state()`生成）
- 包含: `query`, `dialogue_turn`, `prev_reject`, `task_uncertainty`等
- **不包含**: `disclosure_rule`（Policy不能"作弊"）

**输出**: 动作token (`Clarify` 或 `Execute`)

**关键设计**: Policy模型**不生成代码**，代码生成由单独的模型完成（避免DPO污染代码生成能力）。

#### 3.2 DPO训练

使用TRL的DPOTrainer:
- **Prompt**: `render_state(state)` (State的文本表示)
- **Chosen**: `chosen_action` (例如 "Execute")
- **Rejected**: `rejected_action` (例如 "Clarify")
- **Loss**: DPO loss (最大化chosen的logit，最小化rejected的logit)

**输出**: 训练好的Policy模型（LoRA adapter）
- 保存到 `outputs/policy_model/`

---

## 4️⃣ 模型评估 (Step 4: Evaluate)

### 目标
评估训练好的Policy模型性能。

### 过程

#### 4.1 评估流程

1. **加载Policy模型**: 从 `outputs/policy_model/` 加载
2. **加载测试States**: 来自 `data/seeds/` 或 `data/eval/`
3. **对每个State**:
   - 使用Policy模型预测动作（Clarify/Execute）
   - 根据动作生成代码（使用单独的代码生成模型）
   - 执行测试，计算成功率
4. **统计指标**:
   - 任务成功率
   - 平均对话轮次
   - 中断成本
   - Pareto曲线（成功率 vs 中断成本）

#### 4.2 评估设计

**当前设计**: **Single-turn评估**
- 每个测试State只评估一次（不是多轮对话）
- 原因: 简化评估，快速验证模型性能
- 与数据生成的Multi-turn不同:
  - **数据生成**: Multi-turn（完整对话模拟，用于生成训练数据）
  - **评估**: Single-turn（简化评估，用于快速验证）

---

## 🔑 核心概念总结

### State（状态）
- **定义**: 对话的当前状态快照
- **包含**: `id`, `domain`, `query`（累积）, `dialogue_turn`, `prev_reject`, `task_uncertainty`, `test`, `canonical_solution`, `disclosure_rule`等
- **变化**: `dialogue_turn`递增，`query`累积，`prev_reject`可能变化

### Turn（轮次）
- **定义**: 一轮对话（Assistant说话 → User回复）
- **编号**: 从1开始（`turn: 1, 2, 3...`）
- **对应State**: `dialogue_turn: 0, 1, 2...`（从0开始）

### Trajectory（轨迹/决策记录）
- **定义**: 一个Turn的完整记录
- **格式**: JSONL文件中的一行
- **包含**: `state`, `action`, `assistant_msg`, `user_reaction`, `turn`等
- **注意**: 在RL术语中，这是"single-step trajectory"或"decision record"，不是完整的episode

### Policy模型
- **作用**: 根据State预测动作（Clarify/Execute）
- **输入**: State的文本表示（不包含`disclosure_rule`）
- **输出**: 动作token (`Clarify` 或 `Execute`)
- **训练**: DPO（Direct Preference Optimization）
- **不负责**: 代码生成（由单独的模型完成）

### Reward计算
- **公式**: `R = w_task × R_task - w_interrupt × C_interrupt`
- **R_task**: 任务成功分数（0.0/0.5/1.0）
- **C_interrupt**: 中断成本（基于用户反应）
- **分组**: 按`(state_id, dialogue_turn)`分组，只在同一turn内比较

---

## 📝 完整示例

### 数据生成示例

假设一个3-turn对话（`state_id="task_001"`）:

```
初始State:
  id: "task_001"
  dialogue_turn: 0
  query: "写一个函数计算两个数的和"

Turn 1 (Clarify):
  state.dialogue_turn: 0
  state.query: "写一个函数计算两个数的和\n\n[Assistant]: 请问这两个数是什么类型？整数还是浮点数？\n[User]: 整数"
  action: "Clarify"
  assistant_msg: "请问这两个数是什么类型？整数还是浮点数？"
  user_reaction: {answered: 1, reject: 0}

Turn 2 (Clarify):
  state.dialogue_turn: 1
  state.query: "写一个函数计算两个数的和\n\n[Assistant]: 请问这两个数是什么类型？整数还是浮点数？\n[User]: 整数\n\n[Assistant]: 是否需要处理负数？\n[User]: 需要"
  action: "Clarify"
  assistant_msg: "是否需要处理负数？"
  user_reaction: {answered: 1, reject: 0}

Turn 3 (Execute):
  state.dialogue_turn: 2
  state.query: "写一个函数计算两个数的和\n\n[Assistant]: 请问这两个数是什么类型？整数还是浮点数？\n[User]: 整数\n\n[Assistant]: 是否需要处理负数？\n[User]: 需要\n\n[Assistant]: ```python\ndef add(a, b):\n    return a + b\n```"
  action: "Execute"
  assistant_msg: "```python\ndef add(a, b):\n    return a + b\n```"
  user_reaction: {answered: 0, reject: 0}
  task_completed: true
```

### Reward计算示例

**Turn 1**:
- R_task = 0.0 (无代码)
- C_interrupt = -0.3 × 2 = -0.6 (2个问题，有效澄清)
- R = 1.0 × 0.0 - 0.15 × (-0.6) = **0.09**

**Turn 2**:
- R_task = 0.0 (无代码)
- C_interrupt = -0.3 × 1 = -0.3 (1个问题，有效澄清)
- R = 1.0 × 0.0 - 0.15 × (-0.3) = **0.045**

**Turn 3**:
- R_task = 1.0 (代码通过测试)
- C_interrupt = 0.0 (无提问)
- R = 1.0 × 1.0 - 0.15 × 0.0 = **1.0**

**Preference Pair生成**:
- 如果Turn 1和Turn 3在同一个`(state_id, dialogue_turn)`组（实际上不在，因为`dialogue_turn`不同）
- 实际分组: 每个turn单独一组（如果只有一个轨迹）
- **问题**: 如果每个`(state_id, dialogue_turn)`只有一个轨迹，无法生成preference pairs

---

## ⚠️ 当前限制

### 数据生成限制

**问题**: 当前数据生成只产生**单一路径**（每个state一条对话路径）
- 对于每个`(state_id, dialogue_turn)`，只有**一个轨迹**
- **无法生成preference pairs**（需要至少2个轨迹才能比较）

**解决方案**:
1. **多Persona生成**: 使用不同的Persona生成多条路径
2. **显式分支**: 在`generate_trajectories.py`中为每个turn生成分支（Clarify和Execute两种）

---

## 🎯 总结

**项目核心**: 训练Policy模型学会"校准"（Calibration）——在不同情况下选择最合适的策略（Clarify/Execute）。

**数据流程**:
1. 生成多轮对话数据（Multi-turn）
2. 计算每个turn的reward
3. 生成DPO偏好对
4. 训练Policy模型
5. 评估模型性能（Single-turn）

**关键设计**:
- Policy模型只预测动作，不生成代码
- State.query是累积的（包含完整对话历史）
- Reward分组按`(state_id, dialogue_turn)`（避免特征偏移）
- 评估使用Single-turn（简化评估流程）
