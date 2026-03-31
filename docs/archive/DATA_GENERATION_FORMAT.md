# 数据生成格式说明

**生成时间**: 2026-03-17  
**数据生成脚本**: `scripts/generate_trajectories.py`  
**Reward计算脚本**: `reward/compute_rewards.py`

---

## 📊 数据生成流程

### Step 1: 生成Trajectories

**脚本**: `scripts/generate_trajectories.py`  
**输入**: States JSONL (`data/seeds/*.jsonl`)  
**输出**: Trajectories JSONL (`data/logs/traj_*.jsonl`)

### Step 2: 计算Rewards并生成Preference Pairs

**脚本**: `reward/compute_rewards.py`  
**输入**: Trajectories JSONL (`data/logs/traj_*.jsonl`)  
**输出**: Preference Pairs JSONL (`data/dpo/prefs_*.jsonl`)

---

## 1️⃣ Trajectory数据格式

**文件位置**: `data/logs/traj_*.jsonl`  
**格式**: 每行一个JSON对象，代表一个对话轮次（turn）

### 字段说明

```json
{
  "trajectory_id": "unique_id_123",  // 唯一ID，用于链接同一对话的所有轮次
  "state": {
    "id": "BigCodeBench/0",           // 任务ID
    "domain": "coding",                // 领域（coding/planning）
    "query": "任务描述...",            // 用户查询（可能是masked或full）
    "dialogue_turn": 0,                // 对话轮次（从0开始）
    "prev_reject": 0,                  // 之前是否被拒绝（0/1）
    "task_uncertainty": 0.8,           // 任务不确定性（0-1）
    "convcodeworld_tests": "...",      // 测试用例（Python代码）
    "original_instruct_prompt": "...", // 原始完整指令
    "canonical_solution": "...",       // 标准答案
    "disclosure_rule": {...}           // 信息披露规则
  },
  "action": "Execute",                 // 动作：Clarify 或 Execute
  "action_prompt": "You are...",      // 用于生成action的prompt
  "assistant_msg": "```python\n...",  // 助手回复（代码或问题）
  "persona": {
    "name": "Busy-Developer",          // Persona名称
    "domain": "coding",
    "expertise": "mid",                // 专业水平：low/mid/high
    "patience": "low"                  // 耐心程度：low/mid/high
  },
  "user_reaction": {
    "user_reply": "好的。",            // 用户回复文本
    "meta": {
      "answered_clarification": 1,     // 是否回答了澄清问题（0/1）
      "reject_signal": 0,              // 是否拒绝（0/1）
      "silence": 0,                    // 是否沉默（0/1）
      "off_topic_flag": 0,             // 是否偏离主题（0/1）
      "satisfaction": 0.7              // 满意度（0-1）
    }
  },
  "turn": 1,                          // 当前轮次（从1开始）
  "is_mainline": true,                 // 是否为主线轨迹
  "is_terminal": false,                // 是否终止状态
  "task_completed": false,             // 任务是否完成（仅Execute时）
  "has_edge_cases_info": false         // 是否获得了边界情况信息
}
```

### 关键特点

1. **多轮对话**: 每个对话轮次是一个独立的trajectory entry
2. **trajectory_id**: 同一对话的所有轮次共享相同的`trajectory_id`
3. **状态更新**: `dialogue_turn`会递增，`prev_reject`会根据用户反应更新
4. **终止条件**: 
   - Execute后必须终止（无论任务是否完成）
   - 用户拒绝（`reject_signal > 0`）时终止
   - 达到最大轮次（`max_turns`）时终止

---

## 2️⃣ Preference Pair数据格式

**文件位置**: `data/dpo/prefs_*.jsonl`  
**格式**: 每行一个JSON对象，代表一个preference pair（用于DPO训练）

### 字段说明

```json
{
  "state": {
    "id": "BigCodeBench/0",
    "domain": "coding",
    "query": "任务描述...",
    "dialogue_turn": 0,
    "prev_reject": 0,
    "task_uncertainty": 0.8,
    "convcodeworld_tests": "...",
    // ... 其他state字段
  },
  "persona": {
    "name": "Busy-Developer",
    "domain": "coding",
    "expertise": "mid",
    "patience": "low"
  },
  "chosen_action": "Execute",          // 被选择的动作（reward更高）
  "rejected_action": "Clarify",        // 被拒绝的动作（reward更低）
  "chosen_assistant_msg": "```python\n...",  // chosen动作的完整回复
  "rejected_assistant_msg": "Could you clarify...",  // rejected动作的完整回复
  "chosen_reward": 0.2,                // chosen动作的总reward
  "rejected_reward": -0.212,           // rejected动作的总reward
  "chosen_task_score": 0.0,            // chosen动作的task_score（0-1）
  "rejected_task_score": 0.0,          // rejected动作的task_score（0-1）
  "chosen_interrupt_cost": 0.0,        // chosen动作的interrupt_cost
  "rejected_interrupt_cost": 0.08,    // rejected动作的interrupt_cost
  "prev_action": null,                 // 上一个动作（如果有）
  "dialogue_turn": 0                   // 对话轮次
}
```

### Reward计算

**总Reward公式**:
```
total_reward = w_task * task_score - w_interrupt * interrupt_cost
```

其中：
- `w_task = 1.0` (任务成功权重)
- `w_interrupt = 0.3` (中断成本权重)
- `task_score`: 0-1，基于测试用例通过率
- `interrupt_cost`: 基于用户反应（clarify成本、拒绝成本等）

### Preference Pair选择

1. **按trajectory分组**: 同一`trajectory_id`的所有turns为一组
2. **计算reward**: 为每个turn计算`total_reward`
3. **排序选择**: 
   - 选择reward最高的作为`chosen_action`
   - 选择reward最低的作为`rejected_action`
   - 如果reward相同，使用tie-break机制（偏好Execute）

---

## 📋 数据生成示例

### 示例1: 单轮Execute

```json
{
  "trajectory_id": "traj_001",
  "state": {
    "id": "BigCodeBench/0",
    "query": "Write a function to reverse a string",
    "dialogue_turn": 0,
    "prev_reject": 0,
    "task_uncertainty": 0.3
  },
  "action": "Execute",
  "assistant_msg": "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
  "persona": {"name": "Busy-Developer", "patience": "low"},
  "user_reaction": {"user_reply": "Continue.", "meta": {...}},
  "turn": 1,
  "task_completed": true,
  "is_terminal": true
}
```

### 示例2: 多轮对话（Clarify → Execute）

**Turn 1 (Clarify)**:
```json
{
  "trajectory_id": "traj_002",
  "state": {"dialogue_turn": 0, ...},
  "action": "Clarify",
  "assistant_msg": "请问需要处理空字符串吗？",
  "turn": 1,
  "is_terminal": false
}
```

**Turn 2 (Execute)**:
```json
{
  "trajectory_id": "traj_002",  // 相同的trajectory_id
  "state": {"dialogue_turn": 1, ...},  // dialogue_turn递增
  "action": "Execute",
  "assistant_msg": "```python\ndef reverse_string(s):\n    if not s:\n        return ''\n    return s[::-1]\n```",
  "turn": 2,
  "task_completed": true,
  "is_terminal": true
}
```

---

## 🔍 数据统计

### 当前数据集

- **Trajectory文件**: `data/logs/traj_colm_3turn_persona_150states_*.jsonl`
  - 150个states
  - 3个personas
  - 每个(state, persona)组合生成2个samples（heuristic sampling）
  - 总计: 150 × 3 × 2 = 900个trajectories

- **Preference文件**: `data/dpo/traj_colm_3turn_persona_150states_*_prefs.jsonl`
  - 从trajectories计算reward并生成preference pairs
  - 通常每个trajectory生成1个preference pair
  - 总计: ~900个preference pairs

### 数据分割

- **Train/Test分割**: 通常80/20
- **Train文件**: `*_train_prefs.jsonl`
- **Test文件**: `*_test_prefs.jsonl`

---

## 📝 关键设计点

### 1. Trajectory-Level Reward

- 同一对话的所有轮次共享最终任务结果
- 如果最终任务完成，所有轮次都会受益
- 如果最终任务失败，所有轮次都会受到惩罚

### 2. Persona-Aware

- 不同persona有不同的行为模式
- Novice-Learner: 更倾向于Clarify
- Busy-Developer: 更倾向于Execute
- Experienced-Engineer: 平衡Clarify和Execute

### 3. Multi-Turn Support

- 支持多轮对话（最多5轮）
- 每轮都会更新state（`dialogue_turn++`, `prev_reject`等）
- Execute后必须终止

### 4. Reward设计

- **Task Score**: 基于测试用例通过率（0-1）
- **Interrupt Cost**: 基于用户反应（clarify成本、拒绝成本等）
- **Total Reward**: `w_task * task_score - w_interrupt * interrupt_cost`

---

## 📁 相关文件

- **数据生成脚本**: `scripts/generate_trajectories.py`
- **Reward计算脚本**: `reward/compute_rewards.py`
- **Trajectory示例**: `data/logs/traj_colm_3turn_persona_150states_*.jsonl`
- **Preference示例**: `data/dpo/traj_colm_3turn_persona_150states_*_prefs.jsonl`

---

*文档生成时间: 2026-03-17*
