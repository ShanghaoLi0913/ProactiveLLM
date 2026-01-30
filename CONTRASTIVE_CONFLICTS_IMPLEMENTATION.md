# 对比冲突实现文档

## 目标

确保轨迹生成包含**"对比的冲突"**，为DPO训练提供有效的信号。同样的任务（Task），在不同的画像（Persona）下，由于采取了不同的行动（Action），导致截然不同的结局（Reward）。

## 实现内容

### 1. 修改 `reward/compute.py`

**增强 `compute_task_score` 函数**：
- 新增参数：`persona_name`, `has_edge_cases_info`, `disclosure_rule`
- 根据persona类型和是否获得edge_cases信息调整task_score：

  - **Busy-Developer**: 容忍小瑕疵（效率优先）
    - 如果代码运行良好（base_score >= 0.7），给予小幅提升
    - 如果部分测试通过（base_score >= 0.5），给予更多容忍度
  
  - **Novice-Learner**: 需要edge_cases信息
    - 如果任务有隐藏edge_cases但Execute时没有通过Clarify获得信息 → 严重惩罚（score * 0.3）
    - 如果通过Clarify获得了edge_cases信息 → 正常评分
  
  - **Experienced-Engineer**: 平衡，奖励精准提问
    - 如果通过精准Clarify获得了edge_cases信息 → 小幅提升（+0.05）

### 2. 修改 `simulator/simulate.py`

**增强reject机制**：
- 对于**Experienced-Engineer**，检测"啰嗦提问"（obvious questions）：
  - 如果问题太短（<10词）或询问prompt中已有的信息 → 标记为obvious
  - 对于obvious问题，将effective_patience降低50%，增加reject概率
  - 给出更具体的reject消息："This information is already in the prompt. Please proceed with the implementation."

### 3. 修改 `scripts/generate_trajectories.py`

**跟踪edge_cases信息获取**：
- 在轨迹中新增字段：`has_edge_cases_info`
- 当action为Clarify且用户回答时，检查回答中是否包含edge_cases关键词
- 将`has_edge_cases_info`传递到state中，供后续turn使用

### 4. 修改 `reward/compute_rewards.py`

**传递persona和edge_cases信息**：
- 在计算task_score时，从trajectory中提取：
  - `persona_name`: 从`traj["persona"]["name"]`获取
  - `has_edge_cases_info`: 从`traj["has_edge_cases_info"]`获取
  - `disclosure_rule`: 从`traj["state"]["disclosure_rule"]`获取
- 支持single-interaction和multi-interaction两种模式

## 理想的数据分布

### Busy-Developer (低耐心/高专业度)
- **Chosen轨迹**: Turn 1直接Execute → 即使代码有小瑕疵，Reward依然很高（1.0分）
- **Rejected轨迹**: Turn 1选择Clarify → Simulator判定user_stopped: true（0分 - 熔断）
- **训练信号**: "在忙碌的人面前，提问是有风险的罪过。"

### Novice-Learner (高耐心/低专业度)
- **Chosen轨迹**: Turn 1 Clarify → Turn 2获得edge_cases → Turn 3 Execute完美代码 → Success=1（0.9分）
- **Rejected轨迹**: Turn 1直接Execute → 由于不知道隐藏的edge_cases，代码在评测时Fail → Success=0（0.2分）
- **训练信号**: "对小白不耐烦，会导致任务彻底失败。"

### Experienced-Engineer (中耐心/极高专业度)
- **Chosen轨迹**: 精准提问（问深层次的逻辑约束）→ User给高质量回复 → Execute成功（0.95分）
- **Rejected轨迹**: 啰嗦提问（问original_prompt里已写明的信息）→ User回复reject_signal: 1（0.6分 - 浪费时间）
- **训练信号**: "只问关键问题，不要问显而易见的问题。"

## 使用方式

### 生成轨迹
```bash
python scripts/generate_trajectories.py \
  --mode dataset \
  --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 10 \
  --out logs/traj_contrastive.jsonl \
  --max_turns 3 \
  --persona_idx 0 \
  --llm_model gpt-4o-mini
```

### 计算Reward
```bash
python reward/compute_rewards.py \
  --trajectories logs/traj_contrastive.jsonl \
  --out dpo/prefs_contrastive.jsonl
```

## 验证

生成轨迹后，检查：
1. **Busy-Developer**: 直接Execute的轨迹reward应该 > Clarify的轨迹reward
2. **Novice-Learner**: 先Clarify再Execute的轨迹reward应该 > 直接Execute的轨迹reward（对于有hidden edge_cases的任务）
3. **Experienced-Engineer**: 精准提问的轨迹reward应该 > 啰嗦提问的轨迹reward

## 注意事项

1. **disclosure_rule必须存在**: 确保state中包含`disclosure_rule`字段，用于判断任务是否有隐藏的edge_cases
2. **edge_cases检测**: 当前使用关键词匹配（"edge case", "empty", "negative"等），可能需要根据实际数据调整
3. **obvious question检测**: 当前使用简单的启发式规则，可能需要根据实际LLM输出调整



