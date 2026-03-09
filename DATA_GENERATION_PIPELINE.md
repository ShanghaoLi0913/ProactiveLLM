# 数据生成流程总结

## 📋 总体流程（5个步骤）

### Step 1: 生成多轮轨迹数据

**脚本**: `scripts/generate_trajectories.py`  
**调用**: `GENERATE_COLM_DATA_V2.sh` Step 1

**输入**:
- 数据源: `data/seeds/bigcodebench_masked_states.jsonl`
- States数量: 150
- Personas: 全部3个 (Busy-Developer, Experienced-Engineer, Novice-Learner)
- 每个(state, persona)组合: 4个样本
- 最大轮次: 3轮
- LLM模型: `gpt-4o-mini` (用于代码生成)

**处理流程**:

1. **加载States**
   - 从JSONL文件读取states
   - 每个state包含: `query`, `original_instruct_prompt`, `test`, `task_uncertainty`等

2. **对每个(state, persona)组合生成trajectories**
   - 使用heuristic采样策略:
     - 2个强制Execute（第一轮执行）
     - 2个强制Clarify（第一轮澄清）
   - 目的: 确保有足够的多样性，可以生成Turn 1+的pairs

3. **多轮对话生成**
   - Turn 0: 
     - 选择action（根据采样策略或persona逻辑）
     - 生成assistant_msg（使用GPT-4o-mini）
     - 获取user_reaction（通过simulator）
   - Turn 1+: 
     - 根据persona和state选择action（`select_mainline_action_from_persona`）
     - 生成 → 反应
     - 继续直到任务完成或达到max_turns

4. **Action选择逻辑** (`select_mainline_action_from_persona`):
   - 基于`task_uncertainty`和persona的patience阈值
   - **Busy-Developer** (low patience): 
     - Clarify阈值: 0.7
     - 很少Clarify，平均~1.16轮
   - **Experienced-Engineer** (mid patience):
     - Clarify阈值: 0.5
     - 适度Clarify，平均~1.52轮
   - **Novice-Learner** (high patience):
     - Clarify阈值: 0.3
     - 经常Clarify，平均~1.98轮

**输出**:
- 轨迹文件: `data/logs/traj_colm_3turn_persona_150states_*.jsonl`
- 格式: 每行一个turn的trajectory
- 包含字段: `{state, action, assistant_msg, persona, user_reaction, turn, trajectory_id}`

---

### Step 2: 分析轨迹质量（验证persona差异）

**脚本**: `GENERATE_COLM_DATA_V2.sh` Step 2 (内嵌Python代码)

**检查指标**:
- 每个persona的平均轮次
- 第1轮action分布（Clarify vs Execute）
- 轨迹长度分布（1轮、2轮、3轮）
- 任务完成率

**期望值**:
- Busy-Developer: ~1.16轮, Clarify@T0 ~15%
- Experienced-Engineer: ~1.52轮, Clarify@T0 ~40%
- Novice-Learner: ~1.98轮, Clarify@T0 ~70%

**目的**: 验证persona差异是否符合设计预期

---

### Step 3: 计算Trajectory-level Rewards

**脚本**: `reward/compute_rewards.py`  
**调用**: `GENERATE_COLM_DATA_V2.sh` Step 3

**输入**:
- 轨迹文件: `data/logs/traj_*.jsonl`
- 权重配置:
  - `w_task = 1.0` (任务成功权重)
  - `w_interrupt = 0.15` (中断成本权重)
- 使用trajectory-level奖励（整个对话的最终结果）

**处理流程**:

1. **按trajectory_id分组**
   - 同一对话的所有turns归为一组
   - 确保trajectory-level奖励计算

2. **对每个trajectory计算奖励**
   - **task_score**: 
     - 如果Execute，运行测试用例
     - 使用`score_code_passfail`计算通过率
     - 只有通过所有测试才得1.0
   - **interrupt_cost**: 
     - 基于Clarify次数和用户反应
     - 使用`compute_interrupt_cost_v2`
   - **total_reward**: 
     - `w_task * task_score - w_interrupt * interrupt_cost`

3. **生成preference pairs**
   - 对每个decision point（state + turn）:
     - 收集所有可能的actions及其rewards
     - 选择reward最高的作为`chosen_action`
     - 选择reward较低的作为`rejected_action`
     - 生成preference pair

**输出**:
- Preference文件: `data/dpo/traj_*_prefs.jsonl`
- 格式: 每行一个preference pair
- 包含字段: `{state, chosen_action, rejected_action, chosen_reward, rejected_reward, chosen_task_score, rejected_task_score, ...}`

---

### Step 4: 分析Preference数据质量

**脚本**: `GENERATE_COLM_DATA_V2.sh` Step 4 (内嵌Python代码)

**检查指标**:
- 总preference pairs数量
- 测试用例覆盖率（是否所有prefs都有测试用例）
- Chosen/Rejected action分布
- Reward margin（chosen_reward - rejected_reward）
- Persona分布

**目的**: 确保数据质量符合训练要求

---

### Step 5: 分割训练集和测试集

**脚本**: `GENERATE_COLM_DATA_V2.sh` Step 5 (内嵌Python代码)

**处理**:
- 按`state_id`分组（确保同一state的所有prefs在同一集合中）
- 按20%比例分割测试集
- 确保训练集和测试集不重叠（避免数据泄漏）

**输出**:
- 训练集: `data/dpo/traj_*_train_prefs.jsonl` (~120 states)
- 测试集: `data/dpo/traj_*_test_prefs.jsonl` (~30 states)

---

## 🔑 关键设计要点

### 1. Persona差异设计
- **实现方式**: 通过`task_uncertainty`阈值和persona `patience`实现
- **效果**: 不同persona有不同的Clarify倾向
- **验证**: 体现在平均轮次和action分布上

### 2. 采样策略（heuristic）
- **设计**: 每个(state, persona)生成4个样本
- **分布**: 2×Execute + 2×Clarify
- **目的**: 确保有足够的多样性，可以生成Turn 1+的pairs用于训练

### 3. Trajectory-level奖励
- **设计**: 整个对话的最终结果决定奖励
- **优势**: 
  - 避免中间turn的噪声影响
  - 更符合实际任务完成情况
  - 奖励信号更稳定

### 4. 数据质量保证
- **数据源**: 使用`bigcodebench_masked_states.jsonl`（包含测试用例）
- **测试用例**: 确保所有prefs都有测试用例用于计算`task_score`
- **数据分割**: 按state分割避免数据泄漏

### 5. 两阶段代码生成设计（方案2）

**核心思想**: 同时生成两个版本的代码，用于不同的目的。

#### 5.1 代码生成策略

- **assistant_code** (实际代码):
  - 使用 `masked query + 澄清问题`（真实场景）
  - 用于DPO训练（通过`assistant_msg`字段）
  - 模拟真实的信息获取过程
  - 体现澄清的价值：澄清后代码质量提升

- **teacher_code** (理想代码):
  - 使用 `full query`（完整信息）
  - 用于分析和对比实验
  - 提供高质量的参考代码
  - 不会破坏因果解释（因为不用于训练）

#### 5.2 数据格式

每个Execute的trajectory包含：
```json
{
  "assistant_msg": "Execute\n{assistant_code}",  // 用于DPO训练
  "assistant_code": "...",  // 实际代码（masked query + 澄清问题）
  "teacher_code": "...",    // 理想代码（full query，可选）
  ...
}
```

#### 5.3 研究价值

**对比实验**: 比较 `success(masked execution)` vs `success(teacher execution)`

- **实验设计**:
  - 对每个Execute动作，同时评估`assistant_code`和`teacher_code`的成功率
  - 计算两个版本的`task_score`
  - 对比分析：`success(teacher_code)` vs `success(assistant_code)`

- **研究假设**:
  - 如果 `success(teacher_code) >> success(assistant_code)`
  - 可以证明 **clarification matters**（澄清的价值）
  - 说明通过澄清获取信息确实能提高代码质量

- **Paper价值**:
  - 这是reviewer非常喜欢的结果
  - 提供了强有力的证据证明澄清的价值
  - 避免了privileged baseline问题（teacher_code不用于训练）

#### 5.4 实现细节

- **生成时机**: Execute动作时同时生成两个版本
- **成本**: 需要两次LLM调用（但提供了重要的研究价值）
- **Pipeline影响**: DPO训练流程完全不变（继续使用`assistant_msg`）
- **分析工具**: 可以使用`teacher_code`进行详细的对比分析

---

## 📁 输入输出文件

### 输入
- `data/seeds/bigcodebench_masked_states.jsonl`
  - 包含: `query`, `original_instruct_prompt`, `test`, `task_uncertainty`等

### 中间文件
- `data/logs/traj_colm_3turn_persona_150states_*.jsonl`
  - 轨迹数据（多轮对话）
  - 格式: 每行一个turn的trajectory

### 输出
- `data/dpo/traj_*_prefs.jsonl` (完整preference pairs)
- `data/dpo/traj_*_train_prefs.jsonl` (训练集)
- `data/dpo/traj_*_test_prefs.jsonl` (测试集)

---

## ⚙️ 关键参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_STATES` | 150 | 总states数（120 train + 30 test） |
| `TEST_RATIO` | 0.2 | 20%作为测试集 |
| `MAX_TURNS` | 3 | 最大轮次 |
| `LLM_MODEL` | `gpt-4o-mini` | 代码生成模型 |
| `SEED` | 42 | 随机种子（可复现） |
| `n_samples` | 4 | 每个(state, persona)组合的样本数 |
| `sampling_strategy` | `heuristic` | 采样策略（2×Execute + 2×Clarify） |
| `w_task` | 1.0 | 任务成功权重 |
| `w_interrupt` | 0.15 | 中断成本权重 |

---

## 🔄 数据流图

```
bigcodebench_masked_states.jsonl
    ↓
[Step 1] generate_trajectories.py
    ↓ (生成多轮对话)
traj_*.jsonl (轨迹数据)
    ↓
[Step 2] 轨迹质量分析
    ↓ (验证persona差异)
[Step 3] compute_rewards.py
    ↓ (计算trajectory-level奖励)
traj_*_prefs.jsonl (preference pairs)
    ↓
[Step 4] Preference质量分析
    ↓ (验证数据质量)
[Step 5] 数据分割
    ↓
traj_*_train_prefs.jsonl (训练集)
traj_*_test_prefs.jsonl (测试集)
```

---

## 📝 使用方式

### 运行完整流程
```bash
bash GENERATE_COLM_DATA_V2.sh
```

### 单独运行某个步骤
```bash
# Step 1: 生成轨迹
python scripts/generate_trajectories.py \
  --mode dataset \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --domain coding \
  --n_states 150 \
  --all_personas \
  --n_samples 4 \
  --sampling_strategy heuristic \
  --max_turns 3 \
  --llm_model gpt-4o-mini \
  --out logs/traj_*.jsonl \
  --seed 42

# Step 3: 计算奖励
python reward/compute_rewards.py \
  --trajectories data/logs/traj_*.jsonl \
  --out data/dpo/traj_*_prefs.jsonl \
  --w_task 1.0 \
  --w_interrupt 0.15 \
  --use_trajectory_level \
  --target_execute_ratio 0.7 \
  --rebalance_seed 42
```

---

## ⚠️ 注意事项

1. **数据源**: 确保使用`bigcodebench_masked_states.jsonl`（包含测试用例）
2. **API Key**: 需要设置`OPENAI_API_KEY`环境变量
3. **数据泄漏**: 按state分割确保训练集和测试集不重叠
4. **代码生成**: 使用GPT-4o-mini进行知识蒸馏，但action选择基于masked query
5. **Persona差异**: 通过调整`task_uncertainty`阈值和persona patience实现

---

## 🔍 质量检查点

1. **轨迹质量**: 检查persona差异是否符合预期
2. **测试用例**: 确保所有prefs都有测试用例
3. **Reward分布**: 检查reward margin是否合理
4. **数据分割**: 验证训练集和测试集不重叠
5. **Persona分布**: 确保三个persona都有足够的样本
