# ProactiveLLM 开发日志 - 2026年2月10日

## 项目目标
训练一个具有上下文感知主动性校准能力的代码生成LLM，能够根据用户persona和任务不确定度动态决定Clarify还是Execute，目标是发表COLM 2026论文。

---

## 重要尝试和迭代历程

### **V14 - 初始评估与问题发现**
**时间**：2月初
**问题**：
- Action Accuracy = 0%，Task Success Rate = 0%
- 发现多个关键bug

**修复**：
1. ❌ **Action标签不匹配**：评估脚本期望"LOW/MID/HIGH"，但模型训练用"Clarify/Execute"
   - 修复：统一使用"Clarify/Execute"
2. ❌ **数据泄露**：Task score使用ground truth而非模型生成的response
   - 修复：改为使用模型实际生成的代码进行评分
3. ❌ **Embedding size不匹配**：加载PEFT模型时报错
   - 修复：添加`tokenizer.add_special_tokens()`和`model.resize_token_embeddings()`

**结果**：修复后仍有问题，V14未能正常工作

**文档**：
- `V14_EXECUTIVE_SUMMARY.md`
- `outputs/eval_results/V14_CRITICAL_ISSUE_SUMMARY.md`

---

### **V15 - Chat Template集成**
**时间**：2月6日
**关键改进**：集成Llama-3.1-Instruct的chat template

**问题根源**：模型生成的是prompt continuation而非response
- DPO训练时没有明确的prompt/response边界
- 评估时也没有使用chat template

**修复**：
1. **训练侧**（`policy/train_dpo.py`）：
   ```python
   prompt = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True  # 添加<|start_header_id|>assistant<|end_header_id|>
   )
   ```

2. **评估侧**（`eval/evaluate_v13_persona.py`）：
   - 同样使用`apply_chat_template`确保格式一致

**结果**：模型开始正常生成response，但仍需进一步优化

**文档**：
- `V15_CHANGES_AND_FIX.md`
- `V14_vs_V15_COMPARISON.md`
- `V15_FINAL_SUMMARY.md`

---

### **V16 - Action推断逻辑**
**时间**：2月7-8日
**核心洞察**：模型不应该生成"Clarify"/"Execute"标签，而是直接生成代码或问题

**设计哲学**：
```
模型 → 生成自然语言response（代码/问题）
Agent → 推断action（根据response内容）
```

**修复**（`eval/evaluate_v13_persona.py`）：
```python
def extract_action_from_response(response: str) -> str:
    # 如果包含代码块 → Execute
    if '```python' in response or 'def ' in response:
        return "Execute"
    
    # 如果包含问号/澄清词 → Clarify  
    if '?' in response or 'clarify' in response.lower():
        return "Clarify"
    
    return "Execute"  # 默认
```

**结果**：V16达到86.8% action accuracy

**文档**：
- Training/evaluation logs in `outputs/`

---

### **多轮交互探索**
**时间**：2月9-10日

#### **问题分析**
发现现有训练数据存在严重问题：
- 所有trajectory都是单轮的（`dialogue_turn: 0`）
- 缺少`trajectory_id`字段
- Clarify的trajectory-level reward无法正确计算

**数据检查**：
```python
# traj_multiturn_100states_20260205_105256.jsonl
# traj_bigcode_100states_20260206_050454.jsonl
所有条目: dialogue_turn = 0
结论: 没有真正的多轮对话数据
```

#### **技术方案讨论**
1. **完全多轮**（5+轮）：
   - 优势：创新性强，可能发顶会
   - 劣势：实现难度⭐⭐⭐⭐⭐，DPO有distribution shift问题，失败概率60%
   
2. **2轮简化**：
   - 优势：技术可行，成本$2，调试简单
   - 劣势：创新性稍弱，persona差异不够明显
   
3. **3轮折中**（最终选择）：
   - 优势：平衡可行性和创新性，persona差异明显（平均轮次差0.82）
   - 成本：$1.75（500 states），2小时
   - 失败风险：35%（可控）

**预期效果**（3轮设计）：
```
Busy-Developer: 平均1.16轮（3轮占比1%）
Experienced-Engineer: 平均1.52轮（3轮占比12%）
Novice-Learner: 平均1.98轮（3轮占比28%）
```

---

## 核心代码改进

### **1. Action选择逻辑**（`scripts/generate_trajectories.py`）
```python
def select_mainline_action_from_persona(persona, state):
    # Turn 0: 基于persona + task_uncertainty
    if dialogue_turn == 0:
        clarify_threshold = {
            "low": 0.7,    # Busy: 很少Clarify
            "mid": 0.5,    # Experienced: 适度Clarify
            "high": 0.3,   # Novice: 经常Clarify
        }[persona.patience]
        
        return "Clarify" if task_uncertainty > clarify_threshold else "Execute"
    
    # Turn 1: 更高阈值（已经问过一次了）
    elif dialogue_turn == 1:
        clarify_threshold = {"low": 0.9, "mid": 0.75, "high": 0.6}[persona.patience]
        return "Clarify" if task_uncertainty > clarify_threshold else "Execute"
    
    # Turn 2+: 必须Execute
    else:
        return "Execute"
```

### **2. Task Uncertainty更新**（`scripts/generate_trajectories.py`）
```python
# Equation 9: U_{t+1} = U_t * (1 - 0.5 * answer_clarity)
if answered_clarification > 0 and answer_clarity > 0:
    new_uncertainty = current_uncertainty * (1 - 0.5 * answer_clarity)
    new_state["task_uncertainty"] = max(0.0, min(1.0, new_uncertainty))
```

### **3. Chat Template集成**（`policy/train_dpo.py` & `eval/evaluate_v13_persona.py`）
确保训练和评估都使用Llama-3.1-Instruct的标准格式。

### **4. Trajectory-level Reward**（`reward/compute_rewards.py`）
```python
def compute_trajectory_level_rewards(trajectory_turns, cfg):
    # 整个trajectory共享最终的task_score
    final_task_score = trajectory_turns[-1]["task_score"]
    
    # 每一轮累积interrupt_cost
    for turn in trajectory_turns:
        turn["reward"] = final_task_score - accumulated_interrupt_cost
```

---

## 文件组织

### **核心代码文件**
- `policy/train_dpo.py` - DPO训练（已修复chat template）
- `eval/evaluate_v13_persona.py` - 评估脚本（已修复action推断）
- `scripts/generate_trajectories.py` - 轨迹生成（已改进action选择逻辑）
- `reward/compute_rewards.py` - Reward计算（支持trajectory-level）
- `simulator/simulate.py` - 用户模拟器（已修复indentation error）

### **训练/评估脚本**
- `TRAIN_V14.sh`, `TRAIN_V15.sh`, `TRAIN_V16.sh`
- `EVAL_V14.sh`, `EVAL_V15.sh`, `EVAL_V16.sh`

### **文档**
- `COLM_2TURN_DESIGN.md` - COLM 2026论文的3轮设计方案
- `TRAJECTORY_LEVEL_QUICKSTART.md` - Trajectory-level reward快速入门
- `V14_EXECUTIVE_SUMMARY.md` - V14问题总结
- `V15_FINAL_SUMMARY.md` - V15改进总结

### **数据生成脚本**（新）
- `GENERATE_COLM_DATA_V2.sh` - 3轮数据生成脚本
- `DISK_CLEANUP_PLAN.md` - 磁盘清理计划
- `cleanup_disk.sh` - 清理脚本

---

## 当前状态（2026-02-10）

### **已完成✓**
1. ✅ V14-V16迭代，解决了chat template、action推断等关键问题
2. ✅ V16达到86.8% action accuracy
3. ✅ 识别了训练数据的单轮问题
4. ✅ 设计了3轮方案（平衡可行性和创新性）
5. ✅ 改进了action选择逻辑（基于persona + task_uncertainty）
6. ✅ 验证了成本可控（$1.75 for 500 states）

### **下一步（待实施）**
1. ⏭ 生成3轮轨迹数据（100 states测试 → 500 states完整）
2. ⏭ 训练V17模型（基于3轮数据）
3. ⏭ 评估persona差异（验证平均轮次差异）
4. ⏭ 撰写COLM 2026论文

### **技术债务**
1. ⚠️ 需要清理大量临时评估脚本（`scripts/eval_v*.py`）
2. ⚠️ 需要整理outputs目录
3. ⚠️ 需要删除废弃的文档（已删除部分）

---

## 关键学习

### **方法论**
1. **DPO可以用于序列决策**，但需要trajectory-level reward
2. **Chat template很关键**，必须在训练和评估中保持一致
3. **Action推断**比显式生成标签更自然
4. **3轮是sweet spot**：既能展示persona差异，又技术可控

### **调试技巧**
1. 数据泄露很隐蔽（用ground truth而非模型输出）
2. Embedding mismatch需要在加载PEFT前resize
3. 先小规模测试（100 states）再扩展

### **论文策略**
1. COLM比EMNLP稍好投，比NeurIPS容易
2. 3轮设计足够发COLM（预计70%接受概率）
3. Limitation中诚实讨论单步DPO vs 完整RL

---

## 成本统计

### **已花费**
- V14-V16训练：~$30（GPU租用）
- 数据生成测试：~$2
- 总计：~$32

### **预计**
- 3轮数据生成（500 states）：$1.75
- V17训练：$10
- 论文撰写阶段额外实验：$5
- 总预算：~$50

---

## 致谢
感谢在调试过程中发现的每一个bug，它们让我们对DPO、chat template、trajectory-level reward有了更深的理解。

---

**版本**：Pre-V17（准备生成3轮数据）
**日期**：2026-02-10
**状态**：设计完成，待实施
