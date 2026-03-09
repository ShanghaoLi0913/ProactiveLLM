# 评估指标设计文档

## 一、代码生成质量指标

### 1. Task Success Rate (已有)
- **定义**: `task_score >= 1.0` 的比例
- **含义**: 完全通过所有测试用例的比例
- **适用场景**: 主要评估指标，符合BigCodeBench等标准基准

### 2. Soft Task Success Rate (已有)
- **定义**: `task_score >= 0.5` 的比例
- **含义**: 至少通过50%测试用例的比例
- **适用场景**: 评估部分成功的任务

### 3. Average Task Score / Pass Rate (已有)
- **定义**: 所有Execute动作的平均`task_score`
- **含义**: 平均测试通过率
- **适用场景**: 评估代码质量的整体水平

### 4. **Pass@K** (新增)
- **定义**: 生成K个代码候选，至少有一个通过所有测试的比例
- **计算**: 
  ```
  pass@k = (至少1个候选通过的任务数) / 总任务数
  ```
- **适用场景**: 评估模型在多次尝试下的成功率（容错性）
- **实现**: 对每个任务生成K个候选代码，分别测试

### 5. **First Attempt Success Rate** (新增)
- **定义**: 第一次Execute就成功的比例
- **计算**: 
  ```
  first_attempt_success = (第一次Execute就task_score>=1.0的任务数) / 总任务数
  ```
- **含义**: 评估模型在信息不完整时的代码生成能力
- **适用场景**: 评估是否需要多轮澄清

### 6. **Code Quality Metrics** (可选)
- **代码可读性**: 代码长度、注释率、命名规范性
- **代码复杂度**: 圈复杂度、嵌套深度
- **实现难度**: 需要额外的代码分析工具

---

## 二、用户体验指标

### 1. **Time to First Code (TTC)** (新增)
- **定义**: 从对话开始到第一次生成代码的轮次
- **计算**: 
  ```
  TTC = 第一个Execute动作出现的轮次
  ```
- **含义**: 用户等待代码的时间，越小越好
- **适用场景**: 评估对话效率，特别是Busy-Developer persona

### 2. **Efficiency Score** (新增)
- **定义**: 成功完成任务的平均轮次
- **计算**: 
  ```
  efficiency = (所有成功任务的轮次总和) / 成功任务数
  ```
- **含义**: 评估完成任务所需的对话轮次，越小越好
- **适用场景**: 评估对话效率

### 3. **Over-clarification Rate** (新增)
- **定义**: 过度澄清导致失败的比例
- **计算**: 
  ```
  over_clarify = (有多次Clarify但最终task_score<1.0的任务数) / 总任务数
  ```
- **含义**: 评估是否在不必要的时候过度澄清
- **适用场景**: 评估Clarify策略的合理性

### 4. **Under-clarification Rate** (新增)
- **定义**: 澄清不足导致失败的比例
- **计算**: 
  ```
  under_clarify = (没有Clarify直接Execute但task_score<1.0的任务数) / 总任务数
  ```
- **含义**: 评估是否在应该澄清时直接执行
- **适用场景**: 评估Execute策略的合理性

### 5. **Clarification Quality** (新增)
- **定义**: 澄清问题的质量评分
- **评估维度**:
  - **相关性**: 澄清问题是否与任务相关
  - **有效性**: 澄清后是否提高了成功率
  - **清晰度**: 问题是否清晰易懂
- **计算**: 
  ```
  clarification_quality = (澄清后成功率提升) / (澄清次数)
  ```
- **实现难度**: 需要NLP分析或人工评估

### 6. **User Interruption Rate** (已有，在reward中)
- **定义**: 用户中断对话的比例
- **含义**: 评估用户对模型行为的不满意度
- **适用场景**: 评估用户体验

### 7. **Conversation Satisfaction Score** (新增，综合指标)
- **定义**: 综合满意度评分
- **计算公式**: 
  ```
  satisfaction = w1 * task_success_rate 
               + w2 * (1 - avg_turns / max_turns)
               + w3 * (1 - over_clarify_rate)
               + w4 * (1 - under_clarify_rate)
               - w5 * user_interruption_rate
  ```
- **权重建议**: 
  - w1 = 0.4 (任务成功最重要)
  - w2 = 0.2 (效率)
  - w3 = 0.15 (避免过度澄清)
  - w4 = 0.15 (避免澄清不足)
  - w5 = 0.1 (避免用户中断)

---

## 三、Persona一致性指标

### 1. **Persona Alignment Score** (新增)
- **定义**: 模型行为是否符合persona特征
- **评估维度**:
  - **Busy-Developer**: 应该更倾向于Execute，轮次更少
  - **Experienced-Engineer**: 平衡Clarify和Execute
  - **Novice-Learner**: 应该更倾向于Clarify，轮次可能更多
- **计算**: 
  ```
  alignment = 1 - |actual_clarify_rate - expected_clarify_rate|
  ```
- **适用场景**: 评估模型是否学会了persona-aware行为

### 2. **Action Selection Consistency** (新增)
- **定义**: 相同情况下动作选择的一致性
- **计算**: 对相同state和persona，多次运行的动作选择一致性
- **适用场景**: 评估模型的稳定性

### 3. **Clarify Rate by Persona** (已有)
- **定义**: 不同persona的Clarify率
- **含义**: 评估模型是否根据persona调整行为

---

## 四、对话质量指标

### 1. **Average Turns per Conversation** (已有)
- **定义**: 平均对话轮次
- **含义**: 评估对话长度，需要平衡效率和成功率

### 2. **Multi-turn Clarify Count** (已有)
- **定义**: 有多轮Clarify的对话数
- **含义**: 评估是否需要多轮澄清

### 3. **Dialogue Efficiency** (新增)
- **定义**: 对话效率 = 任务成功率 / 平均轮次
- **计算**: 
  ```
  efficiency = task_success_rate / avg_turns
  ```
- **含义**: 评估在最少轮次内完成任务的能力

---

## 五、实现优先级

### 高优先级（立即实现）
1. ✅ Task Success Rate (已有)
2. ✅ Average Task Score (已有)
3. ✅ Clarify Rate (已有)
4. ✅ Average Turns (已有)
5. 🔲 **Time to First Code (TTC)** - 简单，重要
6. 🔲 **Efficiency Score** - 简单，重要
7. 🔲 **Over-clarification Rate** - 中等，重要
8. 🔲 **Under-clarification Rate** - 中等，重要

### 中优先级（后续实现）
9. 🔲 **Pass@K** - 需要修改代码生成逻辑
10. 🔲 **First Attempt Success Rate** - 简单
11. 🔲 **Persona Alignment Score** - 需要定义期望值
12. 🔲 **Dialogue Efficiency** - 简单

### 低优先级（可选）
13. 🔲 **Clarification Quality** - 需要NLP分析
14. 🔲 **Code Quality Metrics** - 需要额外工具
15. 🔲 **Conversation Satisfaction Score** - 综合指标，需要权重调优

---

## 六、对比实验：证明Clarification的价值

### 实验设计：Masked Execution vs Teacher Execution

**核心思想**: 通过对比实验证明澄清（clarification）的价值。

#### 6.1 实验设置

- **数据来源**: 轨迹数据中的`assistant_code`和`teacher_code`字段
- **assistant_code**: 使用`masked query + 澄清问题`生成的代码（实际场景）
- **teacher_code**: 使用`full query`生成的代码（理想目标）

#### 6.2 评估指标

对每个Execute动作，同时评估两个版本的代码：

1. **Success Rate (Masked)**: `assistant_code`的成功率
   ```
   success_rate_masked = (assistant_code通过所有测试的任务数) / 总任务数
   ```

2. **Success Rate (Teacher)**: `teacher_code`的成功率
   ```
   success_rate_teacher = (teacher_code通过所有测试的任务数) / 总任务数
   ```

3. **Gap Analysis**: 两个版本的差距
   ```
   gap = success_rate_teacher - success_rate_masked
   ```

#### 6.3 研究假设

**如果 `success_rate_teacher >> success_rate_masked`**:
- ✅ 可以证明 **clarification matters**（澄清的价值）
- ✅ 说明通过澄清获取信息确实能提高代码质量
- ✅ 提供了强有力的证据支持我们的研究假设

**如果差距很小**:
- 可能说明澄清获取的信息不够关键
- 或者masked query已经包含了足够的信息
- 需要进一步分析哪些类型的澄清最有效

#### 6.4 分析维度

1. **整体对比**: 所有任务的总体成功率对比
2. **按Persona对比**: 不同persona下的成功率对比
3. **按澄清次数对比**: 有澄清 vs 无澄清的差距
4. **按澄清类型对比**: 不同澄清类型（edge_cases, output_format, constraints）的效果

#### 6.5 Paper价值

- **Reviewer友好**: 这是reviewer非常喜欢的结果
- **因果解释**: 避免了privileged baseline问题（teacher_code不用于训练）
- **证据强度**: 提供了强有力的证据证明澄清的价值
- **可复现性**: 实验设计清晰，结果可复现

#### 6.6 实现建议

```python
# 评估脚本示例
def compare_masked_vs_teacher(trajectories):
    masked_success = 0
    teacher_success = 0
    total = 0
    
    for traj in trajectories:
        if traj["action"] == "Execute":
            total += 1
            # 评估assistant_code
            masked_score = evaluate_code(traj["assistant_code"], traj["state"]["test"])
            if masked_score >= 1.0:
                masked_success += 1
            
            # 评估teacher_code（如果存在）
            if traj.get("teacher_code"):
                teacher_score = evaluate_code(traj["teacher_code"], traj["state"]["test"])
                if teacher_score >= 1.0:
                    teacher_success += 1
    
    return {
        "success_rate_masked": masked_success / total,
        "success_rate_teacher": teacher_success / total,
        "gap": (teacher_success - masked_success) / total
    }
```

---

## 七、指标对比表

| 指标 | 类型 | 重要性 | 实现难度 | 适用场景 |
|------|------|--------|----------|----------|
| Task Success Rate | 代码质量 | ⭐⭐⭐⭐⭐ | 简单 | 主要评估指标 |
| **Masked vs Teacher Success** | **代码质量** | **⭐⭐⭐⭐⭐** | **简单** | **证明clarification价值** |
| Pass@K | 代码质量 | ⭐⭐⭐⭐ | 中等 | 容错性评估 |
| Time to First Code | 用户体验 | ⭐⭐⭐⭐ | 简单 | 效率评估 |
| Efficiency Score | 用户体验 | ⭐⭐⭐⭐ | 简单 | 效率评估 |
| Over-clarification | 用户体验 | ⭐⭐⭐ | 中等 | 策略评估 |
| Under-clarification | 用户体验 | ⭐⭐⭐ | 中等 | 策略评估 |
| Persona Alignment | Persona一致性 | ⭐⭐⭐⭐ | 中等 | Persona评估 |
| Dialogue Efficiency | 综合 | ⭐⭐⭐ | 简单 | 综合评估 |
