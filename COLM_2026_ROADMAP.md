# COLM 2026 论文路线图
## 目标：多轮对话 + 3 Persona + 平衡Task Success与用户偏好

**投稿目标**：COLM 2026（预计4-5月投稿，8月会议）  
**核心贡献**：Persona-Aware Proactive LLM with Multi-turn Trajectory-Level DPO

---

## 📊 当前状态分析

### ✅ 已完成
1. **多轮对话框架**：3轮设计，trajectory-level rewards
2. **Persona系统**：3个persona（Busy/Experienced/Novice）
3. **DPO训练流程**：Persona-aware训练已实现
4. **用户模拟器**：支持patience decay和expertise-based clarity

### ❌ 关键问题
1. **Task Success Rate = 0%**：数据缺少测试用例（`convcodeworld_tests: null`）
2. **Execute率过高（94.7%）**：persona差异不明显
3. **Reward不平衡**：Task success和interrupt cost权重需要调优
4. **评估不一致**：两个评估脚本结果差异大（94.7% vs 45%）

---

## 🎯 论文核心指标（必须达到）

### 1. Persona差异（关键创新点）
- **Busy-Developer**: Execute率 >80%, 平均轮次 <1.3
- **Experienced-Engineer**: Execute率 50-70%, 平均轮次 1.4-1.6
- **Novice-Learner**: Execute率 <50%, 平均轮次 >1.7

### 2. Task Success Rate（必须 >0）
- **Execute动作**: Task success rate >60%
- **Overall**: Task success rate >50%
- **Persona差异**: Novice通过Clarify获得更高success rate

### 3. 平衡性指标
- **Reward分布**: 不同persona的reward有明显差异
- **Action Accuracy**: >70%（模型预测与chosen action匹配度）

---

## 📅 分阶段计划（8周）

### **Phase 1: 数据修复与验证（Week 1-2）**

#### Week 1: 修复数据源问题
**目标**：确保所有数据包含测试用例

**任务**：
1. ✅ 修复生成脚本，使用`bigcodebench_masked_states.jsonl`（有测试用例）
2. ✅ 重新生成100 states的trajectories（包含测试用例）
3. ✅ 重新计算rewards和preference pairs
4. ✅ 验证prefs中所有state都有`convcodeworld_tests`

**验证标准**：
- [ ] 100%的prefs包含测试用例
- [ ] 可以成功计算task_score

**预计时间**：2-3天

#### Week 2: 统一评估流程
**目标**：确保评估结果一致且准确

**任务**：
1. ✅ 合并评估脚本（`evaluate_v17_persona_aware.py` + `evaluate_dpo_model.py`）
2. ✅ 统一action选择逻辑
3. ✅ 统一代码提取和测试执行逻辑
4. ✅ 创建综合评估脚本：`evaluate_comprehensive.py`

**验证标准**：
- [ ] 两个评估脚本结果一致（Execute率差异<5%）
- [ ] Task success rate可以正确计算
- [ ] Persona差异可以清晰展示

**预计时间**：3-4天

---

### **Phase 2: Reward优化（Week 3-4）**

#### Week 3: Reward公式调优
**目标**：平衡task success和interrupt cost，实现persona差异化

**当前公式**：
```
R = w_task * R_task - w_interrupt * C_interrupt
```

**优化方向**：
1. **Persona-aware权重**：
   - Busy: w_interrupt更高（惩罚多轮对话）
   - Novice: w_interrupt更低（允许更多clarify）
   
2. **Task success奖励**：
   - 通过Clarify获得edge_cases_info后Execute：+0.2 bonus
   - 直接Execute但失败：-0.1 penalty
   
3. **Trajectory-level优化**：
   - 早期Clarify的价值通过最终success体现
   - 确保reward信号足够强（差异>0.1）

**任务**：
1. ✅ 实现persona-aware reward weights
2. ✅ 添加edge_cases_info bonus机制
3. ✅ 调整w_interrupt范围（0.1-0.5）
4. ✅ 验证reward分布：不同persona有明显差异

**验证标准**：
- [ ] Busy的reward分布偏向Execute
- [ ] Novice的reward分布偏向Clarify
- [ ] Reward差异足够大（>0.15）

**预计时间**：4-5天

#### Week 4: 数据质量提升
**目标**：生成高质量训练数据（500-1000 states）

**任务**：
1. ✅ 扩展到500 states（从bigcodebench）
2. ✅ 确保persona分布均衡
3. ✅ 确保task_uncertainty分布合理（低/中/高）
4. ✅ 分析preference pairs质量：
   - Reward差异分布
   - Action分布
   - Persona分布

**验证标准**：
- [ ] 500+ preference pairs
- [ ] Persona分布均衡（每个~33%）
- [ ] Execute/Clarify分布合理（~60/40或70/30）
- [ ] Reward差异平均>0.15

**预计时间**：3-4天（包括API调用时间）

---

### **Phase 3: 模型训练与调优（Week 5-6）**

#### Week 5: V19训练（Persona-Aware + Optimized Rewards）
**目标**：训练第一个完整版本

**配置**：
- Base model: Llama-3.1-8B-Instruct
- Training data: 500 states, ~800-1000 preference pairs
- DPO beta: 0.1
- Learning rate: 5e-5
- Epochs: 3

**任务**：
1. ✅ 训练V19模型
2. ✅ 监控训练loss和reward
3. ✅ 每epoch后评估persona差异

**验证标准**：
- [ ] Training loss收敛
- [ ] Persona差异开始显现（Execute率差异>10%）
- [ ] Task success rate >0

**预计时间**：2-3天（训练）+ 1天（评估）

#### Week 6: 超参数调优
**目标**：优化模型性能

**调优方向**：
1. **DPO beta**: 0.05, 0.1, 0.2
2. **Learning rate**: 3e-5, 5e-5, 8e-5
3. **Reward weights**: w_interrupt范围
4. **Temperature**: 代码生成温度

**任务**：
1. ✅ 网格搜索关键超参数
2. ✅ 训练3-5个候选模型
3. ✅ 选择最佳配置

**验证标准**：
- [ ] Persona差异最大化
- [ ] Task success rate >50%
- [ ] Action accuracy >70%

**预计时间**：4-5天

---

### **Phase 4: 评估与验证（Week 7）**

#### Week 7: 全面评估
**目标**：验证所有论文指标

**评估指标**：
1. **Persona差异**：
   - Execute率（按persona）
   - 平均轮次（按persona）
   - Clarify率（按persona）

2. **Task Success**：
   - Overall task success rate
   - Execute-only task success rate
   - Persona-specific task success rate

3. **平衡性**：
   - Reward分布（按persona）
   - Action accuracy
   - Reward margin

4. **Baseline对比**：
   - Always Execute
   - Always Clarify
   - Zero-shot Llama-3.1-8B

**任务**：
1. ✅ 运行完整评估套件
2. ✅ 生成可视化图表
3. ✅ 统计分析显著性
4. ✅ 案例研究（挑选典型样本）

**验证标准**：
- [ ] 所有核心指标达到论文要求
- [ ] Persona差异显著（p<0.05）
- [ ] Task success rate >50%
- [ ] 明显优于baseline

**预计时间**：3-4天

---

### **Phase 5: 论文撰写（Week 8）**

#### Week 8: 论文初稿
**目标**：完成论文初稿

**论文结构**：
1. **Introduction** (1页)
   - 问题：LLM需要平衡proactivity和task success
   - 贡献：Persona-aware + Multi-turn DPO

2. **Related Work** (1页)
   - Proactive LLM
   - DPO for sequential decision
   - Persona modeling

3. **Method** (2-3页)
   - Problem formulation
   - Persona-aware reward design
   - Trajectory-level DPO
   - User simulator

4. **Experiments** (2-3页)
   - Dataset
   - Baselines
   - Metrics
   - Results（重点：persona差异、task success、平衡性）

5. **Analysis** (1页)
   - Ablation studies
   - Case studies
   - Error analysis

6. **Conclusion** (0.5页)

**任务**：
1. ✅ 撰写论文初稿
2. ✅ 生成所有图表
3. ✅ 完善实验部分
4. ✅ 内部review

**预计时间**：5-7天

---

## 🔧 关键技术细节

### 1. Reward公式（最终版本）

```python
# Persona-aware weights
w_interrupt_persona = {
    "Busy-Developer": 0.4,      # 高惩罚多轮对话
    "Experienced-Engineer": 0.2, # 中等
    "Novice-Learner": 0.1        # 低惩罚，允许clarify
}

# Base reward
R_base = w_task * R_task - w_interrupt_persona[persona] * C_interrupt

# Edge cases bonus
if has_edge_cases_info and R_task > 0:
    R_final = R_base + 0.2
else:
    R_final = R_base
```

### 2. Persona差异实现

**Action选择阈值**（在trajectory generation时）：
- Busy: Clarify if uncertainty > 0.7
- Experienced: Clarify if uncertainty > 0.5
- Novice: Clarify if uncertainty > 0.3

**Reward权重**（在DPO训练时）：
- 通过persona-aware weights实现差异化

### 3. 评估统一化

创建`evaluate_comprehensive.py`，包含：
- Persona-aware评估
- Task success评估
- Action accuracy评估
- Reward分布分析
- 可视化生成

---

## 📈 成功标准（论文接受的最低要求）

### 必须达到（Hard Requirements）
- [ ] Persona差异显著：Execute率差异>15%
- [ ] Task success rate >50%
- [ ] 明显优于baseline（至少2个指标）

### 理想达到（Nice to Have）
- [ ] Task success rate >60%
- [ ] Persona差异Execute率差异>20%
- [ ] Action accuracy >75%
- [ ] 有统计显著性（p<0.01）

---

## 🚨 风险与应对

### 风险1：Persona差异不明显
**应对**：
- 调整reward weights（增大w_interrupt差异）
- 调整action选择阈值
- 增加persona-specific penalty/bonus

### 风险2：Task success rate太低
**应对**：
- 检查代码生成质量
- 调整代码生成temperature
- 增加best-of-N采样

### 风险3：Reward不平衡
**应对**：
- 调整w_task和w_interrupt比例
- 添加edge_cases_info bonus
- 优化trajectory-level reward计算

### 风险4：时间不够
**应对**：
- 优先完成Phase 1-3（核心功能）
- Phase 4可以简化（减少评估指标）
- Phase 5可以并行（边实验边写）

---

## 📝 每周检查点

### Week 1结束
- [ ] 数据源已修复
- [ ] 可以计算task_score

### Week 2结束
- [ ] 评估脚本已统一
- [ ] 评估结果一致

### Week 3结束
- [ ] Reward公式已优化
- [ ] Persona差异开始显现

### Week 4结束
- [ ] 500 states数据已生成
- [ ] Preference pairs质量良好

### Week 5结束
- [ ] V19模型已训练
- [ ] 初步评估通过

### Week 6结束
- [ ] 超参数已优化
- [ ] 最佳模型已选择

### Week 7结束
- [ ] 全面评估完成
- [ ] 所有指标达标

### Week 8结束
- [ ] 论文初稿完成
- [ ] 准备投稿

---

## 🎯 立即行动（本周）

1. **今天**：
   - [ ] 修复`scripts/ops/GENERATE_COLM_DATA_V2.sh`，使用`bigcodebench_masked_states.jsonl`
   - [ ] 重新生成100 states数据（包含测试用例）

2. **明天**：
   - [ ] 重新计算rewards和prefs
   - [ ] 验证数据质量

3. **本周内**：
   - [ ] 统一评估脚本
   - [ ] 完成Phase 1验证

---

**最后更新**：2026-02-12  
**负责人**：[Your Name]  
**状态**：Phase 1进行中
