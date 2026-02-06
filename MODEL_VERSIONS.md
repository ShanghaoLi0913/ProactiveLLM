# DPO模型版本记录

## 版本总览

| 版本 | 训练数据 | 数据量 | Execute | Clarify | TSR(Test) | 说明 | 状态 |
|------|----------|--------|---------|---------|-----------|------|------|
| V1 | 50 states全部 | ~50对 | 100% | 0% | N/A | 基线版本 | 已删除 |
| V2 | 100 states全部 | ~100对 | ~73% | ~27% | N/A | 增加数据但未筛选 | 已删除 |
| V3 | 100 states筛选 | 304对 | 100% | 0% | 25.68% | 允许部分通过(score>0) | ✅保留 |
| V4 | 完美+修复 | 135对 | 100% | 0% | 32.30% | 严格质量控制(score=1.0) | ✅保留 |
| V5A | Execute+Clarify | 686对 | 19.7% | 80.3% | 待测试 | 包含所有Clarify | 已生成 |
| V5B | 平衡版本 | 168对 | 80.4% | 19.6% | 待测试 | **推荐版本** ⭐ | 已生成 |

---

## 详细版本信息

### V1 (Baseline)
**时间**: 2026-02-05  
**数据**: `data/dpo/prefs_50states_*.jsonl` (约50对)  
**策略**: 使用所有50 states的preference pairs，无筛选  
**结果**: TSR ~18%，性能较差  
**模型**: 已删除  
**问题**: 数据质量低，包含大量失败案例

---

### V2 (More Data)
**时间**: 2026-02-05  
**数据**: `data/dpo/prefs_100states_wi0.3_no_rebalance.jsonl` (约100对)  
**策略**: 增加到100 states，但未筛选质量  
**结果**: TSR ~17%，反而下降  
**模型**: 已删除  
**问题**: 数据量增加但质量更低

---

### V3 (Filtered)
**时间**: 2026-02-06  
**数据**: `data/dpo/prefs_bigcode_100_filtered.jsonl` (304对)  
**策略**: 筛选条件 `chosen_task_score > 0`（允许部分通过）  
**数据构成**:
- Execute成功: 70对 (pass_rate = 1.0)
- Execute部分成功: 234对 (0 < pass_rate < 1.0)
- **Clarify: 0对** ⚠️
- 完美数据比例: 23% (70/304)

**评估结果**:
- Test Split TSR: **25.68%**
- Action Accuracy: 50.19%
- Execute Rate: 84.44%
- Persona Discrimination Score: 0.1556

**模型位置**: `outputs/dpo_bigcode_100_filtered/` (4.6G)  
**评估文件**: `outputs/persona_evaluation/v3_eval_results.json`

**关键问题**: 虽然TSR提升，但Clarify样本被完全过滤（因为Clarify单步score=0）

---

### V4 (High Quality + Repaired)
**时间**: 2026-02-06  
**数据**: `data/dpo/prefs_bigcode_100_repaired.jsonl` (135对)  
**策略**: 
1. 严格筛选：只用pass_rate = 1.0的代码
2. 代码修复：对失败的234个代码尝试LLM修复，成功65个

**数据构成**:
- Execute完美成功: 70对 (原始)
- Execute修复成功: 65对 (35%修复成功率)
- **Clarify: 0对** ⚠️
- 完美数据比例: **100%** (135/135)

**评估结果**:
- Test Split TSR: **32.30%** (+25.8% vs V3)
- Action Accuracy: 47.86%
- Execute Rate: **100%** ⚠️
- Persona Discrimination Score: **0.0** ⚠️

**模型位置**: `outputs/dpo_bigcode_repaired/` (4.6G)  
**评估文件**: `outputs/persona_evaluation/v4_eval_results.json`

**核心成就**: 
- ✅ 验证了"质量>数量"：用44%的数据获得125.8%的效果
- ✅ 验证了代码修复策略的有效性

**核心问题**:
- ❌ 完全失去Action多样性（100% Execute）
- ❌ 失去Persona区分能力（PDS=0）
- ❌ 模型学会了"总是Execute"

**根本原因**: 训练数据100%是Execute成功案例，没有任何Clarify样本

---

### V5A (All Clarify Samples)
**时间**: 2026-02-06  
**数据**: `data/dpo/prefs_bigcode_v5_all.jsonl` (686对)  
**策略**: **Trajectory-Level奖励** - 使用多轮对话的最终成功来评估Clarify价值

**核心创新**:
```python
# ❌ 旧方法（单步）
Clarify → task_score = 0 → 被过滤

# ✅ 新方法（多轮）
Clarify → ... → Execute → task_score = 1.0 → Clarify得到正反馈
```

**数据构成**:
- Execute完美成功: 135对 (19.7%)，来自V4
- Clarify后最终成功: 551对 (80.3%)，从多轮轨迹提取
- 总计: 686对

**Clarify样本来源**:
- 分析100个任务的多轮轨迹
- 发现96个任务在Clarify后最终成功
- 588个Clarify turns（占所有turns的49.7%）
- 提取Clarify步骤，使用最终task completion作为奖励

**数据文件**: `data/dpo/prefs_bigcode_v5_all.jsonl`  
**模型**: 未训练  
**适用场景**: 实验性，想要模型更倾向于Clarify

---

### V5B (Balanced - RECOMMENDED ⭐)
**时间**: 2026-02-06  
**数据**: `data/dpo/prefs_bigcode_v5_balanced.jsonl` (168对)  
**策略**: 平衡Execute和Clarify比例（目标20% Clarify）

**数据构成**:
- Execute完美成功: 135对 (80.4%)，来自V4
- Clarify后最终成功: 33对 (19.6%)，精选自V5A
- 总计: 168对
- **完美平衡**: 既保证质量，又有action多样性

**数据质量**:
- Execute样本: 100% pass_rate = 1.0
- Clarify样本: 100%来自最终成功的轨迹
- 对比样本: Execute失败 (task_score = 0)

**数据文件**: `data/dpo/prefs_bigcode_v5_balanced.jsonl`  
**模型**: 未训练  
**推荐原因**:
1. ✅ 保持高质量Execute样本（来自V4）
2. ✅ 引入适量Clarify样本（~20%）
3. ✅ 预期恢复Action多样性和Persona区分能力
4. ✅ 数据量适中，训练效率高

**预期效果**:
- TSR: ~30-35%（保持V4水平）
- Execute Rate: ~75-85%（恢复多样性）
- Action Accuracy: ~55-65%（更准确判断）
- PDS: >0.10（恢复persona区分能力）

---

## 数据生成脚本

### 关键脚本
1. **轨迹生成**: `scripts/generate_trajectories.py`
   - 生成多轮对话轨迹
   - 修复了Execute without break和indentation bugs

2. **Preference Pairs计算**: `reward/compute_rewards.py`
   - 计算chosen/rejected pairs
   - 使用task_score和interrupt_cost

3. **代码修复**: `scripts/repair_all_failed_code.py`
   - 使用GPT-4o-mini修复失败代码
   - 35%修复成功率

4. **V5数据生成**: `scripts/generate_v5_balanced_prefs.py` ⭐
   - **核心创新**: Trajectory-level奖励
   - 从多轮轨迹中提取Clarify样本
   - 生成V5A和V5B两个版本

### 分析脚本
1. `scripts/analyze_clarify_samples.py` - 分析Clarify样本分布
2. `scripts/compare_persona_metrics.py` - 对比V3 vs V4性能
3. `eval/evaluate_persona_metrics.py` - 计算persona相关指标

---

## 评估数据

### Training Split (100 states)
- 轨迹文件: `data/data/logs/traj_bigcode_100states_20260206_050454.jsonl`
- 总turns: 1184
- Action分布: 596 Execute (50.3%) + 588 Clarify (49.7%)

### Test Split (47 states)
- 轨迹文件: `data/data/logs/traj_test_split_47states_20260206_111249.jsonl`
- 评估pairs: `data/dpo/prefs_test_split_all_trajs.jsonl` (257对)
- 用于无数据泄露的真实评估

---

## 关键发现与洞察

### 1. 质量 > 数量 (V3 → V4)
```
V3: 304对（23%完美） → TSR 25.68%
V4: 135对（100%完美）→ TSR 32.30%
```
**结论**: 用44%的数据量，获得了125.8%的效果

### 2. Clarify被过滤的根本原因 (V4分析)
- Clarify action本身不生成代码
- 单步task_score永远是0
- 筛选条件`chosen_task_score > 0`导致所有Clarify被过滤
- 结果: 模型学会"总是Execute"

### 3. Clarify的真正价值 (V5创新)
- **发现**: 96个任务在Clarify后最终成功（成功率96%）
- **洞察**: Clarify的价值在于**多轮对话的最终收益**
- **解决**: 使用trajectory-level的最终task completion作为Clarify的奖励

### 4. 代码修复策略可行 (V4验证)
- 尝试修复234个失败代码
- 成功修复65个（35%成功率）
- 在不泄露ground truth的前提下扩充数据92.8%

### 5. 单一指标优化的风险 (V4教训)
- V4在TSR上显著提升
- 但失去了action多样性和persona区分能力
- **教训**: 需要多目标优化，而不仅仅是TSR

---

## 下一步计划

### 立即任务
1. ✅ 备份当前版本到GitHub
2. 🔄 清理磁盘空间（方案C）
3. ⏳ 训练V5B模型
4. ⏳ 评估V5B vs V4性能

### 后续优化方向
1. **Persona-Conditioned Training**: 在训练时显式加入persona信息
2. **Multi-turn DPO**: 优化整个对话序列而不是单步
3. **Curriculum Learning**: 分阶段训练（完美数据→Clarify→失败案例）
4. **Iterative DPO**: 用V5生成新轨迹，持续迭代

---

## 重要文件位置

### 模型文件
- V3: `outputs/dpo_bigcode_100_filtered/` (4.6G)
- V4: `outputs/dpo_bigcode_repaired/` (4.6G)
- V5A/V5B: 未训练

### 训练数据
- V3: `data/dpo/prefs_bigcode_100_filtered.jsonl` (304对)
- V4: `data/dpo/prefs_bigcode_100_repaired.jsonl` (135对)
- V5A: `data/dpo/prefs_bigcode_v5_all.jsonl` (686对)
- V5B: `data/dpo/prefs_bigcode_v5_balanced.jsonl` (168对) ⭐

### 评估结果
- V3: `outputs/persona_evaluation/v3_eval_results.json`
- V4: `outputs/persona_evaluation/v4_eval_results.json`
- 对比: `outputs/persona_comparison_v3_v4.json`

### 文档
- 完整分析: `outputs/FINAL_ANALYSIS.md`
- V5方案: `outputs/V5_SOLUTION_SUMMARY.md`
- 本文档: `MODEL_VERSIONS.md`

---

**最后更新**: 2026-02-06  
**当前推荐**: V5B (平衡版本) - 待训练 ⭐  
**核心创新**: Trajectory-level奖励，让模型学会"何时Execute、何时Clarify"
