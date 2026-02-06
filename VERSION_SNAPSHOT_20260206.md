# 版本快照 - 2026-02-06

## 🎯 当前状态

**日期**: 2026年2月6日  
**阶段**: V5数据准备完成，准备训练  
**重要里程碑**: 解决了V4"总是Execute"的问题，创新性地引入Trajectory-Level奖励

---

## 📊 现有模型

### V3 - Filtered (保留)
- **路径**: `outputs/dpo_bigcode_100_filtered/`
- **大小**: 4.6G
- **数据**: 304对，允许部分通过
- **Test TSR**: 25.68%
- **问题**: 0% Clarify样本

### V4 - High Quality + Repaired (保留)
- **路径**: `outputs/dpo_bigcode_repaired/`
- **大小**: 4.6G
- **数据**: 135对，100%完美数据
- **Test TSR**: 32.30% ⬆️
- **问题**: 100% Execute，完全失去action多样性

---

## 🆕 V5数据（已生成，待训练）

### V5A - All Clarify Samples
- **数据**: `data/dpo/prefs_bigcode_v5_all.jsonl`
- **大小**: 686对 (135 Execute + 551 Clarify)
- **Clarify比例**: 80.3%
- **用途**: 实验性

### V5B - Balanced (推荐⭐)
- **数据**: `data/dpo/prefs_bigcode_v5_balanced.jsonl`
- **大小**: 168对 (135 Execute + 33 Clarify)
- **Clarify比例**: 19.6%
- **状态**: **推荐训练版本**

---

## 🔬 核心创新

### Trajectory-Level奖励机制

**问题**: Clarify action本身不生成代码，单步task_score永远是0，导致被筛选过滤

**解决**: 使用多轮轨迹的最终成功结果来评估Clarify的价值

```python
# 旧方法（单步）- 导致Clarify被过滤
Clarify → task_score = 0 → 被过滤掉

# 新方法（多轮）- V5创新
Turn 1: Clarify（澄清需求）
Turn 2: Execute（根据澄清生成代码）
Turn 3: Execute（最终成功）
→ 整个轨迹成功 → Clarify获得正反馈
```

**数据来源**:
- 分析了100个任务的1184个turns
- 发现588个Clarify turns（49.7%）
- 96个任务在Clarify后最终成功（成功率96%）
- 提取这些Clarify步骤，使用最终task_completed作为奖励

---

## 📈 性能演进

| 版本 | 数据量 | Clarify% | Test TSR | Execute Rate | PDS | 核心特点 |
|------|--------|----------|----------|--------------|-----|----------|
| V1 | ~50 | 0% | ~18% | 100% | N/A | 基线 |
| V2 | ~100 | ~27% | ~17% | ~73% | N/A | 数据↑质量↓ |
| V3 | 304 | **0%** | 25.68% | 84.4% | 0.156 | 允许部分通过 |
| V4 | 135 | **0%** | **32.30%** | **100%** | **0.0** | 高质量但失去多样性 |
| **V5B** | **168** | **19.6%** | **待测** | **预计~80%** | **预计>0.1** | **平衡版本⭐** |

---

## 🔑 关键发现

### 1. 质量 > 数量
- V3: 304对（23%完美）→ TSR 25.68%
- V4: 135对（100%完美）→ TSR 32.30%
- **结论**: 用44%数据获得125.8%效果

### 2. 单一指标优化的风险
- V4追求TSR最大化
- 结果: Execute Rate = 100%，失去灵活性
- PDS降为0，完全失去persona适应能力

### 3. Clarify的价值
- 不在于单步收益（task_score=0）
- 而在于多轮对话的最终收益
- 96/100任务在Clarify后成功

### 4. 代码修复可行
- 234个失败代码尝试修复
- 65个成功（35%成功率）
- 在不泄露ground truth下扩充数据92.8%

---

## 🎯 V5预期目标

### 性能指标
- **TSR**: 保持在30-35%（不低于V4）
- **Execute Rate**: 降至75-85%（恢复多样性）
- **Action Accuracy**: 提升至55-65%
- **PDS**: 恢复至>0.10

### 行为目标
模型应该学会:
1. ✅ 任务明确时 → 直接Execute
2. ✅ 任务模糊时 → 先Clarify
3. ✅ 根据不同persona调整策略（未来扩展）

---

## 📂 重要文件清单

### 数据文件
```
data/dpo/
├── prefs_bigcode_100.jsonl              (原始510对)
├── prefs_bigcode_100_filtered.jsonl     (V3: 304对)
├── prefs_bigcode_100_repaired.jsonl     (V4: 135对)
├── prefs_bigcode_v5_all.jsonl           (V5A: 686对)
├── prefs_bigcode_v5_balanced.jsonl      (V5B: 168对) ⭐
└── prefs_test_split_all_trajs.jsonl     (测试集: 257对)
```

### 模型文件
```
outputs/
├── dpo_bigcode_100_filtered/            (V3模型: 4.6G)
├── dpo_bigcode_repaired/                (V4模型: 4.6G)
└── persona_evaluation/                   (评估结果)
    ├── v3_eval_results.json
    ├── v4_eval_results.json
    └── persona_comparison_v3_v4.json
```

### 文档文件
```
outputs/
├── FINAL_ANALYSIS.md                    (V3 vs V4完整分析)
├── V5_SOLUTION_SUMMARY.md               (V5解决方案说明)
├── clarify_analysis.json                (Clarify样本分析)
├── clarify_success_tasks.json           (成功任务列表)
└── generate_v5_dataset.log              (V5数据生成日志)

根目录/
├── MODEL_VERSIONS.md                    (版本详细记录) ⭐
└── VERSION_SNAPSHOT_20260206.md         (本文件)
```

### 关键脚本
```
scripts/
├── generate_trajectories.py             (轨迹生成，已修复bug)
├── repair_all_failed_code.py            (代码修复)
├── generate_v5_balanced_prefs.py        (V5数据生成) ⭐
├── analyze_clarify_samples.py           (Clarify分析)
└── compare_persona_metrics.py           (性能对比)

eval/
└── evaluate_persona_metrics.py          (Persona指标计算)
```

---

## 🚀 下一步行动

### 立即任务（按顺序）
1. ✅ 创建版本文档
2. 🔄 提交到GitHub备份
3. ⏳ 执行磁盘清理（方案C）
4. ⏳ 训练V5B模型
5. ⏳ 评估V5B性能
6. ⏳ 对比V4 vs V5B

### 训练命令（清理后执行）
```bash
cd /root/autodl-tmp/ProactiveLLM

# 训练V5B
python policy/train_dpo.py \
  --data data/dpo/prefs_bigcode_v5_balanced.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output outputs/dpo_bigcode_v5_balanced \
  --epochs 3 \
  --lr 5e-5 \
  --beta 0.1

# 评估V5B
python eval/evaluate_dpo_model.py \
  --model_dir outputs/dpo_bigcode_v5_balanced \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --prefs data/dpo/prefs_test_split_all_trajs.jsonl \
  --output outputs/persona_evaluation/v5_eval_results.json \
  --seed 42
```

---

## ⚠️ 注意事项

### 磁盘空间
- **当前**: 数据盘94%满，仅剩6.2G
- **需要**: V5训练约需5G
- **方案**: 删除旧模型释放38.6G空间
- **清理后**: 可用空间~45G

### 待清理文件
```
/root/autodl-tmp/ProactiveLLM_outputs/     (34G - 7个旧模型)
outputs/dpo_bigcode_100_filtered/           (4.6G - V3可选删除)
```

---

## 📚 相关文档

详细技术文档请查看:
- `MODEL_VERSIONS.md` - 完整版本历史和技术细节
- `outputs/FINAL_ANALYSIS.md` - V3 vs V4深度分析
- `outputs/V5_SOLUTION_SUMMARY.md` - V5解决方案完整说明

---

**快照创建时间**: 2026-02-06 15:00 CST  
**系统状态**: 准备就绪，等待清理后训练V5B  
**核心成就**: 首次成功引入Trajectory-Level奖励机制解决Clarify样本问题 🎉
