# TactfulLLM 论文实验设计

> Target: NeurIPS 2026 (Deadline: 2026-05-06)
> Last updated: 2026-04-13

---

## Overview

| Experiment | Core Question | New Eval Needed? |
|---|---|---|
| Exp 1: Main Performance | TactfulLLM vs baselines, 谁的 trade-off 最好？ | Baselines 需跑 |
| Exp 2: Recovery Analysis | Clarification 恢复了多少被 mask 掉的信息？ | Full Query baseline 需跑 |
| Exp 3: Persona Sensitivity | 模型是否真正理解 persona 并据此调整行为？ | Cross-persona swap 需跑 |

---

## Shared Setup

### Backbones
- Llama-3.1-8B-Instruct (primary)
- Qwen2.5-7B-Instruct (secondary, if time permits)

### Test Set
- 200 BigCodeBench states (50 existing + 150 extra)
- 与训练 109 states 零重叠
- 每个 state 评估 3 personas = 600 组对话

### User Simulator
- gpt-4o-mini (与训练轨迹一致)

### Code Evaluation
- BigCodeBench test cases, pass@1 / pass@5

---

## Experiment 1: Main Task Performance

### Research Question
TactfulLLM 能否在 task success 和 user interruption 之间取得比 baselines 更好的 trade-off？

### Methods

| Method | Description | Training | Clarify Strategy |
|---|---|---|---|
| **TactfulLLM-DPO** (ours) | Persona-aware DPO | 500 pairs, QLoRA | Learned: persona-dependent |
| **Direct Execution** | Masked query 直接生成代码 | None | Never clarify |
| **Prompt-only** | System prompt 描述 persona，指示何时 clarify | None | Prompted |
| **Always-Clarify** | 固定先问 K 轮再执行 | None | Fixed K turns |
| **Base LLM** | Base Llama, no LoRA, no persona instruction | None | Model default |

### Metrics

**Primary:**
- **pass@1** — 首次代码生成通过率 (task success)
- **Avg Clarification Turns** — 平均澄清轮数 (interruption cost)

**Secondary:**
- **pass@5** — top-5 代码通过率 (code quality ceiling)
- **Clarify Rate** — clarify actions / total actions

**Composite (三种方案都算，选效果最好的):**
- **方案 A: Utility Score** — U = pass@1 - lambda * avg_turns, 画 lambda-sensitivity 曲线
- **方案 B: Pareto Plot** — x=avg_turns, y=pass@1, 展示 Pareto frontier
- **方案 C: Reward-based Utility** — 复用训练 reward 公式 (放 appendix)

### Main Table (Overall, 不按 persona 拆)

| Method | pass@1 | pass@5 | Avg Turns | Clarify% | Utility |
|---|:---:|:---:|:---:|:---:|:---:|
| TactfulLLM-DPO (ours) | | | | | |
| Direct Execution | | | | | |
| Prompt-only | | | | | |
| Always-Clarify (K=2) | | | | | |
| Base LLM | | | | | |

### Figures

**Figure 1: Pareto Plot**
- x=avg_turns, y=pass@1, 每个 method 一个点
- 展示谁在 Pareto frontier 上

**Figure 2: Lambda-Sensitivity Curve**
- x=lambda, y=utility, 每个 method 一条线
- 展示在大多数合理 lambda 下 DPO 都赢

**Figure 3: Success vs Turns Budget Curve (Efficiency Curve)**
- x = turn budget (1, 2, 3, 4, 5, 6), y = pass@1
- 每个 method 一条线
- 含义：如果只允许最多 K 轮对话，各方法的 pass@1 是多少
- 做法：对每个 method，截断到 turn budget K（超过 K 轮的对话在第 K 轮强制 Execute），统计 pass@1
- 讲的故事：
  - Direct Execution: 水平线（不 clarify，turns 无关）
  - Always-Clarify: 在 K=固定轮 处突然跳升
  - TactfulLLM: 曲线更平滑，early turns 就有收益（因为会根据 persona 选择最佳 clarify 时机）
  - "我们不是靠多问赢的，是靠在正确时机问赢的"
- 数据来源：从 Exp 1 评估结果中按 turn 截断重新计算，无需额外跑实验

### Statistical Tests

对 Exp 1 主表中所有 method 两两对比：
- **Fisher's exact test** on pass@1 (200 states, binary pass/fail)
- **Bootstrap confidence interval** (1000 resamples) for pass@1 差异
- 主表加星号标注 significance: * p<0.05, ** p<0.01, *** p<0.001
- Power analysis 已确认 200 states 可达 p~0.001 (见 v29_experiment_log.md §13.1)

### GPU Cost Estimate
- Direct Execution: ~6h (每 state 1 轮 Execute)
- Prompt-only: ~20h (类似 DPO 评估)
- Always-Clarify: ~20h
- Base LLM: ~22h (50-state 已完成, 150-extra 待跑)

---

## Experiment 2: Information Recovery Analysis

### Research Question
Clarification 能恢复多少因 masking 丢失的信息？恢复量与 clarification 轮次的关系？

### Design

固定代码生成器为 **Base Llama**（控制变量，唯一变化是输入信息量）。

| Condition | Input | 说明 |
|---|---|---|
| Full Query | 完整 instruct_prompt | 上界：没有信息缺失 |
| Masked + Ideal Disclosed | masked query + 被 mask 原文完美还原 | 理论最优恢复 |
| Masked + Clarified (Novice) | masked query + 6 轮 Clarify 获得的信息 | 多轮实际恢复 |
| Masked + Clarified (Experienced) | masked query + 1-2 轮 Clarify 获得的信息 | 少轮实际恢复 |
| Masked + Clarified (Busy) | masked query + 0 轮 Clarify | = Masked Direct |
| Masked Direct | masked query, 不 Clarify | 下界：零信息恢复 |

### Metrics

- **pass@1** — 每个条件下的代码通过率
- **Delta vs Full** — 与 Full Query 的差距
- **Recovery Rate** — (Clarified - Direct) / (Full - Direct) * 100%

### Main Table

| Condition | pass@1 | Delta vs Full | Recovery Rate |
|---|:---:|:---:|:---:|
| Full Query (upper bound) | | — | 100% |
| Masked + Ideal Disclosed | | | |
| Masked + Clarified (Novice, ~6 turns) | | | |
| Masked + Clarified (Experienced, ~2 turns) | | | |
| Masked + Clarified (Busy, 0 turns) | | | |
| Masked Direct (lower bound) | | — | 0% |

### Data Source
- Masked + Clarified 各 persona: 从 Exp 1 的 DPO 评估结果提取（无需额外跑）
- Masked Direct: 从 Exp 1 的 Direct Execution baseline 提取（无需额外跑）
- **Full Query: 需额外跑** — Base Llama + 200 unmasked states
- Masked + Ideal Disclosed: 需额外跑或从 v29 Layer 2 数据推算

### Cross-Experiment Connection (Exp 2 <-> Exp 1)

将 Recovery Rate 与 Exp 1 的 downstream task success 关联分析：
- 画 scatter plot: x = Recovery Rate, y = pass@1 improvement over Direct
- 每个 persona 一个点（或每个 state 一个点）
- 叙事: "We further correlate recovery rate with downstream task success to understand when additional interaction is beneficial."
- 期望发现: Recovery Rate 与 pass@1 improvement 正相关，但存在边际递减（后几轮 clarify 恢复信息多但 pass@1 提升小）
- 这给 reviewer 展示 cross-experiment reasoning，而非三个孤立实验

### GPU Cost Estimate
- Full Query baseline: ~6h (每 state 1 轮 Execute, no clarify)
- Masked + Ideal Disclosed: ~6h (同上)
- 其他条件: 复用 Exp 1 数据, 0h

---

## ~~Experiment 3: Persona Sensitivity~~ (降级，融入 Exp 1)

> 2026-04-16 决定：Part A 行为分析融入 Exp 1 正文（per-persona 数据已在 Table 1），
> Part B Cross-Persona Swap 砍掉（42h GPU 成本高、结果可预测、加分有限）。
> 省下的时间用于 Qwen backbone 和 Ablation Study。

### Ablation Study (替代 Exp 3)

精简 ablation，只用 Llama backbone，验证两个核心 design choices：

| Variant | 改了什么 | 训练成本 | 评估成本 |
|---|---|:---:|:---:|
| TactfulLLM (full, v30) | — | done | 进行中 |
| w/o disclosure-aware (v29 pairs) | 固定轮数规则，无 disclosure 感知 | 0 (已有模型) | 0 (已有结果) |
| w/o behavior-first | pairs 按 reward 排序，不按 persona 设计 | 17min | ~7h |

v29 vs v30 对比免费（两边数据都有），再加一个 reward-based ablation 只需 ~7h。

---

## Timeline (updated 2026-04-16)

| Date | Task | Status |
|---|---|---|
| 4/10 - 4/12 | v29: 轨迹生成 + DPO 训练 + 50-state 评估 | ✅ |
| 4/12 - 4/14 | 200-state 扩大评估 (DPO + Base) | ✅ |
| 4/14 - 4/15 | Exp 1 baselines: Prompt-only, Direct, Clarify-first | ✅ |
| 4/15 | Exp 2: Oracle + Ideal Disclosed 实现 | 🏃 Oracle 运行中 |
| **4/16** | **v30: disclosure-aware pairs + 训练** | **✅ 评估中** |
| 4/17 | v30 50-state 结果分析 | Pending |
| 4/17 - 4/18 | v30 200-state 评估 (如 50-state 结果好) | Pending |
| 4/18 - 4/19 | Exp 2: Full Query + Ideal Disclosed 评估 | Pending |
| 4/19 - 4/20 | Ablation: w/o behavior-first | Pending |
| 4/20 - 4/25 | Qwen backbone (masking → 轨迹 → DPO → 评估) | Pending |
| 4/25 - 5/05 | 论文写作 + 补充实验 | Pending |
| 5/06 | NeurIPS deadline | |

---

## File References

| File | Description |
|---|---|
| `docs/v29_experiment_log.md` | v29 详细实验记录 (masking, training, eval) |
| `docs/work_log.md` | 每日工作记录 |
| `eval/evaluate_multi_turn_persona.py` | 多轮评估脚本 (已加增量写入+断点续跑) |
| `models/v29_100states/` | DPO LoRA adapter |
| `data/seeds/test_states_v29_eval_50.jsonl` | 50-state 测试集 |
| `data/seeds/test_states_v29_eval_150extra.jsonl` | 150-extra 测试集 |
| `outputs/eval_v29_100states_50test.json` | DPO 50-state 评估结果 |
| `outputs/eval_v29_base_llama_50test.json` | Base 50-state 评估结果 |
