# Generalization Experiment — Plan

> 2026-04-23. Exp1 主实验之外加一块 generalization 验证，证明 TactfulLLM 学到的是 transferable skill 而非 task-distribution artifact。

---

## 动机

主实验（Exp1/Exp2）全部在 BigCodeBench 上评测——模型也是在 BigCodeBench trajectory 上训练的。Reviewer 可预期的质疑：

- "指标漂亮是不是因为 overfit 到了 BigCodeBench 的 masking 模板？"
- "Clarify-execute 行为会不会只在日常工具代码上成立，换个 domain 就崩？"

需要一个**零额外训练**、**out-of-distribution** 的实验来拦住这个质疑。

---

## 核心思路

**不训练、不动模型**：TactfulLLM-Llama (v29) 权重完全不变。

**流程**：

```
BigCodeBench (训练) ──── TactfulLLM-Llama ────┐
                                              ├──→ 评测
OOD benchmark (新) ──── 跑 masking pipeline ──┘
```

**具体步骤**：

1. 选一个 domain 明显不同的 code benchmark
2. 对该 benchmark 的 instructions 应用主实验的 masking 协议（删 output_format / edge_cases / validation_rules 等字段）
3. 用主实验一致的 eval pipeline 跑 TactfulLLM / Direct / (可选) Clarify-first 三条件
4. 报告主实验一致的指标：pass@1 / OGR / Disc. / Rej. Rate / Avg Turns

**关键 claim**：
> Trained on BigCodeBench (general utility code), TactfulLLM generalizes to [Domain X] without fine-tuning, showing that persona-aware clarification is a transferable skill.

---

## Benchmark 候选 · 按 ROI 排序

### Option A — HumanEval + example-masking（smoke test）⭐ 先跑这个

- 164 任务，Llama-3.1-8B pass@1 ~55%（信号强，不会地板崩）
- **Masking 方法最便宜**：HumanEval docstring 自带 `>>> example()` 块 → 删掉这个块当作 mask
- **工程代价**：~0.5 天；regex 删示例 + 产出 masked states 文件
- **局限**：spec 短，mask 维度少（只有"示例"一个），generalization 说服力中等
- **用途**：验证 "masking → pass@1 掉 → clarify 能 recover" 这条链在 OOD 上成立

### Option B — DS-1000 ⭐ 主推的完整版本

- 1000 任务，Llama-3.1-8B pass@1 ~20–28%
- **Domain 明显 OOD**：data science (pandas/numpy/scipy/sklearn/...) vs BigCodeBench 的 general utility
- Spec 天然富含可 mask 字段：
  - 期望 library / API (e.g., "using pandas.groupby")
  - Output shape / dtype
  - Input 约束
  - Edge case handling
- **工程代价**：~3-5 天；masking pipeline 需要按 DS-1000 spec 格式调整，eval runner 需要适配
- **故事最好**：从工具代码泛化到数据科学，domain shift 清楚

### Option C — APPS-Intro subset（备选）

- 从 APPS intro 抽 200-500 任务
- Rich spec（input format / output format / constraints / examples）
- Llama-3.1-8B Intro pass@1 ~15%
- 和 BigCodeBench、DS-1000 都不像，算第二层 OOD 验证
- **工程代价**：中等（stdin/stdout-style test runner 需要适配）
- **何时用**：DS-1000 跑完还有时间，想加一层 robustness 证据

### 已排除

- **RECODE-H**：research code 难度太高，8B 模型 pass@1 会被压到个位数，floor effect 吞掉方法差异
- **APPS-Competition / LiveCodeBench 难题**：同上
- **ClassEval**：class-level 任务对 8B 太难
- **BigCodeBench-Hard**：同分布，只能叫 "difficulty generalization" 不是 domain generalization

---

## 执行计划

### Phase 1 — HumanEval smoke test（0.5-1 天）

**目的**：验证方向对不对，low risk。

- [ ] 写 `scripts/humaneval_mask.py`：读 HumanEval 164 条，regex 删 docstring 里的 `>>> example` 块，产出和 `test_states_v29_eval_200.jsonl` 同格式的 masked states 文件
- [ ] 跑 Direct baseline → 确认 pass@1 相对原始 HumanEval 下降（下降 = masking 有信息量）
- [ ] 跑 TactfulLLM → 看 clarify 是否 recover 一部分 pass@1
- [ ] 跑 Oracle (原始 HumanEval) → 定 ceiling

**决策点**：
- 若 Direct pass@1 vs Oracle 差距 ≥ 10pp → masking 有效，继续 Phase 2
- 若 TactfulLLM 相对 Direct 改善 ≥ 3pp → generalization signal 成立
- 若 Direct 掉得少（< 5pp）→ HumanEval 例子信息量不够，**撤 → 直接上 DS-1000**

### Phase 2 — DS-1000（3-5 天，Phase 1 通过后）

**目的**：强 generalization claim。

- [ ] 下载 DS-1000 数据 + 评测框架
- [ ] 调整 `scripts/mask_task_details.py` 适配 DS-1000 spec 格式
  - 新 mask 字段：`expected_library`、`output_shape`、`input_constraints`
- [ ] 在 200-task subset 上跑 Direct + TactfulLLM smoke test（~1 天）
  - 若 Direct pass@1 < 10% → 撤退或降难度
- [ ] 全量 1000 任务跑 Direct / TactfulLLM / Clarify-first / Oracle（~2-3 天 eval 时间）
- [ ] 按主实验模板生成表格：pass@1 / pass@5 / Avg Turns / Rej. Rate + OGR / Disc.

---

## 指标

和主实验完全一致，便于对照读：

| 指标 | 说明 |
|---|---|
| pass@1 / pass@5 | 代码正确性 |
| Avg Turns | 交互轮数 |
| Rej. Rate | 澄清问题被拒比例（per-clarification pooled） |
| OGR | (method − direct) / (oracle − direct) |
| Disc. | 被恢复的 mask 信息占比 |

**持论**：若 TactfulLLM 在 OOD benchmark 上 OGR 仍 ≥ 30%、per-persona 行为分化模式（Busy 少问 / Novice 多问）保留，就是 strong generalization 证据。

---

## 科学风险 & 缓解

| 风险 | 缓解 |
|---|---|
| OOD benchmark pass 率太低，方法差异被 floor 吃掉 | Phase 1/2 都先跑小规模 smoke test，若 Direct 太低立即撤 |
| 新 benchmark 的 masking 字段不好定义（spec 格式不同） | HumanEval 用"删示例"这一 trivial mask；DS-1000 的 mask 字段有几个自然候选，先从这些入手 |
| DS-1000 eval 时间长（7 个 library × 1000 tasks） | 先跑 200-task subset，若有信号再扩到全量 |
| Persona 行为分化可能不如 in-domain 明显 | 可接受——只要 OGR / pass@1 改善显著，行为分化减弱可以归因为 OOD；prose 里坦承这一点 |

---

## 论文定位

放在 **Experiment 4: Generalization to Out-of-Distribution Tasks**（或 appendix "Additional Experiments"），一个表 + 两段 narrative：

- 段 1：setup（TactfulLLM 权重不变，只换评测 benchmark）
- 段 2：结论（OGR / pass@1 numbers + 回到 intro 的 "transferable skill" claim）

---

## 下一步

**等用户拍板**：Phase 1 (HumanEval smoke test) 立刻开工？还是先写完主实验 narrative 再来？
