# TactfulLLM Experiment Design

> Target: NeurIPS 2026 (deadline 2026-05-06). Last updated 2026-04-25.
>
> 这个文档只讲 **final plan + why**。执行进度、调试记录、失败尝试在 `work_log.md` 和各 `v*_experiment_log.md` 里。

---

## Overview

4 个实验,每个回答一个独立问题,互不重复。

| § | Experiment | 回答什么 | 为什么一定要做 | 状态 |
|---|---|---|---|---|
| 5.1 | Main Performance | TactfulLLM vs baselines 的 pass@1 / turns 取舍 | Headline claim | ✓ Runs done, 表待整 |
| 5.2 | Recovery Analysis | Clarification 真的恢复 masked 信息了吗? | 证明 gain 来自 info recovery 而非 policy artifact | ✓ |
| 5.3 | Generalization (OOD) | 零训练换 benchmark,行为还成立吗? | 挡 "overfit to BigCodeBench masking" 质疑 | **Pending (4/25-28)** |
| 5.4 | Ablation | Persona 和 Uncertainty 两个信号各做了多少工? | 证明两个信号都必要 | ✓ |

### 明确不做(附理由)

| 砍掉 | 理由 |
|---|---|
| Qwen backbone | 时间不够。单 backbone + generalization(§5.3)比双 backbone 更回应 reviewer concern |
| Cross-Persona Swap (原 §5.3 方案,4/16 砍) | 要翻倍 eval 成本,信号反而弱。Ablation 对 "哪个信号起作用" 的回答更直接 |
| HumanEval | 和 BigCodeBench 同 domain(general Python utility),只有 1 个可 mask 字段(删示例),generalization claim 太弱 |
| RECODE-H | 8B floor effect(论文自己的数字:GPT-5 L0=6.0%,DeepSeek-V3.1 L0=5.1%)会压死 method 差异;且 paradigm 不对等(他们的 multi-turn 是 code-debug feedback,不是 spec clarification) |
| APPS-Competition / LiveCodeBench-hard / ClassEval / BigCodeBench-Hard | 要么 8B 地板崩,要么同分布只能叫 difficulty generalization |

---

## Shared Setup

| | |
|---|---|
| Backbone | Llama-3.1-8B-Instruct |
| Test set | `data/seeds/test_states_v29_eval_200.jsonl` — 200 canonical states,和 107 训练 states **零重叠**(4/21 verified) |
| Personas | Novice-Learner / Experienced-Engineer / Busy-Developer(覆盖 tolerance × expertise 矩阵) |
| User simulator | gpt-4o-mini(和训练轨迹一致) |
| Code eval | BigCodeBench test cases, pass@1 / pass@5 |

---

## §5.1 Main Performance

### Research Question
TactfulLLM 在 pass@1 和 avg_turns 的 trade-off 上是否优于 baselines?

### 5 个 conditions — 每个 ablate 一个轴

| Method | Ablates | Clarify 策略 |
|---|---|---|
| **TactfulLLM-DPO** (ours) | — | Learned, persona-conditioned |
| Direct Execution | Clarification | 从不问 |
| Prompt-only | Training | Persona 进 system prompt,无训练 |
| Always-Clarify (K=1) | Learned policy | 固定先问 1 轮 |
| Base LLM | Persona + training | 原版 Llama |

这 5 个组合分别挡掉 reviewer "不用 clarify 行不行 / prompt 就够了吧 / 固定问一轮更简单 / LoRA 有用吗" 的四种反问。

### Metrics
- **Primary**:pass@1、Avg Clarification Turns
- **Secondary**:pass@5、Clarify Rate
- **Composite (主图)**:Pareto plot `x=avg_turns, y=pass@1`,看谁在 frontier
- **Appendix**:λ-sensitivity 曲线 `U = pass@1 − λ · turns` — 给想看 scalar 的读者

### Figures
1. **Pareto plot** — 一 method 一点
2. **Success vs turn budget** — K ∈ [1, 6],截断对话到最多 K 轮(K 时 force Execute),重算 pass@1。数据从 Exp 1 复用,**不用重跑**。故事:TactfulLLM 的曲线平滑,赢在 "在对的时间问" 而非 "问得多"
3. λ-sensitivity(appendix)

### Statistical tests
Pairwise Fisher exact on pass@1(200 states)+ bootstrap CI(1000 resamples)。Power analysis 已确认 200 states 能到 p ≈ 0.001(`v29_experiment_log.md §13.1`)。

---

## §5.2 Recovery Analysis

### Research Question
Clarification 实际恢复了多少 masked 信息?恢复量如何随轮数变化?

### Design
**代码生成器固定 Base Llama(单轮 Execute),唯一变量 = 输入信息量**。

| Condition | Input | 作用 |
|---|---|---|
| Full Query | 未 mask `instruct_prompt` | 上界 |
| Ideal Disclosed v2 | masked query + bullet list 原始 spec | Clarification ceiling |
| TactfulLLM Clarified (per persona) | masked + 学到的 clarify | 实际恢复值 |
| Masked Direct | masked query | 下界 |

### 为什么 Ideal Disclosed v2,不是 v1
v1 用 `masked_fields`(含 `Return type:` 等人造标签)+ `"; "` 单行分号。实测 pass@1 < TactfulLLM Overall,**违反 ceiling 预期** — 原因不是信息缺失,是 eval prompt 格式不自然。

v2 改用 `disclosure_info.specification`(原始 BigCodeBench spec 文本)+ bullet list + 明确的 `Additional information from clarification:` 表头。修正后:
- v2 pass@1 = 16.0% **精确等于** TactfulLLM Overall(canonical-200,McNemar p ≈ 1.00)
- 仍和 Full Query 可区分:信息在代码模板**之后**、明示为澄清结果,vs Full Query 代码模板**之前**的单段 prose

### Metrics
pass@1、Δ vs Full、**OGR = (method − direct) / (full − direct) × 100%**、Disc. rate(disclosure 比例)。

### Cross-experiment link (Exp 2 → Exp 1)
Scatter:x=recovery rate,y=pass@1 improvement over Direct,每 persona 一点。预期正相关 + 高 disclosure 边际递减。**给 reviewer 一个 cross-experiment 故事,而非三个孤立 table。**

---

## §5.3 Generalization to Out-of-Distribution Tasks

### Research Question
**零额外训练**,TactfulLLM 的 persona-aware clarification 在另一个 code domain 上是否仍然有效?

### 为什么一定要做
§5.1 和 §5.2 训练和评测全在 BigCodeBench。审稿人一定会写:
> "Are the gains an artifact of the BigCodeBench masking template? Does the clarify-execute behavior survive a domain shift?"

不跑这个实验就是把这个质疑主动让出去。

### 为什么选 DS-1000(benchmark 对比)

| 候选 | 否决理由 |
|---|---|
| HumanEval + 删示例 mask | 和 BigCodeBench 同是 Python utility domain(**weak domain shift**),只有 1 个 maskable 字段 |
| **RECODE-H** | (a) **8B 地板崩**:论文自己的 Table 2,GPT-5 L0=6.0%,DeepSeek-V3.1 L0=5.1%,L4 最高 21% — 我们的 8B 大概率 0-3%,floor effect 吞所有 method 差异。(b) **Paradigm 不对等**:他们的 multi-turn 是 code-debugging feedback(L0-L4 hierarchy),不是 spec clarification,我们的 masking + persona 协议套不进去 |
| APPS-Competition / LiveCodeBench-hard / ClassEval | 同 8B 地板问题 |
| BigCodeBench-Hard | 同分布,只能叫 difficulty generalization,挡不住 domain 质疑 |
| **DS-1000** ✓ | **Strong domain shift**(data science: pandas/numpy/scipy/…)+ 8B pass@1 20-28%(有信号)+ 天然多个 maskable 字段 |

### Design(零训练)
- TactfulLLM-Llama (v29) 权重不变
- 对 DS-1000 problem description 应用同构 masking 协议
- 同一套 eval pipeline,同一套指标

**Conditions — 只跑 3 个,minimal viable table**:

| Condition | 次数 |
|---|:---:|
| Direct Execution | 200 |
| TactfulLLM v29 | 200 × 3 personas = 600 |
| Oracle (Full Query) | 200 |

共 1000 conversations,单 GPU ~15-20h。不跑 Clarify-first / Prompt-only(nice-to-have,非 must-have)。

### Subset
全量 1000 超预算。**分层采样 200 个**(按库占比,seed=42):
Pandas 60 / NumPy 45 / Matplotlib 30 / Sklearn 25 / SciPy 20 / PyTorch 15 / TensorFlow 5。
200 和主实验规模一致,统计功效足够。

### Masking 字段(对齐 BigCodeBench 风格)

| 字段 | BigCodeBench 对应 | DS-1000 样例 |
|---|---|---|
| `expected_output_shape` | output_format | "DataFrame with columns [...]" / "shape (N, 3) array" |
| `expected_library_api` | (新增) | "using pandas.groupby" |
| `input_constraints` | input_constraints | "assume x non-negative" |
| `edge_case_behavior` | validation_rules(可选) | "return None on empty input" |

DS-1000 是自然 prose,regex 不稳。用 **gpt-4o-mini LLM-assisted span extraction** + **20 条人工抽检 gate**,再全量。

### Success criteria
两条都满足 → strong generalization:
1. Direct vs Oracle ≥ 5pp(masking 在 OOD 上有信息量)
2. TactfulLLM vs Direct ≥ 3pp 且 **OGR ≥ 30%**(主实验 Overall 48%)

若再满足:
3. Per-persona turns 分化保留(Busy < Exp < Novice)

若 (3) 不成立但 (1)+(2) 成立:prose 坦承 "clarification skill transfers; behavioral differentiation weakens under OOD" — 仍可写。

---

## §5.4 Ablation

### Research Question
Persona(user signal)和 Uncertainty(task signal)两个输入信号各贡献多少?

### Design
保持 v29 DPO pairs 不变,只改 `render_state.py` 的 prompt,重训 + 评估。

| Variant | Prompt 改动 |
|---|---|
| TactfulLLM (full) | Persona block + Uncertainty |
| w/o Persona | 移除 `[User Profile]` |
| w/o Uncertainty | 移除 `Task Uncertainty` |

通过 `ABLATION_MODE` env var 传进训练和评估脚本。

### Results(100-state,4/17 完成)
- **w/o Persona** → turns ≈ 1.0(三 persona 分化完全消失),pass@1 14.0% → 11.0%
- **w/o Uncertainty** → turns 8.0 / 2.6 / 1.0(分化在),pass@1 14.0% → 10.7%

**解读**:Persona 控制 *何时问*,Uncertainty 控制 *问了有没有用*。

### Narrative
> Calibrated proactivity requires both user-conditioned (persona) and task-conditioned (uncertainty) decision signals.

---

## Timeline(剩余)

| 日期 | 任务 |
|---|---|
| ✓ by 4/24 | §5.1 / §5.2 / §5.4 runs done(详见 `work_log.md`) |
| **4/25** (今天,GPU 在跑 Qwen v31.4) | 下 DS-1000 + inspect 数据 + 起草 `scripts/ds1000_mask.py` |
| 4/26 | Qwen 完毕 → LLM mask 200 条 + 20 条人工抽检 + eval adapter + 10-task dry run |
| 4/27 – 4/28 am | DS-1000 eval 1000 conversations(~20h) |
| 4/28 pm | 算 OGR + 建表 + figure |
| 4/29 – 5/5 | 论文写作 + polish |
| **5/6** | Submit |

---

## File References

| File | Purpose |
|---|---|
| `docs/v29_experiment_log.md` | v29 详细实验记录(masking / training / eval) |
| `docs/v31_experiment_log.md` | v31 诊断 + Busy T1 Execute 分析 |
| `docs/generalization_experiment_plan_20260423.md` | §5.3 的早期草稿(已被本文档合并) |
| `docs/work_log.md` | 每日工作记录 |
| `eval/evaluate_multi_turn_persona.py` | 多轮评估脚本(增量写入 + 断点续跑) |
| `data/seeds/test_states_v29_eval_200.jsonl` | Canonical-200 测试集 |
| `models/v29_100states/` | TactfulLLM DPO LoRA adapter |
