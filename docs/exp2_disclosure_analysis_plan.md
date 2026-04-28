# Exp 2 Disclosure-OGR Asymmetry: Verification Plan

> 创建日期: 2026-04-28
> 启动时机: Qwen Base 150extra 完成后
> 目标 venue: NeurIPS 2026

## 1. 背景

Exp 2 (Information Recovery) 当前主 finding 之一：
- Novice: 88.6% Disclosure → 82.4% OGR
- Experienced: 78.0% Disclosure → 40.0% OGR
- **42pp OGR gap from only 10pp Disclosure difference** —— 不成比例

当前 paper 措辞 (claim H):
> "the residual unrecovered fields likely concern critical specifications—such as edge cases or validation rules—that surface only through extended probing"

**问题**：这是 hypothesis，没有实证支撑。reviewer 大概率会问 "evidence?"。

## 2. 验证目标

把 H 从 speculation 升级为 empirical claim，给 paper 加 rigor。

## 3. 三个 verification angles

### Angle A — 按 category 拆 Disclosure（核心验证）

**测什么**：
TactfulLLM Llama 200-state 数据中，每个 persona 在每个 disclosed category 上的 disclosure rate。

四个 category（来自 `disclosure_rule`）:
- `input_constraints`
- `output_format`
- `edge_cases`
- `validation_rules`

**算法**：
```python
for persona in ['Novice', 'Experienced', 'Busy']:
    per_category_rate = {cat: [] for cat in CATEGORIES}
    for state in 200_states:
        masked = state['disclosure_rule']['masked_fields']
        # 从 conversation 累积所有 turn 的 disclosed_items
        disclosed = aggregate_disclosed_items_across_turns(state, persona)
        for cat in CATEGORIES:
            n_masked_in_cat = len(masked[cat]) if isinstance(masked[cat], list) else int(bool(masked[cat]))
            if n_masked_in_cat > 0:
                rate = len(disclosed[cat]) / n_masked_in_cat
                per_category_rate[cat].append(min(rate, 1.0))
```

**预期数据形态**：

| Persona | Input | Output | Edge cases | Validation |
|---|:---:|:---:|:---:|:---:|
| Novice | ~95% | ~92% | ~85% | ~80% |
| Experienced | ~92% | ~88% | ~50% | ~55% |
| Busy | ~0% | ~0% | ~0% | ~0% |

**支持 H 的 pattern**：Exp 在 edge_cases / validation_rules 上明显低于 Novice，但在 input/output 上接近持平。

**反驳 H 的 pattern**：Exp 在所有 4 个 category 上均匀低于 Novice ~10pp（说明只是数量少，不是 category 选择问题）。

**数据来源**：
- `outputs/eval_v29_100states_50test.json` (50 states)
- `outputs/eval_v29_dpo_150extra.json` (150 states)
- merge 得 200-state 完整数据

**Exit criteria**：表格生成 + 一段对照分析（2-3 句）。

### Angle B — Pass rate conditional on edge_case disclosure

**测什么**：
edge_cases 类别是否被披露 vs pass@1 的关系。

**算法**：
```python
for persona in ['Novice', 'Experienced']:  # Busy 跳过 (always 0 disclosure)
    with_edge_pass = []
    without_edge_pass = []
    for state in states:
        edge_disclosed = (len(disclosed_items_by_state[state]['edge_cases']) > 0)
        pass1 = first_sample_passed(state, persona)
        if edge_disclosed:
            with_edge_pass.append(pass1)
        else:
            without_edge_pass.append(pass1)
    # 比较两组 pass rate
```

**预期**：
- with edge_cases disclosed → pass@1 高
- without → pass@1 低
- gap 越大，证明 edge_cases 是 critical category

**支持 H 的 pattern**：Δ pass@1 ≥ 5pp。
**反驳**：Δ pass@1 < 2pp。

### Angle C — Case studies for appendix

**挑 case**：
1. **Exp 失败 + Novice 通过** —— 看 Exp 漏了哪个 field 导致 fail
2. **Exp 通过 + Novice 通过** —— 78% Disc 也够的 case
3. **Exp 通过 + Novice 失败**（如果存在）—— 反例

**每个 case 展示**：
- BigCodeBench task ID + brief description
- Masked query 内容
- Disclosed items by persona（结构化 dict）
- Generated code 关键 snippet
- Test failure mode（如适用）

**Format**：appendix table + 半页 narrative text。

## 4. 输出 → paper 整合

### Main paper 改动

主 finding 段加 forward reference：

```latex
... a 42 pp OGR gap from only a 10 pp Disclosure difference, indicating 
the residual unrecovered fields concern subtle specifications such as 
edge cases that surface only through extended probing 
(Appendix~\ref{app:disclosure_analysis} provides per-category breakdown 
and case studies).
```

### Appendix 节

```latex
\section{Per-Category Disclosure Analysis}
\label{app:disclosure_analysis}

\subsection{Disclosure rate by category and persona}
[Table from Angle A]

\subsection{Edge-case disclosure vs task success}
[Numbers from Angle B]

\subsection{Case studies}
[2-3 cases from Angle C]
```

## 5. 实施时间估算

| 步骤 | 时间 |
|---|---|
| Angle A 数据提取 + 表生成 | 1h |
| Angle B conditional pass rate | 1-2h |
| Angle C 挑 case + narrative | 3-4h |
| Appendix LaTeX 整理 | 1h |
| Main paper forward reference 加一句 | 10min |
| **总计** | **~6-8h** |

## 6. 决策矩阵

| 数据结果 | paper 措辞 |
|---|---|
| Angle A 强支持 H | "edge_cases / validation_rules 是关键 bottleneck"（坚定 claim）|
| Angle A 部分支持 | "edge_cases 是部分 bottleneck，complete framework 留 future work" |
| Angle A 反驳 | 改 main 段 phrasing："the asymmetry suggests information utilization depends on conversation depth, not just disclosure coverage"（去掉 specific category mechanism） |

## 7. 启动条件

**必须先完成**：
- Qwen Base 150extra 跑完（预计 wall ~17:00 today, 2026-04-28）
- merged 50test + 150extra → Qwen Base 200 完整数据
- 重算 Qwen 主表 Δ vs Base 行（更高优先级）

**完成上面后才动手做这个 disclosure analysis**：
- 优先级 P1（重要但不紧急）
- ddl 充裕的话做完整 A+B+C
- ddl 紧张只做 Angle A（最关键）

## 8. 输出文件

| 文件 | 内容 |
|---|---|
| `data/analysis/disclosure_per_category.csv` | Angle A 原始数据 |
| `data/analysis/edge_case_conditional_pass.csv` | Angle B 原始数据 |
| `data/analysis/exp2_case_studies.md` | Angle C 案例描述 |
| `figures/per_category_disclosure_table.tex` | Angle A 输出表 LaTeX |
| `docs/exp2_appendix_section.tex` | Appendix LaTeX 段落 |

## 9. 风险 / 注意事项

1. **Disclosed_items 数据完整性**：要确认所有 conversation turn 的 disclosed_items 字段都存在且 category 完整
2. **Sampling noise**：n=200 上 per-category sub-sampling 可能噪声大（如 edge_cases 只在 ~50/200 state 出现），需要交代 confidence interval
3. **Hypothesis 可能不成立**：要预先想好"如果 H 不成立"的备选 narrative（见决策矩阵）
4. **不要在 main paper 写满**：核心 finding 在 main，case studies 一定放 appendix
