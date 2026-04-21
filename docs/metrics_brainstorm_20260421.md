# Experiment 1 — Interaction Quality Metrics Brainstorm

> 2026-04-21 session 思考记录。明天来筛选和算数。

---

## 背景

Exp1 分两部分：

- **Part 1 — Task Success**：pass@1 / pass@5（已定，不动）
- **Part 2 — Interaction Quality**：体现 TactfulLLM "问得合理" 的行为

现有的 `avg_turns` 和 `rejection_rate` 作为 Part 2 的 metric 不够：

- **avg_turns 是相对量**：7 轮对 Novice 合理、对 Busy 是噩梦；单数字没语义
- **rejection_rate 是比率**：同样值在 1 轮和 10 轮下含义完全不同

需要 turn-invariant 或 persona-normalized 的指标。

---

## 候选 metric 全清单（9 个）

### 1. Clarification Attempts per Persona

$$\text{CAP} = \frac{\text{\#clarify}}{N_{\text{task}}}$$

- 最初想法，raw count
- **问题**：还是绝对轮数，没解决"多/少"的相对性

---

### 2. Patience Utilization Ratio (PUR)

$$\text{PUR} = \frac{T}{\text{budget}(\text{persona})}, \quad
\text{budget}: \text{Busy}=1,\ \text{Exp}=2,\ \text{Novice}=5$$

- budget 取 patience 概率 `p^k ≈ 0.5` 的 k
- **优点**：直观，"烧了多少预算"
- **缺点**：budget 是启发式（不是数据驱动）；>1 表示超预算但没上限（无 cap）

---

### 3. Rejections per Task (RPT)

$$\text{RPT} = T \cdot r$$

（`T` = avg turns, `r` = rejection rate）

- "每个任务平均被拒绝几次"
- **优点**：绝对量有实际意义（=用户真被惹几次），方向明确（越小越好）
- **缺点**：Direct 和 "不 clarify" 的 Busy 都是 0，无法区分

---

### 4. CAR / CSR — Clarification Acceptance Rate

$$\text{CAR} = \frac{\text{\#answered}}{\text{\#asked}} = 1 - r$$

- turn-invariant
- **优点**：直接衡量"每一问有没有用"
- **缺点**：单独用不够强——Clarify-first 能刷到 0.72，TactfulLLM 反而 0.55（问得多包含很多"已饱和还问"的 rejection）

---

### 5. ACT — Answered Clarify per Task

$$\text{ACT} = T_c \cdot (1 - r)$$

（`T_c` = avg clarify turns）

- "每任务有效澄清次数"
- 相当于 asked × accept rate

---

### 6. NIQ — Net Interaction Quality ⭐

$$\text{NIQ} = \text{ACT} - \text{RPT} = T_c \cdot (1 - 2r)$$

- 用有效澄清次数减掉拒绝次数
- `1 − 2r`：r < 0.5 正值，r > 0.5 负值 → 拒绝率超过一半的方法 NIQ 为负
- **优点**：一个数同时捕捉"问得多 + 问得有用" + "惩罚高拒绝"
- **缺点**：NIQ = 0 有两种情况（从不问 / 问但一半被拒）需要配合 raw 数读

**拍板**：Exp1 Part-2 主 metric 用 **NIQ**。

---

### 7. TAS — Turn Appropriateness Score

$$\text{TAS} = \max\left(0,\ 1 - \frac{|T - \text{ideal}|}{\text{tol}}\right)$$

其中：
- `ideal(task, persona) = w(persona) × N_masked(task)`
- `w(Busy)=0, w(Exp)=0.5, w(Novice)=1`（persona 对澄清的倾向权重）
- `tol` = 容忍度（比如 2）

- **优点**：task-adaptive（mask 字段多的 task 允许问更多轮）
- **缺点**：w 和 tol 都是拍的，定义不干净

---

### 8. PRTD — Persona-Reference Turn Deviation

$$\text{PRTD} = |T_{\text{method}} - \text{ref}(\text{task, persona})|$$

- task-adaptive 的 reference，从 v29 trajectory 的 per-(task, persona) mean 得来
- **致命问题**：v29 轨迹覆盖 task 0–108，eval 集 111+，**零重叠**——没法提供 per-eval-task reference
- 要救只有两条：(A) 用 per-persona 全局 mean 当 reference（牺牲 task 粒度）；(B) 在 eval 集上重跑 teacher rollout（～5h 额外开销）

---

### 9. PTD — Persona Turn Deviation ⭐

$$\text{PTD} = |T_{\text{method}} - \text{ideal}_T(\text{persona})|, \quad
\text{ideal}_T: \text{Busy}=1,\ \text{Exp}=2,\ \text{Novice}=5$$

- PRTD 的简化版：per-persona 固定 ideal，不看 task
- **优点**：定义最干净，一行算出来
- **缺点**：所有 task 等价对待，忽略 task 复杂度

**拍板**：Exp1 Part-2 第二 metric 用 **PTD**（替 PRTD）。

---

## 已拍板方案

Exp1 Part-2 用 **NIQ + PTD 两个 metric**：

| Metric | 焦点 | 方向 |
|---|---|---|
| **NIQ** | 问澄清的"净效益"（有用 − 被拒）| ↑ 好 |
| **PTD** | 轮次相对 persona ideal 的偏差 | ↓ 好 |

两者互补：NIQ 看"问得有用吗"，PTD 看"问得合 persona 习惯吗"。

---

## 未解决的问题（明天想）

1. **NIQ 的 `r` 用哪个分母？** per-clarify rejection（sim 拒答）还是 per-turn？目前 `analyze_disclosure_recovery.py` 算的是 per-clarify。
2. **PTD 的 ideal_T 数值**：现在 Busy=1 / Exp=2 / Novice=5 是拍的。v29 轨迹的 per-persona mean 分别是 2.25 / 2.80 / 3.64——差很多。要不要用轨迹学来的数代替？
3. **是否同时放两个 metric**：paper 版面紧；可能只进主表一个、另一个进 appendix
4. **还没算过实际数值**：NIQ 和 PTD 在 5 个 method × 3 persona 下具体多少，没人知道，得先算一遍看谁有判别力
5. **v30 重跑前要不要冻结 metric**：NIQ 在 v29 的 Novice（7 turns, 50% rej）上会偏低，如果 v30 能把 Novice 降到 4 turns，NIQ 会显著回升——这是 metric 说服力的好测试

---

## 代码 / 数据现状

- **已有**：`scripts/analysis/plot_interaction_quality.py`（CSR + Turn-vs-budget 双图版）
- **缺**：NIQ、PTD 的算子和表；需要补一个 `scripts/analysis/compute_niq_ptd.py`
- **数据源**：
  - Clarify-first / Prompt-only 的 canonical-200 数据还在补跑（今晚 Clarify-first 已启动 PID 146330）
  - Base LLM / Direct / TactfulLLM 已 canonical-200 完整
- **前置依赖**：要出最终表，必须等 Clarify-first + Prompt-only 补跑完（~30h）

---

## 相关参考

- 入口讨论：work_log #86（"Exp1 Part-2: 轮次合理性 metric 设计中断"）
- 出图代码：`scripts/analysis/plot_interaction_quality.py`
- 参考轮次提取：`scripts/analysis/extract_reference_turns.py` → `data/analysis/ref_turns_v29.json`
- 实际 per-persona mean（v29 trajectory）：Busy 2.25 / Exp 2.80 / Novice 3.64
