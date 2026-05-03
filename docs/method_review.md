# Method §Review

> 审查 paper §Method vs code (`reward/compute*.py` + `policy/render_state.py`)
> 投稿前必修 🔴 / 必加 🟡 / 整理 🟢

## 🔴 必改（公式 bug，reviewer 必抓）

### 1. $C_{interrupt}$ 公式漏 Clarify indicator

**你写的**：
$$c(a_j, r_j) = \lambda - \gamma \alpha_j + \delta r_j$$

这个公式里 $\lambda$ 永远加，**不管 $a_j$ 是 Clarify 还是 Execute**。Code 里不是这样 — Execute turn cost = 0。

**改成**：
$$c(a_j, \alpha_j, r_j) = b_j (\lambda + \delta r_j - \gamma \alpha_j),\quad b_j = \mathbb{1}[a_j = \text{Clarify}]$$

或一句话：only Clarify turns contribute to $C_{interrupt}$, Execute turns contribute 0.

### 2. $\gamma$ 在 reward 和 simulator 同名不同义

- §Reward: $\gamma$ = 答了的奖励参数 (code 0.20)
- §Simulator: $\gamma < 1$ = patience decay rate

**Rename simulator 那个 → $\rho$**：
$$P(\text{answer} \mid p, t) = p_{base} \cdot \rho^t \quad (\rho < 1)$$

### 3. $a_j$ 跟 $\alpha_j$ 视觉冲突

$a_j \in \{\text{Clarify}, \text{Execute}\}$（action）vs $\alpha_j$（user 是否回答）— 看一眼以为同一个变量。

**Rename**：user 是否回答 → $u^a_j$，user 是否拒绝 → $u^r_j$。

## 🟡 必加（公式不完整）

### 4. $\kappa$ Clarify-turn penalty 没写

Code 每个 Clarify turn 额外 $-0.2$ 防 always-clarify collapse。完整公式：
$$R(\tau) = w_{task} R_{task} - w_{int} C_{interrupt} - \kappa \sum_j \mathbb{1}[a_j = \text{Clarify}]$$

加一句解释 $\kappa$ 作用。

### 5. `user_stopped` clamp 没写

Footnote 加：
> If the simulated user terminates the dialogue early due to exhausted patience, the per-turn reward is clamped to 0.

### 6. Hyperparameter 值表（appendix 或 footnote）

| Param | Value | Role |
|---|---|---|
| $w_{task}$ | 1.0 | task 权重 |
| $w_{interrupt}$ | 0.2 | interrupt 权重 |
| $\delta$ | 0.8 | 被拒罚 |
| $\lambda$ | 0.12 | 提问基础开销 |
| $\gamma$ | 0.20 | 答了的奖励（$\gamma > \lambda$ 必要）|
| $\kappa$ | 0.2 | per-Clarify-turn collapse-prevention |
| $\beta_{DPO}$ | 0.1 | DPO temperature |

## 🟢 整理（已 flag）

### 7. 重复 `\subsection{Preference Construction and Policy Optimization}`

末尾两段几乎重复（一段被 `%` 注释一段正式）。删第一段。

### 8. §Trajectory Generation bullet list

你自己写了 `% 不要写出 bulletpoints`。改成 prose：
> "The simulation pipeline involves three components: task masking that creates underspecified requests; user personas that model heterogeneous collaboration preferences; and a user simulator that responds to clarification actions."

### 9. Overview 段 "These trajectories allow us to evaluate..." 改写

你自己 flag `% 这句我要再改改 想想我的framework到底优越性在于哪里`。试：
> "Unlike single-turn task benchmarks, our trajectories preserve both clarification decisions and their downstream consequences, enabling joint optimization of task success and persona-aware interaction efficiency."

### 10. Reward 段加 "什么 reward / 什么 penalty" 一句

你自己 flag `% 最好点明`。加：
> "When $\gamma > \lambda$ (our setting), an answered clarification yields net negative cost ($\lambda - \gamma < 0$), effectively rewarding informative clarification; a rejected clarification imposes a strictly positive penalty $\delta + \lambda$."

## ✅ 不用改

- Sequential decision process framing
- State $s_j = (q_j, \sigma_j, d_j, r_j, p)$ 跟 code 一致
- Action $\{Clarify, Execute\}$
- DPO（不 claim RL）
- Persona = (patience, expertise)
- Multi-turn credit assignment（final $R_{task}$ propagate, per-turn $C_{interrupt}$ 累加）
- 三 component 划分（policy / reward / simulator）

## 总表

| # | 严重度 | 项目 |
|---|---|---|
| 1 | 🔴 critical | $C_{interrupt}$ 缺 $b_j$ indicator |
| 2 | 🔴 critical | $\gamma$ 在两处 overload |
| 3 | 🔴 critical | $a_j$ vs $\alpha_j$ 视觉冲突 |
| 4 | 🟡 required | $\kappa$ Clarify-turn penalty |
| 5 | 🟡 required | `user_stopped` clamp |
| 6 | 🟡 required | hyperparameter 值表 |
| 7 | 🟢 cleanup | 删重复 subsection |
| 8 | 🟢 cleanup | bullet → prose |
| 9 | 🟢 polish | overview framework 优越性句 |
| 10 | 🟢 polish | reward / penalty 解释句 |

🔴 必改 ~30 min；全做 1-1.5h。**不算大改**，每条都是 reviewer 必查点。
