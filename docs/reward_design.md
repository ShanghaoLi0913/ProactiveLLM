# Reward 设计

> 实现：`reward/compute.py` + `reward/compute_rewards.py` + `policy/render_state.py`

## State

每个 turn $t$ 的决策状态 $s_t$（policy 输入）：

$$
s_t = (q,\; \sigma,\; d_t,\; r_{t-1},\; p)
$$

| 符号 | 含义 | 来源 |
|---|---|---|
| $q$ | 用户 masked query | `state.query` |
| $\sigma$ | task uncertainty $\in [0,1]$ | 从 query 算（`compute_state_uncertainty`）|
| $d_t$ | dialogue turn 编号（0,1,2,…）| `state.dialogue_turn` |
| $r_{t-1}$ | 上一轮是否被 user 拒（0/1）| `state.prev_reject` |
| $p$ | persona = (name, patience, expertise) | `persona` 字段，3 选 1：Novice/Exp/Busy |

`render_state(s_t, persona)` 把这些 field 拼成纯文本 prompt 喂 policy；policy 输出 action $\in$ {Clarify, Execute}。**不含 action template / system prompt**，让 model 自由决策。

## 总公式

每个 turn $t$ 的 reward：

$$
R_t = w_{\text{task}}\,R_{\text{task}} \;-\; w_{\text{int}}\,C_{\text{Interrupt}} \;-\; \kappa\,\mathbb{1}[\text{Clarify}]
$$

如果 user 半路 stop（耐心耗尽），$R_t$ 直接 clamp 到 0。

## Reward 变量

| 符号 | 含义 |
|---|---|
| $b_t$ | 这个 turn 里 "?" 的数量（`assistant_msg.count("?")`） |
| $a_t$ | user 是否认真回答澄清（0/1） |
| $r_t$ | user 是否明确拒绝澄清（0/1） |
| $R_{\text{task}}$ | Execute 通过 BigCodeBench 测试 = 1，否则 0；Clarify turn 永远 0 |

## Interrupt cost

$$
C_{\text{Interrupt}} = b_t (\delta r_t + \lambda - \gamma a_t)
$$

| 情况 | $C$ 值 | 含义 |
|---|---|---|
| 有效澄清（$a_t=1$）| $b_t(\lambda - \gamma) < 0$ | 净奖励 |
| 被拒澄清（$r_t=1$）| $b_t(\delta + \lambda) > 0$ | 重罚 |
| 沉默无回应 | $b_t \lambda$ | 小 cost |
| 不澄清（$b_t=0$）| 0 | 无 |

## Hyperparameters

| 参数 | 值 | 作用 |
|---|---|---|
| $w_{\text{task}}$ | 1.0 | task 权重 |
| $w_{\text{int}}$ | 0.2 | interrupt 权重（小，让 task 主导） |
| $\delta$ | 0.8 | 被拒罚 |
| $\lambda$ | 0.12 | 提问基础开销 |
| $\gamma$ | 0.20 | 答了的奖励（$\gamma>\lambda$ 才有净奖） |
| $\kappa$ | 0.2 | 每个 Clarify turn 额外罚（防 "always clarify" collapse） |

## 例子（Novice 3-turn 通关）

| $t$ | action | $b_t$ | $a_t$ | $R_{\text{task}}$ | $R_t$ |
|---|---|---|---|---|---|
| 0 | Clarify | 2 | 1 | 0 | $-0.168$ |
| 1 | Clarify | 1 | 1 | 0 | $-0.184$ |
| 2 | Execute | 0 | – | 1 | $+1.000$ |

总 $R = +0.65$（赢任务超过澄清成本）。

对比：Clarify 被拒一次 = $0 - 0.2{\cdot}2(0.8+0.12) - 0.2 = -0.568$（重罚）。

## 为什么能学到 persona-aware 行为

- **Novice**（高耐心）：$a_t=1$ 多 → Clarify 净奖励 → DPO 选 Clarify
- **Busy**（低耐心）：$r_t=1$ 多 → Clarify 重罚 → DPO 选 Execute
- **Exp**（中耐心）：混合，DPO 用 $\sigma$（task uncertainty）+ $d_t$（轮数）+ $p$（persona）联合决策

state 里 $p$ 让 model 区分 user 类型，$\sigma$ 让它判断 task 模糊度，$d_t$ 让它知道已经问几轮了 → 三者结合学到 persona-conditioned + uncertainty-aware 决策。

eval 实测：Novice ~8 turn / Busy ~1 turn / Exp ~2 turn ✓

## 训练算法

不是 multi-turn RL（无 policy gradient on trajectory，无 value function）。是 **SFT-then-DPO**：

1. 用 multi-turn user simulator 跑出 trajectory，按上面公式算每轮 $R_t$
2. 同 state $s_t$ 不同 candidate trajectory 比 reward → 挑 chosen / rejected
3. 标准 DPO loss 在 (prompt=$s_t$, chosen, rejected) 上训
4. multi-turn 只体现在 **数据构造** 阶段，训练 loss 是 single-step preference

## ⚠ Paper §4 公式跟 code 的 4 处差异

| 差异 | Paper 写 | Code 实际 |
|---|---|---|
| $b_t$ 类型 | $\{0,1\}$ 二元 | "?" 数量（可 1, 3, 7…）|
| Interrupt 权重 | 隐含 1 | $w_{\text{int}}=0.2$ |
| $\kappa$ Clarify 罚 | 没写 | 每个 Clarify turn $-0.2$ |
| `user_stopped` clamp | 没写 | 强制 $R_t=0$ |

**Paper 必须补这 4 处**，否则 reviewer 跑 code 算的 reward 跟 paper 公式对不上。
