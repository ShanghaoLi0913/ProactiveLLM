# Classifier Bug & DPO Behavior Finding — 2026-04-28

> 明天 (2026-04-29) 重读 + 决策。所有 sanity 脚本和数据见 `scripts/sanity_classifier/`。

## TL;DR

`policy/infer.py:_pick_action_v1` 只看生成的前 30 token 前缀来判 Clarify/Execute。Llama 和 Qwen DPO 模型在 Novice/Exp persona 上 turn 0 都倾向 emit "I'd be happy to help / Sure! Let's break down..." 这种 friendly preamble，30 token 窗口内全是自然语言 → v1 系统性误判 Clarify。但 200 token 全文里 model 实际上写了完整代码（` ```python\nimport...\ndef task_func`）。

两个 backbone sanity 都 16/24 = 67% v1 vs v2 disagreement。深层问题：**DPO 没让 model 学会论文 claim 的 turn-0 persona-aware Clarify decision**——训练 preferred Clarify response 是直接问句（"What should..."），实际训练后 Llama Novice 40 sampling 0/40 emit 直接问句，仅 7.5% 在 preamble 后接真 clarify question。

---

## 1. Sanity 数据

### 1.1 Greedy（确定性）— 双 backbone 对比

| Backbone | Total | Agree | v1=C, v2=E（v1 误判） | v1=E, v2=C |
|---|:---:|:---:|:---:|:---:|
| Qwen DPO (`models/v29_qwen_100states/`) | 24 | 8/24 (33%) | **16/24 (67%)** | 0 |
| Llama DPO (`models/v29_100states/`) | 24 | 8/24 (33%) | **16/24 (67%)** | 0 |

两个 backbone 完全一致 pattern：
- Novice 8/8 误判（preamble + code）
- Exp 8/8 误判（preamble + code）
- Busy 0/8 误判（直接 ` ```python` 起手）

### 1.2 Sampling（T=0.7, top_p=0.9）— Llama DPO Novice 40 sample

```
v1 = Clarify:        40/40 = 100%   ← 100% 误判
v2 = Execute:        37/40 = 92.5%  ← preamble 后真写代码
v2 = Clarify:         3/40 =  7.5%  ← preamble 后真问 clarify question
starts-with-question:  0/40 =   0%  ← 没一个起手是 What/How/Could
```

3 个 v2=Clarify 例子（preamble 后嵌入真问题）：
- state2 sample0: "I'd be happy to help. Before we begin, I just want to confirm... Is that correct? Also, what would you like the categories to be?"
- state4 sample0: "I'd be happy to help. Before we start, I just want to clarify what you're trying to achieve..."
- state4 sample4: "I'd be happy to help. What does the 'data' DataFrame look like? Is it empty or does it already have some columns?"

### 1.3 训练 preferred response vs 实际 model emission

DPO 训练 preferred Novice Clarify response（来自 `data/dpo/prefs_v29_100states.jsonl`，剥掉 "Clarify\n" 前缀后）：
```
"What should the function return: a single average value or a list...?"
"Should the function handle cases where the input list has fewer than two elements...?"
"What is the expected output format for the character counts...?"
```

但训练后的 Llama DPO 在 Novice persona turn 0 实际 emit：
```
"I'd be happy to help you with your task.\n\nTo clarify, I'll make sure to explain
each step of the code in detail. Don't worry if you're new to programming...
\n\n### Step 1: Import necessary libraries\n```python\nimport numpy as np..."
```

**Gap：DPO 训练目标与实际行为完全不同。** Llama 的 pretrained "friendly assistant" tendency 压过了 DPO learned signal。

---

## 2. v1 实际看到 vs model 实际想做

### Llama Novice idx_0 例子

**text30（v1 看到，30 token）：**
```
I'd be happy to help you with your task.

To clarify, I'll m
```
→ v1 verdict: **Clarify**（前缀不是 ` ```/def/import`）

**text200（v2 看到，200 token，同一次 generate）：**
```
I'd be happy to help you with your task.

To clarify, I'll make sure to explain each step of the code in detail.
Don't worry if you're new to programming; I'm here to guide you through it.

### Step 1: Import necessary libraries
We need to import the `numpy` library for numerical operations and the
`random` library for generating random numbers.

```python
import numpy as np
import random
```
##
```
→ v2 verdict: **Execute**（` ```python` + `\nimport ` 命中）

**Model 真实 intent**: Execute（写完整代码，带教学说明）

---

## 3. Eval pipeline 流程：为什么 "Llama Novice 7 turns" 不是 model 自己的 Clarify decision

```
turn 0:
  classifier 看 prompt → model emit "I'd be happy to help...```python\n..."
  v1 看前 30 token → 判 Clarify （误判：model intent 是 Execute）
  generate_assistant_message(action="Clarify", ...) 被调用
  → 用 Clarify template + temperature=0.7 重新生成 → assistant_msg = 真 question
  user simulator 回答 → state.disclosed_info 增加 → next turn

turn 1: 重复 → 又判 Clarify → 又强制生成 question → ...
turn 7: 撞 max_turns 顶 → forced final Execute → 真出代码
```

**关键**：conversation 里看到的 turn 0 assistant_msg "What is the expected output format..." 不是 model 自己的输出，是 **v1 误判后 Clarify template 强制重新生成的产物**。Model 自己的 30/200 token classifier 输出从未被用作 assistant_msg。

---

## 4. 受影响的 baselines（双 backbone 同样规则）

| Baseline | Action 来源 | 受 v1 误判影响？|
|---|---|:---:|
| Direct Execution | hardcoded `Execute` | ❌ |
| Clarify-First (CF) | hardcoded turn-based | ❌ |
| Oracle | hardcoded 单 turn Execute | ❌ |
| Ideal Disclosed | hardcoded 单 turn Execute | ❌ |
| **Base** (no_lora) | classifier (`select_action_with_model`) | ✅ |
| **Prompt-only** | classifier (`select_action_prompt_only`) | ✅ |
| **DPO / TactfulLLM** | classifier (`select_action_with_model`) | ✅ |
| **Ablations** (no_persona, no_uncertainty) | classifier | ✅ |

---

## 5. 论文 narrative 的影响（三层分析）

| 层 | 状态 | 解释 |
|---|---|---|
| **数字层**：DPO Llama 14% pass@1 / Qwen 11.67% | ✅ 真实 | v1 pipeline 的 reproducible 数字 |
| **行为层**：Novice 7 turn / Exp 3.4 / Busy 1 | ✅ 真实 | eval pipeline 实际产生的 turn 数 |
| **Mechanism 层**："DPO 学到 persona-aware Clarify decision" | ❌ 反驳 | sanity 显示 turn 0 都倾向 Execute |
| **对比层**："DPO 14% > Direct 12.3% 是 DPO 的功劳" | ⚠ 存疑 | 可能是 forced multi-turn 副作用而非 DPO learned 提升 |

**核心 paper claim "TactfulLLM learns persona-aware proactive Clarify decision-making" 数据不支持。**

---

## 6. Confidence Levels（决策时记得权衡）

### High confidence（数据无可辩驳）
- v1 classifier 30 token 前缀规则确实把 friendly preamble 读成 Clarify
- Llama Novice 100% emit "I'd be happy to help" 起手（40 stochastic sample 一致）
- Llama / Qwen 双 backbone 都 16/24 turn-0 disagreement
- DPO 训练 preferred Novice Clarify 是直接问句，但 model 学不到这个风格

### Moderate confidence（推断需要更多证据）
- "DPO 训练失败 / 完全没用"——只在 8 个 state 上测，且没 Base Llama 对照
- "Llama 14% pass@1 是 forced multi-turn 副作用"——没用 v2 重跑过 Llama
- "v2 重跑会让 DPO 数字跌到 ~Direct 水平"——纯推断

### 可能错的点（明天要验证）
1. **LoRA 真的加载上了吗？** — 没显式 verify，可能 sanity 跑的是 Base 不是 DPO
2. **Sanity prompt 跟训练完全一致？** — 代码层面看一致，没 byte-level diff
3. **8 state 代表性？** — n_masked = 1-3，可能其他 n_masked 范围下 DPO Clarify rate 不同

---

## 7. 性价比最高的 4 个验证（明天先做）

| Check | 时间 | 决定什么 |
|---|:---:|---|
| **A. Base Llama (no LoRA) Novice sampling** | 5 min | 区分 DPO learned vs Llama pretrained tendency。如果 Base 也 emit "I'd be happy to help"，DPO 在 Novice 上 ≈ 没改变；如果 Base 不一样，DPO 至少改了 Novice |
| **B. 真实 eval JSON 抽查** (`eval_v29_dpo_150extra.json` Llama Novice turn 0) | 1 min | 看 turn 0 action 分布 + assistant_msg 是不是直接问句 |
| **C. LoRA 加载 verify** | 1 min | 比较 Base vs DPO 同 prompt logits，确认 LoRA 生效 |
| **D. 50 state × 3 persona greedy** | 30 min | 拓宽 state sample，得到稳定 Clarify rate 估计 |

A + B + C 总计 < 10 min，明早做了再决定后续。

---

## 8. 决策点

### D1：选哪条路

| Option | 工作量 | 风险 |
|---|---|---|
| **A. 保 v1 数字 + pivot narrative** | ~2 天写作 | 贡献从 "learned proactiveness" 缩到 "persona-style + multi-turn recovery"，可能 reviewer 觉得 incremental |
| **B. 重训 DPO**（更高 rank / KL penalty / 更长训练） | ~3-5 天 train + eval | 不保证能让 model 学会直接问句；ddl 风险 |
| **C. v2 重跑所有 classifier-using baseline** | ~80-120h GPU (双 backbone × 4 行) | 数字大概率下跌；claim 进一步崩 |
| **D. 弃 paper 这个 cycle** | -- | 高心理代价但 ddl 风险最低 |

### D2：v2 重跑 scope（如果选 C）

| Backbone | Baseline | N | 估时 |
|---|---|:---:|:---:|
| Llama | DPO | 200 | ~14h |
| Llama | Base | 200 | ~16h |
| Llama | Prompt-only | 200 | ~12h |
| Llama | no_persona | 50 | ~5h |
| Llama | no_uncertainty | 50 | ~5h |
| Qwen | DPO | 100 | ~14h |
| Qwen | Base | 100 | ~13h |
| Qwen | Prompt-only | 100 | ~10h |

**Direct / CF / Oracle / Ideal Disclosed 不动**（不调 classifier）。

最小重跑：DPO 双 backbone（核心 claim 行）= ~28h GPU = 1 个 overnight + 半天。

### D3：Narrative pivot 选项（如果选 A）

| Pivot | 还能 claim 什么 | 弱点 |
|---|---|---|
| **Honest pivot** | "DPO teaches persona-conditional response style; multi-turn pipeline enables disclosure recovery" | 跟 CF baseline 作用相似，TactfulLLM 增量小 |
| **Mechanism pivot** | "强调 forced multi-turn → disclosure recovery 是真 mechanism，DPO 是 catalyst" | 这其实就是 CF 的 mechanism，非 novel |
| **Negative result pivot** | "DPO 难以 override pretrained friendly-assistant tendency on cooperative personas; instructive failure mode for proactive LLM training" | 转成 negative-result paper |

---

## 9. 当前未提交的代码

**`policy/infer.py`**（已写好，未 commit）：
- `_pick_action_v1` — legacy 30-token prefix 规则，bit-exact reproduce 之前所有 Llama 实验
- `_pick_action_v2` — 200-token 全文扫描，code marker anywhere → Execute；fallback to `?` → Clarify
- `pick_action_from_generation` — env var `CLASSIFIER_VERSION` 切换（默认 v1）

是否 commit 看 D1 选择：
- 选 A：commit + 把 v2 留 opt-in（保 Llama 现有数字 reproducibility）
- 选 C：commit + 默认切 v2（重跑用 v2）
- 选 B/D：暂不 commit

---

## 10. Sanity Files（已保存）

`scripts/sanity_classifier/`:
- `qwen_classifier_sanity.py` + `.json` — Qwen DPO 24 verdict
- `llama_classifier_sanity.py` + `.json` — Llama DPO 24 verdict
- `llama_sampling_sanity.py` + `.json` — Llama DPO Novice 40 sample
- 原始 logs 在 `/tmp/*sanity*.log`（重启会清，必要时复制）

---

## 11. 明天的最小工作流建议

1. 重读这份 doc（10 min）
2. 跑 Verification A + B + C（10 min 总）—— 把"Llama 是否真有问题"钉死
3. 决定 D1 路径
4. 决定 D2/D3（如果选 C/A）
5. Commit `policy/infer.py`（按 D1 选择决定默认）
6. 写 `work_log.md` §112 记 finding
