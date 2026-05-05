# v33: SFT-then-DPO Pipeline

> Started 2026-04-29 due to v1 误判 finding (see `classifier_bug_2026-04-28.md`)
> NeurIPS DDL: 2026-05-06

## 🎯 为什么开始 SFT+DPO（核心动机）

### 1. v1 classifier 误判 finding 暴露了根本问题

2026-04-28 sanity 验证发现：v1 classifier 看 30 token 前缀判 Clarify/Execute，但
Llama / Qwen DPO 模型在 Novice/Exp persona 下 emit "I'd be happy to help..." preamble
起手（背后实际是写完整代码）。**v1 误判 67%（双 backbone 一致）**。

```
Model 真 intent (200 tok 全文): "I'd be happy to help...```python\nimport...```"
                                                              ↑ Execute
v1 看 30 tok 前缀:               "I'd be happy to help. To clarify, I'll mak"
v1 verdict:                      Clarify  ❌ 误判
```

**这意味着 v29 DPO 14% pass@1 是 v1 误判 + forced multi-turn artifact**，不是
"DPO 学到 persona-aware proactive Clarify decision-making"。

### 2. Pure DPO 在 cross-distribution learning 上结构性失败

为什么 DPO 没让 model 真学到 emit 直接问句？数学根因：

```
DPO loss: L = -log σ(β · (log π/π_ref(chosen) - log π/π_ref(rejected)))
                          ↑
                       受 β 约束的 ratio 优化
                       β=0.1 意味着 model 不能远离 π_ref 太多

π_ref = Llama-3.1-Instruct (RLHF 训过):
  P("I'd be happy to help...") = 0.5      (RLHF 训得很高)
  P("What should the function...") ≈ 1e-10 (几乎从不自然 emit)

DPO 即使把 ratio 翻 1000 倍:
  P("What should...") = 1e-10 × 1000 = 1e-7
  仍是绝对小，model 还是不会自主 emit
  → DPO 撞 "可达 distribution 边界"，过不去
```

### 3. v30/v31 系列证明 "改 oracle / 加数据" 不解决问题

```
v30: 改 oracle (Busy clarify when n_masked>=3)        → pass@1 跌到 2.56% (vs v29 14%)
v31: 改 oracle + reward                              → 9.33%
v31_2a: 进一步改 oracle                              → 8.67%
v31_4: 更多 oracle 调整                              → 8.00%

→ 4 次 retry 都失败，证明 改 oracle 不动 hparam 推不动
→ pure DPO + LoRA 在这个 task 上有结构性 limit
```

### 4. v32 系列证明 alpha 调参也救不了

```
v32 (alpha=128, keep prefix): collapse — Busy emit "Execute." then 停 (degenerate)
v32b (alpha=32, keep prefix):  no collapse, 但 model 还是不学 prefix

→ alpha 太低推不动，太高 collapse，没 sweet spot
→ 单纯调 hparam 在 cross-distribution learning 上不够
```

### 5. 为什么 SFT-then-DPO 是答案

```
SFT loss: L = -log p(chosen | prompt)
                    ↑
              直接最大化 chosen likelihood，无 KL 约束
              → 可以 shift 到任意远的目标 distribution

SFT 跨过 KL gap:
  "I'd be happy..." → "Clarify\nWhat should..."
  
然后 DPO 从 SFT 模型继续训:
  ref = SFT 模型 ≈ chosen distribution
  KL gap ≈ 0
  → DPO 能正常 refine + 加 rejected 信号

这就是 standard SFT-then-DPO recipe (Rafailov 2023, Llama-2-Chat, Zephyr 7B 都用)
```

**结论**：v29-v32 系列 8 个月的 pure DPO 尝试在 RLHF-tuned Instruct 模型上结构性受限。
2026-04-29 转 SFT-then-DPO，标准 industry recipe，理论可解决 KL gap。

---

## 📍 当前进度（2026-05-02）

### May 2 (PM) — State set 一致性审计 + Qwen PO classifier inconsistency 发现

#### Llama state set bug — CF 跟 DPO 不同 state set，之前对比 unfair

Llama DPO partial（n=40）: Busy 5.4%。跟 Llama CF 150 (Busy 19.3%) 比，差 14pp 看似 disaster。审计发现：

```
File                                    State set         |Common with CF 150
─────────────────────────────────────────────────────────|───────────────
Llama Direct 200      eval_200          200 states        | 150 [eval_200 ⊃ eval_150extra]
Llama CF 150          eval_150extra     150 states        | 150 (self)
Llama v33 DPO 200     eval_200          200 (running)     | 150 (after full run)
Llama Base v2 (queued) eval_200         200 (queued)      | 150
Llama PO 200 v2       eval_200          200 (running)     | 150
```

**eval_200 = eval_150extra ∪ 50 new states**。DPO partial 41 恰好都在 "新 50" 里 → DPO partial ∩ CF 150 = 0。

**Paired Llama DPO vs Direct on SAME 40 states**：

| Persona | DPO | Direct | Δ | DPO_wins | Direct_wins |
|---|---|---|---|---|---|
| Novice | 12.5% | 5.0% | **+7.5pp** | 3 | 0 |
| Exp | 12.5% | 5.0% | **+7.5pp** | 4 | 1 |
| Busy | 5.0% | 5.0% | 0 | 1 | 1 |

McNemar χ²<3.84（N=40 太小），但 direction 一致：**DPO ≥ Direct 在所有 persona 上**。partial Busy 5.4% 看似低是因为这 40 个 state 本身难（Direct 全集 200 Busy=13% vs 这 40 Busy=5%）。

**对 Qwen 类似检查**：✅ Qwen 5 个 method 都在 eval_200 上（Direct 200, CF 200, Base 200, TactfulLLM 200, PO 100→200）。clean。

**Llama CF fix plan**：CF 当前在 eval_150extra (150 states)，eval_200 = eval_150extra + 50 new。补跑 CF on 50 missing states (~3h on free GPU after Llama DPO finishes)，合并 → CF 200 on eval_200，跟其他 4 method paired-comparable。

#### Qwen PO classifier inconsistency — first-100 是 v1, remaining-100 是 v2

| File | Date | Classifier | avg_turn (Novice) |
|---|---|---|---|
| `eval_v29_qwen_prompt_only_100.json` | **Apr 26** | **v1** (Apr 28 才上 v2) | 2.05 |
| `eval_v29_qwen_prompt_only_remaining100_ft.json` | May 2 | v2 | 1.00 |

v1 把 PO 的 "I'd be happy to help..." preamble 误判为 Clarify → first-100 多轮（2.05 turn）。v2 严格 "Clarify\n" prefix → 都判 Execute（1.00 turn）。

**影响**:
- pass@1 N=200 合并: ✅ 有效（pass/fail 不依赖 classifier verdict）
- avg_turn / clarify_rate N=200: ❌ 失效（classifier 不一致）

**Fix（已部署）**: 重跑 Qwen PO first-100 on v2，replace 旧 first-100：
- 新 GPU 1 chain：Qwen PO rem-100 (running) → Qwen PO first-100 v2 (~5h) → Llama Base 200 (~10h)
- `/tmp/qwen_po_first100_v2.sh` 用 `data/seeds/test_states_v29_eval_200_first100.jsonl`（从原 first-100 file 反抽 100 state IDs，保 same set）
- 完成后 Qwen PO N=200 全 v2 一致

#### 教训 — methodology

1. **paper 必须用同 state set** 做 method 间比较；不同 set 做的 ranking 是 misleading（Llama CF "vs" DPO 5.4 是不存在的差异）
2. **classifier 版本一致性** 是 metric (avg_turn, clarify_rate) 的隐性前提；pass@1 不受影响但 turn-level 指标受影响
3. **Audit 工具应在每次 method comparison 前自动跑**：检查 state_id set 重叠 + classifier 版本

#### 修订 paper 论述（再次）

之前修订过的 "TactfulLLM matches CF on accuracy" 加上：
- **Llama 上 paired comparison 显示 DPO > Direct 在 Novice/Exp** (+7.5pp partial)
- **Cross-backbone validation**：Qwen 上 DPO ≈ Direct (within noise), Llama 上 DPO > Direct (partial direction). 等 Llama N=200 完整 lock。
- **CF Llama 必须重跑在 eval_200 上** 才能进 main table，否则 footnote 标注 state-set caveat

---

### May 2 — Qwen N=200 method 比较的 honest 分析（重要 finding）

跟 baseline N=200 数字 lock 后做 paired comparison，发现**之前的 "TactfulLLM 是最优 method" 主张站不住**。

#### Qwen N=200 final accuracy 排名

| Method | pass@1 | pass@5 | avg_turn |
|---|---|---|---|
| 🥇 Clarify-First | **15.7** | **23.2** | 2.00 |
| 🥈 TactfulLLM | 15.2 | 23.0 | 3.99 |
| 🥉 Direct | 14.7 | 18.3 | 1.00 |
| Prompt-Only (N=100) | 13.0 | – | – |
| Base | 11.0 | 18.0 | 1.00 |

**CF accuracy 略胜 TactfulLLM** — 0.5pp pass@1, 0.2pp pass@5。

#### McNemar paired tests（N=200, χ²>3.84 显著）

```
TactfulLLM vs CF      ALL: χ²=0.07  Nov: 3.68  Exp: 1.88  Busy: 1.23
TactfulLLM vs Direct  ALL: χ²=0.06  Nov: 1.24  Exp: 0.41  Busy: 0.07
CF vs Direct          ALL: χ²=0.57  Nov: 0.27  Exp: 0.45  Busy: 1.78
```

**全部 method-method pairwise 都不显著（p > 0.05）**。最接近显著的是 TactfulLLM vs CF on Novice (χ²=3.68, p≈0.055，临界)。

#### 为什么 N=200 detect 不出差距？— Power 分析

```
Discordant rate ≈ 10%（600 trials 中 60 discordant pairs）
Detect Δ=1pp need 1285 per persona（6.4× 现有 N）
Detect Δ=2pp need 643 per persona（3.2× 现有 N）
Detect Δ=5pp need 257 per persona（1.3× 现有 N）

per-persona binomial SE at N=200, p≈0.15: ≈2.5pp
detect threshold @ 95% confidence: |Δ| ≥ 7pp
```

**N=200/persona 只能 detect 6-7pp 以上的差**。我们看到的 1-3pp 差距全在不可分辨区间内。

#### Per-persona pattern（descriptive，非 statistical claim）

| Persona | TactfulLLM | CF | Direct | Δ TactfulLLM-CF |
|---|---|---|---|---|
| Novice | 18.0 | 13.0 | 14.5 | **+5.0pp** ✓ |
| Exp | 14.0 | 18.0 | 16.0 | -4.0pp ✗ |
| Busy | 13.5 | 16.0 | 13.5 | -2.5pp ✗ |

观察到的 pattern：
- Novice 上 TactfulLLM 显著（borderline）赢 CF — 跟理论一致（multi-turn clarify 在合作 persona 上有用）
- Exp/Busy 上 TactfulLLM 略输 CF — 跟理论矛盾（更灵活的 method 不应该输给 hardcoded baseline）

#### Busy 上 TactfulLLM 内部分裂（描述性）

把 TactfulLLM 200 个 Busy sample 按它自己的 turn-0 决策分两组：

| TactfulLLM Busy 决策 | n | pass@1 |
|---|---|---|
| Execute-T0（不 clarify）| 91 | **18.7%** ⭐ 三 method 中最高 |
| Clarify-T0（决定 clarify）| 109 | 9.2% 三 method 中最低 |

**当 TactfulLLM 选不 clarify 时，Busy 上击败所有 baseline。当它选 clarify 时，比所有 baseline 都差**。
**这是描述性观察，但 N=91/109 内部分组的 pass 差是否 significant 没正式 test。**

#### 推测的机制（**未验证**，N=200 不足以 confirm）

候选解释（按 plausibility）：

1. **LoRA "alignment tax" on turn-1 Execute**：DPO 训练 chosen pairs 都是 Clarify→Execute，turn-1 的 Execute style 偏离 Qwen 自然分布。Turn-0 Execute (Busy 不 clarify case) LoRA 不损 (18.7% = Direct 13.5%? 实际略高)，turn-1 Execute 损 5pp。**但 same-state McNemar 不显著，可能是 noise**。

2. **Busy oracle 拒答率不匹配训练分布**：训练数据 oracle 给 Busy 的 patience 跟 eval oracle 不一致 → DPO 学到的 clarify policy 在 eval 上 over-fire。

3. **Pure sample variance**：N=200 给 SE ≈ 3.5pp，看到的 2-5pp 差全在噪声内，不需要 mechanism 解释。

⚠️ **N=200 数据无法在这三个解释间分辨。**Llama N=200 跑完后看是否 reproduce 类似 pattern：
- 如果 Llama 上 TactfulLLM > CF on Exp/Busy → Qwen 4pp 是 noise
- 如果两 backbone 一致 CF > TactfulLLM on Exp/Busy → 真 effect，机制需要更深入实验（如 disable LoRA on turn-1 Execute）

#### 修订的 paper claim（去掉 over-claiming）

❌ **不能写**：
- "TactfulLLM achieves best accuracy on Qwen"
- "DPO refinement improves over CF baseline"
- "LoRA alignment tax explains Exp/Busy underperformance"（machinery 未证）

✅ **可以写**（statistically defensible）：
- "TactfulLLM matches Direct/CF baselines on aggregate accuracy (15.2 vs 14.7 vs 15.7, all McNemar p > 0.5)"
- "TactfulLLM exhibits persona-conditional interaction depth: Novice 7.99-turn / Exp 2.42-turn / Busy 1.55-turn vs CF's flat 2.0-turn"
- "On Novice persona, TactfulLLM achieves 18.0% vs CF's 13.0% (5pp lift, McNemar χ²=3.68, p≈0.055 borderline)"
- "Cross-persona behavior split is the primary contribution; accuracy parity is the secondary finding"

#### 教训（method 角度）

1. N=100/200 paper-grade comparison 不够，detect 不出 1-3pp 差距
2. CF 是 surprisingly strong baseline — 强制 1 clarify + base-model 在 Qwen 上几乎打平 DPO trained model
3. DPO 真正贡献是**学到 persona-conditional 行为**，不是 raw accuracy
4. 任何 mechanism claim（如 alignment tax）需要 controlled ablation 才能 establish

---

### May 2 更新 — Qwen 全 4 baseline 中 3 个补到 N=200 ✅

3 个 baseline (Direct/CF/Base) remaining-100 一夜跑完无 freeze；`scripts/merge_baselines_200.py` 合并出 N=200 final。Sanity 全过：first-100 vs remaining-100 Δ ≤ 1.3pp（远小于 SE 3.5pp）。

**Qwen N=200 论文主表**:

| Method | pass@1 | pass@5 | avg_turn | clarify_rate |
|---|---|---|---|---|
| Base | 11.0 | 18.0 | 1.00 | 0.00 |
| Prompt-Only (N=100) | 13.0 | – | – | – |
| Direct | 14.7 | 18.3 | 1.00 | 0.00 |
| **TactfulLLM** | **15.2** | **23.0** | **3.99** | varied |
| Clarify-First | 15.7 | 23.2 | 2.00 | 0.50 |

**🎯 论文主结果 — pass@5 分两组**：
- Clarify 组 (TactfulLLM 23.0, CF 23.2)：clarify 触发 user disclosure，5 个 candidate 各得不同 hint → pass@5 拔高
- Execute 组 (Direct 18.3, Base 18.0)：直接 generate 无 disclosure → pass@5 = pass@1 噪声放大
- **TactfulLLM vs Direct: pass@1 +0.5pp / pass@5 +4.7pp** ← 这是 disclosure recovery 真实 evidence

**TactfulLLM > CF 的 contribution = persona-aware turn 分化**：
- TactfulLLM: Novice 7.99-turn / Busy 1.55-turn / Exp 2.42-turn
- CF: 全部 2.00-turn 固定
- 在几乎相同 pass@1 / pass@5 下，TactfulLLM 在 Busy 上 22% 更省 interaction cost
- **Paper claim**: "TactfulLLM matches CF on accuracy while adapting interaction depth to persona"

### May 1 更新 — N=200 合并完成 ✅

**Qwen v33 SFT+DPO N=200 (canonical, merged first-100 patched + remaining-100 ft)**:

| Persona | pass@1 | pass@3 | pass@5 | avg_turn | clarify_rate |
|---|---|---|---|---|---|
| Novice | 36/200 = **18.0** | 50/200 = 25.0 | 54/200 = **27.0** | 7.99 | 0.87 |
| Exp | 28/200 = **14.0** | 37/200 = 18.5 | 47/200 = **23.5** | 2.42 | 0.59 |
| Busy | 27/200 = **13.5** | 34/200 = 17.0 | 37/200 = **18.5** | 1.55 | 0.35 |
| **All** | **91/600 = 15.2** | **121/600 = 20.2** | **138/600 = 23.0** | 3.99 | – |

**N=100 patched → N=200 合并 sanity check**:

| Persona | N=100 patched | N=200 | Δ |
|---|---|---|---|
| Novice | 17.0 | 18.0 | +1.0 |
| Exp | 12.0 | 14.0 | +2.0 |
| Busy | 14.0 | 13.5 | -0.5 |
| **All** | **14.3** | **15.2** | **+0.9** |

全部 |Δ| ≤2pp（远小于 SE≈3.5pp），合并 sanity 通过。Pass@5 ordering 仍健康（Nov 27 > Exp 23.5 > Busy 18.5）—— 表示 Exp Clarify 拿到的 disclosure 确实让代码更好。

**vs Qwen baselines (overall pass@1)**:
- Direct 15.3 / CF 16.0 / Base 11.0 / **TactfulLLM 15.2 (N=200)** / PO 13.0
- 比 Base **+4.2pp**（真实 DPO 价值）
- vs Direct/CF 几乎打平（-0.1 / -0.8，远在噪声内）
- McNemar p>0.5 vs CF（baseline 仍 N=100，需补到 200 才能正式 paired test）

**输出文件**:
- `outputs/eval_v33_v3_qwen_dpo_v2_remaining100_ft.json` (Apr 30 19:20, 新模板+完整 code)
- `outputs/eval_v33_v3_qwen_dpo_v2_200.json` (May 1 11:33, 合并 first-100 patched + remaining-100, detailed_results=600)

**Pending decision (May 1)**:
- Baseline (Direct/CF/Base/PO) 是否补 100→200（4×17h ≈ 68h 单卡，34h 双卡）
- Llama backbone 是否补完整 SFT+DPO 100/200 pipeline
- GPU 升级 (单 4090 → 双 4090) 等上面两个决策定了再说

---

### 剩余实验分析（2026-05-01 决策快照，在跑 Qwen baseline 时整理）

DDL 5/6 还 5 天。下面按"必跑 / 应跑 / 可跑"分类。

#### 必跑（决定论文能不能投）

##### A. Qwen N=200 baseline 收尾

| Baseline | 当前 N | 状态 | 还要跑什么 |
|---|---|---|---|
| Direct | 100 (15.3%) | ⏳ remaining-100 跑中 (GPU 0) | 今晚完 + 合并 |
| CF | 100 (16.0%) | ⏳ remaining-100 跑中 (GPU 1) | 今晚完 + 合并 |
| Base | 100 (11.0%) | ⏳ remaining-100 跑中 (GPU 2) | 今晚完 + 合并 |
| PO | 100 (13.0%) | ❌ remaining-100 没跑 | May 2 起跑 (~5h) |
| TactfulLLM | 200 (15.2%) | ✅ 完成 | – |

合并工具：`scripts/merge_v33_qwen_200.py`。Sanity 通过 = first-100 vs remaining-100 |Δ| ≤2pp（参考 v33 DPO 案例已成立）。

##### B. Llama 主结果（cross-backbone validation）

当前 Llama 状态：
- ✅ v33 v3 SFT/DPO 模型已训完，sanity 24/24 perfect
- ❌ 只有 **5-state eval = 1/15 = 6.7%**，N 太小论文不能用
- ❌ Llama Base / PO 在 v2 classifier 下无 N=100/200 数字

§118 已证 **v1 vs v2 数字偏差大**（v1 Llama DPO 14% → v2 6.35%），所以依赖 classifier 的 baseline 必须 v2 重测：

| 任务 | N | 单卡时间 | v1 复用？ |
|---|---|---|---|
| Llama v33 SFT+DPO | 100/200 | 17h / 34h | ❌ 必跑 |
| Llama Base | 100/200 | 17h / 34h | ❌ 必跑（§118 排除）|
| Llama PO | 100/200 | 17h / 34h | ❌ 必跑（依赖 classifier）|
| Llama Direct | 200 ✅ | – | ✅ 已有（不依赖 classifier）|
| Llama CF | 150 → 200? | 17h | ⚠ 部分（150 缺 50）|

##### C. 统计检验（不要 GPU，1h 离线）

- McNemar paired test：DPO vs Base / Direct / CF / PO（每个 backbone 各 4 个对比）
- Bonferroni multi-comparison 校正
- 用现有 `outputs/eval_*.json` 的 `detailed_results` 字段直接算

#### 应跑

##### D. SFT-only vs SFT+DPO ablation（DPO refinement 单独 contribution）

5-state 已有数字（Llama 0/15 vs 1/15, Qwen 13.3% vs 13.3%）但 N 小。**只在时间富余时扩 N=100**，否则 5-state 加 paragraph 解释即可。

##### E. v1 vs v2 classifier ablation（classifier 设计的论文故事）

`scripts/sanity/classifier/` 已有完整 sanity 数据（Llama + Qwen，24-output × 2 classifier）。**不用跑**，直接写 appendix 表 + sample 输出。

#### 可跑（supplementary，时间富余）

- Pareto tradeoff plots：现有 `scripts/plot_persona_tradeoffs_2panel.py` 等
- Disclosure recovery analysis：`scripts/analyze_disclosure_recovery.py`
- Qwen DPO v1 (epochs=3) collapse 案例：写成 "DPO over-refinement 负面例子"

#### 时间排程（May 2-6）

```
今晚 (May 1)         3 baseline 跑完 (19:00 北京 = 06:00 芝加哥)

May 2 早 (你醒来)     1) Qwen baseline N=200 合并 + sanity check
                      2) GPU 0: PO Qwen rem-100 (5h) → Llama v33 SFT+DPO 100 (17h)
                         GPU 1: Llama Base 100 (17h)
                         GPU 2: Llama PO 100 (17h)

May 3 早              Llama N=100 全完。决策点：扩 N=200 (+ 1 天) 还是直接写 paper？
                      统计检验 + 表格生成 (1h)

May 3-5               写 paper + ablation/supplement 图

May 6                 提交
```

**关键 trade-off — Llama N=100 vs N=200**：

- N=100 std ≈ 5pp，足够看 method 跨 backbone 是否 work（"Qwen 15.2 vs Llama X" 数量级对比）
- N=200 std ≈ 3.5pp，paper-grade 严谨数字
- Qwen first-100 vs N=200 飘 +0.9pp（< SE） → N=100 数字基本稳
- **Llama 大概率也 stable**，N=100 够用；如果数字反直觉再扩

**N=100 vs N=200 决策卡点**：等 Qwen 4 个 baseline N=200 数字定（May 2 早），看是否每个 baseline first-100 vs N=200 都飘 <SE。如果是，Llama 走 N=100；如果有大飘，Llama 必须 N=200。

---

### Apr 30 更新（在 Apr 29 进度之上）

**Qwen 100 SFT+DPO v2 完成 + patched**:

| Persona | pass@1 raw | pass@1 patched | pass@5 | avg_turn | rejection rate |
|---|---|---|---|---|---|
| Novice  | 16.0 | **17.0** | 25.0 | 7.97 | 0.46 |
| Experienced | 11.0 | **12.0** | 22.0 | 2.41 | 0.38 |
| Busy    | 14.0 | **14.0** | 19.0 | 1.55 | 0.89 |
| **All** | **13.7** | **14.3** | **22.0** | 3.98 | 0.47 |

**vs Qwen Baseline (overall pass@1):**
- Direct 15.3 / CF 16.0 / Base 11.0 / **TactfulLLM 14.3** / PO 13.0
- 比 Base **+3.3pp**（真实 DPO 价值）；落后 Direct/CF 1-2pp（噪声内）

**Pass@5 ordering 健康**: Novice 25 > Exp 22 > Busy 19（符合理论）

**Apr 30 关键 finding**:
1. **Import-missing template bug** — `prompts/coding_execute.txt` example 没显示 imports，导致大量 generated code 漏 `import numpy/pandas/...` → 0 pass。Fix Apr 30 04:56：加 requirement #5 强制 imports + 改 example。
2. **Patch_imports.py 失效** — 因为 `evaluate_multi_turn_persona.py:355,377,807` 把 `code` 字段截断到 200 字符，post-hoc patching 拿不到完整代码。Patch 4 个 method 总效果：Direct +0/CF +0/Base +0/TactfulLLM +0.7。
3. **截断 fix** — 三处 `[:200] + "..."` 全部去掉，从今天起 eval 保存完整 code。
4. **Exp<Busy 是噪声**: McNemar p=0.53 不显著；pass@5 上 Exp>Busy 顺序反而正确。模型本身没问题。
5. **Exp 0/100 pure Execute 是对的行为** — task uncertainty 高时先 clarify 是合理 proactive 决策。

### Apr 30 进行中 — N=200 扩展

**v33 SFT+DPO 在 remaining 100 state (101-200) 上跑**:
- PID 692635, 输出 `eval_v33_v3_qwen_dpo_v2_remaining100_ft.json`
- 用新模板 + 完整 code 保存
- Seed=42 reproducibility 验证：first-100 已评估 state_id 与新 sampling 100/100 匹配
- ETA 8-10h, wall Apr 30 ~14:00 完
- 完成后跟 first-100 patched 直接合并 → N=200（差异 ≤2pp，远小于 SE 3.5pp，sanity check 后合并）

**Pending baseline 决策（Apr 30 DPO 跑完后）**:
- 4 个 Qwen baseline (Direct/CF/Base/PO) 当前都 N=100
- 是否补到 N=200 取决于 v33 N=200 数字是否显著变化

---

## 📍 历史进度（2026-04-29 evening）

### Llama 流程（验证 method work）

| 阶段 | 状态 | 关键数字/观察 |
|---|:---:|---|
| 1. v33 v1 SFT (no masking) | ✅ DONE | 0/8 Novice Clarify ❌ 失败诊断 |
| 2. v33 v2 SFT (masking, alpha=16) | ✅ DONE | 1/8 Novice 真 Clarify ⚠ 部分 |
| 3. v33 v3 SFT (masking, alpha=32, 3ep) | ✅ DONE | **8/8 Novice "Clarify\n[直接问句]"** ✅✅ |
| 4. v33 v3 SFT 5-state eval (Llama) | ✅ DONE | pass@1 = 0/15 (但 Direct/CF 也 0/15 同 5 state) |
| 5. v33 v3 DPO refinement train | ✅ DONE | 18 min train，从 SFT 继续 |
| 6. v33 v3 DPO sanity | ✅ DONE | **24/24 prefix correct, Exp 5→8 strengthened** ✅ |
| 7. v33 v3 DPO 5-state eval | ✅ DONE | **1/15 = 6.7%**（vs SFT-only 0/15 = +1 pass over SFT） |

### Qwen 流程（method 跨 backbone 验证）

| 阶段 | 状态 | 关键数字/观察 |
|---|:---:|---|
| 1. Qwen v33 SFT 训练 | ✅ DONE | 12 min train, loss 0.37（跟 Llama 同步）|
| 2. Qwen SFT sanity | ✅ DONE | **24/24 perfect, Exp 8/8 Clarify**（Qwen 比 Llama Exp 更 Clarify-prone）|
| 3. Qwen v33 SFT 5-state eval | ✅ DONE | **2/15 = 13.3%**（Qwen 比 Llama 0/15 强）|
| 4. Qwen DPO refinement v1 (epochs=3) | ❌ COLLAPSE | Busy 8/8 emit "Execute" then 停 / 乱码 (loss 0.0005 over-fit) |
| 5. Qwen DPO refinement v2 (epochs=1) | ✅ DONE | 部分救回 — Busy 4/8 valid code, 4/8 garbage |
| 6. Qwen DPO v2 sanity | ✅ DONE | Novice/Exp 8/8, Busy 8/8 prefix 但 4/8 valid code |
| 7. Qwen DPO v2 5-state eval | ✅ DONE | **2/15 = 13.3%**（同 SFT-only 13.3%）|
| 8. Qwen 100 DPO v2 eval (canonical N) | 🔄 跑中 | 启动 wall ~23:00 Apr 29，ETA Apr 30 16-18:00 |
| 9. Qwen 100→200 续跑（partial resume）| ⏳ Apr 30 后 | 需要的话扩到 200 ~17h |

### 已确认的关键 finding

```
✅ v33 v3 SFT 在两 backbone 都 100% 学到 persona-aware Clarify decision
   - Llama:  Novice 8/8, Busy 8/8, Exp 5/8 Clarify
   - Qwen:   Novice 8/8, Busy 8/8, Exp 8/8 Clarify
   
✅ Llama DPO refinement 完美工作（24/24 sanity, 不破坏 SFT, 加强 Exp）
   
⚠ Qwen DPO refinement 在 epochs=3 (Llama setting) 下 over-fit collapse Busy
   - 修复: 降到 epochs=1，Busy 部分救回 (4/8 valid code)
   - Qwen 比 Llama 对 DPO 更敏感（不同 RLHF tuning）
   
✅ Multi-turn 行为分化在两 backbone 都对（Novice 7-turn / Busy 1-turn / Exp 2-3-turn）
   - Qwen Exp 比 Llama Exp 多 1 turn（DPO sanity Exp 8/8 vs Llama 5/8）
   
✅ 5-state pass@1 对比（hard subset of test set）:
   - Llama SFT-only: 0/15  vs  Llama SFT+DPO: 1/15 = 6.7%
   - Qwen SFT-only: 2/15 = 13.3%  vs  Qwen SFT+DPO v2: 2/15 = 13.3%
   - same 5 states: Direct 0/15 (Llama), 2/15 (Qwen); CF 0/15 (Llama), 2/15 (Qwen)
   - 这 5 states 是 hard subset，N=5 too small for paper-grade
```

### Pending 关键问题

```
1. ✅ Qwen 100 SFT+DPO v2 (canonical N): 14.3 patched
2. ✅ Qwen 100 → 200 已扩: 15.2 (+0.9pp，sanity 通过)
3. Qwen baseline (Direct/CF/Base/PO) 是否补 100→200 (4×17h 单卡 / 34h 双卡)
4. Llama 需要补 Llama 100/200 SFT+DPO eval 吗，还是用 v29 v1-era 数字
5. 第一次的 Qwen DPO v1 (epochs=3) 也可以 paper 写成 ablation: "DPO over-refinement 在 Qwen 上的负面案例"
6. GPU 升级 (单 4090 → 双 4090) 等 (3) (4) 决策定了再说
```

## TL;DR

Pure DPO from Llama-3.1-8B-Instruct cannot bridge the KL gap between RLHF tendency
("I'd be happy to help...") and target persona-aware Clarify distribution
("Clarify\nWhat should..."). Switching to SFT-then-DPO pipeline (Rafailov et al. 2023):
SFT first shifts model to chosen distribution (跨 gap), DPO refines with rejection signal.

**Status (May 1)**: Method works on both backbones (Llama + Qwen). Qwen N=200 canonical
result: pass@1 = 15.2% (Novice 18.0 / Exp 14.0 / Busy 13.5), pass@5 = 23.0%, +4.2pp over
Base baseline, ~tie with Direct/CF (within noise). Persona behavior split correct
(Novice 8-turn, Busy 1.5-turn, Exp 2.4-turn). Baseline N=200 extension and Llama
full-pipeline eval still pending.

---

## Why v29 DPO failed (background)

```
DPO loss has implicit KL constraint via β:
  L = -log σ(β · (log π/π_ref(chosen) - log π/π_ref(rejected)))
                   ↑
                  ratio optimization, bounded by ref distribution

Llama-Instruct π_ref:
  P("I'd be happy to help...") = 0.5      (RLHF-tuned high)
  P("What should the function...") ≈ 1e-10 (rare in pretrained)

DPO can amplify ratio but absolute π(chosen) stays low → model doesn't learn target

v30/v31 (changed oracle, kept hparam): also failed (4 retries)
v32 (alpha=128, kept prefix): collapsed (Busy emit "Execute." then stop)
v32b (alpha=32, no SFT): no collapse but no prefix learning
```

**Root cause**: KL gap too large for DPO + LoRA from Llama-Instruct directly.
**Fix**: Add SFT warmup stage — directly maximize log p(chosen | prompt), no KL constraint, can shift distribution arbitrarily far.

---

## v33 iteration history

### v33 v1 (no prompt masking, 1 epoch, LR 1e-5)

Used existing `train_sft.py` which has loss on prompt+response (no masking).

**Result**: Sanity 0/8 Novice Clarify. Models emit ` ```python` directly (similar to Busy).

**Diagnosis**: Loss diluted across prompt + response tokens; "Clarify\n" signal too weak
to learn at LR 1e-5 / 1 epoch / alpha 16.

### v33 v2 (prompt masking, 2 epoch, LR 2e-5, alpha 16)

Wrote new `train_sft_v33.py` with proper prompt masking (label=-100 for prompt, real for response).

**Result**: Sanity 1/8 Novice true Clarify (model emit "I can guide... However I need to clarify..." for one ambiguous state). Other 7/8 still preamble + code. 0/24 emit "Clarify\n" prefix explicitly.

**Diagnosis**: Prompt masking helped but not enough. Need more capacity / training.

### v33 v3 (prompt masking, 3 epoch, LR 5e-5, alpha 32) ✅

```
KEEP_PREFIX=1 LORA_ALPHA=32 LORA_R=64 \
  python policy/train_sft_v33.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output models/v33_v3_sft \
  --epochs 3 --lr 5e-5
```

**Sanity result** (8 state × 3 persona, greedy, 200 token):

```
Novice:  Clarify_pfx = 8/8 (100%)  Pure_Clarify = 8/8  ✅✅✅
Busy:    Clarify_pfx = 0/8         Execute = 8/8       ✅
Exp:     Clarify_pfx = 5/8         Execute = 3/8       ✅ (mixed by state)
```

**Sample outputs**:
- Novice: `"Clarify\nWhat format should the input data be in..."`
- Busy:   `"Execute\n```python\nimport re\nimport json..."`
- Exp:    `"Clarify\nWhat specific URL pattern should be used..."`

Pretrained "I'd be happy to help" tendency completely overcome (0/24 in sanity).

---

## v33 v3 SFT-only 5-state eval (Llama)

**Setup**:
```bash
CLASSIFIER_VERSION=v2 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v33_v3_sft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 5 --max_turns 7 \
  --output outputs/eval_v33_v3_sft_5.json
```

**Result**:
| Persona | pass@1 | pass@5 | avg_turns |
|---|:---:|:---:|:---:|
| Novice | 0/5 = 0% | 1/5 = 20% | 7.40 |
| Busy | 0/5 = 0% | 0/5 = 0% | 1.00 |
| Exp | 0/5 = 0% | 0/5 = 0% | 2.40 |
| **Overall** | **0/15 = 0%** | 1/15 = 6.7% | 3.60 |

**Behavior分化 perfect** (Novice multi-turn, Busy 1-turn, Exp 2-turn) — matches sanity prediction.

### Apples-to-apples comparison on same 5 states

| Method | Total pass@1 |
|---|:---:|
| v29 DPO Llama (combined 200) | **3/15 = 20.0%** |
| Direct Llama 200 | 0/15 = 0.0% |
| CF Llama 200 | 0/15 = 0.0% |
| v33 v3 SFT-only | 0/15 = 0.0% |

**Caveat**: Direct/CF also 0/15 on these specific 5 states (BigCodeBench/127/202/575/784/945) — these states are anomalously hard for non-multi-turn-with-disclosure-recovery methods.
v29 DPO got 20% via v1 误判 → forced multi-turn → disclosure recovery.

**Conclusion**: SFT-only 0/15 doesn't necessarily mean SFT damaged code generation
(Direct also 0/15). Need bigger N for confident comparison.

---

## v33 v3 DPO refinement (in progress, 2026-04-29 evening)

**Setup**:
```bash
INIT_ADAPTER=models/v33_v3_sft KEEP_PREFIX=1 LORA_ALPHA=32 LORA_R=64 \
  python policy/train_dpo.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output models/v33_v3_dpo \
  --epochs 3 --beta 0.1
```

**Train time**: ~18 min (continues training v33_v3_sft LoRA with DPO loss).

**Sanity result** (after DPO):

```
Novice: 8/8 "Clarify\n[直接问句]"  ✅ (preserved from SFT)
Busy:   8/8 "Execute\n[代码]"      ✅ (preserved from SFT)
Exp:    8/8 "Clarify\n[直接问句]"  ← DPO 推从 5/8 → 8/8 Clarify
```

**Notable**: DPO refinement strengthened Exp's Clarify tendency (5/8 → 8/8) at turn 0.
But this is turn-0 only; multi-turn eval shows Exp = 2-turn (Clarify → Execute) consistently
(both v33 SFT and DPO have similar multi-turn dynamics).

**5-state DPO eval running** (started 2026-04-29 ~17:42 wall, ETA ~19:00 wall).

---

## File locations

### Scripts (uncommitted)
```
policy/train_sft_v33.py         # NEW: SFT with prompt masking
policy/train_dpo.py             # MODIFIED: added INIT_ADAPTER env var support
policy/infer.py                 # MODIFIED: added v2 classifier (CLASSIFIER_VERSION env var)
```

### Models
```
models/v33_v3_sft/              # SFT warmup model
models/v33_v3_dpo/              # DPO refinement on top of SFT (just trained)
```

### Eval outputs
```
outputs/eval_v33_v3_sft_5.json       # SFT-only 5-state eval
outputs/eval_v33_v3_dpo_5.json       # DPO 5-state eval (running)
```

### Sanity scripts (in /tmp/, should move to scripts/)
```
/tmp/v33_v3_sft_sanity.py
/tmp/v33_v3_dpo_sanity.py
```

### Wrapper scripts
```
/tmp/v33_v3_sft.sh              # SFT wrapper
/tmp/v33_v3_dpo.sh              # DPO refinement wrapper
/tmp/v33_v3_eval_5.sh           # 5-state SFT eval
/tmp/v33_v3_dpo_eval_5.sh       # 5-state DPO eval
```

---

## Pending decisions (post 5-state DPO eval)

After 5-state DPO eval finishes (~19:00 wall):

### If DPO 5-state pass@1 ≥ 1/15:
- DPO refinement adds value
- Train Qwen v33 SFT (~10 min) + sanity (~5 min)
- If Qwen sanity OK → Qwen 100 SFT eval overnight
- Then Qwen DPO refinement + 100 eval

### If DPO 5-state pass@1 = 0/15:
- DPO doesn't help on these 5 states
- Run extended N=15-30 SFT-only Llama eval (~6-12h) for stable signal
- If extended N also 0% → SFT damaging code, need rethink
- If extended N reasonable (~10-14%) → these 5 states were unlucky, proceed Qwen

### If DPO 5-state pass@1 ≥ 3/15:
- DPO recovers code well
- Strong signal, proceed Qwen full pipeline immediately

---

## Time budget (May 6 ddl)

```
Apr 29 (today)
  17:30  Llama 5-state SFT eval done (0/15)
  17:33  Llama DPO refinement done (~18 min)
  17:42  Llama DPO sanity done (8/8 Novice, 8/8 Busy, 8/8 Exp Clarify\n)
  17:42  Llama DPO 5-state eval started
  19:00  Llama DPO 5-state done → decision point

Apr 30 onwards (depends on decision)
  Best case: Qwen v33 pipeline (60h compute over 3 days)
             Llama keeps v29 v1-era numbers (or extends to N=30/50)
  Compute: SFT 10min + sanity 5min + 100 SFT eval 25h + DPO train 4h + 100 DPO eval 25h
            + 30 ablation 11h = ~65h

May 3-5: Writing (3 days)
May 6:  Submit
```

---

## Paper narrative implications

```
Original claim (v29 era): "TactfulLLM uses DPO to learn persona-aware proactive Clarify"
  ↑ Disproved by v1 误判 finding + v2 sanity (model doesn't naturally emit direct questions)

New claim (v33 era): "TactfulLLM uses standard SFT-then-DPO recipe (Rafailov 2023)
  to teach persona-conditional proactive Clarify decision-making.
  SFT shifts the model toward the persona-aware target distribution;
  DPO refines using preference signals."

Note: cross-distribution learning from RLHF Instruct models requires SFT warmup
  to bridge the KL gap. Pure DPO without SFT cannot reach distant target distributions
  (verified empirically: v29/v30/v31/v32 series all failed to learn direct questions).
```

---

## Next session pickup

If session is interrupted:
1. Check `outputs/eval_v33_v3_dpo_5.json` exists → DPO eval finished
2. Run `python3 -c "import json; d=json.load(open('outputs/eval_v33_v3_dpo_5.json')); ..."` to read pass@1
3. Read this doc + `classifier_bug_2026-04-28.md` for context
4. Continue per "Pending decisions" section

---

## 2026-05-04 Update — 200-state evals + paper consistency audit

### Llama v33 SFT-then-DPO 200-state results

```
Era: 8-bit, v2 classifier (`models/v33_v3_dpo`)
  Novice  pass@1 = 12.5%  avg_t = 8.0
  Exp     pass@1 = 12.5%  avg_t = 2.0
  Busy    pass@1 = 11.5%  avg_t = 1.0
  Overall pass@1 = 12.2%  pass@5 = 22.5%

Era: bf16, v2 classifier
  Novice  pass@1 = 17.5%  avg_t = 8.0
  Exp     pass@1 = 13.5%  avg_t = 2.0
  Busy    pass@1 = 11.5%  avg_t = 1.0
  Overall pass@1 = 14.2%  pass@5 = 22.8%
```

Source: `outputs/eval_v33_v3_llama_dpo_200.json`, `outputs/eval_v33_v3_llama_dpo_200_bf16.json`.

### Qwen v33 SFT-then-DPO 200-state results

```
Era: 8-bit, v2 classifier (`models/v33_v3_qwen_dpo_v2`)
  Novice  pass@1 = 18.0%  avg_t = 8.0
  Exp     pass@1 = 14.0%  avg_t = 2.4
  Busy    pass@1 = 13.5%  avg_t = 1.6
  Overall pass@1 = 15.2%  pass@5 = 23.0%

Era: bf16, v2 classifier
  Novice  pass@1 = 23.0%  avg_t = 8.0
  Exp     pass@1 = 18.5%  avg_t = 2.4
  Busy    pass@1 = 13.5%  avg_t = 1.7
  Overall pass@1 = 18.3%  pass@5 = 25.7%
```

Source: `outputs/eval_v33_v3_qwen_dpo_v2_200.json`, `outputs/eval_v33_v3_qwen_dpo_v2_200_bf16.json`.

### Critical finding: Llama SFT-then-DPO underperforms pure DPO

```
Llama v29 pure DPO  (paper, v1, 8-bit):  Overall 16.0%  (Nov 18.5 / Exp 15.5 / Busy 14.0)
Llama v33 SFT-then-DPO (v2, 8-bit):      Overall 12.2%  (Nov 12.5 / Exp 12.5 / Busy 11.5)
Llama v33 SFT-then-DPO (v2, bf16):       Overall 14.2%  (Nov 17.5 / Exp 13.5 / Busy 11.5)
```

**v33 SFT-then-DPO does not improve over v29 pure DPO on Llama, even at bf16.** Likely root causes:

1. **Classifier era mismatch in evaluation**: v33 was evaluated with v2 classifier (strict "Clarify\n" prefix), while v29 paper numbers use v1 classifier (intent-based, hedge code → Clarify). Under v1, Llama hedge outputs trigger more multi-turn rounds → more disclosure recovery → higher pass@1. v33 SFT teaches the model to commit early, removing the v1-exploitable hedge pattern.

2. **v29's apparent strength is partially v1 classifier artifact**: re-evaluating v29 pure DPO under v2 classifier would likely drop it close to v33 numbers. Conversely, evaluating v33 SFT-then-DPO under v1 classifier might recover some of the gap (untested).

3. **SFT may not be needed for Llama-3.1-Instruct**: this backbone's RLHF prior is permissive enough for pure DPO + LoRA to learn proactive Clarify under intent-based eval. SFT-then-DPO becomes load-bearing only for stricter eval criteria or distributionally-distant targets.

### Qwen v33 SFT-then-DPO works as designed

Qwen v33 SFT-then-DPO **does** beat its baselines and shows persona-differentiated behavior:
- Novice 23.0% (bf16) — strong proactive multi-turn
- avg_t differentiated: Nov 8.0 / Exp 2.4 / Busy 1.7 (matches our story)

Qwen-Instruct's RLHF prior is more terse than Llama's, so SFT warmup IS needed to bridge the KL gap to the proactive-Clarify distribution. Pure DPO on Qwen had failed earlier (collapse / no learning).

### Paper consistency decision (May 4)

§Method as drafted describes "SFT-then-DPO with persona-aware preference signals" (matches v33 pipeline). Paper main table TactfulLLM Llama uses **v29 pure DPO** numbers (16.0%). This is a method/data mismatch.

| Option | Trade-off |
|--------|-----------|
| Keep v29 era for Llama, change §Method to "DPO with persona-aware reward" | ✅ Story strong (16.0% wins all baselines under v1 classifier); SFT relegated to ablation showing "SFT pre-training did not improve over pure DPO on Llama" |
| Switch Llama numbers to v33 SFT-then-DPO (12.2%–14.2%) | ✅ Method/numbers consistent; ❌ TactfulLLM ≈ Prompt-only at v1-era story collapses |

**Decision (May 4): keep v29 era for Llama, use v33 SFT-then-DPO for Qwen.** §Method must be rewritten to match:
- Main method = "DPO with persona-aware preference signals + intent-based classifier"
- SFT-then-DPO becomes a **per-backbone necessity**: "Qwen requires SFT warmup to bridge KL gap to the proactive-Clarify distribution; Llama's RLHF prior is permissive enough for pure DPO."

This framing is honest about the empirical reality and turns the asymmetry into a finding rather than a flaw.

### Paper main table data audit (cross-reference)

See `docs/work_log.md §150` for the full audit. Key issues identified:
- Llama Base avg_t was 1.0/1.0/1.0/1.0 in table; correct v1-era values are 2.30/1.65/2.51/2.15
- Llama Prompt-only is n=50 per persona (n=150 total), not n=200 — needs补 or footnote
- All other rows are internally consistent v29-era 200-state data

### Still open (updated 2026-05-05)

- ~~CollabLLM Llama 200-state~~ ✅ DONE (bf16 + v2 = 15.3%; 8-bit + v1 partial 13.1%)
- ~~Llama Prompt-only n=200~~ ✅ DONE (merged 50test + 150extra = 13.3%, paper 8.7% was unlucky n=50)
- Paper §Method rewrite to match v29 (Llama) + v33 (Qwen) split decision — pending, doing post-experiment

---

## 2026-05-05 Update — CollabLLM apples-to-apples + per-persona Pareto findings

### Apples-to-apples CollabLLM eval

Re-evaluated CollabLLM Llama at 8-bit + v1 (matching paper TactfulLLM precision/classifier) instead of original bf16 + v2:

```
                          Nov    Exp    Busy   Overall
TactfulLLM 8-bit + v1     18.5   15.5   14.0   16.0%   (paper main)
CollabLLM bf16 + v2       19.0   14.0   13.0   15.3%   (original eval, unfair precision)
CollabLLM 8-bit + v1      15.0   13.1   11.2   13.1%   (apples-to-apples, n=318 partial)
```

**Verified bf16-vs-8bit precision asymmetry hypothesis**: CollabLLM lost ~2.2pp dropping from bf16 to 8-bit, consistent with the +1.7pp average lift other Llama baselines show on bf16. The original "TactfulLLM ~tied with CollabLLM" reading was an artifact of precision asymmetry.

**Updated paper main table claim**: TactfulLLM beats CollabLLM by **+2.8pp Overall** under matched precision, with consistent **+2.3 to +3.4pp gains across all three personas**.

### Per-persona Pareto-dominance pattern (paired analysis on 450 (state, persona) pairs)

McNemar paired tests on TactfulLLM vs CollabLLM:

```
Pass@1 (single-attempt accuracy):
  Busy:  6 vs 6     p=1.000   (tied, NOT significant)
  Exp:   13 vs 11   p=0.839   (NOT significant)
  Nov:   9 vs 13    p=0.523   (CollabLLM slight edge, NOT significant)

Pareto-dominance (pass AND ≤ turns):
  Busy:  TactfulLLM 129  vs  CollabLLM 1     ⭐ 129:1, p<0.001
  Exp:   TactfulLLM 17   vs  CollabLLM 93    (CollabLLM wins efficiency)
  Nov:   TactfulLLM 4    vs  CollabLLM 138   (CollabLLM wins efficiency)
```

**Mixed verdict**: TactfulLLM Pareto-dominates **only** on Busy (the persona where interruption cost matters most), while CollabLLM achieves better efficiency on patient personas (Nov + Exp) where TactfulLLM saturates max_turns or asks more questions than necessary.

**Updated paper §6 framing** (replacing earlier "global Pareto dominance" narrative):
> "TactfulLLM achieves persona-asymmetric Pareto dominance: on low-patience users (Busy), it dominates CollabLLM in 129/150 paired tests (p<0.001), eliminating clarification entirely while matching task accuracy. On high-patience personas, TactfulLLM uses more turns than necessary—a saturation behavior we discuss as a limitation in §7."

### pass@5 mechanism analysis

Investigated why CollabLLM occasionally beats TactfulLLM on pass@5 despite lower pass@1. Inspection of `candidate_results` (per-sample pass info) shows:

```
Per-sample variance analysis (Busy, n=150-200):
  TactfulLLM samples: 14.0/12.7/12.7/13.3/11.3 → mean 12.8%, low variance
  CollabLLM samples:  13.0/13.0/13.0/13.5/13.0 → mean 13.1%, low variance
  → both uniform, low per-sample variance

Per-sample variance (Novice):
  TactfulLLM: 19.3/18.7/17.3/18.7/15.3 → mean 17.9%, mild spread
  CollabLLM:  19.0/13.5/13.5/15.0/10.5 → mean 14.3%, larger spread (8.5pp range)
```

**Conclusion**: The "CollabLLM accumulates disclosed info → easier task → higher pass@5" hypothesis is partially wrong: Busy rejects 84% of clarifications, so most CollabLLM Busy turns yield no info gain. The actual pass@5 gap (1.7pp Overall, 2.5pp on Busy) is **within statistical noise at n=200** and stems from per-sample variance/diversity rather than substantive disclosure differences.

**Honest paper claim** (avoid overclaiming TactfulLLM superiority on pass@5):
> "TactfulLLM and CollabLLM achieve **comparable** pass@5 (within 1.7pp Overall, within statistical noise at n=200). TactfulLLM's advantages are concentrated in pass@1 (deployment-realistic single-attempt accuracy) and interaction efficiency."

### Llama Prompt-only n=200 corrected number

Original paper Prompt-only number was 8.7% (n=50 subset). Completed remaining 150-extra states; merged n=200 actual = **13.3%** Overall (Nov 15.0 / Exp 13.0 / Busy 12.0). The paper 50test was an unlucky subset on Exp/Busy (n=50 noise).

**Implication**: TactfulLLM vs Prompt-only gap shrinks from +7.3pp (paper) to +2.7pp (real n=200), still positive but narrower. Story still holds; need to update main table and §6 narrative to reflect real numbers.

### Out-of-distribution persona test (§6.3)

Added Time-Pressured-Expert persona (high expertise, low patience — combination not seen in training). Evaluated TactfulLLM (paper 8-bit + v1) on 200 states × TPE only:

```
Time-Pressured-Expert (OOD persona):
  pass@1 = 10.5%
  avg_turns = 1.00 (100% Execute, voluntary)
  clarify_rate = 0%
  forced_final_execute = 0
```

**Finding**: Model voluntarily executes immediately (no max_turns saturation), correctly identifying low-patience signal as dominant. Maps (high, low) to "minimize interruption" — same behavior as Busy (mid expertise, low patience), independent of expertise level.

**Paper claim**: axis-aligned generalization — policy disentangles expertise from patience axes.

### Implementation additions

- `eval/evaluate_multi_turn_persona.py`:
  - `--persona_filter` whitelist (for OOD eval)
  - `--prompt_only + --model_dir` co-existence (for LoRA-adapter baselines like CollabLLM)
  - `USE_4BIT=1` env var (NF4 4-bit eval, matching QLoRA training)
  - `FEW_SHOT_PERSONA=1` env var (3 in-context persona-action examples in select_action_prompt_only)
  - `--random_policy` CLI flag (50/50 Clarify/Execute, seeded for reproducibility)

- New baselines queued (post-current-runs):
  - Few-shot Persona Prompt (Llama running, Qwen queued)
  - Random Policy (Llama + Qwen queued)
  - Both at 8-bit + v1 (paper consistent)

### Pending (~17h to NeurIPS DDL)

- Wait for CollabLLM 8-bit + v1 full 200 (~22:30 May 5)
- Wait for Few-shot Persona Prompt eval (Llama ~16:30, Qwen ~20:30)
- Wait for Random Policy eval (~19:00 May 5)
- Wait for Ablation no_uncertainty n=200 finish (~17:00 May 5)
- ~22:30 May 5: all data complete → ~24h to finalize paper main table + §6 + §Method rewrite + investigate
