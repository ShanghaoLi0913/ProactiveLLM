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

## 📍 当前进度（2026-05-01）

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
