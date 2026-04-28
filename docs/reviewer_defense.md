# Reviewer Defense Checklist

> 2026-04-25。按杀伤力分 Tier,每条注明状态。
>
> **Status**:`✓ 已防` / `⚠ 部分` / `✗ 无防御` / `📋 需补` / `🚫 不做` / `🏃 in progress`

---

## 快速索引

| Tier | 条数 | 还没防住 |
|---|:---:|:---:|
| 1 必须硬回应 | 7 | 4 |
| 2 appendix 级 | 5 | 3 |
| 3 Limitations 一带 | 6 | 全部 |

**两个 Tier 1 新增(2026-04-25 评审后)**:T1-6 novelty positioning、T1-7 threshold baseline。

---

## Tier 1:必须硬回应

### T1-1. User simulator 是 gpt-4o-mini —— 测的是 persona adaptation 还是 roleplay artifact? ✗ → 📋

Train 和 eval 共用一个 LLM。行为可能是 gpt-4o-mini 的 prompt interpretation,不是真实 persona effect。
**补**:Claude Haiku 做 simulator,50-state 子集跑 TactfulLLM 看行为分化是否保留。**必做**,半天 + ~$50。

### T1-2. 部署时 persona 从哪来? ✗ → 📋

Eval 时 persona 是 ground-truth 给定。真实用户没 label。
**补**:Limitations 段 + Discussion 一段 "persona inference as future work,本文 scope 限于 persona-conditional policy"。**不临时做 persona classifier**(引新变量更糟)。

### T1-3. ~~SFT-on-rules baseline~~ 🚫

被 T1-7 threshold baseline 替代,ROI 更高。Method 章节仍需写清 "behavior-first 决定 label,reward 决定选哪条轨迹" 的分工。

### T1-4. 只测 Llama-3.1-8B ⚠ → 🏃

单 backbone 自动质疑。
**状态**:v31.4 Qwen 今晚出结果。只要有 50-state 数字就挂主表。**必做**。

### T1-5. Masking 协议是你们自己设计的 ⚠ → 📋

选 output_format / input_constraints / validation_rules 这三类,正好是 clarify 能恢复的 —— 像是设计容易赢的游戏。
**部分防**:Exp 3 DS-1000 换 domain。
**补**:Method 写明选这三类的理由(真实用户最常省略的信息类型);Limitations 坦白 mask-field 先验 bias。

### T1-6. 【新增 · 最大漏】Novelty depth:"state + reward + DPO 的工程扩展" ✗ → 📋

> "This looks like a reasonable engineering extension of prior proactive LLM work. Where is the fundamentally new learning insight?"

这是 NeurIPS 最致命的 framing 攻击。11 天里不可能新开算法,但可以**重新 framing**。

**三个要当算法贡献来写的点**:
1. **Calibrated proactivity as meta-decision policy** — 不学 clarification content,学 "when to stop asking"(decision-boundary learning)。Figure 1 要突出这点。
2. **Behavior-first pair construction** — 算法贡献,不是工程细节。把 action label(语义决定)和 trajectory selection(reward 决定)**解耦**的 preference construction 方法。给它名字,method 里单独一段。
3. **Cross-axis composition(persona × uncertainty)** — 两轴正交信号组合,Ablation §5.4 证明互不可替代。Intro 提前 preview。

**补**:重写 intro Contributions bullets,method 加 "Behavior-First Pair Construction" 子节,不写代码。**必做**。

### T1-7. 【新增】缺 threshold baseline ✗ → 📋

现有 baselines 都在"方法差异"轴(no clarify / prompt / always / no training),**没一个在"学习 vs handcrafted heuristic"轴**。Reviewer 会问 "是不是就一个 threshold?"。

**做法**:
```
Threshold baseline(和 TactfulLLM 同信号):
  if task_uncertainty > θ:  clarify K=2 turns
  else:                     execute
  θ ∈ {0.3, 0.5, 0.7},报 best
```
核心论点:**learned policy 在用同一信号时仍赢 handcrafted threshold**。1 天 code + 1 天 eval。**必做**。

---

## Tier 2:appendix 级

### T2-1. Prompt-only baseline 真调过吗? ✗ → 📋
Appendix 附完整 system prompt + "tried X variants, best reported"。半天。

### T2-2. Uncertainty = n_masked_items/5 太粗糙 ✗ → 📋
Limitations 坦白 + 提 model-estimated uncertainty 作 future work。

### T2-3. Novice turn 7 被 forced execute = artifact? ⚠ → 📋
Eval Setup 一段:forced execute 防死循环,同时报 rejection rate 显示用户早已失去耐心 —— artifact 在惩罚我们自己。

### T2-4. 为什么不比 CollabLLM? ✗ → 🚫
不实现。Related Work 一句话:CollabLLM 面向 open-ended dialogue,与 spec-driven coding 不直接可比。

### T2-5. 没 qualitative examples ✗ → 📋
**必做**。Appendix 3 success + 3 failure,每 persona 一组。半天。这是最常被抱怨的点。

### T2-6. gpt-4o-mini 数据泄露? ⚠ → 📋
gpt-4o-mini cutoff 2023-10,BigCodeBench 2024 发布。Appendix 一句话 release dates。10 分钟。

---

## Tier 3:Limitations 打包(一句话一条)

只 coding / 3 personas / 无人类 eval / reward hyperparam 无 sensitivity / latency 未报 / 500 pairs 饱和(一句说明)。

**策略**:打包进 Limitations 章节,我们坦白 vs reviewer 挖出来,分差很大。

---

## 待补 artifacts(按 ROI)

| # | 动作 | 防 | 成本 | 状态 |
|---|---|---|:---:|:---:|
| 1 | v31.4 Qwen 跑完进主表 | T1-4 | 0 | 🏃 |
| 2 | **Novelty reframing(intro + method)** | **T1-6** | **1 天写作** | 📋 |
| 3 | **Threshold baseline** | **T1-7** | **2 天** | 📋 |
| 4 | DS-1000 Exp 3 | T1-5 + 同分布质疑 | 4 天 | 📋 |
| 5 | Qualitative examples | T2-5 | 0.5 天 | 📋 |
| 6 | Cross-simulator check(Claude Haiku) | T1-1 | 0.5 天 | 📋 |
| 7 | Limitations 章节 | T1-2、T1-5、T3 | 0.5 天 | 📋 |
| 8 | Prompt-only variants 补记录 | T2-1 | 0.5 天 | 📋 |
| 9 | Release dates 声明 | T2-6 | 10 min | 📋 |

---

## 11 天时间分配

| 任务 | 天 | 类型 |
|---|:---:|---|
| DS-1000 Exp 3 | 4 | 实验 |
| Threshold baseline | 2 | 实验 |
| Cross-simulator + Qualitative + Limitations | 1.5 | 混合 |
| **论文写作(含 1 天 novelty reframe)** | **3.5** | **写作** |
| 5/6 Submit | 0 | |

紧但能赶。**novelty reframing 必须单独留 1 天**,以前 timeline 没这项。

---

## Intro / Discussion 要埋的关键词

Reviewer 读 intro 会对号入座,主动说出来就不会单独挑。

- "simulated users" / "user simulator"(别假装真人)
- "persona-conditional policy"(persona 是输入,不是 inferred)
- "spec-driven coding"(划定 scope)
- "calibrated proactivity as meta-decision policy"(T1-6 novelty framing)
- "behavior-first preference construction with reward-guided trajectory selection"(T1-6 算法点命名)
- "curated masking protocol"(别假装 natural)
