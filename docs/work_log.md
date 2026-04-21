# 工作记录

> 最新记录在前

---

## 2026-04-21

### 82. Canonical test set 审计 + Exp1/Exp2 表格不一致排查

**起因**：用户看到 Experiment 1 主表（"200 test tasks"）和 Experiment 2 matched 表（"151 matched seeds"）数字不一致（如 Direct Execution：7.3% vs 14.1%，差 2×），问"是不是同一个测试集"。

**审计结论**：canonical 测试集是 `data/seeds/test_states_v29_eval_200.jsonl`（200 状态）。但：

| 方法 | canonical 覆盖 | 缺 | 备注 |
|---|---|---|---|
| Direct Execution | 200 ✓ | 0 | `eval_v29_direct_execution_200.json` |
| Oracle (Full Query) | 200 ✓ | 0 | `eval_v29_oracle_200.json` |
| Ideal Disclosed v1 | 200 ✓ | 0 | `eval_v29_ideal_disclosed_200.json` |
| **Ideal Disclosed v2** | 198/200 | 2 | 仍在跑，~5 min 完结 |
| **TactfulLLM** | 200 ✓ | 0 | 分散 3 文件（50test+150extra+100states 里 3 个）|
| **Clarify-first** | 50 | **150** | 只有 `eval_v29_clarify_first_50test.json` |
| **Prompt-only** | 50 | **150** | 只有 `eval_v29_prompt_only_50test.json` |
| **Base LLM** | 93 | **107** | 50test=50 + 150extra_remaining=42 + 老 20-seed=3 |

**Exp1 表数据源真相**（caption 误写 "200 test tasks"）：

| 方法 | 表中数字（Nov/Exp/Busy） | 实际源 |
|---|---|---|
| Direct | 6.0/8.0/8.0 | `50test` 文件 ✓ 完全吻合 |
| Clarify-first | 8.0/12.0/8.0 | `50test` ✓ |
| Prompt-only | 14.0/6.0/6.0 | `50test` ✓ |
| Base LLM | 12.5/12.5/13.0 | 不纯，疑似手拼 |
| TactfulLLM | 18.5/15.5/14.0 | 早期快照 |

**同一 Direct Execution 两表差 2×** 的原因：Exp1 用 50-state 子集 (`direct_execution_50test.json`)，Exp2 用 200-matched-151。50test 恰好难，不宜当主表。

**⚠ 关于老的 20-seed 测试集 `test_states_v29_eval.jsonl`**：只和 canonical-200 重叠 3 个，17 个在外。三个 eval 文件使用它：
- `eval_v29_100states.json`（17 outside，但 TactfulLLM 另外两文件已覆盖这 17 的替代品）
- `eval_v29_oracle_50test.json`（已被 200 文件取代）
- `eval_v29_base_llama.json`（需要补跑 canonical 版本）

### 83. 泄漏虚惊一场 — v29 实际无泄漏

审计 train/eval 是否泄漏：

```
canonical splits (470 masked tasks, 互不重叠):
    train: 376    val: 47    test: 47

eval_200 文件构造：
  ├─ 来自 train: 67    （⚠ 文件级重叠）
  ├─ 来自 val:   13
  ├─ 来自 test:   6
  └─ 孤儿:      114    （不在 470-masked 池里，单独跑 masking）

v29 实际训练 (prefs_v29_100states.jsonl): 107 个任务，id 全部 < 110
eval_200: id 全部 ≥ 111
v29-actually-trained ∩ eval_200 = 0  ← 零泄漏
```

**所有 200 eval 任务都有 masked_fields，schema 与训练一致。** v29 paper 可以放心报 "200 held-out tasks, zero overlap with 107 training tasks"。

**潜在风险（v29 不受影响）**：v30+ 若扩大到 train_split 全集 376，那 67 个文件级重叠会变成真泄漏——须先修 eval_200。

### 84. 项目清理 commit

`git commit 9a3b633`：
- 把 61MB v29 轨迹日志从嵌套的 `data/data/logs/` 移到 `data/logs/`（生成器文档的位置），并修复两个硬编码路径（`analyze_clarify_samples.py`、`generate_v5_balanced_prefs.py`）
- 归档 5 个 Exp1 Part-2 分析脚本到 `scripts/analysis/`：
  - `extract_reference_turns.py` — 提取 v29 per-(state, persona) mainline 轮数
  - `v29_traj_stats.py` — 轨迹轮次全视角统计（per-trajectory、per-(state,persona)、action 分布）
  - `perTask_turn_table.py` — per-task 轮次对比表
  - `plot_turn_lines.py` — Exp1 Part2 折线图
  - `plot_interaction_quality.py` — 交互质量双图（Clarification Success Rate + Turn 分布 vs tolerance budget）
- 删除根目录污染：`qq_plot.png`、`key4_*.txt`、`test_output.json`、`test_data_2/`
- 清理 `.gitignore` 过时条目

### 85. Ideal Disclosed v2 完成 (200/200) + Exp2 表升级到 canonical-200

PID 3648（04:28 启动，耗时 ~5h），**200/200 完成**：

| 方法 (canonical-200) | pass@1 | pass@5 | avg turns |
|---|---|---|---|
| Oracle | 20.0% | 28.0% | 1.0 |
| **Ideal Disclosed v2** | **16.0%** | **27.0%** | 1.0 |
| Ideal Disclosed v1 | 13.5% | 24.5% | 1.0 |
| Masked Direct | 12.3% | 18.5% | 1.0 |

v2 vs v1：pass@1 +2.5pp、pass@5 +2.5pp。OGR：(16.0-12.3)/(20.0-12.3) = **48%**（v1 是 7%）——bullet 格式确实多传了一半 gap 的信息。

**Experiment 2 表重算（canonical-200，取代旧 151-matched）**：

| Group | Condition | pass@1 | pass@5 | Δ | OGR % | Disc. |
|---|---|---|---|---|---|---|
| Bounds | Masked Direct | 12.3 | 18.5 | -- | 0 | 0.00 |
|  | Full Query | 20.0 | 28.0 | +7.7 | 100 | n/a |
| TactfulLLM | Overall | 16.0 | 23.5 | +3.7 | 48 | 0.56 |
|  | Novice | 18.5 | 25.5 | +7.0 | 82 | 0.89 |
|  | Experienced | 15.5 | 25.0 | +3.0 | 40 | 0.78 |
|  | Busy | 14.0 | 20.0 | +1.0 | 14 | 0.00 |
| Oracle | Ideal Disclosed | 16.0 | 27.0 | +3.7 | 48 | 1.00 |

**最强 finding**：TactfulLLM Overall 和 Ideal Disclosed v2 在 200 canonical 上 pass@1 **精确重合 16.0%**。600 matched trials McNemar：b=43、c=43、p≈1.00 → "approach clarification ceiling" 升级为 "fully matches"。

Per-persona OGR 分层变清晰：Novice 82% → Exp 40% → Busy 14%（151-matched 时 Busy=0 现在 =14，不再退化）。

### 86. Exp1 Part-2: 轮次合理性 metric 设计中断

目标：除 avg_turns 和 rejection_rate 之外，设计"轮次是否合理"的 metric。用户要的 reference 来自**实际 v29 轨迹**。

**v29 轨迹统计**（`data/logs/traj_v29_100states_combined.jsonl`，1527 条轨迹 / 109 训练任务）：

| Persona | 按 trajectory 粒度 mean | 分布 |
|---|---|---|
| Busy | 2.25 | {2: 321, 3: 107} |
| Experienced | 2.80 | {2: 109, 3: 426} |
| Novice | 3.64 | {2: 109, 3: 146, 4: 176, 5: 105, 6: 28} |

**但：v29 轨迹覆盖 task 0-108，eval_200 是 111+，完全不重叠**——无法提供 per-(eval_task, persona) reference。需要：
- 方案 A：用 per-persona 平均当 reference（牺牲 per-task 粒度）
- 方案 B：在 eval 集 200 上额外跑一轮 v29 teacher rollout
- 待用户决定

### 87. 补跑 queue（等 Ideal Disclosed v2 完结后串行启动）

为让**所有方法都在 canonical-200 上完整评估**：

| 序 | 任务 | 缺样本 | 预估 GPU 时长 |
|---|---|---|---|
| 1 | 等 Ideal Disclosed v2 收尾 | 2 | ~5 min |
| 2 | Clarify-first 补跑 canonical 150 | 150 | ~50 min |
| 3 | Prompt-only 补跑 canonical 150 | 150 | ~2.5 hr（5-6 turn/样本）|
| 4 | Base LLM 补跑 canonical 107 | 107 | ~1 hr |

总计 ~5 hr。跑完后 Exp1 主表可直接报 "200 seeds" 而不是 "50 test"。

---

## 2026-04-18

### 79. Experiment 2 narrative + 表格定稿（论文段落）

跟用户迭代 Experiment 2 的 LaTeX 段落和表格：
- 表格新增 **`Disc.` 列**（mean disclosure rate），把信息恢复机制做进表内自证
- 砍掉额外 scatter / bin 图（pooled 信号 ρ=+0.088 弱，bin 非单调，per-persona 三联画 2/3 退化），决定只保留 4-condition grouped bar 作为主图
- 段落开头从 "performance loss recovery" 改成两问并列：(i) clarification 恢复多少 mask 信息 (ii) 恢复是否转化为下游成功
- 末段把 "DPO 和 information recovery 混在一起" 的诚实声明改成 disentanglement claim：TactfulLLM vs Ideal Disclosed 隔离 policy effect

### 78. Experiment 2 数据更新 + OGR 计算

凌晨 03:36 启动的 Ideal Disclosed 在跑（PID 751556）。Full Query (Oracle) 已完成：

| Condition | pass@1 | pass@5 | Δ | OGR |
|---|:---:|:---:|:---:|:---:|
| Masked Direct | 12.3% | 18.5% | -- | 0% |
| TactfulLLM Overall | 16.0% | 23.5% | +3.7 | **48%** |
| · Novice | 18.5% | 25.5% | +7.0 | **82%** |
| · Experienced | 15.5% | 25.0% | +3.0 | **40%** |
| · Busy | 14.0% | 20.0% | +1.0 | **14%** |
| Ideal Disclosed | partial 86/200 | -- | -- | -- |
| Full Query | 20.0% | 28.0% | +7.7 | 100% |

OGR 用每个 persona 自己的 Direct 做分母（Novice 11.5, Exp 12.5, Busy 13.0；Full Query 20.0 共用）。Novice 82% 接近 ceiling，Busy 14% 与 policy 学会 Execute 一致。

Ideal Disclosed 进度 86/200（43%），实际 ~1.6 min/conv（5 candidate × Llama 8B 本地推理），剩 ~3h，预计上午 09:00 前后完成。`gpt-4o-mini` API 因为 single-turn execute 几乎不被调用。

### 77. Disclosure 信息回填 — 修 eval bug + replay 脚本

发现 eval JSON **没有记 `disclosed_items`**（simulator `react()` 返回了，但 `evaluate_multi_turn_persona.py` 在 turn_data 构造时漏掉），无法直接算 disclosure_rate。

- **修复**：`evaluate_multi_turn_persona.py:379` 和 `:785` 两处 `turn_data` 加 `"disclosed_items": user_reaction.get("meta", {}).get("disclosed_items", {})`
- **存量数据回填**：写 `scripts/replay_disclosure.py`，用 simulator 的确定性 `get_disclosure_info()` 重放已完成 eval 的 conversation → 重建 disclosed_items timeline。避免 5h 重跑。产出 `data/analysis/disclosure_per_conversation.csv`（600 行）

### 76. Disclosure → pass@1 相关性分析

写 `scripts/analyze_disclosure_recovery.py` 和 `scripts/plot_4condition_grouped.py`。

**Disclosure rate by persona**（验证设计意图）：
- Novice 0.886（expertise=low, 1 item/turn × 多轮累积，常饱和）
- Experienced 0.780（mid, 3 items/turn）
- Busy 0.000（policy 学会 Execute，根本不进 clarify 路径）

**Recovery → success 的相关性**：
- Pooled Spearman ρ=+0.088, p=0.032（弱信号，被 persona confounding）
- Within-Experienced ρ=+0.202, p=0.004（**唯一干净 dynamic range**）
- Within-Novice ρ=+0.041 p=0.57（饱和近 1.0 无变化）
- Busy degenerate（disclosure 全 0）
- Logistic regression `pass1 ~ disclosure_rate + C(persona)`：coef=+0.90, 95% CI [-0.19, +1.99], p=0.106（控制 persona 后边际）

**Bin 图非单调**（[14.2, 11.1, 7.5, 19.0]）是 persona composition shift artifact：低 bin 全是 Busy，中 bin 是 Exp 偶然低段。结论：**不放散点/bin 图主文**，靠主表的 Disc. 列 + grouped bar 自证。

输出图 `data/analysis/fig_recovery_4condition.png`：4 condition × 3 persona grouped bar，TBD bar 用 hatch 标注。

---

## 2026-04-17

### 75. Experiment 2 过夜任务进度检查（12h 后）

PID 603803 仍在跑（已 12h09m）：
- **Direct Execution 200**: ✅ 完成（22:39 出结果）
- **Oracle 200**: 🏃 48/200（约 24%，~1.5min/sample，预计还要 3-4h）
- **Ideal Disclosed 200**: ⏳ 待 Oracle 完成后启动

**Direct Execution 200-state 结果**（与 50-state 相比异常偏高）：

| Persona | pass@1 | pass@5 |
|---|:---:|:---:|
| Novice | 11.5% (23/200) | 19.0% (38/200) |
| Experienced | 12.5% (25/200) | 17.5% (35/200) |
| Busy | 13.0% (26/200) | 19.0% (38/200) |
| **Overall** | **12.3%** (74/600) | **18.5%** (111/600) |

⚠️ 两个发现：
1. **Direct 200 (12.3%) >> Direct 50 (7.3%)**，gap 5%。50-state 抽样可能偏难
2. **Direct 三 persona 不一致** (11.5/12.5/13.0) — 按理无交互应完全相同。说明 `--direct_execution` 模式下 persona 信息可能仍进了 prompt，或采样随机性导致

**对论文叙事的影响**：DPO vs Direct gap 从原本的 +8.7% (16.0 vs 7.3) 缩到 **+3.7%** (16.0 vs 12.3)。明早 Oracle/Ideal Disclosed 出来后填 Recovery 表。

### 74. Experiment 2 过夜任务启动（Recovery Analysis 200-state）

目标填 Recovery 表（Masked Direct / Clarified / Ideal Disclosed / Full Query）。已有：DPO 200-state（16%），Direct 50-state（7.3%）。今晚过夜跑三项，全部 200-state：

1. Direct Execution 200-state（从 50-state resume，补 150 extra）
2. Oracle / Full Query 200-state（persona-independent 单轮，Base Llama）
3. Ideal Disclosed 200-state（persona-independent 单轮，Base Llama）

`nohup /tmp/run_exp2_overnight.sh &`（PID 603803, PPID=1），log 在 `logs/exp2_overnight.log`。`.partial` resume 保护容器休眠。预计 3-5h。产出：
- `outputs/eval_v29_direct_execution_200.json`
- `outputs/eval_v29_oracle_200.json`
- `outputs/eval_v29_ideal_disclosed_200.json`

### 73. w/o Uncertainty 50-extra 中断 + 87-state 合并填表

50-extra 跑到 37/50 states（111/150 对话）中断（疑似容器休眠，无进程），合并 50 + 37 = **87 states** 先填 ablation 表。完整 100-state 明天补跑剩余 13 states。

**合并结果（87 states, 261 对话）**:

| Persona | pass@1 | pass@5 | Avg Turns | Rej Rate |
|---|:---:|:---:|:---:|:---:|
| Novice | 9/87 (10.3%) | 20.7% | 7.6 | 45.4% |
| Experienced | 12/87 (13.8%) | 19.5% | 2.7 | 37.5% |
| Busy | 8/87 (9.2%) | 12.6% | 1.0 | 0% |
| **Overall** | **11.1%** | 17.6% | 3.7 | — |

**注意**：w/o Unc Exp 13.8% > full Exp 12.0%（test 规模不一致：full=50, w/o Unc=87）。待 full 扩到 100-state 或 w/o Unc 补到 100 后重新比较。

### 72. Ablation w/o Uncertainty 50-extra 评估启动

w/o Persona 50-extra 完成后，启动 w/o Uncertainty 50-extra（50 states × 3 personas）。Novice 跑满 8 轮，预计 3-5h。完成后合并为 100-state 最终结果。

### 71. Ablation w/o Persona 100-state 结果确认

50 + 50-extra 合并：pass@1 11.0%（33/300），三 persona turns 全部 ≈1.02。行为分化完全消失的结论在 100 states 下稳定。

### 70. Ablation w/o Persona 50-extra 评估完成

50 extra states 完成。结果：Novice 12.0%, Exp 12.0%, Busy 10.0%，turns 全部 1.0。与 50-state 一致，全部退化为 Direct Execution。

### 69. Ablation 综合对比表整理

详见 `docs/v29_experiment_log.md` Ablation Study 部分。

核心结论：
- **w/o Persona**: 行为分化消失（turns≈1.0），pass@1 14.0%→11.0%。Persona 是行为分化的必要条件。
- **w/o Uncertainty**: 行为分化保持（8.0/2.6/1.0），pass@1 14.0%→10.7%。Uncertainty 不影响行为模式但影响代码质量。
- 两个组件互补：persona 控制"何时问"，uncertainty 帮助"问得更有效"。

---

## 2026-04-16

### 68. Ablation w/o Uncertainty 训练完成

`ABLATION_MODE=no_uncertainty`，17min 完成，loss 0.597→0.006，accuracy 100%（与 full v29 一致）。模型保存至 `models/v29_ablation_no_uncertainty/`。两个 ablation 的 50-state 评估过夜完成。

### 67. Ablation w/o Persona 训练完成

基于 v29 pairs（500 对），`ABLATION_MODE=no_persona` 训练 DPO。16min 完成，loss 0.642→0.386，accuracy 59.4%→87.5%。模型保存至 `models/v29_ablation_no_persona/`。

### 66. Ablation Study 设计确定 + 代码实现

Exp 3 (Cross-Persona Swap) 砍掉，替换为 Ablation Study：w/o Persona + w/o Uncertainty。
`render_state.py` 加 `ablation_mode` 参数，`train_dpo.py` 和 `evaluate_multi_turn_persona.py` 通过 `ABLATION_MODE` 环境变量传入。代码已验证三种模式输出正确。

### 65. v30 失败，回退 v29

v30 50-state 评估中间结果（13 states）：pass@1 全面崩盘（Novice 7.7%, Exp 0%, Busy 0%，vs v29 18.5/15.5/14.0%）。Busy 过度 Clarify（avg 2.9 turns vs v29 1.0）。
决定停止 v30 评估，回退到 v29 代码和模型。v30 commit 保留在 git history 中不 revert。

### 64. v30 DPO 训练完成 + 50-state 评估启动

详见 `docs/v29_experiment_log.md` v30 部分。

509 pairs, 同 v29 配置训练 17min。Loss 0.505→0.130, accuracy 97.5%（vs v29 100%，因为 pairs 更多样化）。模型保存至 `models/v30_100states/`。50-state 评估进行中，预计 5-7h。

### 63. v30 Preference Pairs 生成完成

详见 `docs/v29_experiment_log.md` v30 部分。

最终 509 pairs（v29: 500）。关键变化：
- **Busy**: turn 0 出现 Clarify=73 + Execute=34（复杂 task 问一轮），新增 turn 1 Execute=17
- **Novice**: turn 2 从全 Clarify(36) 变为 Clarify=3 + Execute=25（信息够了就停），turn 3 全 Execute
- **Experienced**: 不变

### 62. v30 Busy 条件性 Clarify 设计

分析 masked items 分布：1 item(1%), 2 items(31%), 3 items(40%), 4 items(23%), 5 items(5%)。

设计：n_masked_items >= 3 时（68% task），Busy 在 turn 0 Clarify 一轮。解决 v29 中 Busy 行为 = Direct Execution 的问题，让 Busy 与 Direct baseline 有区分。

### 61. v30 Disclosure-Aware 停止条件实现

详见 `docs/v29_experiment_log.md` v30 部分。

`compute_rewards.py` 核心改动：
1. 新增 `compute_disclosure_info()` — 计算 disclosure_ratio 和 n_masked_items
2. `get_correct_action()` — Novice: disclosure >= 50% 且 turn >= 2 → Execute；Busy: n_masked_items >= 3 → turn 0 Clarify
3. Method B 跳过已充分披露的 Novice Clarify pairs
4. 新增 Method B2 为 Busy 生成 turn 1 Execute pairs

初版用 disclosure_ratio >= 1.0 阈值，发现 Clarify turns 的 disclosure 从未到 100%（因为 state 记录的是问问题之前的状态），调整为 >= 0.5 + turn >= 2。

### 60. v30 计划：修复 Novice 过拟合 + Busy 行为极端

v29 两个问题：
1. **Novice 过拟合**：100% 跑满 7 轮，从不提前 Execute。根因：227 个 Novice pairs 几乎全是 chosen=Clarify
2. **Busy = Direct Execution**：107 pairs 全是 chosen=Execute，与 Direct Execution baseline 无区别

v30 方案：disclosure-aware 停止条件（Novice 信息够了就停）+ Busy 条件性 Clarify（复杂 task 问一轮）。

---

## 2026-04-15

### 59. Experiment 2: Oracle & Ideal Disclosed 实现

详见 `docs/v29_experiment_log.md` §22。

在 `eval/evaluate_multi_turn_persona.py` 新增 `--oracle` 和 `--ideal_disclosed` flag。两者都是 persona-independent 单轮 Execute（无用户交互），用 Base Llama 衡量不同信息量下的代码质量。Oracle 50-state 运行中，Ideal Disclosed 待启动。新增 pass@10 评估。

### 58. Clarify-first (K=1) baseline 50-state 完成

详见 `docs/v29_experiment_log.md` §20。

结果：**pass@1 9.3%**, **pass@5 19.3%**, Avg Turns 2.0, Rejection Rate 52.0%。比 Direct Execution (+2.0% pass@1) 略有提升，说明 1 轮 clarify 能获取少量信息。所有 persona 固定 2 turns 无行为分化。

### 57. 论文 Table 设计确定

详见 `docs/v29_experiment_log.md` §21。

**Table 1 (Main Results)**: pass@1 / pass@5 / Avg Turns × 4 personas (Nov/Exp/Busy/All) = 12 列。按 backbone 分组（Llama + Qwen）。Rejection Rate 在正文一句话提及，不单独成表。C_interrupt 不报（DPO Novice 过度 Clarify 导致成本反而高于 Base）。

### 56. Clarify-first (K=1) baseline 实现并启动

详见 `docs/v29_experiment_log.md` §20。

实现 `--always_clarify K` flag，Turn 0 强制 Clarify → Turn 1 强制 Execute。命名从 Always-Clarify 改为 **Clarify-first**（更准确描述 K=1 的"先问一轮再写代码"行为）。50-state 评估进行中。

### 55. Direct Execution baseline 50-state 完成

详见 `docs/v29_experiment_log.md` §19。

实现 `--direct_execution` flag，强制 Turn 0 Execute。结果：**pass@1 7.3%**（所有方法最低），三 persona 无差异。确认为 zero-interaction lower bound，证明 clarification 有价值（DPO 16.0% vs Direct 7.3%）。

### 54. Baseline 对比框架确定

5 个 baseline + ours:
- **Direct Execution**: 从不问（lower bound）✅
- **Clarify-first**: 必问一轮 → 进行中
- **Base LLM**: 裸模型自由决定 ✅
- **Prompt-only**: persona prompt 无训练 ✅
- **CollabLLM**: 外部方法 → 待实现
- **TactfulLLM (ours)**: DPO 训练 ✅

Appendix baseline 描述已写完（实现细节 + 设计理由）。

### 53. Prompt-only baseline 50-state 完成

详见 `docs/v29_experiment_log.md` §18。

结果：**pass@1 8.7%**（比 Base 12.7% 还低），行为零分化（三 persona avg turns 5.3-5.8），Busy rejection rate 89.4%。直接回应 reviewer "why not just prompt?"。

---

## 2026-04-14

### 52. Prompt-only baseline 50-state 评估启动

详见 `docs/v29_experiment_log.md` §16。

实现 `--prompt_only` flag（`eval/evaluate_multi_turn_persona.py`），Base Llama + persona 描述 system prompt（无决策规则）。Sanity check 2 states 确认：**Prompt-only 几乎无行为分化**，Busy 甚至比 Novice 问得更多。50-state 评估进行中，预计 3-5 小时。

### 51. 200-state DPO vs Base 完整对比完成

详见 `docs/v29_experiment_log.md` §13.7。

Base 150-extra 分两批跑完（108 + 42 states），合并后 200-state 结果：
- **DPO pass@1 16.0% vs Base 12.7%**（+3.3%），趋势正确但 Fisher exact p=0.059 未达 0.05
- Gap 比 50-state 预期小（50-state 时 gap=6%，200-state 实际 3.3%）
- **行为分化依然完美**：DPO Novice 7.0 > Experienced 2.6 > Busy 1.0
- 决定：不再扩大测试集，转向 baseline 对比丰富论文叙事

### 50. Base 150-extra 剩余 42 states 补跑完成

Base 150-extra 跑到 108/150 时中断（疑似容器休眠），提取已完成 state，生成 `test_states_v29_eval_150extra_remaining.jsonl`（42 states），利用 resume 机制补跑完成。

---

## 2026-04-12

### 49. 论文完整实验计划确定

详见 `docs/v29_experiment_log.md` §14。

**2 backbones**: Llama-3.1-8B + Qwen2.5-7B。**5 个方法**: TactfulLLM-DPO (ours), Direct Execution, Prompt-only, Always-Clarify, CollabLLM。**3 个实验**: Main performance, Recovery analysis, Persona sensitivity。

当前阶段只做 Llama，200-state 评估确认显著性后，优先加 Prompt-only 和 Always-Clarify baseline（不需要训练，最快出结果），然后 Qwen backbone 和 CollabLLM。

### 48. 200-state 扩大评估启动（50 已有 + 150 新增）

详见 `docs/v29_experiment_log.md` §13。

50-state 结果 DPO 14% vs Base 8% 趋势正确但不显著（p=0.139）。Power analysis 显示 200 states 可达 p≈0.001, power 89%。

从 981 个可用 state 中采样 150 个（seed=43，与训练 109 + 已测试 50 零重叠），文件 `data/seeds/test_states_v29_eval_150extra.jsonl`。DPO 150-state 评估已启动（预计 ~9h），Base 待 DPO 完成后启动。完成后与 50-state 结果合并为 200-state 最终结果。

**训练集不增大**：500 pairs 已饱和（accuracy 100%），pass@1 瓶颈在代码生成能力非数据量。

**200-state 后下一步**：加 SFT + Prompting baseline，用同一 200-state 测试集评估，构成最终论文实验。

### 47. v29 DPO vs Base 50-state 评估完成

详见 `docs/v29_experiment_log.md` §12.4。

**行为分化**: DPO 完全成功 — Novice(85.7% clarify, 7.0 turns) > Experienced(62.4%, 2.66) > Busy(0%, 1.0)。Base 完全 persona-blind（三 persona clarify 49-62%，无差异）。

**pass@1: DPO 14% vs Base 8%**（+75% 相对提升）。Novice 差距最大：DPO 16% vs Base 4%（+300%）。20-state 时 DPO=Base=15% 是样本噪声，50-state 揭示了真实差距。

**pass@5 持平**: DPO 20% vs Base 19.3%，说明 DPO 学的是决策策略而非代码生成能力。

**20-state 的 Busy 担忧消除**: DPO Busy 14% > Base 10%，之前 20-state 的 -5% 是噪声。

**Base 乱问反而差**: Base Busy clarify 62% 但 pass@1 仅 10%，不分 persona 乱问不如 DPO 的 persona-aware 策略。

---

## 2026-04-11

### 46. Busy 表现不佳的深入分析

详见 `docs/v29_experiment_log.md` §11.5。

DPO Busy 永远 T0 Execute → 只有 masked query → 从未获得额外信息。控制变量对比（两者都 0-clarify 时）DPO Busy 11.8% < Base 17.6%，暗示 DPO LoRA 轻微损害纯代码生成能力。

两个叠加因素：①永远不 Clarify 导致信息不足 ②LoRA 可能伤害 code gen。Novice 通过多轮 Clarify 恢复信息弥补了这个损失。

**论文意义**: 这体现了 persona-aware 的核心 trade-off — Clarify 有代价（打断用户）也有收益（恢复信息），不同 persona 偏好下产生不同的 task success trade-off。

### 45. 扩大评估至 50 states（进行中）

详见 `docs/v29_experiment_log.md` §12。

20-state 评估 pass@1 DPO = Base = 15%，样本量太少（1 task 差异 = 5%）。从 1031 个未使用 state 中采样 50 个（seed=42，与 109 训练 state 零重叠），文件 `data/seeds/test_states_v29_eval_50.jsonl`。

DPO 50-state 评估进行中，Base Llama 待 DPO 完成后启动。

### 44. v29 DPO vs Base Llama 对比（20 states）

详见 `docs/v29_experiment_log.md` §11.2-11.4。

**DPO 行为学习成功**: Novice clarify 85.7%、Experienced 68.3%、Busy 0%。Base 完全 persona-blind。

**代码质量对比（20 states）**:
- pass@1: DPO 15% = Base 15%（总体无差异）
- pass@5: DPO **18.3%** > Base 15%（+3.3%，Novice 贡献最大 25% vs 15%）
- Novice pass@1: DPO 20% > Base 15%（多轮 Clarify 有效）
- Busy pass@1: DPO 10% < Base 15%（仅差 1 task，采样噪声）

结论：行为分化成功，pass@5 有提升趋势但 20 samples 不够显著 → 扩大到 50 states。

### 43. v29 多轮评估启动

详见 `docs/v29_experiment_log.md` §11。

20 个 BigCodeBench states（与训练轨迹 109 states 零重叠），3 personas × 20 states = 60 对话。本地 Llama 端到端生成，gpt-4o-mini 用户模拟。

初步观察（前 8 states）：行为差异化正确——Busy 全部 T0 Execute，Novice 全部多轮 Clarify，Experienced 1-2 轮 Clarify 后 Execute。

评估完成后需跑 Base Llama (`--no_lora`) 对照，验证 v29 masking 下的 baseline pass rate。

### 42. v29 DPO 训练完成

详见 `docs/v29_experiment_log.md` §10。

500 pairs, beta=0.1, epochs=3, QLoRA (r=64)。训练 17 分钟，loss 0.597→0.006，accuracy 59%→100%。模型保存至 `models/v29_100states/`。

数据泄露检查：初始测试集有 2 个 state 出现在轨迹数据中，已替换为干净 states。

### 41. 逆信号与 reward gap 深入分析 → 决定直接训练

详见 `docs/v29_experiment_log.md` §9。

**逆信号本质**：72 个逆信号中 52 个是 interrupt bonus 噪声（task_score 相同，γ bonus 翻转方向），20 个是真逆信号（Clarify 确实帮了代码，但 behavior-first 选了 Execute/另一方）。真逆信号是论文核心 tradeoff 的体现，应保留。

**γ=λ 方案否决**：模拟后 Busy 只剩 12 个 pairs 且 8 个逆信号。γ=λ 破坏了"打断有成本"的核心机制，不可行。

**关键发现：reward gap 不进 DPO loss**。`train_dpo.py` 用标准 `trl.DPOTrainer`，只需 (prompt, chosen, rejected)，reward gap 不影响梯度。逆信号对训练实际无害——chosen/rejected 方向由 behavior-first 保证正确。

**决定：停止调 reward 参数，直接用 500 pairs 训练 v29。** 继续调 reward 不会改善最终 pass rate。真正决定 pass rate 的是模型行为学习（已验证可行）和代码生成能力（非 reward 问题）。

### 40. v29 100-state 4 层分析完成

详见 `docs/v29_experiment_log.md` §7-8。

**数据规模**: 109 unique states, 2794 trajectory turns, 1527 trajectories → 500 preference pairs (107 complete states)。

**Layer 1 — 轨迹行为** ✅: Novice(2.30) > Experienced(1.80) > Busy(1.25)，与 10 states 一致。

**Layer 2 — Pass Rate** ⚠️: direct=0.373, clarified=**0.399**, ideal_disclosed=0.385, oracle=0.436。**clarified > direct 确认**（10 states 时 clarified < direct 是样本噪声）。整体 pass rate 比 10 states 低（0.37 vs 0.56），10 states 抽到的 task 偏简单。clarified > ideal_disclosed 反直觉，可能是 `"; ".join(items)` 格式比原始 spec 更结构化。

**Layer 3 — 信号质量** ⚠️: 正确信号率 75.6%（10-state 84.8%，v28 ~60%）。逆信号率 14.4%（10-state 8.7%，v28 ~25%）。Novice 96% 正确，Experienced T1 逆信号 **64%**（39/61，其中 35 个是 interrupt bonus 噪声），Busy T0 只有 34% 正确+46% zero gap。

**Layer 4 — Gap 来源**: 66.6% 纯 interrupt bonus 驱动，只有 23.4% 有 task_score 差异。Scale up 没有改善——不是样本量问题，是 gpt-4o-mini 在多数 task 上 pass rate 差异不足。强信号 pairs (gap≥0.05) 有 146 个 (29.2%)。

**待决策**:
1. Experienced T1 逆信号：设 γ=λ 去掉 bonus / 过滤 |gap|<0.05 / 两者结合
2. Busy zero gap (46%)：暂不处理或过滤
3. 是否直接用 500 pairs 训练

---

## 2026-04-10

### 39. v29 100-state 生成启动

10-state 验证通过后，启动 100-state 生成（~5.5h, gpt-4o-mini, n_samples=4）。分两批生成（part1: 57 states, part2: 52 states），合并后 109 unique states。

### 38. v29 10-state 4 层分析完成

详见 `docs/v29_experiment_log.md`。

**Layer 1 — 轨迹行为** ✅: Novice(2.22) > Experienced(1.80) > Busy(1.25)，排序正确。

**Layer 2 — Pass Rate**: direct=0.563, clarified=0.541, ideal_disclosed=0.589, oracle=0.612。masking 修复有效（v28≈0→v29 0.56+）。但 clarified < direct（差异 0.022，样本小待确认）。

**Layer 3 — 信号质量**: 总正确信号率 84.8%（v28 ~60%），逆信号率 8.7%（v28 ~25%）。Novice 100% 正确。Experienced T1 逆信号 3/7，其中 2 个是 interrupt bonus 噪声（gap=0.032），1 个真逆信号。

**Layer 4 — Gap 深入分析**: 46 pairs 中 27 个 gap 在 0.032-0.048 之间，全部来自 interrupt bonus（γ-λ=0.08 × w_interrupt=0.2），不是 task_score 差异。根因是 10 states 太少，task_score 大多全 0 或全 1.0。讨论了方案 B（过滤小 gap）和方案 C（去掉 bonus），决定先 scale up 到 100 states 再定。

---

## 2026-04-09

### 37. v29 计划：基于结构化 masking 重新生成数据

详见下方 #35、#36 分析。核心改动：用 BigCodeBench 官方 `instruct_prompt` 的固定结构边界重写 masking 逻辑，彻底解决断句残留和 regex 跨行吞内容问题。

v29 步骤：
1. 重写 `mask_task_details.py`：按 `"The function should output with:\n"` → `"You should write self-contained"` 边界精确切割
2. 用 `instruct_prompt`（而非自建 query 字段）作为 masked prompt 基础，保证结构一致
3. 重新生成 `bigcodebench_masked_states.jsonl`（470 states）
4. 重新生成轨迹 → compute_rewards → 训练 v29 DPO → 评估

### 36. Masking 根因：应直接按结构边界切割

BigCodeBench 全部 1140 个 task 的 `instruct_prompt` 都有固定结构：

```
[函数描述]
The function should raise the exception for: [异常]（可选）
The function should output with:
    [返回值类型和描述]
You should write self-contained code starting with:
```[代码模板]```
```

output_format 就是 `"The function should output with:\n"` 到 `"You should write self-contained"` 之间的内容，边界 100% 固定，不需要任何复杂 regex。

正确的 mask 方式：
```python
masked = re.sub(
    r'The function should output with:\n.*?(?=You should write self-contained)',
    '',
    instruct_prompt,
    flags=re.DOTALL
)
```

mask 后效果（以 BigCodeBench/1 为例）：
- 原始：`"...ValueError if ...\nThe function should output with:\n    dict: A dictionary...\nYou should write..."`
- masked：`"...ValueError if ...\nYou should write..."` ← 零残留，干净

现有 masking（regex 猜边界）的问题：
1. 断句残留 `"The function \n"`：470/470（100%）受影响
2. input_constraints regex 跨行吞掉 output_format（Bug #2）：14/470
3. disclosure_info 只存截断内容（Bug #1）：456/470

### 35. Task success rate 低的根本原因确认

**实验**：Base Llama-3.1-8B-Instruct 在这 20 个 v28 评估 task 上：

| 条件 | Pass@1 |
|------|--------|
| Unmasked（完整 instruct_prompt） | **30% (6/20)** |
| Masked（现有 masking） | **0% (0/20)** |
| v28 DPO + Masked（Busy，直接 Execute） | **5% (1/20)** |

与官方 leaderboard 对齐：Llama-3.1-8B-Instruct 在 BigCodeBench Full Set 的官方 solve rate = 32.8%，我们的 30% 完全一致。

**结论**：
- 模型能力上限 ~30%，这批 task 本身不算特别难
- Masking 把 pass rate 从 30% 打到 0%，是主要原因
- v28 Experienced persona 多轮 Clarify 只恢复到 5%，说明 Clarify 几乎没有有效恢复 output_format 信息
- 根本原因是 masking 方式有问题（断句残留 + regex 不精确），导致 masked query 质量差，即使 Clarify 也难以弥补

### 34. v28 task success 低的根因澄清

#### 之前的结论（#33）需要修正

#33 认为 `disclosure_info.output_format` 为空是 v28 task success 低的关键原因。经过对轨迹数据和代码的深入排查，发现这个结论**不准确**。

#### 实际情况

**轨迹生成没有问题**：用户模拟器（`simulator/disclosure.py:94`）读取的是 `disclosure_rule.masked_fields.output_format`，而非 `disclosure_rule.disclosure_info.output_format`。`masked_fields` 存有完整的被 mask 文本，所以轨迹生成时 Clarify 能正确披露返回值信息。

验证数据：
- `trajectories_145states_combined.jsonl` 中 1417 条 Clarify 轨迹
- 其中 466 条的 `user_reaction.meta.disclosed_items.output_format` 包含完整内容（如 `"should output with:\n    float: The average of..."`)
- `disclosure_info.output_format.specification` 确实只有 `"should output with:"`（截断），但**不影响轨迹生成**

**评估脚本是真正的根因**：#23 Bug #2——`react()` 未传 `disclosure_rule` 参数，导致评估时用户模拟器拿不到任何 masked_fields 信息，Clarify 完全无效。这解释了为什么：
- Novice（5 轮 Clarify）pass rate 反而不如 Busy（直接 Execute）
- Experienced Clarify 轮数多但 task success 无提升

#### 修正后的根因归因

| 因素 | 影响程度 | 说明 |
|------|---------|------|
| 评估脚本 `react()` 缺 `disclosure_rule`（#23 Bug #2） | **关键** | Clarify 无效，多轮对话白问 |
| 8B 模型编码能力 | **根本瓶颈** | unmasked task 也只有 43.3% |
| `disclosure_info` 截断 bug | **不影响轨迹生成** | 仅影响 disclosure_info 字段，用户模拟器用的是 masked_fields |

#### 结论

- `disclosure_info` 的 bug 仍应修复（保持数据一致性），但不是 v28 低 pass rate 的原因
- 评估脚本已在 #23 中修复（两处 `react()` 加上 `disclosure_rule`），需要**用修复后的评估脚本重跑 v28 评估**验证
- 轨迹生成和训练数据质量没有问题，不需要重新训练

### 33. Masking 代码深入审计与修复

#### 发现的 3 个 Bug

**Bug 1（HIGH）：`disclosure_info.output_format` 未存完整内容**

`create_mask_rule` 调用 `extract_output_spec(task)` 提取 output_format，但该函数用 `r'should output.*?(?:\n|$)'` 只匹配到第一个换行，结果 456/470 个 state 的 `disclosure_info.output_format.specification` 都是 `"should output with:"`（无实际内容）。

修复：`disclosure_info.output_format.specification` 直接使用 `masked_fields["output_format"]`（mask_prompt 实际删掉的完整文本），不再调用 `extract_output_spec`。

**Bug 2（HIGH）：`mask_prompt` 执行顺序导致 14 个 state output_format 丢失**

input_constraints regex（含 `$`）先于 output_format 执行，`r'\b(?:if|when).*?(?:empty|negative|zero).*?(?:\.|$)'` 会跨行吞掉后续的 `"should output with:..."` 内容。14/470 个 state 的 output_format 被错误地存入 `masked_fields.input_constraints`。

修复：
1. output_format masking 先于 input_constraints 执行
2. input_constraints regex 用 `[^\n]` 替代 `.`，禁止跨行匹配

**Bug 3（MEDIUM）：断句残留 `"The function \n"`**

mask_prompt 删除 `"should output with:..."` 后，前面的 `"The function "` 残留在 query 中，456/470 个 state 受影响。

修复：删除 output_format 时同时清理前面的不完整句子片段。

#### 修复验证

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| output_format 正确 mask | 456/470 | **470/470** |
| 断句残留 `"The function \n"` | 456/470 | **0/470** |
| disclosure_info 有完整内容 | 0/470 | **470/470** |

#### v28 task success 低的根因分析

v28 pass@1 = 3.3%（2/60），逐层归因：

| 因素 | 影响程度 | 分析 |
|------|---------|------|
| 断句残留 `"The function \n"` | **低** | cosmetic noise，模型基本能忽略 |
| output_format 被 mask（对 Busy） | **有限** | 77% 的返回类型可从上下文推断（描述+import+函数签名）；Busy 不 Clarify，信息缺失是设计意图 |
| disclosure_info 为空（对 Experienced/Novice） | **关键** | Clarify 完全无效，用户模拟器无法披露返回值信息。Novice 5 轮 Clarify pass@1=0%，反而不如 Busy（5%） |
| 8B 模型编码能力 | **根本瓶颈** | validate_llama_gap_v20 在 90 个 unmasked task 上也只有 43.3%；gpt-4o-mini 在这些 masked task 上同样 0% |

**output_format 复杂度分布**：
- 简单（1-2 行，类型基本可猜）：288/456（63%）
- 复杂（3+ 行，tuple 结构/dict key-value 规范）：168/456（37%），测试 assert 严格匹配这些细节

**结论**：disclosure_info 为空是必须修的（否则无法验证"Clarify 能提升 task success"这一核心论点），但不应期望 pass rate 从 3% 跳到 40%。修复后的预期：
- Busy 基本不变（不依赖 Clarify）
- Experienced/Novice 应有明显提升（Clarify 能恢复返回值信息）
- 复杂 output_format（37%）的 task 提升最大

### 23. 多轮评估脚本系统性 Bug 修复

审计 `eval/evaluate_multi_turn_persona.py`，发现并修复 7 个 Bug：

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | **HIGH** | Execute 时用膨胀的对话历史 query 生成代码（未用 `build_clean_execute_query`） | 导入并使用 `build_clean_execute_query`，传入 `initial_state_snapshot` |
| 2 | **HIGH** | `react()` 未传 `disclosure_rule`，用户模拟器无法披露结构化信息，Clarify 完全无效 | 两处 `react()` 调用加上 `disclosure_rule=current_state.get("disclosure_rule")` |
| 3 | **MEDIUM** | `total_questions_asked` 训练时数问号、评估时数 Clarify 轮数 | 改为 `assistant_msg.count("?")` 累计 |
| 4 | **MEDIUM** | Clarify prompt 缺少 `disclosure_rule` 的 masked_fields 引导 | `generate_assistant_message` 中加 disclosure_rule 引导（与训练一致） |
| 5 | **LOW→HIGH** | 测试数据缺 `disclosed_info` 字段，修 Bug2 后会 KeyError | `load_jsonl` 初始化 `disclosed_info` |
| 6 | **LOW** | `initial_state.copy()` 浅拷贝导致跨 persona 状态污染 | 改为 `copy.deepcopy` |
| 7 | **INFO** | 死代码：本地 `render_state` 与训练用的格式不同 | 删除 |

额外修复 `scripts/generate_trajectories.py`：`update_state_for_next_turn` 中 `dialogue_turn` 默认值 1→0（latent bug，实际数据都有该字段不触发）。

### 24. 修复 `generate_with_template_local` prompt 回显 Bug

**问题**：纯 Llama 评估（不加 `--use_openai_for_generation`）时 pass@1 = 0%，所有代码都是 system prompt 回显。

**根因**：`generate_with_template_local` (line 85-87) 用 `skip_special_tokens=True` decode 整个序列后按 `"Assistant:"` split，但 Llama-3.1 chat template 用特殊 token 标记 assistant turn，不含 `"Assistant:"` 文本，split 没生效，返回完整 prompt + 生成内容。

**修复**：只 decode 新生成的 tokens：
```python
input_len = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_len:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

### 25. v28 多轮评估结果（修复后）

**配置 A：`--use_openai_for_generation`**（Llama 选 action，gpt-4o-mini 生成内容）

| Persona | Avg turns | Clarify rate | pass@1 |
|---------|-----------|-------------|--------|
| Busy | 1.0 | 0% | 0% (0/5) |
| Experienced | 4.6 | 78.3% | 0% (0/5) |
| Novice | 6.0 | 83.3% | 20% (1/5) |

行为模式：
- Busy：5/5 Turn 0 Execute ✓
- Experienced：2/5 早期切 Execute（Turn 1-2），3/5 和 Novice 一样多轮 Clarify
- Novice：5/5 多轮 Clarify → forced Execute
- 唯一 pass 的 case：BigCodeBench/678 Novice（多轮 Clarify 获取信息后通过）

**配置 B：纯 Llama**（修复 prompt 回显前）— 全 0%，无效结果。修复后重跑中。

### 26. v28 训练数据分析

训练数据：`prefs_v28_145states.jsonl`，727 pairs

**Pair 分布：**

| Persona | Turn 0 | Turn 1 | Turn 2 | Turn 3 | Turn 4 | 总计 |
|---------|--------|--------|--------|--------|--------|------|
| Busy | 145 | - | - | - | - | 145 |
| Experienced | 138 | 77 | 1 | - | - | 216 |
| Novice | 145 | 98 | 69 | 37 | 17 | 366 |

Behavior-first 方向：724/727 正确（99.6%）。

**关键问题：Experienced turn≥1 逆信号率 53.8%（42/78）**

逆信号构成分析：

| 类型 | 数量 | 占总78 | 原因 |
|------|------|--------|------|
| γ 奖励 artifact | 36 | 46.2% | task_score 相同，但 Clarify 有负 interrupt_cost（-0.08），多了 0.016 reward |
| 真正 Clarify 更好 | 6 | 7.7% | 多轮澄清确实提升了代码质量 |
| 正常信号 | 36 | 46.2% | Execute reward ≥ Clarify reward |

**根因**：4/07 修复 γ=0.20 > λ=0.12 后，有效澄清的 interrupt_cost = λ - γ = -0.08（奖励）。Experienced turn≥1 的 Clarify 轨迹中，用户回答了问题 → cost 为负 → Clarify reward 比 Execute 高 0.016。DPO 训练时近一半 pairs 在说"Clarify 更好"，模型学不到 turn≥1 该 Execute。

**解决方案（待实施）**：
1. **过滤方案**：drop `|margin| < 0.05` 的逆信号 pairs（78→42 pairs，逆信号率降到 14.3%）
2. **修 reward 方案**：turn≥1 时不给 Clarify 负 interrupt_cost 奖励

### 27. method.tex 更新

新增 Evaluation Metrics 小节：
- Task Success Rate：BigCodeBench pass@k 定义（k 个候选中至少一个通过全部测试 = success）
- Proactive Behavior Metrics：action accuracy、avg turns、clarify rate

### 28. v28 纯 Llama 端到端评估（修复 prompt 回显后）

5 samples × 3 personas，Llama DPO 生成所有内容，gpt-4o-mini 只做用户模拟。

**Proactive behavior（行为模式 — 与 OpenAI 版一致）：**

| Persona | Avg turns | Clarify rate | 行为 |
|---------|-----------|-------------|------|
| Busy | 1.0 | 0% | 5/5 Turn 0 Execute ✓ |
| Experienced | 4.6 | 78.3% | 2/5 早切 Execute (Turn 1-2)，3/5 多轮 Clarify |
| Novice | 6.0 | 83.3% | 5/5 多轮 Clarify → forced Execute |

**Task success rate（pass@1）：**

| Persona | pass@1 | pass@5 |
|---------|--------|--------|
| Busy | 0% (0/5) | 0% (0/5) |
| Experienced | **20% (1/5)** | 20% (1/5) |
| Novice | 0% (0/5) | 0% (0/5) |
| **Overall** | **6.7% (1/15)** | 6.7% (1/15) |

唯一 PASS：BigCodeBench/1133 Experienced（Clarify 1 轮获取 API 返回类型和错误处理信息 → Turn 1 Execute → candidate 1/5 通过全部测试）。同一 task 的 Busy（直接 Execute）和 Novice（5 轮 Clarify）均 FAIL。

### 29. Task success rate 偏低的排查（初步）

**初步排查结论：不是模型 bug，是这批 task 本身难。**

验证步骤：
1. canonical solution + imports + def → score=1.0 ✓（测试执行管线正确）
2. `extract_code_from_text` 对 markdown 代码块提取正确 ✓
3. **gpt-4o-mini 在这 5 个 masked task 上 pass@1 也是 0%**（部分分：484=33%, 678=80%，但无一全过）
4. `validate_llama_gap_v20` 的 43.3% 用的是不同的 90 个 task + base Llama（无 DPO）+ `max_new_tokens=512`，那批 task 不含这 5 个

### 30. v28 扩大评估（20 samples）

20 samples × 3 personas，纯 Llama DPO 端到端：

| Persona | Avg turns | Clarify rate | pass@1 | 行为正确率 |
|---------|-----------|-------------|--------|-----------|
| Busy | 1.0 | 0% | 5% (1/20) | 20/20 (100%) ✓ |
| Experienced | 5.5 | 81.8% | 5% (1/20) | 2/20 (10%) ✗ |
| Novice | 6.0 | 83.3% | 0% (0/20) | 20/20 (100%) ✓ |
| **Overall** | - | - | **3.3% (2/60)** | - |

PASS cases：Busy/BigCodeBench/415（直接 Execute）、Experienced/BigCodeBench/1133（Clarify 1 轮→Execute）。

### 31. DPO 是否影响代码生成能力？

同样 20 个 masked task，Base Llama（无 DPO）直接 Execute：

| 模型 | pass@1 |
|------|--------|
| Base Llama（无 DPO） | **0% (0/20)** |
| v28 DPO Busy（直接 Execute） | **5% (1/20)** |

**结论：DPO 没有降低代码生成能力**，反而略有提升。问题出在 masked 数据质量。

### 32. Masked 数据质量问题深入排查

发现两个严重问题导致 task success rate 极低：

#### 问题 1：断句残留（96% 数据受影响）

masking 删除 output_format 内容后，留下残缺的句子开头：

```
原始：The function should output with:\n    str: The filename...
masked 后：The function \nYou should write self-contained code...
```

`"The function \n"` 是一个无意义的断句残留，333 条测试数据中 321 条（96%）都有此问题。

**已修复**：简单文本清理，删除 `\nThe function \n` → 保存为 `test_states_clean_for_eval_fixed.jsonl`。种子数据 `bigcodebench_masked_states.jsonl` 中 455/470（97%）也有同样问题。

#### 问题 2（根本原因）：output_format 被 mask 但 disclosure_info 未存完整内容

被 mask 的 `output_format` 包含**返回值类型和格式**，是测试通过的关键信息：

| Task | masked 掉的 output_format | 测试会检查 |
|------|--------------------------|-----------|
| 1133 | `str: The filename into which the JSON data was written` | assert 返回值是文件名字符串 |
| 484 | `pd.DataFrame: Generated sensor readings` | assert 返回 DataFrame |
| 138 | `Axes object, x-axis 'Letters', y-axis 'Frequency', title 'Letter Frequency'` | assert 图表标签和标题 |
| 678 | `DataFrame containing data from all processed files` | assert 返回 DataFrame |
| 630 | `str: The full file path` | assert 返回路径字符串 |

但 `disclosure_rule.disclosure_info.output_format` 只存了 `{'specification': 'should output with:'}` — **没有具体内容**。即使模型 Clarify 了 output_format，用户模拟器也无法披露完整的返回值信息，Clarify 对 task success 的提升被严重限制。

**影响**：
- 模型不知道返回什么 → 测试 assert 返回值 fail
- Clarify 无法恢复此信息 → 多轮对话也没用
- 这解释了为什么 Novice（多轮 Clarify）pass rate 反而不如 Experienced/Busy

**修复方案**：把 `masked_fields.output_format` 的完整内容填入 `disclosure_info.output_format`，使用户模拟器能通过 Clarify 披露返回值信息。不需要重做 masking，只需修补 disclosure_info 字段。

### 下一步

1. **修补 disclosure_info**：把 masked_fields.output_format 完整内容填入 disclosure_info，使 Clarify 能恢复返回值信息
2. **清理断句残留**：对种子数据和测试数据都清理 `"The function \n"` 断句
3. **修复 Experienced turn≥1 逆信号**：过滤 `|margin| < 0.05` 的 artifact pairs
4. 用修复后的数据重新生成轨迹 → 训练 v29 → 评估
5. 对比修复前后的 task success rate

---

## 2026-04-08

### 15. 修复 Execute prompt 拼接问题（clean execute query）

**问题**：Novice 多轮 Clarify 后，`current_state['query']` 包含完整对话历史（3500+ chars），传给 gpt-4o-mini 生成代码时噪声太多（assistant 废话、重复内容、空代码块），导致 clarified 代码质量反而比 direct 差。

对比数据（20 states 旧数据）：15 个 Novice case 中只有 2 个 clarified > direct，5 个 worse。

**根因**：`generate_trajectories.py:565` 每轮把完整 assistant_msg + user_reply 拼进 query，Execute 时直接用这个膨胀的 query。

**修复**：新增 `build_clean_execute_query()` 函数：
- Execute 时用 `initial_masked_query` + `disclosure_rule.disclosed_info`（结构化追踪的澄清信息）构造干净 prompt
- 格式和 `ideal_disclosed` 一致：`"Key requirements: item1; item2; ..."`
- Clarify 仍用原始 dialogue history（需要上下文来生成问题）

修改了两处 Execute 代码生成：
1. loop 内 Execute（~line 690）
2. force Execute（~line 860）

**关键 insight**：`disclosed_info` 已经在 `update_state_for_next_turn` 中按 category 结构化追踪了每轮披露的信息，不需要从对话文本中提取。

### 16. 3 states 验证

结果：71 条轨迹，17 pairs。逆信号率 29.4%（vs 之前 33.1%）。

Execute prompt 长度从 2600+ chars 降到 ~600 chars。Clarified 代码不再被对话噪声污染。

### 17. 同步 method.tex 与 NeurIPS PDF

对比 `docs/method.tex`（旧）与 `docs/NeurIPS__TactfulLLM.pdf`（新），同步了三处：
- R_task：5 档分段函数 → 直接用 pass_rate
- C_interrupt：简化为 `c = λ - γ·α + δ·r`
- w_interrupt：0.3 → 0.2

新增 **Behavior-First Pair Construction** 段落到 method.tex：
- 三个 persona 的理想行为模式表格
- target action 函数 $a^*(p, t)$ 公式化
- Trajectory diversity（forced first action + fork）和 state-aligned rebalancing 的实现说明

### 18. 30 states 验证完成

轨迹文件：`data/logs/trajectories_20260407_231005.jsonl`（640 条轨迹）

**逆信号率：21.6%**（vs 之前 33.1%），clean execute query 有效。

分析发现 Novice 高 turn 逆信号 68% 来自**被用户拒绝的 Clarify 轨迹被选为 chosen**。

### 19. 过滤被拒绝 Clarify 轨迹

修改 `reward/compute_rewards.py`，三处过滤：
1. 选 Clarify best trajectory 时，优先选未被拒绝的（`interrupt_cost < 0.8`）
2. Method B 多轮 pairs，跳过被拒绝的 Clarify turn
3. Method A fork pairs，跳过被拒绝的 mainline Clarify

**过滤后**：176 → 155 pairs，逆信号率 21.6% → **16.8%**

| Persona | Turn | Before | After |
|---------|------|--------|-------|
| Novice | 1 | 24.0% | **13.6%** |
| Novice | 2 | 31.8% | **7.1%** |
| Novice | 3 | 21.4% | **0%** |

Pairs 文件：`data/dpo/prefs_v28_clean_30states_filtered.jsonl`

### 20. 新增 `--skip_states` 参数

`generate_trajectories.py` 新增 `--skip_states N`，跳过前 N 个 states，用于续跑不重复。

### 21. 150 states 正式数据生成（进行中）

先跑 30 states 已完成，再跑 120 states（skip 前 30）拼接。

120 states 正在跑，预计 ~6 小时。

### 22. 训练和评估脚本检查

确认 `train_dpo.py` 和 `evaluate_dpo_model.py` 均 ready for v28：
- action prefix stripping ✓
- pick_action_from_generation ✓
- persona 传 render_state ✓
- 默认参数：beta=0.1, epochs=3, lr=5e-5, lora_rank=64

### 下一步

1. 120 states 跑完后拼接 30 states 数据，compute_rewards 生成 pairs
2. 训练 v28（Llama-3.1-8B-Instruct）
3. 评估 v28 persona 区分度
4. 同样 pairs 训练 Qwen2.5-7B-Instruct 作为第二个 base model

---

## 2026-04-07（续）

### 9. 排查 20 states pair 为空问题

之前 `traj_v27_real_20states_20260407_042623.jsonl` 的 compute_rewards 产出 0 pairs，原因：**轨迹生成时未传 `--llm_model`，所有 assistant_msg 为同一条 dummy 输出**。compute_rewards 的 `same_message` 检查正确地跳过了这些 pair。

### 10. 重新生成 20 states 轨迹（gpt-4o-mini）

```
python scripts/generate_trajectories.py --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 20 --llm_model gpt-4o-mini --all_personas --max_turns 5 --n_samples 3
```

结果：434 条轨迹，207 个不同 assistant_msg，消息多样性正常。

轨迹文件：`data/data/logs/traj_v27_20states_llm_20260407_093902.jsonl`

### 11. 修复 `get_correct_action`：Novice turn≥3 切换 Execute

之前 Novice 在所有 turn 都返回 Clarify，但实际轨迹最后一轮一定是 Execute。

修改 `reward/compute_rewards.py` 的 `get_correct_action`：
- Novice-Learner：turn < 3 → Clarify，turn ≥ 3 → Execute

### 12. 修复 interrupt_cost γ 参数：有效澄清应为奖励

论文设计意图：用户回答了问题 → cost 为负（即加分）。但代码中 γ=0.08 < λ=0.12，有效澄清 cost = +0.04（还在扣分），与设计矛盾。

修改 `reward/compute.py` 的 `compute_interrupt_cost_v2`：
- γ: 0.08 → **0.20**（γ > λ，使有效澄清 cost = 0.12 - 0.20 = -0.08，即奖励）

三种情况：
- 用户回答了：cost = n_q × (0.12 - 0.20) = **-0.08/问题**（奖励）
- 用户没回答：cost = n_q × 0.12（轻罚）
- 用户拒绝了：cost = n_q × (0.80 + 0.12) = 0.92（重罚）

### 13. 20 states compute_rewards 验证

最终结果（两个修复叠加后）：121 pairs

| Persona | Turn 0 | Turn 1 | Turn 2 | Turn 3 | Turn 4 | 合计 |
|---------|--------|--------|--------|--------|--------|------|
| Busy | 20 | - | - | - | - | 20 |
| Experienced | 20 | 13 | - | - | - | 33 |
| Novice | 20 | 18 | 14 | 10 | 6 | 68 |

Pair 方向全部正确。逆信号（chosen_reward < rejected_reward）从 51.7% 降到 **33.1%**。

关键改善：
- Novice turn=0 逆信号：85% → **30%**
- Experienced turn=0 逆信号：64% → **21%**
- Experienced turn=0 覆盖率：14/20 → **19/20**

### 14. 发现：Novice clarified 代码质量不稳定

对比同一 state 的代码 pass rate：

| 版本 | 说明 | 表现 |
|------|------|------|
| direct | 不澄清直接写 | baseline |
| clarified | Novice 多轮澄清后写 | 2/15 better, 5/15 worse |
| ideal_disclosed | 一次性给全部信息 | 多数 ≥ direct |
| oracle | 完整原始 query | 天花板 |

信息本身有用（ideal_disclosed 验证过），但 **Novice 多轮对话后 context 太长，gpt-4o-mini 代码生成质量反而下降**。

例：BigCodeBench/22: direct=1.00, ideal=1.00, 但 clarified=0.00

**待排查**：是否是 prompt 拼接方式的问题（多轮对话历史干扰代码生成）。

### 下一步

1. **排查 Novice clarified 代码质量问题**：检查最终 Execute 的 prompt，看对话历史是否干扰代码生成
2. 修复后重新生成 20 states 验证
3. 跑 150 states 正式数据（gpt-4o-mini）
4. 训练 v27

---

## 2026-04-07

### 1. Reward 重设计：删除所有 persona 参数

决定从 reward 函数中删除 persona_adjustment 和 Busy additional_penalty，改为 unified reward：

    R = task_score - w_interrupt * interrupt_cost

理由：reward 衡量客观代码质量，persona 偏好不应影响 reward 计算。否则论文里难以自圆其说。

修改文件：reward/compute_rewards.py
- 删除 compute_rewards_for_group 里的 Busy additional_penalty (~12行)
- 删除 compute_trajectory_level_rewards 里的 persona_adjustment 块 (~100行)
- 两处均替换为统一公式

---

### 2. Behavior-First Pair Construction

Unified reward 导致新问题：task_score 天然偏高 Execute，所有 turn=0 对都变成 Execute chosen，persona-blind。

解决方案：preference 方向由 persona 设计决定，不由 reward 大小决定。

新增 get_correct_action(persona_name, dialogue_turn) 函数：
- Busy-Developer：任何 turn 都返回 Execute
- Novice-Learner：任何 turn 都返回 Clarify
- Experienced-Engineer：turn=0 返回 Clarify，turn>=1 返回 Execute

主循环逻辑：先查 get_correct_action，chosen = correct action 对应的最优轨迹，rejected = 另一个 action 的最优轨迹。reward 只用于在同 action 内选最优轨迹，不再决定 chosen/rejected 方向。

Method B：加了 persona_name == "Novice-Learner" 过滤，去掉 reward gate。
Method A：改为 get_correct_action 决定方向。
Method C：强制 chosen=Execute rej=Clarify（Experienced turn>=1 应 Execute）。

---

### 3. Rebalance 改为按 (state, persona, turn) 分组

旧版按 (state, persona) 分组，会把 Experienced 的 turn=0 pair 和 turn=1 pair 合并取一个最优，导致 turn=1 Execute pair 被丢弃。

新版按 (state, persona, turn) 三元组分组，保留所有 turn 的 pair，再筛选三个 persona 都有 pair 的 complete states。

---

### 4. 验证：3 states + 18 states 小规模测试

跑了 3 states 和 18 states 的轨迹生成 + compute_rewards，结果符合预期：

18 states 的 pair 分布（64 pairs）：
- Busy turn=0：17 pairs，其中 4/17 behavior-first 覆盖（reward 建议 Clarify 但强制 Execute）
- Experienced turn=0：17 pairs，0/17 覆盖（reward 自然支持 Clarify，无需覆盖）
- Experienced turn=1：13 pairs，12/13 behavior-first 强制 Execute（reward 建议 Clarify）
- Novice turn=0：17 pairs，17/17 behavior-first 强制 Clarify（reward 建议 Execute）

Experienced turn=1 的 Execute pair 由 Method C fork 提供，每个 state 都正常触发。

---

### 5. 发现：task_uncertainty 全部为 0.3

所有 18 个测试 state 的 task_uncertainty 都是固定值 0.3。

后果：Novice 在 turn=1 的判断条件 task_uncertainty > 0.3 不成立（0.3 > 0.3 = False），所以 Novice 也在 turn=1 直接 Execute，没有出现 Clarify -> Clarify -> Execute 的多轮模式。

当前三个 persona 的轨迹模式完全相同（都是 Execute 或 Clarify -> Execute），persona 差异完全靠 pair 标签方向体现，不靠轨迹本身长度体现。

待决策：是否修复 task_uncertainty 分布或调低 Novice turn=1 阈值，让多轮 Clarify 出现。

---

### 6. 修复 target_execute_ratio 默认值

默认值 0.8 会在 state-aligned rebalance 之后再做 action rebalance，把 Experienced turn=0 Clarify pair 和 Novice Clarify pair 全部删掉。

修复：将默认值改为 -1（禁用），compute_rewards.py 第 1168 行。

---

### 7. 调查 task_uncertainty=0.3 问题：虚惊一场

debug 数据（18/20 states）用的是 dummy query "帮我写个 Python 爬虫"，导致 task_uncertainty 固定为 0.3。

真实 bigcodebench 数据的 task_uncertainty 分布：
- 范围：0.80 ~ 0.90，均值 0.82
- Novice 阈值 0.3：0.82 > 0.3 = True，多轮 Clarify 正常触发
- task_uncertainty 问题是 debug dummy query 的假象，无需修复

---

### 8. 真实数据端到端验证（5 states + gpt-4o-mini）

用真实 bigcodebench 数据 + gpt-4o-mini API 跑了 5 states 完整流程：

**轨迹生成结果（113 条，5 states × 3 personas × 3 samples）：**
- Busy 主线：Execute（10条），Clarify→Execute（5条，强制生成用于对比）
- Experienced 主线：Clarify→Execute（10条），Execute（5条，强制生成）
- Novice 主线：Clarify→Clarify→Clarify→Clarify→Execute（6条），Clarify→Clarify→Clarify→Execute（2条）等多轮模式

生成内容真实有效：
- Execute 轨迹包含真实 Python 代码（```python ... def task_func ...）
- Clarify 轨迹包含真实澄清问题（"To better understand your requirements, could you clarify..."）

**DPO pair 生成结果（29 pairs，rebalance 后）：**

| Persona | Turn | Chosen | Rejected | 数量 |
|---------|------|--------|----------|------|
| Busy | 0 | Execute | Clarify | 5 |
| Experienced | 0 | Clarify | Execute | 3 |
| Experienced | 1 | Execute | Clarify | 4 |
| Novice | 0 | Clarify | Execute | 5 |
| Novice | 1 | Clarify | Execute | 5 |
| Novice | 2 | Clarify | Execute | 4 |
| Novice | 3 | Clarify | Execute | 3 |

关键验证点：
- Novice turn=1/2/3 的多轮 Clarify pairs 全部出现 ✓（之前 debug 数据缺失的部分）
- Experienced turn=1 切换 Execute ✓
- behavior-first 覆盖：12/29 pairs 的 chosen_reward < rejected_reward，方向被强制纠正 ✓

**结论：完整流程验证通过，可以跑 150 states 正式数据。**

关键文件：
- 轨迹：`data/logs/traj_v27_real_5states_llm_20260407_044214.jsonl`
- Pairs：`data/dpo/prefs_v27_real_5states.jsonl`

---

### 下一步

1. 跑正式 150 states 轨迹生成（gpt-4o-mini，预计 1-2 小时）
2. compute_rewards 生成 pair 数据
3. 训练 v27

---

## 2026-04-06

### 1. 定位并修复三个核心 Bug

#### Bug 1：action selection 使用 single-token argmax（已修复）

**现象**：v21 / v23 评估 100% Execute，Clarify F1 = 0%

**根因**：`pick_action_from_logits` 比较 logit[Clarify=128256] 与 logit[Execute=17617]。
`Clarify` 是新加的 special token，lm_head 权重弱（~0.5）；`Execute` 是预训练原有 token（~6.3）。
gap = -5.9，LoRA 无法弥合，模型永远选 Execute。

**修复**：新增 `pick_action_from_generation`（`policy/infer.py`）：
生成 30 token，检测开头是否为代码标志（` ``` `、`def`、`import` 等）→ Execute，否则 → Clarify。

---

#### Bug 2：DPO 训练数据含人工 action 前缀（已修复）

**现象**：DPO model 与 BASE model 生成完全相同，LoRA adapter 无效。

**根因**：chosen / rejected 以 `"Clarify\n..."` / `"Execute\n..."` 开头。
Llama 从不自然生成这种前缀，DPO 在极低概率区间训练，梯度近乎为零。

**修复**：`to_dpo_format`（`policy/train_dpo.py`）里 strip 掉 action 前缀，
同时去掉 special token 注册和 `resize_token_embeddings`，去掉 `_init_new_token_lm_head` 调用。

**效果（v24）**：Clarify F1 从 0% → 45.7%，Execute Rate 从 100% → 73.3%。

---

#### Bug 3：rebalance 步骤三个 persona 各自独立抽样（已修复）

**现象**：v24 / v25 三个 persona 行为完全相同（persona-blind）。

**根因**：`rebalance_prefs_by_persona`（`reward/compute_rewards.py`）三个 persona 各自
按 hash 独立选 150 pair，导致大量 state 只有 Busy pair，缺少同一任务不同 persona 对比信号。

**修复**：重写为 state-aligned 版本：
1. 找三个 persona 都有 pair 的 state 交集（149 / 150 个 state）
2. 每个 state × 每个 persona 各保留 chosen_reward 最高的一条 pair
3. 禁用 action rebalance（`target_execute_ratio=-1`）

**新数据**：`data/dpo/prefs_method_abc_150states_aligned.jsonl`，449 pairs，149 states 三 persona 全覆盖。

---

### 2. 发现更深层问题：reward 函数未区分 dialogue_turn

通过分析 aligned 训练数据按 turn 的分布：

| Persona | Turn | pairs 数 | chosen=Execute |
|---------|------|---------|----------------|
| Novice | 0 | 54 | 17%（Clarify ✓）|
| Busy | 0 | 150 | 98%（Execute ✓）|
| Experienced | 0 | 140 | 5%（Clarify ✓）|
| Novice | 1 | 92 | 10%（Clarify ✓）|
| Experienced | **1** | **9** | **11%（Clarify ✗ 应为 Execute！）**|

**根本原因**：`persona_adjustment`（`reward/compute_rewards.py`）没有考虑 `dialogue_turn`。
Experienced 在 turn=1（已 Clarify 一轮）应切换到 Execute，但 reward 函数仍给 Clarify 更高分。
- turn=1 Experienced pairs 极少（只有 9 条 vs Novice 92 条）
- 仅有的 9 条数据也在教模型对 Experienced 选 Clarify

**结论**：v26 就算训练成功也无法区分 Experienced 与 Novice，数据本身有误。

---

### 3. 模型训练历史

| 版本 | 数据 | beta | epoch | 状态 / 问题 |
|------|------|------|-------|------------|
| v21 | prefs_method_abc | 0.3 | 3 | Bug1+2，100% Execute |
| v23 | 同上 + lmhead init | 0.3 | 3 | Bug1+2，100% Execute |
| v24 | natural format | 0.3 | 3 | Bug3，Clarify F1=45.7% 但 persona-blind |
| v25 | aligned | 0.3 | 3 | Bug3 修复，但 persona-blind |
| **v26** | **aligned** | **0.05** | **5** | **训练完成，评估中** |

---

### 4. v26 结果

**Quick action check（30 states × 3 persona）**：

| Persona | Clarify | Execute |
|---------|---------|---------|
| Novice-Learner | 29/30 (97%) | 1/30 |
| Busy-Developer | 7/30 (23%) | **23/30 (77%)** |
| Experienced-Engineer | 28/30 (93%) | 2/30 |

Busy 已与 Novice 拉开差距，但 Experienced 仍与 Novice 几乎相同（符合 reward bug 预期）。

**完整评估（eval_v26_lowbeta.json）** 显示三个 persona 全部 Clarify，原因：`evaluate_dpo_model.py`
调用 `render_state(state)` 未传 persona 参数。已修复，重跑中 → `outputs/eval_v26_lowbeta_fixed.json`

---

### 5. 代码修改汇总

| 文件 | 改动 |
|------|------|
| `policy/infer.py` | 新增 `pick_action_from_generation`；`pick_action_from_logits` 标注已废弃 |
| `policy/train_dpo.py` | strip action prefix；去掉 special token 注册和 resize；去掉 `_init_new_token_lm_head` |
| `eval/evaluate_dpo_model.py` | 改用 `pick_action_from_generation`；修复 `render_state` 未传 persona 的 bug |
| `eval/evaluate_multi_turn_persona.py` | 改用 `pick_action_from_generation` + `render_state_with_persona` |
| `reward/compute_rewards.py` | 重写 `rebalance_prefs_by_persona` 为 state-aligned 版本 |

---

### 下一步

1. **修复 `reward/compute_rewards.py`**：Experienced + `dialogue_turn >= 1` → Execute reward > Clarify reward
2. 修完后重新生成轨迹和 pairs，重新训练
3. 查看 `outputs/eval_v26_lowbeta_fixed.json` 结果

---

## 2026-04-02

### 1. 检查轨迹数据

文件：`data/logs/traj_colm_3turn_persona_150states_20260402_053113_20260402_053116.jsonl`

**基本情况**：
- 150 个 tasks，1586 条记录，3 个 persona 各 450 条轨迹
- 所有记录都有 `test` / `convcodeworld_tests` 字段（数据源问题已解决）
- task_score 全部为空（reward 还没计算）

**Persona 差异（Execute rate）**：

| Persona | Execute 率 |
|---------|-----------|
| Busy-Developer | 66.7% |
| Experienced-Engineer | 66.5% |
| Novice-Learner | 43.8% |

- Busy vs Novice 差距 22.9%，超过论文要求的 15% ✓
- **问题：Busy 和 Experienced 几乎一样**，论文需要三者有区分

---

### 2. 澄清后 task success 分析

- 直接 Execute：16.7%
- Clarify-first：5.6%（整体更低）
- Clarify + 获得 edge_cases_info：9.6%
- Clarify 但**没获得** edge_cases_info：**0%**（三个 persona 全是 0）

只有 Busy 在 Clarify+EdgeInfo 时超过直接 Execute（25.9% vs 17.3%）。

---

### 3. validate_llama_gap 实验（v20）

n=90，每个 persona 各 30 个，对比 Llama 在 5 种条件下的代码生成成功率：

| 条件 | pass_rate |
|------|-----------|
| direct | 43.3% |
| old_clarified | 35.6% |
| new_clarified | 44.4% |
| ideal_disclosed | **56.7%** |
| oracle | 53.3% |

- 方向依然成立（`DIRECTION VALID`）
- gap 缩小原因：新数据 direct baseline 更高（43.3% vs 40.0%），模型本身更强
- `ideal_disclosed` 和 `direct` 差距 +13.3%，是论文的强动机论据

输出文件：`outputs/validate_llama_gap_v20.json`

---

### 4. 代码修改

**`scripts/generate_trajectories.py`**：
- 新增 `ideal_disclosed` 版本（masked query + 所有 masked_fields 直接展示）
- 重命名 `code_versions` 字段：`masked_with_clarification` → `clarified`，`masked_only` → `direct`，`full_query` → `oracle`，新增 `ideal_disclosed`

**`reward/compute_rewards.py`**：
- 同步更新 `code_versions["full_query"]` → `code_versions["oracle"]`

**`scripts/validate_llama_gap.py`**：
- 更新 `TRAJ_PATH` 指向新轨迹文件

---

### 关键文件

| 文件 | 说明 |
|------|------|
| `data/logs/traj_colm_3turn_persona_150states_20260402_053113_20260402_053116.jsonl` | 2026-04-02 生成的主轨迹 |
| `outputs/validate_llama_gap_v20.json` | v20 validate 实验结果 |
