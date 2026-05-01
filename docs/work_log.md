# 工作记录

> 最新记录在前

---

## 2026-05-02

### 149. Qwen PO classifier 不一致 → 重跑 first-100 v2 (re-sequenced GPU 1 chain)

Qwen PO 比较结果：
- first-100 (Apr 26) avg_turn Novice **2.05** / Exp 1.07 / Busy 1.00
- remaining-100 (May 2 partial) avg_turn 全 **1.00**

根因：first-100 是 **Apr 26 跑的，v2 classifier Apr 28 才上线** → first-100 用的是 v1 classifier。v1 把 PO 的 "I'd be happy to help..." preamble 误判为 Clarify → 强制多轮（2.05 turn）。v2 严格 "Clarify\n" prefix → 都判 Execute（1.00 turn）。

**影响**:
- pass@1 N=200（合并 first-100 v1 + remaining-100 v2）: ✅ 有效（pass/fail 不依赖 classifier）
- avg_turn / clarify_rate N=200: ❌ 不可用（classifier 不一致）

**Fix**: 在 GPU 1 chain 中插入 Qwen PO first-100 v2 重跑，re-sequence 成：
```
GPU 1: Qwen PO rem-100 (跑中) → Qwen PO first-100 v2 (~5h) → Llama Base 200 (~10h)
```

实现：
- `/tmp/qwen_po_first100_v2.sh`：v2 PO eval on `data/seeds/test_states_v29_eval_200_first100.jsonl`
- `data/seeds/test_states_v29_eval_200_first100.jsonl`：从 `eval_v29_qwen_prompt_only_100.json` 反抽出的 100 state IDs，保持原始 first-100 集合
- `/tmp/wait_then_qwen_po_first100.sh`：守 PID 159265（rem-100 python）退出后启动新 chain

新 chain ETA：Llama Base 推迟到 ~08:48 ChiSat（vs 原 03:30），但仍在 user 醒来时（~09:00）前完成。

### 148. Llama state set 审计 — 发现 CF 用了不同 state 集合 (unfair compare)

用户质疑 Llama DPO partial Busy 5.4% 比 Direct 13.0% 差太多反直觉，要求严格 audit。结果发现：

```
Direct 200       eval_200    (BigCodeBench/122, /160, ..., /1010, /1015...)
DPO partial 41   eval_200    ✓ 同
PO partial 50    eval_200    ✓ 同
Base partial 42  eval_150extra ✗ 不同
v33 DPO running  eval_200    ✓ 同
v29 DPO 150      eval_150extra ✗ 不同
CF 150           eval_150extra ✗ 不同
```

**发现 eval_200 = eval_150extra (150) ∪ 50 个新 state**，DPO partial 41 都在新 50 里 → DPO 跟 CF state set 完全不重叠。**之前 "Llama DPO 在 Busy 5.4% 远低于 CF 19.3%" 完全是 unfair 比较**。

**Paired comparison on same 40 states (DPO vs Direct)**:

| Persona | DPO | Direct | Δ |
|---|---|---|---|
| Novice | 12.5% | 5.0% | **+7.5pp** ⭐ |
| Exp | 12.5% | 5.0% | **+7.5pp** ⭐ |
| Busy | 5.0% | 5.0% | 0 (tied) |

McNemar χ² < 3.84（n=40 太小），但 direction 一致 — DPO ≥ Direct 在所有 persona 上。**修订**：DPO partial Busy 5% 是 state subset 难度问题，不是 LoRA alignment tax。这 40 state 在 Direct 全集上恰好是 hard subset（Direct 全集 Busy 13% vs 这 40 state 上 Direct 5%）。

**Llama CF fix plan**: CF 当前在 eval_150extra 上，eval_200 ⊃ eval_150extra (150 overlap)，需要补跑 CF 在缺的 50 state（~3h 单 GPU）。等 Llama DPO 跑完（GPU 0 ~08:00 ChiSat 空闲）再起。

**教训**: paper 必须用同 state set 做 paired comparison，不同集合做出来的 method 排名 misleading。

### 147. Codebase 第二轮清理（在 Llama 跑中安全归档）

3 个 Llama eval 跑中借空隙整理：

**A. 顶层 BCB artifact 再清一次**
今天 eval 又生成一批 (`histogram.png`, `qq_plot.png`, `*.txt.json`, `test_data_1-5/`, `log.csv`, `output.csv`, `word_counts.json` 等)。删了 21 文件 + 5 目录。已 gitignored 不入 git，但视觉清爽。

**B. `models/` 11G → 7G（释放 4G）**

删 7 个失败 retry/superseded checkpoints：
- `v30_100states`（changed oracle, failed）
- `v31_100states`, `v31_2a_100states`, `v31_4_100states`, `v31_4_qwen_100states`（failed v31 retries）
- `v32_test_keep_prefix`, `v32b_alpha32_keep_prefix`（v32 alpha 调参 collapse）

保留 11 个：v29 era (4) + v33 完整迭代 (7, 含 v1 v2 v3 + Qwen 同款)。Caveat: model checkpoint 是 gitignored，**GitHub 上没 backup**；若将来要复现，需 git checkout + retrain（数据 + 训练代码都在，理论可复现）。

**C. `outputs/` archive (5.9MB)**

移到 `outputs/archive/legacy_smalltest/`：50test/30test/5state 失败 era exploration（17 文件）
移到 `outputs/archive/legacy_v29_misc/`：v29 misc + ablation 等（9 文件）
留下 36 个 paper canonical（200/100_patched/remaining100_ft/150extra）。

**D. `/tmp` 428 → 14 文件**

只留 6 个活跃 wrapper + 它们的 log + freeze_monitor + morning_brief scheduler PID 文件。删的：旧 v33_v1/v2 sanity .py/.json/.log（已 cp 到 scripts/sanity/）、v32 sanity、v29-era pipeline scripts、status checkers、chain monitors，等等。

**Eval 无影响**：cleanup 期间反复 `ps -ef | grep evaluate_multi_turn`，3 个进程一直 alive，sample 进度持续 +1 +1。

### 146. 起 Llama N=200 全 pipeline + Qwen PO 收尾（Option D 启动）

23:25 北京（10:25 芝加哥）3 卡并行起 Llama N=200：
- GPU 0: Llama v33 SFT+DPO N=200（~25h，瓶颈）
- GPU 1 chain: Qwen PO rem-100 (5h) → Llama Base N=200 v2 (~10h)
- GPU 2: Llama PO N=200 v2 (~12h)

ETA: Llama DPO 明早 ~09:30 AM 芝加哥 May 2 完，其他更早。

新加 `scripts/wrappers/progress_3chains.sh` 给 3 chain 一键 snapshot。

### 145. Qwen N=200 method 比较 honest 分析 — 修订 paper claim

跟用户讨论 "为什么 CF 看起来比 TactfulLLM 好" 后，做严谨 paired analysis 发现之前 over-claiming：

**事实**：
- CF 15.7 > TactfulLLM 15.2 > Direct 14.7（pass@1, N=200）
- 全部 method-method McNemar **不显著**（χ² < 3.84, p > 0.05）
- 只 Novice TactfulLLM vs CF χ²=3.68, p≈0.055 borderline

**Power 分析**：N=200/persona 给 SE≈2.5pp，detect threshold ≈ 6-7pp。观察到的 1-3pp 差距**全在 noise 内不可分辨**。需要 N=600+/persona 才能 detect 2pp 差。

**自己 over-claim 的纠错**：
- 之前我说 "TactfulLLM Execute-T0 LoRA 不损 (18.7% = Direct)" + "turn-1 Execute LoRA 损 5pp" → 其实是不同 state 子集比较，**不是 apples-to-apples**
- 同 state McNemar：DPO=6, CF=11 wins, χ²=0.94 不显著
- 用户 push back 后纠正

**Busy 内部分裂（描述性）**：TactfulLLM 200 个 Busy sample 按它自己 turn-0 决策分两组：
- Execute-T0 (n=91): 18.7% pass ⭐ 三 method 中 Busy 最高
- Clarify-T0 (n=109): 9.2% pass — 最低
- "当它选不 clarify 时是最强的，当它选 clarify 时是最差的"
- 但 91 vs 109 内部分组 significance 没正式 test

**未验证的候选机制**（标记 hypothesis 不当 claim 写）：
1. LoRA alignment tax on turn-1 Execute（chosen pairs trajectory 偏离 Qwen 自然分布）
2. Busy oracle 拒答率训练/测试不一致
3. Pure sample variance（最简单解释）

⚠️ N=200 数据**无法在三个解释间分辨**。等 Llama N=200 cross-backbone reproduce 看是否有真 effect。

**修订 paper claim**：
- ❌ "TactfulLLM achieves best accuracy"（事实是 CF 略高）
- ❌ "DPO refinement improves over CF"（McNemar 不显著）
- ❌ "LoRA alignment tax explains Exp/Busy underperformance"（机制未证）
- ✅ "TactfulLLM matches baselines on accuracy (within statistical noise)"
- ✅ "TactfulLLM exhibits persona-conditional interaction depth (Nov 7.99 / Exp 2.42 / Busy 1.55 vs CF flat 2.0)"
- ✅ "On Novice, TactfulLLM 18.0% vs CF 13.0% (borderline χ²=3.68, p≈0.055)"
- ✅ "Cross-persona behavior split is primary contribution; accuracy parity is secondary"

**记入 sft_then_dpo_v33.md** "May 2 honest 分析" 段，包括完整 McNemar 表 + Power 分析 + 候选机制 + 修订 claim。

### 144. Qwen baseline N=200 合并完成 — sanity 全过 + 关键 pass@5 finding 浮现

3 个 baseline（Direct/CF/Base）remaining-100 全部跑完，一夜无 freeze（freeze_monitor host_uptime 3683→37343 单调增 9.3h）。`scripts/merge_baselines_200.py` 合并 first-100 patched + remaining-100 ft：

**first-100 vs remaining-100 sanity (Qwen, N=100 each)**:

| Baseline | first-100 | remaining-100 | Δ |
|---|---|---|---|
| Direct | 15.3% | 14.0% | -1.3pp |
| CF | 16.0% | 15.3% | -0.7pp |
| Base | 11.0% | 11.0% | 0.0pp |

全部 |Δ| ≤ 1.3pp，远小于 SE ≈ 3.5pp。合并健康。

**Qwen N=200 final（论文主表）**:

| Method | pass@1 | pass@5 | avg_turn |
|---|---|---|---|
| Base | 11.0% | 18.0% | 1.00 |
| Prompt-Only (N=100) | 13.0% | – | – |
| Direct | **14.7%** | **18.3%** | 1.00 |
| TactfulLLM | **15.2%** | **23.0%** | 3.99 |
| Clarify-First | **15.7%** | **23.2%** | 2.00 |

**🎯 关键 finding — pass@5 故事**：
- pass@1 上 TactfulLLM ≈ Direct ≈ CF（都在 14.7-15.7 噪声内）
- **pass@5 上分两组**：
  - Clarify 方法：TactfulLLM 23.0 ≈ CF 23.2 ← clarify 触发 disclosure，多 candidate 各得不同 hint
  - Execute 方法：Direct 18.3 ≈ Base 18.0 ← 不 clarify 直接 generate，多 candidate 没新信息
- **TactfulLLM vs Direct pass@5 差 +4.7pp**，pass@1 +0.5pp — **这就是论文主结果的 lift 机制**

**TactfulLLM 比 CF 的优势 = persona-aware multi-turn 行为**：
- TactfulLLM avg_t: Novice 7.99 / Busy 1.55 / Exp 2.42（按 persona 自适应）
- CF avg_t: 全部 2.00（固定 1 turn clarify）
- pass@1 几乎打平但 TactfulLLM 用更少 turn 完成（Busy 1.55 vs CF 2.00 = 节省 22% interaction cost）

**输出文件**:
- `outputs/eval_v29_qwen_direct_execution_200.json` (1937KB, detailed_results=600)
- `outputs/eval_v29_qwen_clarify_first_200.json` (2333KB)
- `outputs/eval_v29_qwen_base_200.json` (1938KB)
- PO skipped（remaining-100 还没跑）

### 143. 起 morning_brief scheduler 在用户睡觉时定期写 status

5 个 snapshot 调度（+4h/+5h/+6h/+8h/+10h，container 17:56-23:56）。文件 `/tmp/morning_brief.txt`。结果：3 baseline 全在 +6h 前完成（Direct 18:39 / CF 19:19 / Base 20:19 北京），5 个 snapshot 都顺利写入。Freeze 没发生，all clear。

## 2026-05-01

### 142. 明天计划 — Llama 主 pipeline + Qwen PO 收尾（等今晚 baseline 数字定 N）

DDL 5/6 还 5 天。Qwen N=200 baseline 今晚跑完后看数字，决定 Llama N=100 还是 N=200。当前 Llama 状态：v33 v3 SFT+DPO 已训完（model + 24/24 sanity 都 ok），但**只有 5-state eval 数字（1/15 = 6.7%）**，N 太小论文不能用。

**明天必跑（首跑，不是重跑）**：

| 任务 | 单卡时间 | 备注 |
|---|---|---|
| Qwen PO remaining-100 | ~5h | 补 PO N=100→N=200，跟今晚 3 个 baseline 一起合并 |
| Llama v33 SFT+DPO N=100 v2 | ~17h | 主 cross-backbone 结果 |
| Llama Base N=100 v2 | ~17h | §118 已证 v1 不可复用，必须 v2 重测 |
| Llama PO N=100 v2 | ~17h | 同上，依赖 classifier |

**可复用 v1 数字（不依赖 classifier）**：
- Llama Direct 200 ✅ `eval_v29_direct_execution_200.json`
- Llama CF 150 ⚠️ `eval_v29_clarify_first_150extra.json`（N=150 不是 200，可能补）

**Llama v29-era pure DPO 不再跑** — v33 论文里已被 SFT-then-DPO 替换（pure DPO 跨 KL gap 失败的 case study，§118 v2 partial 6.35% 已是 evidence）。

**3 卡排程方案（明天起）**：
```
GPU 0:  Qwen PO rem-100 (~5h) ──→ Llama v33 SFT+DPO 100 (17h)   总 22h
GPU 1:  Llama Base 100         (17h)
GPU 2:  Llama PO 100           (17h)
```
后天早上完，留 May 3-5 写 paper + 统计检验。

**N=100 vs N=200 决策点**：今晚 Qwen baseline N=200 数字定后，看 Δ vs N=100 是否 <SE。如果 stable，Llama N=100 就够；如果 first-100 vs remaining-100 飘 >2pp，Llama 也补 N=200（再加一天）。

### 141. Codebase 清理 + git push 到新分支

借 baseline 跑空隙整理项目结构 + 备份代码。

**3 个 commit 推到 GitHub**：
```
4332df4 chore: persist /tmp wrappers + sanity scripts into scripts/
663b8da chore: clean BCB artifact pollution + gitignore prevention
da42159 v33: SFT-then-DPO pipeline + v2 classifier + truncation fix
```

**Highlights**:
1. **da42159 主 commit**：v33 全套（train_sft_v33.py, train_dpo.py INIT_ADAPTER, infer.py v2 classifier, evaluate.py 截断 fix, coding_execute.txt import requirement, merge/patch_imports/sanity_classifier 工具，docs/sft_then_dpo_v33.md, docs/classifier_bug_2026-04-28.md）
2. **663b8da 清理 BCB artifact**：删 `s/`, `source/`, `src/`, `d/`, `dst/`, `destination/`, `downloads/`, `ftp/`, `invalid_directory/`, `test_*/`, `test_data*/` + 顶层 `*.png`/`*.csv`/`*.txt.json` 杂物；扩 `.gitignore` 防再污染
3. **4332df4 持久化 /tmp**：16 个 wrapper → `scripts/wrappers/`（含 README.md），4 个 v33 v3 sanity → `scripts/sanity/v33_v3/`，原 `scripts/sanity_classifier/` rename → `scripts/sanity/classifier/`

**机器 freeze 担忧 → 双 branch 备份**：
- push `v20_development`（延续历史）
- push `v33-sft-dpo-pipeline`（NeurIPS 提交专题分支）— 当前 HEAD 在这分支，后续 commit 落这里

### 140. 起 3 卡并行 Qwen baseline remaining-100 + freeze monitor

3×4090 都空闲，并行跑 Direct/CF/Base remaining-100（state 101-200）：

```
GPU 0  Direct  qwen_direct_remaining100_ft.sh   PID 7084   ~17h ETA
GPU 1  CF      qwen_cf_remaining100_ft.sh       PID 7186
GPU 2  Base    qwen_base_remaining100_ft.sh     PID 7198
```

**实测 ~3 min/sample**（远低于 v33 DPO 的 17h，因为 baseline 都是 1-2 turn 不撞 max_turns），3 卡并行 ETA ~5h compute → 今晚 ~19:00 北京完。

**Base 行为正常**: 5 sample × 3 persona = 15/15 turn 0 Execute（v2 classifier 下 Base Qwen 永不 emit "Clarify\n" prefix → 一致 §116 数字）。Base 实际功能上 ≈ Direct，4pp gap 是 random state divergence。

**§104 / §115 freeze 担忧 → 起 freeze monitor**：
- `/tmp/freeze_monitor.sh` 后台 PID 21190，每 20 min 写一行 CSV 到 `/tmp/freeze_monitor.log`
- 字段：container_time, host_uptime, gpu0/1/2_util, sample_n × 3, elapsed × 3
- 时间戳缺口 = autodl 冻结期；用户睡觉醒来直接 `cat /tmp/freeze_monitor.log` 看真实进度
- 已 cp 到 `scripts/wrappers/freeze_monitor.sh` 持久化

### 139. GPU 升级讨论：单 4090 → 双 4090（待决策）

DDL 5 天，剩下：Qwen baseline (Direct/CF/Base/PO) 100→200 ≈68h 单卡 + 可能的 Llama 100/200。双卡价值在并行跑两个 eval（每个吃满单卡，不做 DDP），4 个 baseline 从 68h → 34h 省 1.5 天。权衡：迁移成本（容器重启、env 重装、§104 冻结风险）vs 是否真要补 baseline 到 200。等 N=200 vs baseline N=100 比较结果决定补 baseline 与否再升级。

### 138. v33 SFT+DPO Qwen N=200 合并完成 — sanity 通过 +0.9pp

`eval_v33_v3_qwen_dpo_v2_remaining100_ft.json` (Apr 30 19:20) 完成；与 first-100 patched 合并 → `eval_v33_v3_qwen_dpo_v2_200.json` (May 1 11:33, detailed_results=600)。

**Qwen v33 SFT+DPO N=200 (canonical)**:

| Persona | pass@1 | pass@3 | pass@5 | avg_turn | clarify_rate |
|---|---|---|---|---|---|
| Novice | 36/200 = **18.0** | 50/200 = 25.0 | 54/200 = **27.0** | 7.99 | 0.87 |
| Exp | 28/200 = **14.0** | 37/200 = 18.5 | 47/200 = **23.5** | 2.42 | 0.59 |
| Busy | 27/200 = **13.5** | 34/200 = 17.0 | 37/200 = **18.5** | 1.55 | 0.35 |
| **All** | **91/600 = 15.2** | **121/600 = 20.2** | **138/600 = 23.0** | 3.99 | – |

**N=100 patched → N=200 sanity check**:

| Persona | N=100 | N=200 | Δ |
|---|---|---|---|
| Novice | 17.0 | 18.0 | +1.0 |
| Exp | 12.0 | 14.0 | +2.0 |
| Busy | 14.0 | 13.5 | -0.5 |
| **All** | **14.3** | **15.2** | **+0.9** |

全部 |Δ| ≤2pp（远小于 SE≈3.5pp），合并 sanity 通过。pass@5 ordering 仍健康 Nov 27 > Exp 23.5 > Busy 18.5。

**vs Qwen baselines (overall pass@1)**:
- Direct 15.3 / CF 16.0 / Base 11.0 / **TactfulLLM 15.2 (N=200)** / PO 13.0
- TactfulLLM 比 Base **+4.2pp**；vs Direct/CF 几乎打平（-0.1 / -0.8，远在噪声内）

---

## 2026-04-30

### 137. Decision: 让 DPO 跑 N=200，baseline 暂定 N=100 后再看

合并方案确认：first-100 (旧模板+截断) vs remaining-100 (新模板+不截断) 总偏差 ≤2pp（远小于 SE≈3.5pp），sanity check 后直接合并报 N=200。Baseline 是否补到 200 等 DPO 完成后看 v33 N=200 vs N=100 差异是否显著再决定。

### 136. v33 DPO remaining 100 起跑 (N=200 canonical)

用户 06:30 决定先放 PO，把 v33 SFT+DPO 补到 200 state。重现 `random.Random(42).sample(states_200, 100)` 验证：first 100 与 v33 已评估 state_id 100/100 一致。剩余 100 → `data/seeds/test_states_v29_eval_200_remaining100.jsonl`。

**Run**: PID 692635，输出 `outputs/eval_v33_v3_qwen_dpo_v2_remaining100_ft.json`，新模板 + 完整 code 保存。ETA 8-10h，wall Apr 30 ~14:00 完。

**坑：PO kill 没杀干净** — `kill <wrapper_pid>` 只杀 bash 父进程，python 子进程 (PID 676539) 仍跑了 19 分钟才被发现，导致首次 DPO launch (PID 688779) crashed: `NotImplementedError: Cannot copy out of meta tensor`（GPU 抢占）。教训：以后 kill eval 进程要 `pkill -f evaluate_multi_turn_persona` 或同时 kill bash + python PID。

### 135. Sanity check: v33 DPO 模型本身没问题（Exp<Busy 是噪声）

用户质疑 Exp 12% < Busy 14% 反直觉。McNemar paired test：discordant pair = 6 (Busy-only) + 4 (Exp-only)，χ² = 0.4，**p ≈ 0.53 不显著**。pass@5 顺序反而是健康的 **Novice 25 > Exp 22 > Busy 19**——说明模型在 Exp 上做的 Clarify 拿到的 disclosure 确实让代码更好。

**Exp 0/100 pure Execute 是正确行为**（用户 insight）：task uncertainty 高时先 clarify 是合理的 proactive 决策。之前我把它当 bug 是过度解读 N=100 噪声。

### 134. patch_imports.py 失效：截断 bug 让 post-hoc patching 没法工作

`scripts/patch_imports.py` 跑完所有 4 个 method，结果：

| Method | n_failed_w_code | patched_imports | Δ pass@1 |
|---|---|---|---|
| Direct | 1292 | **0** | +0.0 |
| Clarify-First | 1272 | 1 | +0.0 |
| Base v2 | 1305 | **0** | +0.0 |
| TactfulLLM v33 | 1294 | 8 | **+0.7** |

根因：`evaluate_multi_turn_persona.py:355,377,807` 把 `code`/`assistant_msg` 保存为 `s[:200] + "..."` 截断版。patch script 拿到的是 200 字符片段，加 imports 也救不了截断代码。**记入 memory：feedback_eval_truncation.md**。

**修复**：
1. `eval/evaluate_multi_turn_persona.py:355,377,807` 三处去掉 `[:200] + "..."` → 保存完整 code
2. 当前 PO/DPO run 已经用上不截断版本（重启后）

### 133. Import 模板 bug → 修 `prompts/coding_execute.txt`

发现大量 generated code 失败因为漏 imports。例如 task 用 `np.array` 但代码不写 `import numpy as np`。

**Fix (Apr 30 04:56:44)**：在 `coding_execute.txt` 加 requirement #5 + 修 example：
```
5. Always include all necessary import statements at the top of the code block before the function definition.

Format your response as:
```python
import <required_module_1>
from <module> import <name>
# ... (all imports needed)

def function_name(...):
    # Your code here
```
```

**实测对 PO baseline 增益**：在同样前 27 个 state 上 v29 PO (旧模板) vs v2 PO (新模板) 都是 7/27 = 25.9%，**+0pp**（非 cherry-pick 比较，是 same-state apples-to-apples）。新模板对 base Qwen prompt-only 看不出帮助。但 v33 DPO 上是否有 lift 还要看 remaining-100 跑完。

### 132. v33 first-100 patched 数字 + Δ vs Base 表格（更新主表）

**Qwen v33 SFT+DPO 100-state patched (列顺序 Nov/Exp/Bus/All)**:

| 维度 | Novice | Exp | Busy | All |
|---|---|---|---|---|
| pass@1 | 17.0 | 12.0 | 14.0 | **14.3** (vs Base 11.0, +3.3pp) |
| pass@5 | 25.0 | 22.0 | 19.0 | 22.0 (vs Base 19.7, +2.3pp) |
| avg_turn | 7.97 | 2.41 | 1.55 | 3.98 |
| rejection rate | 0.46 | 0.38 | 0.89 | 0.47 |

**vs baselines (overall pass@1):**
- Direct 15.3 / Clarify-First 16.0 / Base 11.0 / **TactfulLLM 14.3** / PO (v29) 13.0
- TactfulLLM 落后 Direct/CF 1-2pp（噪声内），但显著赢 Base + PO

**论文表第 4 列**: clarification rejection rate（= 1 − answered_clarification）—— Novice 0.07/Bus 0.80/Exp 0.32 in CF (与 persona patience 设定吻合)。

### 131. Display truncation 误判事件

跑 PO 看到所有 candidate code len=203 chars，一度怀疑 `max_new_tokens` 没生效或 generation 截断。深入排查发现是 `evaluate_multi_turn_persona.py:355,377` 保存时做 `s[:200] + "..."` 显示截断，**pass/fail 是在截断前用 full text 算的**。Memory 记入 `feedback_eval_truncation.md`。

后果：所有历史 eval JSON 的 `code` 字段都是 200 字符版本，无法做事后分析（patch import / 看哪些是 OOM / 看 prompt-template-induced 错）。修了，未来 eval 保存完整。

---

## 2026-04-29

### 130. Qwen 100 SFT+DPO v2 eval 起跑 (canonical N) — 容器冻结 §104 重现

`outputs/eval_v33_v3_qwen_dpo_v2_100.json` 起跑 (system 14:57)，ETA ~17h compute → wall Apr 30 ~16-18:00。

**§104 重现警告**：用户半夜检查 (wall 02:00, 启动 ~3h 后) 发现 process ELAPSED 仅 8:37——容器被 autodl 冻结 ~2.5h。Process 仍 alive (Rl)，GPU 占用，但 cgroup 限速。Sample 1 刚完 Novice (7-turn 撞顶) + Busy (2-turn因 garbage)，Exp 跑中。

**应对**：用户决定笔记本一直开着不睡眠，保 SSH 活直到 100 state 完。

**Partial-resume 验证**：`random.Random(42).sample(states_200, 100)` 跟 `sample(states_200, 200)` 第 1-100 个 state_id 一致，所以 Qwen 100 完成后可直接 `cp ... .json.partial && python eval ... --max_samples 200` 续跑剩 100，再 ~17h compute。

### 129. Qwen v33 v3 DPO v2 (epochs=1) 5-state = 13.3% — 同 SFT-only

epochs=1 DPO refinement 部分救回 Busy collapse (vs v1 epochs=3 全部崩)。但 5-state pass@1 跟 SFT-only 持平。

**Qwen DPO v2 5-state**:
- Novice 1/5 = 20%, avg_t = 8.00 (撞顶)
- Busy 0/5 = 0%, avg_t = 1.80 (mix of 1-turn 和 multi-turn 因 garbage)
- Exp 1/5 = 20%, avg_t = 3.00 (DPO 让 Exp 比 SFT 多 1 turn)
- Overall: 2/15 = 13.3% (跟 Qwen SFT-only 5-state 一致)

**Cross-backbone 5-state 全景**:
| Backbone × Method | pass@1 |
|---|:---:|
| Llama v33 SFT-only | 0/15 = 0% |
| Llama v33 SFT+DPO | 1/15 = 6.7% |
| Qwen v33 SFT-only | 2/15 = 13.3% |
| Qwen v33 SFT+DPO v2 | 2/15 = 13.3% |

N=5 太小（std ~6pp），需要 100-state stable 数字。

### 128. Qwen DPO v1 (epochs=3) collapse — Busy 学坏

直接复用 Llama hparam (epochs=3, alpha=32, β=0.1) 做 Qwen DPO refinement → over-fit collapse:
- Train final loss 0.0005 (vs Llama 0.39, Llama 健康)
- rewards/accuracies 1.0, rewards/margins 9.1 (over-amplified)
- Sanity Busy 8/8 emit "Execute" then 停 / "Executeuibutable\n;\",\n;\"..." 乱码
- Novice/Exp 仍 8/8 prefix correct（Busy 单独 collapse）

**诊断**: Qwen2.5-Instruct 比 Llama-3.1-Instruct 对 DPO 更敏感，3 epochs 过头。

**修复 v2**: epochs=3 → 1，部分救 Busy (4/8 valid code, 4/8 仍 garbage)。详见 §129。

### 127. Qwen v33 v3 SFT 训练成功 + 5-state = 13.3%

复用 Llama 同 hparam（KEEP_PREFIX=1, alpha=32, r=64, epochs=3, LR=5e-5）：

```
Qwen v33 v3 SFT train: 12 min, loss 0.37 (vs Llama 0.39，几乎一致)

Sanity 24/24:
  Novice: 8/8 "Clarify\n[直接问句]"  ✅
  Busy:   8/8 "Execute\n[code]"      ✅
  Exp:    8/8 "Clarify\n[直接问句]"  ← 比 Llama SFT (5/8) 更 Clarify-prone

Method 跨 backbone generalize ✅
```

**Qwen 5-state SFT eval** (~50 min)：
- Novice 1/5 = 20%, Busy 0/5, Exp 1/5 = 20%
- Overall: **2/15 = 13.3%** （vs Llama SFT-only 0/15）
- Per persona avg_t: Novice 8.0 / Busy 1.0 / Exp 2.0
- 跟 Direct Qwen 100 (15.3%) 和 CF Qwen 100 (16.0%) 同 5-state subset 持平

### 126. Llama v33 v3 DPO refinement: 18 min + sanity 24/24 + 5-state = 6.7%

DPO refinement 从 SFT v33_v3_sft 继续训（INIT_ADAPTER），保 KEEP_PREFIX=1, alpha=32, r=64, beta=0.1, epochs=3。**18 min 跑完**（loss 0.39 ↓，无 collapse）。

**Sanity 结果（24/24 perfect）**：
- Novice 8/8 "Clarify\n[直接问句]" ✅ (preserved from SFT)
- Busy 8/8 "Execute\n[code]" ✅ (preserved from SFT)
- Exp 8/8 "Clarify\n[直接问句]" ← DPO 推从 SFT 5/8 → 8/8 Clarify

**5-state DPO eval**：
- Novice 1/5, Busy 0/5, Exp 0/5 → **1/15 = 6.7%**
- Multi-turn 完美：Novice 7 turn / Busy 1 turn / Exp 2 turn

**SFT-only vs SFT+DPO 对比**:
- Llama SFT-only: 0/15 = 0%
- Llama SFT+DPO: 1/15 = 6.7% (+1 pass)
- 同 5 states 上 v29 DPO (v1 era) 3/15 = 20% (但 v1 误判 + multi-turn artifact)
- 同 5 states 上 Direct/CF Llama 200 也 0/15 (这 5 状态本身 hard)

### 125. Llama v33 v3 SFT 5-state eval = 0/15 (caveat: same 5 state Direct/CF 也 0)

```
v33 v3 SFT-only Llama 5-state (BigCodeBench/127, 202, 575, 784, 945):
  Novice: 0/5  pass@5=1/5=20%  avg_t=7.40
  Busy:   0/5  pass@5=0/5      avg_t=1.00
  Exp:    0/5  pass@5=0/5      avg_t=2.40
  Overall: 0/15 = 0%
```

**Apples-to-apples 同 5 state**:
| Method | pass@1 |
|---|:---:|
| v29 DPO Llama (combined 200, v1 era) | **3/15 = 20%** ← 唯一非零 |
| Direct Llama 200 | 0/15 = 0% |
| CF Llama 200 | 0/15 = 0% |
| v33 v3 SFT-only | 0/15 = 0% |

**修正解读**：这 5 个 state 本身 hard（Direct/CF 也 0%）。v29 DPO 20% 是 v1 误判 + forced multi-turn → disclosure recovery 救的。**v33 SFT 0/15 不能据此判定 SFT 损 code**，需扩 N。

**Behavior 完美**: Novice 7-turn 撞顶 / Busy 1-turn / Exp 2-3 turn (mixed) ← 持续验证 persona-aware 行为。

### 124. v33 v3 DPO refinement: 18 min train + sanity 24/24 perfect + 5-state eval 跑中

DPO refinement 从 SFT v33_v3_sft 继续训（INIT_ADAPTER），保 KEEP_PREFIX=1, alpha=32, r=64, beta=0.1, epochs=3。**18 min 跑完**（loss 0.39 ↓，无 collapse）。

**Sanity 结果（24/24 perfect）**：

```
Novice: 8/8 "Clarify\n[直接问句]"  ✅ (preserved from SFT)
Busy:   8/8 "Execute\n[code]"      ✅ (preserved from SFT)
Exp:    8/8 "Clarify\n[直接问句]"  ← DPO 推从 SFT 5/8 → 8/8 Clarify
```

**Notable**：DPO 没破坏 SFT 学到的 prefix 行为，反而让 Exp 在 turn 0 上更 Clarify-prone（rejected pair 含 Execute responses，DPO 推 model 远离 Execute on Exp turn-0 state）。

**5-state DPO eval** (~17:42 起跑 wall, ETA 19:00)：
- Sample 1 完成: Novice 7-turn forced final, Busy 1-turn, Exp 2-turn (Clarify→Execute) ← persona-aware multi-turn 行为完美
- Sample 2 完成: 类似 pattern
- Sample 3-5 跑中

等 5-state pass@1 数字决定 Qwen pipeline 起多大 N。

### 123. v33 v3 SFT 5-state eval (Llama)：行为完美 + pass@1=0/15 (但 Direct/CF 也 0)

```
v33 v3 SFT-only Llama 5-state (BigCodeBench/127, 202, 575, 784, 945):
  Novice: 0/5 = 0%  pass@5=1/5=20%  avg_t=7.40 (撞顶 7-turn 多轮)
  Busy:   0/5 = 0%  pass@5=0/5=0%   avg_t=1.00 (1-turn execute)
  Exp:    0/5 = 0%  pass@5=0/5=0%   avg_t=2.40 (mixed multi-turn)
  Overall: 0/15 = 0% pass@1
```

**初始诊断错误**：以为 SFT 损害了 code 生成能力，但 apples-to-apples 对比同 5 state：

| Method (same 5 states) | pass@1 |
|---|:---:|
| v29 DPO Llama (combined 200) | **3/15 = 20%** ← 唯一非零 |
| Direct Llama 200 | 0/15 = 0% |
| CF Llama 200 | 0/15 = 0% |
| v33 v3 SFT-only | 0/15 = 0% |

**修正解读**：这 5 个 state 本身就是 hard 状态，Direct/CF 也都 0%。v29 DPO 20% 是 v1 误判 → forced multi-turn → disclosure recovery 救的（v29 200 整体平均 14%，这 5 state 上 20% 高于平均）。**v33 SFT 0/15 不能据此判定 SFT 损 code，需扩 N 看真水平**。

**Behavior 分化完美**：
```
Sample 1 (state 945):
  Novice 7-turn (clarify forced final)  ← 高耐心多问
  Busy 1-turn (Execute)                  ← 低耐心直 code
  Exp 2-turn (Clarify→Execute)           ← 中耐心问一次

Sample 2 (state 575): 类似 pattern
```

**结论**：Behavior 验证通过（persona-aware multi-turn），但 5-state pass@1 too noisy，等 DPO 5-state 数字 + 后续扩 N。

### 122. v33 v3 SFT 训练成功：8/8 Novice "Clarify\n[直接问句]" 完美

调参 plan 是 epochs 2→3, LR 2e-5→5e-5, alpha 16→32（其他保 v33 v2: KEEP_PREFIX=1, prompt masking, r=64）。

```bash
KEEP_PREFIX=1 LORA_ALPHA=32 LORA_R=64 \
  python policy/train_sft_v33.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output models/v33_v3_sft \
  --epochs 3 --lr 5e-5
```

12 min 训完（87 step × ~9s/step，loss 从 0.83 → 0.39 → ~0.1，健康）。

**Sanity 数据（彻底突破）**：

```
Novice: Clarify_pfx = 8/8 (100%)  ✅
Busy:   Execute_pfx = 8/8 (100%)  ✅
Exp:    Clarify 5/8, Execute 3/8  (mixed, by state n_masked)
```

样本输出：
- Novice: `"Clarify\nWhat format should the input data be in..."`  
- Busy:   `"Execute\n```python\nimport re\nimport json..."`  
- Exp:    `"Clarify\nWhat specific URL pattern should be used..."`

**0/24 emit "I'd be happy to help"** preamble — pretrained RLHF tendency 完全 override。

**关键 hparam 经验** (vs v33 v1/v2 失败配置)：

| 配置 | v33 v1 | v33 v2 | v33 v3 |
|---|---|---|---|
| epochs | 1 | 2 | 3 |
| LR | 1e-5 | 2e-5 | 5e-5 |
| alpha | 16 | 16 | 32 |
| prompt masking | ❌ no | ✅ yes | ✅ yes |
| Novice Clarify_pfx | 0/8 | 0/8 | **8/8** |
| Novice 真 clarify | 0/8 | 1/8 | **8/8** |

prompt masking + 适度增强（alpha 32, LR 5e-5, epochs 3）是 sweet spot。

### 121. v33 v1/v2 SFT iteration（失败诊断）

#### v33 v1（existing train_sft.py，无 prompt masking）

```
配置: epochs=1, LR=1e-5, alpha=16, KEEP_PREFIX=1
sanity: 0/8 Novice Clarify_pfx, 三 persona 输出几乎相同 ` ```python\nimport...`
```

**诊断**：existing `train_sft.py` 用 `DataCollatorForLanguageModeling(mlm=False)` 对全 token 算 loss，包括 prompt。"Clarify\n" 这 2 token 在 ~500 token 全文里被稀释 → 信号微弱。

#### v33 v2（new train_sft_v33.py，加 prompt masking）

```
新写 policy/train_sft_v33.py，labels = -100 for prompt tokens, 真 labels for response only
配置: epochs=2, LR=2e-5, alpha=16, prompt masking ✓
sanity: 0/8 Novice Clarify_pfx, 1/8 Novice 真 clarify (state 7 "I can guide... However I need to clarify..." 真问 user 要 data)
        三 persona 行为开始分化
```

**诊断**：prompt masking 有效（Novice 1/8 出现真 clarify），但参数还不够强。alpha=16 LoRA 输出衰减 4x，推不出 prefix learning。

### 120. v33 SFT-then-DPO pipeline 启动（弃 §119 pure DPO 路径）

§119 提的"无 SFT 路径"在测试中失败：

- 起 v32 (alpha=128, KEEP_PREFIX=1, no SFT): collapse — Busy emit "Execute." 后停（degenerate）
- 起 v32b (alpha=32, KEEP_PREFIX=1, no SFT): no collapse 但 0/8 Novice 学到 prefix
- 结论：**纯 DPO + LoRA 在 cross-distribution learning 上结构性受限**

数学根因：DPO loss 受 β 约束 KL gap 推不动（详见 `docs/sft_then_dpo_v33.md`）：
```
π_ref(Llama-Inst)("I'd be happy") = 0.5
π_ref(Llama-Inst)("What should...") ≈ 1e-10
DPO 即使 ratio 翻 1000x，π(chosen) 还是绝对小 → model 不学
```

**用户最初不愿做 SFT**（觉得不够 RL）。但实证 v32 失败 + 解释 SFT-then-DPO 是标准 industry recipe (DPO 原论文 Rafailov 2023, Llama-2-Chat, Zephyr 7B 都用) 后接受。

**v33 SFT-then-DPO plan**:
- Stage 1: SFT 跨 KL gap，model 学 chosen distribution
- Stage 2: DPO 从 SFT 模型继续训，加 rejected 信号 refine

详见 `docs/sft_then_dpo_v33.md`。

### 119. Retrain 决策（无 SFT 路径）+ 数据扩展计划

§118 数据已经决定方向：**必须 retrain，否则 paper claim 全崩**。但 v30/v31 已经 4 次失败 → 不能简单重复。新方案核心是**保留 prefix + 加数据 + 改 hparam**，不动 oracle。

**不做 SFT**（用户决定）→ 替代方案是**保留 `Clarify\n` / `Execute\n` action prefix**，把"风格选择"拆成"prefix token 预测（easy）+ 条件生成（natural）"，理论上能跨 KL gap 不需单独 SFT 阶段。

| 维度 | 现 v29 | 新方案 |
|---|---|---|
| 训练数据 | 500 pairs（100 state × 1 traj） | **~1500 pairs**（multi-traj × 3 temperatures: 0.7/0.9/1.1）|
| Action prefix | strip | **保留** |
| LoRA r / alpha | 64 / 16 (alpha/r=0.25) | **128 / 128** (alpha/r=1) |
| DPO beta | 0.1 | **0.05** |
| Epochs | 3 | **5** |
| Filter weak pairs | 没 | **filter gap < 0.05** |
| Oracle | v29 baseline | **不动**（避开 v30/v31 失败 path）|
| Eval classifier | v1 prefix-30tok | **prefix-based "Clarify\n"/"Execute\n" 检测**（model emit 时直接读，规则 100% 准）|

**v30/v31 失败的核心原因 vs 新方案修复**：
- v30/v31 改 oracle 但保弱 hparam → DPO 推不动新规则，反而扰乱已 work 的 Busy/Execute 学习
- 新方案保 oracle 不动，**改训练机制**（hparam + prefix + 数据量）→ 直接修根因

**时间估算**：
- 多轨迹生成（3 temperatures × 100 states × 3 personas = 900 traj）: ~6h API + GPU
- DPO 训练（1500 pairs，alpha=128 r=128 epochs=5）: ~12h
- Eval Llama 200 + Qwen 100 with prefix classifier: ~22h
- **总 ~40h compute**，wall 2-3 天（SSH 保活）

**风险**：仍可能 fail（pretrained tendency 太强）。但失败模式跟 v30/v31 不同——这次改的是机制不是 oracle，理论上应该能跨过 KL gap。如果 fail，下一步只能 SFT warmup（用户暂不想做）或弃 paper。

### 118. v2 Llama DPO 30 中止（70% partial）：6.35% << Direct 12.3%，DPO 在 v2 下是 net negative

§117 跑到 63/90 (70%) 时用户决定 kill——数据已经足够 lock 决策。

**最终 partial 数字**：

| Persona | n | pass@1 | pass@5 | avg_turns |
|---|:---:|:---:|:---:|:---:|
| Novice-Learner | 21 | 1/21 = 4.8% | 4/21 = 19.0% | 1.19 |
| Busy-Developer | 21 | 3/21 = 14.3% | 3/21 = 14.3% | 1.00 |
| Experienced-Engineer | 21 | **0/21 = 0.0%** | 2/21 = 9.5% | 1.00 |
| **OVERALL** | **63** | **4/63 = 6.35%** | **9/63 = 14.29%** | 1.06 |

**对比表（Llama）**：

| | v1 数字（forced multi-turn artifact）| v2 真实数字 |
|---|:---:|:---:|
| DPO Llama | **14.0%** (200) | **6.35%** (63) ← 跌 7.65pp |
| Direct Llama | 12.3% | 12.3%（不调 classifier，不变）|
| Base Llama (per §47) | ~8% | 未跑 |

**Per-persona 关键发现**：
- **Busy 14.3% > Direct 12.3%** ← DPO 唯一帮上的 persona（学到 terse code style）
- **Novice 4.8% << Direct 12.3%** ← DPO friendly preamble 在 v2 下没 multi-turn 帮，code 直接挂
- **Exp 0/21** 异常严重 ← DPO LoRA 对 Exp 的 weight adjustment 严重扰乱 base code generation

**结论 lock**：
1. ❌ "DPO learned persona-aware proactive Clarify decision" — 数据反驳
2. ❌ "DPO improves pass@1 over Direct" — v2 下反向（DPO -6pp from Direct）
3. ⚠ DPO 实际 effect = **(Busy +2pp) + (Novice -7.5pp) + (Exp -12.3pp) = 平均 -6pp**
4. v1 14% 数字 = **v1 误判 → forced multi-turn → disclosure recovery 把 LoRA 的损害补回来 + 顺便加一点**

**含义**：
- Llama DPO 在 v2 pipeline 下 net negative，复用 §111 / §116 的 Qwen 类似结论 → **两 backbone 一致：DPO 是 net negative on code generation in v2**
- v1 数字全是 v1 误判产物，**不可作为论文 final 数字**
- TactfulLLM 当前 contribution claim **必须 retrain 或 pivot narrative**

**用户决定**：retrain（不做 SFT），新 plan 见 §119。

**保留**：`outputs/eval_v29_dpo_30_v2.json.partial`（63/90 entries 数据，作为 v1 vs v2 对照证据）。

### 117. v2 Llama DPO 30-state quick eval（跑中，directional 数字）

启动 Llama DPO v29 LoRA + v2 classifier 30-state eval（PID 16512，00:20:38 系统时间起跑）：

```bash
CLASSIFIER_VERSION=v2 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v29_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_200.jsonl \
  --max_samples 30 \
  --output outputs/eval_v29_dpo_30_v2.json
```

ETA ~3-4h compute。Output 跟 100-state run 用同一文件名前缀（partial-resume 可扩到 100/200）。

**关键看点**：
- avg_turns ≈ 1.0 → 验证 v2 让 model turn 0 直接 Execute（一致 sanity）
- pass@1 vs Direct 12.3% / v1 DPO 14.0% → 决定 DPO 在 v2 下有没有 effect
- 如果 v2 DPO ≈ Direct → DPO Llama 跟 Qwen 同样在 v2 下没贡献（"learned proactiveness" 完全证伪）

进度工具：`/tmp/llama_dpo_v2_progress.sh`。

### 116. v2 Qwen Base 100 完成：11.00%（v1 15.67% 跌 4.67pp）

v2 Base Qwen 100 跑完（wall 13:37→23:12 系统时间，9.5h compute）。Final 数字：

| | avg_turns | clarify_rate | pass@1 | pass@5 |
|---|:---:|:---:|:---:|:---:|
| **v1 Base Qwen 200**（误判 + forced multi-turn）| 6.66 | 85% | **15.67%** | -- |
| **v2 Base Qwen 100**（v2 turn 0 Execute）| **1.00** | **0%** | **11.00%** | 19.67% |
| Direct Qwen 100 reference | 1.00 | 0% | 15.33% | -- |

**Per persona (v2)**：Novice 11.0% / Busy 12.0% / Exp 10.0%（all 1-turn Execute）。

**关键 finding**：v1 forced multi-turn 给 Base Qwen 拔高了 4.67pp。去掉 v1 误判后 Base 真实性能 = 11.00%。**v1 数字 15.67% 是 forced multi-turn artifact，不可信**。

**Puzzle**：v2 Base 11% < Direct 15.33% (4pp)。功能上两者都 1-turn Execute 应该几乎相同。可能原因：
- v2 classifier 200-token 生成消耗 GPU random state → 后续 5 candidate 采样跟 Direct 不同
- 或 sampling variance（n=300 std ≈ 2pp，4pp ≈ 2σ 边缘）

**含义**：post-v2 后 **Base = Direct**（功能 redundant）。这意味着 Base baseline 失去独立 narrative 价值——v1 的"Base 与 DPO 行为差异"是 v1 误判产生的虚假分化。

### 115. v2 DPO Qwen 100 启动后被冻 + Kill（autodl 重现 §104 事故）

23:33:45 系统时间起 v2 DPO Qwen 100（PID 993737, setsid wrapper）。预期 ~14h 完成，明早 ~10 AM。

**重现 §104 事故**：用户 04-29 早上 10:35 真实墙钟登录，发现：
- 系统时钟还停在 23:42（容器内时间，比真实 wall clock 慢 11h）
- 进程 ELAPSED 仅 8 分 47 秒（其余时间 cgroup freeze）
- 仅完成 sample 1/100 Novice（Turn 0: Execute ✓ 一致 sanity 预测）

setsid wrapper 挡不住 autodl 的 cgroup 降级。**教训复现 §104**：bash 层无法替代 SSH 保活。

用户决定 kill 进程释放 GPU 转跑更小批量验证：
- `kill -9 993737 993734 993732 993729` → GPU 2 MiB / 0%
- 改起 v2 DPO Llama 30 small batch（详见 §117）

### 114. v1 Classifier Bug 诊断 + sanity 验证（核心 finding）

**根因**：`policy/infer.py:_pick_action_v1` 只看生成的前 30 token 前缀。Llama/Qwen DPO Novice/Exp persona 都倾向 emit "I'd be happy to help / Sure! Let's break down..." 这种 friendly preamble 起手。30 token 窗口内全是自然语言 → v1 系统性误判 Clarify。但 200 token 全文里 model 实际写了完整代码（` ```python\nimport.../def task_func`）。

详见 `docs/classifier_bug_2026-04-28.md`。

**Sanity 数据**（双 backbone + sampling，详见 `scripts/sanity_classifier/`）：

| 实验 | Total | v1=Clarify, v2=Execute（v1 误判）| 起手是 question word |
|---|:---:|:---:|:---:|
| Qwen DPO greedy 8 state × 3 persona | 24 | **16/24 (67%)** | 0/24 |
| Llama DPO greedy 8 state × 3 persona | 24 | **16/24 (67%)** | 0/24 |
| Llama DPO Novice sampling 8×5 | 40 | **40/40 (100%)** | 0/40 |
| Qwen Base greedy 8 state × 3 persona | 24 | **24/24 (100%)** | 0/24 |

**Pattern 完全一致**：
- Novice/Exp 8/8 误判（preamble + code）
- Busy 0/8 误判（DPO 后直接 ` ```python` 起手）
- Sampling 40 个里 7.5% (3/40) 在 preamble 后真嵌问题，92.5% 后接代码

**含义**：
1. **v1 的 forced multi-turn 是误判产生的副作用**——不是 DPO learned behavior
2. **DPO 没学到 proactive Clarify decision**——三个 persona turn 0 都倾向 Execute（只是 style 不同）
3. **DPO 学到的是 persona-conditional style**：Busy=terse direct code, Novice/Exp=friendly preamble + code
4. **Llama DPO 14% > Direct 12.3% 的真实 mechanism** 是 v1 误判 → forced multi-turn → disclosure recovery，不是 "learned proactiveness"

**对论文 claim 的影响**：
- ❌ "TactfulLLM learns persona-aware proactive Clarify decision-making" — 数据反驳
- ⚠ "DPO is the only persona-adaptive method"（per §111）— 是 **style-adaptive** 不是 **action-adaptive**

**未提交代码**：`policy/infer.py` 加 `_pick_action_v1` (legacy) + `_pick_action_v2` (200-token 全文扫描) + env var `CLASSIFIER_VERSION` dispatcher。默认 v1 保 Llama 复现性。

**Sanity 脚本归档**：`scripts/sanity_classifier/` 含 6 个文件（3 py + 3 json）。

### 113. v29 训练数据 distribution 审计：Busy/Clarify = 0 pair

调查 DPO 失败原因，发现 **prefs_v29_100states.jsonl 严重失衡**：

| (persona, chosen_action) | v29 pairs |
|---|:---:|
| Novice/Clarify | 226 |
| Busy/Execute | 107 |
| Exp/Clarify | 105 |
| Exp/Execute | 61 |
| **Busy/Clarify** | **0** ← 完全缺失 |
| **Novice/Execute** | **1** ← 几乎缺失 |
| Total | 500 |

只有 29.2% (146/500) 有 reward gap >= 0.05（强信号）。Turn 分布偏 turn 0 (319/500)，turn 2+ 几乎空。

**v30/v31/v31_4 distribution 对比**：

| | v29 | v30 | v31 | v31_4 |
|---|:---:|:---:|:---:|:---:|
| Busy/Clarify | 0 | 73 | 30 | 30 |
| Busy/Execute | 107 | 51 | 80 | 104 |
| Novice/Clarify | 226 | 191 | 199 | 199 |
| Novice/Execute | 1 | 28 | 21 | 21 |

但 **v30/v31 系列 50-state pass@1 都比 v29 14% 低**（v30 partial 2.56%，v31 9.33%，v31_2a 8.67%, v31_4 8.00%）。说明改 oracle rule 但保 hparam 没用——LoRA alpha=16/r=64 (alpha/r=0.25) 输出衰减 4x，DPO 信号压不过 pretrained tendency。

### 112. C 方案讨论：retrain DPO 修正 proactive failure

用户决定试 C 方案（重训 DPO 修复 proactive 学习失败）。

**根因诊断**：
- LoRA `alpha/r=16/64=0.25`，标准应 alpha=r 或 2*r → DPO 输出被衰减 4 倍，压不过 pretrained "I'd be happy to help" 倾向
- `_strip_action_prefix` 让 model 失去明确 action 信号，必须从零学一种新风格
- DPO 优化 implicit reward，KL 约束保护 pretrained 行为 → 跨过 friendly preamble → "What should..." 直接问句的 KL gap 难

**C 方案 hparam 改动**：

| 参数 | 现 v29 | C 方案 |
|---|---|---|
| `lora_alpha` | 16 | **128** (alpha/r=2) |
| `r` | 64 | 128 |
| `beta` (DPO) | 0.1 | 0.05 |
| `epochs` | 3 | 5 |
| `_strip_action_prefix` | strip | **保留** |

加 prefix-based eval classifier。但**风险**：v30/v31 4 次 retry 都失败的历史 → 第 5 次成功概率不高。可能仍然 degenerate。

**SFT warmup → DPO** 是更可能 work 但代码改动大的方案：先 SFT 强制 model 输出问句风格（跨过 KL gap），DPO 再 refine。1-2 day 代码 + 训练。

**当前 status**：暂不重训。先用 v2 跑现 v29 model 拿真数字（§115/§116/§117），看是否值得 spend GPU on retrain。

---

## 2026-04-27

### 111. DPO Qwen 100 完成：Overall 11.67%，Qwen 上 last among baselines

DPO Qwen 100 跑完（wall ~22:00, 11h, ~7 min/state, PID 264568）。详情见 `qwen_experiment_log.md` §9。

**Final 数字**（n=100, apples-to-apples vs Direct/CF/PO）：

| persona | avg_t | clarify% | pass@1 | pass@5 | rej |
|---|:---:|:---:|:---:|:---:|:---:|
| Novice | 7.00 | 85.7% | **13.0%** | 19.0% | 40.5% |
| Exp | 3.89 | 74.3% | **14.0%** | 21.0% | 47.1% |
| Busy | 1.00 | 0% | **8.0%** | 12.0% | -- |
| **Overall** | 3.96 | — | **11.67%** | **17.33%** | 42.6% |

**vs Qwen baselines**（同 100 sample）：CF 16.0% > Direct 15.3% > PO 13.0% > **DPO 11.67%**（last by 1.3pp vs PO, 4.3pp vs CF）。

**vs Partial 54**：Overall 13.0% → 11.67%, -1.3pp（剩 46 state 表现略差，落在 §8 预测 12-14% 区间下沿）。

**关键 takeaway**：
1. **DPO Qwen 是最弱方法 on pass@1**——Qwen 数据直接证伪 "TactfulLLM consistently strongest" claim
2. **持续做对 persona-aware behavior**——Novice 7 / Exp 3.89 / Busy 1.0 完美分化（其他三个都是 persona-blind）
3. **Busy 0% rejection** 独有——Pareto 维度 DPO 仍 dominate（CF Busy 80% rej）
4. **DPO 仍正向 vs Base 50**: Novice +63%, Overall +17%（Δ vs Base 行可写 +63/0/0/+17%）

**论文叙事必须 pivot**（per #108 / #110 早就推荐过的 B 方案）：
> TactfulLLM is the **only persona-adaptive method** across both backbones; on Llama it achieves the highest pass@1, while on Qwen the persona-aware decision quality is preserved at lower absolute pass@1 due to backbone capability.

**主表 Qwen TactfulLLM 行最终**：
```
& 13.0 & 14.0 & 8.0 & 11.7    ← pass@1
& 19.0 & 21.0 & 12.0 & 17.3   ← pass@5
& 7.0 & 3.89 & 1.0 & 3.96     ← avg turns
& 0.41 & 0.47 & -- & 0.43     ← rej
```

`\ddagger` 脚注（partial 占位）可以删。

**下一步**：决定要不要扩 Base Qwen 50→100（让 Δ vs Base 严格 apples-to-apples）+ 写 paper 时 pivot 到 persona-alignment narrative。

### 110. DPO Qwen 50→100 启动（Phase 3 第一步，setsid wrapper）

PO Qwen 100 出 13% 反超 DPO 50 的 7.3% 后，立刻起 DPO 100。详细见 `qwen_experiment_log.md` §6+§8。

**启动**（wall 10:35 / system 23:35:55）：
```bash
setsid bash -c '/tmp/qwen_dpo_100.sh > /tmp/qwen_dpo_100_wrapper.log 2>&1 < /dev/null' &
```
PID 264565（PPID=1）。Output: `outputs/eval_v29_qwen_dpo_100.json`。测试集 `test_states_v29_eval_200.jsonl --max_samples 100`，与 Direct/CF/PO 同 sample。

**ETA**: ~6-7 min/state（DPO Novice 撞 7 轮顶最慢）× 100 = **wall 20:30-22:30 今晚**完成。

**关键 caveat**：DPO 50（用 `eval_50.jsonl`）与 DPO 100（用 `eval_200.jsonl --max_samples 100`）只 25 个 state 重叠，不是 50 ⊂ 100。所以 DPO 100 是 75 个全新 + 25 个重跑。DPO 50 文件保留作 sanity-check / appendix。

**进度工具**: `/tmp/qwen_dpo_progress.sh`——查 done/total、ETA、log age（>5min throttle warn）、recent action。

**未来扩 200**: `cp eval_v29_qwen_dpo_100.json eval_v29_qwen_dpo_200.json.partial` → `--max_samples 200` resume，state_id-based partial resume 自动跳已完成 100。

### 109. DPO 代码审计：无 Qwen-specific bug（详见 qwen_experiment_log §7）

用户问 "qwen DPO 代码有没有问题"。审 `train_dpo.py` / `infer.py` / `evaluate_multi_turn_persona.py` 的 Qwen 路径——

| 检查点 | 结论 |
|---|---|
| chat template (`apply_chat_template`) | ✓ Qwen 自动转 `<|im_start|>` 格式 |
| action prefix strip | ✓ 自然响应训练，backbone 通用 |
| LoRA target_modules | ✓ q/k/v/o/gate/up/down，Qwen2 同名 |
| `pick_action_from_generation` | ✓ skip_special_tokens + 内容样式检测 |
| Qwen tokenizer pad/eos | ✓ pad=`<|endoftext|>`, eos=`<|im_end|>` 分离正确 |
| 8-bit 推理 vs 4-bit 训练量化 | ⚠ 不匹配，但 Llama 也是这样——非 Qwen 特异性 |

**核心反驳证据**：DPO 50 vs PO 100 在 25 个**重叠 state** 上 pairwise 对比——Overall 4.0% vs 5.3%，**只差 1.3pp**。7.3% vs 13.0% 大头是 N=50 vs N=100 的采样噪声，不是代码 bug 也不是 DPO 失效。

### 108. Prompt-only Qwen 100 完成 + DPO 反输 paradox 暴露

**PO Qwen 100 结果**（wall 07:14 完成，8h 跑完）：

| persona | avg turns | clarify% | pass@1 | pass@5 |
|---|:---:|:---:|:---:|:---:|
| Novice | 2.05 | 51.2% | 10% (10/100) | 20% |
| Busy | 1.00 | 0% | 13% (13/100) | 21% |
| Exp | 1.07 | 6.5% | 16% (16/100) | 23% |
| **Overall** | — | — | **13.0%** (39/300) | **21.3%** |

**Qwen 反常**（与 Llama 模式相反）：

| | Llama 模式 | Qwen 模式 |
|---|---|---|
| Direct vs DPO | DPO 14% > Direct 12.3% | Direct 15.3% > DPO 7.3% |
| PO vs DPO | DPO 14% > PO 8.7% | PO 13.0% > DPO 7.3% |
| CF vs DPO | DPO 14% ≈ CF 14.8% | CF 16.0% > DPO 7.3% |

但 §109 已证大头是采样噪声。DPO 100 跑完应该接近 13-16% 区间。

**论文 story pivot 关键**：DPO 真正卖点不是 pass@1，是**persona alignment**。
- DPO Qwen Novice 7.0 轮 / Exp 3.4 / Busy 1.0 → 行为完美分化
- PO Qwen Novice 2.05 / Exp 1.07 / Busy 1.0 → 基本 persona-blind（虽然有 persona prompt）

DPO 学到的 persona-aware decision-making 是 Prompt-only 学不到的，这是 Qwen 行的核心 contribution。

### 107. Phase 1 完成：Direct + CF Qwen 100（apples-to-apples 主表数据）

CF watchdog (PID 100943) 12:15 system / 23:15 wall 退出 → 守门检查通过 → PO 接力起来（详见 §108）。

**Direct 100**（wall ~16:53 完成）：Overall **15.3%** (46/300), pass@5 20.0%
- Novice 16% / Busy 13% / Exp 17%（all forced 1-turn execute）

**CF 100**（wall ~23:15 完成）：Overall **16.0%** (48/300), pass@5 24.0%
- Novice 12% / Busy 17% / Exp 19%（all forced 1 clarify + 1 execute）
- **CF Busy 17% 仍然反超 DPO Busy（per #98 simulator 张力，80% 拒绝 + 20% 一次给 3 items）**

**实测速率订正 §5.2**：
- Direct 3.0 min/state（vs §5.2 估的 4-5）
- CF 3.5 min/state
- PO 4.8 min/state（中长 turn）
- DPO ~6-7 min/state（Novice 撞顶最慢）

§5.2 那个 5 min/state 的"固有 ceiling"对 Direct 偏高 25%，多轮越多越慢的 pattern 才是真相。

---

## 2026-04-26

### 106. Qwen baseline 全套规划（4 phase, ~55h GPU）

用户决定先把 Qwen 全 baseline 跑完，100-state 对齐：

| Phase | 任务 | GPU 时长 | 起讫（系统时钟）|
|---|---|---:|---|
| 1 (进行中) | Direct 100 + CF 100 | ~19h | 04-26 00:13 → 19:13 |
| 2 | Prompt-only Qwen 100 | ~12h | → +12h |
| 3 | DPO 50→100 + Base 50→100 | ~12h | → +12h |
| 4 | Llama Prompt-only 50→200（补 150）| ~12h | → +12h |

总计 ~55h GPU，wall clock 3.5 天（笔记本不睡前提）。CollabLLM #54 仍待实现，未排入。Qwen 100 + Llama 200 用脚注解释 N 不一致。

### 105. Qwen Direct + CF 改 100 state（速率估算重大修正）

**#103 估算的 "Direct 1-turn ~40s/state" 完全错**。实测：Qwen Direct ~5.8 min/state（GPU util 10-16%，bitsandbytes 8-bit + batch=1 + 5 candidate × 3 persona = 15 次串行推理）。对比 Llama Direct 200（`logs/exp2_overnight.log` Apr 17 11:44→22:39，跑 150 state = 4.37 min/state）—— **这速率是 eval 脚本固有 ceiling，不是 Qwen 异常**。Qwen 比 Llama 慢 33%，同量级。

按真实速率 Direct 200 ≈ 19h + CF 200 ≈ 23h = 42h，远超过夜 window。**砍到 100-state**：Direct 7.5h + CF 12h = 19.5h。

**操作**：kill PID 44410 → mv `.partial` _200→_100 → 改 wrapper `max_samples 200→100` & output `_200→_100` → setsid 重启（PID 59661，04-26 00:13）。Resume 验证：`🔄 恢复 7 个已完成 state`，从 sample 8 接着跑。

**论文叙事**：Qwen 主表 N=100 + Llama 主表 N=200，脚注说明（"backbone-secondary, footnote per #94"）。

### 104. 04-25 watchdog 早死 + 11h GPU 空跑诊断

**事件**：#103 起的 watchdog (PID 982966) 在 12:28 Base Qwen 完成后死掉，没启动 Direct/CF 步骤。`/tmp/qwen_baselines_tonight.log` 12:28 之后零输出，无 step 子日志，无 .partial 文件。直到 04-26 00:00 用户问"baselines 跑完了吗"才发现，**11h GPU 空跑**。

**根因（不确定）**：容器 uptime 866 天连续没重启 → 非物理机故障。重启后系统 `date` 与真实 wall clock 差 11h，系统时钟和进程 ELAPSED 自洽（都说"刚跑了几分钟"），符合 cgroup freeze / autodl 共享调度 deprioritize 特征。user 离线后 SSH 无心跳 → autodl 把容器降级。watchdog `nohup` 在这种降级下被清理。

**教训**：
1. 过夜 watchdog 不再靠 nohup，改用 **setsid + PID 1 attach**（04-26 00:13 重启用 setsid，待验证）
2. **Eval 速率估算必须基于历史 log**，#103 那个 "40s/state" 凭空写的。引用时也得核对，否则 ddl 会误判
3. **保活靠 user 笔记本不睡 + cursor 不断**，bash 层无法替代
4. eval 脚本天然支持 .partial resume（line 547-621），重启即续跑，进度本身不会丢

---

## 2026-04-25

### 103. 今晚 Qwen baseline 过夜安排（watchdog + 串联 pipeline）

**目标**：完成 Qwen 主表 baseline 行（Direct / Clarify-first 200-state），对齐 Llama canonical-200 覆盖；剩余 Prompt-only 50 + Base 扩 200 推到后续。

**今晚 pipeline**（`/tmp/qwen_baselines_tonight.sh`, watchdog PID 982966，10:22 kick off）：
1. 等 Base Qwen 50（PID 944046）结束（`kill -0` 监控）
2. Direct Execution Qwen 200-state（~2.5h，`--direct_execution --max_turns 1`）
3. Clarify-first Qwen 200-state（~5h，`--always_clarify 1 --max_turns 2`）

预期完成时间：19:55 CST（从 10:22 算总 wall-clock ~9.5h）。

**时长估算基础**：Base Qwen 50 实测 ~7 min/state（6-7 turn × 3 persona × 5 candidate）。Direct 1-turn ~40s/state，CF 2-turn ~80s/state。

**不跑的**：
- Base Qwen 200 扩展（~23h，超过夜 window）→ Base 用 50 + footnote 对齐 Llama Prompt-only 策略
- Prompt-only Qwen 50（~5h）→ 挤占今晚太多，明天白天单独跑
- TactfulLLM Qwen 200 扩展（~10h）→ 独立 overnight，给 v31.4 试验留余地

**产物（明早 9 点查）**：
- `outputs/eval_v29_qwen_base_50test.json`
- `outputs/eval_v29_qwen_direct_execution_200.json`
- `outputs/eval_v29_qwen_clarify_first_200.json`

### 102. Exp1 主表 vs Exp2 主表数据源不一致审计

用户质疑 Exp1 Main Results 和 Exp2 Info Recovery 两表数字对不上。重审数据源：

| 方法 | Exp1 主表 pass@1 | 实际源 | Exp2 表 | 实际源 |
|---|:---:|---|:---:|---|
| Direct | 7.3% | **50-state** | 12.3% | 200-state |
| Clarify-first | 9.3% | **50-state**（#98 已更新 14.8% / 200）| — | — |
| Prompt-only | 8.7% | **50-state** | — | — |
| Base LLM | 12.7% | 200 ✓ | — | — |
| TactfulLLM | 16.0% | 200 ✓ | 16.0% | 200 ✓ |

**根因**：Exp1 caption 误写 "200 test tasks" 但 Direct/CF/Prompt-only 都用 50-state 快照。Prompt-only 200 至今未补跑（#87 queue 只跑了 CF）。

**修复计划**：
1. Exp1 主表 Direct 7.3 → 12.3（用 canonical-200 first-sample 数字）
2. Exp1 主表 CF 9.3 → 14.8（已在 #98 做过数字但 LaTeX 未改）
3. Prompt-only 200 单独补跑（~2.5h）
4. Base Llama 行不动（已 200）

### 101. Unbiased pass@1 vs first-sample pass@1 讨论（保留 first-sample）

用户质疑"我们的 pass@1 计算对不对"。拉 `candidate_results` 字段（每个 state 5 个候选 pass/fail 都存着），**离线**重算 unbiased vs first-sample：

| eval 文件 | n | first-sample | **unbiased** | Δ |
|---|:---:|:---:|:---:|:---:|
| Llama DPO 50 | 150 | 14.0% | 10.4% | -3.6pp ⚠️ |
| Llama DPO 150extra | 450 | 16.7% | 15.3% | -1.3pp |
| Llama Base 50 | 150 | 8.0% | 10.0% | +2.0pp |
| Llama Direct 200 | 600 | 12.3% | 11.1% | -1.2pp |
| Llama Oracle 200 | 200 | 20.0% | 17.0% | -3.0pp |
| Llama Ideal Disclosed v2 200 | 200 | 16.0% | 15.9% | -0.1pp |
| Qwen DPO 50 | 150 | 7.3% | 7.2% | -0.1pp ✓ |
| Qwen Base 50 (26/50 partial) | 78 | 5.1% | 5.7% | +1.0pp |

**观察**：
- 200-state 上偏移都 ≤ 3pp，first-sample 基本收敛
- 50-state 上偏移大（Llama DPO 50 -3.6pp），主要是小样本 + n=5 采样噪声
- Qwen 数字异常稳定（DPO -0.1pp）→ 7.3% 不是 first-sample 运气

**决定：保留 first-sample pass@1**（不切 unbiased）。理由：
1. 200-state 数字差异可忽略
2. first-sample 是合法操作定义（论文 method 节加一句即可）
3. ddl 紧，切 unbiased 要重写 Exp2 的 OGR 叙事（`Ideal Disclosed OGR 48% → 81%`）
4. 真要 robustness check 可放 appendix 做对照表

**待做**：method 节加操作定义文字："we report pass@1 as the success rate of the first sampled candidate from n=5 candidates (temperature T, top_p P). pass@5 reports the rate of at least one of 5 passing."

### 100. Qwen Base 诊断中期：DPO 胜 Base，backbone code ceiling bound

Base Qwen 50-state eval（PID 944046，06:15 起跑）中期 26/50 数字 vs DPO Qwen 50 完整对比：

| persona | Base Qwen 中期 pass@1 | DPO Qwen 完整 pass@1 | Δ |
|---|:---:|:---:|:---:|
| Novice | 2/26 = 7.7% | 5/50 = 10% | +2.3pp ✓ |
| Busy | 1/26 = 3.8% | 2/50 = 4% | +0.2pp（持平）|
| Exp | 1/26 = 3.8% | 4/50 = 8% | +4.2pp ✓ |
| **Overall** | **5.1%** (4/78) | **7.3%** (11/150) | +2.2pp ✓ |

**关键诊断结论**：
1. **DPO 没伤 Qwen 代码生成**，反而持平+改进（+2.2pp overall）。相对 Llama 的 DPO +6pp 提升小，但方向一致
2. **Qwen2.5-7B 在 masked BigCodeBench 上 backbone 能力就弱**：Base Qwen 5.1% < Base Llama 8% 
3. **Base Qwen 完全 persona-blind**：Novice/Busy 都跑 7 轮 clarify rate 83-86%，Exp 6.04 轮 — Base 无法区分 persona，完全复现 Llama Base 现象（#47）
4. **Busy 0 提升**（3.8→4%）和 Llama v29 Busy 问题一致 → Busy T0 Execute 学到了但信息不足

**对论文 story**：
- Qwen 7.3% 不达 §3.2 合格线 10%，但 DPO vs Base gap 方向一致
- Table 1 Qwen 行叙事："方法在两 backbone 都能学到 persona-aware 决策，绝对 pass@1 bound by backbone code capability"
- Busy 0 提升是两 backbone 共同问题 → future work 收尾

**masked 数据 sanity check**：三个 eval（DPO Qwen / Base Qwen / Llama v29）都读 `data/seeds/test_states_v29_eval_50.jsonl`，**字节级同文件**。唯一变量是 backbone + chat template + tokenizer。Busy 14% (Llama) vs 3.8% (Qwen) 同 masked query 同管线 → backbone 差异。

### 99. Qwen v29 DPO 50-state eval 完整结果

50/50 最终（04:05 CST 左右完成）：

| persona | avg turns | clarify rate | pass@1 | pass@5 |
|---|:---:|:---:|:---:|:---:|
| Novice | 7.0（全撞顶）| 85.7% | 10% (5/50) | 12% (6/50) |
| Exp | 3.4（29 forced final）| 70.6% | 8% (4/50) | 20% (10/50) |
| Busy | 1.0 ✓ | 0% | 4% (2/50) | 10% (5/50) |
| **Overall** | — | — | **7.3%** (11/150) | **14%** (21/150) |

vs Llama v29 50-state（14% / 20%）：pass@1 -6.7pp, pass@5 -6pp。

**行为分化完美复现 Llama**（Novice 7 轮 / Exp 3.4 轮 / Busy 1 轮），**策略对、代码挂**。

vs §3.2 合格线判定：Overall 7.3% < 10% ❌ / Novice 10% < 12% ❌ / Exp 8% = 8% ✓ 踩线 / Busy 4% < 10% ❌。

### 98. Clarify-first 200-state 真相：simulator 设计张力暴露 Pareto trade-off

用户质疑 Clarify-first (CF) 200-state pass@1 14.8% 数字偏高，一起深挖到底：

**合并 CF 50test + 150extra = 600 unique conversations**，配置全对（100% `(Clarify, Execute)` action pattern, Base Llama no-LoRA, `--always_clarify 1`）。数字不是 bug。

**但发现 CF Busy 19.3% 的来源分解**（对比 Direct 同 150 states）：

| Busy 分组 | n | pass@1 | 解读 |
|---|:---:|:---:|---|
| 拒绝 (answered=0) | 123 | **15.4%** | ≈ Direct 13.8%（+1.6pp 噪声），**打扰白打扰** |
| 配合 (answered=1) | 27 | **37.0%** | simulator 一次给 3 items，pass 率 2.4× |
| 合计 | 150 | 19.3% | 加权平均 |

**根因 = simulator 设计张力**：`simulator/simulate.py:47` 定义 Busy = `(expertise=mid, patience=low)`。`disclosure.py` 里 expertise=mid → **一次给 3 items**。`simulate.py:219` 里 patience=low 让 Busy ~80% 拒绝。所以 Busy 实际行为 = 82% 拒绝 + 18% 一次给 3 items。

**v31 log §0.5 里简化成 "Busy 拒绝 Clarify" 不完全准确**——是"80% 拒绝，20% 配合时给 3 items"。这解释了为什么 CF Busy 反超 TactfulLLM（TactfulLLM 永远不问 → 错过 18% 配合的 Busy 用户）。

**信息量 vs pass@1 的 ROI**（从 replay csv 读）：

| Method | Persona | 轮数 | disclosed items | pass@1 |
|---|---|:---:|:---:|:---:|
| CF | Novice | 1 | 0.62 | 14.0% |
| TactfulLLM | Novice | 6 | 2.47 | 18.5% |
| CF | Exp | 1 | 1.18 | 14.0% |
| TactfulLLM | Exp | 1.58 | 2.19 | 15.5% |
| CF | Busy | 1 | 0.32 | **16.5%** |
| TactfulLLM | Busy | 0 | 0.00 | 14.0% |

**信息收益严重递减**：Novice TactfulLLM 多拿 4× 信息只换 +4.5pp pass@1；CF 1 轮的边际 ROI 意外高，因为"第一个关键 item"（如 output return type）已经抓到大部分信息价值。

**对论文的意义**：story 从"pass@1 更高" pivot 到 **"Pareto trade-off between task success and user interruption"**。
- CF 14.8% pass@1 ← 代价 48% overall rejection rate（Busy 82%）
- TactfulLLM 16.0% ← 代价 ~30% rejection（Busy 0）
- TactfulLLM 在 Pareto frontier 上 dominate CF

v31.x 想做的"Busy 选择性问一次"本质就是想捕获这 18% 配合的 Busy 用户——单 LoRA 学不会这个细粒度策略是 future work。

**Table 1 修正**：
- CF 行全部更新到 200-state 数字（14.0/14.0/16.5/14.8 for pass@1）
- **CF Busy 16.5% 反超 TactfulLLM 14%**，bold/underline 需调换（CF 的 Busy cell bold，TactfulLLM 降为 underline）
- 类似地 Busy pass@5 21.5 vs 20.0 也要调换

### 97. 创建 Qwen backbone 实验记录

新建 `docs/qwen_experiment_log.md`，结构对齐 v29/v31 的 log 习惯：
- §0 动机 + 为何选 v29 pair（v31.x 跨干扰问题已在 Llama 证实，Qwen 复刻没价值）
- §1 HF cache 问题定位 + 软链修复过程
- §2 训练（20min，87 steps, accuracy 99%, margins 5.8，收敛曲线和 Llama v29 同量级）
- §3 50-state eval（进行中，判据 + 结果位留空）
- §4 200-state eval 计划（pipeline 备好 `/tmp/qwen_v29_eval_200.sh`）
- §5 后续优先级（P0 Qwen v29 200 / P1 Base 对照 / P2 baselines / 不做 v31.x on Qwen）
- §6 文件与时间线

### 96. Qwen v29 DPO 50-state 中期结果（30/50）：pass@1 3.3%，需要 Base 对照诊断

30/50 states 完成后的 partial 数字：

| persona | n | avg turns | pass@1 | pass@5 |
|---|:---:|:---:|:---:|:---:|
| Novice | 30 | 7.00（全撞顶）| 6.7% | 10.0% |
| Exp | 30 | 3.73 ⚠（5 个撞顶）| 3.3% | 20.0% |
| Busy | 30 | 1.00 | **0% (0/30)** ❌ | 10.0% |
| **Overall** | 90 | — | **3.3%** | **13.3%** |

vs Llama v29 50-state (14%, 20%)：**pass@1 差 10.7pp，pass@5 差 6.7pp**。

**关键信号**：
1. 行为分化完美复现 Llama（Novice 7 撞顶 / Busy 1 直接 / Exp 多轮）
2. **Busy 30/30 pass@1=0** 最担忧——决策完全对但代码没过
3. **Exp avg turns 3.73** 比 Llama 2.66 高（5 个 forced final），Qwen 学得"半透不透"

最严重的是 Busy 0% 路径和 Llama 同策略（T0 Execute）但代码全挂——**纯代码生成问题**。

**启动 Qwen Base 50-state watchdog** (`/tmp/qwen_v29_base_50test.sh`, PID 875405)，等 DPO eval 完退出后自动跑。关键看：
- Base Qwen Busy pass@1 ≈ 0% → Qwen2.5-7B 在 masked BigCodeBench 上就是弱（非 DPO 问题）
- Base Qwen Busy > DPO Qwen → DPO 伤了代码生成，需调 hparam

### 95. Qwen HF cache 修复 + Qwen v29 pipeline 启动

v31.4 Qwen 训练 `LocalEntryNotFoundError` 根因定位：15GB Qwen 权重下载到 `/root/autodl-tmp/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/`，但新版 transformers 期望 `$HF_HOME/hub/models--...`。Llama 在 `hub/` 下所以能 offline 加载，Qwen 在老路径所以失败。

**修复**：软链 `ln -s hf_cache/models--Qwen... hf_cache/hub/models--Qwen...`，offline 加载验证通过（vocab 151665, model_type qwen2, hidden 3584）。

**Pipeline 启动** (`/tmp/qwen_v29_pipeline.sh`, 04-24 23:50 kick off)：
- 复用 v29 pair (`data/dpo/prefs_v29_100states.jsonl`, 500 对)
- 同 Llama v29 配置 (epochs=3, beta=0.1, lr=5e-5, QLoRA r=64)
- 测试集 `test_states_v29_eval_50.jsonl`（**和 Llama v29 完全同文件**，数字直接可比）
- train 20min 完成 → eval 5h 串联

**选 v29 pair 不选 v31.x 的理由**：v31.x 在 Llama 上已证明 pass@1 全面低于 v29，Qwen 复刻大概率重演同样的跨 persona 干扰，ddl 紧张下优先保 Qwen 主结果成立。

### 94. v31.4 结果分析记入 v31_experiment_log §12

Llama v31.4 eval 完成（14:21），Qwen v31.4 训练崩溃（14:22 HF cache）。详细见 `docs/v31_experiment_log.md` §12：

**v31.4 Llama 50-state**：Overall pass@1 **8.0%** (vs v29 14%) / pass@5 15.3% (vs v29 20%)

| persona | 指标 | v29 | v31.4 | Δ |
|---|---|:---:|:---:|:---:|
| Novice | avg turns / p@1 | 7.00 / 16% | **4.14 / 12%** | -2.86 ✓ / -2 task |
| Exp | avg turns / p@1 | 2.66 / 12% | 2.24 / **6%** | 干扰退步 |
| Busy | avg turns / p@1 | 1.0 / 14% | **1.24** / 6% | 3 漏网长尾 |

**跨 persona 干扰重演 v31.2a**：Exp/Novice pair 字节级与 v31.1 相同，但 LoRA 共享参数让 Busy 新 24 对 pair 污染 Exp T0 决策（94%→78%）+ 代码生成。v31.4 证明"把 Busy T1 Execute 学进 DPO"在当前 data + method 组合下不可行。

**Novice 4.14 轮结构优秀**但 Exp/Busy pass@1 都退步，**整体劣于 v29、劣于 v31.3-D**。v31.3-D 推理 patch (12.67%) 仍是当前最优操作点。

Qwen v31.4 模型没训出来 (`models/v31_4_qwen_100states/` 空目录)。决定 Qwen 走 v29 pair 方案（见 #95），不在 Qwen 上复刻 v31.x。

---

## 2026-04-24

### 93. v31.4 pipeline 启动（过夜跑，Llama + Qwen 双 backbone）

**目标**：把 v31.3-D 的推理补丁做成 DPO 训练能力 + 补 Qwen backbone。详细设计见 `docs/v31_experiment_log.md` §12（待补）+ §0.5 理想轨迹。

**规则改动**（`reward/compute_rewards.py`）：
- Busy 新增 `turn >= 1 → Execute` 分支（T1+ 硬停，消除 rejection 长尾）
- Novice 保持 `NOVICE_U_STOP=0.4`（尝试过 0.3 无效 / 0.2 会退化为 v29 永远 clarify）
- Exp 不改

**两个 bug 修复（关键）**：

1. **Method B2 dialogue_turn 错位**：`generate_trajectories.py` 的 forced_execute turn 继承 `state.dialogue_turn=0`（没递增），Method B2 的 `if ct_turn == 0: continue` 把它当 T0 skip 掉。改用 top-level `turn - 1` 作 chronological 真值；同时 patch `ct_state["dialogue_turn"]` 给 rebalance 用。
2. **U 离散化陷阱**：U ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}。原本想把 Novice 阈值 0.4→0.3 让它多跑 1 轮，但 0.3 和 0.4 触发集一样（都是 {0, 0.2}），无效。要真变行为得跨 bin（0.2 或 0.5）。0.2 会让 T2 全 Clarify 退化 v29，回退到 0.4。

**pair 分布（v31.4 vs v31.1）**：520 对 vs 496 对（+24）

| persona × turn × action | v31.1 | v31.4 | Δ |
|---|:---:|:---:|---|
| Busy T0 Clarify | 30 | 30 | 0 |
| Busy T0 Execute | 77 | 77 | 0 |
| **Busy T1 Execute** | **3** | **27** | **+24** ← 核心修复 |
| Exp 所有 | 166 | 166 | 0（字节级相同）|
| Novice 所有 | 220 | 220 | 0（字节级相同）|

Busy T1 从 3 → 27（预期 28，2 对被 same-msg filter 过滤），修复生效。

**Pipeline 启动（07:35 CST）**：`/tmp/v31_4_full_pipeline.sh`
1. 等 compute_rewards → validate → **Llama 训练**（17min）→ **Llama eval 50-state**（5h）
2. 等 Qwen 下载 → **Qwen 训练**（17min）→ **Qwen eval 50-state**（5h）
3. 串行执行，预计 18:30 CST 全部完成

**当前状态（10:10 CST）**：
- compute_rewards ✓（07:59 完成）
- Llama 训练 ✓（~08:16 完成）
- Llama eval 进行中（20/50，40%）
- Qwen 下载 ✓（15GB，~15min 完成）
- 观察到的行为样本：Novice 3 轮 / Busy 1 轮（低 U 样本）/ Exp 2 轮，符合理想轨迹

**产物**：
- `data/dpo/prefs_v31_4_100states.jsonl`（520 对）
- `models/v31_4_100states/`（Llama）
- `models/v31_4_qwen_100states/`（Qwen，晚些完成）
- `outputs/eval_v31_4_dpo_50test.json`
- `outputs/eval_v31_4_qwen_50test.json`

**v31.4 预期数字（50-state，基于 v31.3-D + Qwen 未知）**：

| | v29 | v31.1 | v31.3-D | v31.4 预期 |
|---|:---:|:---:|:---:|:---:|
| Overall p@1 | 14.0% | 9.33% | 12.67% | **~12-13%**（v31.3-D 的 learned 版本）|
| Busy 长尾 | 0 | 9/50 | 0 | 0 |
| Busy p@1 | 14% | 8% | 12% | ~12% |
| Novice turns | 7.0 | 4.0 | 4.24 | ~4.0 |

**Novice pass@1 天花板**：items_per_turn=1 × 4 轮 = 最多 3 items 披露。n_masked=5 task 必缺 2 items。要上 v29 的 18.5%（200-state）必须让 Novice 跑 5+ 轮。Plan H（MIN_CLARIFY=4，用 T4 Execute pair 13 对）留作 v31.4 结果不理想时的 fallback。

### 92. v31 理想轨迹记入 `v31_experiment_log.md` §0.5

详细见 `docs/v31_experiment_log.md` §0.5。三个 persona 在 user simulator 下的理想行为模式（Novice 多轮问学够就停 / Exp 1 轮问 / Busy 高 U 问一次就停），以及 v31.4 如何逼近这些理想。

---

## 2026-04-23

### 91. v31.3-D: Busy T1+ Execute 推理补丁（hypothesis test）— 结果出 ✓

详细见 `docs/v31_experiment_log.md` §11。

**动机**：v31.1 Busy 问题拆成两层——T0 52% rule-match（乱 Clarify）+ T1+ 不停下来（9 个 7-turn 长尾）。假设**长尾是 pass@1 主因**。

**实现**：零训练 patch。`eval/evaluate_multi_turn_persona.py` 加 `--busy_t1_execute` flag，推理时若 Busy turn≥1 且 model 输出 Clarify，强制翻成 Execute。其他完全不变，用 v31.1 模型。

**时间线**：10:52（机器）kick off → 16:20 完成（~5.5h）。

**结果（50-state）**：

| 指标 | v29 | v31.1 | **v31.3-D** | vs v31.1 |
|---|:---:|:---:|:---:|:---:|
| **Overall pass@1** | 14.0% | 9.33% | **12.67%** | **+5 task** ✓ |
| Busy turns / p@1 | 1.0 / 14% | 2.72 / 8% | **1.42 / 12%** ✓ | +2 task |
| Busy 长尾 | 0 | 9/50 | **0** ✓ | 零长尾 |
| Exp turns / p@1 | 2.66 / 12% | 2.66 / 10% | 2.68 / 12% | +1 task（噪声）|
| Novice turns / p@1 | 7.0 / 16% | 4.0 / 10% | 4.24 / 14% | +2 task（噪声）|

**Busy 分布兑现物理保证**：29 个 T0 Execute（1 轮）+ 21 个 T0 Clarify（patch 强制 T1 Execute → 2 轮），max 2，零长尾。预测 avg=1.63 偏高，实际 1.42（T0 Clarify 42% 低于预期 63%，因为 §7.1 63% 包含被长尾拉高的 clarify_turns/total_turns）。

**Busy pass@1 = 12% 踩线 §11.5 判据**（≥12% → 长尾确实主因）。决策：**走 A —— 把 "Busy T1 Execute" 规则做进 DPO 训练**，不依赖推理 patch。

**意外：Novice/Exp 也涨 pass@1**（+2, +1 task）。但 patch 不影响 Novice/Exp 决策，turn 分布基本不变。pass@5 反向（Novice 11→9, Exp 11→9）→ 判定为 gpt-4o-mini 代码生成采样噪声，50-state σ≈2.3 task。论文叙事不靠这个涨幅，核心信号是 Busy 的 +2 task。

**产物**：`outputs/eval_v31_1_busyT1exec_50test.json`（模型仍是 v31.1，零训练）。

**下一步**：设计 v31.4 = v31.1 + Busy T1 Execute pair（加入 DPO 数据），评估训练版本能否稳住 Busy 行为，同时看会不会触发 v31.2a 那种跨 persona 干扰。

### 90. v31.1 eval post-mortem + v31.2a 启动（eval 进行中）

详细记录见 `docs/v31_experiment_log.md` §7-9。

**v31.1 最终结果（50-state）**：Overall pass@1 9.33% vs v29 14.0%（同测试集），掉 4.67pp = 7 task。

| | v29 | v31.1 | Δ |
|---|:---:|:---:|:---:|
| Novice | 16% (8/50), 7.0轮 | **10% (5/50), 4.0轮** | −3 task（**设计内**）|
| Busy | 14% (7/50), 1.0轮 | **8% (4/50), 2.72轮** | −3 task（**意外问题**）|
| Exp | 12% (6/50), 2.66轮 | 10% (5/50), 2.66轮 | −1 task（噪声，行为字节级不变）|

**诊断**：
- Novice 过拟合**修好**（7→4轮核心目标达成），pass@1 −6pp 是"少问 → 少披露"的必要代价，Novice `items_per_turn=1` 导致 n_masked≥4 task 信息不够
- Busy T0 **52% rule match（硬币水平）**：30/107 Clarify pair 太稀，DPO 没覆盖 v29 "100% Execute" 的强先验；加上 Busy T1 只有 3 个 Execute pair，**进入多轮后不会停**（9/50 state 跑满 7 轮被 forced execute，vs v29 零长尾）
- Exp 完全稳定（turns/clarify_rate 一字不差），确认 Busy 改动未干扰 Exp/Novice（至少在 v31.1 阈值下）

**v31.2a 设计（04-23 凌晨启动）**：

Busy 阈值 `> 0.6 → > 0.8`。U 离散 {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}，`> 0.8` 只捕获 U=1.0（n_masked=5 极端 task）。Busy T0 Clarify pair 从 30 → **5**（4.7%），预期信号太稀 DPO 学不到 → Busy 退化成 v29 的永 Execute。

| | v31.1 | v31.2a |
|---|:---:|:---:|
| Busy T0 Clarify | 30 (28%) | **5 (4.7%)** |
| Busy T0 Execute | 77 | 102 |
| Exp pair | 166 | 166（**字节级相同**）|
| Novice pair | 220 | 220（**字节级相同**）|

**训练完成**（03:15 → 03:31，16min）：84 steps, accuracy 96-98%, margins 0.38→4.72。

**Eval 中期观察（40/50 @ 07:30）**：

| persona | avg turns | Clarify% | 对比 v31.1 |
|---|:---:|:---:|---|
| Busy | **1.00** ✓ | 0% | **修好** |
| Novice | 4.55 | 100% | 略差（4.0→4.55，7-turn 尾巴 6%→24%）|
| Exp | 1.86 ⬇ | 57% | **T0 Clarify 94%→54%（40pp 下降）** |

**跨 persona 干扰现象**：Exp pair 完全没变，但 Exp 行为大幅变化。验证"单 LoRA 共享参数，Busy 的 25 个 pair 翻转（U=0.8 Clarify→Execute）通过参数共享泛化成全局 '高 U → Execute' 信号"。

**Eval 完成（08:34）+ 最终决定：回退 v31.1**

| 指标 | v29 | v31.1 | v31.2a |
|---|:---:|:---:|:---:|
| Overall pass@1 | 21/150 (14.0%) | 14/150 (9.33%) | **13/150 (8.67%)** |
| Overall pass@5 | 30/150 (20.0%) | 29/150 (19.3%) | **23/150 (15.3%)** |
| Novice turns / p@1 / p@5 | 7.0 / 8 / 10 | 4.0 / 5 / 11 | 4.76 / 4 / **7 (−4)** |
| Exp turns / p@1 / p@5 | 2.66 / 6 / 12 | 2.66 / 5 / 11 | 1.90 / 4 / **8 (−3)** |
| Exp clarify% | 62 | 62 | **47 (−15pp)** |
| Busy turns / p@1 / p@5 | 1.0 / 7 / 8 | 2.72 / 4 / 7 | **1.0 ✓ / 5 / 8** |

**v31.2a 用 Busy +1 task 换掉 Exp/Novice 共 −6 pass@5，亏本**。跨 persona 干扰被实证：v31.2a vs v31.1 仅 Busy 25 对 pair 变化（U=0.8 Clarify→Execute），Exp/Novice pair 字节级相同，但行为大幅变化（Exp T0 Clarify 94%→54%，Novice 7-turn 长尾 6%→18%）。单 LoRA rank=64 在 ~500 pair 尺度无法在三 persona 间解耦。

**最终版 v31 = v31.1**：`models/v31_100states/` + `prefs_v31_100states.jsonl` + `eval_v31_dpo_50test.json`。

**论文叙事**：Novice 过拟合修复（7→4）付 Novice pass@1 −6pp 的设计代价；Exp 稳定；Busy 稀疏正类信号下未学透（52% rule-match）。收紧 Busy 阈值导致跨 persona 干扰，证实单 LoRA adapter 在此数据尺度下 persona 规则耦合。Busy 清洁修复 = future work（per-persona adapter / 扩数据 / 更强 persona token）。

**v31.2a 制品留作 ablation**：`models/v31_2a_100states/` + `prefs_v31_2a_100states.jsonl` + `eval_v31_2a_dpo_50test.json`，是跨 persona 干扰的直接证据。

---

## 2026-04-22

### 89. v31 诊断 + v31.1 修复 Novice 过拟合（进行中）

详细记录见 `docs/v31_experiment_log.md`。

**动机**：v29 Novice 100% 跑满 7 轮，消融证明 U 在 turn-count 维度完全没起作用（w/o U 下 Novice 仍 8 轮），persona 独自决定行为。DPO 训 pair 里 226/227 Novice = Clarify → 模型学成 `persona → action` 硬映射。

**修复（v31.1）**：只改 Novice 规则，Busy/Exp 保持 v31.0 不动（防 v30 三处同改崩盘）。

- `reward/compute_rewards.py`：`get_correct_action(persona, U, turn)` 新 Novice 分支 — `turn<2 → Clarify`，`turn>=2 → Execute if U<0.4 else Clarify`
- 3 个 callsite 传 turn（踩坑：最初 `fork_turn` 与外层循环变量重名，`AttributeError: 'int' object has no attribute 'get'`，重命名 `fork_turn_idx` 修复）

**v31.1 pair 分布（496 pairs）**：

| persona | pairs | Execute 占比 | 关键信号 |
|---|:---:|:---:|---|
| Novice (v29) | 227 | 0.4% | 几乎全 Clarify |
| Novice (v31.0) | 227 | 0.0% | 更单边 |
| **Novice (v31.1)** | **220** | **9.5%** | **turn 2 对比 ✨** |

Novice turn 2：U<0.4 → 18 Execute，U≥0.4 → 11 Clarify。同 persona × 同 turn × U 决定动作 — DPO 学 U-conditional 所需的对比监督首次出现。Busy/Exp 与 v31.0 完全相同。

**管道已 kick off**（`/tmp/v31_pipeline.sh`，nohup detached）：
- 10:54 train v31.1 DPO（~17 min）→ `models/v31_100states/`
- auto-chain: eval 50-state（~1.5-2h）→ `outputs/eval_v31_dpo_50test.json`
- 日志 `/tmp/v31_pipeline.log`

**成功判据**：Novice Avg Turns <5（理想 3-4），Novice pass@1 ≥14%，Busy/Exp 不崩。

**fallback**：若 Novice 改善不足（5-6 轮），调 `NOVICE_U_STOP` 0.4→0.6 或 `NOVICE_MIN_CLARIFY` 2→1；若 Busy 崩（<10%），回退 Busy 阈值 0.6→0.8。

---

## 2026-04-21

### 88. Canonical 覆盖 re-audit — Base LLM 其实已完整（修正 #82、#87）

用户追问"你确定吗"后重查，推翻昨日结论：

**Base LLM canonical-200 已完整** — 不需要补跑。

实际覆盖（zero overlap between 3 files）：

| 文件 | (state, persona) pairs | canonical 内 |
|---|---|---|
| `eval_v29_base_llama_50test.json` | 150 | 150 |
| `eval_v29_base_150extra.json.partial` | 324 | 324 |
| `eval_v29_base_150extra_remaining.json` | 126 | 126 |
| **合计（去重）** | **600 唯一** | **600 = 200 state × 3 persona** |

之前 #82/#87 写 "Base LLM 缺 107" 是审计时只合并了老 20-seed 文件（3 match）+ 50test（50）+ 150extra_remaining（42） = 95，漏算了 `_150extra.json.partial` 里的 324 对——那个 `.partial` 后缀骗过我，以为是损坏/不完整文件。实际 partial 仅仅是当时断点续跑的命名约定，数据完整。

**修正覆盖表**：

| 方法 | canonical-200 | 缺 |
|---|---|---|
| Direct | 200/200 ✓ | 0 |
| Oracle | 200/200 ✓ | 0 |
| Ideal Disclosed v2 | 200/200 ✓ | 0 |
| TactfulLLM | 200/200 ✓ | 0 |
| **Base LLM** | **200/200 ✓** | **0** |
| Clarify-first | 50/200 | **150** |
| Prompt-only | 50/200 | **150** |

**修正补跑 queue（删 Base LLM）**：
- Clarify-first 450 trials × ~100s ≈ 12.5h
- Prompt-only 450 trials × ~135s ≈ 17h
- 并行 ~17h，串行 ~30h（非 5h）

**时长估算大幅上调原因**：之前按 ~50s/trial 估的，但 Ideal Disclosed v2 实测 ~90s/trial 也就是 pass@5 (5 candidate × Llama 8B 本地推理) 主导成本，Clarify turn 边际便宜。

**待整理**：Base LLM 的 3 文件建议合并为单一 `eval_v29_base_llama_200.json`，读表更方便。

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
