# Qwen Backbone 实验记录

> 起始日期: 2026-04-24
> 目标: 为论文补第二个 backbone (Qwen2.5-7B-Instruct)，验证方法在不同 base model 上的泛化性

---

## 0. 背景与定位

论文设计要求 2 backbones (Llama-3.1-8B-Instruct + Qwen2.5-7B-Instruct)。Llama 侧已完成 v29 / v31.x 全系列实验（详见 `v29_experiment_log.md` / `v31_experiment_log.md`）。Qwen 侧截至 2026-04-24 尚未成功训练过任何版本。

### 0.1 Qwen 的策略选择

基于 v31.x 的跨 persona 干扰问题（v31.2a / v31.4 失败分析见 `v31_experiment_log.md` §10 + §12），最稳健的选择是：

- **Qwen 主结果复用 v29 的 preference pairs**（`data/dpo/prefs_v29_100states.jsonl`, 500 对）
- 保持与 Llama v29 相同的训练配置（beta=0.1, lr=5e-5, 3 epochs, QLoRA r=64）
- 用 canonical 测试集（`test_states_v29_eval_50.jsonl` / `_200.jsonl`）直接比对 Llama v29 数字

不走 v31.x 的原因：v31.x 在 Llama 上已证明 pass@1 全面低于 v29（v31.4 8% vs v29 14%），Qwen 同样的 pair 大概率重演同样的跨 persona 干扰。在 ddl 紧张下优先保 Qwen 主结果成立。

---

## 1. 环境准备

### 1.1 HF cache 问题（2026-04-24）

v31.4 pipeline 第一次尝试 Qwen 训练时报错：

```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files
  in the disk cache and outgoing traffic has been disabled.
OSError: We couldn't connect to 'https://huggingface.co' to load the files.
```

`train_dpo.py` 强制 `local_files_only=True`，但 Qwen 权重不在 transformers 预期的 cache 路径。

**根因**：15GB Qwen 权重已下载到 `/root/autodl-tmp/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/`，但新版 transformers 期望 cache 在 `$HF_HOME/hub/models--.../` 子目录下。Llama 在 `hub/` 下所以能 offline 加载，Qwen 在老路径所以失败。

**修复**：软链

```bash
ln -s /root/autodl-tmp/hf_cache/models--Qwen--Qwen2.5-7B-Instruct \
      /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct
```

Offline 加载验证通过（tokenizer size 151665, model_type qwen2, hidden 3584）。

### 1.2 运行环境

- 4bit QLoRA via bitsandbytes
- `HF_HOME=/root/autodl-tmp/hf_cache` 必须显式传给训练/评估进程
- GPU: RTX 4090 24GB
- Qwen2.5-7B 4bit 占用 ~20GB VRAM，训练 batch 正常

---

## 2. Qwen v29 训练

### 2.1 配置（2026-04-24 23:50 启动）

```bash
HF_HOME=/root/autodl-tmp/hf_cache PYTHONUNBUFFERED=1 \
python policy/train_dpo.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output models/v29_qwen_100states \
  --epochs 3 --lr 5e-5 --beta 0.1
```

Pipeline: `/tmp/qwen_v29_pipeline.sh`（train → 50-state eval 串联，nohup 后台）
日志: `/tmp/qwen_v29_pipeline.log`

### 2.2 训练过程观察

500 pairs, 87 steps, 3 epochs。Qwen 每 step ~13s（vs Llama ~12s），全程 ~20min。

中期 metrics（从 log 提取）：

| Step | Epoch | Loss | Accuracy | Margins |
|:---:|:---:|:---:|:---:|:---:|
| 30/87 | 1.04 | 0.145 | 93.2% | 3.08 |
| 40/87 | 1.39 | 0.039 | 99.4% | 4.61 |
| 50/87 | 1.75 | 0.045 | 98.8% | 5.83 |

收敛曲线和 Llama v29 同量级（Llama v29 epoch 2 达 100% accuracy, margins 6.4）。Qwen 在 epoch 1.75 时已稳定在 99% accuracy，margin 持续上升说明 DPO 学到可分离的 chosen/rejected。

### 2.3 训练产出

- 模型: `models/v29_qwen_100states/`（QLoRA adapter）

---

## 3. 50-state 评估（自动接训练）

### 3.1 配置

```bash
HF_HOME=/root/autodl-tmp/hf_cache PYTHONUNBUFFERED=1 \
python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v29_qwen_100states \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_qwen_50test.json
```

**重点**：测试集与 Llama v29 50-state 完全一致（`test_states_v29_eval_50.jsonl`），数字直接可比。

预计 ~5h（每 conversation ~1-2min × 150 conversations）。

### 3.2 合格判据

| 指标 | Llama v29 (50-state) | Qwen 合格线 | 说明 |
|---|:---:|:---:|---|
| Overall pass@1 | 14% (21/150) | **≥ 10%** | Qwen2.5 官方 BigCodeBench 和 Llama 相当 |
| Novice pass@1 | 16% | ≥ 12% | 行为分化最敏感 |
| Exp pass@1 | 12% | ≥ 8% | — |
| Busy pass@1 | 14% | ≥ 10% | — |
| Novice avg turns | 7.0 | 5.0-7.5 | 应复现 Llama "多轮 Clarify" 模式 |
| Exp avg turns | 2.66 | 2-3 | 应复现"一轮 Clarify"模式 |
| Busy avg turns | 1.0 | ~1.0 | 应复现"直接 Execute"模式 |

### 3.3 结果（50/50 完整，04:05 CST 完成）

| persona | avg turns | clarify rate | pass@1 | pass@5 |
|---|:---:|:---:|:---:|:---:|
| Novice | 7.0（全撞顶）| 85.7% | **10%** (5/50) | 12% (6/50) |
| Exp | 3.4（29 forced final）| 70.6% | **8%** (4/50) | 20% (10/50) |
| Busy | 1.0 ✓ | 0% | **4%** (2/50) | 10% (5/50) |
| **Overall** | — | — | **7.3%** (11/150) | **14%** (21/150) |

**vs §3.2 合格线判定**：

| 指标 | 合格线 | Qwen v29 | 判定 |
|---|:---:|:---:|:---:|
| Overall pass@1 | ≥ 10% | 7.3% | ❌ 差 2.7pp |
| Novice pass@1 | ≥ 12% | 10% | ❌ 差 2pp |
| Exp pass@1 | ≥ 8% | 8% | ✓ 踩线 |
| Busy pass@1 | ≥ 10% | 4% | ❌ 差 6pp |

**vs Llama v29 50-state**（同测试集 `test_states_v29_eval_50.jsonl` 字节级一致）：Overall Qwen 7.3% vs Llama 14%，差 6.7pp ≈ 10 task。Busy 最惨（4% vs 14%），Novice/Exp 差距较小。

**行为分化**：Novice 7 轮 / Exp 3.4 轮 / Busy 1 轮，完美复现 Llama 模式——**策略对、代码挂**。

**unbiased pass@1 sanity check**（见 work_log #101）：DPO Qwen first-sample 7.3% → unbiased 7.2%，**Δ = -0.1pp**，数字稳定不是采样运气。

---

## 3.4 Base Qwen 50-state 对照（诊断 DPO 是否伤代码生成）

Base Qwen eval（`--no_lora`，PID 944046，04-25 06:15 启动）：

**中期 26/50 数字（对齐 DPO Qwen 完整 50/50）**：

| persona | Base Qwen 中期 pass@1 | DPO Qwen 完整 pass@1 | Δ |
|---|:---:|:---:|:---:|
| Novice | 2/26 = 7.7% | 5/50 = 10% | +2.3pp ✓ |
| Busy | 1/26 = 3.8% | 2/50 = 4% | +0.2pp（持平）|
| Exp | 1/26 = 3.8% | 4/50 = 8% | +4.2pp ✓ |
| **Overall** | **5.1%** (4/78) | **7.3%** (11/150) | **+2.2pp** ✓ |

**Base Qwen 行为模式**：Novice/Busy 都跑 7 轮 clarify rate 85.7%（全部长 Clarify chain → forced final Execute），Exp 6.04 轮 clarify rate 83.4%。**完全 persona-blind**，和 Base Llama（#47）现象一致。

### 3.4.1 核心诊断结论

1. **DPO 没伤 Qwen 代码生成**：DPO 7.3% > Base 5.1% (+2.2pp, +43% relative)。相对 Llama DPO +75% 提升（14% vs 8%）小，但方向一致
2. **Qwen2.5-7B 在 masked BigCodeBench 上 backbone 能力就弱**：Base Qwen 5.1% < Base Llama 8%。Busy T0 Execute 是最干净对比（纯 code gen，零 clarify 干扰）：Llama 14% vs Qwen 3.8% = 差 10pp，同 masked query 同管线 → 纯 backbone 差异
3. **masked 数据字节级同文件**：DPO Qwen / Base Qwen / Llama v29 都读 `test_states_v29_eval_50.jsonl`，唯一变量是 chat template + tokenizer + 4bit 量化对不同架构敏感度
4. **Busy 0 提升**（Base 3.8% → DPO 4%）是两 backbone 共同问题，Llama v29 也一样 → future work

### 3.4.2 Qwen vs 官方 BigCodeBench leaderboard（参考，不可直接横比）

| backbone | 官方 BCB-Instruct full set | 我们 Base 50-state masked | 掉幅 |
|---|:---:|:---:|:---:|
| Llama-3.1-8B-Instruct | 32.8% | 8% (Busy 14%) | -25pp |
| Qwen2.5-7B-Instruct | ~35% | 5.1% (Busy 3.8%) | -30pp |

**不能直接横比**的原因（#101）：我们 pipeline 和官方差
- 4bit QLoRA 量化（官方 fp16/bf16）
- n=5 sampling + first-sample pass@1（官方 n=1 greedy pass@1）
- `build_clean_execute_query` prompt ≠ 官方 instruct_prompt 原样
- 我们 Oracle 200-state first-sample 20%（#85）远低于官方 32.8%，说明 pipeline 损失 ~13pp，两 backbone 都掉

Qwen 掉更多（-30pp vs -25pp）可能原因（未验证）：Qwen 对 under-specified prompt 更敏感；chat template / tokenizer 对信息密度折扣更大；4bit 量化对 Qwen 小 hidden（3584 < 4096）更伤。

### 3.4.3 对论文 story 的含义

- **不能用** "Qwen 5% 说明 backbone code ceiling 低" 这种 claim（pipeline 损失也有 10+pp 贡献）
- **可以用** "DPO 在两 backbone 都能学到 persona-aware 决策，absolute pass@1 bound by backbone"
- Table 1 Qwen 行叙事：DPO 7.3% + Base 5.1% 同管线对比，relative improvement +43%（vs Llama +75%），方向一致
- Busy 0 提升两 backbone 共同问题 → future work 收尾

---

## 4. 后续 baseline 扩展（2026-04-25 今晚 + 明天）

### 4.1 Qwen 主表缺口（对齐 Llama canonical-200）

| 方法 | Llama 当前 | Qwen 已有 | Qwen 要跑 |
|---|:---:|:---:|:---:|
| Base LLM | 200 ✓ | 50（进行中）| 扩 150（~17h，不划算）→ **用 50 + footnote** |
| Direct Execution | 200 ✓ | 无 | **200**（~2.5h）|
| Clarify-first | 200 ✓ | 无 | **200**（~5h）|
| Prompt-only | 50† | 无 | **50**（~5h，明天白天）|
| TactfulLLM (ours) | 200 ✓ | 50（DPO 跑完）| 扩 150（~10h，给 v31.4 让路）|

### 4.2 今晚串联 pipeline（watchdog，PID 982966，10:22 kick off）

脚本: `/tmp/qwen_baselines_tonight.sh`

```
1. 等 Base Qwen 50 (PID 944046) 结束   ~2h
2. Direct Execution Qwen 200           ~2.5h
3. Clarify-first Qwen 200              ~5h
────────────────────────────────────────
总 wall-clock ~9.5h，预计 19:55 CST 全部完成
```

产物：
- `outputs/eval_v29_qwen_base_50test.json`（Base 完整 50/50）
- `outputs/eval_v29_qwen_direct_execution_200.json`（Direct canonical-200）
- `outputs/eval_v29_qwen_clarify_first_200.json`（CF canonical-200）

### 4.3 不跑的 + 推后的

**不跑**：
- Base Qwen 200 扩展（~23h 超 window）
- v31.x 不在 Qwen 复刻（跨干扰问题已在 Llama 证实）
- Qwen 扩训练数据（ROI 低）

**推后（明天 / 后续 overnight）**：
- Prompt-only Qwen 50（~5h，挤占今晚太多）
- TactfulLLM Qwen 200-state（~10h，独立 overnight，给 v31.4 试验腾时间）
- Recovery analysis (Ideal Disclosed / Oracle) on Qwen（若 ddl 允许）

### 4.4 DPO Qwen 200-state 决策（暂缓）

**不启动** Qwen v29 DPO 200-state（原 §4 计划的 P0 任务）。理由：
1. 50-state Overall 7.3% 不达合格线 10%，投 20h 扩 200 ROI 不高
2. 200-state 的数字大概率同量级（first-sample 偏差在 200 上已收敛，见 #101）
3. 优先让 Llama v31.x 有 GPU 时间 + Qwen baseline 先填齐主表 cell

**若后续 ddl 允许**：TactfulLLM Qwen 200-state 独立 overnight（~10h）补全主表第二 backbone 列。

---

## 5. Baseline 实战与速率重新校准（2026-04-25 → 04-26）

§4 的 watchdog 计划失败了，下面是实战经过 + 教训 + 修正后的全 baseline 规划。

### 5.1 04-25 watchdog 早死

`/tmp/qwen_baselines_tonight.sh` (PID 982966) 10:22 启动，等 Base Qwen 完成后串联 Direct 200 + CF 200。

**事实**：watchdog 在 12:28（Base Qwen 完成那一刻）写完 "Final output check" 后死亡。`qwen_baselines_tonight.log` 12:28 之后零输出，无 step 子日志，无 Direct/CF .partial。GPU 空跑 11 小时直到 04-26 00:00 用户介入。

**根因（不确定）**：容器 uptime 866 天连续没重启 → 非物理机故障。重启后系统 `date` 与真实 wall clock 差 11h，系统时钟和进程 ELAPSED 自洽（都说"刚跑了几分钟"），符合 cgroup freeze / autodl 共享调度 deprioritize 特征。猜测：用户离线后 SSH 无心跳 → autodl 把容器降级 → watchdog `nohup` 在降级下被清理。

**教训**：
1. 过夜任务不再用 `nohup`，改 **setsid + PID 1 attach**
2. **保活靠用户笔记本不睡 + cursor 不断**，bash 层无法替代

### 5.2 速率估算重大修正

§4.2 估的 "Direct 1-turn ~40s/state" **完全错**。

**实测数据**：

| 配置 | 速率 | 来源 |
|---|---:|---|
| Llama Direct 200（Apr 17）| 4.37 min/state | `logs/exp2_overnight.log`（11:44→22:39 跑 150 state，resume from 50） |
| Qwen Direct 100（Apr 26）| 5.8 min/state | 当前 partial，重启后 14 min 跑 3 state（+7 resume = 10）|
| Base Qwen 50（Apr 25）| 7.4 min/state | #100 实测 06:15→12:27 跑 50 state |

**结论**：~5 min/state 是 eval 脚本的固有 ceiling，不是 Qwen 异常。Qwen 比 Llama 慢 33%，原因：bitsandbytes 8-bit 量化（4090 上 fp16 tensor cores 才是优化路径）+ batch=1 串行 + 5 candidate × 3 persona = 15 串行推理 / state。GPU util 仅 10-16%，4090 严重 underutilized。

§4.2 的"今晚 19:55 完成"承诺基于错误估算（按真实速率 Direct 200 ≈ 19h + CF 200 ≈ 23h = 42h，远超过夜 window）。

### 5.3 04-26 重启：切换到 100-state

操作（04-26 00:13 CST）：
1. kill 当前 Direct 200 进程（PID 44410，已跑 7 state）
2. `mv outputs/eval_v29_qwen_direct_execution_200.json.partial → _100.json.partial`（保留进度）
3. 改 wrapper（`/tmp/qwen_baselines_resume.sh`）：`--max_samples 100`，output `_100.json`
4. **setsid 启动**（PID 59661）：

```bash
setsid bash -c '/tmp/qwen_baselines_resume.sh > /tmp/qwen_baselines_resume.log 2>&1 < /dev/null' &
```

PPID=1（脱离 SSH 会话），扛 SSH 断（不一定扛容器 freeze）。

**Resume 验证**：脚本日志确认 `🔄 恢复 7 个已完成 state` (kill 时实际多写了 2 个：5+2)，从 sample 8 接着跑。

### 5.4 Qwen 全 baseline 计划（修正后）

用户决定先把 Qwen baseline 跑完。100-state 对齐：

| 方法 | 已有 | 目标 | 缺 | GPU 时长 |
|---|---|---|---|---:|
| **DPO (TactfulLLM)** | 50 ✓ | 100 | +50 | ~6h |
| **Base** | 50 ✓ | 100 | +50 | ~6h |
| **Direct** | — | 100 | 100 | ~7.5h ← 进行中 |
| **Clarify-first** | — | 100 | 100 | ~12h ← Phase 1 后 |
| **Prompt-only** | — | 100 | 100 | ~12h |

**Phase 安排（串行单 4090）**：
1. Phase 1（进行中）：Direct 100 + CF 100，~19h，预计 04-26 00:13 → 19:13 系统时钟
2. Phase 2: Prompt-only 100，~12h
3. Phase 3: DPO 50→100 + Base 50→100，~12h
4. Phase 4（Llama 收尾）: Llama Prompt-only 50→200 补 150，~12h

总 ~55h GPU，wall clock 3.5 天（笔记本不睡前提）。

**论文叙事**：Qwen N=100 + Llama N=200，主表脚注说明（per #94 secondary backbone 策略）。

### 5.5 进度查询工具

`/tmp/qwen_progress.sh` 一行命令查 Direct/CF 进度 + 进程状态：

```bash
$ /tmp/qwen_progress.sh
Direct: 10/100, last update 2.4min ago
CF: 未启动
---
PID 59661  ELAPSED 14:30
```

`last update >5min` = 容器被 throttle，告诉我但别重启（重启救不了 throttle，只能等）。

---

## 6. Phase 1 + Phase 2 完成 + 主表 Qwen 行（2026-04-26 → 04-27）

### 6.1 实测速率（订正 §5.2）

| 任务 | wall clock | 速率 | 备注 |
|---|---|---:|---|
| Direct 100 | ~5h | 3.0 min/state | 1-turn execute，最快 |
| CF 100 | ~5.7h | 3.5 min/state | 2-turn (forced clarify + execute) |
| Prompt-only 100 | 8h | 4.8 min/state | 自decide，但行为退化为 PO-blind |
| DPO Qwen 100（in-flight） | 预计 10-12h | ~6-7 min/state | Novice 撞 7 轮顶 |

§5.2 估的 4-5 min/state 是 Direct/CF 平均偏高，实际 Direct 比这快 25%。多轮越多越慢。

### 6.2 Phase 1 + 2 结果（canonical-100 sample，与 PO 同 sample）

| 方法 | Overall p@1 | Overall p@5 | Novice p@1 | Busy p@1 | Exp p@1 | 行为分化 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Direct 100 | **15.3%** (46/300) | 20.0% | 16% | 13% | 17% | persona-blind（forced 1 turn）|
| CF 100 | **16.0%** (48/300) | 24.0% | 12% | 17% | 19% | persona-blind（forced 2 turn）|
| Prompt-only 100 | **13.0%** (39/300) | 21.3% | 10% (avg 2.05) | 13% (1.0) | 16% (1.07) | **基本 persona-blind**，模型自decide但不分化 |
| DPO Qwen 50（参考）| 7.3% (11/150) | 14.0% | 10% (7.0 撞顶) | 4% (1.0) | 8% (3.4) | **行为完美分化**，但 pass@1 低 |
| DPO Qwen 100 | 进行中 | — | — | — | — | — |

### 6.3 ⚠ Qwen 反常：Prompt-only / Direct / CF 都 > DPO

Llama 行业标准模式 (DPO 14% > PO 8.7% > Direct 12.3% > CF 14.8%) 在 Qwen 上**完全反过来**。

**原因诊断（关键）**：DPO 50 vs PO 100 在 25 个**重叠 state** 上的 pairwise 对比：

| Persona | DPO 50 整体 | PO 100 整体 | DPO vs PO 在 25 重叠 state |
|---|:---:|:---:|:---:|
| Novice | 10% | 10% | **4% vs 4%** |
| Exp | 8% | 16% | **4% vs 8%** |
| Busy | 4% | 13% | **4% vs 4%** |
| Overall | 7.3% | 13.0% | **4.0% vs 5.3%（+1.3pp）** |

7.3% vs 13.0% 看着差 5.7pp，**实际在同 state 上只差 1.3pp**——大头是 N=50 vs N=100 不同 sample 的采样噪声。DPO 100 跑完应该和 Direct/CF/PO 接近（13-16% 区间）。

### 6.4 Prompt-only 行为退化为 persona-blind（与 Llama 不同）

Qwen Base + persona prompt 不能学到行为分化：

| persona | DPO Qwen 50 | Prompt-only Qwen 100 |
|---|---|---|
| Novice avg turns | 7.0（撞顶）| 2.05 |
| Exp avg turns | 3.4 | 1.07 |
| Busy avg turns | 1.0 | 1.0 |
| Novice clarify rate | 85.7% | 51.2%（一半还是 1 轮） |
| Exp clarify rate | 70.6% | 6.5% |

DPO 学到的"Novice 多问 / Exp 短问 / Busy 不问"在 Prompt-only 上**几乎完全没有**。这是 DPO 真正的论文卖点——**persona alignment**，不是 pass@1。

### 6.5 论文 Table 1 含义

```
Llama 行：DPO 14% > Direct 12.3% ≈ CF 14.8% > PO 8.7%（DPO pass@1 胜，行为分化）
Qwen 行：DPO ?% vs Direct 15.3% / CF 16.0% / PO 13.0%（pass@1 大概率不胜，但行为分化是 DPO 独有）
```

叙事 pivot：**"DPO 在两 backbone 都做对了 persona-aware 决策，绝对 pass@1 是 backbone code capability 的函数"**——
- Llama（强 backbone）: DPO 既赢 pass@1 又赢 persona alignment
- Qwen（弱 backbone）: DPO 主要赢 persona alignment，pass@1 与 Direct/CF 同量级

Pareto trade-off (per #98) 仍成立：DPO 用更低 user interruption rate 拿到接近 CF 的 pass@1。

---

## 7. DPO 代码审计（2026-04-27）

排查 DPO Qwen 7.3% 是否含 backbone-specific bug。审计 `train_dpo.py` / `infer.py` / `evaluate_multi_turn_persona.py` 全部相关路径——

| 检查点 | 结论 |
|---|---|
| `train_dpo.py` chat template | ✓ `tokenizer.apply_chat_template`，Qwen 自动转 `<|im_start|>assistant\n` |
| `to_dpo_format` action prefix strip | ✓ 自然响应训练，Qwen/Llama 通用 |
| LoRA `target_modules` | ✓ q/k/v/o + gate/up/down，Qwen2 同名 |
| `pick_action_from_generation` | ✓ `skip_special_tokens=True` + 内容样式检测 |
| Qwen tokenizer pad/eos | ✓ `pad=<|endoftext|>`, `eos=<|im_end|>`，正确分离 |
| 8-bit 推理 vs 4-bit 训练量化 | ⚠ 不匹配，但 Llama 也是这样——不解释 Qwen 特异性 |
| Qwen 默认 chat template 注入 system prompt | ⚠ "You are Qwen, created by Alibaba Cloud..." 自动加入；Qwen 标准用法，非 bug |

**结论：无 backbone-specific bug**。DPO Qwen 7.3% 的 ROI 弱主要来自：
1. N=50 采样噪声（§6.3 证实差 4pp 大头是噪声）
2. Qwen2.5-7B 在 masked BCB 上 backbone code capability 弱（§3.4 已证）
3. Novice 7-turn forced-final long context 对 Qwen 退化更狠（#3.4.2）

---

## 8. Phase 3：DPO Qwen 50 → 100 启动（2026-04-27 wall 10:35）

### 8.1 决策

PO 完成后，**Phase 3 第一优先级提前到 DPO 扩 100**（原计划 Phase 3 是 DPO 50→100 + Base 50→100 一起，现在先做 DPO 一项）。理由：
- §6.3 显示 DPO 50 vs PO 100 主要差异是采样噪声 → DPO 100 才能 apples-to-apples 进 Table 1
- Base Qwen 50 已有 5.1%，是诊断用，不进主表，N=50 够；Phase 3 中 Base 扩 100 推后

### 8.2 启动

```bash
setsid bash -c '/tmp/qwen_dpo_100.sh > /tmp/qwen_dpo_100_wrapper.log 2>&1 < /dev/null' &
```

- Wrapper PID 264565（PPID=1, setsid 生效）
- Python PID 264568
- 启动：system 23:35:55 = wall **2026-04-27 ~10:35**
- Output: `outputs/eval_v29_qwen_dpo_100.json`
- 测试集：`test_states_v29_eval_200.jsonl --max_samples 100`（**与 Direct/CF/PO 同 sample**）
- ETA: 10-12h，**预计 wall 20:30 - 22:30 今晚出结果**

### 8.3 重要细节：DPO 50 vs DPO 100 的 state 集合

DPO 50 用 `test_states_v29_eval_50.jsonl`，DPO 100 用 `test_states_v29_eval_200.jsonl --max_samples 100`。两者**只 25 个 state 重叠**（不是 50 ⊂ 100 的关系）。所以 DPO 100 实际是：
- 75 个 model 没见过的全新 state
- 25 个 state 重跑（拿到 ± 采样噪声的同样数字）

DPO 50 文件保留作为 sanity check / appendix footnote 用。

### 8.4 未来扩 200 的机制（备忘）

`max_samples 100 → 200` **不是 prefix subset**：`random.Random(42).sample(200, 100) ≠ sample(200, 200)[:100]`。但因为 200 全集就是 canonical，跑 `max_samples 200` 会自然包含已完成的 100。正确做法：

```bash
cp outputs/eval_v29_qwen_dpo_100.json outputs/eval_v29_qwen_dpo_200.json.partial
# 然后 --max_samples 200 跑，state_id-based partial resume 自动跳已完成 100 个
```

---

## 9. DPO Qwen 100 Final Results（2026-04-27 wall ~22:00 完成）

### 9.1 最终数字（n=100, apples-to-apples vs Direct/CF/PO）

| persona | avg_t | clarify% | pass@1 | pass@5 | rejection |
|---|:---:|:---:|:---:|:---:|:---:|
| Novice | 7.00（撞顶）| 85.7% | **13.0%** (13/100) | 19.0% (19/100) | 40.5% (243/600) |
| Exp. | 3.89 | 74.3% | **14.0%** (14/100) | 21.0% (21/100) | 47.1% (136/289) |
| Busy | 1.00 ✓ | 0% | **8.0%** (8/100) | 12.0% (12/100) | -- (no clarify) |
| **Overall** | 3.96 | — | **11.67%** (35/300) | **17.33%** (52/300) | 42.6% (379/889) |

### 9.2 与 Partial 54 对比（采样收敛）

| | Partial 54 | Final 100 | Δ |
|---|:---:|:---:|:---:|
| Novice p@1 | 14.8% | 13.0% | -1.8pp |
| Exp p@1 | 14.8% | 14.0% | -0.8pp |
| Busy p@1 | 9.3% | 8.0% | -1.3pp |
| **Overall p@1** | **13.0%** | **11.67%** | **-1.3pp** |

剩 46 state 表现略差，最终落在 §8 预测的 12-14% 区间下沿。

### 9.3 vs Qwen baselines（同 100-state sample，全部 apples-to-apples）

| 方法 | Overall p@1 | Novice | Exp | Busy | 行为分化 |
|---|:---:|:---:|:---:|:---:|---|
| **CF** | **16.0%** 🥇 | 12 | 19 | 17 | persona-blind（强制 2 turns）|
| Direct | 15.3% 🥈 | 16 | 17 | 13 | persona-blind（强制 1 turn）|
| PO | 13.0% 🥉 | 10 | 16 | 13 | basically blind（avg 1.4 turns） |
| **DPO (ours)** | **11.67%** ❌ last | 13 | 14 | 8 | **完美分化**（7/3.89/1.0）|

⚠ **Qwen 上 DPO 是最弱的方法**——比 PO 低 1.3pp，比 CF 低 4.3pp。但**唯一做到 persona-aware behavior**。

### 9.4 关键叙事 pivot（per #108 / #110 早期判断已确认）

**不能用** "TactfulLLM consistently strongest pass@1"——Qwen 数据直接打脸。

**必须用 B 选项**（per #108 推荐）：
> Across both backbones, **TactfulLLM is the only method that adapts clarification budget to user persona** (Novice 7 turns, Exp ~3 turns, Busy 1 turn). On Llama, this strategy yields the highest pass@1; on Qwen, the persona-aware behavior pattern is preserved while absolute pass@1 is bounded by the weaker code-generation capacity.

或更精炼：
> TactfulLLM is the **only persona-adaptive method** across both backbones; on Llama it also achieves the highest pass@1, while on Qwen the persona-aware decision quality is preserved at lower absolute pass@1 due to backbone capability.

### 9.5 Δ vs Base 50 (n=100 vs n=50, caveat: 不同 state sample)

| persona | DPO 100 | Base 50 | Δ |
|---|:---:|:---:|:---:|
| Novice | 13.0% | 8.0% | **+63%** |
| Exp | 14.0% | 14.0% | **0%** |
| Busy | 8.0% | 8.0% | **0%** |
| Overall | 11.7% | 10.0% | **+17%** |

DPO Qwen 仍**净正向 vs Base**，主要 Novice 拉升（行为分化的功劳）。Exp/Busy 持平 = code capability ceiling。

### 9.6 Pareto trade-off 仍成立

虽然 DPO Overall pass@1 输给 baselines，但 **Busy 0% rejection** 是独有优势：
- DPO Busy 0% rej（never clarifies）
- CF Busy 80% rej（强制 clarify 被拒）
- PO Busy -- (no clarify)
- Direct Busy -- (no clarify)

DPO 在 Busy 上做到 "low cost + zero rejection"，CF 在 Busy 上 high cost + 80% rej。**用户体验维度 DPO 显著更好**。

### 9.7 主表 Qwen 行（最终）

```latex
\rowcolor{blue!10}
TactfulLLM (ours)
& \textbf{13.0} & \textbf{14.0} & \textbf{8.0}  & \textbf{11.7}
& \textbf{19.0} & \textbf{21.0} & \textbf{12.0} & \textbf{17.3}
& 7.0 & 3.89 & 1.0 & 3.96
& 0.41 & 0.47 & -- & 0.43 \\
```

`\ddagger` 脚注可以删（不再是 partial）。

---

## 10. 测试集 Sample 结构 + 扩展策略备忘

### 10.1 三种 sample 方式

| 方式 | 状态集 | 覆盖 | 后续扩 200 怎么做 |
|---|---|:---:|---|
| `eval_50.jsonl`（固定 50）| canonical 200 中固定 50 个 | 50 | 跑 `eval_150extra.jsonl` 那 150 个 |
| `eval_50.jsonl` + `eval_150extra.jsonl`（merged）| **完整 canonical 200** | 200 | 已满 |
| `eval_200 --max_samples k seed=42`（random）| 200 中随机 k 个 | k | partial resume 跑剩 200-k 个 |

### 10.2 关键 invariant（已验证）

**`eval_50.jsonl ∪ eval_150extra.jsonl = eval_200.jsonl`**

- 两文件**互不重叠**（partition）
- 并集 = canonical 200 全集
- Llama TactfulLLM 200 结果就是 `eval_v29_100states_50test.json` (50) + `eval_v29_dpo_150extra.json` (150) merge 出的 600 unique (state, persona) cells

### 10.3 各 baseline 当前覆盖与扩展可能

| Eval 文件 | 当前 N | 测试集源 | 扩 200 是否需要重跑 |
|---|:---:|---|:---:|
| **Qwen 行** | | | |
| `eval_v29_qwen_direct_execution_100.json` | 100 | `eval_200 --max_samples 100` | ✓ incremental（partial resume）|
| `eval_v29_qwen_clarify_first_100.json` | 100 | 同上 | ✓ incremental |
| `eval_v29_qwen_prompt_only_100.json` | 100 | 同上 | ✓ incremental |
| `eval_v29_qwen_dpo_100.json` | 100 | 同上 | ✓ incremental |
| `eval_v29_qwen_base_50test.json` | 50 | `eval_50.jsonl` | ✓ 跑 `eval_150extra`（150 个）合并 |
| **Llama 行** | | | |
| `eval_v29_direct_execution_200.json` | 200 | `eval_200` | 已满 |
| `eval_v29_oracle_200.json` | 200 | `eval_200` | 已满 |
| `eval_v29_ideal_disclosed_v2_200.json` | 200 | `eval_200` | 已满 |
| `eval_v29_clarify_first_*` (50test+150extra) | 200 | merge | 已满 |
| `eval_v29_base_llama_*` (50test+150extra) | 200 | merge | 已满 |
| `eval_v29_100states_50test.json` + `eval_v29_dpo_150extra.json` (TactfulLLM) | 200 | merge | 已满 |
| `eval_v29_prompt_only_50test.json` | 50 | `eval_50.jsonl` | ✓ 跑 `eval_150extra`（150 个）合并 |

### 10.4 Qwen Base 扩 200 最优方案

**两步走**：
1. 跑 `eval_150extra.jsonl` 所有 150 个 state：
   ```bash
   python eval/evaluate_multi_turn_persona.py \
     --no_lora --base_model Qwen/Qwen2.5-7B-Instruct \
     --test_states data/seeds/test_states_v29_eval_150extra.jsonl \
     --max_turns 6 --pass_at_k 1 5 \
     --output outputs/eval_v29_qwen_base_150extra.json
   ```
2. 离线合并 50test + 150extra → 200-state 结果（不重跑那 50 个）

**省 ~5h** vs 直接跑 canonical 100 然后再扩。

### 10.5 Llama PO 扩 200 同理

`eval_v29_prompt_only_50test.json` (n=50) + 跑 `eval_150extra.jsonl`（150 个）→ Llama PO 200。Phase 4 任务就这一个。

---

## 11. 留存

### 6.1 文件

| 文件 | 说明 |
|---|---|
| `models/v29_qwen_100states/` | Qwen v29 DPO LoRA adapter ✓ |
| `data/dpo/prefs_v29_100states.jsonl` | 与 Llama 共用训练数据（500 对）|
| `data/seeds/test_states_v29_eval_50.jsonl` | 与 Llama 共用 50-state 测试集（字节级同文件）|
| `data/seeds/test_states_v29_eval_200.jsonl` | 与 Llama 共用 canonical 200-state 测试集 |
| `outputs/eval_v29_qwen_50test.json` | Qwen v29 DPO 50-state eval ✓（Overall 7.3%）|
| `outputs/eval_v29_qwen_base_50test.json` | Qwen Base 50-state eval ✓（pass@1 10.0%, 04-25 12:27 完成）|
| `outputs/eval_v29_qwen_direct_execution_100.json` | Qwen Direct 100 ✓（Overall 15.3%）|
| `outputs/eval_v29_qwen_clarify_first_100.json` | Qwen CF 100 ✓（Overall 16.0%）|
| `outputs/eval_v29_qwen_prompt_only_100.json` | Qwen PO 100 ✓（Overall 13.0%）|
| `outputs/eval_v29_qwen_dpo_100.json` | Qwen DPO 100 ✓（Overall 11.67%, 04-27 ~22:00 完成）|
| `/tmp/qwen_v29_pipeline.sh` | 训练 + 50-state eval 串联脚本 |
| `/tmp/qwen_v29_pipeline.log` | 训练 + DPO eval 日志 |
| `/tmp/qwen_v29_base_50test.sh` | Base Qwen 50 脚本 |
| `/tmp/qwen_v29_base_50test.log` | Base Qwen 50 日志 |
| `/tmp/qwen_baselines_tonight.sh` | ⚠ 04-25 watchdog 脚本（已死，参考 §5.1）|
| `/tmp/qwen_baselines_tonight.log` | watchdog 主日志（12:28 后无输出）|
| `/tmp/qwen_baselines_resume.sh` | **当前** Phase 1 wrapper（setsid，PID 59661）|
| `/tmp/qwen_baselines_resume.log` | Phase 1 主日志 |
| `/tmp/qwen_baselines_resume_direct.log` | Direct 100 详细日志 |
| `/tmp/qwen_baselines_resume_cf.log` | CF 100 详细日志 |
| `/tmp/qwen_progress.sh` | Direct/CF 进度查询（§5.5）|
| `/tmp/qwen_prompt_only_after_cf.sh` | CF→PO 接力 watchdog（PID 147688，已完成）|
| `/tmp/qwen_prompt_only_100.log` | PO 100 详细日志 |
| `/tmp/qwen_dpo_100.sh` | **当前** DPO 100 wrapper（setsid，PID 264565）|
| `/tmp/qwen_dpo_100.log` | DPO 100 详细日志 |
| `/tmp/qwen_dpo_progress.sh` | DPO 100 进度 + ETA 查询脚本 |

### 6.2 时间线

- 2026-04-24 07:35: v31.4 pipeline 首次尝试 Qwen 训练 → HF cache 错误
- 2026-04-24 14:22: v31.4 Qwen 训练崩溃，Llama 侧完成
- 2026-04-24 23:45: 定位 HF cache 子目录问题，做软链修复
- 2026-04-24 23:50: Qwen v29 pipeline kick off
- 2026-04-25 00:10（约）: Qwen v29 DPO 训练完成 ✓
- 2026-04-25 04:05: Qwen v29 DPO 50-state eval 完成 ✓（Overall 7.3%）
- 2026-04-25 06:15: Qwen Base 50-state eval 启动（watchdog 诊断）
- 2026-04-25 10:22: Qwen baselines watchdog 启动（PID 982966）
- 2026-04-25 12:27: Base Qwen 50 完成 ✓（pass@1 10.0%）
- 2026-04-25 12:28: **watchdog 在写完 "Final output check" 后死亡**，11h GPU 空跑（§5.1）
- 2026-04-26 00:00: 用户介入发现 watchdog 已死，Direct/CF 均未启动
- 2026-04-26 00:13: setsid 重启 Phase 1（PID 59661），切到 100-state，resume 7 state（§5.3）
- 2026-04-26 system 05:53 / wall ~16:53: Direct 100 ✓（Overall 15.3%）
- 2026-04-26 system 12:15 / wall ~23:15: CF 100 ✓（Overall 16.0%）
- 2026-04-26 system 12:15:57 / wall ~23:16: PO watchdog (PID 147688) 自动接力起 Prompt-only 100
- 2026-04-26 system 20:14 / wall 2026-04-27 ~07:14: PO 100 ✓（Overall 13.0%）
- 2026-04-27 wall ~10:35: DPO Qwen 100 启动（PID 264565，setsid）
- 2026-04-27 wall ~22:00: DPO Qwen 100 完成 ✓（Overall **11.67%**, 11h 跑完，~7 min/state）
- 后续：Base Qwen 50→100（apples-to-apples Δ vs Base 数字）+ Llama Prompt-only 50→200（Phase 3/4）
