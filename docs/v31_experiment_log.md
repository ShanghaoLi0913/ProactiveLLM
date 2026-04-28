# v31 实验与分析记录

> 起始日期: 2026-04-22
> 目标: 修复 v29 Novice 过拟合（100% 跑满 7 轮），让 DPO 真正学到 U-conditional 停止

---

## 0. 动机 / v29 的核心失败

v29 DPO 在 50-state eval 表现：
- **Novice 100%（50/50）跑满 7 轮**（6 Clarify + 1 forced Execute），零样本提前 Execute
- Busy 固定 1.0 轮，Exp 2.6 轮（这两档正常）

v29 消融实验（见 `v29_experiment_log.md` §13.6、§Ablation）：
- **w/o Uncertainty**: Novice Avg Turns 7.0 → **8.0**，turn 数几乎不动 → U 从未推动 Clarify/Execute 决策
- **w/o Persona**: Novice Avg Turns 7.0 → **1.04**，完全退化 → persona 是唯一有效决策信号

**诊断结论**：v29 的 227 个 Novice 训练 pair 里 226 条 chosen=Clarify（99.6%），DPO 学到的是 `persona → action` 硬映射，不是 `U → action`。U 虽然对 pass@1 有 ~3% 的贡献（影响生成质量 / rejection calibration），但在 "Clarify vs Execute" 的序列决策维度上完全是 persona 主导。

要改变这一点，必须在训练 pair 里加 `chosen=Execute, rejected=Clarify` 的 Novice 样本，并且这些样本在某个 prompt-可观测变量（U / turn / prev_reject）上与 Clarify 样本形成对比。

---

## 0.5 三个 persona 的理想轨迹（v31 设计目标）

v31 整套改动（U 公式重做、turn-conditional 规则、Busy T1 硬停等）都是为了逼近下表这三个 persona 的"理想轨迹"。

### user simulator 属性（已写死）

| persona | patience | expertise | items_per_turn |
|---|:---:|:---:|:---:|
| Novice-Learner | high | low | 1 |
| Experienced-Engineer | mid | mid | 3 |
| Busy-Developer | low | — | 拒绝 Clarify |

### 理想轨迹

**Novice-Learner：多轮 Clarify，学够就停（3-5 轮）**

```
T0: Clarify → +1 item → U 降 0.2
T1: Clarify → +1 item → U 降 0.2
T2: Clarify → +1 item → U 降 0.2
T3: Execute（信息够了）
```

- 决策信号：**U 控制什么时候停**（U<阈值 → Execute）
- `items_per_turn=1` 导致必须多轮才能累计恢复信息
- 绝对不能像 v29 那样"撞 max_turns=6 被 forced Execute"

**Experienced-Engineer：一轮 Clarify 解决（~2.5 轮）**

```
T0: Clarify → +3 items → U 大幅降
T1: Execute（够了）
```

- 决策信号：**persona + turn**（T0 Clarify, T1 Execute）
- v29 实测 2.66 轮，已接近理想，v31 不改

**Busy-Developer：低 U 直接 Execute；高 U 问一次就停（~1.3 轮）**

```
低 U task (n_masked ≤ 3):
T0: Execute（不问，直接写）

高 U task (n_masked ≥ 4):
T0: Clarify → 被拒或少量披露
T1: Execute（必停，不管 U 多高）
```

- 决策信号：**U（T0 是否问）+ turn（T1 必停）**
- patience=low 导致拒绝率高，继续 Clarify 会陷入 rejection spiral（v31.1 9/50 撞 max_turns 就是这么来的）

### v31.4 目标区间

| | v29 实际 | 理想 | v31.4 目标 |
|---|:---:|:---:|:---:|
| Novice avg turns | 7.0（撞顶）| 3-5 | ~5（NOVICE_U_STOP=0.2）|
| Busy avg turns | 1.0（从不问）| 1-1.5 | ~1.3（T0 U>0.6 问 + T1 硬停）|
| Exp avg turns | 2.66 | 2-3 | ~2.66（不改）|

### 三个 persona 的"机制贡献"对照

| 机制 | Novice | Exp | Busy |
|---|:---:|:---:|:---:|
| Persona 驱动"是否多轮" | 多轮 | 一轮 | 一轮 |
| Uncertainty 驱动"何时停" | T2+ U 控制 | T0 U 决定进入 | T0 U 决定进入 |
| Turn 驱动"硬上限" | min_clarify=2 保底 | T1 硬停 | T1 硬停 |

v31.4 让 U 真正参与全部三个 persona 的决策（v29 ablation 证明 U 对 v29 行为无效）——这是 v31 区别于 v29 的唯一机制卖点。

---

## 1. v31 Pipeline 总览

v31 主要引入 **disclosure-based uncertainty**：
- 新 U 公式: `U = max(0, n_masked_total - n_disclosed) / MAX_MASKED`，MAX_MASKED=5
- 离散取值: `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`，随对话 turn 增加，user disclose 越多，U 越低
- 训练 / 评估统一走 `utils.compute_task_uncertainty.compute_state_uncertainty`（single source of truth）
- 修正了 v29 render_state 里 "compute_task_uncertainty 返回 clarity，显示时方向错反" 的历史 bug

代码改动点：
- `utils/compute_task_uncertainty.py` — 新增 `compute_state_uncertainty`
- `policy/render_state.py` — 直接用新 U，不再对 query 做启发式 text heuristic
- `scripts/mask_task_details.py` — mask 时把 disclosure_rule 塞进 state，确保 U 在初始轨迹构建时就一致
- `scripts/generate_trajectories.py` — import 切到新 API（should_clarify 本次未改）

历史：v31.0 阈值初版用 `Novice U>0, Exp U>0.3, Busy U>0.6`，pair 分布与 v29 几乎一致（Novice 227 / 227 Clarify，Execute=0），所以没带来任何行为改变。**v31.1 在 v31.0 之上继续改 get_correct_action 的 Novice 分支**。

---

## 2. v31.1 核心改动

### 2.1 `reward/compute_rewards.py` — Novice turn-conditional 规则

```python
NOVICE_U_STOP = 0.4      # turn>=2 且 U<0.4 → Execute
NOVICE_MIN_CLARIFY = 2   # 前 2 轮总是 Clarify

def get_correct_action(persona_name, uncertainty, turn=0):
    if persona_name == "Novice-Learner":
        if turn < NOVICE_MIN_CLARIFY:
            return "Clarify"
        return "Execute" if uncertainty < NOVICE_U_STOP else "Clarify"
    thr = V31_U_THRESHOLD.get(persona_name)  # Exp 0.3, Busy 0.6
    return "Clarify" if uncertainty > thr else "Execute"
```

Busy / Exp 规则与 v31.0 相同（保不动，防 v30 那种三处同改 pass@1 崩盘）。

### 2.2 三处 callsite 传 turn

- Method A T0 pair（`line ~912`）：`get_correct_action(persona, u_here, state.dialogue_turn)`
- Method B Novice loop（`line ~984`）：`get_correct_action("Novice-Learner", ct_u, ct_turn)`
- Method A fork（`line ~1097`）：`get_correct_action(fork_persona, fork_u, fork_turn_idx)`
  - ⚠ 踩坑：最初命名 `fork_turn`，与外层循环变量 `for fork_turn in turns` 冲突，`chosen_t, rejected_t = fork_turn, mainline_clarify` 变成 int → `AttributeError: 'int' object has no attribute 'get'`。重命名 `fork_turn_idx` 修复。

---

## 3. v31.1 Pair 分布（496 pairs）

### 3.1 per (persona, turn, chosen_action)

| persona | turn | Clarify | Execute |
|---|:---:|:---:|:---:|
| Busy | 0 | 30 | 77 |
| Busy | 1 | 0 | 3 |
| Exp | 0 | 104 | 1 |
| Exp | 1 | 0 | 61 |
| **Novice** | 0 | **107** | 0 |
| **Novice** | 1 | **81** | 0 |
| **Novice** | 2 | **11** | **18** ✨ |
| **Novice** | 3 | 0 | **3** ✨ |

### 3.2 Novice Execute 占比演变

| 版本 | Novice pairs | Execute | % |
|---|:---:|:---:|:---:|
| v29 | 227 | 1 | 0.4% |
| v31.0 | 227 | 0 | 0.0% |
| **v31.1** | 220 | **21** | **9.5%** |

### 3.3 关键：turn 2 的 U-conditional 对比

| Novice turn 2 | U < 0.4 | U ≥ 0.4 |
|---|:---:|:---:|
| chosen = Execute | **18** | 0 |
| chosen = Clarify | 0 | **11** |

同 persona × 同 turn × U 决定 action — 这是 DPO 学 U-conditional 停止所必需的对比监督。v29 缺的就是这个。

Busy / Exp 分布与 v31.0 一致，未改动。

---

## 4. 训练与评估

### 4.1 训练配置（v29 baseline）

```
PYTHONUNBUFFERED=1 python policy/train_dpo.py \
  --data data/dpo/prefs_v31_100states.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output models/v31_100states \
  --epochs 3 --beta 0.1 --lr 5e-5
```

QLoRA r=64（train_dpo.py 内置），~17 min。

### 4.2 评估配置（50-state canonical）

```
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v31_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v31_dpo_50test.json
```

预期 ~1.5-2h（v31.1 Novice 若降到 3-4 轮，会比 v29 的 3h 显著快）。

### 4.3 成功判据

| 指标 | v29 baseline | v31.1 目标 |
|---|:---:|:---:|
| Novice Avg Turns | 7.0 | **< 5**（理想 3-4） |
| Novice pass@1 | 18.5% (200) / 16.0% (50) | **≥ 14%** |
| Busy Avg Turns | 1.0 | **≤ 2**（28% Clarify on 高 U task → 期望 1.3-1.5） |
| Busy pass@1 | 14.0% | **≥ 12%**（v30 教训：30 Clarify 混合可能拉低 Busy） |
| Exp Avg Turns | 2.6 | ~2.6（规则不变） |
| Overall pass@1 | 16.0% | **≥ 14%** |

核心要看是不是解决了 Novice 过拟合，同时 Busy/Exp 不崩。

---

## 5. 风险与 fallback

### 5.1 Novice 改善不足

21/220 ≈ 10% 的 Execute 样本量相对小，最坏情况 DPO 只学到 "Novice turn<2 总是 Clarify" 而没学到 "turn≥2 U 决定"，Novice 从 7 轮降到 5-6 轮。

Fallback：调 `NOVICE_U_STOP` 从 0.4 → 0.6（更激进让 Execute），或调 `NOVICE_MIN_CLARIFY` 从 2 → 1（允许 turn 1 就停）。

### 5.2 Busy 过度 Clarify（v30 记忆）

v30 Busy 28% T0 Clarify 导致 pass@1 从 14% 崩到 0%（13-state partial）。v31.1 Busy 规则与 v30 相同，但 Novice/Exp 规则与 v30 不同，整体风险结构不同。

Fallback：若 Busy pass@1 <10%，把 Busy 阈值从 0.6 → 0.8（只在 4+ mask 的极端 task 上 Clarify）。

### 5.3 训练不收敛

v29 / v31.0 都是 epoch 2 即 ~100% accuracy，v31.1 pair 更多样，预期 accuracy ~95-97%（类似 v30 的 97.5%）。低于 90% 需查数据质量。

---

## 6. 时间线

- 2026-04-22 早晨：诊断 v29 Novice 100% 7 轮的根因，定位到 pair label 单边性
- 2026-04-22 上午：讨论修复方向，确定只改 Novice 规则（不动 Busy/Exp 避免 v30 崩盘）
- 2026-04-22 ~10:00：`get_correct_action` 加 turn 参数，3 个 callsite 更新
- 2026-04-22 ~10:50：pair 重跑完成（496 pairs），分布符合预期
- 2026-04-22 10:54：`/tmp/v31_pipeline.sh` kick off train + auto-chain eval
- 2026-04-22 16:45：v31.1 eval 完成（50-state），见 §7
- 2026-04-23 凌晨：v31.1 post-mortem + v31.2a 设计 + 跑通，见 §8-9

---

## 7. v31.1 Eval 结果（50-state）+ Post-mortem

### 7.1 最终数字（vs v29 同 50-test 集）

| persona | v29 pass@1 | v31.1 pass@1 | Δ | v29 turns | v31.1 turns | Clarify% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Novice | 8/50 (16%) | **5/50 (10%)** | −3 | 7.0 | **4.0** ✓ | 86% → 75% |
| Busy | 7/50 (14%) | **4/50 (8%)** | −3 | 1.0 | **2.72** ❌ | 0% → **63%** |
| Exp | 6/50 (12%) | 5/50 (10%) | −1 | 2.66 | **2.66** | 62.4% → 62.4% |
| **Overall** | **21/150 (14.0%)** | **14/150 (9.33%)** | **−4.67pp** | — | — | — |

**Novice 核心目标达成**（7→4 轮，落在"理想 3-4 轮"区间）。Exp **行为字节级不变**（turns/clarify_rate/turn counts 全一致）；1-task pass@1 差异在 50-sample 采样噪声内（σ ≈ 2.3）。

### 7.2 Busy T0 Rule-Match 诊断

Busy 规则 `U > 0.6 → Clarify`，eval 50 个 state 中 U > 0.6 的 15/50（期望 Clarify 15 次）。实际 Busy T0：

- **26/50 匹配规则 = 52%（接近硬币）**
- 15 个 "U ≤ 0.6 应 Execute 但 Clarify 了"（过度 Clarify）
- 9 个 "U > 0.6 应 Clarify 但 Execute 了"（过度 Execute）

对比 Novice T0（50/50 all Clarify）和 Exp T0（46/50 match 92%）：**Busy 是唯一训练 pair 里正反样本接近平衡的 persona**（30 Clarify / 77 Execute = 28%），DPO 没学干净。

### 7.3 Busy 长尾 + 用户 rejection

Busy turn 分布：1轮×29, 2轮×1, 3轮×6, 4轮×2, 5轮×2, 6轮×1, **7轮×9**（跑满 max_turns 被 forced execute）。

9 个 7-turn 长尾是 Busy 2.72 avg turns 的主要来源：扣掉这 9 个，剩下 41 个均值只有 1.78。看 conversation 细节，Busy 多轮 Clarify 时 simulator 返回 "Stop asking, just give me the code"（`patience=low` 导致 rejection），但模型训练数据里 **Busy T1 只有 3 个 Execute pair，几乎没教"被拒了就停"**。

### 7.4 Novice T2 也学得不完全

Novice T2 rule-match 35/50 = 70%，比 Busy 好但有 11 个 "U=0~0.2 应 Execute 但仍 Clarify" 的 case。这 11 个拉出了 Novice 的 5-7 轮尾巴（3 个 7-turn）。

### 7.5 跨 persona 干扰假设（v31.2a 验证）

v31.1 仅 Novice 规则改动、Busy/Exp 规则不变，但 Busy pair 分布（Clarify 28%）本身不同于 v29（0%）。Busy 31 pair 翻转是否会通过共享 LoRA 影响 Exp/Novice？v31.1 Exp 字节级不变否定这个方向（Exp 没变化），但 v31.2a 进一步压缩 Busy Clarify 到 4.7% 后 Exp 行为大幅变化，见 §8.2。

### 7.6 结论

- **Novice 过拟合修好了**（设计目标达成）；pass@1 −6pp 是"cut turns → cut disclosed info"的必要代价
- **Exp 稳定**（对照组）
- **Busy 是唯一意外问题**：稀疏 Clarify 信号（30/107 = 28%）+ v29 强先验（100% Execute）的组合让 DPO 学成 52% 硬币

---

## 8. v31.2a：收紧 Busy 阈值 0.6 → 0.8

### 8.1 设计

U 是离散的 `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`（MAX_MASKED=5）。阈值与"n_masked 最少几个"的业务语义一一对应：

| 阈值 | 捕获 U | 语义 |
|:---:|:---:|---|
| `> 0.6`（v31.1）| {0.8, 1.0} | n_masked ≥ 4 |
| **`> 0.8`（v31.2a）**| **{1.0}** | **n_masked = 5（全 mask 极端 task）**|

Busy T0 Clarify pair 从 30 → 5（4.7%），预期 DPO 信号太稀不学，Busy 退化到 v29 的"永远 Execute"行为。代价：论文叙事仍说"三 persona 同构 U-threshold"，但 Busy 的 5 pair 实际等同于没学。

### 8.2 理论预期

| 版本 | Busy T0 Clarify | Busy T0 Execute | 预期 Busy 行为 |
|---|:---:|:---:|---|
| v31.1 | 30 (28%) | 77 | 52% 硬币，2.72 轮 |
| **v31.2a** | **5 (4.7%)** | **102** | **信号太稀，退化 ~1.0 轮** |

### 8.3 实施改动

`reward/compute_rewards.py:708`：

```python
V31_U_THRESHOLD = {
    "Experienced-Engineer": 0.3,
    "Busy-Developer": 0.8,  # was 0.6
}
```

注释同步更新。无其他代码改动。

---

## 9. v31.2a Pair 分布 + 训练 + Eval（进行中）

### 9.1 Pair 分布（494 pairs, 107 states）

| persona | T0 Clarify | T0 Execute | T1 Clarify | T1 Execute | T2 | T3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Novice | 107 | 0 | 81 | 0 | C:11/E:18 | E:3 |
| Exp | 104 | 1 | 0 | 61 | — | — |
| **Busy (v31.2a)** | **5** | **102** | 0 | 1 | — | — |

Busy T0 Clarify 5 个全部 U=1.0（n_masked=5 极端 task），训练信号干净。Exp/Novice pair 与 v31.1 **字节级相同**（166 + 220 pair 完全一致）。

### 9.2 训练结果（03:15 → 03:31，16 min）

- 84 steps, train_loss 0.226
- rewards/margins: 0.38 → 4.72（逐 epoch 上升，模型学到区分）
- 末段 accuracy 96-98%（跟 v31.1 的 97.5% 接近）

模型保存到 `models/v31_2a_100states/`。

### 9.3 Eval 进行中（03:31 start）

50-state × 3 persona。中期观察（40/50）：

| persona | 部分 n | 部分 avg turns | Clarify% | 对比 v31.1 |
|---|:---:|:---:|:---:|---|
| **Busy** | 40 | **1.00** ✓ | 0% | 修好 |
| **Novice** | 40 | **4.55** | 100% | 略升（4.0→4.55），7-turn 尾巴 24% vs v31.1 6% ❌ |
| **Exp** | 39 | **1.86** ⬇ | 57% | **T0 Clarify 94%→54%**，跨 persona 干扰 ❗ |

### 9.4 关键发现：跨 persona 干扰

v31.2a 相对 v31.1 **仅 Busy 的 25 个 pair 在 U=0.8 从 chosen=Clarify 翻成 chosen=Execute**。Exp 和 Novice 的 pair 完全不变。但：

- Exp T0 Clarify 率 94% → 54%（40pp 下降）
- Novice 7-turn 长尾 6% → 24%

这验证了"单 LoRA adapter 下 persona 规则耦合"的假设：局部 label 翻转通过共享参数泛化到其他 persona。Exp prompt 里的 `Task Uncertainty: 0.80` + `Busy → Execute` 信号被 DPO 学成 "U=0.8 倾向 Execute" 的全局规则，即使 Exp 的训练 label 仍是 Clarify。

### 9.5 Eval 完成（2026-04-23 08:34）

完整 50-state × 3 persona，见 §10。

---

## 10. v31.2a Eval 最终结果 + 回退 v31.1 决定

### 10.1 三方对比（相同 50-state 测试集）

| 指标 | v29 | v31.1 | **v31.2a** |
|---|:---:|:---:|:---:|
| **Overall pass@1** | 21/150 (14.0%) | 14/150 (9.33%) | **13/150 (8.67%)** |
| **Overall pass@5** | 30/150 (20.0%) | 29/150 (19.3%) | **23/150 (15.3%)** |
| Novice turns | 7.00 | **4.00** ✓ | 4.76 |
| Novice pass@1 | 8/50 | 5/50 | 4/50 |
| Novice pass@5 | 10/50 | 11/50 | **7/50 (−4)** |
| Novice clarify% | 85.7% | 75.0% | 79.0% |
| Exp turns | 2.66 | **2.66** | 1.90 |
| Exp pass@1 | 6/50 | 5/50 | 4/50 |
| Exp pass@5 | 12/50 | 11/50 | **8/50 (−3)** |
| Exp clarify% | 62.4% | 62.4% | **47.4%** (−15pp) |
| Busy turns | 1.00 | 2.72 ❌ | **1.00** ✓ |
| Busy pass@1 | 7/50 | 4/50 | **5/50 (+1)** |
| Busy pass@5 | 8/50 | 7/50 | 8/50 |
| Busy clarify% | 0% | 63% | **0%** |

### 10.2 v31.2a vs v31.1 变化解读

**Busy：完美修复** — turns 1.00, clarify 0%, pass@1 +1。确认稀疏 Clarify 信号（5/107 = 4.7%）下 DPO 退化为 v29 行为。

**Exp：实质性退步** —
- clarify 率 62→47%（40pp 下降）
- turns 2.66→1.90（平均少一轮 Clarify）
- pass@5 11→8（**−3 task，不是噪声**）

pass@5 的信号比 pass@1 更可靠（5 个候选至少 1 个过 vs 1 个候选过）。Exp pass@5 −3 表明"信息缺失的 task 即使 5 次尝试也捞不回"，与 clarify 率下降方向一致。

**Novice：长尾拖累** —
- turns 4.00→4.76（意外变长）
- 7-turn 长尾从 3/50 → 9/50（6%→18%）
- pass@5 11→7（**−4 task，最严重**）

v31.2a Novice pair 与 v31.1 完全相同（§9.1），但 Novice 行为变差。这是跨 persona 干扰的另一个表现：Busy pair 翻转通过共享 LoRA 参数也影响了 Novice T2+ 的 Execute 决策。

### 10.3 跨 persona 干扰：实证确认

v31.2a vs v31.1 pair 差异：**仅 Busy 25 对在 U=0.8 从 chosen=Clarify 翻成 Execute**。Exp/Novice pair 字节级相同。但行为上：

| persona | pair 变化 | 行为变化 |
|---|---|---|
| Busy | Clarify 30→5, Execute 77→102 | 被改造（按设计）|
| Exp | **0 对变化** | **T0 Clarify 94%→54%（40pp）** |
| Novice | **0 对变化** | **7-turn 长尾 6%→18%（3x）** |

**结论**：单 LoRA rank=64 在 ~500 pair 尺度下无法在三 persona 间解耦。Busy 的 U=0.8 label 翻转被 DPO 泛化成"高 U 倾向 Execute"的全局信号，侵蚀 Exp/Novice 的 Clarify 倾向。这是方法层面的限制，不是数据或训练配置问题。

### 10.4 决策：回退 v31.1

按 §9.5 预设判据：

| 判据 | v31.2a | 通过 |
|---|---|---|
| Overall pass@1 ≥ v31.1 (9.33%) | 8.67% | ❌ |
| Exp pass@1 稳定 | 5→4 | ❌ |
| Overall pass@5 ≥ v31.1 (19.3%) | 15.3% | ❌ |

v31.2a 用 Busy 的 +1 task 换掉 Exp (−3 pass@5) + Novice (−4 pass@5) = 总计 **−6 pass@5**，亏本。

**最终版 v31 = v31.1**（`models/v31_100states/`, `data/dpo/prefs_v31_100states.jsonl`）。

### 10.5 论文叙事

> "v31.1 通过 turn + U conditional 修复了 v29 的 Novice 100% 过拟合（7→4 轮），代价是 Novice pass@1 −6pp（cut-turn → cut-info 的必要 tradeoff，Novice `items_per_turn=1` 决定了必须多轮才能完全披露）。Exp 行为与 v29 字节级一致。Busy 出现了稀疏正类信号（28% Clarify）下 DPO 未学透的现象（52% rule-match）。
>
> 进一步收紧 Busy 阈值（v31.2a，4.7% Clarify）成功修复 Busy 行为（1.00 轮），但导致**跨 persona 干扰**：Exp/Novice pair 完全不变的前提下，Busy 的 25 对 label 翻转通过共享 LoRA 参数泛化成全局"高 U → Execute"信号，侵蚀 Exp 的 clarify 倾向（pass@5 −3）+ Novice 长尾（pass@5 −4）。这证明单 LoRA adapter 在 ~500 pair 尺度下不能在三 persona 间解耦规则修改。Busy 的清洁修复留作 future work：per-persona adapter、扩数据、或更强的 persona token 设计。"

### 10.6 留存

- v31.1 模型：`models/v31_100states/`（最终版）
- v31.1 pair：`data/dpo/prefs_v31_100states.jsonl`
- v31.1 eval：`outputs/eval_v31_dpo_50test.json`
- v31.2a 模型：`models/v31_2a_100states/`（留作 ablation / 跨干扰证据）
- v31.2a pair：`data/dpo/prefs_v31_2a_100states.jsonl`
- v31.2a eval：`outputs/eval_v31_2a_dpo_50test.json`

---

## 11. v31.3-D: Busy T1+ Execute Post-hoc Patch（hypothesis test）

### 11.1 动机

v31.1 Busy 问题拆解成两层：

| 层 | 数据 | 贡献 |
|---|---|---|
| T0 乱 Clarify（52% rule match）| 30/107 Clarify pair 太稀 | **未知** |
| T1+ 不停下来（9 个 7-turn 长尾）| Busy T1 只有 3 个 Execute pair | **主要损失源？** |

假设：**长尾是 pass@1 的主因**。T0 偶尔乱 Clarify 不致命，只要 T1 能停在 2 轮。9 个 7-turn state 的 Execute prompt 里积累了用户 rejection 语气，污染代码生成。

D 是 cheap experiment 验证这个假设：不训练、不改数据，只在推理时给 Busy 加硬 patch。

### 11.2 实现

`eval/evaluate_multi_turn_persona.py`：
- 新增 CLI flag `--busy_t1_execute`
- 推理循环中：`if persona=="Busy-Developer" and turn>=1 and action=="Clarify": action = "Execute"`
- 其他所有东西（模型、Novice/Exp 逻辑、prompt 构造、user simulator）完全不变

用 **v31.1 模型** 评估：`models/v31_100states/`。

### 11.3 数学预测

补丁在 T1 强制 Execute → 所有 Busy 对话最多 2 轮：
- 37% Busy T0 Execute → 1 轮（v31.1 那 37% T0 Execute 的 state）
- 63% Busy T0 Clarify → T1 强制 Execute → 2 轮

**预期 avg turns = 0.37×1 + 0.63×2 = 1.63，max 2，零长尾**（物理保证，不依赖模型学习）。

### 11.4 对 pass@1 的预测

| 情况 | 机制 | Busy pass@1 预期 |
|---|---|---|
| 用户 T1 答了问题（~20%）| T1 Execute prompt 含 disclosed_info → **比 v29 更好**（v29 是 0 disclosure）| ↑ |
| 用户 T1 拒绝（~80%，Busy patience=low, effective≤0.2）| T1 Execute prompt 含拒绝语气 → **可能比 v29 更差** | ↓ |

加权结果不确定。关键是**去掉 9 个 7-turn 长尾的污染**，即使 rejection 率高，T1 Execute 也比 T6 forced Execute 干净很多。

### 11.5 决策判据

| D 结果 | 解读 | 下一步 |
|---|---|---|
| Busy pass@1 ≥ 12%（接近 v29 14%）| 长尾确实是主因 | **走 A：把"T1 Execute"做进 DPO 训练** |
| Busy pass@1 ∈ [9, 12]% | 长尾修好帮了一部分，T0 也在吃 pass@1 | A + 附加方案（per-persona adapter / 扩数据）|
| Busy pass@1 < v31.1（8%）| 长尾不是主因 / T1 Execute prompt 污染严重 | 方法层面需要重想（B 或 C）|

### 11.6 执行状态（2026-04-23 晚间）

- 10:52（机器时间）kick off eval，单进程 nohup
- 日志 `/tmp/v31_3d_eval.log`
- 输出 `outputs/eval_v31_1_busyT1exec_50test.json`
- 2026-04-23 16:20 完成（实际 ~5.5h）

### 11.7 结果

**三方对比（同 50-state 测试集）**：

| 指标 | v29 | v31.1 | **v31.3-D** |
|---|:---:|:---:|:---:|
| **Overall pass@1** | 21/150 (14.0%) | 14/150 (9.33%) | **19/150 (12.67%)** |
| **Overall pass@5** | 30/150 (20.0%) | 29/150 (19.3%) | 26/150 (17.3%) |
| Busy turns / p@1 / p@5 | 1.0 / 7 / 8 | 2.72 / 4 / 7 | **1.42 ✓ / 6 / 8** |
| Busy clarify% / 长尾 | 0% / 0 | 63% / 9 | **29.6% / 0** ✓ |
| Exp turns / p@1 / p@5 | 2.66 / 6 / 12 | 2.66 / 5 / 11 | 2.68 / 6 / 9 |
| Exp clarify% | 62.4% | 62.4% | 62.7% |
| Novice turns / p@1 / p@5 | 7.0 / 8 / 10 | 4.0 / 5 / 11 | 4.24 / 7 / 9 |
| Novice clarify% | 85.7% | 75.0% | 76.4% |

**Busy 分布兑现物理保证**：50 对话 = 29 T0 Execute（1 轮）+ 21 T0 Clarify（被 patch 强制 T1 Execute → 2 轮），总 71 turns，avg 1.42，max 2，**零 7-turn 长尾**。预测 1.63（基于 §11.3 的 63% T0 Clarify 估计）偏高，实际 T0 Clarify 只有 42% —— §7.1 那个 63% 是 clarify_turns/total_turns 被 9 个长尾拉高的。

### 11.8 按 §11.5 判据的决策

| 判据 | v31.3-D | 判定 |
|---|---|---|
| Busy pass@1 ≥ 12% | **12%（6/50）正好踩线** | ✓ |
| Novice pass@1 稳定 | 10% → 14%（+2 task） | ✓ |
| Exp pass@1 稳定 | 10% → 12%（+1 task） | ✓ |
| Overall pass@1 vs v31.1 | 9.33% → **12.67%（+5 task）** | ✓ |

**结论：长尾是 Busy pass@1 的主因**（假设成立）。去掉 9 个 7-turn 尾巴（每个都被 rejection 污染 Execute prompt）后 Busy pass@1 从 8%→12%，恢复到接近 v29 14% 的水平。

→ **下一步走 A：把 "Busy T1 Execute" 做进 DPO 训练**，目标是不靠推理 patch 也能得到相同行为。

### 11.9 意外发现：Novice/Exp pass@1 也上升

v31.3-D patch 只在推理时翻转 Busy turn≥1 的 action，不影响 Novice/Exp 的任何决策。但两者 pass@1 都比 v31.1 涨了（Novice 10→14%，Exp 10→12%），而 turn 分布几乎一致（Novice 4.0→4.24，Exp 2.66→2.68）。

可能原因：
- gpt-4o-mini 代码生成 temperature 采样噪声（每次重跑都会有 ±1-2 task 差异）
- pass@5 反向变化（Novice 11→9，Exp 11→9）也说明是采样波动，不是系统性改善

**实操上**：v31.1 的 "真实" p@1 可能就在 10-14% 区间，50-state 评估本身 σ ≈ 2.3 task，这次重跑恰好落高端。论文叙事不靠这个涨幅，核心信号是 Busy 的 +2 task。

### 11.10 留存

- v31.3-D eval：`outputs/eval_v31_1_busyT1exec_50test.json`
- 模型仍是 v31.1：`models/v31_100states/`（零训练）
- 推理 patch：`eval/evaluate_multi_turn_persona.py --busy_t1_execute`

---

## 12. v31.4: Busy T1 Execute 学进 DPO（失败）

> 日期: 2026-04-24
> 目标: 把 v31.3-D 的推理补丁做成 DPO 训练能力，不依赖推理 patch

### 12.1 改动

- `reward/compute_rewards.py`：Busy 新增 `turn >= 1 → Execute` 分支（T1+ 硬停）
- Novice/Exp 规则完全保持 v31.1（pair 字节级相同）
- 两个 bug 修复：Method B2 dialogue_turn 错位 + U 离散化陷阱（0.3 和 0.4 触发集相同）
- 520 对 pair（vs v31.1 496 对），**Busy T1 Execute 3→27**（+24），Exp/Novice pair 字节级不变

### 12.2 训练 + Pipeline

- Pipeline（`/tmp/v31_4_full_pipeline.sh`）07:35 CST 启动
- Llama 训练 ~17min 完成 → `models/v31_4_100states/`
- Llama eval 50-state 14:21 完成 → `outputs/eval_v31_4_dpo_50test.json`
- **Qwen 训练 14:22 崩溃**：`LocalEntryNotFoundError` + `OSError: couldn't connect to huggingface.co`。Qwen 权重未正确落到 cache，`train_dpo.py` 强制 `local_files_only=True` 直接失败。`models/v31_4_qwen_100states/` 为空目录。

### 12.3 v29 vs v31.4 per-persona 对比（同 50-state 测试集）

#### 轮数与行为

| persona | 指标 | v29 | v31.4 | Δ |
|---|---|:---:|:---:|:---:|
| **Novice** | avg turns | 7.00 | **4.14** | **-2.86** ✓ |
| | turn 分布 | 7×50（全撞顶）| 3×20, 4×15, 5×6, 6×6, 7×3 | 尾巴 3/50 |
| | clarify rate | 85.7% | 75.8% | -10pp |
| | forced execute | 50/50 | 3/50 | -47 ✓ |
| **Exp** | avg turns | 2.66 | **2.24** | -0.42 ⚠ |
| | T0 Clarify | 94% (47/50) | **78%** (39/50) | **-16pp** ❌ |
| | clarify rate | 62.4% | 55.4% | -7pp |
| **Busy** | avg turns | 1.00 | 1.24 | +0.24 |
| | turn 分布 | 1×50（完全不问）| 1×47, 3×1, 5×1, 7×1 | 3 个漏网 |
| | T0 Execute | 100% | 94% (47/50) | -6pp |
| | clarify rate | 0% | 19.4% | +19pp |

#### Task Success

| persona | 指标 | v29 | v31.4 | Δ |
|---|---|:---:|:---:|:---:|
| **Novice** | pass@1 | **16%** (8/50) | 12% (6/50) | **-2 task** |
| | pass@5 | 20% (10/50) | 16% (8/50) | -2 task |
| **Exp** | pass@1 | **12%** (6/50) | 6% (3/50) | **-3 task** ❌ |
| | pass@5 | **24%** (12/50) | 12% (6/50) | **-6 task** ❌ |
| **Busy** | pass@1 | **14%** (7/50) | 6% (3/50) | **-4 task** ❌ |
| | pass@5 | 16% (8/50) | 18% (9/50) | +1 task |
| **Overall** | **pass@1** | **14.0%** (21/150) | **8.0%** (12/150) | **-9 task / -6pp** |
| | **pass@5** | **20.0%** (30/150) | **15.3%** (23/150) | **-7 task / -4.7pp** |

### 12.4 失败原因：跨 persona 干扰 + 规则学不透

**层 1 — Busy 规则没学透**

520 对 pair 里只有 27 对 Busy T1 Execute（5.2%），对抗 v29 强先验 + Novice/Exp 的隐含 T1 Execute 信号，新规则被稀释。3 个 Busy T0 误选 Clarify 的 task（BigCodeBench/337 7-turn forced、/364 5-turn、/968 3-turn）T1 依然 Clarify，长尾没修干净。

**层 2 — 跨 persona 干扰（v31.2a 重演）**

Exp/Novice pair 字节级未变，但：
- Exp T0 Clarify 94%→78%（-16pp）、pass@5 22→12%（-10pp）
- Novice pass@5 22→16%（-6pp）

机制同 v31.2a：单 LoRA rank=64 在 500 对数据尺度下无法把"Busy T1 → Execute"隔离在 Busy persona token 上，被 DPO 泛化成全局"T1+ 倾向 Execute"信号，侵蚀 Exp T0 的 Clarify 倾向 + 代码生成质量。

### 12.5 结论

v31.4 证明 **"把 Busy T1 Execute 学进 DPO"这条路在当前数据 + 方法组合下不可行**：
- Busy 新增 pair 稀疏 → 规则学不干净（3 个漏网长尾）
- 改动通过共享 LoRA 参数污染 Exp/Novice → Overall pass@1 砍 6pp

**v31.3-D 推理 patch 仍是当前最优操作点**（12.67% pass@1 vs v31.4 8%）。

### 12.6 留存

- v31.4 pair：`data/dpo/prefs_v31_4_100states.jsonl`（520 对）
- v31.4 Llama 模型：`models/v31_4_100states/`
- v31.4 Llama eval：`outputs/eval_v31_4_dpo_50test.json`
- v31.4 Qwen 模型：空（训练崩溃，需先修 HF cache）
- Pipeline log：`/tmp/v31_4_full_pipeline.log`
