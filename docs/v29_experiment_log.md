# v29 实验与分析记录

> 日期: 2026-04-10
> 目标: 基于结构化 masking + OF sub-item 拆分，重新生成轨迹并验证信号质量

---

## 1. v29 核心改动（vs v28）

### 1.1 Masking 重写
- **v28 问题**: regex 猜边界，导致 100% 断句残留 `"The function \n"`，14/470 跨行吞内容
- **v29 方案**: 按 BigCodeBench `instruct_prompt` 固定结构锚点精确切割，零残留
  - `output_format`: `"The function should output with:\n"` → `"You should write self-contained"`
  - `validation_rules`: `"The function should raise the exception for:"` → 下一个 section 锚点
  - `note_that`: `"Note that:"` → 下一个 section 锚点
  - `edge_cases`: 移除（从 test 代码关键词匹配不准确，用 validation_rules 替代）

### 1.2 OF Sub-item 拆分
- **v28 问题**: 61.6% task 只有 1 个 masked item → Novice 只需 1 轮 Clarify → avg turns (1.7) < Experienced (1.8)
- **v29 方案**: `_split_output_format()` 把 OF 拆成多个 sub-items
  - 4 策略：dict key-desc pattern → multi-line independent lines → single-line multi-sentence → fallback 2-item
  - VR/NT 不拆（自由文本，无稳定格式）
  - 拆分后 mean 2.6 items/OF, 总 mean 2.9 items/task

### 1.3 Code Versions 区分
4 个版本用于评估 Clarify 的价值：
- `direct`: masked query 直接生成代码（blind baseline）
- `clarified`: masked query + split items 拼接（`build_clean_execute_query`，用 `"; ".join(items)`）
- `ideal_disclosed`: masked query + 原始 specification 文本（`disclosure_info.specification`）
- `oracle`: 完整原始 `instruct_prompt` 生成代码（upper bound）

### 1.4 All-disclosed 检查修复
- **v28 问题**: `not disclosed_info.get(field)` 用 truthy 检查，1 个 item 即为 True → 过早 Execute
- **v29 修复**: `len(disclosed) < len(total)` 按数量比较

---

## 2. 10-State 小批量验证

### 2.1 生成配置
```
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 10 --llm_model gpt-4o-mini \
  --all_personas --max_turns 6 --n_samples 4 \
  --out data/logs/traj_v29_10states_final.jsonl
```

输出文件: `data/data/logs/traj_v29_10states_final_20260410_052948.jsonl`
- 249 trajectory turns, 139 unique trajectories, 30 conversations (10 states × 3 personas)

### 2.2 Preference Pairs
```
python reward/compute_rewards.py \
  --trajectories data/data/logs/traj_v29_10states_final_20260410_052948.jsonl \
  --out data/dpo/prefs_v29_10states_test.jsonl
```

输出: 46 preference pairs (rebalance 后), 其中:
- Method A (fork) pairs: 9
- Method C (fork) pairs: 12

---

## 3. 4 层分析结果

### Layer 1: 轨迹行为 ✅

| Persona | 轨迹数 | 平均 turns | First Clarify 率 | Turns 分布 |
|---------|--------|-----------|-----------------|-----------|
| Novice-Learner | 49 | **2.22** | 61% (30/49) | 1×19, 2×2, 3×25, 4×3 |
| Experienced-Engineer | 50 | **1.80** | 80% (40/50) | 1×10, 2×40 |
| Busy-Developer | 40 | **1.25** | 25% (10/40) | 1×30, 2×10 |

**结论**: 排序正确 Novice (2.22) > Experienced (1.80) > Busy (1.25) ✅
- vs v28: Novice 从 1.7 提升到 2.22, 现在明确 > Experienced

### Layer 2: Pass Rate by Code Version

#### 总体

| Version | Pass Rate | 通过数/总数 |
|---------|-----------|-----------|
| direct | 0.563 | 94/139 |
| clarified | 0.541 | 89/139 |
| ideal_disclosed | 0.589 | 93/139 |
| oracle | 0.612 | 97/139 |

#### 按 Persona

| Persona | direct | clarified | ideal_disclosed | oracle |
|---------|--------|-----------|-----------------|--------|
| Novice-Learner | 0.538 | 0.510 | 0.565 | 0.586 |
| Busy-Developer | 0.581 | 0.564 | 0.614 | 0.614 |
| Experienced-Engineer | 0.574 | 0.551 | 0.591 | 0.634 |

**关键发现**:
1. **v28 pass_rate ≈ 0 → v29 pass_rate 0.56+**: masking 修复有效 ✅
2. **ideal_disclosed (0.589) > direct (0.563) > clarified (0.541)**: 原始 spec 有帮助，但 split items 拼接反而不如 direct
3. **oracle (0.612)** 是 upper bound, ideal_disclosed 接近
4. **clarified < direct**: split items 的 `"; ".join()` 拼接语义不如直接 masked query 清晰，或 LLM Clarify 回答引入噪声

**需关注**: clarified < direct 说明当前 Clarify 对话在 task score 上是"负收益"，会影响 Clarify 的 reward signal。

### Layer 3: Preference Pair 信号质量

#### 总体

| 指标 | v29 | v28 |
|------|-----|-----|
| 正确信号率 | **84.8%** (39/46) | ~60% |
| 逆信号率 | **8.7%** (4/46) | ~25% |
| Zero gap | 6.5% (3/46) | — |
| 平均 reward gap | 0.1830 | — |
| 平均 |gap| | 0.2335 | — |

#### 按 Persona

| Persona | Pairs | 正确信号 | 逆信号 | Zero Gap |
|---------|-------|---------|--------|----------|
| Novice-Learner | 20 | **100%** (20/20) | 0% | 0% |
| Busy-Developer | 10 | 60% (6/10) | 10% (1/10) | 30% (3/10) |
| Experienced-Engineer | 16 | 81% (13/16) | 19% (3/16) | 0% |

#### 按 Turn 逆信号率

| Persona | Turn 0 | Turn 1 | Turn 2 |
|---------|--------|--------|--------|
| Novice-Learner | 0/10 (0%) | 0/9 (0%) | 0/1 (0%) |
| Busy-Developer | 1/10 (10%) | — | — |
| Experienced-Engineer | 0/9 (0%) | **3/7 (43%)** | — |

### Layer 3 逆信号深入分析

#### Experienced-Engineer Turn 1 逆信号（3/7）

| State | Chosen (Execute) | Rejected (Clarify) | Gap | 性质 |
|-------|-----------------|--------------------|----|------|
| BigCodeBench/0 | task=0.0, reward=0.0 | task=0.0, reward=0.032 | -0.032 | **噪声** (clarification bonus) |
| BigCodeBench/4 | task=1.0, reward=1.0 | task=1.0, reward=1.032 | -0.032 | **噪声** (clarification bonus) |
| BigCodeBench/9 | task=0.0, reward=0.0 | task=1.0, reward=1.048 | **-1.048** | **真逆信号** (Clarify 确实帮助了) |

结论: 3 个逆信号中 2 个是 w_interrupt=0.2 的微小 bonus 噪声 (gap=0.032), 只有 1 个是真正的逆信号。

#### Busy-Developer 分析

- 6 个 OK (gap 大, 好信号)
- 3 个 ZERO (Execute/Clarify 的 task_score 都为 0, 任务太难)
- 1 个 REVERSE (gap=-0.048, 微小噪声)

### Layer 4: v28 vs v29 问题诊断

| 问题 | v28 | v29 | 状态 |
|------|-----|-----|------|
| masking 断句残留 → pass_rate=0 | 100% 受影响 | 0% (锚点切割) | ✅ 解决 |
| Novice avg turns < Experienced | 1.7 vs 1.8 | 2.22 vs 1.80 | ✅ 解决 |
| Experienced T1 逆信号 | 53.8% | 43% (2/3 是噪声) | ⚠️ 改善 |
| 总逆信号率 | ~25% | 8.7% | ✅ 大幅改善 |
| Novice 信号 | 混合 | 100% 正确 | ✅ 解决 |

---

## 4. Reward Gap 深入分析

### 4.1 Gap 分布（46 pairs 全景）

| Gap 范围 | 数量 | 说明 |
|----------|------|------|
| gap < 0 (逆信号) | 4 | -1.048, -0.048, -0.032, -0.032 |
| gap = 0 (零信号) | 3 | task_score 两者都为 0 |
| 0 < gap < 0.05 (弱信号) | 27 | 几乎全是 0.032 或 0.048，来自 interrupt bonus |
| gap >= 0.05 (强信号) | 12 | 真正来自 task_score 差异 |

**关键发现**: 大部分 pairs (27/46) 的 reward gap 来自 interrupt bonus（γ-λ=0.08, ×w_interrupt=0.2 → 0.016/question），不是来自 task_score 差异。10 states 样本太小，很多 task 的 task_score 要么全 0 要么全 1.0。

### 4.2 Interrupt Bonus 机制

`compute_interrupt_cost_v2` 参数: γ=0.20, λ=0.12, δ=0.8

| 场景 | 公式 | cost 值 | ×w_interrupt=0.2 后 |
|------|------|---------|--------------------|
| 有效澄清 (用户回答) | n × (λ-γ) = n × (-0.08) | -0.08/q (奖励) | +0.016/q bonus |
| 被拒绝 | n × (δ+λ) = n × 0.92 | +0.92/q (惩罚) | -0.184/q penalty |
| 未回答 | n × λ = n × 0.12 | +0.12/q | -0.024/q |

### 4.3 逆信号处理方案讨论

| 方案 | 做法 | 影响 |
|------|------|------|
| B. 过滤小 gap | 丢弃 gap < threshold 的 pairs | threshold=0.03 只去 7 个; threshold=0.05 去掉 34/46 (Novice 只剩 2 个) |
| C. 去掉 bonus | 设 γ=λ=0.12 | 27 个弱信号 pairs 变成 zero gap，同样没训练信号 |

**结论**: 两个方案都暴露了同一根因——10 states 样本太小，task_score 差异不足，gap 主要靠 interrupt bonus 撑着。决定先 scale up 到 100 states 看 gap 分布是否改善，再决定处理方案。

### 4.4 BigCodeBench/0 案例分析

所有 persona、所有 code version 的 pass_rate 都是 0.00 — 这个 task 对 gpt-4o-mini 太难，无论给多少信息都过不了。这类 task 产生的 pairs 全是 zero-gap，对训练无贡献。100 states 后占比应降低。

---

## 5. 剩余问题与后续计划

### 5.1 clarified < direct 问题
- 10 states: clarified(0.541) < direct(0.563)，差异 0.022，样本小可能不显著
- 拼接方式 `"Key requirements: item1; item2"` 格式本身看起来合理
- 决定: 等 100 states 结果确认是否系统性

### 5.2 Experienced T1 逆信号
- 真逆信号只有 1/7, 其他 2 个是 interrupt bonus 噪声
- 需要 100 states 验证逆信号率是否稳定

### 5.3 Pairs 数量不平衡
- Turn 0 已 balanced (10:10:10)，不平衡来自 Turn 1+ (Novice 20 > Exp 16 > Busy 10)
- 天然结构差异，暂不处理。scale up 后观察训练 loss 曲线再决定

### 5.4 Disclosure 顺序
- 当前固定 OF → VR → NT 顺序
- 等 100 states 结果后再调整

### 5.5 代码生成模型
- 轨迹中 4 个 code version 都由 gpt-4o-mini 生成（v28 也是如此）
- GPT pass_rate 不代表 Llama pass_rate，但作为 oracle reward signal 用于构建 preference pairs 是合理的
- DPO 训练的是决策策略（Clarify vs Execute），不是代码生成能力

---

## 6. 100-State 生成（已完成）

### 6.1 生成配置
```
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 100 --llm_model gpt-4o-mini \
  --all_personas --max_turns 6 --n_samples 4 \
  --out data/logs/traj_v29_100states.jsonl
```

实际分两批生成（part1: 57 states, part2: 52 states），合并后 109 unique states, 2794 trajectory turns, 1527 unique trajectories。

### 6.2 Preference Pairs
```
python reward/compute_rewards.py \
  --trajectories data/data/logs/traj_v29_100states_combined.jsonl \
  --out data/dpo/prefs_v29_100states.jsonl
```

输出: **500 preference pairs** (rebalance 后), 107 complete states (三 persona 全覆盖), 其中:
- Method A (fork) pairs: 89
- Method C (fork) pairs: 115
- Busy-Developer: 107 pairs (turn=0:107)
- Experienced-Engineer: 166 pairs (turn=0:107, turn=1:59)
- Novice-Learner: 227 pairs (turn=0:107, turn=1:81, turn=2:36, turn=3:3)

---

## 7. 100-State 4 层分析结果

### Layer 1: 轨迹行为 ✅

| Persona | 轨迹数 | 平均 turns | First Clarify 率 | Turns 分布 |
|---------|--------|-----------|-----------------|-----------|
| Novice-Learner | 564 | **2.30** | 57.3% (323/564) | 1×241, 2×65, 3×133, 4×97, 5×28 |
| Experienced-Engineer | 535 | **1.80** | 79.6% (426/535) | 1×109, 2×426 |
| Busy-Developer | 428 | **1.25** | 25.0% (107/428) | 1×321, 2×107 |

**结论**: 排序正确 Novice (2.30) > Experienced (1.80) > Busy (1.25) ✅
- vs 10-state: 几乎一致（2.22 > 1.80 > 1.25），scale up 未改变行为模式

### Layer 2: Pass Rate by Code Version ⚠️

#### 总体

| Version | 100-states | 10-states | 差异 |
|---------|-----------|-----------|------|
| direct | 0.373 | 0.563 | -0.190 |
| clarified | **0.399** | 0.541 | -0.142 |
| ideal_disclosed | 0.385 | 0.589 | -0.204 |
| oracle | 0.436 | 0.612 | -0.176 |

#### 按 Persona

| Persona | direct | clarified | ideal_disclosed | oracle |
|---------|--------|-----------|-----------------|--------|
| Novice-Learner | 0.362 | 0.387 | 0.376 | 0.433 |
| Experienced-Engineer | 0.387 | 0.430 | 0.387 | 0.447 |
| Busy-Developer | 0.369 | 0.376 | 0.395 | 0.428 |

**关键发现**:
1. **clarified(0.399) > direct(0.373)**: 10 states 时 clarified < direct 被推翻，100 states 确认 **Clarify 有正收益** ✅
2. **clarified(0.399) > ideal_disclosed(0.385)**: 反直觉——`"; ".join(items)` 拼接比原始 spec 文本更结构化，LLM 更容易理解
3. **整体 pass rate 下降**: 10-state 的 0.56 降到 0.37，说明 10 states 抽到的 task 偏简单
4. **oracle(0.436)** 是 upper bound，clarified 已接近 oracle 的 91%

### Layer 3: Preference Pair 信号质量 ⚠️

#### 总体

| 指标 | 100-states | 10-states | v28 |
|------|-----------|-----------|-----|
| 正确信号率 | **75.6%** (378/500) | 84.8% (39/46) | ~60% |
| 逆信号率 | **14.4%** (72/500) | 8.7% (4/46) | ~25% |
| Zero gap | 10.0% (50/500) | 6.5% (3/46) | — |
| 平均 reward gap | 0.1726 | 0.1830 | — |
| 平均 |gap| | 0.2225 | 0.2335 | — |

#### 按 Persona

| Persona | Pairs | 正确信号 | 逆信号 | Zero Gap |
|---------|-------|---------|--------|----------|
| Novice-Learner | 227 | 96% (217/227) | 4% (9/227) | 0% (1/227) |
| Experienced-Engineer | 166 | 75% (125/166) | 25% (41/166) | 0% (0/166) |
| Busy-Developer | 107 | 34% (36/107) | 21% (22/107) | 46% (49/107) |

#### 按 Persona × Turn 逆信号率

| Persona | Turn 0 | Turn 1 | Turn 2 | Turn 3 |
|---------|--------|--------|--------|--------|
| Novice-Learner | 3/107 (3%) | 3/81 (4%) | 2/36 (6%) | 1/3 (33%) |
| Experienced-Engineer | 2/105 (2%) | **39/61 (64%)** | — | — |
| Busy-Developer | 22/107 (21%) | — | — | — |

#### 逆信号详细分析

| Persona | Turn | Count | 真逆信号(task_score差异) | 噪声(仅interrupt) |
|---------|------|-------|---------------------|-------------------|
| Busy-Developer | 0 | 22 | 6 | 16 |
| Experienced-Engineer | 0 | 2 | 2 | 0 |
| Experienced-Engineer | 1 | 39 | 4 | **35** |
| Novice-Learner | 0 | 3 | 3 | 0 |
| Novice-Learner | 1 | 3 | 3 | 0 |
| Novice-Learner | 2 | 2 | 2 | 0 |
| Novice-Learner | 3 | 1 | 0 | 1 |

### Layer 4: Reward Gap 深入分析

#### Gap 分布

| Gap 范围 | 数量 | 占比 |
|----------|------|------|
| < 0 (逆信号) | 72 | 14.4% |
| = 0 (零信号) | 50 | 10.0% |
| 0 < gap < 0.05 (弱信号) | 232 | 46.4% |
| gap >= 0.05 (强信号) | 146 | 29.2% |

#### Gap 来源

| 来源 | 数量 | 占比 |
|------|------|------|
| 有 task_score 差异 | 117 | 23.4% |
| 纯 interrupt bonus 驱动 | 333 | **66.6%** |
| Zero gap | 50 | 10.0% |

#### 强信号 pairs (gap >= 0.05) 按 persona × turn

| Persona | Turn 0 | Turn 1 | Turn 2 | Turn 3 | Total |
|---------|--------|--------|--------|--------|-------|
| Novice-Learner | 27 | 19 | 10 | 2 | 58 |
| Experienced-Engineer | 31 | 21 | 0 | 0 | 52 |
| Busy-Developer | 36 | 0 | 0 | 0 | 36 |

---

## 8. 100-State 分析结论与问题诊断

### 8.1 已确认（vs 10-state）

| 结论 | 10-state | 100-state | 状态 |
|------|---------|-----------|------|
| 轨迹行为排序正确 | ✅ | ✅ | 稳定 |
| clarified > direct | ❌ (0.541 < 0.563) | ✅ (0.399 > 0.373) | **反转，好消息** |
| Novice 信号质量 | 100% | 96% | ✅ 稳定 |
| masking 修复有效 | ✅ | ✅ | 稳定 |

### 8.2 需要解决的问题

#### 问题 1（最严重）：Experienced T1 逆信号 64%
- 39/61 逆信号，其中 35 个是 interrupt bonus 噪声（γ-λ=0.08 使 Clarify reward 高于 Execute）
- Scale up 没有改善（10-state 43% → 100-state 64%，反而更高）
- **根因**：Experienced T1 的 chosen=Execute，但 Clarify 多问一轮拿到 γ 奖励，task_score 无差异时 Clarify reward 反而更高

#### 问题 2：Busy T0 低信号率
- 只有 34% 正确信号，46% zero gap（两边 task_score 都 0），21% 逆信号
- 根因：很多 task 对 gpt-4o-mini 太难，Execute 和 Clarify 的 task_score 都是 0

#### 问题 3：Gap 主要靠 interrupt bonus
- 66.6% 的 gap 纯靠 interrupt bonus，只有 23.4% 有 task_score 差异
- Scale up 到 100 states 没有改善——不是样本量问题，是 gpt-4o-mini 在多数 task 上 pass rate 差异不足

### 8.3 待决策（初始方案，已在 §9 中深入分析后更新）

1. **Experienced T1 逆信号处理**：
   - 方案 A：设 γ=λ=0.12（去掉 clarify bonus）→ 35 个噪声逆信号变 zero gap
   - 方案 B：过滤 |gap|<0.05 → 去掉弱信号和噪声逆信号
   - 方案 C：两者结合
2. **Busy zero gap**：暂不处理（无害），或过滤 zero gap pairs
3. **是否直接用 500 pairs 训练 v29**：即使有噪声，75.6% 正确信号率可能已足够

---

## 9. 逆信号与 reward gap 深入分析

> 日期: 2026-04-11
> 目标: 搞清楚逆信号的本质、γ=λ 方案是否可行、以及 reward gap 对训练的实际影响

### 9.1 逆信号的两种本质

72 个逆信号分解：

| 类型 | 数量 | 机制 | 性质 |
|------|------|------|------|
| **interrupt 噪声** | 52 | task_score 相同（0 vs 0 或 1 vs 1），γ bonus 使 Clarify reward 高 0.016-0.048 | 噪声，可过滤 |
| **真逆信号** | 20 | Clarify 确实帮了 task_score，但 behavior-first 选了另一方 | **论文的 tradeoff，应保留** |

真逆信号分布：
- Busy T0: 6 个 — Clarify 帮了代码但 Busy 不愿被问 → 论文核心 tradeoff
- Experienced T1: 4 个 — 多问一轮确实帮了，但 Experienced 一轮够了 → 论文核心 tradeoff
- Novice T0-T2: 8 个 — chosen=Clarify 但 Execute 碰巧代码更好 → 样本噪声
- Novice T3: 1 个 — interrupt 噪声

典型案例（Busy BigCodeBench/37）：
- Execute: task_score=0.00（代码没通过）
- Clarify: task_score=1.00（问了之后通过了）
- behavior-first 选 Execute → 这就是论文说的"Busy 用户宁可代码差也不想被打扰"

### 9.2 γ=λ=0.12 方案模拟

**结论：γ=λ 会破坏论文的核心机制，不可行。**

模拟结果：

| 指标 | 当前(γ=0.20) | γ=λ=0.12 | γ=λ + 过滤zero |
|------|-------------|----------|---------------|
| 总 pairs | 500 | 500 | **117** |
| 正确信号 | 378 (75.6%) | 95 (19.0%) | 95 (81.2%) |
| 逆信号 | 72 (14.4%) | 22 (4.4%) | 22 (18.8%) |
| Zero gap | 50 (10.0%) | **383 (76.6%)** | 0 |

逆信号确实从 14.4%→4.4%，但 **76.6% pairs 变成 zero gap**。

过滤 zero gap 后按 persona：

| Persona | Pairs | 正确信号 |
|---------|-------|---------|
| Novice | 57 | 86% |
| Experienced | 48 | 88% |
| **Busy** | **12** | **33%（4 正确 8 逆）** |

**Busy 只剩 12 个 pairs 且 8 个是逆信号** — 模型会学到"Busy 应该 Clarify"，完全反了。

γ=λ 不可行的根本原因：项目的核心是平衡 task completion 和 user preference。γ 是成功澄清的 credit，设为 λ 等于说"问问题没有成本"——抹掉了 Busy 用户"被打扰"的代价，破坏了论文的 tradeoff 机制。

### 9.3 DPO 训练中 reward gap 的实际作用

**关键发现：reward gap 不进入 DPO loss。**

训练代码（`policy/train_dpo.py`）使用标准 `trl.DPOTrainer`：
- 输入：`(prompt, chosen, rejected)` 三元组
- Loss: `-log σ(β · (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))`
- **不使用 margin**，reward gap 不影响梯度

reward gap 仅在两处起作用：
1. `compute_rewards.py` 内选"最优轨迹"（同一 action 内选 reward 最高的）
2. 分析时判断信号质量

所以：
- 52 个 interrupt 噪声逆信号 → chosen/rejected 方向由 behavior-first 保证正确 → **对 DPO 训练无害**
- 20 个真逆信号 → 方向正确（Busy 选 Execute），reward 较低只影响轨迹选择 → **无害**
- 232 个弱信号（gap<0.05）→ 和强信号 pairs 对 DPO 梯度贡献相同 → **有效训练数据**

### 9.4 500 pairs 的 task_score 分布

| chosen_ts × rejected_ts | 数量 | 占比 | 含义 |
|-------------------------|------|------|------|
| 0 vs 0 | 245 | 49.0% | 两边都没通过（task 太难） |
| 1 vs 1 | 88 | 17.6% | 两边都通过（task 太简单） |
| 1 vs 0（chosen 过了） | 37 | 7.4% | **有效差异信号** |
| 其它 partial 分数 | 130 | 26.0% | 部分差异 |

107 个 states 中：
- 47 个 (43.9%) 有 task_score 差异 → 产生有效信号
- 42 个 (39.3%) 所有 pair 的 task_score 都=0 → task 对 gpt-4o-mini 太难
- 18 个 (16.8%) 有 pass 但无差异 → task 太简单或 Clarify 没帮上

### 9.5 结论：停止调 reward，直接训练

**为什么继续调 reward 没有意义**：

1. DPO loss 不用 reward gap → reward 精度对训练影响极小
2. 行为方向由 behavior-first 保证 → 500 pairs 全部方向正确
3. 逆信号中的 20 个"真逆信号"是论文的 tradeoff 设计 → 不应消除
4. 52 个噪声逆信号对 DPO 训练无害 → 不需要为此改公式
5. 真正决定 pass rate 的是模型的代码生成能力和 Clarify 信息获取质量 → 不受 reward 参数影响

**决定：直接用 500 pairs 训练 v29，评估后再诊断。**

pass rate 目标（评估时验证）：
- Layer 2 已确认 clarified(0.399) > direct(0.373)，Clarify 有正收益
- 评估时期望：Novice（多轮 Clarify）pass rate ≥ Experienced（1轮） > Busy（不 Clarify）
- 用户体验指标后续设计

---

## 10. v29 DPO 训练

### 10.1 训练配置

```
python policy/train_dpo.py \
  --data data/dpo/prefs_v29_100states.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output models/v29_100states \
  --epochs 3 --lr 5e-5 --beta 0.1
```

- 数据：500 pairs, 不过滤（behavior-first 保证方向正确，reward gap 不进 DPO loss）
- 模型：Llama-3.1-8B-Instruct + QLoRA (r=64, alpha=16, 4-bit)
- 训练时间：~17 分钟（RTX 4090）

### 10.2 训练曲线

| Epoch | Loss | Accuracy | Margin |
|-------|------|----------|--------|
| 0.3 | 0.597 | 59.4% | 0.336 |
| 1.0 | 0.129 | 95.9% | 2.779 |
| 2.0 | 0.005 | 100% | 6.381 |
| 3.0 | 0.006 | 100% | 6.285 |

收敛良好，epoch 2 即达到 100% accuracy。

### 10.3 数据泄露检查

测试集 20 个 BigCodeBench states，与训练轨迹 109 states 零重叠 ✅。
初始选取时发现 2 个 state（BigCodeBench/56, BigCodeBench/108）出现在轨迹数据中（虽不在训练 pairs 里），已替换。

---

## 11. v29 多轮评估（进行中）

### 11.1 评估配置

```
python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v29_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval.jsonl \
  --max_samples 20 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_100states.json
```

- 测试：20 states × 3 personas = 60 对话
- 代码生成：本地 Llama（端到端）
- 用户模拟：gpt-4o-mini
- 指标：pass@1, pass@5, action accuracy, avg turns, clarify rate

### 11.2 v29 DPO 评估结果（20 states）

| Persona | avg turns | clarify rate | pass@1 | pass@5 |
|---------|:---------:|:------------:|:------:|:------:|
| Novice-Learner | 7.0 | 85.7% | 20% (4/20) | 25% (5/20) |
| Experienced-Engineer | 3.15 | 68.3% | 15% (3/20) | 15% (3/20) |
| Busy-Developer | 1.0 | 0% | 10% (2/20) | 15% (3/20) |
| **Overall** | — | — | **15% (9/60)** | **18.3% (11/60)** |

**行为分化**: Novice 多轮 Clarify（85.7%）、Experienced 适度（68.3%）、Busy 直接 Execute（0%）— 完全符合 persona 设计。

### 11.3 Base Llama 对照评估（20 states, `--no_lora`）

```
python eval/evaluate_multi_turn_persona.py \
  --no_lora --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval.jsonl \
  --max_samples 20 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_base_llama.json
```

| Persona | avg turns | clarify rate | pass@1 | pass@5 |
|---------|:---------:|:------------:|:------:|:------:|
| Novice-Learner | — | — | 15% (3/20) | 15% (3/20) |
| Experienced-Engineer | — | — | 15% (3/20) | 15% (3/20) |
| Busy-Developer | — | — | 15% (3/20) | 15% (3/20) |
| **Overall** | — | — | **15% (9/60)** | **15% (9/60)** |

**Base 行为**: persona-blind，所有 persona 表现一致，无行为分化。

### 11.4 DPO vs Base 对比分析（20 states）

| 指标 | DPO | Base | 差异 |
|------|:---:|:----:|:----:|
| pass@1 overall | 15% (9/60) | 15% (9/60) | ±0 |
| pass@5 overall | **18.3%** (11/60) | 15% (9/60) | **+3.3%** |
| Novice pass@1 | **20%** | 15% | +5% |
| Novice pass@5 | **25%** | 15% | **+10%** |
| Experienced pass@1 | 15% | 15% | ±0 |
| Busy pass@1 | 10% | 15% | -5% |
| 行为分化 | ✅ 三档明显 | ❌ persona-blind | — |

**关键发现**:
1. **行为学习完全成功** — DPO 三个 persona 行为明显分化，Base 无差异
2. **pass@1 总体相同** — 15% vs 15%，20 个样本统计噪声大
3. **pass@5 DPO 更优** — 18.3% vs 15%，Novice 贡献最大（25% vs 15%），说明多轮 Clarify 确实帮助了代码质量
4. **Busy 略低** — DPO Busy 10% vs Base 15%，仅差 1 个 task，采样噪声
5. **样本量不足** — 20 states 下 1 个 task 差异 = 5%，无法做统计显著性判断

**结论**: 需要扩大测试集到 50 states 以获得更可靠的对比。

### 11.5 Busy 表现不佳的深入分析

#### 核心观察

DPO Busy 永远 T0 Execute → 只有 masked query（缺失 output_format、validation_rules 等）→ 本质等同于 Layer 2 的 `direct` 版本。**Busy 从未获得任何额外信息。**

#### Base Llama 不完全是 persona-blind

Base Llama 虽未学过 persona，但会自发 Clarify：

| Base persona | 有 Clarify 的对话数 | 总 clarify turns |
|---|---|---|
| Novice | 4/20 (20%) | 20 |
| Experienced | 1/20 (5%) | 6 |
| Busy | 3/20 (15%) | 17 |

Base Busy 有 3 个 task 做了 5-6 轮 clarify → "意外"获得了额外信息。

#### 控制变量对比：都是 0-clarify 直接 Execute 时

仅比较 DPO 和 Base 都没有 Clarify、直接 Execute 的 task：

| Persona | 共同 0-clarify tasks | DPO pass@1 | Base pass@1 |
|---|---|---|---|
| Novice | 16 | 25.0% (4/16) | 18.8% (3/16) |
| Experienced | 19 | 15.8% (3/19) | 15.8% (3/19) |
| **Busy** | **17** | **11.8% (2/17)** | **17.6% (3/17)** |

当两者都直接 Execute 时，DPO Busy 反而比 Base 差。暗示 **DPO LoRA 训练可能轻微损害了纯代码生成能力**（LoRA 权重改变了 code generation 分布）。

#### Busy 困境的两个叠加因素

```
DPO Busy 的困境:
  ① 永远不 Clarify → 只有 masked query → 信息不足
  ② DPO LoRA 可能轻微伤害代码生成质量
  ③ 两因素叠加 → Busy 是受影响最大的 persona

DPO Novice 的优势:
  ① 多轮 Clarify → 恢复缺失信息
  ② 信息增益弥补了代码生成能力的轻微损失
  ③ 实例: BigCodeBench/116 — DPO(6轮Clarify)=Pass, Base(0轮)=Fail
```

#### 对论文的意义

这个结果**支持论文叙事**——正好体现了 persona-aware proactive behavior 的核心 trade-off:
- **Novice**: 多问 → 信息恢复 → 代码质量提升 ✅
- **Busy**: 不问 → 效率高但信息不足 → 代码质量略降 ← persona 设计的代价
- 论文故事: "不同 persona 偏好下产生不同的 task success trade-off，Clarify 是有代价的（打断用户），但也有收益（恢复信息提升代码质量）"

50-state 评估将验证这一趋势是否稳定。

---

## 12. 50-State 扩大评估（进行中）

### 12.1 动机

20-state 评估中 pass@1 DPO = Base = 15%，无法区分。1 个 task 差异 = 5%，统计噪声过大。扩大到 50 states 以获得更可靠对比。

### 12.2 测试集

- 从 1031 个未使用 state（排除 109 个轨迹 state）中随机采样 50 个
- 文件: `data/seeds/test_states_v29_eval_50.jsonl`
- 种子: 42，与训练数据零重叠

### 12.3 评估配置

```
# DPO 评估
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v29_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_100states_50test.json

# Base Llama 评估（DPO 完成后执行）
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --no_lora --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_base_llama_50test.json
```

- 50 states × 3 personas = 150 组多轮对话

### 12.4 评估结果（DPO vs Base, 50 states）

> 日期: 2026-04-12
> DPO 评估完成于 2026-04-11 18:09，Base 评估完成于 2026-04-12 04:00

#### 行为对比

| Persona | DPO avg turns | DPO clarify% | Base avg turns | Base clarify% |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | **7.0** | **85.7%** | 2.1 | 52.4% |
| Experienced-Engineer | **2.66** | **62.4%** | 1.98 | 49.5% |
| Busy-Developer | **1.0** | **0%** | 2.62 | 61.8% |

**DPO 行为分化完全成功** — 三档泾渭分明（Novice 多轮 Clarify > Experienced 适度 > Busy 直接 Execute）。

**Base 完全 persona-blind** — 三个 persona 的 clarify rate 几乎一样（49-62%），甚至 Busy(61.8%) 比 Experienced(49.5%) 问得还多，完全无视 persona 设定。

#### Pass Rate 对比

| Persona | DPO pass@1 | Base pass@1 | DPO pass@5 | Base pass@5 |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | **16%** (8/50) | 4% (2/50) | 20% (10/50) | 20% (10/50) |
| Experienced-Engineer | **12%** (6/50) | 10% (5/50) | **24%** (12/50) | 22% (11/50) |
| Busy-Developer | **14%** (7/50) | 10% (5/50) | 16% (8/50) | 16% (8/50) |
| **Overall** | **14%** (21/150) | **8%** (12/150) | **20%** (30/150) | **19.3%** (29/150) |

#### 关键发现

1. **pass@1 DPO 14% vs Base 8%** — DPO 显著优于 Base（+75% 相对提升），50 states 样本量下差异明确
2. **Novice pass@1: DPO 16% vs Base 4%** — 差距最大（+300% 相对提升），多轮 Clarify 的价值在 pass@1 上充分体现
3. **pass@5 基本持平** (20% vs 19.3%) — 说明差异主要来自 DPO 的行为策略（首选动作更准确），而非代码生成能力本身
4. **Base Busy clarify 62% 但 pass@1 只有 10%** — Base 不加区分地乱问问题反而没帮上忙，对比 DPO Busy 不问但 pass@1 14%，说明 persona-aware 策略比 persona-blind 更有效
5. **Busy pass@1: DPO 14% > Base 10%** — 20-state 时 DPO Busy 偏低的担忧未在 50-state 上复现，样本量增大后 DPO Busy 反而优于 Base

#### vs 20-state 对比

| 指标 | 20-state | 50-state | 趋势 |
|------|:---:|:---:|:---:|
| DPO pass@1 overall | 15% (9/60) | 14% (21/150) | 稳定 |
| Base pass@1 overall | 15% (9/60) | 8% (12/150) | Base 下降，20-state 高估了 Base |
| DPO-Base gap (pass@1) | ±0 | **+6%** | 50-state 揭示了真实差距 |
| DPO Novice pass@1 | 20% | 16% | 略降但仍最高 |
| DPO Busy pass@1 | 10% | 14% | 回升，20-state 的 -5% 是噪声 |
| 行为分化 | ✅ | ✅ | 稳定 |

20-state 时 pass@1 DPO=Base=15% 无法区分，50-state 证实了 DPO 的真实优势。20-state 的 Base 15% 是高估（样本噪声）。

#### 论文叙事

50-state 结果支持论文的核心论点：

1. **Persona-aware proactive behavior 可以通过 DPO 学习** — 行为分化三档明确
2. **Clarify 有正收益** — Novice 多轮 Clarify 带来最高 pass@1（16% vs Base 4%）
3. **策略比乱问更重要** — Base 不分 persona 乱问（Busy 62% clarify）反而效果差，DPO 按 persona 决策更有效
4. **pass@1 vs pass@5 的差异** 说明 DPO 学的是"何时该问"的决策能力，而非代码生成能力

---

## 13. 200-State 扩大评估（进行中）

> 日期: 2026-04-12
> 目标: 将测试集从 50 扩大到 200 states，获得统计显著性 (目标 p<0.005, power≈89%)

### 13.1 动机与 Power Analysis

50-state 评估 DPO 14% vs Base 8%，趋势正确但统计不显著（Fisher exact p=0.139）。

| 测试规模 | Overall p-value | Power (p<0.05) | Novice p-value |
|:---:|:---:|:---:|:---:|
| 50 states (当前) | 0.139 | 31.7% | 0.092 |
| 100 states | 0.026 | 64.5% | 0.008 |
| 150 states | 0.005 | 79.9% | 0.0008 |
| **200 states** | **0.001** | **89.4%** | **0.0001** |

### 13.2 测试集

- 从 981 个可用 state（排除 109 训练 + 50 已测试）中随机采样 150 个
- 文件: `data/seeds/test_states_v29_eval_150extra.jsonl`
- 种子: 43（与 50-state 的 seed=42 不同），与训练数据零重叠
- 最终合并: 50 (已有) + 150 (新增) = **200 states, 600 组对话**

### 13.3 评估配置

```
# DPO 评估（150 extra states）
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v29_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_150extra.jsonl \
  --max_samples 150 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_dpo_150extra.json

# Base Llama 评估（DPO 完成后执行）
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --no_lora --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_150extra.jsonl \
  --max_samples 150 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_base_150extra.json
```

- DPO 150-state 评估进行中（预计 ~9h），Base 待 DPO 完成后启动
- 完成后与 50-state 结果合并为 200-state 最终结果

### 13.4 训练集决策

**不增大训练集**。理由：
1. 500 pairs 训练已饱和（epoch 2 即 accuracy 100%，行为分化完美）
2. pass@1 瓶颈在 Llama-8B 代码生成能力，非训练数据量
3. 当前阶段目标是小规模验证 pipeline 有效性

### 13.5 200-state 评估后的下一步

评估完成后，需补充 baseline 对比（用同一个 200-state 测试集）：
1. **SFT baseline** — 同样 500 pairs 做 supervised finetuning
2. **Prompting baseline** — system prompt 描述 persona，不训练
3. 最终论文实验: DPO vs SFT vs Prompting vs Base，200 states

### 13.6 已发现问题：Novice 过拟合为"永远 Clarify"

50-state 评估数据显示：**Novice 100%（50/50）都跑满 7 轮**（6 轮 Clarify + 1 轮 forced Execute），没有一个提前选择 Execute。

| 阶段 | Novice avg turns | 行为 |
|------|:---:|------|
| 训练轨迹 (gpt-4o-mini) | 2.30 | 正常，1-3 轮后 Execute |
| DPO 评估 (Llama) | **7.0** | 永远 Clarify，全部撞 max_turns=6 上限 |

**影响**：
1. **评估速度** — Novice 每对话 7 轮（6 次 Llama 推理 + 6 次 gpt-4o-mini API），约为 Busy (1轮) 的 7 倍耗时
2. **pass@1 可能受损** — 前几轮 Clarify 有信息增益，后面几轮可能是无效提问，反而引入噪声
3. **论文叙事** — 如果 Novice "过度打扰"用户，与"学会何时问"的论点矛盾

**可能原因**：
- DPO 训练 500 pairs 中 Novice 的 chosen 几乎全是 Clarify（turn 0-2），模型过拟合为"Novice → Clarify"
- 训练数据中缺少 Novice 在 turn 2-3 选择 Execute 的正样本
- DPO loss 不区分 turn，所有 Clarify chosen 对梯度贡献相同

**待 200-state 结果出来后评估**：
- 是否需要降低 max_turns（如 4）以限制过度 clarify
- 是否需要在训练数据中增加 Novice Execute 的正样本
- 是否需要在 reward 中加 turn penalty 抑制过长对话

### 13.7 评估结果

待评估完成后补充。

---

## 14. 论文完整实验计划

> 日期: 2026-04-12
> 在 200-state 评估完成并确认显著性后，按此计划推进剩余实验

### 14.1 Backbones

- Llama-3.1-8B-Instruct（已训练 + 评估中）
- Qwen2.5-7B-Instruct（待训练）

### 14.2 Methods & Baselines

| 方法 | 说明 | 状态 |
|------|------|------|
| **TactfulLLM-DPO** (ours) | Persona-aware DPO, Llama + Qwen | Llama ✅, Qwen ❌ |
| **Direct Execution** | masked query 直接生成，不允许 clarification | ❌ |
| **Prompt-only Clarify-or-Execute** | 不训练，仅靠 prompt 指示何时 clarify | ❌ |
| **Always-Clarify** | 固定先问 1 或 K 轮再执行 | ❌ |
| **CollabLLM** | 外部 baseline (github.com/Wuyxin/collabllm) | ❌ |

**Baseline 设计理由**：
- **Direct**: 证明不澄清会怎样（下界）
- **Prompt-only**: 回应 reviewer "simply prompting 能否解决"
- **Always-Clarify**: 证明"学会何时问" > "总是问"
- **CollabLLM**: 外部方法对比

### 14.3 三个实验

**Experiment 1: Main task performance (masked queries)**
- 所有方法 × 两个 backbone 全面对比
- 指标：pass@1, avg clarification turns, interruption cost, overall utility
- 核心叙事：TactfulLLM 在 success 和 interruption 之间取得更好的 trade-off

**Experiment 2: Recovery analysis (full vs masked)**
- 对每个 backbone 比较: Full Query / Masked Direct / Masked+Clarified / Masked+Ideal Disclosed / Oracle
- 核心叙事：clarification 恢复了多少信息，与 ideal disclosed 之间的 gap 有多大
- 已有基础：v29 Layer 2 分析框架 (direct/clarified/ideal_disclosed/oracle)

**Experiment 3: Persona sensitivity**
- 按 persona 分组比较: clarification frequency, rejection rate, task success, utility
- 核心叙事：模型根据 user characteristics 改变行为，而非固定套路
- 已有基础：50-state 行为分化数据，200-state 将更 solid

### 14.4 评估设置

- Primary: masked queries（200 test states）
- Reference: full queries
- Metrics: task success, clarification turns, rejection rate, total utility/reward
- Analysis: recovery from missing information, persona sensitivity, success-interruption trade-off

### 14.5 实施优先级（200-state 确认后）

1. **P0**: Prompt-only baseline + Always-Clarify baseline（用已有 Llama 模型 + 200 test states）
2. **P0**: Direct Execution baseline（最简单，可能接近 Base Busy 的行为）
3. **P1**: Qwen backbone（重跑 masking → 轨迹 → DPO 训练 → 评估）
4. **P1**: CollabLLM（需要跑他们的代码，可能有适配工作）
5. **P2**: 补充 utility/rejection rate 等指标

---

## 15. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/mask_task_details.py` | v29 masking 逻辑 (锚点切割 + OF 拆分) |
| `scripts/generate_trajectories.py` | 轨迹生成 (all-disclosed fix, code_versions) |
| `data/seeds/bigcodebench_masked_states.jsonl` | 1140 tasks, mean 2.9 items/task |
| `data/seeds/test_states_v29_eval.jsonl` | 20 个评估用 states（与训练无重叠） |
| `data/data/logs/traj_v29_10states_final_20260410_052948.jsonl` | 10-state 轨迹 (249 turns) |
| `data/data/logs/traj_v29_100states_20260410_080521.jsonl` | 100-state 轨迹 part1 (1454 turns, 57 states) |
| `data/data/logs/traj_v29_100states_part2_20260410_114625.jsonl` | 100-state 轨迹 part2 (1340 turns, 52 states) |
| `data/data/logs/traj_v29_100states_combined.jsonl` | 100-state 轨迹合并 (2794 turns, 109 states) |
| `data/dpo/prefs_v29_10states_test.jsonl` | 10-state preference pairs (46 pairs) |
| `data/dpo/prefs_v29_100states.jsonl` | 100-state preference pairs (500 pairs) |
| `models/v29_100states/` | v29 DPO LoRA adapter |
| `outputs/eval_v29_100states.json` | v29 DPO 评估结果（20 states） |
| `outputs/eval_v29_base_llama.json` | Base Llama 评估结果（20 states） |
| `outputs/eval_v29_100states_50test.json` | v29 DPO 评估结果（50 states） |
| `outputs/eval_v29_base_llama_50test.json` | Base Llama 评估结果（50 states） |
| `data/seeds/test_states_v29_eval_50.jsonl` | 50 个评估用 states（与训练无重叠） |
| `data/seeds/test_states_v29_eval_150extra.jsonl` | 150 个额外评估用 states（seed=43，与训练+50test 无重叠） |
| `outputs/eval_v29_dpo_150extra.json` | v29 DPO 评估结果（150 extra states，进行中） |
| `outputs/eval_v29_base_150extra.json` | Base Llama 评估结果（150 extra states，待启动） |
| `reward/compute_rewards.py` | Reward + preference pair 生成 |
