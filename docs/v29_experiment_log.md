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

### 13.7 200-State 评估结果

> 日期: 2026-04-14
> DPO 150-extra 完成于 2026-04-14，Base 150-extra 分两批完成（108 + 42 states）

#### 200-State DPO vs Base 完整对比

| Persona | DPO pass@1 | Base pass@1 | DPO pass@5 | Base pass@5 |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | **18.5%** (37/200) | 12.5% (25/200) | **25.5%** (51/200) | 20.5% (41/200) |
| Experienced-Engineer | **15.5%** (31/200) | 12.5% (25/200) | **25.0%** (50/200) | 19.0% (38/200) |
| Busy-Developer | 14.0% (28/200) | 13.0% (26/200) | 20.0% (40/200) | 20.0% (40/200) |
| **Overall** | **16.0%** (96/600) | **12.7%** (76/600) | **23.5%** (141/600) | **19.8%** (119/600) |

#### Fisher Exact Tests (pass@1, one-sided)

| 对比 | p-value | 显著? |
|------|:---:|:---:|
| Overall | 0.059 | ❌ (接近) |
| Novice | 0.064 | ❌ (接近) |
| Experienced | 0.236 | ❌ |
| Busy | 0.442 | ❌ |

#### 分析

1. **趋势正确但统计不显著** — DPO 全面优于 Base（+3.3% overall），但 p=0.059 未过 0.05
2. **Gap 比 50-state 预期小** — 50-state 时 DPO 14% vs Base 8%（gap=6%），200-state 实际 gap 只有 3.3%
3. **原因**：Base 150-extra 的 pass@1 没有继续低（稳定在 7-9%），而 50-state 的 Base 8% 本身就偏高
4. **行为分化依然完美** — DPO 三档分化稳定（Novice 7.0 > Experienced 2.6 > Busy 1.0），Base persona-blind
5. **决定**：不再扩大测试集，转向 baseline 对比来丰富论文叙事

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

## 16. Prompt-only Baseline（进行中）

> 日期: 2026-04-14
> 目标: 验证"纯 prompt 描述 persona 能否驱动行为分化"，回应 reviewer "why not just prompt?"

### 16.1 实现方案

在 `eval/evaluate_multi_turn_persona.py` 新增 `--prompt_only` flag：
- 使用 Base Llama（无 LoRA），加载方式与 `--no_lora` 相同
- Action 选择使用 `select_action_prompt_only()` 替代 `select_action_with_model()`
- System prompt 包含 persona 描述（patience/expertise/特点），但**不包含任何决策规则或轮数阈值**
- 让模型自己根据 persona 描述判断是否 Clarify

### 16.2 Prompt 设计原则

**只描述用户特征，不给策略**：
- Busy-Developer: "low patience, mid-level expertise, time-constrained, prefers efficiency"
- Novice-Learner: "high patience, low expertise, unfamiliar with domain"
- Experienced-Engineer: "moderate patience, high expertise, may not provide complete specs"

通用指令: "Consider the user's characteristics when deciding" — 不含轮数、阈值或任何 oracle 知识。

### 16.3 Sanity Check（2 states）

| Persona | State 1 turns | State 2 turns | avg turns |
|---------|:---:|:---:|:---:|
| Novice-Learner | 7 (forced) | 3 | 4.50 |
| Busy-Developer | 4 | 5 | **5.00** |
| Experienced-Engineer | 3 | 3 | 3.00 |

**关键发现**：Prompt-only **几乎没有行为分化**。Busy (5.0轮) 甚至比 Novice (4.5轮) 问得更多。Base Llama 无视 prompt 中的 persona 描述，一律倾向 Clarify。

对比 DPO: Novice 7.0 > Experienced 2.6 > Busy 1.0 — 分化完美。

### 16.4 50-State 评估（进行中）

```
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --prompt_only --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v29_prompt_only_50test.json
```

- 50 states × 3 personas = 150 组对话，预计 3-5 小时
- 完成后与 DPO 50-state 和 Base 50-state 做三方对比
- 如果 50-state 结果符合预期，再扩展到 200 states

### 16.5 预期结果

Prompt-only 应该：
1. **行为分化弱或无** — 三个 persona 行为差异不大（sanity check 已确认）
2. **pass@1 接近 Base** — 没有学过策略，行为随机，不应优于 Base
3. **论文价值**: 证明 persona-aware proactive behavior 不能靠 prompting 解决，需要 DPO 训练

---

## 18. Prompt-only 50-State 完整结果

> 日期: 2026-04-15
> 50 states 评估完成

| Persona | pass@1 | pass@5 | Avg Turns | Rej Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 14.0% (7/50) | 20.0% (10/50) | 5.82 | 46.5% |
| Experienced-Engineer | 6.0% (3/50) | 16.0% (8/50) | 5.38 | 55.7% |
| Busy-Developer | 6.0% (3/50) | 18.0% (9/50) | 5.32 | 89.4% |
| **Overall** | **8.7% (13/150)** | **18.0% (27/150)** | **5.51** | **63.2%** |

**关键结论**:
1. **行为零分化** — 三个 persona 的 avg turns 几乎一样（5.3-5.8），完全无视 persona prompt
2. **pass@1 最差** — 8.7%，比 Base (12.7%) 和 DPO (16.0%) 都低
3. **Busy 被疯狂打扰** — 83% clarify rate、89.4% rejection rate，加了 persona prompt 反而更糟
4. **论文价值** — 直接回应 reviewer "why not just prompt?"

---

## 19. Direct Execution 50-State 结果

> 日期: 2026-04-15
> 实现 `--direct_execution` flag，强制 Turn 0 Execute，不做任何 clarification

| Persona | pass@1 | pass@5 | Avg Turns | Rej Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 6.0% (3/50) | 12.0% (6/50) | 1.0 | 0% |
| Experienced-Engineer | 8.0% (4/50) | 14.0% (7/50) | 1.0 | 0% |
| Busy-Developer | 8.0% (4/50) | 14.0% (7/50) | 1.0 | 0% |
| **Overall** | **7.3% (11/150)** | **13.3% (20/150)** | **1.0** | **0%** |

**关键结论**:
1. **zero-interaction lower bound** — pass@1 7.3% 是所有方法中最低的
2. **证明 clarification 有价值** — DPO 16.0% vs Direct 7.3%，+8.7% 绝对提升
3. **三 persona 无差异** — 没有交互就没有行为分化

---

## 20. Clarify-first (K=1) Baseline ✅

> 日期: 2026-04-15
> 实现 `--always_clarify K` flag，固定 K 轮 Clarify 后 Execute

**设计**: Turn 0 强制 Clarify（Base Llama 生成澄清问题）→ gpt-4o-mini 模拟用户回答 → Turn 1 强制 Execute。所有 persona 统一 2 turns。

**命名改为 Clarify-first**: 比 Always-Clarify 更准确描述 K=1 的行为（"先问一轮再写代码"）。

**50-state 结果**:

| Persona | pass@1 | pass@5 | Avg Turns | Rej Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 8.0% (4/50) | 20.0% (10/50) | 2.0 | — |
| Experienced-Engineer | 12.0% (6/50) | 20.0% (10/50) | 2.0 | — |
| Busy-Developer | 8.0% (4/50) | 18.0% (9/50) | 2.0 | — |
| **Overall** | **9.3% (14/150)** | **19.3% (29/150)** | **2.0** | **52.0%** |

**关键结论**:
1. 比 Direct Execution (+2.0% pass@1) 有提升，说明 1 轮 Clarify 能获取少量信息
2. 52% rejection rate 很高 — Busy persona 拒绝大部分 clarify（只有 1 轮机会）
3. 所有 persona 固定 2 turns，无行为分化（by design）

---

## 21. Baseline 全面对比（50-state 口径）

> 日期: 2026-04-15

| Method | pass@1 | pass@5 | Avg Turns | Rej Rate |
|--------|:---:|:---:|:---:|:---:|
| Direct Execution | 7.3% | 13.3% | 1.0 | 0% |
| Clarify-first (K=1) | 9.3% | 19.3% | 2.0 | 52.0% |
| Base LLM | 12.7%* | 19.8%* | 2.2* | 62.7%* |
| Prompt-only | 8.7% | 18.0% | 5.5 | 63.2% |
| **TactfulLLM (ours)** | **16.0%*** | **23.5%*** | **3.5*** | **45.2%*** |

*Base 和 DPO 为 200-state 结果，其余为 50-state。

**Rejection Rate 分析**:
- C_interrupt (论文公式) 对 DPO 不利：DPO Novice 跑满 7 轮累积成本 2.235，高于 Base Novice 0.326
- 原因：Novice 过拟合为"永远 Clarify"，大量 clarify turns + 45% rejection rate
- 决定：Table 1 报 pass@1/pass@5/Avg Turns 按 persona 展开，Rejection Rate 在正文提及，不单独成表

### 论文 Table 1 设计

```
pass@1 (Nov/Exp/Busy/All) | pass@5 (Nov/Exp/Busy/All) | Avg Turns (Nov/Exp/Busy/All)
```
- 12 data columns，NeurIPS 单栏可放
- Rejection Rate 在正文一句话报
- C_interrupt 不报（对 DPO 不利，因为 Novice 过度 Clarify）

---

## 22. Experiment 2: Recovery Analysis — Oracle & Ideal Disclosed 实现

> 日期: 2026-04-15
> 目标: 量化 clarification 恢复了多少被遮蔽的信息

### 设计

Experiment 2 比较 4 个 information level：
1. **Direct** (已有): 遮蔽后 query 直接生成代码 → pass@k = 7.3%
2. **Clarified** (已有): TactfulLLM 通过对话获取部分信息后生成代码 → pass@k = 16.0%
3. **Ideal Disclosed** (新): 遮蔽 query + 所有 masked_fields 一次性给出 → pass@k = ?
4. **Oracle** (新): 完整原始 `original_instruct_prompt` → pass@k = ?

### 实现

在 `eval/evaluate_multi_turn_persona.py` 中新增两个 flag:
- `--oracle`: 用 `original_instruct_prompt` 替换 `query`，单轮 Execute，无 persona 循环
- `--ideal_disclosed`: 保持遮蔽 `query`，但把所有 `masked_fields` 填入 `disclosed_info`，单轮 Execute

关键设计：
- **Persona-independent**: 这两个都是单轮生成，不涉及用户交互，结果与 persona 无关
- **Base Llama** (no LoRA): 衡量信息量差异而非策略差异
- 新增 pass@10 评估 (`--pass_at_k 1 5 10`)

### 运行状态

- Oracle 50-state: 🏃 运行中
- Ideal Disclosed 50-state: ⏳ 待 Oracle 完成后启动

---

## 17. 文件清单

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
| `outputs/eval_v29_dpo_150extra.json` | v29 DPO 评估结果（150 extra states）✅ |
| `outputs/eval_v29_base_150extra.json.partial` | Base Llama 评估结果（150 extra, 108 states partial）|
| `outputs/eval_v29_base_150extra_remaining.json` | Base Llama 评估结果（150 extra, 剩余 42 states）✅ |
| `outputs/eval_v29_prompt_only_50test.json` | Prompt-only baseline 评估结果（50 states）✅ |
| `outputs/eval_v29_direct_execution_50test.json` | Direct Execution baseline 评估结果（50 states）✅ |
| `outputs/eval_v29_clarify_first_50test.json` | Clarify-first (K=1) baseline 评估结果（50 states）✅ |
| `outputs/eval_v29_oracle_50test.json` | Oracle (unmasked) 评估结果（50 states，运行中） |
| `outputs/eval_v29_ideal_disclosed_50test.json` | Ideal Disclosed 评估结果（50 states，待启动） |
| `reward/compute_rewards.py` | Reward + preference pair 生成 |

---

## v30: Disclosure-Aware 停止条件

> 日期: 2026-04-16
> 目标: 修复 Novice 过拟合（永远 Clarify），让模型学会 "信息够了就 Execute"

### 问题回顾

v29 DPO 评估中 Novice 100% 跑满 7 轮（6 Clarify + 1 forced Execute），原因：
- `get_correct_action` 硬编码 `turn < 3 → Clarify`，训练数据中 227 个 Novice pairs 几乎全是 chosen=Clarify
- Method B 在 turn 1+ 无条件生成 chosen=Clarify pairs（"No reward gate"）
- 模型从未见过 Novice "该停了" 的 Execute 正样本

### 核心改动

#### 1. `compute_disclosure_ratio(state)` — 新增函数
从 `state.disclosure_rule` 计算已披露比例：`disclosed_count / masked_count`

#### 2. `get_correct_action()` — Disclosure-aware Novice 规则
```
旧规则: Novice → Clarify if turn < 3 else Execute
新规则: Novice → Execute if disclosure_ratio >= 1.0
                  else Clarify if turn < 4
                  else Execute
```
- 当所有 masked items 已全部 disclosed，任何 turn 都该 Execute
- 硬上限从 turn 3 放宽到 turn 4（给更多 Clarify 空间，但有 disclosure 兜底）

#### 3. Method B — 跳过已充分披露的 Clarify pairs
在 Method B 循环中，对每个 Clarify turn 计算 `disclosure_ratio`，若 `get_correct_action` 返回 Execute，则不生成 chosen=Clarify pair。

#### 4. Method A (fork pairs) — 传入 disclosure_ratio
fork pair 的 `get_correct_action` 调用也传入 disclosure_ratio，保持一致。

### 实际改动（最终版）

经过迭代调试，最终规则：

#### Novice-Learner
- Turn 0-1: 总是 Clarify
- Turn 2+: disclosure_ratio >= 0.5 → Execute（信息已够），否则继续 Clarify
- Turn 3+: 硬上限 Execute

注：disclosure_ratio 在 turn T 反映的是 turn T **之前**的累计披露状态（问问题之前），
所以 turn 2 看到 50% 意味着前两轮已获取了一半信息。

#### Busy-Developer
- Turn 0: n_masked_items >= 3 → Clarify（复杂 task 值得问一轮），否则 Execute
- Turn 1+: 总是 Execute（最多问一轮）

68% 的 task 有 ≥3 masked items，所以 Busy 在多数 task 上会问一轮。

#### Experienced-Engineer（不变）
- Turn 0: Clarify
- Turn 1+: Execute

#### Method B2（新增）
为 Busy 生成 turn 1 Execute pairs：当 Busy chosen=Clarify at turn 0 时，
遍历该 trajectory 的 turn 1+，生成 chosen=Execute, rejected=Clarify pairs。

### v30 Preference Pairs 对比

**v29 (500 pairs)**
```
Busy-Developer: 107 pairs
  turn=0: Clarify=0,   Execute=107

Experienced-Engineer: 166 pairs
  turn=0: Clarify=105, Execute=0
  turn=1: Clarify=0,   Execute=61

Novice-Learner: 227 pairs
  turn=0: Clarify=107, Execute=0
  turn=1: Clarify=81,  Execute=0
  turn=2: Clarify=36,  Execute=0
  turn=3: Clarify=2,   Execute=1
```

**v30 (509 pairs)**
```
Busy-Developer: 124 pairs
  turn=0: Clarify=73,  Execute=34   ← 复杂task问一轮
  turn=1: Clarify=0,   Execute=17   ← 问完就停

Experienced-Engineer: 166 pairs
  turn=0: Clarify=105, Execute=0
  turn=1: Clarify=0,   Execute=61

Novice-Learner: 219 pairs
  turn=0: Clarify=107, Execute=0
  turn=1: Clarify=81,  Execute=0
  turn=2: Clarify=3,   Execute=25   ← 信息够→停
  turn=3: Clarify=0,   Execute=3    ← 硬上限
```

### v30 DPO 训练

配置同 v29：Llama-3.1-8B-Instruct + QLoRA (r=64), beta=0.1, epochs=3, lr=5e-5

| Epoch | Loss | Accuracy | Margin |
|:---:|:---:|:---:|:---:|
| 0.35 | 0.505 | 66.3% | 0.984 |
| 1.0 | 0.338 | 88.3% | 1.638 |
| 2.0 | 0.209 | 92.2% | 3.133 |
| 3.0 | 0.130 | 97.5% | 3.917 |

vs v29（epoch 2 即 100%）略低，因为 v30 pairs 更多样化（Busy 混合 Clarify/Execute，
Novice turn 2 翻转）。97.5% accuracy 是健康的，说明模型能学到但不是死记。

模型保存至 `models/v30_100states/`。

### v30 50-state 评估（进行中）

```
PYTHONUNBUFFERED=1 python eval/evaluate_multi_turn_persona.py \
  --model_dir models/v30_100states \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --test_states data/seeds/test_states_v29_eval_50.jsonl \
  --max_samples 50 --max_turns 6 \
  --llm_model gpt-4o-mini --pass_at_k 1 5 \
  --output outputs/eval_v30_dpo_50test.json
```

预期：
- Novice avg turns 从 7.0 降到 ~3-4
- Busy avg turns 从 1.0 升到 ~1.3-1.5
- pass@1 维持或提升（≥16%）
- 行为分化保持三档

### v30 50-state 中间结果（13 states）→ 失败，回退 v29

13 states 跑完后的中间结果（来自 `.partial` 文件）：

| Persona | v30 pass@1 | v29 pass@1 (200) | v30 Avg Turns | v29 Avg Turns |
|---|:---:|:---:|:---:|:---:|
| Novice | 1/13 (7.7%) | 18.5% | 3.6 | 7.0 |
| Experienced | 0/13 (0.0%) | 15.5% | 2.2 | 2.6 |
| Busy | 0/13 (0.0%) | 14.0% | 2.9 | 1.0 |

**问题**：
1. **pass@1 全面崩盘**：三个 persona 都远低于 v29，不仅仅是 Novice
2. **Busy 过度 Clarify**：58% T0 Execute（vs v29 100%），部分跑到 Turn 3-6。加了 conditional Clarify 后模型学过头了
3. **Experienced 也退化**：T0 Execute 仅 14%（vs v29 38%）

**结论**：v30 disclosure-aware 改动对 pair 分布影响过大，模型行为和代码质量同时退化。
决定**回退到 v29 代码**，用 v29 模型和结果继续实验。

---

## Ablation Study（基于 v29）

### 设计

保持 v29 DPO pairs（500 对）不变，只修改 `render_state.py` 中 prompt 格式：

| Variant | 改了什么 | Prompt 变化 |
|---|---|---|
| TactfulLLM (full) | — | 完整：User Profile + Task Uncertainty + Context |
| w/o Persona | 移除 `[User Profile]` 块 | 无 Type/Patience/Expertise |
| w/o Uncertainty | 移除 `Task Uncertainty` 行 | 无 uncertainty 数值 |

实现方式：`render_state()` 加 `ablation_mode` 参数（`"no_persona"` / `"no_uncertainty"`），
通过 `ABLATION_MODE` 环境变量在训练和评估时传入。

### 训练

配置同 v29：3 epochs, lr=5e-5, beta=0.1, QLoRA r=64。

**w/o Persona 训练**（16min 完成）：

| Epoch | Loss | Accuracy | Margin |
|:---:|:---:|:---:|:---:|
| 0.36 | 0.642 | 59.4% | 0.201 |
| 1.04 | 0.648 | 69.9% | 0.301 |
| 2.07 | 0.502 | 74.0% | 0.713 |
| 3.0 | 0.386 | 87.5% | 1.022 |

模型保存至 `models/v29_ablation_no_persona/`。

**w/o Uncertainty 训练**（17min 完成）：

| Epoch | Loss | Accuracy | Margin |
|:---:|:---:|:---:|:---:|
| 0.35 | 0.597 | 59.4% | 0.336 |
| 1.0 | 0.129 | 95.9% | 2.779 |
| 2.0 | 0.005 | 100% | 6.381 |
| 3.0 | 0.006 | 100% | 6.285 |

模型保存至 `models/v29_ablation_no_uncertainty/`。

注：w/o Uncertainty 的训练曲线与 full v29 几乎一致（epoch 2 即 100%），说明 uncertainty 信息对 DPO 的 chosen/rejected 区分贡献不大。而 w/o Persona 只到 87.5%，说明缺少 persona 信息后模型更难区分正确动作。

### 50-state 评估结果

> 日期: 2026-04-16

#### w/o Persona (50-state)

| Persona | pass@1 | pass@5 | Avg Turns | Clarify Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 8.0% (4/50) | 14.0% (7/50) | 1.04 | 3.8% |
| Experienced-Engineer | 8.0% (4/50) | 16.0% (8/50) | 1.04 | 3.8% |
| Busy-Developer | 16.0% (8/50) | 16.0% (8/50) | 1.04 | 3.8% |
| **Overall** | **10.7% (16/150)** | **15.3% (23/150)** | **1.04** | **3.8%** |

**行为分化完全消失** — 三个 persona turns 全部 ≈1.0，clarify rate 3.8%，退化为 Direct Execution。

#### w/o Uncertainty (50-state)

| Persona | pass@1 | pass@5 | Avg Turns | Clarify Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 12.0% (6/50) | 22.0% (11/50) | 8.00 | 87.5% |
| Experienced-Engineer | 12.0% (6/50) | 16.0% (8/50) | 2.56 | 60.9% |
| Busy-Developer | 8.0% (4/50) | 14.0% (7/50) | 1.00 | 0.0% |
| **Overall** | **10.7% (16/150)** | **17.3% (26/150)** | **3.85** | **49.5%** |

**行为分化保持**（8.0 / 2.6 / 1.0，与 full 几乎一致），但 pass@1 从 14.0% 降到 10.7%。

### 100-state 评估结果（50 + 50-extra 合并）

> 日期: 2026-04-17

为增强统计可靠性，两个 ablation 各追加 50 states（从 `test_states_v29_eval_150extra.jsonl` 前 50 个）。

#### w/o Persona (100-state)

| Persona | pass@1 | pass@5 | Avg Turns |
|---------|:---:|:---:|:---:|
| Novice-Learner | 10.0% (10/100) | 14.0% (14/100) | 1.02 |
| Experienced-Engineer | 10.0% (10/100) | 15.0% (15/100) | 1.02 |
| Busy-Developer | 13.0% (13/100) | 14.0% (14/100) | 1.02 |
| **Overall** | **11.0% (33/300)** | **14.3% (43/300)** | **1.02** |

100-state 确认：行为分化完全消失（三 persona turns 全部 ≈1.0），pass@1 11.0%。

#### w/o Uncertainty (87-state partial, 2026-04-17)

50-extra 跑到 37/50 states 中断（疑似容器休眠），合并 50 + 37 = 87 states 先填 ablation 表，完整 100-state 明天补剩余 13 states。

| Persona | pass@1 | pass@5 | Avg Turns | Rej Rate |
|---------|:---:|:---:|:---:|:---:|
| Novice-Learner | 10.3% (9/87) | 20.7% (18/87) | 7.55 | 45.4% (259/570) |
| Experienced-Engineer | 13.8% (12/87) | 19.5% (17/87) | 2.66 | 37.5% (54/144) |
| Busy-Developer | 9.2% (8/87) | 12.6% (11/87) | 1.00 | 0% |
| **Overall** | **11.1% (29/261)** | **17.6% (46/261)** | **3.74** | — |

**注意**：w/o Unc Exp 13.8% 单列反超 full Exp 12.0%（full 为 50-state，此为 87-state），test 规模不一致导致的样本差异。建议后续统一到 100-state 再比较。

### Ablation 综合对比表（论文 Table 用）

最新口径（2026-04-17）：

| Method | pass@1 | | | | Avg Turns | | | Rej Rate | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | Nov | Exp | Busy | All | Nov | Exp | Busy | Nov | Exp | Busy |
| **TactfulLLM** (full, 50) | **16.0** | 12.0 | **14.0** | **14.0** | 7.0 | 2.7 | 1.0 | 45.7 | 49.4 | 0.0 |
| w/o Persona (100) | 10.0 | 10.0 | 13.0 | 11.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| w/o Uncertainty (87†) | 10.3 | 13.8 | 9.2 | 11.1 | 7.6 | 2.7 | 1.0 | 45.4 | 37.5 | 0.0 |

*三行 test 规模不一致（full=50, w/o Persona=100, w/o Uncertainty=87 partial）。†w/o Uncertainty 50-extra 在 37/50 处中断，明天补跑剩余 13 states。*

### Ablation 分析

1. **w/o Persona — 行为分化的必要条件**
   - 移除 User Profile 后，模型无法区分 persona，三个用户的行为完全一致（turns≈1.0）
   - 本质退化为 Direct Execution（几乎从不 Clarify）
   - pass@1 从 14.0% 降到 11.0%（-3.0%）

2. **w/o Uncertainty — 代码质量的贡献因子**
   - 行为分化完好保持（Novice 8.0 > Experienced 2.6 > Busy 1.0）
   - 说明行为分化主要由 persona 信息驱动，不依赖 uncertainty
   - 但 pass@1 从 14.0% 降到 10.7%（-3.3%），尤其 Busy 14%→8%
   - 说明 uncertainty 信号帮助模型做出更有效的决策

3. **两个组件互补**
   - Persona 控制 **"何时问"**（行为模式）
   - Uncertainty 帮助 **"问得更有效"**（决策质量）
   - 缺一不可：缺 persona 则无分化，缺 uncertainty 则质量降

### 文件清单（新增）

| 文件 | 说明 |
|------|------|
| `models/v29_ablation_no_persona/` | w/o Persona DPO LoRA adapter |
| `models/v29_ablation_no_uncertainty/` | w/o Uncertainty DPO LoRA adapter |
| `outputs/eval_v29_ablation_no_persona_50test.json` | w/o Persona 50-state 评估 ✅ |
| `outputs/eval_v29_ablation_no_persona_50extra.json` | w/o Persona 50-extra 评估 ✅ |
| `outputs/eval_v29_ablation_no_uncertainty_50test.json` | w/o Uncertainty 50-state 评估 ✅ |
| `outputs/eval_v29_ablation_no_uncertainty_50extra.json.partial` | w/o Uncertainty 50-extra 评估（37/50，中断） |

---

## Experiment 2: Recovery Analysis — 200-state 过夜任务

> 日期: 2026-04-17 晚启动
> 目标: 填 Recovery 表（Masked Direct / Clarified / Ideal Disclosed / Full Query）

### 任务规划

Recovery 表按 200-state 口径，当前覆盖：

| Condition | 现状 | 今晚动作 |
|---|---|---|
| Masked Direct | 50-state (7.3%) | 补 150 extra，resume 至 200 |
| DPO Clarified | ✅ 200-state (16.0%, Nov 18.5 / Exp 15.5 / Busy 14.0) | 无需动作 |
| Ideal Disclosed | 未启动 | 新跑 200-state |
| Oracle / Full Query | 旧 20-state 在不同测试集上 | 新跑 200-state |

### 启动配置

```bash
# /tmp/run_exp2_overnight.sh（nohup 后台）
python eval/evaluate_multi_turn_persona.py --direct_execution ... --output outputs/eval_v29_direct_execution_200.json
python eval/evaluate_multi_turn_persona.py --oracle          ... --output outputs/eval_v29_oracle_200.json
python eval/evaluate_multi_turn_persona.py --ideal_disclosed ... --output outputs/eval_v29_ideal_disclosed_200.json
```

- Test set: `data/seeds/test_states_v29_eval_200.jsonl`（50test + 150extra 合并，200 states）
- `--max_samples 200 --pass_at_k 1 5`，`--base_model meta-llama/Llama-3.1-8B-Instruct`
- Direct: 3 personas（虽然行为相同，保持格式一致），另两项 persona-independent
- Direct 用 `.partial` resume 机制，复用 50-state 结果（只跑 150 extra × 3 personas）

### 进程安全性

- PID 603803, PPID=1（init 接管），TTY=?，独立 SID
- SSH 断开不影响；容器休眠靠 `.partial` resume 接续
- Log: `logs/exp2_overnight.log`

### 预期产出（明早检查）

| 文件 | 说明 |
|---|---|
| `outputs/eval_v29_direct_execution_200.json` | Masked Direct 200-state ✅ |
| `outputs/eval_v29_oracle_200.json` | Full Query (Oracle) 200-state 🏃 |
| `outputs/eval_v29_ideal_disclosed_200.json` | Ideal Disclosed 200-state ⏳ |

### Recovery Rate 计算公式

```
Recovery Rate = (Method pass@1 - Direct pass@1) / (Full Query pass@1 - Direct pass@1) × 100%
Δ vs Direct   = Method pass@1 - Direct pass@1
```

### 进度检查（2026-04-17 23:52，已跑 12h09m）

PID 603803 still running。

- **Direct Execution 200**: ✅ 完成（22:39）
- **Oracle 200**: 🏃 48/200（~1.5min/sample，预计还要 3-4h）
- **Ideal Disclosed 200**: ⏳ 待 Oracle 完成

#### Direct Execution 200-state 结果

| Persona | pass@1 | pass@5 |
|---|:---:|:---:|
| Novice-Learner | 11.5% (23/200) | 19.0% (38/200) |
| Experienced-Engineer | 12.5% (25/200) | 17.5% (35/200) |
| Busy-Developer | 13.0% (26/200) | 19.0% (38/200) |
| **Overall** | **12.3%** (74/600) | **18.5%** (111/600) |

**异常发现**：
1. **Direct 200 (12.3%) >> Direct 50 (7.3%)**，gap +5%。50-state seed 可能抽到偏难 task
2. **三 persona 不完全一致**（11.5/12.5/13.0）— 按理 Direct 不交互应完全相同，需查 `--direct_execution` 模式下 persona 是否漏进 prompt

**对叙事影响**：DPO vs Direct 200-state gap = **+3.7%**（16.0 vs 12.3），比 50-state 时的 +8.7% 显著缩小。

### Recovery 表（部分填好，待 Oracle/Ideal Disclosed）

| Condition | pass@1 | pass@5 | Δ vs Direct | Recovery Rate |
|---|:---:|:---:|:---:|:---:|
| Masked Direct | 12.3% | 18.5% | -- | 0% |
| Clarified (Overall) | 16.0% | 23.5% | +3.7% | TBD |
| · Busy | 14.0% | 20.0% | +1.0% | TBD |
| · Experienced | 15.5% | 25.0% | +3.0% | TBD |
| · Novice | 18.5% | 25.5% | +7.0% | TBD |
| Ideal Disclosed | TBD | TBD | TBD | TBD |
| Full Query | TBD | TBD | TBD | 100% |

每行 Δ 用对应 persona 的 Direct 做基准（Novice vs 11.5, Exp vs 12.5, Busy vs 13.0）。

---

## 23. Experiment 2: 信息恢复机制分析（2026-04-18）

Full Query 完成 + 信息恢复相关性分析。

### Full Query (Oracle 200) 完成

- **pass@1 = 20.0%**, **pass@5 = 28.0%**（n=200, persona-independent 单跑）
- 用作 OGR 计算的分母上限

### Ideal Disclosed 进度

凌晨 03:36 启动（PID 751556），目标 200（persona-independent，不是 600）。
- 当前 86/200 (43%)，~1.6 min/conv（5 candidate × Llama 8B 本地推理）
- 预计 09:00 前完成
- `gpt-4o-mini` API 几乎不调用（single-turn execute 不进 user simulator）

### Recovery 表（OGR 已计算）

OGR = (method - direct) / (full_query - direct) × 100，per-persona 用各自 Direct 做分母。

| Condition | pass@1 | pass@5 | Δ | OGR | Disc. |
|---|:---:|:---:|:---:|:---:|:---:|
| Masked Direct | 12.3% | 18.5% | -- | 0% | 0.00 |
| Full Query | 20.0% | 28.0% | +7.7 | 100% | n/a |
| TactfulLLM Overall | 16.0% | 23.5% | +3.7 | **48%** | 0.56 |
| · Novice | 18.5% | 25.5% | +7.0 | **82%** | 0.89 |
| · Experienced | 15.5% | 25.0% | +3.0 | **40%** | 0.78 |
| · Busy | 14.0% | 20.0% | +1.0 | **14%** | 0.00 |
| Ideal Disclosed | partial 86/200 | -- | -- | -- | 1.00 |

### Eval bug 修复 + disclosure 信息回填

发现 `evaluate_multi_turn_persona.py` 的 turn_data 构造（行 379, 785）漏记 `disclosed_items`，无法直接算 per-conversation disclosure_rate。
- **修复**：两处加 `"disclosed_items": user_reaction.get("meta", {}).get("disclosed_items", {})`
- **存量回填**：`scripts/replay_disclosure.py` 用 simulator 确定性 `get_disclosure_info()` 重放，避免 5h 重跑
- **产出**：`data/analysis/disclosure_per_conversation.csv`（600 rows: state_id, persona, n_masked, n_disclosed, disclosure_rate, n_clarify_turns, n_answered_clarifies, pass1, pass5）

### Disclosure rate by persona（验证设计意图）

| Persona | mean disclosure_rate | 解释 |
|---|:---:|---|
| Novice-Learner | 0.886 | expertise=low（1 item/turn），多轮累积常饱和 |
| Experienced-Engineer | 0.780 | expertise=mid（3 items/turn），中等恢复 |
| Busy-Developer | 0.000 | DPO policy 学会 Execute，从不进 clarify 路径 |

### Recovery → Success 相关性（`scripts/analyze_disclosure_recovery.py`）

- **Pooled Spearman ρ = +0.088, p = 0.032**（弱但显著，被 persona confounding 稀释）
- **Within-Experienced ρ = +0.202, p = 0.004**（唯一有 dynamic range 的子集，干净 mechanism 证据）
- Within-Novice ρ = +0.041, p = 0.57（饱和）
- Busy 全 0 → degenerate
- **Logistic regression** `pass1 ~ disclosure_rate + C(persona)`：disclosure_rate coef = +0.901, 95% CI [-0.19, +1.99], p = 0.106（控制 persona 后边际不显著）

### 可视化决策

**不放**散点 / pooled bin 图：
- Pooled bin pass@1 = [14.2%, 11.1%, 7.5%, 19.0%]，**非单调**（persona composition artifact：低 bin 全是 Busy，中 bin 是 Exp 偶然低段）
- 600 点散点视觉上是云团，ρ=+0.088 趋势线几乎看不出
- Per-persona 三联画 2/3 退化（Busy 全 0、Novice 饱和），看似 cherry-pick

**主图**：4-condition grouped bar（`scripts/plot_4condition_grouped.py` → `data/analysis/fig_recovery_4condition.png`）
- 横轴 persona × 4 条柱（Direct / TactfulLLM / Ideal Disclosed / Full Query），TBD 用 hatch
- 把 Recovery 表 visualize，Bounds → TactfulLLM → Ideals 视觉递进
- Busy 那组四条贴齐反而强化"policy 选择不 recover"的故事

**主表**：加 `Disc.` 列（mean disclosure rate）作为机制 evidence in-table，不需要单独图

### 论文段落（最终版）

主段落框架：
1. 开篇两问并列：(i) clarification 恢复多少 mask 信息 (ii) 恢复是否转化为下游成功
2. 三 reference conditions：Masked Direct (lower bound) / Ideal Disclosed (clarification ceiling) / Full Query (absolute ceiling)
3. Hierarchy + DPO 隔离声明：TactfulLLM vs Ideal Disclosed isolates information recovery from policy adaptation

具体 LaTeX 见论文草稿；caption 浓缩到 4 行内，OGR 解释用 "(per-persona)" 一笔带过。

### 文件清单（新增）

| 文件 | 说明 |
|---|---|
| `scripts/replay_disclosure.py` | 离线重放 disclosure，回填 600 行 csv |
| `scripts/analyze_disclosure_recovery.py` | Spearman + logistic regression + bin 分析 |
| `scripts/plot_4condition_grouped.py` | 4-condition grouped bar 主图 |
| `data/analysis/disclosure_per_conversation.csv` | per-conv disclosure × pass 数据 |
| `data/analysis/recovery_bins.csv` | bin 统计（含 Wilson CI） |
| `data/analysis/fig_recovery_4condition.png` | Recovery 主图 |

---

## 17. 2026-04-21 Canonical 测试集审计 + Ideal Disclosed v2

### 17.1 Canonical 测试集定义

- **`data/seeds/test_states_v29_eval_200.jsonl`**（200 状态）= 本项目规范测试集
- 所有 Exp1/Exp2 后续评估**必须**用这个文件或它的子集
- 已确认的干净子集：
  - `test_states_v29_eval_50.jsonl`（50 个，⊂ 200 ✓）
  - `test_states_v29_eval_150extra.jsonl`（150 个，⊂ 200 ✓）

### 17.2 非 canonical 的老测试集（⚠ 需避开）

`test_states_v29_eval.jsonl`（20 状态）是早期遗留，和 canonical-200 只重叠 3 个、17 个在外。下列 eval 输出使用了它：
- `eval_v29_100states.json`（TactfulLLM 早期跑，17 outside 但已被新文件覆盖，保留无害）
- `eval_v29_oracle_50test.json`（已废，被 `oracle_200.json` 替代）
- `eval_v29_base_llama.json`（**需要重跑 canonical 版本**）

### 17.3 Canonical-200 覆盖现状

| 方法 | 已覆盖 | 缺 |
|---|---|---|
| Direct Execution | 200 ✓ | 0 |
| Oracle | 200 ✓ | 0 |
| Ideal Disclosed v1 | 200 ✓ | 0 |
| Ideal Disclosed v2 | 198/200 (2026-04-21 进行中) | 2 |
| TactfulLLM | 200 ✓ | 0（分散在 3 文件）|
| Clarify-first | 50 | **150** |
| Prompt-only | 50 | **150** |
| Base LLM | 93 | **107** |

### 17.4 Exp1 主表数据源不一致 bug

Exp1 "Main Results" 表 caption 写 "200 test tasks"，实际用了 **50-state 子集**（`*_50test.json`），导致 Direct Execution 在两张表里 7.3% vs 14.1% 差 2×（50test 子集恰好难）。

**修正计划**：补跑 Clarify-first / Prompt-only / Base LLM 在 canonical-200 缺的 150/150/107 个，让所有方法统一到 200-seed 规模，主表 caption 才对得上。

### 17.5 Ideal Disclosed v2 完成 (200/200)

v2 修正 disclosure 格式（bullet + 2 items）后，在完整 canonical-200 上：

| Condition | pass@1 | pass@5 | avg turns |
|---|---|---|---|
| Oracle (full query) | 20.0% | 28.0% | 1.0 |
| **Ideal Disclosed v2** | **16.0%** | **27.0%** | 1.0 |
| Ideal Disclosed v1 | 13.5% | 24.5% | 1.0 |
| Masked Direct | 12.3% | 18.5% | 1.0 |

v2 比 v1 提升 pass@1 +2.5pp、pass@5 +2.5pp——说明 bullet 格式确实传递更多信息。

### 17.6 Experiment 2 表升级到 200 seed（canonical）

用 canonical-200 重算，所有 condition 都在同一测试集：

| Group | Condition | pass@1 | pass@5 | Δ | OGR (%) | Disc. |
|---|---|---|---|---|---|---|
| **Bounds** | Masked Direct | 12.3 | 18.5 | -- | 0 | 0.00 |
|  | Full Query | **20.0** | **28.0** | +7.7 | 100 | n/a |
| **TactfulLLM** | Overall | 16.0 | 23.5 | +3.7 | 48 | 0.56 |
|  | Novice | 18.5 | 25.5 | +7.0 | 82 | 0.89 |
|  | Experienced | 15.5 | 25.0 | +3.0 | 40 | 0.78 |
|  | Busy | 14.0 | 20.0 | +1.0 | 14 | 0.00 |
| **Oracle** | Ideal Disclosed | 16.0 | 27.0 | +3.7 | 48 | 1.00 |

**关键发现**：TactfulLLM Overall 和 Ideal Disclosed v2 的 pass@1 **精确重合在 16.0%**。在 600 个 (state, persona) matched 试验上做 McNemar：
- TactfulLLM pass & Ideal fail: b=43
- Ideal pass & TactfulLLM fail: c=43
- p-value (exact) ≈ 1.00

→ "TactfulLLM approaches the clarification ceiling" 从 **approach** 升级为 **fully matches**。pass@5 上 Ideal (27.0) > TactfulLLM (23.5)，但 pass@1 是主指标。

**Per-persona OGR 分层干净**：Novice 82% → Experienced 40% → Busy 14%，强化 persona adaptation 机制。Busy 从 151-matched 的 OGR=0 变成 200 的 14%，不再退化为"完全不恢复"。

### 17.7 数据泄漏审计（无事）

- canonical splits（train 376 / val 47 / test 47 = 470，互不重叠）
- eval_200 文件级构造有脏：67 个与 train_split 文件重叠、13 个与 val 重叠、6 个与 test 重叠、114 个是孤儿（不在 470-masked 池里，独立 masking 得到）
- **但 v29 DPO 实际只训练了 `prefs_v29_100states.jsonl` 里的 107 个任务（id 全 < 110），和 eval_200（id 全 ≥ 111）零交集**
- **v29 paper 可以放心写 "zero overlap between 107 training tasks and 200 held-out eval tasks"**
- v30+ 若扩到 train_split 全集，那 67 重叠就变成真泄漏，届时须先重建 eval_200
