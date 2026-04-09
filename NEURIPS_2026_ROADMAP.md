# NeurIPS 2026 论文路线图
## 目标：多轮对话 + 3 Persona + 平衡 Task Success 与用户偏好

**投稿目标**：NeurIPS 2026（Abstract ~5月16日，Full paper ~5月23日，会议12月）  
**核心贡献**：Persona-Aware Proactive LLM with Multi-turn Trajectory-Level DPO

---

## 📊 当前状态（更新于 2026-04-06）

### ✅ 已完成
1. **多轮对话框架**：3轮设计，trajectory-level rewards
2. **Persona 系统**：3个 persona（Busy / Experienced / Novice）
3. **DPO 训练流程**：Persona-aware 训练已实现，v21–v26 共6个版本
4. **用户模拟器**：支持 patience decay 和 expertise-based clarity
5. **三个关键 Bug 修复**（2026-04-06）：
   - action selection 改为 generation-based（`pick_action_from_generation`）
   - DPO 训练数据 strip action 前缀
   - rebalance 改为 state-aligned 版本
6. **v26 训练完成**：beta=0.05, epoch=5；Busy Execute 率达 77%（quick check）

### ❌ 当前阻塞问题
1. **Experienced 与 Novice 无法区分**：`reward/compute_rewards.py` 的 `persona_adjustment` 未区分 `dialogue_turn`，Experienced 在 turn=1 时仍给 Clarify 更高 reward（应切换到 Execute）
2. **Task Success Rate ≈ 0%**：模型大量 Clarify，代码生成极少
3. **Experienced turn=1 训练数据极少且方向错误**：仅 9 条，且 8/9 为 Clarify chosen

---

## 🎯 论文核心指标（必须达到）

### 1. Persona 差异（关键创新点）
- **Busy-Developer**：Execute 率 >80%，平均轮次 <1.3
- **Experienced-Engineer**：Execute 率 50–70%，平均轮次 1.4–1.6
- **Novice-Learner**：Execute 率 <50%，平均轮次 >1.7

### 2. Task Success Rate
- Execute 动作：success rate >60%
- Overall：success rate >50%
- Novice 通过 Clarify 获得比直接 Execute 更高的 success rate

### 3. 平衡性指标
- Reward 分布：不同 persona 有明显差异
- Action Accuracy：>70%

---

## 🚀 当前优先级（4–5月冲刺）

### Step 1：修复 reward 函数 ← 阻塞所有后续

**文件**：`reward/compute_rewards.py`，函数：`persona_adjustment`

**修复内容**：Experienced-Engineer + `dialogue_turn >= 1` 时，Execute reward > Clarify reward。

**完成后**：重新生成轨迹 → 重新计算 reward → 重新 rebalance → 重新训练。

### Step 2：重新生成训练数据

```bash
python scripts/generate_trajectories.py --n_states 150   # 修复后跑
python reward/compute_rewards.py
# rebalance 用 state-aligned 版本
```

可选：扩展到 500 states 以增强信号（需额外 2–3 小时）。

### Step 3：训练新版本（v27+）

沿用 v26 参数：beta=0.05, epoch=5, lr=1e-5

### Step 4：完整多轮评估

```bash
python eval/evaluate_multi_turn_persona.py --model_dir outputs/proactive_llm_v27 ...
```

验证三个 persona 的行为差异（Busy >> Experienced > Novice）。

### Step 5：论文撰写（并行推进）

先写 Method 和 Experiment setup，等实验结果填数字。NeurIPS 截稿前 2 周留给写作润色。

---

## 🔧 关键技术细节

### Reward 公式

```python
w_interrupt_persona = {
    "Busy-Developer": 0.4,
    "Experienced-Engineer": 0.2,
    "Novice-Learner": 0.1,
}

R_base = pass_rate - w_interrupt_persona[persona] * interrupt_cost

# persona_adjustment（待修复）：
# Experienced + dialogue_turn >= 1 → Execute bonus
```

### Action Detection（v24+）

生成 30 token，检测开头是否为代码标志（` ``` `、`def`、`import`）→ Execute，否则 → Clarify。

### 训练数据构建

1. 生成轨迹（`scripts/generate_trajectories.py`）
2. 计算 reward（`reward/compute_rewards.py`）
3. State-aligned rebalance：找三个 persona 都有 pair 的 state 交集，每个 state 每个 persona 保留最优 pair
4. 输出：`data/dpo/prefs_method_abc_Nstates_aligned.jsonl`

### Base Model

`/root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/`

---

## 📈 成功标准

### Hard Requirements
- [ ] Persona Execute 率差异 >15%（Busy vs Novice）
- [ ] Task success rate >50%
- [ ] 明显优于 baseline（至少 2 个指标）

### Nice to Have
- [ ] Task success rate >60%
- [ ] Execute 率差异 >20%
- [ ] Action accuracy >75%
- [ ] 统计显著性（p<0.05）

---

## 🚨 风险与应对

| 风险 | 应对 |
|------|------|
| Persona 差异不明显 | 增大 w_interrupt 差异；调整 dialogue_turn reward |
| Task success rate 太低 | 检查代码生成质量；增加 best-of-N 采样 |
| 时间不够（截稿 5月） | 优先 reward 修复 + 训练，论文同步写 Method |

---

## 📅 里程碑

| 日期 | 目标 |
|------|------|
| 2026-04-10 | reward 修复 + 新数据生成完成 |
| 2026-04-15 | v27 训练完成，persona 差异验证 |
| 2026-04-30 | 完整评估通过，所有指标达标 |
| 2026-05-10 | 论文初稿完成 |
| 2026-05-16 | NeurIPS abstract 提交 |
| 2026-05-23 | NeurIPS full paper 提交 |

---

**最后更新**：2026-04-06  
**状态**：v26 评估中；reward bug 待修复后重新生成数据
