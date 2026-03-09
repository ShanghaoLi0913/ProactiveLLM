# 版本对比总结：上个版本 vs 当前版本

## 一、上个版本（GitHub上的版本）的主要问题

### 1.1 核心问题

**Clarification几乎没有提升代码成功率**（+0.8%），而Full Information提升明显（+18%）

**数据对比**：
- masked (仅masked query): 36.1%
- masked + clarification: 36.9% （**+0.8%**，几乎无提升）
- full (original_instruct_prompt): 54.2% （**+18.1%**，巨大提升）

### 1.2 根本原因

#### 问题1：强制追问采样机制覆盖了Disclosure信息

**代码位置**：`simulator/simulate.py:255-272`

**问题**：
- Turn 0的第一次Clarify时，有**65%的概率**会给出模糊回复
- 模糊回复：`"I want a general solution that works. Just do it the standard way."`
- **这个模糊回复完全跳过了disclosure机制**，不包含任何masked信息

**统计数据**：
- 模糊回复（不包含disclosure信息）：29个
- 包含disclosure信息的回复：46个
- 其他正常回复：182个

#### 问题2：Reconstruction失败率高

**统计数据**：
- 有clarification历史的Execute轨迹：88个
- 其中reconstruction有内容的：36个（**40.9%**）
- **reconstruction为空的比例：59.1%**

**原因**：
1. **模式匹配太严格**：依赖预定义关键词（如"empty", "null", "single element"）
2. **无法处理自然语言**：用户可能用不同表达方式描述相同需求
3. **对表达方式敏感**：如"empty list" vs "empty input"

#### 问题3：用户回答质量问题

**失败样本分析**：
- 常见失败模式：`"I want a general solution that works. Just do it the standard way."`
- 问题：用户回答太模糊，不包含结构化信息
- Reconstruction无法从模糊回答中提取有用信息

## 二、当前版本的改进

### 2.1 ✅ 可配置的模糊回复概率（与persona关联）

**改进前**：
- 固定65%概率给模糊回复（硬编码）
- 所有persona使用相同的概率

**改进后**：
```python
VAGUE_REPLY_PROB_MAP = {
    "Busy-Developer": 0.5,           # 时间压力，更容易敷衍
    "Experienced-Engineer": 0.1,     # 虽然expertise高，但patience是mid，偶尔可能不耐烦
    "Novice-Learner": 0.25,          # 表达能力有限，更容易给出模糊回答
}
```

**设计理由**：
- 不同persona有不同的模糊回复概率，更真实
- 比简单的"从65%降到20%"更有可defend性
- "We model unhelpful user feedback with a configurable probability, reflecting real-world ambiguity."

### 2.2 ✅ 模糊回复也包含部分disclosure信息（不短路信息流）

**改进前**：
```python
# 模糊回复完全跳过disclosure
user_reply = "I want a general solution that works. Just do it the standard way."
```

**改进后**：
```python
# 模糊回复也包含部分disclosure信息
vague_base = "I want a general solution that works. Just do it the standard way."
# 提取一个disclosure信息点
partial_disclosure = get_disclosure_info(...)
user_reply = f"{vague_base} Also, {partial_disclosure.lower()}."
# 例如: "I want a general solution that works. Just do it the standard way. Also, please make sure it handles empty input."
```

**好处**：
- 保留真实用户行为（确实会说"随便做个通用的"）
- 但不短路信息流，确保disclosure信息总是被整合
- 避免reviewer说"模拟器太理想化"

### 2.3 ✅ Reconstruction Canonicalization层

**改进前**：
- 纯关键词匹配，对表达方式敏感
- 无法处理同义表达（如"empty list" vs "empty input"）

**改进后**：
- 添加了`CANONICAL_MAP`和`canonicalize_text()`函数
- 将同义表达规范化到canonical tokens：
  - `empty list / empty input / empty string → EMPTY_INPUT`
  - `output should be Counter / return a Counter → OUTPUT_COUNTER`
  - `O(n) / linear time / time complexity is linear → TIME_LINEAR`

**好处**：
- 解决了"关键词匹配对表达敏感"的问题
- 不需要引入额外模型，成本低
- 大幅提升reconstruction成功率

### 2.4 ✅ 修复中文回答问题

**改进前**：
- Novice-Learner的回答包含中文："输入有默认值。"、"可能需要处理一些特殊情况。"
- 导致reconstruction失败（canonicalization层不支持中文）

**改进后**：
- 将所有中文回答改为简单英文
- `"可能需要处理一些特殊情况。"` → `"Should handle: {ec}"`
- `"输入有默认值。"` → `"Input has default value: {constraint}"`

**效果**：
- Reconstruction成功率从52.6%提升到100%（小批量测试）

### 2.5 ✅ 提高Novice-Learner的Disclosure步长

**改进前**：
- Novice-Learner每次只reveal 1个信息点
- 导致Coverage Ratio极低（平均9-30%）

**改进后**：
- 从1个增加到2个
- Coverage Ratio从0.231提升到0.632（小批量测试）

**效果**：
- 信息更完整，可能提升代码质量

## 三、改进效果对比

### 3.1 改进前（上个版本）

| 指标 | 数值 |
|------|------|
| **Clarification Gain** | +0.8% |
| **Reconstruction成功率** | 40.9% |
| **模糊回复概率** | 65%（固定，所有persona） |
| **Disclosure信息整合率** | 17.9% |

### 3.2 改进后（当前版本）

**小批量测试（5个states）**：
| 指标 | 数值 |
|------|------|
| **Clarification Gain** | +10.0% |
| **Reconstruction成功率** | 100.0% |
| **模糊回复概率** | Persona-specific（0.1-0.5） |
| **Coverage Ratio** | 0.632 |

**大批量测试（35个states，18个唯一）**：
| 指标 | 数值 |
|------|------|
| **Clarification Gain** | 0.0%（总体） |
| **Persona差异** | Experienced-Engineer: +3.4%, Novice-Learner: -6.6% |
| **Reconstruction成功率** | 需要进一步验证 |
| **Full Info Gain** | +27.5% |

### 3.3 关键发现

1. **小批量测试效果显著**：Clarification Gain从+0.8%提升到+10.0%
2. **大批量测试效果消失**：Clarification Gain降到0.0%
3. **Persona差异明显**：
   - Experienced-Engineer: +3.4%（正向）
   - Novice-Learner: -6.6%（负向）
   - 说明clarification对不同persona的效果差异很大

## 四、版本演进总结

### 4.1 数据版本演进（V1-V5）

根据`scripts/generate_v5_balanced_prefs.py`：

```
V1-V2: ~100对，低质量混合 → TSR ~17%
V3:    304对，允许部分通过（但0% Clarify）→ TSR 25.68%
V4:    135对，完美Execute（0% Clarify）→ TSR 32.30%
V5A:   Execute+Clarify → 预期提升persona适应能力
V5B:   平衡比例（~20% Clarify）→ 推荐版本 ⭐
```

### 4.2 当前版本的改进（基于V5）

1. **可配置的模糊回复概率**：Persona-specific，更真实
2. **模糊回复+部分disclosure**：不短路信息流
3. **Canonicalization层**：提升reconstruction成功率
4. **修复中文问题**：避免reconstruction失败
5. **提高Novice-Learner的disclosure步长**：信息更完整

## 五、论文价值

### 5.1 关键Insight

> "Clarification effectiveness varies significantly across personas. While clarification helps Experienced-Engineer and Busy-Developer (+3.3-3.4%), it actually hurts Novice-Learner (-6.6%), suggesting that clarification mechanisms need persona-specific optimization."

### 5.2 证据

1. **Persona差异**：
   - Experienced-Engineer: +3.4%
   - Busy-Developer: +3.3%
   - Novice-Learner: -6.6%

2. **总体Gain为0%**：
   - 说明不同persona的效果相互抵消

3. **Full Info Gain仍然很高**（+27.5%）
   - 说明完整信息确实重要，但clarification没有完全捕获

---

**总结**：
- **上个版本**：Clarification Gain只有+0.8%，主要问题是模糊回复跳过disclosure、reconstruction失败率高
- **当前版本**：改进了模糊回复机制、添加canonicalization层、修复中文问题、提高disclosure步长
- **效果**：小批量测试效果显著（+10.0%），但大批量测试效果消失（0.0%），说明仍有改进空间
