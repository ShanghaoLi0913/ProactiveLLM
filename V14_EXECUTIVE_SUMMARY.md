# V14 Executive Summary: 训练成功但格式错误

**日期**: 2026-02-09  
**状态**: ⚠️ 需要修复并重新训练V15

---

## 🎯 一句话总结

**V14的数据和训练过程都是成功的，但由于没有使用Llama-3.1-Instruct的chat template，模型学会了生成prompt延续而不是response。**

---

## ✅ 成功的部分

### 1. 数据生成 (100%成功)
```
✅ Trajectory-level rewards
✅ Persona-aware preferences  
✅ Balanced distribution (6.9%-15.5% Clarify)
✅ No train/test overlap
✅ 143 train prefs, 38 test prefs
```

### 2. DPO训练 (技术成功)
```
✅ Loss: 0.619 → 0.488 (↓21%)
✅ Accuracy: 38.7% → 69.4% (↑80%)
✅ Margins: -0.006 → +0.428
✅ 训练时间: 4分4秒
```

---

## ❌ 失败的部分

### 模型生成 (完全失败)
```
❌ Action Accuracy: 0%
❌ Task Success Rate: 0%
❌ 模型生成prompt延续，不是response
```

**示例：**
```
Input: "[Domain] coding\n[Persona] Novice-Learner\n...\n[User Request]\nFind the best-selling product..."

Expected: "```python\ndef task_func():..."

Actual: "to find the best-selling product from a given CSV file with sales data. 
[User Request]
Find the best-selling product..."  ← 继续生成prompt!
```

---

## 🔍 根本原因

**缺少Chat Template**

Llama-3.1-Instruct是用这个格式训练的：
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Your message here<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Response starts here ← 模型知道从这里开始生成
```

V14的格式（错误）：
```
[Domain] coding
[Persona] Novice-Learner
...
[User Request]
write a function...```python
def task_func():... ← 没有边界标记！
```

**结果**: 模型不知道prompt在哪里结束，response从哪里开始。

---

## 🚀 解决方案 (V15)

### 修改代码（简单）

**`policy/train_dpo.py`:**
```python
# 修改 to_dpo_format 函数
def to_dpo_format(records, tokenizer):  # ← 添加tokenizer参数
    for ex in records:
        state_text = render_state(state_with_persona)
        
        # ✅ 使用chat template
        messages = [
            {"role": "user", "content": state_text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True  # ← 添加<|start_header_id|>assistant<|end_header_id|>
        )
        
        dataset["prompt"].append(prompt)
        dataset["chosen"].append(ex["chosen_assistant_msg"])
        ...
```

### 预期结果
```
V14 (现在):     V15 (预期):
Action: 0%  →   Action: >50%
Task: 0%    →   Task: >30%
```

### 工作量
```
修改代码: 30分钟
重新训练: 5分钟
评估验证: 10分钟
──────────────────
总计: ~45分钟
```

---

## 💡 关键教训

1. **Fine-tuning instruction-tuned models必须使用其chat template**
2. **数据质量 < 训练格式** - 再好的数据，格式错了也白搭
3. **仔细评估很重要** - 我们发现了3个不同的评估bug
4. **失败也有价值** - V14的数据生成流程是正确的，可以直接用于V15

---

## 📊 V14数据统计（可重用）

```
Persona             Train (Clarify%)    Test (Clarify%)
─────────────────────────────────────────────────────
Busy-Developer         6.9% (2/29)        0.0% (0/5)
Experienced-Engineer  10.7% (6/56)        6.7% (1/15)
Novice-Learner        15.5% (9/58)       22.2% (4/18)
─────────────────────────────────────────────────────
Total                 11.9% (17/143)     13.2% (5/38)
```

**✅ 这些数据质量很高，可以直接用于V15训练！**

---

## 🎯 下一步

**立即修复并训练V15**

1. 修改`policy/train_dpo.py`添加chat template
2. 修改`eval/evaluate_v13_persona.py`使用chat template
3. 重新训练（使用相同的数据）
4. 评估并验证

**预期**: V15将是第一个真正工作的persona-aware proactivity model！

---

## 📂 关键文件

**数据** (可重用):
- `data/dpo/prefs_100states_balanced_train.jsonl`
- `data/dpo/prefs_100states_balanced_test.jsonl`

**需要修改**:
- `policy/train_dpo.py`
- `eval/evaluate_v13_persona.py`

**详细文档**:
- `outputs/eval_results/V14_CRITICAL_ISSUE_SUMMARY.md`
