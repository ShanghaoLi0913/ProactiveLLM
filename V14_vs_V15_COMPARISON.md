# V14 vs V15: 格式修复对比

## 问题：V14的训练格式

### DPO训练时模型看到的：
```
[Domain] coding
[Persona] Novice-Learner
[Task Uncertainty] 0.80
[Dialogue Turn] 0
[Previous Reject] 0
[User Request]
write a python function...```python
def task_func():
    # implementation
    return result
```
```

**问题**: 
- ❌ 没有明确的prompt/response边界
- ❌ 模型不知道从哪里开始生成
- ❌ 结果：模型生成prompt延续

### V14生成示例：
```
Input: [Domain] coding\n[Persona] Novice\n[User Request]\nFind best product...

Generated:
"to find the best-selling product from a given CSV file with sales data. 
[User Request]
Find the best-selling product from a given CSV file..." 
← 继续扩展prompt！
```

---

## 解决方案：V15的训练格式

### DPO训练时模型看到的：
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

[Domain] coding
[Persona] Novice-Learner
[Task Uncertainty] 0.80
[Dialogue Turn] 0
[Previous Reject] 0
[User Request]
write a python function...<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```python
def task_func():
    # implementation
    return result
```
```

**优势**:
- ✅ 明确的边界标记：`<|eot_id|><|start_header_id|>assistant<|end_header_id|>`
- ✅ 模型知道从special token后开始生成
- ✅ 符合Llama-3.1-Instruct的原始训练格式

### V15生成示例（预期）：
```
Input: <|begin_of_text|><|start_header_id|>user<|end_header_id|>
       [Domain] coding
       [Persona] Novice
       [User Request]
       Find best product...<|eot_id|>
       <|start_header_id|>assistant<|end_header_id|>

Generated:
"```python
import csv
import collections

def task_func(csv_file_path):
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        sales = collections.Counter()
        for row in reader:
            sales[row['product']] += int(row['quantity'])
        return sales.most_common(1)[0][0]
```"
← 正确生成代码！
```

---

## 代码修改对比

### V14 (错误的实现)

```python
# policy/train_dpo.py
def to_dpo_format(records):
    dataset = {"prompt": [], "chosen": [], "rejected": []}
    
    for ex in records:
        state_with_persona = ex["state"].copy()
        if "persona" in ex:
            state_with_persona["persona"] = ex["persona"]
        
        # ❌ 直接使用render_state，没有边界标记
        dataset["prompt"].append(render_state(state_with_persona))
        dataset["chosen"].append(ex["chosen_assistant_msg"])
        dataset["rejected"].append(ex["rejected_assistant_msg"])
    
    return Dataset.from_dict(dataset)
```

### V15 (正确的实现)

```python
# policy/train_dpo.py
def to_dpo_format(records, tokenizer):  # ← 添加tokenizer参数
    dataset = {"prompt": [], "chosen": [], "rejected": []}
    
    for ex in records:
        state_with_persona = ex["state"].copy()
        if "persona" in ex:
            state_with_persona["persona"] = ex["persona"]
        
        # ✅ 使用chat template
        state_text = render_state(state_with_persona)
        messages = [
            {"role": "user", "content": state_text}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ← 添加assistant header
        )
        
        dataset["prompt"].append(prompt)
        dataset["chosen"].append(ex["chosen_assistant_msg"])
        dataset["rejected"].append(ex["rejected_assistant_msg"])
    
    return Dataset.from_dict(dataset)
```

**关键差异**：
1. 添加`tokenizer`参数
2. 将state包装成`messages`格式
3. 使用`tokenizer.apply_chat_template()`
4. 设置`add_generation_prompt=True`

---

## 评估代码修改

### V14 (错误的评估)

```python
# eval/evaluate_v13_persona.py
prompt = render_state(state_with_persona)  # ← 没有chat template

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
```

### V15 (正确的评估)

```python
# eval/evaluate_v13_persona.py
state_text = render_state(state_with_persona)
messages = [{"role": "user", "content": state_text}]

# ✅ 使用chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
```

---

## 预期结果对比

| 指标 | V14 (现在) | V15 (预期) | 改进 |
|------|-----------|-----------|------|
| **Action Accuracy** | 0% | >50% | +50%+ |
| **Task Success Rate** | 0% | >30% | +30%+ |
| **生成质量** | Prompt延续 | 真实Response | ✅ |
| **训练时间** | 4分钟 | 4分钟 | 相同 |
| **数据** | 143/38 prefs | 143/38 prefs | 相同 |

---

## 迁移检查清单

### 需要修改的文件：

- [ ] `policy/train_dpo.py`
  - [ ] 修改`to_dpo_format`函数签名
  - [ ] 添加chat template逻辑
  - [ ] 在`main()`中传递tokenizer

- [ ] `policy/train_dpo.py` (main函数)
  - [ ] 在加载数据前初始化tokenizer
  - [ ] 调用`to_dpo_format(records, tokenizer)`

- [ ] `eval/evaluate_v13_persona.py`
  - [ ] 在生成prompt时使用chat template
  - [ ] 确保与训练时格式一致

### 不需要修改的：

- ✅ 数据文件 (`prefs_100states_balanced_*.jsonl`)
- ✅ `render_state.py` (state渲染逻辑)
- ✅ `reward/compute_rewards.py` (reward计算)
- ✅ `scripts/generate_trajectories.py` (trajectory生成)

---

## 风险分析

### 低风险 ✅
- Chat template是标准做法
- Llama-3.1-Instruct就是用这个格式训练的
- 代码修改很小（<20行）

### 中等风险 ⚠️
- Chat template会增加prompt长度（~100 tokens）
  - 影响：context window可能更快填满
  - 缓解：当前state很短，影响有限

- 需要在训练和评估时保持格式一致
  - 影响：格式不一致会导致性能下降
  - 缓解：使用相同的chat template函数

### 高风险 ❌
- 无

---

## 总结

### V14的问题：
```
没有prompt/response边界 → 模型不知道该生成什么 → 生成prompt延续
```

### V15的解决：
```
使用chat template → 明确的边界标记 → 模型生成真实response
```

### 关键insight：
**Fine-tuning instruction-tuned models时，保持原始训练格式至关重要！**

---

## 时间线

```
V13: Persona-aware DPO (有bug)
  ↓
V14: Trajectory-level + 修复persona bug (数据✅, 格式❌)
  ↓
V15: 添加chat template (预期全部✅)
  ↓
未来: 在更多states上训练，提高性能
```

**当前状态**: V14完成，准备V15  
**预计完成**: V15修复后45分钟
