# 方案A训练进行中

## 训练配置

- **训练数据**: `data/dpo/prefs_150_taskdom_v2.jsonl` (138个preference pairs)
- **Base模型**: `meta-llama/Llama-3.1-8B-Instruct`
- **输出目录**: `outputs/policy_scheme_a_150`
- **训练参数**:
  - Epochs: 3
  - Learning rate: 5e-5
  - Beta: 0.1
  - Batch size: 1 (gradient accumulation: 16)

## 训练目标

**Scheme A: 分离架构**
- Policy模型只学习预测action token (LOW/MID/HIGH)
- Code generation由独立模型处理（不受DPO影响）
- 避免语言分布污染

## 监控训练

```bash
# 查看训练日志
tail -f /tmp/train_scheme_a.log

# 检查训练进度
ps aux | grep train_dpo

# 检查输出目录
ls -lh outputs/policy_scheme_a_150/
```

## 训练完成后

1. **评估模型**:
   ```bash
   export OPENAI_API_KEY=sk-...
   python eval/evaluate_dpo_model.py \
       --model_dir outputs/policy_scheme_a_150 \
       --base_model meta-llama/Llama-3.1-8B-Instruct \
       --prefs data/dpo/prefs_150_taskdom_v2.jsonl \
       --max_samples 50 \
       --output data/eval/scheme_a_results.json
   ```

2. **对比结果**:
   - 与之前的模型对比（policy_taskdom, policy_instruct_150）
   - 检查action准确率是否提升
   - 检查代码质量是否改善（因为使用独立code generation）

## 预期改进

1. **Action准确率**: 应该提升（因为训练目标更明确）
2. **代码质量**: 应该显著提升（因为code generation不受DPO污染）
3. **Task Success Rate**: 应该接近或超过baseline的42%


