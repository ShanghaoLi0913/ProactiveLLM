# Git Commit Message 建议

## 推荐版本（简洁版）

```
feat: implement Scheme A separated architecture for action prediction and code generation

- Separate policy model (action prediction) from code generation
- Policy model only learns to predict action tokens (LOW/MID/HIGH)
- Code generation handled by independent model (not affected by DPO)
- Add unified render_state() function for consistent training/inference
- Update training, inference, and evaluation code for separated architecture
- Add comprehensive documentation (ARCHITECTURE_SCHEME_A.md, etc.)
- Clean up obsolete models and documentation files

This addresses the root cause: assistant_msg language distribution pollution
that was degrading code generation quality. The separated architecture ensures
clean code generation while allowing DPO to optimize action selection.
```

## 详细版本（如果需要更详细的说明）

```
feat: implement Scheme A separated architecture

Major Changes:
- Architecture: Separate action prediction from code generation
  * Policy model: Predicts action tokens (LOW/MID/HIGH) only
  * Code generation: Handled by independent model (OpenAI API or dedicated code model)
  * Prevents DPO training from polluting code generation capabilities

- Training (policy/train_dpo.py):
  * Restore action token training (not full responses)
  * Add LOW/MID/HIGH as special tokens
  * Train state → action_token mapping

- Inference (policy/infer.py):
  * Implement separated inference flow
  * select_action(): Policy model predicts action
  * generate_code(): Independent code generation
  * execute_action(): Integrate both steps

- Evaluation (eval/evaluate_dpo_model.py):
  * Use separated architecture for evaluation
  * Support OpenAI API for code generation

- Core improvements:
  * Add unified render_state() function (policy/render_state.py)
  * Ensure training/inference prompt consistency
  * No action_prompts in state rendering (allows free action selection)

- Documentation:
  * ARCHITECTURE_SCHEME_A.md: Architecture design
  * SCHEME_A_IMPLEMENTATION.md: Implementation details
  * CRITICAL_DETAILS.md: Key training requirements
  * ROOT_CAUSE_ANALYSIS.md: Problem analysis

- Cleanup:
  * Remove obsolete models (policy_llama_base_150, policy_llama_instruct_150_optimized)
  * Remove outdated documentation files
  * Remove PEFT-generated README templates

Rationale:
This addresses the root cause identified: assistant_msg in trajectories contained
too much natural language (MID: 35.4%, HIGH: 59.9%), which polluted the training
data language distribution. DPO training then reinforced this wrong distribution,
leading to degraded code generation (12% vs 42% baseline task success rate).

The separated architecture ensures:
1. Code generation quality is not affected by DPO training
2. Action selection can be independently optimized
3. Clean code output (no language distribution pollution)
```

## 中文版本（如果需要）

```
feat: 实现方案A分离架构 - 分离action预测和代码生成

主要改动：
- 架构：将action预测与代码生成分离
  * Policy模型：只预测action token (LOW/MID/HIGH)
  * 代码生成：由独立模型处理（不受DPO影响）
  * 避免DPO训练污染代码生成能力

- 训练代码：恢复action token训练，添加特殊token
- 推理代码：实现分离架构的推理流程
- 评估代码：使用分离架构进行评估
- 核心改进：统一render_state函数，确保训练/推理一致性
- 文档：添加架构设计、实现细节等文档
- 清理：删除过期模型和文档

解决的问题：
解决了根本原因：轨迹中的assistant_msg包含过多自然语言，污染了训练数据
的语言分布，导致代码生成质量下降（12% vs 42% baseline）。
```

## 使用建议

**推荐使用简洁版**，因为：
1. 清晰明了，概括了主要改动
2. 符合conventional commits格式
3. 包含了关键信息（解决了什么问题）

如果需要更详细的说明，可以使用详细版本。

