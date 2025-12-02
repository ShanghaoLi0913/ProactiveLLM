# 项目结构说明

## 目录结构

```
ProactiveLLM/
├── data/                    # 数据目录
│   ├── seeds/              # 原始种子数据（states）
│   │   ├── convcodeworld_states_150.jsonl
│   │   └── mbpp_states.jsonl
│   ├── logs/               # 生成的轨迹数据
│   │   └── traj_convcodeworld_150.jsonl
│   ├── dpo/                # DPO训练用的preference pairs
│   │   └── prefs_150.jsonl
│   ├── eval/               # 评估结果
│   │   ├── base_model_results.json
│   │   └── instruct_model_results.json
│   └── external/           # 外部数据集（只读）
│       └── ConvCodeWorld/
│
├── outputs/                # 训练好的模型
│   ├── policy_llama_base_150/
│   └── policy_llama_instruct_150/
│
├── policy/                 # 策略训练和推理
│   ├── train_dpo.py       # DPO训练脚本
│   └── infer.py           # 推理脚本
│
├── reward/                 # 奖励计算
│   ├── compute.py         # 奖励计算函数
│   ├── compute_rewards.py # 生成preference pairs
│   └── mbpp_eval.py       # 代码执行评估
│
├── eval/                   # 评估脚本
│   ├── evaluate.py        # 基础评估
│   ├── evaluate_dpo_model.py  # DPO模型评估
│   └── plot_pareto.py     # 可视化
│
├── scripts/                # 数据生成脚本
│   ├── convert_convcodeworld_to_states.py
│   ├── generate_trajectories.py
│   └── run_experiment.sh  # 完整实验流程
│
├── simulator/             # 用户模拟器
│   └── simulate.py
│
├── llm/                    # LLM接口
│   └── provider.py
│
├── prompts/                # 行为模板
│   ├── coding_low.txt
│   ├── coding_mid.txt
│   ├── coding_high.txt
│   └── planning_*.txt
│
├── README.md               # 项目说明
├── EXPERIMENT_RESULTS.md   # 实验结果
└── requirements.txt       # 依赖
```

## 数据流程

1. **原始数据** → `data/seeds/`
   - 从ConvCodeBench等数据集转换而来
   - 格式: JSONL，每行一个state

2. **轨迹生成** → `data/logs/`
   - 使用 `scripts/generate_trajectories.py`
   - 为每个state生成3个actions的轨迹
   - 格式: JSONL，每行一个trajectory

3. **Reward计算** → `data/dpo/`
   - 使用 `reward/compute_rewards.py`
   - 计算每个trajectory的reward
   - 生成preference pairs (chosen, rejected)

4. **模型训练** → `outputs/`
   - 使用 `policy/train_dpo.py`
   - 训练DPO模型
   - 保存LoRA adapter

5. **模型评估** → `data/eval/`
   - 使用 `eval/evaluate_dpo_model.py`
   - 计算task success rate等指标
   - 保存评估结果

## 关键文件说明

### 训练脚本
- `policy/train_dpo.py`: DPO训练主脚本
  - 支持QLoRA 4-bit量化
  - 支持LoRA参数高效微调
  - 配置了4090 GPU优化参数

### 评估脚本
- `eval/evaluate_dpo_model.py`: DPO模型评估
  - 计算task success rate
  - 计算action accuracy
  - 生成详细评估报告

### 数据生成
- `scripts/generate_trajectories.py`: 生成轨迹
  - Mainline + Branches策略
  - 支持3种persona
  - 生成3个actions的对比轨迹

- `reward/compute_rewards.py`: 计算reward
  - R_task: 任务成功分数
  - C_interrupt: 中断成本
  - R = R_task - C_interrupt

## 实验流程

完整实验流程见 `scripts/run_experiment.sh`:

```bash
bash scripts/run_experiment.sh
```

或手动执行:

1. 训练Base模型
2. 训练Instruct模型
3. 评估Base模型
4. 评估Instruct模型
5. 对比结果

## 环境配置

- Python: 3.11 (bitsandbytes需要)
- CUDA: 11.8+
- 主要依赖:
  - transformers
  - trl (DPO)
  - peft (LoRA)
  - bitsandbytes (4-bit量化)


