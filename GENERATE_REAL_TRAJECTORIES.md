# 生成真实轨迹的步骤

## 前置要求

### 1. 安装依赖
```bash
pip install openai python-dotenv
```

### 2. 设置 OpenAI API Key

**方式1（推荐）：使用环境变量**
```bash
export OPENAI_API_KEY='sk-...'  # 替换为你的实际API key
```

**方式2：使用 .env 文件**
```bash
# 在项目根目录创建 .env 文件
echo 'OPENAI_API_KEY=sk-...' > .env  # 替换为你的实际API key
```

### 3. 验证 API Key
```bash
python llm/test_openai_key.py
```

## 生成真实轨迹

### 基本命令
```bash
python scripts/generate_trajectories.py \
  --mode dataset \
  --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 2 \
  --out logs/test_traj_real.jsonl \
  --max_turns 3 \
  --persona_idx 0 \
  --llm_model gpt-4o-mini
```

### 参数说明
- `--n_states 2`: 处理2个初始状态（可以根据需要调整）
- `--max_turns 3`: 每个对话最多3轮
- `--persona_idx 0`: 使用 Novice-Learner (0), Busy-Developer (1), 或 Experienced-Engineer (2)
- `--llm_model gpt-4o-mini`: 使用 gpt-4o-mini 模型（成本较低）

### 测试不同 Persona
```bash
# Novice-Learner (高耐心，低专业度)
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 2 --out logs/test_traj_novice.jsonl \
  --max_turns 3 --persona_idx 0 --llm_model gpt-4o-mini

# Busy-Developer (低耐心，中专业度)
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 2 --out logs/test_traj_busy.jsonl \
  --max_turns 3 --persona_idx 1 --llm_model gpt-4o-mini

# Experienced-Engineer (中耐心，高专业度)
python scripts/generate_trajectories.py \
  --mode dataset --domain coding \
  --dataset_path data/seeds/bigcodebench_masked_states.jsonl \
  --n_states 2 --out logs/test_traj_expert.jsonl \
  --max_turns 3 --persona_idx 2 --llm_model gpt-4o-mini
```

## 查看生成结果

```bash
# 查看轨迹数量
wc -l data/logs/test_traj_real.jsonl

# 查看第一条轨迹的JSON结构
head -1 data/logs/test_traj_real.jsonl | python -m json.tool

# 查看轨迹摘要
python3 << 'EOF'
import json
with open('data/logs/test_traj_real.jsonl', 'r') as f:
    trajs = [json.loads(line) for line in f if line.strip()]
    print(f"总轨迹数: {len(trajs)}")
    for i, traj in enumerate(trajs[:5], 1):  # 只看前5条
        state = traj.get('state', {})
        print(f"\n轨迹 {i}:")
        print(f"  State ID: {state.get('id')}")
        print(f"  Turn: {traj.get('turn')}")
        print(f"  Action: {traj.get('action')}")
        print(f"  Assistant: {traj.get('assistant_msg', '')[:100]}...")
        print(f"  User Reply: {traj.get('user_reaction', {}).get('user_reply', '')[:100]}...")
        print(f"  Terminal: {traj.get('is_terminal')}")
EOF
```

## 注意事项

1. **成本控制**: 使用 `gpt-4o-mini` 而不是 `gpt-4` 可以显著降低成本
2. **批量生成**: 如果需要生成大量轨迹，建议分批处理，避免一次性处理过多
3. **错误处理**: 如果API调用失败，脚本会输出错误信息，可以根据错误调整参数或重试
