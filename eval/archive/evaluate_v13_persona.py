"""
评估V13模型的Persona-Aware性能
按不同persona分别统计：
1. Action准确率
2. Task Success Rate
3. Interrupt Cost
4. Total Reward
"""
import json
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policy.render_state import render_state
from reward.compute import compute_task_score, compute_interrupt_cost_v2


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_action_from_response(response: str) -> str:
    """从模型生成的内容推断其行为 (Clarify/Execute)
    
    不要求显式标签，而是从内容判断：
    - 生成代码 → Execute
    - 生成问题 → Clarify
    """
    response = response.strip().lower()
    
    # Execute行为特征：生成代码
    execute_indicators = [
        '```python',           # Python代码块
        'def ',                # 函数定义
        'import ',             # import语句
        'class ',              # 类定义
        'return ',             # return语句
        "here's a python",     # 常见开头
        "here is a python",
        "here's the python",
        "here is the python",
    ]
    
    # Clarify行为特征：提出问题
    clarify_indicators = [
        '?',                   # 问号
        'clarif',              # clarify相关词
        'could you',           # 礼貌询问
        'can you',
        'would you',
        'please provide',
        'need more information',
        'not clear',
        'ambiguous',
        'specify',
        'which ',              # 选择疑问
        'what ',               # 疑问词
        'how ',
        'when ',
    ]
    
    # 检查Execute
    for indicator in execute_indicators:
        if indicator in response:
            return "Execute"
    
    # 检查Clarify
    for indicator in clarify_indicators:
        if indicator in response:
            return "Clarify"
    
    # 默认：如果长度>50且没有明显问题特征，可能是Execute（描述+代码）
    if len(response) > 50:
        return "Execute"
    
    # 其他情况默认Execute（因为测试集主要是Execute）
    return "Execute"


def extract_code_from_text(text: str) -> Optional[str]:
    """从文本中提取Python代码块"""
    # 尝试提取markdown代码块
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        valid_blocks = []
        for code in matches:
            code = code.strip()
            # 跳过错误信息
            if any(keyword in code for keyword in ["Traceback", "Error:", "Exception"]):
                continue
            valid_blocks.append(code)
        
        if valid_blocks:
            return max(valid_blocks, key=len).strip()
    
    # 尝试提取函数定义
    def_match = re.search(r'def\s+\w+.*?(?=\n(?:def\s+|\Z))', text, re.DOTALL)
    if def_match:
        return def_match.group(0).strip()
    
    return None


def score_code_simple(code: str, tests: str) -> float:
    """简单的代码评分（基于启发式规则）"""
    if not code:
        return 0.0
    
    # 检查是否有基本的代码结构
    has_def = "def " in code
    has_return = "return" in code
    code_length = len(code.strip().split('\n'))
    
    # 基础分数
    score = 0.0
    if has_def:
        score += 0.3
    if has_return:
        score += 0.3
    if code_length >= 3:
        score += 0.4
    
    return min(score, 1.0)


def evaluate_model(
    model_dir: str,
    base_model: str,
    test_data_path: str,
    output_path: str,
    max_samples: Optional[int] = None
):
    """评估模型性能，按persona分组统计"""
    
    print("=" * 60)
    print("🚀 V13 Persona-Aware Evaluation")
    print("=" * 60)
    
    # 1. 加载模型
    print(f"\n📂 Loading model from: {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加特殊tokens（与训练时保持一致）
    special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"✅ Added {num_added} special tokens: Clarify, Execute")
    
    # 使用4-bit量化配置
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Resize token embeddings to match the tokenizer
    base.resize_token_embeddings(len(tokenizer))
    print(f"✅ Resized model embeddings to {len(tokenizer)} tokens")
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    print("✅ Model loaded")
    
    # 2. 加载测试数据
    print(f"\n📊 Loading test data from: {test_data_path}")
    test_data = load_jsonl(Path(test_data_path))
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # 3. 按persona分组
    persona_groups = defaultdict(list)
    for sample in test_data:
        persona_name = sample.get("persona", {}).get("name", "Unknown")
        persona_groups[persona_name].append(sample)
    
    print(f"\n📋 Persona Distribution:")
    for persona, samples in persona_groups.items():
        print(f"  - {persona}: {len(samples)} samples")
    
    # 4. 评估每个样本
    print(f"\n🔍 Evaluating model predictions...")
    
    results_by_persona = defaultdict(lambda: {
        "total": 0,
        "action_correct": 0,
        "task_success": 0,
        "total_reward": 0.0,
        "total_interrupt_cost": 0.0,
        "predictions": []
    })
    
    for sample in tqdm(test_data, desc="Evaluating"):
        state = sample["state"]
        persona = sample.get("persona", {})
        persona_name = persona.get("name", "Unknown")
        
        # 渲染state为prompt
        state_with_persona = state.copy()
        state_with_persona["persona"] = persona
        state_text = render_state(state_with_persona)
        
        # ✅ V15 Fix: Use chat template for generation (matching training format)
        messages = [{"role": "user", "content": state_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds <|start_header_id|>assistant<|end_header_id|>
        )
        
        # 生成预测
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Increased for full code generation
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_action = extract_action_from_response(response)
        
        # 获取ground truth
        chosen_action = sample["chosen_action"]
        chosen_msg = sample.get("chosen_assistant_msg", "")
        
        # 计算指标
        action_correct = (predicted_action == chosen_action)
        
        # 计算task score - 使用模型生成的response而不是ground truth!
        task_score = 0.0
        if chosen_action == "Execute":
            # 从模型生成的response中提取代码并评分
            code = extract_code_from_text(response)  # 修改：使用response而不是chosen_msg
            if code:
                tests = state.get("convcodeworld_tests", "")
                task_score = score_code_simple(code, tests)
        
        # 计算interrupt cost - 使用模型生成的response
        meta = {
            "answered_clarification": 1 if predicted_action == "Clarify" else 0,  # 修改：使用predicted_action
            "reject_signal": 0
        }
        n_questions = 1 if predicted_action == "Clarify" else 0  # 修改：使用predicted_action
        interrupt_cost = compute_interrupt_cost_v2(meta, n_questions, response)  # 修改：使用response
        
        # 计算total reward
        total_reward_val = task_score - interrupt_cost
        
        # 记录结果
        results_by_persona[persona_name]["total"] += 1
        results_by_persona[persona_name]["action_correct"] += int(action_correct)
        results_by_persona[persona_name]["task_success"] += int(task_score > 0.5)
        results_by_persona[persona_name]["total_reward"] += total_reward_val
        results_by_persona[persona_name]["total_interrupt_cost"] += interrupt_cost
        
        results_by_persona[persona_name]["predictions"].append({
            "state_id": state.get("id", "unknown"),
            "predicted_action": predicted_action,
            "chosen_action": chosen_action,
            "action_correct": action_correct,
            "task_score": task_score,
            "interrupt_cost": interrupt_cost,
            "total_reward": total_reward_val
        })
    
    # 5. 计算统计指标
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS BY PERSONA")
    print("=" * 60)
    
    summary = {}
    overall_stats = {
        "total": 0,
        "action_correct": 0,
        "task_success": 0,
        "total_reward": 0.0,
        "total_interrupt_cost": 0.0
    }
    
    for persona, stats in sorted(results_by_persona.items()):
        total = stats["total"]
        action_acc = stats["action_correct"] / total * 100
        task_success_rate = stats["task_success"] / total * 100
        avg_reward = stats["total_reward"] / total
        avg_interrupt_cost = stats["total_interrupt_cost"] / total
        
        print(f"\n🎭 {persona} ({total} samples)")
        print(f"  Action Accuracy:      {action_acc:.2f}% ({stats['action_correct']}/{total})")
        print(f"  Task Success Rate:    {task_success_rate:.2f}% ({stats['task_success']}/{total})")
        print(f"  Avg Reward:           {avg_reward:.4f}")
        print(f"  Avg Interrupt Cost:   {avg_interrupt_cost:.4f}")
        
        summary[persona] = {
            "total_samples": total,
            "action_accuracy": action_acc,
            "task_success_rate": task_success_rate,
            "avg_reward": avg_reward,
            "avg_interrupt_cost": avg_interrupt_cost
        }
        
        # 累加到overall
        overall_stats["total"] += total
        overall_stats["action_correct"] += stats["action_correct"]
        overall_stats["task_success"] += stats["task_success"]
        overall_stats["total_reward"] += stats["total_reward"]
        overall_stats["total_interrupt_cost"] += stats["total_interrupt_cost"]
    
    # 6. 总体统计
    print("\n" + "=" * 60)
    print("📊 OVERALL STATISTICS")
    print("=" * 60)
    
    overall_action_acc = overall_stats["action_correct"] / overall_stats["total"] * 100
    overall_task_success = overall_stats["task_success"] / overall_stats["total"] * 100
    overall_avg_reward = overall_stats["total_reward"] / overall_stats["total"]
    overall_avg_interrupt = overall_stats["total_interrupt_cost"] / overall_stats["total"]
    
    print(f"\n📈 Total Samples:         {overall_stats['total']}")
    print(f"✅ Action Accuracy:       {overall_action_acc:.2f}%")
    print(f"🎯 Task Success Rate:     {overall_task_success:.2f}%")
    print(f"💰 Avg Reward:            {overall_avg_reward:.4f}")
    print(f"⚠️  Avg Interrupt Cost:   {overall_avg_interrupt:.4f}")
    
    summary["overall"] = {
        "total_samples": overall_stats["total"],
        "action_accuracy": overall_action_acc,
        "task_success_rate": overall_task_success,
        "avg_reward": overall_avg_reward,
        "avg_interrupt_cost": overall_avg_interrupt
    }
    
    # 7. 保存结果
    output = {
        "model_dir": model_dir,
        "test_data": test_data_path,
        "summary": summary,
        "detailed_results": dict(results_by_persona)
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/prefs_bigcode/dpo_v13_llama31_8b")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--test_data", type=str, default="data/dpo/prefs_bigcode_persona_aware_test.jsonl")
    parser.add_argument("--output", type=str, default="outputs/eval_results/v13_persona_eval.json")
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        base_model=args.base_model,
        test_data_path=args.test_data,
        output_path=args.output,
        max_samples=args.max_samples
    )
