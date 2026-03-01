#!/usr/bin/env python3
"""
快速验证：使用Few-shot Inference测试不同persona的行为差异

不训练模型，直接用base model + few-shot examples测试不同persona的prompt是否产生不同行为
"""

import json
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

# Set environment variables for HuggingFace cache
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.render_state import render_state


def extract_action_from_response(response: str) -> str:
    """Extract action from model response."""
    response_lower = response.lower().strip()
    
    # Check for explicit action markers
    if response.startswith("Clarify\n") or response.startswith("clarify\n"):
        return "Clarify"
    if response.startswith("Execute\n") or response.startswith("execute\n"):
        return "Execute"
    
    # Check for question marks (indicates Clarify)
    if "?" in response[:200]:
        return "Clarify"
    
    # Check for code (indicates Execute)
    if "```" in response or "def " in response or "class " in response:
        return "Execute"
    
    # Default: if contains clarifying words
    clarify_keywords = ["could you", "can you", "please clarify", "what", "which", "how"]
    if any(keyword in response_lower[:100] for keyword in clarify_keywords):
        return "Clarify"
    
    return "Execute"  # Default


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate model response."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Extract only the generated part
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def create_few_shot_examples(prefs_file: Path, n_examples: int = 2) -> Dict[str, List[Dict]]:
    """Create few-shot examples from training data, grouped by persona."""
    with open(prefs_file) as f:
        prefs = [json.loads(line) for line in f]
    
    # Group by persona
    persona_examples = defaultdict(list)
    for p in prefs:
        persona_name = p.get("persona", {}).get("name", "Unknown")
        if len(persona_examples[persona_name]) < n_examples:
            persona_examples[persona_name].append(p)
    
    return dict(persona_examples)


def build_few_shot_prompt(state: Dict, persona: Dict, examples: List[Dict]) -> str:
    """Build few-shot prompt with examples."""
    persona_name = persona.get("name", "Unknown")
    
    prompt = f"""You are an AI assistant helping a user. The user has a specific persona: {persona_name}

Here are some examples of how to respond based on the user's persona:

"""
    
    # Add examples
    for i, ex in enumerate(examples, 1):
        ex_state = ex.get("state", {})
        ex_persona = ex.get("persona", {})
        chosen_action = ex.get("chosen_action", "")
        chosen_msg = ex.get("chosen_assistant_msg", "")[:200]  # Truncate
        
        prompt += f"""Example {i}:
Task: {ex_state.get('query', '')[:100]}
Persona: {ex_persona.get('name', 'Unknown')}
Response: {chosen_action}
{chosen_msg}

"""
    
    # Add current task
    state_text = render_state(state, persona=persona)
    prompt += f"""Now, respond to this task based on the user's persona ({persona_name}):

{state_text}

Response:"""
    
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Quick verify persona differences with few-shot inference")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model to use")
    parser.add_argument("--test_states", type=str, default="data/seeds/bigcodebench_masked_states.jsonl",
                       help="Test states file")
    parser.add_argument("--prefs_file", type=str, default="data/dpo/test_fixed_prefs_final.jsonl",
                       help="Training preferences file for few-shot examples")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of test samples")
    parser.add_argument("--n_examples", type=int, default=2,
                       help="Number of few-shot examples per persona")
    parser.add_argument("--output", type=str, default="eval_results/quick_verify_persona.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("快速验证：Few-shot Inference测试不同persona的行为差异")
    print("=" * 70)
    
    # Load model
    print(f"\n[Step 1] 加载模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✅ 模型加载完成")
    
    # Load test states
    print(f"\n[Step 2] 加载测试数据: {args.test_states}")
    test_states = []
    with open(args.test_states) as f:
        for line in f:
            if line.strip():
                test_states.append(json.loads(line))
    
    test_states = test_states[:args.max_samples]
    print(f"✅ 加载了 {len(test_states)} 个测试states")
    
    # Load few-shot examples
    print(f"\n[Step 3] 加载Few-shot示例: {args.prefs_file}")
    prefs_file = Path(args.prefs_file)
    if prefs_file.exists():
        persona_examples = create_few_shot_examples(prefs_file, n_examples=args.n_examples)
        print(f"✅ 为每个persona准备了 {args.n_examples} 个示例")
    else:
        print(f"⚠️  Few-shot示例文件不存在，将不使用few-shot")
        persona_examples = {}
    
    # Define personas
    personas = [
        {"name": "Busy-Developer", "patience": "low", "expertise": "mid"},
        {"name": "Experienced-Engineer", "patience": "mid", "expertise": "high"},
        {"name": "Novice-Learner", "patience": "high", "expertise": "low"}
    ]
    
    # Test
    print(f"\n[Step 4] 开始测试...")
    results = []
    persona_stats = defaultdict(lambda: {"total": 0, "Clarify": 0, "Execute": 0})
    task_persona_stats = defaultdict(lambda: defaultdict(lambda: {"Clarify": 0, "Execute": 0}))
    
    for state in test_states:
        state_id = state.get("id", "unknown")
        print(f"\n测试 Task: {state_id}")
        
        for persona in personas:
            persona_name = persona.get("name")
            
            # Build prompt
            examples = persona_examples.get(persona_name, [])
            if examples:
                prompt = build_few_shot_prompt(state, persona, examples)
            else:
                # No examples, use simple prompt
                state_text = render_state(state, persona=persona)
                prompt = f"""You are an AI assistant helping a user. The user has a specific persona: {persona_name}

{state_text}

Response:"""
            
            # Generate response
            response = generate_response(model, tokenizer, prompt, max_new_tokens=256)
            action = extract_action_from_response(response)
            
            # Record results
            persona_stats[persona_name]["total"] += 1
            persona_stats[persona_name][action] += 1
            task_persona_stats[state_id][persona_name][action] += 1
            
            results.append({
                "state_id": state_id,
                "persona": persona_name,
                "action": action,
                "response": response[:200]  # Truncate
            })
            
            print(f"  {persona_name}: {action}")
    
    # Calculate metrics
    print(f"\n{'=' * 70}")
    print("📊 验证结果")
    print(f"{'=' * 70}")
    
    print(f"\n【按Persona的Action分布】")
    for persona in personas:
        persona_name = persona.get("name")
        stats = persona_stats[persona_name]
        total = stats["total"]
        clarify_ratio = stats["Clarify"] / total * 100 if total > 0 else 0
        execute_ratio = stats["Execute"] / total * 100 if total > 0 else 0
        
        print(f"  {persona_name}:")
        print(f"    Clarify: {stats['Clarify']} ({clarify_ratio:.1f}%)")
        print(f"    Execute: {stats['Execute']} ({execute_ratio:.1f}%)")
        print(f"    预期: {'更多Execute' if persona_name == 'Busy-Developer' else '更多Clarify' if persona_name == 'Novice-Learner' else '平衡'}")
    
    # Calculate PDS
    p_clarify_busy = persona_stats["Busy-Developer"]["Clarify"] / persona_stats["Busy-Developer"]["total"] if persona_stats["Busy-Developer"]["total"] > 0 else 0
    p_clarify_novice = persona_stats["Novice-Learner"]["Clarify"] / persona_stats["Novice-Learner"]["total"] if persona_stats["Novice-Learner"]["total"] > 0 else 0
    pds = abs(p_clarify_busy - p_clarify_novice)
    
    print(f"\n【Persona Discrimination Score (PDS)】")
    print(f"  P(Clarify|Busy-Developer): {p_clarify_busy:.2%}")
    print(f"  P(Clarify|Novice-Learner): {p_clarify_novice:.2%}")
    print(f"  PDS (Busy vs Novice): {pds:.2%}")
    
    if pds > 0.2:
        print(f"  ✅ 模型表现出明显的persona差异（PDS > 20%）")
    else:
        print(f"  ⚠️  模型的persona差异较小（PDS < 20%）")
    
    # Check task-specific differences
    print(f"\n【按(Task, Persona)的Action分布（前5个tasks）】")
    for state_id in list(sorted(task_persona_stats.keys()))[:5]:
        print(f"  {state_id}:")
        for persona in personas:
            persona_name = persona.get("name")
            if persona_name in task_persona_stats[state_id]:
                stats = task_persona_stats[state_id][persona_name]
                total = stats["Clarify"] + stats["Execute"]
                if total > 0:
                    clarify_ratio = stats["Clarify"] / total * 100
                    print(f"    {persona_name}: Clarify={clarify_ratio:.0f}%")
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "results": results,
            "persona_stats": dict(persona_stats),
            "task_persona_stats": {k: dict(v) for k, v in task_persona_stats.items()},
            "pds": pds
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"\n{'=' * 70}")
    print("验证完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
