"""
多轮交互评估脚本：测试训练好的模型在不同persona下的多轮行为差异

若在 max_turns 内从未选择 Execute，会在当前 state 上追加一轮强制 Execute，
使每条对话都以代码测评分结束（见 _build_execute_turn_data(..., forced_final_execute=True)）。

Clarify 与 Execute 的**正文/代码**默认由**同一套已加载的 DPO checkpoint** 本地生成；
若需对照实验，可加 --use_openai_for_generation 改为 gpt-4o-mini。

User 模拟（react）默认与轨迹生成一致：`--llm_model gpt-4o-mini`（见 GENERATE_COLM_DATA_V2.sh）。
离线/省钱可加 `--user_dummy`，等价于轨迹脚本里不传 llm_model。

使用方法:
    python eval/evaluate_multi_turn_persona.py \
        --model_dir checkpoints/v18_uniform_dpo_llama/ \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --test_states data/test_states.jsonl \
        --max_samples 20 \
        --max_turns 5 \
        --output eval_results/multi_turn_persona.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_trajectories import (
    PERSONAS, 
    build_action_prompts,
    select_mainline_action_from_persona,
    update_state_for_next_turn,
    sanitize_clarify_message,
    check_task_completion,
)
from simulator.simulate import react
from llm.provider import chat_complete


def generate_with_template_local(
    model,
    tokenizer,
    template: str,
    task_prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response from a local HF model (same pattern as evaluate_dpo_model)."""
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": f"[Task]\n{task_prompt}"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{template}\n\n[Task]\n{task_prompt}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 多轮评估：内容与动作一致，使用 LoRA（训练后的策略）；勿 disable_adapter。
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()
    return generated_text


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def render_state(state: Dict) -> str:
    """Render state to text for model input."""
    query = state.get("query", "")
    dialogue_turn = state.get("dialogue_turn", 0)
    prev_reject = state.get("prev_reject", False)
    task_uncertainty = state.get("task_uncertainty", 0.5)
    
    persona = state.get("persona", {})
    persona_name = persona.get("name", "Unknown")
    
    text = f"""Task: {query}

Dialogue Turn: {dialogue_turn}
Previous Reject: {prev_reject}
Task Uncertainty: {task_uncertainty:.2f}
Persona: {persona_name}
"""
    
    # Add user's previous answer if available
    if "[User]:" in query:
        text += "\nUser has answered your previous clarification question.\n"
    
    return text


def select_action_with_model(
    state: Dict,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    persona: Dict,
) -> str:
    """Use trained model to select action."""
    # Render state to text
    state_text = render_state(state)
    
    # Add persona info to prompt
    persona_name = persona.get("name", "Unknown")
    full_prompt = f"""{state_text}

Based on the persona ({persona_name}), should you Clarify or Execute?

Action:"""
    
    messages = [{"role": "user", "content": full_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Extract action
    response_lower = response.lower()
    if "clarify" in response_lower:
        return "Clarify"
    elif "execute" in response_lower:
        return "Execute"
    else:
        # Default: use persona-based selection
        persona_obj = next((p for p in PERSONAS if p.name == persona_name), PERSONAS[0])
        return select_mainline_action_from_persona(persona_obj, state)


def generate_assistant_message(
    action: str,
    state: Dict,
    domain: str = "coding",
    model=None,
    tokenizer=None,
    use_openai: bool = False,
    base_model: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """Generate assistant message for the given action (Clarify or Execute-style text)."""
    prompts = build_action_prompts(domain)
    action_prompt = prompts.get(action, prompts["Execute"])
    task_prompt = state.get("query", "")
    
    if use_openai:
        response = chat_complete(
            action_prompt,
            f"[Task]\n{task_prompt}",
            model="gpt-4o-mini",
            max_tokens=400,
            temperature=temperature,
        )
    else:
        if model is None or tokenizer is None:
            raise ValueError("generate_assistant_message: model and tokenizer required when use_openai=False")
        response = generate_with_template_local(
            model,
            tokenizer,
            action_prompt,
            task_prompt,
            max_new_tokens=400,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
    
    if action == "Clarify":
        response = sanitize_clarify_message(response)
    
    return response


def generate_multiple_code_candidates(
    state: Dict,
    domain: str = "coding",
    k: int = 5,
    model=None,
    tokenizer=None,
    use_openai: bool = False,
    base_model: Optional[str] = None,
) -> List[str]:
    """Generate k different code candidates for pass@k (local policy model or OpenAI)."""
    candidates = []
    prompts = build_action_prompts(domain)
    action_prompt = prompts.get("Execute", "")
    task_prompt = state.get("query", "")
    
    for i in range(k):
        temperature = 0.7 + (i * 0.1)
        
        if use_openai:
            response = chat_complete(
                action_prompt,
                f"[Task]\n{task_prompt}",
                model="gpt-4o-mini",
                max_tokens=400,
                temperature=temperature,
            )
        else:
            if model is None or tokenizer is None:
                raise ValueError("generate_multiple_code_candidates: model and tokenizer required when use_openai=False")
            response = generate_with_template_local(
                model,
                tokenizer,
                action_prompt,
                task_prompt,
                max_new_tokens=400,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        candidates.append(response)
    
    return candidates


def _build_execute_turn_data(
    turn: int,
    current_state: Dict,
    persona_obj,
    pass_at_k: List[int],
    use_openai: bool,
    base_model: Optional[str],
    llm_model: Optional[str],
    model=None,
    tokenizer=None,
    total_questions_asked: int = 0,
    forced_final_execute: bool = False,
) -> Dict:
    """Run one Execute turn: code candidates, tests, pass@k, user reaction stub, turn record."""
    max_k = max(pass_at_k) if pass_at_k else 1
    code_candidates = generate_multiple_code_candidates(
        current_state,
        domain=current_state.get("domain", "coding"),
        k=max_k,
        model=model,
        tokenizer=tokenizer,
        use_openai=use_openai,
        base_model=base_model,
    )
    assistant_msg = code_candidates[0] if code_candidates else ""
    candidate_results = []
    for candidate_msg in code_candidates:
        candidate_completed = check_task_completion(
            current_state,
            candidate_msg,
            current_state.get("domain", "coding"),
        )
        candidate_results.append({
            "code": candidate_msg[:200] + "..." if len(candidate_msg) > 200 else candidate_msg,
            "passed": candidate_completed,
        })
    pass_at_k_results = {}
    for k in pass_at_k:
        if k <= len(candidate_results):
            passed_count = sum(1 for r in candidate_results[:k] if r["passed"])
            pass_at_k_results[f"pass@{k}"] = passed_count > 0
        else:
            pass_at_k_results[f"pass@{k}"] = False
    user_reaction = react(
        current_state["query"],
        assistant_msg,
        persona_obj,
        llm_model=llm_model,
        total_questions_asked=total_questions_asked,
        dialogue_turn=current_state.get("dialogue_turn", 0),
    )
    turn_data = {
        "turn": turn,
        "action": "Execute",
        "assistant_msg": assistant_msg[:200] + "..." if len(assistant_msg) > 200 else assistant_msg,
        "user_reaction": user_reaction.get("user_reply", "")[:100] + "..." if len(user_reaction.get("user_reply", "")) > 100 else user_reaction.get("user_reply", ""),
        "answered_clarification": user_reaction.get("meta", {}).get("answered_clarification", 0),
        "forced_final_execute": forced_final_execute,
    }
    task_completed = candidate_results[0]["passed"] if candidate_results else False
    turn_data["task_completed"] = task_completed
    turn_data["pass_at_k"] = pass_at_k_results
    turn_data["candidate_results"] = candidate_results
    return turn_data


def evaluate_multi_turn_conversation(
    model_dir: Optional[str],
    base_model: str,
    test_states_path: str,
    max_samples: Optional[int] = None,
    max_turns: int = 5,
    output_path: Optional[str] = None,
    llm_model: Optional[str] = None,
    seed: int = 42,
    pass_at_k: List[int] = [1, 3, 5],
    use_openai_for_generation: bool = False,
    no_lora: bool = False,
):
    """Evaluate model's multi-turn behavior with different personas.
    
    no_lora=True: 只加载 base_model（用于 Base Llama 对照），不加载 LoRA。
    """
    
    print("=" * 80)
    print("🔍 多轮交互评估：测试不同Persona下的行为差异")
    print("=" * 80)
    
    hf_token = os.getenv("HF_TOKEN")
    
    # Load model
    if no_lora:
        print(f"\n📊 加载 Base 模型（无 LoRA）: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            token=hf_token if hf_token else None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token if hf_token else None,
            )
        except Exception:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token if hf_token else None,
            )
        model = base_model_obj
    else:
        if not model_dir:
            raise ValueError("model_dir 必填（除非 no_lora=True）")
        print(f"\n📊 加载模型（LoRA）: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir if Path(model_dir).exists() else base_model,
            use_fast=True,
            token=hf_token if hf_token else None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token if hf_token else None,
            )
        except Exception:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token if hf_token else None,
            )
        
        if len(tokenizer) != base_model_obj.get_input_embeddings().num_embeddings:
            base_model_obj.resize_token_embeddings(len(tokenizer))
        
        try:
            model = PeftModel.from_pretrained(base_model_obj, model_dir)
        except Exception:
            model = base_model_obj
    
    model.eval()
    print("✅ 模型加载完成", flush=True)
    
    # Load test states
    print("📂 开始加载测试数据...", flush=True)
    test_states = load_jsonl(Path(test_states_path))
    print(f"✅ 已加载 {len(test_states)} 条测试数据", flush=True)
    if max_samples:
        import random
        rng = random.Random(seed)
        test_states = rng.sample(test_states, min(max_samples, len(test_states)))
        print(f"✅ 采样后剩余 {len(test_states)} 条", flush=True)
    
    print(f"\n📊 评估 {len(test_states)} 个测试样本", flush=True)
    print(f"📊 每个样本测试 {len(PERSONAS)} 个personas", flush=True)
    print(f"📊 最大轮次: {max_turns}", flush=True)
    
    if use_openai_for_generation:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "use_openai_for_generation=True 需要环境变量 OPENAI_API_KEY（或 .env）"
            )
        print("📌 Clarify / Execute 正文与代码：OpenAI gpt-4o-mini（对照用）", flush=True)
    else:
        print(
            "📌 Clarify / Execute 正文与代码：本地已加载策略模型（Base 或 LoRA，与选动作同一套权重）",
            flush=True,
        )
    use_openai = use_openai_for_generation

    if llm_model:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "User 模拟使用 OpenAI（llm_model 已设置）需要 OPENAI_API_KEY。"
                "若无需 API，请传 --user_dummy。"
            )
        print(f"📌 User 模拟（react）：OpenAI `{llm_model}`（与 generate_trajectories --llm_model 对齐）", flush=True)
    else:
        print("📌 User 模拟（react）：dummy / 规则（--user_dummy 或未设置 llm_model）", flush=True)
    
    print("\n🚀 开始评估...", flush=True)
    # Evaluate each state with each persona
    results = []
    persona_stats = defaultdict(lambda: {
        "total_conversations": 0,
        "total_turns": 0,
        "clarify_turns": 0,
        "execute_turns": 0,
        "avg_turns_per_conversation": 0.0,
        "clarify_rate": 0.0,
        "multi_turn_clarify_count": 0,  # 多轮clarify的conversations数量
        "pass_at_k": {f"pass@{k}": {"total": 0, "passed": 0} for k in pass_at_k},
    })
    
    for state_idx, initial_state in enumerate(test_states):
        print(f"\n{'='*80}", flush=True)
        print(f"样本 {state_idx + 1}/{len(test_states)}: {initial_state.get('id', 'unknown')}", flush=True)
        print(f"{'='*80}", flush=True)
        
        for persona_idx, persona_obj in enumerate(PERSONAS):
            persona_name = persona_obj.name
            print(f"\n  Persona: {persona_name}", flush=True)
            
            # Initialize state with persona
            current_state = initial_state.copy()
            current_state["persona"] = {
                "name": persona_name,
                "patience": persona_obj.patience,
                "expertise": persona_obj.expertise,
            }
            
            conversation = []
            actions_taken = []
            
            for turn in range(max_turns):
                # Select action using trained model
                action = select_action_with_model(
                    current_state,
                    tokenizer,
                    model,
                    current_state["persona"],
                )
                
                # Generate assistant message
                if action == "Execute":
                    turn_data = _build_execute_turn_data(
                        turn,
                        current_state,
                        persona_obj,
                        pass_at_k,
                        use_openai,
                        base_model,
                        llm_model,
                        model=model,
                        tokenizer=tokenizer,
                        total_questions_asked=sum(1 for a in actions_taken if a == "Clarify"),
                        forced_final_execute=False,
                    )
                    conversation.append(turn_data)
                    actions_taken.append(action)
                    print(f"    Turn {turn}: {action}")
                    break
                else:
                    # For Clarify, generate single message
                    assistant_msg = generate_assistant_message(
                        action,
                        current_state,
                        domain=current_state.get("domain", "coding"),
                        model=model,
                        tokenizer=tokenizer,
                        use_openai=use_openai,
                        base_model=base_model,
                    )
                    user_reaction = react(
                        current_state["query"],
                        assistant_msg,
                        persona_obj,
                        llm_model=llm_model,
                        total_questions_asked=sum(1 for a in actions_taken if a == "Clarify"),
                        dialogue_turn=current_state.get("dialogue_turn", 0),
                    )
                    turn_data = {
                        "turn": turn,
                        "action": action,
                        "assistant_msg": assistant_msg[:200] + "..." if len(assistant_msg) > 200 else assistant_msg,
                        "user_reaction": user_reaction.get("user_reply", "")[:100] + "..." if len(user_reaction.get("user_reply", "")) > 100 else user_reaction.get("user_reply", ""),
                        "answered_clarification": user_reaction.get("meta", {}).get("answered_clarification", 0),
                        "forced_final_execute": False,
                    }
                    conversation.append(turn_data)
                    actions_taken.append(action)
                    print(f"    Turn {turn}: {action}")
                    current_state = update_state_for_next_turn(
                        current_state,
                        user_reaction,
                        assistant_msg,
                        is_same_turn=False,
                    )

            # If the policy never chose Execute within max_turns, force one final Execute
            # on the latest state so every conversation is scored on code.
            if not any(a == "Execute" for a in actions_taken):
                total_q = sum(1 for a in actions_taken if a == "Clarify")
                turn_data = _build_execute_turn_data(
                    len(conversation),
                    current_state,
                    persona_obj,
                    pass_at_k,
                    use_openai,
                    base_model,
                    llm_model,
                    model=model,
                    tokenizer=tokenizer,
                    total_questions_asked=total_q,
                    forced_final_execute=True,
                )
                conversation.append(turn_data)
                actions_taken.append("Execute")
                print(f"    Turn {turn_data['turn']}: Execute (forced final)")
            
            # Record results
            result = {
                "state_id": initial_state.get("id", "unknown"),
                "persona": persona_name,
                "conversation": conversation,
                "total_turns": len(conversation),
                "actions": actions_taken,
                "clarify_count": sum(1 for a in actions_taken if a == "Clarify"),
                "execute_count": sum(1 for a in actions_taken if a == "Execute"),
                "has_multi_turn_clarify": sum(1 for a in actions_taken if a == "Clarify") > 1,
            }
            results.append(result)
            
            # Update stats
            stats = persona_stats[persona_name]
            stats["total_conversations"] += 1
            stats["total_turns"] += len(conversation)
            stats["clarify_turns"] += result["clarify_count"]
            stats["execute_turns"] += result["execute_count"]
            if result["has_multi_turn_clarify"]:
                stats["multi_turn_clarify_count"] += 1
            
            # Update pass@k stats
            for turn_data in conversation:
                if turn_data.get("action") == "Execute" and "pass_at_k" in turn_data:
                    for k in pass_at_k:
                        key = f"pass@{k}"
                        if key in turn_data["pass_at_k"]:
                            stats["pass_at_k"][key]["total"] += 1
                            if turn_data["pass_at_k"][key]:
                                stats["pass_at_k"][key]["passed"] += 1
    
    # Calculate final stats
    for persona_name, stats in persona_stats.items():
        if stats["total_conversations"] > 0:
            stats["avg_turns_per_conversation"] = stats["total_turns"] / stats["total_conversations"]
            total_actions = stats["clarify_turns"] + stats["execute_turns"]
            if total_actions > 0:
                stats["clarify_rate"] = stats["clarify_turns"] / total_actions
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 评估结果总结")
    print("=" * 80)
    
    for persona_name in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona_name in persona_stats:
            stats = persona_stats[persona_name]
            print(f"\n{persona_name}:")
            print(f"  总对话数: {stats['total_conversations']}")
            print(f"  平均轮次: {stats['avg_turns_per_conversation']:.2f}")
            print(f"  Clarify率: {stats['clarify_rate']:.1%}")
            print(f"  多轮Clarify对话数: {stats['multi_turn_clarify_count']} ({stats['multi_turn_clarify_count']/stats['total_conversations']*100:.1f}%)")
            
            # Print pass@k metrics
            if "pass_at_k" in stats:
                print(f"  Pass@K指标:")
                for k in pass_at_k:
                    key = f"pass@{k}"
                    if key in stats["pass_at_k"]:
                        total = stats["pass_at_k"][key]["total"]
                        passed = stats["pass_at_k"][key]["passed"]
                        rate = (passed / total * 100) if total > 0 else 0.0
                        print(f"    {key}: {rate:.2f}% ({passed}/{total})")
    
    # Save results
    if output_path:
        output = {
            "summary": dict(persona_stats),
            "detailed_results": results,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {output_path}")
    
    return results, persona_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="LoRA 适配器目录（DPO/SFT）。评估纯 Base 时请使用 --no_lora，可不填。",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="只评估 base_model（不加载 LoRA），用于 Base Llama 对照。",
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--test_states", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="User 侧 react()：与 generate_trajectories --llm_model 相同（默认 gpt-4o-mini）。",
    )
    parser.add_argument(
        "--user_dummy",
        action="store_true",
        help="User 用规则/dummy，不调 OpenAI（等价于轨迹脚本里 --llm_model 留空）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass_at_k", type=int, nargs="+", default=[1, 3, 5],
                        help="List of k values for pass@k evaluation (e.g., --pass_at_k 1 3 5)")
    parser.add_argument(
        "--use_openai_for_generation",
        action="store_true",
        help="用 gpt-4o-mini 生成 Clarify/Execute 内容（默认用本地 DPO 模型）",
    )
    
    args = parser.parse_args()

    if not args.no_lora and not args.model_dir:
        parser.error("需要 --model_dir（LoRA 目录），或使用 --no_lora 评估 Base 模型")
    if args.no_lora and args.model_dir:
        print("ℹ️  已指定 --no_lora，忽略 --model_dir", flush=True)

    # 与 scripts/generate_trajectories 一致：默认 gpt-4o-mini；--user_dummy 则 llm_model=None
    llm_for_user = None if args.user_dummy else (args.llm_model.strip() or None)
    
    evaluate_multi_turn_conversation(
        model_dir=args.model_dir,
        base_model=args.base_model,
        test_states_path=args.test_states,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        output_path=args.output,
        llm_model=llm_for_user,
        seed=args.seed,
        pass_at_k=args.pass_at_k,
        use_openai_for_generation=args.use_openai_for_generation,
        no_lora=args.no_lora,
    )
