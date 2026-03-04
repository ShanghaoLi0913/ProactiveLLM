"""
多轮交互评估脚本（新版本）：测试训练好的模型在不同persona下的多轮行为差异
包含 task_score 计算和 Task Success Rate 统计

注意：如果test_states中包含original_instruct_prompt字段，将自动使用它替代query
（提供完整信息，更公平地测试代码生成能力）

使用方法:
    python eval/evaluate_multi_turn_persona.py \
        --model_dir checkpoints/dpo_colm_150states_persona \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --test_states data/seeds/bigcodebench_masked_states.jsonl \
        --max_samples 20 \
        --max_turns 3 \
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
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
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
from policy.render_state import render_state
from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail
from eval.reconstruct_state import reconstruct_state_for_execute


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class ForceActionPrefixProcessor(LogitsProcessor):
    """LogitsProcessor that forces the first generated token to be 'Clarify' or 'Execute'."""
    
    def __init__(self, tokenizer: AutoTokenizer, prompt_length: int):
        """
        Args:
            tokenizer: Tokenizer to get token IDs for Clarify and Execute
            prompt_length: Length of the prompt (input_ids before generation)
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        
        # Get token IDs for Clarify and Execute
        clarify_id = tokenizer.convert_tokens_to_ids("Clarify")
        execute_id = tokenizer.convert_tokens_to_ids("Execute")
        
        # Handle cases where tokens might not be found
        self.allowed_token_ids = []
        if clarify_id is not None and clarify_id != tokenizer.unk_token_id:
            self.allowed_token_ids.append(clarify_id)
        if execute_id is not None and execute_id != tokenizer.unk_token_id:
            self.allowed_token_ids.append(execute_id)
        
        # Fallback: try encoding the full strings
        if not self.allowed_token_ids:
            clarify_encoded = tokenizer.encode("Clarify", add_special_tokens=False)
            execute_encoded = tokenizer.encode("Execute", add_special_tokens=False)
            if clarify_encoded:
                self.allowed_token_ids.extend(clarify_encoded)
            if execute_encoded:
                self.allowed_token_ids.extend(execute_encoded)
        
        # Remove duplicates and ensure we have valid IDs
        self.allowed_token_ids = list(set([tid for tid in self.allowed_token_ids if tid is not None]))
        
        if not self.allowed_token_ids:
            print("⚠️  Warning: Could not find token IDs for 'Clarify' or 'Execute'. "
                  "Action prefix forcing will be disabled.")
        else:
            print(f"✅ ForceActionPrefixProcessor initialized:")
            print(f"   Prompt length: {prompt_length}")
            print(f"   Allowed token IDs: {self.allowed_token_ids}")
            print(f"   Clarify ID: {clarify_id}, Execute ID: {execute_id}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask all tokens except Clarify/Execute at the first generation step.
        
        Args:
            input_ids: Current input sequence (including prompt + generated tokens so far)
            scores: Logits for next token prediction (shape: [batch_size, vocab_size])
        
        Returns:
            Modified scores with non-action tokens masked
        """
        # Only apply at the first generation step (when input_ids length equals prompt_length)
        # This means we're generating the very first token after the prompt
        if input_ids.shape[1] == self.prompt_length and self.allowed_token_ids:
            # Create mask: set all tokens to -inf except allowed ones
            # This forces the model to choose between Clarify and Execute
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.allowed_token_ids] = 0.0
            scores = scores + mask
            # Debug: print when mask is applied (only first few times to avoid spam)
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                print(f"🔧 LogitsProcessor applied: input_ids.shape={input_ids.shape}, "
                      f"prompt_length={self.prompt_length}, allowed_ids={self.allowed_token_ids}")
                self._debug_count += 1
        
        return scores


def select_action_with_model(
    state: Dict,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    persona: Dict,
) -> str:
    """Use trained model to select action by free generation (matching training format).
    
    This method simulates real inference:
    - Prompt: render_state(state, persona) (only state info, no action question)
    - Model generates full response: 'Clarify\n...' or 'Execute\n...'
    - Extract action from first line of response
    
    ⭐ SCHEME 1: Uses LogitsProcessor to force the first token to be 'Clarify' or 'Execute'.
    This ensures 100% prefix generation, addressing the issue where the model
    learned content preferences but not format constraints.
    
    This is 'free generation' with format constraint, not pure classification.
    """
    # Render state to text (using unified render_state function)
    # This matches training format: only state info, no action instruction
    state_text = render_state(state, persona=persona)
    
    # Use chat template (matching training format)
    messages = [{"role": "user", "content": state_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    # ⭐ SCHEME 1: Create LogitsProcessor to force action prefix
    force_prefix_processor = ForceActionPrefixProcessor(tokenizer, prompt_length)
    
    # Generate full response (matching training: model generates complete message)
    # ⭐ SCHEME 1: Use logits_processor to force first token to be Clarify/Execute
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Generate enough tokens for full response
            temperature=0.7,  # Use sampling for more natural generation
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=[force_prefix_processor],  # ⭐ SCHEME 1: Force action prefix
        )
    
    # ⭐ SCHEME 1: Don't skip special tokens to preserve "Clarify"/"Execute" prefix
    # Note: skip_special_tokens=False to keep the action prefix tokens
    generated_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    
    # Debug: Check first generated token
    if len(generated_token_ids) > 0:
        first_token_id = generated_token_ids[0].item()
        first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=False)
        if not hasattr(select_action_with_model, '_debug_count'):
            select_action_with_model._debug_count = 0
        if select_action_with_model._debug_count < 3:
            print(f"🔍 First generated token: ID={first_token_id}, text='{first_token_text}'")
            print(f"   Allowed IDs: {force_prefix_processor.allowed_token_ids}")
            print(f"   Is in allowed: {first_token_id in force_prefix_processor.allowed_token_ids}")
            select_action_with_model._debug_count += 1
    
    response = tokenizer.decode(
        generated_token_ids, 
        skip_special_tokens=False  # ⭐ Changed: preserve special tokens to see action prefix
    )
    
    # ⭐ SCHEME 1: Extract action from first token (since LogitsProcessor forces it)
    # The first token should always be Clarify or Execute
    if len(generated_token_ids) > 0:
        first_token_id = generated_token_ids[0].item()
        if first_token_id in force_prefix_processor.allowed_token_ids:
            # Decode just the first token to get the action
            first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=False).strip()
            if first_token_text.lower() == "clarify":
                return "Clarify"
            elif first_token_text.lower() == "execute":
                return "Execute"
    
    # Fallback: Extract action from first line of response
    # Training format: 'Clarify\n...' or 'Execute\n...'
    # ⭐ SCHEME 1: With forced prefix, first token should always be Clarify or Execute
    response_lines = response.strip().split('\n')
    if not response_lines:
        # Fallback if empty response
        persona_name = persona.get("name", "Unknown")
        persona_obj = next((p for p in PERSONAS if p.name == persona_name), PERSONAS[0])
        return select_mainline_action_from_persona(persona_obj, state)
    
    first_line = response_lines[0].strip()
    first_line_lower = first_line.lower()
    
    # Check if first line starts with action token
    # ⭐ SCHEME 1: With forced prefix, this should always match
    if first_line_lower.startswith("clarify"):
        return "Clarify"
    elif first_line_lower.startswith("execute"):
        return "Execute"
    
    # Fallback: check if response contains action indicators
    # (in case model generates without explicit prefix - should be rare with SCHEME 1)
    response_lower = response.lower()
    
    # Check for code indicators first (Execute)
    if "```" in response_lower[:200] or "def " in response_lower[:200] or "import " in response_lower[:200]:
        # Contains code block, function definition, or import → Execute
        return "Execute"
    
    # Check for Clarify indicators (more comprehensive)
    # 1. Check for "clarify" keyword (extend to 200 chars)
    if "clarify" in response_lower[:200]:
        return "Clarify"
    
    # 2. Check for question marks (extend to 300 chars, as questions may come later)
    if "?" in response_lower[:300] and "```" not in response_lower[:300]:
        return "Clarify"
    
    # 3. Check for clarification keywords (extend to 200 chars)
    clarify_keywords = ["could you", "can you", "would you", "please clarify", 
                       "i need to", "i'd like to", "to ensure", "to provide"]
    if any(keyword in response_lower[:200] for keyword in clarify_keywords):
        # But make sure it's not code
        if "```" not in response_lower[:200] and "def " not in response_lower[:200]:
            return "Clarify"
    
    # 4. Check for "execute" keyword (less common, but check anyway)
    if "execute" in response_lower[:200]:
        return "Execute"
    
    # Final fallback: use persona-based selection
    persona_name = persona.get("name", "Unknown")
    persona_obj = next((p for p in PERSONAS if p.name == persona_name), PERSONAS[0])
    return select_mainline_action_from_persona(persona_obj, state)


def generate_assistant_message(
    action: str,
    state: Dict,
    domain: str = "coding",
    use_openai: bool = True,
    base_model: Optional[str] = None,
    base_model_obj: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    policy_model: Optional[torch.nn.Module] = None,
) -> str:
    """Generate assistant message for the given action.
    
    ⭐ 改进：Execute阶段使用结构化state reconstruction，而不是直接拼接对话历史。
    
    Args:
        policy_model: Trained model (Llama + PEFT) - should be used for code generation
        base_model_obj: Base model (fallback if policy_model not provided)
    """
    prompts = build_action_prompts(domain)
    action_prompt = prompts.get(action, prompts["Execute"])
    
    # ⭐ 改进：Execute时使用结构化state reconstruction
    if action == "Execute":
        # 重构state：提取原始query + 结构化clarified requirements
        reconstructed = reconstruct_state_for_execute(state)
        original_query = reconstructed["original_query"]
        clarified_requirements = reconstructed["clarified_requirements"]
        
        # 构建结构化的task prompt
        if clarified_requirements:
            # 有澄清信息：使用结构化格式
            task_prompt = f"{original_query}\n\n[Clarified Requirements]\n{clarified_requirements}\n\n[Instruction]\nWrite the implementation.\nDo not ask further questions."
        else:
            # 没有澄清信息：直接使用原始query
            task_prompt = original_query
    else:
        # Clarify时仍使用原始query（包含对话历史，用于上下文）
        task_prompt = state.get("query", "")
    
    if use_openai:
        response = chat_complete(
            action_prompt,
            f"[Task]\n{task_prompt}" if action == "Execute" else task_prompt,
            model="gpt-4o-mini",
            max_tokens=400
        )
    else:
        # Use trained policy model for code generation (preferred)
        # If not available, fallback to base model
        model_to_use = policy_model if policy_model is not None else base_model_obj
        
        if model_to_use is not None and tokenizer is not None:
            # ⭐ 改进：Execute时使用结构化的task prompt
            if action == "Execute":
                user_content = f"[Task]\n{task_prompt}"
            else:
                user_content = task_prompt
            
            messages = [
                {"role": "system", "content": action_prompt},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            response = f"[{action} action - code generation would go here]"
    
    if action == "Clarify":
        response = sanitize_clarify_message(response)
    
    return response


def evaluate_multi_turn_conversation(
    model_dir: str,
    base_model: str,
    test_states_path: str,
    max_samples: Optional[int] = None,
    max_turns: int = 5,
    output_path: Optional[str] = None,
    llm_model: Optional[str] = None,
    seed: int = 42,
    use_original_query: bool = False,
):
    """Evaluate model's multi-turn behavior with different personas."""
    
    print("=" * 80)
    print("🔍 多轮交互评估：测试不同Persona下的行为差异（新版本）")
    print("=" * 80)
    
    # Load model
    print(f"\n📊 加载模型: {model_dir}")
    hf_token = os.getenv("HF_TOKEN")
    
    # Load tokenizer (should have special tokens)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir if Path(model_dir).exists() else base_model,
            use_fast=True,
            token=hf_token if hf_token else None,
        )
        print(f"✅ Loaded tokenizer from {model_dir if Path(model_dir).exists() else base_model}")
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer from {model_dir}: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            token=hf_token if hf_token else None,
        )
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
        tokenizer.add_special_tokens(special_tokens)
        print(f"✅ Loaded tokenizer from base model and added special tokens")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get target vocab size
    target_vocab_size = len(tokenizer)
    print(f"📏 Target vocab size: {target_vocab_size}")
    
    # Load base model
    print(f"🔄 Loading base model...", flush=True)
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
        )
    except Exception as e:
        print(f"⚠️  8-bit量化失败 ({e}), 使用bfloat16", flush=True)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None,
        )
    
    # Resize embeddings BEFORE loading PEFT
    if len(tokenizer) != base_model_obj.get_input_embeddings().num_embeddings:
        print(f"⚠️  Resizing token embeddings from {base_model_obj.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
        base_model_obj.resize_token_embeddings(len(tokenizer))
    
    # Load PEFT adapter
    print("🔄 Loading PEFT adapter...", flush=True)
    try:
        policy_model = PeftModel.from_pretrained(base_model_obj, model_dir)
        print("✅ Loaded PEFT adapter", flush=True)
    except Exception as e:
        print(f"⚠️  Failed to load PEFT adapter: {e}, using base model", flush=True)
        policy_model = base_model_obj
    
    policy_model.eval()
    print("✅ 模型加载完成", flush=True)
    
    # Load test states
    print(f"\n📂 Loading test states from: {test_states_path}", flush=True)
    try:
        test_states = load_jsonl(Path(test_states_path))
        print(f"✅ Loaded {len(test_states)} test states", flush=True)
    except Exception as e:
        print(f"❌ Failed to load test states: {e}", flush=True)
        raise
    
    if max_samples:
        import random
        rng = random.Random(seed)
        original_count = len(test_states)
        test_states = rng.sample(test_states, min(max_samples, len(test_states)))
        print(f"📊 Sampled {len(test_states)} from {original_count} test states", flush=True)
    
    print(f"\n📊 评估 {len(test_states)} 个测试样本", flush=True)
    print(f"📊 每个样本测试 {len(PERSONAS)} 个personas", flush=True)
    print(f"📊 最大轮次: {max_turns}", flush=True)
    
    # Use local base model for code generation (not OpenAI API)
    # The trained policy model is used for action selection, base model for code generation
    use_openai = False  # Always use local base model for code generation
    if use_openai:
        print("✅ 使用OpenAI API进行代码生成", flush=True)
    else:
        print("✅ 使用本地Base模型（Llama）进行代码生成", flush=True)
    
    # Evaluate each state with each persona
    results = []
    persona_stats = defaultdict(lambda: {
        "total_conversations": 0,
        "total_turns": 0,
        "clarify_turns": 0,
        "execute_turns": 0,
        "avg_turns_per_conversation": 0.0,
        "clarify_rate": 0.0,
        "multi_turn_clarify_count": 0,
        # New metrics
        "task_success_count": 0,  # task_score >= 1.0
        "task_evaluated_count": 0,  # conversations with code execution
        "task_success_rate": 0.0,
        "avg_task_score": 0.0,
        "avg_test_pass_rate": 0.0,
        "soft_task_success_count": 0,  # task_score >= 0.5
        "soft_task_success_rate": 0.0,
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
            # Use original_instruct_prompt if requested and available
            if use_original_query and "original_instruct_prompt" in current_state and current_state["original_instruct_prompt"]:
                current_state["query"] = current_state["original_instruct_prompt"]
                print(f"    ✅ Using original query (length: {len(current_state['query'])}) instead of masked query")
            else:
                print(f"    📝 Using masked query (length: {len(current_state.get('query', ''))})")
            current_state["persona"] = {
                "name": persona_name,
                "patience": persona_obj.patience,
                "expertise": persona_obj.expertise,
            }
            
            conversation = []
            actions_taken = []
            task_score = None
            task_completed = False
            
            for turn in range(max_turns):
                # Select action using trained model
                action = select_action_with_model(
                    current_state,
                    tokenizer,
                    policy_model,
                    current_state["persona"],
                )
                
                # Generate assistant message
                assistant_msg = generate_assistant_message(
                    action,
                    current_state,
                    domain=current_state.get("domain", "coding"),
                    use_openai=use_openai,
                    base_model=base_model,
                    base_model_obj=base_model_obj,
                    tokenizer=tokenizer,
                    policy_model=policy_model,  # Use trained model for code generation
                )
                
                # Get user reaction
                user_reaction = react(
                    current_state["query"],
                    assistant_msg,
                    persona_obj,
                    llm_model=llm_model,
                    total_questions_asked=sum(1 for a in actions_taken if a == "Clarify"),
                    dialogue_turn=current_state.get("dialogue_turn", 0),
                    disclosure_rule=current_state.get("disclosure_rule"),
                )
                
                # Calculate task_score if Execute action
                if action == "Execute" and current_state.get("domain") == "coding":
                    code = extract_code_from_text(assistant_msg)
                    # Support both "convcodeworld_tests" and "test" field names
                    tests = current_state.get("convcodeworld_tests") or current_state.get("test")
                    if code and tests:
                        task_score = score_code_passfail(code, tests, timeout=30, debug=(state_idx < 2))
                        task_completed = (task_score is not None and task_score >= 1.0)
                    else:
                        task_score = 0.0
                        task_completed = False
                
                # Record turn
                turn_data = {
                    "turn": turn,
                    "action": action,
                    "assistant_msg": assistant_msg[:200] + "..." if len(assistant_msg) > 200 else assistant_msg,
                    "user_reaction": user_reaction.get("user_reply", "")[:100] + "..." if len(user_reaction.get("user_reply", "")) > 100 else user_reaction.get("user_reply", ""),
                    "answered_clarification": user_reaction.get("meta", {}).get("answered_clarification", 0),
                }
                if action == "Execute":
                    turn_data["task_score"] = task_score
                    turn_data["task_completed"] = task_completed
                conversation.append(turn_data)
                actions_taken.append(action)
                
                print(f"    Turn {turn}: {action}", end="", flush=True)
                if action == "Execute" and task_score is not None:
                    print(f" (task_score: {task_score:.3f}, success: {task_completed})", flush=True)
                else:
                    print(flush=True)
                
                # Update state for next turn
                if action == "Execute":
                    # Execute ends conversation
                    break
                else:
                    # Clarify: update state and continue
                    current_state = update_state_for_next_turn(
                        current_state,
                        user_reaction,
                        assistant_msg,
                        is_same_turn=False,
                    )
            
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
                "task_score": task_score,
                "task_completed": task_completed,
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
            
            # Update task success stats
            if task_score is not None:
                stats["task_evaluated_count"] += 1
                stats["avg_task_score"] += task_score
                stats["avg_test_pass_rate"] += task_score
                if task_score >= 1.0:
                    stats["task_success_count"] += 1
                if task_score >= 0.5:
                    stats["soft_task_success_count"] += 1
    
    # Calculate final stats
    for persona_name, stats in persona_stats.items():
        if stats["total_conversations"] > 0:
            stats["avg_turns_per_conversation"] = stats["total_turns"] / stats["total_conversations"]
            total_actions = stats["clarify_turns"] + stats["execute_turns"]
            if total_actions > 0:
                stats["clarify_rate"]ß = stats["clarify_turns"] / total_actions
        
        # Calculate task success metrics
        if stats["task_evaluated_count"] > 0:
            stats["task_success_rate"] = (stats["task_success_count"] / stats["task_evaluated_count"]) * 100
            stats["soft_task_success_rate"] = (stats["soft_task_success_count"] / stats["task_evaluated_count"]) * 100
            stats["avg_task_score"] = stats["avg_task_score"] / stats["task_evaluated_count"]
            stats["avg_test_pass_rate"] = stats["avg_test_pass_rate"] / stats["task_evaluated_count"]
    
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
            if stats["task_evaluated_count"] > 0:
                print(f"  Task Success Rate: {stats['task_success_rate']:.2f}% ({stats['task_success_count']}/{stats['task_evaluated_count']})")
                print(f"  Soft Task Success (>=50%): {stats['soft_task_success_rate']:.2f}%")
                print(f"  平均Task Score: {stats['avg_task_score']:.4f}")
                print(f"  平均通过率: {stats['avg_test_pass_rate']:.4f}")
    
    # Save results
    if output_path:
        output = {
            "summary": dict(persona_stats),
            "detailed_results": results,
        }
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {output_path}")
    
    return results, persona_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--test_states", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_original_query", action="store_true", 
                        help="Use original_instruct_prompt instead of masked query")
    
    args = parser.parse_args()
    
    evaluate_multi_turn_conversation(
        model_dir=args.model_dir,
        base_model=args.base_model,
        test_states_path=args.test_states,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        output_path=args.output,
        llm_model=args.llm_model,
        seed=args.seed,
        use_original_query=args.use_original_query,
    )
