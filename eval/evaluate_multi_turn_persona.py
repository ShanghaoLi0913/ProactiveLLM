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
import copy
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
    build_clean_execute_query,
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
    # Only decode newly generated tokens (exclude the input prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return generated_text


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Ensure disclosure_rule has disclosed_info field
                # (aligned with generate_trajectories state initialization)
                dr = item.get("disclosure_rule")
                if dr and "disclosed_info" not in dr:
                    dr["disclosed_info"] = {
                        "edge_cases": [],
                        "input_constraints": [],
                        "output_format": [],
                        "validation_rules": [],
                    }
                data.append(item)
    return data


def select_action_with_model(
    state: Dict,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    persona: Dict,
) -> str:
    """Use trained model to select action via natural generation.

    Uses the same prompt format as training (render_state → chat template → generate),
    then detects Clarify vs Execute from the style of the generated text:
    code starters (```, def, import...) → Execute; natural language → Clarify.
    """
    from policy.infer import build_action_selection_chat_prompt, pick_action_from_generation
    from policy.render_state import render_state as render_state_with_persona
    state_text = render_state_with_persona(state, persona=persona)
    prompt = build_action_selection_chat_prompt(state_text, tokenizer)
    return pick_action_from_generation(model, tokenizer, prompt, max_new_tokens=30)


def select_action_prompt_only(
    state: Dict,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    persona: Dict,
) -> str:
    """Prompt-only baseline: base model + explicit persona-aware instructions.

    Instead of DPO-trained weights, we give the base model a system prompt that
    explicitly describes how each persona should decide between Clarify and Execute.
    The model generates a short response; action is detected from content style
    (same as select_action_with_model).
    """
    from policy.infer import pick_action_from_generation
    from policy.render_state import render_state as render_state_with_persona

    persona_name = persona.get("name", "Unknown")
    patience = persona.get("patience", "mid")

    # Build persona description — no strategy rules, only user characteristics
    if patience == "low":  # Busy-Developer
        persona_desc = (
            "The user is a Busy-Developer. They have low patience and mid-level expertise. "
            "They are time-constrained and prefer efficiency over thoroughness."
        )
    elif patience == "high":  # Novice-Learner
        persona_desc = (
            "The user is a Novice-Learner. They have high patience but low expertise. "
            "They are unfamiliar with the domain and may benefit from a more careful approach."
        )
    else:  # Experienced-Engineer (mid patience)
        persona_desc = (
            "The user is an Experienced-Engineer. They have moderate patience and high expertise. "
            "They are technically skilled but may not always provide complete specifications."
        )

    state_text = render_state_with_persona(state, persona=persona)

    # System prompt: persona description + task framing, no decision rules
    system_msg = (
        f"{persona_desc}\n\n"
        "You are a coding assistant. Given the user's profile and their task request, "
        "decide how to respond:\n"
        "- If you need more information to write correct code, ask a clarifying question.\n"
        "- If you have enough information, write the code directly "
        "(start with ```python or def or import).\n\n"
        "Consider the user's characteristics when deciding."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": state_text},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = f"{system_msg}\n\n{state_text}\n\nAssistant:"

    return pick_action_from_generation(model, tokenizer, prompt, max_new_tokens=30)


def generate_assistant_message(
    action: str,
    state: Dict,
    domain: str = "coding",
    model=None,
    tokenizer=None,
    use_openai: bool = False,
    base_model: Optional[str] = None,
    temperature: float = 0.7,
    initial_state: Optional[Dict] = None,
) -> str:
    """Generate assistant message for the given action (Clarify or Execute-style text)."""
    prompts = build_action_prompts(domain)
    action_prompt = prompts.get(action, prompts["Execute"])
    # For Execute, use clean query (initial query + structured disclosed info)
    # to avoid polluting code generation with conversation history.
    if action == "Execute" and initial_state is not None:
        task_prompt = build_clean_execute_query(initial_state, state)
    else:
        task_prompt = state.get("query", "")

    # Enhance Clarify prompt with disclosure_rule guidance (aligned with training pipeline)
    if action == "Clarify":
        disclosure_rule = state.get("disclosure_rule")
        if disclosure_rule:
            masked_fields = disclosure_rule.get("masked_fields", {})
            guidance_parts = []
            if masked_fields.get("input_constraints"):
                guidance_parts.append("- Input constraints or default values")
            if masked_fields.get("output_format"):
                guidance_parts.append("- Output format or return type")
            if masked_fields.get("edge_cases"):
                guidance_parts.append("- Edge cases to handle")
            if masked_fields.get("validation_rules"):
                guidance_parts.append("- Validation rules or error handling")
            if guidance_parts:
                guidance_text = "\n".join(guidance_parts)
                action_prompt = f"""{action_prompt}

IMPORTANT: The task description may be missing some information. Consider asking about:
{guidance_text}

Generate 1-2 specific questions that would help clarify these missing aspects."""
    
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
    initial_state: Optional[Dict] = None,
) -> List[str]:
    """Generate k different code candidates for pass@k (local policy model or OpenAI)."""
    candidates = []
    prompts = build_action_prompts(domain)
    action_prompt = prompts.get("Execute", "")
    # Use clean query (initial query + structured disclosed info) to avoid
    # polluting code generation with conversation history noise.
    if initial_state is not None:
        task_prompt = build_clean_execute_query(initial_state, state)
    else:
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
    initial_state: Optional[Dict] = None,
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
        initial_state=initial_state,
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
        disclosure_rule=current_state.get("disclosure_rule"),
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
    prompt_only: bool = False,
    direct_execution: bool = False,
    always_clarify: Optional[int] = None,
    oracle: bool = False,
    ideal_disclosed: bool = False,
):
    """Evaluate model's multi-turn behavior with different personas.

    no_lora=True: 只加载 base_model（用于 Base Llama 对照），不加载 LoRA。
    prompt_only=True: base model + explicit persona instructions (no DPO training).
    direct_execution=True: always execute on turn 0, no clarification (lower bound baseline).
    always_clarify=K: clarify for exactly K rounds, then execute (total K+1 turns).
    oracle=True: use original unmasked instruct_prompt, single-turn Execute, no persona (upper bound).
    ideal_disclosed=True: masked query + all masked_fields disclosed at once, single-turn Execute, no persona.
    """
    
    print("=" * 80)
    print("🔍 多轮交互评估：测试不同Persona下的行为差异")
    print("=" * 80)
    
    hf_token = os.getenv("HF_TOKEN")
    
    # Load model
    # Oracle / Ideal Disclosed are persona-independent single-turn evaluations
    persona_independent = oracle or ideal_disclosed

    if no_lora:
        if oracle:
            mode_label = "Oracle (unmasked) upper bound"
        elif ideal_disclosed:
            mode_label = "Ideal Disclosed upper bound"
        elif direct_execution:
            mode_label = "Direct Execution baseline"
        elif always_clarify is not None:
            mode_label = f"Always-Clarify (K={always_clarify}) baseline"
        elif prompt_only:
            mode_label = "Prompt-only baseline"
        else:
            mode_label = "Base 模型（无 LoRA）"
        print(f"\n📊 加载 {mode_label}: {base_model}")
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
    if persona_independent:
        print(f"📊 Persona-independent 模式：每个样本测试 1 次（无 persona）", flush=True)
    else:
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
    
    # --- Resume support: load already-completed state_ids from output or partial file ---
    completed_state_ids = set()
    results = []
    _incremental_path = output_path + ".partial" if output_path else None
    # Prefer partial file (has latest progress), fall back to final output
    _resume_file = None
    if _incremental_path and os.path.exists(_incremental_path):
        _resume_file = _incremental_path
    elif output_path and os.path.exists(output_path):
        _resume_file = output_path
    if _resume_file:
        try:
            with open(_resume_file, 'r') as f:
                existing = json.load(f)
            existing_results = existing.get("detailed_results", [])
            # A state is complete only if all personas finished (or 1 for persona-independent)
            from collections import Counter as _Counter
            _state_counts = _Counter(r["state_id"] for r in existing_results)
            _required = 1 if persona_independent else len(PERSONAS)
            completed_state_ids = {sid for sid, cnt in _state_counts.items() if cnt >= _required}
            results = [r for r in existing_results if r["state_id"] in completed_state_ids]
            print(f"🔄 Resume: 从 {_resume_file} 恢复 {len(completed_state_ids)} 个已完成 state，跳过", flush=True)
        except (json.JSONDecodeError, KeyError):
            print("⚠️  恢复文件损坏，从头开始", flush=True)
            results = []

    def _save_incremental(results_so_far, persona_stats_so_far):
        """Save current progress after each state (all 3 personas done)."""
        if not _incremental_path:
            return
        # Recompute summary stats for saving
        _ps = {}
        for pn, st in persona_stats_so_far.items():
            _ps[pn] = dict(st)
            if st["total_conversations"] > 0:
                _ps[pn]["avg_turns_per_conversation"] = st["total_turns"] / st["total_conversations"]
                total_actions = st["clarify_turns"] + st["execute_turns"]
                _ps[pn]["clarify_rate"] = st["clarify_turns"] / total_actions if total_actions else 0.0
        out = {"summary": _ps, "detailed_results": results_so_far}
        with open(_incremental_path, 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n🚀 开始评估...", flush=True)
    # Evaluate each state with each persona
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

    # Rebuild persona_stats from resumed results
    for r in results:
        stats = persona_stats[r["persona"]]
        stats["total_conversations"] += 1
        stats["total_turns"] += r["total_turns"]
        stats["clarify_turns"] += r["clarify_count"]
        stats["execute_turns"] += r["execute_count"]
        if r.get("has_multi_turn_clarify"):
            stats["multi_turn_clarify_count"] += 1
        for td in r.get("conversation", []):
            if td.get("action") == "Execute" and "pass_at_k" in td:
                for k in pass_at_k:
                    key = f"pass@{k}"
                    if key in td["pass_at_k"]:
                        stats["pass_at_k"][key]["total"] += 1
                        if td["pass_at_k"][key]:
                            stats["pass_at_k"][key]["passed"] += 1

    for state_idx, initial_state in enumerate(test_states):
        # Skip already-completed states (resume support)
        if initial_state.get("id", "unknown") in completed_state_ids:
            print(f"\n⏭️  跳过已完成: {state_idx + 1}/{len(test_states)}: {initial_state.get('id', 'unknown')}", flush=True)
            continue
        print(f"\n{'='*80}", flush=True)
        print(f"样本 {state_idx + 1}/{len(test_states)}: {initial_state.get('id', 'unknown')}", flush=True)
        print(f"{'='*80}", flush=True)

        # --- Oracle / Ideal Disclosed: persona-independent, single-turn Execute ---
        if persona_independent:
            initial_state_snapshot = copy.deepcopy(initial_state)
            current_state = copy.deepcopy(initial_state)
            persona_label = "Oracle" if oracle else "Ideal_Disclosed"

            if oracle:
                # Replace masked query with full original instruct_prompt
                current_state["query"] = current_state["original_instruct_prompt"]
                initial_state_snapshot["query"] = current_state["original_instruct_prompt"]
                print(f"  Mode: Oracle (unmasked prompt)", flush=True)
            else:
                # Ideal Disclosed: pre-fill all masked_fields into disclosed_info
                dr = current_state.get("disclosure_rule", {})
                masked = dr.get("masked_fields", {})
                # Build disclosed_info from all masked fields
                disclosed_info = {}
                for category, items in masked.items():
                    if isinstance(items, list):
                        disclosed_info[category] = list(items)
                    elif isinstance(items, str):
                        disclosed_info[category] = [items]
                    else:
                        disclosed_info[category] = []
                dr["disclosed_info"] = disclosed_info
                current_state["disclosure_rule"] = dr
                print(f"  Mode: Ideal Disclosed (all masked fields given)", flush=True)

            # Single-turn Execute
            from scripts.generate_trajectories import PERSONAS as _PERSONAS
            _dummy_persona = _PERSONAS[0]  # persona doesn't matter, just needed for react()
            turn_data = _build_execute_turn_data(
                0,
                current_state,
                _dummy_persona,
                pass_at_k,
                use_openai,
                base_model,
                llm_model,
                model=model,
                tokenizer=tokenizer,
                total_questions_asked=0,
                forced_final_execute=False,
                initial_state=initial_state_snapshot,
            )
            result = {
                "state_id": initial_state.get("id", "unknown"),
                "persona": persona_label,
                "conversation": [turn_data],
                "total_turns": 1,
                "actions": ["Execute"],
                "clarify_count": 0,
                "execute_count": 1,
                "has_multi_turn_clarify": False,
            }
            results.append(result)

            stats = persona_stats[persona_label]
            stats["total_conversations"] += 1
            stats["total_turns"] += 1
            stats["execute_turns"] += 1
            for k in pass_at_k:
                key = f"pass@{k}"
                if key in turn_data.get("pass_at_k", {}):
                    stats["pass_at_k"][key]["total"] += 1
                    if turn_data["pass_at_k"][key]:
                        stats["pass_at_k"][key]["passed"] += 1
            print(f"    Turn 0: Execute | pass@1={turn_data.get('pass_at_k', {}).get('pass@1', False)}")

            _save_incremental(results, persona_stats)
            continue
        # --- End Oracle / Ideal Disclosed ---

        for persona_idx, persona_obj in enumerate(PERSONAS):
            persona_name = persona_obj.name
            print(f"\n  Persona: {persona_name}", flush=True)

            # Initialize state with persona; keep a clean snapshot for Execute query
            initial_state_snapshot = copy.deepcopy(initial_state)
            current_state = copy.deepcopy(initial_state)
            current_state["persona"] = {
                "name": persona_name,
                "patience": persona_obj.patience,
                "expertise": persona_obj.expertise,
            }

            conversation = []
            actions_taken = []
            total_questions_asked = 0

            for turn in range(max_turns):
                # Select action using trained model, prompt-only, direct execution, or always-clarify
                if direct_execution:
                    action = "Execute"
                elif always_clarify is not None:
                    action = "Clarify" if turn < always_clarify else "Execute"
                elif prompt_only:
                    action = select_action_prompt_only(
                        current_state,
                        tokenizer,
                        model,
                        current_state["persona"],
                    )
                else:
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
                        total_questions_asked=total_questions_asked,
                        forced_final_execute=False,
                        initial_state=initial_state_snapshot,
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
                    # Track total questions asked (count '?' like training pipeline)
                    total_questions_asked += assistant_msg.count("?")
                    user_reaction = react(
                        current_state["query"],
                        assistant_msg,
                        persona_obj,
                        llm_model=llm_model,
                        total_questions_asked=total_questions_asked,
                        disclosure_rule=current_state.get("disclosure_rule"),
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
                    total_questions_asked=total_questions_asked,
                    forced_final_execute=True,
                    initial_state=initial_state_snapshot,
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

        # Incremental save after each state (all 3 personas done)
        _save_incremental(results, persona_stats)
        print(f"  💾 已保存进度 ({state_idx + 1}/{len(test_states)})", flush=True)

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
        # Clean up partial file
        if _incremental_path and os.path.exists(_incremental_path):
            os.rename(_incremental_path, output_path)
            print(f"  (partial → final: {output_path})")
    
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
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        help="Prompt-only baseline: base model + explicit persona instructions for action selection (no DPO training).",
    )
    parser.add_argument(
        "--direct_execution",
        action="store_true",
        help="Direct Execution baseline: always execute on turn 0, no clarification.",
    )
    parser.add_argument(
        "--always_clarify",
        type=int,
        default=None,
        metavar="K",
        help="Always-Clarify baseline: clarify for exactly K rounds, then execute (total K+1 turns).",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Oracle upper bound: use original unmasked instruct_prompt, single-turn Execute, no persona.",
    )
    parser.add_argument(
        "--ideal_disclosed",
        action="store_true",
        help="Ideal Disclosed upper bound: masked query + all masked_fields disclosed at once, single-turn Execute.",
    )

    args = parser.parse_args()

    if not args.no_lora and not args.prompt_only and not args.direct_execution and args.always_clarify is None and not args.oracle and not args.ideal_disclosed and not args.model_dir:
        parser.error("需要 --model_dir（LoRA 目录），或使用 --no_lora / --prompt_only / --direct_execution / --always_clarify / --oracle / --ideal_disclosed 评估 Base 模型")
    if args.no_lora and args.model_dir:
        print("ℹ️  已指定 --no_lora，忽略 --model_dir", flush=True)

    # 与 scripts/generate_trajectories 一致：默认 gpt-4o-mini；--user_dummy 则 llm_model=None
    llm_for_user = None if args.user_dummy else (args.llm_model.strip() or None)
    
    # prompt_only, direct_execution, always_clarify, oracle, ideal_disclosed imply no_lora (uses base model)
    if args.prompt_only or args.direct_execution or args.always_clarify is not None or args.oracle or args.ideal_disclosed:
        args.no_lora = True

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
        prompt_only=args.prompt_only,
        direct_execution=args.direct_execution,
        always_clarify=args.always_clarify,
        oracle=args.oracle,
        ideal_disclosed=args.ideal_disclosed,
    )
