"""
Step 1: Generate trajectories (state + action + assistant output + simulator reaction)

Multi-turn conversation mode: Generates full conversations until task completion.
Each turn is a separate trajectory entry with its own state and action.

Input: States (from synthetic or dataset)
Output: Trajectories JSONL to data/logs/
Each trajectory contains: {state, action, action_prompt, assistant_msg, persona, user_reaction, turn}
"""
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import nullcontext
import time

# Ensure project root is on sys.path for package imports when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import PERSONAS, react
from utils.compute_task_uncertainty import compute_task_uncertainty_from_state


def set_global_seed(seed: int) -> None:
    """Best-effort seeding for reproducible trajectory generation."""
    import random
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class LocalHFChatGenerator:
    """Local HF chat completion (system+user) with one-time model load + reuse."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # NOTE: some Transformers builds reject `generator=` kwarg in generate().
        # We rely on global torch seeding for reproducibility instead.
        if seed is not None:
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    def chat_complete(self, system_prompt: str, user_prompt: str) -> str:
        # Match the rest of the codebase: prompts are system + user
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
        return text


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_action_prompts(domain: str) -> Dict[str, str]:
    """Load behavior templates from prompts/ directory.
    
    Sequential Decision Process: actions are Clarify or Execute.
    """
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        return {
            "Clarify": load_prompt(base / "coding_clarify.txt"),
            "Execute": load_prompt(base / "coding_execute.txt"),
        }
    elif domain == "planning":
        return {
            "Clarify": load_prompt(base / "planning_clarify.txt"),
            "Execute": load_prompt(base / "planning_execute.txt"),
        }
    else:
        raise ValueError(f"Unknown domain: {domain}")


def build_interaction_prompts(domain: str) -> Dict[str, Dict[str, str]]:
    """Load interaction-specific prompts for sequential decision process.
    
    Returns a dict like:
    {
        "Clarify": {"clarify": "..."},
        "Execute": {"execute": "..."}
    }
    """
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        return {
            "Clarify": {
                "clarify": load_prompt(base / "coding_clarify.txt"),
            },
            "Execute": {
                "execute": load_prompt(base / "coding_execute.txt"),
            },
        }
    else:
        return {
            "Clarify": {
                "clarify": load_prompt(base / "planning_clarify.txt"),
            },
            "Execute": {
                "execute": load_prompt(base / "planning_execute.txt"),
            },
        }


def synth_states(domain: str, n: int) -> List[Dict]:
    """Generate synthetic states for quick testing."""
    samples = []
    for i in range(n):
        # Generate initial state (dialogue_turn=0 for first turn)
        query = "帮我写个 Python 爬虫" if domain == "coding" else "帮我规划今天的待办"
        # Compute task_uncertainty from query
        temp_state = {"query": query}
        task_uncertainty = compute_task_uncertainty_from_state(temp_state)
        
        samples.append(
            {
                "id": f"{domain}-{i}",
                "domain": domain,
                "query": query,
                "dialogue_turn": 0,  # Start from 0 for initial state
                "prev_reject": 0,
                "task_uncertainty": task_uncertainty,  # Computed from query
            }
        )
    return samples


def load_states_from_dataset(dataset_path: Path, domain: str, limit: Optional[int] = None) -> List[Dict]:
    """Load states from JSONL dataset file.
    
    If limit > dataset size, cycles through the dataset to reach the limit.
    """
    # First, load all available states
    all_states = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            row = json.loads(line)
            all_states.append(row)
    
    if not all_states:
        return []
    
    # If limit is None or <= dataset size, return up to limit
    if limit is None or limit <= len(all_states):
        states_to_use = all_states[:limit] if limit else all_states
        need_cycling = False
    else:
        # Cycle through dataset to reach limit
        states_to_use = []
        cycles = limit // len(all_states)
        remainder = limit % len(all_states)
        
        for cycle in range(cycles):
            for row in all_states:
                states_to_use.append((row, cycle))
        
        for i in range(remainder):
            states_to_use.append((all_states[i], cycles))
        
        need_cycling = True
    
    # Convert to state format
    result = []
    for i, item in enumerate(states_to_use):
        if need_cycling:
            row, cycle = item
            original_id = row.get("id", f"ds-{i % len(all_states)}")
            # Generate unique ID for each cycle
            state_id = f"{original_id}-{cycle}" if cycle > 0 else original_id
        else:
            row = item
            original_id = row.get("id", f"ds-{i}")
            state_id = original_id
        
        # Compute task_uncertainty if not provided
        query = row["query"]
        if "task_uncertainty" in row:
            task_uncertainty = float(row["task_uncertainty"])
        else:
            # Compute from query if not provided
            temp_state = {"query": query}
            task_uncertainty = compute_task_uncertainty_from_state(temp_state)
        
        # Support both "test" (BigCodeBench) and "convcodeworld_tests" field names
        tests = row.get("convcodeworld_tests") or row.get("test")
        
        # Build state dict, preserving important fields
        state_dict = {
            "id": state_id,
            "domain": domain,
            "query": query,
            "dialogue_turn": int(row.get("dialogue_turn", 0)),  # Start from 0
            "prev_reject": int(row.get("prev_reject", 0)),
            "task_uncertainty": task_uncertainty,
            "convcodeworld_tests": tests,  # Support both "test" and "convcodeworld_tests"
        }
        
        # Preserve additional fields that may be needed
        for key in ["original_instruct_prompt", "canonical_solution", "test", "entry_point"]:
            if key in row:
                state_dict[key] = row[key]
        
        # Preserve disclosure_rule (critical for disclosure mechanism)
        if "disclosure_rule" in row:
            disclosure_rule = row["disclosure_rule"].copy()
            # Ensure disclosed_info is initialized if not present
            if "disclosed_info" not in disclosure_rule:
                disclosure_rule["disclosed_info"] = {
                    "edge_cases": [],
                    "input_constraints": [],
                    "output_format": [],
                    "validation_rules": [],
                }
            state_dict["disclosure_rule"] = disclosure_rule
        
        result.append(state_dict)
    return result


def dummy_llm_output(state: Dict, action_prompt: str) -> str:
    """Generate a dummy LLM output for testing (synthetic mode).
    
    Returns a simple placeholder that indicates the action type.
    For Clarify action, includes a question mark to trigger user reaction logic.
    """
    # Check if this is a Clarify prompt (contains "ask" or "question" or "clarify")
    prompt_lower = action_prompt.lower()
    if "ask" in prompt_lower or "question" in prompt_lower or "clarify" in prompt_lower:
        # Return a dummy clarifying question
        return "What specific requirements do you have for this task? Do you need any special features?"
    else:
        # Return dummy code/execution output
        return "```python\ndef solution():\n    # Implementation here\n    pass\n```"
    """Placeholder LLM output for testing without API calls."""
    lower = action_prompt.lower()
    if "ask up to two clarifying" in lower:
        return "请问目标网站与输出格式？随后给出实现步骤与代码。"
    if "ask exactly one" in lower or "one key clarifying" in lower:
        return "请问需要爬取哪个网站的数据？然后我会给出代码。"
    return "这是一个最小可运行的示例代码/计划。"


def llm_output(
    state: Dict,
    action_prompt: str,
    model: str,
    conversation_history: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate assistant output using OpenAI API.
    
    Args:
        state: Current state dict
        action_prompt: Action template/prompt
        model: LLM model name
        conversation_history: Optional list of previous turns [{role, content}, ...]
    """
    from llm.provider import chat_complete
    system = action_prompt
    
    # Build user message from state query
    # If query contains conversation history (multi-turn), use it directly
    # Otherwise, format as initial task
    user = f"[Task]\n{state['query']}"
    
    return chat_complete(system, user, model=model, max_tokens=400, temperature=temperature, top_p=top_p)


def sanitize_clarify_message(assistant_msg: str) -> str:
    """Ensure Clarify action does not include code output."""
    # Remove fenced code blocks
    sanitized = re.sub(r"```.*?```", "", assistant_msg, flags=re.DOTALL)
    # If an opening fence exists without a closing fence, drop everything after it
    if "```" in sanitized:
        sanitized = sanitized.split("```", 1)[0]
    # Remove inline code-ish lines
    sanitized_lines = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("import "):
            continue
        sanitized_lines.append(line)
    sanitized = "\n".join(sanitized_lines).strip()
    if "?" not in sanitized:
        sanitized = (sanitized + "\n" if sanitized else "")
        sanitized += "Could you clarify any edge cases or constraints I should handle?"
    return sanitized


def select_mainline_action_from_persona(persona, state: Optional[Dict] = None) -> str:
    """
    Select mainline action based on persona characteristics and state.
    
    Sequential Decision Process: actions are Clarify or Execute.
    
    COLM 2026 Strategy: Different personas have different Clarify thresholds
    - Busy-Developer (low patience): Only Clarify if task_uncertainty > 0.7 (rarely)
    - Experienced-Engineer (mid patience): Clarify if task_uncertainty > 0.5 (sometimes)
    - Novice-Learner (high patience): Clarify if task_uncertainty > 0.3 (often)
    
    This creates natural persona differentiation in average trajectory length:
    - Busy: ~1.2 turns (mostly Execute)
    - Experienced: ~1.5 turns (context-dependent)
    - Novice: ~1.8 turns (mostly Clarify first)
    """
    prev_reject = state.get("prev_reject", 0) if state else 0
    
    # Condition 1: If user rejected in previous turn, execute (don't ask more questions)
    if prev_reject > 0:
        return "Execute"
    
    # Get task uncertainty from state
    task_uncertainty = state.get("task_uncertainty", 0.5) if state else 0.5
    dialogue_turn = state.get("dialogue_turn", 0) if state else 0
    
    # Condition 2: First turn decision based on persona + task_uncertainty
    if dialogue_turn == 0:
        # Different personas have different Clarify thresholds
        if persona.patience == "low":  # Busy-Developer
            # Very high threshold: almost never Clarify (only on extremely uncertain tasks)
            # Raised from 0.7 to 0.9 to create clear gap vs Experienced (73% tasks have uncertainty>0.7)
            clarify_threshold = 0.9
        elif persona.patience == "high":  # Novice-Learner
            # Low threshold: Clarify even if moderately uncertain
            clarify_threshold = 0.3
        else:  # Experienced-Engineer (mid patience)
            # Medium threshold: Clarify if reasonably uncertain
            # Raised from 0.5 to 0.6 for slightly more separation from Novice
            clarify_threshold = 0.6
        
        # Decision: Clarify if uncertainty exceeds threshold
        if task_uncertainty > clarify_threshold:
            return "Clarify"
        else:
            return "Execute"
    
    # Condition 3: After first turn (dialogue_turn > 0)
    # Multi-turn conversation: persona differences should persist
    # Check if previous turn was Clarify and user answered
    query = state.get("query", "") if state else ""
    prev_was_clarify = "[User]:" in query  # If query contains "[User]:", previous turn was Clarify and user answered
    
    if prev_was_clarify:
        # Previous turn was Clarify and user answered
        # Check if uncertainty is still high enough to continue Clarify (based on persona)
        # Different personas have different thresholds for continuing Clarify
        if persona.patience == "low":  # Busy-Developer
            # Even if uncertainty is high, Execute after one Clarify (don't keep asking)
            return "Execute"
        elif persona.patience == "high":  # Novice-Learner
            # Continue Clarify if uncertainty is still high
            clarify_threshold = 0.3
            if task_uncertainty > clarify_threshold:
                return "Clarify"
            else:
                return "Execute"
        else:  # Experienced-Engineer (mid patience)
            # Continue Clarify only if uncertainty is very high
            clarify_threshold = 0.5
            if task_uncertainty > clarify_threshold:
                return "Clarify"
            else:
                return "Execute"
    else:
        # Previous turn was not Clarify (or user didn't answer)
        # This shouldn't happen if Execute ends conversation, but handle it anyway
        return "Execute"


def check_task_completion(state: Dict, assistant_msg: str, domain: str) -> bool:
    """Check if task is completed based on assistant output and tests.
    
    For coding tasks, task is completed ONLY if:
    1. Code is present in assistant message
    2. All test cases pass (score == 1.0)
    
    If tests are available but execution fails or times out, task is NOT completed.
    """
    if domain == "coding":
        # Check if code is present
        has_code = (
            "```" in assistant_msg or
            "def " in assistant_msg or
            "class " in assistant_msg
        )
        if not has_code:
            return False
        
        # Check for tests in multiple possible field names
        # Some datasets use "convcodeworld_tests", others use "test"
        tests = state.get("convcodeworld_tests") or state.get("test")
        if tests:
            try:
                from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail
                code = extract_code_from_text(assistant_msg)
                if code:
                    score = score_code_passfail(code, tests, timeout=30)
                    # Task completed ONLY if ALL test cases pass (score == 1.0)
                    return score == 1.0
                else:
                    # Code extraction failed
                    return False
            except Exception:
                # Test execution failed or timed out - task NOT completed
                return False
        
        # If no tests available, cannot verify completion - return False
        # (Previously returned has_code, but this is too lenient)
        return False
    
    # For planning domain, assume task completed if response is long enough
    return len(assistant_msg.split()) > 50


def update_state_for_next_turn(current_state: Dict, user_reaction: Dict, assistant_msg: str, is_same_turn: bool = False) -> Dict:
    """Update state for next dialogue turn based on user reaction.
    
    Updates:
    - dialogue_turn: increment by 1 (only if is_same_turn=False, i.e., moving to next turn)
    - prev_reject: set to 1 if user rejected
    - query: append conversation history
    - task_uncertainty: recalculate based on updated query (task becomes clearer after user answers)
    
    Args:
        current_state: Current state dict
        user_reaction: User reaction dict
        assistant_msg: Assistant message
        is_same_turn: If True, this is within the same turn (multi-interaction), don't increment dialogue_turn
    """
    new_state = current_state.copy()
    
    # Update dialogue turn (only if moving to next turn, not within same turn)
    if not is_same_turn:
        new_state["dialogue_turn"] = current_state.get("dialogue_turn", 1) + 1
    # If is_same_turn=True, keep the same dialogue_turn (same turn, multiple interactions)
    
    # Update prev_reject if user rejected
    if user_reaction.get("meta", {}).get("reject_signal", 0) > 0:
        new_state["prev_reject"] = 1
    else:
        # Reset prev_reject if user didn't reject (might want to keep history)
        # For now, keep it as is unless explicitly rejected
        pass
    
    # Update query to include conversation history
    user_reply = user_reaction.get("user_reply", "")
    meta = user_reaction.get("meta", {})
    
    # Persona → Behavior Mapping: Task Uncertainty Update (Equation 9)
    # If Assistant Clarifies and user answers:
    #   U_{t+1} = U_t (1 - 0.5 · answer_clarity)
    # If Assistant Executes:
    #   task_uncertainty does not update
    
    # Check if user answered clarification (has answer_clarity > 0)
    answer_clarity = meta.get("answer_clarity", 0.0)
    answered_clarification = meta.get("answered_clarification", 0)
    
    # Update disclosure_rule.disclosed_info to track what has been disclosed
    disclosed_items = meta.get("disclosed_items", {})
    if disclosed_items and current_state.get("disclosure_rule"):
        # 深拷贝disclosure_rule，避免修改原始state
        import copy
        disclosure_rule = copy.deepcopy(current_state["disclosure_rule"])
        
        # 确保disclosure_rule存在disclosed_info字段
        if "disclosed_info" not in disclosure_rule:
            disclosure_rule["disclosed_info"] = {
                "edge_cases": [],
                "input_constraints": [],
                "output_format": [],
                "validation_rules": [],
            }
        
        # 更新disclosed_info，添加本次披露的信息（去重）
        disclosed_info = disclosure_rule["disclosed_info"]
        for category in ["edge_cases", "input_constraints", "output_format", "validation_rules"]:
            if category in disclosed_items:
                existing = set(disclosed_info.get(category, []))
                new_items = set(disclosed_items[category])
                disclosed_info[category] = list(existing | new_items)  # 并集，去重
        
        # 更新state中的disclosure_rule
        new_state["disclosure_rule"] = disclosure_rule
    
    if answered_clarification > 0 and answer_clarity > 0:
        # User answered clarification: update task_uncertainty using Equation 9
        # U_{t+1} = U_t (1 - 0.5 · answer_clarity)
        current_uncertainty = current_state.get("task_uncertainty", 0.5)
        new_task_uncertainty = current_uncertainty * (1 - 0.5 * answer_clarity)
        new_state["task_uncertainty"] = max(0.0, min(1.0, new_task_uncertainty))  # Clamp to [0, 1]
        
        # Update query to include conversation history
        new_state["query"] = f"{current_state['query']}\n\n[Assistant]: {assistant_msg}\n[User]: {user_reply}"
    else:
        # User rejected or no answer: task_uncertainty does not update
        # If Assistant Executes: task_uncertainty does not update
        new_state["query"] = f"{current_state['query']}\n\n[Assistant]: {assistant_msg}"
        # task_uncertainty保持不变
    
    return new_state


def generate_multi_turn_conversation(initial_state: Dict, domain: str,
                                     llm_model: Optional[str] = None,
                                     persona_idx: int = 0,
                                     max_turns: int = 5,
                                     action_selection_fn=None,
                                     force_first_action: Optional[str] = None,
                                     local_generator: Optional[LocalHFChatGenerator] = None,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9,
                                     trajectory_id: Optional[str] = None) -> List[Dict]:
    """
    Generate a multi-turn conversation until task completion or max turns.
    
    Args:
        initial_state: Starting state
        domain: "coding" or "planning"
        llm_model: LLM model name (None for dummy)
        persona_idx: Index of persona to use
        max_turns: Maximum number of dialogue turns
        action_selection_fn: Function(state) -> action (LOW/MID/HIGH). If None, uses persona-based selection.
        force_first_action: If provided ("Execute" or "Clarify"), force the first action instead of auto-selecting.
        trajectory_id: Optional unique ID for this trajectory. If None, will be auto-generated.
    
    Returns:
        List of trajectory dicts, one per turn
    """
    import time
    prompts = build_action_prompts(domain)
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    trajectories = []
    current_state = initial_state.copy()
    total_questions_asked = 0
    
    # Generate unique trajectory ID if not provided
    if trajectory_id is None:
        state_id = initial_state.get("id", "unknown")
        timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
        trajectory_id = f"{state_id}_{persona.name}_{timestamp}"
    
    for turn in range(max_turns):
        # Select action for this turn
        if turn == 0 and force_first_action:
            # Force first action if specified (for sampling strategy)
            action = force_first_action
        elif action_selection_fn:
            action = action_selection_fn(current_state)
        else:
            action = select_mainline_action_from_persona(persona, current_state)
        
        # Guard against invalid/None action from selection functions
        if action not in prompts:
            action = "Execute"
        action_prompt = prompts[action]
        
        # Enhance Clarify prompt with disclosure_rule information (混合方案)
        if action == "Clarify":
            disclosure_rule = current_state.get("disclosure_rule")
            if disclosure_rule:
                # Extract key information from disclosure_rule to guide question generation
                disclosure_info = disclosure_rule.get("disclosure_info", {})
                masked_fields = disclosure_rule.get("masked_fields", {})
                
                # Build guidance text for the model
                guidance_parts = []
                
                # Check what information is available but masked
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
                    # Enhance prompt with disclosure guidance
                    action_prompt = f"""{action_prompt}

IMPORTANT: The task description may be missing some information. Consider asking about:
{guidance_text}

Generate 1-2 specific questions that would help clarify these missing aspects."""
        
        # Generate assistant message
        if local_generator is not None:
            assistant_msg = local_generator.chat_complete(action_prompt, f"[Task]\n{current_state['query']}")
        elif llm_model:
            assistant_msg = llm_output(current_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
        else:
            assistant_msg = dummy_llm_output(current_state, action_prompt)
        
        # Enforce action/message consistency: Clarify should not include code
        if action == "Clarify":
            assistant_msg = sanitize_clarify_message(assistant_msg)
        
        # Track total questions asked to influence user patience over multiple clarifications
        total_questions_asked += assistant_msg.count("?")
        
        # Get simulator reaction
        # Note: react() supports llm_model=None (dummy mode), but for realistic testing, llm_model is recommended
        # Pass dialogue_turn to enable patience decay
        reaction = react(
            current_state["query"],
            assistant_msg,
            persona,
            llm_model=llm_model,
            total_questions_asked=total_questions_asked,
            disclosure_rule=current_state.get("disclosure_rule"),
            dialogue_turn=current_state.get("dialogue_turn", 0),
        )
        
        # Track if edge_cases info was obtained through Clarify (for "contrastive conflicts")
        has_edge_cases_info = current_state.get("has_edge_cases_info", False)
        if action == "Clarify" and reaction.get("meta", {}).get("answered_clarification", 0) > 0:
            # Check if the answer contains edge_cases information
            user_reply = reaction.get("user_reply", "")
            edge_case_keywords = ["edge case", "empty", "negative", "zero", "null", "none", "constraint", "special", "exception", "invalid", "error", "boundary", "extreme"]
            if any(keyword.lower() in user_reply.lower() for keyword in edge_case_keywords):
                has_edge_cases_info = True
            
            # If disclosure_rule has edge_cases, assume edge_cases info was obtained
            disclosure_rule = current_state.get("disclosure_rule")
            if disclosure_rule:
                disclosure_info = disclosure_rule.get("disclosure_info", {})
                input_constraints = disclosure_info.get("input_constraints", {})
                if input_constraints.get("edge_cases"):
                    has_edge_cases_info = True
        
        # Create trajectory for this turn
        traj = {
            "trajectory_id": trajectory_id,  # Unique ID to link all turns in this trajectory
            "state": current_state.copy(),
            "action": action,
            "action_prompt": action_prompt,
            "assistant_msg": assistant_msg,
            "persona": {
                "name": persona.name,
                "domain": persona.domain,
                "expertise": persona.expertise,
                "patience": persona.patience,
            },
            "user_reaction": reaction,
            "turn": turn + 1,
            "is_mainline": True,  # All turns in multi-turn are mainline
            "is_terminal": False,  # Will be set to True if task completed or user stopped
            "has_edge_cases_info": has_edge_cases_info,  # Track if edge_cases info was obtained
        }
        trajectories.append(traj)
        
        # Check if task is completed (only for Execute action)
        # Use enhanced task completion check that considers persona and edge_cases info
        if action == "Execute":
            # Generate 3 versions of code for comparison
            code_versions = {}
            
            # Version 1: masked query + 用户澄清得到的信息 (current version)
            code_versions["clarified"] = assistant_msg
            
            # Version 2: full query生成的代码
            original_query = current_state.get("original_instruct_prompt", "")
            if original_query:
                # Create a temporary state with full query
                full_query_state = current_state.copy()
                full_query_state["query"] = original_query
                if local_generator is not None:
                    full_query_code = local_generator.chat_complete(action_prompt, f"[Task]\n{original_query}")
                elif llm_model:
                    full_query_code = llm_output(full_query_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
                else:
                    full_query_code = dummy_llm_output(full_query_state, action_prompt)
                code_versions["oracle"] = full_query_code
            else:
                code_versions["oracle"] = None
            
            # Version 3: masked query本身生成的代码 (初始masked query，无澄清信息)
            initial_masked_query = initial_state.get("query", "")
            if initial_masked_query and initial_masked_query != current_state.get("query", ""):
                # Create a temporary state with initial masked query
                masked_query_state = initial_state.copy()
                masked_query_state["query"] = initial_masked_query
                if local_generator is not None:
                    masked_query_code = local_generator.chat_complete(action_prompt, f"[Task]\n{initial_masked_query}")
                elif llm_model:
                    masked_query_code = llm_output(masked_query_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
                else:
                    masked_query_code = dummy_llm_output(masked_query_state, action_prompt)
                code_versions["direct"] = masked_query_code
            else:
                # If no clarification happened, masked_only is same as masked_with_clarification
                code_versions["direct"] = assistant_msg

            # Version 4: ideal_disclosed — masked query + all masked_fields directly revealed
            disclosure_rule = initial_state.get("disclosure_rule", {})
            masked_fields = disclosure_rule.get("masked_fields", {}) if disclosure_rule else {}
            ideal_parts = []
            for field_items in masked_fields.values():
                for item in (field_items if isinstance(field_items, list) else [field_items]):
                    if str(item).strip():
                        ideal_parts.append(str(item).strip())
            if ideal_parts:
                ideal_ctx = "Key requirements: " + "; ".join(ideal_parts)
                ideal_query = f"{initial_masked_query}\n\nAdditional information from clarification:\n{ideal_ctx}"
                ideal_state = initial_state.copy()
                ideal_state["query"] = ideal_query
                if local_generator is not None:
                    ideal_code = local_generator.chat_complete(action_prompt, f"[Task]\n{ideal_query}")
                elif llm_model:
                    ideal_code = llm_output(ideal_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
                else:
                    ideal_code = dummy_llm_output(ideal_state, action_prompt)
                code_versions["ideal_disclosed"] = ideal_code
            else:
                code_versions["ideal_disclosed"] = code_versions["direct"]

            # Add code_versions to trajectory
            traj["code_versions"] = code_versions
            
            task_completed = check_task_completion(current_state, assistant_msg, domain)
            traj["task_completed"] = task_completed
            traj["is_terminal"] = True  # Mark as terminal state
            # Execute之后必须结束对话（无论任务是否完成）
            break
        
        # Check if user wants to stop (reject signal)
        if reaction.get("meta", {}).get("reject_signal", 0) > 0:
            traj["user_stopped"] = True
            traj["task_completed"] = False  # Explicitly mark as not completed
            traj["is_terminal"] = True  # Mark as terminal state
            break
        
        # Update state for next turn
        # is_same_turn=False because this is moving to the next dialogue turn (not within same turn)
        current_state = update_state_for_next_turn(current_state, reaction, assistant_msg, is_same_turn=False)
        
        # Update has_edge_cases_info in state for next turn
        current_state["has_edge_cases_info"] = has_edge_cases_info
    
    # If loop ended without break (reached max_turns or user stopped), check if we need to force Execute
    has_executed = any(t.get("action") == "Execute" for t in trajectories)
    
    # If we haven't executed yet (regardless of reason: max_turns or user_stopped), force Execute
    # This ensures every trajectory ends with code generation using masked query + clarification answers
    if not has_executed:
        # Force Execute with current query (masked query + clarification answers)
        action = "Execute"
        action_prompt = prompts[action]
        
        # Generate assistant message (code) using current state's query
        if local_generator is not None:
            assistant_msg = local_generator.chat_complete(action_prompt, f"[Task]\n{current_state['query']}")
        elif llm_model:
            assistant_msg = llm_output(current_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
        else:
            assistant_msg = dummy_llm_output(current_state, action_prompt)
        
        # Generate 3 versions of code for comparison
        code_versions = {}
        
        # Version 1: masked query + 用户澄清得到的信息 (current version)
        code_versions["clarified"] = assistant_msg
        
        # Version 2: full query生成的代码
        original_query = current_state.get("original_instruct_prompt", "")
        if original_query:
            # Create a temporary state with full query
            full_query_state = current_state.copy()
            full_query_state["query"] = original_query
            if local_generator is not None:
                full_query_code = local_generator.chat_complete(action_prompt, f"[Task]\n{original_query}")
            elif llm_model:
                full_query_code = llm_output(full_query_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
            else:
                full_query_code = dummy_llm_output(full_query_state, action_prompt)
            code_versions["oracle"] = full_query_code
        else:
            code_versions["oracle"] = None
        
        # Version 3: masked query本身生成的代码 (初始masked query，无澄清信息)
        initial_masked_query = initial_state.get("query", "")
        if initial_masked_query and initial_masked_query != current_state.get("query", ""):
            # Create a temporary state with initial masked query
            masked_query_state = initial_state.copy()
            masked_query_state["query"] = initial_masked_query
            if local_generator is not None:
                masked_query_code = local_generator.chat_complete(action_prompt, f"[Task]\n{initial_masked_query}")
            elif llm_model:
                masked_query_code = llm_output(masked_query_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
            else:
                masked_query_code = dummy_llm_output(masked_query_state, action_prompt)
            code_versions["direct"] = masked_query_code
        else:
            # If no clarification happened, masked_only is same as masked_with_clarification
            code_versions["direct"] = assistant_msg

        # Version 4: ideal_disclosed — masked query + all masked_fields directly revealed
        disclosure_rule = initial_state.get("disclosure_rule", {})
        masked_fields = disclosure_rule.get("masked_fields", {}) if disclosure_rule else {}
        ideal_parts = []
        for field_items in masked_fields.values():
            for item in (field_items if isinstance(field_items, list) else [field_items]):
                if str(item).strip():
                    ideal_parts.append(str(item).strip())
        if ideal_parts:
            ideal_ctx = "Key requirements: " + "; ".join(ideal_parts)
            ideal_query = f"{initial_masked_query}\n\nAdditional information from clarification:\n{ideal_ctx}"
            ideal_state = initial_state.copy()
            ideal_state["query"] = ideal_query
            if local_generator is not None:
                ideal_code = local_generator.chat_complete(action_prompt, f"[Task]\n{ideal_query}")
            elif llm_model:
                ideal_code = llm_output(ideal_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
            else:
                ideal_code = dummy_llm_output(ideal_state, action_prompt)
            code_versions["ideal_disclosed"] = ideal_code
        else:
            code_versions["ideal_disclosed"] = code_versions["direct"]

        # Check task completion
        task_completed = check_task_completion(current_state, assistant_msg, domain)
        
        # Create final Execute trajectory
        final_traj = {
            "trajectory_id": trajectory_id,
            "state": current_state.copy(),
            "action": action,
            "action_prompt": action_prompt,
            "assistant_msg": assistant_msg,
            "persona": {
                "name": persona.name,
                "domain": persona.domain,
                "expertise": persona.expertise,
                "patience": persona.patience,
            },
            "user_reaction": {},  # No user reaction for forced Execute
            "turn": len(trajectories) + 1,
            "is_mainline": True,
            "is_terminal": True,
            "has_edge_cases_info": current_state.get("has_edge_cases_info", False),
            "code_versions": code_versions,
            "task_completed": task_completed,
            "forced_execute": True,  # Mark as forced Execute (due to max_turns or user_stopped)
        }
        trajectories.append(final_traj)
    else:
        # If loop ended without forced Execute, mark last trajectory as terminal
        if trajectories and not trajectories[-1].get("is_terminal", False):
            trajectories[-1]["is_terminal"] = True
            # If task wasn't completed and user didn't stop, this is a timeout scenario
            if not trajectories[-1].get("task_completed", False) and not trajectories[-1].get("user_stopped", False):
                trajectories[-1]["task_completed"] = False  # Explicitly mark as not completed
    
    return trajectories


def generate_trajectories(states: List[Dict], domain: str, 
                         llm_model: Optional[str] = None,
                         persona_idx: int = 0,
                         max_turns: int = 5,
                         out_file=None,
                         all_personas: bool = False,
                         force_first_action: Optional[str] = None,
                         sampling_strategy: str = "heuristic",
                         n_samples_per_state_persona: int = 1,
                         local_generator: Optional[LocalHFChatGenerator] = None,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         progress_every_states: int = 10,
                         resume_done: Optional[set] = None) -> List[Dict]:
    """
    Generate multi-turn conversation trajectories.
    
    For each state, generates a full conversation until task completion or max_turns.
    Each turn is a separate trajectory entry with its own state and action.
    
    Args:
        states: List of initial states
        domain: "coding" or "planning"
        llm_model: LLM model name (None for dummy)
        persona_idx: Index of persona to use (ignored if all_personas=True)
        max_turns: Maximum number of dialogue turns per conversation
        out_file: Optional file object for streaming write
        all_personas: If True, generate trajectories for all personas for each state
        force_first_action: Force first action ("Execute"/"Clarify"/None). Used for sampling strategy.
        sampling_strategy: "heuristic" (deterministic) or "free" (random with temperature)
        n_samples_per_state_persona: Number of samples to generate per (state, persona) combination
        
    Returns:
        List of trajectory dicts, one per turn
    """
    import random
    trajectories = []
    persona_indices = range(len(PERSONAS)) if all_personas else [persona_idx]
    t0 = time.time()
    total_states = len(states)
    
    for si, st in enumerate(states, start=1):
        for pid in persona_indices:
            # Determine first action based on sampling strategy
            first_actions = []
            
            if force_first_action:
                # Explicitly specified action
                first_actions = [force_first_action]
            elif sampling_strategy == "heuristic":
                # Heuristic strategy: generate samples with different forced actions
                # Sample 1: Force Execute (blind guess)
                # Sample 2: Force Clarify (ask questions)
                # Sample 3 & 4: Free (auto-select from persona)
                if n_samples_per_state_persona >= 1:
                    first_actions.append("Execute")  # Sample 1: Force Execute
                if n_samples_per_state_persona >= 2:
                    first_actions.append("Clarify")  # Sample 2: Force Clarify
                # Sample 3+ : Free (None = auto-select)
                for _ in range(max(0, n_samples_per_state_persona - 2)):
                    first_actions.append(None)
            elif sampling_strategy == "free":
                # Free strategy: random first action or auto-select
                for _ in range(n_samples_per_state_persona):
                    if random.random() < 0.5:
                        first_actions.append(random.choice(["Execute", "Clarify"]))
                    else:
                        first_actions.append(None)  # Auto-select
            else:
                # Default: auto-select (None)
                first_actions = [None] * n_samples_per_state_persona
            
            # Generate trajectories for each first action
            for first_action in first_actions[:n_samples_per_state_persona]:
                # Resume support: in our heuristic n_samples=2 case, turn==1 and action == forced first action.
                if resume_done is not None:
                    persona_name = PERSONAS[pid].name
                    action_key = first_action if first_action is not None else None
                    # When first_action is None (auto), we cannot know; don't skip.
                    if action_key is not None:
                        sid = st.get("id", "unknown")
                        if (sid, persona_name, action_key) in resume_done:
                            continue

                multi_turn_trajs = generate_multi_turn_conversation(
                    st, domain, llm_model, pid, max_turns,
                    force_first_action=first_action,
                    local_generator=local_generator,
                    temperature=temperature,
                    top_p=top_p,
                )
                trajectories.extend(multi_turn_trajs)

                # Stream write if out_file is provided
                if out_file is not None:
                    for traj in multi_turn_trajs:
                        out_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                    out_file.flush()
                    persona_name = PERSONAS[pid].name
                    action_str = f" (first_action={first_action})" if first_action else " (auto)"
                    print(
                        f"  ✓ Generated {len(multi_turn_trajs)} turn trajectories for state {st.get('id', 'unknown')} with {persona_name}{action_str}",
                        flush=True,
                    )

                # ── Method A: fork Execute at each intermediate Clarify turn ──────
                # For Clarify mainlines with 2+ Clarify turns, generate an "Execute now"
                # branch at each Clarify decision point after T0.  This creates pairs:
                #   Clarify-again (mainline) vs Execute-now (fork) at the SAME state,
                # giving direct per-turn proactivity signal for DPO training.
                # T0 Execute branch is already handled by heuristic sampling (n_samples=2).
                if first_action == "Clarify" and len(multi_turn_trajs) > 1:
                    mainline_id = multi_turn_trajs[0]["trajectory_id"]
                    for t_idx, turn_data in enumerate(multi_turn_trajs):
                        if t_idx == 0:
                            continue  # T0 Execute branch already covered by heuristic sampling
                        if turn_data.get("action") != "Clarify":
                            continue  # only fork at Clarify decisions
                        fork_turn_num = turn_data["turn"]   # 1-based turn index in mainline
                        fork_state = turn_data["state"].copy()
                        fork_traj_id = f"{mainline_id}_fork_t{fork_turn_num}"
                        fork_trajs = generate_multi_turn_conversation(
                            fork_state, domain, llm_model, pid, max_turns=1,
                            force_first_action="Execute",
                            local_generator=local_generator,
                            temperature=temperature,
                            top_p=top_p,
                            trajectory_id=fork_traj_id,
                        )
                        for ft in fork_trajs:
                            ft["is_fork"] = True
                            ft["fork_at_turn"] = fork_turn_num
                            ft["parent_trajectory_id"] = mainline_id
                            ft["turn"] = fork_turn_num  # align with mainline turn number
                        trajectories.extend(fork_trajs)
                        if out_file is not None:
                            for ft in fork_trajs:
                                out_file.write(json.dumps(ft, ensure_ascii=False) + "\n")
                            out_file.flush()
                            print(
                                f"  ✓ [Method A] Fork Execute@turn{fork_turn_num} "
                                f"for {PERSONAS[pid].name} (parent={mainline_id[:24]}...)",
                                flush=True,
                            )

        # Coarse progress update per state (works well in redirected logs; avoids tqdm carriage-return issues)
        if progress_every_states > 0 and (si % progress_every_states == 0 or si == total_states):
            elapsed = time.time() - t0
            rate = si / elapsed if elapsed > 0 else 0.0
            remaining = (total_states - si) / rate if rate > 0 else float("inf")
            eta_min = remaining / 60.0 if remaining != float("inf") else float("inf")
            print(
                f"[Progress] states={si}/{total_states} ({si/total_states*100:.2f}%) "
                f"elapsed={elapsed/60:.1f}m eta={eta_min:.1f}m",
                flush=True,
            )
    
    return trajectories




def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate trajectories (state + action + assistant output + simulator reaction)"
    )
    parser.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic",
                       help="synthetic: quick test without dataset; dataset: load from JSONL")
    parser.add_argument("--domain", choices=["coding", "planning"], default="coding")
    parser.add_argument("--n_states", type=int, default=50, help="Number of states to process")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to states JSONL (required for dataset mode)")
    parser.add_argument("--out", type=str, default="logs/trajectories.jsonl",
                       help="Output path relative to data/ directory")
    parser.add_argument("--llm_model", type=str, default="",
                       help="OpenAI model name (e.g., gpt-4o-mini). If empty, uses dummy output.")
    parser.add_argument("--local_model", type=str, default="",
                       help="HF model name for local generation (if set, overrides --llm_model and uses local Transformers).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation (default: 42)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling for generation")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Max new tokens for generation")
    parser.add_argument("--progress_every", type=int, default=10, help="Print a coarse progress line every N states (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume by appending to an existing output file path (no timestamp added) and skipping completed (state, persona, action) combos.")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum number of dialogue turns per conversation (default: 5)")
    parser.add_argument("--persona_idx", type=int, default=0,
                       help="Index of persona to use (0=Novice-Learner, 1=Busy-Developer, 2=Experienced-Engineer, default: 0). Ignored if --all_personas is set.")
    parser.add_argument("--all_personas", action="store_true",
                       help="Generate trajectories for all personas for each state (default: False)")
    parser.add_argument("--force_first_action", type=str, choices=["Execute", "Clarify"], default=None,
                       help="Force the first action (Execute/Clarify) instead of auto-selecting. If not specified, auto-selects based on persona and state (default)")
    parser.add_argument("--sampling_strategy", type=str, choices=["heuristic", "free"], default="heuristic",
                       help="Sampling strategy: 'heuristic' (deterministic with forced actions) or 'free' (random with temperature, default: heuristic)")
    parser.add_argument("--n_samples", type=int, default=1,
                       help="Number of samples to generate per (state, persona) combination (default: 1). For heuristic strategy, generates: 1=Execute, 2=Execute+Clarify, 3+=Execute+Clarify+Free...")
    args = parser.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load states
    if args.mode == "dataset":
        if not args.dataset_path:
            raise SystemExit("--dataset_path is required in dataset mode")
        states = load_states_from_dataset(Path(args.dataset_path), domain=args.domain, limit=args.n_states)
    else:
        states = synth_states(args.domain, args.n_states)
    
    # Open output file for streaming write
    # Add timestamp to filename if not already present (unless resuming)
    from datetime import datetime
    out_basename = args.out
    if (not args.resume) and (not any(token in out_basename for token in ["%Y", "%m", "%d", "%H", "%M", "%S", "timestamp"])):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "." in out_basename:
            name, ext = out_basename.rsplit(".", 1)
            out_basename = f"{name}_{timestamp}.{ext}"
        else:
            out_basename = f"{out_basename}_{timestamp}"
    out_path = out_dir / out_basename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine persona count
    n_personas = len(PERSONAS) if args.all_personas else 1
    total_combinations = len(states) * n_personas * args.n_samples
    
    print(f"📝 Starting trajectory generation (streaming to {out_path})...")
    print(f"   - States: {len(states)}")
    print(f"   - Personas: {'All (' + str(len(PERSONAS)) + ')' if args.all_personas else f'Index {args.persona_idx} ({PERSONAS[args.persona_idx].name})'}")
    print(f"   - Samples per (state, persona): {args.n_samples}")
    print(f"   - Sampling strategy: {args.sampling_strategy}")
    if args.force_first_action:
        print(f"   - Force first action: {args.force_first_action}")
    print(f"   - Total combinations: {total_combinations} (states × personas × samples)")
    print(f"   - Expected trajectories: variable (depends on conversation length)")
    print()
    
    # Resume: load completed combos from existing output file
    resume_done = None
    if args.resume:
        if not out_path.exists():
            raise SystemExit(f"--resume requires an existing output file path, but not found: {out_path}")
        resume_done = set()
        with out_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                st = row.get("state") or {}
                sid = st.get("id")
                persona = (row.get("persona") or {}).get("name")
                action = row.get("action")
                turn = row.get("turn")
                if sid and persona and action and turn == 1:
                    resume_done.add((sid, persona, action))
        print(f"🔁 Resume enabled: found {len(resume_done)} completed (state, persona, action) combos in {out_path}", flush=True)

    # Open file for streaming write
    local_gen = None
    if args.local_model:
        print(f"🤖 Using local HF model for generation: {args.local_model}", flush=True)
        local_gen = LocalHFChatGenerator(
            args.local_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        llm_model = None
    else:
        llm_model = args.llm_model if args.llm_model else None

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        trajectories = generate_trajectories(
            states, 
            args.domain, 
            llm_model,
            persona_idx=args.persona_idx,
            max_turns=args.max_turns,
            out_file=f,  # Pass file object for streaming write
            all_personas=args.all_personas,
            force_first_action=args.force_first_action,
            sampling_strategy=args.sampling_strategy,
            n_samples_per_state_persona=args.n_samples,
            local_generator=local_gen,
            temperature=args.temperature,
            top_p=args.top_p,
            progress_every_states=args.progress_every,
            resume_done=resume_done,
        )

    # Print summary
    n_completed = sum(1 for t in trajectories if t.get("task_completed", False))
    avg_turns = sum(t.get("turn", 1) for t in trajectories) / len(trajectories) if trajectories else 0
    
    # Count unique conversations (by state_id + persona)
    from collections import defaultdict
    conversations = defaultdict(set)
    for traj in trajectories:
        state_id = traj.get("state", {}).get("id", "unknown")
        persona_name = traj.get("persona", {}).get("name", "unknown")
        conversations[(state_id, persona_name)].add(traj.get("turn", 1))
    n_conversations = len(conversations)
    
    # Count actions in first turn
    first_actions = defaultdict(int)
    for traj in trajectories:
        if traj.get("turn", 0) == 1:
            first_actions[traj.get("action", "unknown")] += 1
    
    print(f"\n{'='*60}")
    print(f"Generation Summary")
    print(f"{'='*60}")
    print(f"Wrote {len(trajectories)} trajectory turns to {out_path}")
    print(f"  - Total conversations: {n_conversations}")
    print(f"  - Average turns per conversation: {avg_turns:.2f}")
    print(f"  - Completed conversations: {n_completed}/{n_conversations}")
    print(f"  - First action distribution:")
    for action, count in first_actions.items():
        print(f"      {action}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

