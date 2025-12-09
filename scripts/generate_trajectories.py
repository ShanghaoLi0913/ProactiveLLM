"""
Step 1: Generate trajectories (state + action + assistant output + simulator reaction)

Uses mainline+branches strategy: 1 mainline trajectory + 2 branches per state = 3 trajectories total (cost-efficient).

Input: States (from synthetic or dataset)
Output: Trajectories JSONL to data/logs/
Each trajectory contains: {state, action, action_prompt, assistant_msg, persona, user_reaction, is_mainline, decision_point}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Ensure project root is on sys.path for package imports when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import PERSONAS, react
from utils.compute_task_uncertainty import compute_task_uncertainty_from_state


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
        
        result.append(
            {
                "id": state_id,
                "domain": domain,
                "query": query,
                "dialogue_turn": int(row.get("dialogue_turn", 0)),  # Start from 0
                "prev_reject": int(row.get("prev_reject", 0)),
                "task_uncertainty": task_uncertainty,
                "convcodeworld_tests": row.get("convcodeworld_tests"),  # preserve if present
            }
        )
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


def llm_output(state: Dict, action_prompt: str, model: str, conversation_history: Optional[List[Dict]] = None) -> str:
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
    
    return chat_complete(system, user, model=model, max_tokens=400)


def select_mainline_action_from_persona(persona, state: Optional[Dict] = None) -> str:
    """
    Select mainline action based on persona characteristics and state.
    
    Sequential Decision Process: actions are Clarify or Execute.
    
    Logic:
    - Low patience → Execute (direct, no questions)
    - High patience + low task_uncertainty → Clarify (can ask questions)
    - Previous reject → Execute (don't ask more questions)
    - High dialogue_turn → Execute (already asked many questions)
    - Otherwise → Execute (default to execution)
    """
    task_uncertainty = state.get("task_uncertainty", 0.5) if state else 0.5
    dialogue_turn = state.get("dialogue_turn", 0) if state else 0
    prev_reject = state.get("prev_reject", 0) if state else 0
    
    # If user rejected in previous turn, execute (don't ask more questions)
    if prev_reject > 0:
        return "Execute"
    
    # If already asked many questions (high dialogue_turn), execute
    if dialogue_turn >= 2:
        return "Execute"
    
    # Low patience → Execute (direct, no questions)
    if persona.patience == "low":
        return "Execute"
    # High patience + low task_uncertainty → Clarify (can ask questions)
    elif persona.patience == "high" and task_uncertainty < 0.5:
        return "Clarify"
    # Otherwise → Execute (default to execution)
    else:
        return "Execute"


def check_task_completion(state: Dict, assistant_msg: str, domain: str) -> bool:
    """Check if task is completed based on assistant output and tests."""
    if domain == "coding":
        # Check if code is present
        has_code = (
            "```" in assistant_msg or
            "def " in assistant_msg or
            "class " in assistant_msg
        )
        if not has_code:
            return False
        
        # If tests are available, try to execute them
        tests = state.get("convcodeworld_tests")
        if tests:
            try:
                from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail
                code = extract_code_from_text(assistant_msg)
                if code:
                    score = score_code_passfail(code, tests, timeout=30)
                    return score > 0.5  # Task completed if tests pass
            except Exception:
                pass
        
        # If no tests or execution failed, assume task completed if code is present
        return has_code
    
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
                                     action_selection_fn=None) -> List[Dict]:
    """
    Generate a multi-turn conversation until task completion or max turns.
    
    Args:
        initial_state: Starting state
        domain: "coding" or "planning"
        llm_model: LLM model name (None for dummy)
        persona_idx: Index of persona to use
        max_turns: Maximum number of dialogue turns
        action_selection_fn: Function(state) -> action (LOW/MID/HIGH). If None, uses persona-based selection.
    
    Returns:
        List of trajectory dicts, one per turn
    """
    prompts = build_action_prompts(domain)
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    trajectories = []
    current_state = initial_state.copy()
    
    for turn in range(max_turns):
        # Select action for this turn
        if action_selection_fn:
            action = action_selection_fn(current_state)
        else:
            action = select_mainline_action_from_persona(persona, current_state)
        
        action_prompt = prompts[action]
        
        # Generate assistant message
        if llm_model:
            assistant_msg = llm_output(current_state, action_prompt, llm_model)
        else:
            assistant_msg = dummy_llm_output(current_state, action_prompt)
        
        # Get simulator reaction
        if llm_model is None:
            raise ValueError("llm_model is required for user response generation. Please provide --llm_model argument.")
        reaction = react(current_state["query"], assistant_msg, persona, llm_model=llm_model)
        
        # Create trajectory for this turn
        traj = {
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
        }
        trajectories.append(traj)
        
        # Check if task is completed
        if check_task_completion(current_state, assistant_msg, domain):
            traj["task_completed"] = True
            break
        
        # Check if user wants to stop (reject signal)
        if reaction.get("meta", {}).get("reject_signal", 0) > 0:
            traj["user_stopped"] = True
            break
        
        # Update state for next turn
        # is_same_turn=False because this is moving to the next dialogue turn (not within same turn)
        current_state = update_state_for_next_turn(current_state, reaction, assistant_msg, is_same_turn=False)
    
    return trajectories


def generate_branch_at_state(state: Dict, action: str, action_prompt: str, domain: str, 
                             llm_model: Optional[str] = None, persona_idx: int = 0,
                             use_interactions: bool = True, max_interactions: int = 5) -> Dict:
    """Generate a single branch (trajectory) at a given state with specified action.
    
    Sequential Decision Process: action is either Clarify or Execute.
    
    - Clarify: Ask 1-2 clarifying questions
    - Execute: Provide code/solution directly
    
    If use_interactions=True, supports multi-interaction turns within the same action.
    """
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    
    if not use_interactions:
        # Original single-interaction mode (backward compatibility)
        assistant_msg = llm_output(state, action_prompt, model=llm_model) if llm_model else dummy_llm_output(state, action_prompt)
        # For synthetic mode (llm_model=None), use dummy user reactions
        reaction = react(state["query"], assistant_msg, persona, llm_model=llm_model)
        return {
            "state": state,
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
        }
    
    # Sequential decision mode: Clarify or Execute
    interaction_prompts = build_interaction_prompts(domain)
    current_state = state.copy()
    interactions = []
    all_assistant_msgs = []
    
    # Execute interactions based on action
    interaction_idx = 0
    questions_asked = 0
    
    if action == "Clarify":
        # Clarify: Ask 1-2 questions, then provide code
        max_questions = 2
    else:  # Execute
        # Execute: Provide code directly
        max_questions = 0
    
    while interaction_idx < max_interactions:
        interaction_idx += 1
        
        # Determine interaction type based on action
        if action == "Clarify":
            if questions_asked < max_questions:
                # Ask a clarifying question
                interaction_type = "clarify"
                prompt = interaction_prompts["Clarify"]["clarify"]
            else:
                # After asking questions, provide code
                interaction_type = "execute"
                prompt = interaction_prompts["Execute"]["execute"]
        else:  # Execute
            # Execute: Provide code directly
            interaction_type = "execute"
            prompt = interaction_prompts["Execute"]["execute"]
        
        # Generate assistant message
        assistant_msg = llm_output(current_state, prompt, model=llm_model) if llm_model else dummy_llm_output(current_state, prompt)
        all_assistant_msgs.append(assistant_msg)
        
        # Get user reaction
        # For synthetic mode (llm_model=None), use dummy user reactions
        # For dataset mode (llm_model provided), use LLM-generated reactions
        # Calculate total_questions_asked: if current interaction is a question, include it in the count
        total_questions_for_react = questions_asked + (1 if interaction_type == "clarify" else 0)
        user_reaction = react(current_state["query"], assistant_msg, persona, llm_model=llm_model, total_questions_asked=total_questions_for_react)
        interactions.append({
            "interaction_num": interaction_idx,
            "type": interaction_type,
            "assistant_msg": assistant_msg,
            "user_reaction": user_reaction,
            "state_at_this_point": current_state.copy(),
        })
        
        # Check if task is completed (if code is provided and passes tests)
        if check_task_completion(current_state, assistant_msg, domain):
            if interaction_type == "clarify":
                questions_asked += 1
            break
        
        # Check if user wants to stop (reject signal)
        if user_reaction.get("meta", {}).get("reject_signal", 0) > 0:
            # User rejected, provide code in next interaction (if not already providing code)
            if interaction_type == "clarify":
                # Skip remaining questions, go to execute
                interaction_idx += 1
                execute_prompt = interaction_prompts["Execute"]["execute"]
                execute_assistant_msg = llm_output(current_state, execute_prompt, model=llm_model) if llm_model else dummy_llm_output(current_state, execute_prompt)
                all_assistant_msgs.append(execute_assistant_msg)
                if llm_model is None:
                    raise ValueError("llm_model is required for user response generation. Please provide --llm_model argument.")
                execute_user_reaction = react(current_state["query"], execute_assistant_msg, persona, llm_model=llm_model, total_questions_asked=questions_asked)
                interactions.append({
                    "interaction_num": interaction_idx,
                    "type": "execute",
                    "assistant_msg": execute_assistant_msg,
                    "user_reaction": execute_user_reaction,
                    "state_at_this_point": current_state.copy(),
                })
                if check_task_completion(current_state, execute_assistant_msg, domain):
                    break
            if interaction_type == "clarify":
                questions_asked += 1
            break
        
        # Count questions asked
        if interaction_type == "clarify":
            questions_asked += 1
        
        # Update state for next interaction
        # is_same_turn=True because we're still in the same turn (multi-interaction)
        current_state = update_state_for_next_turn(current_state, user_reaction, assistant_msg, is_same_turn=True)
        
        # If we just provided code, we're done
        if interaction_type == "execute":
            break
    
    # Combine all assistant messages for final assistant_msg field (for backward compatibility)
    final_assistant_msg = "\n\n".join(all_assistant_msgs)
    
    # Get final user reaction (last one)
    final_user_reaction = interactions[-1]["user_reaction"] if interactions else {}
    
    return {
        "state": state,  # Original state (for grouping)
        "action": action,
        "action_prompt": action_prompt,  # Keep for backward compatibility
        "assistant_msg": final_assistant_msg,  # Combined message
        "interactions": interactions,  # Detailed interaction history
        "total_interactions": len(interactions),
        "task_completed": check_task_completion(current_state, final_assistant_msg, domain),
        "persona": {
            "name": persona.name,
            "domain": persona.domain,
            "expertise": persona.expertise,
            "patience": persona.patience,
        },
        "user_reaction": final_user_reaction,  # Final reaction (for backward compatibility)
    }


def generate_trajectories(states: List[Dict], domain: str, 
                         llm_model: Optional[str] = None,
                         mainline_action: Optional[str] = None,
                         persona_idx: int = 0,
                         multi_turn: bool = False,
                         max_turns: int = 5,
                         use_interactions: bool = True,
                         max_interactions: int = 5,
                         out_file=None) -> List[Dict]:  # Optional file object for streaming write
    """
    Generate trajectories using mainline+branches strategy (cost-efficient).
    
    If multi_turn=True: Generate full multi-turn conversations until task completion.
    If multi_turn=False: Generate single-turn trajectories (original behavior).
    
    Single-turn mode:
    Phase 1: Generate 1 mainline trajectory
      - If mainline_action provided: use it
      - Otherwise: auto-select based on persona (patience) and state (task_uncertainty, dialogue_turn, prev_reject)
    Phase 2: At each decision point, generate 2 branches:
      - Branch 1: The other action (Clarify if mainline is Execute, or vice versa)
      - Branch 2: Mainline action variant (regenerate with same action but different output)
    
    For each state: 1 mainline + 2 branches = 3 trajectories total.
    Mainline serves as reference; branches provide contrastive actions/variants for DPO learning.
    
    Multi-turn mode:
    - Generate full conversation for each state until task completion or max_turns
    - Only generates mainline trajectory (no branches to reduce cost)
    - Each turn is a separate trajectory entry
    """
    if multi_turn:
        # Multi-turn mode: generate full conversations
        trajectories = []
        for st in states:
            multi_turn_trajs = generate_multi_turn_conversation(
                st, domain, llm_model, persona_idx, max_turns
            )
            trajectories.extend(multi_turn_trajs)
            
            # Stream write if out_file is provided
            if out_file is not None:
                for traj in multi_turn_trajs:
                    out_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                out_file.flush()
                print(f"  ✓ Generated {len(multi_turn_trajs)} turn trajectories for state {st.get('id', 'unknown')}", flush=True)
        return trajectories
    
    # Original single-turn mode
    prompts = build_action_prompts(domain)
    trajectories = []
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    
    # Determine mainline action (can vary per state based on persona and state)
    mainline_selected_by = "manual" if mainline_action else None
    
    for st in states:
        # Determine mainline action for this state if not manually specified
        if mainline_action is None:
            mainline_action = select_mainline_action_from_persona(persona, st)
            if mainline_selected_by is None:
                mainline_selected_by = "persona+state"
        # Phase 1: Generate mainline trajectory
        mainline_prompt = prompts[mainline_action]
        mainline_traj = generate_branch_at_state(
            st, mainline_action, mainline_prompt, domain, llm_model, persona_idx,
            use_interactions=use_interactions, max_interactions=max_interactions
        )
        
        # Mark as mainline
        mainline_traj["is_mainline"] = True
        mainline_traj["decision_point"] = 0  # initial decision point
        mainline_traj["mainline_action_selected_by"] = mainline_selected_by
        trajectories.append(mainline_traj)
        
        # Stream write if out_file is provided
        if out_file is not None:
            out_file.write(json.dumps(mainline_traj, ensure_ascii=False) + "\n")
            out_file.flush()  # Ensure immediate write
            print(f"  ✓ Generated mainline trajectory for state {st.get('id', 'unknown')} (action: {mainline_action})", flush=True)
        
        # Phase 2: Generate branches at this decision point
        # Generate 2 branches: 1) the other action, 2) mainline action variant (regenerate)
        # Store the mainline action for this state (may vary per state)
        state_mainline_action = mainline_action
        
        # Branch 1: Generate the other action (Clarify if mainline is Execute, or vice versa)
        for action, tpl in prompts.items():
            # Skip mainline action (will generate as variant in Branch 2)
            if action == state_mainline_action:
                continue
            
            branch_traj = generate_branch_at_state(
                st, action, tpl, domain, llm_model, persona_idx,
                use_interactions=use_interactions, max_interactions=max_interactions
            )
            branch_traj["is_mainline"] = False
            branch_traj["decision_point"] = 0  # same decision point as mainline
            branch_traj["mainline_action"] = state_mainline_action  # reference to mainline
            trajectories.append(branch_traj)
            
            # Stream write if out_file is provided
            if out_file is not None:
                out_file.write(json.dumps(branch_traj, ensure_ascii=False) + "\n")
                out_file.flush()  # Ensure immediate write
                print(f"  ✓ Generated branch 1 trajectory for state {st.get('id', 'unknown')} (action: {action})", flush=True)
            break  # Only generate one other action
        
        # Branch 2: Generate mainline action variant (regenerate with same action but different output)
        mainline_tpl = prompts[state_mainline_action]
        branch_variant_traj = generate_branch_at_state(
            st, state_mainline_action, mainline_tpl, domain, llm_model, persona_idx,
            use_interactions=use_interactions, max_interactions=max_interactions
        )
        branch_variant_traj["is_mainline"] = False
        branch_variant_traj["decision_point"] = 0  # same decision point as mainline
        branch_variant_traj["mainline_action"] = state_mainline_action  # reference to mainline
        branch_variant_traj["is_variant"] = True  # Mark as variant of mainline
        trajectories.append(branch_variant_traj)
        
        # Stream write if out_file is provided
        if out_file is not None:
            out_file.write(json.dumps(branch_variant_traj, ensure_ascii=False) + "\n")
            out_file.flush()  # Ensure immediate write
            print(f"  ✓ Generated branch 2 trajectory for state {st.get('id', 'unknown')} (action: {state_mainline_action}, variant)", flush=True)
        
        # Reset mainline_action for next state if it was auto-selected
        # (so each state can have different mainline action based on persona and state)
        if mainline_selected_by != "manual":
            mainline_action = None
    
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
    parser.add_argument("--mainline_action", choices=["Clarify", "Execute"], default=None,
                       help="Action to use for mainline trajectory. If not provided, auto-selects based on persona (patience) and state (task_uncertainty, dialogue_turn, prev_reject).")
    parser.add_argument("--multi_turn", action="store_true",
                        help="Enable multi-turn conversation mode. Generates full conversations until task completion.")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum number of dialogue turns in multi-turn mode (default: 5)")
    parser.add_argument("--use_interactions", action="store_true", default=True,
                        help="Enable multi-interaction turns within a single turn (default: True). "
                             "LOW: 1 interaction, MID: 2 interactions, HIGH: multiple interactions.")
    parser.add_argument("--no_interactions", action="store_true",
                        help="Disable multi-interaction turns (use single-interaction mode for backward compatibility).")
    parser.add_argument("--max_interactions", type=int, default=5,
                       help="Maximum number of interactions within a single turn (default: 5)")
    parser.add_argument("--persona_idx", type=int, default=0,
                       help="Index of persona to use (0=Impatient-Novice, 1=Neutral-Intermediate, 2=Busy-Manager, default: 0)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load states
    if args.mode == "dataset":
        if not args.dataset_path:
            raise SystemExit("--dataset_path is required in dataset mode")
        states = load_states_from_dataset(Path(args.dataset_path), domain=args.domain, limit=args.n_states)
    else:
        states = synth_states(args.domain, args.n_states)

    # Generate trajectories (mainline+branches strategy to reduce LLM calls)
    # Determine use_interactions flag
    use_interactions = args.use_interactions and not args.no_interactions
    
    # Open output file for streaming write
    out_path = out_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Starting trajectory generation (streaming to {out_path})...")
    print(f"   - States: {len(states)}")
    print(f"   - Expected trajectories: {len(states) * 3 if not args.multi_turn else 'variable'}")
    print()
    
    # Open file for streaming write
    with out_path.open("w", encoding="utf-8") as f:
        trajectories = generate_trajectories(
            states, 
            args.domain, 
            args.llm_model if args.llm_model else None,
            mainline_action=args.mainline_action if args.mainline_action else None,
            persona_idx=args.persona_idx,
            multi_turn=args.multi_turn,
            max_turns=args.max_turns,
            use_interactions=use_interactions,
            max_interactions=args.max_interactions,
            out_file=f  # Pass file object for streaming write
        )

    # Print summary
    if args.multi_turn:
        n_completed = sum(1 for t in trajectories if t.get("task_completed", False))
        avg_turns = sum(t.get("turn", 1) for t in trajectories) / len(trajectories) if trajectories else 0
        print(f"Wrote {len(trajectories)} trajectory turns to {out_path}")
        print(f"  - Mode: Multi-turn conversation")
        print(f"  - {len(states)} initial states")
        print(f"  - Average turns per conversation: {avg_turns:.2f}")
        print(f"  - Completed conversations: {n_completed}/{len(states)}")
    else:
        n_mainline = sum(1 for t in trajectories if t.get("is_mainline", False))
        n_branches = len(trajectories) - n_mainline
        mainline_actions_used = set(t.get("action") for t in trajectories if t.get("is_mainline", False))
        print(f"Wrote {len(trajectories)} trajectories to {out_path}")
        if args.mainline_action:
            print(f"  - Strategy: mainline+branches (manual: {args.mainline_action} as mainline)")
        else:
            print(f"  - Strategy: mainline+branches (auto-selected from persona: {mainline_actions_used})")
        print(f"  - {len(states)} states × (1 mainline + 2 branches) = {len(trajectories)} trajectories")
        print(f"  - Expected: {len(states)} × 3 = {len(states) * 3} trajectories")
        print(f"  - Mainline: {n_mainline}, Branches: {n_branches}")


if __name__ == "__main__":
    main()

