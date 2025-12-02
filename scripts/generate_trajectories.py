"""
Step 1: Generate trajectories (state + action + assistant output + simulator reaction)

Uses mainline+branches strategy: 1 mainline trajectory + 2 branches per state (cost-efficient).

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


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_action_prompts(domain: str) -> Dict[str, str]:
    """Load behavior templates from prompts/ directory."""
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        return {
            "LOW": load_prompt(base / "coding_low.txt"),
            "MID": load_prompt(base / "coding_mid.txt"),
            "HIGH": load_prompt(base / "coding_high.txt"),
        }
    elif domain == "planning":
        return {
            "LOW": load_prompt(base / "planning_low.txt"),
            "MID": load_prompt(base / "planning_mid.txt"),
            "HIGH": load_prompt(base / "planning_high.txt"),
        }
    else:
        raise ValueError(f"Unknown domain: {domain}")


def synth_states(domain: str, n: int) -> List[Dict]:
    """Generate synthetic states for quick testing."""
    samples = []
    for i in range(n):
        samples.append(
            {
                "id": f"{domain}-{i}",
                "domain": domain,
                "query": "帮我写个 Python 爬虫" if domain == "coding" else "帮我规划今天的待办",
                "dialogue_turn": 1,
                "query_clarity": 0.6,
                "task_uncertainty": 0.6,
                "time_pressure": "low",  # default to low
                "prev_reject": 0,
            }
        )
    return samples


def load_states_from_dataset(dataset_path: Path, domain: str, limit: Optional[int] = None) -> List[Dict]:
    """Load states from JSONL dataset file."""
    states: List[Dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            states.append(
                {
                    "id": row.get("id", f"ds-{i}"),
                    "domain": domain,
                    "query": row["query"],
                    "dialogue_turn": int(row.get("dialogue_turn", 1)),
                    "query_clarity": float(row.get("query_clarity", 0.5)),
                    "task_uncertainty": float(row.get("task_uncertainty", 0.5)),
                    "time_pressure": row.get("time_pressure", "low"),  # default to low
                    "prev_reject": int(row.get("prev_reject", 0)),
                    "convcodeworld_tests": row.get("convcodeworld_tests"),  # preserve if present
                }
            )
    return states


def dummy_llm_output(state: Dict, action_prompt: str) -> str:
    """Placeholder LLM output for testing without API calls."""
    lower = action_prompt.lower()
    if "ask up to two clarifying" in lower:
        return "请问目标网站与输出格式？随后给出实现步骤与代码。"
    if "ask exactly one" in lower or "one key clarifying" in lower:
        return "请问需要爬取哪个网站的数据？然后我会给出代码。"
    return "这是一个最小可运行的示例代码/计划。"


def llm_output(state: Dict, action_prompt: str, model: str) -> str:
    """Generate assistant output using OpenAI API."""
    from llm.provider import chat_complete
    system = action_prompt
    user = f"[Task]\n{state['query']}"
    return chat_complete(system, user, model=model, max_tokens=400)


def select_mainline_action_from_persona(persona, state: Optional[Dict] = None) -> str:
    """
    Select mainline action based on persona characteristics and state.
    
    Logic:
    - Low patience + high time_pressure (from state) → LOW (direct, no questions)
    - High patience → HIGH (can ask more)
    - Otherwise → MID (balanced)
    """
    # Get time_pressure from state if available, otherwise default to "low"
    time_pressure = state.get("time_pressure", "low") if state else "low"
    
    if persona.patience < 0.4 and time_pressure == "high":
        return "LOW"
    elif persona.patience > 0.7:
        return "HIGH"
    else:
        return "MID"


def generate_branch_at_state(state: Dict, action: str, action_prompt: str, domain: str, 
                             llm_model: Optional[str] = None, persona_idx: int = 0) -> Dict:
    """Generate a single branch (trajectory) at a given state with specified action."""
    # Generate assistant message
    assistant_msg = llm_output(state, action_prompt, model=llm_model) if llm_model else dummy_llm_output(state, action_prompt)
    
    # Get simulator reaction
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    reaction = react(state["query"], assistant_msg, persona)
    
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
            "style": persona.style,
        },
        "user_reaction": reaction,
    }


def generate_trajectories(states: List[Dict], domain: str, 
                         llm_model: Optional[str] = None,
                         mainline_action: Optional[str] = None,
                         persona_idx: int = 0) -> List[Dict]:
    """
    Generate trajectories using mainline+branches strategy (cost-efficient).
    
    Phase 1: Generate 1 mainline trajectory
      - If mainline_action provided: use it
      - Otherwise: auto-select based on persona (patience) and state (time_pressure)
    Phase 2: At each decision point, generate 2 branches (the other 2 actions)
    
    For each state: 1 mainline + 2 branches = 3 trajectories total.
    Mainline serves as reference; branches provide contrastive actions for DPO learning.
    """
    prompts = build_action_prompts(domain)
    trajectories = []
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    
    # Determine mainline action (can vary per state if time_pressure differs)
    mainline_selected_by = "manual" if mainline_action else None
    
    for st in states:
        # Determine mainline action for this state if not manually specified
        if mainline_action is None:
            mainline_action = select_mainline_action_from_persona(persona, st)
            if mainline_selected_by is None:
                mainline_selected_by = "persona+state"
        # Phase 1: Generate mainline trajectory
        mainline_prompt = prompts[mainline_action]
        mainline_traj = generate_branch_at_state(st, mainline_action, mainline_prompt, domain, llm_model, persona_idx)
        
        # Mark as mainline
        mainline_traj["is_mainline"] = True
        mainline_traj["decision_point"] = 0  # initial decision point
        mainline_traj["mainline_action_selected_by"] = mainline_selected_by
        trajectories.append(mainline_traj)
        
        # Phase 2: Generate branches at this decision point
        # Generate all 3 actions (LOW/MID/HIGH) as branches
        for action, tpl in prompts.items():
            # Skip mainline action (already generated above)
            if action == state_mainline_action:
                continue
            
            branch_traj = generate_branch_at_state(st, action, tpl, domain, llm_model, persona_idx)
            branch_traj["is_mainline"] = False
            branch_traj["decision_point"] = 0  # same decision point as mainline
            branch_traj["mainline_action"] = state_mainline_action  # reference to mainline
            trajectories.append(branch_traj)
    
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
    parser.add_argument("--mainline_action", choices=["LOW", "MID", "HIGH"], default=None,
                       help="Action to use for mainline trajectory. If not provided, auto-selects based on persona (patience) and state (time_pressure).")
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
    trajectories = generate_trajectories(
        states, 
        args.domain, 
        args.llm_model if args.llm_model else None,
        mainline_action=args.mainline_action if args.mainline_action else None
    )

    # Write to data/logs/
    out_path = out_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    # Print summary
    n_mainline = sum(1 for t in trajectories if t.get("is_mainline", False))
    n_branches = len(trajectories) - n_mainline
    mainline_actions_used = set(t.get("action") for t in trajectories if t.get("is_mainline", False))
    print(f"Wrote {len(trajectories)} trajectories to {out_path}")
    if args.mainline_action:
        print(f"  - Strategy: mainline+branches (manual: {args.mainline_action} as mainline)")
    else:
        print(f"  - Strategy: mainline+branches (auto-selected from persona: {mainline_actions_used})")
    print(f"  - {len(states)} states × (1 mainline + 2 branches) = {len(trajectories)} trajectories")
    print(f"  - Mainline: {n_mainline}, Branches: {n_branches}")


if __name__ == "__main__":
    main()

