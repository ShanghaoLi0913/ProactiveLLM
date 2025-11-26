"""
Step 2: Compute rewards from trajectories and generate DPO preference pairs

Input: Trajectories JSONL from data/logs/
Output: Preference pairs JSONL to data/dpo/
Each pair contains: {state, chosen_action, rejected_action, chosen_text, rejected_text, rewards, task_scores, interrupt_costs}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Ensure project root is on sys.path for package imports when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward.compute import compute_task_score, compute_interrupt_cost, total_reward


def load_trajectories(trajectories_path: Path) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with trajectories_path.open("r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories


def compute_rewards_for_trajectories(trajectories: List[Dict]) -> Dict[str, Dict]:
    """
    Compute reward for each (state_id, action) pair.
    
    Returns: {(state_id, action): {reward, task_score, interrupt_cost, ...}}
    """
    scored = {}
    for traj in trajectories:
        state = traj["state"]
        action = traj["action"]
        assistant_msg = traj["assistant_msg"]
        user_reaction = traj["user_reaction"]
        meta = user_reaction["meta"]
        
        # Extract features for interrupt cost
        n_questions = assistant_msg.count("?")
        length_tokens = len(assistant_msg.split())
        off_topic = 0  # placeholder
        
        # Compute rewards
        task_score = compute_task_score(state, state["domain"], assistant_output=assistant_msg)
        interrupt_cost = compute_interrupt_cost(meta, n_questions, length_tokens, off_topic)
        reward = total_reward(task_score, interrupt_cost)
        
        key = (state.get("id", "unknown"), action)
        scored[key] = {
            "state": state,
            "action": action,
            "assistant_msg": assistant_msg,
            "meta": meta,
            "n_questions": n_questions,
            "length_tokens": length_tokens,
            "task_score": task_score,
            "interrupt_cost": interrupt_cost,
            "reward": reward,
        }
    return scored


def build_preference_pairs(scored: Dict, trajectories: List[Dict]) -> List[Dict]:
    """
    Group trajectories by state_id, rank actions by reward, create (chosen, rejected) pairs.
    
    For each state, picks the highest-reward action as chosen and lowest as rejected.
    """
    # Group by state_id
    by_state = defaultdict(list)
    for traj in trajectories:
        state_id = traj["state"].get("id", "unknown")
        action = traj["action"]
        key = (state_id, action)
        if key in scored:
            by_state[state_id].append((action, scored[key]))
    
    pairs = []
    for state_id, actions_with_scores in by_state.items():
        if len(actions_with_scores) < 2:
            continue  # need at least 2 actions to compare
        
        # Rank by reward
        ranked = sorted(actions_with_scores, key=lambda x: x[1]["reward"], reverse=True)
        (a_plus, plus_obj), (a_minus, minus_obj) = ranked[0], ranked[-1]
        
        # Get original trajectory for full context
        state = plus_obj["state"]
        pairs.append({
            "state": state,
            "chosen_action": a_plus,
            "rejected_action": a_minus,
            "chosen_text": plus_obj["assistant_msg"],
            "rejected_text": minus_obj["assistant_msg"],
            "rewards": {a: obj["reward"] for a, obj in actions_with_scores},
            "task_scores": {a: obj["task_score"] for a, obj in actions_with_scores},
            "interrupt_costs": {a: obj["interrupt_cost"] for a, obj in actions_with_scores},
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Compute rewards from trajectories and generate DPO preference pairs"
    )
    parser.add_argument("--trajectories", type=str, required=True,
                       help="Path to trajectories JSONL file (relative to data/ or absolute)")
    parser.add_argument("--out", type=str, default="dpo/prefs.jsonl",
                       help="Output path for preference pairs (relative to data/)")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    trajectories_path = data_dir / args.trajectories if not Path(args.trajectories).is_absolute() else Path(args.trajectories)
    
    if not trajectories_path.exists():
        raise SystemExit(f"Trajectories file not found: {trajectories_path}")

    # Load trajectories
    trajectories = load_trajectories(trajectories_path)
    print(f"Loaded {len(trajectories)} trajectories")

    # Compute rewards
    scored = compute_rewards_for_trajectories(trajectories)
    print(f"Computed rewards for {len(scored)} (state, action) pairs")

    # Build preference pairs
    pairs = build_preference_pairs(scored, trajectories)
    print(f"Generated {len(pairs)} preference pairs")

    # Write to data/dpo/
    out_path = data_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} preference pairs to {out_path}")


if __name__ == "__main__":
    main()

