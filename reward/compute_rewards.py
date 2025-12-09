"""
Step 2: Compute rewards & preference pairs for DPO.

Input:
  - Trajectories JSONL from data/logs/, each line:
      {
        "state": {...},
        "action": "LOW" | "MID" | "HIGH",
        "assistant_msg": "...",
        "user_reaction": {
            "meta": {
                "answered_clarification": int,
                "reject_signal": int,
                "silence": int,
                "off_topic_flag": int,
                "satisfaction": float,
            },
            ...
        },
        "is_mainline": bool,
        "decision_point": int
      }

Output:
  - Preference pairs JSONL to data/dpo/, each line:
      {
        "state": {...},                  # original state
        "chosen_action": "LOW/MID/HIGH",
        "rejected_action": "LOW/MID/HIGH",
        "chosen_assistant_msg": "...",    # 完整回复（用于训练）
        "rejected_assistant_msg": "...",  # 完整回复（用于训练）
        "chosen_reward": float,
        "rejected_reward": float,
        "chosen_task_score": float,
        "rejected_task_score": float,
        "chosen_interrupt_cost": float,
        "rejected_interrupt_cost": float
      }

Design:
  - We **do not** assume mainline action is best.
  - For each state + decision_point, we:
      1) Compute task_score using reward.compute_task_score (runs tests for coding)
      2) Compute interrupt_cost using reward.compute_interrupt_cost
      3) Combine into total_reward = w_task * task_score - w_int * interrupt_cost
  - Then we select preference pairs based on total_reward ranking.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a script: add project root to sys.path and import reward.compute
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward.compute import compute_task_score, compute_interrupt_cost, compute_interrupt_cost_v2, compute_clarification_bonus, total_reward


@dataclass
class RewardConfig:
    """Weights for reward aggregation.
    
    优化：使用激进设置以最大化奖励差异，解决MID塌陷问题。
    - w_task: 任务成功权重（保持高优先级）
    - w_interrupt: 中断成本权重（提高以放大有效/无效澄清的差异）
    """

    # Strongly prioritize task success (pass tests)
    w_task: float = 1.0
    # 提高w_interrupt以放大有效澄清（奖励）和无效澄清（惩罚）的差异
    # 这有助于解决MID塌陷问题：让有效澄清的MID/HIGH获得更高reward
    w_interrupt: float = 0.15  # 从0.1提高到0.15，放大interrupt_cost的影响


def load_trajectories(path: Path) -> List[Dict]:
    """Load trajectories JSONL."""
    trajs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajs.append(json.loads(line))
    return trajs


def group_by_state(trajs: List[Dict]) -> Dict[Tuple[str, int], List[Dict]]:
    """
    Group trajectories by (state_id, decision_point).

    Assumes each trajectory has:
      - traj["state"]["id"]
      - traj.get("decision_point", 0)
    """
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for t in trajs:
        st = t.get("state", {})
        sid = st.get("id")
        if sid is None:
            # Fallback: hashable repr
            sid = json.dumps(st, sort_keys=True)
        dp = int(t.get("decision_point", 0))
        key = (sid, dp)
        groups.setdefault(key, []).append(t)
    return groups


def compute_rewards_for_group(
    trajs: List[Dict],
    cfg: RewardConfig,
) -> List[Dict]:
    """
    Compute task_score / interrupt_cost / total_reward for each trajectory in a group.
    
    Supports both single-interaction and multi-interaction trajectories:
    - Single-interaction: uses assistant_msg and user_reaction directly
    - Multi-interaction: accumulates interrupt_cost across all interactions,
      and computes task_score based on the final interaction with code.
    """
    scored: List[Dict] = []
    for t in trajs:
        state = t["state"]
        domain = state.get("domain", "coding")
        
        # Check if this is a multi-interaction trajectory
        if "interactions" in t and len(t["interactions"]) > 0:
            # Multi-interaction mode: accumulate costs across all interactions
            total_interrupt_cost = 0.0
            final_assistant_msg = ""
            
            # Accumulate interrupt_cost from all interactions
            for interaction in t["interactions"]:
                assistant_msg = interaction.get("assistant_msg", "")
                user_reaction = interaction.get("user_reaction", {})
                meta = user_reaction.get("meta", {})
                n_questions = assistant_msg.count("?")
                
                # Compute interrupt_cost for this interaction
                interaction_cost = compute_interrupt_cost_v2(
                    meta,
                    n_questions=n_questions,
                    assistant_msg=assistant_msg,
                )
                total_interrupt_cost += interaction_cost
                
                # Track the last interaction that contains code
                has_code = (
                    "```" in assistant_msg or
                    "def " in assistant_msg or
                    "class " in assistant_msg
                )
                if has_code:
                    final_assistant_msg = assistant_msg
            
            # If no code found, use the last interaction's message
            if not final_assistant_msg and t["interactions"]:
                final_assistant_msg = t["interactions"][-1].get("assistant_msg", "")
            
            # Compute task_score based on final code (if any)
            task_score = compute_task_score(state, domain, assistant_output=final_assistant_msg)
            interrupt_cost = total_interrupt_cost
            
        else:
            # Single-interaction mode (backward compatibility)
            assistant_msg = t.get("assistant_msg", "")
            
            # Task score (0/1 for coding with tests)
            task_score = compute_task_score(state, domain, assistant_output=assistant_msg)
            
            # Interrupt cost (Reward_version2: 新公式)
            # C_Interrupt = Σ_{t=1}^{T} (δb_t r_t + λb_t - γb_t a_t)
            meta = (t.get("user_reaction") or {}).get("meta", {})
            n_questions = assistant_msg.count("?")
            
            # 使用新版本的interrupt cost计算
            interrupt_cost = compute_interrupt_cost_v2(
                meta,
                n_questions=n_questions,
                assistant_msg=assistant_msg,
            )
        
        # 新公式: R = R_task - C_interrupt
        # 注意：在新公式中，有效澄清的奖励已经包含在C_Interrupt的计算中
        # （通过-γb_t a_t项减少成本，相当于奖励）
        r = cfg.w_task * task_score - cfg.w_interrupt * interrupt_cost
        total_r = float(r)

        scored.append(
            {
                **t,
                "task_score": float(task_score),
                "interrupt_cost": float(interrupt_cost),
                "total_reward": total_r,
            }
        )
    return scored


def build_prefs_from_group(scored_trajs: List[Dict]) -> List[Dict]:
    """
    Given scored trajectories for a single (state, decision_point),
    generate preference pairs (chosen, rejected) based on total_reward.

    Strategy:
      - Sort by total_reward descending
      - Take top-1 as chosen, bottom-1 as rejected (if distinct)
      - Optional: if there are 3 actions, you can also create multiple pairs:
          best > middle, best > worst, middle > worst
        For simplicity and stability we start with just (best, worst).
    """
    if len(scored_trajs) < 2:
        return []

    scored_sorted = sorted(scored_trajs, key=lambda t: t["total_reward"], reverse=True)
    best = scored_sorted[0]
    worst = scored_sorted[-1]

    # If equal reward, skip to avoid noisy / conflicting pairs
    if best["total_reward"] <= worst["total_reward"]:
        return []

    state = best["state"]
    pref = {
        "state": state,
        "chosen_action": best["action"],
        "rejected_action": worst["action"],
        "chosen_assistant_msg": best.get("assistant_msg", ""),  # 完整回复
        "rejected_assistant_msg": worst.get("assistant_msg", ""),  # 完整回复
        "chosen_reward": best["total_reward"],
        "rejected_reward": worst["total_reward"],
        "chosen_task_score": best["task_score"],
        "rejected_task_score": worst["task_score"],
        "chosen_interrupt_cost": best["interrupt_cost"],
        "rejected_interrupt_cost": worst["interrupt_cost"],
    }
    return [pref]


def compute_preferences(
    traj_path: Path,
    out_path: Path,
    cfg: RewardConfig,
) -> None:
    """High-level function: trajectories → prefs JSONL."""
    print(f"📂 Loading trajectories from: {traj_path}")
    trajs = load_trajectories(traj_path)
    print(f"📊 Loaded {len(trajs)} trajectories")

    groups = group_by_state(trajs)
    print(f"📊 Grouped into {len(groups)} (state, decision_point) groups")

    prefs: List[Dict] = []
    for (sid, dp), g in groups.items():
        scored = compute_rewards_for_group(g, cfg)
        group_prefs = build_prefs_from_group(scored)
        prefs.extend(group_prefs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(prefs)} preference pairs to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute rewards and preference pairs from trajectories (task-dominated reward)."
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help="Path to trajectories JSONL (e.g., data/logs/traj_convcodeworld_150.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output prefs JSONL path (e.g., data/dpo/prefs_150_taskdom.jsonl)",
    )
    parser.add_argument(
        "--w_task",
        type=float,
        default=1.0,
        help="Weight for task_score (default: 1.0, dominant)",
    )
    parser.add_argument(
        "--w_interrupt",
        type=float,
        default=0.1,
        help="Weight for interrupt cost penalty (default: 0.1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.trajectories)
    out_path = Path(args.out)
    cfg = RewardConfig(w_task=args.w_task, w_interrupt=args.w_interrupt)

    print("⚙️  Reward config:")
    print(f"  - w_task = {cfg.w_task}")
    print(f"  - w_interrupt = {cfg.w_interrupt}")

    compute_preferences(traj_path, out_path, cfg)


if __name__ == "__main__":
    main()

