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
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    w_interrupt: float = 0.2  # 0.2: interrupt_cost有区分力但不过激翻转正确pair
    # When True: Execute uses code_versions["full_query"] for task_score + DPO prefs (if present).
    full_query_for_execute: bool = False


def resolve_execute_code_for_training_and_reward(turn: Dict, *, use_full_query: bool) -> str:
    """Text used for Execute in task_score and preference training.

    If use_full_query and trajectory has code_versions['full_query'], use it (full-spec code).
    Otherwise assistant_msg. Clarify always uses assistant_msg.
    """
    if not use_full_query or turn.get("action") != "Execute":
        return turn.get("assistant_msg", "") or ""
    cv = turn.get("code_versions")
    if not isinstance(cv, dict):
        return turn.get("assistant_msg", "") or ""
    fq = cv.get("oracle")
    if isinstance(fq, str) and fq.strip():
        return fq
    return turn.get("assistant_msg", "") or ""


def pref_assistant_text(turn: Dict) -> str:
    """Assistant string for prefs/DPO; prefers assistant_msg_training when set."""
    return (turn.get("assistant_msg_training") or turn.get("assistant_msg") or "").strip()


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


def group_by_trajectory(trajs: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group trajectories by trajectory_id for trajectory-level reward computation.
    
    Each trajectory_id represents a complete multi-turn conversation.
    All turns in the same trajectory will share the final task outcome.
    
    Args:
        trajs: List of trajectory dicts, each should have "trajectory_id"
    
    Returns:
        Dict mapping trajectory_id to list of turns (sorted by dialogue_turn)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    
    for t in trajs:
        traj_id = t.get("trajectory_id")
        if traj_id is None:
            # Fallback for old data without trajectory_id
            # Group by (state_id, persona) - not perfect but better than nothing
            state_id = t.get("state", {}).get("id", "unknown")
            persona_name = t.get("persona", {}).get("name", "unknown")
            traj_id = f"{state_id}_{persona_name}"
        
        groups[traj_id].append(t)
    
    # Sort each trajectory by dialogue_turn
    for traj_id in groups:
        groups[traj_id] = sorted(groups[traj_id], key=lambda x: x.get("state", {}).get("dialogue_turn", 0))
    
    return groups


def group_by_state(trajs: List[Dict]) -> Dict[Tuple[str, int], List[Dict]]:
    """
    Group trajectories by (state_id, dialogue_turn) for step-level processing.
    
    DEPRECATED: Use group_by_trajectory for trajectory-level reward.
    This function is kept for backward compatibility.
    
    This ensures that preference pairs are generated within the same dialogue turn,
    preventing the model from learning feature drift (e.g., longer query = higher reward)
    that would occur when comparing trajectories across different dialogue turns.

    Assumes each trajectory has:
      - traj["state"]["id"]
      - traj["state"]["dialogue_turn"]
    """
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for t in trajs:
        st = t.get("state", {})
        sid = st.get("id")
        if sid is None:
            # Fallback: hashable repr
            sid = json.dumps(st, sort_keys=True)
        # Use dialogue_turn instead of decision_point for multi-turn mode
        dialogue_turn = int(st.get("dialogue_turn", 0))
        key = (sid, dialogue_turn)
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
            # 需要传递完整的trajectory信息（包括action和has_edge_cases_info）给compute_task_score
            state_with_traj_info = state.copy()
            state_with_traj_info["action"] = t.get("action", "")
            state_with_traj_info["has_edge_cases_info"] = t.get("has_edge_cases_info", False)
            task_code = (
                resolve_execute_code_for_training_and_reward(t, use_full_query=cfg.full_query_for_execute)
                if t.get("action") == "Execute"
                else final_assistant_msg
            )
            if not task_code.strip():
                task_code = final_assistant_msg
            task_score = compute_task_score(state_with_traj_info, domain, assistant_output=task_code)
            interrupt_cost = total_interrupt_cost
            assistant_msg_for_pref = (
                resolve_execute_code_for_training_and_reward(t, use_full_query=cfg.full_query_for_execute)
                if t.get("action") == "Execute"
                else final_assistant_msg
            )
            
        else:
            # Single-interaction mode (backward compatibility)
            assistant_msg = t.get("assistant_msg", "")
            assistant_msg_for_pref = resolve_execute_code_for_training_and_reward(
                t, use_full_query=cfg.full_query_for_execute
            )
            
            # Task score (0/1 for coding with tests)
            # 需要传递完整的trajectory信息（包括action和has_edge_cases_info）给compute_task_score
            state_with_traj_info = state.copy()
            state_with_traj_info["action"] = t.get("action", "")
            state_with_traj_info["has_edge_cases_info"] = t.get("has_edge_cases_info", False)
            task_score = compute_task_score(state_with_traj_info, domain, assistant_output=assistant_msg_for_pref)
            
            # Interrupt cost (Reward_version2: 新公式)
            # C_Interrupt = Σ_{t=1}^{T} (δb_t r_t + λb_t - γb_t a_t)
            meta = (t.get("user_reaction") or {}).get("meta", {})
            n_questions = assistant_msg.count("?")
            # NOTE:
            # Do NOT "promote" Clarify into task_score. In this project, task_score is reserved for
            # actually completing coding tasks (i.e., passing tests under Execute).
            # Clarify is already accounted for via interrupt_cost_v2 where answered clarifications
            # can reduce (or even invert) the interrupt cost. Boosting task_score here double-counts
            # Clarify and empirically collapses preferences to almost-all Clarify.
            
            # 使用新版本的interrupt cost计算
            interrupt_cost = compute_interrupt_cost_v2(
                meta,
                n_questions=n_questions,
                assistant_msg=assistant_msg,
            )
        
        # 状况四：确保Reward有显著分差
        
        # 1. 失败项判定更狠：user_stopped时Reward一律0
        user_stopped = t.get("user_stopped", False)
        if user_stopped:
            total_r = 0.0  # 用户气跑了，无论代码多好，Reward一律0
        else:
            # 2. 对Busy角色，每多一轮对话，加大扣分（从0.1到0.4）
            persona_name = (t.get("persona", {}) or {}).get("name", "")
            dialogue_turn = state.get("dialogue_turn", 0)
            
            # 额外的对话轮次惩罚（仅对Busy-Developer）
            additional_penalty = 0.0
            if persona_name == "Busy-Developer" and dialogue_turn > 0:
                # 每多一轮，扣0.4分（从0.1提高到0.4）
                additional_penalty = 0.4 * dialogue_turn
            
            # 新公式: R = R_task - C_interrupt - additional_penalty
            # 注意：在新公式中，有效澄清的奖励已经包含在C_Interrupt的计算中
            # （通过-γb_t a_t项减少成本，相当于奖励）
            r = cfg.w_task * task_score - cfg.w_interrupt * interrupt_cost - additional_penalty

            # For 1-turn coding trajectories, Clarify cannot complete the task by design.
            # Without an explicit penalty, Clarify tends to dominate because:
            # - Execute often fails tests → task_score=0
            # - answered clarifications can reduce interrupt_cost (sometimes negative)
            # This penalty prevents the "always Clarify" collapse for 1-turn datasets.
            action = t.get("action", "")
            clarify_incomplete_penalty = 0.0
            if domain == "coding" and action == "Clarify":
                clarify_incomplete_penalty = 0.2

            # IMPORTANT: do NOT clamp to non-negative; clamping creates many 0-ties and
            # makes preference generation sensitive to JSONL ordering.
            total_r = float(r - clarify_incomplete_penalty)
        
        # 3. 成功项奖励明确：只有task_completed且has_edge_cases_info为True才给满分
        # 这个逻辑已经在compute_task_score中实现了

        scored.append(
            {
                **t,
                "task_score": float(task_score),
                "interrupt_cost": float(interrupt_cost),
                "total_reward": total_r,
                # Ensure preference text is available for multi-turn traces
                "assistant_msg": assistant_msg_for_pref,
                "assistant_msg_training": assistant_msg_for_pref,
            }
        )
    return scored


def compute_trajectory_level_rewards(
    trajectory_turns: List[Dict],
    cfg: RewardConfig,
) -> List[Dict]:
    """
    Compute trajectory-level rewards for a complete multi-turn conversation.
    
    Key idea: All turns in a trajectory share the FINAL task outcome (R_task),
    but each turn has its own interrupt cost (C_interrupt).
    
    Formula for each turn:
        R_turn = R_final - C_interrupt_turn
    
    Where:
        R_final = task_score from the last turn (or best Execute turn)
        C_interrupt_turn = interrupt cost for this specific turn
    
    Args:
        trajectory_turns: List of turns in a single trajectory (sorted by dialogue_turn)
        cfg: Reward configuration
    
    Returns:
        List of trajectory dicts with computed rewards
    """
    if not trajectory_turns:
        return []
    
    domain = trajectory_turns[0]["state"].get("domain", "coding")
    
    # Step 1: Find the final task score for this trajectory
    # Look for the last Execute action or the terminal turn
    final_task_score = 0.0
    
    for turn in reversed(trajectory_turns):
        action = turn.get("action", "")
        code_for_score = resolve_execute_code_for_training_and_reward(
            turn, use_full_query=cfg.full_query_for_execute
        )
        
        # Only Execute actions can have task_score > 0
        if action == "Execute" and code_for_score:
            # Compute task score for this Execute
            # Need to pass state + action info
            state_with_action = turn["state"].copy()
            state_with_action["action"] = action  # Add action to state for compute_task_score
            state_with_action["has_edge_cases_info"] = turn.get("has_edge_cases_info", False)
            
            task_score = compute_task_score(
                state_with_action,
                domain,
                code_for_score,
            )
            
            # Use the first (last in reverse) Execute's score as final
            if task_score > 0:
                final_task_score = task_score
                break
    
    # Step 2: Compute reward for each turn
    scored = []
    for i, turn in enumerate(trajectory_turns):
        state = turn["state"]
        action = turn.get("action", "")
        assistant_msg = turn.get("assistant_msg", "")
        dialogue_turn = state.get("dialogue_turn", 0)
        
        # Find previous action for multi-turn behavior learning
        prev_action = None
        if dialogue_turn > 0 and i > 0:
            # Get the previous turn's action
            prev_turn = trajectory_turns[i - 1]
            prev_action = prev_turn.get("action")
        
        # Compute interrupt cost for this turn
        user_reaction = turn.get("user_reaction", {})
        meta = user_reaction.get("meta", {})
        n_questions = assistant_msg.count("?") if assistant_msg else 0
        
        interrupt_cost = compute_interrupt_cost_v2(
            meta,
            n_questions=n_questions,
            assistant_msg=assistant_msg,
        )
        
        # Check if user stopped
        user_stopped = turn.get("user_stopped", False)
        if user_stopped:
            # User stopped: trajectory failed, reward = 0
            total_r = 0.0
        else:
            # Persona-aware reward adjustment
            persona_name = (turn.get("persona", {}) or {}).get("name", "")
            task_uncertainty = state.get("task_uncertainty", 0.0)
            
            # Base reward: R = R_final - C_interrupt
            base_reward = float(cfg.w_task * final_task_score - cfg.w_interrupt * interrupt_cost)
            
            # Persona-specific adjustments
            persona_adjustment = 0.0
            
            if persona_name == "Novice-Learner":
                # Novice-Learner: Clarify multiple turns, then Execute
                # More tolerant of Clarify, especially for high uncertainty tasks
                if dialogue_turn == 0:
                    # Turn 0: Strong preference for Clarify
                    if action == "Clarify":
                        uncertainty_bonus = task_uncertainty * 0.20  # Up to 0.20 bonus for high uncertainty at Turn 0
                        persona_adjustment = uncertainty_bonus
                    elif action == "Execute":
                        if task_uncertainty >= 0.7:
                            persona_adjustment = -0.15  # Strong penalty for Execute on high uncertainty at Turn 0
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = -0.10  # Penalty for Execute on medium-high uncertainty at Turn 0
                        else:
                            persona_adjustment = -0.05  # Small penalty for Execute at Turn 0
                else:
                    # Turn 1+: Continue Clarify, then Execute
                    if action == "Clarify":
                        # Reduce interrupt cost penalty for Clarify (make it more attractive)
                        # Higher uncertainty → more benefit from Clarify
                        uncertainty_bonus = task_uncertainty * 0.15  # Up to 0.15 bonus for high uncertainty
                        persona_adjustment = uncertainty_bonus
                        
                        # Multi-turn bonus: If prev_action was Clarify, continue Clarify is good
                        if prev_action == "Clarify" and task_uncertainty > 0.5:
                            persona_adjustment += 0.15  # Increased bonus for continuing Clarify
                    elif action == "Execute":
                        # Execute is acceptable after multiple Clarify turns
                        if task_uncertainty >= 0.7:
                            persona_adjustment = -0.05  # Small penalty for Execute on high uncertainty
                        # Multi-turn bonus: If prev_action was Clarify, Execute is acceptable
                        if prev_action == "Clarify":
                            persona_adjustment += 0.05  # Small bonus for Execute after Clarify
            
            elif persona_name == "Busy-Developer":
                # Busy-Developer: Almost always Execute at Turn 0, strong penalty for multi-turn
                if dialogue_turn == 0:
                    # Turn 0: Strong preference for Execute
                    if action == "Execute":
                        persona_adjustment = 0.20  # Strong bonus for Execute at Turn 0
                    elif action == "Clarify":
                        persona_adjustment = -0.20  # Strong penalty for Clarify at Turn 0
                elif dialogue_turn > 0:
                    # Additional penalty for each extra turn (0.3 per turn)
                    persona_adjustment = -0.3 * dialogue_turn
                    # Also: slight bonus for Execute on low uncertainty (encourage direct action)
                    if action == "Execute" and task_uncertainty < 0.4:
                        persona_adjustment += 0.05
                
                # Multi-turn bonus: If prev_action was Clarify, Execute is strongly preferred
                if prev_action == "Clarify" and action == "Execute":
                    persona_adjustment += 0.15  # Strong bonus for Execute after Clarify
            
            elif persona_name == "Experienced-Engineer":
                # Experienced-Engineer: Clarify ONCE at Turn 0, then Execute immediately.
                # Turn 0: Clarify is correct (ask the key question once).
                # Turn 1+: Execute is always correct (stop over-clarifying).
                if dialogue_turn == 0:
                    # Turn 0: Strong preference for Clarify (especially for high uncertainty)
                    if action == "Clarify":
                        if task_uncertainty >= 0.7:
                            persona_adjustment = 0.25  # Strong bonus for Clarify on high uncertainty at Turn 0
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = 0.20  # Bonus for Clarify on medium-high uncertainty at Turn 0
                        elif task_uncertainty >= 0.4:
                            persona_adjustment = 0.15  # Bonus for Clarify on medium uncertainty at Turn 0
                    elif action == "Execute":
                        if task_uncertainty >= 0.7:
                            persona_adjustment = -0.20  # Strong penalty for Execute on high uncertainty at Turn 0
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = -0.15  # Penalty for Execute on medium-high uncertainty at Turn 0
                        elif task_uncertainty < 0.4:
                            persona_adjustment = 0.05  # Small bonus for Execute on low uncertainty at Turn 0
                else:
                    # Turn 1+: Experienced should EXECUTE now, not keep clarifying.
                    # Flip direction vs Turn 0: Execute is rewarded, Clarify is penalized.
                    if action == "Execute":
                        if prev_action == "Clarify":
                            persona_adjustment = 0.20  # Strong bonus: execute right after clarifying once
                        else:
                            persona_adjustment = 0.10  # Bonus for execute at turn 1+
                    elif action == "Clarify":
                        # Over-clarifying: always penalize for Experienced at Turn 1+
                        if task_uncertainty >= 0.7:
                            persona_adjustment = -0.15  # Strong penalty for over-clarifying
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = -0.10  # Penalty for over-clarifying
                        else:
                            persona_adjustment = -0.05  # Small penalty even on low uncertainty
            
            total_r = float(base_reward + persona_adjustment)
        
        training_text = resolve_execute_code_for_training_and_reward(
            turn, use_full_query=cfg.full_query_for_execute
        )
        # Store results
        scored.append({
            **turn,
            "task_score": float(final_task_score),  # All turns share final score
            "interrupt_cost": float(interrupt_cost),  # Each turn has its own cost
            "total_reward": total_r,
            "prev_action": prev_action,  # Store prev_action for multi-turn learning
            "dialogue_turn": dialogue_turn,  # Store dialogue_turn for multi-turn learning
            "assistant_msg_training": training_text,
        })
    
    return scored


def build_multi_turn_prefs_for_group(
    scored_trajs: List[Dict],
    persona_name: str,
    dialogue_turn: int,
) -> List[Dict]:
    """
    为Turn 1+生成persona-aware的preference pairs。
    
    注意：不修改reward值，只调整排序策略。
    Reward已经在compute_trajectory_level_rewards中考虑了persona特性。
    
    这里只是通过调整排序来强化persona-aware的preference，但不改变reward本身。
    """
    if len(scored_trajs) < 2:
        return []
    
    # 获取task_uncertainty
    task_uncertainty = scored_trajs[0].get("state", {}).get("task_uncertainty", 0.5)
    
    # Persona-aware preference selection (只用于排序，不改变reward值)
    # 通过调整排序的tie-break来体现persona偏好
    if persona_name == "Novice-Learner":
        # Novice-Learner: 如果uncertainty仍然高，倾向于继续Clarify
        if task_uncertainty > 0.5:
            preferred_action = "Clarify"
        else:
            preferred_action = "Execute"
    elif persona_name == "Experienced-Engineer":
        # Experienced-Engineer: 倾向于Execute（快速执行）
        preferred_action = "Execute"
    elif persona_name == "Busy-Developer":
        # Busy-Developer: 强烈倾向于Execute（避免多轮）
        preferred_action = "Execute"
    else:
        preferred_action = None
    
    # 使用标准方法生成preference pairs，但传入preferred_action用于tie-break
    # build_prefs_from_group已经支持tie-break，我们只需要确保它使用preferred_action
    return build_prefs_from_group(scored_trajs, preferred_action=preferred_action)


def build_prefs_from_group(scored_trajs: List[Dict], preferred_action: Optional[str] = None) -> List[Dict]:
    """
    Given scored trajectories for a single (state_id, dialogue_turn),
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

    # Deterministic + meaningful sorting:
    # - Primary: total_reward
    # - Secondary (tie-break): prefer Execute when task is clear, prefer Clarify when unclear.
    # This prevents 0-ties from defaulting to whatever appears first in the JSONL.
    state = scored_trajs[0].get("state", {})
    # For coding, default to preferring Execute in tie-breaks. The BigCodeBench masked
    # states tend to have very high task_uncertainty (often ~0.8+), which would otherwise
    # make tie-breaks always prefer Clarify and collapse training.
    prefer_action = "Execute" if state.get("domain", "coding") == "coding" else "Clarify"
    eps = 1e-3

    def adjusted_score(t: Dict) -> float:
        bonus = eps if t.get("action") == prefer_action else 0.0
        return float(t["total_reward"]) + bonus

    scored_sorted = sorted(scored_trajs, key=adjusted_score, reverse=True)

    # Build multiple cross-action pairs to strengthen signal.
    # We only emit pairs where:
    # - actions differ
    # - assistant messages differ
    # - either reward margin is meaningful OR tie-break decides (very small margin)
    max_pairs = 3
    min_margin = 0.02
    prefs: List[Dict] = []

    for i in range(len(scored_sorted)):
        for j in range(i + 1, len(scored_sorted)):
            if len(prefs) >= max_pairs:
                break
            hi = scored_sorted[i]
            lo = scored_sorted[j]
            if hi.get("action") == lo.get("action"):
                continue
            if pref_assistant_text(hi) == pref_assistant_text(lo):
                continue

            # Decide preference by adjusted score (includes tie-break).
            hi_adj = adjusted_score(hi)
            lo_adj = adjusted_score(lo)
            if hi_adj <= lo_adj:
                continue

            # If raw rewards are too close, only keep if tie-break is the reason.
            raw_gap = float(hi["total_reward"]) - float(lo["total_reward"])
            if raw_gap < min_margin and hi.get("action") != prefer_action:
                # Too close and not aligned with tie-break preference → skip noisy pair.
                continue

            # Extract multi-turn information for training
            # Both hi and lo are from the same group, so they share the same prev_action and dialogue_turn
            prev_action = hi.get("prev_action")  # Can be None for Turn 0
            dialogue_turn = hi.get("dialogue_turn", hi.get("state", {}).get("dialogue_turn", 0))

            prefs.append(
                {
                    "state": hi["state"],
                    "persona": hi.get("persona", {}),  # ← 添加persona信息！
                    "chosen_action": hi["action"],
                    "rejected_action": lo["action"],
                    # V16: 添加action前缀，让模型学会先决策再生成
                    "chosen_assistant_msg": f"{hi['action']}\n{pref_assistant_text(hi)}",
                    "rejected_assistant_msg": f"{lo['action']}\n{pref_assistant_text(lo)}",
                    "chosen_reward": hi["total_reward"],
                    "rejected_reward": lo["total_reward"],
                    "chosen_task_score": hi["task_score"],
                    "rejected_task_score": lo["task_score"],
                    "chosen_interrupt_cost": hi["interrupt_cost"],
                    "rejected_interrupt_cost": lo["interrupt_cost"],
                    "prev_action": prev_action,  # For multi-turn learning: previous action (None for Turn 0)
                    "dialogue_turn": dialogue_turn,  # For multi-turn learning: current dialogue turn
                }
            )
        if len(prefs) >= max_pairs:
            break

    return prefs


def _stable_hash_to_unit_interval(text: str, seed: int) -> float:
    """Deterministic hash -> [0, 1)."""
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16 ** 16)


def rebalance_prefs_by_action(
    prefs: List[Dict],
    target_execute_ratio: float,
    seed: int,
) -> List[Dict]:
    """Downsample over-represented action to reach a target Execute ratio."""
    if not prefs:
        return prefs
    if not (0.0 < target_execute_ratio < 1.0):
        return prefs

    execute = [p for p in prefs if p.get("chosen_action") == "Execute"]
    clarify = [p for p in prefs if p.get("chosen_action") == "Clarify"]
    total = len(prefs)
    exec_ratio = len(execute) / total

    # Already close enough: keep as is
    if abs(exec_ratio - target_execute_ratio) < 0.02:
        return prefs

    if exec_ratio > target_execute_ratio:
        # Downsample Execute relative to existing Clarify count
        desired_exec = max(1, int(round(len(clarify) * target_execute_ratio / (1 - target_execute_ratio))))
        keep_prob = min(1.0, desired_exec / max(1, len(execute)))
        kept_execute = [
            p for p in execute
            if _stable_hash_to_unit_interval(p["state"].get("id", str(p["state"])), seed) < keep_prob
        ]
        return kept_execute + clarify
    else:
        # Downsample Clarify relative to existing Execute count
        desired_clarify = max(1, int(round(len(execute) * (1 - target_execute_ratio) / target_execute_ratio)))
        keep_prob = min(1.0, desired_clarify / max(1, len(clarify)))
        kept_clarify = [
            p for p in clarify
            if _stable_hash_to_unit_interval(p["state"].get("id", str(p["state"])), seed) < keep_prob
        ]
        return execute + kept_clarify


def rebalance_prefs_by_persona(
    prefs: List[Dict],
    seed: int,
) -> List[Dict]:
    """Downsample over-represented personas so each persona has equal pair count.

    Groups by persona name (pref["persona"]["name"] or pref["persona"] string).
    Keeps min_count pairs per persona, chosen deterministically by stable hash.
    """
    if not prefs:
        return prefs

    def _get_persona(p: Dict) -> str:
        persona = p.get("persona")
        if isinstance(persona, dict):
            return persona.get("name", "Unknown")
        return str(persona) if persona else "Unknown"

    from collections import defaultdict
    by_persona: Dict[str, List[Dict]] = defaultdict(list)
    for p in prefs:
        by_persona[_get_persona(p)].append(p)

    if len(by_persona) <= 1:
        return prefs

    min_count = min(len(v) for v in by_persona.values())
    if min_count == 0:
        return prefs

    result = []
    for persona_name, persona_prefs in by_persona.items():
        if len(persona_prefs) <= min_count:
            result.extend(persona_prefs)
        else:
            # Deterministically keep min_count pairs via stable hash
            sorted_prefs = sorted(
                persona_prefs,
                key=lambda p: _stable_hash_to_unit_interval(p["state"].get("id", str(p["state"])), seed),
            )
            result.extend(sorted_prefs[:min_count])

    print(f"🔧 Per-persona rebalance: {len(prefs)} -> {len(result)} "
          f"({min_count} pairs × {len(by_persona)} personas)")
    for persona_name, persona_prefs in sorted(by_persona.items()):
        kept = min(len(persona_prefs), min_count)
        print(f"   {persona_name}: {len(persona_prefs)} -> {kept}")
    return result


def compute_preferences(
    traj_path: Path,
    out_path: Path,
    cfg: RewardConfig,
    target_execute_ratio: float,
    rebalance_seed: int,
    use_trajectory_level: bool = True,
) -> None:
    """High-level function: trajectories → prefs JSONL.
    
    Args:
        traj_path: Path to trajectories JSONL
        out_path: Output path for preferences JSONL
        cfg: Reward configuration
        target_execute_ratio: Target ratio for Execute in chosen prefs
        rebalance_seed: Seed for deterministic rebalancing
        use_trajectory_level: If True, use trajectory-level reward (recommended).
                             If False, use step-level reward (legacy).
    """
    print(f"📂 Loading trajectories from: {traj_path}")
    trajs = load_trajectories(traj_path)
    print(f"📊 Loaded {len(trajs)} trajectory turns")

    prefs: List[Dict] = []
    
    if use_trajectory_level:
        print("🎯 Using TRAJECTORY-LEVEL reward computation")
        # Group by trajectory_id to get complete trajectories
        trajectory_groups = group_by_trajectory(trajs)
        print(f"📊 Grouped into {len(trajectory_groups)} complete trajectories")
        
        # Compute trajectory-level rewards
        all_scored = []
        total_trajectories = len(trajectory_groups)
        for idx, (traj_id, turns) in enumerate(trajectory_groups.items(), 1):
            scored_turns = compute_trajectory_level_rewards(turns, cfg)
            all_scored.extend(scored_turns)
            
            # Print progress every 50 trajectories or at milestones
            if idx % 50 == 0 or idx == total_trajectories:
                progress_pct = (idx / total_trajectories) * 100
                print(f"📊 Progress: {idx}/{total_trajectories} trajectories ({progress_pct:.1f}%) - Computed rewards for {len(all_scored)} turns", flush=True)
        
        print(f"📊 Computed rewards for {len(all_scored)} turns across all trajectories")
        
        # NEW: Trajectory-level comparison for preference pair generation
        # Group by (state_id, persona) and compare final rewards of complete trajectories
        # This ensures each persona has enough pairs for training
        trajectory_info = {}  # trajectory_id -> {reward, turn0_turn, persona, state}
        
        # Group all_scored by trajectory_id to find Turn 0 and final reward
        traj_turns_dict = defaultdict(list)
        for t in all_scored:
            traj_id = t.get("trajectory_id")
            if traj_id is None:
                continue
            traj_turns_dict[traj_id].append(t)
        
        # For each trajectory, find Turn 0 and use Turn 0's reward for comparison
        # This ensures we compare actions at the decision point (Turn 0), not final outcome
        for traj_id, turns in traj_turns_dict.items():
            # Sort by dialogue_turn
            turns_sorted = sorted(turns, key=lambda x: x["state"].get("dialogue_turn", 0))
            
            # Find Turn 0 (for action/message and reward)
            turn0 = None
            for t in turns_sorted:
                if t["state"].get("dialogue_turn", 0) == 0:
                    turn0 = t
                    break
            
            if turn0 is None:
                continue  # Skip if no Turn 0
            
            # Use Turn 0's reward for comparison (not final reward)
            # This makes sense because we want to compare the immediate reward of choosing
            # Clarify vs Execute at Turn 0, not the final trajectory outcome
            turn0_reward = turn0.get("total_reward", 0)
            
            trajectory_info[traj_id] = {
                "reward": turn0_reward,  # Use Turn 0 reward for comparison
                "turn0": turn0,  # Use Turn 0 for action/message
                "persona": turn0.get("persona", {}),
                "state": turn0["state"],
            }
        
        # Group trajectories by (state_id, persona) for comparison
        state_persona_groups = defaultdict(list)
        for traj_id, traj_info in trajectory_info.items():
            state_id = traj_info["state"].get("id", "unknown")
            persona = traj_info["persona"]
            persona_name = persona.get("name", "Unknown") if isinstance(persona, dict) else "Unknown"
            key = (state_id, persona_name)
            state_persona_groups[key].append({
                "trajectory_id": traj_id,
                "reward": traj_info["reward"],
                "turn0": traj_info["turn0"],  # Use Turn 0 for action/message
            })
        
        print(f"📊 Grouped into {len(state_persona_groups)} (state_id, persona) groups for trajectory-level preference generation")
        
        # Generate preference pairs by comparing trajectories within each (state, persona) group
        # FIXED: Group by Turn 0 action first, then compare across action groups
        # This ensures we always compare different actions (Clarify vs Execute)
        total_groups_processed = 0
        total_pairs_generated = 0
        skipped_reasons = defaultdict(int)
        
        for (state_id, persona_name), traj_list in state_persona_groups.items():
            if len(traj_list) < 2:
                skipped_reasons["less_than_2"] += 1
                continue
            
            total_groups_processed += 1
            
            # Group by Turn 0 action
            action_groups = defaultdict(list)
            for traj in traj_list:
                action = traj["turn0"].get("action", "unknown")
                action_groups[action].append(traj)
            
            # Need at least 2 different actions to generate pairs
            if len(action_groups) < 2:
                skipped_reasons["only_one_action"] += 1
                continue
            
            # For each action group, find the best trajectory (highest reward)
            action_best = {}
            for action, trajs in action_groups.items():
                if trajs:
                    best = max(trajs, key=lambda x: x["reward"])
                    action_best[action] = best
            
            # Also find worst trajectories for comparison
            action_worst = {}
            for action, trajs in action_groups.items():
                if trajs:
                    worst = min(trajs, key=lambda x: x["reward"])
                    action_worst[action] = worst
            
            # Generate pairs: compare best trajectories from different action groups
            # Strategy: For each pair of actions, compare their best trajectories
            max_pairs = min(3, len(action_groups) * (len(action_groups) - 1))
            generated_pairs = 0
            
            actions_list = list(action_groups.keys())
            for i, action1 in enumerate(actions_list):
                if generated_pairs >= max_pairs:
                    break
                for action2 in actions_list[i+1:]:
                    if generated_pairs >= max_pairs:
                        break
                    
                    # Compare best of action1 vs best of action2
                    best_traj1 = action_best[action1]
                    best_traj2 = action_best[action2]
                    best_turn0_1 = best_traj1["turn0"]
                    best_turn0_2 = best_traj2["turn0"]
                    
                    # Check if messages are different
                    if pref_assistant_text(best_turn0_1) == pref_assistant_text(best_turn0_2):
                        skipped_reasons["same_message"] += 1
                        continue
                    
                    # Determine which is chosen based on reward
                    if best_traj1["reward"] > best_traj2["reward"]:
                        chosen_traj = best_traj1
                        rejected_traj = best_traj2
                        chosen_turn0 = best_turn0_1
                        rejected_turn0 = best_turn0_2
                    elif best_traj2["reward"] > best_traj1["reward"]:
                        chosen_traj = best_traj2
                        rejected_traj = best_traj1
                        chosen_turn0 = best_turn0_2
                        rejected_turn0 = best_turn0_1
                    else:
                        # Reward相同，跳过
                        skipped_reasons["reward_diff_too_small"] += 1
                        continue
                    
                    # Check reward difference (use smaller threshold for trajectory-level)
                    reward_diff = chosen_traj["reward"] - rejected_traj["reward"]
                    if reward_diff < 0.0001:  # Very small threshold to allow more pairs
                        skipped_reasons["reward_diff_too_small"] += 1
                        continue
                    
                    # Extract dialogue_turn and prev_action from Turn 0
                    dialogue_turn = chosen_turn0["state"].get("dialogue_turn", 0)
                    prev_action = chosen_turn0.get("prev_action")  # Should be None for Turn 0
                    
                    prefs.append({
                        "state": chosen_turn0["state"],
                        "persona": chosen_turn0.get("persona", {}),
                        "chosen_action": chosen_turn0["action"],
                        "rejected_action": rejected_turn0["action"],
                        "chosen_assistant_msg": f"{chosen_turn0['action']}\n{pref_assistant_text(chosen_turn0)}",
                        "rejected_assistant_msg": f"{rejected_turn0['action']}\n{pref_assistant_text(rejected_turn0)}",
                        "chosen_reward": chosen_traj["reward"],
                        "rejected_reward": rejected_traj["reward"],
                        "chosen_task_score": chosen_turn0.get("task_score", 0),
                        "rejected_task_score": rejected_turn0.get("task_score", 0),
                        "chosen_interrupt_cost": chosen_turn0.get("interrupt_cost", 0),
                        "rejected_interrupt_cost": rejected_turn0.get("interrupt_cost", 0),
                        "prev_action": prev_action,
                        "dialogue_turn": dialogue_turn,
                    })
                    generated_pairs += 1
                    total_pairs_generated += 1

                    # --- Method B: multi-turn pairs for Turn 1+ Clarify turns ---
                    # For the chosen Clarify trajectory, generate additional pairs at each
                    # subsequent Clarify turn using the rejected Execute trajectory's Turn 0
                    # response as the counterfactual "what if we executed right now".
                    if chosen_turn0["action"] == "Clarify":
                        clarify_traj_id = chosen_traj["trajectory_id"]
                        execute_turn0 = rejected_turn0  # the Execute-at-T0 response

                        full_clarify_turns = sorted(
                            traj_turns_dict.get(clarify_traj_id, []),
                            key=lambda x: x["state"].get("dialogue_turn", 0),
                        )
                        for ct in full_clarify_turns:
                            ct_turn = ct["state"].get("dialogue_turn", 0)
                            if ct_turn == 0:
                                continue  # already handled above
                            if ct.get("action") != "Clarify":
                                continue  # only Clarify turns get multi-turn pairs

                            ct_reward = ct.get("total_reward", chosen_traj["reward"])
                            exe_reward = rejected_traj["reward"]
                            if ct_reward - exe_reward < 0.0001:
                                continue  # no useful signal

                            ct_msg = pref_assistant_text(ct)
                            exe_msg = pref_assistant_text(execute_turn0)
                            if not ct_msg or not exe_msg or ct_msg == exe_msg:
                                continue

                            prefs.append({
                                "state": ct["state"],
                                "persona": ct.get("persona", chosen_turn0.get("persona", {})),
                                "chosen_action": "Clarify",
                                "rejected_action": "Execute",
                                "chosen_assistant_msg": f"Clarify\n{ct_msg}",
                                "rejected_assistant_msg": f"Execute\n{exe_msg}",
                                "chosen_reward": ct_reward,
                                "rejected_reward": exe_reward,
                                "chosen_task_score": ct.get("task_score", 0),
                                "rejected_task_score": rejected_turn0.get("task_score", 0),
                                "chosen_interrupt_cost": ct.get("interrupt_cost", 0),
                                "rejected_interrupt_cost": rejected_turn0.get("interrupt_cost", 0),
                                "prev_action": "Clarify",
                                "dialogue_turn": ct_turn,
                                "multi_turn_pair": True,
                            })
                            total_pairs_generated += 1

        # --- Method A: pairs from fork Execute trajectories ---
        # Fork trajectories are forked from a Clarify mainline at each intermediate
        # Clarify turn.  For each fork, create a pair:
        #   Clarify-again (mainline) vs Execute-now (fork) at the SAME dialogue state.
        method_a_pairs = 0
        for _traj_id, turns in traj_turns_dict.items():
            for fork_turn in turns:
                if not fork_turn.get("is_fork"):
                    continue
                parent_id = fork_turn.get("parent_trajectory_id")
                fork_at = fork_turn.get("fork_at_turn")
                if parent_id is None or fork_at is None:
                    continue
                # Find mainline Clarify turn at the fork position
                parent_turns = traj_turns_dict.get(parent_id, [])
                mainline_clarify = None
                for pt in parent_turns:
                    if pt.get("turn", 0) == fork_at and pt.get("action") == "Clarify":
                        mainline_clarify = pt
                        break
                if mainline_clarify is None:
                    continue
                clarify_reward = mainline_clarify.get("total_reward", 0)
                execute_reward = fork_turn.get("total_reward", 0)
                if abs(clarify_reward - execute_reward) < 0.0001:
                    continue  # no useful signal
                clarify_msg = pref_assistant_text(mainline_clarify)
                execute_msg = pref_assistant_text(fork_turn)
                if not clarify_msg or not execute_msg or clarify_msg == execute_msg:
                    continue
                if clarify_reward > execute_reward:
                    chosen_t, rejected_t = mainline_clarify, fork_turn
                    chosen_a, rejected_a = "Clarify", "Execute"
                else:
                    chosen_t, rejected_t = fork_turn, mainline_clarify
                    chosen_a, rejected_a = "Execute", "Clarify"
                prefs.append({
                    "state": mainline_clarify["state"],
                    "persona": mainline_clarify.get("persona", {}),
                    "chosen_action": chosen_a,
                    "rejected_action": rejected_a,
                    "chosen_assistant_msg": f"{chosen_a}\n{pref_assistant_text(chosen_t)}",
                    "rejected_assistant_msg": f"{rejected_a}\n{pref_assistant_text(rejected_t)}",
                    "chosen_reward": chosen_t.get("total_reward", 0),
                    "rejected_reward": rejected_t.get("total_reward", 0),
                    "chosen_task_score": chosen_t.get("task_score", 0),
                    "rejected_task_score": rejected_t.get("task_score", 0),
                    "chosen_interrupt_cost": chosen_t.get("interrupt_cost", 0),
                    "rejected_interrupt_cost": rejected_t.get("interrupt_cost", 0),
                    "prev_action": "Clarify",
                    "dialogue_turn": fork_at - 1,
                    "multi_turn_pair": True,
                    "method_a_pair": True,
                })
                total_pairs_generated += 1
                method_a_pairs += 1
        if method_a_pairs > 0:
            print(f"   - Method A (fork) pairs: {method_a_pairs}")

        # Debug output
        print(f"📊 Trajectory-level preference generation:")
        print(f"   - Processed {total_groups_processed} groups with 2+ trajectories")
        print(f"   - Generated {total_pairs_generated} pairs")
        if skipped_reasons:
            print(f"   - Skipped reasons: {dict(skipped_reasons)}")
    
    else:
        print("⚠️  Using STEP-LEVEL reward computation (legacy mode)")
        groups = group_by_state(trajs)
        print(f"📊 Grouped into {len(groups)} (state_id, dialogue_turn) groups")

        for (sid, dialogue_turn), g in groups.items():
            scored = compute_rewards_for_group(g, cfg)
            group_prefs = build_prefs_from_group(scored)
            prefs.extend(group_prefs)

    print(f"📊 Generated {len(prefs)} preference pairs")

    # Step 1: per-persona rebalance (equal pairs per persona, prevents one persona dominating)
    prefs = rebalance_prefs_by_persona(prefs, rebalance_seed)

    # Step 2: optional global action rebalance (Execute vs Clarify ratio within the balanced set)
    if target_execute_ratio is not None:
        before = len(prefs)
        prefs = rebalance_prefs_by_action(prefs, target_execute_ratio, rebalance_seed)
        after = len(prefs)
        print(f"🔧 Action rebalanced: {before} -> {after} (target Execute ratio={target_execute_ratio})")

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
        default=0.2,
        help="Weight for interrupt cost penalty (default: 0.2)",
    )
    parser.add_argument(
        "--target_execute_ratio",
        type=float,
        default=0.8,
        help="Target ratio for Execute in chosen prefs (default: 0.8; set <0 or >1 to disable)",
    )
    parser.add_argument(
        "--rebalance_seed",
        type=int,
        default=42,
        help="Seed for deterministic preference rebalancing (default: 42)",
    )
    parser.add_argument(
        "--use_trajectory_level",
        action="store_true",
        default=True,
        help="Use trajectory-level reward computation (default: True, recommended)",
    )
    parser.add_argument(
        "--use_step_level",
        dest="use_trajectory_level",
        action="store_false",
        help="Use step-level reward computation (legacy mode)",
    )
    parser.add_argument(
        "--full_query_for_execute",
        action="store_true",
        help=(
            "For Execute turns, use trajectory code_versions['full_query'] for task_score and "
            "preference text when present (Clarify unchanged; falls back to assistant_msg)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.trajectories)
    out_path = Path(args.out)
    cfg = RewardConfig(
        w_task=args.w_task,
        w_interrupt=args.w_interrupt,
        full_query_for_execute=args.full_query_for_execute,
    )

    print("⚙️  Reward config:")
    print(f"  - w_task = {cfg.w_task}")
    print(f"  - w_interrupt = {cfg.w_interrupt}")
    print(f"  - full_query_for_execute = {cfg.full_query_for_execute}")

    target_ratio = args.target_execute_ratio
    if not (0.0 < target_ratio < 1.0):
        target_ratio = None

    compute_preferences(
        traj_path,
        out_path,
        cfg,
        target_execute_ratio=target_ratio,
        rebalance_seed=args.rebalance_seed,
        use_trajectory_level=args.use_trajectory_level,
    )


if __name__ == "__main__":
    main()

