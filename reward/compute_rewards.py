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
    w_interrupt: float = 0.3  # 提高到0.3，放大interrupt_cost的影响以拉开分差


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
            task_score = compute_task_score(state_with_traj_info, domain, assistant_output=final_assistant_msg)
            interrupt_cost = total_interrupt_cost
            assistant_msg_for_pref = final_assistant_msg
            
        else:
            # Single-interaction mode (backward compatibility)
            assistant_msg = t.get("assistant_msg", "")
            assistant_msg_for_pref = assistant_msg
            
            # Task score (0/1 for coding with tests)
            # 需要传递完整的trajectory信息（包括action和has_edge_cases_info）给compute_task_score
            state_with_traj_info = state.copy()
            state_with_traj_info["action"] = t.get("action", "")
            state_with_traj_info["has_edge_cases_info"] = t.get("has_edge_cases_info", False)
            task_score = compute_task_score(state_with_traj_info, domain, assistant_output=assistant_msg)
            
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
        assistant_msg = turn.get("assistant_msg", "")
        
        # Only Execute actions can have task_score > 0
        if action == "Execute" and assistant_msg:
            # Compute task score for this Execute
            # Need to pass state + action info
            state_with_action = turn["state"].copy()
            state_with_action["action"] = action  # Add action to state for compute_task_score
            state_with_action["has_edge_cases_info"] = turn.get("has_edge_cases_info", False)
            
            task_score = compute_task_score(
                state_with_action,
                domain,
                assistant_msg
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
                # Experienced-Engineer: Clarify at Turn 0, then Execute at Turn 1
                # Should Clarify for high/medium uncertainty, Execute for low uncertainty
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
                    # Turn 1+: Use original logic
                    if action == "Clarify":
                        if task_uncertainty >= 0.7:
                            persona_adjustment = 0.15  # Bonus for Clarify on high uncertainty
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = 0.10  # Bonus for Clarify on medium-high uncertainty
                        elif task_uncertainty >= 0.4:
                            persona_adjustment = 0.05  # Small bonus for medium uncertainty
                    elif action == "Execute":
                        if task_uncertainty >= 0.7:
                            persona_adjustment = -0.10  # Penalty for Execute on high uncertainty
                        elif task_uncertainty >= 0.5:
                            persona_adjustment = -0.05  # Penalty for Execute on medium-high uncertainty
                        elif task_uncertainty < 0.4:
                            persona_adjustment = 0.05  # Bonus for Execute on low uncertainty
                    
                    # Multi-turn bonus: If prev_action was Clarify, Execute is preferred
                    if prev_action == "Clarify":
                        persona_adjustment += 0.10  # Bonus for Execute after Clarify
            
            total_r = float(base_reward + persona_adjustment)
        
        # Store results
        scored.append({
            **turn,
            "task_score": float(final_task_score),  # All turns share final score
            "interrupt_cost": float(interrupt_cost),  # Each turn has its own cost
            "total_reward": total_r,
            "prev_action": prev_action,  # Store prev_action for multi-turn learning
            "dialogue_turn": dialogue_turn,  # Store dialogue_turn for multi-turn learning
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
            # Allow same-action pairs if rewards differ (for learning better Clarify/Execute)
            # But still prefer cross-action pairs for Turn 0
            same_action = hi.get("action") == lo.get("action")
            dialogue_turn = hi.get("dialogue_turn", hi.get("state", {}).get("dialogue_turn", 0))
            if same_action and dialogue_turn == 0:
                # Turn 0: prefer cross-action pairs (Clarify vs Execute)
                continue
            if hi.get("assistant_msg", "").strip() == lo.get("assistant_msg", "").strip():
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
                    "chosen_assistant_msg": f"{hi['action']}\n{hi.get('assistant_msg', '')}",
                    "rejected_assistant_msg": f"{lo['action']}\n{lo.get('assistant_msg', '')}",
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
    base_execute_ratio: float,
    seed: int,
) -> List[Dict]:
    """
    Rebalance preference pairs by persona, allowing different personas to have different action ratios.
    
    Persona-specific target ratios:
    - Busy-Developer: 0.8-0.9 Execute (high, prefers direct execution)
    - Experienced-Engineer: 0.3-0.4 Execute at Turn 0, 0.7-0.8 at Turn 1+ (clarify first, then execute)
    - Novice-Learner: 0.2-0.3 Execute (low, prefers multiple clarifications)
    """
    if not prefs:
        return prefs
    
    from collections import defaultdict
    
    # Group by persona
    persona_groups = defaultdict(list)
    for pref in prefs:
        persona_name = pref.get("persona_name") or pref.get("persona", {}).get("name", "Unknown")
        persona_groups[persona_name].append(pref)
    
    rebalanced_prefs = []
    
    # First pass: rebalance each persona to target ratio
    persona_rebalanced = {}
    for persona_name, persona_prefs in persona_groups.items():
        # Determine persona-specific target ratio
        if persona_name == "Busy-Developer":
            target_ratio = 0.85  # High Execute ratio
        elif persona_name == "Experienced-Engineer":
            # Check if Turn 0 or Turn 1+
            turn0_count = sum(1 for p in persona_prefs if p.get("dialogue_turn", 0) == 0)
            if turn0_count > len(persona_prefs) * 0.5:
                target_ratio = 0.35  # Low Execute at Turn 0 (prefer Clarify)
            else:
                target_ratio = 0.75  # Higher Execute at Turn 1+ (prefer Execute after Clarify)
        elif persona_name == "Novice-Learner":
            target_ratio = 0.25  # Low Execute ratio (prefer multiple Clarify)
        else:
            # Default: use base ratio
            target_ratio = base_execute_ratio
        
        # Check actual ratio first
        execute_count = sum(1 for p in persona_prefs if p.get("chosen_action") == "Execute")
        clarify_count = sum(1 for p in persona_prefs if p.get("chosen_action") == "Clarify")
        actual_ratio = execute_count / len(persona_prefs) if persona_prefs else 0.0
        
        # If actual ratio is very different from target, adjust target to be more lenient
        # This prevents over-filtering when data doesn't match expected distribution
        if abs(actual_ratio - target_ratio) > 0.3:
            # If gap is too large, adjust target to be closer to actual to avoid losing too much data
            # Use a more lenient target (closer to actual ratio)
            adjusted_target = actual_ratio + 0.1 * (target_ratio - actual_ratio)  # Move 10% towards target
            print(f"  {persona_name}: Large gap detected (actual={actual_ratio:.2f}, target={target_ratio:.2f}), "
                  f"using adjusted target={adjusted_target:.2f} to preserve data")
            target_ratio = adjusted_target
        
        # Separate Turn 0 and Turn 1+ pairs before rebalancing
        # Priority: Preserve all Turn 1+ pairs (they are crucial for multi-turn learning)
        turn0_persona_prefs = [p for p in persona_prefs if p.get("dialogue_turn", 0) == 0]
        turn1plus_persona_prefs = [p for p in persona_prefs if p.get("dialogue_turn", 0) > 0]
        
        # Rebalance only Turn 0 pairs
        rebalanced_turn0 = rebalance_prefs_by_action(turn0_persona_prefs, target_ratio, seed)
        
        # Always keep all Turn 1+ pairs (they are important for multi-turn learning)
        rebalanced = rebalanced_turn0 + turn1plus_persona_prefs
        
        # If rebalancing resulted in too few pairs, keep original data
        # This prevents over-filtering when data doesn't match expected distribution
        min_pairs_threshold = 20  # Minimum pairs to keep per persona (reduced from 50 to preserve more data)
        if len(rebalanced) < min_pairs_threshold and len(persona_prefs) >= min_pairs_threshold:
            print(f"  {persona_name}: Rebalance resulted in too few pairs ({len(rebalanced)} < {min_pairs_threshold}), "
                  f"keeping original {len(persona_prefs)} pairs")
            persona_rebalanced[persona_name] = persona_prefs
        else:
            persona_rebalanced[persona_name] = rebalanced
        
        # Log rebalancing results
        final_prefs = persona_rebalanced[persona_name]
        execute_count = sum(1 for p in final_prefs if p.get("chosen_action") == "Execute")
        clarify_count = sum(1 for p in final_prefs if p.get("chosen_action") == "Clarify")
        actual_ratio = execute_count / len(final_prefs) if final_prefs else 0.0
        print(f"  {persona_name}: {len(persona_prefs)} -> {len(final_prefs)} pairs "
              f"(target Execute={target_ratio:.2f}, actual={actual_ratio:.2f})")
    
    # Second pass: Balance data volume across personas (only if needed)
    # Priority: Preserve all Turn 0 pairs (should be 121 per persona)
    # Then balance Turn 1+ data if needed
    
    for persona_name, persona_prefs in persona_rebalanced.items():
        # Separate Turn 0 and Turn 1+ pairs
        turn0_pairs = [p for p in persona_prefs if p.get("dialogue_turn", 0) == 0]
        turn1plus_pairs = [p for p in persona_prefs if p.get("dialogue_turn", 0) > 0]
        
        # Always keep all Turn 0 pairs (they should be balanced by design)
        # For Turn 0, if there are multiple pairs per state, keep only the best one per state
        # But don't enforce a strict limit - keep all unique state pairs
        state_best = {}
        for p in turn0_pairs:
            state_id = p.get("state", {}).get("id", "")
            if state_id not in state_best or p.get("chosen_reward", 0) > state_best[state_id].get("chosen_reward", 0):
                state_best[state_id] = p
        turn0_pairs = list(state_best.values())
        # Log if significantly different from expected, but don't enforce limit
        expected_turn0 = len([p for p in persona_prefs if p.get("dialogue_turn", 0) == 0])
        if abs(len(turn0_pairs) - expected_turn0) > 10:
            print(f"  {persona_name}: Turn 0 pairs adjusted from {expected_turn0} to {len(turn0_pairs)} (keeping all unique state pairs)")
        
        rebalanced_prefs.extend(turn0_pairs)
        rebalanced_prefs.extend(turn1plus_pairs)
    
    # Third pass: Balance Turn 1+ data across personas if needed
    # IMPORTANT: Preserve all Turn 3+ pairs (they are crucial for learning multi-turn behavior)
    persona_turn1plus = {}
    turn3plus_pairs = []  # Separate Turn 3+ pairs to preserve them
    for p in rebalanced_prefs:
        dialogue_turn = p.get("dialogue_turn", 0)
        if dialogue_turn > 0:
            if dialogue_turn >= 3:
                # Preserve all Turn 3+ pairs (they are important and rare)
                turn3plus_pairs.append(p)
            else:
                # Turn 1-2 pairs can be balanced
                persona_name = p.get("persona_name") or p.get("persona", {}).get("name", "Unknown")
                if persona_name not in persona_turn1plus:
                    persona_turn1plus[persona_name] = []
                persona_turn1plus[persona_name].append(p)
    
    # Remove Turn 1+ pairs from rebalanced_prefs, we'll add them back after balancing
    rebalanced_prefs = [p for p in rebalanced_prefs if p.get("dialogue_turn", 0) == 0]
    
    # Always keep all Turn 3+ pairs
    rebalanced_prefs.extend(turn3plus_pairs)
    if len(turn3plus_pairs) > 0:
        print(f"  Preserved {len(turn3plus_pairs)} Turn 3+ pairs (not balanced)")
    
    if persona_turn1plus:
        sizes_list = [len(prefs) for prefs in persona_turn1plus.values()]
        if sizes_list and len(sizes_list) > 1:
            max_size = max(sizes_list)
            min_size = min(sizes_list)
            # Only balance if the ratio is more than 3:1 AND min_size > 0 (changed from 2:1 to 3:1 to preserve more data)
            # If min_size is 0, keep all Turn 1+ pairs (they are important for multi-turn learning)
            if min_size > 0 and max_size > min_size * 3:
                target_size = int(min_size * 2)  # Increased from 1.5 to 2 to preserve more data
                print(f"  Balancing Turn 1-2 data: max={max_size}, min={min_size}, target={target_size}")
                
                for persona_name, turn1plus_prefs in persona_turn1plus.items():
                    if len(turn1plus_prefs) > target_size:
                        # Deterministic downsampling
                        keep_prob = target_size / len(turn1plus_prefs)
                        sampled = [
                            p for p in turn1plus_prefs
                            if _stable_hash_to_unit_interval(
                                f"{persona_name}:{p.get('state', {}).get('id', '')}:{p.get('chosen_action', '')}:{p.get('dialogue_turn', 0)}",
                                seed
                            ) < keep_prob
                        ]
                        if len(sampled) < target_size:
                            sampled = sorted(turn1plus_prefs, key=lambda p: _stable_hash_to_unit_interval(
                                f"{persona_name}:{p.get('state', {}).get('id', '')}:{p.get('chosen_action', '')}:{p.get('dialogue_turn', 0)}",
                                seed
                            ))[:target_size]
                        print(f"  {persona_name}: Turn 1-2 downsampled from {len(turn1plus_prefs)} to {len(sampled)} pairs")
                        rebalanced_prefs.extend(sampled)
                    else:
                        rebalanced_prefs.extend(turn1plus_prefs)
            else:
                # No need to balance, keep all Turn 1-2 data
                # This is especially important when min_size=0 (some personas have no Turn 1+ data)
                for turn1plus_prefs in persona_turn1plus.values():
                    rebalanced_prefs.extend(turn1plus_prefs)
        else:
            # Only one persona has Turn 1-2 data, keep all
            for turn1plus_prefs in persona_turn1plus.values():
                rebalanced_prefs.extend(turn1plus_prefs)
    
    return rebalanced_prefs


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
        for traj_id, turns in trajectory_groups.items():
            scored_turns = compute_trajectory_level_rewards(turns, cfg)
            all_scored.extend(scored_turns)
        
        print(f"📊 Computed rewards for {len(all_scored)} turns across all trajectories")
        
        # Now group by (state_id, dialogue_turn, persona, prev_action) for preference pair generation
        # This ensures we compare actions at the same turn within the SAME persona
        # For Turn 1+, we also consider the previous action to enable multi-turn learning
        # Different personas should have different preferences!
        turn_groups = {}
        for t in all_scored:
            state_id = t["state"]["id"]
            dialogue_turn = t["state"]["dialogue_turn"]
            persona_name = t.get("persona", {}).get("name", "Unknown")
            
            # For Turn 1+, include prev_action in the key to enable multi-turn learning
            # Use prev_action from scored turn (already computed in compute_trajectory_level_rewards)
            # This ensures consistency and avoids duplicate logic
            prev_action = t.get("prev_action")  # Can be None for Turn 0
            
            # For Turn 0: group by (state_id, dialogue_turn, persona) - no prev_action needed
            # For Turn 1+: group by (state_id, dialogue_turn, persona, prev_action) to allow comparing
            # different trajectories at the same dialogue_turn with the same prev_action
            # This ensures we can generate pairs for Turn 1+ when multiple trajectories exist
            if dialogue_turn > 0:
                # For Turn 1+, include dialogue_turn to separate different turns
                # This allows comparing different trajectories at the same turn
                key = (state_id, dialogue_turn, persona_name, prev_action)
            else:
                # For Turn 0, don't include prev_action (it's always None) to allow comparing Clarify vs Execute
                key = (state_id, dialogue_turn, persona_name)
            if key not in turn_groups:
                turn_groups[key] = []
            turn_groups[key].append(t)
        
        # Special handling for Turn 3+ Execute: if a state has only 1 Execute turn,
        # allow grouping across states to enable pair generation
        # This is a special case for the automatically added Execute turns
        turn3plus_execute_groups = defaultdict(list)
        for key, g in turn_groups.items():
            if len(key) == 4:  # Turn 1+ format
                state_id, dialogue_turn, persona_name, prev_action = key
                if dialogue_turn >= 3:  # Turn 3 or later
                    # Check if this group has Execute actions
                    execute_turns = [t for t in g if t.get("action") == "Execute"]
                    if len(execute_turns) == 1 and len(g) == 1:  # Only 1 Execute turn in this state, and it's the only turn
                        # Group by (dialogue_turn, persona_name, prev_action) across states
                        cross_state_key = (dialogue_turn, persona_name, prev_action)
                        turn3plus_execute_groups[cross_state_key].extend(execute_turns)
        
        # Add cross-state groups to turn_groups if they have at least 2 turns
        for cross_state_key, execute_turns in turn3plus_execute_groups.items():
            if len(execute_turns) >= 2:
                # Use a special key format to indicate cross-state grouping
                dialogue_turn, persona_name, prev_action = cross_state_key
                # Use a special state_id to indicate cross-state grouping
                cross_state_group_key = (f"__CROSS_STATE__{dialogue_turn}", dialogue_turn, persona_name, prev_action)
                turn_groups[cross_state_group_key] = execute_turns
        
        print(f"📊 Regrouped into {len(turn_groups)} (state_id, dialogue_turn, persona, prev_action) groups for preference generation")
        
        # Generate preference pairs with persona-aware multi-turn logic
        for key, g in turn_groups.items():
            # Handle different key formats for Turn 0 vs Turn 1+
            # Turn 0: (state_id, dialogue_turn, persona_name) - 3 elements, dialogue_turn is int (0)
            # Turn 1+: (state_id, dialogue_turn, persona_name, prev_action) - 4 elements, dialogue_turn is int (>0)
            if len(key) == 3:
                # Turn 0 grouping: (state_id, dialogue_turn, persona_name)
                state_id, dialogue_turn, persona_name = key
                prev_action = None  # Turn 0 has no previous action
            elif len(key) == 4:
                # Turn 1+ grouping: (state_id, dialogue_turn, persona_name, prev_action)
                state_id, dialogue_turn, persona_name, prev_action = key
                # Handle cross-state grouping (special state_id format)
                if isinstance(state_id, str) and state_id.startswith("__CROSS_STATE__"):
                    # This is a cross-state group, pairs will be generated across states
                    # Use the first turn's state for preference pairs (for consistency)
                    if g:
                        state_id = g[0]["state"]["id"]
            else:
                # Fallback: try to parse as Turn 1+ format (old format for backward compatibility)
                if len(key) == 3 and not isinstance(key[1], int):
                    # Old Turn 1+ format: (state_id, persona_name, prev_action)
                    state_id, persona_name, prev_action = key
                    dialogue_turn = None  # Not available in old format
                else:
                    # Unknown format, skip
                    continue
            
            # Use standard method for all cases
            # Reward already considers prev_action in compute_trajectory_level_rewards
            group_prefs = build_prefs_from_group(g)
            
            prefs.extend(group_prefs)
    
    else:
        print("⚠️  Using STEP-LEVEL reward computation (legacy mode)")
        groups = group_by_state(trajs)
        print(f"📊 Grouped into {len(groups)} (state_id, dialogue_turn) groups")

        for (sid, dialogue_turn), g in groups.items():
            scored = compute_rewards_for_group(g, cfg)
            group_prefs = build_prefs_from_group(scored)
            prefs.extend(group_prefs)

    print(f"📊 Generated {len(prefs)} preference pairs")

    # Optional persona-aware rebalancing to avoid action collapse
    # Different personas have different ideal action ratios:
    # - Busy-Developer: high Execute ratio (~0.8-0.9)
    # - Experienced-Engineer: medium Execute ratio (~0.3-0.4 at Turn 0, higher at Turn 1+)
    # - Novice-Learner: low Execute ratio (~0.2-0.3)
    if args.no_rebalance:
        print(f"⏭️  Rebalancing disabled, using all {len(prefs)} generated pairs")
    elif target_execute_ratio is not None:
        before = len(prefs)
        prefs = rebalance_prefs_by_persona(prefs, target_execute_ratio, rebalance_seed)
        after = len(prefs)
        print(f"🔧 Rebalanced prefs by persona: {before} -> {after} (base Execute ratio={target_execute_ratio})")

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
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.trajectories)
    out_path = Path(args.out)
    cfg = RewardConfig(w_task=args.w_task, w_interrupt=args.w_interrupt)

    print("⚙️  Reward config:")
    print(f"  - w_task = {cfg.w_task}")
    print(f"  - w_interrupt = {cfg.w_interrupt}")

    target_ratio = args.target_execute_ratio
    if args.no_rebalance or not (0.0 < target_ratio < 1.0):
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

