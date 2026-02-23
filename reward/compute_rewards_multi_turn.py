"""
支持多轮行为学习的Preference Pair生成

关键改进：
1. 生成Turn 1+的preference pairs，考虑Turn 0的action
2. 在state中包含对话历史信息
3. 根据persona和uncertainty选择chosen/rejected
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from reward.compute_rewards import (
    load_trajectories,
    group_by_trajectory,
    compute_trajectory_level_rewards,
    build_prefs_from_group,
    RewardConfig,
)


def build_multi_turn_prefs(
    trajectory_groups: Dict[str, List[Dict]],
    cfg: RewardConfig,
) -> List[Dict]:
    """
    生成支持多轮行为学习的preference pairs。
    
    策略：
    1. 对于Turn 0: 按(state_id, dialogue_turn, persona)分组，比较Clarify vs Execute
    2. 对于Turn 1+: 按(state_id, dialogue_turn, persona, prev_action)分组
       - prev_action: Turn 0的action
       - 比较Turn 1的Clarify vs Execute
       - 但需要考虑Turn 0的action和persona特性
    """
    prefs = []
    
    # Step 1: 计算所有turns的reward
    all_scored = []
    for traj_id, turns in trajectory_groups.items():
        scored_turns = compute_trajectory_level_rewards(turns, cfg)
        all_scored.extend(scored_turns)
    
    # Step 2: 按turn分组
    turn_groups = defaultdict(lambda: defaultdict(list))
    
    for t in all_scored:
        state_id = t["state"]["id"]
        dialogue_turn = t["state"]["dialogue_turn"]
        persona_name = t.get("persona", {}).get("name", "Unknown")
        
        # 对于Turn 0，不需要prev_action
        if dialogue_turn == 0:
            key = (state_id, dialogue_turn, persona_name, None)
        else:
            # 对于Turn 1+，需要找到Turn 0的action
            # 从同一个trajectory中找到Turn 0的action
            traj_id = t.get("trajectory_id")
            prev_action = None
            if traj_id:
                traj_turns = trajectory_groups.get(traj_id, [])
                for turn in traj_turns:
                    if turn.get("state", {}).get("dialogue_turn", 0) == 0:
                        prev_action = turn.get("action")
                        break
            
            key = (state_id, dialogue_turn, persona_name, prev_action)
        
        turn_groups[dialogue_turn][key].append(t)
    
    # Step 3: 生成preference pairs
    for dialogue_turn, groups in turn_groups.items():
        for key, group in groups.items():
            # 对于Turn 1+，应用persona-aware的preference选择
            if dialogue_turn > 0:
                group_prefs = build_multi_turn_prefs_for_group(
                    group, key, dialogue_turn, cfg
                )
            else:
                # Turn 0: 使用标准方法
                group_prefs = build_prefs_from_group(group)
            
            prefs.extend(group_prefs)
    
    return prefs


def build_multi_turn_prefs_for_group(
    scored_trajs: List[Dict],
    key: Tuple,
    dialogue_turn: int,
    cfg: RewardConfig,
) -> List[Dict]:
    """
    为Turn 1+生成persona-aware的preference pairs。
    
    关键逻辑：
    - 如果Turn 0是Clarify，Turn 1应该根据persona选择：
      * Novice-Learner: 倾向于继续Clarify（如果uncertainty仍然高）
      * Experienced-Engineer: 倾向于Execute（快速执行）
      * Busy-Developer: 倾向于Execute（避免多轮）
    """
    if len(scored_trajs) < 2:
        return []
    
    state_id, turn, persona_name, prev_action = key
    
    # 获取task_uncertainty
    task_uncertainty = scored_trajs[0].get("state", {}).get("task_uncertainty", 0.5)
    
    # Persona-aware preference selection
    if prev_action == "Clarify":
        # Turn 0是Clarify，Turn 1的选择应该根据persona
        if persona_name == "Novice-Learner":
            # Novice-Learner: 如果uncertainty仍然高，倾向于继续Clarify
            if task_uncertainty > 0.5:
                # 偏好Clarify
                preferred_action = "Clarify"
            else:
                # Uncertainty降低了，可以Execute
                preferred_action = "Execute"
        elif persona_name == "Experienced-Engineer":
            # Experienced-Engineer: 倾向于Execute（快速执行）
            preferred_action = "Execute"
        elif persona_name == "Busy-Developer":
            # Busy-Developer: 强烈倾向于Execute（避免多轮）
            preferred_action = "Execute"
        else:
            # 默认：根据reward选择
            preferred_action = None
    else:
        # Turn 0是Execute（不应该发生，因为Execute会结束对话）
        # 但为了安全，使用标准方法
        preferred_action = None
    
    # 如果有preferred_action，调整reward
    if preferred_action:
        for t in scored_trajs:
            if t.get("action") == preferred_action:
                # 给preferred action一个小的bonus
                t["total_reward"] = float(t["total_reward"]) + 0.1
    
    # 使用标准方法生成preference pairs
    return build_prefs_from_group(scored_trajs)


def add_conversation_history_to_state(
    state: Dict,
    trajectory_turns: List[Dict],
    current_turn: int,
) -> Dict:
    """
    在state中添加对话历史信息，帮助模型理解多轮上下文。
    """
    state_with_history = state.copy()
    
    # 添加对话历史
    history = []
    for i in range(current_turn):
        if i < len(trajectory_turns):
            turn = trajectory_turns[i]
            action = turn.get("action", "")
            assistant_msg = turn.get("assistant_msg", "")[:200]  # 截断
            user_reply = turn.get("user_reaction", {}).get("user_reply", "")[:100]
            
            history.append({
                "turn": i,
                "action": action,
                "assistant_msg": assistant_msg,
                "user_reply": user_reply,
            })
    
    state_with_history["conversation_history"] = history
    
    # 添加Turn 0的action信息
    if current_turn > 0 and len(trajectory_turns) > 0:
        turn0_action = trajectory_turns[0].get("action", "")
        state_with_history["prev_action"] = turn0_action
    
    return state_with_history


def compute_preferences_multi_turn(
    traj_path: Path,
    out_path: Path,
    cfg: RewardConfig,
    target_execute_ratio: float = None,
    rebalance_seed: int = 42,
) -> None:
    """
    生成支持多轮行为学习的preference pairs。
    """
    print(f"📂 Loading trajectories from: {traj_path}")
    trajs = load_trajectories(traj_path)
    print(f"📊 Loaded {len(trajs)} trajectory turns")
    
    # Group by trajectory
    trajectory_groups = group_by_trajectory(trajs)
    print(f"📊 Grouped into {len(trajectory_groups)} complete trajectories")
    
    # 为Turn 1+的turns添加对话历史
    for traj_id, turns in trajectory_groups.items():
        for i, turn in enumerate(turns):
            if i > 0:  # Turn 1+
                turn["state"] = add_conversation_history_to_state(
                    turn["state"],
                    turns,
                    i,
                )
    
    # 生成multi-turn preference pairs
    print("🎯 Generating multi-turn preference pairs...")
    prefs = build_multi_turn_prefs(trajectory_groups, cfg)
    print(f"📊 Generated {len(prefs)} preference pairs")
    
    # Optional rebalancing
    if target_execute_ratio is not None:
        from reward.compute_rewards import rebalance_prefs_by_action
        before = len(prefs)
        prefs = rebalance_prefs_by_action(prefs, target_execute_ratio, rebalance_seed)
        print(f"📊 Rebalanced: {before} → {len(prefs)} pairs")
    
    # Save
    print(f"💾 Saving to: {out_path}")
    with open(out_path, "w") as f:
        for pref in prefs:
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")
    
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--w_task", type=float, default=1.0)
    parser.add_argument("--w_interrupt", type=float, default=0.3)
    parser.add_argument("--target_execute_ratio", type=float, default=None)
    parser.add_argument("--rebalance_seed", type=int, default=42)
    
    args = parser.parse_args()
    
    cfg = RewardConfig(
        w_task=args.w_task,
        w_interrupt=args.w_interrupt,
    )
    
    compute_preferences_multi_turn(
        traj_path=Path(args.traj_path),
        out_path=Path(args.out_path),
        cfg=cfg,
        target_execute_ratio=args.target_execute_ratio,
        rebalance_seed=args.rebalance_seed,
    )
