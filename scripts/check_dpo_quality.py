#!/usr/bin/env python3
"""
检查轨迹数据是否符合DPO训练的理想比例要求。

检查清单：
1. Busy角色下Clarify被Rejected的比例 > 80%
2. Novice角色下Execute失败的比例 > 60%
3. Turn 3或更长的轨迹比例 10% - 20%
4. Reward分差 > 0.5的Pair比例 > 70%
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_trajectories(path: Path) -> List[Dict]:
    """Load trajectories JSONL."""
    trajs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajs.append(json.loads(line))
    return trajs


def check_dpo_quality(traj_path: Path) -> Dict:
    """检查轨迹数据是否符合DPO训练的理想比例要求。"""
    
    trajectories = load_trajectories(traj_path)
    
    results = {
        "total_tasks": len(trajectories),
        "checks": {}
    }
    
    # 按对话分组（仅使用Turn 1作为对话起点，避免错误折叠）
    conversations = defaultdict(list)
    turn1_trajs = [t for t in trajectories if t.get('turn') == 1]
    for t in turn1_trajs:
        state_id = t.get('state', {}).get('id', 'unknown')
        persona_name = t.get('persona', {}).get('name', 'unknown')
        first_action = t.get('action', 'unknown')
        conv_key = (state_id, persona_name, first_action)
        conversations[conv_key].append(t)
    
    # 将后续turn尽量归并到Clarify分支（更符合多轮对话的实际路径）
    for traj in trajectories:
        if traj.get('turn', 0) <= 1:
            continue
        state_id = traj.get('state', {}).get('id', 'unknown')
        persona_name = traj.get('persona', {}).get('name', 'unknown')
        clarify_key = (state_id, persona_name, "Clarify")
        execute_key = (state_id, persona_name, "Execute")
        if clarify_key in conversations:
            conversations[clarify_key].append(traj)
        elif execute_key in conversations:
            conversations[execute_key].append(traj)
        else:
            # Fallback if no turn1 found
            conversations[(state_id, persona_name, "unknown")].append(traj)
    
    results["total_conversations"] = len(conversations)
    
    # 检查1: Busy角色下Clarify被Rejected的比例
    busy_clarify_total = 0
    busy_clarify_rejected = 0
    for conv_key, trajs in conversations.items():
        state_id, persona_name, first_action = conv_key
        if persona_name == "Busy-Developer" and first_action == "Clarify":
            busy_clarify_total += 1
            # 检查是否被拒绝
            first_traj = [t for t in trajs if t.get('turn') == 1][0]
            if first_traj.get('user_stopped', False) or \
               first_traj.get('user_reaction', {}).get('meta', {}).get('reject_signal', 0) > 0:
                busy_clarify_rejected += 1
    
    busy_reject_rate = busy_clarify_rejected / busy_clarify_total * 100 if busy_clarify_total > 0 else 0
    results["checks"]["busy_clarify_rejected"] = {
        "rate": busy_reject_rate,
        "count": busy_clarify_rejected,
        "total": busy_clarify_total,
        "passed": busy_reject_rate > 80,
        "target": "> 80%"
    }
    
    # 检查2: Novice角色下Execute失败的比例
    novice_execute_total = 0
    novice_execute_failed = 0
    for conv_key, trajs in conversations.items():
        state_id, persona_name, first_action = conv_key
        if persona_name == "Novice-Learner" and first_action == "Execute":
            novice_execute_total += 1
            # 检查是否失败（没有edge_cases_info或task_completed=False）
            first_traj = [t for t in trajs if t.get('turn') == 1][0]
            has_edge_info = first_traj.get('has_edge_cases_info', False)
            task_completed = first_traj.get('task_completed', False)
            if not has_edge_info or not task_completed:
                novice_execute_failed += 1
    
    novice_fail_rate = novice_execute_failed / novice_execute_total * 100 if novice_execute_total > 0 else 0
    results["checks"]["novice_execute_failed"] = {
        "rate": novice_fail_rate,
        "count": novice_execute_failed,
        "total": novice_execute_total,
        "passed": novice_fail_rate > 60,
        "target": "> 60%"
    }
    
    # 检查3: Turn 3或更长的轨迹比例
    conversation_turns = []
    for conv_key, trajs in conversations.items():
        max_turn = max(t.get('turn', 1) for t in trajs) if trajs else 1
        conversation_turns.append(max_turn)
    
    long_convs = [t for t in conversation_turns if t >= 3]
    long_conv_rate = len(long_convs) / len(conversation_turns) * 100 if conversation_turns else 0
    results["checks"]["turn3_or_longer"] = {
        "rate": long_conv_rate,
        "count": len(long_convs),
        "total": len(conversation_turns),
        "passed": 10 <= long_conv_rate <= 20,
        "target": "10% - 20%"
    }
    
    # 检查4: Reward分差 > 0.5的Pair比例（需要先计算reward）
    # 这个检查需要在compute_rewards之后进行，这里先跳过
    results["checks"]["reward_margin"] = {
        "note": "需要先运行compute_rewards.py来计算reward",
        "target": "> 70% pairs with margin > 0.5"
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check DPO training data quality")
    parser.add_argument("--trajectories", type=str, required=True,
                       help="Path to trajectories JSONL file")
    args = parser.parse_args()
    
    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        print(f"Error: {traj_path} does not exist")
        return
    
    results = check_dpo_quality(traj_path)
    
    print("=" * 80)
    print("DPO训练数据质量检查")
    print("=" * 80)
    print(f"\n总轨迹数: {results['total_tasks']}")
    print(f"总对话数: {results['total_conversations']}")
    
    print("\n" + "=" * 80)
    print("检查结果:")
    print("=" * 80)
    
    for check_name, check_result in results["checks"].items():
        if check_name == "reward_margin":
            print(f"\n{check_name}:")
            print(f"  {check_result['note']}")
            print(f"  目标: {check_result['target']}")
            continue
            
        status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
        print(f"\n{check_name}: {status}")
        print(f"  当前: {check_result['rate']:.1f}% ({check_result['count']}/{check_result['total']})")
        print(f"  目标: {check_result['target']}")
    
    # 总结
    passed_checks = sum(1 for k, v in results["checks"].items() 
                       if k != "reward_margin" and v.get("passed", False))
    total_checks = sum(1 for k in results["checks"] if k != "reward_margin")
    
    print("\n" + "=" * 80)
    print(f"总结: {passed_checks}/{total_checks} 项检查通过")
    if passed_checks == total_checks:
        print("✅ 所有检查通过！数据质量符合DPO训练要求。")
    else:
        print("⚠️  部分检查未通过，建议调整生成参数。")
    print("=" * 80)


if __name__ == "__main__":
    main()

