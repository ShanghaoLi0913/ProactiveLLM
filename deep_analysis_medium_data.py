#!/usr/bin/env python3
"""
中等规模数据深度分析脚本
包括轨迹模式、reward分布、persona行为差异等
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import statistics

def analyze_trajectory_patterns(traj_groups):
    """分析轨迹模式"""
    print("=" * 70)
    print("📊 轨迹模式深度分析")
    print("=" * 70)
    
    # Action序列模式
    action_sequences = defaultdict(int)
    persona_sequences = defaultdict(lambda: defaultdict(int))
    
    # 对话模式统计
    dialogue_patterns = {
        "single_execute": 0,  # 单轮Execute
        "single_clarify": 0,  # 单轮Clarify（应该很少，因为会补Execute）
        "clarify_then_execute": 0,  # Clarify -> Execute
        "multiple_clarify_then_execute": 0,  # 多个Clarify -> Execute
        "execute_only": 0,  # 只有Execute（单轮或多轮）
    }
    
    persona_patterns = defaultdict(lambda: defaultdict(int))
    
    # Task completion统计
    completion_by_persona = defaultdict(lambda: {"completed": 0, "total": 0})
    completion_by_pattern = defaultdict(lambda: {"completed": 0, "total": 0})
    
    for traj_id, turns in traj_groups.items():
        persona_name = turns[0]["persona"]["name"]
        actions = [t.get("action") for t in turns]
        sequence = " -> ".join(actions)
        action_sequences[sequence] += 1
        persona_sequences[persona_name][sequence] += 1
        
        # 分类对话模式
        if len(actions) == 1:
            if actions[0] == "Execute":
                dialogue_patterns["single_execute"] += 1
                persona_patterns[persona_name]["single_execute"] += 1
            elif actions[0] == "Clarify":
                dialogue_patterns["single_clarify"] += 1
                persona_patterns[persona_name]["single_clarify"] += 1
        elif len(actions) == 2:
            if actions == ["Clarify", "Execute"]:
                dialogue_patterns["clarify_then_execute"] += 1
                persona_patterns[persona_name]["clarify_then_execute"] += 1
        elif len(actions) > 2:
            if actions[-1] == "Execute" and all(a == "Clarify" for a in actions[:-1]):
                dialogue_patterns["multiple_clarify_then_execute"] += 1
                persona_patterns[persona_name]["multiple_clarify_then_execute"] += 1
        
        # 检查是否有Execute
        if any(a == "Execute" for a in actions):
            dialogue_patterns["execute_only"] += 1
            persona_patterns[persona_name]["execute_only"] += 1
        
        # Task completion
        task_completed = any(t.get("task_completed", False) for t in turns)
        completion_by_persona[persona_name]["total"] += 1
        if task_completed:
            completion_by_persona[persona_name]["completed"] += 1
        
        # 按模式统计completion
        pattern_key = sequence
        completion_by_pattern[pattern_key]["total"] += 1
        if task_completed:
            completion_by_pattern[pattern_key]["completed"] += 1
    
    print("\n1. Action序列模式（Top 10）:")
    for seq, count in sorted(action_sequences.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(traj_groups) * 100
        print(f"   {seq}: {count} ({pct:.1f}%)")
    
    print("\n2. 对话模式分布:")
    for pattern, count in dialogue_patterns.items():
        pct = count / len(traj_groups) * 100
        print(f"   {pattern}: {count} ({pct:.1f}%)")
    
    print("\n3. 按Persona的对话模式:")
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_patterns:
            continue
        persona_total = len([turns for tid, turns in traj_groups.items() 
                             if turns[0]["persona"]["name"] == persona])
        print(f"\n   {persona}:")
        for pattern, count in persona_patterns[persona].items():
            pct = count / persona_total * 100 if persona_total > 0 else 0
            print(f"     {pattern}: {count} ({pct:.1f}%)")
    
    print("\n4. Task Completion统计:")
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in completion_by_persona:
            continue
        stats = completion_by_persona[persona]
        rate = stats["completed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"   {persona}: {stats['completed']}/{stats['total']} ({rate:.1f}%)")
    
    return action_sequences, dialogue_patterns

def analyze_user_reactions(traj_groups):
    """分析用户反应模式"""
    print("\n" + "=" * 70)
    print("📊 用户反应模式分析")
    print("=" * 70)
    
    persona_reactions = defaultdict(lambda: {
        "satisfaction": [],
        "answered_clarification": [],
        "reject_signal": [],
        "silence": [],
        "off_topic": [],
    })
    
    for traj_id, turns in traj_groups.items():
        persona_name = turns[0]["persona"]["name"]
        
        for turn in turns:
            reaction = turn.get("user_reaction", {})
            meta = reaction.get("meta", {})
            
            if "satisfaction" in meta:
                persona_reactions[persona_name]["satisfaction"].append(meta["satisfaction"])
            if "answered_clarification" in meta:
                persona_reactions[persona_name]["answered_clarification"].append(meta["answered_clarification"])
            if "reject_signal" in meta:
                persona_reactions[persona_name]["reject_signal"].append(meta["reject_signal"])
            if "silence" in meta:
                persona_reactions[persona_name]["silence"].append(meta["silence"])
            if "off_topic_flag" in meta:
                persona_reactions[persona_name]["off_topic"].append(meta["off_topic_flag"])
    
    print("\n按Persona的用户反应统计:")
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_reactions:
            continue
        reactions = persona_reactions[persona]
        print(f"\n   {persona}:")
        
        if reactions["satisfaction"]:
            avg_sat = statistics.mean(reactions["satisfaction"])
            print(f"     平均satisfaction: {avg_sat:.3f}")
        
        if reactions["answered_clarification"]:
            total_answered = sum(reactions["answered_clarification"])
            total_clarify = len([r for r in reactions["answered_clarification"] if r >= 0])
            print(f"     回答澄清问题: {total_answered}/{total_clarify} ({total_answered/max(total_clarify,1)*100:.1f}%)")
        
        if reactions["reject_signal"]:
            total_reject = sum(reactions["reject_signal"])
            print(f"     拒绝信号: {total_reject}次")
        
        if reactions["silence"]:
            total_silence = sum(reactions["silence"])
            print(f"     沉默: {total_silence}次")
        
        if reactions["off_topic"]:
            total_off_topic = sum(reactions["off_topic"])
            print(f"     离题: {total_off_topic}次")

def analyze_reward_distribution(prefs):
    """分析reward分布"""
    print("\n" + "=" * 70)
    print("📊 Reward分布深度分析")
    print("=" * 70)
    
    # 按persona分组
    persona_rewards = defaultdict(lambda: {
        "chosen_rewards": [],
        "rejected_rewards": [],
        "margins": [],
        "chosen_task_scores": [],
        "rejected_task_scores": [],
        "chosen_interrupt_costs": [],
        "rejected_interrupt_costs": [],
    })
    
    # 按action分组
    action_rewards = defaultdict(lambda: {
        "chosen": [],
        "rejected": [],
    })
    
    # 按turn分组
    turn_rewards = defaultdict(lambda: {
        "chosen_rewards": [],
        "rejected_rewards": [],
        "margins": [],
    })
    
    for pref in prefs:
        persona_name = pref.get("persona", {}).get("name", "Unknown")
        chosen_action = pref.get("chosen_action")
        rejected_action = pref.get("rejected_action")
        dialogue_turn = pref.get("state", {}).get("dialogue_turn", 0)
        
        chosen_reward = pref.get("chosen_reward", 0)
        rejected_reward = pref.get("rejected_reward", 0)
        margin = chosen_reward - rejected_reward
        
        persona_rewards[persona_name]["chosen_rewards"].append(chosen_reward)
        persona_rewards[persona_name]["rejected_rewards"].append(rejected_reward)
        persona_rewards[persona_name]["margins"].append(margin)
        persona_rewards[persona_name]["chosen_task_scores"].append(pref.get("chosen_task_score", 0))
        persona_rewards[persona_name]["rejected_task_scores"].append(pref.get("rejected_task_score", 0))
        persona_rewards[persona_name]["chosen_interrupt_costs"].append(pref.get("chosen_interrupt_cost", 0))
        persona_rewards[persona_name]["rejected_interrupt_costs"].append(pref.get("rejected_interrupt_cost", 0))
        
        action_rewards[chosen_action]["chosen"].append(chosen_reward)
        action_rewards[rejected_action]["rejected"].append(rejected_reward)
        
        turn_rewards[dialogue_turn]["chosen_rewards"].append(chosen_reward)
        turn_rewards[dialogue_turn]["rejected_rewards"].append(rejected_reward)
        turn_rewards[dialogue_turn]["margins"].append(margin)
    
    print("\n1. 按Persona的Reward统计:")
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_rewards:
            continue
        rewards = persona_rewards[persona]
        print(f"\n   {persona}:")
        print(f"     平均chosen reward: {statistics.mean(rewards['chosen_rewards']):.3f}")
        print(f"     平均rejected reward: {statistics.mean(rewards['rejected_rewards']):.3f}")
        print(f"     平均margin: {statistics.mean(rewards['margins']):.3f}")
        print(f"     平均chosen task_score: {statistics.mean(rewards['chosen_task_scores']):.3f}")
        print(f"     平均rejected task_score: {statistics.mean(rewards['rejected_task_scores']):.3f}")
        print(f"     平均chosen interrupt_cost: {statistics.mean(rewards['chosen_interrupt_costs']):.3f}")
        print(f"     平均rejected interrupt_cost: {statistics.mean(rewards['rejected_interrupt_costs']):.3f}")
    
    print("\n2. 按Action的Reward统计:")
    for action in ["Execute", "Clarify"]:
        if action not in action_rewards:
            continue
        rewards = action_rewards[action]
        if rewards["chosen"]:
            print(f"\n   {action} (as chosen):")
            print(f"     平均reward: {statistics.mean(rewards['chosen']):.3f}")
            print(f"     最小: {min(rewards['chosen']):.3f}")
            print(f"     最大: {max(rewards['chosen']):.3f}")
        if rewards["rejected"]:
            print(f"   {action} (as rejected):")
            print(f"     平均reward: {statistics.mean(rewards['rejected']):.3f}")
            print(f"     最小: {min(rewards['rejected']):.3f}")
            print(f"     最大: {max(rewards['rejected']):.3f}")
    
    print("\n3. 按Turn的Reward统计:")
    for turn in sorted(turn_rewards.keys()):
        rewards = turn_rewards[turn]
        print(f"\n   Turn {turn}:")
        print(f"     平均chosen reward: {statistics.mean(rewards['chosen_rewards']):.3f}")
        print(f"     平均rejected reward: {statistics.mean(rewards['rejected_rewards']):.3f}")
        print(f"     平均margin: {statistics.mean(rewards['margins']):.3f}")
        print(f"     Pairs数: {len(rewards['margins'])}")
    
    return persona_rewards, action_rewards, turn_rewards

def analyze_uncertainty_action_relationship(prefs):
    """分析task_uncertainty与action选择的关系"""
    print("\n" + "=" * 70)
    print("📊 Task Uncertainty与Action选择关系分析")
    print("=" * 70)
    
    # 按uncertainty范围分组
    uncertainty_ranges = {
        "low (0.0-0.3)": [],
        "medium (0.3-0.7)": [],
        "high (0.7-1.0)": [],
    }
    
    persona_uncertainty = defaultdict(lambda: {
        "chosen_clarify": [],
        "chosen_execute": [],
    })
    
    for pref in prefs:
        uncertainty = pref.get("state", {}).get("task_uncertainty", 0.0)
        chosen_action = pref.get("chosen_action")
        
        if uncertainty < 0.3:
            key = "low (0.0-0.3)"
        elif uncertainty < 0.7:
            key = "medium (0.3-0.7)"
        else:
            key = "high (0.7-1.0)"
        
        uncertainty_ranges[key].append((uncertainty, chosen_action))
        
        persona_name = pref.get("persona", {}).get("name", "Unknown")
        if chosen_action == "Clarify":
            persona_uncertainty[persona_name]["chosen_clarify"].append(uncertainty)
        elif chosen_action == "Execute":
            persona_uncertainty[persona_name]["chosen_execute"].append(uncertainty)
    
    print("\n1. 按Uncertainty范围的Action选择:")
    for range_key, pairs in uncertainty_ranges.items():
        if not pairs:
            continue
        clarify_count = sum(1 for _, action in pairs if action == "Clarify")
        execute_count = sum(1 for _, action in pairs if action == "Execute")
        total = len(pairs)
        print(f"\n   {range_key} ({total} pairs):")
        print(f"     Clarify: {clarify_count} ({clarify_count/total*100:.1f}%)")
        print(f"     Execute: {execute_count} ({execute_count/total*100:.1f}%)")
    
    print("\n2. 按Persona的Uncertainty分布:")
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_uncertainty:
            continue
        stats = persona_uncertainty[persona]
        print(f"\n   {persona}:")
        if stats["chosen_clarify"]:
            avg_unc_clarify = statistics.mean(stats["chosen_clarify"])
            print(f"     选择Clarify时的平均uncertainty: {avg_unc_clarify:.3f}")
        if stats["chosen_execute"]:
            avg_unc_execute = statistics.mean(stats["chosen_execute"])
            print(f"     选择Execute时的平均uncertainty: {avg_unc_execute:.3f}")

def analyze_code_quality(traj_groups):
    """分析代码质量"""
    print("\n" + "=" * 70)
    print("📊 代码质量分析")
    print("=" * 70)
    
    execute_turns = []
    for traj_id, turns in traj_groups.items():
        for turn in turns:
            if turn.get("action") == "Execute":
                execute_turns.append(turn)
    
    if not execute_turns:
        print("\n   未找到Execute turns")
        return
    
    code_lengths = []
    has_original_query = 0
    knowledge_distillation_count = 0
    
    for turn in execute_turns:
        assistant_msg = turn.get("assistant_msg", "")
        state = turn.get("state", {})
        
        # 估算代码长度（提取代码块）
        if "```" in assistant_msg:
            code_blocks = assistant_msg.split("```")
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    code = code_blocks[i]
                    # 跳过语言标识符
                    lines = code.split("\n")
                    if lines and lines[0].strip() in ["python", "Python", "py"]:
                        code = "\n".join(lines[1:])
                    code_lengths.append(len(code))
        
        # 检查是否使用了original_instruct_prompt（知识蒸馏）
        if state.get("original_instruct_prompt"):
            has_original_query += 1
            # 检查assistant_msg是否包含original query的内容（间接验证）
            # 这里我们假设如果state有original_instruct_prompt，那么Execute应该使用了它
            knowledge_distillation_count += 1
    
    print(f"\n1. 代码统计:")
    print(f"   Execute turns总数: {len(execute_turns)}")
    print(f"   有代码块的turns: {len(code_lengths)}")
    if code_lengths:
        print(f"   平均代码长度: {statistics.mean(code_lengths):.0f} 字符")
        print(f"   最小代码长度: {min(code_lengths)} 字符")
        print(f"   最大代码长度: {max(code_lengths)} 字符")
    
    print(f"\n2. 知识蒸馏验证:")
    print(f"   有original_instruct_prompt的Execute turns: {has_original_query}/{len(execute_turns)} ({has_original_query/len(execute_turns)*100:.1f}%)")
    print(f"   知识蒸馏覆盖率: {knowledge_distillation_count}/{len(execute_turns)} ({knowledge_distillation_count/len(execute_turns)*100:.1f}%)")

def main():
    # 找到最新的文件
    log_dir = Path("data/logs")
    dpo_dir = Path("data/dpo")
    
    traj_files = list(log_dir.glob("medium_traj_*.jsonl"))
    prefs_files = list(dpo_dir.glob("medium_traj_*_prefs.jsonl"))
    
    if not traj_files:
        print("❌ 未找到轨迹文件")
        return
    
    traj_file = sorted(traj_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"📁 分析轨迹文件: {traj_file.name}\n")
    
    # 加载轨迹数据
    with open(traj_file) as f:
        trajs = [json.loads(line) for line in f]
    
    # 按trajectory_id分组
    traj_groups = defaultdict(list)
    for t in trajs:
        traj_id = t.get("trajectory_id", "unknown")
        traj_groups[traj_id].append(t)
    
    # 执行各项分析
    analyze_trajectory_patterns(traj_groups)
    analyze_user_reactions(traj_groups)
    analyze_code_quality(traj_groups)
    
    if prefs_files:
        prefs_file = sorted(prefs_files, key=lambda p: p.stat().st_mtime)[-1]
        print(f"\n📁 分析Preference文件: {prefs_file.name}\n")
        
        with open(prefs_file) as f:
            prefs = [json.loads(line) for line in f]
        
        analyze_reward_distribution(prefs)
        analyze_uncertainty_action_relationship(prefs)
    else:
        print("\n⏳ Preference文件尚未生成")
    
    print("\n" + "=" * 70)
    print("✅ 深度分析完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
