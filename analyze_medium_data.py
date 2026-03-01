#!/usr/bin/env python3
"""
中等规模数据系统性分析脚本
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

def analyze_trajectories(traj_file):
    """分析轨迹数据"""
    print("=" * 70)
    print("📊 轨迹数据系统性分析")
    print("=" * 70)
    
    with open(traj_file) as f:
        trajs = [json.loads(line) for line in f]
    
    print(f"\n1. 基本统计:")
    print(f"   - 总trajectory turns: {len(trajs)}")
    
    # 按trajectory_id分组
    traj_groups = defaultdict(list)
    for t in trajs:
        traj_id = t.get("trajectory_id", "unknown")
        traj_groups[traj_id].append(t)
    
    print(f"   - 总trajectories: {len(traj_groups)}")
    print(f"   - 平均轮次: {len(trajs) / len(traj_groups):.2f}")
    
    # 按persona统计
    print(f"\n2. Persona统计:")
    persona_stats = defaultdict(lambda: {
        "trajs": [], "total_turns": 0, "execute_turns": 0, 
        "clarify_turns": 0, "has_code": 0, "has_original": 0
    })
    
    for traj_id, turns in traj_groups.items():
        persona_name = turns[0]["persona"]["name"]
        traj_length = len(turns)
        execute_count = sum(1 for t in turns if t.get("action") == "Execute")
        clarify_count = sum(1 for t in turns if t.get("action") == "Clarify")
        
        # 检查代码和original_instruct_prompt
        has_code = any("```" in t.get("assistant_msg", "") or "def " in t.get("assistant_msg", "") 
                      for t in turns if t.get("action") == "Execute")
        has_original = any(t.get("state", {}).get("original_instruct_prompt") for t in turns)
        
        persona_stats[persona_name]["trajs"].append(traj_length)
        persona_stats[persona_name]["total_turns"] += traj_length
        persona_stats[persona_name]["execute_turns"] += execute_count
        persona_stats[persona_name]["clarify_turns"] += clarify_count
        if has_code:
            persona_stats[persona_name]["has_code"] += 1
        if has_original:
            persona_stats[persona_name]["has_original"] += 1
    
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_stats:
            continue
        stats = persona_stats[persona]
        n_trajs = len(stats["trajs"])
        avg_length = sum(stats["trajs"]) / n_trajs if n_trajs > 0 else 0
        
        print(f"\n   {persona}:")
        print(f"     - 轨迹数: {n_trajs}")
        print(f"     - 平均轮次: {avg_length:.2f}")
        print(f"     - Execute turns: {stats['execute_turns']}")
        print(f"     - Clarify turns: {stats['clarify_turns']}")
        print(f"     - Execute有代码: {stats['has_code']}/{stats['execute_turns']} ({stats['has_code']/max(stats['execute_turns'],1)*100:.1f}%)")
        print(f"     - 有original_instruct_prompt: {stats['has_original']}/{n_trajs} ({stats['has_original']/n_trajs*100:.1f}%)")
    
    # Action分布
    print(f"\n3. Action分布:")
    actions = [t.get("action") for t in trajs]
    for action, count in Counter(actions).items():
        print(f"   - {action}: {count} ({count/len(trajs)*100:.1f}%)")
    
    # 轮次分布
    print(f"\n4. 轮次分布:")
    lengths = [len(turns) for turns in traj_groups.values()]
    length_dist = Counter(lengths)
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        print(f"   - {length}轮: {count}个 ({count/len(traj_groups)*100:.1f}%)")
    
    # 检查补Execute
    print(f"\n5. 补Execute统计:")
    added_execute = sum(1 for tid, turns in traj_groups.items() 
                       if len(turns) > 1 and 
                       turns[-1].get("action") == "Execute" and
                       not any(t.get("action") == "Execute" for t in turns[:-1]))
    print(f"   - 补Execute的trajectories: {added_execute}/{len(traj_groups)} ({added_execute/len(traj_groups)*100:.1f}%)")
    
    return trajs, traj_groups

def analyze_preferences(prefs_file):
    """分析preference pairs"""
    print("\n" + "=" * 70)
    print("📊 Preference Pairs系统性分析")
    print("=" * 70)
    
    with open(prefs_file) as f:
        prefs = [json.loads(line) for line in f]
    
    print(f"\n1. 基本统计:")
    print(f"   - 总pairs: {len(prefs)}")
    
    # 按persona统计
    print(f"\n2. Persona统计:")
    persona_prefs = defaultdict(list)
    for pref in prefs:
        persona_name = pref.get("persona", {}).get("name", "Unknown")
        persona_prefs[persona_name].append(pref)
    
    for persona in ["Busy-Developer", "Experienced-Engineer", "Novice-Learner"]:
        if persona not in persona_prefs:
            continue
        prefs_list = persona_prefs[persona]
        n_pairs = len(prefs_list)
        
        chosen_clarify = sum(1 for p in prefs_list if p["chosen_action"] == "Clarify")
        chosen_execute = sum(1 for p in prefs_list if p["chosen_action"] == "Execute")
        
        avg_margin = sum(p["chosen_reward"] - p["rejected_reward"] for p in prefs_list) / n_pairs
        avg_uncertainty = sum(p["state"].get("task_uncertainty", 0) for p in prefs_list) / n_pairs
        
        print(f"\n   {persona}:")
        print(f"     - Pairs数: {n_pairs}")
        print(f"     - Chosen Clarify: {chosen_clarify} ({chosen_clarify/n_pairs*100:.1f}%)")
        print(f"     - Chosen Execute: {chosen_execute} ({chosen_execute/n_pairs*100:.1f}%)")
        print(f"     - 平均reward margin: {avg_margin:.3f}")
        print(f"     - 平均task_uncertainty: {avg_uncertainty:.2f}")
    
    # Action分布
    print(f"\n3. Action分布:")
    chosen_actions = [p["chosen_action"] for p in prefs]
    for action, count in Counter(chosen_actions).items():
        print(f"   - Chosen {action}: {count} ({count/len(prefs)*100:.1f}%)")
    
    # Reward margin
    margins = [p["chosen_reward"] - p["rejected_reward"] for p in prefs]
    print(f"\n4. Reward Margin:")
    print(f"   - 平均: {sum(margins)/len(margins):.3f}")
    print(f"   - 最小: {min(margins):.3f}")
    print(f"   - 最大: {max(margins):.3f}")
    
    # 数据质量
    print(f"\n5. 数据质量:")
    has_tests = sum(1 for p in prefs if p.get("state", {}).get("convcodeworld_tests") or p.get("state", {}).get("test"))
    has_original = sum(1 for p in prefs if p.get("state", {}).get("original_instruct_prompt"))
    print(f"   - 有测试用例: {has_tests}/{len(prefs)} ({has_tests/len(prefs)*100:.1f}%)")
    print(f"   - 有original_instruct_prompt: {has_original}/{len(prefs)} ({has_original/len(prefs)*100:.1f}%)")
    
    return prefs

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
    
    trajs, traj_groups = analyze_trajectories(traj_file)
    
    if prefs_files:
        prefs_file = sorted(prefs_files, key=lambda p: p.stat().st_mtime)[-1]
        print(f"\n📁 分析Preference文件: {prefs_file.name}\n")
        prefs = analyze_preferences(prefs_file)
    else:
        print("\n⏳ Preference文件尚未生成")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
