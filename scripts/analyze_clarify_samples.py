#!/usr/bin/env python3
"""
分析训练数据中的Clarify样本，找出"Clarify后成功"的正样本
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

def load_jsonl(path: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_trajectories(path: Path) -> Dict[str, Dict]:
    """加载轨迹数据，建立state_id到完整轨迹的映射"""
    trajectories = load_jsonl(path)
    traj_map = {}
    for traj in trajectories:
        for turn in traj.get("turns", []):
            state_id = turn.get("state_id")
            if state_id:
                traj_map[state_id] = {
                    "trajectory": traj,
                    "turn": turn,
                    "task_id": traj.get("task_id"),
                    "persona": traj.get("persona", "Unknown"),
                }
    return traj_map

def analyze_preference_pairs(prefs_path: Path, traj_map: Dict) -> Dict:
    """分析preference pairs中的Clarify样本"""
    prefs = load_jsonl(prefs_path)
    
    stats = {
        "total_pairs": len(prefs),
        "chosen_execute": 0,
        "chosen_clarify": 0,
        "clarify_success": [],  # Clarify后最终成功的案例
        "clarify_partial": [],  # Clarify后部分成功的案例
        "clarify_fail": [],     # Clarify后失败的案例
        "persona_action_distribution": defaultdict(lambda: {"Execute": 0, "Clarify": 0}),
        "persona_clarify_success": defaultdict(int),
    }
    
    for pref in prefs:
        chosen_action = pref.get("chosen_action")
        state_id = pref.get("state_id")
        chosen_score = pref.get("chosen_task_score", 0)
        
        # 获取persona信息
        persona = "Unknown"
        if state_id and state_id in traj_map:
            persona = traj_map[state_id]["persona"]
        
        # 统计action分布
        if chosen_action == "Execute":
            stats["chosen_execute"] += 1
            stats["persona_action_distribution"][persona]["Execute"] += 1
        elif chosen_action == "Clarify":
            stats["chosen_clarify"] += 1
            stats["persona_action_distribution"][persona]["Clarify"] += 1
            
            # 分析Clarify的结果
            sample = {
                "state_id": state_id,
                "persona": persona,
                "task_score": chosen_score,
                "chosen_prompt": pref.get("chosen_prompt", "")[:200],
                "chosen_response": pref.get("chosen_response", "")[:200],
            }
            
            if chosen_score >= 1.0:
                stats["clarify_success"].append(sample)
                stats["persona_clarify_success"][persona] += 1
            elif chosen_score > 0:
                stats["clarify_partial"].append(sample)
            else:
                stats["clarify_fail"].append(sample)
    
    return stats

def main():
    # 路径设置
    base_dir = Path("/root/autodl-tmp/ProactiveLLM")
    
    # 100 states训练数据
    traj_path = base_dir / "data/logs/traj_bigcode_100states_20260206_050454.jsonl"
    prefs_v3_path = base_dir / "data/dpo/prefs_bigcode_100_filtered.jsonl"
    prefs_v4_path = base_dir / "data/dpo/prefs_bigcode_100_repaired.jsonl"
    
    print("="*80)
    print("🔍 分析训练数据中的Clarify样本")
    print("="*80)
    
    # 加载轨迹数据
    print("\n📂 加载轨迹数据...")
    traj_map = load_trajectories(traj_path)
    print(f"   加载了 {len(traj_map)} 个state的轨迹信息")
    
    # 分析V3数据
    print("\n" + "="*80)
    print("📊 V3数据分析（允许部分通过，304对）")
    print("="*80)
    stats_v3 = analyze_preference_pairs(prefs_v3_path, traj_map)
    
    print(f"\n总样本数: {stats_v3['total_pairs']}")
    print(f"  • Chosen = Execute: {stats_v3['chosen_execute']} ({stats_v3['chosen_execute']/stats_v3['total_pairs']*100:.1f}%)")
    print(f"  • Chosen = Clarify: {stats_v3['chosen_clarify']} ({stats_v3['chosen_clarify']/stats_v3['total_pairs']*100:.1f}%)")
    
    print(f"\nClarify样本详情:")
    print(f"  • Clarify后完全成功（score=1.0）: {len(stats_v3['clarify_success'])} 个 ⭐")
    print(f"  • Clarify后部分成功（0<score<1.0）: {len(stats_v3['clarify_partial'])} 个")
    print(f"  • Clarify后失败（score=0）: {len(stats_v3['clarify_fail'])} 个")
    
    print(f"\n按Persona统计Action分布:")
    for persona, actions in sorted(stats_v3['persona_action_distribution'].items()):
        total = actions['Execute'] + actions['Clarify']
        print(f"  {persona}:")
        print(f"    Execute: {actions['Execute']}/{total} ({actions['Execute']/total*100:.1f}%)")
        print(f"    Clarify: {actions['Clarify']}/{total} ({actions['Clarify']/total*100:.1f}%)")
        if persona in stats_v3['persona_clarify_success']:
            print(f"    Clarify成功: {stats_v3['persona_clarify_success'][persona]} 个")
    
    # 分析V4数据
    print("\n" + "="*80)
    print("📊 V4数据分析（完美数据+修复，135对）")
    print("="*80)
    stats_v4 = analyze_preference_pairs(prefs_v4_path, traj_map)
    
    print(f"\n总样本数: {stats_v4['total_pairs']}")
    print(f"  • Chosen = Execute: {stats_v4['chosen_execute']} ({stats_v4['chosen_execute']/stats_v4['total_pairs']*100:.1f}%)")
    print(f"  • Chosen = Clarify: {stats_v4['chosen_clarify']} ({stats_v4['chosen_clarify']/stats_v4['total_pairs']*100:.1f}%)")
    
    print(f"\nClarify样本详情:")
    print(f"  • Clarify后完全成功（score=1.0）: {len(stats_v4['clarify_success'])} 个 ⭐")
    print(f"  • Clarify后部分成功（0<score<1.0）: {len(stats_v4['clarify_partial'])} 个")
    print(f"  • Clarify后失败（score=0）: {len(stats_v4['clarify_fail'])} 个")
    
    # 关键发现
    print("\n" + "="*80)
    print("🎯 关键发现")
    print("="*80)
    
    v3_clarify_rate = stats_v3['chosen_clarify'] / stats_v3['total_pairs'] * 100
    v4_clarify_rate = stats_v4['chosen_clarify'] / stats_v4['total_pairs'] * 100
    
    print(f"\n1. Clarify样本比例:")
    print(f"   V3: {stats_v3['chosen_clarify']}/{stats_v3['total_pairs']} = {v3_clarify_rate:.1f}%")
    print(f"   V4: {stats_v4['chosen_clarify']}/{stats_v4['total_pairs']} = {v4_clarify_rate:.1f}%")
    
    print(f"\n2. 可用的高质量Clarify样本:")
    print(f"   V3中的完美Clarify: {len(stats_v3['clarify_success'])} 个 ⭐")
    print(f"   V4中的完美Clarify: {len(stats_v4['clarify_success'])} 个 ⭐")
    
    if len(stats_v3['clarify_success']) > 0:
        print(f"\n3. 示例 - Clarify成功案例:")
        for i, sample in enumerate(stats_v3['clarify_success'][:3], 1):
            print(f"\n   示例{i}:")
            print(f"   Persona: {sample['persona']}")
            print(f"   State ID: {sample['state_id']}")
            print(f"   Task Score: {sample['task_score']}")
            print(f"   Prompt片段: {sample['chosen_prompt'][:100]}...")
    
    # 保存详细数据
    output_path = base_dir / "outputs/clarify_analysis.json"
    output_data = {
        "v3_stats": {
            "total_pairs": stats_v3['total_pairs'],
            "chosen_execute": stats_v3['chosen_execute'],
            "chosen_clarify": stats_v3['chosen_clarify'],
            "clarify_success_count": len(stats_v3['clarify_success']),
            "clarify_success_samples": stats_v3['clarify_success'],
            "persona_distribution": dict(stats_v3['persona_action_distribution']),
        },
        "v4_stats": {
            "total_pairs": stats_v4['total_pairs'],
            "chosen_execute": stats_v4['chosen_execute'],
            "chosen_clarify": stats_v4['chosen_clarify'],
            "clarify_success_count": len(stats_v4['clarify_success']),
            "clarify_success_samples": stats_v4['clarify_success'],
        }
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细分析已保存到: {output_path}")
    
    # 建议
    print("\n" + "="*80)
    print("💡 V5数据集构建建议")
    print("="*80)
    
    clarify_success = len(stats_v3['clarify_success'])
    execute_success_v4 = stats_v4['chosen_execute']
    
    if clarify_success > 0:
        print(f"\n✅ 好消息：找到了 {clarify_success} 个高质量的Clarify成功案例！")
        print(f"\n推荐的V5数据集构成:")
        print(f"  • Execute成功: {execute_success_v4} 个（来自V4）")
        print(f"  • Clarify成功: {clarify_success} 个（来自V3）")
        print(f"  • 总计: {execute_success_v4 + clarify_success} 对")
        
        clarify_ratio = clarify_success / (execute_success_v4 + clarify_success) * 100
        print(f"\nClarify比例: {clarify_ratio:.1f}%")
        
        if clarify_ratio < 15:
            print(f"⚠️  Clarify比例偏低（<15%），可以考虑:")
            print(f"   1. 增加部分通过的Clarify案例（0.5 <= score < 1.0）")
            print(f"   2. 生成更多需要Clarify的轨迹")
    else:
        print(f"\n⚠️  V3中没有找到完美Clarify成功案例")
        print(f"   建议：")
        print(f"   1. 检查部分成功的Clarify案例（{len(stats_v3['clarify_partial'])}个）")
        print(f"   2. 生成新的需要Clarify的轨迹数据")

if __name__ == "__main__":
    main()
