#!/usr/bin/env python3
"""
生成V5平衡数据集：包含Execute和Clarify样本
关键创新：使用trajectory-level的最终结果来评估Clarify的价值
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_jsonl(path: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], path: Path):
    """保存JSONL文件"""
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_trajectories(traj_path: Path) -> Dict[str, List[Dict]]:
    """按task_id组织轨迹数据"""
    turns = load_jsonl(traj_path)
    
    trajectories = defaultdict(list)
    for turn in turns:
        task_id = turn.get("state", {}).get("id")
        if task_id:
            trajectories[task_id].append(turn)
    
    # 排序每个轨迹的turns
    for task_id in trajectories:
        trajectories[task_id].sort(key=lambda x: x.get("turn", 0))
    
    return trajectories

def extract_clarify_prefs(trajectories: Dict[str, List[Dict]], 
                          execute_prefs: List[Dict]) -> List[Dict]:
    """
    从成功的多轮轨迹中提取Clarify preference pairs
    
    策略：
    1. 找到"Clarify后最终成功"的轨迹
    2. 提取Clarify那一步的state和response
    3. 构建preference pair: (Clarify,最终成功) vs (Execute,立即失败)
    """
    clarify_prefs = []
    
    # 建立execute失败样本的索引（用于构建对比）
    execute_fail_by_state = {}
    for pref in execute_prefs:
        if pref.get("rejected_action") == "Execute" and pref.get("rejected_task_score", 0) == 0:
            # state可能是dict或string
            state = pref.get("state", {})
            if isinstance(state, dict):
                state_id = state.get("id")
            else:
                state_id = pref.get("state_id")
            
            if state_id and state_id not in execute_fail_by_state:
                execute_fail_by_state[state_id] = pref
    
    print(f"\n构建Clarify preference pairs...")
    print(f"  可用的Execute失败样本: {len(execute_fail_by_state)} 个")
    
    clarify_success_count = 0
    clarify_fail_count = 0
    
    for task_id, task_turns in trajectories.items():
        # 检查最终是否成功
        final_turn = task_turns[-1]
        task_completed = final_turn.get("task_completed", False)
        
        if not task_completed:
            clarify_fail_count += 1
            continue
        
        # 找出所有Clarify turns
        for turn_idx, turn in enumerate(task_turns):
            action = turn.get("action")
            if action != "Clarify":
                continue
            
            # 这是一个Clarify turn，且最终任务成功了
            state_id = turn.get("state", {}).get("id")
            state = turn.get("state", {})
            persona = turn.get("persona", {})
            if isinstance(persona, dict):
                persona_type = persona.get("type", "Unknown")
            else:
                persona_type = str(persona)
            
            # 构建chosen (Clarify)
            chosen_prompt = turn.get("action_prompt", "")
            chosen_response = turn.get("assistant_msg", "")
            
            # 尝试找到一个对比的Execute失败样本
            # 优先使用同一个state的Execute失败
            rejected_response = ""
            rejected_prompt = ""
            rejected_assistant_msg = ""
            
            if state_id in execute_fail_by_state:
                exec_fail = execute_fail_by_state[state_id]
                # 原始数据中可能用的是assistant_msg字段
                rejected_response = exec_fail.get("rejected_response", "") or exec_fail.get("rejected_assistant_msg", "")
                rejected_prompt = exec_fail.get("rejected_prompt", "") or exec_fail.get("action_prompt", "")
            
            # 如果没有精确匹配，使用任意一个Execute失败样本
            if not rejected_response and len(execute_fail_by_state) > 0:
                some_fail = list(execute_fail_by_state.values())[0]
                rejected_response = some_fail.get("rejected_response", "") or some_fail.get("rejected_assistant_msg", "")
                rejected_prompt = some_fail.get("rejected_prompt", "") or some_fail.get("action_prompt", "")
            
            if not chosen_response or not rejected_response:
                continue
            
            # 关键：使用最终的task completion作为奖励
            # Clarify本身不生成代码，但它导致了最终成功
            chosen_reward = 1.0 - 0.1  # 成功(1.0) - interrupt cost(0.1)
            chosen_task_score = 1.0    # 最终任务完成
            
            rejected_reward = 0.0 - 0.4  # 失败(0.0) - 高interrupt(0.4)
            rejected_task_score = 0.0
            
            pref = {
                "state_id": state_id,
                "persona": persona_type,
                "chosen_action": "Clarify",
                "rejected_action": "Execute",
                "chosen_prompt": chosen_prompt,
                "chosen_response": chosen_response,
                "rejected_prompt": rejected_prompt,
                "rejected_response": rejected_response,
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "chosen_task_score": chosen_task_score,
                "rejected_task_score": rejected_task_score,
                "trajectory_based": True,  # 标记这是基于轨迹的Clarify样本
                "final_turn_idx": len(task_turns),
                "clarify_turn_idx": turn_idx,
            }
            
            clarify_prefs.append(pref)
            clarify_success_count += 1
    
    print(f"  Clarify后成功的任务: {clarify_success_count}")
    print(f"  Clarify后失败的任务: {clarify_fail_count}")
    print(f"  生成的Clarify preference pairs: {len(clarify_prefs)}")
    
    return clarify_prefs

def main():
    base_dir = Path("/root/autodl-tmp/ProactiveLLM")
    
    # 加载数据
    traj_path = base_dir / "data/data/logs/traj_bigcode_100states_20260206_050454.jsonl"
    execute_prefs_path = base_dir / "data/dpo/prefs_bigcode_100_repaired.jsonl"  # V4的高质量Execute数据
    all_prefs_path = base_dir / "data/dpo/prefs_bigcode_100.jsonl"  # 原始数据（包含失败样本）
    
    print("="*80)
    print("🔧 生成V5平衡数据集（Execute + Clarify）")
    print("="*80)
    
    print(f"\n📂 加载数据...")
    trajectories = load_trajectories(traj_path)
    print(f"   轨迹数: {len(trajectories)} 个任务")
    
    execute_prefs = load_jsonl(execute_prefs_path)
    print(f"   Execute样本（V4高质量）: {len(execute_prefs)} 对")
    
    all_prefs = load_jsonl(all_prefs_path)
    print(f"   原始样本（包含失败）: {len(all_prefs)} 对")
    
    # 提取Clarify preference pairs（使用原始数据中的失败样本作为对比）
    clarify_prefs = extract_clarify_prefs(trajectories, all_prefs)
    
    # 构建平衡数据集
    print(f"\n" + "="*80)
    print(f"📊 构建平衡数据集")
    print(f"="*80)
    
    # 策略A: 全部包含（Execute + Clarify）
    v5_all = execute_prefs + clarify_prefs
    print(f"\n策略A - 包含所有数据:")
    print(f"  Execute: {len(execute_prefs)} 对 ({len(execute_prefs)/len(v5_all)*100:.1f}%)")
    print(f"  Clarify: {len(clarify_prefs)} 对 ({len(clarify_prefs)/len(v5_all)*100:.1f}%)")
    print(f"  总计: {len(v5_all)} 对")
    
    # 策略B: 平衡比例（保持Clarify在15-25%）
    target_clarify_ratio = 0.20  # 20% Clarify
    if len(clarify_prefs) > 0:
        # 如果Clarify太多，需要下采样Clarify或上采样Execute
        # 计算需要多少Execute才能达到目标比例
        # clarify / (execute + clarify) = 0.2
        # clarify = 0.2 * (execute + clarify)
        # clarify = 0.2 * execute + 0.2 * clarify
        # 0.8 * clarify = 0.2 * execute
        # execute = 4 * clarify
        
        target_execute_count = int(len(clarify_prefs) * (1 - target_clarify_ratio) / target_clarify_ratio)
        
        if len(execute_prefs) >= target_execute_count:
            # Execute够多，使用全部Execute
            execute_sampled = execute_prefs
            # 下采样Clarify
            clarify_sampled = clarify_prefs[:int(len(execute_prefs) * target_clarify_ratio / (1 - target_clarify_ratio))]
        else:
            # Execute不够，使用全部Execute，下采样Clarify
            execute_sampled = execute_prefs
            clarify_sampled = clarify_prefs[:int(len(execute_prefs) * target_clarify_ratio / (1 - target_clarify_ratio))]
        
        v5_balanced = execute_sampled + clarify_sampled
    else:
        execute_sampled = execute_prefs
        clarify_sampled = []
        v5_balanced = execute_sampled
    
    if len(clarify_prefs) > 0:
        actual_clarify_ratio = len(clarify_sampled) / len(v5_balanced) * 100
        print(f"\n策略B - 平衡比例（目标20% Clarify）:")
        print(f"  Execute: {len(execute_sampled)} 对 ({len(execute_sampled)/len(v5_balanced)*100:.1f}%)")
        print(f"  Clarify: {len(clarify_sampled)} 对 ({actual_clarify_ratio:.1f}%)")
        print(f"  总计: {len(v5_balanced)} 对")
    
    # 保存数据集
    output_all = base_dir / "data/dpo/prefs_bigcode_v5_all.jsonl"
    output_balanced = base_dir / "data/dpo/prefs_bigcode_v5_balanced.jsonl"
    
    save_jsonl(v5_all, output_all)
    save_jsonl(v5_balanced, output_balanced)
    
    print(f"\n✅ V5数据集已保存:")
    print(f"   策略A（全部）: {output_all}")
    print(f"   策略B（平衡）: {output_balanced}")
    
    # 统计信息
    print(f"\n" + "="*80)
    print(f"📈 数据演进总结")
    print(f"="*80)
    print(f"""
V1-V2: ~100对，低质量混合 → TSR ~17%
V3:    304对，允许部分通过（但0% Clarify）→ TSR 25.68%
V4:    135对，完美Execute（0% Clarify）→ TSR 32.30%
V5A:   {len(v5_all)}对，Execute+Clarify → 预期提升persona适应能力
V5B:   {len(v5_balanced)}对，平衡比例（~20% Clarify）→ 推荐版本 ⭐

关键改进：
  ✅ V5包含了Clarify样本（基于多轮轨迹的最终成功）
  ✅ 模型可以学习"何时Clarify、何时Execute"
  ✅ 为persona-aware行为奠定基础
    """)

if __name__ == "__main__":
    main()
