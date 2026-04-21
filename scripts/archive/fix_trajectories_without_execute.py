"""
修复没有Execute的轨迹：为最终没有Execute的轨迹添加Execute turn

策略：
1. 保留原始数据不变
2. 识别没有Execute的轨迹（用户停止或达到max_turns）
3. 为每个需要修复的轨迹添加一个Execute turn
4. 保存修复后的轨迹到新文件
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_trajectories import (
    LocalHFChatGenerator,
    build_action_prompts,
    check_task_completion,
    llm_output,
    dummy_llm_output,
)
from simulator import react, PERSONAS


def find_persona_by_name(persona_name: str):
    """根据persona名称找到对应的persona对象"""
    for persona in PERSONAS:
        if persona.name == persona_name:
            return persona
    # 默认返回第一个persona
    return PERSONAS[0]


def generate_execute_turn(
    last_turn: Dict,
    domain: str,
    llm_model: Optional[str] = None,
    local_generator: Optional[LocalHFChatGenerator] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict:
    """
    为轨迹生成一个Execute turn
    
    Args:
        last_turn: 最后一个turn的dict
        domain: "coding" or "planning"
        llm_model: LLM model name (None for dummy)
        local_generator: LocalHFChatGenerator instance
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
    
    Returns:
        新的Execute turn dict
    """
    # 获取当前state（最后一个turn的state）
    current_state = last_turn["state"].copy()
    
    # 更新dialogue_turn
    current_state["dialogue_turn"] = current_state.get("dialogue_turn", 0) + 1
    
    # 获取persona信息
    persona_dict = last_turn.get("persona", {})
    persona_name = persona_dict.get("name", "Novice-Learner")
    persona = find_persona_by_name(persona_name)
    
    # 获取trajectory_id
    trajectory_id = last_turn.get("trajectory_id", "unknown")
    
    # 获取action prompts
    prompts = build_action_prompts(domain)
    action = "Execute"
    action_prompt = prompts[action]
    
    # 生成Execute响应
    if local_generator is not None:
        assistant_msg = local_generator.chat_complete(action_prompt, f"[Task]\n{current_state['query']}")
    elif llm_model:
        assistant_msg = llm_output(current_state, action_prompt, llm_model, temperature=temperature, top_p=top_p)
    else:
        assistant_msg = dummy_llm_output(current_state, action_prompt)
    
    # 获取simulator reaction（对于Execute，用户应该接受）
    # 注意：这里我们不再需要真实的user reaction，因为这是修复后的turn
    # 但为了保持数据结构一致，我们创建一个默认的reaction
    reaction = {
        "user_reply": "",  # Execute后用户不需要回复
        "meta": {
            "answered_clarification": 0,
            "reject_signal": 0,
            "answer_clarity": 0.0,
        }
    }
    
    # 检查task_completed
    task_completed = check_task_completion(current_state, assistant_msg, domain)
    
    # 获取has_edge_cases_info（从之前的turns继承）
    has_edge_cases_info = last_turn.get("has_edge_cases_info", False)
    
    # 创建新的Execute turn
    new_turn = {
        "trajectory_id": trajectory_id,
        "state": current_state,
        "action": action,
        "action_prompt": action_prompt,
        "assistant_msg": assistant_msg,
        "persona": persona_dict,
        "user_reaction": reaction,
        "turn": last_turn.get("turn", 1) + 1,
        "is_mainline": True,
        "is_terminal": True,  # Execute后必须结束
        "task_completed": task_completed,
        "has_edge_cases_info": has_edge_cases_info,
        "fixed": True,  # 标记这是修复后的turn
    }
    
    return new_turn


def fix_trajectories(
    traj_path: Path,
    out_path: Path,
    domain: str = "coding",
    llm_model: Optional[str] = None,
    local_model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 400,
) -> None:
    """
    修复没有Execute的轨迹
    
    Args:
        traj_path: 原始轨迹文件路径
        out_path: 输出文件路径
        domain: "coding" or "planning"
        llm_model: LLM model name (None for dummy)
        local_model: Local HF model name
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        max_new_tokens: Max new tokens for generation
    """
    print(f"📂 Loading trajectories from: {traj_path}")
    
    # 读取所有trajectories
    with open(traj_path, 'r', encoding='utf-8') as f:
        all_trajs = [json.loads(line) for line in f if line.strip()]
    
    print(f"📊 Loaded {len(all_trajs)} trajectory turns")
    
    # 按trajectory_id分组
    traj_groups = defaultdict(list)
    for traj in all_trajs:
        traj_id = traj.get("trajectory_id", "unknown")
        traj_groups[traj_id].append(traj)
    
    # 按turn排序
    for traj_id in traj_groups:
        traj_groups[traj_id].sort(key=lambda x: x.get("turn", 0))
    
    print(f"📊 Grouped into {len(traj_groups)} conversations")
    
    # 识别需要修复的轨迹
    needs_fix = []
    for traj_id, turns in traj_groups.items():
        if not turns:
            continue
        
        last_turn = turns[-1]
        last_action = last_turn.get("action", "")
        
        # 检查是否有Execute
        has_execute = any(t.get("action") == "Execute" for t in turns)
        
        if not has_execute:
            # 没有Execute，需要修复
            needs_fix.append((traj_id, turns))
    
    print(f"🔧 Found {len(needs_fix)} conversations without Execute (need fixing)")
    print(f"   - Total conversations: {len(traj_groups)}")
    print(f"   - Need fixing: {len(needs_fix)} ({len(needs_fix)/len(traj_groups)*100:.1f}%)")
    
    if len(needs_fix) == 0:
        print("✅ All conversations already have Execute, no fixing needed!")
        # 仍然复制原始数据到输出文件
        with open(out_path, 'w', encoding='utf-8') as f:
            for traj in all_trajs:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        print(f"✅ Copied {len(all_trajs)} trajectories to {out_path}")
        return
    
    # 初始化模型生成器
    local_gen = None
    if local_model:
        print(f"🤖 Using local HF model for generation: {local_model}")
        local_gen = LocalHFChatGenerator(
            local_model,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=42,
        )
        llm_model = None
    elif not llm_model:
        print("⚠️  No model specified, using dummy output")
    
    # 修复轨迹
    fixed_count = 0
    all_fixed_trajs = []
    
    print(f"\n🔧 Starting to fix trajectories...")
    for i, (traj_id, turns) in enumerate(needs_fix, 1):
        if i % 50 == 0 or i == len(needs_fix):
            print(f"  Progress: {i}/{len(needs_fix)} ({i/len(needs_fix)*100:.1f}%)", flush=True)
        
        # 添加所有原始turns
        all_fixed_trajs.extend(turns)
        
        # 生成Execute turn
        last_turn = turns[-1]
        try:
            new_execute_turn = generate_execute_turn(
                last_turn,
                domain,
                llm_model=llm_model,
                local_generator=local_gen,
                temperature=temperature,
                top_p=top_p,
            )
            all_fixed_trajs.append(new_execute_turn)
            fixed_count += 1
        except Exception as e:
            print(f"  ⚠️  Error fixing trajectory {traj_id}: {e}")
            # 即使出错，也保留原始turns
            continue
    
    # 添加不需要修复的轨迹
    for traj_id, turns in traj_groups.items():
        if (traj_id, turns) not in needs_fix:
            all_fixed_trajs.extend(turns)
    
    # 保存修复后的轨迹
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for traj in all_fixed_trajs:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Fixed {fixed_count}/{len(needs_fix)} conversations")
    print(f"✅ Wrote {len(all_fixed_trajs)} trajectory turns to {out_path}")
    
    # 统计修复后的情况
    fixed_groups = defaultdict(list)
    for traj in all_fixed_trajs:
        traj_id = traj.get("trajectory_id", "unknown")
        fixed_groups[traj_id].append(traj)
    
    execute_count = sum(1 for traj_id, turns in fixed_groups.items() 
                       if any(t.get("action") == "Execute" for t in turns))
    
    print(f"\n📊 Final statistics:")
    print(f"   - Total conversations: {len(fixed_groups)}")
    print(f"   - Conversations with Execute: {execute_count}/{len(fixed_groups)} ({execute_count/len(fixed_groups)*100:.1f}%)")
    print(f"   - Fixed conversations: {fixed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix trajectories without Execute by adding Execute turns"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help="Path to original trajectories JSONL",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for fixed trajectories JSONL",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["coding", "planning"],
        default="coding",
        help="Domain (default: coding)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="",
        help="OpenAI model name (e.g., gpt-4o-mini). If empty, uses dummy output.",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        default="",
        help="HF model name for local generation (if set, overrides --llm_model)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Max new tokens for generation (default: 400)",
    )
    
    args = parser.parse_args()
    
    traj_path = Path(args.trajectories)
    out_path = Path(args.out)
    
    if not traj_path.exists():
        raise SystemExit(f"Trajectories file not found: {traj_path}")
    
    llm_model = args.llm_model if args.llm_model else None
    local_model = args.local_model if args.local_model else None
    
    fix_trajectories(
        traj_path,
        out_path,
        domain=args.domain,
        llm_model=llm_model,
        local_model=local_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
