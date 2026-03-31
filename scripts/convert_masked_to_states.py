"""
Convert masked BigCodeBench tasks to state format with disclosure rules

将masked任务转换为项目使用的state格式，并在state中包含disclosure_rule
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.compute_task_uncertainty import compute_task_uncertainty


def convert_masked_task_to_state(masked_task: Dict, domain: str = "coding") -> Dict:
    """
    将masked任务转换为state格式
    
    Args:
        masked_task: masked任务字典（包含disclosure_rule）
        domain: 任务领域
        
    Returns:
        state: 项目使用的state格式，包含disclosure_rule
    """
    # 使用masked版本的instruct_prompt作为query
    query = masked_task.get("instruct_prompt", "")  # 这是masked版本
    
    # 计算task_uncertainty（masked版本应该不确定性更高）
    task_uncertainty = compute_task_uncertainty(query)
    
    # 初始化disclosure_rule，添加disclosed_info字段用于跟踪已披露信息
    disclosure_rule = masked_task.get("disclosure_rule", {}).copy()
    if "disclosed_info" not in disclosure_rule:
        disclosure_rule["disclosed_info"] = {
            "edge_cases": [],
            "input_constraints": [],
            "output_format": [],
            "validation_rules": [],
        }
    
    state = {
        "id": masked_task.get("task_id", ""),
        "domain": domain,
        "query": query,  # masked版本的query
        "dialogue_turn": 0,  # 初始状态
        "prev_reject": 0,  # 初始状态
        "task_uncertainty": task_uncertainty,
        # 保留原始信息用于评估
        "original_instruct_prompt": masked_task.get("original_instruct_prompt", ""),
        "canonical_solution": masked_task.get("canonical_solution", ""),
        "test": masked_task.get("test", ""),
        "entry_point": masked_task.get("entry_point", ""),
        # disclosure_rule用于simulator，包含disclosed_info用于跟踪已披露信息
        "disclosure_rule": disclosure_rule,
    }
    
    return state


def convert_tasks(
    input_path: Path,
    output_path: Path,
    domain: str = "coding",
    limit: Optional[int] = None
) -> None:
    """
    转换masked任务为state格式
    
    Args:
        input_path: masked任务文件（JSONL）
        output_path: 输出state文件（JSONL）
        domain: 任务领域
        limit: 限制转换的任务数量
    """
    print(f"Loading masked tasks from {input_path}...")
    tasks = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} masked tasks")
    
    states = []
    for i, task in enumerate(tasks):
        if limit and len(states) >= limit:
            break
        
        state = convert_masked_task_to_state(task, domain)
        states.append(state)
        
        if (i + 1) % 50 == 0:
            print(f"Converted {i + 1}/{len(tasks)} tasks")
    
    # 保存states
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for state in states:
            f.write(json.dumps(state, ensure_ascii=False) + '\n')
    
    print(f"\nConversion complete!")
    print(f"Converted {len(states)} tasks to states")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert masked BigCodeBench tasks to state format")
    parser.add_argument("--input", type=str, required=True,
                       help="Input masked tasks JSONL file")
    parser.add_argument("--output", type=str,
                       default="data/seeds/bigcodebench_masked_states.jsonl",
                       help="Output path for states")
    parser.add_argument("--domain", type=str, default="coding",
                       choices=["coding", "planning"])
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to convert")
    
    args = parser.parse_args()
    
    convert_tasks(
        Path(args.input),
        Path(args.output),
        domain=args.domain,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

