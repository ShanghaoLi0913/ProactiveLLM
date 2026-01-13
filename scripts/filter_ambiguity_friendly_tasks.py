"""
Step 1: Filter ambiguity-friendly tasks from BigCodeBench

筛选那些"天然可以被写得不完整"的题目，例如：
- 输入约束未写清（是否有空输入、负数、重复等）
- 输出格式是否严格
- edge case 行为是否可多种合理解释

可以使用规则或LLM判断。
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Delay import of chat_complete until needed (only for LLM method)


def rule_based_filter(task: Dict) -> tuple[bool, str]:
    """
    Rule-based filtering: 基于规则的筛选
    
    检查任务是否包含可能产生歧义的方面：
    - 输入约束不明确（空输入、负数、重复等）
    - 输出格式不明确
    - edge case 行为可多种解释
    """
    instruct_prompt = task.get("instruct_prompt", "").lower()
    test = task.get("test", "").lower()
    
    ambiguity_signals = []
    
    # 1. 检查输入约束
    # 测试用例中包含edge cases，但instruct_prompt中未明确说明
    has_edge_cases_in_test = any(
        keyword in test for keyword in ["empty", "negative", "zero", "single", "identical", "duplicate"]
    )
    mentions_constraints_in_prompt = any(
        keyword in instruct_prompt for keyword in ["empty", "negative", "zero", "constraint", "assume", "handle"]
    )
    
    if has_edge_cases_in_test and not mentions_constraints_in_prompt:
        ambiguity_signals.append("input_constraints_ambiguous")
    
    # 2. 检查输出格式
    # 如果测试用例检查了类型但prompt中没有明确说明
    has_type_check = any(
        keyword in test for keyword in ["isinstance", "assertisinstance", "type", "should be"]
    )
    mentions_output_format = any(
        keyword in instruct_prompt for keyword in ["return", "output", "format", "type", "should output"]
    )
    
    if has_type_check and not mentions_output_format:
        ambiguity_signals.append("output_format_ambiguous")
    
    # 3. 检查是否有多种合理行为
    # 如果测试用例包含多种边界情况，可能表示有歧义
    edge_case_count = sum(1 for keyword in ["empty", "negative", "zero", "single", "null", "none"] if keyword in test)
    if edge_case_count >= 3:
        ambiguity_signals.append("multiple_edge_cases")
    
    # 4. 检查是否缺少关键信息
    # 如果prompt很短但测试很复杂，可能有信息缺失
    prompt_len = len(instruct_prompt)
    test_len = len(test)
    if prompt_len < 200 and test_len > 1000:
        ambiguity_signals.append("missing_information")
    
    is_ambiguous = len(ambiguity_signals) > 0
    reason = ", ".join(ambiguity_signals) if is_ambiguous else "clear_specification"
    
    return is_ambiguous, reason


def llm_based_filter(task: Dict, llm_model: Optional[str] = None) -> tuple[bool, str]:
    """
    LLM-based filtering: 使用LLM判断任务是否适合做澄清
    
    Args:
        task: BigCodeBench任务字典
        llm_model: LLM模型名称（如"gpt-4o-mini"），如果为None则使用规则方法
        
    Returns:
        (is_ambiguous, reason): 是否适合做澄清，以及原因
    """
    if not llm_model:
        return rule_based_filter(task)
    
    # Import here to avoid requiring openai package when using rule method
    from llm.provider import chat_complete
    
    instruct_prompt = task.get("instruct_prompt", "")
    test = task.get("test", "")
    
    system_prompt = """You are an expert in code generation tasks. Your task is to determine if a coding task specification is underspecified or ambiguous, making it suitable for clarification questions.

A task is considered "underspecified" if it lacks critical details that could lead to multiple reasonable implementations, such as:
- Unclear input constraints (empty inputs, negative numbers, duplicates, etc.)
- Unclear output format requirements
- Unspecified edge case behavior
- Missing validation requirements

Respond with JSON: {"is_underspecified": true/false, "reason": "brief explanation"}"""

    user_prompt = f"""Task specification:
{instruct_prompt}

Test cases (for reference):
{test[:1000]}  # Limit test length

Is this task underspecified? Would an assistant need to ask clarifying questions?"""

    try:
        response = chat_complete(system_prompt, user_prompt, model=llm_model, max_tokens=200)
        # Try to parse JSON response
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])
            is_ambiguous = result.get("is_underspecified", False)
            reason = result.get("reason", "llm_judgment")
            return is_ambiguous, reason
        else:
            # Fallback: check if response indicates ambiguity
            ambiguous_keywords = ["yes", "underspecified", "ambiguous", "unclear", "missing"]
            is_ambiguous = any(keyword in response.lower() for keyword in ambiguous_keywords)
            return is_ambiguous, response[:100]
    except Exception as e:
        print(f"Error in LLM filtering for task {task.get('task_id', 'unknown')}: {e}")
        # Fallback to rule-based
        return rule_based_filter(task)


def filter_tasks(
    input_path: Path,
    output_path: Path,
    method: str = "rule",
    llm_model: Optional[str] = None,
    limit: Optional[int] = None
) -> None:
    """
    筛选适合做澄清的题目
    
    Args:
        input_path: BigCodeBench数据集路径（JSONL格式）
        output_path: 输出路径（JSONL格式，只包含筛选后的题目）
        method: 筛选方法，"rule" 或 "llm"
        llm_model: LLM模型名称（如果method为"llm"时需要）
        limit: 限制输出的最大数量（用于测试）
    """
    print(f"Loading tasks from {input_path}...")
    tasks = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks")
    print(f"Filtering method: {method}")
    
    ambiguous_tasks = []
    filter_stats = {
        "total": len(tasks),
        "ambiguous": 0,
        "clear": 0,
        "reasons": {}
    }
    
    for i, task in enumerate(tasks):
        if limit and len(ambiguous_tasks) >= limit:
            break
            
        if method == "llm" and llm_model:
            is_ambiguous, reason = llm_based_filter(task, llm_model)
        else:
            is_ambiguous, reason = rule_based_filter(task)
        
        if is_ambiguous:
            ambiguous_tasks.append(task)
            filter_stats["ambiguous"] += 1
            # Count reasons
            reason_key = reason.split(",")[0].strip()  # Get first reason
            filter_stats["reasons"][reason_key] = filter_stats["reasons"].get(reason_key, 0) + 1
        else:
            filter_stats["clear"] += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(tasks)} tasks, found {len(ambiguous_tasks)} ambiguous tasks")
    
    # Save filtered tasks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for task in ambiguous_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"\nFiltering complete!")
    print(f"Total tasks: {filter_stats['total']}")
    print(f"Ambiguous tasks (selected): {filter_stats['ambiguous']}")
    print(f"Clear tasks (filtered out): {filter_stats['clear']}")
    print(f"\nAmbiguity reasons:")
    for reason, count in sorted(filter_stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nSaved {len(ambiguous_tasks)} tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter ambiguity-friendly tasks from BigCodeBench")
    parser.add_argument("--input", type=str, required=True,
                       help="Input BigCodeBench JSONL file")
    parser.add_argument("--output", type=str,
                       default="data/external/BigCodeBench/ambiguous_tasks.jsonl",
                       help="Output path for filtered tasks")
    parser.add_argument("--method", type=str, choices=["rule", "llm"], default="rule",
                       help="Filtering method: rule-based or LLM-based")
    parser.add_argument("--llm_model", type=str, default=None,
                       help="LLM model name (required if method=llm)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of output tasks (for testing)")
    
    args = parser.parse_args()
    
    if args.method == "llm" and not args.llm_model:
        parser.error("--llm_model is required when --method=llm")
    
    filter_tasks(
        Path(args.input),
        Path(args.output),
        method=args.method,
        llm_model=args.llm_model,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

