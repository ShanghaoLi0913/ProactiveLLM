"""
Step 2: Mask critical details to create underspecified tasks

对选中的任务，有意识地删掉/mask一些信息，比如：
- 输入范围/约束（空输入、负数、重复等）
- 特殊值处理规则
- 返回值细节
- 是否需要排序、过滤、去重等

结果是：用户最初看到的task = 不完整specification
ground truth和hidden tests不变，只是assistant在初始状态看不到
"""
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_maskable_info(task: Dict) -> Dict[str, any]:
    """
    从完整任务中提取可以mask的信息
    
    Returns:
        maskable_info: 包含可mask信息的字典
        {
            "input_constraints": {...},
            "output_format": {...},
            "edge_cases": [...],
            "validation_rules": [...]
        }
    """
    instruct_prompt = task.get("instruct_prompt", "")
    test = task.get("test", "")
    doc_struct_str = task.get("doc_struct", "{}")
    
    maskable_info = {
        "input_constraints": {},
        "output_format": {},
        "edge_cases": [],
        "validation_rules": [],
        "missing_details": []
    }
    
    # 从测试用例中提取edge cases
    edge_case_keywords = {
        "empty": "empty input",
        "negative": "negative numbers",
        "zero": "zero value",
        "single": "single element",
        "identical": "identical elements",
        "duplicate": "duplicate values",
        "null": "null/None values",
        "large": "large inputs"
    }
    
    test_lower = test.lower()
    for keyword, description in edge_case_keywords.items():
        if keyword in test_lower:
            maskable_info["edge_cases"].append(description)
    
    # 从doc_struct中提取参数约束
    try:
        doc_struct = json.loads(doc_struct_str) if isinstance(doc_struct_str, str) else doc_struct_str
        params = doc_struct.get("params", [])
        if params:
            maskable_info["input_constraints"]["param_details"] = params
    except:
        pass
    
    # 从instruct_prompt中提取显式提到的约束
    prompt_lower = instruct_prompt.lower()
    
    # 提取输入约束
    if "raise" in prompt_lower or "exception" in prompt_lower:
        # 提取异常处理要求
        exception_match = re.search(r'raise.*?(?:if|when|for).*?(?:\.|$)', instruct_prompt, re.IGNORECASE)
        if exception_match:
            maskable_info["validation_rules"].append(exception_match.group(0))
    
    # 提取输出格式要求
    output_match = re.search(r'should output.*?(?:\.|$)', instruct_prompt, re.IGNORECASE | re.DOTALL)
    if output_match:
        maskable_info["output_format"]["specification"] = output_match.group(0)
    
    return maskable_info


def mask_prompt(instruct_prompt: str, maskable_info: Dict, mask_level: str = "moderate") -> tuple[str, Dict]:
    """
    Mask关键信息，生成不完整的specification
    
    Args:
        instruct_prompt: 原始指令提示
        maskable_info: 可mask的信息
        mask_level: "light", "moderate", "heavy" - mask的程度
        
    Returns:
        (masked_prompt, masked_fields): mask后的提示和被mask的字段信息
    """
    masked_prompt = instruct_prompt
    masked_fields = {
        "input_constraints": [],
        "output_format": [],
        "edge_cases": [],
        "validation_rules": []
    }
    
    # Light masking: 只mask部分细节
    # Moderate masking: mask输入约束和edge cases
    # Heavy masking: mask大部分约束和格式要求

    # ── 阶段 1：先 mask output_format（必须在 input_constraints 之前）──
    # 原因：input_constraints 的 regex 含 `$`，会跨行吞掉 output_format 内容
    if mask_level in ["moderate", "heavy"]:
        output_patterns = [
            r'should output with:\s*\n.*?(?=\n\s*You should|$)',
        ]

        for pattern in output_patterns:
            matches = list(re.finditer(pattern, masked_prompt, re.IGNORECASE | re.DOTALL))
            for match in reversed(matches):
                masked_fields["output_format"].append(match.group(0).strip())
                # 清理 match 前面的残留不完整句子（如 "The function "）
                before = masked_prompt[:match.start()]
                trailing_fragment = re.search(r'(?:^|\n)(\s*\w[\w\s]{0,30})\s*$', before)
                if trailing_fragment:
                    fragment = trailing_fragment.group(1).strip()
                    # 如果残留片段不是完整句子（不以句号结尾且很短），一起删掉
                    if not fragment.endswith('.') and len(fragment.split()) <= 5:
                        before = before[:trailing_fragment.start(1)]
                masked_prompt = before + masked_prompt[match.end():]

    # ── 阶段 2：mask input_constraints ──
    if mask_level in ["moderate", "heavy"]:
        # 使用 [^\n] 代替 .，防止跨行匹配吞掉后续内容
        patterns_to_remove = [
            r'\b(?:handle|process|support)[^\n]*?(?:empty|negative|zero|null|none)[^\n]*?(?:\.|\n|$)',
            r'\b(?:if|when)[^\n]*?(?:empty|negative|zero)[^\n]*?(?:\.|\n|$)',
            r'default[^\n]*?(?:is|are)[^\n]*?(?:\.|\n|$)',
        ]

        for pattern in patterns_to_remove:
            matches = list(re.finditer(pattern, masked_prompt, re.IGNORECASE))
            for match in reversed(matches):  # Reverse to maintain indices
                masked_prompt = masked_prompt[:match.start()] + masked_prompt[match.end():]
                masked_fields["input_constraints"].append(match.group(0).strip())
    
    if mask_level == "heavy":
        # Heavy masking: 移除更多细节
        detail_patterns = [
            r'Requirements:.*?\n',
            r'Example:.*?(?=\n\n|\nYou should|$)',
        ]
        
        for pattern in detail_patterns:
            masked_prompt = re.sub(pattern, '', masked_prompt, flags=re.IGNORECASE | re.DOTALL)
    
    # 清理多余的空行
    masked_prompt = re.sub(r'\n\s*\n\s*\n+', '\n\n', masked_prompt).strip()
    
    # 从maskable_info中提取实际被mask的信息（用于disclosure）
    if maskable_info["edge_cases"]:
        masked_fields["edge_cases"] = maskable_info["edge_cases"]
    if maskable_info["validation_rules"]:
        masked_fields["validation_rules"] = maskable_info["validation_rules"]
    
    return masked_prompt, masked_fields


def create_mask_rule(masked_fields: Dict, task: Dict) -> Dict:
    """
    创建disclosure rule数据结构
    
    存储被mask的信息，供simulator在assistant问澄清问题时使用
    
    Returns:
        disclosure_rule: 包含被mask信息的字典，用于Step 3
    """
    disclosure_rule = {
        "masked_fields": masked_fields,
        "disclosure_info": {}
    }
    
    # 为每个被mask的类别创建disclosure信息
    test = task.get("test", "")
    instruct_prompt = task.get("instruct_prompt", "")
    
    # 输入约束的disclosure信息
    if masked_fields.get("input_constraints") or masked_fields.get("edge_cases"):
        disclosure_rule["disclosure_info"]["input_constraints"] = {
            "edge_cases": masked_fields.get("edge_cases", []),
            "hints": extract_constraint_hints(test, instruct_prompt)
        }
    
    # 输出格式的disclosure信息 — 直接使用 masked_fields 中的完整内容
    if masked_fields.get("output_format"):
        disclosure_rule["disclosure_info"]["output_format"] = {
            "specification": masked_fields["output_format"]
        }
    
    # 验证规则的disclosure信息
    if masked_fields.get("validation_rules"):
        disclosure_rule["disclosure_info"]["validation_rules"] = {
            "rules": masked_fields.get("validation_rules", [])
        }
    
    return disclosure_rule


def extract_constraint_hints(test: str, prompt: str) -> List[str]:
    """从测试用例中提取约束提示"""
    hints = []
    test_lower = test.lower()
    
    if "empty" in test_lower or "[]" in test:
        hints.append("Should handle empty inputs")
    if "negative" in test_lower or "-" in test:
        hints.append("Should handle negative numbers")
    if "zero" in test_lower:
        hints.append("Should handle zero values")
    if "single" in test_lower:
        hints.append("Should handle single element inputs")
    
    return hints


def extract_output_spec(task: Dict) -> str:
    """提取输出格式规范"""
    instruct_prompt = task.get("instruct_prompt", "")
    
    # 尝试提取输出类型和格式
    output_match = re.search(r'should output.*?(?:\n|$)', instruct_prompt, re.IGNORECASE | re.DOTALL)
    if output_match:
        return output_match.group(0).strip()
    
    # 从doc_struct提取
    doc_struct_str = task.get("doc_struct", "{}")
    try:
        doc_struct = json.loads(doc_struct_str) if isinstance(doc_struct_str, str) else doc_struct_str
        returns = doc_struct.get("returns", [])
        if returns:
            return " ".join(returns)
    except:
        pass
    
    return ""


def process_tasks(
    input_path: Path,
    output_path: Path,
    mask_level: str = "moderate",
    limit: Optional[int] = None
) -> None:
    """
    处理任务：mask关键细节
    
    Args:
        input_path: 筛选后的任务文件（JSONL）
        output_path: 输出路径（JSONL，包含masked版本和disclosure rule）
        mask_level: "light", "moderate", "heavy"
        limit: 限制处理的任务数量
    """
    print(f"Loading tasks from {input_path}...")
    tasks = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks")
    print(f"Mask level: {mask_level}")
    
    masked_tasks = []
    
    for i, task in enumerate(tasks):
        if limit and len(masked_tasks) >= limit:
            break
        
        # 提取可mask信息
        maskable_info = extract_maskable_info(task)
        
        # 创建masked版本
        original_prompt = task.get("instruct_prompt", "")
        masked_prompt, masked_fields = mask_prompt(original_prompt, maskable_info, mask_level)
        
        # 创建disclosure rule
        disclosure_rule = create_mask_rule(masked_fields, task)
        
        # 创建新任务对象
        masked_task = task.copy()
        masked_task["original_instruct_prompt"] = original_prompt  # 保留原始版本
        masked_task["instruct_prompt"] = masked_prompt  # masked版本
        masked_task["disclosure_rule"] = disclosure_rule  # disclosure rule for Step 3
        
        masked_tasks.append(masked_task)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(tasks)} tasks")
    
    # 保存masked任务
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for task in masked_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"\nMasking complete!")
    print(f"Processed {len(masked_tasks)} tasks")
    print(f"Saved to {output_path}")
    
    # 打印统计信息
    total_masked_fields = sum(len(task.get("disclosure_rule", {}).get("masked_fields", {}).get("input_constraints", [])) 
                              for task in masked_tasks)
    print(f"Total masked input constraints: {total_masked_fields}")


def main():
    parser = argparse.ArgumentParser(description="Mask critical details from tasks to create underspecified versions")
    parser.add_argument("--input", type=str, required=True,
                       help="Input filtered tasks JSONL file")
    parser.add_argument("--output", type=str,
                       default="data/external/BigCodeBench/masked_tasks.jsonl",
                       help="Output path for masked tasks")
    parser.add_argument("--mask_level", type=str, choices=["light", "moderate", "heavy"], default="moderate",
                       help="Level of masking: light, moderate, or heavy")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to process (for testing)")
    
    args = parser.parse_args()
    
    process_tasks(
        Path(args.input),
        Path(args.output),
        mask_level=args.mask_level,
        limit=args.limit
    )


if __name__ == "__main__":
    main()


