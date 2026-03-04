"""
State Reconstruction Module

将Clarify得到的用户回答转换为结构化的Spec，用于Execute阶段。

关键设计：
1. 只保留用户提供的信息（去掉Assistant的问题）
2. 将用户回答映射到结构化字段（Edge cases, Output format, Constraints等）
3. Execute时使用structured state而不是chat replay
"""

import re
from typing import Dict, List, Optional


def extract_user_answers_from_query(query: str) -> List[str]:
    """
    从query中提取所有用户回答，移除Assistant的问题。
    
    Args:
        query: 可能包含对话历史的query字符串
        
    Returns:
        user_answers: 用户回答列表（按时间顺序）
    """
    if "[User]:" not in query:
        return []  # 没有对话历史
    
    user_answers = []
    # 提取所有[User]:之后的内容
    parts = query.split("[User]:")
    for part in parts[1:]:  # 跳过第一个部分（原始query）
        # 提取到下一个[Assistant]:之前的内容，或者到字符串末尾
        if "[Assistant]:" in part:
            user_answer = part.split("[Assistant]:")[0].strip()
        else:
            user_answer = part.strip()
        if user_answer:
            user_answers.append(user_answer)
    
    return user_answers


def extract_original_query(query: str) -> str:
    """
    从包含对话历史的query中提取原始query（第一个[Assistant]:或[User]:之前的内容）。
    
    Args:
        query: 可能包含对话历史的query字符串
        
    Returns:
        original_query: 原始query（不包含对话历史）
    """
    if "[Assistant]:" not in query and "[User]:" not in query:
        return query  # 没有对话历史，直接返回
    
    # 找到第一个[Assistant]:或[User]:的位置
    first_marker = min(
        query.find("[Assistant]:"),
        query.find("[User]:"),
        key=lambda x: x if x >= 0 else len(query)
    )
    
    if first_marker >= 0:
        return query[:first_marker].strip()
    return query


def parse_user_answer_to_structured_spec(user_answer: str, disclosure_rule: Optional[Dict] = None) -> Dict[str, List[str]]:
    """
    将用户回答解析为结构化的spec字段。
    
    映射规则：
    - Edge cases: "empty", "null", "single", "large", "negative"等关键词
    - Output format: "should output", "output format", "return"等
    - Constraints: "time complexity", "space complexity", "recursion", "iteration"等
    - Input constraints: "default", "range", "type"等
    
    Args:
        user_answer: 用户回答文本
        disclosure_rule: 可选的disclosure_rule，用于验证和补充信息
        
    Returns:
        structured_spec: 结构化的spec字典
        {
            "edge_cases": [...],
            "output_format": [...],
            "constraints": [...],
            "input_constraints": [...]
        }
    """
    structured_spec = {
        "edge_cases": [],
        "output_format": [],
        "constraints": [],
        "input_constraints": []
    }
    
    user_answer_lower = user_answer.lower()
    
    # 提取Edge cases
    edge_case_patterns = [
        (r"empty\s+(?:string|input|list|array|dict)", "empty input"),
        (r"null|none", "null/None values"),
        (r"single\s+element", "single element"),
        (r"large\s+input", "large inputs"),
        (r"negative", "negative numbers"),
        (r"zero", "zero value"),
        (r"duplicate", "duplicate values"),
    ]
    
    for pattern, description in edge_case_patterns:
        if re.search(pattern, user_answer_lower):
            if description not in structured_spec["edge_cases"]:
                structured_spec["edge_cases"].append(description)
    
    # 提取Output format
    output_patterns = [
        (r"should\s+output\s+with[:\s]+([^\.]+)", "output specification"),
        (r"output\s+format[:\s]+([^\.]+)", "output format"),
        (r"return\s+(?:a\s+)?(?:dict|list|tuple|counter|dataframe)", "return type"),
    ]
    
    for pattern, label in output_patterns:
        match = re.search(pattern, user_answer_lower, re.IGNORECASE)
        if match:
            spec_text = match.group(1).strip() if match.groups() else label
            # 提取完整规格（如果匹配到具体内容）
            if ":" in user_answer and "output" in user_answer_lower:
                # 尝试提取"should output with: ..."后的完整内容
                output_match = re.search(r"output\s+(?:format|specification|with)[:\s]+(.+?)(?:\.|$)", user_answer, re.IGNORECASE | re.DOTALL)
                if output_match:
                    spec_text = output_match.group(1).strip()
            if spec_text and spec_text not in structured_spec["output_format"]:
                structured_spec["output_format"].append(spec_text)
    
    # 提取Constraints（时间/空间复杂度、算法要求等）
    # 优先匹配完整的复杂度描述（支持中英文）
    time_complexity_keywords = r"(?:time\s+complexity|时间复杂度|时间复杂)"
    if re.search(time_complexity_keywords, user_answer_lower, re.IGNORECASE):
        # 提取 "time complexity O(n)" 或 "时间复杂度O(n)"
        time_complexity_match = re.search(rf"{time_complexity_keywords}[:\s]*O\(([^\)]+)\)", user_answer, re.IGNORECASE)
        if time_complexity_match:
            constraint_text = f"time complexity O({time_complexity_match.group(1)})"
        else:
            constraint_text = "time complexity specified"
        if constraint_text not in structured_spec["constraints"]:
            structured_spec["constraints"].append(constraint_text)
    
    space_complexity_keywords = r"(?:space\s+complexity|空间复杂度|空间复杂)"
    if re.search(space_complexity_keywords, user_answer_lower, re.IGNORECASE):
        space_complexity_match = re.search(rf"{space_complexity_keywords}[:\s]*O\(([^\)]+)\)", user_answer, re.IGNORECASE)
        if space_complexity_match:
            constraint_text = f"space complexity O({space_complexity_match.group(1)})"
        else:
            constraint_text = "space complexity specified"
        if constraint_text not in structured_spec["constraints"]:
            structured_spec["constraints"].append(constraint_text)
    
    # 匹配独立的O(...)（如果没有明确标注time/space complexity）
    if not any("complexity" in c for c in structured_spec["constraints"]):
        o_notation_match = re.search(r"O\(([^\)]+)\)", user_answer, re.IGNORECASE)
        if o_notation_match:
            constraint_text = f"O({o_notation_match.group(1)})"
            if constraint_text not in structured_spec["constraints"]:
                structured_spec["constraints"].append(constraint_text)
    
    # 算法要求
    algorithm_patterns = [
        (r"use\s+recursion|使用递归|recursive", "use recursion"),
        (r"use\s+iteration|使用迭代|iterative", "use iteration"),
    ]
    
    for pattern, label in algorithm_patterns:
        if re.search(pattern, user_answer_lower, re.IGNORECASE):
            if label not in structured_spec["constraints"]:
                structured_spec["constraints"].append(label)
    
    # 提取Input constraints
    input_patterns = [
        (r"default\s+(?:is|value)[:\s]+([^\.]+)", "default value"),
        (r"range[:\s]+([^\.]+)", "input range"),
        (r"type[:\s]+([^\.]+)", "input type"),
    ]
    
    for pattern, label in input_patterns:
        match = re.search(pattern, user_answer_lower, re.IGNORECASE)
        if match:
            constraint_text = match.group(1).strip() if match.groups() else label
            if constraint_text and constraint_text not in structured_spec["input_constraints"]:
                structured_spec["input_constraints"].append(constraint_text)
    
    return structured_spec


def merge_structured_specs(specs: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    合并多个structured spec（来自多轮Clarify的用户回答）。
    
    Args:
        specs: structured spec列表
        
    Returns:
        merged_spec: 合并后的structured spec
    """
    merged = {
        "edge_cases": [],
        "output_format": [],
        "constraints": [],
        "input_constraints": []
    }
    
    for spec in specs:
        for key in merged:
            for item in spec.get(key, []):
                if item not in merged[key]:
                    merged[key].append(item)
    
    return merged


def reconstruct_state_for_execute(state: Dict) -> Dict[str, str]:
    """
    重构state，为Execute阶段生成结构化的prompt。
    
    关键改进：
    1. 提取原始query（去掉对话历史）
    2. 提取所有用户回答
    3. 将用户回答解析为结构化spec
    4. 生成结构化的[Clarified Requirements]部分
    
    Args:
        state: 当前state（可能包含对话历史）
        
    Returns:
        reconstructed: 包含以下字段的字典
        {
            "original_query": 原始masked query,
            "clarified_requirements": 结构化的requirements文本,
            "has_clarifications": 是否有澄清信息
        }
    """
    query = state.get("query", "")
    disclosure_rule = state.get("disclosure_rule", {})
    
    # 提取原始query
    original_query = extract_original_query(query)
    
    # 提取用户回答
    user_answers = extract_user_answers_from_query(query)
    
    if not user_answers:
        # 没有澄清信息，直接返回原始query
        return {
            "original_query": original_query,
            "clarified_requirements": "",
            "has_clarifications": False
        }
    
    # 解析每个用户回答为结构化spec
    structured_specs = []
    for user_answer in user_answers:
        spec = parse_user_answer_to_structured_spec(user_answer, disclosure_rule)
        structured_specs.append(spec)
    
    # 合并所有spec
    merged_spec = merge_structured_specs(structured_specs)
    
    # 生成结构化的[Clarified Requirements]文本
    requirements_lines = []
    
    if merged_spec["edge_cases"]:
        requirements_lines.append("Edge cases:")
        for edge_case in merged_spec["edge_cases"]:
            requirements_lines.append(f"- {edge_case}")
        requirements_lines.append("")
    
    if merged_spec["output_format"]:
        requirements_lines.append("Output format:")
        for output_spec in merged_spec["output_format"]:
            requirements_lines.append(f"- {output_spec}")
        requirements_lines.append("")
    
    if merged_spec["constraints"]:
        requirements_lines.append("Constraints:")
        for constraint in merged_spec["constraints"]:
            requirements_lines.append(f"- {constraint}")
        requirements_lines.append("")
    
    if merged_spec["input_constraints"]:
        requirements_lines.append("Input constraints:")
        for input_constraint in merged_spec["input_constraints"]:
            requirements_lines.append(f"- {input_constraint}")
        requirements_lines.append("")
    
    clarified_requirements = "\n".join(requirements_lines).strip()
    
    return {
        "original_query": original_query,
        "clarified_requirements": clarified_requirements,
        "has_clarifications": len(user_answers) > 0
    }
