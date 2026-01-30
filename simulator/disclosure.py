"""
Disclosure Rule Module

实现disclosure rule：只有当assistant ASK clarification时，用户（模拟器）才会补充被mask的信息。

Key principle:
- assistant = EXECUTE → 用户不会主动给缺失信息
- assistant = ASK → user simulator从被mask的字段中选相关项，给出补充说明
"""
from typing import Dict, List, Optional
import random


def get_disclosure_info(
    assistant_question: str,
    disclosure_rule: Dict,
    expertise: str = "mid",
    dialogue_turn: int = 0
) -> Optional[str]:
    """
    根据assistant的问题，从disclosure_rule中提取相关的被mask信息
    
    实现渐进式披露机制（更慢的披露速度，让对话持续更久）:
    - Novice-Learner: 每次固定1个信息点（始终不变）
    - Busy-Developer: Turn 0-2: 每次1个, Turn 3+: 2个
    - Experienced-Engineer: Turn 0-1: 每次1个, Turn 2: 2个, Turn 3+: 2个
    - 这样Turn 0-2累计：Novice 3个, Busy 3个, Experienced 4个
    - 预期能看到更多Turn 3+的对话
    
    Args:
        assistant_question: Assistant的澄清问题
        disclosure_rule: 包含masked_fields和disclosure_info的字典
        expertise: 用户专业水平，影响披露步长K
        dialogue_turn: 当前对话轮次（0-indexed），用于逐步披露
        
    Returns:
        disclosure_text: 补充信息的文本，如果没有相关信息则返回None
    """
    if not disclosure_rule:
        return None
    
    disclosure_info = disclosure_rule.get("disclosure_info", {})
    if not disclosure_info:
        return None
    
    # 渐进式披露机制（更慢的披露速度，让对话持续更久）
    # 策略：进一步减少前期披露数量，确保需要更多轮才能获得足够信息
    # - Novice-Learner: 每次固定1个信息点（始终不变）
    # - Busy-Developer: Turn 0-2: 每次1个, Turn 3+: 2个
    # - Experienced-Engineer: Turn 0-1: 每次1个, Turn 2: 2个, Turn 3+: 2个
    # 这样Turn 0-2累计：Novice 3个, Busy 3个, Experienced 4个
    # 预期能看到更多Turn 3+的对话
    import math
    
    if expertise == "low":
        # Novice-Learner: 每次固定1个
        max_points = 1
    elif expertise == "mid":
        # Busy-Developer: Turn 0-2: 每次1个, Turn 3+: 2个
        if dialogue_turn <= 2:
            max_points = 1
        else:  # dialogue_turn >= 3
            max_points = 2
    else:  # expertise == "high"
        # Experienced-Engineer: Turn 0-1: 每次1个, Turn 2: 2个, Turn 3+: 2个
        if dialogue_turn <= 1:
            max_points = 1
        elif dialogue_turn == 2:
            max_points = 2
        else:  # dialogue_turn >= 3
            max_points = 2
    
    question_lower = assistant_question.lower()
    disclosure_parts = []
    info_points_count = 0
    
    # 收集所有可用的信息点（按类别）
    available_info_points = []
    
    # 输入约束相关的信息点
    input_keywords = ["input", "empty", "negative", "zero", "null", "none", "constraint", "range", "value", "default"]
    if any(keyword in question_lower for keyword in input_keywords):
        input_constraints = disclosure_info.get("input_constraints", {})
        if input_constraints:
            edge_cases = input_constraints.get("edge_cases", [])
            hints = input_constraints.get("hints", [])
            masked_fields = disclosure_rule.get("masked_fields", {}).get("input_constraints", [])
            
            # 将edge_cases和hints作为信息点
            if edge_cases:
                for i, ec in enumerate(edge_cases[:max_points]):
                    if expertise == "high":
                        available_info_points.append(f"Edge case: {ec}")
                    elif expertise == "low":
                        available_info_points.append("可能需要处理一些特殊情况。")
                        break  # 新手只说一个模糊的点
                    else:
                        available_info_points.append(f"Should handle: {ec}")
            
            # 如果有masked的input_constraints，也作为信息点
            if masked_fields and info_points_count < max_points:
                for constraint in masked_fields[:max_points - info_points_count]:
                    if expertise == "high":
                        available_info_points.append(f"Input constraint: {constraint}")
                    elif expertise == "low":
                        available_info_points.append("输入有默认值。")
                        break
                    else:
                        available_info_points.append(f"Note: {constraint}")
    
    # 输出格式相关的信息点
    output_keywords = ["output", "return", "format", "type", "result", "should return"]
    if any(keyword in question_lower for keyword in output_keywords):
        output_format = disclosure_info.get("output_format", {})
        if output_format:
            spec = output_format.get("specification", "")
            masked_fields = disclosure_rule.get("masked_fields", {}).get("output_format", [])
            
            if spec:
                if expertise == "high":
                    available_info_points.append(f"Output specification: {spec}")
                elif expertise == "low":
                    available_info_points.append("需要返回正确格式的结果。")
                else:
                    # 提取关键部分
                    if "dict" in spec.lower():
                        available_info_points.append("Should return a dictionary.")
                    elif "list" in spec.lower():
                        available_info_points.append("Should return a list.")
                    elif "float" in spec.lower() or "int" in spec.lower():
                        available_info_points.append("Should return a number.")
            
            # 如果有masked的output_format，也作为信息点
            if masked_fields and len(available_info_points) < max_points:
                for fmt in masked_fields[:max_points - len(available_info_points)]:
                    if expertise == "high":
                        available_info_points.append(f"Output format: {fmt}")
                    elif expertise == "low":
                        available_info_points.append("输出有特定格式要求。")
                        break
                    else:
                        available_info_points.append(f"Output: {fmt}")
    
    # 验证规则相关的信息点
    validation_keywords = ["error", "exception", "validate", "check", "raise"]
    if any(keyword in question_lower for keyword in validation_keywords):
        validation_rules = disclosure_info.get("validation_rules", {})
        if validation_rules:
            rules = validation_rules.get("rules", [])
            if rules:
                if expertise == "high":
                    available_info_points.append(f"Validation: {rules[0]}")
                else:
                    available_info_points.append("需要处理错误情况。")
    
    # 如果没有匹配的问题类别，按顺序从masked_fields中提取
    if not available_info_points:
        masked_fields = disclosure_rule.get("masked_fields", {})
        # 按顺序提取：input_constraints, output_format, edge_cases
        if masked_fields.get("input_constraints"):
            constraint = masked_fields["input_constraints"][0]
            if expertise == "high":
                available_info_points.append(f"Input constraint: {constraint}")
            elif expertise == "low":
                available_info_points.append("输入有默认值。")
            else:
                available_info_points.append(f"Note: {constraint}")
        
        if masked_fields.get("output_format") and len(available_info_points) < max_points:
            fmt = masked_fields["output_format"][0]
            if expertise == "high":
                available_info_points.append(f"Output format: {fmt}")
            elif expertise == "low":
                available_info_points.append("输出有特定格式要求。")
            else:
                available_info_points.append(f"Output: {fmt}")
    
    # 限制信息点数量为max_points
    selected_points = available_info_points[:max_points]
    
    if selected_points:
        return " ".join(selected_points)
    
    return None


def generate_answer_with_disclosure(
    assistant_question: str,
    user_query: str,
    disclosure_rule: Optional[Dict],
    expertise: str = "mid",
    base_answer: Optional[str] = None,
    dialogue_turn: int = 0
) -> str:
    """
    生成包含disclosure信息的回答
    
    实现渐进式披露机制（更慢的披露速度，让对话持续更久）:
    - Novice-Learner: 每次固定1个信息点（始终不变）
    - Busy-Developer: Turn 0-2: 每次1个, Turn 3+: 2个
    - Experienced-Engineer: Turn 0-1: 每次1个, Turn 2: 2个, Turn 3+: 2个
    - 这样Turn 0-2累计：Novice 3个, Busy 3个, Experienced 4个
    - 预期能看到更多Turn 3+的对话
    
    Args:
        assistant_question: Assistant的澄清问题
        user_query: 原始用户查询
        disclosure_rule: disclosure rule字典
        expertise: 用户专业水平，影响披露步长K
        base_answer: 基础回答（如果没有disclosure rule时的默认回答）
        dialogue_turn: 当前对话轮次（0-indexed），用于逐步披露
        
    Returns:
        answer: 包含disclosure信息的回答
    """
    # 首先尝试从disclosure_rule获取信息
    disclosure_info = None
    if disclosure_rule:
        disclosure_info = get_disclosure_info(assistant_question, disclosure_rule, expertise, dialogue_turn)
    
    if disclosure_info:
        # 如果有disclosure信息，将其整合到回答中
        if base_answer:
            return f"{base_answer} {disclosure_info}"
        else:
            return disclosure_info
    
    # 如果没有disclosure信息，返回基础回答或默认回答
    if base_answer:
        return base_answer
    
    # 默认回答（基于expertise）
    if expertise == "low":
        return "可能是这样的吧，我也不太确定。"
    elif expertise == "high":
        return "需要根据具体情况处理。"
    else:
        return "好的。"


