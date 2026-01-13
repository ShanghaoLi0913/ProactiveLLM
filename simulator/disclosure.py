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
    expertise: str = "mid"
) -> Optional[str]:
    """
    根据assistant的问题，从disclosure_rule中提取相关的被mask信息
    
    Args:
        assistant_question: Assistant的澄清问题
        disclosure_rule: 包含masked_fields和disclosure_info的字典
        expertise: 用户专业水平，影响回答的清晰度
        
    Returns:
        disclosure_text: 补充信息的文本，如果没有相关信息则返回None
    """
    if not disclosure_rule:
        return None
    
    disclosure_info = disclosure_rule.get("disclosure_info", {})
    if not disclosure_info:
        return None
    
    question_lower = assistant_question.lower()
    disclosure_parts = []
    
    # 检查问题是否涉及输入约束
    input_keywords = ["input", "empty", "negative", "zero", "null", "none", "constraint", "range", "value"]
    if any(keyword in question_lower for keyword in input_keywords):
        input_constraints = disclosure_info.get("input_constraints", {})
        if input_constraints:
            edge_cases = input_constraints.get("edge_cases", [])
            hints = input_constraints.get("hints", [])
            
            if edge_cases:
                if expertise == "high":
                    # 专家：详细说明
                    disclosure_parts.append(f"Edge cases to handle: {', '.join(edge_cases)}")
                elif expertise == "low":
                    # 新手：模糊回答
                    disclosure_parts.append("可能需要处理一些特殊情况。")
                else:
                    # 中等：部分信息
                    if edge_cases:
                        disclosure_parts.append(f"Should handle: {edge_cases[0]}")
    
    # 检查问题是否涉及输出格式
    output_keywords = ["output", "return", "format", "type", "result", "should return"]
    if any(keyword in question_lower for keyword in output_keywords):
        output_format = disclosure_info.get("output_format", {})
        if output_format:
            spec = output_format.get("specification", "")
            if spec:
                if expertise == "high":
                    disclosure_parts.append(f"Output specification: {spec}")
                elif expertise == "low":
                    disclosure_parts.append("需要返回正确格式的结果。")
                else:
                    # 提取关键部分
                    if "dict" in spec.lower():
                        disclosure_parts.append("Should return a dictionary.")
                    elif "list" in spec.lower():
                        disclosure_parts.append("Should return a list.")
                    elif "float" in spec.lower() or "int" in spec.lower():
                        disclosure_parts.append("Should return a number.")
    
    # 检查问题是否涉及验证规则
    validation_keywords = ["error", "exception", "validate", "check", "raise"]
    if any(keyword in question_lower for keyword in validation_keywords):
        validation_rules = disclosure_info.get("validation_rules", {})
        if validation_rules:
            rules = validation_rules.get("rules", [])
            if rules:
                if expertise == "high":
                    disclosure_parts.append(f"Validation: {rules[0]}")
                else:
                    disclosure_parts.append("需要处理错误情况。")
    
    if disclosure_parts:
        return " ".join(disclosure_parts)
    
    return None


def generate_answer_with_disclosure(
    assistant_question: str,
    user_query: str,
    disclosure_rule: Optional[Dict],
    expertise: str = "mid",
    base_answer: Optional[str] = None
) -> str:
    """
    生成包含disclosure信息的回答
    
    Args:
        assistant_question: Assistant的澄清问题
        user_query: 原始用户查询
        disclosure_rule: disclosure rule字典
        expertise: 用户专业水平
        base_answer: 基础回答（如果没有disclosure rule时的默认回答）
        
    Returns:
        answer: 包含disclosure信息的回答
    """
    # 首先尝试从disclosure_rule获取信息
    disclosure_info = None
    if disclosure_rule:
        disclosure_info = get_disclosure_info(assistant_question, disclosure_rule, expertise)
    
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


