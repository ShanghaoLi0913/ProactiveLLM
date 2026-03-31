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
    dialogue_turn: int = 0,  # 保留参数但不使用（向后兼容）
    n_questions: int = 1  # 新增参数：assistant问的问题数量
) -> tuple[Optional[str], Dict]:
    """
    根据assistant的问题，从disclosure_rule中提取相关的被mask信息
    
    简化版披露机制：
    - 根据assistant问的问题数量决定披露数量
    - Novice: 最多1个（即使问了多个问题，也只回答1个）
    - Busy/Experienced: 回答所有问题（通常1-2个）
    
    改进：
    - 检查disclosed_info，避免重复披露相同信息
    - 移除dialogue_turn依赖，简化逻辑
    
    Args:
        assistant_question: Assistant的澄清问题
        disclosure_rule: 包含masked_fields、disclosure_info和disclosed_info的字典
        expertise: 用户专业水平，影响最大允许披露数
        dialogue_turn: 当前对话轮次（保留但不使用，向后兼容）
        n_questions: Assistant问的问题数量（通常1-2个）
        
    Returns:
        (disclosure_text, disclosed_items): 
            - disclosure_text: 补充信息的文本，如果没有相关信息则返回None
            - disclosed_items: 本次披露的信息项（用于更新disclosed_info），格式：
              {
                  "edge_cases": [...],
                  "input_constraints": [...],
                  "output_format": [...],
                  "validation_rules": [...]
              }
    """
    if not disclosure_rule:
        return None, {}
    
    disclosure_info = disclosure_rule.get("disclosure_info", {})
    if not disclosure_info:
        return None, {}
    
    # 获取已披露信息，避免重复披露
    disclosed_info = disclosure_rule.get("disclosed_info", {})
    disclosed_edge_cases = set(disclosed_info.get("edge_cases", []))
    disclosed_input_constraints = set(disclosed_info.get("input_constraints", []))
    disclosed_output_format = set(disclosed_info.get("output_format", []))
    disclosed_validation_rules = set(disclosed_info.get("validation_rules", []))
    
    # 用于记录本次披露的信息项
    new_disclosed_items = {
        "edge_cases": [],
        "input_constraints": [],
        "output_format": [],
        "validation_rules": [],
    }
    
    # 简化版披露机制：根据问题数量和expertise限制
    # - Novice: 最多1个（即使问了多个问题，也只回答1个）
    # - Busy/Experienced: 回答所有问题（通常1-2个）
    if expertise == "low":
        # Novice-Learner: 最多1个
        max_allowed = 1
    else:
        # Busy-Developer 和 Experienced-Engineer: 回答所有问题
        # 设置一个合理的上限（比如10），防止异常情况
        max_allowed = min(n_questions, 10)
    
    # 根据问题数量和expertise限制，取较小值
    max_points = min(n_questions, max_allowed)
    
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
            
            # 将edge_cases和hints作为信息点（只选择未披露的）
            if edge_cases:
                for ec in edge_cases:
                    if len(available_info_points) >= max_points:
                        break
                    # 检查是否已披露
                    if ec not in disclosed_edge_cases:
                        if expertise == "high":
                            available_info_points.append(f"Edge case: {ec}")
                            new_disclosed_items["edge_cases"].append(ec)
                        elif expertise == "low":
                            # Novice: give a vague point, but only once
                            if not new_disclosed_items["edge_cases"]:
                                available_info_points.append("You might need to handle some special cases.")
                                new_disclosed_items["edge_cases"].append(ec)  # Record the actual edge case disclosed
                            break  # Novice only gives one vague point
                        else:
                            available_info_points.append(f"Should handle: {ec}")
                            new_disclosed_items["edge_cases"].append(ec)
            
            # 如果有masked的input_constraints，也作为信息点（只选择未披露的）
            if masked_fields and len(available_info_points) < max_points:
                for constraint in masked_fields:
                    if len(available_info_points) >= max_points:
                        break
                    # 检查是否已披露（使用字符串匹配）
                    constraint_str = str(constraint)
                    if constraint_str not in disclosed_input_constraints:
                        if expertise == "high":
                            available_info_points.append(f"Input constraint: {constraint}")
                            new_disclosed_items["input_constraints"].append(constraint_str)
                        elif expertise == "low":
                            if not new_disclosed_items["input_constraints"]:
                                available_info_points.append("The input has default values.")
                                new_disclosed_items["input_constraints"].append(constraint_str)
                            break
                        else:
                            available_info_points.append(f"Note: {constraint}")
                            new_disclosed_items["input_constraints"].append(constraint_str)
    
    # 输出格式相关的信息点
    output_keywords = ["output", "return", "format", "type", "result", "should return"]
    if any(keyword in question_lower for keyword in output_keywords):
        output_format = disclosure_info.get("output_format", {})
        if output_format:
            spec = output_format.get("specification", "")
            masked_fields = disclosure_rule.get("masked_fields", {}).get("output_format", [])
            
            # 检查spec是否已披露（使用spec的关键部分作为标识）
            if spec and len(available_info_points) < max_points:
                spec_key = spec[:50]  # 使用前50个字符作为标识
                if spec_key not in disclosed_output_format:
                    if expertise == "high":
                        available_info_points.append(f"Output specification: {spec}")
                        new_disclosed_items["output_format"].append(spec_key)
                    elif expertise == "low":
                        if not new_disclosed_items["output_format"]:
                            available_info_points.append("You need to return results in the correct format.")
                            new_disclosed_items["output_format"].append(spec_key)
                    else:
                        # 提取关键部分
                        if "dict" in spec.lower():
                            available_info_points.append("Should return a dictionary.")
                            new_disclosed_items["output_format"].append(spec_key)
                        elif "list" in spec.lower():
                            available_info_points.append("Should return a list.")
                            new_disclosed_items["output_format"].append(spec_key)
                        elif "float" in spec.lower() or "int" in spec.lower():
                            available_info_points.append("Should return a number.")
                            new_disclosed_items["output_format"].append(spec_key)
            
            # 如果有masked的output_format，也作为信息点（只选择未披露的）
            if masked_fields and len(available_info_points) < max_points:
                for fmt in masked_fields:
                    if len(available_info_points) >= max_points:
                        break
                    fmt_str = str(fmt)
                    if fmt_str not in disclosed_output_format:
                        if expertise == "high":
                            available_info_points.append(f"Output format: {fmt}")
                            new_disclosed_items["output_format"].append(fmt_str)
                        elif expertise == "low":
                            if not new_disclosed_items["output_format"]:
                                available_info_points.append("The output has specific format requirements.")
                                new_disclosed_items["output_format"].append(fmt_str)
                            break
                        else:
                            available_info_points.append(f"Output: {fmt}")
                            new_disclosed_items["output_format"].append(fmt_str)
    
    # 验证规则相关的信息点
    validation_keywords = ["error", "exception", "validate", "check", "raise"]
    if any(keyword in question_lower for keyword in validation_keywords):
        validation_rules = disclosure_info.get("validation_rules", {})
        if validation_rules:
            rules = validation_rules.get("rules", [])
            if rules and len(available_info_points) < max_points:
                rule_str = str(rules[0])
                if rule_str not in disclosed_validation_rules:
                    if expertise == "high":
                        available_info_points.append(f"Validation: {rules[0]}")
                        new_disclosed_items["validation_rules"].append(rule_str)
                    else:
                        if not new_disclosed_items["validation_rules"]:
                            available_info_points.append("You need to handle error cases.")
                            new_disclosed_items["validation_rules"].append(rule_str)
    
    # 如果没有匹配的问题类别，按顺序从masked_fields中提取（只选择未披露的）
    if not available_info_points:
        masked_fields = disclosure_rule.get("masked_fields", {})
        # 按顺序提取：input_constraints, output_format, edge_cases
        if masked_fields.get("input_constraints"):
            for constraint in masked_fields["input_constraints"]:
                constraint_str = str(constraint)
                if constraint_str not in disclosed_input_constraints:
                    if expertise == "high":
                        available_info_points.append(f"Input constraint: {constraint}")
                        new_disclosed_items["input_constraints"].append(constraint_str)
                    elif expertise == "low":
                        if not new_disclosed_items["input_constraints"]:
                            available_info_points.append("The input has default values.")
                            new_disclosed_items["input_constraints"].append(constraint_str)
                    else:
                        available_info_points.append(f"Note: {constraint}")
                        new_disclosed_items["input_constraints"].append(constraint_str)
                    break  # 只取第一个未披露的
        
        if masked_fields.get("output_format") and len(available_info_points) < max_points:
            for fmt in masked_fields["output_format"]:
                fmt_str = str(fmt)
                if fmt_str not in disclosed_output_format:
                    if expertise == "high":
                        available_info_points.append(f"Output format: {fmt}")
                        new_disclosed_items["output_format"].append(fmt_str)
                    elif expertise == "low":
                        if not new_disclosed_items["output_format"]:
                            available_info_points.append("The output has specific format requirements.")
                            new_disclosed_items["output_format"].append(fmt_str)
                    else:
                        available_info_points.append(f"Output: {fmt}")
                        new_disclosed_items["output_format"].append(fmt_str)
                    break  # 只取第一个未披露的
    
    # 限制信息点数量为max_points
    selected_points = available_info_points[:max_points]
    
    if selected_points:
        disclosure_text = " ".join(selected_points)
        return disclosure_text, new_disclosed_items
    
    return None, {}


def generate_answer_with_disclosure(
    assistant_question: str,
    user_query: str,
    disclosure_rule: Optional[Dict],
    expertise: str = "mid",
    base_answer: Optional[str] = None,
    dialogue_turn: int = 0
) -> tuple[str, Dict]:
    """
    生成包含disclosure信息的回答
    
    简化版披露机制：
    - 根据assistant问的问题数量决定披露数量
    - Novice: 最多1个（即使问了多个问题，也只回答1个）
    - Busy/Experienced: 回答所有问题（通常1-2个）
    
    改进：返回本次披露的信息项，用于更新disclosed_info
    
    Args:
        assistant_question: Assistant的澄清问题
        user_query: 原始用户查询
        disclosure_rule: disclosure rule字典
        expertise: 用户专业水平，影响最大允许披露数
        base_answer: 基础回答（如果没有disclosure rule时的默认回答）
        dialogue_turn: 当前对话轮次（保留但不使用，向后兼容）
        
    Returns:
        (answer, disclosed_items): 
            - answer: 包含disclosure信息的回答
            - disclosed_items: 本次披露的信息项（用于更新disclosed_info）
    """
    # 统计assistant问的问题数量
    n_questions = assistant_question.count("?")
    # 至少为1（如果没有问号，可能是陈述句，但通常clarify会有问号）
    n_questions = max(1, n_questions)
    
    # 首先尝试从disclosure_rule获取信息
    disclosure_info = None
    disclosed_items = {}
    if disclosure_rule:
        disclosure_info, disclosed_items = get_disclosure_info(
            assistant_question, 
            disclosure_rule, 
            expertise, 
            dialogue_turn,
            n_questions=n_questions  # 传递问题数量
        )
    
    if disclosure_info:
        # 如果有disclosure信息，将其整合到回答中
        if base_answer:
            answer = f"{base_answer} {disclosure_info}"
        else:
            answer = disclosure_info
        return answer, disclosed_items
    
    # 如果没有disclosure信息，返回基础回答或默认回答
    if base_answer:
        return base_answer, {}
    
    # Default answer (based on expertise)
    if expertise == "low":
        return "Maybe that's the case, I'm not quite sure.", {}
    elif expertise == "high":
        return "It depends on the specific situation.", {}
    else:
        return "Okay.", {}


