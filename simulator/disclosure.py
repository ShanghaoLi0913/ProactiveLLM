"""
Disclosure Rule Module

实现 disclosure rule：只有当 assistant Clarify 时，用户（模拟器）才会补充被 mask 的信息。

Key principle:
- assistant = EXECUTE → 用户不主动给缺失信息
- assistant = CLARIFY → user simulator 从 masked_fields 中选相关项，按关键词匹配披露

披露优先级（v29）：
1. output_format    — 返回值类型/结构，最关键
2. validation_rules — 异常处理要求
3. note_that        — 特殊规则/命名约定
"""
from typing import Dict, List, Optional, Tuple


def get_disclosure_info(
    assistant_question: str,
    disclosure_rule: Dict,
    expertise: str = "mid",
    dialogue_turn: int = 0,   # 保留参数，向后兼容
    n_questions: int = 1,
) -> Tuple[Optional[str], Dict]:
    """
    根据 assistant 的问题，从 disclosure_rule 中提取相关被 mask 信息。

    Args:
        assistant_question: assistant 的澄清问题文本
        disclosure_rule:    包含 masked_fields、disclosure_info、disclosed_info 的字典
        expertise:          用户专业水平 ("low" / "mid" / "high")
        dialogue_turn:      当前对话轮次（保留，不使用）
        n_questions:        assistant 问的问题数量

    Returns:
        (disclosure_text, new_disclosed_items)
        - disclosure_text:     本次披露的文本，None 表示无匹配
        - new_disclosed_items: 本次新披露的内容，用于更新 disclosed_info
    """
    if not disclosure_rule:
        return None, {}

    masked_fields = disclosure_rule.get("masked_fields", {})
    disclosed_info = disclosure_rule.get("disclosed_info", {})

    # 已披露内容（避免重复）
    disclosed_output_format    = set(disclosed_info.get("output_format", []))
    disclosed_validation_rules = set(disclosed_info.get("validation_rules", []))
    disclosed_note_that        = set(disclosed_info.get("note_that", []))

    new_disclosed_items = {
        "output_format":    [],
        "validation_rules": [],
        "note_that":        [],
    }

    # 披露上限：expertise 越高给越多
    # low (Novice):      1条/轮 — 认知负担高，分步消化
    # mid (Busy):        3条/轮 — 时间有限，给够用的
    # high (Experienced):6条/轮 — 能处理完整技术细节
    if expertise == "low":
        max_points = 1
    elif expertise == "mid":
        max_points = 3
    else:
        max_points = 6

    question_lower = assistant_question.lower()
    available = []  # 收集本次可披露的信息点

    # ── 优先级 1：output_format ───────────────────────────────────────────
    output_keywords = [
        "output", "return", "format", "type", "result", "should return",
        "give", "produce", "expect", "what", "structure", "shape", "form",
        "value", "yield",
    ]
    if any(kw in question_lower for kw in output_keywords):
        for item in masked_fields.get("output_format", []):
            if len(available) >= max_points:
                break
            item_str = str(item)
            if item_str not in disclosed_output_format:
                if expertise == "high":
                    available.append(f"Output format: {item_str}")
                elif expertise == "low":
                    available.append(f"Output should be: {item_str}")
                else:
                    available.append(f"Output: {item_str}")
                new_disclosed_items["output_format"].append(item_str)
                if expertise == "low":
                    break  # Novice 每轮只披露 1 条

    # ── 优先级 2：validation_rules ────────────────────────────────────────
    validation_keywords = [
        "error", "exception", "raise", "handle", "fail", "throw",
        "invalid", "wrong", "edge", "case", "what if", "when",
    ]
    if any(kw in question_lower for kw in validation_keywords):
        for item in masked_fields.get("validation_rules", []):
            if len(available) >= max_points:
                break
            item_str = str(item)
            if item_str not in disclosed_validation_rules:
                if expertise == "high":
                    available.append(f"Exception: {item_str}")
                elif expertise == "low":
                    available.append(f"Error handling: {item_str}")
                else:
                    available.append(f"Raises: {item_str}")
                new_disclosed_items["validation_rules"].append(item_str)
                if expertise == "low":
                    break

    # ── 优先级 3：note_that ───────────────────────────────────────────────
    note_keywords = [
        "note", "special", "specific", "always", "fixed", "threshold",
        "rule", "requirement", "constraint", "name", "convention",
    ]
    if any(kw in question_lower for kw in note_keywords):
        for item in masked_fields.get("note_that", []):
            if len(available) >= max_points:
                break
            item_str = str(item)
            if item_str not in disclosed_note_that:
                if expertise == "high":
                    available.append(f"Note: {item_str}")
                elif expertise == "low":
                    available.append(f"Important: {item_str}")
                else:
                    available.append(f"Note: {item_str}")
                new_disclosed_items["note_that"].append(item_str)
                if expertise == "low":
                    break

    # ── 兜底：若无关键词命中，按优先级依次披露未披露的内容 ─────────────────
    if not available:
        fallback_order = [
            ("output_format",    disclosed_output_format,    "Output format: "),
            ("validation_rules", disclosed_validation_rules, "Raises: "),
            ("note_that",        disclosed_note_that,        "Note: "),
        ]
        for field_key, disclosed_set, prefix in fallback_order:
            if len(available) >= max_points:
                break
            for item in masked_fields.get(field_key, []):
                if len(available) >= max_points:
                    break
                item_str = str(item)
                if item_str not in disclosed_set:
                    available.append(f"{prefix}{item_str}")
                    new_disclosed_items[field_key].append(item_str)
                    if expertise == "low":
                        break

    selected = available[:max_points]
    if selected:
        return " ".join(selected), new_disclosed_items

    return None, {}


def generate_answer_with_disclosure(
    assistant_question: str,
    user_query: str,
    disclosure_rule: Optional[Dict],
    expertise: str = "mid",
    base_answer: Optional[str] = None,
    dialogue_turn: int = 0,
) -> Tuple[str, Dict]:
    """
    生成包含 disclosure 信息的用户回答。

    Returns:
        (answer, disclosed_items)
    """
    n_questions = max(1, assistant_question.count("?"))

    disclosure_text, disclosed_items = None, {}
    if disclosure_rule:
        disclosure_text, disclosed_items = get_disclosure_info(
            assistant_question,
            disclosure_rule,
            expertise,
            dialogue_turn,
            n_questions=n_questions,
        )

    if disclosure_text:
        answer = f"{base_answer} {disclosure_text}" if base_answer else disclosure_text
        return answer, disclosed_items

    if base_answer:
        return base_answer, {}

    # 无任何信息时的默认回答
    if expertise == "low":
        return "Maybe that's the case, I'm not quite sure.", {}
    elif expertise == "high":
        return "It depends on the specific situation.", {}
    else:
        return "Okay.", {}
