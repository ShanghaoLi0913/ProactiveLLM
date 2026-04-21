"""
计算task_uncertainty（任务不确定性）

基于query文本的启发式评估任务不确定性。
返回值范围：0.0-1.0
- 0.0-0.3: 非常清晰，可以直接生成代码
- 0.3-0.7: 中等不确定性，可能需要澄清
- 0.7-1.0: 非常不清晰，需要多个澄清问题

[2026-04-22 修正] v29 版本内部逻辑实际是 clarity（与函数名相反），
下游 reward 代码的语义注释（"高 uncertainty → prefer Clarify"）一直
在反向运行。此版翻正：低 U = 清晰，高 U = 不确定。
"""
from typing import Dict


def compute_task_uncertainty(query: str) -> float:
    """
    基于query文本的启发式规则评估任务不确定性。

    语义：返回值越大表示越不确定 / 需要越多澄清。

    Args:
        query: 用户的任务描述（应该只包含初始query，不包含对话历史）

    Returns:
        task_uncertainty: 0.0-1.0，值越大表示越不确定。
    """
    if not query or len(query.strip()) == 0:
        return 0.8  # 空query，非常不确定

    query_lower = query.lower()
    # 启发式先算 clarity-style 得分，最后翻转为 uncertainty。
    # 保持原有加减号与因子，只在返回时 1 - x，便于 diff/回溯。
    clarity = 0.5

    # 1. 长度检查
    query_len = len(query)
    if 200 <= query_len <= 1500:
        clarity += 0.1  # 合适的长度，更清晰
    elif query_len < 100:
        clarity -= 0.2  # 太短，可能不清晰
    elif query_len > 3000:
        clarity -= 0.1  # 太长，可能包含太多信息，不够聚焦

    # 2. 问号检查（包含问号可能表示不清晰）
    question_count = query.count("?")
    if question_count > 0:
        clarity -= 0.1 * min(question_count, 3)

    # 3. 模糊词检查
    ambiguous_words = ["maybe", "perhaps", "could", "might", "uncertain",
                       "not sure", "not clear", "unclear"]
    ambiguous_count = sum(1 for word in ambiguous_words if word in query_lower)
    if ambiguous_count > 0:
        clarity -= 0.15 * min(ambiguous_count, 2)

    # 4. 具体性检查（包含技术细节表示更清晰）
    specific_indicators = [
        "def ", "function", "import ", "class ",
        "should output", "should return", "should raise",
        "test", "test case", "example",
        "parameter", "argument", "input", "output",
        "error", "exception", "traceback",
    ]
    specific_count = sum(1 for ind in specific_indicators if ind in query_lower)
    if specific_count >= 3:
        clarity += 0.2
    elif specific_count >= 1:
        clarity += 0.1

    # 5. 错误信息 / traceback 通常意味着问题很明确
    if "error" in query_lower or "traceback" in query_lower or "exception" in query_lower:
        clarity += 0.1

    # 6. 测试用例 / 示例
    if "test" in query_lower and ("case" in query_lower or "example" in query_lower):
        clarity += 0.1

    # 7. 明确输入输出格式
    if "input" in query_lower and "output" in query_lower:
        clarity += 0.1

    clarity = max(0.0, min(1.0, clarity))
    return 1.0 - clarity


def compute_task_uncertainty_from_state(state: Dict) -> float:
    """
    从state中提取query并计算task_uncertainty。
    
    Args:
        state: 包含query字段的state字典
        
    Returns:
        task_uncertainty: 0.0-1.0
    """
    query = state.get("query", "")
    return compute_task_uncertainty(query)


