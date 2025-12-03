"""
计算task_uncertainty（任务不确定性/清晰度）

根据输入（query）进行打分，评估任务的清晰度。
返回值范围：0.0-1.0
- 0.0-0.3: 非常不清晰，需要多个澄清问题（HIGH action）
- 0.3-0.7: 中等清晰度，可能需要一个澄清问题（MID action）
- 0.7-1.0: 非常清晰，可以直接生成代码（LOW action）
"""
from typing import Dict


def compute_task_uncertainty(query: str) -> float:
    """
    基于query文本的启发式规则评估任务不确定性/清晰度。
    
    Args:
        query: 用户的任务描述
        
    Returns:
        task_uncertainty: 0.0-1.0，值越小表示越不清晰（不确定性越高）
    """
    if not query or len(query.strip()) == 0:
        return 0.2  # 空query，非常不清晰
    
    query_lower = query.lower()
    uncertainty = 0.5  # 默认值
    
    # 1. 长度检查
    query_len = len(query)
    if 200 <= query_len <= 1500:
        uncertainty += 0.1  # 合适的长度，更清晰
    elif query_len < 100:
        uncertainty -= 0.2  # 太短，可能不清晰
    elif query_len > 3000:
        uncertainty -= 0.1  # 太长，可能包含太多信息，不够聚焦
    
    # 2. 问号检查（包含问号可能表示不清晰）
    question_count = query.count("?")
    if question_count > 0:
        uncertainty -= 0.1 * min(question_count, 3)
    
    # 3. 模糊词检查
    ambiguous_words = ["maybe", "perhaps", "could", "might", "uncertain", "not sure", "not clear", "unclear"]
    ambiguous_count = sum(1 for word in ambiguous_words if word in query_lower)
    if ambiguous_count > 0:
        uncertainty -= 0.15 * min(ambiguous_count, 2)
    
    # 4. 具体性检查（包含技术细节表示更清晰）
    specific_indicators = [
        "def ", "function", "import ", "class ",  # 代码相关
        "should output", "should return", "should raise",  # 输出规范
        "test", "test case", "example",  # 测试相关
        "parameter", "argument", "input", "output",  # 参数相关
        "error", "exception", "traceback",  # 错误信息（通常表示问题明确）
    ]
    specific_count = sum(1 for indicator in specific_indicators if indicator in query_lower)
    if specific_count >= 3:
        uncertainty += 0.2  # 非常具体
    elif specific_count >= 1:
        uncertainty += 0.1  # 有一定具体性
    
    # 5. 完整性检查
    # 包含错误信息通常表示任务更清晰（用户知道问题在哪里）
    if "error" in query_lower or "traceback" in query_lower or "exception" in query_lower:
        uncertainty += 0.1
    
    # 6. 包含测试用例或示例
    if "test" in query_lower and ("case" in query_lower or "example" in query_lower):
        uncertainty += 0.1
    
    # 7. 包含明确的输入输出格式
    if "input" in query_lower and "output" in query_lower:
        uncertainty += 0.1
    
    # 确保返回值在0.0-1.0范围内
    uncertainty = max(0.0, min(1.0, uncertainty))
    
    return uncertainty


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


