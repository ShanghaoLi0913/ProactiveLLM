"""
统一的state渲染函数，确保训练和评估使用完全相同的prompt格式。

关键原则：
1. 只包含纯state信息，不包含任何action_prompt或模板内容
2. 让模型自由判断：要问几个问题、怎么问、是否该问
3. 训练和评估必须使用完全相同的render_state函数
"""
from typing import Dict


def render_state(state: Dict) -> str:
    """
    将state转换为纯文本prompt，不包含任何action指令或模板。
    
    格式示例：
    [Domain] coding
    [Turn] 1
    [User Clarity] 0.3
    [Prev Reject] 1
    [Task] write a python script that scrapes data...
    
    Args:
        state: 包含domain, dialogue_turn, query_clarity, prev_reject, query等字段的字典
        
    Returns:
        格式化的纯state文本，不包含任何action指令
    """
    domain = state.get("domain", "coding")
    turn = state.get("dialogue_turn", 1)
    clarity = state.get("query_clarity", 0.5)
    prev_reject = state.get("prev_reject", 0)
    query = state.get("query", "")
    
    # 可选字段
    task_complexity = state.get("task_complexity")
    
    # 构建prompt - 使用清晰的格式，不包含任何action指令
    lines = [
        f"[Domain] {domain}",
        f"[Turn] {turn}",
        f"[User Clarity] {clarity:.2f}",
        f"[Prev Reject] {prev_reject}",
    ]
    
    if task_complexity is not None:
        lines.append(f"[Task Complexity] {task_complexity:.2f}")
    
    lines.append(f"[Task]\n{query}")
    
    return "\n".join(lines)

