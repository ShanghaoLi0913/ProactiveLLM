"""
统一的state渲染函数，确保训练和评估使用完全相同的prompt格式。

序列决策问题（Sequential Decision Process）：
S_t = {task_uncertainty, dialogue_turn, prev_reject}

根据问题定义：
- 这是一个序列决策问题，助手在每轮对话t做出高层次决策
- 每轮决策：A_t ∈ {Clarify (提出澄清问题), Execute (执行任务)}
- 多轮对话持续直到任务完成
- 助手根据上下文状态动态调整主动性

关键原则：
1. 只包含纯state信息，不包含任何action_prompt或模板内容
2. 让模型自由判断：当前轮次应该澄清还是执行
3. State只包含任务相关信息，不包含persona信息（patience等）
4. 训练和评估必须使用完全相同的render_state函数

设计理念：
- 序列决策：需要dialogue_turn（当前是第几轮对话）
- 需要prev_reject（上一轮是否被用户拒绝）
- task_uncertainty从初始user request推断，帮助模型判断是否需要澄清
- 关键：task_uncertainty在决策时基于初始query计算，dialogue_turn和prev_reject反映对话历史
  （这是multi-step RL (sequential decision process)，每轮做一次决策）
"""
from typing import Dict
import sys
from pathlib import Path

# 导入task_uncertainty计算函数
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.compute_task_uncertainty import compute_task_uncertainty_from_state
except ImportError:
    # 如果导入失败，使用fallback
    def compute_task_uncertainty_from_state(state: Dict) -> float:
        query = state.get("query", "")
        # 简单的fallback计算
        if not query:
            return 0.2
        return 0.5  # 默认值


def render_state(state: Dict) -> str:
    """
    将state转换为纯文本prompt，不包含任何action指令或模板。
    
    序列决策问题格式：
    [Domain] coding
    [Task Uncertainty] 0.3
    [Dialogue Turn] 2
    [Previous Reject] 0
    [User Request]
    write a python script that scrapes data...
    
    注意：
    - State不包含persona信息（patience等）
    - 这是序列决策问题，需要dialogue_turn（当前是第几轮对话）
    - 需要prev_reject（上一轮是否被用户拒绝，0或1）
    - task_uncertainty基于初始query计算，帮助模型判断是否需要澄清
    
    Args:
        state: 包含以下字段的字典：
               - domain: 任务领域（如 "coding"）
               - query: 用户请求（可能包含对话历史）
               - task_uncertainty: 如果不存在，会根据初始query自动计算
               - dialogue_turn: 当前对话轮次（0, 1, 2, ...），默认为0
               - prev_reject: 上一轮是否被拒绝（0或1），默认为0
        
    Returns:
        格式化的纯state文本，不包含任何action指令
    """
    domain = state.get("domain", "coding")
    query = state.get("query", "")
    dialogue_turn = state.get("dialogue_turn", 0)
    prev_reject = state.get("prev_reject", 0)
    
    # 提取初始query（如果query包含对话历史，只使用原始部分）
    # task_uncertainty应该基于初始query计算，不会动态更新
    initial_query = query
    if "[Assistant]:" in query or "[User]:" in query:
        # 如果query包含对话历史，只提取原始部分（第一个[Assistant]:或[User]:之前的内容）
        lines = query.split("\n")
        initial_lines = []
        for line in lines:
            if line.strip().startswith("[Assistant]:") or line.strip().startswith("[User]:"):
                break
            initial_lines.append(line)
        initial_query = "\n".join(initial_lines).strip()
    
    # 基于初始query计算task_uncertainty（决策时固定）
    # 这样确保task_uncertainty有区分度，帮助模型判断，但不会在执行过程中改变
    temp_state = {"query": initial_query}
    task_uncertainty = compute_task_uncertainty_from_state(temp_state)
    
    # 构建prompt - 使用清晰的格式，不包含任何action指令
    # State只包含任务相关信息，不包含persona信息
    # 
    # 格式设计让LLM容易理解：
    # - task_uncertainty: 0.0-1.0，值越小表示越不清晰（需要更多澄清）
    # - dialogue_turn: 当前对话轮次，帮助模型判断是否应该继续澄清
    # - prev_reject: 上一轮是否被拒绝，帮助模型判断用户耐心
    # - query: 用户的请求（可能包含对话历史）
    lines = [
        f"[Domain] {domain}",
        f"[Task Uncertainty] {task_uncertainty:.2f}",  # 明确显示不确定性（0.0=非常不清晰，1.0=非常清晰）
        f"[Dialogue Turn] {dialogue_turn}",  # 当前对话轮次
        f"[Previous Reject] {prev_reject}",  # 上一轮是否被拒绝（0=否，1=是）
        f"[User Request]\n{query}",  # 用户的请求（可能包含对话历史）
    ]
    
    return "\n".join(lines)

