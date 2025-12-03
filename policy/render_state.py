"""
统一的state渲染函数，确保训练和评估使用完全相同的prompt格式。

State_version2设计（修正版）：
S = {task_uncertainty, dialogue_turn, prev_reject}

关键原则：
1. 只包含纯state信息，不包含任何action_prompt或模板内容
2. 让模型自由判断：要问几个问题、怎么问、是否该问
3. State只包含任务相关信息，不包含persona信息（patience等）
4. 模型需要从dialogue history（prev_reject, dialogue_turn）推断用户偏好
5. 训练和评估必须使用完全相同的render_state函数

设计理念：
- persona、state、action应该不重复
- 如果state里直接包含persona信息，项目就没意义了（太简单）
- 模型应该从dialogue signals推断用户偏好，这才是项目的意义
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
    
    State_version2格式（修正版）：
    [Domain] coding
    [Turn] 1
    [Task Uncertainty] 0.3
    [Prev Reject] 1
    [Task] write a python script that scrapes data...
    
    注意：State不包含persona信息（patience等）
    - 模型需要从dialogue history（prev_reject, dialogue_turn）推断用户偏好
    - 例如：prev_reject=1可能表示用户不耐烦
    - 例如：dialogue_turn>1可能表示用户有耐心（愿意继续对话）
    
    Args:
        state: 包含domain, dialogue_turn, task_uncertainty, prev_reject, query等字段的字典
               - task_uncertainty: 如果不存在，会根据query自动计算
               - dialogue_turn: 0/1/2（第几轮）
               - prev_reject: 0/1（上一轮是否被拒绝）
        
    Returns:
        格式化的纯state文本，不包含任何action指令
    """
    domain = state.get("domain", "coding")
    turn = state.get("dialogue_turn", 1)
    prev_reject = state.get("prev_reject", 0)
    query = state.get("query", "")
    
    # 获取task_uncertainty，如果不存在则根据query计算
    task_uncertainty = state.get("task_uncertainty")
    if task_uncertainty is None:
        # 自动计算task_uncertainty
        task_uncertainty = compute_task_uncertainty_from_state(state)
    
    # 构建prompt - 使用清晰的格式，不包含任何action指令
    # State只包含任务相关信息，不包含persona信息
    # 模型需要从dialogue signals推断用户偏好
    lines = [
        f"[Domain] {domain}",
        f"[Turn] {turn}",
        f"[Task Uncertainty] {task_uncertainty:.2f}",
        f"[Prev Reject] {prev_reject}",
        f"[Task]\n{query}",
    ]
    
    return "\n".join(lines)

