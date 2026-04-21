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
3. 可选包含persona信息（如name），用于区分不同用户风格
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
    from utils.compute_task_uncertainty import compute_state_uncertainty
except ImportError:
    # 如果导入失败，使用fallback
    def compute_state_uncertainty(state: Dict) -> float:
        return 0.5


def render_state(state: Dict, persona: Dict = None, ablation_mode: str = None) -> str:
    """
    将state转换为纯文本prompt，不包含任何action指令或模板。
    
    序列决策问题格式：
    [Domain] coding
    [User Profile]
    Type: Busy-Developer
    Patience: low
    Expertise: mid
    [Task Uncertainty] 0.3
    [Dialogue Turn] 2
    [Previous Reject] 0
    [User Request]
    write a python script that scrapes data...
    
    注意：
    - 显式传递persona信息（name, patience, expertise），让模型学会persona-aware策略
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
        persona: Optional persona dict with:
               - name: Persona name (e.g., "Busy-Developer")
               - patience: low/mid/high
               - expertise: low/mid/high
        
    Returns:
        格式化的纯state文本，不包含任何action指令
    """
    domain = state.get("domain", "coding")
    query = state.get("query", "")
    dialogue_turn = state.get("dialogue_turn", 0)
    prev_reject = state.get("prev_reject", 0)
    
    # ⭐ Persona优先从参数获取，否则从state获取
    if persona is None:
        persona = state.get("persona", {})
    
    # 提取persona信息
    if isinstance(persona, dict):
        persona_name = persona.get("name", "") or "Unknown"
        persona_patience = persona.get("patience", "mid")
        persona_expertise = persona.get("expertise", "mid")
    elif isinstance(persona, str):
        persona_name = persona or "Unknown"
        persona_patience = "mid"
        persona_expertise = "mid"
    else:
        persona_name = "Unknown"
        persona_patience = "mid"
        persona_expertise = "mid"
    
    # v31: task_uncertainty is disclosure-based and DOES evolve across turns.
    # U = (n_masked - n_disclosed) / MAX_MASKED (see utils/compute_task_uncertainty.py).
    # As the user answers clarifying questions, disclosed_info grows and U drops.
    # This is the single source of truth for both prompt display AND label
    # generation (reward/compute_rewards.get_correct_action), so training and
    # eval see identical numeric values.
    task_uncertainty = compute_state_uncertainty(state)
    
    # 构建prompt - 使用清晰的格式，显式包含persona信息
    # 
    # 格式设计让LLM容易理解：
    # - persona: 用户类型、耐心度、专业水平（关键：让模型学会persona-aware策略）
    # - task_uncertainty: 0.0-1.0，值越高表示越不确定（需要更多澄清）
    #   [2026-04-22 修正] 之前 compute_task_uncertainty 内部返回的是 clarity，
    #   此处显示的数字方向与变量名相反。已翻正，现在数值与语义一致。
    # - dialogue_turn: 当前对话轮次，帮助模型判断是否应该继续澄清
    # - prev_reject: 上一轮是否被拒绝，帮助模型判断用户耐心
    # - query: 用户的请求（可能包含对话历史）
    lines = [f"[Domain] {domain}", ""]

    # Ablation: "no_persona" removes [User Profile] block entirely
    if ablation_mode != "no_persona":
        lines += [
            "[User Profile]",
            f"Type: {persona_name}",
            f"Patience: {persona_patience}",
            f"Expertise: {persona_expertise}",
            "",
        ]

    lines.append("[Context]")
    # Ablation: "no_uncertainty" removes Task Uncertainty line
    if ablation_mode != "no_uncertainty":
        lines.append(f"Task Uncertainty: {task_uncertainty:.2f}")
    lines += [
        f"Dialogue Turn: {dialogue_turn}",
        f"Previous Reject: {prev_reject}",
        "",
        "[User Request]",
        query,
    ]
    
    return "\n".join(lines)

