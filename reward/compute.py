from typing import Dict, Optional


def compute_task_score(sample: Dict, domain: str, assistant_output: Optional[str] = None) -> float:
    """Compute task success score.

    Args:
        sample: State dict (may contain convcodeworld_tests for coding tasks)
        domain: "coding" or "planning"
        assistant_output: Generated assistant message (optional, for test execution)

    Returns:
        Task score (0.0-1.0). Higher is better.
    """
    # If no assistant output, task is not completed
    if not assistant_output:
        return 0.0
    
    # Check if assistant output contains code (for coding tasks)
    if domain == "coding":
        # Simple heuristic: check if output contains code blocks or function definitions
        has_code = (
            "```" in assistant_output or
            "def " in assistant_output or
            "class " in assistant_output or
            "import " in assistant_output
        )
        
        # If no code, task is not completed (e.g., just asking questions)
        if not has_code:
            return 0.0
        
        # If has code, try to execute tests if available
        tests = sample.get("convcodeworld_tests")
        if tests:
            # Import here to avoid circular dependency
            try:
                from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail
                code = extract_code_from_text(assistant_output)
                if code:
                    # Execute tests and return pass/fail score
                    return score_code_passfail(code, tests, timeout=30)
                else:
                    # Code extraction failed, assume task not completed
                    return 0.0
            except Exception:
                # If test execution fails, return 0.0 (task not completed)
                return 0.0
        else:
            # No tests available, use heuristic: if has code, assume 0.5 (uncertain)
            # This is a fallback for cases without tests
            return 0.5
    
    # For planning domain, use placeholder
    # TODO: Implement planning task score calculation
    return 0.6


def compute_interrupt_cost(meta: Dict, n_questions: int, length_tokens: int, off_topic: int, alpha: float = 0.6) -> float:
    """
    计算interrupt cost（旧版本，保留用于兼容性）。
    
    新版本应该使用 compute_interrupt_cost_v2。
    """
    w1, w2, w3, w4 = 0.4, 1.0, 0.3, 0.7
    cost = (
        w1 * n_questions
        + w2 * int(meta.get("reject_signal", 0))
        + w3 * int(length_tokens > 200)
        + w4 * int(off_topic)
    )
    return float(alpha * cost)


def compute_interrupt_cost_v2(meta: Dict, n_questions: int, assistant_msg: str = "") -> float:
    """
    Reward_version2: 计算C_interrupt（总成本）。
    
    新公式: C_Interrupt = Σ_{t=1}^{T} (δb_t r_t + λb_t - γb_t a_t)
    
    其中：
    - b_t ∈ {0,1}: 是否提出澄清问题
    - a_t ∈ {0,1}: 如果提出澄清，用户是否认真回答
    - r_t ∈ {0,1}: 如果提出澄清，用户是否明确拒绝
    - δ > 0: 每一次被拒绝的澄清所带来的成本
    - λ ≥ 0: 提出澄清的基本开销，用于抑制过度提问
    - γ > 0: 成功澄清所带来的成本抵消，相当于奖励有效澄清
    
    对于单轮对话:
    C_Interrupt = δ * b * r + λ * b - γ * b * a
    
    Args:
        meta: 包含answered_clarification, reject_signal等字段
        n_questions: 澄清问题的数量
        assistant_msg: assistant的消息（用于检查是否包含问题）
        
    Returns:
        C_interrupt: 累积的中断成本
    """
    # 参数设置（激进设置，最大化奖励差异，解决MID塌陷问题）
    # 高γ：有效澄清带来显著奖励
    # 高δ：无效澄清带来显著惩罚
    # λ设为0：不惩罚提问本身，只惩罚无效提问
    gamma = 0.3   # γ: 有效澄清的成本抵消（奖励）- 提高以激励有效澄清
    delta = 0.7   # δ: 无效澄清的惩罚 - 提高以惩罚无效澄清
    lambda_param = 0.0  # λ: 提出澄清的基本开销（设为0，不惩罚提问本身）
    
    # b_t: 是否提出澄清问题
    b = 1 if n_questions > 0 else 0
    
    if b == 0:
        # 没有提出澄清，成本为0
        return 0.0
    
    # a_t: 用户是否认真回答
    a = meta.get("answered_clarification", 0)  # 1 if answered, 0 if not
    # r_t: 用户是否明确拒绝
    r = meta.get("reject_signal", 0)  # 1 if rejected, 0 if not
    
    # 对于多个问题，需要对每个问题分别计算
    # 简化处理：如果所有问题状态相同，可以统一计算
    # 否则需要对每个问题分别判断
    
    # 简化版本：假设所有问题的状态相同
    # 如果answered，所有问题都有效
    # 如果rejected，所有问题都无效
    # 否则，所有问题都未回答
    
    if a > 0:
        # 所有问题都有效澄清
        # C_Interrupt = n_questions * (δ*0 + λ*1 - γ*1) = n_questions * (λ - γ)
        cost = n_questions * (lambda_param - gamma)
    elif r > 0:
        # 所有问题都无效澄清（被拒绝）
        # C_Interrupt = n_questions * (δ*1 + λ*1 - γ*0) = n_questions * (δ + λ)
        cost = n_questions * (delta + lambda_param)
    else:
        # 所有问题都未回答
        # C_Interrupt = n_questions * (δ*0 + λ*1 - γ*0) = n_questions * λ
        cost = n_questions * lambda_param
    
    return float(cost)


def compute_clarification_bonus(meta: Dict, n_questions: int) -> float:
    """
    计算有效澄清的奖励（bonus）。
    
    注意：这个函数在新公式中不再需要，因为有效澄清的奖励已经包含在
    C_Interrupt的计算中（通过γ参数）。保留此函数是为了向后兼容。
    
    新公式中，有效澄清通过减少C_Interrupt来实现奖励：
    C_Interrupt = δb_t r_t + λb_t - γb_t a_t
    当a_t=1时，-γb_t a_t项会减少成本，相当于奖励。
    
    Args:
        meta: 包含answered_clarification, reject_signal等字段
        n_questions: 澄清问题的数量
        
    Returns:
        B_clarify: 有效澄清的奖励分数（在新公式中应该返回0，因为奖励已包含在C_Interrupt中）
    """
    # 在新公式中，有效澄清的奖励已经通过C_Interrupt中的-γb_t a_t项实现
    # 所以这里返回0，避免重复计算
    return 0.0


def total_reward(task_score: float, interrupt_cost: float, clarification_bonus: float = 0.0) -> float:
    """
    Reward_version2: R = R_task - C_interrupt
    
    新公式：总奖励 = 任务成功奖励 - 中断成本
    
    其中C_Interrupt = Σ_{t=1}^{T} (δb_t r_t + λb_t - γb_t a_t)
    - 有效澄清（a_t=1）会减少成本（通过-γb_t a_t项）
    - 无效澄清（r_t=1）会增加成本（通过+δb_t r_t项）
    - 每次提问都有基本开销（通过+λb_t项）
    
    Args:
        task_score: R_task（任务成功奖励，sparse final reward）
        interrupt_cost: C_interrupt（中断成本）
        clarification_bonus: B_clarify（已废弃，在新公式中不再需要）
        
    Returns:
        Total reward
    """
    # 新公式：R = R_task - C_interrupt
    # clarification_bonus参数保留是为了向后兼容，但不再使用
    return float(task_score - interrupt_cost)


