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
    
    规则：
    - 如果澄清有效（answered）→ 不扣分
    - 如果澄清无效（rejected）→ 扣分
    
    C_interrupt是整个对话中所有无效澄清问题的累积成本。
    
    Args:
        meta: 包含answered_clarification, reject_signal等字段
        n_questions: 澄清问题的数量
        assistant_msg: assistant的消息（用于检查是否包含问题）
        
    Returns:
        C_interrupt: 累积的无效澄清成本
    """
    # 检查澄清是否有效
    answered = meta.get("answered_clarification", 0)  # 1 if answered, 0 if not
    rejected = meta.get("reject_signal", 0)  # 1 if rejected, 0 if not
    
    # 如果澄清被拒绝（rejected），则扣分
    if rejected:
        # 被拒绝的澄清问题，每个扣分
        cost = n_questions * 0.5  # 每个无效澄清问题扣0.5分
    elif answered:
        # 澄清有效（answered），不扣分
        cost = 0.0
    else:
        # 既没有answered也没有rejected，可能是第一次提问
        # 这种情况下，如果问了问题但还没有得到回答，暂时不扣分
        # 但如果有多个问题，可能表示过度提问
        if n_questions > 1:
            cost = (n_questions - 1) * 0.2  # 超过1个问题，每个额外问题扣0.2分
        else:
            cost = 0.0
    
    return float(cost)


def total_reward(task_score: float, interrupt_cost: float) -> float:
    """
    Reward_version2: R = R_task - C_interrupt
    
    Args:
        task_score: R_task（任务成功奖励，sparse final reward）
        interrupt_cost: C_interrupt（总成本，无效澄清问题的累积成本）
        
    Returns:
        Total reward
    """
    return float(task_score - interrupt_cost)


