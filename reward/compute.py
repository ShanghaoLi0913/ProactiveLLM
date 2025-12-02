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
    # For coding with tests (ConvCodeWorld), use test execution if available
    if domain == "coding" and assistant_output is not None:
        tests = sample.get("convcodeworld_tests")
        if tests:
            # Test execution is handled in eval/evaluate_dpo_model.py via score_code_passfail
            # This function is kept for compatibility but returns placeholder
            pass
    
    # Placeholder for MVP
    return 0.6


def compute_interrupt_cost(meta: Dict, n_questions: int, length_tokens: int, off_topic: int, alpha: float = 0.6) -> float:
    w1, w2, w3, w4 = 0.4, 1.0, 0.3, 0.7
    cost = (
        w1 * n_questions
        + w2 * int(meta.get("reject_signal", 0))
        + w3 * int(length_tokens > 200)
        + w4 * int(off_topic)
    )
    return float(alpha * cost)


def total_reward(task_score: float, interrupt_cost: float) -> float:
    return float(task_score - interrupt_cost)


