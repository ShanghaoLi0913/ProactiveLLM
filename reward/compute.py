from typing import Dict


def compute_task_score(sample: Dict, domain: str) -> float:
    """MVP task score.

    coding: placeholder 0.6
    planning: placeholder 0.6
    Replace with tests (coding) or LLM judge (planning) later.
    """
    base = 0.6
    return float(base)


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


