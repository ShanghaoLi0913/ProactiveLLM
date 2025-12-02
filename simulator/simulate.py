from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Persona:
    name: str
    domain: str  # "coding" | "planning"
    expertise: str  # "low" | "mid" | "high"
    patience: float  # 0..1
    style: str  # "direct" | "polite" | "concise" | "verbose"


PERSONAS = [
    Persona("Impatient-Novice", "coding", "low", 0.25, "direct"),
    Persona("Neutral-Intermediate", "coding", "mid", 0.6, "polite"),
    Persona("Busy-Manager", "planning", "high", 0.35, "direct"),
]


def react(user_msg: str, assistant_msg: str, persona: Persona) -> Dict[str, Any]:
    """Minimal deterministic rules for MVP.

    Returns a dict with keys: user_reply, meta
    meta contains: answered_clarification, reject_signal, silence, off_topic_flag, satisfaction
    """
    # Naive heuristics for MVP
    asked_count = assistant_msg.lower().count("?")
    length_tokens = max(1, len(assistant_msg.split()))

    answered = 1 if (asked_count > 0 and persona.patience >= 0.35) else 0

    reject_signal = 1 if (asked_count >= 2 and persona.patience < 0.4) else 0
    silence = 1 if (length_tokens > 200 and persona.patience < 0.3) else 0
    off_topic_flag = 0  # placeholder: add semantic check later

    # Satisfaction: simple function for MVP
    satisfaction = max(0.0, min(1.0, 0.6 + 0.1 * answered - 0.3 * reject_signal - 0.1 * (length_tokens > 200)))

    if reject_signal:
        user_reply = "Stop asking, just give me the plan." if persona.domain == "planning" else "Stop asking, just give me the code."
    elif answered:
        user_reply = "OK." if persona.style != "direct" else "Ok."
    elif silence:
        user_reply = "..."
    else:
        user_reply = "Continue."

    return {
        "user_reply": user_reply,
        "meta": {
            "answered_clarification": int(answered),
            "reject_signal": int(reject_signal),
            "silence": int(silence),
            "off_topic_flag": int(off_topic_flag),
            "satisfaction": float(satisfaction),
        },
    }