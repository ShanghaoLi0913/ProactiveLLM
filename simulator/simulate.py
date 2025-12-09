from dataclasses import dataclass
from typing import Dict, Any, Optional
import random
import math


@dataclass
class Persona:
    name: str
    domain: str  # "coding" | "planning"
    expertise: str  # "low" | "mid" | "high" - affects answer clarity
    patience: str  # "low" | "mid" | "high" - affects reject probability


# Mapping from patience level to numeric value (for calculations)
PATIENCE_MAP = {
    "low": 0.3,
    "mid": 0.6,
    "high": 0.9,
}

# Mapping from expertise level to numeric value (for calculations)
# expertise → 回答清晰度（answer clarity）
EXPERTISE_MAP = {
    "low": 0.3,
    "mid": 0.6,
    "high": 0.9,
}


PERSONAS = [
    # 1. Novice Learner: 新手学习者
    # - 刚学编程，需要帮助和理解
    # - 愿意回答澄清问题，但回答可能不够清晰（low expertise）
    # - 行为：高耐心（high patience），不容易拒绝
    Persona("Novice-Learner", "coding", "low", "high"),
    
    # 2. Busy Developer: 忙碌的开发者
    # - 有编程经验但时间紧迫
    # - 回答清晰（mid expertise），但低耐心（low patience）
    # - 行为：如果问太多会拒绝
    Persona("Busy-Developer", "coding", "mid", "low"),
    
    # 3. Experienced Engineer: 经验丰富的工程师
    # - 技术能力强，回答非常清晰（high expertise）
    # - 中等耐心（mid patience），有一定容忍度
    # - 行为：如果问题合理会回答，但如果问太多会不耐烦
    Persona("Experienced-Engineer", "coding", "high", "mid"),
]


def generate_specific_answer_dummy(assistant_msg: str, user_query: str, domain: str, 
                                     expertise: str = "mid") -> str:
    """Generate a dummy specific answer for testing (synthetic mode).
    
    Args:
        assistant_msg: The assistant's message (may contain a question)
        user_query: The original user query/task
        domain: "coding" or "planning"
        expertise: "low" | "mid" | "high" - affects answer clarity
    
    Returns:
        A dummy answer based on expertise level.
    """
    expertise_value = EXPERTISE_MAP[expertise]
    
    if expertise == "low":
        # Novice: vague or incomplete answer
        return "可能是这样的吧，我也不太确定。"
    elif expertise == "high":
        # Expert: very clear, detailed answer
        return "需要处理空字符串的情况，使用递归实现，时间复杂度O(n)。"
    else:  # mid
        # Intermediate: clear, specific answer
        return "需要处理空字符串。"
    
    return "好的。"  # fallback


def generate_specific_answer_llm(assistant_msg: str, user_query: str, domain: str, 
                                  llm_model: str, expertise: str = "mid") -> str:
    """Generate a specific answer using LLM based on the assistant's question.
    
    This makes multi-interaction turns meaningful by providing actual information
    that is relevant to the question and the original task.
    
    Expertise affects answer clarity:
    - low: May provide vague or incomplete answers (novice user)
    - mid: Provides clear, specific answers (intermediate user)
    - high: Provides very clear, detailed, professional answers (expert user)
    
    Args:
        assistant_msg: The assistant's message (may contain a question)
        user_query: The original user query/task
        domain: "coding" or "planning"
        llm_model: LLM model name (e.g., "gpt-4o-mini"). Required.
        expertise: "low" | "mid" | "high" - affects answer clarity
    
    Returns:
        A specific answer to the assistant's question (clarity depends on expertise).
    """
    from llm.provider import chat_complete
    
    # Adjust prompt based on expertise level
    if expertise == "low":
        # Novice: may provide vague or incomplete answers
        clarity_instruction = "You are a beginner. Your answer may be somewhat vague or incomplete, but try to be helpful."
    elif expertise == "high":
        # Expert: very clear, detailed, professional answers
        clarity_instruction = "You are an expert. Provide a very clear, detailed, and professional answer with specific technical details."
    else:  # mid
        # Intermediate: clear, specific answers
        clarity_instruction = "Provide a clear and specific answer."
    
    # Create a prompt for the LLM to generate a user response
    system_prompt = f"""You are a user who has asked for help with a {domain} task.
The assistant has asked you a clarifying question. {clarity_instruction}
Keep your answer concise (1-2 sentences or a short phrase)."""
    
    user_prompt = f"""Original task: {user_query}

Assistant's question: {assistant_msg}

Provide a brief, specific answer to the assistant's question:"""
    
    answer = chat_complete(system_prompt, user_prompt, model=llm_model, max_tokens=100)
    return answer.strip()




def react(user_msg: str, assistant_msg: str, persona: Persona, 
          llm_model: Optional[str] = None, total_questions_asked: int = 0) -> Dict[str, Any]:
    """Generate user reaction based on assistant message and persona.
    
    Implements Persona → Behavior Mapping:
    - If Assistant Clarifies:
      * P(answer) = patience (Equation 6)
      * P(reject) = 1 - patience (Equation 7)
      * answer_clarity = EXPERTISE_MAP[expertise] (Equation 8)
    - If Assistant Executes:
      * No immediate user reaction (until final code checking)
    
    Args:
        user_msg: The original user query/task
        assistant_msg: The assistant's message (may contain a question or code)
        persona: User persona (expertise affects answer clarity, patience affects reject probability)
        llm_model: LLM model name (e.g., "gpt-4o-mini"). Required.
        total_questions_asked: Total number of questions asked so far (for context)

    Returns a dict with keys: user_reply, meta
    meta contains: answered_clarification, reject_signal, answer_clarity, silence, off_topic_flag, satisfaction
    """
    # Check if assistant is asking a question in this message
    asked_count_this_msg = assistant_msg.lower().count("?")
    length_tokens = max(1, len(assistant_msg.split()))
    
    # Check if assistant is providing code (Execute action)
    has_code = (
        "```" in assistant_msg or
        "def " in assistant_msg or
        "class " in assistant_msg or
        assistant_msg.strip().startswith("import ")
    )
    
    # Convert persona attributes to numeric values
    patience_value = PATIENCE_MAP[persona.patience]
    expertise_value = EXPERTISE_MAP[persona.expertise]
    
    # If Assistant Executes (provides code):
    # - No immediate user reaction (until final code checking)
    # - Return minimal reaction
    if has_code:
        return {
            "user_reply": "Continue.",
            "meta": {
                "answered_clarification": 0,
                "reject_signal": 0,
                "answer_clarity": 0.0,  # No answer when executing
                "silence": 0,
                "off_topic_flag": 0,
                "satisfaction": 0.6,
            },
        }
    
    # If Assistant Clarifies (asks question):
    # - P(answer) = patience (Equation 6)
    # - P(reject) = 1 - patience (Equation 7)
    # - answer_clarity = EXPERTISE_MAP[expertise] (Equation 8)
    
    if asked_count_this_msg > 0:
        # Determine if user answers or rejects based on patience
        # P(answer) = patience, P(reject) = 1 - patience
        if random.random() < patience_value:
            # User answers (with probability = patience)
            # Generate answer with clarity based on expertise
            if llm_model:
                # Use LLM to generate realistic answer
                specific_answer = generate_specific_answer_llm(
                    assistant_msg, user_msg, persona.domain, llm_model=llm_model, expertise=persona.expertise
                )
            else:
                # Use dummy answer for synthetic mode (testing)
                specific_answer = generate_specific_answer_dummy(
                    assistant_msg, user_msg, persona.domain, expertise=persona.expertise
                )
            user_reply = specific_answer
            answered = 1
            reject_signal = 0
            answer_clarity = expertise_value  # answer_clarity = f(expertise)
        else:
            # User rejects (with probability = 1 - patience)
            user_reply = "Stop asking, just give me the plan." if persona.domain == "planning" else "Stop asking, just give me the code."
            answered = 0
            reject_signal = 1
            answer_clarity = 0.0  # No answer when rejected
    else:
        # No question asked, default response
        user_reply = "Continue."
        answered = 0
        reject_signal = 0
        answer_clarity = 0.0
    
    # Silence check (if message is too long and patience is low)
    silence = 1 if (length_tokens > 200 and patience_value < 0.3) else 0
    off_topic_flag = 0  # placeholder: add semantic check later

    # Satisfaction: enhanced to provide clearer signal for successful clarifications
    # When answered_clarification=1, satisfaction should be higher to reinforce positive feedback
    base_satisfaction = 0.6
    if answered == 1:
        # Successful clarification: higher satisfaction (0.8-0.9) to reinforce positive feedback
        satisfaction = 0.85  # High satisfaction for effective clarifications
    elif reject_signal == 1:
        # User rejected: low satisfaction
        satisfaction = 0.3
    elif silence == 1:
        # User silent: lower satisfaction
        satisfaction = 0.5
    else:
        # Default satisfaction
        satisfaction = base_satisfaction
    
    # Ensure satisfaction is in [0.0, 1.0] range
    satisfaction = max(0.0, min(1.0, satisfaction))

    return {
        "user_reply": user_reply,
        "meta": {
            "answered_clarification": int(answered),
            "reject_signal": int(reject_signal),
            "answer_clarity": float(answer_clarity),  # answer_clarity = f(expertise)
            "silence": int(silence),
            "off_topic_flag": int(off_topic_flag),
            "satisfaction": float(satisfaction),
        },
    }