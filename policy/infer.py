"""
Sequential Decision Process Inference

This module implements the sequential decision process where:
1. Policy model predicts action (Clarify/Execute) at each dialogue turn
2. Code generation is handled by a separate model (not affected by DPO)
"""
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import for code generation (can use base model or separate code model)
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_action_selection_chat_prompt(state_text: str, tokenizer) -> str:
    """One user message + assistant header, matching policy/train_dpo.to_dpo_format."""
    messages = [{"role": "user", "content": state_text}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return state_text


def pick_action_from_logits(tokenizer, next_token_logits: torch.Tensor) -> str:
    """Argmax over single-token ids for Clarify / Execute.

    NOTE: This function is kept for backward compatibility but is no longer used
    by the main evaluation path. Use pick_action_from_generation() instead,
    which generates a short response and parses the first word. The single-token
    argmax approach fails when 'Clarify' is a new special token (id=128256) with
    weak lm_head weights (~0.5 logit) vs 'Execute' (id=17617) with strong
    pretrained weights (~6.3 logit) — LoRA cannot close this gap.
    """
    pairs = []
    for tok in ("Clarify", "Execute"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None:
            pairs.append((tok, tid))
    if not pairs:
        return "Execute"
    idx = torch.tensor(
        [t for _, t in pairs],
        device=next_token_logits.device,
        dtype=torch.long,
    )
    best = torch.argmax(next_token_logits[idx]).item()
    return pairs[best][0]


def pick_action_from_generation(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
    """Generate a short response and detect Clarify vs Execute from content style.

    After stripping "Clarify\\n"/"Execute\\n" prefixes from training data, the model
    learns to prefer natural responses: clarification questions (natural language) vs
    code (starts with ```python / def / import). We detect the action from the style
    of the generated text rather than looking for artificial keyword prefixes.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Code indicators → Execute
    code_starters = ("```", "def ", "import ", "from ", "class ", "#!", "#!/")
    if any(generated.startswith(s) for s in code_starters):
        return "Execute"
    # Natural language → Clarify (questions, clarifications, acknowledgements)
    return "Clarify"


def select_action(state_text: str, policy_model_dir: str, base_model: Optional[str] = None) -> str:
    """
    Use policy model to predict action (Clarify/Execute) based on state.
    
    Args:
        state_text: Rendered state text (contains task_uncertainty, dialogue_turn, prev_reject)
        policy_model_dir: Directory containing trained policy model (LoRA adapter)
        base_model: Base model name (if None, will try to load from policy_model_dir)
    
    Returns:
        Action token: "Clarify" or "Execute"
    """
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(policy_model_dir, use_fast=True)
    except:
        if base_model:
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            # Add special tokens
            special_tokens = {"additional_special_tokens": ["Clarify", "Execute"]}
            tokenizer.add_special_tokens(special_tokens)
        else:
            raise ValueError("Cannot load tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    if base_model:
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        # Resize embeddings if needed
        if len(tokenizer) != base_model_obj.get_input_embeddings().num_embeddings:
            base_model_obj.resize_token_embeddings(len(tokenizer))
        
        # Load LoRA adapter
        try:
            model = PeftModel.from_pretrained(base_model_obj, policy_model_dir)
        except:
            model = base_model_obj
    else:
        model = AutoModelForCausalLM.from_pretrained(policy_model_dir)
    
    model.eval()
    
    prompt = build_action_selection_chat_prompt(state_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        next_token_logits = logits[0, -1, :]
    return pick_action_from_logits(tokenizer, next_token_logits)


def get_template(action: str, domain: str) -> str:
    """Get template for the given action and domain."""
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        fname = {
            "Clarify": "coding_clarify.txt",
            "Execute": "coding_execute.txt",
        }[action]
    else:
        fname = {
            "Clarify": "planning_clarify.txt",
            "Execute": "planning_execute.txt",
        }[action]
    return (base / fname).read_text(encoding="utf-8").strip()


def generate_code(
    task_prompt: str,
    template: str,
    code_model_name: Optional[str] = None,
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
) -> str:
    """
    Generate code using a separate code generation model.
    
    This is NOT affected by DPO training, ensuring clean code generation.
    
    Args:
        task_prompt: The task description
        template: The action template (from get_template)
        code_model_name: Name of code generation model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        use_openai: Whether to use OpenAI API for code generation
        openai_model: OpenAI model name if use_openai=True
    
    Returns:
        Generated code response
    """
    if use_openai:
        # Use OpenAI API for code generation
        from llm.provider import chat_complete
        system = template
        user = f"[Task]\n{task_prompt}"
        return chat_complete(system, user, model=openai_model, max_tokens=400)
    elif code_model_name:
        # Use a local model (e.g., Llama) for code generation
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(code_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            code_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Format prompt using chat template
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": f"[Task]\n{task_prompt}"}
        ]
        
        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback: simple formatting
            prompt = f"{template}\n\n[Task]\n{task_prompt}\n\nAssistant:"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()
        
        return generated_text
    else:
        # Fallback: just return template + task
        # In production, you should use a proper code generation model
        return f"{template}\n\n[Task]\n{task_prompt}"


def execute_action(
    action: str,
    task_prompt: str,
    domain: str,
    code_model_name: Optional[str] = None,
    use_openai: bool = False,
) -> str:
    """
    Execute action using separated architecture.
    
    This function:
    1. Gets template based on action
    2. Generates code using separate code generation model
    3. Returns clean code (not affected by DPO training)
    
    Args:
        action: Action token (Clarify/Execute)
        task_prompt: The task description
        domain: Domain ("coding" or "planning")
        code_model_name: Code generation model name
        use_openai: Whether to use OpenAI API
    
    Returns:
        Generated code response
    """
    template = get_template(action, domain)
    return generate_code(
        task_prompt,
        template,
        code_model_name=code_model_name,
        use_openai=use_openai,
    )


