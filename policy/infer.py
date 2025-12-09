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
    
    # Tokenize state text
    inputs = tokenizer(state_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Use logits directly to select action token (instead of generating text)
    with torch.no_grad():
        logits = model(**inputs).logits
        # Get logits for the last token position
        next_token_logits = logits[0, -1, :]
        
        # Get token IDs for action tokens
        action_tokens = ["Clarify", "Execute"]
        action_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in action_tokens]
        
        # Filter out None values (in case token not found)
        valid_actions = []
        valid_ids = []
        for token, token_id in zip(action_tokens, action_token_ids):
            if token_id is not None:
                valid_actions.append(token)
                valid_ids.append(token_id)
        
        if not valid_ids:
            # Fallback if no action tokens found in vocabulary
            return "Execute"
        
        # Get logits for action tokens only
        action_logits = next_token_logits[valid_ids]
        
        # Select action with highest logit
        best_action_idx = torch.argmax(action_logits).item()
        best_action = valid_actions[best_action_idx]
        
        return best_action


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
    openai_model: str = "gpt-4o-mini"
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
    use_openai: bool = False
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
    return generate_code(task_prompt, template, code_model_name, use_openai)


