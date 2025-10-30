from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def select_action(state_text: str, model_dir: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    inputs = tokenizer(state_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=3)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # naive: extract last token in {LOW,MID,HIGH}
    for token in ["LOW", "MID", "HIGH"]:
        if token in text:
            return token
    return "MID"


def execute_action(action: str, task_prompt: str, domain: str) -> str:
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        fname = {
            "LOW": "coding_low.txt",
            "MID": "coding_mid.txt",
            "HIGH": "coding_high.txt",
        }[action]
    else:
        fname = {
            "LOW": "planning_low.txt",
            "MID": "planning_mid.txt",
            "HIGH": "planning_high.txt",
        }[action]
    template = (base / fname).read_text(encoding="utf-8").strip()
    # For MVP we just return the template + task prompt; plug real LLM later
    return f"{template}\n\n[Task]\n{task_prompt}"


