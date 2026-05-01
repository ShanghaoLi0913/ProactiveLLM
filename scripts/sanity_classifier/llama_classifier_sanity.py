"""Llama DPO sanity: same protocol as qwen_classifier_sanity.py, but on
Llama-3.1-8B-Instruct + v29 LoRA. Goal: measure v1 misclassification rate on
Llama to decide whether Llama experiments need rerunning under v2.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path("/root/autodl-tmp/ProactiveLLM")
sys.path.insert(0, str(PROJECT_ROOT))

from policy.infer import build_action_selection_chat_prompt  # noqa: E402
from policy.render_state import render_state as render_state_with_persona  # noqa: E402
from simulator.simulate import PERSONAS  # noqa: E402

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = str(PROJECT_ROOT / "models/v29_100states")
TEST_PATH = PROJECT_ROOT / "data/seeds/test_states_v29_eval_50.jsonl"
N_STATES = 8
OUT_PATH = Path("/tmp/llama_classifier_sanity.json")


def apply_v1_rule(text: str) -> str:
    code_starters = ("```", "def ", "import ", "from ", "class ", "#!", "#!/")
    if any(text.startswith(s) for s in code_starters):
        return "Execute"
    return "Clarify"


def apply_v2_rule(text: str) -> str:
    code_anywhere = (
        "```python", "```py\n", "```py ",
        "\ndef ", "\nimport ", "\nfrom ",
        "def task_func", "import numpy", "import pandas",
    )
    if any(m in text for m in code_anywhere):
        return "Execute"
    code_starters = ("```", "def ", "import ", "from ", "class ", "#!", "#!/")
    if any(text.startswith(s) for s in code_starters):
        return "Execute"
    if "?" in text:
        return "Clarify"
    return "Clarify"


def load_llama_dpo():
    print(f"Loading tokenizer from {LORA_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading base {BASE_MODEL} (8-bit)", flush=True)
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=qcfg,
        device_map="auto",
    )
    if len(tokenizer) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tokenizer))
    print(f"Attaching LoRA from {LORA_DIR}", flush=True)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return model, tokenizer


def generate_greedy(model, tokenizer, prompt: str, max_new: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_states(n: int):
    states = []
    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            states.append(json.loads(line))
            if len(states) >= n:
                break
    return states


def main():
    states = load_states(N_STATES)
    print(f"Loaded {len(states)} states from {TEST_PATH.name}", flush=True)

    model, tokenizer = load_llama_dpo()

    rows = []
    for si, state in enumerate(states):
        for persona in PERSONAS:
            persona_dict = {
                "name": persona.name,
                "patience": persona.patience,
                "expertise": persona.expertise,
            }
            state_text = render_state_with_persona(state, persona=persona_dict)
            prompt = build_action_selection_chat_prompt(state_text, tokenizer)

            text30 = generate_greedy(model, tokenizer, prompt, max_new=30)
            text200 = generate_greedy(model, tokenizer, prompt, max_new=200)

            v1_verdict = apply_v1_rule(text30)
            v2_verdict = apply_v2_rule(text200)

            row = {
                "state_id": state.get("state_id", f"idx_{si}"),
                "persona": persona.name,
                "text30": text30,
                "text200": text200,
                "v1": v1_verdict,
                "v2": v2_verdict,
                "agree": v1_verdict == v2_verdict,
            }
            rows.append(row)
            tag = "==" if row["agree"] else "!="
            print(
                f"[{si+1:2d}/{len(states)}] {persona.name:22s} "
                f"v1={v1_verdict:7s} {tag} v2={v2_verdict:7s}  "
                f"text30={text30[:60]!r}",
                flush=True,
            )

    n_total = len(rows)
    n_agree = sum(1 for r in rows if r["agree"])
    n_v1_clarify_v2_execute = sum(
        1 for r in rows if r["v1"] == "Clarify" and r["v2"] == "Execute"
    )
    n_v1_execute_v2_clarify = sum(
        1 for r in rows if r["v1"] == "Execute" and r["v2"] == "Clarify"
    )
    print()
    print("=" * 72)
    print(f"TOTAL: {n_total}   AGREE: {n_agree}/{n_total}   "
          f"DISAGREE v1=C/v2=E: {n_v1_clarify_v2_execute}   "
          f"v1=E/v2=C: {n_v1_execute_v2_clarify}")
    print("=" * 72)
    print()
    print("Disagreement cases:")
    for r in rows:
        if not r["agree"]:
            print(f"  state={r['state_id']} persona={r['persona']}  v1={r['v1']} v2={r['v2']}")
            print(f"    text30  : {r['text30']!r}")
            print(f"    text200 : {r['text200'][:300]!r}")
            print()

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"Wrote full results to {OUT_PATH}")


if __name__ == "__main__":
    main()
