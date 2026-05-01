"""Llama DPO Novice turn-0 with sampling (T=0.7, top_p=0.9), 8 states × 5 samples
each. Goal: check whether sampling reveals the trained behavior (direct clarify
question) that greedy decoding may have missed.
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
N_SAMPLES = 5
OUT_PATH = Path("/tmp/llama_sampling_sanity.json")


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


def main():
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading {BASE_MODEL} (8-bit)", flush=True)
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=qcfg, device_map="auto",
    )
    if len(tokenizer) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    print("Loaded.", flush=True)

    states = []
    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))
            if len(states) >= N_STATES:
                break

    persona = PERSONAS[0]  # Novice-Learner
    persona_dict = {
        "name": persona.name,
        "patience": persona.patience,
        "expertise": persona.expertise,
    }

    rows = []
    for si, state in enumerate(states):
        state_text = render_state_with_persona(state, persona=persona_dict)
        prompt = build_action_selection_chat_prompt(state_text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        for sample_idx in range(N_SAMPLES):
            torch.manual_seed(1000 + si * 100 + sample_idx)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text30 = tokenizer.decode(new_tokens[:30], skip_special_tokens=True).strip()

            v1 = apply_v1_rule(text30)
            v2 = apply_v2_rule(text)
            row = {
                "state_idx": si,
                "sample": sample_idx,
                "text30": text30,
                "text200": text,
                "v1": v1,
                "v2": v2,
                "starts_with_question": text.lstrip().startswith(("What", "How", "Could", "Can", "Should", "Do you", "Are there", "Is the")),
            }
            rows.append(row)
            print(
                f"[s{si} #{sample_idx}] v1={v1:7s} v2={v2:7s} qstart={row['starts_with_question']}  "
                f"text30={text30[:70]!r}",
                flush=True,
            )

    n_total = len(rows)
    n_v1_clarify = sum(1 for r in rows if r["v1"] == "Clarify")
    n_v2_execute = sum(1 for r in rows if r["v2"] == "Execute")
    n_q_start = sum(1 for r in rows if r["starts_with_question"])
    print()
    print("=" * 72)
    print(f"TOTAL: {n_total}  v1=Clarify: {n_v1_clarify}  v2=Execute: {n_v2_execute}  "
          f"starts-with-question: {n_q_start}")
    print("=" * 72)

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
