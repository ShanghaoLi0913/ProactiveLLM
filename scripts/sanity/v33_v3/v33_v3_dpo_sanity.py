"""v33 v3 DPO sanity (after refinement on SFT)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path("/root/autodl-tmp/ProactiveLLM")
sys.path.insert(0, str(PROJECT_ROOT))

from policy.infer import build_action_selection_chat_prompt  # noqa
from policy.render_state import render_state as render_state_with_persona  # noqa
from simulator.simulate import PERSONAS  # noqa

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = str(PROJECT_ROOT / "models/v33_v3_dpo")
TEST_PATH = PROJECT_ROOT / "data/seeds/test_states_v29_eval_50.jsonl"
N_STATES = 8
OUT_PATH = Path("/tmp/v33_v3_dpo_sanity.json")


def main():
    print(f"Loading {LORA_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    rows = []
    for si, state in enumerate(states):
        for persona in PERSONAS:
            persona_dict = {
                "name": persona.name, "patience": persona.patience, "expertise": persona.expertise,
            }
            state_text = render_state_with_persona(state, persona=persona_dict)
            prompt = build_action_selection_chat_prompt(state_text, tokenizer)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text200 = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text30 = tokenizer.decode(new_tokens[:30], skip_special_tokens=True).strip()

            row = {
                "state_idx": si, "persona": persona.name,
                "text30": text30, "text200_first300": text200[:300],
                "starts_with_Clarify": text200.startswith("Clarify"),
                "starts_with_Execute": text200.startswith("Execute"),
                "has_question_mark": "?" in text200,
                "has_code_marker": ("```python" in text200) or ("\nimport " in text200),
            }
            rows.append(row)
            tag = "Clarify" if row["starts_with_Clarify"] else (
                  "Execute" if row["starts_with_Execute"] else "OTHER")
            print(f"[{si+1}/8] {persona.name:22s} prefix={tag:8s}  q={row['has_question_mark']}  code={row['has_code_marker']}  text30={text30[:55]!r}", flush=True)

    n = len(rows)
    cp = sum(1 for r in rows if r["starts_with_Clarify"])
    ep = sum(1 for r in rows if r["starts_with_Execute"])
    print()
    print("=" * 72)
    print(f"TOTAL: {n}")
    print(f"  starts with 'Clarify\\n': {cp}/{n} ({100*cp/n:.1f}%)")
    print(f"  starts with 'Execute\\n': {ep}/{n} ({100*ep/n:.1f}%)")
    print("=" * 72)
    print()
    for p_name in ('Novice-Learner', 'Busy-Developer', 'Experienced-Engineer'):
        sub = [r for r in rows if r['persona'] == p_name]
        if not sub: continue
        cp_p = sum(1 for r in sub if r['starts_with_Clarify'])
        ep_p = sum(1 for r in sub if r['starts_with_Execute'])
        q = sum(1 for r in sub if r['has_question_mark'])
        code = sum(1 for r in sub if r['has_code_marker'])
        print(f"  {p_name:22s}: Clarify={cp_p}/{len(sub)}  Execute={ep_p}/{len(sub)}  q={q}/{len(sub)}  code={code}/{len(sub)}")

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
