"""Spot-check A 方案: Llama DPO + short-context structured prompt.

Hypothesis: Llama loses pass@1 in long-context multi-turn forced-execute.
Fix: re-generate Execute code using single-turn short prompt with
structured hints from actually-disclosed items (not all masked_fields).

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/llama_smart_execute_spotcheck.py \
      --n_samples 20 --persona Novice-Learner

Outputs spot-check pass rate vs original DPO eval.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail


def build_smart_query(state: dict, conversation: list) -> str:
    """v1: Build masked query + structured hint of actually-disclosed items only.

    Excludes raw user_reaction text (which contains gpt-4o hallucinated specs and
    repetitive content that bloats prompt and confuses model).
    """
    query = state.get("query", "")
    seen = set()
    hints = []
    for t in conversation:
        di = t.get("disclosed_items") or {}
        for cat, items in di.items():
            if isinstance(items, list):
                for it in items:
                    s = str(it).strip()
                    if s and s not in seen:
                        seen.add(s)
                        hints.append(s)
    if not hints:
        return query
    hint_block = "Key requirements:\n" + "\n".join(f"- {h}" for h in hints)
    return f"{query}\n\nAdditional information from clarification:\n{hint_block}"


def build_execute_prompt(query: str, template: str) -> str:
    """Mimic the Execute prompt format used in evaluate_multi_turn_persona.py."""
    return f"{template}\n\nUser's request:\n{query}\n\nYour code:\n"


def gen_one(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    """Generate one completion."""
    msgs = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_dpo_eval", default="outputs/eval_v33_v3_llama_dpo_200.json")
    ap.add_argument("--seed_states", default="data/seeds/test_states_v29_eval_200.jsonl")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora_dir", default="models/v33_v3_dpo")
    ap.add_argument("--persona", default="Novice-Learner")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--filter", choices=["failed", "all"], default="failed",
                    help="failed: only re-test originally-failed samples; all: re-test all")
    ap.add_argument("--n_candidates", type=int, default=5, help="for pass@k")
    ap.add_argument("--out", default="outputs/llama_smart_execute_spotcheck.json")
    args = ap.parse_args()

    print(f"Loading DPO eval: {args.llama_dpo_eval}")
    dpo = json.load(open(args.llama_dpo_eval))
    # state map
    state_map = {}
    for line in open(args.seed_states):
        s = json.loads(line)
        state_map[s["id"]] = s

    rows = [r for r in dpo["detailed_results"] if r["persona"] == args.persona]
    # Restrict to states present in seed file (so a small seed acts as a filter set)
    rows = [r for r in rows if r["state_id"] in state_map]

    if args.filter == "failed":
        rows = [r for r in rows if not r["conversation"][-1].get("pass_at_k",{}).get("pass@1")]
    rows = rows[:args.n_samples]
    print(f"Selected {len(rows)} samples (persona={args.persona}, filter={args.filter}, seed-restricted={len(state_map)} states)")

    # template
    template = (PROJECT_ROOT / "prompts" / "coding_execute.txt").read_text(encoding="utf-8").strip()

    # load model
    print(f"Loading {args.base_model} + LoRA {args.lora_dir}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    # Decoupled: skip LoRA — use base model directly for Execute generation
    # (Idea: DPO LoRA may slightly degrade base model code generation; test if dropping helps)
    model = base
    model.eval()
    print("Model loaded (DECOUPLED: no LoRA at Execute time).")

    results = []
    n_orig_pass = 0
    n_new_pass1 = 0
    n_new_pass5 = 0

    for idx, r in enumerate(rows):
        sid = r["state_id"]
        state = state_map.get(sid)
        if state is None:
            print(f"[{idx+1}/{len(rows)}] {sid}: state not found, skip"); continue

        # original pass
        orig_p1 = bool(r["conversation"][-1].get("pass_at_k", {}).get("pass@1"))
        orig_p5 = bool(r["conversation"][-1].get("pass_at_k", {}).get("pass@5"))
        if orig_p1: n_orig_pass += 1

        # build smart query
        smart_q = build_smart_query(state, r["conversation"])
        n_hints = smart_q.count("- ")
        prompt = build_execute_prompt(smart_q, template)

        # generate K candidates: 1 greedy + K-1 sampled
        codes = []
        scores = []
        tests = state.get("convcodeworld_tests") or state.get("test")
        for k in range(args.n_candidates):
            do_sample = (k > 0)
            temp = 0.7 if do_sample else 1.0
            try:
                resp = gen_one(model, tok, prompt, max_new_tokens=600, do_sample=do_sample, temperature=temp)
                code = extract_code_from_text(resp)
                if code:
                    sc = score_code_passfail(code, tests, timeout=10) if tests else 0.0
                else:
                    sc = 0.0
                codes.append(resp[:200])
                scores.append(sc)
            except Exception as e:
                codes.append(f"<error: {e}>")
                scores.append(0.0)

        new_p1 = scores[0] >= 1.0
        new_p5 = any(s >= 1.0 for s in scores)
        if new_p1: n_new_pass1 += 1
        if new_p5: n_new_pass5 += 1

        marker = "✓" if new_p1 else ("[5]" if new_p5 else "✗")
        print(f"[{idx+1}/{len(rows)}] {sid} hints={n_hints} orig_p1={orig_p1} new_p1={new_p1} {marker}", flush=True)

        results.append({
            "state_id": sid,
            "n_hints": n_hints,
            "orig_pass_at_1": orig_p1,
            "orig_pass_at_5": orig_p5,
            "new_pass_at_1": new_p1,
            "new_pass_at_5": new_p5,
            "scores": scores,
        })

    # summary
    n = len(results)
    print()
    print("=" * 70)
    print(f"SPOT CHECK SUMMARY (persona={args.persona}, filter={args.filter}, n={n})")
    print("=" * 70)
    print(f"  Original DPO pass@1: {n_orig_pass}/{n} = {100*n_orig_pass/n if n else 0:.1f}%")
    print(f"  Smart-Execute pass@1: {n_new_pass1}/{n} = {100*n_new_pass1/n if n else 0:.1f}%")
    print(f"  Smart-Execute pass@5: {n_new_pass5}/{n} = {100*n_new_pass5/n if n else 0:.1f}%")
    print(f"  Δ pass@1 vs original: {100*(n_new_pass1-n_orig_pass)/n if n else 0:+.1f}pp")

    Path(args.out).write_text(json.dumps({
        "persona": args.persona,
        "filter": args.filter,
        "n": n,
        "n_orig_pass1": n_orig_pass,
        "n_new_pass1": n_new_pass1,
        "n_new_pass5": n_new_pass5,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
