"""Merge two pending baseline pairs into N=200:
  1. Llama CF: eval_150extra + missing-50 → CF on full eval_200
  2. Qwen PO: first-100 v2 + remaining-100 v2 → PO N=200 (v2 classifier consistent)
"""
import json
from pathlib import Path

MERGES = [
    ("Llama CF",
     "outputs/eval_v29_clarify_first_150extra.json",
     "outputs/eval_v29_clarify_first_missing50.json",
     "outputs/eval_v29_clarify_first_200.json"),
    ("Qwen PO v2",
     "outputs/eval_v29_qwen_prompt_only_first100_v2.json",
     "outputs/eval_v29_qwen_prompt_only_remaining100_ft.json",
     "outputs/eval_v29_qwen_prompt_only_200.json"),
]

PERSONAS = ["Novice-Learner", "Busy-Developer", "Experienced-Engineer"]


def pass_flags(entry):
    return entry["conversation"][-1]["pass_at_k"]


def merge_one(name, fa, fb, fout):
    a_path, b_path, out_path = Path(fa), Path(fb), Path(fout)
    if not a_path.exists():
        print(f"[SKIP {name}] missing: {a_path}")
        return None
    if not b_path.exists():
        print(f"[SKIP {name}] missing: {b_path}")
        return None

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    merged = a["detailed_results"] + b["detailed_results"]

    keys = {(e["state_id"], e["persona"]) for e in merged}
    if len(keys) != len(merged):
        print(f"[ERROR {name}] duplicates: {len(merged)} entries vs {len(keys)} unique keys")
        return None

    n_states = len(keys) // len(PERSONAS)
    print(f"\n[{name}] merged: {len(merged)} entries, {n_states} unique states")

    summary = {}
    for p in PERSONAS:
        es = [e for e in merged if e["persona"] == p]
        n = len(es)
        total_turns = sum(e["total_turns"] for e in es)
        clarify_turns = sum(e["clarify_count"] for e in es)
        execute_turns = sum(e["execute_count"] for e in es)
        multi_turn_clarify = sum(1 for e in es if e.get("has_multi_turn_clarify", False))
        p1 = sum(1 for e in es if pass_flags(e).get("pass@1"))
        p3 = sum(1 for e in es if pass_flags(e).get("pass@3"))
        p5 = sum(1 for e in es if pass_flags(e).get("pass@5"))
        summary[p] = {
            "total_conversations": n,
            "total_turns": total_turns,
            "clarify_turns": clarify_turns,
            "execute_turns": execute_turns,
            "avg_turns_per_conversation": round(total_turns / n, 4) if n else 0.0,
            "clarify_rate": clarify_turns / total_turns if total_turns else 0.0,
            "multi_turn_clarify_count": multi_turn_clarify,
            "pass_at_k": {
                "pass@1": {"total": n, "passed": p1},
                "pass@3": {"total": n, "passed": p3},
                "pass@5": {"total": n, "passed": p5},
            },
        }

    out = {"summary": summary, "detailed_results": merged}
    out_path.write_text(json.dumps(out, indent=2))
    return summary


def fmt_row(name, summary):
    if summary is None:
        return f"{name:<14}  FAILED"
    total_n = sum(summary[p]["total_conversations"] for p in PERSONAS)
    total_p1 = sum(summary[p]["pass_at_k"]["pass@1"]["passed"] for p in PERSONAS)
    total_p5 = sum(summary[p]["pass_at_k"]["pass@5"]["passed"] for p in PERSONAS)
    parts = []
    for p in PERSONAS:
        s = summary[p]
        pk = s["pass_at_k"]["pass@1"]
        avg_t = s["avg_turns_per_conversation"]
        short = p.split("-")[0][:3]
        parts.append(f"{short}={100*pk['passed']/pk['total']:>4.1f}%(t={avg_t:.1f})")
    return (f"{name:<14}  All={total_p1:>3}/{total_n} ({100*total_p1/total_n:>4.1f}%)  "
            f"p@5={100*total_p5/total_n:>4.1f}%   " + "  ".join(parts))


def main():
    print("=" * 100)
    print("Merging two pending baselines → N=200")
    print("=" * 100)
    results = []
    for name, fa, fb, fout in MERGES:
        s = merge_one(name, fa, fb, fout)
        results.append((name, s))
    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    for name, s in results:
        print(fmt_row(name, s))
    print()
    print("Output files:")
    for _, _, _, fout in MERGES:
        p = Path(fout)
        if p.exists():
            print(f"  {fout}  ({p.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
