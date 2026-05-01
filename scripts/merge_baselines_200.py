"""Merge first-100 (patched) + remaining-100 (ft) Qwen baseline eval files into N=200.

Same logic as merge_v33_qwen_200.py but for the 4 baselines: Direct, CF, Base, PO.
Skips PO if its remaining-100 file does not yet exist.
"""
import json
from pathlib import Path

MERGES = [
    ("Direct",
     "outputs/eval_v29_qwen_direct_execution_100_patched.json",
     "outputs/eval_v29_qwen_direct_execution_remaining100_ft.json",
     "outputs/eval_v29_qwen_direct_execution_200.json"),
    ("Clarify-First",
     "outputs/eval_v29_qwen_clarify_first_100_patched.json",
     "outputs/eval_v29_qwen_clarify_first_remaining100_ft.json",
     "outputs/eval_v29_qwen_clarify_first_200.json"),
    ("Base",
     "outputs/eval_v29_qwen_base_100_v2_patched.json",
     "outputs/eval_v29_qwen_base_remaining100_ft.json",
     "outputs/eval_v29_qwen_base_200.json"),
    ("Prompt-Only",
     "outputs/eval_v29_qwen_prompt_only_100_patched.json",
     "outputs/eval_v29_qwen_prompt_only_remaining100_ft.json",
     "outputs/eval_v29_qwen_prompt_only_200.json"),
]

PERSONAS = ["Novice-Learner", "Busy-Developer", "Experienced-Engineer"]


def pass_flags(entry):
    last = entry["conversation"][-1]
    return last["pass_at_k"]


def merge_one(name, fa, fb, fout):
    a_path, b_path, out_path = Path(fa), Path(fb), Path(fout)
    if not b_path.exists():
        print(f"[SKIP {name}] remaining-100 file missing: {b_path}")
        return None
    if not a_path.exists():
        print(f"[SKIP {name}] first-100 file missing: {a_path}")
        return None

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    merged = a["detailed_results"] + b["detailed_results"]

    keys = {(e["state_id"], e["persona"]) for e in merged}
    assert len(keys) == len(merged), f"{name}: duplicates {len(merged)} vs unique {len(keys)}"

    summary = {}
    for p in PERSONAS:
        es = [e for e in merged if e["persona"] == p]
        n = len(es)
        total_turns = sum(e["total_turns"] for e in es)
        clarify_turns = sum(e["clarify_count"] for e in es)
        execute_turns = sum(e["execute_count"] for e in es)
        multi_turn_clarify = sum(1 for e in es if e["has_multi_turn_clarify"])
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
        return f"{name:<14}  (skipped — file missing)"
    total_n = sum(summary[p]["total_conversations"] for p in PERSONAS)
    total_p1 = sum(summary[p]["pass_at_k"]["pass@1"]["passed"] for p in PERSONAS)
    total_p5 = sum(summary[p]["pass_at_k"]["pass@5"]["passed"] for p in PERSONAS)
    parts = []
    for p in PERSONAS:
        s = summary[p]
        pk = s["pass_at_k"]["pass@1"]
        short = p.split("-")[0][:3]
        parts.append(f"{short}={pk['passed']:>3}/{pk['total']:>3}({100*pk['passed']/pk['total']:>4.1f}%)")
    return (f"{name:<14}  All={total_p1:>3}/{total_n} ({100*total_p1/total_n:>4.1f}%)  "
            f"pass@5={total_p5:>3}/{total_n} ({100*total_p5/total_n:>4.1f}%)  "
            + "  ".join(parts))


def main():
    print("=" * 100)
    print("Merging Qwen baselines first-100 + remaining-100 → N=200")
    print("=" * 100)
    print()
    for name, fa, fb, fout in MERGES:
        s = merge_one(name, fa, fb, fout)
        print(fmt_row(name, s))
    print()
    print("Output files:")
    for _, _, _, fout in MERGES:
        p = Path(fout)
        if p.exists():
            print(f"  {fout}  ({p.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
