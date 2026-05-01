"""Merge first-100 (patched) + remaining-100 (ft) Qwen v33 SFT+DPO eval files into N=200."""
import json
from pathlib import Path

A = Path("outputs/eval_v33_v3_qwen_dpo_v2_100_patched.json")
B = Path("outputs/eval_v33_v3_qwen_dpo_v2_remaining100_ft.json")
OUT = Path("outputs/eval_v33_v3_qwen_dpo_v2_200.json")

a = json.loads(A.read_text())
b = json.loads(B.read_text())

merged = a["detailed_results"] + b["detailed_results"]

# verify disjoint
keys = {(e["state_id"], e["persona"]) for e in merged}
assert len(keys) == len(merged), f"duplicates: {len(merged)} entries vs {len(keys)} unique"

# recompute per-persona summary from per-entry pass_at_k (last turn of each conversation)
def pass_flags(entry):
    last = entry["conversation"][-1]
    return last["pass_at_k"]

personas = ["Novice-Learner", "Busy-Developer", "Experienced-Engineer"]
summary = {}
for p in personas:
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
        "avg_turns_per_conversation": round(total_turns / n, 4),
        "clarify_rate": clarify_turns / total_turns if total_turns else 0.0,
        "multi_turn_clarify_count": multi_turn_clarify,
        "pass_at_k": {
            "pass@1": {"total": n, "passed": p1},
            "pass@3": {"total": n, "passed": p3},
            "pass@5": {"total": n, "passed": p5},
        },
    }

out = {"summary": summary, "detailed_results": merged}
OUT.write_text(json.dumps(out, indent=2))

# print
total_n = sum(summary[p]["total_conversations"] for p in personas)
total_p1 = sum(summary[p]["pass_at_k"]["pass@1"]["passed"] for p in personas)
total_p5 = sum(summary[p]["pass_at_k"]["pass@5"]["passed"] for p in personas)

print(f"merged: {len(merged)} entries -> {OUT}")
print()
print(f"{'Persona':<22} {'N':>4} {'pass@1':>10} {'pass@3':>10} {'pass@5':>10} {'avg_t':>7} {'clar%':>7}")
for p in personas:
    s = summary[p]
    pk = s["pass_at_k"]
    print(f"{p:<22} {s['total_conversations']:>4} "
          f"{pk['pass@1']['passed']:>3}/{pk['pass@1']['total']} ({100*pk['pass@1']['passed']/pk['pass@1']['total']:>4.1f})  "
          f"{pk['pass@3']['passed']:>3}/{pk['pass@3']['total']} ({100*pk['pass@3']['passed']/pk['pass@3']['total']:>4.1f})  "
          f"{pk['pass@5']['passed']:>3}/{pk['pass@5']['total']} ({100*pk['pass@5']['passed']/pk['pass@5']['total']:>4.1f})  "
          f"{s['avg_turns_per_conversation']:>6.2f}  "
          f"{100*s['clarify_rate']:>6.1f}")
print(f"{'OVERALL':<22} {total_n:>4} {total_p1:>3}/{total_n} ({100*total_p1/total_n:>4.1f})  "
      f"{'':>11}  {total_p5:>3}/{total_n} ({100*total_p5/total_n:>4.1f})")
