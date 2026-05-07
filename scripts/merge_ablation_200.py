"""Merge ablation eval files (50test + 50extra + missing) into N=200 canonical.

For each ablation (no_persona, no_uncertainty), three sources have zero state-set
overlap on the 3 main personas (Novice / Experienced / Busy), summing to N=200.
Time-Pressured-Expert (OOD) only exists in the missing file at its native N.
"""
import json
from pathlib import Path
from collections import defaultdict


def aggregate_summary(detailed_results, source_summaries, claimed_state_keys):
    """Aggregate per-persona summary.

    detailed_results give us turn / clarify / execute counts.
    pass@1 / pass@5 must come from each source's own summary block (since
    detailed_results don't carry per-candidate pass info), summed across the
    disjoint state sets that we actually merged in.

    source_summaries: list of (label, summary_dict, set_of_state_ids_per_persona).
    claimed_state_keys: set of (state_id, persona) actually used in the merged file.
    """
    by = defaultdict(lambda: {
        "n": 0, "turns": 0,
        "clarify_turns": 0, "execute_turns": 0, "multi_turn_clarify": 0,
    })
    for r in detailed_results:
        p = r["persona"]
        by[p]["n"] += 1
        by[p]["turns"] += r.get("total_turns", 0)
        by[p]["clarify_turns"] += r.get("clarify_count", 0)
        by[p]["execute_turns"] += r.get("execute_count", 0)
        by[p]["multi_turn_clarify"] += int(r.get("has_multi_turn_clarify", False))

    # Aggregate pass@1 / pass@5 per persona by summing per-source 'passed' values.
    # Since each source covers disjoint state-sets and we used all entries from
    # each source, sum is exact.
    pass_by_persona = defaultdict(lambda: {"p1": 0, "p5": 0})
    for label, src_summary, src_state_sets in source_summaries:
        for p, s in src_summary.items():
            # Only count personas present in the merged set.
            if p not in by:
                continue
            pass_by_persona[p]["p1"] += s["pass_at_k"]["pass@1"]["passed"]
            pass_by_persona[p]["p5"] += s["pass_at_k"]["pass@5"]["passed"]

    summary = {}
    for p, v in by.items():
        n = v["n"]
        total_turns = v["turns"]
        pp = pass_by_persona[p]
        summary[p] = {
            "total_conversations": n,
            "total_turns": total_turns,
            "avg_turns_per_conversation": total_turns / n if n else 0.0,
            "clarify_turns": v["clarify_turns"],
            "execute_turns": v["execute_turns"],
            "clarify_rate": v["clarify_turns"] / total_turns if total_turns else 0.0,
            "multi_turn_clarify_count": v["multi_turn_clarify"],
            "pass_at_k": {
                "pass@1": {"passed": pp["p1"], "total": n},
                "pass@5": {"passed": pp["p5"], "total": n},
            },
        }
    return summary


def merge_files(sources, out_path):
    seen = set()  # (state_id, persona)
    merged = []
    source_summaries = []  # for pass@k aggregation
    for label, path in sources:
        d = json.load(open(path))
        added = 0
        src_state_sets = defaultdict(set)
        for r in d.get("detailed_results", []):
            key = (r["state_id"], r["persona"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
            added += 1
            src_state_sets[r["persona"]].add(r["state_id"])
        source_summaries.append((label, d.get("summary", {}), src_state_sets))
        print(f"  {label:10s} {path.split('/')[-1]:60s} +{added} (running total {len(merged)})")

    summary = aggregate_summary(merged, source_summaries, seen)
    out = {"summary": summary, "detailed_results": merged}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  -> {out_path}  ({len(merged)} convs)")
    return summary


def report(label, summary):
    print(f"\n=== {label} merged N=200 ===")
    main = ["Novice-Learner", "Experienced-Engineer", "Busy-Developer"]
    p1_t = p5_t = n_t = turns_t = 0
    for p in main + ["Time-Pressured-Expert"]:
        if p not in summary:
            continue
        s = summary[p]
        n = s["total_conversations"]
        p1 = s["pass_at_k"]["pass@1"]["passed"]
        p5 = s["pass_at_k"]["pass@5"]["passed"]
        avg = s["avg_turns_per_conversation"]
        cr = s["clarify_rate"]
        print(f"  {p[:22]:<22s} n={n:3d}  pass@1={100*p1/n:5.1f}%  pass@5={100*p5/n:5.1f}%  avg_t={avg:4.2f}  clarify={cr:.2f}")
        if p in main:
            p1_t += p1; p5_t += p5; n_t += n; turns_t += s["total_turns"]
    print(f"  {'OVERALL (3 main)':<22s} n={n_t:3d}  pass@1={100*p1_t/n_t:5.1f}%  pass@5={100*p5_t/n_t:5.1f}%  avg_t={turns_t/n_t:4.2f}")


if __name__ == "__main__":
    np_sources = [
        ("50test",  "outputs/archive/legacy_v29_misc/eval_v29_ablation_no_persona_50test.json"),
        ("50extra", "outputs/archive/legacy_v29_misc/eval_v29_ablation_no_persona_50extra.json"),
        ("missing", "outputs/eval_v29_ablation_no_persona_missing_prefix.json"),
    ]
    nu_sources = [
        ("50test",  "outputs/archive/legacy_smalltest/eval_v29_ablation_no_uncertainty_50test.json"),
        ("50extra", "outputs/archive/legacy_v29_misc/eval_v29_ablation_no_uncertainty_50extra.json.partial"),
        ("missing", "outputs/eval_v29_ablation_no_uncertainty_missing_prefix.json"),
    ]

    print("=== merging no_persona ===")
    np_summary = merge_files(np_sources, "outputs/eval_v29_ablation_no_persona_200.json")

    print("\n=== merging no_uncertainty ===")
    nu_summary = merge_files(nu_sources, "outputs/eval_v29_ablation_no_uncertainty_200.json")

    report("no_persona", np_summary)
    report("no_uncertainty", nu_summary)
