"""
Tier-1 validation for v31 proposal: can a disclosure-based uncertainty
formula drive persona-aware label flips without breaking Experienced?

Reads existing v29 artefacts, applies the PROPOSED formula + rule, and
prints four diagnostic checks. No code changes, no training.

Proposal (v2, disclosure-only):
    uncertainty = (n_masked_total - n_disclosed) / MAX_MASKED
  with MAX_MASKED = 5 (observed maximum across v29 states).
  Dropped the text-based heuristic term; uncertainty now derives
  purely from masking state — matches the paper's framing that
  uncertainty IS "unrevealed masked information".

    get_correct_action(persona, uncertainty):
        Novice:  Clarify if U > THRESH_NOVICE else Execute
        Exp:     Clarify if U > THRESH_EXP    else Execute
        Busy:    Clarify if U > THRESH_BUSY   else Execute

Checks:
  1.1 Proposed-U distribution per (persona, turn): stdev > 0.15 for
      Novice T2/T3, > 0.20 for Busy T0.
  1.2 Novice Execute-flips (target 15-40), Busy Clarify-flips (10-25).
  1.3 Exp label change < 10 pairs (< 6% of Exp pairs).
  1.4 Correlation: proposed-U vs pass@1 on v29 Novice-Learner Execute
      trajectories (expect U higher when Execute fails).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.compute_task_uncertainty import compute_task_uncertainty


PAIRS = PROJECT_ROOT / "data/dpo/prefs_v29_100states.jsonl"
TRAJ = PROJECT_ROOT / "data/logs/traj_v29_100states_combined.jsonl"


def count_items(section: dict) -> int:
    return sum(len(v) for v in section.values() if isinstance(v, list))


def initial_query(query: str) -> str:
    """Strip appended dialogue history, matching render_state.py."""
    if "[Assistant]:" not in query and "[User]:" not in query:
        return query
    out = []
    for line in query.split("\n"):
        if line.strip().startswith(("[Assistant]:", "[User]:")):
            break
        out.append(line)
    return "\n".join(out).strip()


MAX_MASKED = 5  # observed max across v29 states


def proposed_uncertainty(state: dict) -> tuple[float, float, int, int]:
    """Returns (U, _reserved, n_disclosed, n_masked_total).

    U = (n_masked_total - n_disclosed) / MAX_MASKED.  Higher = more unrevealed.
    """
    dr = state.get("disclosure_rule", {}) or {}
    n_masked_total = count_items(dr.get("masked_fields", {}))
    n_disclosed = count_items(dr.get("disclosed_info", {}))
    remaining = max(0, n_masked_total - n_disclosed)
    u = min(1.0, remaining / MAX_MASKED)
    return u, 0.0, n_disclosed, n_masked_total


# Initial proposed thresholds (tune after seeing real distribution)
THRESH_NOVICE = 0.20
THRESH_EXP = 0.30
THRESH_BUSY = 0.60


def proposed_rule(persona_name: str, u: float) -> str:
    if persona_name == "Novice-Learner":
        return "Clarify" if u > THRESH_NOVICE else "Execute"
    if persona_name == "Busy-Developer":
        return "Clarify" if u > THRESH_BUSY else "Execute"
    return "Clarify" if u > THRESH_EXP else "Execute"


# ---------- load data ----------
pairs = []
with PAIRS.open() as f:
    for line in f:
        pairs.append(json.loads(line))

trajs_by_id = defaultdict(list)
with TRAJ.open() as f:
    for line in f:
        d = json.loads(line)
        trajs_by_id[d["trajectory_id"]].append(d)
for tid in trajs_by_id:
    trajs_by_id[tid].sort(key=lambda x: x["turn"])

print(f"Loaded {len(pairs)} pairs, {len(trajs_by_id)} trajectories.\n")


# ---------- 1.1 U distribution per (persona, turn) ----------
print("=" * 70)
print("[1.1] Proposed-U distribution by (persona, dialogue_turn)")
print("=" * 70)
print(f"{'persona':<22}{'turn':>5}{'n':>6}{'mean':>8}{'stdev':>8}{'min':>7}{'max':>7}")
print("-" * 70)

bucket = defaultdict(list)
for p in pairs:
    u, _, _, _ = proposed_uncertainty(p["state"])
    bucket[(p["persona"]["name"], p["dialogue_turn"])].append(u)

targets = {("Novice-Learner", 2): 0.15, ("Novice-Learner", 3): 0.15,
           ("Busy-Developer", 0): 0.20}
pass_11 = {k: False for k in targets}

for (name, t), vals in sorted(bucket.items()):
    if len(vals) < 2:
        sd = 0.0
    else:
        sd = statistics.stdev(vals)
    tgt = targets.get((name, t))
    mark = ""
    if tgt is not None:
        ok = sd > tgt
        mark = f"  [target >{tgt} → {'PASS' if ok else 'FAIL'}]"
        pass_11[(name, t)] = ok
    print(f"{name:<22}{t:>5}{len(vals):>6}{statistics.mean(vals):>8.3f}"
          f"{sd:>8.3f}{min(vals):>7.2f}{max(vals):>7.2f}{mark}")

print()
verdict_11 = all(pass_11.values())
print(f"[1.1 verdict] {'PASS' if verdict_11 else 'FAIL'} "
      f"(required buckets: {sum(pass_11.values())}/{len(pass_11)})")


# ---------- 1.2 / 1.3 label flips under proposed rule ----------
print()
print("=" * 70)
print("[1.2/1.3] Label flips: v29 chosen_action vs proposed_rule(persona, U)")
print("=" * 70)

flip_counter = defaultdict(Counter)  # persona -> Counter("C->E", "E->C", "same")
total_by_persona = Counter()
for p in pairs:
    u, _, _, _ = proposed_uncertainty(p["state"])
    proposed = proposed_rule(p["persona"]["name"], u)
    orig = p["chosen_action"]
    name = p["persona"]["name"]
    total_by_persona[name] += 1
    if proposed == orig:
        flip_counter[name]["same"] += 1
    elif orig == "Clarify" and proposed == "Execute":
        flip_counter[name]["C->E"] += 1
    else:
        flip_counter[name]["E->C"] += 1

for name in ["Novice-Learner", "Experienced-Engineer", "Busy-Developer"]:
    c = flip_counter[name]
    tot = total_by_persona[name]
    print(f"\n  {name} (n={tot})")
    print(f"    same : {c['same']}")
    print(f"    C->E : {c['C->E']} (Clarify→Execute)")
    print(f"    E->C : {c['E->C']} (Execute→Clarify)")
    print(f"    total flips: {c['C->E'] + c['E->C']} "
          f"({(c['C->E']+c['E->C'])/tot*100:.1f}%)")

novice_exec_flips = flip_counter["Novice-Learner"]["C->E"]
busy_clarify_flips = flip_counter["Busy-Developer"]["E->C"]
exp_flips = flip_counter["Experienced-Engineer"]["C->E"] + flip_counter["Experienced-Engineer"]["E->C"]
exp_total = total_by_persona["Experienced-Engineer"]

print()
print(f"  Novice C->E flips : {novice_exec_flips}   target 15-40  → "
      f"{'PASS' if 15 <= novice_exec_flips <= 40 else 'FAIL'}")
print(f"  Busy   E->C flips : {busy_clarify_flips}   target 10-25  → "
      f"{'PASS' if 10 <= busy_clarify_flips <= 25 else 'FAIL'}")
pct = exp_flips / exp_total * 100 if exp_total else 0
print(f"  Exp    any flips  : {exp_flips}/{exp_total} ({pct:.1f}%)   "
      f"target <6%     → {'PASS' if pct < 6.0 else 'FAIL'}")

verdict_12 = 15 <= novice_exec_flips <= 40 and 10 <= busy_clarify_flips <= 25
verdict_13 = pct < 6.0


# ---------- 1.4 U vs pass@1 on v29 Novice Execute actions ----------
print()
print("=" * 70)
print("[1.4] Correlation: proposed-U vs code-pass on Novice Execute turns")
print("=" * 70)

# For each Novice-Learner trajectory, take the (first) Execute turn and
# record (U_at_execute, task_completed).
rows = []
for tid, turns in trajs_by_id.items():
    persona = turns[0]["persona"]["name"]
    if persona != "Novice-Learner":
        continue
    for t in turns:
        if t["action"] != "Execute":
            continue
        u, _, nd, nm = proposed_uncertainty(t["state"])
        passed = bool(t.get("task_completed"))
        rows.append((u, passed, nd, nm))
        break

print(f"  Novice Execute rows : {len(rows)}")
if len(rows) < 10:
    print("  [1.4 verdict] INSUFFICIENT DATA")
    verdict_14 = None
else:
    # bin by U
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    print(f"  {'U range':<12}{'n':>5}{'pass_rate':>12}{'mean_U':>9}"
          f"{'mean_disc':>11}")
    print("  " + "-" * 50)
    rates = []
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r[0] < hi]
        if not sub:
            print(f"  [{lo:.2f},{hi:.2f})  {0:>5}{'-':>12}{'-':>9}{'-':>11}")
            continue
        pr = sum(r[1] for r in sub) / len(sub)
        mu = sum(r[0] for r in sub) / len(sub)
        mdisc = sum((r[2] / r[3] if r[3] else 0) for r in sub) / len(sub)
        print(f"  [{lo:.2f},{hi:.2f})  {len(sub):>5}{pr:>12.2%}{mu:>9.3f}"
              f"{mdisc:>11.2%}")
        rates.append((mu, pr, len(sub)))

    # Spearman-lite: simple Pearson on U vs pass (0/1)
    us = [r[0] for r in rows]
    ps = [1.0 if r[1] else 0.0 for r in rows]
    mu_u = sum(us) / len(us)
    mu_p = sum(ps) / len(ps)
    num = sum((u - mu_u) * (p - mu_p) for u, p in zip(us, ps))
    du = (sum((u - mu_u) ** 2 for u in us)) ** 0.5
    dp = (sum((p - mu_p) ** 2 for p in ps)) ** 0.5
    r = num / (du * dp) if du and dp else 0.0
    print(f"\n  Pearson(U, pass@1)  = {r:+.3f}   (expect NEGATIVE: higher U → lower pass)")
    verdict_14 = r < -0.10
    print(f"  [1.4 verdict] {'PASS' if verdict_14 else 'WEAK'} "
          "(threshold: r < -0.10)")


# ---------- overall verdict ----------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
lines = [
    ("1.1  U variance per bucket", verdict_11),
    ("1.2  Novice/Busy flip counts", verdict_12),
    ("1.3  Exp label preservation <6%", verdict_13),
    ("1.4  U vs pass@1 correlation", verdict_14),
]
for name, v in lines:
    tag = "PASS" if v is True else ("WEAK" if v is False else "N/A")
    print(f"  {tag:<5} {name}")

crit = [verdict_11, verdict_12, verdict_13]
if all(crit):
    print("\n  => GO: proceed to implement v31 pipeline.")
elif verdict_12 and verdict_13 and not verdict_11:
    print("\n  => PARTIAL: flips look right but U variance thin — tune thresholds.")
else:
    print("\n  => NO-GO: rework formula or thresholds before training.")
