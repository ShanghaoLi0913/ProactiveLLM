"""
Extract per-(state, persona) reference turn count from v29 trajectory tree.

Reference = mainline trajectory's final turn number (i.e., is_mainline=True & is_terminal=True).
This is the canonical "how many turns this task×persona should take" learned from DPO training.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/ProactiveLLM')
TRAJ = ROOT / 'data/logs/traj_v29_100states_combined.jsonl'

# Group events by trajectory_id
trajs = defaultdict(list)
with open(TRAJ) as f:
    for line in f:
        e = json.loads(line)
        trajs[e['trajectory_id']].append(e)

# Find mainline terminal per (state, persona)
ref_turns = {}  # (state_id, persona) -> turn count
mainline_info = defaultdict(list)
for tid, events in trajs.items():
    events.sort(key=lambda x: x.get('turn', 0))
    # A trajectory is mainline if all its events are mainline (or contains mainline terminal)
    mainline_events = [e for e in events if e.get('is_mainline')]
    if not mainline_events:
        continue
    terminal = [e for e in mainline_events if e.get('is_terminal')]
    if not terminal:
        continue
    term = terminal[-1]
    st = term['state']
    sid = st.get('id') if isinstance(st, dict) else st
    if not sid:
        continue
    p = term['persona']
    persona = p['name'] if isinstance(p, dict) else p
    turns = term['turn'] + 1  # turn is 0-indexed; total turns = last turn + 1
    mainline_info[(sid, persona)].append((tid, turns, term.get('task_completed')))

# If multiple mainline terminals for same (state, persona), prefer task_completed=True
for key, runs in mainline_info.items():
    completed = [r for r in runs if r[2]]
    chosen = completed[0] if completed else runs[0]
    ref_turns[key] = chosen[1]

# Summary by persona
from collections import Counter
by_persona = defaultdict(list)
for (sid, p), t in ref_turns.items():
    by_persona[p].append(t)

print(f"Total (state, persona) refs: {len(ref_turns)}")
print(f"Unique states: {len(set(k[0] for k in ref_turns))}\n")
print(f"{'Persona':<25} {'N':>4} {'mean':>6} {'min':>4} {'max':>4}  dist")
for p in sorted(by_persona):
    ts = by_persona[p]
    dist = Counter(ts)
    print(f"{p:<25} {len(ts):>4} {sum(ts)/len(ts):>6.2f} {min(ts):>4} {max(ts):>4}  {dict(sorted(dist.items()))}")

# Save for the table-builder
OUT = ROOT / 'data/analysis/ref_turns_v29.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in ref_turns.items()}, indent=2))
print(f"\nSaved → {OUT}")
