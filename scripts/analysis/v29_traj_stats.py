"""Comprehensive v29 trajectory turn statistics."""
import json
from collections import Counter, defaultdict
from pathlib import Path

TRAJ = Path('/root/autodl-tmp/ProactiveLLM/data/logs/traj_v29_100states_combined.jsonl')

events = []
with open(TRAJ) as f:
    for line in f:
        events.append(json.loads(line))

print(f'Total events: {len(events)}')
print(f'Unique trajectory_ids: {len(set(e["trajectory_id"] for e in events))}')
print(f'Unique states: {len(set(e["state"]["id"] for e in events))}')

# Group by trajectory_id
trajs = defaultdict(list)
for e in events:
    trajs[e['trajectory_id']].append(e)
for tid in trajs:
    trajs[tid].sort(key=lambda x: x['turn'])

# ============================================================
# View 1: per-trajectory_id (each branch counted separately)
# ============================================================
print('\n' + '='*70)
print('VIEW 1: per-trajectory_id (each DPO branch is one trajectory)')
print('='*70)
by_persona_all = defaultdict(list)
by_persona_mainline = defaultdict(list)
for tid, evs in trajs.items():
    persona = evs[0]['persona']['name']
    n_turns = evs[-1]['turn'] + 1
    by_persona_all[persona].append(n_turns)
    if all(e['is_mainline'] for e in evs):
        by_persona_mainline[persona].append(n_turns)

print(f'\n{"Persona":<22} {"N":>4} {"mean":>6} {"min":>4} {"max":>4}  distribution')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    ts = by_persona_all[p]
    d = dict(sorted(Counter(ts).items()))
    print(f'{p:<22} {len(ts):>4} {sum(ts)/len(ts):>6.2f} {min(ts):>4} {max(ts):>4}  {d}')

print('\n-- all-mainline trajectories only --')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    ts = by_persona_mainline[p]
    if not ts:
        continue
    d = dict(sorted(Counter(ts).items()))
    print(f'{p:<22} {len(ts):>4} {sum(ts)/len(ts):>6.2f} {min(ts):>4} {max(ts):>4}  {d}')

# ============================================================
# View 2: per-(state, persona) — mainline reference
# ============================================================
print('\n' + '='*70)
print('VIEW 2: per (state, persona) — mainline-terminal turn count')
print('='*70)
ref_turns = defaultdict(list)
for tid, evs in trajs.items():
    term = [e for e in evs if e['is_terminal'] and e['is_mainline']]
    if not term:
        continue
    sid = term[-1]['state']['id']
    persona = term[-1]['persona']['name']
    ref_turns[(sid, persona)].append(term[-1]['turn'] + 1)

ref_final = {k: max(v) for k, v in ref_turns.items()}  # deepest mainline terminal

by_persona = defaultdict(list)
for (sid, p), n in ref_final.items():
    by_persona[p].append(n)
print(f'\n{"Persona":<22} {"N":>4} {"mean":>6} {"min":>4} {"max":>4}  distribution')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    ts = by_persona[p]
    d = dict(sorted(Counter(ts).items()))
    print(f'{p:<22} {len(ts):>4} {sum(ts)/len(ts):>6.2f} {min(ts):>4} {max(ts):>4}  {d}')

# ============================================================
# View 3: per-(state, persona) — deepest branch (any trajectory)
# ============================================================
print('\n' + '='*70)
print('VIEW 3: per (state, persona) — DEEPEST trajectory (any branch)')
print('='*70)
deepest = defaultdict(int)
for tid, evs in trajs.items():
    sid = evs[0]['state']['id']
    persona = evs[0]['persona']['name']
    n = evs[-1]['turn'] + 1
    if n > deepest[(sid, persona)]:
        deepest[(sid, persona)] = n

by_persona = defaultdict(list)
for (sid, p), n in deepest.items():
    by_persona[p].append(n)
print(f'\n{"Persona":<22} {"N":>4} {"mean":>6} {"min":>4} {"max":>4}  distribution')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    ts = by_persona[p]
    d = dict(sorted(Counter(ts).items()))
    print(f'{p:<22} {len(ts):>4} {sum(ts)/len(ts):>6.2f} {min(ts):>4} {max(ts):>4}  {d}')

# ============================================================
# View 4: action distribution per persona
# ============================================================
print('\n' + '='*70)
print('VIEW 4: action distribution per persona (across all events)')
print('='*70)
actions = defaultdict(Counter)
for e in events:
    actions[e['persona']['name']][e['action']] += 1
print(f'\n{"Persona":<22} {"Clarify":>8} {"Execute":>8} {"Other":>8} {"ClarRate":>10}')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    cs = actions[p]
    clar = cs.get('Clarify', 0)
    exe  = cs.get('Execute', 0)
    other = sum(cs.values()) - clar - exe
    total = sum(cs.values())
    print(f'{p:<22} {clar:>8} {exe:>8} {other:>8} {clar/total:>10.2%}')

# ============================================================
# View 5: mainline only — per (state, persona) Clarify count
# ============================================================
print('\n' + '='*70)
print('VIEW 5: mainline-only Clarify count per (state, persona)')
print('='*70)
mainline_clar = defaultdict(int)
mainline_states = set()
for e in events:
    if e['is_mainline']:
        key = (e['state']['id'], e['persona']['name'])
        mainline_states.add(key)
        if e['action'] == 'Clarify':
            mainline_clar[key] += 1

by_p = defaultdict(list)
for key in mainline_states:
    by_p[key[1]].append(mainline_clar[key])
print(f'\n{"Persona":<22} {"N":>4} {"avg Clar":>9}  distribution')
for p in ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']:
    cs = by_p[p]
    d = dict(sorted(Counter(cs).items()))
    print(f'{p:<22} {len(cs):>4} {sum(cs)/len(cs):>9.2f}  {d}')
