"""
Per-(task, persona) turn reasonableness.

Reference = v29 mainline trajectory's total_turns for that (state_id, persona).
For each method & task present in that method's eval, compute:
  abs_dev = |actual - ref|
  signed_dev = actual - ref    (negative = cut short, positive = over-asking)
Aggregate per (method, persona) and report exact-match rate.
Also show 8 example tasks per persona where ≥3 methods overlap.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/ProactiveLLM')
REF_FILE = ROOT / 'data/analysis/ref_turns_v29.json'

ref_raw = json.loads(REF_FILE.read_text())
ref_turns = {tuple(k.split('|')): v for k, v in ref_raw.items()}

METHOD_FILES = {
    'Direct':        ['outputs/eval_v29_direct_execution_200.json'],
    'Clarify-1st':   ['outputs/eval_v29_clarify_first_50test.json'],
    'Base-LLM':      ['outputs/eval_v29_base_llama.json',
                      'outputs/eval_v29_base_150extra_remaining.json'],
    'Prompt-only':   ['outputs/eval_v29_prompt_only_50test.json'],
    'TactfulLLM':    ['outputs/eval_v29_100states.json',
                      'outputs/eval_v29_dpo_150extra.json'],
}
METHOD_ORDER = list(METHOD_FILES)
PERSONAS = ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']
SHORT = {'Busy-Developer': 'Busy', 'Experienced-Engineer': 'Exp', 'Novice-Learner': 'Novice'}

turns = defaultdict(dict)
for m, paths in METHOD_FILES.items():
    for p in paths:
        for c in json.load(open(ROOT / p))['detailed_results']:
            turns[(m, c['persona'])][c['state_id']] = c['total_turns']

# -------- Aggregate per (method, persona) against reference --------
print('=== AGGREGATE: each method vs v29 mainline reference ===')
print(f'{"Method":<14} {"Persona":<8} {"N":>4} {"ref μ":>6} {"actual μ":>9} {"mean |Δ|":>10} {"mean Δ":>9} {"match%":>7}')
for persona in PERSONAS:
    for m in METHOD_ORDER:
        sids = [s for s in turns[(m, persona)] if (s, persona) in ref_turns]
        if not sids:
            continue
        actual, ref_vals, absd, sd = [], [], [], []
        exact = 0
        for s in sids:
            r = ref_turns[(s, persona)]
            t = turns[(m, persona)][s]
            ref_vals.append(r); actual.append(t)
            absd.append(abs(t - r)); sd.append(t - r)
            if t == r:
                exact += 1
        print(f'{m:<14} {SHORT[persona]:<8} {len(sids):>4} '
              f'{np.mean(ref_vals):>6.2f} {np.mean(actual):>9.2f} '
              f'{np.mean(absd):>10.2f} {np.mean(sd):>+9.2f} {100*exact/len(sids):>6.1f}%')
    print()

# -------- Example tasks: overlap ≥3 methods --------
print('\n=== PER-TASK EXAMPLES (tasks with ≥3 methods + ref available) ===')
for persona in PERSONAS:
    coverage = defaultdict(list)
    for m in METHOD_ORDER:
        for s in turns[(m, persona)]:
            if (s, persona) in ref_turns:
                coverage[s].append(m)
    candidates = sorted(s for s, ms in coverage.items() if len(ms) >= 3)
    print(f'\n--- {SHORT[persona]}: {len(candidates)} tasks with ≥3 methods ---')
    if not candidates:
        continue
    hdr = f'{"state_id":<22} {"ref":>4} | ' + ' '.join(f'{m:>11}' for m in METHOD_ORDER)
    print(hdr); print('-' * len(hdr))
    for sid in candidates[:8]:
        ref = ref_turns[(sid, persona)]
        row = f'{sid:<22} {ref:>4} | '
        cells = []
        for m in METHOD_ORDER:
            t = turns[(m, persona)].get(sid)
            cells.append('—' if t is None else f'{t:>2} (Δ{t-ref:+d})')
        row += ' '.join(f'{c:>11}' for c in cells)
        print(row)
