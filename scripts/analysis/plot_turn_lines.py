"""
Exp 1 Part 2: per-task turn-count lines.
One subplot per persona; one line per method (x=task index sorted by state_id, y=turns).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/root/autodl-tmp/ProactiveLLM')

METHOD_FILES = {
    'Direct':        ['outputs/eval_v29_direct_execution_200.json'],
    'Clarify-first': ['outputs/eval_v29_clarify_first_50test.json'],
    'Base LLM':      ['outputs/eval_v29_base_llama.json', 'outputs/eval_v29_base_150extra_remaining.json'],
    'Prompt-only':   ['outputs/eval_v29_prompt_only_50test.json'],
    'TactfulLLM':    ['outputs/eval_v29_100states.json', 'outputs/eval_v29_dpo_150extra.json'],
}
METHOD_ORDER = ['Direct', 'Clarify-first', 'Base LLM', 'Prompt-only', 'TactfulLLM']
PERSONAS = ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']
PERSONA_SHORT = {'Novice-Learner': 'Novice', 'Experienced-Engineer': 'Experienced', 'Busy-Developer': 'Busy'}

COLORS = {
    'Direct':        '#999999',
    'Clarify-first': '#1f77b4',
    'Base LLM':      '#ff9f1c',
    'Prompt-only':   '#9467bd',
    'TactfulLLM':    '#e63946',
}
LINESTYLE = {
    'Direct':        ':',
    'Clarify-first': '-',
    'Base LLM':      '-',
    'Prompt-only':   '-',
    'TactfulLLM':    '-',
}
LINEWIDTH = {
    'Direct':        1.6,
    'Clarify-first': 1.2,
    'Base LLM':      1.2,
    'Prompt-only':   1.2,
    'TactfulLLM':    2.2,
}


def load_turns():
    """Returns dict[(method, persona)] -> list of (state_id, turns) sorted by state_id."""
    out = defaultdict(list)
    for m, paths in METHOD_FILES.items():
        tmp = defaultdict(dict)
        for p in paths:
            with open(ROOT / p) as f:
                for c in json.load(f)['detailed_results']:
                    tmp[c['persona']][c['state_id']] = c['total_turns']
        for persona, d in tmp.items():
            out[(m, persona)] = sorted(d.items(), key=lambda kv: kv[0])
    return out


data = load_turns()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, persona in zip(axes, PERSONAS):
    for m in METHOD_ORDER:
        pairs = data[(m, persona)]
        if not pairs:
            continue
        xs = np.arange(len(pairs))
        ys = [t for _, t in pairs]
        mean = np.mean(ys)
        ax.plot(xs, ys,
                color=COLORS[m], linestyle=LINESTYLE[m], linewidth=LINEWIDTH[m],
                alpha=0.85, label=f'{m} (μ={mean:.1f}, n={len(ys)})')

    ax.set_title(f'{PERSONA_SHORT[persona]} (patience={["low","mid","high"][PERSONAS.index(persona)]})',
                 fontsize=12)
    ax.set_xlabel('Task index (sorted by state_id)')
    ax.set_ylim(0, 8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

axes[0].set_ylabel('Turns per conversation')
fig.suptitle('Per-task turn count by method × persona', fontsize=13)
plt.tight_layout()

OUT = ROOT / 'data' / 'analysis' / 'fig_turn_per_task.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT}")

# Also print summary
print(f"\n{'Method':<14} {'Persona':<14} {'N':>4} {'Turn μ':>7} {'σ':>6} {'min':>4} {'max':>4}")
for m in METHOD_ORDER:
    for persona in PERSONAS:
        pairs = data[(m, persona)]
        if not pairs:
            continue
        turns = [t for _, t in pairs]
        print(f'{m:<14} {PERSONA_SHORT[persona]:<14} {len(turns):>4} '
              f'{np.mean(turns):>7.2f} {np.std(turns):>6.2f} {min(turns):>4} {max(turns):>4}')
