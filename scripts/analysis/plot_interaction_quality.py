"""
Exp 1 Part 2: interaction-quality metrics.
Turn-count is relative; use two turn-invariant metrics:
  (A) Clarification Success Rate = answered / asked
  (B) Turn-count distribution per persona (violin) — read in context of persona
      "tolerance budget" reference line.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/root/autodl-tmp/ProactiveLLM')

# Patience level → canonical "tolerance budget" (in turns).
# Derived from patience probabilities: floor at which P(survive)=~0.5
# Novice (p=0.98): ≈ 30 turns before random reject, but capped by disclosure
# Experienced (p=0.75): 0.75^2≈0.56, so ~2 turns
# Busy (p=0.20): 0.2^1=0.2, so effectively ~1 turn (ideal 0)
PATIENCE_BUDGET = {
    'Novice-Learner': 5,
    'Experienced-Engineer': 2,
    'Busy-Developer': 1,
}
PERSONAS = ['Busy-Developer', 'Experienced-Engineer', 'Novice-Learner']
PERSONA_SHORT = {'Novice-Learner': 'Novice', 'Experienced-Engineer': 'Exp', 'Busy-Developer': 'Busy'}

METHOD_FILES = {
    'Direct':        ['outputs/eval_v29_direct_execution_200.json'],
    'Clarify-first': ['outputs/eval_v29_clarify_first_50test.json'],
    'Base LLM':      ['outputs/eval_v29_base_llama.json', 'outputs/eval_v29_base_150extra_remaining.json'],
    'Prompt-only':   ['outputs/eval_v29_prompt_only_50test.json'],
    'TactfulLLM':    ['outputs/eval_v29_100states.json', 'outputs/eval_v29_dpo_150extra.json'],
}
METHOD_ORDER = ['Direct', 'Clarify-first', 'Base LLM', 'Prompt-only', 'TactfulLLM']

COLORS = {
    'Direct':        '#999999',
    'Clarify-first': '#66b3ff',
    'Base LLM':      '#ffcc66',
    'Prompt-only':   '#ff9999',
    'TactfulLLM':    '#e74c3c',
}


def load(paths):
    rows = []
    for p in paths:
        with open(ROOT / p) as f:
            d = json.load(f)
        rows.extend(d['detailed_results'])
    return rows


def compute_metrics(convos):
    """Returns dict[persona] -> {turns: [...], clar_asked: int, clar_answered: int}"""
    out = defaultdict(lambda: {'turns': [], 'clar_asked': 0, 'clar_answered': 0})
    for c in convos:
        p = c['persona']
        out[p]['turns'].append(c['total_turns'])
        # count clarification turns + answered
        for turn in c.get('conversation', []):
            if turn.get('action') == 'Clarify':
                out[p]['clar_asked'] += 1
                if turn.get('answered_clarification') == 1:
                    out[p]['clar_answered'] += 1
    return out


# --- Load everything ---
data = {m: compute_metrics(load(paths)) for m, paths in METHOD_FILES.items()}

# --- Print summary table for sanity ---
print(f"{'Method':<14} {'Persona':<22} {'N':>4} {'Turns(mean)':>11} {'ClarAsk':>8} {'ClarAns':>8} {'ClarSR':>8}")
for m in METHOD_ORDER:
    for p in PERSONAS:
        st = data[m][p]
        n = len(st['turns'])
        mt = np.mean(st['turns']) if n else 0
        a, b = st['clar_answered'], st['clar_asked']
        sr = a / b if b else float('nan')
        print(f"{m:<14} {p:<22} {n:>4} {mt:>11.2f} {b:>8} {a:>8} {sr:>8.2f}")
    print()

# ========================================================================
# PLOT 1 — Clarification Success Rate (grouped bar, per persona × method)
# ========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

# Clarification success rate
x = np.arange(len(PERSONAS))
width = 0.16
for i, m in enumerate(METHOD_ORDER):
    vals = []
    for p in PERSONAS:
        st = data[m][p]
        a, b = st['clar_answered'], st['clar_asked']
        vals.append(100 * a / b if b else np.nan)
    offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
    bars = ax1.bar(x + offset, vals, width, color=COLORS[m], edgecolor='black',
                   linewidth=0.6, alpha=0.9, label=m)
    for b_, v in zip(bars, vals):
        if not np.isnan(v):
            ax1.text(b_.get_x() + b_.get_width() / 2, v + 1.5, f'{v:.0f}',
                     ha='center', fontsize=7, color='dimgray')
        else:
            ax1.text(b_.get_x() + b_.get_width() / 2, 2, 'N/A',
                     ha='center', fontsize=7, color='gray', style='italic')

ax1.set_xticks(x)
ax1.set_xticklabels([PERSONA_SHORT[p] for p in PERSONAS])
ax1.set_ylabel('Clarification Success Rate (%)')
ax1.set_title('(a) Clarification Success Rate — answered / asked')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.legend(fontsize=8, loc='upper left', ncol=2)

# ========================================================================
# PLOT 2 — Turn distribution per persona × method, with budget line
# ========================================================================
positions = []
labels = []
turn_data = []
colors_for_violin = []
gap = 0
pos_center_per_persona = {}
idx = 0
for p_i, p in enumerate(PERSONAS):
    persona_positions = []
    for m_i, m in enumerate(METHOD_ORDER):
        idx += 1
        positions.append(idx)
        labels.append(m if p_i == 0 else '')
        turn_data.append(data[m][p]['turns'])
        colors_for_violin.append(COLORS[m])
        persona_positions.append(idx)
    pos_center_per_persona[p] = np.mean(persona_positions)
    idx += 1  # gap between persona groups

# Violin plot
parts = ax2.violinplot(turn_data, positions=positions, widths=0.85,
                        showmeans=True, showmedians=False, showextrema=False)
for body, color in zip(parts['bodies'], colors_for_violin):
    body.set_facecolor(color)
    body.set_edgecolor('black')
    body.set_alpha(0.75)
parts['cmeans'].set_edgecolor('black')
parts['cmeans'].set_linewidth(1.0)

# Horizontal budget reference lines per persona
for p in PERSONAS:
    c = pos_center_per_persona[p]
    budget = PATIENCE_BUDGET[p]
    ax2.axhline(budget, xmin=(c - 2.8 - 0.5) / idx, xmax=(c + 2.8 - 0.5) / idx,
                color='green', linestyle='--', linewidth=1.3, alpha=0.7)
    ax2.text(c + 2.9, budget, f'budget={budget}', fontsize=8, color='green',
             va='center')

# Persona group labels on x-axis (below ticks)
ax2.set_xticks([pos_center_per_persona[p] for p in PERSONAS])
ax2.set_xticklabels([f'{PERSONA_SHORT[p]}\n(patience {PATIENCE_BUDGET[p]})' for p in PERSONAS],
                     fontsize=10)

ax2.set_ylabel('Turns per conversation')
ax2.set_title('(b) Turn distribution vs persona tolerance budget')
ax2.set_ylim(0, max(max(t) if t else 0 for t in turn_data) * 1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Custom legend for ax2 (reuse method colors)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=COLORS[m], edgecolor='black', label=m) for m in METHOD_ORDER]
ax2.legend(handles=legend_handles, fontsize=8, loc='upper left', ncol=2)

plt.tight_layout()
OUT = ROOT / 'data' / 'analysis' / 'fig_interaction_quality.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUT}")
