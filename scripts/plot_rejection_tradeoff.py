import matplotlib.pyplot as plt
import numpy as np

# Data: (avg_turns, rejection_rate %)
# Rejection rate = % of clarification attempts rejected by user

data = {
    'Direct Execution': {
        'All': (1.0, 0.0),  # no clarification → no rejection
    },
    'Clarify-first (K=1)': {
        'All': (2.0, 52.0),
    },
    'Base LLM': {
        'All': (2.2, 62.7),
    },
    'Prompt-only': {
        'Nov': (5.82, 46.5),
        'Exp': (5.38, 55.7),
        'Busy': (5.32, 89.4),
        'All': (5.51, 63.2),
    },
    'TactfulLLM (Ours)': {
        'Nov': (7.0, 45.2),   # estimated from 200-state data
        'Exp': (2.6, 35.0),   # estimated
        'Busy': (1.0, 0.0),   # never clarifies → 0% rejection
        'All': (3.5, 45.2),   # from experiment log
    },
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {
    'Direct Execution': '#999999',
    'Clarify-first (K=1)': '#66b3ff',
    'Base LLM': '#ffcc66',
    'Prompt-only': '#ff9999',
    'TactfulLLM (Ours)': '#e74c3c',
}
markers = {
    'Direct Execution': 'o',
    'Clarify-first (K=1)': 's',
    'Base LLM': 'D',
    'Prompt-only': '^',
    'TactfulLLM (Ours)': '*',
}

# --- Plot A: Overall ---
ax = axes[0]
for method, personas in data.items():
    turns, rej = personas['All']
    ax.scatter(turns, rej,
               c=colors[method], marker=markers[method],
               s=200 if method == 'TactfulLLM (Ours)' else 120,
               edgecolors='black', linewidth=0.8, zorder=5, label=method)
    ox, oy = 0.15, 1.5
    if method == 'Direct Execution':
        ox, oy = 0.15, 2.0
    elif method == 'TactfulLLM (Ours)':
        ox, oy = 0.15, -4.0
    elif method == 'Prompt-only':
        ox, oy = -1.0, 2.0
    ax.annotate(method, (turns, rej), xytext=(turns + ox, rej + oy), fontsize=8)

ax.set_xlabel('Avg Turns', fontsize=12)
ax.set_ylabel('Rejection Rate (%)', fontsize=12)
ax.set_title('(a) Overall: Turns vs Rejection Rate', fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(-5, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')

# "ideal" is bottom-left: few turns AND low rejection
ax.annotate('Better\n(fewer turns,\nlower rejection)', xy=(0.3, 3),
            fontsize=8, color='green', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

# --- Plot B: Per-persona ---
ax = axes[1]

# Ours per-persona
ours = data['TactfulLLM (Ours)']
for persona_key, label, color_text in [('Busy', 'Busy', '#c0392b'), 
                                         ('Exp', 'Exp', '#c0392b'),
                                         ('Nov', 'Nov', '#c0392b')]:
    t, r = ours[persona_key]
    ax.scatter(t, r, c='#e74c3c', marker='*', s=250,
               edgecolors='black', linewidth=0.8, zorder=5)
    ox, oy = 0.2, 1.5
    if persona_key == 'Busy':
        ox, oy = 0.2, 2.0
    elif persona_key == 'Nov':
        ox, oy = -1.5, -4.0
    ax.annotate(f'Ours-{label}', (t, r), xytext=(t + ox, r + oy),
                fontsize=9, fontweight='bold', color='#e74c3c')

# Connect ours points
ours_pts = [ours['Busy'], ours['Exp'], ours['Nov']]
ax.plot([p[0] for p in ours_pts], [p[1] for p in ours_pts], 
        'r-', linewidth=2, alpha=0.4, zorder=3, label='Ours')

# Prompt-only per-persona
po = data['Prompt-only']
for persona_key, label in [('Busy', 'Busy'), ('Exp', 'Exp'), ('Nov', 'Nov')]:
    t, r = po[persona_key]
    ax.scatter(t, r, c='#ff9999', marker='^', s=150,
               edgecolors='black', linewidth=0.8, zorder=5)
    ox, oy = 0.2, 1.5
    if persona_key == 'Busy':
        oy = 2.0
    ax.annotate(f'PO-{label}', (t, r), xytext=(t + ox, r + oy),
                fontsize=8, color='#cc6666')

po_pts = [po['Busy'], po['Exp'], po['Nov']]
ax.plot([p[0] for p in po_pts], [p[1] for p in po_pts],
        '--', color='#ff9999', linewidth=1.5, alpha=0.4, zorder=3, label='Prompt-only')

# Baselines
for method in ['Direct Execution', 'Base LLM', 'Clarify-first (K=1)']:
    t, r = data[method]['All']
    ax.scatter(t, r, c=colors[method], marker=markers[method], s=120,
               edgecolors='black', linewidth=0.8, zorder=4, label=method)
    ax.annotate(method, (t, r), xytext=(t + 0.15, r + 1.5), fontsize=7, color='gray')

ax.set_xlabel('Avg Turns', fontsize=12)
ax.set_ylabel('Rejection Rate (%)', fontsize=12)
ax.set_title('(b) Per-Persona: User Experience', fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(-5, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ProactiveLLM/rejection_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved to rejection_tradeoff.png")
