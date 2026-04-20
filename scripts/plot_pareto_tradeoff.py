import matplotlib.pyplot as plt
import numpy as np

# Data from experiment results
# Format: (method, persona, avg_turns, pass_at_1)

# Baselines (50-state, no per-persona differentiation for Direct/Clarify-first)
data = {
    'Direct Execution': {
        'All': (1.0, 7.3),
    },
    'Clarify-first (K=1)': {
        'All': (2.0, 9.3),
    },
    'Base LLM': {
        'All': (2.2, 12.7),
    },
    'Prompt-only': {
        'Nov': (5.82, 14.0),
        'Exp': (5.38, 6.0),
        'Busy': (5.32, 6.0),
        'All': (5.51, 8.7),
    },
    'TactfulLLM (Ours)': {
        'Nov': (7.0, 18.5),
        'Exp': (2.6, 15.5),
        'Busy': (1.0, 14.0),
        'All': (3.5, 16.0),
    },
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot A: All methods, "All" persona only ---
ax = axes[0]
colors_all = {
    'Direct Execution': '#999999',
    'Clarify-first (K=1)': '#66b3ff',
    'Base LLM': '#ffcc66',
    'Prompt-only': '#ff9999',
    'TactfulLLM (Ours)': '#e74c3c',
}
markers_all = {
    'Direct Execution': 'o',
    'Clarify-first (K=1)': 's',
    'Base LLM': 'D',
    'Prompt-only': '^',
    'TactfulLLM (Ours)': '*',
}

for method, personas in data.items():
    turns, p1 = personas['All']
    ax.scatter(turns, p1, 
               c=colors_all[method], 
               marker=markers_all[method],
               s=200 if method == 'TactfulLLM (Ours)' else 120, 
               edgecolors='black', linewidth=0.8,
               zorder=5, label=method)
    offset_x = 0.1
    offset_y = 0.3
    if method == 'Direct Execution':
        offset_x = 0.1
        offset_y = -1.0
    elif method == 'Clarify-first (K=1)':
        offset_y = -1.0
    elif method == 'Prompt-only':
        offset_x = -0.5
        offset_y = -1.2
    ax.annotate(method, (turns, p1), 
                xytext=(turns + offset_x, p1 + offset_y),
                fontsize=8, ha='left')

ax.set_xlabel('Avg Turns (Cost)', fontsize=12)
ax.set_ylabel('pass@1 (%)', fontsize=12)
ax.set_title('(a) Overall Comparison', fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(4, 20)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='lower right')

# --- Plot B: Per-persona for Ours vs Prompt-only + baselines ---
ax = axes[1]

# Ours per-persona
ours = data['TactfulLLM (Ours)']
ours_turns = [ours['Busy'][0], ours['Exp'][0], ours['Nov'][0]]
ours_p1 = [ours['Busy'][1], ours['Exp'][1], ours['Nov'][1]]
ax.plot(ours_turns, ours_p1, 'r-', linewidth=2, alpha=0.5, zorder=3)
ax.scatter(ours_turns, ours_p1, c='#e74c3c', marker='*', s=250, 
           edgecolors='black', linewidth=0.8, zorder=5, label='Ours')
for persona, (t, p) in [(k, v) for k, v in ours.items() if k != 'All']:
    ox, oy = (0.15, 0.3)
    if persona == 'Busy':
        ox, oy = (0.15, -1.0)
    elif persona == 'Nov':
        ox, oy = (-1.5, -1.2)
    ax.annotate(persona, (t, p), xytext=(t + ox, p + oy), fontsize=9, 
                fontweight='bold', color='#e74c3c')

# Prompt-only per-persona
po = data['Prompt-only']
po_turns = [po['Busy'][0], po['Exp'][0], po['Nov'][0]]
po_p1 = [po['Busy'][1], po['Exp'][1], po['Nov'][1]]
ax.plot(po_turns, po_p1, '--', color='#ff9999', linewidth=1.5, alpha=0.5, zorder=3)
ax.scatter(po_turns, po_p1, c='#ff9999', marker='^', s=150,
           edgecolors='black', linewidth=0.8, zorder=5, label='Prompt-only')
for persona, (t, p) in [(k, v) for k, v in po.items() if k != 'All']:
    ox, oy = (0.15, 0.3)
    if persona == 'Nov':
        oy = 0.5
    elif persona == 'Busy':
        oy = -1.0
    ax.annotate(persona, (t, p), xytext=(t + ox, p + oy), fontsize=8, color='#cc6666')

# Baselines as reference points
for method in ['Direct Execution', 'Base LLM', 'Clarify-first (K=1)']:
    t, p = data[method]['All']
    ax.scatter(t, p, c=colors_all[method], marker=markers_all[method], s=120,
               edgecolors='black', linewidth=0.8, zorder=4, label=method)
    ox = 0.15
    oy = -1.0 if method == 'Direct Execution' else 0.4
    ax.annotate(method, (t, p), xytext=(t + ox, p + oy), fontsize=7, color='gray')

# Arrow showing "ideal direction"
ax.annotate('', xy=(0.5, 19), xytext=(2.0, 19),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.text(0.4, 19.5, 'Better\n(less cost)', fontsize=7, color='green', ha='center')
ax.annotate('', xy=(0.5, 19), xytext=(0.5, 17),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.set_xlabel('Avg Turns (Cost)', fontsize=12)
ax.set_ylabel('pass@1 (%)', fontsize=12)
ax.set_title('(b) Per-Persona Trade-off', fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(4, 21)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ProactiveLLM/pareto_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved to pareto_tradeoff.png")
