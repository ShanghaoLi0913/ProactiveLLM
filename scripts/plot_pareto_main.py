"""Pareto plot for paper main figure: pass@1 vs avg_turns.

Higher pass@1 = better; lower avg_turns = better.
Methods on the Pareto frontier are not dominated.
TactfulLLM on Llama Busy dominates CollabLLM (same pass@1, 4.5x fewer turns).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Llama numbers (n=200) ──
llama_all = {
    'Direct':           {'p1': 12.3, 'avg_t': 1.00},
    'Base':             {'p1': 12.7, 'avg_t': 2.15},
    'Clarify-first':    {'p1': 14.8, 'avg_t': 2.00},
    'Prompt-only':      {'p1': 13.3, 'avg_t': 5.73},
    'Few-shot Persona': {'p1': 13.7, 'avg_t': 3.80},
    'CollabLLM':        {'p1': 15.0, 'avg_t': 4.26},
    'TactfulLLM':       {'p1': 16.0, 'avg_t': 3.50},
}

llama_busy = {
    'Direct':           {'p1': 13.0, 'avg_t': 1.00},
    'Base':             {'p1': 13.0, 'avg_t': 2.51},
    'Clarify-first':    {'p1': 16.5, 'avg_t': 2.00},
    'Prompt-only':      {'p1': 12.0, 'avg_t': 5.71},
    'Few-shot Persona': {'p1': 11.5, 'avg_t': 1.00},
    'CollabLLM':        {'p1': 14.0, 'avg_t': 4.55},
    'TactfulLLM':       {'p1': 14.0, 'avg_t': 1.00},
}


def is_pareto(point, others):
    """Return True if point is not strictly dominated by any other (higher p1, lower avg_t)."""
    p1, t = point
    for op1, ot in others:
        if op1 >= p1 and ot <= t and (op1 > p1 or ot < t):
            return False
    return True


def plot_panel(ax, data, title, show_legend=False):
    # Highlight TactfulLLM
    style = {
        'Direct':           {'c': '#888', 'marker': 's', 'size': 110},
        'Base':             {'c': '#888', 'marker': 'D', 'size': 110},
        'Clarify-first':    {'c': '#3c8c3c', 'marker': '^', 'size': 130},
        'Prompt-only':      {'c': '#bbb', 'marker': 'v', 'size': 110},
        'Few-shot Persona': {'c': '#5e8db8', 'marker': 'P', 'size': 130},
        'CollabLLM':        {'c': '#d97757', 'marker': 'X', 'size': 150},
        'TactfulLLM':       {'c': '#1f4e8c', 'marker': '*', 'size': 320},
    }

    points = [(v['p1'], v['avg_t']) for v in data.values()]

    # Compute Pareto frontier
    pareto_methods = []
    for name, v in data.items():
        others = [(d['p1'], d['avg_t']) for n2, d in data.items() if n2 != name]
        if is_pareto((v['p1'], v['avg_t']), others):
            pareto_methods.append(name)

    # Draw Pareto frontier line (sorted by avg_t ascending)
    pareto_points = sorted([(data[m]['avg_t'], data[m]['p1']) for m in pareto_methods])
    px, py = zip(*pareto_points)
    ax.plot(px, py, '--', color='#888', alpha=0.5, linewidth=1.4, zorder=1, label='Pareto frontier')

    # Plot each method
    for name, v in data.items():
        s = style[name]
        edgecolor = 'black' if name == 'TactfulLLM' else 'none'
        lw = 1.5 if name == 'TactfulLLM' else 0
        ax.scatter(v['avg_t'], v['p1'], c=s['c'], marker=s['marker'], s=s['size'],
                   edgecolors=edgecolor, linewidths=lw, zorder=3, label=name)

    # Annotate methods (offset so they don't overlap markers)
    annot_offsets = {
        'Direct':           (0.10,  -0.55),
        'Base':             (0.10,   0.30),
        'Clarify-first':    (0.10,  -0.55),
        'Prompt-only':      (0.05,  -0.65),
        'Few-shot Persona': (0.10,   0.40),
        'CollabLLM':        (0.10,  -0.55),
        'TactfulLLM':       (0.10,   0.50),
    }
    for name, v in data.items():
        dx, dy = annot_offsets.get(name, (0.10, 0.30))
        ax.annotate(
            name, (v['avg_t'], v['p1']),
            xytext=(v['avg_t'] + dx, v['p1'] + dy),
            fontsize=9,
            color='#1f4e8c' if name == 'TactfulLLM' else '#222',
            fontweight='bold' if name == 'TactfulLLM' else 'normal',
        )

    # Highlight TactfulLLM region with subtle annotation arrow on right panel
    if 'Busy' in title and 'TactfulLLM' in data and 'CollabLLM' in data:
        t = data['TactfulLLM']
        c = data['CollabLLM']
        ax.annotate(
            '', xy=(t['avg_t'], t['p1']), xytext=(c['avg_t'], c['p1']),
            arrowprops=dict(arrowstyle='->', color='#1f4e8c', lw=1.5, alpha=0.7),
        )
        ax.text(
            (t['avg_t'] + c['avg_t']) / 2 + 0.1, (t['p1'] + c['p1']) / 2 - 0.5,
            r'$4.5\times$ fewer turns,' '\nsame pass@1',
            fontsize=8.5, color='#1f4e8c', fontweight='bold', ha='left',
        )

    ax.set_xlabel('Avg.\\ Turns  (lower is better →)', fontsize=11)
    ax.set_ylabel('pass@1 (\\%)  (↑ higher is better)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

plot_panel(axes[0], llama_all,
           r'(a) Llama-3.1-8B  $\cdot$  Aggregate (Nov + Exp + Busy)')
plot_panel(axes[1], llama_busy,
           r'(b) Llama-3.1-8B  $\cdot$  Busy-Developer (low patience)')

axes[0].set_xlim(0.3, 6.5)
axes[0].set_ylim(11.3, 17.2)
axes[1].set_xlim(0.3, 6.5)
axes[1].set_ylim(10.3, 17.7)

# Suptitle
fig.suptitle(
    r'\textbf{Pareto trade-off: task accuracy vs.\ user interruption}',
    fontsize=13, y=1.02,
)

plt.tight_layout()
plt.savefig('docs/fig_pareto_main.png', dpi=200, bbox_inches='tight')
plt.savefig('docs/fig_pareto_main.pdf', bbox_inches='tight')
print('Saved: docs/fig_pareto_main.png  +  .pdf')
