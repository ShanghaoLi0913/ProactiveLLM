"""
Combined 2-panel figure for Experiment 2 (Info Recovery / OGR).

(a) Per-persona recovery uplift (Direct → TactfulLLM, with Ideal/Full ceilings).
(b) Per-persona disclosure rate (% of masked info revealed via clarification).

Story:
  (a) shows the OUTCOME — how much pass@1 each persona gains.
  (b) shows the MECHANISM — how much info each persona reveals.
  Together: persona OGR is driven by disclosure obtained.

Output:
  data/analysis/fig_recovery_combined.pdf  (vector, paper)
  data/analysis/fig_recovery_combined.png  (raster, review)
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (Llama canonical-200)
# ---------------------------------------------------------------------------

PERSONAS = ['Novice', 'Experienced', 'Busy']

DIRECT  = {'Novice': 11.5, 'Experienced': 12.5, 'Busy': 13.0}
TACTFUL = {'Novice': 18.5, 'Experienced': 15.5, 'Busy': 14.0}

# Disclosure rate: % of masked items revealed via clarification (TactfulLLM, n=200).
DISCLOSURE = {'Novice': 88.6, 'Experienced': 78.0, 'Busy': 0.0}

PERSONA_COLOR = {
    'Novice':      '#F4A261',  # light orange / peach
    'Experienced': '#94D2BD',  # light mint green
    'Busy':        '#A8DADC',  # light sky blue
}
PERSONA_TEXT_COLOR = {
    'Novice':      '#C77A37',  # darker caramel for label text
    'Experienced': '#4F9485',  # darker mint
    'Busy':        '#4A8DAB',  # darker slate blue
}

IDEAL_Y = 16.0
FULL_Y  = 20.0

def ogr(persona):
    return (TACTFUL[persona] - DIRECT[persona]) / (FULL_Y - DIRECT[persona]) * 100

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6),
                         gridspec_kw={'width_ratios': [1.15, 1.0], 'wspace': 0.35})

# =============================================================================
# Panel (a): Recovery uplift arrows
# =============================================================================
ax = axes[0]
x_positions = list(range(len(PERSONAS)))

# Reference ceilings
ax.axhline(y=FULL_Y,  linestyle='--', color='#444444', linewidth=1.0, alpha=0.7, zorder=1)
ax.axhline(y=IDEAL_Y, linestyle='--', color='#888888', linewidth=0.9, alpha=0.7, zorder=1)
ax.text(2.45, FULL_Y + 0.20, f'Full Query ({FULL_Y:.0f}%)',
        ha='right', va='bottom', fontsize=7, color='#222222', fontweight='bold')
ax.text(2.45, IDEAL_Y + 0.20, f'Ideal Disclosed ({IDEAL_Y:.0f}%)',
        ha='right', va='bottom', fontsize=7, color='#555555', fontweight='bold')

for xi, persona in enumerate(PERSONAS):
    color = PERSONA_COLOR[persona]
    text_color = PERSONA_TEXT_COLOR[persona]
    d_y = DIRECT[persona]
    t_y = TACTFUL[persona]

    arrow = FancyArrowPatch(
        (xi, d_y), (xi, t_y),
        arrowstyle='-|>', mutation_scale=14,
        color=color, linewidth=2.2, alpha=0.9, zorder=3,
    )
    ax.add_patch(arrow)
    ax.scatter(xi, d_y, c='#cccccc', marker='o', s=55,
               edgecolors='black', linewidth=0.6, zorder=4)
    ax.scatter(xi, t_y, c=color, marker='*', s=170,
               edgecolors='black', linewidth=0.7, zorder=5)

    # Value labels
    ax.annotate(f'{d_y:.1f}', (xi, d_y), xytext=(xi - 0.10, d_y),
                ha='right', va='center', fontsize=7, color='#555555')
    ax.annotate(f'{t_y:.1f}', (xi, t_y), xytext=(xi + 0.10, t_y),
                ha='left', va='center', fontsize=8, color=text_color, fontweight='bold')

    # Uplift + OGR
    mid_y = (d_y + t_y) / 2
    uplift = t_y - d_y
    ogr_pct = ogr(persona)
    if persona == 'Novice':
        lx, ly, ha = xi + 0.05, mid_y + 1.8, 'left'
    elif persona == 'Busy':
        lx, ly, ha = xi - 0.20, mid_y - 1.7, 'left'
    else:
        lx, ly, ha = xi - 0.12, mid_y, 'right'
    ax.annotate(f'+{uplift:.1f} pp\n({ogr_pct:.0f}% OGR)',
                (xi, mid_y), xytext=(lx, ly),
                ha=ha, va='center', fontsize=7, color=text_color,
                fontweight='bold', style='italic')

# Callouts
ax.annotate(
    'exceeds\ndisclosure ceiling',
    xy=(0.05, 17.3), xytext=(0.20, 20.4),
    fontsize=7, color='#a02020', style='italic', ha='left',
    arrowprops=dict(arrowstyle='->', color='#a02020', alpha=0.8, linewidth=0.8),
    zorder=6,
)
ax.annotate(
    'by design:\nlow patience',
    xy=(2, 13.5), xytext=(1.20, 9.7),
    fontsize=7, color='#666666', style='italic', ha='left',
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', alpha=0.8, linewidth=0.8),
    zorder=6,
)

ax.set_xticks(x_positions)
ax.set_xticklabels(['Novice', 'Exp.', 'Busy'], fontsize=9, fontweight='bold')
ax.set_xlabel('User Persona', fontsize=9)
ax.set_ylabel('Task Success (pass@1, %)', fontsize=9)
ax.tick_params(axis='y', labelsize=7)
ax.set_ylim(9, 22)
ax.set_xlim(-0.5, 2.6)
ax.grid(True, axis='y', alpha=0.22, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.text(0.02, 0.97, '(a)', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top', ha='left')

# =============================================================================
# Panel (b): Disclosure rate horizontal bars
# =============================================================================
ax = axes[1]

# Y positions: Novice top, Exp middle, Busy bottom
y_positions = list(range(len(PERSONAS)))[::-1]  # reverse so Novice is at top
disclosure_vals = [DISCLOSURE[p] for p in PERSONAS]
colors = [PERSONA_COLOR[p] for p in PERSONAS]

bars = ax.barh(y_positions, disclosure_vals, color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.7, height=0.55)

# Reference line at 100% (oracle disclosure ceiling)
ax.axvline(x=100, linestyle='--', color='#888888', linewidth=1.0, alpha=0.7, zorder=1)
ax.text(91, 2.85, 'Ideal Disclosed\n(100%)',
        ha='center', va='top', fontsize=7, color='#555555', style='italic')

# Value labels at end of each bar
for yi, (persona, val) in enumerate(zip(PERSONAS, disclosure_vals)):
    text_color = PERSONA_TEXT_COLOR[persona]
    label = f'{val:.1f}%' if val > 0 else '0.0%  (no clarify)'
    ax.text(val + 2 if val > 5 else 5, y_positions[yi], label,
            ha='left', va='center', fontsize=8.5,
            color=text_color if val > 0 else '#666666',
            fontweight='bold', style='italic' if val == 0 else 'normal')

ax.set_yticks(y_positions)
ax.set_yticklabels(['Novice', 'Exp.', 'Busy'], fontsize=9, fontweight='bold')
ax.set_xlabel('Disclosure Rate (%)', fontsize=9)
ax.tick_params(axis='x', labelsize=7)
ax.set_xlim(0, 115)
ax.set_ylim(-0.6, len(PERSONAS) - 0.05)  # extra room for 2-line ceiling label
ax.grid(True, axis='x', alpha=0.22, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.text(0.02, 0.97, '(b)', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top', ha='left')

# =============================================================================
# Shared legend at bottom
# =============================================================================
direct_proxy = mlines.Line2D([], [], color='#cccccc', marker='o', markersize=6,
                              markeredgecolor='black', linestyle='None',
                              label='Direct Execution')
tactful_proxy = mlines.Line2D([], [], color='gray', marker='*', markersize=10,
                               markeredgecolor='black', linestyle='None',
                               label='TactfulLLM (Ours)')
fig.legend(handles=[direct_proxy, tactful_proxy],
           loc='upper center', bbox_to_anchor=(0.5, -0.04),
           fontsize=8, frameon=True, ncol=2, columnspacing=1.5,
           handletextpad=0.4)

plt.tight_layout(rect=[0, 0.11, 1, 1])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path('/root/autodl-tmp/ProactiveLLM/data/analysis')
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / 'fig_recovery_combined.pdf'
png_path = out_dir / 'fig_recovery_combined.png'

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=200, bbox_inches='tight')

print(f'Saved {pdf_path}')
print(f'Saved {png_path}')
