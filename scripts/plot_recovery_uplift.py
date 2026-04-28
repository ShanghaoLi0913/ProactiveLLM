"""
Per-persona uplift arrows for Experiment 2 (Info Recovery / OGR).

Each persona has a colored vertical arrow from Direct Execution → TactfulLLM,
arrow length = task-success uplift. Two horizontal dashed reference lines
(Ideal Disclosed at 16%, Full Query at 20%) sit in the background.

Story:
- Novice's red arrow crosses the Ideal Disclosed line — TactfulLLM exceeds
  oracle freebie disclosure for high-patience users.
- Experienced stops just below Ideal — close but not over.
- Busy barely moves — by design (low patience → minimal clarification).

Single-column NeurIPS sizing (figsize ~3.5×3.5 in).

Output:
  data/analysis/fig_recovery_uplift.pdf  (vector, paper)
  data/analysis/fig_recovery_uplift.png  (raster, review)
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

PERSONA_COLOR = {
    'Novice':      '#d62728',
    'Experienced': '#1f77b4',
    'Busy':        '#ff7f0e',
}

IDEAL_Y = 16.0
FULL_Y  = 20.0

# Oracle Gap Recovery: fraction of (Full Query - Direct) gap that TactfulLLM closes.
def ogr(persona):
    return (TACTFUL[persona] - DIRECT[persona]) / (FULL_Y - DIRECT[persona]) * 100

# ---------------------------------------------------------------------------
# Render — single-column NeurIPS sizing
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

x_positions = list(range(len(PERSONAS)))

# --- Reference ceilings (background, subtle) ---
ax.axhline(y=FULL_Y,  linestyle='--', color='#444444', linewidth=1.0, alpha=0.7, zorder=1)
ax.axhline(y=IDEAL_Y, linestyle='--', color='#888888', linewidth=0.9, alpha=0.7, zorder=1)
ax.text(2.45, FULL_Y + 0.20, f'Full Query ({FULL_Y:.0f}%)',
        ha='right', va='bottom', fontsize=7, color='#222222', fontweight='bold')
ax.text(2.45, IDEAL_Y + 0.20, f'Ideal Disclosed ({IDEAL_Y:.0f}%)',
        ha='right', va='bottom', fontsize=7, color='#555555', fontweight='bold')

# --- Per-persona uplift arrows ---
for xi, persona in enumerate(PERSONAS):
    color = PERSONA_COLOR[persona]
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
                ha='left', va='center', fontsize=8, color=color, fontweight='bold')

    # Uplift + OGR label (per-persona placement to avoid overlap and emphasize)
    mid_y = (d_y + t_y) / 2
    uplift = t_y - d_y
    ogr_pct = ogr(persona)
    if persona == 'Novice':
        # Right of arrow, upper portion (clear of Exp star)
        lx, ly, ha = xi + 0.15, mid_y + 1.8, 'left'
    elif persona == 'Busy':
        # Lower-left of arrow base
        lx, ly, ha = xi - 0.10, mid_y - 1.7, 'left'
    else:
        # Default (Experienced): left of arrow, midpoint
        lx, ly, ha = xi - 0.12, mid_y, 'right'
    ax.annotate(f'+{uplift:.1f} pp\n({ogr_pct:.0f}% OGR)',
                (xi, mid_y), xytext=(lx, ly),
                ha=ha, va='center', fontsize=7.0, color=color,
                fontweight='bold', style='italic')

# --- Highlight: Novice exceeds Ideal Disclosed ---
ax.annotate(
    'exceeds\ndisclosure ceiling',
    xy=(0.05, 17.3), xytext=(0.45, 19.2),
    fontsize=7, color='#a02020', style='italic', ha='left',
    arrowprops=dict(arrowstyle='->', color='#a02020', alpha=0.8, linewidth=0.8),
    zorder=6,
)

# --- Busy "by design" framing ---
ax.annotate(
    'by design:\nlow patience',
    xy=(2, 13.5), xytext=(1.20, 9.8),
    fontsize=7, color='#666666', style='italic', ha='left',
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', alpha=0.8, linewidth=0.8),
    zorder=6,
)

# --- Axes formatting ---
ax.set_xticks(x_positions)
ax.set_xticklabels(['Novice', 'Exp.', 'Busy'], fontsize=9, fontweight='bold')
ax.set_xlabel('User Persona', fontsize=9)
ax.set_ylabel('Task Success (pass@1, %)', fontsize=9)
ax.tick_params(axis='y', labelsize=7.5)
ax.set_ylim(9, 22)
ax.set_xlim(-0.5, 2.6)
ax.grid(True, axis='y', alpha=0.22, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# --- Legend below figure ---
direct_proxy = mlines.Line2D([], [], color='#cccccc', marker='o', markersize=6,
                              markeredgecolor='black', linestyle='None',
                              label='Direct Execution')
tactful_proxy = mlines.Line2D([], [], color='gray', marker='*', markersize=10,
                               markeredgecolor='black', linestyle='None',
                               label='TactfulLLM (Ours)')
ax.legend(handles=[direct_proxy, tactful_proxy],
          loc='upper center', bbox_to_anchor=(0.5, -0.18),
          fontsize=8, frameon=True, ncol=2, columnspacing=1.2,
          handletextpad=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 1])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path('/root/autodl-tmp/ProactiveLLM/data/analysis')
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / 'fig_recovery_uplift.pdf'
png_path = out_dir / 'fig_recovery_uplift.png'

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=200, bbox_inches='tight')

print(f'Saved {pdf_path}')
print(f'Saved {png_path}')
