"""
Recovery slope chart for Experiment 2 (Info Recovery / OGR).

Per-persona pass@1 trajectory along the clarification recovery path
(Direct Execution → TactfulLLM → Ideal Disclosed → Full Query),
Llama-3.1-8B 200-state.

Story:
- Direct Execution: tight cluster (persona variation = sampling noise on identical prompt)
- TactfulLLM: maximum spread (persona-aware adaptation)
- Ideal Disclosed / Full Query: oracle reference ceilings (persona-independent)

Output:
  data/analysis/fig_recovery_slope.pdf  (vector, paper)
  data/analysis/fig_recovery_slope.png  (raster, review)
"""
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (Llama canonical-200)
# ---------------------------------------------------------------------------

CONDITIONS = ['Direct\nExecution', 'TactfulLLM\n(Ours)', 'Ideal\nDisclosed', 'Full\nQuery']

# Per-persona pass@1 (only columns 0 and 1 carry per-persona data;
# columns 2 and 3 are persona-independent reference ceilings).
DATA = {
    'Novice':       [11.5, 18.5],
    'Experienced':  [12.5, 15.5],
    'Busy':         [13.0, 14.0],
}

PERSONA_COLOR = {
    'Novice':      '#d62728',
    'Experienced': '#1f77b4',
    'Busy':        '#ff7f0e',
}
PERSONA_MARKER = {
    'Novice':      'o',
    'Experienced': 's',
    'Busy':        'D',
}

IDEAL_Y = 16.0
FULL_Y  = 20.0

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))

x = list(range(len(CONDITIONS)))

# --- Reference ceilings, full-width dashed lines ---
ax.axhline(y=FULL_Y,  linestyle='--', color='#444444', linewidth=1.6, alpha=0.85, zorder=2)
ax.axhline(y=IDEAL_Y, linestyle='--', color='#888888', linewidth=1.4, alpha=0.85, zorder=2)

# Anchor each reference line label DIRECTLY ABOVE its corresponding stage column
# (Ideal Disclosed → column 2, Full Query → column 3). This visually links the
# x-axis stage label with the horizontal ceiling.
ax.text(2, IDEAL_Y + 0.35, f'Ideal Disclosed  ({IDEAL_Y:.0f}%)',
        ha='center', va='bottom', fontsize=10.5, color='#555555', fontweight='bold')
ax.text(3, FULL_Y + 0.35, f'Full Query  ({FULL_Y:.0f}%)',
        ha='center', va='bottom', fontsize=10.5, color='#222222', fontweight='bold')

# --- Persona dashed slope lines (Direct Execution → TactfulLLM) ---
for persona, values in DATA.items():
    color = PERSONA_COLOR[persona]
    marker = PERSONA_MARKER[persona]
    ax.plot([0, 1], values, '--', color=color, linewidth=2.2, alpha=0.85,
            zorder=4, label=persona)
    ax.scatter([0, 1], values, c=color, marker=marker, s=150,
               edgecolors='black', linewidth=0.9, zorder=5)

# --- Value labels: Direct Execution column (left of markers) ---
for persona, values in DATA.items():
    color = PERSONA_COLOR[persona]
    ax.annotate(f'{values[0]:.1f}', (0, values[0]),
                xytext=(-0.10, values[0]),
                ha='right', va='center', fontsize=10, color=color, fontweight='bold')

# --- Value labels: TactfulLLM column (left of markers, leaving room for persona names) ---
for persona, values in DATA.items():
    color = PERSONA_COLOR[persona]
    ax.annotate(f'{values[1]:.1f}', (1, values[1]),
                xytext=(0.88, values[1]),
                ha='right', va='center', fontsize=10, color=color, fontweight='bold')

# --- Persona name annotations at TactfulLLM column (right of markers) ---
for persona, values in DATA.items():
    color = PERSONA_COLOR[persona]
    ax.annotate(persona, (1, values[1]), xytext=(1.18, values[1]),
                ha='left', va='center', fontsize=11.5, color=color, fontweight='bold')

# --- Busy "by design" framing: small annotation BELOW the Busy line ---
ax.annotate(
    'Busy gets a small recovery\nby design (low patience)',
    xy=(1.05, 14.0), xytext=(1.55, 9.5),
    fontsize=9, color='#666666', style='italic', ha='left',
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', alpha=0.8, linewidth=1.0),
    zorder=3,
)

# --- Axes formatting ---
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS, fontsize=11)
ax.set_xlabel('Clarification Recovery Path', fontsize=13, fontweight='bold')
ax.set_ylabel('Task Success (pass@1, %)', fontsize=13)
ax.set_ylim(8, 23)
ax.set_xlim(-0.4, 3.5)
ax.grid(True, axis='y', alpha=0.25, linestyle='--')
ax.set_axisbelow(True)

# --- Persona legend only (reference lines self-labeled in-plot) ---
ax.legend(loc='upper left', fontsize=10, frameon=True, ncol=1, title='Persona')

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path('/root/autodl-tmp/ProactiveLLM/data/analysis')
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / 'fig_recovery_slope.pdf'
png_path = out_dir / 'fig_recovery_slope.png'

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=150, bbox_inches='tight')

print(f'Saved {pdf_path}')
print(f'Saved {png_path}')
