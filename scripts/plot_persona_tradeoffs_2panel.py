"""
Merge per-persona panels of pareto_tradeoff (b) + rejection_tradeoff (b)
into one 2-panel figure for the paper. Shared X axis = Avg Turns.

Output:
  data/analysis/persona_tradeoffs_2panel.pdf  (vector, for paper)
  data/analysis/persona_tradeoffs_2panel.png  (raster, for review)

Data: Llama-3.1-8B canonical-200 (matches Table 1's Llama row).
"""
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (Llama canonical-200; PO is 50-state per current paper footnote)
# ---------------------------------------------------------------------------

pass_data = {
    'Direct Execution': {
        'Nov':  (1.0, 11.5),
        'Exp':  (1.0, 12.5),
        'Busy': (1.0, 13.0),
        'All':  (1.0, 12.3),
    },
    'Clarify-first (K=1)': {
        'Nov':  (2.0, 14.0),
        'Exp':  (2.0, 14.0),
        'Busy': (2.0, 16.5),
        'All':  (2.0, 14.8),
    },
    'Base LLM': {
        'Nov':  (2.29, 12.5),
        'Exp':  (1.65, 12.5),
        'Busy': (2.51, 13.0),
        'All':  (2.15, 12.7),
    },
    'Prompt-only': {
        'Nov':  (5.82, 14.0),
        'Exp':  (5.38, 6.0),
        'Busy': (5.32, 6.0),
        'All':  (5.51, 8.7),
    },
    'TactfulLLM (Ours)': {
        'Nov':  (7.0, 18.5),
        'Exp':  (2.6, 15.5),
        'Busy': (1.0, 14.0),
        'All':  (3.5, 16.0),
    },
}

rej_data = {
    'Clarify-first (K=1)': {
        'Nov':  (2.0, 19.5),
        'Exp':  (2.0, 41.0),
        'Busy': (2.0, 82.5),
        'All':  (2.0, 48.0),
    },
    'Base LLM': {
        'Nov':  (2.29, 33.2),
        'Exp':  (1.65, 59.2),
        'Busy': (2.51, 89.4),
        'All':  (2.15, 62.0),
    },
    'Prompt-only': {
        'Nov':  (5.82, 46.0),
        'Exp':  (5.38, 56.0),
        'Busy': (5.32, 89.0),
        'All':  (5.51, 63.0),
    },
    'TactfulLLM (Ours)': {
        'Nov':  (7.0, 45.3),
        'Exp':  (2.6, 44.8),
        'Busy': (1.0, 0.0),
        'All':  (3.5, 45.0),
    },
}

COLOR_SCHEMES = {
    'red': {
        'Direct Execution': '#999999',
        'Clarify-first (K=1)': '#66b3ff',
        'Base LLM': '#ffcc66',
        'Prompt-only': '#ff9999',
        'TactfulLLM (Ours)': '#e74c3c',
        '_ours_text': '#c0392b',  # darker red for TactfulLLM persona text labels
    },
    'blue': {
        'Direct Execution': '#999999',         # gray
        'Clarify-first (K=1)': '#2ca02c',      # green
        'Base LLM': '#ffcc66',
        'Prompt-only': '#ff9999',
        'TactfulLLM (Ours)': '#66b3ff',        # light blue (matches red-scheme CF)
        '_ours_text': '#1f4d75',               # darker shade for persona text labels
    },
}
MARKER = {
    'Direct Execution': 'o',
    'Clarify-first (K=1)': 's',
    'Base LLM': 'D',
    'Prompt-only': '^',
    'TactfulLLM (Ours)': '*',
}

PERSONA_ORDER = ['Busy', 'Exp', 'Nov']  # left-to-right on X axis (low-to-high turns)


def _annotate(ax, text, xy, spec, **kwargs):
    """spec = (ox, oy[, ha, va]). Defaults ha='left', va='baseline'."""
    x, y = xy
    ox, oy, *align = spec
    ha = align[0] if len(align) >= 1 else 'left'
    va = align[1] if len(align) >= 2 else 'baseline'
    ax.annotate(text, (x, y), xytext=(x + ox, y + oy), ha=ha, va=va, **kwargs)


# Per-(panel, kind, persona/method) hand-tuned offsets so labels never collide.
# spec: (ox, oy) or (ox, oy, ha, va).
LABEL_OFFSETS = {
    # ---- Panel (a) pass@1 ----
    ('pass', 'ours', 'Busy'): (0.00,  0.70, 'center', 'bottom'),
    ('pass', 'ours', 'Exp'):  (0.00,  0.70, 'center', 'bottom'),
    ('pass', 'ours', 'Nov'):  (-0.25, 0.00, 'right', 'center'),
    ('pass', 'po', 'Busy'):   (-0.25, -0.20, 'right', 'top'),
    ('pass', 'po', 'Exp'):    (-0.25,  0.40, 'right', 'bottom'),
    ('pass', 'po', 'Nov'):    (-0.25, -0.20, 'right', 'top'),
    ('pass', 'baseline', 'Direct Execution'):    (-0.20, 0.00, 'right', 'center'),
    ('pass', 'baseline', 'Clarify-first (K=1)'): (-0.20, 0.00, 'right', 'center'),
    ('pass', 'baseline', 'Base LLM'):            (0.20,  0.00, 'left', 'center'),
    # ---- Panel (b) rejection ----
    ('rej', 'ours', 'Busy'):  (0.25,  0.00, 'left', 'center'),
    ('rej', 'ours', 'Exp'):   (0.25, -3.00, 'left', 'top'),
    ('rej', 'ours', 'Nov'):   (0.30,  3.00, 'right', 'bottom'),
    ('rej', 'po', 'Busy'):    (-0.25, -1.00, 'right', 'top'),
    ('rej', 'po', 'Exp'):     (-0.25,  1.50, 'right', 'bottom'),
    ('rej', 'po', 'Nov'):     (-0.40, -7.00, 'center', 'top'),
    ('rej', 'baseline', 'Clarify-first (K=1)'):  (-0.20, 0.00, 'right', 'center'),
    ('rej', 'baseline', 'Base LLM'):             (0.20,  0.00, 'left', 'center'),
}


PERSONA_FULL = {'Busy': 'Busy', 'Exp': 'Exp', 'Nov': 'Novice'}


def plot_persona_panel(ax, data, panel_key, ylabel, ylim, baselines_show, panel_label, COLOR):
    """One panel: per-persona points for adaptive methods (TactfulLLM, Prompt-only);
    single 'All' aggregate point for pooled baselines (Direct, Clarify-first, Base LLM).
    """
    ours_color = COLOR['TactfulLLM (Ours)']
    ours_text_color = COLOR['_ours_text']
    po_color = COLOR['Prompt-only']

    # Pooled-only baselines (single "All" point per method).
    # Identification via shared bottom legend — no in-plot text label, avoids overlap.
    for method in baselines_show:
        if method not in data:
            continue
        t, y = data[method]['All']
        ax.scatter(t, y, c=COLOR[method], marker=MARKER[method], s=180,
                   edgecolors='black', linewidth=1.0, zorder=4, label=method)

    # Prompt-only frontier
    po = data['Prompt-only']
    po_pts = [po[p] for p in PERSONA_ORDER]
    ax.plot([p[0] for p in po_pts], [p[1] for p in po_pts],
            '--', color=po_color, linewidth=2.0, alpha=0.6, zorder=3)
    ax.scatter([p[0] for p in po_pts], [p[1] for p in po_pts],
               c=po_color, marker='^', s=220,
               edgecolors='black', linewidth=1.0, zorder=5, label='Prompt-only')
    for p in PERSONA_ORDER:
        _annotate(ax, f'Prompt-only ({PERSONA_FULL[p]})', po[p],
                  LABEL_OFFSETS[(panel_key, 'po', p)],
                  fontsize=11, color='#cc6666')

    # Ours frontier — drawn last so it sits on top and appears last in legend
    ours = data['TactfulLLM (Ours)']
    ours_pts = [ours[p] for p in PERSONA_ORDER]
    ax.plot([p[0] for p in ours_pts], [p[1] for p in ours_pts],
            '--', color=ours_color, linewidth=2.5, alpha=0.6, zorder=3)
    ax.scatter([p[0] for p in ours_pts], [p[1] for p in ours_pts],
               c=ours_color, marker='*', s=380,
               edgecolors='black', linewidth=1.0, zorder=5, label='TactfulLLM (Ours)')
    for p in PERSONA_ORDER:
        _annotate(ax, PERSONA_FULL[p], ours[p], LABEL_OFFSETS[(panel_key, 'ours', p)],
                  fontsize=14, fontweight='bold', color=ours_text_color)

    ax.set_xlabel('Avg Turns (Cost)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(0, 8)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    # Panel sub-label inside top-left of axes (paper convention)
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')


# ---------------------------------------------------------------------------
# Render — emit one figure per color scheme.
# ---------------------------------------------------------------------------
out_dir = Path('/root/autodl-tmp/ProactiveLLM/data/analysis')
out_dir.mkdir(parents=True, exist_ok=True)

# Each scheme name → output filename suffix.
# 'red' → persona_tradeoffs_2panel.pdf (canonical)
# 'blue' → persona_tradeoffs_2panel_blue.pdf (alt)
SCHEME_SUFFIX = {'red': '', 'blue': '_blue'}

for scheme_name, scheme_color in COLOR_SCHEMES.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    plot_persona_panel(
        axes[0], pass_data, panel_key='pass',
        ylabel='Task Success (pass@1, %)',
        ylim=(4, 21),
        baselines_show=['Direct Execution', 'Clarify-first (K=1)', 'Base LLM'],
        panel_label='(a)',
        COLOR=scheme_color,
    )
    plot_persona_panel(
        axes[1], rej_data, panel_key='rej',
        ylabel='User Rejection Rate (%)',
        ylim=(-8, 100),
        baselines_show=['Clarify-first (K=1)', 'Base LLM'],
        panel_label='(b)',
        COLOR=scheme_color,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    suffix = SCHEME_SUFFIX[scheme_name]
    pdf_path = out_dir / f'persona_tradeoffs_2panel{suffix}.pdf'
    png_path = out_dir / f'persona_tradeoffs_2panel{suffix}.png'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')
    print(f'Saved {png_path}')
