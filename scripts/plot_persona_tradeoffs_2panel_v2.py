"""
Per-persona Pareto trade-off plot (2 panels: pass@1 and Rejection Rate vs Avg Turns).
Shows TactfulLLM, Prompt-only, Few-shot Persona, CollabLLM as per-persona frontiers
(reveals which methods achieve persona-conditional behavior); Direct/Clarify-first/Base
LLM as single aggregate "All" points.

Output:
  data/analysis/persona_tradeoffs_2panel.pdf  (red scheme, canonical)
  data/analysis/persona_tradeoffs_2panel_blue.pdf  (blue scheme, alt)
  + .png raster versions

Data: Llama-3.1-8B canonical N=200 (matches Table 1 Llama row).
"""
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (Llama canonical-200; all rows match Table 1)
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
        'Nov':  (2.30, 12.5),
        'Exp':  (1.65, 12.5),
        'Busy': (2.51, 13.0),
        'All':  (2.15, 12.7),
    },
    'Prompt-only': {
        'Nov':  (5.99, 15.0),
        'Exp':  (5.50, 13.0),
        'Busy': (5.71, 12.0),
        'All':  (5.73, 13.3),
    },
    'Few-shot Persona': {
        'Nov':  (5.79, 15.0),
        'Exp':  (4.60, 14.5),
        'Busy': (1.00, 11.5),
        'All':  (3.80, 13.7),
    },
    'CollabLLM': {
        'Nov':  (3.73, 17.0),
        'Exp':  (4.50, 14.0),
        'Busy': (4.55, 14.0),
        'All':  (4.26, 15.0),
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
        'Nov':  (2.30, 28.6),
        'Exp':  (1.65, 51.6),
        'Busy': (2.51, 91.0),
        'All':  (2.15, 62.0),
    },
    'Prompt-only': {
        'Nov':  (5.99, 45.0),
        'Exp':  (5.50, 54.0),
        'Busy': (5.71, 88.0),
        'All':  (5.73, 62.0),
    },
    'Few-shot Persona': {
        'Nov':  (5.79, 42.0),
        'Exp':  (4.60, 55.0),
        # Busy never clarifies — omit from rejection panel (undefined)
        'All':  (3.80, 47.0),
    },
    'CollabLLM': {
        'Nov':  (3.73, 30.0),
        'Exp':  (4.50, 55.0),
        'Busy': (4.55, 84.0),
        'All':  (4.26, 59.0),
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
        'Direct Execution':     '#999999',
        'Clarify-first (K=1)':  '#66b3ff',
        'Base LLM':             '#ffcc66',
        'Prompt-only':          '#ff9999',
        'Few-shot Persona':     '#bb88dd',     # purple
        'CollabLLM':            '#7cb87c',     # sage green
        'TactfulLLM (Ours)':    '#e74c3c',
        '_ours_text':           '#c0392b',
        '_po_text':             '#cc6666',
        '_fs_text':             '#7a4ea0',
        '_collab_text':         '#3a7a3a',
    },
    'blue': {
        'Direct Execution':     '#999999',
        'Clarify-first (K=1)':  '#2ca02c',
        'Base LLM':             '#ffcc66',
        'Prompt-only':          '#ff9999',
        'Few-shot Persona':     '#bb88dd',
        'CollabLLM':            '#ff7f50',     # coral orange (distinct from CF green)
        'TactfulLLM (Ours)':    '#66b3ff',
        '_ours_text':           '#1f4d75',
        '_po_text':             '#cc6666',
        '_fs_text':             '#7a4ea0',
        '_collab_text':         '#cc5530',
    },
}

MARKER = {
    'Direct Execution':     'o',
    'Clarify-first (K=1)':  's',
    'Base LLM':             'D',
    'Prompt-only':          '^',
    'Few-shot Persona':     'P',     # filled plus
    'CollabLLM':            'X',     # filled cross
    'TactfulLLM (Ours)':    '*',
}

PERSONA_ORDER = ['Busy', 'Exp', 'Nov']
PERSONA_FULL = {'Busy': 'Busy', 'Exp': 'Exp', 'Nov': 'Novice'}


def _annotate(ax, text, xy, spec, **kwargs):
    x, y = xy
    ox, oy, *align = spec
    ha = align[0] if len(align) >= 1 else 'left'
    va = align[1] if len(align) >= 2 else 'baseline'
    ax.annotate(text, (x, y), xytext=(x + ox, y + oy), ha=ha, va=va, **kwargs)


# Hand-tuned label offsets to avoid collisions.
# Coordinates: data units; (dx, dy, ha, va) — dx/dy applied to (x_marker, y_marker).
LABEL_OFFSETS = {
    # ─────────────── Panel (a) pass@1 ───────────────
    # TactfulLLM (bold blue) — landmark, sits ABOVE markers
    ('pass', 'ours', 'Busy'):     (0.00,  0.75, 'center', 'bottom'),
    ('pass', 'ours', 'Exp'):      (0.00,  0.65, 'center', 'bottom'),
    ('pass', 'ours', 'Nov'):      (-0.20, 0.00, 'right', 'center'),

    # Prompt-only (pink, x≈5.5–6.0 cluster, y range 12–15)
    ('pass', 'po', 'Busy'):       (0.25,  -0.55, 'left',  'top'),     # below-right (5.96, 11.45)
    ('pass', 'po', 'Exp'):        (-0.25, -0.55, 'right', 'top'),     # below-left  (5.25, 12.45)
    ('pass', 'po', 'Nov'):        (0.30,   0.00, 'left',  'center'),  # right       (6.29, 15.00)

    # Few-shot Persona (purple) — Busy at left edge, Exp/Nov in middle cluster
    ('pass', 'fs', 'Busy'):       (0.00,  -0.65, 'center', 'top'),    # below marker (1.00, 10.85)
    ('pass', 'fs', 'Exp'):        (-0.30,  0.55, 'right', 'bottom'),  # upper-left  (4.30, 15.05)
    ('pass', 'fs', 'Nov'):        (-0.30,  0.30, 'right', 'bottom'),  # upper-left  (5.49, 15.30)

    # CollabLLM (orange) — Exp/Busy nearly co-located at (4.5, 14)
    ('pass', 'collab', 'Busy'):   (0.30,   0.50, 'left',  'bottom'),  # upper-right (4.85, 14.50)
    ('pass', 'collab', 'Exp'):    (-0.30, -0.50, 'right', 'top'),     # lower-left  (4.20, 13.50)
    ('pass', 'collab', 'Nov'):    (0.30,   0.00, 'left',  'center'),  # right       (4.03, 17.00)

    # ─────────────── Panel (b) rejection ───────────────
    # TactfulLLM
    ('rej', 'ours', 'Busy'):      (0.25,   0.00, 'left',  'center'),
    ('rej', 'ours', 'Exp'):       (0.25,  -3.50, 'left',  'top'),
    ('rej', 'ours', 'Nov'):       (0.00,   5.00, 'center', 'bottom'),

    # Prompt-only
    ('rej', 'po', 'Busy'):        (0.25,   2.50, 'left',  'bottom'),  # upper-right (5.96, 90.5)
    ('rej', 'po', 'Exp'):         (0.30,  -3.00, 'left',  'top'),     # lower-right (5.80, 51.0)
    ('rej', 'po', 'Nov'):         (-0.30, -3.50, 'right', 'top'),     # lower-left  (5.69, 41.5)

    # Few-shot Persona
    ('rej', 'fs', 'Exp'):         (0.30,   3.50, 'left',  'bottom'),  # upper-right (4.90, 58.5)
    ('rej', 'fs', 'Nov'):         (0.30,   3.00, 'left',  'bottom'),  # upper-right (6.09, 45.0)

    # CollabLLM
    ('rej', 'collab', 'Busy'):    (-0.30, -3.50, 'right', 'top'),     # lower-left  (4.25, 80.5)
    ('rej', 'collab', 'Exp'):     (-0.30, -3.50, 'right', 'top'),     # lower-left  (4.20, 51.5)
    ('rej', 'collab', 'Nov'):     (0.30,  -3.00, 'left',  'top'),     # lower-right (4.03, 27.0)
}


def _plot_method_per_persona(ax, data, key, panel_key, color, marker, size,
                             text_color, label_kind, frontier_line=True,
                             show_label_text=True, fontsize=10, fontweight='normal',
                             label_text_fn=None, smooth_curve=False, linewidth=1.8):
    """Plot a method's per-persona points, optional connecting Pareto frontier line.

    When smooth_curve=True and >=3 points, draws a monotone cubic spline (PCHIP)
    instead of straight segments — gentler aesthetic for the highlighted method.
    """
    if key not in data:
        return None
    pts = []
    for p in PERSONA_ORDER:
        if p not in data[key]:
            continue
        pts.append((p, data[key][p]))
    if frontier_line and len(pts) >= 2:
        xs = [pt[1][0] for pt in pts]
        ys = [pt[1][1] for pt in pts]
        # Sort by x for spline; use a smooth curve if requested.
        sx, sy = zip(*sorted(zip(xs, ys)))
        if smooth_curve and len(sx) >= 3:
            try:
                import numpy as np
                from scipy.interpolate import PchipInterpolator
                spline = PchipInterpolator(sx, sy)
                x_fine = np.linspace(sx[0], sx[-1], 80)
                y_fine = spline(x_fine)
                ax.plot(x_fine, y_fine, '--', color=color,
                        linewidth=linewidth, alpha=0.55, zorder=2)
            except ImportError:
                ax.plot(sx, sy, '--', color=color, linewidth=linewidth, alpha=0.5, zorder=2)
        else:
            ax.plot(sx, sy, '--', color=color, linewidth=linewidth, alpha=0.5, zorder=2)
    handle = None
    for p, xy in pts:
        h = ax.scatter(xy[0], xy[1], c=color, marker=marker, s=size,
                       edgecolors='black', linewidth=0.8, zorder=5)
        if handle is None:
            handle = h
        if show_label_text:
            text = label_text_fn(p) if label_text_fn else f'{key} ({PERSONA_FULL[p]})'
            offset_key = (panel_key, label_kind, p)
            if offset_key in LABEL_OFFSETS:
                _annotate(ax, text, xy, LABEL_OFFSETS[offset_key],
                          fontsize=fontsize, color=text_color, fontweight=fontweight)
    return handle


def plot_persona_panel(ax, data, panel_key, ylabel, ylim, baselines_show, panel_label, COLOR):
    """One panel.
    Pooled-only (single 'All' point): Direct, Clarify-first, Base LLM.
    Per-persona frontiers: TactfulLLM, Prompt-only, Few-shot Persona, CollabLLM.
    """
    legend_handles = {}

    # Pooled-only baselines: single 'All' point.
    for method in baselines_show:
        if method not in data:
            continue
        t, y = data[method]['All']
        h = ax.scatter(t, y, c=COLOR[method], marker=MARKER[method], s=180,
                       edgecolors='black', linewidth=1.0, zorder=4)
        legend_handles[method] = h

    # Per-persona methods (in order: ours drawn last → on top).
    # Persona text labels: only for TactfulLLM and Prompt-only (the original story);
    # for CollabLLM and Few-shot Persona, we show only persona symbols/markers
    # (full text would over-crowd; legend entry suffices).
    legend_handles['Prompt-only'] = _plot_method_per_persona(
        ax, data, 'Prompt-only', panel_key,
        color=COLOR['Prompt-only'], marker=MARKER['Prompt-only'], size=200,
        text_color=COLOR['_po_text'], label_kind='po',
        show_label_text=True, fontsize=9,
        label_text_fn=lambda p: PERSONA_FULL[p],
    )
    legend_handles['Few-shot Persona'] = _plot_method_per_persona(
        ax, data, 'Few-shot Persona', panel_key,
        color=COLOR['Few-shot Persona'], marker=MARKER['Few-shot Persona'], size=160,
        text_color=COLOR['_fs_text'], label_kind='fs',
        show_label_text=True, fontsize=9,
        label_text_fn=lambda p: PERSONA_FULL[p],
    )
    legend_handles['CollabLLM'] = _plot_method_per_persona(
        ax, data, 'CollabLLM', panel_key,
        color=COLOR['CollabLLM'], marker=MARKER['CollabLLM'], size=170,
        text_color=COLOR['_collab_text'], label_kind='collab',
        show_label_text=True, fontsize=9,
        label_text_fn=lambda p: PERSONA_FULL[p],
    )
    # TactfulLLM last → on top
    legend_handles['TactfulLLM (Ours)'] = _plot_method_per_persona(
        ax, data, 'TactfulLLM (Ours)', panel_key,
        color=COLOR['TactfulLLM (Ours)'], marker=MARKER['TactfulLLM (Ours)'], size=380,
        text_color=COLOR['_ours_text'], label_kind='ours',
        show_label_text=True, fontsize=14, fontweight='bold',
        label_text_fn=lambda p: PERSONA_FULL[p],
    )

    ax.set_xlabel('Avg Turns (Cost)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(0, 8)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    return legend_handles


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
out_dir = Path('/root/autodl-tmp/ProactiveLLM/data/analysis')
out_dir.mkdir(parents=True, exist_ok=True)

SCHEME_SUFFIX = {'red': '_v2', 'blue': '_v2_blue'}

# Legend order: pooled baselines first, then per-persona methods, ours last.
LEGEND_ORDER = [
    'Direct Execution',
    'Base LLM',
    'Prompt-only',
    'Few-shot Persona',
    'CollabLLM',
    'TactfulLLM (Ours)',
]

for scheme_name, scheme_color in COLOR_SCHEMES.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    h_a = plot_persona_panel(
        axes[0], pass_data, panel_key='pass',
        ylabel='Task Success (pass@1, %)',
        ylim=(8, 21),
        baselines_show=['Direct Execution', 'Base LLM'],
        panel_label='(a)',
        COLOR=scheme_color,
    )
    h_b = plot_persona_panel(
        axes[1], rej_data, panel_key='rej',
        ylabel='User Rejection Rate (%)',
        ylim=(-8, 100),
        baselines_show=['Base LLM'],
        panel_label='(b)',
        COLOR=scheme_color,
    )

    # Combined legend at bottom
    handles, labels = [], []
    for name in LEGEND_ORDER:
        if name in h_a:
            handles.append(h_a[name])
            labels.append(name)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    suffix = SCHEME_SUFFIX[scheme_name]
    pdf_path = out_dir / f'persona_tradeoffs_2panel{suffix}.pdf'
    png_path = out_dir / f'persona_tradeoffs_2panel{suffix}.png'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')
    print(f'Saved {png_path}')
