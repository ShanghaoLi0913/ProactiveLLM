"""
4-condition grouped bar chart for Experiment 2 (Recovery).
Conditions: Masked Direct | TactfulLLM | Ideal Disclosed | Full Query
Groups:     Novice-Learner / Experienced-Engineer / Busy-Developer

TBD bars (Ideal Disclosed not finished, Full Query persona-agnostic) are shown
hatched with the partial / pooled value as placeholder.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "analysis" / "fig_recovery_4condition.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

personas = ["Novice", "Experienced", "Busy"]

# pass@1 (%) — real numbers from outputs/
data = {
    "Masked Direct":    [11.5, 12.5, 13.0],
    "TactfulLLM":       [18.5, 15.5, 14.0],
    "Ideal Disclosed":  [17.1, 17.1, 17.1],   # partial overall; no persona split yet
    "Full Query":       [20.0, 20.0, 20.0],   # persona-agnostic
}
tbd = {"Ideal Disclosed": True, "Full Query": True}
colors = {
    "Masked Direct":   "#999999",
    "TactfulLLM":      "#1f77b4",
    "Ideal Disclosed": "#2ca02c",
    "Full Query":      "#d62728",
}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(personas))
width = 0.2

for i, (cond, vals) in enumerate(data.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(
        x + offset, vals, width,
        color=colors[cond], edgecolor="black", linewidth=0.6,
        hatch="//" if tbd.get(cond) else None,
        alpha=0.85, label=cond + (" (TBD)" if tbd.get(cond) else ""),
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=8, color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels(personas)
ax.set_ylabel("pass@1 (%)")
ax.set_title("Information availability vs task success (TactfulLLM, n=200/persona)")
ax.set_ylim(0, 25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved → {OUT}")
