"""
Analyze whether information recovery (disclosure_rate) predicts task success (pass@1).

Inputs:  data/analysis/disclosure_per_conversation.csv
Outputs: data/analysis/recovery_bins.csv
         data/analysis/recovery_scatter.png
         stdout: Spearman ρ, logistic regression coef + 95% CI
"""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "analysis" / "disclosure_per_conversation.csv"
BINS_OUT = ROOT / "data" / "analysis" / "recovery_bins.csv"
PLOT_OUT = ROOT / "data" / "analysis" / "recovery_scatter.png"

PERSONAS = ["Novice-Learner", "Experienced-Engineer", "Busy-Developer"]
PERSONA_COLOR = {"Novice-Learner": "#1f77b4", "Experienced-Engineer": "#2ca02c", "Busy-Developer": "#d62728"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0, center - half), min(1, center + half))


def main():
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} conversations\n")

    # --- 1. Aggregate counts -------------------------------------------------
    print("=== Disclosure summary by persona ===")
    print(df.groupby("persona")[["disclosure_rate", "pass1"]].mean().round(3))

    # --- 2. Spearman correlation --------------------------------------------
    print("\n=== Spearman ρ: disclosure_rate vs pass@1 ===")
    rho, p = stats.spearmanr(df["disclosure_rate"], df["pass1"])
    print(f"All conversations:    ρ={rho:+.3f}  p={p:.4g}  n={len(df)}")
    for persona in PERSONAS:
        sub = df[df.persona == persona]
        if len(sub) > 5 and sub["disclosure_rate"].nunique() > 1:
            r, pp = stats.spearmanr(sub["disclosure_rate"], sub["pass1"])
            print(f"{persona:<22s} ρ={r:+.3f}  p={pp:.4g}  n={len(sub)}")
        else:
            print(f"{persona:<22s} skipped (degenerate)")

    # --- 3. Logistic regression with persona control ------------------------
    try:
        import statsmodels.formula.api as smf
        print("\n=== Logistic regression: pass1 ~ disclosure_rate + C(persona) ===")
        df_lr = df.copy()
        m = smf.logit("pass1 ~ disclosure_rate + C(persona)", data=df_lr).fit(disp=False)
        print(m.summary().tables[1])
        coef = m.params["disclosure_rate"]
        ci = m.conf_int().loc["disclosure_rate"].values
        print(f"\ndisclosure_rate coef = {coef:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={m.pvalues['disclosure_rate']:.4g}")
    except ImportError:
        print("statsmodels not installed; skip logistic regression")

    # --- 4. Binned bar chart -------------------------------------------------
    bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0001]
    bin_labels = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1.0]"]
    df["bin"] = pd.cut(df.disclosure_rate, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)

    print("\n=== Binned pass@1 ===")
    bin_rows = []
    for label in bin_labels:
        sub = df[df.bin == label]
        n = len(sub)
        k = int(sub.pass1.sum())
        mean = k / n if n else 0
        lo, hi = wilson_ci(k, n)
        bin_rows.append({"bin": label, "n": n, "passed": k, "pass1": mean, "ci_lo": lo, "ci_hi": hi})
        print(f"  {label:<14s} n={n:3d} pass@1={mean:.3f}  CI=[{lo:.3f}, {hi:.3f}]")

    BINS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bin_rows).to_csv(BINS_OUT, index=False)
    print(f"Saved bin stats → {BINS_OUT}")

    # --- 5. Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(bin_labels))
    means = [r["pass1"] for r in bin_rows]
    los = [r["pass1"] - r["ci_lo"] for r in bin_rows]
    his = [r["ci_hi"] - r["pass1"] for r in bin_rows]
    ax.bar(xs, means, yerr=[los, his], capsize=5, color="#4c72b0", alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, r in enumerate(bin_rows):
        ax.text(i, r["pass1"] + (his[i] + 0.01), f"n={r['n']}", ha="center", fontsize=9, color="dimgray")
    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Disclosure rate (information recovered through clarification)")
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, max(means) * 1.6 + 0.05)
    ax.set_title("Information recovery vs. task success (TactfulLLM, n=600)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    print(f"Saved plot   → {PLOT_OUT}")


if __name__ == "__main__":
    main()
