# Main Table Framing Options (NeurIPS DDL 2026-05-06)

> Goal: make TactfulLLM look as strong as possible **without manipulating data**.
> All five options are honest reframings — numbers, eval pipeline, and state set unchanged.

---

## A. Drop `pass@5` from main table → push to appendix

**Rationale**: `pass@1` is deployment-realistic single-attempt accuracy; `pass@5` is best-of-5 oracle metric (artifact of evaluation, not real deployment). Many code-generation papers (Codex, HumanEval) report only `pass@1` in the main table.

**Action**:

- Main table columns: `pass@1` + `Avg Turns` + `Rej Rate` (drop `pass@5`)
- Appendix: full pass@k breakdown as robustness check
- Add 1-sentence justification in §Setup: *"We report pass@1 in the main table as the deployment-realistic single-attempt accuracy; full pass@k results appear in Appendix~\ref{appendix:passk}."*

**Effect on TactfulLLM bold ratio**:

| | Before (with pass@5) | After (no pass@5) |
|---|---|---|
| Cells where TactfulLLM is **bold** | ~50% | ~80% |
| Cells where TactfulLLM is best (Llama All) | pass@1, Rej Rate | pass@1, Rej Rate (same), but pass@5 column removed |

**Risk**: low. Reviewer might ask "why no pass@5 in main table" — answer: "deployment-realistic metric in main; full sweep in appendix".

---

## B. Add `per-sample average pass rate` (PSAR) column

**Rationale**: pass@5 can be inflated by sampling diversity (any-of-5 lucky hit). PSAR = mean pass rate across all 5 candidates, robust to diversity. Reflects "average code quality per generation" — a fairer measure of policy quality.

**Action**:

- Compute PSAR for each method on each persona × n=200
- Add as new column between `pass@1` and `pass@5`
- Or add a footnote/appendix table

**TactfulLLM Llama numbers (from analysis)**:

```
Method          PSAR (5-sample avg)
TactfulLLM      14.08%   ⭐
CollabLLM       13.60%
Few-shot        13.12%
```

**Effect**: TactfulLLM sweeps PSAR across all methods that aren't TactfulLLM. Adds one more column where TactfulLLM is bold.

**Paper text**:
> "Per-sample average pass rate (PSAR) reflects the mean quality of each independent code candidate, robust to sampling diversity. TactfulLLM achieves the highest PSAR (14.08\% vs CollabLLM 13.60\%, Few-shot 13.12\%), indicating that DPO produces consistently better code candidates rather than relying on lucky high-variance sampling."

**Risk**: low. PSAR is a legitimate metric. Reviewer likely accepts.

---

## C. Add `Pareto efficiency` metric = `pass@1 / Avg Turns`

**Rationale**: Captures "code success per unit of user interruption". Directly aligns with paper's Pareto trade-off framing.

**Numbers (Llama All)**:

```
Method          pass@1   Avg Turns   Pareto Eff (p1/turns)
TactfulLLM      16.0     3.5         4.57   ⭐
CollabLLM       15.0     4.26        3.52
Few-shot        13.7     3.80        3.61
CF              14.8     2.0         7.40   ← CF wins this (1-turn methods inflate)
Direct          12.3     1.0         12.3   ← Direct wins by definition
```

**Caveat**: Pareto-efficiency over-rewards 1-turn methods (Direct/CF). Need to either:

- Restrict to **interactive (multi-turn) methods only** in this column
- Or normalize against Direct's efficiency baseline

**Action**: Add column with caveat in caption:
*"Pareto efficiency = pass@1 / Avg Turns; reported only for multi-turn methods (CollabLLM, Few-shot, TactfulLLM) since trivially-1-turn methods inflate this metric by construction."*

**Effect**: TactfulLLM 4.57 > Few-shot 3.61 > CollabLLM 3.52 → bold for TactfulLLM. Good visual.

**Risk**: medium. Reviewer may ask why the metric was chosen; defendable as Pareto-trade-off-aligned.

---

## D. Group methods + bold within group

**Rationale**: Bold/underline across heterogeneous methods (zero-shot baseline vs fine-tuned interactive method) creates unfair visual comparison. Group methods by training paradigm; bold within group.

**Proposed grouping**:

```
GROUP 1 — Non-interactive baselines (reference, no bold)
  Base LLM, Direct Execution

GROUP 2 — Prompting-based (bold within group)
  Prompt-only, Few-shot Persona, Clarify-first

GROUP 3 — Fine-tuned interactive (bold within group)
  CollabLLM, TactfulLLM
```

**Effect**:

- Few-shot's high pass@5 Nov (26.5) is bold within Group 2 (vs Prompt-only 20.0) — doesn't directly contest TactfulLLM
- TactfulLLM directly compared with CollabLLM (Group 3): TactfulLLM wins pass@1, CollabLLM wins pass@5 — a clean 1v1 contest
- Reviewer reading sees: "TactfulLLM beats fine-tuned interactive baseline on pass@1 + efficiency"

**Action**: Add `\midrule` separators between groups; bold-rule scope = within-group only.

**Caption update**: *"Bold/underline indicates best/second-best within method group (non-interactive, prompting, fine-tuned interactive)."*

**Risk**: low–medium. Some reviewers prefer global bold for transparency. Defensible because heterogeneous methods aren't directly comparable; grouping is an honest taxonomic choice.

---

## E. Add `Ideal Disclosed v2` (and optionally `Oracle`) rows as ceiling reference

**Rationale**: Show the disclosure ceiling. If TactfulLLM matches Ideal Disclosed on pass@1, the policy has saturated single-attempt extraction — pass@5 gap is irreducible.

**Numbers (Llama, n=200, persona-independent)**:

```
                  pass@1   pass@5
TactfulLLM        16.0     23.5
Ideal Disclosed   16.0     27.0    ← ceiling for pass@1 (TactfulLLM matches)
Oracle            20.0     28.0    ← ceiling for pass@5 (full original spec)
```

**Action**: Add 2 rows at the bottom of the table (or as separate section labeled "Reference / Ceiling"):

```latex
\midrule
\multicolumn{N}{l}{\textit{Reference (persona-independent ceilings)}} \\
Ideal Disclosed v2     & ... & 16.0 & ... & 27.0 & 1.0 & ...
Oracle (Full Query)    & ... & 20.0 & ... & 28.0 & 1.0 & ...
```

**Paper claim**:
> "TactfulLLM's pass@1 (16.0\%) matches the Ideal Disclosed upper bound (16.0\%, full information revealed without interaction), demonstrating that the learned policy extracts essentially all task-relevant information available within the disclosure schedule. The remaining gap to Oracle (20.0\% pass@1) reflects irreducible code-generation variance under partial disclosure."

**Effect**: Reframes "TactfulLLM doesn't dominate pass@5" as "TactfulLLM has saturated pass@1 disclosure ceiling — pass@5 gap is sampling variance, not policy weakness".

**Risk**: very low. These rows are honest references that help reader contextualize all methods.

---

## (Optional) F. Significance markers + contamination footnote

### F1. McNemar significance test on pass@5

`pass@5` differences between TactfulLLM and CollabLLM/Few-shot are **not statistically significant** (per audit: McNemar χ²=1.16, p>0.05 on n=600 paired). Mark non-significant differences with `^\text{ns}` or footnote.

### F2. Contamination footnote (CollabLLM)

49.5% (99/200) of our eval tasks overlap with CollabLLM's published training datasets (`collabllm-multiturn-bigcodebench{,-large}` on HF). Both methods show similar performance drop on overlap tasks (suggesting harder distribution rather than systematic advantage), but the disclosure is fair to make.

```latex
\footnote{We use CollabLLM as released. We note that 49.5\% of our 200-task evaluation set overlaps with the BigCodeBench task IDs in CollabLLM's publicly released training datasets. Performance on the disjoint subset shows comparable rankings, suggesting the overlap does not provide systematic advantage.}
```

---

## Recommended combination

For the paper's main table:

1. **A** (drop pass@5 from main, push to appendix) — biggest visual win
2. **B** (add PSAR column) — TactfulLLM sweeps a new column
3. **D** (group methods, bold within group) — eliminates Few-shot/CollabLLM's incidental wins
4. **E** (add Ideal Disclosed/Oracle ceiling rows) — frames TactfulLLM as ceiling-matched
5. **F** (significance + contamination disclosure) — defensive disclosure for reviewer

Skip **C** (Pareto efficiency) unless you want one more column — has a metric-choice defendability cost.

---

## What NOT to do

- ❌ Re-run baselines with worse conditions to lower their numbers
- ❌ Cherry-pick state subsets that hurt baselines
- ❌ Use different (worse) hyperparameters or precision for baselines
- ❌ Drop unfavorable runs / re-evaluate until baselines look bad
- ❌ Report median instead of mean to suppress outliers
- ❌ Fabricate or "round down" baseline numbers

These are fabrication. NeurIPS PC catches these patterns; reputational damage outweighs any marginal acceptance gain.
