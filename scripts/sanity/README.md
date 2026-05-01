# Sanity verification scripts

Probe scripts that run a model on the 8-state × 3-persona test grid (24 outputs)
to verify expected behavior before launching expensive 100/200-state evals.

Persisted from `/tmp` so reproducibility is preserved.

## `classifier/` — v1 vs v2 prefix-classifier sanity

Verified the 2026-04-28 finding that v1 (30-token prefix scan) misclassifies 67%
of Qwen DPO outputs. Both Llama and Qwen sampling outputs included.

See `docs/classifier_bug_2026-04-28.md` for findings.

## `v33_v3/` — v33 v3 SFT/DPO behavior sanity

24-output sanity grid: 8 states × 3 personas (Novice / Busy / Experienced).

| Script | Result | Interpretation |
|---|---|---|
| `v33_v3_sft_sanity.py` | Llama SFT 24/24 perfect | Novice 8/8 Clarify, Busy 8/8 Execute, Exp 5/8 Clarify |
| `v33_v3_dpo_sanity.py` | Llama SFT+DPO 24/24 | Same as SFT but Exp 5/8 → 8/8 Clarify (DPO refinement) |
| `v33_v3_qwen_sft_sanity.py` | Qwen SFT 24/24 perfect | All three personas 8/8 expected behavior |
| `v33_v3_qwen_dpo_v2_sanity.py` | Qwen SFT+DPO v2 partial | Novice/Exp 8/8, Busy 4/8 valid (epochs=1 was needed to avoid total collapse) |

`.json` files are the structured result; `.log` files (gitignored) capture stdout.
