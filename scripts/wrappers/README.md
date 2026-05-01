# Run Wrappers

Bash launchers for training and evaluation runs. Persisted from `/tmp` so
container restarts don't lose them. Each wrapper sets the offline HF env,
chooses classifier version, and invokes the right Python entrypoint.

Caller is expected to pin the GPU via `CUDA_VISIBLE_DEVICES=N bash <wrapper>`.

## v33 SFT-then-DPO pipeline (canonical, May 2026)

| Wrapper | Stage | Backbone |
|---|---|---|
| `v33_v3_sft.sh` | SFT (prompt masking, alpha=32, 3 ep, LR 5e-5) | Llama-3.1-8B-Instruct |
| `v33_v3_qwen_sft.sh` | SFT (same hparams) | Qwen2.5-7B-Instruct |
| `v33_v3_dpo.sh` | DPO refinement (INIT_ADAPTER=SFT, 3 ep, β=0.1) | Llama |
| `v33_v3_qwen_dpo_v2.sh` | DPO refinement (1 ep — Qwen needed reduced epochs to avoid Busy collapse) | Qwen |

## v33 evaluation

| Wrapper | N | Notes |
|---|---|---|
| `v33_v3_eval_5.sh` | 5 | Llama SFT-only smoke |
| `v33_v3_dpo_eval_5.sh` | 5 | Llama SFT+DPO smoke |
| `v33_v3_qwen_eval_5.sh` | 5 | Qwen SFT-only smoke |
| `v33_v3_qwen_dpo_v2_eval_5.sh` | 5 | Qwen SFT+DPO smoke |
| `v33_v3_qwen_dpo_v2_eval_100_ft.sh` | 100 | First 100 states (canonical) |
| `v33_v3_qwen_dpo_v2_eval_remaining100.sh` | 100 | State 101-200 (extends to N=200) |

## Qwen baselines (remaining-100 = state 101-200)

`baseline_*_remaining100_ft.sh` invoke `evaluate.py` with new template + non-truncated
saving. Run alongside first-100 patched files; merge with `scripts/merge_v33_qwen_200.py`.

| Wrapper | Method |
|---|---|
| `qwen_direct_remaining100_ft.sh` | `--direct_execution`, max_turns=1 |
| `qwen_cf_remaining100_ft.sh` | `--always_clarify 1`, max_turns=2 |
| `qwen_base_remaining100_ft.sh` | `--no_lora`, max_turns=7 (v2 classifier → 1-turn execute in practice) |
| `qwen_po_remaining100_ft.sh` | Prompt-only persona instructions, no DPO |

## Monitors

| Wrapper | Purpose |
|---|---|
| `qwen_baselines_progress.sh` | One-shot snapshot: PIDs, GPU util, sample counts |
| `freeze_monitor.sh` | Long-running CSV recorder (20 min cadence) — reveals autodl freeze gaps via timestamp deltas |

See `docs/sft_then_dpo_v33.md` for pipeline rationale and
`docs/work_log.md` for run-by-run history.
