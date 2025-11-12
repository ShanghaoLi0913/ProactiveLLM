ProactiveLLM - Context-Aware Proactivity Calibration

Setup

- Create env (example):
  - conda create -n proact python=3.10 -y
  - conda activate proact
  - pip install -r requirements.txt
- Set up OpenAI API key (required for LLM generation):
  - See llm/SETUP_API_KEY.md for detailed instructions
  - Quick setup: `export OPENAI_API_KEY='sk-...'`
  - Test: `python llm/test_openai_key.py`

Structure

- prompts/: behavior templates for coding and planning (low/mid/high)
- llm/: OpenAI API provider and API key setup (see llm/SETUP_API_KEY.md)
- simulator/: user personas and reaction rules
- reward/: task success scoring and interrupt cost
- policy/: DPO training and action selection
- scripts/: data generation utilities
- eval/: evaluation and plots
- data/: datasets (input JSONL/CSV) and generated trajectories/preferences
  - seeds/: curated seed tasks (human-written or selected)
  - external/: raw downloaded datasets (never modified in place)
  - logs/: generated multi-turn logs (LLM + simulator)
  - dpo/: preference tuples after reward calc (s, a+, a-)
  - eval/: final test runs, tables, figures

Quickstart

- Option A (synthetic MVP):
  - Generate small dataset
    - python scripts/generate_data.py --mode synthetic --domain coding --n_states 50 --out dpo/prefs_coding.jsonl
    - python scripts/generate_data.py --mode synthetic --domain planning --n_states 50 --out dpo/prefs_planning.jsonl
- Option B (use your dataset):
  - Prepare a JSONL with fields per row: {"query": str, "dialogue_turn": int, "query_clarity": float, "task_complexity": float, "prev_reject": int}
  - Run
    - python scripts/generate_data.py --mode dataset --domain coding --dataset_path /abs/path/to/your.jsonl --n_states 200 --out dpo/prefs_from_ds.jsonl
  - To use OpenAI for assistant generation (e.g., GPT-4o-mini):
    - export OPENAI_API_KEY=sk-...
    - add flag: --llm_model gpt-4o-mini
    - example: python scripts/generate_data.py --mode dataset --domain coding --dataset_path /abs/path/to/your.jsonl --n_states 200 --out dpo/prefs_from_ds.jsonl --llm_model gpt-4o-mini

MBPP quickstart (phase-1)

- Convert MBPP → states JSONL (derived, so saved under seeds/):
  - python scripts/convert_mbpp_to_states.py  # writes data/seeds/mbpp_states.jsonl
- Generate preference pairs from MBPP states:
  - export OPENAI_API_KEY=sk-...
  - python scripts/generate_data.py --mode dataset --domain coding --dataset_path data/seeds/mbpp_states.jsonl --n_states 100 --out dpo/prefs_mbpp_phase1.jsonl --llm_model gpt-4o-mini
- Train policy with policy/train_dpo.py
- Evaluate with eval/evaluate.py and plot with eval/plot_pareto.py

Data generation overview

- Purpose of modes
  - synthetic: quick sanity check to validate the full pipeline (state → 3 behaviors → simulator feedback → reward → preference pairs). No external dataset required.
  - dataset: use your coding/planning dataset (JSONL) as states to obtain realistic distributions and train usable policies.

- Key principle
  - Never modify `external/` datasets in place. Pipeline transforms: `seeds/ → logs/ → dpo/`.

- Flow (both modes)
  1) State source
     - synthetic: scripts/generate_data.py → synth_states(...)
     - dataset: pass --dataset_path (JSONL with fields {query, dialogue_turn, query_clarity, task_complexity, prev_reject})
  2) Behavior candidates (LOW/MID/HIGH)
     - prompts/: coding_*.txt, planning_*.txt
     - generation: dummy_llm_output(...) or llm/provider.py → chat_complete(..., model=gpt-4o-mini)
       - enable real LLM via --llm_model gpt-4o-mini
  3) Simulator feedback
     - simulator/simulate.py → react(...) returns structured META: answered_clarification, reject_signal, silence, off_topic_flag, satisfaction
  4) Reward & preference pairs
     - reward/compute.py: compute_task_score(...) (stub), compute_interrupt_cost(...), total_reward(...)
     - rank 3 behaviors by reward → pick (chosen_action, rejected_action)
  5) Output
     - data/*.jsonl, each line contains: {state, chosen_action, rejected_action, chosen_text, rejected_text, rewards}