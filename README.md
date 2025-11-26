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

Quickstart (Two-Step Pipeline)

The data generation is split into two steps for flexibility: generate trajectories first, then compute rewards separately.

Step 1: Generate Trajectories
- Synthetic mode (quick test, no API calls):
  ```bash
  python scripts/generate_trajectories.py --mode synthetic --domain coding --n_states 50 --out logs/traj_synth_coding.jsonl
  ```
- Dataset mode (with OpenAI):
  ```bash
  export OPENAI_API_KEY=sk-...
  python scripts/generate_trajectories.py --mode dataset --domain coding \
    --dataset_path data/seeds/mbpp_states.jsonl --n_states 100 \
    --out logs/traj_mbpp.jsonl --llm_model gpt-4o-mini
  ```

**Strategy**: Mainline+Branches (default, cost-efficient)
- Generates 1 mainline trajectory + 2 branches (other actions) per state
- Mainline action is **auto-selected from persona** (patience/clarity/time_pressure):
  - Low patience + high time_pressure → LOW
  - High patience + low clarity → HIGH
  - Otherwise → MID
- Manual override: `--mainline_action LOW|MID|HIGH` (optional)

Step 2: Compute Rewards
- Compute rewards and generate preference pairs:
  ```bash
  python reward/compute_rewards.py --trajectories logs/traj_synth_coding.jsonl --out dpo/prefs_synth_coding.jsonl
  python reward/compute_rewards.py --trajectories logs/traj_mbpp.jsonl --out dpo/prefs_mbpp_phase1.jsonl
  ```
- **Benefit**: You can recompute rewards multiple times with different weights without regenerating trajectories!

Then:
- Train policy: `python policy/train_dpo.py`
- Evaluate: `python eval/evaluate.py`
- Plot: `python eval/plot_pareto.py`

MBPP Quickstart (Phase-1)

1. Convert MBPP → states JSONL:
   ```bash
   python scripts/convert_mbpp_to_states.py  # writes data/seeds/mbpp_states.jsonl
   ```

2. Generate trajectories:
   ```bash
   export OPENAI_API_KEY=sk-...
   python scripts/generate_trajectories.py --mode dataset --domain coding \
     --dataset_path data/seeds/mbpp_states.jsonl --n_states 100 \
     --out logs/traj_mbpp.jsonl --llm_model gpt-4o-mini
   ```

3. Compute rewards:
   ```bash
   python reward/compute_rewards.py --trajectories logs/traj_mbpp.jsonl --out dpo/prefs_mbpp_phase1.jsonl
   ```

Data Generation Pipeline (Two-Step)

Key Principle
- Never modify `external/` datasets in place. Pipeline transforms: `seeds/ → logs/ → dpo/`.
- Two-step design allows recomputing rewards without regenerating expensive LLM calls.

---

### Step 1: Generate Trajectories (`scripts/generate_trajectories.py`)

**Purpose**: Generate assistant outputs and simulator reactions for each (state, action) pair.

**Input**:
- **Synthetic mode**: No input file needed (generates states internally)
- **Dataset mode**: States JSONL from `data/seeds/` (e.g., `mbpp_states.jsonl`)
  - Required fields: `{query, dialogue_turn, query_clarity, task_complexity, prev_reject, mbpp_tests?}`

**Process** (Mainline+Branches strategy):
- **Phase 1**: Generate 1 mainline trajectory
  - If `--mainline_action` provided: use it
  - Otherwise: **auto-select from persona** (patience/clarity/time_pressure)
    - Low patience + high time_pressure → LOW (direct, no questions)
    - High patience + low clarity → HIGH (can ask more)
    - Otherwise → MID (balanced)
- **Phase 2**: At each decision point, generate 2 branches (the other 2 actions)
- **Benefit**: Mainline matches persona preferences; branches provide contrastive actions for learning

**Steps**:
1. **Load states**:
   - Synthetic: `synth_states()` generates simplified states
   - Dataset: `load_states_from_dataset()` reads from JSONL

2. **Generate assistant outputs**:
   - Load templates: `prompts/coding_{low,mid,high}.txt` or `planning_{...}.txt`
   - Generate `assistant_msg`:
     - Without `--llm_model`: `dummy_llm_output()` (placeholder for testing)
     - With `--llm_model`: `llm_output()` → `llm/provider.py` → OpenAI API

3. **Get simulator reaction**:
   - `simulator/simulate.py` → `react(user_msg, assistant_msg, persona)`
   - Returns: `{user_reply, meta}` where `meta` contains:
     - `answered_clarification`, `reject_signal`, `silence`, `off_topic_flag`, `satisfaction`

**Output**: `data/logs/*.jsonl`
- Each line: `{state, action, action_prompt, assistant_msg, persona, user_reaction, is_mainline?, decision_point?, mainline_action?}`
- Format: One trajectory per line (JSONL)
- For mainline+branches strategy: trajectories include `is_mainline` flag and `decision_point` index

**Files Used**:
- `scripts/generate_trajectories.py` (main script)
- `prompts/coding_*.txt`, `prompts/planning_*.txt` (behavior templates)
- `llm/provider.py` (OpenAI API calls, if `--llm_model` provided)
- `simulator/simulate.py` (user reaction simulation)

---

### Step 2: Compute Rewards (`reward/compute_rewards.py`)

**Purpose**: Compute rewards for each trajectory and generate DPO preference pairs.

**Input**: Trajectories JSONL from `data/logs/` (generated in Step 1)

**Process**:
1. **Load trajectories** from JSONL file

2. **For each trajectory, compute reward**:
   - Extract features:
     - `n_questions = assistant_msg.count("?")`
     - `length_tokens = len(assistant_msg.split())`
     - `meta` from `user_reaction`
   - Compute `R_task`:
     - `reward/compute.py` → `compute_task_score(state, domain, assistant_output)`
     - For coding: can use `mbpp_tests` if present (via `reward/mbpp_eval.py`)
   - Compute `C_interrupt`:
     - `reward/compute.py` → `compute_interrupt_cost(meta, n_questions, length_tokens, off_topic)`
     - Formula: `α * (w1*n_questions + w2*reject_signal + w3*long_turn + w4*off_topic)`
   - Total reward: `R = R_task − C_interrupt`

3. **Build preference pairs**:
   - Group trajectories by `state_id`
   - Rank 3 actions (LOW/MID/HIGH) by reward
   - Pick highest as `chosen_action`, lowest as `rejected_action`

**Output**: `data/dpo/*.jsonl`
- Each line: `{state, chosen_action, rejected_action, chosen_text, rejected_text, rewards, task_scores, interrupt_costs}`
- Format: One preference pair per line (JSONL)

**Files Used**:
- `reward/compute_rewards.py` (main script)
- `reward/compute.py` (reward computation functions)
- `reward/mbpp_eval.py` (optional: MBPP test execution for `R_task`)

**Benefits of Two-Step Design**:
- ✅ Recompute rewards without regenerating trajectories (adjust weights, fix bugs)
- ✅ Audit intermediate trajectories for debugging
- ✅ Reuse expensive LLM calls when tuning reward parameters
- ✅ Separate concerns: generation vs. scoring