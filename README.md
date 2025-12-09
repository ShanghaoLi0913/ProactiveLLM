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

- prompts/: behavior templates for coding and planning (clarify/execute)
- llm/: OpenAI API provider and API key setup (see llm/SETUP_API_KEY.md)
- simulator/: user personas and reaction rules
- reward/: task success scoring and interrupt cost
- policy/: DPO training and action selection (sequential decision: Clarify/Execute)
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

**Single-turn mode (default, for DPO training)**:
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

**Multi-turn mode (full conversations until task completion)**:
- Generate complete conversations with sequential decisions:
  ```bash
  python scripts/generate_trajectories.py --mode dataset --domain coding \
    --dataset_path data/seeds/your_dataset.jsonl --n_states 10 \
    --out logs/traj_multiturn.jsonl --llm_model gpt-4o-mini \
    --multi_turn --max_turns 5
  ```
- This mode generates full conversations where:
  - At each turn, model decides: **Clarify** (ask questions) or **Execute** (provide solution)
  - State updates after each turn: `dialogue_turn++`, `prev_reject` (if user rejected)
  - Task completion is detected (code passes tests or planning is complete)
  - Each turn is saved as a separate trajectory entry with its own state and action

**Strategy**: Mainline+Branches (single-turn mode, cost-efficient)
- Generates 1 mainline trajectory + 2 branches per state = 3 trajectories total
- Branch 1: The other action (Clarify if mainline is Execute, or vice versa)
- Branch 2: Mainline action variant (regenerate with same action but different output)
- Sequential Decision Process: actions are **Clarify** (ask questions) or **Execute** (provide solution)
- Mainline action is **auto-selected from persona and state**:
  - Low patience ("low") → Execute
  - High patience ("high") + low task_uncertainty → Clarify
  - Previous reject → Execute (don't ask more)
  - High dialogue_turn → Execute (already asked many questions)
  - Otherwise → Execute (default)
- Manual override: `--mainline_action Clarify|Execute` (optional)

Step 2: Compute Rewards
- Compute rewards and generate preference pairs:
  ```bash
  python reward/compute_rewards.py --trajectories logs/traj_synth_coding.jsonl --out dpo/prefs_synth_coding.jsonl
  python reward/compute_rewards.py --trajectories logs/traj_mbpp.jsonl --out dpo/prefs_mbpp_phase1.jsonl
  ```
- **Benefit**: You can recompute rewards multiple times with different weights without regenerating trajectories!

Then:
- Train policy: 
  ```bash
  python policy/train_dpo.py \
    --data data/dpo/prefs_synth_coding.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output outputs/policy_model \
    --epochs 3 --lr 5e-5 --beta 0.1
  ```
- Evaluate: 
  ```bash
  python eval/evaluate_dpo_model.py \
    --model_dir outputs/policy_model \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --prefs_path data/dpo/prefs_synth_coding.jsonl \
    --output outputs/eval_results.json
  ```
- Plot: `python eval/plot_pareto.py` (if available)

Dataset Quickstart

**MBPP** (if conversion script available):
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
   python reward/compute_rewards.py --trajectories logs/traj_mbpp.jsonl --out dpo/prefs_mbpp.jsonl
   ```

**Custom Dataset**:
1. Prepare your dataset in JSONL format with required fields:
   ```json
   {
     "id": "task-0",
     "domain": "coding",
     "query": "User's task description",
     "dialogue_turn": 0,
     "prev_reject": 0,
     "task_uncertainty": 0.5
   }
   ```

2. Generate trajectories:
   ```bash
   export OPENAI_API_KEY=sk-...
   python scripts/generate_trajectories.py --mode dataset --domain coding \
     --dataset_path data/seeds/your_dataset.jsonl --n_states 100 \
     --out logs/traj_your_dataset.jsonl --llm_model gpt-4o-mini
   ```

3. Compute rewards:
   ```bash
   python reward/compute_rewards.py --trajectories logs/traj_your_dataset.jsonl --out dpo/prefs_your_dataset.jsonl
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
  - Required fields: `{query, dialogue_turn, prev_reject, task_uncertainty?}`
  - Optional fields: `{domain, id, ...}`

**Process** (Mainline+Branches strategy):
- **Phase 1**: Generate 1 mainline trajectory
  - If `--mainline_action` provided: use it (Clarify or Execute)
  - Otherwise: **auto-select from persona and state**
    - Low patience → Execute (direct, no questions)
    - High patience + low task_uncertainty → Clarify (can ask questions)
    - Previous reject → Execute (don't ask more)
    - High dialogue_turn → Execute (already asked many questions)
    - Otherwise → Execute (default)
- **Phase 2**: Generate 1 branch (the other action: Clarify if mainline is Execute, or vice versa)
- **Benefit**: Mainline matches persona preferences; branch provides contrastive action for learning

**Steps**:
1. **Load states**:
   - Synthetic: `synth_states()` generates simplified states
   - Dataset: `load_states_from_dataset()` reads from JSONL

2. **Generate assistant outputs**:
   - Load templates: `prompts/coding_{clarify,execute}.txt` or `planning_{clarify,execute}.txt`
   - Generate `assistant_msg`:
     - Without `--llm_model`: `dummy_llm_output()` (placeholder for testing)
     - With `--llm_model`: `llm_output()` → `llm/provider.py` → OpenAI API
   - **Sequential Decision**: Each action (Clarify/Execute) generates appropriate response
     - Clarify: Ask 1-2 clarifying questions
     - Execute: Provide code/solution directly

3. **Get simulator reaction**:
   - `simulator/simulate.py` → `react(user_msg, assistant_msg, persona)`
   - Returns: `{user_reply, meta}` where `meta` contains:
     - `answered_clarification`, `reject_signal`, `silence`, `off_topic_flag`, `satisfaction`

**Output**: `data/logs/*.jsonl`
- Each line: `{state, action, action_prompt, assistant_msg, persona, user_reaction, is_mainline?, decision_point?, mainline_action?}`
- Format: One trajectory per line (JSONL)
- **State format**: `{domain, query, dialogue_turn, prev_reject, task_uncertainty}`
- **Action**: `"Clarify"` or `"Execute"`
- For mainline+branches strategy: trajectories include `is_mainline` flag and `decision_point` index

**Files Used**:
- `scripts/generate_trajectories.py` (main script)
- `prompts/coding_{clarify,execute}.txt`, `prompts/planning_{clarify,execute}.txt` (behavior templates)
- `llm/provider.py` (OpenAI API calls, if `--llm_model` provided)
- `simulator/simulate.py` (user reaction simulation)
- `policy/render_state.py` (state rendering for consistency)

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
   - Group trajectories by `state_id` (and `decision_point` if multi-turn)
   - Rank 2 actions (Clarify/Execute) by reward
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