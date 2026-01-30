# Scripts Directory

Scripts for data generation and dataset conversion.

## Files

- **`generate_trajectories.py`**: Generate trajectories (Step 1 of pipeline)
- **`convert_mbpp_to_states.py`**: Convert MBPP dataset to our state format

---

## Trajectory Generation (`generate_trajectories.py`)

### Overview

Generates multi-turn conversation trajectories:
- For each state: Generate a full conversation until task completion or max_turns
- Each turn is a separate trajectory entry with its own state and action
- Sequential Decision Process: actions are **Clarify** (ask questions) or **Execute** (provide solution)
- Action selection is auto-selected from persona and state

### Process Flow

```
Input: States (from synthetic or dataset)
  ↓
For each state:
  Generate multi-turn conversation:
    Turn 1: Select action (auto from persona + state) → Generate → Simulator reaction → Update state
    Turn 2: Select action (auto from persona + state) → Generate → Simulator reaction → Update state
    ...
    Until: Task completed OR max_turns reached OR user rejects
  ↓
Output: Trajectories JSONL (data/logs/*.jsonl)
```

### Detailed Steps

1. **Load States**
   - Synthetic mode: Generate simplified states internally
   - Dataset mode: Read from JSONL file (`data/seeds/*.jsonl`)

2. **For Each State:**
   - **Phase 1 - Mainline**:
     - **Determine mainline action**:
       - If `--mainline_action` provided: use it (Clarify or Execute)
       - Otherwise: auto-select from persona and state using `select_mainline_action_from_persona()`
         - Low patience ("low") → Execute
         - High patience ("high") + low task_uncertainty (<0.5) → Clarify
         - Previous reject → Execute
         - Otherwise → Execute (default)
     - Load behavior template: `prompts/{domain}_{mainline_action}.txt` (clarify or execute)
     - Generate `assistant_msg`:
       - With `--llm_model`: Call OpenAI API (`llm/provider.py`)
       - Without: Use `dummy_llm_output()` (placeholder)
     - Get user reaction: `simulator/simulate.py` → `react()`
     - Mark: `is_mainline=True, decision_point=0, mainline_action_selected_by="persona"|"manual"`
   
   - **Phase 2 - Branches**:
     - **Branch 1**: The other action (Clarify if mainline is Execute, or vice versa)
       - Same process as mainline
       - Mark: `is_mainline=False, decision_point=0, mainline_action="Execute"` (or "Clarify")
     - **Branch 2**: Mainline action variant (regenerate with same action)
       - Same process as mainline but generates different output
       - Mark: `is_mainline=False, decision_point=0, mainline_action="Execute"` (or "Clarify"), `is_variant=True`

3. **Output**: Write trajectories to `data/logs/*.jsonl`

### Usage

```bash
# Synthetic mode (quick test, no API calls)
python scripts/generate_trajectories.py --mode synthetic --domain coding \
  --n_states 50 --out logs/traj_synth_coding.jsonl

# Dataset mode with OpenAI
export OPENAI_API_KEY=sk-...
python scripts/generate_trajectories.py --mode dataset --domain coding \
  --dataset_path data/seeds/mbpp_states.jsonl --n_states 100 \
  --out logs/traj_mbpp.jsonl --llm_model gpt-4o-mini

# Auto-select mainline from persona (default behavior)
python scripts/generate_trajectories.py --mode dataset ...

# Manual override mainline action
python scripts/generate_trajectories.py --mainline_action Clarify ...
# or
python scripts/generate_trajectories.py --mainline_action Execute ...
```

### Arguments

- `--mode`: `synthetic` (test) or `dataset` (from file)
- `--domain`: `coding` or `planning`
- `--n_states`: Number of states to process
- `--dataset_path`: Path to states JSONL (required for dataset mode)
- `--out`: Output path relative to `data/` (default: `logs/trajectories.jsonl`)
- `--llm_model`: OpenAI model name (e.g., `gpt-4o-mini`). If empty, uses dummy output.
- `--mainline_action`: `Clarify` or `Execute` (optional). If not provided, auto-selects based on persona and state (task_uncertainty, dialogue_turn, prev_reject)

### Output Format

Each trajectory (one per line in JSONL):
```json
{
  "state": {
    "id": "mbpp-0",
    "domain": "coding",
    "query": "Write a Python function...",
    "dialogue_turn": 0,
    "prev_reject": 0,
    "task_uncertainty": 0.5
  },
  "action": "Clarify",
  "action_prompt": "You are an assistant...",
  "assistant_msg": "请问需要处理空字符串吗？然后我会给出代码。",
  "persona": {
    "name": "Busy-Developer",
    "expertise": "mid",
    "patience": "low",
    ...
  },
  "user_reaction": {
    "user_reply": "好的。",
    "meta": {
      "answered_clarification": 1,
      "reject_signal": 0,
      "silence": 0,
      "off_topic_flag": 0,
      "satisfaction": 0.7
    }
  },
  "is_mainline": true,  // or false for branches
  "decision_point": 0,
  "mainline_action": "Execute"  // only for branches
}
```

### Example: Multi-Turn Conversation

**Initial State**: `{id: "mbpp-0", query: "Write a function to reverse a string", dialogue_turn: 0, prev_reject: 0, task_uncertainty: 0.4}`

**Generated Conversation** (example):
1. Turn 1: `{action: "Clarify", turn: 1, assistant_msg: "请问需要处理空字符串吗？", state: {...dialogue_turn: 0...}}`
2. Turn 2: `{action: "Execute", turn: 2, assistant_msg: "def reverse_string(s): return s[::-1]", state: {...dialogue_turn: 1...}, task_completed: true}`

Each turn is a separate trajectory entry with updated state.

### Files Used

- `prompts/coding_{clarify,execute}.txt`, `prompts/planning_{clarify,execute}.txt`: Behavior templates
- `llm/provider.py`: OpenAI API calls (if `--llm_model` provided)
- `simulator/simulate.py`: User reaction simulation
- `policy/render_state.py`: State rendering for consistency

---

## Dataset Conversion Scripts

### State Format

All datasets should be converted to the following state format for sequential decision process:

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

**Required fields**:
- `domain`: "coding" or "planning"
- `query`: User's task description
- `dialogue_turn`: Current dialogue turn (0 for initial state)
- `prev_reject`: Whether previous turn was rejected (0 or 1)

**Optional fields**:
- `id`: Unique identifier for the task
- `task_uncertainty`: Computed automatically if not provided (0.0-1.0, lower = more unclear)

**Note**: The state format supports sequential decision making where each turn updates `dialogue_turn` and `prev_reject` based on user reactions.

