# Scripts Directory

Scripts for data generation and dataset conversion.

## Files

- **`generate_trajectories.py`**: Generate trajectories (Step 1 of pipeline)
- **`convert_mbpp_to_states.py`**: Convert MBPP dataset to our state format

---

## Trajectory Generation (`generate_trajectories.py`)

### Overview

Generates trajectories using **mainline+branches strategy** (cost-efficient):
- For each state: 1 mainline trajectory + 2 branches = 3 trajectories total
- Mainline serves as reference; branches provide contrastive actions for DPO learning

### Process Flow

```
Input: States (from synthetic or dataset)
  ↓
For each state:
  ├─ Phase 1: Generate mainline (auto-selected from persona, or manual)
  │   ├─ Determine mainline action:
  │   │   - If --mainline_action provided: use it
  │   │   - Otherwise: auto-select based on persona
  │   │     * Low patience + high time_pressure → LOW
  │   │     * High patience + low clarity → HIGH
  │   │     * Otherwise → MID
  │   └─ Load prompt template → Call LLM → Get simulator reaction
  │
  └─ Phase 2: Generate 2 branches (other actions, e.g., LOW/HIGH)
      └─ For each branch: Load prompt → Call LLM → Get simulator reaction
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
       - If `--mainline_action` provided: use it
       - Otherwise: auto-select from persona using `select_mainline_action_from_persona()`
         - Low patience (<0.4) + high time_pressure → LOW
         - High patience (>0.7) + low clarity (<0.5) → HIGH
         - Otherwise → MID
     - Load behavior template: `prompts/{domain}_{mainline_action}.txt`
     - Generate `assistant_msg`:
       - With `--llm_model`: Call OpenAI API (`llm/provider.py`)
       - Without: Use `dummy_llm_output()` (placeholder)
     - Get user reaction: `simulator/simulate.py` → `react()`
     - Mark: `is_mainline=True, decision_point=0, mainline_action_selected_by="persona"|"manual"`
   
   - **Phase 2 - Branches**:
     - For each remaining action (e.g., LOW/HIGH if mainline is MID):
       - Same process as mainline
       - Mark: `is_mainline=False, decision_point=0, mainline_action="MID"`

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
python scripts/generate_trajectories.py --mainline_action LOW ...
```

### Arguments

- `--mode`: `synthetic` (test) or `dataset` (from file)
- `--domain`: `coding` or `planning`
- `--n_states`: Number of states to process
- `--dataset_path`: Path to states JSONL (required for dataset mode)
- `--out`: Output path relative to `data/` (default: `logs/trajectories.jsonl`)
- `--llm_model`: OpenAI model name (e.g., `gpt-4o-mini`). If empty, uses dummy output.
- `--mainline_action`: `LOW`, `MID`, or `HIGH` (optional). If not provided, auto-selects based on persona characteristics

### Output Format

Each trajectory (one per line in JSONL):
```json
{
  "state": {
    "id": "mbpp-0",
    "domain": "coding",
    "query": "Write a Python function...",
    "dialogue_turn": 1,
    "query_clarity": 0.6,
    "task_complexity": 0.6,
    "prev_reject": 0,
    "mbpp_tests": "..."
  },
  "action": "MID",
  "action_prompt": "You are an assistant...",
  "assistant_msg": "请问需要处理空字符串吗？然后我会给出代码。",
  "persona": {
    "name": "Impatient-Novice",
    "patience": 0.25,
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
  "mainline_action": "MID"  // only for branches
}
```

### Example: 1 State → 3 Trajectories

**State**: `{id: "mbpp-0", query: "Write a function to reverse a string"}`

**Generated**:
1. Mainline (MID): `{action: "MID", is_mainline: true, assistant_msg: "请问需要处理空字符串吗？..."}`
2. Branch 1 (LOW): `{action: "LOW", is_mainline: false, mainline_action: "MID", assistant_msg: "这是一个最小可运行的示例代码..."}`
3. Branch 2 (HIGH): `{action: "HIGH", is_mainline: false, mainline_action: "MID", assistant_msg: "请问目标网站与输出格式？..."}`

### Files Used

- `prompts/coding_*.txt`, `prompts/planning_*.txt`: Behavior templates
- `llm/provider.py`: OpenAI API calls (if `--llm_model` provided)
- `simulator/simulate.py`: User reaction simulation

---

## MBPP Conversion (`convert_mbpp_to_states.py`)

### Purpose

Convert raw MBPP dataset to our state format for trajectory generation.

### Usage

```bash
python scripts/convert_mbpp_to_states.py
# Output: data/seeds/mbpp_states.jsonl
```

### Process

1. Load MBPP dataset from Hugging Face (`datasets` library)
2. Extract fields: `{text, test_list, test_imports}`
3. Convert to state format:
   - `query`: "Write a Python function: {text}"
   - `mbpp_tests`: Combined `test_imports` + `test_list`
   - Default values: `query_clarity=0.6`, `task_complexity=0.6`, `prev_reject=0`
4. Write to `data/seeds/mbpp_states.jsonl`

### Output Format

Each line:
```json
{
  "id": "mbpp-0",
  "domain": "coding",
  "query": "Write a Python function: ...",
  "dialogue_turn": 1,
  "query_clarity": 0.6,
  "task_complexity": 0.6,
  "prev_reject": 0,
  "mbpp_tests": "import ...\nassert ..."
}
```

