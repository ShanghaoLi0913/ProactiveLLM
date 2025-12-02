"""
Convert ConvCodeWorld/convcodebench dataset to our state format.

Dataset: https://huggingface.co/datasets/ConvCodeWorld/convcodebench
Format: Parquet with nested structure (different configurations: CF_EF_UNIT_SNF, CF_EF_FULL_SNF, etc.)
"""
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError as e:
    HAS_DATASETS = False
    IMPORT_ERROR = str(e)


def extract_tasks_from_config(config_data: Dict, config_name: str) -> List[Dict]:
    """
    Extract individual tasks from ConvCodeWorld nested structure.
    
    Structure: {"ITER=1": {"task_id": [...], "previous_code": [...], "verbal_feedback": [...], ...}, ...}
    All fields are parallel lists, need to pair them by index.
    """
    tasks = []
    if not isinstance(config_data, dict):
        return tasks
    
    # Iterate over ITER keys (e.g., "ITER=1", "ITER=2", ...)
    for iter_key, iter_data in config_data.items():
        if not isinstance(iter_data, dict):
            continue
        
        task_ids = iter_data.get("task_id", [])
        if not task_ids:
            continue
        
        # Get all list fields and pair them by index
        n_tasks = len(task_ids)
        
        for idx in range(n_tasks):
            # Extract each field's value at this index
            task = {
                "task_id": task_ids[idx] if idx < len(task_ids) else None,
                "iter": iter_key,
                "config": config_name,
            }
            
            # Extract other fields by index
            for key, value in iter_data.items():
                if isinstance(value, list) and idx < len(value):
                    task[key] = value[idx]
                elif not isinstance(value, list):
                    # Non-list fields are shared across all tasks
                    task[key] = value
            
            tasks.append(task)
    
    return tasks


def to_state(task_rec: Dict[str, Any], i: int, dataset_name: str = "convcodeworld") -> Dict[str, Any]:
    """
    Convert ConvCodeWorld task record to our state format.
    
    ConvCodeWorld fields:
    - task_id: Task identifier (e.g., "BigCodeBench/0")
    - verbal_feedback: User's feedback/query (this is the conversation input)
    - previous_code: Previous code attempt (if any)
    - compilation_feedback: Compilation errors
    - execution_feedback: Execution errors
    - label: pass/fail
    """
    task_id = task_rec.get("task_id", f"task-{i}")
    
    # Extract query from verbal_feedback (this is the user's request/feedback)
    query = task_rec.get("verbal_feedback", "")
    if not query:
        # Fallback: use task_id as query
        query = f"Task: {task_id}"
    
    # Extract tests/feedback information
    # ConvCodeWorld doesn't have explicit test cases, but we can use feedback
    # For now, we'll store compilation and execution feedback as "tests"
    tests_parts = []
    if task_rec.get("compilation_feedback"):
        tests_parts.append(f"# Compilation feedback:\n{task_rec['compilation_feedback']}")
    if task_rec.get("execution_feedback"):
        tests_parts.append(f"# Execution feedback:\n{task_rec['execution_feedback']}")
    tests = "\n\n".join(tests_parts) if tests_parts else None
    
    return {
        "id": f"{dataset_name}-{i}",
        "domain": "coding",
        "query": query.strip(),
        "dialogue_turn": 1,
        "query_clarity": 0.6,  # TODO: compute from query if possible
        "task_uncertainty": 0.6,  # TODO: compute from task if possible
        "time_pressure": "low",  # TODO: compute from context if possible
        "prev_reject": 0,
        "convcodeworld_tests": tests,  # compilation/execution feedback
        "convcodeworld_task_id": task_id,  # preserve original task ID
        "convcodeworld_previous_code": task_rec.get("previous_code"),  # preserve previous code if available
        "convcodeworld_label": task_rec.get("label"),  # preserve pass/fail label
    }


def load_from_hf(dataset_name: str, split: str = "train", config: Optional[str] = None, limit: Optional[int] = None, skip_raw_save: bool = False):
    """
    Load dataset from Hugging Face.
    
    ConvCodeWorld has nested structure with different configurations.
    We'll extract tasks from the specified config (or use first available).
    """
    if not HAS_DATASETS:
        error_msg = "datasets library required. Install: pip install datasets"
        if 'IMPORT_ERROR' in globals():
            error_msg += f"\nImport error: {IMPORT_ERROR}"
        raise ImportError(error_msg)
    
    ds = load_dataset(dataset_name, split=split)
    
    # Save raw dataset to data/external/ for reproducibility
    external_dir = Path("data/external/ConvCodeWorld")
    external_dir.mkdir(parents=True, exist_ok=True)
    raw_file = external_dir / f"convcodebench_raw_{split}.jsonl"
    
    # Save raw dataset to data/external/ for reproducibility (unless skipped)
    # Note: This dataset is very large (single row ~154MB JSON), but we save it for reproducibility
    if not skip_raw_save:
        # Check if raw file already exists and is not empty
        if raw_file.exists() and raw_file.stat().st_size > 0:
            print(f"Raw dataset already exists at {raw_file}, skipping save")
        else:
            print(f"Saving raw dataset to {raw_file}...")
            print("Note: This dataset is very large (single row ~154MB), using datasets library's efficient save...")
            try:
                # Use datasets library's to_json method which is more memory-efficient
                ds.to_json(str(raw_file), force_ascii=False)
                file_size_mb = raw_file.stat().st_size / 1024 / 1024
                print(f"✓ Saved raw dataset to {raw_file} ({file_size_mb:.2f} MB)")
            except Exception as e:
                print(f"Warning: Failed to save using datasets.to_json: {e}")
                print("Falling back to manual JSON serialization...")
                # Fallback: manual serialization with progress
                count = 0
                with raw_file.open("w", encoding="utf-8") as f:
                    for row in ds:
                        json_str = json.dumps(row, ensure_ascii=False, separators=(',', ':'))
                        f.write(json_str + "\n")
                        count += 1
                        print(f"  Saved row {count}...")
                file_size_mb = raw_file.stat().st_size / 1024 / 1024
                print(f"✓ Saved {count} raw record(s) to {raw_file} ({file_size_mb:.2f} MB)")
    
    # Extract tasks from configurations
    # Note: Each config field can contain 10k+ tasks, so we apply limit early
    all_tasks = []
    for row in ds:
        # Row contains multiple config fields (CF_EF_UNIT_SNF, CF_EF_FULL_SNF, etc.)
        for field_name, field_value in row.items():
            # If config filter specified, only process that config
            if config and field_name != config:
                continue
            if field_name.startswith("CF_") and isinstance(field_value, dict):
                print(f"Processing config: {field_name}...")
                config_tasks = extract_tasks_from_config(field_value, field_name)
                print(f"  Extracted {len(config_tasks)} tasks from {field_name}")
                # Apply limit early to avoid creating too many objects
                if limit:
                    remaining = limit - len(all_tasks)
                    if remaining <= 0:
                        break
                    config_tasks = config_tasks[:remaining]
                all_tasks.extend(config_tasks)
                if limit and len(all_tasks) >= limit:
                    break
        if limit and len(all_tasks) >= limit:
            break
    
    return all_tasks


def load_from_jsonl(file_path: Path, limit: Optional[int] = None):
    """Load dataset from local JSONL file."""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert ConvCodeWorld/convcodebench dataset to state format")
    parser.add_argument("--source", type=str, default="hf:ConvCodeWorld/convcodebench",
                       help="Source: 'hf:<dataset_name>' for Hugging Face (default: ConvCodeWorld/convcodebench), or path to local file")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (default: train)")
    parser.add_argument("--out", type=str, default="data/seeds/convcodeworld_states.jsonl",
                       help="Output path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to convert")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect dataset structure and exit (for debugging)")
    parser.add_argument("--skip_raw_save", action="store_true",
                       help="Skip saving raw dataset to data/external/ (default: save raw data for reproducibility)")
    parser.add_argument("--config", type=str, default=None,
                       help="Process only specific config field (e.g., CF_EF_UNIT_SNF). If not specified, processes all configs.")
    args = parser.parse_args()

    # Load dataset
    if args.source.startswith("hf:") or not Path(args.source).exists():
        if args.source.startswith("hf:"):
            dataset_name = args.source[3:]
        else:
            dataset_name = args.source
        print(f"Loading from Hugging Face: {dataset_name} (split: {args.split})")
        records = load_from_hf(dataset_name, args.split, config=args.config, limit=args.limit, skip_raw_save=args.skip_raw_save)
        
        # Inspect mode: print structure and exit
        if args.inspect:
            print(f"\nLoaded {len(records)} tasks")
            if records:
                print("\nFirst task structure (summary):")
                first_task = records[0]
                print(f"  task_id: {first_task.get('task_id', 'N/A')}")
                print(f"  iter: {first_task.get('iter', 'N/A')}")
                print(f"  config: {first_task.get('config', 'N/A')}")
                raw_data = first_task.get('raw_data', {})
                print(f"\n  Raw data keys: {list(raw_data.keys())[:10]}...")  # Show first 10 keys
                # Show sample of task_id if available
                if 'task_id' in raw_data:
                    task_ids = raw_data['task_id']
                    if isinstance(task_ids, list) and len(task_ids) > 0:
                        print(f"  Sample task_id: {task_ids[0]}")
                print(f"\n  Full raw_data structure (first 500 chars):")
                print(json.dumps(raw_data, indent=2, ensure_ascii=False)[:500] + "...")
            return
    else:
        file_path = Path(args.source)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        print(f"Loading from local file: {file_path}")
        records = load_from_jsonl(file_path, args.limit)
        
        # If loading from local file, we still need to extract tasks
        # But the structure might be different - adjust as needed
        if records and isinstance(records[0], dict):
            # Check if it's already in the extracted format or raw format
            if "task_id" in records[0] and "iter" in records[0]:
                # Already extracted, use as is
                pass
            else:
                # Raw format, need to extract
                all_tasks = []
                for row in records:
                    for field_name, field_value in row.items():
                        if field_name.startswith("CF_") and isinstance(field_value, dict):
                            config_tasks = extract_tasks_from_config(field_value, field_name)
                            all_tasks.extend(config_tasks)
                records = all_tasks

    # Convert to states
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_name = "convcodeworld"
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            state = to_state(rec, i, dataset_name)
            f.write(json.dumps(state, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(records)} states to {out_path}")
    print("\nNote: If fields are missing, run with --inspect to see actual structure, then adjust to_state() function.")


if __name__ == "__main__":
    main()

