import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset


def to_state(rec: Dict[str, Any], i: int) -> Dict[str, Any]:
    desc = (rec.get("text") or rec.get("task_description") or "").strip()
    return {
        "id": f"mbpp-{i}",
        "domain": "coding",
        "query": f"Write a Python function: {desc}",
        "dialogue_turn": 1,
        "query_clarity": 0.6,
        "task_complexity": 0.6,
        "prev_reject": 0,
    }


def main(split: str = "test", out: str = "data/seeds/mbpp_states.jsonl", limit: int = 200) -> None:
    ds = load_dataset("mbpp", split=split)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(ds):
            if i >= limit:
                break
            f.write(json.dumps(to_state(rec, i), ensure_ascii=False) + "\n")
    print("Wrote:", out_path)


if __name__ == "__main__":
    # default args for quick run; change via CLI if needed
    main()


