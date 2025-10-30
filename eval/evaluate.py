import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize_rewards(rows: List[Dict]) -> Dict:
    # expects fields: rewards dict per example
    import numpy as np

    all_rewards = []
    for ex in rows:
        for v in ex.get("rewards", {}).values():
            all_rewards.append(v)
    if not all_rewards:
        return {"mean_reward": None}
    arr = np.array(all_rewards)
    return {"mean_reward": float(arr.mean()), "std_reward": float(arr.std())}


def main():
    data_path = Path(__file__).resolve().parent.parent / "data/prefs_mvp.jsonl"
    rows = load_jsonl(data_path)
    stats = summarize_rewards(rows)
    out_dir = Path(__file__).resolve().parent.parent / "eval"
    out = out_dir / "results.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()


