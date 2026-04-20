"""
Replay disclosure for completed DPO eval JSONs to recover per-conversation
disclosure_rate (eval JSON missed logging `disclosed_items` per turn).

Output: data/analysis/disclosure_per_conversation.csv
Columns: state_id, persona, n_masked, n_disclosed, disclosure_rate,
         n_clarify_turns, n_answered_clarifies, pass1, pass5
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.disclosure import get_disclosure_info

PERSONA_TO_EXPERTISE = {
    "Novice-Learner": "low",
    "Busy-Developer": "mid",
    "Experienced-Engineer": "high",
}

EVAL_FILES = [
    ROOT / "outputs" / "eval_v29_100states_50test.json",
    ROOT / "outputs" / "eval_v29_dpo_150extra.json",
]
TEST_STATE_FILES = [
    ROOT / "data" / "seeds" / "test_states_v29_eval_50.jsonl",
    ROOT / "data" / "seeds" / "test_states_v29_eval_150extra.jsonl",
]
OUT_CSV = ROOT / "data" / "analysis" / "disclosure_per_conversation.csv"


def load_test_states():
    """state_id → masked_fields dict"""
    state_map = {}
    for path in TEST_STATE_FILES:
        with open(path) as f:
            for line in f:
                s = json.loads(line)
                dr = s.get("disclosure_rule", {})
                mf = dr.get("masked_fields", {})
                state_map[s["id"]] = mf
    return state_map


def n_items(masked_fields):
    return sum(len(v) for v in masked_fields.values() if isinstance(v, list))


def fresh_disclosure_rule(masked_fields):
    return {
        "masked_fields": masked_fields,
        "disclosed_info": {k: [] for k in masked_fields.keys()},
    }


def replay_conversation(conv_record, masked_fields):
    """Returns (n_disclosed, n_clarify_turns, n_answered_clarifies)."""
    expertise = PERSONA_TO_EXPERTISE[conv_record["persona"]]
    disclosure_rule = fresh_disclosure_rule(masked_fields)

    n_clarify = 0
    n_answered = 0
    for turn in conv_record["conversation"]:
        if turn["action"] != "Clarify":
            continue
        n_clarify += 1
        if not turn.get("answered_clarification"):
            continue
        n_answered += 1
        assistant_msg = turn.get("assistant_msg", "")
        n_q = max(1, assistant_msg.count("?"))
        _, new_disclosed = get_disclosure_info(
            assistant_msg,
            disclosure_rule,
            expertise=expertise,
            dialogue_turn=turn.get("turn", 0),
            n_questions=n_q,
        )
        # Merge new disclosures into running disclosed_info
        for cat, items in new_disclosed.items():
            disclosure_rule["disclosed_info"].setdefault(cat, [])
            for item in items:
                if item not in disclosure_rule["disclosed_info"][cat]:
                    disclosure_rule["disclosed_info"][cat].append(item)

    n_disclosed = sum(len(v) for v in disclosure_rule["disclosed_info"].values())
    return n_disclosed, n_clarify, n_answered


def main():
    state_map = load_test_states()
    print(f"Loaded {len(state_map)} test states")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_no_state = 0
    skipped_no_masked = 0

    for eval_path in EVAL_FILES:
        with open(eval_path) as f:
            data = json.load(f)
        print(f"\n=== {eval_path.name}: {len(data['detailed_results'])} conversations ===")
        for conv in data["detailed_results"]:
            sid = conv["state_id"]
            if sid not in state_map:
                skipped_no_state += 1
                continue
            mf = state_map[sid]
            nm = n_items(mf)
            if nm == 0:
                skipped_no_masked += 1
                continue
            nd, nc, na = replay_conversation(conv, mf)
            disclosure_rate = min(nd, nm) / nm

            # pass@k from final turn
            last = conv["conversation"][-1]
            pass_dict = last.get("pass_at_k", {})
            p1 = bool(pass_dict.get("pass@1", False))
            p5 = bool(pass_dict.get("pass@5", False))

            rows.append({
                "state_id": sid,
                "persona": conv["persona"],
                "n_masked": nm,
                "n_disclosed": nd,
                "disclosure_rate": disclosure_rate,
                "n_clarify_turns": nc,
                "n_answered_clarifies": na,
                "pass1": int(p1),
                "pass5": int(p5),
            })

    print(f"\nTotal rows: {len(rows)} | skipped no_state={skipped_no_state}, no_masked={skipped_no_masked}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
