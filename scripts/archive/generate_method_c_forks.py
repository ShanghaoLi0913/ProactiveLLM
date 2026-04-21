"""
Generate Method C forks for an existing trajectory file.

Method C: For Experienced-Engineer Clarify mainline trajectories, fork a Clarify
action at each T1+ Execute turn. Creates pairs:
  Execute-now (mainline, chosen) vs Clarify-again (fork, rejected)

This script reads an existing trajectory file, finds eligible turns, and appends
the new fork turns to the same file (or a new file if --out is specified).

Usage:
    python scripts/generate_method_c_forks.py \
        --trajectories data/logs/traj_method_a_150states_20260403_044930.jsonl \
        --llm_model gpt-4o-mini
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_trajectories import (
    generate_multi_turn_conversation,
    PERSONAS,
    set_global_seed,
)


def find_experienced_persona_idx():
    for i, p in enumerate(PERSONAS):
        if p.name == "Experienced-Engineer":
            return i
    raise ValueError("Experienced-Engineer persona not found")


def main():
    parser = argparse.ArgumentParser(description="Append Method C forks to an existing trajectory file.")
    parser.add_argument("--trajectories", required=True, help="Existing trajectory JSONL file")
    parser.add_argument("--out", default="", help="Output file path. If empty, appends to input file.")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    set_global_seed(args.seed)

    traj_path = Path(args.trajectories)
    out_path = Path(args.out) if args.out else traj_path

    print(f"📂 Loading trajectories from: {traj_path}")
    lines = traj_path.read_text(encoding="utf-8").splitlines()
    trajs = [json.loads(l) for l in lines if l.strip()]
    print(f"📊 Loaded {len(trajs)} trajectory turns")

    # Group by trajectory_id
    by_traj = defaultdict(list)
    for t in trajs:
        by_traj[t["trajectory_id"]].append(t)

    # Collect already-existing Method C fork IDs to avoid duplicates
    existing_mc_ids = {t["trajectory_id"] for t in trajs if t.get("method_c")}
    print(f"   Existing Method C forks: {len(existing_mc_ids)}")

    exp_pid = find_experienced_persona_idx()

    candidates = []
    for tid, turns in by_traj.items():
        if not turns or turns[0].get("is_fork"):
            continue
        persona = turns[0].get("persona", {})
        pname = persona.get("name") if isinstance(persona, dict) else persona
        if pname != "Experienced-Engineer":
            continue
        if turns[0].get("action") != "Clarify":
            continue
        for t in turns[1:]:
            if t.get("action") == "Execute":
                fork_traj_id = f"{tid}_mc_t{t['turn']}"
                if fork_traj_id in existing_mc_ids:
                    continue
                candidates.append((tid, t))

    print(f"🎯 {len(candidates)} Method C fork candidates")

    if not candidates:
        print("✅ Nothing to generate.")
        return

    new_forks = []
    with out_path.open("a", encoding="utf-8") as f:
        for i, (parent_id, execute_turn) in enumerate(candidates):
            fork_turn_num = execute_turn["turn"]
            fork_state = execute_turn["state"].copy()
            fork_traj_id = f"{parent_id}_mc_t{fork_turn_num}"

            fork_trajs = generate_multi_turn_conversation(
                fork_state, "coding", args.llm_model, exp_pid, max_turns=1,
                force_first_action="Clarify",
                temperature=args.temperature,
                top_p=args.top_p,
                trajectory_id=fork_traj_id,
            )
            for ft in fork_trajs:
                ft["is_fork"] = True
                ft["fork_at_turn"] = fork_turn_num
                ft["parent_trajectory_id"] = parent_id
                ft["turn"] = fork_turn_num
                ft["method_c"] = True
                f.write(json.dumps(ft, ensure_ascii=False) + "\n")
            f.flush()
            new_forks.extend(fork_trajs)

            if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
                print(f"  [{i+1}/{len(candidates)}] Generated fork for {parent_id[:40]}...", flush=True)

    print(f"\n✅ Appended {len(new_forks)} Method C fork turns to {out_path}")
    print(f"   Total file size: {sum(1 for _ in out_path.open())} lines")


if __name__ == "__main__":
    main()
