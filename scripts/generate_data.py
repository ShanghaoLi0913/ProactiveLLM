import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

from simulator import PERSONAS, react
from reward import compute_task_score, compute_interrupt_cost, total_reward
from llm.provider import chat_complete


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_action_prompts(domain: str) -> Dict[str, str]:
    base = Path(__file__).resolve().parent.parent / "prompts"
    if domain == "coding":
        return {
            "LOW": load_prompt(base / "coding_low.txt"),
            "MID": load_prompt(base / "coding_mid.txt"),
            "HIGH": load_prompt(base / "coding_high.txt"),
        }
    elif domain == "planning":
        return {
            "LOW": load_prompt(base / "planning_low.txt"),
            "MID": load_prompt(base / "planning_mid.txt"),
            "HIGH": load_prompt(base / "planning_high.txt"),
        }
    else:
        raise ValueError(f"Unknown domain: {domain}")


def synth_states(domain: str, n: int) -> List[Dict]:
    samples = []
    for i in range(n):
        samples.append(
            {
                "id": f"{domain}-{i}",
                "domain": domain,
                "query": "帮我写个 Python 爬虫" if domain == "coding" else "帮我规划今天的待办",
                "dialogue_turn": 1,
                "query_clarity": 0.6,
                "task_complexity": 0.6,
                "prev_reject": 0,
            }
        )
    return samples


def load_states_from_dataset(dataset_path: Path, domain: str, limit: Optional[int] = None) -> List[Dict]:
    states: List[Dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            # Expected minimal fields in dataset JSONL:
            # {"query": str, "dialogue_turn": int, "query_clarity": float, "task_complexity": float, "prev_reject": int}
            states.append(
                {
                    "id": row.get("id", f"ds-{i}"),
                    "domain": domain,
                    "query": row["query"],
                    "dialogue_turn": int(row.get("dialogue_turn", 1)),
                    "query_clarity": float(row.get("query_clarity", 0.5)),
                    "task_complexity": float(row.get("task_complexity", 0.5)),
                    "prev_reject": int(row.get("prev_reject", 0)),
                }
            )
    return states


def score_branch(state: Dict, action: str, assistant_msg: str, persona_idx: int = 0) -> Dict:
    persona = PERSONAS[persona_idx % len(PERSONAS)]
    reaction = react(state["query"], assistant_msg, persona)
    meta = reaction["meta"]
    n_questions = assistant_msg.count("?")
    length_tokens = len(assistant_msg.split())
    off_topic = 0
    task = compute_task_score(state, state["domain"])
    cost = compute_interrupt_cost(meta, n_questions, length_tokens, off_topic)
    rew = total_reward(task, cost)
    return {
        "assistant_msg": assistant_msg,
        "meta": meta,
        "n_questions": n_questions,
        "length_tokens": length_tokens,
        "reward": rew,
    }


def dummy_llm_output(state: Dict, action_prompt: str) -> str:
    # Placeholder generation; replace with real LLM calls
    lower = action_prompt.lower()
    if "ask up to two clarifying" in lower:
        return "请问目标网站与输出格式？随后给出实现步骤与代码。"
    if "ask exactly one" in lower or "one key clarifying" in lower:
        return "请问需要爬取哪个网站的数据？然后我会给出代码。"
    return "这是一个最小可运行的示例代码/计划。"


def llm_output(state: Dict, action_prompt: str, model: str) -> str:
    system = action_prompt
    user = f"[Task]\n{state['query']}"
    return chat_complete(system, user, model=model, max_tokens=400)


def build_preference_pairs(states: List[Dict], domain: str) -> List[Dict]:
    prompts = build_action_prompts(domain)
    pairs = []
    for st in states:
        branches = {}
        for action, tpl in prompts.items():
            assistant_msg = dummy_llm_output(st, tpl)
            branches[action] = score_branch(st, action, assistant_msg)

        ranked = sorted(branches.items(), key=lambda kv: kv[1]["reward"], reverse=True)
        if len(ranked) >= 2:
            (a_plus, plus_obj), (a_minus, minus_obj) = ranked[0], ranked[-1]
            pairs.append(
                {
                    "state": st,
                    "chosen_action": a_plus,
                    "rejected_action": a_minus,
                    "chosen_text": plus_obj["assistant_msg"],
                    "rejected_text": minus_obj["assistant_msg"],
                    "rewards": {a: obj["reward"] for a, obj in branches.items()},
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic")
    parser.add_argument("--domain", choices=["coding", "planning"], default="coding")
    parser.add_argument("--n_states", type=int, default=50)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--out", type=str, default="dpo/prefs_mvp.jsonl")
    parser.add_argument("--llm_model", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dataset":
        if not args.dataset_path:
            raise SystemExit("--dataset_path is required in dataset mode")
        states = load_states_from_dataset(Path(args.dataset_path), domain=args.domain, limit=args.n_states)
    else:
        states = synth_states(args.domain, args.n_states)

    # monkey-patch generator used inside build_preference_pairs
    global dummy_llm_output
    if args.llm_model:
        def generator_wrapper(st, tpl):
            return llm_output(st, tpl, model=args.llm_model)
        gen_func = generator_wrapper
    else:
        def generator_wrapper(st, tpl):
            return dummy_llm_output(st, tpl)
        gen_func = generator_wrapper

    def build_with_gen(states_local: List[Dict], domain_local: str) -> List[Dict]:
        prompts = build_action_prompts(domain_local)
        pairs_local = []
        for st in states_local:
            branches = {}
            for action, tpl in prompts.items():
                assistant_msg = gen_func(st, tpl)
                branches[action] = score_branch(st, action, assistant_msg)
            ranked = sorted(branches.items(), key=lambda kv: kv[1]["reward"], reverse=True)
            if len(ranked) >= 2:
                (a_plus, plus_obj), (a_minus, minus_obj) = ranked[0], ranked[-1]
                pairs_local.append(
                    {
                        "state": st,
                        "chosen_action": a_plus,
                        "rejected_action": a_minus,
                        "chosen_text": plus_obj["assistant_msg"],
                        "rejected_text": minus_obj["assistant_msg"],
                        "rewards": {a: obj["reward"] for a, obj in branches.items()},
                    }
                )
        return pairs_local

    data = build_with_gen(states, args.domain)

    out_path = out_dir / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} preference examples to {out_path}")


if __name__ == "__main__":
    main()