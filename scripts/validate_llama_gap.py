"""
验证脚本：Llama 8B 在 direct / clarified / oracle 三种 prompt 下代码质量是否有差距。

核心问题：用 Llama 生成代码时，clarified > direct 的 gap 是否存在？
如果存在，A-2 方向（换 Llama 生成代码）可行。

运行：
    cd /root/autodl-tmp/ProactiveLLM
    python scripts/validate_llama_gap.py --n_tasks 20 --output outputs/validate_llama_gap.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward.compute import compute_task_score

TRAJ_PATH = PROJECT_ROOT / "data/logs/traj_colm_3turn_persona_150states_20260402_053113_20260402_053116.jsonl"
EXECUTE_PROMPT = (PROJECT_ROOT / "prompts/coding_execute.txt").read_text().strip()


PERSONA_NAMES = ["Novice-Learner", "Experienced-Engineer", "Busy-Developer"]


def load_good_trajs(traj_path: Path, n: int):
    """加载有 Clarify→Execute 的完整轨迹，按 persona 分层采样，每个 persona 取 n//3 个不重复 task。"""
    records = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    trajs = defaultdict(list)
    for r in records:
        trajs[r["trajectory_id"]].append(r)

    # 按 persona 分桶
    per_persona: dict[str, list] = {p: [] for p in PERSONA_NAMES}
    seen_tasks: dict[str, set] = {p: set() for p in PERSONA_NAMES}

    for tid, turns in trajs.items():
        turns = sorted(turns, key=lambda x: x["turn"])
        actions = [t["action"] for t in turns]
        task_id = turns[0]["state"]["id"]
        persona_name = turns[0].get("persona", {}).get("name", "")
        if persona_name not in per_persona:
            continue
        if task_id in seen_tasks[persona_name]:
            continue
        if "Clarify" in actions and actions[-1] == "Execute":
            per_persona[persona_name].append((tid, turns))
            seen_tasks[persona_name].add(task_id)

    # 每个 persona 取 n//3，不足则取全部
    per_n = max(1, n // len(PERSONA_NAMES))
    good = []
    for p in PERSONA_NAMES:
        sample = per_persona[p][:per_n]
        print(f"  {p}: {len(sample)} trajectories (available: {len(per_persona[p])})")
        good.extend(sample)

    print(f"Total loaded: {len(good)} trajectories ({len(PERSONA_NAMES)} personas × ~{per_n})")
    return good


def build_clarification_context(turns: list) -> str:
    """从轨迹里提取 Clarify Q&A，拼成上下文字符串。"""
    qa_parts = []
    for t in turns:
        if t["action"] == "Clarify":
            q = t.get("assistant_msg", "").strip()
            user_reply = t.get("user_reaction", {}).get("user_reply", "").strip()
            if q and user_reply:
                qa_parts.append(f"Assistant asked: {q}\nUser replied: {user_reply}")
    return "\n\n".join(qa_parts)


def build_new_clarification_context(turns: list) -> str:
    """用旧轨迹里的 assistant 问题，重跑新版 get_disclosure_info() 生成回答。
    这样可以在不重新生成轨迹的情况下，单独测试 disclosure.py 修改的效果。
    """
    from simulator.disclosure import get_disclosure_info

    state = turns[0]["state"]
    disclosure_rule = state.get("disclosure_rule")
    if not disclosure_rule:
        return ""

    # 深拷贝 disclosure_rule，避免污染原始数据；重置 disclosed_info
    import copy
    dr = copy.deepcopy(disclosure_rule)
    dr.setdefault("disclosed_info", {
        "edge_cases": [], "input_constraints": [], "output_format": [], "validation_rules": []
    })

    expertise = turns[0].get("persona", {}).get("expertise", "mid")

    def _collect_and_consolidate(exp: str) -> str:
        """收集所有轮次的披露信息，合并为单条 'Key requirements: ...' 列表。
        适用于 Novice（减少多轮噪音）和 Experienced（信息量大但格式干净）。
        """
        all_items = []
        for t in turns:
            if t["action"] != "Clarify":
                continue
            question = t.get("assistant_msg", "").strip()
            if not question:
                continue
            disclosure_text, new_disclosed = get_disclosure_info(
                assistant_question=question,
                disclosure_rule=dr,
                expertise=exp,
            )
            for key, items in new_disclosed.items():
                dr["disclosed_info"].setdefault(key, [])
                dr["disclosed_info"][key].extend(items)
            if disclosure_text:
                all_items.append(disclosure_text)

        if not all_items:
            return ""
        combined = "; ".join(all_items)
        first_question = next(
            (t.get("assistant_msg", "").strip() for t in turns if t["action"] == "Clarify"), ""
        )
        return f"Assistant asked: {first_question}\nUser replied: Key requirements: {combined}"

    # Novice: 多轮 Q&A 噪音大，合并成干净列表
    # Experienced: 信息量多（6条），也用干净列表，对齐 ideal_disclosed 格式
    if expertise in ("low", "high"):
        return _collect_and_consolidate(expertise)

    # Busy: 单轮 Q&A 信息量适中（3条），保留对话格式，已验证有效
    qa_parts = []
    for t in turns:
        if t["action"] != "Clarify":
            continue
        question = t.get("assistant_msg", "").strip()
        if not question:
            continue

        disclosure_text, new_disclosed = get_disclosure_info(
            assistant_question=question,
            disclosure_rule=dr,
            expertise=expertise,
        )
        for key, items in new_disclosed.items():
            dr["disclosed_info"].setdefault(key, [])
            dr["disclosed_info"][key].extend(items)

        if not disclosure_text:
            continue
        qa_parts.append(f"Assistant asked: {question}\nUser replied: {disclosure_text}")

    return "\n\n".join(qa_parts)


def build_prompt(task_query: str, clarification_context: str = "", oracle_prompt: str = "") -> str:
    """构造代码生成 prompt。"""
    if oracle_prompt:
        # Oracle：用完整的原始描述
        user_content = f"Task: {oracle_prompt}\n\nPlease implement the solution."
    elif clarification_context:
        # Clarified：masked query + Q&A
        user_content = (
            f"Task: {task_query}\n\n"
            f"Additional information from clarification:\n{clarification_context}\n\n"
            f"Please implement the solution."
        )
    else:
        # Direct：只有 masked query
        user_content = f"Task: {task_query}\n\nPlease implement the solution."

    return user_content


def generate_code_llama(model, tokenizer, prompt: str, system_prompt: str, max_new_tokens: int = 512) -> str:
    """用 Llama 生成代码。"""
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template returns BatchEncoding (dict-like) or tensor depending on version
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(model.device)
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"].to(model.device)
    else:
        input_ids = encoded.to(model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/validate_llama_gap.json")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    # 加载模型
    print(f"Loading {args.model_name} ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    # 加载轨迹
    good_trajs = load_good_trajs(TRAJ_PATH, args.n_tasks)

    VERSIONS = ["direct", "old_clarified", "new_clarified", "ideal_disclosed", "oracle"]
    results = []
    scores = {v: [] for v in VERSIONS}
    # per-persona 分层统计
    persona_scores: dict[str, dict[str, list]] = {
        p: {v: [] for v in VERSIONS} for p in PERSONA_NAMES
    }

    for i, (tid, turns) in enumerate(good_trajs):
        state = turns[0]["state"]
        task_id = state["id"]
        persona_name = turns[0].get("persona", {}).get("name", "unknown")
        masked_query = state.get("query", "")
        oracle_query = state.get("original_instruct_prompt", masked_query)
        clarification_ctx = build_clarification_context(turns)
        new_clarification_ctx = build_new_clarification_context(turns)

        # ideal_disclosure：直接平铺所有 masked_fields
        disclosure_rule = state.get("disclosure_rule", {})
        masked_fields = disclosure_rule.get("masked_fields", {}) if disclosure_rule else {}
        ideal_parts = []
        for field, items in masked_fields.items():
            for item in (items if isinstance(items, list) else [items]):
                if str(item).strip():
                    ideal_parts.append(str(item).strip())
        ideal_disclosure_ctx = "Key requirements: " + "; ".join(ideal_parts) if ideal_parts else ""

        print(f"\n[{i+1}/{len(good_trajs)}] {task_id} | {persona_name}")
        print(f"  Clarify turns : {sum(1 for t in turns if t['action']=='Clarify')}")
        print(f"  Old Q&A       : {len(clarification_ctx)} chars")
        print(f"  New Q&A       : {len(new_clarification_ctx)} chars | {new_clarification_ctx[:120]!r}")
        print(f"  Ideal         : {ideal_disclosure_ctx[:120]!r}")

        state_for_score = state.copy()
        state_for_score["action"] = "Execute"
        state_for_score["has_edge_cases_info"] = True

        task_scores = {}
        for version, extra_kwargs in [
            ("direct",          {"clarification_context": ""}),
            ("old_clarified",   {"clarification_context": clarification_ctx}),
            ("new_clarified",   {"clarification_context": new_clarification_ctx}),
            ("ideal_disclosed", {"clarification_context": ideal_disclosure_ctx}),
            ("oracle",          {"clarification_context": "", "oracle_prompt": oracle_query}),
        ]:
            prompt = build_prompt(masked_query, **extra_kwargs)
            code = generate_code_llama(model, tokenizer, prompt, EXECUTE_PROMPT)
            score = compute_task_score(state_for_score, "coding", assistant_output=code)
            task_scores[version] = score
            scores[version].append(score)
            if persona_name in persona_scores:
                persona_scores[persona_name][version].append(score)
            print(f"  {version:15s}: {score:.3f}")

        gap_old   = task_scores["old_clarified"]  - task_scores["direct"]
        gap_new   = task_scores["new_clarified"]  - task_scores["direct"]
        gap_ideal = task_scores["ideal_disclosed"] - task_scores["direct"]
        gap_oracle = task_scores["oracle"]         - task_scores["direct"]
        # 诊断：new 比 direct 差时打印原因线索
        if gap_new < 0:
            print(f"  ⚠ new_clarified HURT  (gap={gap_new:+.3f}) | new_ctx={new_clarification_ctx[:80]!r}")
        elif gap_new > 0:
            print(f"  ✓ new_clarified HELPED (gap={gap_new:+.3f})")
        print(f"  gap old={gap_old:+.3f}  new={gap_new:+.3f}  ideal={gap_ideal:+.3f}  oracle={gap_oracle:+.3f}")

        results.append({
            "task_id": task_id,
            "trajectory_id": tid,
            "persona": persona_name,
            "clarify_turns": sum(1 for t in turns if t["action"] == "Clarify"),
            "new_ctx_len": len(new_clarification_ctx),
            "new_ctx_preview": new_clarification_ctx[:200],
            "ideal_ctx_preview": ideal_disclosure_ctx[:200],
            "scores": task_scores,
            "gap_old_clarified":   gap_old,
            "gap_new_clarified":   gap_new,
            "gap_ideal_disclosed": gap_ideal,
            "gap_oracle_vs_direct": gap_oracle,
        })

    # 汇总
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    for v in VERSIONS:
        vals = scores[v]
        avg = sum(vals) / len(vals)
        pr = sum(1 for x in vals if x > 0) / len(vals)
        print(f"  {v:17s}: mean={avg:.3f}, pass_rate={pr:.1%}  (n={len(vals)})")

    gaps_old   = [r["gap_old_clarified"]   for r in results]
    gaps_new   = [r["gap_new_clarified"]   for r in results]
    gaps_ideal = [r["gap_ideal_disclosed"] for r in results]
    gaps_ora   = [r["gap_oracle_vs_direct"] for r in results]
    print(f"\n  old_clarified  - direct: avg={sum(gaps_old)/len(gaps_old):+.3f}  positive={sum(1 for x in gaps_old if x>0)}/{len(gaps_old)}")
    print(f"  new_clarified  - direct: avg={sum(gaps_new)/len(gaps_new):+.3f}  positive={sum(1 for x in gaps_new if x>0)}/{len(gaps_new)}")
    print(f"  ideal_disclosed- direct: avg={sum(gaps_ideal)/len(gaps_ideal):+.3f}  positive={sum(1 for x in gaps_ideal if x>0)}/{len(gaps_ideal)}")
    print(f"  oracle         - direct: avg={sum(gaps_ora)/len(gaps_ora):+.3f}  positive={sum(1 for x in gaps_ora if x>0)}/{len(gaps_ora)}")

    # per-persona 分解
    print("\n" + "="*60)
    print("PER-PERSONA BREAKDOWN")
    print("="*60)
    for p in PERSONA_NAMES:
        pscores = persona_scores[p]
        n = len(pscores["direct"])
        if n == 0:
            continue
        print(f"\n  [{p}]  n={n}")
        for v in VERSIONS:
            vals = pscores[v]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            pr = sum(1 for x in vals if x > 0) / len(vals)
            marker = " ✓" if v == "new_clarified" and avg > sum(pscores["direct"]) / len(pscores["direct"]) else ""
            print(f"    {v:17s}: mean={avg:.3f}, pass_rate={pr:.1%}{marker}")

    # 总结判断
    new_avg = sum(gaps_new) / len(gaps_new)
    old_avg = sum(gaps_old) / len(gaps_old)
    verdict     = "✅ DIRECTION VALID"   if sum(gaps_ideal) / len(gaps_ideal) > 0.03 else "❌ GAP TOO SMALL"
    fix_verdict = "✅ FIX BEATS DIRECT"  if new_avg > 0 else (
                  "↑ FIX BETTER THAN OLD" if new_avg > old_avg else "❌ FIX NOT HELPING")
    print(f"\n  Direction : {verdict}")
    print(f"  Fix status: {fix_verdict}  (new_clarified - direct = {new_avg:+.3f})")

    # 保存
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_persona_summary = {}
    for p in PERSONA_NAMES:
        pscores = persona_scores[p]
        n = len(pscores["direct"])
        if n == 0:
            continue
        per_persona_summary[p] = {
            v: {"mean": sum(pscores[v]) / n, "pass_rate": sum(1 for x in pscores[v] if x > 0) / n}
            for v in VERSIONS if pscores[v]
        }
    with open(out_path, "w") as f:
        json.dump({
            "summary": {v: {"mean": sum(scores[v]) / len(scores[v]),
                            "pass_rate": sum(1 for x in scores[v] if x > 0) / len(scores[v])}
                        for v in VERSIONS},
            "per_persona": per_persona_summary,
            "avg_gap_old_clarified":   sum(gaps_old) / len(gaps_old),
            "avg_gap_new_clarified":   sum(gaps_new) / len(gaps_new),
            "avg_gap_ideal_disclosed": sum(gaps_ideal) / len(gaps_ideal),
            "avg_gap_oracle":          sum(gaps_ora) / len(gaps_ora),
            "total": len(results),
            "verdict": verdict,
            "fix_verdict": fix_verdict,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
