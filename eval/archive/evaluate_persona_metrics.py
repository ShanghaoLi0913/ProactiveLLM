"""
评估DPO模型的Persona区分能力
计算详细的persona相关指标
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict]:
    """加载JSONL文件"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def map_persona_to_category(persona: Dict) -> str:
    """将persona映射到类别 (Cautious/Proactive/Balanced)"""
    expertise = persona.get("expertise", "medium").lower()
    patience = persona.get("patience", "medium").lower()
    
    # Cautious: 低expertise或高patience
    if expertise == "low" or patience == "high":
        return "Cautious"
    # Proactive: 高expertise或低patience
    elif expertise == "high" or patience == "low":
        return "Proactive"
    # Balanced: 其他
    else:
        return "Balanced"


def should_clarify_for_persona(persona_category: str, state: Dict) -> bool:
    """根据persona判断理想情况下是否应该Clarify
    
    这是一个简化的启发式规则：
    - Cautious: 在高不确定性下更倾向于Clarify
    - Proactive: 更倾向于直接Execute
    - Balanced: 中等策略
    """
    uncertainty = state.get("task_uncertainty", 0.5)
    turn = state.get("dialogue_turn", 0)
    
    if persona_category == "Cautious":
        # Cautious在不确定性>0.6时应该Clarify
        return uncertainty > 0.6 and turn == 0
    elif persona_category == "Proactive":
        # Proactive通常直接Execute
        return False
    else:  # Balanced
        # Balanced在不确定性>0.8时才Clarify
        return uncertainty > 0.8 and turn == 0


def compute_persona_metrics(
    trajectories: List[Dict],
    eval_results: Dict,
    verbose: bool = False
) -> Dict:
    """计算persona相关的指标
    
    Args:
        trajectories: trajectory数据（包含persona信息）
        eval_results: 模型评估结果（包含predicted_action等）
        verbose: 是否输出详细信息
    
    Returns:
        包含所有persona指标的字典
    """
    # 建立state_id到trajectory的映射
    traj_map = {}
    for traj in trajectories:
        state_id = traj["state"]["id"]
        if state_id not in traj_map:
            traj_map[state_id] = []
        traj_map[state_id].append(traj)
    
    # 为每个评估结果添加persona信息
    enriched_results = []
    for result in eval_results["detailed_results"]:
        state_id = result["state_id"]
        if state_id in traj_map:
            # 使用第一个轮次的trajectory（turn 1，即第一个action）
            first_turn_traj = min(traj_map[state_id], key=lambda x: x.get("turn", 0))
            persona = first_turn_traj.get("persona", {})
            persona_category = map_persona_to_category(persona)
            
            enriched_results.append({
                **result,
                "persona": persona,
                "persona_category": persona_category,
                "state": traj_map[state_id][0]["state"]
            })
    
    if verbose:
        print(f"\n📊 匹配到 {len(enriched_results)}/{len(eval_results['detailed_results'])} 个样本的persona信息")
    
    # 1️⃣ Persona-Conditioned Action Accuracy
    persona_stats = defaultdict(lambda: {"total": 0, "clarify": 0, "execute": 0, "correct": 0})
    
    for result in enriched_results:
        category = result["persona_category"]
        predicted = result["predicted_action"]
        ideal_action = "Clarify" if should_clarify_for_persona(category, result["state"]) else "Execute"
        
        persona_stats[category]["total"] += 1
        if predicted == "Clarify":
            persona_stats[category]["clarify"] += 1
        else:
            persona_stats[category]["execute"] += 1
        
        if predicted == ideal_action:
            persona_stats[category]["correct"] += 1
    
    persona_accuracy = {}
    for category, stats in persona_stats.items():
        if stats["total"] > 0:
            persona_accuracy[category] = (stats["correct"] / stats["total"]) * 100
        else:
            persona_accuracy[category] = 0.0
    
    # 2️⃣ Persona Discrimination Score (PDS)
    # PDS = |P(Clarify|Cautious) - P(Clarify|Proactive)|
    p_clarify_cautious = (
        persona_stats["Cautious"]["clarify"] / persona_stats["Cautious"]["total"]
        if persona_stats["Cautious"]["total"] > 0 else 0
    )
    p_clarify_proactive = (
        persona_stats["Proactive"]["clarify"] / persona_stats["Proactive"]["total"]
        if persona_stats["Proactive"]["total"] > 0 else 0
    )
    pds = abs(p_clarify_cautious - p_clarify_proactive)
    
    # 3️⃣ Action Distribution by Persona
    action_distribution = {}
    for category, stats in persona_stats.items():
        if stats["total"] > 0:
            action_distribution[category] = {
                "Execute": (stats["execute"] / stats["total"]) * 100,
                "Clarify": (stats["clarify"] / stats["total"]) * 100,
            }
        else:
            action_distribution[category] = {"Execute": 0, "Clarify": 0}
    
    # 4️⃣ TSR by Persona
    persona_tsr = defaultdict(lambda: {"success": 0, "total": 0})
    for result in enriched_results:
        category = result["persona_category"]
        persona_tsr[category]["total"] += 1
        if result["task_score"] >= 1.0:
            persona_tsr[category]["success"] += 1
    
    tsr_by_persona = {}
    for category, stats in persona_tsr.items():
        if stats["total"] > 0:
            tsr_by_persona[category] = (stats["success"] / stats["total"]) * 100
        else:
            tsr_by_persona[category] = 0.0
    
    # 5️⃣ Partial Pass Rate by Persona
    persona_pass_rates = defaultdict(list)
    for result in enriched_results:
        category = result["persona_category"]
        persona_pass_rates[category].append(result["task_score"])
    
    avg_pass_rate_by_persona = {}
    for category, scores in persona_pass_rates.items():
        if scores:
            avg_pass_rate_by_persona[category] = sum(scores) / len(scores)
        else:
            avg_pass_rate_by_persona[category] = 0.0
    
    # 6️⃣ Dialogue Efficiency by Persona
    # Efficiency = TSR / Avg_Turns (从state的dialogue_turn推断)
    persona_turns = defaultdict(list)
    for result in enriched_results:
        category = result["persona_category"]
        turns = result["state"].get("dialogue_turn", 0) + 1  # +1因为至少有1轮
        persona_turns[category].append(turns)
    
    avg_turns_by_persona = {}
    for category, turns_list in persona_turns.items():
        if turns_list:
            avg_turns_by_persona[category] = sum(turns_list) / len(turns_list)
        else:
            avg_turns_by_persona[category] = 1.0
    
    dialogue_efficiency = {}
    for category in tsr_by_persona:
        tsr = tsr_by_persona[category]
        avg_turns = avg_turns_by_persona.get(category, 1.0)
        if avg_turns > 0:
            dialogue_efficiency[category] = tsr / avg_turns
        else:
            dialogue_efficiency[category] = 0.0
    
    # 7️⃣ Overall Metrics (从eval_results获取)
    overall = {
        "task_success_rate": eval_results["summary"]["task_success_rate"],
        "action_accuracy": eval_results["summary"]["action_accuracy"],
        "avg_test_pass_rate": eval_results["summary"]["avg_test_pass_rate"],
        "total_samples": len(enriched_results),
    }
    
    return {
        "overall": overall,
        "persona_accuracy": persona_accuracy,
        "persona_discrimination_score": pds,
        "action_distribution": action_distribution,
        "tsr_by_persona": tsr_by_persona,
        "avg_pass_rate_by_persona": avg_pass_rate_by_persona,
        "avg_turns_by_persona": avg_turns_by_persona,
        "dialogue_efficiency": dialogue_efficiency,
        "persona_stats": dict(persona_stats),
        "p_clarify_by_persona": {
            "Cautious": p_clarify_cautious * 100,
            "Proactive": p_clarify_proactive * 100,
        }
    }


def generate_report(metrics: Dict, output_path: Optional[str] = None) -> str:
    """生成可读的报告"""
    lines = []
    lines.append("╔═══════════════════════════════════════════════════════════════════════╗")
    lines.append("║  Persona-Aware DPO Model Evaluation Report                           ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════════╝")
    lines.append("")
    
    # Overall Metrics
    lines.append("【整体指标】")
    lines.append("━" * 70)
    overall = metrics["overall"]
    lines.append(f"  Task Success Rate:     {overall['task_success_rate']:.2f}%")
    lines.append(f"  Action Accuracy:       {overall['action_accuracy']:.2f}%")
    lines.append(f"  Avg Test Pass Rate:    {overall['avg_test_pass_rate']:.4f}")
    lines.append(f"  Total Samples:         {overall['total_samples']}")
    lines.append("")
    
    # Persona Discrimination Score (最重要的指标)
    lines.append("【1️⃣  Persona Discrimination Score (PDS)】⭐⭐⭐")
    lines.append("━" * 70)
    pds = metrics["persona_discrimination_score"]
    lines.append(f"  PDS = |P(Clarify|Cautious) - P(Clarify|Proactive)|")
    lines.append(f"  PDS = {pds:.4f}")
    lines.append("")
    lines.append("  解释：")
    if pds > 0.3:
        lines.append("    ✅ 优秀 - 模型能很好地区分不同persona")
    elif pds > 0.15:
        lines.append("    ⚠️  中等 - 模型有一定的区分能力")
    else:
        lines.append("    ❌ 较差 - 模型几乎不区分persona")
    lines.append("")
    
    # P(Clarify) by Persona
    lines.append("  详细:")
    p_clarify = metrics["p_clarify_by_persona"]
    lines.append(f"    P(Clarify|Cautious):   {p_clarify['Cautious']:.2f}%")
    lines.append(f"    P(Clarify|Proactive):  {p_clarify['Proactive']:.2f}%")
    lines.append("")
    
    # Persona-Conditioned Action Accuracy
    lines.append("【2️⃣  Persona-Conditioned Action Accuracy】⭐⭐⭐")
    lines.append("━" * 70)
    acc = metrics["persona_accuracy"]
    lines.append("  Persona           Accuracy")
    lines.append("  ─────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in acc:
            lines.append(f"  {persona:<15}   {acc[persona]:>6.2f}%")
    lines.append("")
    
    # Action Distribution by Persona
    lines.append("【3️⃣  Action Distribution by Persona】⭐⭐")
    lines.append("━" * 70)
    dist = metrics["action_distribution"]
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in dist:
            lines.append(f"  {persona}:")
            lines.append(f"    Execute:  {dist[persona]['Execute']:>6.2f}%")
            lines.append(f"    Clarify:  {dist[persona]['Clarify']:>6.2f}%")
            lines.append("")
    
    # TSR by Persona
    lines.append("【4️⃣  Task Success Rate by Persona】⭐⭐")
    lines.append("━" * 70)
    tsr = metrics["tsr_by_persona"]
    lines.append("  Persona           TSR")
    lines.append("  ─────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in tsr:
            lines.append(f"  {persona:<15}   {tsr[persona]:>6.2f}%")
    lines.append("")
    
    # Partial Pass Rate by Persona
    lines.append("【5️⃣  Average Pass Rate by Persona】⭐⭐")
    lines.append("━" * 70)
    pass_rate = metrics["avg_pass_rate_by_persona"]
    lines.append("  Persona           Avg Pass Rate")
    lines.append("  ─────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in pass_rate:
            lines.append(f"  {persona:<15}   {pass_rate[persona]:>6.4f}")
    lines.append("")
    
    # Dialogue Efficiency
    lines.append("【6️⃣  Dialogue Efficiency by Persona】⭐")
    lines.append("━" * 70)
    eff = metrics["dialogue_efficiency"]
    turns = metrics["avg_turns_by_persona"]
    lines.append("  Persona           Avg Turns    Efficiency (TSR/Turns)")
    lines.append("  ───────────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in eff:
            lines.append(f"  {persona:<15}   {turns.get(persona, 0):.2f}        {eff[persona]:.2f}")
    lines.append("")
    
    # Sample Statistics
    lines.append("【样本统计】")
    lines.append("━" * 70)
    stats = metrics["persona_stats"]
    lines.append("  Persona           Total    Clarify    Execute")
    lines.append("  ───────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in stats:
            s = stats[persona]
            lines.append(f"  {persona:<15}   {s['total']:>5}    {s['clarify']:>7}    {s['execute']:>7}")
    lines.append("")
    
    lines.append("╔═══════════════════════════════════════════════════════════════════════╗")
    lines.append("║  Report Generated                                                     ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════════╝")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ 报告已保存到: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="评估DPO模型的Persona相关指标")
    parser.add_argument("--trajectories", type=str, required=True, help="Trajectory文件路径（包含persona信息）")
    parser.add_argument("--eval_results", type=str, required=True, help="评估结果文件路径（evaluate_dpo_model.py的输出）")
    parser.add_argument("--output", type=str, default=None, help="输出报告路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    print("📊 加载数据...")
    trajectories = load_jsonl(Path(args.trajectories))
    with open(args.eval_results, "r", encoding="utf-8") as f:
        eval_results = json.load(f)
    
    print(f"  - Trajectories: {len(trajectories)} 条")
    print(f"  - Eval Results: {len(eval_results['detailed_results'])} 个样本")
    
    print("\n🔍 计算Persona指标...")
    metrics = compute_persona_metrics(trajectories, eval_results, verbose=args.verbose)
    
    print("\n📝 生成报告...")
    report = generate_report(metrics, output_path=args.output)
    
    print("\n" + report)
    
    # 保存完整的metrics JSON
    if args.output:
        json_path = Path(args.output).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"✅ 完整指标已保存到: {json_path}")


if __name__ == "__main__":
    main()
