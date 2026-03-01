"""
扩展评估指标计算
包括用户体验指标、Persona一致性指标等
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def compute_extended_metrics(eval_results_path: str, output_path: Optional[str] = None) -> Dict:
    """
    计算扩展评估指标
    
    Args:
        eval_results_path: 评估结果JSON文件路径
        output_path: 输出文件路径（可选）
    
    Returns:
        包含所有扩展指标的字典
    """
    # 加载评估结果
    with open(eval_results_path) as f:
        eval_data = json.load(f)
    
    summary = eval_data.get("summary", {})
    detailed_results = eval_data.get("detailed_results", [])
    
    # 初始化指标
    extended_metrics = {
        "persona_metrics": {},
        "user_experience_metrics": {},
        "code_quality_metrics": {},
        "overall_metrics": {}
    }
    
    # 按persona分组
    persona_results = defaultdict(list)
    for result in detailed_results:
        persona = result.get("persona", "Unknown")
        persona_results[persona].append(result)
    
    # 1. 计算Persona-specific指标
    for persona_name, results in persona_results.items():
        persona_metrics = compute_persona_metrics(persona_name, results)
        extended_metrics["persona_metrics"][persona_name] = persona_metrics
    
    # 2. 计算用户体验指标
    extended_metrics["user_experience_metrics"] = compute_user_experience_metrics(detailed_results)
    
    # 3. 计算代码质量指标
    extended_metrics["code_quality_metrics"] = compute_code_quality_metrics(detailed_results)
    
    # 4. 计算总体指标
    extended_metrics["overall_metrics"] = compute_overall_metrics(detailed_results, summary)
    
    # 保存结果
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(extended_metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ 扩展指标已保存到: {output_path}")
    
    return extended_metrics


def compute_persona_metrics(persona_name: str, results: List[Dict]) -> Dict:
    """计算Persona-specific指标"""
    metrics = {}
    
    # Persona期望行为（基于设计文档）
    persona_expectations = {
        "Busy-Developer": {
            "expected_clarify_rate": 0.2,  # 低澄清率
            "expected_avg_turns": 1.5,      # 少轮次
        },
        "Experienced-Engineer": {
            "expected_clarify_rate": 0.4,   # 中等澄清率
            "expected_avg_turns": 2.0,      # 中等轮次
        },
        "Novice-Learner": {
            "expected_clarify_rate": 0.6,   # 高澄清率
            "expected_avg_turns": 2.5,      # 多轮次
        }
    }
    
    expectations = persona_expectations.get(persona_name, {})
    
    # 实际行为
    total_conversations = len(results)
    total_turns = sum(r.get("total_turns", 0) for r in results)
    total_clarify = sum(r.get("clarify_count", 0) for r in results)
    total_execute = sum(r.get("execute_count", 0) for r in results)
    total_actions = total_clarify + total_execute
    
    actual_clarify_rate = total_clarify / total_actions if total_actions > 0 else 0
    actual_avg_turns = total_turns / total_conversations if total_conversations > 0 else 0
    
    # Persona Alignment Score
    expected_clarify = expectations.get("expected_clarify_rate", 0.5)
    clarify_alignment = 1.0 - abs(actual_clarify_rate - expected_clarify)
    
    expected_turns = expectations.get("expected_avg_turns", 2.0)
    turns_alignment = 1.0 - min(abs(actual_avg_turns - expected_turns) / expected_turns, 1.0)
    
    persona_alignment = (clarify_alignment + turns_alignment) / 2.0
    
    metrics["persona_alignment_score"] = persona_alignment
    metrics["clarify_rate_alignment"] = clarify_alignment
    metrics["turns_alignment"] = turns_alignment
    metrics["actual_clarify_rate"] = actual_clarify_rate
    metrics["expected_clarify_rate"] = expected_clarify
    metrics["actual_avg_turns"] = actual_avg_turns
    metrics["expected_avg_turns"] = expected_turns
    
    return metrics


def compute_user_experience_metrics(results: List[Dict]) -> Dict:
    """计算用户体验指标"""
    metrics = {}
    
    # 1. Time to First Code (TTC)
    ttc_values = []
    for result in results:
        actions = result.get("actions", [])
        if "Execute" in actions:
            first_execute_turn = actions.index("Execute") + 1  # 1-indexed
            ttc_values.append(first_execute_turn)
    
    metrics["time_to_first_code"] = {
        "avg": sum(ttc_values) / len(ttc_values) if ttc_values else 0,
        "min": min(ttc_values) if ttc_values else 0,
        "max": max(ttc_values) if ttc_values else 0,
        "median": sorted(ttc_values)[len(ttc_values)//2] if ttc_values else 0,
    }
    
    # 2. Efficiency Score (成功任务的平均轮次)
    successful_results = [r for r in results if r.get("task_completed", False)]
    if successful_results:
        successful_turns = [r.get("total_turns", 0) for r in successful_results]
        metrics["efficiency_score"] = {
            "avg_turns_for_success": sum(successful_turns) / len(successful_turns),
            "min_turns_for_success": min(successful_turns) if successful_turns else 0,
            "max_turns_for_success": max(successful_turns) if successful_turns else 0,
        }
    else:
        metrics["efficiency_score"] = {
            "avg_turns_for_success": 0,
            "min_turns_for_success": 0,
            "max_turns_for_success": 0,
        }
    
    # 3. Over-clarification Rate
    over_clarify_count = 0
    for result in results:
        clarify_count = result.get("clarify_count", 0)
        task_completed = result.get("task_completed", False)
        if clarify_count > 1 and not task_completed:
            over_clarify_count += 1
    
    metrics["over_clarification_rate"] = over_clarify_count / len(results) if results else 0
    metrics["over_clarification_count"] = over_clarify_count
    
    # 4. Under-clarification Rate
    under_clarify_count = 0
    for result in results:
        clarify_count = result.get("clarify_count", 0)
        task_completed = result.get("task_completed", False)
        if clarify_count == 0 and not task_completed:
            under_clarify_count += 1
    
    metrics["under_clarification_rate"] = under_clarify_count / len(results) if results else 0
    metrics["under_clarification_count"] = under_clarify_count
    
    # 5. First Attempt Success Rate
    first_attempt_success = 0
    first_attempt_total = 0
    for result in results:
        actions = result.get("actions", [])
        if actions and actions[0] == "Execute":
            first_attempt_total += 1
            if result.get("task_completed", False):
                first_attempt_success += 1
    
    metrics["first_attempt_success_rate"] = (
        first_attempt_success / first_attempt_total if first_attempt_total > 0 else 0
    )
    metrics["first_attempt_success_count"] = first_attempt_success
    metrics["first_attempt_total"] = first_attempt_total
    
    # 6. Dialogue Efficiency
    task_success_count = sum(1 for r in results if r.get("task_completed", False))
    task_success_rate = task_success_count / len(results) if results else 0
    avg_turns = sum(r.get("total_turns", 0) for r in results) / len(results) if results else 0
    
    metrics["dialogue_efficiency"] = (
        task_success_rate / avg_turns if avg_turns > 0 else 0
    )
    
    # 7. Conversation Satisfaction Score (综合指标)
    # 权重: task_success=0.4, efficiency=0.2, over_clarify=0.15, under_clarify=0.15, interruption=0.1
    # 注意: 这里没有user_interruption_rate，因为评估中没有这个数据
    satisfaction = (
        0.4 * task_success_rate +
        0.2 * (1.0 - min(avg_turns / 5.0, 1.0)) +  # 假设max_turns=5
        0.15 * (1.0 - metrics["over_clarification_rate"]) +
        0.15 * (1.0 - metrics["under_clarification_rate"])
    )
    metrics["conversation_satisfaction_score"] = satisfaction
    
    return metrics


def compute_code_quality_metrics(results: List[Dict]) -> Dict:
    """计算代码质量指标"""
    metrics = {}
    
    # 提取所有Execute动作的task_score
    task_scores = []
    for result in results:
        task_score = result.get("task_score")
        if task_score is not None:
            task_scores.append(task_score)
    
    if task_scores:
        metrics["avg_task_score"] = sum(task_scores) / len(task_scores)
        metrics["min_task_score"] = min(task_scores)
        metrics["max_task_score"] = max(task_scores)
        metrics["median_task_score"] = sorted(task_scores)[len(task_scores)//2]
        
        # 分布统计
        perfect_scores = sum(1 for s in task_scores if s >= 1.0)
        high_scores = sum(1 for s in task_scores if s >= 0.7)
        medium_scores = sum(1 for s in task_scores if 0.3 <= s < 0.7)
        low_scores = sum(1 for s in task_scores if s < 0.3)
        
        metrics["score_distribution"] = {
            "perfect (>=1.0)": perfect_scores,
            "high (>=0.7)": high_scores,
            "medium (0.3-0.7)": medium_scores,
            "low (<0.3)": low_scores,
        }
        metrics["score_distribution_percent"] = {
            "perfect (>=1.0)": perfect_scores / len(task_scores) * 100,
            "high (>=0.7)": high_scores / len(task_scores) * 100,
            "medium (0.3-0.7)": medium_scores / len(task_scores) * 100,
            "low (<0.3)": low_scores / len(task_scores) * 100,
        }
    else:
        metrics["avg_task_score"] = 0
        metrics["score_distribution"] = {}
    
    return metrics


def compute_overall_metrics(results: List[Dict], summary: Dict) -> Dict:
    """计算总体指标"""
    metrics = {}
    
    # 从summary中提取已有指标
    if "Busy-Developer" in summary:
        bd_stats = summary["Busy-Developer"]
        metrics["busy_developer"] = {
            "task_success_rate": bd_stats.get("task_success_rate", 0),
            "avg_turns": bd_stats.get("avg_turns_per_conversation", 0),
            "clarify_rate": bd_stats.get("clarify_rate", 0),
        }
    
    if "Experienced-Engineer" in summary:
        ee_stats = summary["Experienced-Engineer"]
        metrics["experienced_engineer"] = {
            "task_success_rate": ee_stats.get("task_success_rate", 0),
            "avg_turns": ee_stats.get("avg_turns_per_conversation", 0),
            "clarify_rate": ee_stats.get("clarify_rate", 0),
        }
    
    if "Novice-Learner" in summary:
        nl_stats = summary["Novice-Learner"]
        metrics["novice_learner"] = {
            "task_success_rate": nl_stats.get("task_success_rate", 0),
            "avg_turns": nl_stats.get("avg_turns_per_conversation", 0),
            "clarify_rate": nl_stats.get("clarify_rate", 0),
        }
    
    # 总体统计
    total_conversations = len(results)
    total_success = sum(1 for r in results if r.get("task_completed", False))
    
    metrics["overall"] = {
        "total_conversations": total_conversations,
        "total_success": total_success,
        "overall_task_success_rate": total_success / total_conversations * 100 if total_conversations > 0 else 0,
    }
    
    return metrics


def print_extended_metrics(extended_metrics: Dict):
    """打印扩展指标"""
    print("\n" + "=" * 80)
    print("📊 扩展评估指标")
    print("=" * 80)
    
    # 1. Persona Metrics
    print("\n1️⃣  Persona一致性指标")
    print("-" * 80)
    for persona, metrics in extended_metrics["persona_metrics"].items():
        print(f"\n{persona}:")
        print(f"  Persona Alignment Score: {metrics['persona_alignment_score']:.3f}")
        print(f"  Clarify Rate Alignment: {metrics['clarify_rate_alignment']:.3f}")
        print(f"  Turns Alignment: {metrics['turns_alignment']:.3f}")
        print(f"  实际 Clarify Rate: {metrics['actual_clarify_rate']:.1%} (期望: {metrics['expected_clarify_rate']:.1%})")
        print(f"  实际平均轮次: {metrics['actual_avg_turns']:.2f} (期望: {metrics['expected_avg_turns']:.2f})")
    
    # 2. User Experience Metrics
    print("\n2️⃣  用户体验指标")
    print("-" * 80)
    ux = extended_metrics["user_experience_metrics"]
    print(f"Time to First Code (TTC):")
    print(f"  平均: {ux['time_to_first_code']['avg']:.2f} 轮")
    print(f"  中位数: {ux['time_to_first_code']['median']:.2f} 轮")
    print(f"  范围: {ux['time_to_first_code']['min']:.0f} - {ux['time_to_first_code']['max']:.0f} 轮")
    
    print(f"\nEfficiency Score:")
    print(f"  成功任务平均轮次: {ux['efficiency_score']['avg_turns_for_success']:.2f}")
    
    print(f"\nClarification策略:")
    print(f"  过度澄清率: {ux['over_clarification_rate']:.1%} ({ux['over_clarification_count']} 个)")
    print(f"  澄清不足率: {ux['under_clarification_rate']:.1%} ({ux['under_clarification_count']} 个)")
    
    print(f"\nFirst Attempt Success Rate: {ux['first_attempt_success_rate']:.1%} ({ux['first_attempt_success_count']}/{ux['first_attempt_total']})")
    
    print(f"\nDialogue Efficiency: {ux['dialogue_efficiency']:.3f}")
    print(f"Conversation Satisfaction Score: {ux['conversation_satisfaction_score']:.3f}")
    
    # 3. Code Quality Metrics
    print("\n3️⃣  代码质量指标")
    print("-" * 80)
    cq = extended_metrics["code_quality_metrics"]
    if cq.get("avg_task_score") is not None:
        print(f"平均Task Score: {cq['avg_task_score']:.3f}")
        print(f"中位数Task Score: {cq['median_task_score']:.3f}")
        print(f"范围: {cq['min_task_score']:.3f} - {cq['max_task_score']:.3f}")
        
        if "score_distribution_percent" in cq:
            print(f"\nScore分布:")
            for range_name, percent in cq["score_distribution_percent"].items():
                count = cq["score_distribution"][range_name]
                print(f"  {range_name}: {percent:.1f}% ({count} 个)")
    
    # 4. Overall Metrics
    print("\n4️⃣  总体指标")
    print("-" * 80)
    overall = extended_metrics["overall_metrics"]
    print(f"总对话数: {overall['overall']['total_conversations']}")
    print(f"总成功数: {overall['overall']['total_success']}")
    print(f"总体Task Success Rate: {overall['overall']['overall_task_success_rate']:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="计算扩展评估指标")
    parser.add_argument("--eval_results", type=str, required=True,
                       help="评估结果JSON文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    extended_metrics = compute_extended_metrics(args.eval_results, args.output)
    print_extended_metrics(extended_metrics)
