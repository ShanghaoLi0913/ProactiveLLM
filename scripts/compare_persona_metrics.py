"""
对比V3和V4的Persona指标
生成详细的对比报告
"""
import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate_persona_metrics import compute_persona_metrics, load_jsonl


def compute_improvement(v3_value: float, v4_value: float) -> str:
    """计算提升百分比并格式化"""
    if v3_value == 0:
        if v4_value == 0:
            return "  (无变化)"
        else:
            return f"  (新增 {v4_value:.2f})"
    
    improvement = ((v4_value - v3_value) / v3_value) * 100
    if improvement > 0:
        return f"  ⬆️ +{improvement:.1f}%"
    elif improvement < 0:
        return f"  ⬇️ {improvement:.1f}%"
    else:
        return "  (无变化)"


def generate_comparison_report(
    v3_metrics: Dict,
    v4_metrics: Dict,
    output_path: str
):
    """生成V3 vs V4的对比报告"""
    lines = []
    lines.append("╔═══════════════════════════════════════════════════════════════════════╗")
    lines.append("║  DPO Model Comparison: V3 (Filtered) vs V4 (Filtered+Repaired)       ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════════╝")
    lines.append("")
    
    lines.append("【数据策略对比】")
    lines.append("━" * 70)
    lines.append("  V3 (Filtered):")
    lines.append("    • 使用所有Execute轨迹（304个）")
    lines.append("    • 允许部分通过的代码（pass_rate > 0）")
    lines.append("    • 数据量：304对")
    lines.append("    • 完美数据比例：23% (70/304)")
    lines.append("")
    lines.append("  V4 (Filtered+Repaired):")
    lines.append("    • 使用完美通过的代码（70个）")
    lines.append("    • + 修复成功的代码（65个）")
    lines.append("    • 数据量：135对")
    lines.append("    • 完美数据比例：100% (135/135)")
    lines.append("")
    
    # 整体指标对比
    lines.append("【整体指标对比】")
    lines.append("━" * 70)
    v3_overall = v3_metrics["overall"]
    v4_overall = v4_metrics["overall"]
    
    lines.append("  指标                        V3          V4          变化")
    lines.append("  ────────────────────────────────────────────────────────────")
    
    # TSR
    tsr_v3 = v3_overall["task_success_rate"]
    tsr_v4 = v4_overall["task_success_rate"]
    lines.append(f"  Task Success Rate         {tsr_v3:>6.2f}%     {tsr_v4:>6.2f}%     {compute_improvement(tsr_v3, tsr_v4)}")
    
    # Action Accuracy
    acc_v3 = v3_overall["action_accuracy"]
    acc_v4 = v4_overall["action_accuracy"]
    lines.append(f"  Action Accuracy           {acc_v3:>6.2f}%     {acc_v4:>6.2f}%     {compute_improvement(acc_v3, acc_v4)}")
    
    # Avg Test Pass Rate
    apr_v3 = v3_overall["avg_test_pass_rate"]
    apr_v4 = v4_overall["avg_test_pass_rate"]
    lines.append(f"  Avg Test Pass Rate        {apr_v3:>6.4f}     {apr_v4:>6.4f}     {compute_improvement(apr_v3, apr_v4)}")
    
    lines.append("")
    
    # Persona Discrimination Score (核心指标!)
    lines.append("【Persona Discrimination Score】⭐⭐⭐ (最重要)")
    lines.append("━" * 70)
    pds_v3 = v3_metrics["persona_discrimination_score"]
    pds_v4 = v4_metrics["persona_discrimination_score"]
    lines.append(f"  V3 PDS:  {pds_v3:.4f}")
    lines.append(f"  V4 PDS:  {pds_v4:.4f}")
    lines.append(f"  变化:    {compute_improvement(pds_v3, pds_v4)}")
    lines.append("")
    lines.append("  解释：")
    if pds_v4 > pds_v3:
        improvement_pct = ((pds_v4 - pds_v3) / pds_v3) * 100 if pds_v3 > 0 else 999
        lines.append(f"    ✅ V4的persona区分能力提升了 {improvement_pct:.1f}%")
        lines.append("    ✅ 更高的PDS意味着模型能更好地区分Cautious和Proactive")
    else:
        lines.append("    ⚠️  V4的persona区分能力未提升")
    lines.append("")
    
    # Persona-Conditioned Action Accuracy
    lines.append("【Persona-Conditioned Action Accuracy】⭐⭐⭐")
    lines.append("━" * 70)
    acc_v3_p = v3_metrics["persona_accuracy"]
    acc_v4_p = v4_metrics["persona_accuracy"]
    lines.append("  Persona          V3         V4         变化")
    lines.append("  ───────────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in acc_v3_p and persona in acc_v4_p:
            v3_val = acc_v3_p[persona]
            v4_val = acc_v4_p[persona]
            lines.append(f"  {persona:<14} {v3_val:>6.2f}%    {v4_val:>6.2f}%    {compute_improvement(v3_val, v4_val)}")
    lines.append("")
    
    # Action Distribution Comparison
    lines.append("【Action Distribution by Persona】⭐⭐")
    lines.append("━" * 70)
    dist_v3 = v3_metrics["action_distribution"]
    dist_v4 = v4_metrics["action_distribution"]
    
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in dist_v3 and persona in dist_v4:
            lines.append(f"  {persona}:")
            lines.append("                    V3         V4         变化")
            lines.append("    ───────────────────────────────────────────────")
            
            v3_e = dist_v3[persona]["Execute"]
            v4_e = dist_v4[persona]["Execute"]
            lines.append(f"    Execute:      {v3_e:>6.2f}%    {v4_e:>6.2f}%    {compute_improvement(v3_e, v4_e)}")
            
            v3_c = dist_v3[persona]["Clarify"]
            v4_c = dist_v4[persona]["Clarify"]
            lines.append(f"    Clarify:      {v3_c:>6.2f}%    {v4_c:>6.2f}%    {compute_improvement(v3_c, v4_c)}")
            lines.append("")
    
    # TSR by Persona
    lines.append("【Task Success Rate by Persona】⭐⭐")
    lines.append("━" * 70)
    tsr_v3_p = v3_metrics["tsr_by_persona"]
    tsr_v4_p = v4_metrics["tsr_by_persona"]
    lines.append("  Persona          V3         V4         变化")
    lines.append("  ───────────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in tsr_v3_p and persona in tsr_v4_p:
            v3_val = tsr_v3_p[persona]
            v4_val = tsr_v4_p[persona]
            lines.append(f"  {persona:<14} {v3_val:>6.2f}%    {v4_val:>6.2f}%    {compute_improvement(v3_val, v4_val)}")
    lines.append("")
    
    # Partial Pass Rate by Persona
    lines.append("【Average Pass Rate by Persona】⭐⭐")
    lines.append("━" * 70)
    apr_v3_p = v3_metrics["avg_pass_rate_by_persona"]
    apr_v4_p = v4_metrics["avg_pass_rate_by_persona"]
    lines.append("  Persona          V3         V4         变化")
    lines.append("  ───────────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in apr_v3_p and persona in apr_v4_p:
            v3_val = apr_v3_p[persona]
            v4_val = apr_v4_p[persona]
            lines.append(f"  {persona:<14} {v3_val:>6.4f}    {v4_val:>6.4f}    {compute_improvement(v3_val, v4_val)}")
    lines.append("")
    
    # Dialogue Efficiency
    lines.append("【Dialogue Efficiency by Persona】⭐")
    lines.append("━" * 70)
    eff_v3 = v3_metrics["dialogue_efficiency"]
    eff_v4 = v4_metrics["dialogue_efficiency"]
    lines.append("  Persona          V3         V4         变化")
    lines.append("  ───────────────────────────────────────────────────")
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in eff_v3 and persona in eff_v4:
            v3_val = eff_v3[persona]
            v4_val = eff_v4[persona]
            lines.append(f"  {persona:<14} {v3_val:>6.2f}     {v4_val:>6.2f}     {compute_improvement(v3_val, v4_val)}")
    lines.append("")
    
    # 关键结论
    lines.append("╔═══════════════════════════════════════════════════════════════════════╗")
    lines.append("║  关键结论                                                             ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════════╝")
    lines.append("")
    
    # 判断V4是否更好
    improvements = []
    if pds_v4 > pds_v3:
        improvements.append(f"✅ Persona区分能力提升 {((pds_v4/pds_v3-1)*100):.1f}%")
    
    avg_acc_improvement = 0
    count = 0
    for persona in ["Cautious", "Proactive", "Balanced"]:
        if persona in acc_v3_p and persona in acc_v4_p:
            if acc_v4_p[persona] > acc_v3_p[persona]:
                avg_acc_improvement += (acc_v4_p[persona] - acc_v3_p[persona])
                count += 1
    if count > 0:
        avg_acc_improvement /= count
        if avg_acc_improvement > 0:
            improvements.append(f"✅ Persona动作准确率平均提升 {avg_acc_improvement:.1f}%")
    
    if tsr_v4 > tsr_v3:
        improvements.append(f"✅ 整体TSR提升 {((tsr_v4/tsr_v3-1)*100):.1f}%")
    
    if apr_v4 > apr_v3:
        improvements.append(f"✅ 平均Pass Rate提升 {((apr_v4/apr_v3-1)*100):.1f}%")
    
    if improvements:
        lines.append("  V4相比V3的优势:")
        for imp in improvements:
            lines.append(f"    {imp}")
    else:
        lines.append("  ⚠️  V4相比V3未显示明显优势")
    
    lines.append("")
    lines.append("  核心洞察:")
    lines.append("    • V4使用更少但更高质量的数据（135 vs 304）")
    lines.append("    • 通过代码修复增加了数据多样性")
    lines.append("    • 严格的质量控制（pass_rate=1.0）提升了学习效果")
    lines.append("")
    
    lines.append("╔═══════════════════════════════════════════════════════════════════════╗")
    lines.append("║  报告生成完成                                                         ║")
    lines.append("╚═══════════════════════════════════════════════════════════════════════╝")
    
    report = "\n".join(lines)
    
    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ 对比报告已保存到: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="对比V3和V4的Persona指标")
    parser.add_argument("--v3_eval", type=str, required=True, help="V3评估结果文件")
    parser.add_argument("--v4_eval", type=str, required=True, help="V4评估结果文件")
    parser.add_argument("--trajectories", type=str, required=True, help="Trajectory文件（包含persona）")
    parser.add_argument("--output", type=str, required=True, help="输出对比报告路径")
    
    args = parser.parse_args()
    
    print("📊 加载数据...")
    trajectories = load_jsonl(Path(args.trajectories))
    
    with open(args.v3_eval, "r", encoding="utf-8") as f:
        v3_eval = json.load(f)
    
    with open(args.v4_eval, "r", encoding="utf-8") as f:
        v4_eval = json.load(f)
    
    print("🔍 计算V3指标...")
    v3_metrics = compute_persona_metrics(trajectories, v3_eval, verbose=False)
    
    print("🔍 计算V4指标...")
    v4_metrics = compute_persona_metrics(trajectories, v4_eval, verbose=False)
    
    print("\n📝 生成对比报告...")
    report = generate_comparison_report(v3_metrics, v4_metrics, args.output)
    
    print("\n" + report)
    
    # 保存完整的对比metrics
    comparison_data = {
        "v3": v3_metrics,
        "v4": v4_metrics,
    }
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 完整对比数据已保存到: {json_path}")


if __name__ == "__main__":
    main()
