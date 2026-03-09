#!/usr/bin/env python3
"""分析每个persona在三个代码版本下的详细统计"""
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail

def safe_score(code, tests, timeout=5):
    """安全的评估函数"""
    try:
        result = score_code_passfail(code, tests, timeout=timeout, debug=False)
        return result if result is not None else 0.0
    except Exception:
        return 0.0

def analyze_persona_by_version(traj_file: str):
    """分析每个persona在三个代码版本下的表现"""
    
    # 统计结构: persona -> version -> {total, success}
    stats = defaultdict(lambda: {
        'masked': {'total': 0, 'success': 0},
        'assistant': {'total': 0, 'success': 0},
        'teacher': {'total': 0, 'success': 0}
    })
    
    # 读取Execute轨迹
    execute_trajs = []
    with open(traj_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if traj.get('action') == 'Execute':
                    execute_trajs.append(traj)
    
    print(f"📊 找到 {len(execute_trajs)} 个Execute轨迹\n")
    print("开始评估（每20个显示一次进度）...")
    
    for i, traj in enumerate(execute_trajs):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(execute_trajs)} ({(i+1)/len(execute_trajs)*100:.1f}%)")
        
        persona_name = traj.get('persona', {}).get('name', 'unknown') if isinstance(traj.get('persona'), dict) else 'unknown'
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        
        if not tests:
            continue
        
        # 评估masked_code
        if 'masked_code' in traj:
            code = extract_code_from_text(traj['masked_code'])
            if code:
                stats[persona_name]['masked']['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score >= 1.0:
                    stats[persona_name]['masked']['success'] += 1
        
        # 评估assistant_code
        if 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                stats[persona_name]['assistant']['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score >= 1.0:
                    stats[persona_name]['assistant']['success'] += 1
        
        # 评估teacher_code
        if 'teacher_code' in traj:
            code = extract_code_from_text(traj['teacher_code'])
            if code:
                stats[persona_name]['teacher']['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score >= 1.0:
                    stats[persona_name]['teacher']['success'] += 1
    
    print("\n" + "=" * 100)
    print("Persona × 代码版本 详细对比分析")
    print("=" * 100)
    
    # 转换为字典以便JSON序列化
    result = {}
    
    personas = sorted(stats.keys())
    versions = ['masked', 'assistant', 'teacher']
    version_names = {
        'masked': 'masked_code',
        'assistant': 'assistant_code',
        'teacher': 'teacher_code'
    }
    
    print("\n1. 每个Persona在三个版本下的Success Rate")
    print("-" * 100)
    print(f"{'Persona':<25} {'masked_code':<15} {'assistant_code':<15} {'teacher_code':<15} {'Clarification Gain':<20} {'Full Info Gain':<20}")
    print("-" * 100)
    
    for persona in personas:
        persona_stats = stats[persona]
        result[persona] = {}
        
        rates = {}
        for version in versions:
            v_stats = persona_stats[version]
            if v_stats['total'] > 0:
                rate = v_stats['success'] / v_stats['total'] * 100
                rates[version] = rate
                result[persona][version] = {
                    'total': v_stats['total'],
                    'success': v_stats['success'],
                    'rate': rate
                }
            else:
                rates[version] = 0.0
                result[persona][version] = {'total': 0, 'success': 0, 'rate': 0.0}
        
        clarification_gain = rates['assistant'] - rates['masked']
        full_info_gain = rates['teacher'] - rates['masked']
        result[persona]['clarification_gain'] = clarification_gain
        result[persona]['full_info_gain'] = full_info_gain
        
        print(f"{persona:<25} {rates['masked']:>6.1f}% ({persona_stats['masked']['success']:>2d}/{persona_stats['masked']['total']:<2d}) "
              f"{rates['assistant']:>6.1f}% ({persona_stats['assistant']['success']:>2d}/{persona_stats['assistant']['total']:<2d}) "
              f"{rates['teacher']:>6.1f}% ({persona_stats['teacher']['success']:>2d}/{persona_stats['teacher']['total']:<2d}) "
              f"{clarification_gain:>+6.1f}%{'':<13} {full_info_gain:>+6.1f}%")
    
    print("\n2. 每个代码版本在三个Persona下的Success Rate")
    print("-" * 100)
    print(f"{'版本':<20} {'Novice-Learner':<20} {'Busy-Developer':<20} {'Experienced-Engineer':<20} {'最大差异':<15}")
    print("-" * 100)
    
    for version in versions:
        version_rates = {}
        for persona in personas:
            if version in result[persona]:
                version_rates[persona] = result[persona][version]['rate']
            else:
                version_rates[persona] = 0.0
        
        max_rate = max(version_rates.values())
        min_rate = min(version_rates.values())
        max_diff = max_rate - min_rate
        
        print(f"{version_names[version]:<20} "
              f"{version_rates.get('Novice-Learner', 0):>6.1f}%{'':<13} "
              f"{version_rates.get('Busy-Developer', 0):>6.1f}%{'':<13} "
              f"{version_rates.get('Experienced-Engineer', 0):>6.1f}%{'':<13} "
              f"{max_diff:>6.1f}%")
    
    print("\n3. Persona间差异分析")
    print("-" * 100)
    
    # 计算每个版本下persona之间的差异
    for version in versions:
        version_rates = {}
        for persona in personas:
            if version in result[persona]:
                version_rates[persona] = result[persona][version]['rate']
        
        if len(version_rates) >= 2:
            rates_list = list(version_rates.values())
            max_rate = max(rates_list)
            min_rate = min(rates_list)
            avg_rate = sum(rates_list) / len(rates_list)
            max_diff = max_rate - min_rate
            
            print(f"\n{version_names[version]}:")
            print(f"  最高: {max_rate:.1f}%, 最低: {min_rate:.1f}%, 平均: {avg_rate:.1f}%, 差异: {max_diff:.1f}%")
            
            # 找出最高和最低的persona
            max_persona = max(version_rates.items(), key=lambda x: x[1])[0]
            min_persona = min(version_rates.items(), key=lambda x: x[1])[0]
            print(f"  最高: {max_persona} ({max_rate:.1f}%)")
            print(f"  最低: {min_persona} ({min_rate:.1f}%)")
    
    print("\n4. 版本间差异分析（按Persona）")
    print("-" * 100)
    
    for persona in personas:
        if persona in result:
            masked_rate = result[persona]['masked']['rate']
            assistant_rate = result[persona]['assistant']['rate']
            teacher_rate = result[persona]['teacher']['rate']
            
            clarification_gain = result[persona]['clarification_gain']
            full_info_gain = result[persona]['full_info_gain']
            gap = teacher_rate - assistant_rate
            
            print(f"\n{persona}:")
            print(f"  masked_code → assistant_code: {masked_rate:.1f}% → {assistant_rate:.1f}% (Gain: {clarification_gain:+.1f}%)")
            print(f"  masked_code → teacher_code: {masked_rate:.1f}% → {teacher_rate:.1f}% (Gain: {full_info_gain:+.1f}%)")
            print(f"  assistant_code → teacher_code: {assistant_rate:.1f}% → {teacher_rate:.1f}% (Gap: {gap:+.1f}%)")
    
    print("\n" + "=" * 100)
    print("✅ 分析完成")
    print("=" * 100)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="分析每个persona在三个代码版本下的详细统计")
    parser.add_argument('--trajectories', type=str, required=True,
                       help='轨迹文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON结果文件路径（可选）')
    
    args = parser.parse_args()
    
    result = analyze_persona_by_version(args.trajectories)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n📄 详细结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
