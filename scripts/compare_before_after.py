#!/usr/bin/env python3
"""对比改进前后的效果"""
import json
import sys
from pathlib import Path
import glob
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.reconstruct_state import reconstruct_state_for_execute
from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail

def count_tokens(text):
    return len(text.split())

def count_clarify_turns(query):
    if '[Assistant]:' not in query:
        return 0
    return query.count('[Assistant]:')

def calculate_coverage_ratio(traj, state):
    disclosure_rule = state.get('disclosure_rule', {})
    disclosure_info = disclosure_rule.get('disclosure_info', {})
    
    masked_points = 0
    if 'input_constraints' in disclosure_info:
        edge_cases = disclosure_info['input_constraints'].get('edge_cases', [])
        masked_points += len(edge_cases) if isinstance(edge_cases, list) else 0
    if 'output_format' in disclosure_info:
        masked_points += 1
    if 'validation_rules' in disclosure_info:
        rules = disclosure_info['validation_rules'].get('rules', [])
        masked_points += len(rules) if isinstance(rules, list) else 0
    
    coverage = traj.get('reconstruction_coverage', [])
    revealed_points = len(coverage)
    
    if masked_points == 0:
        return 0.0
    
    return revealed_points / masked_points

def safe_score(code, tests, timeout=5):
    try:
        return score_code_passfail(code, tests, timeout=timeout, debug=False)
    except:
        return None

def analyze_file(file_path, label):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}\n")
    
    execute_trajs = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if traj.get('action') == 'Execute':
                    execute_trajs.append(traj)
    
    print(f"📊 Execute轨迹数: {len(execute_trajs)}\n")
    
    # 按persona分组
    persona_trajs = defaultdict(list)
    for traj in execute_trajs:
        persona_name = traj.get('persona', {}).get('name', 'unknown') if isinstance(traj.get('persona'), dict) else 'unknown'
        persona_trajs[persona_name].append(traj)
    
    # 重点分析Novice-Learner
    novice_trajs = persona_trajs['Novice-Learner']
    
    # 按是否有clarification分组
    execute_after_clarify = []
    execute_direct = []
    
    for traj in novice_trajs:
        state = traj.get('state', {})
        query = state.get('query', '')
        if '[Assistant]:' in query and '[User]:' in query:
            execute_after_clarify.append(traj)
        else:
            execute_direct.append(traj)
    
    print(f"Novice-Learner统计:")
    print(f"  Execute_after_clarify: {len(execute_after_clarify)}")
    print(f"  Execute_direct: {len(execute_direct)}")
    print()
    
    # 评估success
    after_clarify_success = 0
    after_clarify_total = 0
    direct_success = 0
    direct_total = 0
    
    for traj in execute_after_clarify:
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        if tests and 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                after_clarify_total += 1
                score = safe_score(code, tests, timeout=5)
                if score is not None and score >= 1.0:
                    after_clarify_success += 1
    
    for traj in execute_direct:
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        if tests and 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                direct_total += 1
                score = safe_score(code, tests, timeout=5)
                if score is not None and score >= 1.0:
                    direct_success += 1
    
    print(f"Success对比:")
    if after_clarify_total > 0:
        after_rate = after_clarify_success / after_clarify_total * 100
        print(f"  Execute_after_clarify: {after_clarify_success}/{after_clarify_total} ({after_rate:.1f}%)")
    if direct_total > 0:
        direct_rate = direct_success / direct_total * 100
        print(f"  Execute_direct: {direct_success}/{direct_total} ({direct_rate:.1f}%)")
    
    # Coverage Ratio统计
    coverage_ratios = []
    reconstruction_success = 0
    reconstruction_total = 0
    
    for traj in execute_after_clarify:
        state = traj.get('state', {})
        coverage_ratio = calculate_coverage_ratio(traj, state)
        coverage_ratios.append(coverage_ratio)
        
        coverage = traj.get('reconstruction_coverage', [])
        if coverage:
            reconstruction_success += 1
        reconstruction_total += 1
    
    if coverage_ratios:
        avg_coverage = sum(coverage_ratios) / len(coverage_ratios)
        print(f"\nCoverage Ratio统计:")
        print(f"  平均Coverage Ratio: {avg_coverage:.3f}")
        print(f"  Reconstruction成功率: {reconstruction_success}/{reconstruction_total} ({reconstruction_success/reconstruction_total*100:.1f}%)")
    
    # Prompt长度统计
    prompt_lengths = []
    clarified_lengths = []
    
    for traj in execute_after_clarify:
        state = traj.get('state', {})
        query = state.get('query', '')
        if '[Assistant]:' in query and '[User]:' in query:
            reconstructed = reconstruct_state_for_execute(state)
            original_query = reconstructed['original_query']
            clarified_requirements = reconstructed['clarified_requirements']
            
            if clarified_requirements:
                prompt = f"[Task]\n{original_query}\n\n[Clarified Requirements]\n{clarified_requirements}\n\n[Instruction]\nWrite the implementation.\nDo not ask further questions."
            else:
                prompt = f"[Task]\n{original_query}"
            
            prompt_lengths.append(count_tokens(prompt))
            clarified_lengths.append(count_tokens(clarified_requirements) if clarified_requirements else 0)
    
    if prompt_lengths:
        avg_prompt = sum(prompt_lengths) / len(prompt_lengths)
        avg_clarified = sum(clarified_lengths) / len(clarified_lengths) if clarified_lengths else 0
        print(f"\nPrompt长度统计:")
        print(f"  平均总长度: {avg_prompt:.1f} tokens")
        print(f"  平均clarified部分: {avg_clarified:.1f} tokens")
    
    # 三个代码版本对比
    version_stats = {
        'masked': {'total': 0, 'success': 0},
        'assistant': {'total': 0, 'success': 0},
        'teacher': {'total': 0, 'success': 0}
    }
    
    for traj in novice_trajs:
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        if not tests:
            continue
        
        for version in ['masked', 'assistant', 'teacher']:
            code_key = f'{version}_code'
            if code_key in traj:
                code = extract_code_from_text(traj[code_key])
                if code:
                    version_stats[version]['total'] += 1
                    score = safe_score(code, tests, timeout=5)
                    if score is not None and score >= 1.0:
                        version_stats[version]['success'] += 1
    
    print(f"\n三个代码版本对比（Novice-Learner）:")
    for version in ['masked', 'assistant', 'teacher']:
        stats = version_stats[version]
        if stats['total'] > 0:
            rate = stats['success'] / stats['total'] * 100
            version_name = {
                'masked': 'masked_code',
                'assistant': 'assistant_code',
                'teacher': 'teacher_code'
            }[version]
            print(f"  {version_name:15s}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    if version_stats['masked']['total'] > 0 and version_stats['assistant']['total'] > 0:
        masked_rate = version_stats['masked']['success'] / version_stats['masked']['total'] * 100
        assistant_rate = version_stats['assistant']['success'] / version_stats['assistant']['total'] * 100
        gain = assistant_rate - masked_rate
        print(f"\n  Clarification Gain: {gain:+.1f}%")
    
    return {
        'after_clarify_rate': after_clarify_success / after_clarify_total * 100 if after_clarify_total > 0 else 0,
        'direct_rate': direct_success / direct_total * 100 if direct_total > 0 else 0,
        'avg_coverage': avg_coverage if coverage_ratios else 0,
        'reconstruction_rate': reconstruction_success / reconstruction_total * 100 if reconstruction_total > 0 else 0,
        'avg_clarified_tokens': avg_clarified if clarified_lengths else 0,
        'clarification_gain': gain if version_stats['masked']['total'] > 0 and version_stats['assistant']['total'] > 0 else 0
    }

def main():
    print("=" * 80)
    print("改进前后对比分析")
    print("=" * 80)
    
    # 改进前的文件
    before_files = sorted(glob.glob("data/data/logs/traj_10states_test*.jsonl"), 
                         key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    # 改进后的文件
    after_files = sorted(glob.glob("data/data/logs/traj_5states_improved*.jsonl"), 
                        key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if not before_files:
        print("❌ 未找到改进前的文件")
        return
    
    if not after_files:
        print("❌ 未找到改进后的文件")
        return
    
    before_file = before_files[0]
    after_file = after_files[0]
    
    print(f"\n改进前文件: {before_file}")
    print(f"改进后文件: {after_file}\n")
    
    before_stats = analyze_file(before_file, "改进前（10states）")
    after_stats = analyze_file(after_file, "改进后（5states）")
    
    print("\n" + "=" * 80)
    print("改进效果对比")
    print("=" * 80)
    print()
    
    print(f"{'指标':30s} {'改进前':15s} {'改进后':15s} {'变化':15s}")
    print("-" * 80)
    
    metrics = [
        ('Execute_after_clarify Success', 'after_clarify_rate', '%'),
        ('Execute_direct Success', 'direct_rate', '%'),
        ('平均Coverage Ratio', 'avg_coverage', ''),
        ('Reconstruction成功率', 'reconstruction_rate', '%'),
        ('平均Clarified Tokens', 'avg_clarified_tokens', ' tokens'),
        ('Clarification Gain', 'clarification_gain', '%'),
    ]
    
    for metric_name, key, unit in metrics:
        before_val = before_stats.get(key, 0)
        after_val = after_stats.get(key, 0)
        change = after_val - before_val
        
        if unit == '%':
            print(f"{metric_name:30s} {before_val:13.1f}% {after_val:13.1f}% {change:+13.1f}%")
        elif unit == ' tokens':
            print(f"{metric_name:30s} {before_val:13.1f} {after_val:13.1f} {change:+13.1f}")
        else:
            print(f"{metric_name:30s} {before_val:13.3f} {after_val:13.3f} {change:+13.3f}")
    
    print()
    print("=" * 80)
    print("✅ 对比完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
