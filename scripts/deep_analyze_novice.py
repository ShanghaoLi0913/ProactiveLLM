#!/usr/bin/env python3
"""深入分析Novice-Learner表现异常"""
import json
import sys
from pathlib import Path
import glob
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.reconstruct_state import reconstruct_state_for_execute, extract_user_answers_from_query
from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail

def count_tokens(text):
    """简单统计token数量（按空格分割）"""
    return len(text.split())

def analyze_actionable_specs(clarified_requirements):
    """分析clarified requirements中的actionable specs"""
    if not clarified_requirements:
        return {
            'has_action_verbs': False,
            'has_specific_conditions': False,
            'has_format_strings': False,
            'has_standard_phrases': False,
            'actionable_density': 0.0
        }
    
    text_lower = clarified_requirements.lower()
    
    # 检查行为词
    action_verbs = ['return', 'raise', 'output', 'should be', 'must', 'handle', 'check', 'validate']
    has_action_verbs = any(verb in text_lower for verb in action_verbs)
    
    # 检查具体条件
    specific_conditions = ['if empty', 'if null', 'if negative', 'if zero', 'when', 'empty input', 'null value']
    has_specific_conditions = any(cond in text_lower for cond in specific_conditions)
    
    # 检查格式字符串
    format_strings = ['counter', 'dict', 'list', 'tuple', 'json', 'float', 'int', 'string']
    has_format_strings = any(fmt in text_lower for fmt in format_strings)
    
    # 检查"标准"等无信息片段
    standard_phrases = ['standard', 'general', 'normal', 'usual', 'typical']
    has_standard_phrases = any(phrase in text_lower for phrase in standard_phrases)
    
    # 计算actionable density（actionable词汇数 / 总词汇数）
    actionable_words = sum(1 for word in text_lower.split() if any(keyword in word for keyword in action_verbs + specific_conditions + format_strings))
    total_words = len(text_lower.split())
    actionable_density = actionable_words / total_words if total_words > 0 else 0.0
    
    return {
        'has_action_verbs': has_action_verbs,
        'has_specific_conditions': has_specific_conditions,
        'has_format_strings': has_format_strings,
        'has_standard_phrases': has_standard_phrases,
        'actionable_density': actionable_density
    }

def count_clarify_turns(query):
    """统计clarify轮数"""
    if '[Assistant]:' not in query:
        return 0
    return query.count('[Assistant]:')

def calculate_coverage_ratio(traj, state):
    """计算coverage ratio = revealed_points / masked_points"""
    disclosure_rule = state.get('disclosure_rule', {})
    disclosure_info = disclosure_rule.get('disclosure_info', {})
    
    # 计算masked_points总数
    masked_points = 0
    if 'input_constraints' in disclosure_info:
        edge_cases = disclosure_info['input_constraints'].get('edge_cases', [])
        masked_points += len(edge_cases) if isinstance(edge_cases, list) else 0
    if 'output_format' in disclosure_info:
        masked_points += 1
    if 'validation_rules' in disclosure_info:
        rules = disclosure_info['validation_rules'].get('rules', [])
        masked_points += len(rules) if isinstance(rules, list) else 0
    
    # 计算revealed_points（从reconstruction_coverage）
    coverage = traj.get('reconstruction_coverage', [])
    revealed_points = len(coverage)
    
    if masked_points == 0:
        return 0.0
    
    return revealed_points / masked_points

def main():
    # 读取数据
    files = sorted(glob.glob("data/data/logs/traj_10states_test*.jsonl"), 
                   key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if not files:
        print("❌ 未找到轨迹文件")
        return
    
    latest_file = files[0]
    print("=" * 80)
    print("深入分析Novice-Learner表现异常")
    print("=" * 80)
    print(f"📁 文件: {latest_file}\n")
    
    # 收集所有persona的Execute轨迹
    all_trajs = []
    with open(latest_file, 'r') as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if traj.get('action') == 'Execute':
                    all_trajs.append(traj)
    
    # 按persona分组
    persona_trajs = defaultdict(list)
    for traj in all_trajs:
        persona_name = traj.get('persona', {}).get('name', 'unknown') if isinstance(traj.get('persona'), dict) else 'unknown'
        persona_trajs[persona_name].append(traj)
    
    print("=" * 80)
    print("1. Novice的澄清发生情况统计")
    print("=" * 80)
    print()
    
    novice_trajs = persona_trajs['Novice-Learner']
    print(f"Novice-Learner Execute轨迹总数: {len(novice_trajs)}\n")
    
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
    
    print(f"Execute_after_clarify: {len(execute_after_clarify)} ({len(execute_after_clarify)/len(novice_trajs)*100:.1f}%)")
    print(f"Execute_direct: {len(execute_direct)} ({len(execute_direct)/len(novice_trajs)*100:.1f}%)")
    print()
    
    # 评估两组success
    print("两组success对比:")
    print("-" * 80)
    
    for group_name, group_trajs in [('Execute_after_clarify', execute_after_clarify), ('Execute_direct', execute_direct)]:
        success_count = 0
        total = 0
        
        for traj in group_trajs:
            state = traj.get('state', {})
            tests = state.get('convcodeworld_tests', '')
            
            if tests and 'assistant_code' in traj:
                code = extract_code_from_text(traj['assistant_code'])
                if code:
                    total += 1
                    score = score_code_passfail(code, tests, timeout=5)
                    if score is not None and score >= 1.0:
                        success_count += 1
        
        if total > 0:
            success_rate = success_count / total * 100
            print(f"{group_name:25s}: {success_count}/{total} ({success_rate:.1f}%)")
    
    print()
    print("=" * 80)
    print("2. Novice的Coverage和Answer Clarity分布")
    print("=" * 80)
    print()
    
    coverage_stats = defaultdict(int)
    clarity_stats = {'low': 0, 'mid': 0, 'high': 0}
    
    for traj in execute_after_clarify:
        coverage = traj.get('reconstruction_coverage', [])
        coverage_count = len(coverage)
        coverage_stats[coverage_count] += 1
        
        # 从对话历史中提取answer_clarity（如果有）
        # 这里简化处理，实际应该从meta中提取
    
    print("Reconstruction Coverage分布:")
    for count in sorted(coverage_stats.keys()):
        print(f"  {count} 个类别: {coverage_stats[count]} 个轨迹")
    
    print()
    print("=" * 80)
    print("3. Novice的披露信息规范化程度分析")
    print("=" * 80)
    print()
    
    novice_specs = []
    experienced_specs = []
    
    for traj in execute_after_clarify:
        revealed = traj.get('revealed_requirements', '')
        if revealed and isinstance(revealed, str):
            specs = analyze_actionable_specs(revealed)
            novice_specs.append(specs)
    
    for traj in persona_trajs['Experienced-Engineer']:
        state = traj.get('state', {})
        query = state.get('query', '')
        if '[Assistant]:' in query and '[User]:' in query:
            reconstructed = reconstruct_state_for_execute(state)
            revealed = reconstructed.get('clarified_requirements', '')
            if revealed:
                specs = analyze_actionable_specs(revealed)
                experienced_specs.append(specs)
    
    print("Novice-Learner的Revealed Requirements规范化程度:")
    if novice_specs:
        has_action_verbs = sum(1 for s in novice_specs if s['has_action_verbs'])
        has_specific_conditions = sum(1 for s in novice_specs if s['has_specific_conditions'])
        has_format_strings = sum(1 for s in novice_specs if s['has_format_strings'])
        has_standard_phrases = sum(1 for s in novice_specs if s['has_standard_phrases'])
        avg_density = sum(s['actionable_density'] for s in novice_specs) / len(novice_specs)
        
        print(f"  包含行为词: {has_action_verbs}/{len(novice_specs)} ({has_action_verbs/len(novice_specs)*100:.1f}%)")
        print(f"  包含具体条件: {has_specific_conditions}/{len(novice_specs)} ({has_specific_conditions/len(novice_specs)*100:.1f}%)")
        print(f"  包含格式字符串: {has_format_strings}/{len(novice_specs)} ({has_format_strings/len(novice_specs)*100:.1f}%)")
        print(f"  包含无信息片段: {has_standard_phrases}/{len(novice_specs)} ({has_standard_phrases/len(novice_specs)*100:.1f}%)")
        print(f"  平均Actionable Density: {avg_density:.3f}")
    
    print()
    print("Experienced-Engineer的Revealed Requirements规范化程度:")
    if experienced_specs:
        has_action_verbs = sum(1 for s in experienced_specs if s['has_action_verbs'])
        has_specific_conditions = sum(1 for s in experienced_specs if s['has_specific_conditions'])
        has_format_strings = sum(1 for s in experienced_specs if s['has_format_strings'])
        has_standard_phrases = sum(1 for s in experienced_specs if s['has_standard_phrases'])
        avg_density = sum(s['actionable_density'] for s in experienced_specs) / len(experienced_specs)
        
        print(f"  包含行为词: {has_action_verbs}/{len(experienced_specs)} ({has_action_verbs/len(experienced_specs)*100:.1f}%)")
        print(f"  包含具体条件: {has_specific_conditions}/{len(experienced_specs)} ({has_specific_conditions/len(experienced_specs)*100:.1f}%)")
        print(f"  包含格式字符串: {has_format_strings}/{len(experienced_specs)} ({has_format_strings/len(experienced_specs)*100:.1f}%)")
        print(f"  包含无信息片段: {has_standard_phrases}/{len(experienced_specs)} ({has_standard_phrases/len(experienced_specs)*100:.1f}%)")
        print(f"  平均Actionable Density: {avg_density:.3f}")
    
    print()
    print("=" * 80)
    print("4. Prompt长度和噪声分析")
    print("=" * 80)
    print()
    
    novice_prompts = []
    busy_prompts = []
    experienced_prompts = []
    
    for persona_name, trajs in persona_trajs.items():
        for traj in trajs:
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
                
                prompt_length = count_tokens(prompt)
                clarified_length = count_tokens(clarified_requirements) if clarified_requirements else 0
                
                if persona_name == 'Novice-Learner':
                    novice_prompts.append({'total': prompt_length, 'clarified': clarified_length})
                elif persona_name == 'Busy-Developer':
                    busy_prompts.append({'total': prompt_length, 'clarified': clarified_length})
                elif persona_name == 'Experienced-Engineer':
                    experienced_prompts.append({'total': prompt_length, 'clarified': clarified_length})
    
    print("Prompt长度统计:")
    print("-" * 80)
    for persona_name, prompts in [('Novice-Learner', novice_prompts), ('Busy-Developer', busy_prompts), ('Experienced-Engineer', experienced_prompts)]:
        if prompts:
            avg_total = sum(p['total'] for p in prompts) / len(prompts)
            avg_clarified = sum(p['clarified'] for p in prompts) / len(prompts)
            print(f"{persona_name:25s}: 平均总长度 {avg_total:.1f} tokens, 平均clarified部分 {avg_clarified:.1f} tokens")
    
    print()
    print("=" * 80)
    print("5. Clarify次数分布和Coverage Ratio分析（Novice）")
    print("=" * 80)
    print()
    
    clarify_count_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'coverage_ratios': []})
    
    for traj in execute_after_clarify:
        state = traj.get('state', {})
        query = state.get('query', '')
        tests = state.get('convcodeworld_tests', '')
        
        clarify_count = count_clarify_turns(query)
        coverage_ratio = calculate_coverage_ratio(traj, state)
        
        clarify_count_stats[clarify_count]['total'] += 1
        clarify_count_stats[clarify_count]['coverage_ratios'].append(coverage_ratio)
        
        if tests and 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                score = score_code_passfail(code, tests, timeout=5)
                if score is not None and score >= 1.0:
                    clarify_count_stats[clarify_count]['success'] += 1
    
    print("Novice-Learner: Clarify次数 vs Success vs Coverage Ratio")
    print("-" * 80)
    for clarify_count in sorted(clarify_count_stats.keys()):
        stats = clarify_count_stats[clarify_count]
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_coverage = sum(stats['coverage_ratios']) / len(stats['coverage_ratios']) if stats['coverage_ratios'] else 0
        
        print(f"Clarify {clarify_count} 次:")
        print(f"  轨迹数: {stats['total']}")
        print(f"  Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        print(f"  平均Coverage Ratio: {avg_coverage:.3f}")
        print()
    
    print("=" * 80)
    print("6. Success vs Coverage Ratio详细分析（Novice）")
    print("=" * 80)
    print()
    
    coverage_groups = {
        'low': {'total': 0, 'success': 0},      # < 0.3
        'mid': {'total': 0, 'success': 0},     # 0.3-0.6
        'high': {'total': 0, 'success': 0}     # > 0.6
    }
    
    for traj in execute_after_clarify:
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        coverage_ratio = calculate_coverage_ratio(traj, state)
        
        if coverage_ratio < 0.3:
            group = 'low'
        elif coverage_ratio < 0.6:
            group = 'mid'
        else:
            group = 'high'
        
        coverage_groups[group]['total'] += 1
        
        if tests and 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                score = score_code_passfail(code, tests, timeout=5)
                if score is not None and score >= 1.0:
                    coverage_groups[group]['success'] += 1
    
    print("Success vs Coverage Ratio:")
    print("-" * 80)
    for group_name, group_key in [('Low (<0.3)', 'low'), ('Mid (0.3-0.6)', 'mid'), ('High (>0.6)', 'high')]:
        stats = coverage_groups[group_key]
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total'] * 100
            print(f"{group_name:15s}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    print()
    print("=" * 80)
    print("✅ 分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
