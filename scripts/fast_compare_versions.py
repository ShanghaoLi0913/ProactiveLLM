#!/usr/bin/env python3
"""快速对比三个代码版本（使用更短的超时和进度显示）"""
import json
import sys
import signal
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# 禁用输出缓冲，确保实时显示
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate_dpo_model import extract_code_from_text, score_code_passfail

# 设置超时处理
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("评估超时")

def safe_score(code, tests, timeout=5):
    """安全的评估函数，使用更短的超时，添加额外的保护"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"评估超时（>{timeout}秒）")
    
    # 设置信号超时（仅Unix系统）
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout + 1)  # 比subprocess timeout多1秒
    
    try:
        result = score_code_passfail(code, tests, timeout=timeout, debug=False)
        return result
    except (TimeoutError, subprocess.TimeoutExpired) as e:
        # 超时是正常的，返回0.0
        return 0.0
    except Exception as e:
        # 其他异常也返回None，避免卡住
        return None
    finally:
        # 恢复信号处理
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def main():
    parser = argparse.ArgumentParser(description="快速对比三个代码版本")
    parser.add_argument('--trajectories', type=str, default=None,
                       help='轨迹文件路径（如果不指定，自动查找最新的traj_30states*.jsonl）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON结果文件路径（可选）')
    args = parser.parse_args()
    
    if args.trajectories:
        latest_file = Path(args.trajectories)
        if not latest_file.exists():
            print(f"❌ 文件不存在: {latest_file}")
            return
    else:
        traj_files = sorted(Path("data/logs").glob("traj_30states*.jsonl"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not traj_files:
            print("❌ 未找到轨迹文件")
            return
        
        latest_file = traj_files[0]
    
    print(f"📁 文件: {latest_file}\n")
    
    # 读取Execute轨迹
    execute_trajs = []
    with open(latest_file, 'r') as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if traj.get('action') == 'Execute':
                    execute_trajs.append(traj)
    
    print(f"📊 找到 {len(execute_trajs)} 个Execute轨迹")
    print(f"⚡ 使用快速评估模式（timeout=5秒）\n")
    
    version_stats = {
        'masked': {'total': 0, 'success': 0},
        'assistant': {'total': 0, 'success': 0},
        'teacher': {'total': 0, 'success': 0}
    }
    
    persona_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    
    print("开始评估（每10个显示一次进度）...")
    print(f"总共需要评估: {len(execute_trajs)} 个轨迹 × 3 个版本 = {len(execute_trajs) * 3} 次评估", flush=True)
    for i, traj in enumerate(execute_trajs):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(execute_trajs)} ({(i+1)/len(execute_trajs)*100:.1f}%)", flush=True)
        
        persona_name = traj.get('persona', {}).get('name', 'unknown') if isinstance(traj.get('persona'), dict) else 'unknown'
        state = traj.get('state', {})
        tests = state.get('convcodeworld_tests', '')
        
        if not tests:
            continue
        
        # 评估masked_code
        if 'masked_code' in traj:
            code = extract_code_from_text(traj['masked_code'])
            if code:
                version_stats['masked']['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score is not None and task_score >= 1.0:
                    version_stats['masked']['success'] += 1
        
        # 评估assistant_code
        if 'assistant_code' in traj:
            code = extract_code_from_text(traj['assistant_code'])
            if code:
                version_stats['assistant']['total'] += 1
                persona_stats[persona_name]['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score is not None and task_score >= 1.0:
                    version_stats['assistant']['success'] += 1
                    persona_stats[persona_name]['success'] += 1
        
        # 评估teacher_code
        if 'teacher_code' in traj:
            code = extract_code_from_text(traj['teacher_code'])
            if code:
                version_stats['teacher']['total'] += 1
                task_score = safe_score(code, tests, timeout=5)
                if task_score is not None and task_score >= 1.0:
                    version_stats['teacher']['success'] += 1
    
    print("\n" + "=" * 80)
    print("代码质量对比报告（完整评估）")
    print("=" * 80)
    print()
    
    print("1. Persona之间Task Success Rate（assistant_code）")
    print("-" * 80)
    total_success = 0
    total_evaluated = 0
    for persona in sorted(persona_stats.keys()):
        stats = persona_stats[persona]
        if stats['total'] > 0:
            rate = stats['success'] / stats['total'] * 100
            print(f"  {persona:25s}: {stats['success']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
            total_success += stats['success']
            total_evaluated += stats['total']
    
    if total_evaluated > 0:
        overall_rate = total_success / total_evaluated * 100
        print("-" * 80)
        print(f"  {'总体':25s}: {total_success:3d}/{total_evaluated:3d} ({overall_rate:5.1f}%)")
    
    print("\n2. 三个代码版本对比")
    print("-" * 80)
    for version in ['masked', 'assistant', 'teacher']:
        stats = version_stats[version]
        if stats['total'] > 0:
            rate = stats['success'] / stats['total'] * 100
            version_name = {
                'masked': 'masked_code (仅masked query)',
                'assistant': 'assistant_code (masked + clarifications)',
                'teacher': 'teacher_code (full query)'
            }[version]
            print(f"  {version_name:35s}: {stats['success']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
    
    # 计算差异
    print("\n3. 差异分析")
    print("-" * 80)
    masked_rate = version_stats['masked']['success'] / version_stats['masked']['total'] * 100 if version_stats['masked']['total'] > 0 else 0
    assistant_rate = version_stats['assistant']['success'] / version_stats['assistant']['total'] * 100 if version_stats['assistant']['total'] > 0 else 0
    teacher_rate = version_stats['teacher']['success'] / version_stats['teacher']['total'] * 100 if version_stats['teacher']['total'] > 0 else 0
    
    print(f"  Clarification Gain (assistant vs masked): {assistant_rate - masked_rate:+.1f}%")
    print(f"  Full Info Gain (teacher vs masked): {teacher_rate - masked_rate:+.1f}%")
    print(f"  Gap (teacher vs assistant): {teacher_rate - assistant_rate:+.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ 评估完成")
    print("=" * 80)
    
    # 保存结果到JSON文件（如果指定了输出路径）
    if args.output:
        results = {
            'file': str(latest_file),
            'persona_stats': {k: {'total': v['total'], 'success': v['success'], 
                                 'rate': v['success']/v['total']*100 if v['total'] > 0 else 0}
                            for k, v in persona_stats.items()},
            'version_stats': {k: {'total': v['total'], 'success': v['success'],
                                'rate': v['success']/v['total']*100 if v['total'] > 0 else 0}
                            for k, v in version_stats.items()},
            'gains': {
                'clarification_gain': assistant_rate - masked_rate,
                'full_info_gain': teacher_rate - masked_rate,
                'gap': teacher_rate - assistant_rate
            }
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📄 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
