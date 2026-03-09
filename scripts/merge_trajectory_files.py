#!/usr/bin/env python3
"""
合并多个轨迹文件，用于合并5个states和15个states的数据
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import glob


def load_trajectories(file_path: str) -> List[Dict]:
    """加载轨迹文件"""
    trajectories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traj = json.loads(line)
                    trajectories.append(traj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
    return trajectories


def merge_trajectory_files(input_files: List[str], output_file: str):
    """合并多个轨迹文件"""
    all_trajectories = []
    
    for file_path in input_files:
        print(f"Loading {file_path}...")
        trajectories = load_trajectories(file_path)
        print(f"  Loaded {len(trajectories)} trajectories")
        all_trajectories.extend(trajectories)
    
    print(f"\nTotal trajectories: {len(all_trajectories)}")
    
    # 写入合并后的文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
    
    print(f"✅ Merged trajectories saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return len(all_trajectories)


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory files")
    parser.add_argument('--input_files', nargs='+', required=True,
                       help='Input trajectory files to merge (can use glob patterns)')
    parser.add_argument('--output', required=True,
                       help='Output merged file path')
    
    args = parser.parse_args()
    
    # 展开glob模式
    input_files = []
    for pattern in args.input_files:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            # 如果不是glob模式，直接使用
            if Path(pattern).exists():
                input_files.append(pattern)
            else:
                print(f"Warning: File/pattern not found: {pattern}")
    
    if not input_files:
        print("Error: No input files found!")
        return
    
    # 去重并排序
    input_files = sorted(set(input_files))
    
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f}")
    
    merge_trajectory_files(input_files, args.output)


if __name__ == '__main__':
    main()
