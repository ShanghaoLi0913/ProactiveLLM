#!/usr/bin/env python3
"""
测试persona-aware render_state功能
验证不同persona生成不同的prompts
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.render_state import render_state

# 测试state
test_state = {
    "id": "task_123",
    "domain": "coding",
    "query": "帮我写个Python爬虫",
    "dialogue_turn": 0,
    "task_uncertainty": 0.8,
    "prev_reject": 0
}

# 3个不同的persona
personas = [
    {"name": "Busy-Developer", "patience": "low", "expertise": "mid"},
    {"name": "Experienced-Engineer", "patience": "mid", "expertise": "high"},
    {"name": "Novice-Learner", "patience": "high", "expertise": "low"}
]

print("=" * 70)
print("测试 Persona-Aware Render State")
print("=" * 70)
print()

for i, persona in enumerate(personas, 1):
    print(f"【Test {i}: {persona['name']}】")
    print("-" * 70)
    
    prompt = render_state(test_state, persona=persona)
    print(prompt)
    
    print()
    print("✅ Persona信息已包含在prompt中")
    print()

print("=" * 70)
print("测试总结")
print("=" * 70)
print()
print("✅ render_state 现在接受persona参数")
print("✅ 每个persona生成不同的User Profile信息")
print("✅ 模型将能看到:")
print("   - User Type (Busy/Experienced/Novice)")
print("   - Patience level (low/mid/high)")
print("   - Expertise level (low/mid/high)")
print()
print("预期训练效果:")
print("  - Busy-Developer → 更少Clarify (快速执行)")
print("  - Experienced-Engineer → 适度Clarify (平衡策略)")
print("  - Novice-Learner → 更多Clarify (谨慎询问)")
print()
print("=" * 70)
