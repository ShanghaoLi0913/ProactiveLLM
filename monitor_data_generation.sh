#!/bin/bash
# 监控数据生成进度

echo "=" * 80
echo "📊 数据生成进度监控"
echo "=" * 80

# 检查进程
echo ""
echo "【进程状态】"
ps aux | grep -E "generate_trajectories|GENERATE_COLM" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "运行时间:", $10}'

# 检查日志
echo ""
echo "【日志状态】"
if [ -f /tmp/generate_data_phase1.log ]; then
    LOG_SIZE=$(du -h /tmp/generate_data_phase1.log | cut -f1)
    LOG_LINES=$(wc -l < /tmp/generate_data_phase1.log)
    echo "  日志文件: /tmp/generate_data_phase1.log"
    echo "  大小: $LOG_SIZE"
    echo "  行数: $LOG_LINES"
    echo ""
    echo "  最后10行:"
    tail -10 /tmp/generate_data_phase1.log | sed 's/^/    /'
else
    echo "  ⚠️  日志文件不存在"
fi

# 检查输出文件
echo ""
echo "【输出文件】"
TRAJ_FILES=$(find data/logs -name "traj_colm_3turn_persona_100states_*.jsonl" -type f 2>/dev/null | sort | tail -1)
if [ -n "$TRAJ_FILES" ]; then
    TRAJ_SIZE=$(du -h "$TRAJ_FILES" | cut -f1)
    TRAJ_LINES=$(wc -l < "$TRAJ_FILES" 2>/dev/null || echo "0")
    echo "  轨迹文件: $TRAJ_FILES"
    echo "  大小: $TRAJ_SIZE"
    echo "  行数: $TRAJ_LINES (trajectories)"
    echo "  预计总trajectories: ~800-1000"
    if [ "$TRAJ_LINES" -gt 0 ]; then
        PROGRESS=$(echo "scale=1; $TRAJ_LINES / 1000 * 100" | bc 2>/dev/null || echo "0")
        echo "  进度: ~${PROGRESS}%"
    fi
else
    echo "  ⏳ 轨迹文件尚未生成"
fi

PREFS_FILES=$(find data/dpo -name "*_prefs.jsonl" -type f 2>/dev/null | grep "traj_colm_3turn_persona_100states" | sort | tail -1)
if [ -n "$PREFS_FILES" ]; then
    PREFS_SIZE=$(du -h "$PREFS_FILES" | cut -f1)
    PREFS_LINES=$(wc -l < "$PREFS_FILES" 2>/dev/null || echo "0")
    echo ""
    echo "  Preference文件: $PREFS_FILES"
    echo "  大小: $PREFS_SIZE"
    echo "  行数: $PREFS_LINES (preference pairs)"
fi

TRAIN_FILES=$(find data/dpo -name "*_train_prefs.jsonl" -type f 2>/dev/null | grep "traj_colm_3turn_persona_100states" | sort | tail -1)
TEST_FILES=$(find data/dpo -name "*_test_prefs.jsonl" -type f 2>/dev/null | grep "traj_colm_3turn_persona_100states" | sort | tail -1)

if [ -n "$TRAIN_FILES" ] && [ -n "$TEST_FILES" ]; then
    TRAIN_LINES=$(wc -l < "$TRAIN_FILES" 2>/dev/null || echo "0")
    TEST_LINES=$(wc -l < "$TEST_FILES" 2>/dev/null || echo "0")
    echo ""
    echo "  ✅ 数据分割完成:"
    echo "    训练集: $TRAIN_LINES pairs"
    echo "    测试集: $TEST_LINES pairs"
fi

echo ""
echo "=" * 80
