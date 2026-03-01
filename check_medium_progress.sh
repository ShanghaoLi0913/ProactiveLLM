#!/bin/bash
# 检查中等规模数据生成进度

LOG_FILE="medium_generation.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "=" * 70
echo "📊 数据生成进度检查"
echo "=" * 70

# 检查是否完成
if grep -q "✓ 中等规模数据生成完成" "$LOG_FILE"; then
    echo "✅ 数据生成已完成！"
    echo ""
    tail -50 "$LOG_FILE"
else
    echo "⏳ 数据生成进行中..."
    echo ""
    
    # 显示最新进度
    if grep -q "Progress\]" "$LOG_FILE"; then
        echo "最新进度:"
        grep "Progress\]" "$LOG_FILE" | tail -3
    fi
    
    # 显示最新日志
    echo ""
    echo "最新日志（最后20行）:"
    tail -20 "$LOG_FILE"
    
    # 检查是否有错误
    if grep -qi "error\|exception\|traceback" "$LOG_FILE" | tail -5; then
        echo ""
        echo "⚠️  发现可能的错误:"
        grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5
    fi
fi

# 检查生成的文件
echo ""
echo "=" * 70
echo "📁 生成的文件检查"
echo "=" * 70

TRAJ_FILE=$(ls -t data/logs/medium_traj_*.jsonl 2>/dev/null | head -1)
PREFS_FILE=$(ls -t data/dpo/medium_traj_*_prefs.jsonl 2>/dev/null | head -1)

if [ -n "$TRAJ_FILE" ]; then
    echo "✅ 轨迹文件: $TRAJ_FILE"
    echo "   大小: $(du -h "$TRAJ_FILE" | cut -f1)"
    echo "   行数: $(wc -l < "$TRAJ_FILE")"
else
    echo "⏳ 轨迹文件: 尚未生成"
fi

if [ -n "$PREFS_FILE" ]; then
    echo "✅ Preference文件: $PREFS_FILE"
    echo "   大小: $(du -h "$PREFS_FILE" | cut -f1)"
    echo "   行数: $(wc -l < "$PREFS_FILE")"
else
    echo "⏳ Preference文件: 尚未生成"
fi

