#!/bin/bash
# 检查初步训练进度

echo "📊 初步训练进度检查"
echo "===================="

# 检查进程
PID_FILE="train_preliminary.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 训练进程运行中 (PID: $PID)"
        # 显示资源使用
        ps -p "$PID" -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{print "   CPU: "$2"%  MEM: "$3"%  运行时间: "$4}'
    else
        echo "✅ 训练进程已完成"
    fi
else
    echo "❌ PID文件不存在"
fi

# 检查日志
if [ -f train_preliminary.log ]; then
    echo ""
    echo "📝 最新日志（最后20行）:"
    tail -20 train_preliminary.log | sed 's/^/   /'
    
    # 检查是否完成
    if grep -q "✅.*完成\|Model and tokenizer saved\|训练完成" train_preliminary.log; then
        echo ""
        echo "✅ 训练已完成！"
    fi
    
    # 检查是否有错误
    if grep -q "Error\|error\|Exception\|Traceback" train_preliminary.log; then
        echo ""
        echo "⚠️  检测到错误，请检查日志"
    fi
else
    echo ""
    echo "⏳ 日志文件尚未创建"
fi

# 检查输出目录
OUTPUT_DIR="checkpoints/dpo_colm_preliminary_195pairs"
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "📁 输出目录:"
    echo "   $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -5 | sed 's/^/   /'
    
    # 检查checkpoint
    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ] || [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
        echo ""
        echo "✅ 模型checkpoint已保存"
    fi
fi

echo ""
