#!/bin/bash
# 监控模型下载进度

echo "📊 模型下载监控"
echo "=================="
echo ""

# 检查下载进程
if pgrep -f "snapshot_download" > /dev/null; then
    echo "✅ 下载进程正在运行"
    ps aux | grep "snapshot_download" | grep -v grep | awk '{print "  PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo "❌ 下载进程未运行"
fi

echo ""
echo "📁 已下载的 safetensors 文件："
COMPLETED=$(ls -lh /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/model-*.safetensors 2>/dev/null | wc -l)
echo "  数量: $COMPLETED/4"

echo ""
echo "📥 正在下载的文件："
for f in /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/.cache/huggingface/download/*.incomplete; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
        TIME=$(stat -c %y "$f" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
        NAME=$(basename "$f" .incomplete | cut -c1-20)
        echo "  $NAME... ($SIZE, 更新: $TIME)"
    fi
done

echo ""
echo "💾 磁盘空间："
df -h /root/autodl-tmp | tail -1 | awk '{print "  使用率: " $5 " | 可用: " $4}'

echo ""
echo "📈 缓存目录大小："
du -sh /root/autodl-tmp/hf_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null | awk '{print "  " $1}'

echo ""
echo "🕐 最后更新：$(date '+%Y-%m-%d %H:%M:%S')"
