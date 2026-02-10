#!/bin/bash
# 磁盘清理脚本 - 删除旧版本模型
# 释放空间: ~33G

set -e

cd /root/autodl-tmp/ProactiveLLM/outputs

echo "========================================"
echo "开始清理旧模型..."
echo "========================================"
echo ""

echo "📊 清理前空间使用:"
du -sh .
echo ""

echo "🗑️  删除旧模型 (V10, V11, V13)..."
rm -rf prefs_bigcode/dpo_v13_llama31_8b
rm -rf prefs_bigcode/dpo_v11_llama31_8b
rm -rf prefs_bigcode/dpo_v10_llama31_8b

echo "🗑️  删除旧模型 (V5-V9)..."
rm -rf dpo_v7
rm -rf dpo_v6
rm -rf dpo_bigcode_v5_all
rm -rf dpo_bigcode_repaired
rm -rf dpo_v9

echo ""
echo "✅ 清理完成！"
echo ""

echo "📊 清理后空间使用:"
du -sh .
echo ""

echo "📊 保留的模型:"
ls -lh | grep "^d"
echo ""

echo "========================================"
echo "释放了约 33G 空间"
echo "保留了 v14_final 和 dpo_v14_llama31_8b"
echo "========================================"
