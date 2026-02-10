# 磁盘空间清理计划

**当前状态**: autodl-tmp占用 93G

## 📊 空间占用分析

### 总览
```
/root/autodl-tmp/                      93G
├── ProactiveLLM/                      48G
│   ├── outputs/                       42G  ← 主要清理目标
│   │   ├── prefs_bigcode/             19G  ← 4个模型
│   │   │   ├── dpo_v14_llama31_8b     4.6G  (最新，保留)
│   │   │   ├── dpo_v13_llama31_8b     4.6G  (可删除)
│   │   │   ├── dpo_v11_llama31_8b     4.6G  (可删除)
│   │   │   └── dpo_v10_llama31_8b     4.6G  (可删除)
│   │   ├── v14_final/                 4.6G  (最新，保留)
│   │   ├── dpo_v7/                    4.6G  (可删除)
│   │   ├── dpo_v6/                    4.6G  (可删除)
│   │   ├── dpo_bigcode_v5_all/        4.6G  (可删除)
│   │   ├── dpo_bigcode_repaired/      4.6G  (可删除)
│   │   └── dpo_v9/                    657M  (可删除)
│   └── .git/                          5.7G  (保留)
├── hf_cache/                          38G
│   ├── Llama-3.1-8B-Instruct/         15G  (保留，训练需要)
│   └── 其他缓存/                       23G  (保留)
└── conda_envs/                        7.9G  (保留)
```

---

## 🎯 清理建议

### 方案1: 保守清理（推荐）✅
**删除所有旧版本模型，保留v14和最新的dpo_v14**

可删除：
- `outputs/prefs_bigcode/dpo_v13_llama31_8b/` - 4.6G
- `outputs/prefs_bigcode/dpo_v11_llama31_8b/` - 4.6G
- `outputs/prefs_bigcode/dpo_v10_llama31_8b/` - 4.6G
- `outputs/dpo_v7/` - 4.6G
- `outputs/dpo_v6/` - 4.6G
- `outputs/dpo_bigcode_v5_all/` - 4.6G
- `outputs/dpo_bigcode_repaired/` - 4.6G
- `outputs/dpo_v9/` - 657M

**总计释放**: ~33G

**保留**:
- `outputs/v14_final/` - 4.6G（最新训练的模型）
- `outputs/prefs_bigcode/dpo_v14_llama31_8b/` - 4.6G（最新版本）
- `outputs/eval_results/` - 696K（评估结果）
- `outputs/runlogs/` - 4.9M（日志）

**清理后**: 93G - 33G = **60G** (~35%减少)

---

### 方案2: 激进清理（如果还不够）
**额外删除v14（因为它有格式问题，可以重新训练）**

额外可删除：
- `outputs/v14_final/` - 4.6G
- `outputs/prefs_bigcode/dpo_v14_llama31_8b/` - 4.6G

**总计额外释放**: 9.2G

**清理后**: 60G - 9.2G = **51G** (~45%减少)

---

## 🚀 推荐执行方案

### 方案1的清理命令（推荐）✅

```bash
cd /root/autodl-tmp/ProactiveLLM/outputs

# 删除旧版本模型
echo "删除旧模型..."
rm -rf prefs_bigcode/dpo_v13_llama31_8b
rm -rf prefs_bigcode/dpo_v11_llama31_8b
rm -rf prefs_bigcode/dpo_v10_llama31_8b
rm -rf dpo_v7
rm -rf dpo_v6
rm -rf dpo_bigcode_v5_all
rm -rf dpo_bigcode_repaired
rm -rf dpo_v9

echo "清理完成！"
du -sh .
```

### 检查清理效果

```bash
# 检查总空间使用
du -sh /root/autodl-tmp

# 检查outputs目录
du -h --max-depth=1 /root/autodl-tmp/ProactiveLLM/outputs | sort -hr
```

---

## ⚠️ 注意事项

### 安全提醒
1. **这些旧模型删除后无法恢复**（除非重新训练）
2. **v14_final虽然有格式问题，但我们已经总结了经验，可以保留作为参考**
3. **如果未来需要，可以用相同的数据重新训练这些版本**

### 为什么可以安全删除
1. **V10-V13都是中间版本**，已经被V14改进了
2. **V14的训练数据已经保存**在 `data/dpo/prefs_100states_balanced_*.jsonl`
3. **V15将使用相同的数据重新训练**，只是修复了格式问题

### 保留的重要文件
- ✅ 所有训练数据（`data/dpo/`）
- ✅ 所有评估结果（`outputs/eval_results/`）
- ✅ 所有文档（`docs/`, `*.md`）
- ✅ Base model（`hf_cache/Llama-3.1-8B-Instruct/`）

---

## 📝 V15训练空间需求

训练V15需要的空间：
- Base model: 15G（已有）
- Training artifacts: ~5G（临时）
- Final model: ~4.6G

**总需求**: ~10G（新增）

**清理方案1后剩余空间**: 60G → 足够训练V15 ✅

---

## 🎯 执行建议

**立即执行方案1**：
1. 释放33G空间
2. 保留v14作为参考
3. 有足够空间训练v15
4. 10分钟内完成

**如果方案1后还不够（非常罕见）**：
- 再执行方案2，额外释放9.2G
- V14可以随时用相同数据重新训练
