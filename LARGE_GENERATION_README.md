# 150 States数据生成说明

## ✅ 数据生成已启动

数据生成正在后台运行，**即使SSH断开也会继续执行**。

## 📋 重要文件

- **生成脚本**: `GENERATE_COLM_DATA_150STATES.sh`
- **日志文件**: `large_generation.log`
- **进程ID文件**: `large_generation.pid`
- **进度检查脚本**: `check_large_progress.sh`

## 🔍 查看进度

### 方法1: 使用进度检查脚本（推荐）
```bash
./check_large_progress.sh
```

### 方法2: 查看实时日志
```bash
tail -f large_generation.log
```

### 方法3: 检查进程状态
```bash
ps aux | grep generate_trajectories
```

### 方法4: 查看生成的文件
```bash
ls -lh data/logs/traj_colm_3turn_persona_150states_*.jsonl
ls -lh data/dpo/traj_colm_3turn_persona_150states_*_prefs.jsonl
```

## 📊 生成规模

- **States**: 150
- **Personas**: 3 (Busy-Developer, Experienced-Engineer, Novice-Learner)
- **Samples per (state, persona)**: 4
- **总trajectories**: 150 × 3 × 4 = 1,800个
- **预计Preference pairs**: ~1,432个
- **预计训练集**: ~1,145 pairs
- **预计测试集**: ~286 pairs

## ⏱️ 预计时间

- **预计时间**: 2-3小时（取决于API响应速度）
- **开始时间**: 已记录在日志文件中

## 📁 生成的文件

生成完成后，会创建以下文件：

1. **轨迹数据**: `data/logs/traj_colm_3turn_persona_150states_<timestamp>.jsonl`
2. **Preference pairs**: `data/dpo/traj_colm_3turn_persona_150states_<timestamp>_prefs.jsonl`
3. **训练集**: `data/dpo/traj_colm_3turn_persona_150states_<timestamp>_train_prefs.jsonl`
4. **测试集**: `data/dpo/traj_colm_3turn_persona_150states_<timestamp>_test_prefs.jsonl`

## ✅ 检查是否完成

运行进度检查脚本，如果看到：
```
✅ 数据生成已完成！
```

说明生成已完成。

## 🔧 如果进程意外停止

如果发现进程停止了，可以重新启动：
```bash
nohup bash GENERATE_COLM_DATA_150STATES.sh > large_generation.log 2>&1 &
echo $! > large_generation.pid
```

## 💡 提示

- 数据生成是后台运行，不会阻塞终端
- 可以随时检查进度
- 生成完成后会自动进行质量分析和数据分割
- 所有输出都会记录在 `large_generation.log` 中

