#!/bin/bash
# Overnight watcher: log every 20 min, alert on chain death
LOG=/tmp/overnight_watch.log
while true; do
    {
        echo "==== $(date) ===="
        echo "Active eval procs:"
        pgrep -af "evaluate_multi_turn" | grep -v grep | head
        echo "Active launchers:"
        pgrep -af "launch_bf16" | grep -v grep | head
        echo "GPU util:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
        echo "Recent partial outputs:"
        ls -lt outputs/*_bf16.json* 2>/dev/null | head -3 | awk '{print "  "$6, $7, $8, $9}'
        echo
    } >> $LOG
    sleep 1200  # 20 min
done
