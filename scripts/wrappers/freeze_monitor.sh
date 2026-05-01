#!/bin/bash
# Lightweight freeze monitor — emits one line every ~20 min recording:
#   container_time | gpu0_util | gpu1_util | gpu2_util | direct_n | cf_n | base_n | direct_elapsed | cf_elapsed | base_elapsed
# Gaps in log timestamps reveal autodl freeze periods.
# Output: /tmp/freeze_monitor.log

LOG=/tmp/freeze_monitor.log
INTERVAL=1200  # 20 min

cd /root/autodl-tmp/ProactiveLLM

# header (only if file doesn't exist)
if [ ! -f "$LOG" ]; then
  echo "container_time,host_uptime_s,gpu0_util,gpu1_util,gpu2_util,direct_n,cf_n,base_n,direct_elapsed,cf_elapsed,base_elapsed" > "$LOG"
fi

count_samples() {
  # count completed samples in incremental file (3 entries per sample = 1 per persona)
  local f="$1"
  if [ -f "$f" ]; then
    python3 -c "import json; d=json.load(open('$f')); print(len(d.get('detailed_results',[])) // 3)" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

get_elapsed() {
  # ELAPSED of process matching pattern, e.g. 'direct_execution'
  ps -eo etime,cmd | grep -E "evaluate_multi_turn.*$1" | grep -v grep | head -1 | awk '{print $1}'
}

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  uptime_s=$(awk '{printf "%.0f", $1}' /proc/uptime)

  # GPU util
  gpus=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -3 | tr '\n' ',' | sed 's/,$//' | sed 's/, /,/g' | tr -d ' ')
  g0=$(echo "$gpus" | cut -d, -f1)
  g1=$(echo "$gpus" | cut -d, -f2)
  g2=$(echo "$gpus" | cut -d, -f3)

  # samples per baseline
  d_n=$(count_samples outputs/eval_v29_qwen_direct_execution_remaining100_ft.json.partial)
  c_n=$(count_samples outputs/eval_v29_qwen_clarify_first_remaining100_ft.json.partial)
  b_n=$(count_samples outputs/eval_v29_qwen_base_remaining100_ft.json.partial)

  # process elapsed
  d_e=$(get_elapsed "direct_execution")
  c_e=$(get_elapsed "always_clarify")
  b_e=$(get_elapsed "qwen_base_remaining100\|test_states_v29_eval_200_remaining100" | head -1)
  # base detection: --no_lora w/o --direct_execution and w/o --always_clarify
  b_e=$(ps -eo etime,cmd | grep "evaluate_multi_turn" | grep "remaining100" | grep -v "direct_execution\|always_clarify\|grep" | head -1 | awk '{print $1}')

  echo "$ts,$uptime_s,$g0,$g1,$g2,$d_n,$c_n,$b_n,$d_e,$c_e,$b_e" >> "$LOG"
  sleep $INTERVAL
done
