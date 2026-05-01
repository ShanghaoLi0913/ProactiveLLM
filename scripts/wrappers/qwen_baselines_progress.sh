#!/bin/bash
# Show progress of 3 parallel Qwen baseline runs (Direct/CF/Base remaining-100).
# Usage: bash /tmp/qwen_baselines_progress.sh

cd /root/autodl-tmp/ProactiveLLM

echo "=========================================="
echo "  $(date)  Qwen baseline remaining-100"
echo "=========================================="
echo

# 1) processes alive?
echo "── Processes ──"
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "evaluate_multi_turn_persona.*remaining100" | grep -v grep \
  | awk '{printf "  PID=%s  ELAPSED=%s  CPU=%s%%  MEM=%s%%  %s\n", $1, $2, $3, $4, substr($0, index($0,$5))}'
echo

# 2) GPU
echo "── GPU ──"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo

# 3) sample progress per log
echo "── Per-baseline progress ──"
for tag in direct cf base; do
  log=/tmp/qwen_${tag}_remaining100_ft.log
  out=outputs/eval_v29_qwen_${tag/cf/clarify_first}_remaining100_ft.json
  case "$tag" in
    direct) out=outputs/eval_v29_qwen_direct_execution_remaining100_ft.json ;;
    cf)     out=outputs/eval_v29_qwen_clarify_first_remaining100_ft.json ;;
    base)   out=outputs/eval_v29_qwen_base_remaining100_ft.json ;;
  esac

  printf "  %-7s  " "[$tag]"

  # last sample line in log
  last=$(grep -E "样本 [0-9]+/100" "$log" 2>/dev/null | tail -1)
  if [ -n "$last" ]; then
    n=$(echo "$last" | grep -oE "[0-9]+/100" | head -1)
    printf "log=%s  " "$n"
  else
    printf "log=–  "
  fi

  # detailed_results count from incremental file
  inc="${out}.incremental"
  for f in "$inc" "$out"; do
    if [ -f "$f" ]; then
      n=$(python3 -c "import json,sys
try:
  d=json.load(open('$f'))
  print(len(d.get('detailed_results',[])))
except: print('?')" 2>/dev/null)
      printf "saved=%s/300  file=%s\n" "$n" "$(basename $f)"
      break
    fi
  done
done

echo
echo "── Last log lines (1 each) ──"
for tag in direct cf base; do
  log=/tmp/qwen_${tag}_remaining100_ft.log
  last_status=$(grep -E "样本 [0-9]+/100|Persona:|Turn [0-9]+:|pass@|Done:" "$log" 2>/dev/null | tail -1)
  printf "  [%s] %s\n" "$tag" "$last_status"
done
echo
