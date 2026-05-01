#!/bin/bash
# Generate a human-readable status snapshot for wake-up review.
# Appends one labeled section per call to /tmp/morning_brief.txt.
# Usage: bash scripts/wrappers/morning_brief.sh [LABEL]

LABEL="${1:-snapshot}"
OUT=/tmp/morning_brief.txt
cd /root/autodl-tmp/ProactiveLLM

{
  echo
  echo "=========================================================="
  echo "  [$LABEL]  $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "=========================================================="

  echo
  echo "── Processes ──"
  ps -eo pid,etime,stat,cmd | grep -E "evaluate_multi_turn.*remaining100" | grep -v grep \
    | awk '{printf "  PID=%-6s ELAPSED=%-10s STAT=%s  %s\n", $1, $2, $3, substr($0, index($0,$4))}' \
    | head -8
  if ! pgrep -f "evaluate_multi_turn.*remaining100" >/dev/null; then
    echo "  (no eval processes alive — likely all done OR all crashed)"
  fi

  echo
  echo "── Sample progress ──"
  for tag in direct cf base po; do
    log=/tmp/qwen_${tag}_remaining100_ft.log
    if [ -f "$log" ]; then
      n=$(grep -oE "样本 [0-9]+/100" "$log" 2>/dev/null | tail -1)
      printf "  [%-6s] %s\n" "$tag" "${n:-no data}"
    fi
  done

  echo
  echo "── Output files (sorted by mtime) ──"
  ls -la --time-style=full-iso outputs/eval_v29_qwen_*remaining100_ft* 2>/dev/null \
    | awk '{print $6, $7, $9}'

  echo
  echo "── Saved sample count (3 entries / sample = 1 per persona) ──"
  for f in outputs/eval_v29_qwen_*remaining100_ft.json{,.partial}; do
    [ -f "$f" ] || continue
    n=$(python3 -c "import json; d=json.load(open('$f')); print(len(d.get('detailed_results',[])))" 2>/dev/null)
    printf "  %-65s saved=%s/300\n" "$(basename $f)" "$n"
  done

  echo
  echo "── GPU now ──"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

  echo
  echo "── Freeze monitor recent (last 8 lines) ──"
  if [ -f /tmp/freeze_monitor.log ]; then
    head -1 /tmp/freeze_monitor.log
    tail -8 /tmp/freeze_monitor.log
  else
    echo "  (no freeze monitor log)"
  fi

  echo
  echo "── If all done: quick pass@1 check ──"
  for f in outputs/eval_v29_qwen_*remaining100_ft.json; do
    [ -f "$f" ] || continue
    summary=$(python3 -c "
import json
try:
  d = json.load(open('$f'))
  detailed = d.get('detailed_results', [])
  if not detailed: print('no results'); exit()
  by_p = {}
  for r in detailed:
    p = r.get('persona', '?').split('-')[0]
    by_p.setdefault(p, [0, 0])
    by_p[p][1] += 1
    if r.get('pass_at_1', 0) >= 1: by_p[p][0] += 1
  parts = [f'{p}:{passed}/{total}' for p, (passed, total) in sorted(by_p.items())]
  total_pass = sum(p[0] for p in by_p.values())
  total_n = sum(p[1] for p in by_p.values())
  print(f'  {total_pass}/{total_n} = {100*total_pass/total_n:.1f}%  ({\" / \".join(parts)})')
except Exception as e:
  print(f'  err: {e}')
")
    printf "  %s%s\n" "$(basename $f): " "$summary"
  done

} >> "$OUT"

echo "Wrote snapshot [$LABEL] to $OUT"
