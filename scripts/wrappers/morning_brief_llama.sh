#!/bin/bash
# Llama N=200 + Qwen PO chain status snapshot.
# Appends labeled section to /tmp/morning_brief_llama.txt.
# Usage: bash scripts/wrappers/morning_brief_llama.sh [LABEL]

LABEL="${1:-snapshot}"
OUT=/tmp/morning_brief_llama.txt
cd /root/autodl-tmp/ProactiveLLM

{
  echo
  echo "=========================================================="
  echo "  [$LABEL]  $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "  (Chicago: $(TZ=America/Chicago date '+%Y-%m-%d %H:%M %Z'))"
  echo "=========================================================="

  echo
  echo "── Processes ──"
  ps -eo pid,etime,stat,cmd | grep -E "evaluate_multi_turn" | grep -v grep \
    | awk '{printf "  PID=%-6s ELAPSED=%-10s STAT=%s  %s\n", $1, $2, $3, substr($0, index($0,$4))}' \
    | head -8
  if ! pgrep -f "evaluate_multi_turn" >/dev/null; then
    echo "  (no eval processes alive — all done OR all crashed)"
  fi

  echo
  echo "── Sample progress ──"
  declare -A LOGS
  LOGS[GPU0_Llama_DPO]=/tmp/llama_dpo_200.log
  LOGS[GPU1_chain]=/tmp/gpu1_chain.log
  LOGS[GPU2_Llama_PO]=/tmp/llama_po_200.log

  for tag in GPU0_Llama_DPO GPU1_chain GPU2_Llama_PO; do
    log="${LOGS[$tag]}"
    if [ ! -f "$log" ]; then
      printf "  %-18s (no log)\n" "$tag"
      continue
    fi
    n=$(grep -oE "样本 [0-9]+/[0-9]+" "$log" 2>/dev/null | tail -1)
    step=$(grep -E "STEP [0-9]/[0-9]" "$log" 2>/dev/null | tail -1 | grep -oE "STEP [0-9]/[0-9]")
    done_marker=$(grep -E "^\[.*\] Done:" "$log" 2>/dev/null | tail -1)
    printf "  %-18s %s  %s\n" "$tag" "${n:-loading}" "${step:-}"
    [ -n "$done_marker" ] && printf "    %s\n" "$done_marker"
  done

  echo
  echo "── Output files ──"
  for f in outputs/eval_v33_v3_llama_dpo_200.json{,.partial} \
           outputs/eval_v29_qwen_prompt_only_remaining100_ft.json{,.partial} \
           outputs/eval_v29_llama_base_200_v2.json{,.partial} \
           outputs/eval_v29_llama_prompt_only_200_v2.json{,.partial}; do
    [ -f "$f" ] || continue
    n=$(python3 -c "import json; d=json.load(open('$f')); print(len(d.get('detailed_results',[])))" 2>/dev/null)
    sz=$(stat -c '%s' "$f" 2>/dev/null)
    sz_kb=$((sz / 1024))
    final="          "
    [[ "$f" != *.partial ]] && final=" [FINAL]  "
    printf "  %s%-65s saved=%-6s (%dKB)\n" "$final" "$(basename $f)" "$n" "$sz_kb"
  done

  echo
  echo "── GPU live ──"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

  echo
  echo "── Freeze monitor (last 5 lines) ──"
  if [ -f /tmp/freeze_monitor.log ]; then
    tail -5 /tmp/freeze_monitor.log
  fi

  echo
  echo "── If any FINAL, quick pass@1 ──"
  for f in outputs/eval_v33_v3_llama_dpo_200.json \
           outputs/eval_v29_qwen_prompt_only_remaining100_ft.json \
           outputs/eval_v29_llama_base_200_v2.json \
           outputs/eval_v29_llama_prompt_only_200_v2.json; do
    [ -f "$f" ] || continue
    summary=$(python3 -c "
import json
try:
  d = json.load(open('$f'))
  s = d.get('summary', {})
  if not s:
    detailed = d.get('detailed_results', [])
    print(f'  in_progress saved={len(detailed)}')
    exit()
  parts = []
  total_p = 0; total_n = 0
  for p in ['Novice-Learner','Experienced-Engineer','Busy-Developer']:
    if p not in s: continue
    pk = s[p]['pass_at_k']['pass@1']
    parts.append(f'{p[:3]}={pk[\"passed\"]}/{pk[\"total\"]}({100*pk[\"passed\"]/pk[\"total\"]:.1f}%)')
    total_p += pk['passed']; total_n += pk['total']
  if total_n:
    print(f'  All={total_p}/{total_n}={100*total_p/total_n:.1f}%  ' + ' '.join(parts))
except Exception as e:
  print(f'  err: {e}')
")
    printf "  %-65s %s\n" "$(basename $f):" "$summary"
  done

} >> "$OUT"

echo "Wrote snapshot [$LABEL] to $OUT"
