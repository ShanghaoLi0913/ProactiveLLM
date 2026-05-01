#!/bin/bash
# Show progress of the 3 parallel chains:
#   GPU 0: Llama DPO N=200
#   GPU 1: Qwen PO rem-100 → Llama Base N=200
#   GPU 2: Llama PO N=200

cd /root/autodl-tmp/ProactiveLLM

echo "=========================================="
echo "  $(date)"
echo "=========================================="
echo

echo "── GPU ──"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo

echo "── Processes ──"
ps -eo pid,etime,cmd | grep -E "evaluate_multi_turn" | grep -v grep \
  | awk '{
      cmd=$0; sub(/.*-test_states /, "", cmd); sub(/ .*/, "", cmd);
      if (cmd ~ /direct_execution/) tag="?";
      printf "  PID=%-6s ELAPSED=%-10s\n", $1, $2
    }' | head -8

# Custom labeled progress
echo
echo "── Sample progress ──"
declare -A LOGS
LOGS[GPU0_Llama_DPO]=/tmp/llama_dpo_200.log
LOGS[GPU1_Qwen_PO]=/tmp/gpu1_chain.log
LOGS[GPU2_Llama_PO]=/tmp/llama_po_200.log

for tag in GPU0_Llama_DPO GPU1_Qwen_PO GPU2_Llama_PO; do
  log="${LOGS[$tag]}"
  if [ ! -f "$log" ]; then
    printf "  %-18s (no log)\n" "$tag"
    continue
  fi
  n=$(grep -oE "样本 [0-9]+/[0-9]+" "$log" 2>/dev/null | tail -1)
  step=$(grep -E "STEP [0-9]/[0-9]" "$log" 2>/dev/null | tail -1 | grep -oE "STEP [0-9]/[0-9]")
  printf "  %-18s %s  %s\n" "$tag" "${n:-loading}" "${step:-}"
done

echo
echo "── Saved samples (incremental files) ──"
for f in outputs/eval_v33_v3_llama_dpo_200.json{,.partial} \
         outputs/eval_v29_qwen_prompt_only_remaining100_ft.json{,.partial} \
         outputs/eval_v29_llama_base_200_v2.json{,.partial} \
         outputs/eval_v29_llama_prompt_only_200_v2.json{,.partial}; do
  [ -f "$f" ] || continue
  n=$(python3 -c "import json; d=json.load(open('$f')); print(len(d.get('detailed_results',[])))" 2>/dev/null)
  printf "  %-65s saved=%s\n" "$(basename $f)" "$n"
done

echo
echo "── Recent freeze monitor ──"
if [ -f /tmp/freeze_monitor.log ]; then
  tail -3 /tmp/freeze_monitor.log
fi
