#!/bin/bash
# Watch GPU 0 (Llama DPO PID 159135) to free up; then:
#   1) Run CollabLLM 5-state sanity on GPU 0
#   2) Inspect classifier verdict — confirm Clarify/Execute mix exists
#   3) If sanity OK, launch full 200-state eval on GPU 0
# Logs:
#   /tmp/wait_then_collabllm.log   (this script)
#   /tmp/collabllm_sanity.log      (sanity eval)
#   /tmp/collabllm_200.log         (full eval)

set -u
WATCH_PID=159135                              # Llama v33 SFT+DPO 200 python PID, GPU 0
GPU=0
SANITY_WRAP=/root/autodl-tmp/ProactiveLLM/scripts/wrappers/collabllm_llama_sanity.sh
FULL_WRAP=/root/autodl-tmp/ProactiveLLM/scripts/wrappers/collabllm_llama_200.sh
SANITY_OUT=/root/autodl-tmp/ProactiveLLM/outputs/eval_collabllm_llama_sanity5.json
SANITY_LOG=/tmp/collabllm_sanity.log
FULL_LOG=/tmp/collabllm_200.log
SELF_LOG=/tmp/wait_then_collabllm.log

exec >> "$SELF_LOG" 2>&1
echo "==== started $(date) — watching PID $WATCH_PID (Llama DPO 200) ===="

# Wait for Llama DPO python process to exit (frees GPU 0)
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date)] PID $WATCH_PID exited."

# Confirm GPU 0 is idle
sleep 15
GPU_USE=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "[$(date)] GPU $GPU memory.used = ${GPU_USE} MiB"
if [ "${GPU_USE:-99999}" -gt 2000 ]; then
  echo "[$(date)] WARN: GPU $GPU still has ${GPU_USE} MiB used — waiting 60s"
  sleep 60
fi

# Step 1: Sanity eval (5 state × 3 persona = 15 conversations, ~15 min including model download)
echo "[$(date)] STEP 1: launching CollabLLM sanity-5 on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU bash "$SANITY_WRAP" > "$SANITY_LOG" 2>&1 &
SANITY_PID=$!
echo "[$(date)] sanity bash PID=$SANITY_PID"
wait "$SANITY_PID"
SANITY_RC=$?
echo "[$(date)] sanity exited rc=$SANITY_RC"

if [ $SANITY_RC -ne 0 ] || [ ! -f "$SANITY_OUT" ]; then
  echo "[$(date)] ABORT: sanity failed (rc=$SANITY_RC, out_exists=$([ -f "$SANITY_OUT" ] && echo yes || echo no))"
  echo "==== aborted $(date) ===="
  exit 1
fi

# Step 2: Inspect classifier verdict — degenerate if all 1-turn Execute
echo "[$(date)] STEP 2: inspecting sanity classifier distribution"
PYOUT=$(python3 - <<EOF 2>&1
import json
from collections import Counter
d = json.load(open("$SANITY_OUT"))
acts = []
for r in d["detailed_results"]:
    acts += r["actions"]
turn_dist = Counter(r["total_turns"] for r in d["detailed_results"])
act_dist = Counter(acts)
n = len(d["detailed_results"])
clar_n = act_dist.get("Clarify", 0)
exec_n = act_dist.get("Execute", 0)
print(f"n_conversations={n}")
print(f"action_dist={dict(act_dist)}")
print(f"turn_dist={dict(turn_dist)}")
# Pass criteria: must see at least 1 Clarify (otherwise classifier degenerate)
ok = clar_n >= 1
print(f"VERDICT: {'OK' if ok else 'DEGENERATE'}")
EOF
)
echo "$PYOUT"
if echo "$PYOUT" | grep -q "VERDICT: OK"; then
  echo "[$(date)] sanity OK — proceeding to full 200"
else
  echo "[$(date)] ABORT: sanity DEGENERATE — classifier sees no Clarify."
  echo "          Need to inspect outputs manually before full run."
  echo "          See $SANITY_OUT and $SANITY_LOG"
  echo "==== aborted $(date) ===="
  exit 2
fi

# Step 3: Full 200 eval on GPU 0
echo "[$(date)] STEP 3: launching CollabLLM full 200 on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU bash "$FULL_WRAP" > "$FULL_LOG" 2>&1 &
FULL_PID=$!
echo "[$(date)] full bash PID=$FULL_PID, log=$FULL_LOG"
wait "$FULL_PID"
echo "[$(date)] full exited rc=$?"
echo "==== done $(date) ===="
