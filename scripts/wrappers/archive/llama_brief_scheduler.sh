#!/bin/bash
# Snapshot scheduler covering Llama N=200 run lifecycle.
# 6 snapshots from launch + 6h to launch + 24h.
# Each appends labeled section to /tmp/morning_brief_llama.txt.

cd /root/autodl-tmp/ProactiveLLM
BRIEF=scripts/wrappers/morning_brief_llama.sh

# Initial snapshot
bash "$BRIEF" "+0h_initial"

# Coverage timeline (relative offsets in seconds):
sleep 21600 && bash "$BRIEF" "+6h"   # Qwen PO done; Llama Base started
sleep 14400 && bash "$BRIEF" "+10h"  # Llama PO done
sleep 14400 && bash "$BRIEF" "+14h"  # Llama Base done
sleep 14400 && bash "$BRIEF" "+18h"  # pre-wakeup check
sleep 14400 && bash "$BRIEF" "+22h"  # Llama DPO near done
sleep 14400 && bash "$BRIEF" "+26h"  # final check

echo "All Llama snapshots done at $(date)" >> /tmp/morning_brief_llama.txt
