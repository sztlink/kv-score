#!/bin/bash
# T axis: serve an OpenAI-compatible vLLM config and report engine-counter
# generation throughput at N=1/32/128 (steady windows; ramp/drain excluded by eye).
#
# Usage: measure-tput.sh <label> '<full serve command>'
# The serve command must expose --served-model-name spikemodel on :8001.
# Why engine counter and not client-side tokens/wall-clock: the client window
# includes prefill ramp and drain tail and under-counted 12x in our hands.
set -u
LABEL="${1:?label}"
SCMD="${2:?serve command}"
RDIR="${KVSCORE_OUT:-./out}"; mkdir -p "$RDIR"
LOG="$RDIR/serve-$LABEL.log"

free_vram() { for i in $(seq 1 60); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${U:-9}" -lt 1500 ] && return 0; sleep 3; done; echo "WARN: VRAM not freed"; }
hard_clean() { pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null; sleep 6; free_vram; }

hard_clean
bash -c "$SCMD" > "$LOG" 2>&1 &
SP=$!
for i in $(seq 1 120); do
  curl -sf -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q spikemodel && break
  kill -0 $SP 2>/dev/null || { echo "[$LABEL] SERVE_FAIL"; tail -5 "$LOG"; exit 1; }
  sleep 5
done
grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed "s/^/[$LABEL] /"

for N in 1 32 128; do
  MARK=$(wc -l < "$LOG"); DUR=40; [ "$N" -ge 128 ] && DUR=90
  python3 "$(dirname "$0")/burst.py" http://localhost:8001/v1/completions "$N" "$DUR" > /dev/null 2>&1
  echo "[$LABEL tput N=$N] engine windows:"
  tail -n +"$MARK" "$LOG" | grep -oE "generation throughput: [0-9.]+ tokens/s, Running: [0-9]+" | tail -6
  sleep 5
done
hard_clean
