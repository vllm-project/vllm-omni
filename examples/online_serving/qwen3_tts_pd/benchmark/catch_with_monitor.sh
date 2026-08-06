#!/usr/bin/env bash
# Runs the over-generation catcher and an external freeze monitor. When the
# pipeline wedges (server log stops advancing AND both GPUs idle while the
# catcher is still running), dump the freeze scene: last PD_TRACE lines, the
# resume-vs-submit gap, and per-req submit/decode status for the stuck reqs.
set -u
cd /root/vllm-omni/examples/online_serving/qwen3_tts_pd/benchmark
LOG=results/server_pd_1p1d.log
DUMP=results/catch/freeze_dump.txt
mkdir -p results/catch
: > "$DUMP"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1

# launch catcher (many rounds, c=64, zh)
LOCALE=zh CONC=64 ROUNDS=40 ABS_S=20 python3 catch_overgen.py > results/catch/catch2.log 2>&1 &
CATCH_PID=$!
echo "catcher pid=$CATCH_PID"

strip() { sed -E 's/\x1b\[[0-9;]*m//g'; }

last_log_mtime() { stat -c %Y "$LOG" 2>/dev/null || echo 0; }

FROZEN=0
while kill -0 "$CATCH_PID" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    lm=$(last_log_mtime)
    age=$(( now - lm ))
    util0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    util1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
    echo "[mon $(date +%T)] log_age=${age}s gpu0=${util0}% gpu1=${util1}%"
    # freeze = log stale >45s and both GPUs idle, catcher still running
    if [ "$age" -ge 45 ] && [ "${util0:-0}" -le 2 ] && [ "${util1:-0}" -le 2 ]; then
        FROZEN=1
        echo "===== FREEZE DETECTED $(date +%T) log_age=${age}s =====" | tee -a "$DUMP"
        echo "--- tail 40 server log ---" >> "$DUMP"
        tail -40 "$LOG" | strip | cut -c1-200 >> "$DUMP"
        echo "--- global counts ---" >> "$DUMP"
        echo "prefill_submit_ready=$(grep -c qwen3_tts_prefill_submit_ready $LOG)" >> "$DUMP"
        echo "prefill_to_decode_submitted=$(grep -c prefill_to_decode_submitted $LOG)" >> "$DUMP"
        echo "decode_resume=$(grep -c qwen3_tts_decode_resume $LOG)" >> "$DUMP"
        echo "--- LAST prefill_submit_ready ---" >> "$DUMP"
        grep qwen3_tts_prefill_submit_ready $LOG | strip | tail -3 | cut -c1-200 >> "$DUMP"
        echo "--- LAST prefill_to_decode_submitted ---" >> "$DUMP"
        grep prefill_to_decode_submitted $LOG | strip | tail -3 | cut -c1-200 >> "$DUMP"
        echo "--- LAST decode_resume ---" >> "$DUMP"
        grep qwen3_tts_decode_resume $LOG | strip | tail -3 | cut -c1-200 >> "$DUMP"
        echo "--- any recent orchestrator/forward/error lines (last 60 non-route) ---" >> "$DUMP"
        tail -400 "$LOG" | strip | grep -iE "orchestrator|forward|pick|submit|error|exception|timed out|abort|prewarm" | tail -40 | cut -c1-200 >> "$DUMP"
        echo "wrote $DUMP"
        # let it sit 30s more to confirm permanent, then stop catcher
        sleep 30
        kill "$CATCH_PID" 2>/dev/null
        break
    fi
done
echo "===== monitor done frozen=$FROZEN ====="
echo "--- catcher tail ---"; tail -15 results/catch/catch2.log
