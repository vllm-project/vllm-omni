#!/usr/bin/env bash
# Run the catcher and, the instant the PD pipeline wedges (server log stalls +
# both GPUs idle while the catcher is still running), py-spy dump every live
# vLLM stage process (native stacks) so we can pin the busy-spin. Does NOT kill
# anything — leaves the wedged server up for live inspection.
set -u
cd /root/vllm-omni/examples/online_serving/qwen3_tts_pd/benchmark
LOG=results/server_pd_1p1d.log
OUT=results/catch/pyspy
mkdir -p "$OUT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1

LOCALE=zh CONC=64 ROUNDS=60 ABS_S=20 python3 catch_overgen.py > results/catch/catch_pyspy.log 2>&1 &
CATCH_PID=$!
echo "catcher pid=$CATCH_PID"

dump_all() {
    local tag="$1"
    echo "===== py-spy sweep ($tag) $(date +%T) =====" | tee -a "$OUT/stacks.txt"
    # native VLLM procs: match by comm (VLLM::...) and the vllm serve parent
    for pid in $(ps -eo pid,comm,cmd | grep -iE "VLLM::|vllm serve" | grep -v grep | awk '{print $1}'); do
        local name; name="$(ps -o comm= -p "$pid" 2>/dev/null)"
        local cpu; cpu="$(ps -o pcpu= -p "$pid" 2>/dev/null | tr -d ' ')"
        echo "----- pid=$pid comm=$name cpu=${cpu}% -----" | tee -a "$OUT/stacks.txt"
        timeout 20 py-spy dump --pid "$pid" --nonblocking >> "$OUT/stacks.txt" 2>&1 \
            || echo "  (py-spy failed for $pid)" >> "$OUT/stacks.txt"
    done
    echo "wrote $OUT/stacks.txt ($tag)"
}

while kill -0 "$CATCH_PID" 2>/dev/null; do
    sleep 15
    now=$(date +%s); lm=$(stat -c %Y "$LOG" 2>/dev/null || echo 0); age=$((now-lm))
    u0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null|tr -d ' ')
    u1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null|tr -d ' ')
    echo "[mon $(date +%T)] log_age=${age}s gpu0=${u0}% gpu1=${u1}%"
    if [ "$age" -ge 40 ] && [ "${u0:-0}" -le 2 ] && [ "${u1:-0}" -le 2 ]; then
        echo "===== WEDGE DETECTED $(date +%T) ====="
        dump_all "wedge-t0"
        sleep 8
        dump_all "wedge-t1"   # second sweep 8s later: if same frame -> truly stuck, not slow
        echo "===== stacks captured; killing catcher, leaving server up ====="
        kill "$CATCH_PID" 2>/dev/null
        break
    fi
done
echo "===== done ====="
