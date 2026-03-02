#!/bin/bash
#
# L5 GPU memory monitor - single entry script.
# Subcommands: start | finalize | run -- <command>
#
# start   - Background: nvidia-smi loop, write CSV only
# finalize - Bundle current run (CSV + report.html), print GPU_MONITOR_BUNDLE_DIR=
# run     - Start monitor, run command, then finalize; in CI upload artifacts
#
# Env: GPU_MONITOR_DATA_ROOT, GPU_MONITOR_INTERVAL, GPU_MONITOR_DEVICES,
#      GPU_MONITOR_LOG_INTERVAL, SKIP_DEPS_CHECK
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
SUBCMD="${1:-}"

# ---------- subcommand: start (was moniter.sh) ----------
cmd_start() {
    local INTERVAL="${2:-${GPU_MONITOR_INTERVAL:-5}}"
    local GPU_IDS_RAW="${1:-${GPU_MONITOR_DEVICES:-all}}"

    if [[ -z "${SKIP_DEPS_CHECK:-}" ]]; then
        if ! command -v nvidia-smi &>/dev/null; then
            echo "Error: nvidia-smi not found. Run this script on a Linux machine with NVIDIA drivers."
            exit 1
        fi
    fi

    [[ "$INTERVAL" =~ ^[0-9]+$ ]] && [[ "$INTERVAL" -ge 1 ]] || {
        echo "Error: interval must be a positive integer (seconds)"
        echo "Usage: $0 start [GPU_IDs|all] [interval_seconds]"
        exit 1
    }

    local RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
    local RUN_DIR="$DATA_ROOT/$RUN_ID"
    mkdir -p "$RUN_DIR"
    echo "$RUN_ID" > "$DATA_ROOT/current_run_id"

    local CSV_FILE="$RUN_DIR/gpu_metrics.csv"
    echo "timestamp_iso,timestamp_epoch,gpu_index,memory_used_mb,memory_total_mb,memory_util_pct" > "$CSV_FILE"

    local NVSMI_QUERY="index,memory.used,memory.total"
    local NVSMI_IDS=""
    [[ "$GPU_IDS_RAW" != "all" ]] && NVSMI_IDS="-i $GPU_IDS_RAW"

    trap 'echo "[$(date +%H:%M:%S)] Stopping; data saved to $RUN_DIR"; exit 0' SIGTERM SIGINT

    echo "========================================"
    echo "L5 GPU memory monitor started"
    echo "RUN_ID: $RUN_ID"
    echo "Data dir: $RUN_DIR"
    echo "Interval: ${INTERVAL}s | GPU: $GPU_IDS_RAW"
    echo "========================================"

    while true; do
        local TS_ISO TS_EPOCH RAW
        TS_ISO=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')
        TS_EPOCH=$(date +%s)
        RAW=$(nvidia-smi --query-gpu="$NVSMI_QUERY" --format=csv,noheader,nounits $NVSMI_IDS 2>/dev/null) || true
        if [[ -z "$RAW" ]]; then
            sleep "$INTERVAL"
            continue
        fi

        while IFS= read -r line; do
            line=$(echo "$line" | tr -d ' ')
            [[ -z "$line" ]] && continue
            local idx used total pct
            idx=$(echo "$line" | cut -d',' -f1)
            used=$(echo "$line" | cut -d',' -f2)
            total=$(echo "$line" | cut -d',' -f3)
            used=${used:-0}
            total=${total:-1}
            [[ "$total" -le 0 ]] && total=1
            pct=$((used * 100 / total))
            echo "${TS_ISO},${TS_EPOCH},${idx},${used},${total},${pct}" >> "$CSV_FILE"
        done <<< "$RAW"

        sleep "$INTERVAL"
    done
}

# ---------- subcommand: finalize (was finalize_monitor.sh) ----------
cmd_finalize() {
    local RUN_ID="${1:-}"

    if [[ -z "$RUN_ID" ]]; then
        if [[ -f "$DATA_ROOT/current_run_id" ]]; then
            RUN_ID=$(cat "$DATA_ROOT/current_run_id")
        else
            echo "Error: run_id not specified and $DATA_ROOT/current_run_id does not exist" >&2
            exit 1
        fi
    fi

    local RUN_DIR="$DATA_ROOT/$RUN_ID"
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "Error: run dir does not exist: $RUN_DIR" >&2
        exit 1
    fi

    local CSV_FILE="$RUN_DIR/gpu_metrics.csv"
    if [[ ! -f "$CSV_FILE" ]]; then
        echo "Error: CSV not found: $CSV_FILE" >&2
        exit 1
    fi

    local BUNDLE_DIR="$DATA_ROOT/gpu_monitor_bundle_${RUN_ID}"
    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR"

    cp "$CSV_FILE" "$BUNDLE_DIR/gpu_metrics.csv"

    local REPORT_HTML="$BUNDLE_DIR/report.html"
    if command -v python3 &>/dev/null; then
        if python3 "$SCRIPT_DIR/generate_report.py" "$CSV_FILE" "$REPORT_HTML"; then
            echo "Report generated: $REPORT_HTML"
        else
            echo "Warning: report generation failed; only CSV archived" >&2
        fi
    else
        echo "Warning: python3 not found; skipping report" >&2
    fi

    cat > "$BUNDLE_DIR/README.txt" << EOF
L5 GPU monitor bundle - ${RUN_ID}
- gpu_metrics.csv: raw samples
- report.html: report with charts (open in browser)
Upload this dir as a CI artifact to view after the run.
EOF

    local BUNDLE_ABS
    BUNDLE_ABS=$(cd "$BUNDLE_DIR" && pwd)
    echo "GPU_MONITOR_BUNDLE_DIR=$BUNDLE_ABS"
    echo "Archive path: $BUNDLE_ABS"
}

# ---------- subcommand: run (was run_with_gpu_monitor.sh) ----------
cmd_run() {
    shift
    local CMD=()
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift
            CMD=("$@")
            break
        fi
        shift
    done
    if [[ ${#CMD[@]} -eq 0 ]]; then
        echo "Usage: $0 run -- <command to run>" >&2
        exit 1
    fi

    export GPU_MONITOR_DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$SCRIPT_DIR/gpu_monitor_data}"
    export SKIP_DEPS_CHECK="${SKIP_DEPS_CHECK:-1}"
    export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
    export GPU_MONITOR_DEVICES="${GPU_MONITOR_DEVICES:-all}"
    export GPU_MONITOR_LOG_INTERVAL="${GPU_MONITOR_LOG_INTERVAL:-15}"

    local REPO_ROOT
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    local MONITOR_PID="" LOG_REPORTER_PID=""
    local TEST_EXIT_CODE=0

    cleanup() {
        [[ -n "$LOG_REPORTER_PID" ]] && kill -0 "$LOG_REPORTER_PID" 2>/dev/null && kill "$LOG_REPORTER_PID" 2>/dev/null || true
        [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null && kill "$MONITOR_PID" 2>/dev/null || true
        if [[ -f "$GPU_MONITOR_DATA_ROOT/current_run_id" ]]; then
            echo "--- Finalizing: bundling GPU monitor data ---"
            local TMPF BUNDLE_LINE
            TMPF=$(mktemp)
            "$SCRIPT_DIR/gpu_monitor.sh" finalize 2>&1 | tee "$TMPF"
            BUNDLE_LINE=$(grep '^GPU_MONITOR_BUNDLE_DIR=' "$TMPF" || true)
            rm -f "$TMPF"
            if [[ -n "$BUNDLE_LINE" ]]; then
                eval "$BUNDLE_LINE"
                if [[ -d "${GPU_MONITOR_BUNDLE_DIR:-}" ]]; then
                    echo "--- GPU monitor bundle dir: $GPU_MONITOR_BUNDLE_DIR ---"
                    echo "--- Line chart: open in browser: $GPU_MONITOR_BUNDLE_DIR/report.html ---"
                    if command -v buildkite-agent &>/dev/null; then
                        echo "--- Uploading GPU monitor artifacts ---"
                        for f in "$GPU_MONITOR_BUNDLE_DIR"/*; do
                            [[ -e "$f" ]] && buildkite-agent artifact upload "$f"
                        done
                    fi
                fi
            fi
        fi
        exit "${TEST_EXIT_CODE:-0}"
    }
    trap cleanup EXIT

    if command -v nvidia-smi &>/dev/null; then
        "$SCRIPT_DIR/gpu_monitor.sh" start "$GPU_MONITOR_DEVICES" "$GPU_MONITOR_INTERVAL" &
        MONITOR_PID=$!
        echo "[GPU Monitor] Started (PID $MONITOR_PID), interval=${GPU_MONITOR_INTERVAL}s, devices=$GPU_MONITOR_DEVICES; log every ${GPU_MONITOR_LOG_INTERVAL}s."
    else
        echo "[GPU Monitor] nvidia-smi not found; skipping."
    fi

    (
        sleep 10
        while true; do
            sleep "$GPU_MONITOR_LOG_INTERVAL"
            local RID_FILE RUN_ID CSV LINE
            RID_FILE="$GPU_MONITOR_DATA_ROOT/current_run_id"
            [[ -f "$RID_FILE" ]] || continue
            RUN_ID=$(cat "$RID_FILE" 2>/dev/null)
            CSV="$GPU_MONITOR_DATA_ROOT/$RUN_ID/gpu_metrics.csv"
            [[ -f "$CSV" ]] || continue
            LINE=$(tail -1 "$CSV" 2>/dev/null)
            [[ -n "$LINE" ]] && echo "[GPU] $LINE"
        done
    ) &
    LOG_REPORTER_PID=$!

    (cd "$REPO_ROOT" && "${CMD[@]}") || TEST_EXIT_CODE=$?
    exit $TEST_EXIT_CODE
}

# ---------- dispatch ----------
case "$SUBCMD" in
    start)   cmd_start "$2" "$3" ;;
    finalize) cmd_finalize "$2" ;;
    run)     cmd_run "$@" ;;
    *)
        echo "Usage: $0 { start [gpu_ids] [interval] | finalize [run_id] | run -- <command> }" >&2
        echo "  start   - background nvidia-smi loop (CSV only)" >&2
        echo "  finalize - bundle current run, print GPU_MONITOR_BUNDLE_DIR=" >&2
        echo "  run     - start + command + finalize (+ CI upload)" >&2
        exit 1
        ;;
esac
