#!/bin/bash
#
# Stability resource monitor script (single entry point, extendable to CPU/NPU later)
# Only GPU monitoring is implemented for now; --backend reserves cpu/npu for future expansion.
#
# Subcommands: start | finalize | run -- <command>
#
# start    - collect data in the background (currently only gpu: `nvidia-smi` writes CSV)
# finalize - bundle the current run (CSV + report.html) and print BUNDLE_DIR
# run      - start monitoring -> execute command -> finalize (generate report.html only, no upload)
#
# Argument entry point (reserved for multiple backends):
#   --backend, -b   gpu | cpu | npu  default: gpu; only gpu is implemented right now
#
# Environment variables:
#   RESOURCE_MONITOR_DATA_ROOT     data root directory (compatible with GPU_MONITOR_DATA_ROOT)
#   RESOURCE_MONITOR_INTERVAL      sampling interval in seconds (compatible with GPU_MONITOR_INTERVAL)
#   RESOURCE_MONITOR_LOG_INTERVAL  log print interval in seconds (compatible with GPU_MONITOR_LOG_INTERVAL)
#   GPU_MONITOR_DEVICES            [GPU backend] GPU device IDs to monitor, e.g. 0,1 or all
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Data root directory: placed under the stability directory (the parent of SCRIPT_DIR)
# so it stays alongside stage_configs/tests.
STABILITY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${RESOURCE_MONITOR_DATA_ROOT:-${GPU_MONITOR_DATA_ROOT:-$STABILITY_DIR/gpu_monitor_data}}"
SUBCMD="${1:-}"

# Parse optional --backend|-b arguments, store the result in BACKEND,
# and keep the remaining positional arguments in the REST_ARGS array.
parse_backend_and_rest() {
    BACKEND="gpu"
    REST_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backend|-b)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --backend requires a value (gpu|cpu|npu)" >&2
                    exit 1
                fi
                BACKEND="$2"
                shift 2
                ;;
            *)
                REST_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# ---------- subcommand: start ----------
cmd_start() {
    parse_backend_and_rest "$@"
    local POS1="${REST_ARGS[0]:-${GPU_MONITOR_DEVICES:-all}}"
    local POS2="${REST_ARGS[1]:-${RESOURCE_MONITOR_INTERVAL:-${GPU_MONITOR_INTERVAL:-5}}}"

    case "$BACKEND" in
        gpu)
            start_backend_gpu "$POS1" "$POS2"
            ;;
        cpu|npu)
            echo "Error: backend '$BACKEND' not implemented yet. Currently only 'gpu' is supported." >&2
            exit 1
            ;;
        *)
            echo "Error: invalid backend '$BACKEND'. Use gpu|cpu|npu." >&2
            exit 1
            ;;
    esac
}

start_backend_gpu() {
    local GPU_IDS_RAW="${1:-${GPU_MONITOR_DEVICES:-all}}"
    local INTERVAL="${2:-${RESOURCE_MONITOR_INTERVAL:-${GPU_MONITOR_INTERVAL:-5}}}"

    [[ "$INTERVAL" =~ ^[0-9]+$ ]] && [[ "$INTERVAL" -ge 1 ]] || {
        echo "Error: interval must be a positive integer (seconds)"
        echo "Usage: $0 start [--backend gpu|cpu|npu] [gpu_ids] [interval_seconds]"
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
    echo "Stability resource monitor (backend=gpu) started"
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

# ---------- subcommand: finalize ----------
cmd_finalize() {
    parse_backend_and_rest "$@"
    local RUN_ID="${REST_ARGS[0]:-}"

    case "$BACKEND" in
        gpu)
            finalize_backend_gpu "$RUN_ID"
            ;;
        cpu|npu)
            echo "Error: backend '$BACKEND' not implemented yet. Currently only 'gpu' is supported." >&2
            exit 1
            ;;
        *)
            echo "Error: invalid backend '$BACKEND'. Use gpu|cpu|npu." >&2
            exit 1
            ;;
    esac
}

finalize_backend_gpu() {
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
Stability resource monitor (gpu) bundle - ${RUN_ID}
- gpu_metrics.csv: raw samples
- report.html: report with charts (open in browser to view)
EOF

    local BUNDLE_ABS
    BUNDLE_ABS=$(cd "$BUNDLE_DIR" && pwd)
    echo "GPU_MONITOR_BUNDLE_DIR=$BUNDLE_ABS"
    echo "RESOURCE_MONITOR_BUNDLE_DIR=$BUNDLE_ABS"
    echo "Archive path: $BUNDLE_ABS"
}

# ---------- subcommand: run ----------
cmd_run() {
    shift
    local BACKEND="gpu"
    local CMD=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backend|-b)
                if [[ -n "${2:-}" ]]; then
                    BACKEND="$2"
                    shift 2
                else
                    shift
                fi
                ;;
            --)
                shift
                CMD=("$@")
                break
                ;;
            *)
                shift
                ;;
        esac
    done
    if [[ ${#CMD[@]} -eq 0 ]]; then
        echo "Usage: $0 run [--backend gpu|cpu|npu] -- <command to run>" >&2
        exit 1
    fi

    case "$BACKEND" in
        gpu)
            run_backend_gpu "${CMD[@]}"
            ;;
        cpu|npu)
            echo "Error: backend '$BACKEND' not implemented yet. Currently only 'gpu' is supported." >&2
            exit 1
            ;;
        *)
            echo "Error: invalid backend '$BACKEND'. Use gpu|cpu|npu." >&2
            exit 1
            ;;
    esac
}

run_backend_gpu() {
    local CMD=("$@")
    export RESOURCE_MONITOR_DATA_ROOT="${RESOURCE_MONITOR_DATA_ROOT:-${GPU_MONITOR_DATA_ROOT:-$STABILITY_DIR/gpu_monitor_data}}"
    export GPU_MONITOR_DATA_ROOT="${GPU_MONITOR_DATA_ROOT:-$RESOURCE_MONITOR_DATA_ROOT}"
    export RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-${GPU_MONITOR_INTERVAL:-5}}"
    export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-${GPU_MONITOR_INTERVAL:-5}}"
    export GPU_MONITOR_DEVICES="${GPU_MONITOR_DEVICES:-all}"
    export RESOURCE_MONITOR_LOG_INTERVAL="${RESOURCE_MONITOR_LOG_INTERVAL:-${GPU_MONITOR_LOG_INTERVAL:-15}}"
    export GPU_MONITOR_LOG_INTERVAL="${RESOURCE_MONITOR_LOG_INTERVAL:-${GPU_MONITOR_LOG_INTERVAL:-15}}"

    local REPO_ROOT
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    local MONITOR_PID="" LOG_REPORTER_PID=""
    local TEST_EXIT_CODE=0

    cleanup() {
        [[ -n "$LOG_REPORTER_PID" ]] && kill -0 "$LOG_REPORTER_PID" 2>/dev/null && kill "$LOG_REPORTER_PID" 2>/dev/null || true
        [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null && kill "$MONITOR_PID" 2>/dev/null || true
        if [[ -f "$GPU_MONITOR_DATA_ROOT/current_run_id" ]]; then
            echo "--- Finalizing: bundling resource (gpu) monitor data ---"
            local TMPF BUNDLE_LINE
            TMPF=$(mktemp)
            "$SCRIPT_DIR/resource_monitor.sh" finalize --backend gpu 2>&1 | tee "$TMPF"
            BUNDLE_LINE=$(grep '^GPU_MONITOR_BUNDLE_DIR=' "$TMPF" || true)
            rm -f "$TMPF"
            if [[ -n "$BUNDLE_LINE" ]]; then
                eval "$BUNDLE_LINE"
                if [[ -d "${GPU_MONITOR_BUNDLE_DIR:-}" ]]; then
                    echo "--- Resource monitor bundle dir: $GPU_MONITOR_BUNDLE_DIR ---"
                    echo "--- Line chart: open in browser: $GPU_MONITOR_BUNDLE_DIR/report.html ---"
                fi
            fi
        fi
        exit "${TEST_EXIT_CODE:-0}"
    }
    trap cleanup EXIT

    if command -v nvidia-smi &>/dev/null; then
        "$SCRIPT_DIR/resource_monitor.sh" start --backend gpu "$GPU_MONITOR_DEVICES" "$RESOURCE_MONITOR_INTERVAL" &
        MONITOR_PID=$!
        echo "[Resource Monitor (gpu)] Started (PID $MONITOR_PID), interval=${RESOURCE_MONITOR_INTERVAL}s, devices=$GPU_MONITOR_DEVICES; log every ${RESOURCE_MONITOR_LOG_INTERVAL}s."
    else
        echo "[Resource Monitor (gpu)] nvidia-smi not found; skipping."
    fi

    (
        sleep 10
        while true; do
            sleep "$RESOURCE_MONITOR_LOG_INTERVAL"
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
    start)   cmd_start "${@:2}" ;;
    finalize) cmd_finalize "${@:2}" ;;
    run)     cmd_run "$@" ;;
    *)
        echo "Usage: $0 { start [--backend gpu|cpu|npu] [gpu_ids] [interval] | finalize [--backend gpu|cpu|npu] [run_id] | run [--backend gpu|cpu|npu] -- <command> }" >&2
        echo "  start   - background monitor (currently only backend=gpu)" >&2
        echo "  finalize - bundle current run, print GPU_MONITOR_BUNDLE_DIR= / RESOURCE_MONITOR_BUNDLE_DIR=" >&2
        echo "  run     - start + command + finalize (generate report.html only)" >&2
        echo "  --backend gpu|cpu|npu  (default: gpu; cpu/npu reserved for future use)" >&2
        exit 1
        ;;
esac
