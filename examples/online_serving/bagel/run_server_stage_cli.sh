#!/bin/bash
# Bagel multi-stage online serving startup script.
#
# Usage:
#   ./run_server_stage_cli.sh --stage 0
#   ./run_server_stage_cli.sh --stage 1
#   ./run_server_stage_cli.sh --stage 0 -- --tensor-parallel-size 2
#   ./run_server_stage_cli.sh --stage 1 -- --gpu-memory-utilization 0.9
#
# By default, `--stage all` keeps the old behavior and launches both stages in
# one session. Use `--stage 0` / `--stage 1` to launch each stage separately in
# different terminal sessions, with stage-specific extra CLI arguments passed
# after `--`.

set -euo pipefail

MODEL="${MODEL:-ByteDance-Seed/BAGEL-7B-MoT}"
PORT="${PORT:-8091}"
MASTER_ADDRESS="${MASTER_ADDRESS:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-8092}"
STAGE="all"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-$SCRIPT_DIR/../../../vllm_omni/model_executor/stage_configs/bagel.yaml}"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS...]

Options:
  --stage {0|1|all}          Stage to launch (default: all)
  --model MODEL              Model name/path (default: $MODEL)
  --port PORT                API port for stage 0 (default: $PORT)
  --master-address ADDRESS   Master/orchestrator address (default: $MASTER_ADDRESS)
  --master-port PORT         Master/orchestrator port (default: $MASTER_PORT)
  --stage-configs-path PATH  Stage config YAML path (default: $STAGE_CONFIGS_PATH)
  --help                     Show this help message

Examples:
  $0 --stage 0
  $0 --stage 1
  $0 --stage 0 -- --tensor-parallel-size 2
  $0 --stage 1 -- --gpu-memory-utilization 0.9

Notes:
  - Use different terminal sessions to launch stage 0 and stage 1 separately.
  - Extra args after '--' are forwarded only to the selected stage.
  - When using '--stage all', the extra args are forwarded to both stages.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --master-address)
            MASTER_ADDRESS="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --stage-configs-path)
            STAGE_CONFIGS_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "$STAGE" != "0" && "$STAGE" != "1" && "$STAGE" != "all" ]]; then
    echo "Invalid --stage value: $STAGE" >&2
    usage
    exit 1
fi

print_config() {
    echo "Model: $MODEL"
    echo "API Port: $PORT"
    echo "Master Address: $MASTER_ADDRESS"
    echo "Master Port: $MASTER_PORT"
    echo "Stage Configs: $STAGE_CONFIGS_PATH"
    echo "Selected Stage: $STAGE"
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        echo "Extra Args: ${EXTRA_ARGS[*]}"
    fi
}

run_stage_0() {
    echo "Starting Stage 0 (Thinker) as master..."
    vllm serve "$MODEL" --omni \
        --port "$PORT" \
        --stage-configs-path "$STAGE_CONFIGS_PATH" \
        --stage-id 0 \
        -oma "$MASTER_ADDRESS" \
        -omp "$MASTER_PORT" \
        "${EXTRA_ARGS[@]}"
}

run_stage_1() {
    echo "Starting Stage 1 (DiT) in headless mode..."
    vllm serve "$MODEL" --omni \
        --stage-configs-path "$STAGE_CONFIGS_PATH" \
        --stage-id 1 \
        --headless \
        -oma "$MASTER_ADDRESS" \
        -omp "$MASTER_PORT" \
        "${EXTRA_ARGS[@]}"
}

echo "Starting Bagel multi-stage server..."
print_config

case "$STAGE" in
    0)
        run_stage_0
        ;;
    1)
        run_stage_1
        ;;
    all)
        echo "Launching both stages in one session (legacy mode)..."
        echo "Starting Stage 0 (Thinker) in background first..."
        run_stage_0 &
        STAGE_0_PID=$!

        cleanup() {
            if [[ -n "${STAGE_0_PID:-}" ]]; then
                kill "$STAGE_0_PID" 2>/dev/null || true
                wait "$STAGE_0_PID" 2>/dev/null || true
            fi
        }

        trap cleanup EXIT INT TERM

        echo "Waiting briefly for Stage 0 to initialize..."
        sleep 2
        run_stage_1
        ;;
esac
