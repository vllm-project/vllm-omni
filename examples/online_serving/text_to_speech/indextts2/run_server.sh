#!/bin/bash
# Launch vLLM-Omni server for IndexTTS 2.0 or 2.5.
#
# Usage from repository root:
#   examples/online_serving/text_to_speech/indextts2/run_server.sh
#   MODEL_VERSION=2.5 MODEL=/path/to/native/bundle \
#     examples/online_serving/text_to_speech/indextts2/run_server.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODEL_VERSION="${MODEL_VERSION:-2.0}"
PORT="${PORT:-8092}"

if [[ "$MODEL_VERSION" == "2.5" ]]; then
    if [[ -z "${MODEL:-}" ]]; then
        echo "Usage: MODEL_VERSION=2.5 MODEL=/path/to/native/bundle bash $0" >&2
        exit 2
    fi
    DEFAULT_DEPLOY_CONFIG="$ROOT_DIR/vllm_omni/deploy/indextts2_5.yaml"
elif [[ "$MODEL_VERSION" == "2.0" ]]; then
    MODEL="${MODEL:-IndexTeam/IndexTTS-2}"
    DEFAULT_DEPLOY_CONFIG="$ROOT_DIR/vllm_omni/deploy/indextts2.yaml"
else
    echo "MODEL_VERSION must be 2.0 or 2.5" >&2
    exit 2
fi

DEPLOY_CONFIG="${DEPLOY_CONFIG:-$DEFAULT_DEPLOY_CONFIG}"
MPS_MODE="${INDEXTTS_MPS:-auto}"

if [[ "$MPS_MODE" == "auto" ]]; then
    if [[ "$(basename -- "$DEPLOY_CONFIG")" == "indextts2_5_continuous.yaml" ]]; then
        MPS_MODE=1
    else
        MPS_MODE=0
    fi
fi

cleanup_mps() {
    if [[ "${MPS_OWNED:-0}" == 1 ]]; then
        echo quit | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    if [[ "${MPS_DIR_CREATED:-0}" == 1 ]]; then
        rm -rf -- "$MPS_ROOT"
    fi
}

if [[ "$MPS_MODE" == 1 ]]; then
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "INDEXTTS_MPS=1 requires nvidia-cuda-mps-control" >&2
        exit 2
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        echo "INDEXTTS_MPS=1 requires CUDA_VISIBLE_DEVICES to select exactly one physical GPU or UUID" >&2
        exit 2
    fi

    PHYSICAL_CUDA_DEVICE="$CUDA_VISIBLE_DEVICES"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "INDEXTTS_MPS=1 requires nvidia-smi to verify exclusive GPU access" >&2
        exit 2
    fi
    if ! ACTIVE_COMPUTE_PIDS="$(
        nvidia-smi --id="$PHYSICAL_CUDA_DEVICE" \
            --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
    )"; then
        echo "Unable to inspect physical GPU $PHYSICAL_CUDA_DEVICE before starting MPS" >&2
        exit 2
    fi
    if [[ -n "${ACTIVE_COMPUTE_PIDS//[[:space:]]/}" && "${INDEXTTS_MPS_ALLOW_SHARED_GPU:-0}" != 1 ]]; then
        echo "Refusing to start a private MPS daemon on busy GPU $PHYSICAL_CUDA_DEVICE (PIDs: $ACTIVE_COMPUTE_PIDS)" >&2
        echo "Use an exclusive GPU, disable MPS with INDEXTTS_MPS=0, or explicitly set INDEXTTS_MPS_ALLOW_SHARED_GPU=1" >&2
        exit 2
    fi

    MPS_ROOT="${INDEXTTS_MPS_DIR:-${TMPDIR:-/tmp}/vllm-omni-indextts-mps-${UID}-$$}"
    if [[ -e "$MPS_ROOT" ]]; then
        echo "Refusing to reuse existing MPS directory: $MPS_ROOT" >&2
        exit 2
    fi
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
    mkdir -p -- "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    MPS_DIR_CREATED=1
    chmod 700 "$MPS_ROOT" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    trap cleanup_mps EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    CUDA_VISIBLE_DEVICES="$PHYSICAL_CUDA_DEVICE" \
        nvidia-cuda-mps-control -d
    MPS_OWNED=1
    for _ in {1..100}; do
        [[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]] && break
        sleep 0.05
    done
    if [[ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
        echo "MPS daemon did not create its control socket" >&2
        exit 1
    fi
    # A single physical device managed by MPS is exposed to clients as GPU 0.
    export CUDA_VISIBLE_DEVICES=0
    echo "Scoped NVIDIA MPS enabled for physical device: $PHYSICAL_CUDA_DEVICE"
elif [[ "$MPS_MODE" != 0 ]]; then
    echo "INDEXTTS_MPS must be auto, 1, or 0" >&2
    exit 2
fi

echo "Starting IndexTTS $MODEL_VERSION server with model: $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni \
    --trust-remote-code \
    --deploy-config "$DEPLOY_CONFIG"
