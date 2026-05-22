#!/bin/bash
# Per-config worker for the SP-vs-PF benchmark. Runs inside an srun allocation
# sized to ${ngpus} GPUs. Starts the diffusion server, sweeps {resolution} x
# {num_frames} (2 reps each), then shuts the server down.
#
# Usage (invoked by benchmark_sp_vs_pf.sh):
#   _bench_one_config.sh <label> <ngpus> <port> <server_flags...>
#
# Inherited env (set by dispatcher or by the user):
#   RESULTS_DIR  output directory (per-run timestamped, shared across configs)
#   MODEL, STEPS, ONLY_SHAPES, SERVER_BOOT_TIMEOUT_S, REQUEST_TIMEOUT_S

set -uo pipefail

LABEL="${1:?label required}"
NGPUS="${2:?ngpus required}"
PORT="${3:?port required}"
shift 3
EXTRA_FLAGS=("$@")
SERVER_FLAGS=()
PIPEFUSION_WARMUP_STEPS=""
PIPEFUSION_SPLIT_DIM=""

while (($#)); do
    case "$1" in
        --pipefusion-warmup-steps=*)
            PIPEFUSION_WARMUP_STEPS="${1#*=}"
            shift
            ;;
        --pipefusion-warmup-steps)
            PIPEFUSION_WARMUP_STEPS="${2:?--pipefusion-warmup-steps requires a value}"
            shift 2
            ;;
        --pipefusion-split-dim=*)
            PIPEFUSION_SPLIT_DIM="${1#*=}"
            shift
            ;;
        --pipefusion-split-dim)
            PIPEFUSION_SPLIT_DIM="${2:?--pipefusion-split-dim requires a value}"
            shift 2
            ;;
        *)
            SERVER_FLAGS+=("$1")
            shift
            ;;
    esac
done

# Under sbatch, $0 / BASH_SOURCE point at a spool copy of the script,
# so resolve the project root via SLURM_SUBMIT_DIR. If that directory
# doesn't exist (common in containers with remapped volumes), trust PWD.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        SCRIPT_DIR="${PWD}"
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR must be set by dispatcher}"
MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
STEPS="${STEPS:-40}"
SERVER_BOOT_TIMEOUT_S="${SERVER_BOOT_TIMEOUT_S:-1200}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1800}"
RUN_SERVER_SH="${RUN_SERVER_SH:-${SCRIPT_DIR}/examples/online_serving/text_to_video/run_server.sh}"
BASE_URL="http://localhost:${PORT}"

# Per-config storage path so concurrent jobs don't collide on /tmp/storage.
VLLM_OMNI_STORAGE_PATH="${VLLM_OMNI_STORAGE_PATH:-${RESULTS_DIR}/storage_${LABEL}}"
mkdir -p "${VLLM_OMNI_STORAGE_PATH}"
export VLLM_OMNI_STORAGE_PATH

RESULTS_CSV="${RESULTS_DIR}/results_${LABEL}.csv"
SERVER_LOG="${RESULTS_DIR}/server_${LABEL}.log"

echo "[${LABEL}] running on $(hostname); GPUs=${NGPUS} port=${PORT}"
echo "[${LABEL}] flags: ${EXTRA_FLAGS[*]}"
echo "[${LABEL}] server flags: ${SERVER_FLAGS[*]}"
echo "[${LABEL}] request PipeFusion: warmup_steps=${PIPEFUSION_WARMUP_STEPS:-<runtime-default>} split_dim=${PIPEFUSION_SPLIT_DIM:-<runtime-default>}"

###############################################################################
# Environment (mirrors test_script_5B.sh)
###############################################################################
if nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "[${LABEL}] ERROR: GPUs are occupied by the following processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

# Export PYTHONPATH to ensure our modified vllm-omni code takes precedence
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    AVAILABLE_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"
    echo "[${LABEL}] CUDA_VISIBLE_DEVICES is set, using ${AVAILABLE_GPUS} GPUs"
else
    AVAILABLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi

if (( AVAILABLE_GPUS < NGPUS )); then
    echo "[${LABEL}] ERROR: need ${NGPUS} GPUs, only ${AVAILABLE_GPUS} visible"
    exit 1
fi

###############################################################################
# Sweep matrix
###############################################################################
RESOLUTIONS=("832x480" "1280x704" "1920x1088")
NUM_FRAMES=(41 81 121 161)

PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

in_csv_list() {
    local needle="$1" list="$2"
    [[ -z "${list}" ]] && return 0
    [[ ",${list}," == *",${needle},"* ]]
}

###############################################################################
# Server lifecycle
###############################################################################
SERVER_PID=""

start_server() {
    : > "${SERVER_LOG}"
    echo "[${LABEL}] starting server -> ${SERVER_LOG}"
    # setsid puts the server in its own process group so we can kill the tree.
    MODEL="${MODEL}" PORT="${PORT}" \
    setsid bash "${RUN_SERVER_SH}" \
        --enable-diffusion-pipeline-profiler \
        "${SERVER_FLAGS[@]}" \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "[${LABEL}] server PID=${SERVER_PID}"
}

wait_for_server() {
    local deadline=$(( SECONDS + SERVER_BOOT_TIMEOUT_S ))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[${LABEL}] ERROR: server died during boot (see ${SERVER_LOG})"
            return 1
        fi
        if curl -fsS --max-time 2 "${BASE_URL}/openapi.json" >/dev/null 2>&1; then
            echo "[${LABEL}] server ready"
            return 0
        fi
        sleep 5
    done
    echo "[${LABEL}] ERROR: server boot timeout (${SERVER_BOOT_TIMEOUT_S}s)"
    return 1
}

stop_server() {
    [[ -z "${SERVER_PID}" ]] && return 0
    echo "[${LABEL}] stopping server PID=${SERVER_PID}"
    kill -TERM "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "-${SERVER_PID}" 2>/dev/null || true
    # Best-effort: kill anything still bound to our port within this allocation.
    for pid in $(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $0}' \
                 | grep -oP 'pid=\K[0-9]+' | sort -u); do
        kill -KILL "${pid}" 2>/dev/null || true
    done
    SERVER_PID=""
}
trap 'stop_server; exit 130' INT TERM

###############################################################################
# Single request (async API + polling)
###############################################################################
run_request() {
    local size="$1" frames="$2"
    local create_resp
    local curl_form=(
        -H "Accept: application/json" \
        -F "prompt=${PROMPT}" \
        -F "num_frames=${frames}" \
        -F "size=${size}" \
        -F "negative_prompt=${NEG_PROMPT}" \
        -F "fps=16" \
        -F "num_inference_steps=${STEPS}" \
        -F "guidance_scale=4.0" \
        -F "guidance_scale_2=4.0" \
        -F "boundary_ratio=0.875" \
        -F "flow_shift=5.0" \
        -F "seed=42"
    )
    if [[ -n "${PIPEFUSION_WARMUP_STEPS}" ]]; then
        curl_form+=(-F "pipefusion_warmup_steps=${PIPEFUSION_WARMUP_STEPS}")
    fi
    if [[ -n "${PIPEFUSION_SPLIT_DIM}" ]]; then
        curl_form+=(-F "pipefusion_split_dim=${PIPEFUSION_SPLIT_DIM}")
    fi

    create_resp="$(curl -sS -X POST "${BASE_URL}/v1/videos" "${curl_form[@]}")" || true

    local vid
    vid="$(echo "${create_resp}" | jq -r '.id // empty' 2>/dev/null || true)"
    if [[ -z "${vid}" ]]; then
        echo "{\"status\":\"failed\",\"error\":\"create_failed\",\"raw\":$(jq -Rs . <<<"${create_resp}")}"
        return 1
    fi

    local deadline=$(( SECONDS + REQUEST_TIMEOUT_S ))
    while (( SECONDS < deadline )); do
        local status_resp status
        status_resp="$(curl -sS --max-time 30 "${BASE_URL}/v1/videos/${vid}" || true)"
        status="$(echo "${status_resp}" | jq -r '.status // "unknown"' 2>/dev/null || echo unknown)"
        case "${status}" in
            completed|failed)
                echo "${status_resp}"
                [[ "${status}" == "completed" ]]
                return $?
                ;;
            *)
                sleep 3
                ;;
        esac
    done
    echo "{\"status\":\"failed\",\"error\":\"timeout\",\"id\":\"${vid}\"}"
    return 1
}

###############################################################################
# CSV
###############################################################################
strategy="PF"
[[ " ${EXTRA_FLAGS[*]} " == *"--usp "* ]] && strategy="SP"
cfgp=1
[[ " ${EXTRA_FLAGS[*]} " == *"--cfg-parallel-size=2"* || " ${EXTRA_FLAGS[*]} " == *"--cfg-parallel-size 2"* ]] && cfgp=2

echo "config,ngpus,strategy,cfg_parallel,size,num_frames,rep,status,inference_time_s,peak_memory_mb,diffuse_s,vae_decode_s,text_encoder_s,video_id" > "${RESULTS_CSV}"

append_row() {
    local size="$1" frames="$2" rep="$3" json="$4"
    local status inf_t peak_mem diff_t vae_t te_t vid
    status="$(echo "${json}"   | jq -r '.status // "failed"' 2>/dev/null || echo failed)"
    inf_t="$(echo "${json}"    | jq -r '.inference_time_s // ""' 2>/dev/null)"
    peak_mem="$(echo "${json}" | jq -r '.peak_memory_mb // ""' 2>/dev/null)"
    diff_t="$(echo "${json}"   | jq -r '.stage_durations["Wan22Pipeline.diffuse"] // ""' 2>/dev/null)"
    vae_t="$(echo "${json}"    | jq -r '.stage_durations["Wan22Pipeline.vae.decode"] // ""' 2>/dev/null)"
    te_t="$(echo "${json}"     | jq -r '.stage_durations["Wan22Pipeline.text_encoder.forward"] // ""' 2>/dev/null)"
    vid="$(echo "${json}"      | jq -r '.id // ""' 2>/dev/null)"
    echo "${LABEL},${NGPUS},${strategy},${cfgp},${size},${frames},${rep},${status},${inf_t},${peak_mem},${diff_t},${vae_t},${te_t},${vid}" >> "${RESULTS_CSV}"
}

###############################################################################
# Main
###############################################################################
start_server
if ! wait_for_server; then
    stop_server
    exit 1
fi

first_test=true
for size in "${RESOLUTIONS[@]}"; do
    for frames in "${NUM_FRAMES[@]}"; do
        shape_key="${size}:${frames}"
        if ! in_csv_list "${shape_key}" "${ONLY_SHAPES:-}"; then
            continue
        fi
        if ${first_test}; then
            reps=(1 2)
            first_test=false
        else
            reps=(1)
        fi
        for rep in "${reps[@]}"; do
            echo "[${LABEL}] size=${size} frames=${frames} rep=${rep}"
            json="$(run_request "${size}" "${frames}")" || true
            echo "${json}" > "${RESULTS_DIR}/raw_${LABEL}_${size}_f${frames}_rep${rep}.json"
            append_row "${size}" "${frames}" "${rep}" "${json}"
        done
    done
done

stop_server
echo "[${LABEL}] done. CSV: ${RESULTS_CSV}"
