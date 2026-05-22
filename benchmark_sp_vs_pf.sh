#!/bin/bash
# Dispatcher for SP-vs-PF benchmark on Wan2.2-TI2V-5B-Diffusers.
# Submits one sbatch job per parallelism config (queued, fire-and-forget).
# If Slurm is not available, runs them locally and sequentially.
#
# Run:
#   ./benchmark_sp_vs_pf.sh
#
# Optional env vars:
#   USE_SLURM (0 or 1, default 1 if sbatch exists)
#   STEPS, RESULTS_DIR, ONLY_CONFIGS, ONLY_SHAPES,
#   SERVER_BOOT_TIMEOUT_S, REQUEST_TIMEOUT_S, MODEL,
#   PARTITION, JOB_PREFIX, TIME_LIMIT,
#   CONTAINER_IMAGE, CONTAINER_WORKDIR, CONTAINER_MOUNTS

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/bench_results/${TS}}"
mkdir -p "${RESULTS_DIR}"
export RESULTS_DIR

# Container settings
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/fq9hpsac/fq9hpsacuser05/enroot_images/vllm+vllm-omni+v0.20.0.sqsh}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/app/vllm-omni}"
# Default mounts: project root -> workdir, and huggingface cache
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SCRIPT_DIR}:${CONTAINER_WORKDIR},/scratch/fq9hpsac/fq9hpsacuser05/huggingface:/scratch/fq9hpsac/fq9hpsacuser05/huggingface}"

# Detect Slurm
HAS_SLURM=0
if command -v sbatch >/dev/null 2>&1; then
    HAS_SLURM=1
fi
USE_SLURM="${USE_SLURM:-${HAS_SLURM}}"

PARTITION="${PARTITION:-q-fq9hpsac}"
JOB_PREFIX="${JOB_PREFIX:-sp_vs_pf_bench}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

if [[ "${USE_SLURM}" == "1" ]]; then
    echo "=== SP-vs-PF benchmark dispatch (sbatch) ==="
    echo "partition:    ${PARTITION}"
    echo "time limit:   ${TIME_LIMIT}"
    echo "container:    ${CONTAINER_IMAGE}"
else
    echo "=== SP-vs-PF benchmark (local sequential) ==="
fi
echo "results dir:  ${RESULTS_DIR}"

JOBIDS_FILE="${RESULTS_DIR}/job_ids.txt"
: > "${JOBIDS_FILE}"

# in_csv_list needle list  (empty list = match all)
in_csv_list() {
    local needle="$1" list="$2"
    [[ -z "${list}" ]] && return 0
    [[ ",${list}," == *",${needle},"* ]]
}

dispatch() {
    local label="$1" ngpus="$2" port="$3"
    shift 3
    local extra=("$@")

    if ! in_csv_list "${label}" "${ONLY_CONFIGS:-}"; then
        echo ">>> skipping ${label} (filtered by ONLY_CONFIGS)"
        return 0
    fi

    local logfile="${RESULTS_DIR}/job_${label}.log"

    if [[ "${USE_SLURM}" == "1" ]]; then
        local submit
        submit="$(sbatch \
            --parsable \
            -p "${PARTITION}" \
            --job-name="${JOB_PREFIX}_${label}" \
            --gres="gpu:${ngpus}" \
            --cpus-per-gpu=24 \
            --mem-per-cpu=8G \
            --time="${TIME_LIMIT}" \
            --output="${logfile}" \
            --error="${logfile}" \
            --export=ALL \
            --container-image="${CONTAINER_IMAGE}" \
            --container-workdir="${CONTAINER_WORKDIR}" \
            --container-mounts="${CONTAINER_MOUNTS}" \
            "${SCRIPT_DIR}/_bench_one_config.sh" \
                "${label}" "${ngpus}" "${port}" "${extra[@]}")"
        if [[ -z "${submit}" ]]; then
            echo "!!! failed to submit ${label}"
            return 1
        fi
        echo ">>> submitted ${label} (${ngpus} GPU, port ${port}) JobID=${submit} -> ${logfile}"
        echo "${submit} ${label}" >> "${JOBIDS_FILE}"
    else
        echo ">>> running ${label} locally (${ngpus} GPU, port ${port}) ..."
        echo "    log: ${logfile}"
        # Run sequentially in the foreground, showing output
        bash "${SCRIPT_DIR}/_bench_one_config.sh" \
            "${label}" "${ngpus}" "${port}" "${extra[@]}" 2>&1 | tee "${logfile}"
        local exit_code=${PIPESTATUS[0]}
        if [[ ${exit_code} -ne 0 ]]; then
            echo "!!! ${label} failed with exit code ${exit_code}"
            return ${exit_code}
        fi
        echo ">>> ${label} completed"
    fi
}

# label        ngpus  port  flags...
dispatch PF-4         4 8001 --pipeline-parallel-size=4 --enable-pipefusion --pipefusion-split-dim=temporal --vae-patch-parallel-size=4
dispatch SP-4         4 8002 --usp 4 --vae-patch-parallel-size=4
dispatch PF-8         8 8003 --pipeline-parallel-size=8 --enable-pipefusion --pipefusion-split-dim=temporal --vae-patch-parallel-size=8
dispatch SP-8         8 8004 --usp 8 --vae-patch-parallel-size=8
dispatch PF-4xCFG2    8 8005 --pipeline-parallel-size=4 --enable-pipefusion --pipefusion-split-dim=temporal --cfg-parallel-size=2 --vae-patch-parallel-size=8
dispatch SP-4xCFG2    8 8006 --usp 4 --cfg-parallel-size=2 --vae-patch-parallel-size=8

echo
if [[ "${USE_SLURM}" == "1" ]]; then
    echo "=== all jobs submitted ==="
    echo "JobIDs:          ${JOBIDS_FILE}"
    echo "Track:   squeue -u \$USER --name=${JOB_PREFIX}_PF-4,${JOB_PREFIX}_SP-4,${JOB_PREFIX}_PF-8,${JOB_PREFIX}_SP-8,${JOB_PREFIX}_PF-4xCFG2,${JOB_PREFIX}_SP-4xCFG2"
    echo "Cancel:  scancel \$(awk '{print \$1}' ${JOBIDS_FILE})"
    echo
    echo "You can log out now."
else
    echo "=== all benchmarks completed ==="
fi
echo "results dir:     ${RESULTS_DIR}"
echo "per-config CSVs: ${RESULTS_DIR}/results_<label>.csv"
echo "per-config logs: ${RESULTS_DIR}/job_<label>.log"
echo
