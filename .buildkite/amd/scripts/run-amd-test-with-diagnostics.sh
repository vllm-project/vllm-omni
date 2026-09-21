#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

: "${AMD_CI_TEST_TARGET:?AMD_CI_TEST_TARGET is required}"
: "${AMD_CI_TEST_MARKER:?AMD_CI_TEST_MARKER is required}"
: "${AMD_CI_LOG_PREFIX:?AMD_CI_LOG_PREFIX is required}"
: "${AMD_CI_RUN_LEVEL:=full_model}"
: "${AMD_CI_TEST_TIMEOUT:=30m}"
: "${AMD_CI_LOG_DIR:=amd-ci-diagnostics}"
: "${AMD_CI_ARTIFACT_LOG_DIR:=amd-ci-diagnostics}"
: "${BUILDKITE_BUILD_CHECKOUT_PATH:?BUILDKITE_BUILD_CHECKOUT_PATH is required}"

mkdir -p "${AMD_CI_LOG_DIR}"
metadata_log="${AMD_CI_LOG_DIR}/${AMD_CI_LOG_PREFIX}-metadata.log"
collection_log="${AMD_CI_LOG_DIR}/${AMD_CI_LOG_PREFIX}-collection.log"

stage_diagnostic_logs() {
    local test_status=$?
    local stage_status=0
    local source_dir
    local destination_dir
    local -a logs

    trap - EXIT
    if ! mkdir -p "${BUILDKITE_BUILD_CHECKOUT_PATH}/${AMD_CI_ARTIFACT_LOG_DIR}"; then
        stage_status=1
    else
        source_dir=$(cd "${AMD_CI_LOG_DIR}" && pwd -P)
        destination_dir=$(
            cd "${BUILDKITE_BUILD_CHECKOUT_PATH}/${AMD_CI_ARTIFACT_LOG_DIR}" &&
                pwd -P
        )
        if [[ "${source_dir}" != "${destination_dir}" ]]; then
            shopt -s nullglob
            logs=(
                "${source_dir}"/*-collection.log
                "${source_dir}"/*-metadata.log
            )
            shopt -u nullglob
            if (( ${#logs[@]} > 0 )) &&
                ! cp -v "${logs[@]}" "${destination_dir}/"; then
                stage_status=1
            fi
        fi
    fi

    if (( test_status == 0 && stage_status != 0 )); then
        test_status=${stage_status}
    fi
    exit "${test_status}"
}
trap stage_diagnostic_logs EXIT

{
    echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "branch=${BUILDKITE_BRANCH:-unknown}"
    echo "job_id=${BUILDKITE_JOB_ID:-unknown}"
    echo "working_directory=${PWD}"
    echo "test_target=${AMD_CI_TEST_TARGET}"
    echo "test_marker=${AMD_CI_TEST_MARKER}"
    echo "run_level=${AMD_CI_RUN_LEVEL}"
    echo "expected_gpu_count=${VLLM_CI_EXPECTED_GPU_COUNT:-unknown}"
    echo "hf_home=${HF_HOME:-unset}"
    echo "hf_datasets_cache=${HF_DATASETS_CACHE:-unset}"
    echo "torchinductor_cache_dir=${TORCHINDUCTOR_CACHE_DIR:-unset}"
    echo "triton_cache_dir=${TRITON_CACHE_DIR:-unset}"
    echo "vllm_cache_root=${VLLM_CACHE_ROOT:-unset}"
    echo "xdg_cache_home=${XDG_CACHE_HOME:-unset}"
    echo "--- rocm-smi"
    rocm-smi || true
    echo "--- rocminfo"
    rocminfo || true
} >"${metadata_log}" 2>&1

echo "--- Collecting ${AMD_CI_TEST_TARGET} with -m '${AMD_CI_TEST_MARKER}'"
set +e
pytest --collect-only -q "${AMD_CI_TEST_TARGET}" \
    -m "${AMD_CI_TEST_MARKER}" \
    --run-level "${AMD_CI_RUN_LEVEL}" 2>&1 | tee "${collection_log}"
collection_status=${PIPESTATUS[0]}
set -e
if (( collection_status != 0 )); then
    echo "Test collection failed with status ${collection_status}." >&2
    exit "${collection_status}"
fi

collection_count=$(grep -Ec "::" "${collection_log}" || true)
if (( collection_count == 0 )); then
    echo "The selector collected no tests; refusing to start model execution." >&2
    exit 5
fi
echo "Collected ${collection_count} test item(s)."

echo "--- Running ${AMD_CI_TEST_TARGET}"
timeout --signal=TERM --kill-after=2m "${AMD_CI_TEST_TIMEOUT}" \
    pytest -s -v "${AMD_CI_TEST_TARGET}" \
    -m "${AMD_CI_TEST_MARKER}" \
    --run-level "${AMD_CI_RUN_LEVEL}"
