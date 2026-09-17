#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Run one configured Qwen3-Omni case with fail-closed collection checks.
set -euo pipefail

: "${QWEN3_TEST_TARGET:?QWEN3_TEST_TARGET is required}"
: "${QWEN3_TEST_MARKER:?QWEN3_TEST_MARKER is required}"
: "${QWEN3_LOG_PREFIX:?QWEN3_LOG_PREFIX is required}"
: "${QWEN3_RUN_LEVEL:=advanced_model}"
: "${QWEN3_TEST_TIMEOUT:=30m}"
: "${QWEN3_OMNI_MODEL:=Qwen/Qwen3-Omni-30B-A3B-Instruct}"
: "${QWEN3_OMNI_REVISION:=main}"
: "${HF_HOME:=/home/buildkite-agent/huggingface}"
: "${QWEN3_LOG_DIR:=qwen3-omni-ci}"

mkdir -p "${QWEN3_LOG_DIR}"
metadata_log="${QWEN3_LOG_DIR}/${QWEN3_LOG_PREFIX}-metadata.log"
collection_log="${QWEN3_LOG_DIR}/${QWEN3_LOG_PREFIX}-collection.log"
test_log="${QWEN3_LOG_DIR}/${QWEN3_LOG_PREFIX}-pytest.log"

{
    echo "model=${QWEN3_OMNI_MODEL}"
    echo "revision=${QWEN3_OMNI_REVISION}"
    echo "hf_home=${HF_HOME}"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "branch=${BUILDKITE_BRANCH:-unknown}"
    echo "expected_gpu_count=${VLLM_CI_EXPECTED_GPU_COUNT:-unknown}"
    echo "--- rocm-smi"
    rocm-smi || true
    echo "--- rocminfo"
    rocminfo || true
} >"${metadata_log}" 2>&1

echo "--- Collecting ${QWEN3_TEST_TARGET} with -m '${QWEN3_TEST_MARKER}'"
set +e
pytest --collect-only -q "${QWEN3_TEST_TARGET}" \
    -m "${QWEN3_TEST_MARKER}" \
    --run-level "${QWEN3_RUN_LEVEL}" 2>&1 | tee "${collection_log}"
collection_status=${PIPESTATUS[0]}
set -e
if [[ ${collection_status} -ne 0 ]]; then
    echo "Qwen3 collection failed with status ${collection_status}." >&2
    exit "${collection_status}"
fi

collection_count=$(grep -c "::" "${collection_log}" || true)
if ! [[ "${collection_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Qwen3 selector collected no tests; refusing to run an empty smoke." >&2
    exit 5
fi
echo "Qwen3 collected ${collection_count} test item(s)."

echo "--- Running ${QWEN3_TEST_TARGET}"
set +e
timeout --signal=TERM --kill-after=2m "${QWEN3_TEST_TIMEOUT}" \
    pytest -s -v "${QWEN3_TEST_TARGET}" \
    -m "${QWEN3_TEST_MARKER}" \
    --run-level "${QWEN3_RUN_LEVEL}" 2>&1 | tee "${test_log}"
test_status=${PIPESTATUS[0]}
set -e
exit "${test_status}"
