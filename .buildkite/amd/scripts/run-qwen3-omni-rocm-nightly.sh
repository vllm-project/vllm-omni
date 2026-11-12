#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

required_vars=(
    QWEN3_OMNI_ARTIFACT_DIR
    QWEN3_OMNI_EXPECTED_GPUS
    QWEN3_OMNI_EXPECTED_RESULTS
    QWEN3_OMNI_EXPECTED_TESTS
    QWEN3_OMNI_MARK_EXPRESSION
    QWEN3_OMNI_TEST_TARGET
    QWEN3_OMNI_TIMEOUT
)
for name in "${required_vars[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} is required." >&2
        exit 2
    fi
done

if [[ ! "${QWEN3_OMNI_EXPECTED_GPUS}" =~ ^[1-9][0-9]*$ ]] || \
    [[ ! "${QWEN3_OMNI_EXPECTED_RESULTS}" =~ ^[0-9]+$ ]] || \
    [[ ! "${QWEN3_OMNI_EXPECTED_TESTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: expected GPU/test counts must be positive integers and the result count non-negative." >&2
    exit 2
fi

checkout_path="${BUILDKITE_BUILD_CHECKOUT_PATH:?}"
case "${QWEN3_OMNI_ARTIFACT_DIR}" in
    "${checkout_path}"/artifacts/*) ;;
    *)
        echo "Error: QWEN3_OMNI_ARTIFACT_DIR must be under ${checkout_path}/artifacts." >&2
        exit 2
        ;;
esac

artifact_dir="${QWEN3_OMNI_ARTIFACT_DIR}"
benchmark_dir="${artifact_dir}/benchmark-results"
mkdir -p "${artifact_dir}" "${benchmark_dir}"

baseline_ready=0
test_started_epoch=0
# shellcheck disable=SC2329  # Invoked indirectly by the EXIT trap below.
capture_cleanup() {
    original_status=$?
    trap - EXIT
    set +e

    cleanup_status=0
    if ((baseline_ready)); then
        sleep 5
        ps -eo pid=,ppid=,lstart=,stat=,comm=,args= > "${artifact_dir}/processes-after.txt"
        rocm-smi --showpids > "${artifact_dir}/gpu-holders-after.txt" 2>&1 || true
        {
            if ((test_started_epoch)); then
                echo "runtime_seconds=$(( $(date +%s) - test_started_epoch ))"
            fi
            for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.max; do
                if [[ -r "${path}" ]]; then
                    echo "${path##*/}=$(<"${path}")"
                fi
            done
            rocm-smi --showmeminfo vram
        } > "${artifact_dir}/resource-usage-after.txt" 2>&1 || true

        awk '
            NR == FNR {
                if ($9 != "ps") {
                    baseline[$1 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7] = 1
                }
                next
            }
            $9 != "ps" {
                identity = $1 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7
                if (!(identity in baseline)) {
                    print
                    leaked = 1
                }
            }
            END {
                if (leaked) {
                    print "status=FAIL leaked_processes_detected=1"
                    exit 1
                }
                print "status=PASS leaked_processes=0"
            }
        ' "${artifact_dir}/processes-before.txt" \
            "${artifact_dir}/processes-after.txt" \
            2>&1 | tee "${artifact_dir}/process-cleanup.txt"
        cleanup_pipeline_status=("${PIPESTATUS[@]}")
        for pipeline_code in "${cleanup_pipeline_status[@]}"; do
            if ((pipeline_code != 0)); then
                cleanup_status=${pipeline_code}
                break
            fi
        done
    else
        echo "status=SKIP process_baseline_unavailable=1" > "${artifact_dir}/process-cleanup.txt"
    fi

    if ((original_status != 0)); then
        exit "${original_status}"
    fi
    exit "${cleanup_status}"
}
trap capture_cleanup EXIT

{
    echo "commit=$(git rev-parse HEAD)"
    echo "build_id=${BUILDKITE_BUILD_ID:-unset}"
    echo "job_id=${BUILDKITE_JOB_ID:-unset}"
    echo "agent_name=${BUILDKITE_AGENT_NAME:-unset}"
    echo "agent_queue=${BUILDKITE_AGENT_META_DATA_QUEUE:-unset}"
    echo "test_target=${QWEN3_OMNI_TEST_TARGET}"
    echo "config_file=${QWEN3_OMNI_CONFIG_FILE:-none}"
    echo "marks=${QWEN3_OMNI_MARK_EXPRESSION}"
    echo "expected_gpus=${QWEN3_OMNI_EXPECTED_GPUS}"
    echo "expected_tests=${QWEN3_OMNI_EXPECTED_TESTS}"
    echo "expected_results=${QWEN3_OMNI_EXPECTED_RESULTS}"
    echo "timeout=${QWEN3_OMNI_TIMEOUT}"
    for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.max; do
        if [[ -r "${path}" ]]; then
            echo "${path##*/}=$(<"${path}")"
        fi
    done
    echo "--- rocm-smi"
    rocm-smi
    echo "--- rocminfo"
    rocminfo
    echo "--- PyTorch"
    python3 - <<'PY'
import os

import torch

expected = int(os.environ["QWEN3_OMNI_EXPECTED_GPUS"])
actual = torch.cuda.device_count()
if not torch.version.hip:
    raise SystemExit("PyTorch is not a ROCm build")
if actual != expected:
    raise SystemExit(f"Expected {expected} ROCm GPUs, found {actual}")
print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
print(f"gpus={actual}")
for index in range(actual):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
PY
} > "${artifact_dir}/runtime-inventory.txt" 2>&1
python3 -m pip freeze > "${artifact_dir}/dependencies.txt"
ps -eo pid=,ppid=,lstart=,stat=,comm=,args= > "${artifact_dir}/processes-before.txt"
rocm-smi --showpids > "${artifact_dir}/gpu-holders-before.txt" 2>&1 || true
baseline_ready=1

pytest_args=("${QWEN3_OMNI_TEST_TARGET}")
if [[ -n "${QWEN3_OMNI_CONFIG_FILE:-}" ]]; then
    pytest_args+=(--test-config-file "${QWEN3_OMNI_CONFIG_FILE}")
    export BENCHMARK_DIR="${benchmark_dir}"
fi
pytest_args+=(
    -m "${QWEN3_OMNI_MARK_EXPRESSION}"
    --run-level full_model
)

set +e
bash .buildkite/amd/scripts/assert-pytest-collection.sh \
    "${QWEN3_OMNI_EXPECTED_TESTS}" "${pytest_args[@]}" \
    2>&1 | tee "${artifact_dir}/collection.txt"
collection_pipeline_status=("${PIPESTATUS[@]}")
set -e
for pipeline_code in "${collection_pipeline_status[@]}"; do
    if ((pipeline_code != 0)); then
        exit "${pipeline_code}"
    fi
done

set +e
test_started_epoch=$(date +%s)
timeout --signal=TERM --kill-after=2m "${QWEN3_OMNI_TIMEOUT}" \
    pytest -s -v -ra "${pytest_args[@]}" --durations=0 \
    --junitxml="${artifact_dir}/pytest.xml" \
    2>&1 | tee "${artifact_dir}/pytest.log"
test_pipeline_status=("${PIPESTATUS[@]}")
set -e
test_status=0
for pipeline_code in "${test_pipeline_status[@]}"; do
    if ((pipeline_code != 0)); then
        test_status=${pipeline_code}
        break
    fi
done
tail -n 200 "${artifact_dir}/pytest.log" > "${artifact_dir}/pytest-summary.txt" || true

set +e
python3 - "${artifact_dir}/pytest.xml" "${QWEN3_OMNI_EXPECTED_TESTS}" <<'PY' \
    2>&1 | tee "${artifact_dir}/pytest-result.txt"
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

xml_path = Path(sys.argv[1])
expected = int(sys.argv[2])
root = ET.parse(xml_path).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
passed = tests - failures - errors - skipped
print(f"tests={tests} passed={passed} failures={failures} errors={errors} skipped={skipped}")
if tests != expected or failures or errors or skipped:
    raise SystemExit(1)
PY
junit_pipeline_status=("${PIPESTATUS[@]}")
set -e
if ((test_status == 0)); then
    for pipeline_code in "${junit_pipeline_status[@]}"; do
        if ((pipeline_code != 0)); then
            test_status=5
            break
        fi
    done
fi

if [[ -n "${QWEN3_OMNI_CONFIG_FILE:-}" ]]; then
    find "${benchmark_dir}" -maxdepth 1 -type f -name 'result_*.json' -print | sort > \
        "${artifact_dir}/result-files.txt"
    if ((test_status == 0)); then
        result_count=$(wc -l < "${artifact_dir}/result-files.txt" | tr -d '[:space:]')
        if [[ "${result_count}" != "${QWEN3_OMNI_EXPECTED_RESULTS}" ]]; then
            echo "Expected ${QWEN3_OMNI_EXPECTED_RESULTS} benchmark results, found ${result_count}." >&2
            test_status=5
        else
            set +e
            python3 - "${artifact_dir}/result-files.txt" <<'PY' \
                2>&1 | tee "${artifact_dir}/result-validation.txt"
import json
import math
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
paths = [Path(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
for path in paths:
    result = json.loads(path.read_text(encoding="utf-8"))
    completed = result.get("completed")
    failed = result.get("failed", 0)
    if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
        raise SystemExit(f"{path}: invalid completed count {completed!r}")
    if not isinstance(failed, int) or isinstance(failed, bool) or failed != 0:
        raise SystemExit(f"{path}: benchmark failures {failed!r}")
    for key in ("request_throughput", "mean_ttft_ms", "mean_e2el_ms"):
        value = result.get(key)
        if value is not None and (
            not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        ):
            raise SystemExit(f"{path}: non-finite {key}={value!r}")
    print(f"result={path.name} completed={completed} failed={failed}")
PY
            validation_pipeline_status=("${PIPESTATUS[@]}")
            set -e
            for pipeline_code in "${validation_pipeline_status[@]}"; do
                if ((pipeline_code != 0)); then
                    test_status=5
                    break
                fi
            done
        fi
    fi
fi

exit "${test_status}"
