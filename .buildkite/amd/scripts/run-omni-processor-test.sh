#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -uo pipefail

artifact_root="${BUILDKITE_BUILD_CHECKOUT_PATH:-$PWD}/artifacts/omni-processor"
mkdir -p "${artifact_root}" || exit 1

print_file() {
    local label=$1
    local path=$2
    if [[ -r "${path}" ]]; then
        printf '%s=' "${label}"
        tr '\n' ';' <"${path}"
        printf '\n'
    fi
}

record_cpu_context() {
    local point=$1
    printf 'point=%s\n' "${point}"
    printf 'utc='
    date -u '+%Y-%m-%dT%H:%M:%SZ' || true
    if command -v taskset >/dev/null 2>&1; then
        taskset -pc "$$" || true
    fi
    awk '/^(Cpus_allowed_list|Mems_allowed_list):/' /proc/self/status || true
    print_file cgroup_cpuset /sys/fs/cgroup/cpuset.cpus.effective
    print_file cgroup_cpu_max /sys/fs/cgroup/cpu.max
    print_file cgroup_v1_cpuset /sys/fs/cgroup/cpuset/cpuset.cpus
    print_file cgroup_v1_quota /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    print_file cgroup_v1_period /sys/fs/cgroup/cpu/cpu.cfs_period_us
    print_file load /proc/loadavg
    print_file cpu_pressure /proc/pressure/cpu
}

record_cpu_context before >"${artifact_root}/cpu-context.log" 2>&1 || true

pytest -s -v \
    tests/model_executor/models/test_omni_processing.py \
    -m 'core_model and cpu and omni' \
    --durations=0 \
    --junitxml="${artifact_root}/pytest.xml" \
    2>&1 | tee "${artifact_root}/pytest.log"
pipeline_status=("${PIPESTATUS[@]}")
pytest_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}

record_cpu_context after >>"${artifact_root}/cpu-context.log" 2>&1 || true

if ((pytest_status != 0)); then
    exit "${pytest_status}"
fi
exit "${tee_status}"
