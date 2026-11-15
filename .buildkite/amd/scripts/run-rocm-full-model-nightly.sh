#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# One-GPU, real-weight functional coverage. Run in an isolated AMD CI pod.
set -euo pipefail

test_target=${1:?test file required}
expected_tests=${2:?expected test count required}
model_kind=${3:?omni, tts or diffusion required}
artifact_name=${4:?artifact directory name required}
shift 4
test_targets=("$test_target" "$@")
level_marker=${ROCM_E2E_LEVEL_MARKER:-full_model}
run_level=${ROCM_E2E_RUN_LEVEL:-full_model}
test_timeout_minutes=${ROCM_E2E_TIMEOUT_MINUTES:-105}
[[ "$expected_tests" =~ ^[1-9][0-9]*$ ]]
[[ "$model_kind" == omni || "$model_kind" == tts || "$model_kind" == diffusion ]]
[[ "$artifact_name" =~ ^[a-z0-9-]+$ ]]
[[ "$level_marker" == full_model || "$level_marker" == advanced_model || "$level_marker" == slow || "$level_marker" == all ]]
[[ "$run_level" == core_model || "$run_level" == advanced_model || "$run_level" == full_model ]]
[[ "$test_timeout_minutes" =~ ^[1-9][0-9]*$ ]]
if [[ "$level_marker" == all ]]; then
    marker_expression="$model_kind and rocm and MI325 and cards_1 and not nvidia_only"
else
    marker_expression="$level_marker and $model_kind and rocm and MI325 and cards_1 and not nvidia_only"
fi
artifact_dir="${BUILDKITE_BUILD_CHECKOUT_PATH:?}/artifacts/${artifact_name}"
mkdir -p "$artifact_dir"

# Fixtures own server shutdown. Independently detect lingering processes, even
# after pytest fails or times out; preserve the original failure status.
baseline_ready=0
# shellcheck disable=SC2317,SC2329  # Invoked by the EXIT trap.
check_cleanup() {
    original_status=$?
    trap - EXIT
    set +e
    cleanup_status=0
    if ((baseline_ready)); then
        sleep 5
        ps -eo pid=,ppid=,lstart=,stat=,comm=,args= > "$artifact_dir/processes-after.txt"
        cleanup_status=$?
        if ((cleanup_status == 0)); then
            awk '
                NR == FNR {
                    if ($9 != "ps") baseline[$1 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7] = 1
                    next
                }
                $9 != "ps" && !(($1 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7) in baseline) {
                    print; leaked = 1
                }
                END { print "leaked_processes=" (leaked ? 1 : 0); exit leaked }
            ' "$artifact_dir/processes-before.txt" "$artifact_dir/processes-after.txt" \
                > "$artifact_dir/process-cleanup.txt"
            cleanup_status=$?
            cat "$artifact_dir/process-cleanup.txt"
        fi
        rocm-smi --showpids > "$artifact_dir/gpu-holders-after.txt" 2>&1 || true
    fi
    if ((original_status != 0)); then
        exit "$original_status"
    fi
    exit "$cleanup_status"
}
trap check_cleanup EXIT

{
    printf 'buildkite_commit=%s\n' "${BUILDKITE_COMMIT:?}"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'checkout_head=%s\n' "$(git rev-parse HEAD)"
    fi
    printf 'target=%s\n' "${test_targets[@]}"
    echo "expected_tests=$expected_tests marker=$level_marker run_level=$run_level timeout=${test_timeout_minutes}m"
    rocminfo
    python3 - <<'PY'
import torch

assert torch.version.hip, "A ROCm PyTorch build is required"
assert torch.cuda.device_count() == 1, "Exactly one ROCm GPU is required"
print(f"torch={torch.__version__} hip={torch.version.hip} gpu={torch.cuda.get_device_name(0)}")
PY
} > "$artifact_dir/runtime-inventory.txt" 2>&1
python3 -m pip freeze > "$artifact_dir/dependencies.txt"
ps -eo pid=,ppid=,lstart=,stat=,comm=,args= > "$artifact_dir/processes-before.txt"
baseline_ready=1

pytest_args=(
    "${test_targets[@]}"
    -m "$marker_expression"
    --run-level "$run_level" --shard-id 0 --num-shards 1
)
timeout --signal=TERM --kill-after=1m 10m \
    bash .buildkite/amd/scripts/assert-pytest-collection.sh "$expected_tests" "${pytest_args[@]}" \
    2>&1 | tee "$artifact_dir/collection.txt"

set +e
timeout --signal=TERM --kill-after=2m "${test_timeout_minutes}m" \
    pytest -s -v -ra "${pytest_args[@]}" --durations=0 --junitxml="$artifact_dir/pytest.xml" \
    2>&1 | tee "$artifact_dir/pytest.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
test_status=0
for status in "${pipeline_status[@]}"; do
    if ((status != 0)); then
        test_status=$status
        break
    fi
done

set +e
python3 - "$artifact_dir/pytest.xml" "$expected_tests" <<'PY' \
    > "$artifact_dir/pytest-result.txt" 2>&1
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
print(counts)
if counts["tests"] != int(sys.argv[2]) or any(counts[key] for key in ("failures", "errors", "skipped")):
    raise SystemExit(5)
PY
junit_status=$?
set -e
cat "$artifact_dir/pytest-result.txt"
if ((test_status != 0)); then
    exit "$test_status"
fi
exit "$junit_status"
