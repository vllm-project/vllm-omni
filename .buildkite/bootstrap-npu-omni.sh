#!/bin/bash
# vllm-omni NPU (Ascend) bootstrap
# Determines whether to run NPU CI based on file diff, then uploads the appropriate pipeline.
# Based on: bootstrap-intel-omni.sh + bootstrap-amd-omni.sh

set -euo pipefail

if [[ -z "${NIGHTLY:-}" ]]; then
    NIGHTLY=0
fi

if [[ -z "${DOCS_ONLY_DISABLE:-}" ]]; then
    DOCS_ONLY_DISABLE=0
fi

# --------------------------------------------------------------------------
# Helper: check PR label
# --------------------------------------------------------------------------
check_run_all_label() {
    RUN_ALL_LABEL="ready-run-all-tests"
    if [ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]; then
        PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm-omni/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')
        if [[ $PR_LABELS == *"$RUN_ALL_LABEL"* ]]; then
            echo true
        else
            echo false
        fi
    else
        echo false
    fi
}

# --------------------------------------------------------------------------
# Helper: compute file diff
# --------------------------------------------------------------------------
get_diff() {
    $(git add .)
    echo $(git diff --name-only --diff-filter=ACMDR $(git merge-base origin/main HEAD))
}

get_diff_main() {
    $(git add .)
    echo $(git diff --name-only --diff-filter=ACMDR HEAD~1)
}

# --------------------------------------------------------------------------
# Helper: upload pipeline based on branch
# --------------------------------------------------------------------------
upload_pipeline() {
    if [[ "${NIGHTLY}" == "1" ]]; then
        echo "--- 🌙 Uploading NPU nightly pipeline (L4)"
        buildkite-agent pipeline upload .buildkite/pipeline-npu-nightly.yaml
    elif [[ "${BUILDKITE_BRANCH}" == "main" ]]; then
        echo "--- 🔀 Uploading NPU merge pipeline (L3)"
        buildkite-agent pipeline upload .buildkite/pipeline-npu-merge.yaml
    else
        echo "--- 🔍 Uploading NPU ready pipeline (L2)"
        buildkite-agent pipeline upload .buildkite/pipeline-npu-ready.yaml
    fi
}

# --------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------

file_diff=$(get_diff)
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    file_diff=$(get_diff_main)
fi

# Early exit: skip if all changed files are under docs/
if [[ "${DOCS_ONLY_DISABLE}" != "1" ]] && [[ -n "${file_diff:-}" ]]; then
    docs_only=1
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        if [[ "$f" != docs/* ]]; then
            docs_only=0
            break
        fi
    done < <(printf '%s\n' "$file_diff" | tr ' ' '\n' | tr -d '\r')

    if [[ "$docs_only" -eq 1 ]]; then
        buildkite-agent annotate ":memo: NPU CI skipped — docs only" --style "info" --context "npu-skip" || true
        exit 0
    fi
fi

# Check for ready-run-all-tests label → force run
LABEL_RUN_ALL=$(check_run_all_label)
if [[ $LABEL_RUN_ALL == true ]]; then
    echo "Found 'ready-run-all-tests' label. Running NPU CI."
    upload_pipeline
    exit 0
fi

# Check if any changed file matches NPU-relevant paths
# Patterns that trigger NPU CI
npu_trigger_patterns=(
    "vllm_omni/platforms/npu/"
    "vllm_omni/engine/"
    "vllm_omni/core/"
    "vllm_omni/worker/"
    "vllm_omni/model_executor/"
    "vllm_omni/distributed/"
    "vllm_omni/config/"
    "vllm_omni/inputs/"
    "vllm_omni/entrypoints/"
    "vllm_omni/patch.py"
    "vllm_omni/request.py"
    "vllm_omni/outputs.py"
    "requirements/npu.txt"
    "requirements/common.txt"
    "docker/Dockerfile.npu"
    ".buildkite/pipeline-npu"
    ".buildkite/bootstrap-npu"
    ".buildkite/scripts/hardware_ci/run_npu_test.sh"
    "setup.py"
    "tests/"
)

# Patterns to explicitly ignore (even if they match a trigger pattern prefix)
npu_ignore_patterns=(
    "vllm_omni/platforms/cuda/"
    "vllm_omni/platforms/rocm/"
    "vllm_omni/platforms/xpu/"
    "docker/Dockerfile.ci"
    "docker/Dockerfile.rocm"
    "docker/Dockerfile.xpu"
)

should_run=0
for file in $file_diff; do
    # Check if file matches any trigger pattern
    matches_trigger=0
    for pattern in "${npu_trigger_patterns[@]}"; do
        if [[ $file == $pattern* ]] || [[ $file == $pattern ]]; then
            matches_trigger=1
            break
        fi
    done

    if [[ $matches_trigger -eq 1 ]]; then
        # Check it's not in ignore patterns
        matches_ignore=0
        for ignore in "${npu_ignore_patterns[@]}"; do
            if [[ $file == $ignore* ]] || [[ $file == $ignore ]]; then
                matches_ignore=1
                break
            fi
        done

        if [[ $matches_ignore -eq 0 ]]; then
            should_run=1
            echo "NPU CI triggered by: $file"
            break
        fi
    fi
done

if [[ $should_run -eq 1 ]]; then
    upload_pipeline
else
    buildkite-agent annotate ":fast_forward: NPU CI skipped — no NPU-relevant changes detected" --style "info" --context "npu-skip" || true
    echo "[npu-skip] No NPU-relevant file changes. Skipping NPU CI."
fi
