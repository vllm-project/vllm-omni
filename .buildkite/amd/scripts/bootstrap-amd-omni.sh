#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# vllm-omni customized version
# Based on: https://github.com/vllm-project/ci-infra/blob/main/buildkite/bootstrap-amd.sh
# Last synced: 2025-12-15
# Modifications: Use local template file instead of downloading from ci-infra

set -euo pipefail

# The bootstrap runs from the repository root; ShellCheck does not follow
# sourced files unless invoked with -x.
# shellcheck disable=SC1091
source .buildkite/common/scripts/resolve_skip_ci.sh

AMD_SUITE_SELECTOR=".buildkite/amd/scripts/select_test_suites.py"

if [[ -z "${RUN_ALL:-}" ]]; then
    RUN_ALL=0
fi

if [[ -z "${NIGHTLY:-}" ]]; then
    NIGHTLY=0
fi

if [[ -z "${VLLM_CI_BRANCH:-}" ]]; then
    VLLM_CI_BRANCH="main"
fi

if [[ -z "${AMD_MIRROR_HW:-}" ]]; then
    AMD_MIRROR_HW="amdproduction"
fi

PR_LABELS=""
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
    if ! PR_LABELS=$(curl -fsSL \
        "https://api.github.com/repos/vllm-project/vllm-omni/pulls/$BUILDKITE_PULL_REQUEST" \
        | jq -r '.labels[].name'); then
        echo "ERROR: Could not read PR labels; refusing to select an incorrect AMD test tier." >&2
        exit 1
    fi
fi

fail_fast() {
    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        echo true
    else
        echo false  # not a PR or BUILDKITE_PULL_REQUEST not set
    fi
}

if [[ -z "${COV_ENABLED:-}" ]]; then
    COV_ENABLED=0
fi

resolve_test_specs() {
    local selected_specs
    if ! selected_specs=$(python3 "$AMD_SUITE_SELECTOR" \
        --branch "$BUILDKITE_BRANCH" \
        --labels "$PR_LABELS" \
        --debug-test-yaml "${DEBUG_TEST_YAML:-}" \
        --nightly "$NIGHTLY"); then
        exit 1
    fi

    TEST_SPECS=()
    while IFS= read -r suite_spec; do
        [[ -z "$suite_spec" ]] && continue
        TEST_SPECS+=("$suite_spec")
    done <<< "$selected_specs"
}

filter_test_specs_by_skip_ci() {
    local suite_spec level decision
    local skip_all=0
    local -a runnable_specs=()

    for suite_spec in "${TEST_SPECS[@]}"; do
        case "$suite_spec" in
            READY_TESTS:*) level="l2" ;;
            MERGE_TESTS:*) level="l3" ;;
            NIGHTLY_TESTS:*)
                # Explicit PR and scheduled nightly selections must survive
                # L2/L3 diff gating, including docs-only main commits.
                runnable_specs+=("$suite_spec")
                continue
                ;;
            *)
                echo "ERROR: unknown AMD test suite spec '$suite_spec'" >&2
                exit 1
                ;;
        esac

        if decision=$(python3 "$SKIP_CI_PY" gate amd "$level"); then
            echo "Skipping AMD $level suite due to $decision decision."
            [[ "$decision" == "skip-all" ]] && skip_all=1
        else
            runnable_specs+=("$suite_spec")
        fi
    done

    TEST_SPECS=("${runnable_specs[@]}")
    if [[ ${#TEST_SPECS[@]} -eq 0 ]]; then
        if [[ $skip_all -eq 1 ]]; then
            buildkite-agent annotate \
                ":memo: CI skipped — docs or pytest skip-mark changes only" \
                --style "info" 2>/dev/null || true
        fi
        echo "No AMD suites remain after skip-ci filtering."
        exit 0
    fi
}

upload_pipeline() {
    echo "Uploading pipeline..."
    # Install minijinja
    ls .buildkite || buildkite-agent annotate --style error 'Please merge upstream main branch for buildkite CI'
    curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
    # Installed by the minijinja bootstrap above and only present on the CI agent.
    # shellcheck disable=SC1091
    source /var/lib/buildkite-agent/.cargo/env

    if [[ $BUILDKITE_PIPELINE_SLUG == "fastcheck" ]]; then
        AMD_MIRROR_HW="amdtentative"
    fi

    # Use local template file for vllm-omni
    cp .buildkite/amd/test-template-amd-omni.j2 .buildkite/amd/test-template.j2


    # (WIP) Use pipeline generator instead of jinja template
    if [ -e ".buildkite/amd/pipeline_generator/pipeline_generator.py" ]; then
        python -m pip install click pydantic
        python .buildkite/amd/pipeline_generator/pipeline_generator.py --run_all="$RUN_ALL" --list_file_diff="$LIST_FILE_DIFF" --nightly="$NIGHTLY" --mirror_hw="$AMD_MIRROR_HW"
        buildkite-agent pipeline upload .buildkite/amd/pipeline.yaml
        exit 0
    fi
    echo "List file diff: $LIST_FILE_DIFF"
    echo "Run all: $RUN_ALL"
    echo "Nightly: $NIGHTLY"
    echo "AMD Mirror HW: $AMD_MIRROR_HW"

    FAIL_FAST=$(fail_fast)

    cd .buildkite/amd

    # Multiple label-selected or debug-selected suites share one image build.
    if [[ ${#TEST_SPECS[@]} -eq 1 ]]; then
        TEST_YAML="${TEST_SPECS[0]#*:}"
    else
        TEST_YAML=$(mktemp "${TMPDIR:-/tmp}/amd-selected-tests.XXXXXX.yml")
        python - "$TEST_YAML" "${TEST_SPECS[@]}" <<'PY'
import sys

import yaml


output_path, *suite_specs = sys.argv[1:]
combined = {"env": {}, "steps": []}

for suite_spec in suite_specs:
    group_name, input_path = suite_spec.split(":", 1)
    with open(input_path, encoding="utf-8") as test_file:
        suite = yaml.safe_load(test_file)

    for name, value in (suite.get("env") or {}).items():
        previous = combined["env"].get(name, value)
        if previous != value:
            raise ValueError(
                f"Conflicting environment value for {name}: {previous!r} != {value!r}"
            )
        combined["env"][name] = value

    suite_steps = []
    for entry in suite.get("steps") or []:
        if "group" in entry:
            suite_steps.extend(entry.get("steps") or [])
        else:
            suite_steps.append(entry)
    combined["steps"].append({"group": group_name, "steps": suite_steps})

with open(output_path, "w", encoding="utf-8") as output_file:
    yaml.safe_dump(combined, output_file, sort_keys=False)
PY
    fi
    echo "AMD test suites: ${TEST_SPECS[*]}"

    (
        set -x
        # Output pipeline.yaml with all blank lines removed
        minijinja-cli test-template.j2 "$TEST_YAML" \
            -D branch="$BUILDKITE_BRANCH" \
            -D list_file_diff="$LIST_FILE_DIFF" \
            -D run_all="$RUN_ALL" \
            -D nightly="$NIGHTLY" \
            -D mirror_hw="$AMD_MIRROR_HW" \
            -D fail_fast="$FAIL_FAST" \
            -D vllm_use_precompiled="$VLLM_USE_PRECOMPILED" \
            -D vllm_merge_base_commit="$(git merge-base origin/main HEAD)" \
            -D cov_enabled="$COV_ENABLED" \
            -D vllm_ci_branch="$VLLM_CI_BRANCH" \
            | sed '/^[[:space:]]*$/d' \
            > pipeline.yaml
    )
    cat pipeline.yaml
    if [[ "$TEST_YAML" == "${TMPDIR:-/tmp}/amd-selected-tests."*.yml ]]; then
        rm -f -- "$TEST_YAML"
    fi
    buildkite-agent artifact upload pipeline.yaml
    buildkite-agent pipeline upload pipeline.yaml
    exit 0
}

get_diff() {
    git diff --name-only --diff-filter=ACMDR "$(git merge-base origin/main HEAD)"
}

get_diff_main() {
    git diff --name-only --diff-filter=ACMDR HEAD~1
}

file_diff=$(get_diff)
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    file_diff=$(get_diff_main)
fi

# Resolve PR tier labels before skip-ci so L2/L3 suites can be filtered
# independently while an explicitly selected L4 suite passes through.
resolve_test_specs
filter_test_specs_by_skip_ci

patterns=(
    "docker/Dockerfile"
    "CMakeLists.txt"
    "requirements/common.txt"
    "requirements/cuda.txt"
    "requirements/build.txt"
    "requirements/test.txt"
    "setup.py"
    "csrc/"
    "cmake/"
)

ignore_patterns=(
    "docker/Dockerfile."
    "csrc/cpu"
    "csrc/rocm"
    "cmake/hipify.py"
    "cmake/cpu_extension.cmake"
)

for file in $file_diff; do
    # First check if file matches any pattern
    matches_pattern=0
    for pattern in "${patterns[@]}"; do
        if [[ $file == "$pattern"* ]] || [[ $file == "$pattern" ]]; then
            matches_pattern=1
            break
        fi
    done

    # If file matches pattern, check it's not in ignore patterns
    if [[ $matches_pattern -eq 1 ]]; then
        matches_ignore=0
        for ignore in "${ignore_patterns[@]}"; do
            if [[ $file == "$ignore"* ]] || [[ $file == "$ignore" ]]; then
                matches_ignore=1
                break
            fi
        done

        if [[ $matches_ignore -eq 0 ]]; then
            RUN_ALL=1
            echo "Found changes: $file. Run all tests"
            break
        fi
    fi
done

# Decide whether to use precompiled wheels
# Relies on existing patterns array as a basis.
if [[ -n "${VLLM_USE_PRECOMPILED:-}" ]]; then
    echo "VLLM_USE_PRECOMPILED is already set to: $VLLM_USE_PRECOMPILED"
elif [[ $RUN_ALL -eq 1 ]]; then
    export VLLM_USE_PRECOMPILED=0
    echo "Detected critical changes, building wheels from source"
else
    export VLLM_USE_PRECOMPILED=1
    echo "No critical changes, using precompiled wheels"
fi


LIST_FILE_DIFF=$(get_diff | tr ' ' '|')
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    LIST_FILE_DIFF=$(get_diff_main | tr ' ' '|')
fi
upload_pipeline
