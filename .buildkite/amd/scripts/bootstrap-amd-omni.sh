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

fail_fast() {
    DISABLE_LABEL="ci-no-fail-fast"
    # If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
    if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
        PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm-omni/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')
        if [[ $PR_LABELS == *"$DISABLE_LABEL"* ]]; then
            echo false
        else
            echo true
        fi
    else
        echo false  # not a PR or BUILDKITE_PULL_REQUEST not set
    fi
}

check_run_all_label() {
    RUN_ALL_LABEL="ready-run-all-tests"
    # If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
    if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
        PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm-omni/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')
        if [[ $PR_LABELS == *"$RUN_ALL_LABEL"* ]]; then
            echo true
        else
            echo false
        fi
    else
        echo false  # not a PR or BUILDKITE_PULL_REQUEST not set
    fi
}

if [[ -z "${COV_ENABLED:-}" ]]; then
    COV_ENABLED=0
fi

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
        python .buildkite/amd/pipeline_generator/pipeline_generator.py --run_all=$RUN_ALL --list_file_diff="$LIST_FILE_DIFF" --nightly="$NIGHTLY" --mirror_hw="$AMD_MIRROR_HW"
        buildkite-agent pipeline upload .buildkite/amd/pipeline.yaml
        exit 0
    fi
    echo "List file diff: $LIST_FILE_DIFF"
    echo "Run all: $RUN_ALL"
    echo "Nightly: $NIGHTLY"
    echo "AMD Mirror HW: $AMD_MIRROR_HW"

    FAIL_FAST=$(fail_fast)

    cd .buildkite/amd

    # Select test definition file: merge suite for main, ready suite for PRs.
    # For debugging, DEBUG_TEST_YAML accepts a comma-separated list containing
    # "merge" and/or "ready" (case-insensitive). Multiple suites are combined
    # before rendering so they share one amd-build step.
    if [[ -n "${DEBUG_TEST_YAML:-}" ]]; then
        declare -a DEBUG_TEST_SPECS=()
        declare -A SEEN_DEBUG_TESTS=()
        IFS=',' read -ra REQUESTED_DEBUG_TESTS <<< "${DEBUG_TEST_YAML,,}"

        for requested_test in "${REQUESTED_DEBUG_TESTS[@]}"; do
            # Permit readable values such as "ready, merge," and ignore the
            # empty item produced by a trailing comma.
            requested_test="${requested_test//[[:space:]]/}"
            [[ -z "$requested_test" ]] && continue

            if [[ -n "${SEEN_DEBUG_TESTS[$requested_test]:-}" ]]; then
                echo "ERROR: duplicate DEBUG_TEST_YAML suite '$requested_test'" >&2
                exit 1
            fi
            SEEN_DEBUG_TESTS[$requested_test]=1

            case "$requested_test" in
                ready)
                    DEBUG_TEST_SPECS+=("READY_TESTS:test-amd-ready.yml")
                    ;;
                merge)
                    DEBUG_TEST_SPECS+=("MERGE_TESTS:test-amd-merge.yml")
                    ;;
                *)
                    echo "ERROR: DEBUG_TEST_YAML entries must be 'merge' or 'ready', got '$requested_test'" >&2
                    exit 1
                    ;;
            esac
        done

        if [[ ${#DEBUG_TEST_SPECS[@]} -eq 0 ]]; then
            echo "ERROR: DEBUG_TEST_YAML did not contain a test suite" >&2
            exit 1
        elif [[ ${#DEBUG_TEST_SPECS[@]} -eq 1 ]]; then
            TEST_YAML="${DEBUG_TEST_SPECS[0]#*:}"
        else
            TEST_YAML=$(mktemp "${TMPDIR:-/tmp}/amd-debug-tests.XXXXXX.yml")
            python - "$TEST_YAML" "${DEBUG_TEST_SPECS[@]}" <<'PY'
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
        echo "DEBUG_TEST_YAML override: using ${DEBUG_TEST_SPECS[*]}"
    elif [[ $BUILDKITE_BRANCH == "main" ]]; then
        TEST_YAML="test-amd-merge.yml"
    else
        TEST_YAML="test-amd-ready.yml"
    fi

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
    if [[ "$TEST_YAML" == "${TMPDIR:-/tmp}/amd-debug-tests."*.yml ]]; then
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

# Early exit: unified skip-ci (docs / skip marks) and CI-yaml-only level targeting.
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    gate_bootstrap_ci amd l3
else
    gate_bootstrap_ci amd l2
fi

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

# Check for ready-run-all-tests label
LABEL_RUN_ALL=$(check_run_all_label)
if [[ $LABEL_RUN_ALL == true ]]; then
    RUN_ALL=1
    NIGHTLY=1
    echo "Found 'ready-run-all-tests' label. Running all tests including optional tests."
fi

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
