#!/bin/bash
# vllm-omni Intel bootstrap
# Uses static pipeline-intel.yaml for Intel XPU tests
# Based on: bootstrap-amd-omni.sh

set -euo pipefail

if [[ -z "${DOCS_ONLY_DISABLE:-}" ]]; then
    DOCS_ONLY_DISABLE=0
fi

upload_pipeline() {
    echo "--- 🛠 Preparing Intel pipeline"
    buildkite-agent pipeline upload .buildkite/pipeline-intel.yaml
}

get_diff() {
    $(git add .)
    echo $(git diff --name-only --diff-filter=ACMDR $(git merge-base origin/main HEAD))
}

get_diff_main() {
    $(git add .)
    echo $(git diff --name-only --diff-filter=ACMDR HEAD~1)
}

file_diff=$(get_diff)
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    file_diff=$(get_diff_main)
fi

# ----------------------------------------------------------------------
# Early exit: skip pipeline if all changed files are under docs/
# ----------------------------------------------------------------------
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
        buildkite-agent annotate ":memo: CI skipped — docs only" --style "info"
        exit 0
    fi
fi

upload_pipeline
