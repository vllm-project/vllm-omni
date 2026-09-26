#!/usr/bin/env bash
# Shared Buildkite skip-ci helpers for AMD / Intel bootstrap scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_CI_PY="${SCRIPT_DIR}/skip_ci.py"

# Resolve git diff once. Exit 0 when this bootstrap should stop uploading.
# Decision details are logged by skip_ci.py; this helper only annotates skip-all.
# Usage: gate_bootstrap_ci <platform> <l2|l3>
gate_bootstrap_ci() {
    local platform="$1"
    local level="$2"
    local kind
    if kind="$(python3 "${SKIP_CI_PY}" gate "${platform}" "${level}")"; then
        if [[ "${kind}" == "skip-all" ]]; then
            buildkite-agent annotate \
                ":memo: CI skipped — docs or pytest skip-mark changes only" \
                --style "info" 2>/dev/null || true
        fi
        exit 0
    fi
}
