#!/usr/bin/env bash
# Shared Buildkite skip-ci helpers for AMD / Intel bootstrap scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_CI_PY="${SCRIPT_DIR}/skip_ci.py"

print_annotate() {
    local reason
    reason="$(python3 "${SKIP_CI_PY}" print-annotate 2>/dev/null || true)"
    if [[ -n "${reason}" && "${reason}" == CI\ skipped* ]]; then
        buildkite-agent annotate ":memo: ${reason}" --style "info" 2>/dev/null || true
    fi
}

is_skip_all_ci() {
    if python3 "${SKIP_CI_PY}" check-skip-all; then
        print_annotate
        echo "[skip-ci] Docs/skip-mark-only changes detected. Exiting before pipeline upload."
        exit 0
    fi
}

is_skip_l23_ci() {
    local platform="$1"
    local level="$2"
    if python3 "${SKIP_CI_PY}" check-skip-l2-l3 "${platform}" "${level}"; then
        echo "[ci-yaml-only] skipping ${platform} ${level} pipeline"
        exit 0
    fi
}
