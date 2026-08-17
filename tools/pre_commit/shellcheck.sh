#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Based on vllm/tools/pre_commit/shellcheck.sh: install shellcheck if needed
# and lint bash scripts for undefined vars, quoting, and similar bugs.
set -euo pipefail

scversion="stable"
cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-omni/shellcheck-${scversion}"

is_windows_exe() {
    [[ "$1" == *.exe ]]
}

find_native_shellcheck() {
    local cand
    if [ -x "${cache_dir}/shellcheck" ]; then
        echo "${cache_dir}/shellcheck"
        return 0
    fi
    if [ -x "$(pwd)/shellcheck-${scversion}/shellcheck" ]; then
        echo "$(pwd)/shellcheck-${scversion}/shellcheck"
        return 0
    fi
    cand="$(command -v shellcheck 2>/dev/null || true)"
    if [ -n "$cand" ] && ! is_windows_exe "$cand"; then
        echo "$cand"
        return 0
    fi
    return 1
}

install_linux_x86_64() {
    mkdir -p "$cache_dir"
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion}/shellcheck-${scversion}.linux.x86_64.tar.xz" |
        tar -xJ --strip-components=1 -C "$cache_dir"
}

SHELLCHECK_BIN=""
if SHELLCHECK_BIN="$(find_native_shellcheck)"; then
    :
elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    install_linux_x86_64
    SHELLCHECK_BIN="${cache_dir}/shellcheck"
elif SHELLCHECK_BIN="$(command -v shellcheck.exe 2>/dev/null || true)" && [ -n "$SHELLCHECK_BIN" ]; then
    :
else
    echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
    exit 1
fi

should_lint() {
    local f="${1//\\//}"
    f="${f#./}"
    case "$f" in
        *.sh) ;;
        *) return 1 ;;
    esac
    git check-ignore -q "$f" && return 1
    return 0
}

run_shellcheck() {
    local f
    for f in "$@"; do
        if should_lint "$f"; then
            "$SHELLCHECK_BIN" -s bash "$f"
        fi
    done
}

if [ "$#" -gt 0 ]; then
    run_shellcheck "$@"
    exit 0
fi

# Direct invocation with no args: lint every tracked *.sh.
while IFS= read -r -d '' f || [ -n "$f" ]; do
    git check-ignore -q "$f" && continue
    "$SHELLCHECK_BIN" -s bash "$f"
done < <(find . -path ./.git -prune -o -name "*.sh" -print0)
