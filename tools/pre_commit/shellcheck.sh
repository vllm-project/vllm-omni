#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Lint bash scripts for undefined vars, quoting, and similar bugs.
# Do not download a binary: use a distro/package-manager install so the
# artifact is signed (apt/dnf/brew) or at least an explicit local install.
set -euo pipefail

is_windows_exe() {
    [[ "$1" == *.exe ]]
}

is_git_bash() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

find_native_shellcheck() {
    local cand
    cand="$(command -v shellcheck 2>/dev/null || true)"
    if [ -n "$cand" ] && ! is_windows_exe "$cand"; then
        echo "$cand"
        return 0
    fi
    return 1
}

install_hint() {
    echo "Please install shellcheck with your package manager, then re-run:"
    echo "  Debian/Ubuntu/WSL: sudo apt-get install shellcheck"
    echo "  Fedora:            sudo dnf install ShellCheck"
    echo "  macOS:             brew install shellcheck"
    echo "  Git Bash:          scoop install shellcheck"
    echo "  https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
}

SHELLCHECK_BIN=""
if SHELLCHECK_BIN="$(find_native_shellcheck)"; then
    :
elif is_git_bash && SHELLCHECK_BIN="$(command -v shellcheck.exe 2>/dev/null || true)" && [ -n "$SHELLCHECK_BIN" ]; then
    :
else
    install_hint
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
