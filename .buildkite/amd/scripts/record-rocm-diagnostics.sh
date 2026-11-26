#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Best-effort evidence for GPU aborts; never replace pytest's exit status.
set -u
artifact_dir=${1:?artifact directory required}
phase=${2:?before or after required}
[[ "$phase" == before || "$phase" == after ]] || exit 2
mkdir -p "$artifact_dir" || exit 1
{
    date -u
    printf 'buildkite_commit=%s\nagent=%s\nhost=%s\n' \
        "${BUILDKITE_COMMIT:-unknown}" "${BUILDKITE_AGENT_NAME:-unknown}" "${HOSTNAME:-unknown}"
    uname -a
    timeout --signal=TERM --kill-after=5s 30s rocminfo
    timeout --signal=TERM --kill-after=5s 30s rocm-smi --showall
    ps -eo pid,ppid,stat,comm,args
} > "$artifact_dir/rocm-diagnostics-$phase.txt" 2>&1
if [[ "$phase" == before ]]; then
    timeout --signal=TERM --kill-after=5s 30s python3 -m pip freeze \
        > "$artifact_dir/dependencies.txt" 2>&1
fi
exit 0
