#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Release line: main, or a branch named exactly release/vX.Y.Z (for example release/v0.30.0).
# Keep the version pattern identical to ci_conditions.RELEASE_LINE_VERSION_PATTERN.
#
# Defines a function only. Does not change shell options, so callers that must
# keep set -e off (run_cov_split.sh) can source this file safely.
is_release_line_branch() {
    local branch="${1:-${BUILDKITE_BRANCH:-}}"
    # Store the pattern in a variable. An inline ``=~`` pattern does not match
    # the slash in ``release/v``.
    local release_line_pattern='^release/v[0-9]+\.[0-9]+\.[0-9]+$'
    [[ "${branch}" == "main" || "${branch}" =~ ${release_line_pattern} ]]
}
