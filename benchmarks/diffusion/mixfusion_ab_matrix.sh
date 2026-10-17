#!/usr/bin/env bash
# Full-matrix A/B endorsement run: the 9-case timing matrix of
# mixfusion_ab_bench.sh with RESULTS=results_v3 (the directory name the
# summarizer and parity driver default to).
#
# Record the measured SHAs when re-running (they go into the report):
#   git -C "$A_WT" rev-parse HEAD   /   git -C "$B_WT" rev-parse HEAD
# The candidate should sit on top of the measured baseline commit so the
# comparison isolates the PR's changes.
#
# Usage:
#   A_WT=... B_WT=... bash mixfusion_ab_matrix.sh
#   ONLY=B-W3-on A_WT=... B_WT=... bash mixfusion_ab_matrix.sh   # single case
set -u

export RESULTS=${RESULTS:-$PWD/results_v3}
exec bash "$(dirname "${BASH_SOURCE[0]}")/mixfusion_ab_bench.sh"
