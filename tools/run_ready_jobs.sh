#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Extract steps from .buildkite/test-ready.yml that contain pytest, synthesize
# small bash wrappers (exports + pytest), run them, and tee output to logs named
# after each step's Buildkite "key" when present (otherwise a slug of the label).
#
# Model area (--model-type / MODEL_TYPE), multiple allowed (OR semantics):
#   e.g. --model-type omni,tts
#   omni     — label contains "Omni ·"
#   tts      — label contains "TTS ·"
#   diffusion— label contains "Diffusion"
#   all      — no model filter (default)
#
#   --skip-simple — skip steps in the "Simple Test" Buildkite group and labels
#                   starting with "Simple ·" (L1-style unit tests in the same YAML)
#
# Requirements: bash, python3, PyYAML (pip install pyyaml)
#
# Usage:
#   bash tools/run_ready_jobs.sh
#   REPO_ROOT=/path/to/vllm-omni bash tools/run_ready_jobs.sh --model-type omni --dry-run
#   YML=/path/to/vllm-omni/.buildkite/test-ready.yml bash tools/run_ready_jobs.sh
#
# Repository / YAML (no dependency on where this script lives):
#   • Set REPO_ROOT (or pass --repo-root) — default YAML is $REPO_ROOT/.buildkite/test-ready.yml
#   • Or set YML (or --yaml) — repo root is inferred as parent of the .buildkite directory
#   • Or run from inside the clone: git rev-parse --show-toplevel, else walk up from $PWD,
#     then from the script's directory, until .buildkite/test-ready.yml exists
#
# Optional environment:
#   REPO_ROOT     - vllm-omni root (working directory for pytest); see above
#   YML           - path to test-ready.yml (default: $REPO_ROOT/.buildkite/test-ready.yml)
#   LOG_DIR       - logs + generated job scripts (default: $REPO_ROOT/logs/ready_jobs);
#                   per-job *.log plus timing_summary.log after the run
#   MODEL_TYPE    - comma-separated and/or repeated flags (default: all); see above
#   LABEL_SUBSTR  - substring of Buildkite step label
#   DRY_RUN=1     - print extracted commands only; do not write scripts or run pytest
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDKITE_REL=".buildkite/test-ready.yml"
DEFAULT_LOG_SUBDIR="ready_jobs"

# shellcheck source=tools/run_jobs_common.sh
source "${SCRIPT_DIR}/run_jobs_common.sh"
run_yaml_ci_jobs_main "$@"
