#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Usage: echo '<review_json>' | ./verify_line_numbers.sh <pr_number>
# Requires Python 3 and gh. Exits nonzero for invalid comments or fetch failures.
set -euo pipefail

exec python3 "$(dirname "${BASH_SOURCE[0]}")/verify_line_numbers.py" "$@"
