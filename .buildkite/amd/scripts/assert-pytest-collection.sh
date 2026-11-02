#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 EXPECTED_COUNT PYTEST_ARGS..." >&2
    exit 2
fi

expected_count=$1
shift

collection_file=$(mktemp)
trap 'rm -f "${collection_file}"' EXIT

pytest --collect-only -q "$@" | tee "${collection_file}"

actual_count=$(awk '/^tests\/.*::/ { count += 1 } END { print count + 0 }' "${collection_file}")
if [[ "${actual_count}" != "${expected_count}" ]]; then
    echo "Expected ${expected_count} collected tests, found ${actual_count}." >&2
    exit 1
fi

echo "Verified exact pytest collection: ${actual_count} tests."
