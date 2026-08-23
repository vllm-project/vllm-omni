#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUM_GPUS="${NUM_GPUS:-4}"
exec "${SCRIPT_DIR}/run_all_tasks.sh" "$@"
