#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Compatibility entry for external AMD Buildkite callers. The shared uploader
# owns AMD suite selection and native rendering.
set -euo pipefail

exec python3 .buildkite/common/scripts/upload_pipeline.py --amd --upload "$@"
