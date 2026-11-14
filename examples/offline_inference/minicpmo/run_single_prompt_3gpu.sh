#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Single-prompt offline run on the 3-GPU layout: one GPU per stage
# (Thinker on GPU 0, Talker on GPU 1, Code2Wav on GPU 2).
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../../.. && pwd)"

python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --deploy-config "${REPO_ROOT}/vllm_omni/deploy/minicpmo_4_5_3gpu.yaml" \
                  --stage-init-timeout 300
