#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Launch Breeze-TTS-2 online serving with the async-chunk deploy config.

set -euo pipefail

MODEL="${1:-BreezeBlue/Breeze-TTS-2}"
PORT="${PORT:-8091}"

exec vllm-omni serve "${MODEL}" \
    --deploy-config vllm_omni/deploy/breeze_tts_2.yaml \
    --omni --port "${PORT}"
