#!/bin/bash
# Run multiple Qwen3-Omni requests with async_chunk enabled.
#
# Uses AsyncOmni with --max-in-flight to control request-level
# concurrency (each request still gets true stage-level concurrency
# via async_chunk).
#
# Usage:
#   bash run_multiple_prompts_async_chunk.sh
#   bash run_multiple_prompts_async_chunk.sh --max-in-flight 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/end2end_async_chunk.py" \
    --query-type text \
    --txt-prompts "${SCRIPT_DIR}/text_prompts_10.txt" \
    --deploy-config "${REPO_ROOT}/vllm_omni/deploy/qwen3_omni_moe.yaml" \
    --output-dir output_audio_async_chunk \
    --max-in-flight 2 \
    "$@"
