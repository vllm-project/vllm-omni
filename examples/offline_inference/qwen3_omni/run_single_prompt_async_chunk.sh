#!/bin/bash
# Run a single Qwen3-Omni request with async_chunk enabled.
#
# This uses AsyncOmni (async orchestrator) so that downstream stages
# (Talker, Code2Wav) start *before* stage-0 (Thinker) finishes,
# achieving true stage-level concurrency via chunk-level streaming.
#
# Prerequisites:
#   - A deploy config YAML (e.g. qwen3_omni_moe.yaml)
#   - Hardware matching the config (e.g. 2x H100 for the default 3-stage config)
#
# Usage:
#   bash run_single_prompt_async_chunk.sh
#   bash run_single_prompt_async_chunk.sh --query-type text --modalities text
#   bash run_single_prompt_async_chunk.sh --deploy-config /path/to/custom.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/end2end_async_chunk.py" \
    --query-type use_audio \
    --deploy-config "${REPO_ROOT}/vllm_omni/deploy/qwen3_omni_moe.yaml" \
    --output-dir output_audio_async_chunk \
    "$@"
