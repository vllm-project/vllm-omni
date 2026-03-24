#!/bin/bash
# Run a single Qwen3-Omni request with async_chunk enabled.
#
# This uses AsyncOmni (async orchestrator) so that downstream stages
# (Talker, Code2Wav) start *before* stage-0 (Thinker) finishes,
# achieving true stage-level concurrency via chunk-level streaming.
#
# Prerequisites:
#   - An async_chunk stage config YAML (e.g. qwen3_omni_moe_async_chunk.yaml)
#   - Hardware matching the config (e.g. 2x H100 for the default 3-stage config)
#
# Usage:
#   bash run_single_prompt_async_chunk.sh
#   bash run_single_prompt_async_chunk.sh --query-type text --modalities text
#   bash run_single_prompt_async_chunk.sh --stage-configs-path /path/to/custom.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/end2end_async_chunk.py" \
    --query-type use_audio \
    --stage-configs-path "${REPO_ROOT}/vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml" \
    --output-dir output_audio_async_chunk \
    "$@"
