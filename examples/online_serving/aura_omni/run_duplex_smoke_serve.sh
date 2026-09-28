#!/usr/bin/env bash
# Minimal AURA duplex Realtime server for smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8099}"
MODEL="${MODEL:-/workspace/models/AURA}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$REPO_ROOT/examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml}"

# AURA v1 / Qwen3-VL ChatML ids (do not use AURA v2 248070/248046 here).
export VLLM_AURA_SILENT_TOKEN_ID="${VLLM_AURA_SILENT_TOKEN_ID:-151669}"
export VLLM_AURA_IM_END_TOKEN_ID="${VLLM_AURA_IM_END_TOKEN_ID:-151645}"
export VLLM_AURA_IM_START_TOKEN_ID="${VLLM_AURA_IM_START_TOKEN_ID:-151644}"
export VLLM_AURA_ASSISTANT_TOKEN_ID="${VLLM_AURA_ASSISTANT_TOKEN_ID:-77091}"
# Multi-stage colocated init can make earlier stages' Process.sentinel spuriously
# readable; prefer spawn for stage engine subprocesses (override if needed).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Do NOT set TORCHDYNAMO_DISABLE here: it breaks Stage0/1 AOT when those
# stages are not enforce_eager. Stage2/3 use enforce_eager; code_predictor
# honors that and skips its own torch.compile.

exec "$REPO_ROOT/.venv/bin/vllm-omni" serve "$MODEL" \
  --omni \
  --deploy-config "$DEPLOY_CONFIG" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT"
