#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8099}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/daily_omni}"
BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/.venv/bin/vllm-omni}"

mkdir -p "${OUTPUT_DIR}"
command=(
    "${BENCH_BIN}" bench serve --omni
    --backend openai-chat-omni
    --endpoint /v1/chat/completions
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL}"
    --dataset-name daily-omni
    --num-prompts "${NUM_PROMPTS}"
    --num-warmups 2
    --max-concurrency 1
    --request-rate inf
    --percentile-metrics ttft,e2el
    --extra-body '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":false}}'
    --save-result
    --result-dir "${OUTPUT_DIR}"
)
if [[ -n "${DAILY_OMNI_QA_JSON:-}" ]]; then
    command+=(--daily-omni-qa-json "${DAILY_OMNI_QA_JSON}")
fi
if [[ -n "${DAILY_OMNI_VIDEO_DIR:-}" ]]; then
    command+=(--daily-omni-video-dir "${DAILY_OMNI_VIDEO_DIR}")
fi

printf '%q ' "${command[@]}" | tee "${OUTPUT_DIR}/command.txt"
printf '\n' | tee -a "${OUTPUT_DIR}/command.txt"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
