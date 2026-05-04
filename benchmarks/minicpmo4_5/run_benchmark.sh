#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
MODE="${MODE:-all}"
MODALITIES="${MODALITIES:-text,text+image,text+video}"
NUM_REPEATS="${NUM_REPEATS:-1}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-bench_results/minicpmo4_5}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/minicpmo.yaml}"

CMD=(
  python benchmarks/minicpmo4_5/bench_minicpmo4_5.py
  --model-path "${MODEL}"
  --mode "${MODE}"
  --modalities "${MODALITIES}"
  --num-repeats "${NUM_REPEATS}"
  --seed "${SEED}"
  --temperature "${TEMPERATURE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --stage-config-path "${STAGE_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${CUDA_DEVICES}" ]]; then
  CMD+=(--cuda-visible-devices "${CUDA_DEVICES}")
fi

"${CMD[@]}"
