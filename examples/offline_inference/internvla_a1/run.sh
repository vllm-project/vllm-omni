#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
 : "${INTERNVLA_A1_MODEL_DIR:?Please export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen}"
 : "${INTERNVLA_A1_DATASET_DIR:?Please export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen}"
 : "${INTERNVLA_A1_PROCESSOR_DIR:?Please export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct}"
 : "${INTERNVLA_A1_COSMOS_DIR:?Please export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor}"
INTERNVLA_A1_OUTPUT_DIR="${INTERNVLA_A1_OUTPUT_DIR:-$REPO_ROOT/outputs/internvla_a1/vllm_infer}"

export INTERNVLA_A1_MODEL_DIR
export INTERNVLA_A1_DATASET_DIR
export INTERNVLA_A1_PROCESSOR_DIR
export INTERNVLA_A1_COSMOS_DIR

python "$ROOT_DIR/end2end.py" \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --output-dir "$INTERNVLA_A1_OUTPUT_DIR" \
  --num-episodes "${INTERNVLA_A1_NUM_EPISODES:-1}" \
  --attn-implementation eager \
  "$@"
