#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODEL="${MODEL:-ICTNLP/LLaMA-Omni2-0.5B}"
DECODER_MODEL="${DECODER_MODEL:-ICTNLP/cosy2_decoder}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/llama_omni2_e2e_output}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}"

render_config() {
  local source="$1"
  local destination="$2"
  "${PYTHON}" - "${source}" "${destination}" "${DECODER_MODEL}" <<'PY'
from pathlib import Path
import sys

source, destination, decoder_model = sys.argv[1:]
text = Path(source).read_text()
text = text.replace("ICTNLP/cosy2_decoder", decoder_model)
Path(destination).write_text(text)
PY
}

run_case() {
  local label="$1"
  local mode="$2"
  local deploy="$3"
  "${PYTHON}" \
    "${ROOT_DIR}/tests/e2e/offline_inference/llama_omni2/run_llama_omni2_e2e.py" \
    --model "${MODEL}" \
    --deploy-config "${deploy}" \
    --output-dir "${OUTPUT_DIR}" \
    --label "${label}" \
    --mode "${mode}"
}

TP1_SOURCE="${ROOT_DIR}/tests/e2e/offline_inference/llama_omni2/llama_omni2_tp1.yaml"
TP2_SOURCE="${ROOT_DIR}/tests/e2e/offline_inference/llama_omni2/llama_omni2_tp2.yaml"
TP1_CONFIG="${OUTPUT_DIR}/llama_omni2_tp1.rendered.yaml"
TP2_CONFIG="${OUTPUT_DIR}/llama_omni2_tp2.rendered.yaml"
render_config "${TP1_SOURCE}" "${TP1_CONFIG}"
render_config "${TP2_SOURCE}" "${TP2_CONFIG}"

case "${1:-all}" in
  text)
    run_case tp1 text "${TP1_CONFIG}"
    ;;
  speech)
    run_case tp1 speech "${TP1_CONFIG}"
    ;;
  concurrent)
    run_case tp1 concurrent "${TP1_CONFIG}"
    ;;
  tp)
    run_case tp1 text "${TP1_CONFIG}"
    run_case tp2 text "${TP2_CONFIG}"
    "${PYTHON}" \
      "${ROOT_DIR}/tests/e2e/offline_inference/llama_omni2/compare_tp_results.py" \
      "${OUTPUT_DIR}/tp1-text.json" \
      "${OUTPUT_DIR}/tp2-text.json"
    ;;
  all)
    run_case tp1 text "${TP1_CONFIG}"
    run_case tp1 speech "${TP1_CONFIG}"
    run_case tp1 concurrent "${TP1_CONFIG}"
    run_case tp2 text "${TP2_CONFIG}"
    "${PYTHON}" \
      "${ROOT_DIR}/tests/e2e/offline_inference/llama_omni2/compare_tp_results.py" \
      "${OUTPUT_DIR}/tp1-text.json" \
      "${OUTPUT_DIR}/tp2-text.json"
    ;;
  *)
    echo "usage: $0 [text|speech|concurrent|tp|all]" >&2
    exit 2
    ;;
esac
