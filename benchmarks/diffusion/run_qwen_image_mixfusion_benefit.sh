#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
STEPS="${STEPS:-8}"
ITERS="${ITERS:-3}"
WARMUP="${WARMUP:-1}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}"
BACKEND="${BACKEND:-mp}"
OUTPUT_TYPE="${OUTPUT_TYPE:-latent}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
ATTN_BACKENDS="${ATTN_BACKENDS:-FLASH_ATTN TORCH_SDPA SAGE_ATTN}"
CASES="${CASES:-1024x1024,1024x768 1024x1024,768x1024 1024x1024,512x512 512x512,512x768}"
MIXFUSION_MIN_CHUNK_TOKENS="${MIXFUSION_MIN_CHUNK_TOKENS:-256}"
MIXFUSION_MAX_CHUNKS="${MIXFUSION_MAX_CHUNKS:-128}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/benchmarks/diffusion/qwen_mixfusion_results/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULT_ROOT}"
SUMMARY="${RESULT_ROOT}/summary.tsv"
printf "backend\tcase\tstatus\tserial_avg_s\tbatched_avg_s\tspeedup\tchunk_size\tchunk_count\treason\tjson\n" > "${SUMMARY}"

echo "Writing results to ${RESULT_ROOT}"
echo "Model: ${MODEL}"
echo "Attention backends: ${ATTN_BACKENDS}"
echo "Cases: ${CASES}"

for ATTENTION_BACKEND in ${ATTN_BACKENDS}; do
  for CASE in ${CASES}; do
    SAFE_BACKEND="$(echo "${ATTENTION_BACKEND}" | tr -c 'A-Za-z0-9_' '_')"
    SAFE_CASE="$(echo "${CASE}" | tr -c 'A-Za-z0-9_' '_')"
    JSON_OUT="${RESULT_ROOT}/${SAFE_BACKEND}_${SAFE_CASE}.json"
    LOG_OUT="${RESULT_ROOT}/${SAFE_BACKEND}_${SAFE_CASE}.log"
    IMAGE_DIR_ARG=()
    if [[ "${SAVE_IMAGES}" == "1" ]]; then
      IMAGE_DIR_ARG=(--output-dir "${RESULT_ROOT}/${SAFE_BACKEND}_${SAFE_CASE}_images")
    fi
    QUANT_ARG=()
    if [[ -n "${QUANTIZATION}" ]]; then
      QUANT_ARG=(--quantization "${QUANTIZATION}")
    fi

    echo "==> backend=${ATTENTION_BACKEND} case=${CASE}"
    set +e
    (
      cd "${REPO_ROOT}"
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export DIFFUSION_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
      python benchmarks/diffusion/qwen_image_mixfusion_benefit.py \
        --model "${MODEL}" \
        --image-sizes "${CASE}" \
        --steps "${STEPS}" \
        --iters "${ITERS}" \
        --warmup "${WARMUP}" \
        --dtype "${DTYPE}" \
        --distributed-executor-backend "${BACKEND}" \
        --tensor-parallel-size "${TP}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --output-type "${OUTPUT_TYPE}" \
        --mixfusion-min-chunk-tokens "${MIXFUSION_MIN_CHUNK_TOKENS}" \
        --mixfusion-max-chunks "${MIXFUSION_MAX_CHUNKS}" \
        --json-output "${JSON_OUT}" \
        "${QUANT_ARG[@]}" \
        "${IMAGE_DIR_ARG[@]}"
    ) > "${LOG_OUT}" 2>&1
    STATUS=$?
    set -e

    if [[ ${STATUS} -ne 0 ]]; then
      printf "%s\t%s\tFAILED\t\t\t\t\t\tsee log\t%s\n" \
        "${ATTENTION_BACKEND}" "${CASE}" "${LOG_OUT}" >> "${SUMMARY}"
      tail -n 40 "${LOG_OUT}" || true
      continue
    fi

    python - "${JSON_OUT}" "${ATTENTION_BACKEND}" "${CASE}" "${SUMMARY}" <<'PY'
import json
import sys

json_path, backend, case, summary = sys.argv[1:]
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

candidate = data.get("candidate", {})
if data.get("skipped"):
    row = [
        backend,
        case,
        "SKIPPED",
        "",
        "",
        "",
        str(candidate.get("chunk_size", "")),
        str(candidate.get("chunk_count", "")),
        data.get("skip_reason", ""),
        json_path,
    ]
else:
    row = [
        backend,
        case,
        "OK",
        f"{data.get('serial_avg_s', 0.0):.6f}",
        f"{data.get('batched_avg_s', 0.0):.6f}",
        f"{data.get('speedup', 0.0):.4f}",
        str(candidate.get("chunk_size", "")),
        str(candidate.get("chunk_count", "")),
        candidate.get("reason", ""),
        json_path,
    ]
with open(summary, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print("\t".join(row))
PY
  done
done

echo
echo "Summary:"
cat "${SUMMARY}"
