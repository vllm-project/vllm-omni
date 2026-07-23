#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# One-command HunyuanImage-3.0 MixFusion benefit sweep.
#
# Example:
#   bash benchmarks/diffusion/run_hunyuan_image3_mixfusion_benefit.sh
#
# Useful overrides:
#   MODEL=/path/to/HunyuanImage-3.0-Instruct TP=4 STEPS=8 ITERS=3 \
#     bash benchmarks/diffusion/run_hunyuan_image3_mixfusion_benefit.sh
#
# Output:
#   ${OUT_DIR}/summary.tsv
#   ${OUT_DIR}/<case>.json
#   ${OUT_DIR}/<case>.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
TP="${TP:-4}"
STEPS="${STEPS:-8}"
ITERS="${ITERS:-3}"
WARMUP="${WARMUP:-1}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-fp8}"
BACKEND="${BACKEND:-mp}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-benchmarks/diffusion/hunyuan_mixfusion_results/$(date +%Y%m%d_%H%M%S)}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WARMUP_ARG=()
if [[ "${WARMUP}" != "0" ]]; then
  WARMUP_ARG=(--warmup)
fi

# Use cases whose normalized token lengths should produce a large GCD and
# small chunk count. Small-GCD cases should be guarded by MixFusion fallback and
# are not good benefit candidates.
CASES="${CASES:-1024x1024,1024x768 1024x1024,768x1024 1024x1024,1280x720}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
printf "case\tindependent_mean_s\tmixfusion_mean_s\tspeedup\n" > "${SUMMARY}"

echo "[mixfusion-bench] repo=${ROOT_DIR}"
echo "[mixfusion-bench] model=${MODEL}"
echo "[mixfusion-bench] tp=${TP} steps=${STEPS} iters=${ITERS} warmup=${WARMUP}"
echo "[mixfusion-bench] dtype=${DTYPE} quantization=${QUANTIZATION} guidance=${GUIDANCE_SCALE}"
echo "[mixfusion-bench] out=${OUT_DIR}"

for IMAGE_SIZES in ${CASES}; do
  CASE_NAME="${IMAGE_SIZES//,/__}"
  CASE_NAME="${CASE_NAME//x/-}"
  JSON_OUT="${OUT_DIR}/${CASE_NAME}.json"
  LOG_OUT="${OUT_DIR}/${CASE_NAME}.log"
  IMAGE_OUT_ARG=()
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    IMAGE_OUT_ARG=(--output-dir "${OUT_DIR}/${CASE_NAME}_images")
  fi

  echo
  echo "[mixfusion-bench] running ${IMAGE_SIZES}"
  set +e
  "${PYTHON_BIN}" benchmarks/diffusion/hunyuan_image3_real_mixfusion_benefit.py \
    --model "${MODEL}" \
    --image-sizes "${IMAGE_SIZES}" \
    --steps "${STEPS}" \
    --iters "${ITERS}" \
    "${WARMUP_ARG[@]}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --quantization "${QUANTIZATION}" \
    --tensor-parallel-size "${TP}" \
    --distributed-executor-backend "${BACKEND}" \
    "${IMAGE_OUT_ARG[@]}" \
    > "${JSON_OUT}" 2> "${LOG_OUT}"
  STATUS=$?
  set -e

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[mixfusion-bench] FAILED ${IMAGE_SIZES}; see ${LOG_OUT}"
    printf "%s\tFAILED\tFAILED\tFAILED\n" "${IMAGE_SIZES}" >> "${SUMMARY}"
    continue
  fi

  "${PYTHON_BIN}" - "${JSON_OUT}" "${IMAGE_SIZES}" "${SUMMARY}" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
case = sys.argv[2]
summary = pathlib.Path(sys.argv[3])

data = json.loads(json_path.read_text())
independent = data["time_s"]["independent_mean"]
mixfusion = data["time_s"]["mixfusion_batch_mean"]
speedup = data["speedup"]["mixfusion_batch_vs_independent"]
with summary.open("a", encoding="utf-8") as f:
    f.write(f"{case}\t{independent:.6f}\t{mixfusion:.6f}\t{speedup:.4f}\n")
print(
    f"[mixfusion-bench] {case}: independent={independent:.3f}s "
    f"mixfusion={mixfusion:.3f}s speedup={speedup:.3f}x"
)
PY
done

echo
echo "[mixfusion-bench] summary:"
cat "${SUMMARY}"
