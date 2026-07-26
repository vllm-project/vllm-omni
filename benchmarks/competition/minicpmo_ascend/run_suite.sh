#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
BASE_URL="${BASE_URL:-http://localhost:8099/v1}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/${RUN_ID}}"
VIDEO_INPUT="${VIDEO_INPUT:-}"
STABILITY_CONCURRENCY="${STABILITY_CONCURRENCY:-4}"
STABILITY_REQUESTS="${STABILITY_REQUESTS:-100}"
STABILITY_WARMUPS="${STABILITY_WARMUPS:-2}"
SEED="${SEED:-42}"
THINKER_MAX_TOKENS="${THINKER_MAX_TOKENS:-256}"
TALKER_MAX_TOKENS="${TALKER_MAX_TOKENS:-256}"

if [[ -z "${VIDEO_INPUT}" ]]; then
    echo "VIDEO_INPUT must point to a deterministic local video or official fixture." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
collect_args=(--output "${OUTPUT_DIR}/environment.json")
if [[ -n "${MODEL_PATH:-}" ]]; then
    collect_args+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${STARTER_KIT:-}" ]]; then
    collect_args+=(--starter-kit "${STARTER_KIT}")
fi
if [[ -n "${MODEL_MANIFEST:-}" ]]; then
    collect_args+=(--model-manifest "${MODEL_MANIFEST}")
fi
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.generate_fixtures \
    --output-dir "${OUTPUT_DIR}/fixtures"
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.collect_environment \
    "${collect_args[@]}"
set +e
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.smoke \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --image "${OUTPUT_DIR}/fixtures/competition_smoke.png" \
    --audio "${OUTPUT_DIR}/fixtures/competition_smoke.wav" \
    --video "${VIDEO_INPUT}" \
    --require-all-modalities \
    --output-dir "${OUTPUT_DIR}/smoke"
smoke_status=$?

benchmark_status=0
if [[ ${smoke_status} -eq 0 ]]; then
    "${PYTHON}" -m benchmarks.competition.minicpmo_ascend.benchmark \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --output-dir "${OUTPUT_DIR}/benchmark" \
        "$@"
    benchmark_status=$?
fi

stability_status=0
if [[ ${smoke_status} -eq 0 && ${benchmark_status} -eq 0 ]]; then
    "${PYTHON}" -m benchmarks.competition.minicpmo_ascend.benchmark \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --output-dir "${OUTPUT_DIR}/stability" \
        --modes text_audio \
        --concurrency "${STABILITY_CONCURRENCY}" \
        --num-requests "${STABILITY_REQUESTS}" \
        --warmups "${STABILITY_WARMUPS}" \
        --seed "${SEED}" \
        --thinker-max-tokens "${THINKER_MAX_TOKENS}" \
        --talker-max-tokens "${TALKER_MAX_TOKENS}"
    stability_status=$?
fi

gate_args=(
    --smoke-results "${OUTPUT_DIR}/smoke/smoke_results.json"
    --output "${OUTPUT_DIR}/gate.json"
)
if [[ -f "${OUTPUT_DIR}/benchmark/benchmark_results.json" ]]; then
    gate_args+=(--benchmark-results "${OUTPUT_DIR}/benchmark/benchmark_results.json")
fi
if [[ -f "${OUTPUT_DIR}/stability/benchmark_results.json" ]]; then
    gate_args+=(--stability-results "${OUTPUT_DIR}/stability/benchmark_results.json")
fi
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.correctness_gate "${gate_args[@]}"
gate_status=$?

report_status=0
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.report \
    --artifact-root "${OUTPUT_DIR}" \
    --environment "${OUTPUT_DIR}/environment.json" \
    --smoke-results "${OUTPUT_DIR}/smoke/smoke_results.json" \
    --benchmark-results "${OUTPUT_DIR}/benchmark/benchmark_results.json" \
    --stability-results "${OUTPUT_DIR}/stability/benchmark_results.json" \
    --gate "${OUTPUT_DIR}/gate.json" \
    --output "${OUTPUT_DIR}/baseline_report.md" \
    --manifest-output "${OUTPUT_DIR}/artifact_manifest.sha256"
report_status=$?
set -e

if [[ ${smoke_status} -ne 0 || ${benchmark_status} -ne 0 || ${stability_status} -ne 0 || ${gate_status} -ne 0 || ${report_status} -ne 0 ]]; then
    echo "Competition proxy suite failed: ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "Competition proxy suite passed: ${OUTPUT_DIR}"
