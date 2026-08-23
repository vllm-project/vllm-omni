#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Screen one- and two-GPU DLO residency, then fully validate each winner.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname -- "${REPO_ROOT}")}"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results/sm120-1gpu-2gpu-residency-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-${WORK_ROOT}/.venv/bin/python}"

RESIDENT_LAYER_CANDIDATES="${RESIDENT_LAYER_CANDIDATES:-20 25 30 35 40 45 50}"
PROBE_STEPS="${PROBE_STEPS:-5}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-1}"
FINAL_STEPS="${FINAL_STEPS:-50}"
FINAL_WARMUP_STEPS="${FINAL_WARMUP_STEPS:-2}"
MIN_HEADROOM_MIB="${MIN_HEADROOM_MIB:-4096}"
TWO_GPU_IDS="${TWO_GPU_IDS:-0,1}"
ONE_GPU_ID="${ONE_GPU_ID:-0}"
NUMA_NODE="${NUMA_NODE:-0}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"

RUNNER="${RUNNER:-${SCRIPT_DIR}/run_all_tasks.sh}"
SUMMARIZER="${SUMMARIZER:-${SCRIPT_DIR}/summarize_sm120_results.py}"
SELECTION_JSON="${RESULT_ROOT}/selection.json"
PROBE_CSV="${RESULT_ROOT}/probe_summary.csv"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found: ${PYTHON}" >&2
  exit 1
fi
for required_file in "${RUNNER}" "${SUMMARIZER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
if ! command -v setsid >/dev/null 2>&1; then
  echo "Required command not found: setsid" >&2
  exit 1
fi
for layers in ${RESIDENT_LAYER_CANDIDATES}; do
  if [[ ! ${layers} =~ ^[0-9]+$ ]] || (( layers < 20 || layers > 50 )); then
    echo "Resident-layer candidates must be integers in [20, 50], got: ${layers}" >&2
    exit 2
  fi
done

mkdir -p "${RESULT_ROOT}"

terminate_case_group() {
  local pgid="$1"
  local attempt

  if ! kill -0 -- "-${pgid}" 2>/dev/null; then
    return 0
  fi

  echo "Cleaning up processes left by case process group ${pgid}." >&2
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  for attempt in {1..20}; do
    if ! kill -0 -- "-${pgid}" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  kill -KILL -- "-${pgid}" 2>/dev/null || true
}

ACTIVE_CASE_PGID=""
cleanup_active_case() {
  if [[ -n "${ACTIVE_CASE_PGID}" ]]; then
    terminate_case_group "${ACTIVE_CASE_PGID}"
    ACTIVE_CASE_PGID=""
  fi
  return 0
}
trap cleanup_active_case EXIT

case_passed() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] && "${PYTHON}" - "${summary_path}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
actual = {task.get("task_id") for task in data.get("tasks", [])}
raise SystemExit(data.get("status") != "completed" or not {"t2va", "fl2va_first_frame"} <= actual)
PY
}

run_case() {
  local label="$1" gpu_ids="$2" num_gpus="$3" tp="$4" ulysses="$5"
  local layers="$6" steps="$7" warmup_steps="$8"
  local output_dir="${RESULT_ROOT}/${label}"
  local case_pid
  local case_status

  if case_passed "${output_dir}/summary.json" && [[ -f "${output_dir}/gpu_peak_memory.csv" ]]; then
    echo "=== ${label} already passed; skipping ==="
    return 0
  fi

  echo "=== ${label}: ${num_gpus} GPU, TP${tp} x Ulysses${ulysses}, resident ${layers} ==="
  setsid env \
    PYTHON="${PYTHON}" \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    NUM_GPUS="${num_gpus}" \
    TP_SIZE="${tp}" ULYSSES_DEGREE="${ulysses}" RING_DEGREE=1 \
    TEXT_ENCODER_TP_SIZE="${num_gpus}" \
    VAE_PATCH_PARALLEL_SIZE="${num_gpus}" \
    ATTENTION_BACKEND=CUDNN_ATTN \
    ENABLE_DLO=1 DLO_RESIDENT_LAYERS="${layers}" \
    ENFORCE_EAGER=1 RUN_REF2VA=0 INSTALL_EDITABLE="${INSTALL_EDITABLE}" \
    NUM_INFERENCE_STEPS="${steps}" WARMUP_STEPS="${warmup_steps}" \
    OUTPUT_DIR="${output_dir}" \
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    bash "${RUNNER}" &
  case_pid=$!
  ACTIVE_CASE_PGID="${case_pid}"

  if wait "${case_pid}"; then
    case_status=0
  else
    case_status=$?
  fi
  terminate_case_group "${case_pid}"
  ACTIVE_CASE_PGID=""

  if (( case_status == 0 )); then
    echo "=== ${label} passed ==="
    return 0
  fi
  echo "=== ${label} failed; continuing ===" >&2
  return 1
}

FAILED_CASES=()
for layers in ${RESIDENT_LAYER_CANDIDATES}; do
  run_case "probe-2g-tp1-u2-r${layers}" "${TWO_GPU_IDS}" 2 1 2 \
    "${layers}" "${PROBE_STEPS}" "${PROBE_WARMUP_STEPS}" || FAILED_CASES+=("probe-2g-tp1-u2-r${layers}")
  run_case "probe-2g-tp2-u1-r${layers}" "${TWO_GPU_IDS}" 2 2 1 \
    "${layers}" "${PROBE_STEPS}" "${PROBE_WARMUP_STEPS}" || FAILED_CASES+=("probe-2g-tp2-u1-r${layers}")
done

for layers in ${RESIDENT_LAYER_CANDIDATES}; do
  run_case "probe-1g-tp1-u1-r${layers}" "${ONE_GPU_ID}" 1 1 1 \
    "${layers}" "${PROBE_STEPS}" "${PROBE_WARMUP_STEPS}" || FAILED_CASES+=("probe-1g-tp1-u1-r${layers}")
done

if (( ${#FAILED_CASES[@]} )); then
  printf '%s\n' "${FAILED_CASES[@]}" > "${RESULT_ROOT}/failed_cases.txt"
else
  rm -f "${RESULT_ROOT}/failed_cases.txt"
fi

total_mib="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F',' -v wanted="${ONE_GPU_ID}" '
  {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
    gsub(/[^0-9]/, "", $2)
    if ($1 == wanted) print $2
  }
')"
if [[ -z "${total_mib}" ]]; then
  echo "Unable to read total memory for GPU ${ONE_GPU_ID}." >&2
  exit 1
fi

"${PYTHON}" - "${RESULT_ROOT}" "${total_mib}" "${MIN_HEADROOM_MIB}" "${PROBE_CSV}" "${SELECTION_JSON}" <<'PY'
import csv
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
total_mib = int(sys.argv[2])
min_headroom_mib = int(sys.argv[3])
csv_path = pathlib.Path(sys.argv[4])
selection_path = pathlib.Path(sys.argv[5])
pattern = re.compile(r"probe-(?P<gpus>[12])g-tp(?P<tp>\d+)-u(?P<ulysses>\d+)-r(?P<layers>\d+)")
rows = []

for case_dir in sorted(root.glob("probe-*")):
    match = pattern.fullmatch(case_dir.name)
    summary_path = case_dir / "summary.json"
    peak_path = case_dir / "gpu_peak_memory.csv"
    if match is None or not summary_path.is_file() or not peak_path.is_file():
        continue
    summary = json.loads(summary_path.read_text())
    tasks = summary.get("tasks", [])
    task_ids = {task.get("task_id") for task in tasks}
    complete = summary.get("status") == "completed" and {"t2va", "fl2va_first_frame"} <= task_ids
    with peak_path.open(newline="", encoding="utf-8") as handle:
        peaks = [int(row["peak_memory_mib"]) for row in csv.DictReader(handle)]
    peak_mib = max(peaks, default=0)
    latencies = [float(task["wall_time_s"]) for task in tasks if task.get("task_id") in {"t2va", "fl2va_first_frame"}]
    eligible = complete and len(latencies) == 2 and total_mib - peak_mib >= min_headroom_mib
    rows.append({
        "case": case_dir.name,
        "gpus": int(match["gpus"]),
        "tp": int(match["tp"]),
        "ulysses": int(match["ulysses"]),
        "resident_layers": int(match["layers"]),
        "worst_e2e_s": max(latencies, default=float("inf")),
        "mean_e2e_s": statistics.mean(latencies) if latencies else float("inf"),
        "peak_memory_mib": peak_mib,
        "headroom_mib": total_mib - peak_mib,
        "eligible": eligible,
    })

fieldnames = list(rows[0]) if rows else []
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

selection = {"metric": "minimum worst-case T2VA/FL2VA E2E", "min_headroom_mib": min_headroom_mib}
for gpus in (2, 1):
    eligible = [row for row in rows if row["gpus"] == gpus and row["eligible"]]
    if not eligible:
        raise SystemExit(f"No eligible {gpus}-GPU candidate passed with {min_headroom_mib} MiB headroom")
    selection[f"winner_{gpus}gpu"] = min(eligible, key=lambda row: (row["worst_e2e_s"], row["mean_e2e_s"]))

selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
print(json.dumps(selection, indent=2, sort_keys=True))
PY

read_winner() {
  local key="$1" field="$2"
  "${PYTHON}" - "${SELECTION_JSON}" "${key}" "${field}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]][sys.argv[3]])
PY
}

for gpu_count in 2 1; do
  key="winner_${gpu_count}gpu"
  tp="$(read_winner "${key}" tp)"
  ulysses="$(read_winner "${key}" ulysses)"
  layers="$(read_winner "${key}" resident_layers)"
  if (( gpu_count == 2 )); then
    gpu_ids="${TWO_GPU_IDS}"
  else
    gpu_ids="${ONE_GPU_ID}"
  fi
  run_case "final-${gpu_count}g-tp${tp}-u${ulysses}-r${layers}" \
    "${gpu_ids}" "${gpu_count}" "${tp}" "${ulysses}" "${layers}" \
    "${FINAL_STEPS}" "${FINAL_WARMUP_STEPS}"
done

"${PYTHON}" - "${RESULT_ROOT}" "${SELECTION_JSON}" <<'PY' | tee "${RESULT_ROOT}/conclusion.txt"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
selection = json.load(open(sys.argv[2], encoding="utf-8"))
print("Best validated SM120 deployment plans")
for gpu_count in (2, 1):
    winner = selection[f"winner_{gpu_count}gpu"]
    case = f"final-{gpu_count}g-tp{winner['tp']}-u{winner['ulysses']}-r{winner['resident_layers']}"
    summary = json.loads((root / case / "summary.json").read_text())
    tasks = {task["task_id"]: task for task in summary["tasks"]}
    print(
        f"{gpu_count} GPU: TP{winner['tp']} x Ulysses{winner['ulysses']}, "
        f"resident={winner['resident_layers']}; "
        f"T2VA={tasks['t2va']['wall_time_s']:.3f}s, "
        f"FL2VA={tasks['fl2va_first_frame']['wall_time_s']:.3f}s"
    )
print(f"Artifacts: {root}")
PY

"${PYTHON}" "${SUMMARIZER}" "${RESULT_ROOT}"/final-*
