#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Select the largest one-GPU DLO resident-layer count with a measured margin.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname -- "${REPO_ROOT}")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUMA_NODE="${NUMA_NODE:-0}"
DLO_AUTO_HEADROOM_MIB="${DLO_AUTO_HEADROOM_MIB:-6144}"
DLO_AUTO_CANDIDATES="${DLO_AUTO_CANDIDATES:-35 30 25 20}"
DLO_AUTO_PROBE_STEPS="${DLO_AUTO_PROBE_STEPS:-5}"
DLO_AUTO_ROOT="${DLO_AUTO_ROOT:-${WORK_ROOT}/results/minimax-h3-dlo-auto-$(date -u +%Y%m%dT%H%M%SZ)}"
FINAL_OUTPUT_DIR="${OUTPUT_DIR:-${DLO_AUTO_ROOT}/final}"
FINAL_STEPS="${NUM_INFERENCE_STEPS:-50}"
FINAL_WARMUPS="${WARMUP_STEPS:-2}"

IFS=',' read -r -a SELECTED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#SELECTED_GPUS[@]}" -ne 1 ]]; then
  echo "DLO auto selection supports exactly one physical GPU." >&2
  exit 2
fi

total_mib="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F',' -v wanted="${SELECTED_GPUS[0]}" '
  $1 ~ "^" wanted "[[:space:]]*$" { gsub(/[^0-9]/, "", $2); print $2 }
')"
if [[ -z "${total_mib}" ]]; then
  echo "GPU ${SELECTED_GPUS[0]} was not found by nvidia-smi." >&2
  exit 1
fi

mkdir -p "${DLO_AUTO_ROOT}"
selected_layers=""
selected_peak_mib=""
for layers in ${DLO_AUTO_CANDIDATES}; do
  probe_dir="${DLO_AUTO_ROOT}/probe-r${layers}"
  echo "[DLO auto] probing resident_layers=${layers}"
  if env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_GPUS=1 TP_SIZE=1 ULYSSES_DEGREE=1 RING_DEGREE=1 \
    TEXT_ENCODER_TP_SIZE=1 VAE_PATCH_PARALLEL_SIZE=1 \
    ENABLE_DLO=1 DLO_RESIDENT_LAYERS="${layers}" \
    NUM_INFERENCE_STEPS="${DLO_AUTO_PROBE_STEPS}" WARMUP_STEPS=1 RUN_REF2VA=0 \
    OUTPUT_DIR="${probe_dir}" \
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    bash "${SCRIPT_DIR}/run_all_tasks.sh"; then
    peak_mib="$(awk -F',' 'NR > 1 && $2 > peak { peak = $2 } END { print int(peak) }' "${probe_dir}/gpu_peak_memory.csv")"
    headroom_mib="$((total_mib - peak_mib))"
    echo "[DLO auto] resident_layers=${layers}: peak=${peak_mib} MiB, headroom=${headroom_mib} MiB"
    if (( headroom_mib >= DLO_AUTO_HEADROOM_MIB )); then
      selected_layers="${layers}"
      selected_peak_mib="${peak_mib}"
      break
    fi
    echo "[DLO auto] rejecting resident_layers=${layers}: requires ${DLO_AUTO_HEADROOM_MIB} MiB headroom"
  else
    echo "[DLO auto] resident_layers=${layers} failed; trying the next lower candidate"
  fi
done

if [[ -z "${selected_layers}" ]]; then
  echo "No DLO resident-layer candidate fit with ${DLO_AUTO_HEADROOM_MIB} MiB headroom." >&2
  exit 1
fi

echo "[DLO auto] selected resident_layers=${selected_layers}"
env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_GPUS=1 TP_SIZE=1 ULYSSES_DEGREE=1 RING_DEGREE=1 \
  TEXT_ENCODER_TP_SIZE=1 VAE_PATCH_PARALLEL_SIZE=1 \
  ENABLE_DLO=1 DLO_RESIDENT_LAYERS="${selected_layers}" \
  NUM_INFERENCE_STEPS="${FINAL_STEPS}" WARMUP_STEPS="${FINAL_WARMUPS}" \
  OUTPUT_DIR="${FINAL_OUTPUT_DIR}" \
  numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  bash "${SCRIPT_DIR}/run_all_tasks.sh"

final_peak_mib="$(awk -F',' 'NR > 1 && $2 > peak { peak = $2 } END { print int(peak) }' "${FINAL_OUTPUT_DIR}/gpu_peak_memory.csv")"
"${PYTHON:-python3}" - "${FINAL_OUTPUT_DIR}/summary.json" "${selected_layers}" "${selected_peak_mib}" "${final_peak_mib}" "${total_mib}" "${DLO_AUTO_HEADROOM_MIB}" <<'PYTHON'
import json
import sys

path, layers, probe_peak, final_peak, total, required = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)
summary["dlo_autotune"] = {
    "mode": "safe",
    "selected_resident_layers": int(layers),
    "probe_peak_memory_mib": int(probe_peak),
    "probe_headroom_mib": int(total) - int(probe_peak),
    "peak_memory_mib": int(final_peak),
    "headroom_mib": int(total) - int(final_peak),
    "required_headroom_mib": int(required),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYTHON
