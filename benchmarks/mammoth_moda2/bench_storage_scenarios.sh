#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# MODEL_LOCAL=/local/model MODEL_NFS=/nfs/model OUT_DIR=./startup-bench bash "$0"
# Optional SCENARIOS: local-cold local-warm nfs-cold nfs-warm.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUT_DIR:-./startup-bench}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-vllm_omni/deploy/mammoth_moda2.yaml}
STEPS=${STEPS:-50}
SIZE=${SIZE:-1024}
REPEAT=${REPEAT:-2}
SEED=${SEED:-42}
EXTRA=${EXTRA:-'{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": '$STEPS'}'}
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/json" "$OUT_DIR/images"
rm -f "$OUT_DIR/summary.json" "$OUT_DIR/summary.md"

prepare_cache() {
  python - "$1" "$2" <<'PY'
import os
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).glob("*.safetensors"))
if not files:
    raise SystemExit("no checkpoint shards found")
for path in files:
    with path.open("rb") as f:
        if sys.argv[2] == "cold":
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        else:
            while f.read(8 * 1024 * 1024):
                pass
print("[bench]", "requested eviction of" if sys.argv[2] == "cold" else "read", len(files), "shards")
PY
}

default_scenarios=""
if [ -n "${MODEL_LOCAL:-}" ]; then default_scenarios="local-cold local-warm"; fi
if [ -n "${MODEL_NFS:-}" ]; then default_scenarios="$default_scenarios nfs-cold nfs-warm"; fi
read -r -a scenarios <<< "${SCENARIOS:-$default_scenarios}"
if [ "${#scenarios[@]}" -eq 0 ]; then echo "Set MODEL_LOCAL or MODEL_NFS" >&2; exit 1; fi
logs=()
for label in "${scenarios[@]}"; do
  case "$label" in
    local-cold|local-warm) model=${MODEL_LOCAL:?Set MODEL_LOCAL} ;;
    nfs-cold|nfs-warm) model=${MODEL_NFS:?Set MODEL_NFS} ;;
    *) echo "Unknown scenario: $label" >&2; exit 1 ;;
  esac
  prepare_cache "$model" "${label##*-}"
  log="$OUT_DIR/logs/$label.log"
  # Remove previous outputs for this label so a failed rerun cannot leave stale results.
  rm -f "$OUT_DIR/json/$label.json" "$OUT_DIR/images/$label.png"
  if python "$HERE/bench_startup.py" --model "$model" --deploy-config "$DEPLOY_CONFIG" \
    --height "$SIZE" --width "$SIZE" --seed "$SEED" --extra-body "$EXTRA" --repeat "$REPEAT" \
    --label "$label" --output-json "$OUT_DIR/json/$label.json" --save-image "$OUT_DIR/images/$label.png" \
    > "$log" 2>&1; then
    logs+=("$log")
  else
    rc=$?
    echo "[bench] $label failed (exit $rc); see $log" >&2
    exit "$rc"
  fi
done
python "$HERE/parse_startup_log.py" "${logs[@]}" --require-complete --markdown \
  --json-out "$OUT_DIR/summary.json" | tee "$OUT_DIR/summary.md"
echo BENCH_ALL_DONE
