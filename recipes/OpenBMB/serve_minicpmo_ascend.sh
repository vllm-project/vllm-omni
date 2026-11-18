#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the prepared original/thinker-w8a8 model tree}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the MiniCPM-o Thinker DSpark checkpoint}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="${RUNTIME_DIR:-$(mktemp -d -t minicpmo-ascend.XXXXXXXX)}"
mkdir -p "$runtime_dir"
runtime_dir="$(cd "$runtime_dir" && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export MINICPMO_TALKER_STATIC_EXCLUDED_OP=ScatterPaKvCache
export MINICPMO_NPU_DSPARK_NATIVE_GATHER=1 MINICPMO_NPU_KSTEP_SKIP_TERMINAL_TAIL=1
export MINICPMO_C8_SECONDARY_VOCODER_HOST_CACHE=1 MINICPMO_C8_SECONDARY_VOCODER_TIGHT_MEMORY=1
export MINICPMO_NPU_TALKER_FUSED_MODE_FILE="$runtime_dir/talker-mode"
export MINICPMO_NPU_TALKER_NARROW_C8_MODE_FILE="$runtime_dir/talker-batch-mode"
export MINICPMO_NPU_TALKER_HYBRID_CONTEXT_FILE="$runtime_dir/talker-hybrid"
export MINICPMO_C8_TALKER_ROUTE_FILE="$runtime_dir/talker-route"
export MINICPMO_C8_VOCODER_ROUTE_FILE="$runtime_dir/vocoder-route"
printf 'active_thinker_prefill\n' > "$MINICPMO_NPU_TALKER_FUSED_MODE_FILE"
printf 'active_fixed\n' > "$MINICPMO_NPU_TALKER_NARROW_C8_MODE_FILE"
touch "$MINICPMO_NPU_TALKER_HYBRID_CONTEXT_FILE"
printf '%s deployment\n' "${TALKER_ROUTE:-quarter}" > "$MINICPMO_C8_TALKER_ROUTE_FILE"
printf '%s deployment\n' "${VOCODER_ROUTE:-quarter}" > "$MINICPMO_C8_VOCODER_ROUTE_FILE"

python3 - "$repo_root" "$runtime_dir" "$MODEL_ROOT" "$DRAFT_MODEL" "${MAX_NUM_SEQS:-16}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

repo, runtime, model, draft = (Path(p).resolve() for p in sys.argv[1:5])
seqs = int(sys.argv[5])
if seqs <= 0:
    raise ValueError("MAX_NUM_SEQS must be positive")
for directory in (model, model / "original", model / "thinker-w8a8", draft):
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing prepared model config: {directory / 'config.json'}")
quant = json.loads((model / "thinker-w8a8/config.json").read_text()).get("quantization_config", {})
if quant.get("quant_method") != "compressed-tensors":
    raise ValueError("thinker-w8a8 must contain the qualified compressed-tensors W8A8 checkpoint")
config = yaml.safe_load((repo / "vllm_omni/deploy/minicpmo_4_5_ascend.yaml").read_text())
for stage in config["stages"]:
    stage["max_num_seqs"] = seqs
    if stage["stage_id"] == 0:
        stage["speculative_config"]["model"] = str(draft)
(runtime / "deploy.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Rendered deployment: {runtime / 'deploy.yaml'}", flush=True)
PY

cd "$repo_root"
exec numactl --cpunodebind="${NUMA_NODE:-0}" --preferred="${NUMA_NODE:-0}" \
  python3 -m vllm_omni.entrypoints.cli.main serve "$MODEL_ROOT" \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --omni --trust-remote-code --interleave-mm-strings \
  --media-io-kwargs '{"video":{"fps":1,"num_frames":128}}' \
  --stage-init-timeout 1000 --init-timeout 1200 --log-stats \
  --deploy-config "$runtime_dir/deploy.yaml" "$@"
