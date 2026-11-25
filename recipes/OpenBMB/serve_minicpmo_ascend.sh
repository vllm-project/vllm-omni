#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

model="${MODEL_ROOT:-openbmb/MiniCPM-o-4_5}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="${RUNTIME_DIR:-$(mktemp -d -t minicpmo-ascend.XXXXXXXX)}"
mkdir -p "$runtime_dir"
runtime_dir="$(cd "$runtime_dir" && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MINICPMO_TALKER_STATIC_EXCLUDED_OP=ScatterPaKvCache
export MINICPMO_NPU_DSPARK_NATIVE_GATHER=1 MINICPMO_NPU_KSTEP_SKIP_TERMINAL_TAIL=1
export MINICPMO_NPU_THINKER_PREFILL_STARTUP="$runtime_dir/thinker-prefill-startup"
export MINICPMO_NPU_TALKER_PREFILL_STARTUP="$runtime_dir/talker-prefill-startup"
export MINICPMO_NPU_TALKER_FUSED_MODE_FILE="$runtime_dir/talker-mode"
export MINICPMO_NPU_TALKER_NARROW_C8_MODE_FILE="$runtime_dir/talker-batch-mode"
export MINICPMO_NPU_TALKER_HYBRID_CONTEXT_FILE="$runtime_dir/talker-hybrid"
export MINICPMO_C8_TALKER_ROUTE_FILE="$runtime_dir/talker-route"
export MINICPMO_C8_VOCODER_ROUTE_FILE="$runtime_dir/vocoder-route"
printf 'tail_capacity\n' > "$MINICPMO_NPU_TALKER_FUSED_MODE_FILE"
printf '%s\n' "${TALKER_BATCH_MODE:-active_fixed_padded}" > "$MINICPMO_NPU_TALKER_NARROW_C8_MODE_FILE"
touch "$MINICPMO_NPU_TALKER_HYBRID_CONTEXT_FILE"
printf '%s deployment\n' "${TALKER_ROUTE:-half}" > "$MINICPMO_C8_TALKER_ROUTE_FILE"
printf '%s deployment\n' "${VOCODER_ROUTE:-adaptive_secondary}" > "$MINICPMO_C8_VOCODER_ROUTE_FILE"

python3 - "$repo_root" "$runtime_dir" "${DRAFT_MODEL:-}" "${DEPLOY_CONFIG:-}" "${MAX_NUM_SEQS:-}" "${THINKER_W8A8:-0}" <<'PY'
import sys
from pathlib import Path

import yaml
from vllm_omni.config.stage_config import resolve_deploy_yaml

repo, runtime = (Path(p).resolve() for p in sys.argv[1:3])
draft, deployment, seqs_arg, thinker_w8a8 = sys.argv[3:]
default_deploy = repo / "vllm_omni/deploy/minicpmo_4_5_2gpu.yaml"
deploy_path = Path(deployment).resolve() if deployment else default_deploy
seqs = int(seqs_arg) if seqs_arg else None
if seqs is not None and seqs <= 0:
    raise ValueError("MAX_NUM_SEQS must be positive")
config = resolve_deploy_yaml(deploy_path)
if seqs is not None:
    for stage in config["stages"]:
        stage["max_num_seqs"] = seqs
for stage in config["platforms"]["npu"]["stages"]:
    if seqs is not None:
        stage["max_num_seqs"] = seqs
    if stage["stage_id"] != 0:
        continue
    if thinker_w8a8 == "1":
        stage["model_subdir"] = "thinker-w8a8"
        stage.setdefault("additional_config", {})["minicpmo_visual_a16_prefill"] = True
    if draft:
        public_draft = "Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm"
        local_draft = Path(draft).expanduser()
        spec = {
            "method": "dspark",
            "model": str(local_draft.resolve()) if local_draft.is_dir() else draft,
            "num_speculative_tokens": 7,
            "draft_tensor_parallel_size": 1,
        }
        if draft == public_draft:
            spec["revision"] = "c4c310aca2c768343165261cb22c21b90c1f79e0"
        stage["speculative_config"] = spec
        stage["async_scheduling"] = False
(runtime / "deploy.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Rendered deployment: {runtime / 'deploy.yaml'}", flush=True)
PY

cd "$repo_root"
exec numactl --cpunodebind="${NUMA_NODE:-0}" --preferred="${NUMA_NODE:-0}" \
  python3 -m vllm_omni.entrypoints.cli.main serve "$model" \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --omni --trust-remote-code \
  --stage-init-timeout 1000 --init-timeout 1200 --log-stats \
  --deploy-config "$runtime_dir/deploy.yaml" "$@"
