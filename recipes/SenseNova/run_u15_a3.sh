#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

# Run from the repository root. MODEL may also point to a local base checkpoint.
MODE="${1:-serve}"
MODEL="${MODEL:-sensenova/SenseNova-U1.5-8B-MoT}"
TP="${TP:-1}"
PORT="${PORT:-8091}"
PROMPT="${PROMPT:-A mountain lake at sunrise, detailed reflections, natural colors.}"
LORA_PATH="${LORA_PATH:-}"
OUTPUT="${OUTPUT:-sensenova-u15-tp${TP}}"

case "$TP" in
    1|2) ;;
    *) echo "TP must be 1 or 2" >&2; exit 2 ;;
esac
PROFILE_ARGS=()
if [[ "${PROFILE:-0}" == 1 ]]; then
    PROFILE_ARGS=(--enable-diffusion-pipeline-profiler)
fi
LORA_ARGS=()
STEPS=50
CFG=4.0
if [[ -n "$LORA_PATH" ]]; then
    if [[ ! -f "$LORA_PATH" || "${LORA_PATH##*/}" != "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors" ]]; then
        echo "LORA_PATH must name the original SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors file" >&2
        exit 2
    fi
    LORA_ARGS=(--lora-path "$LORA_PATH" --lora-backend distill)
    STEPS=8
    CFG=1.0
fi

case "$MODE" in
    serve)
        exec vllm serve "$MODEL" --omni --port "$PORT" \
            --dtype bfloat16 --tensor-parallel-size "$TP" --enforce-eager \
            "${PROFILE_ARGS[@]}" "${LORA_ARGS[@]}"
        ;;
    generate)
        exec python examples/online_serving/sensenova_u1/openai_chat_client.py \
            --server "http://127.0.0.1:$PORT" --modality text2img --prompt "$PROMPT" \
            --width 2560 --height 1760 --num-steps "$STEPS" --cfg-scale "$CFG" \
            --cfg-norm none --timestep-shift 3.0 --seed 42 --output "${OUTPUT}.png"
        ;;
    benchmark)
        if [[ -z "$LORA_PATH" ]]; then
            echo "The 8-step benchmark requires LORA_PATH (and the same adapter on the running server)" >&2
            exit 2
        fi
        # Metadata is captured where this script runs. Run beside the server in
        # the same software environment; add CANN/driver details to SERVER_METADATA.
        SERVER_METADATA="${SERVER_METADATA:-}"
        if [[ -z "$SERVER_METADATA" ]]; then SERVER_METADATA='{}'; fi
        METADATA=$(python - "$TP" "$LORA_PATH" "$SERVER_METADATA" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

metadata = json.loads(sys.argv[3])
metadata.update(tp=int(sys.argv[1]), dtype="bfloat16", python=platform.python_version())
metadata["lora_file"] = Path(sys.argv[2]).name
with open(sys.argv[2], "rb") as stream:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
metadata["lora_sha256"] = digest.hexdigest()
metadata["omni_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
metadata["omni_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
metadata["packages"] = {name: importlib.metadata.version(name) for name in
                        ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "diffusers")}
print(json.dumps(metadata))
PY
)
        exec python -m benchmarks.diffusion.bench_stage_latency \
            --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --prompt "$PROMPT" \
            --width 2560 --height 1760 --num-inference-steps 8 --seed 42 \
            --warmup "${WARMUP:-2}" --repeats "${REPEATS:-5}" \
            --text-stage SenseNovaU1Pipeline._prepare_t2i \
            --generation-stage SenseNovaU1Pipeline._run_denoising_loop \
            --extra-body '{"think":false,"cfg_scale":1.0,"cfg_norm":"none","timestep_shift":3.0,"t_eps":0.02}' \
            --metadata "$METADATA" --output-prefix "$OUTPUT"
        ;;
    *) echo "Usage: bash recipes/SenseNova/run_u15_a3.sh {serve|generate|benchmark}" >&2; exit 2 ;;
esac
