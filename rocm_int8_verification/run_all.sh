#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ID="${ROCM_VERIFY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESULT_DIR="$SCRIPT_DIR/results/$RUN_ID"
CACHE_DIR="$SCRIPT_DIR/cache"
RUNTIME_ROOT="$SCRIPT_DIR/runtime"
GPU_IDS="${GPU_IDS:-0,1}"
SINGLE_GPU_ID="${GPU_IDS%%,*}"
BAGEL_MODEL="${BAGEL_MODEL:-ByteDance-Seed/BAGEL-7B-MoT}"
RUN_AITER="${RUN_AITER:-auto}"

if [[ "$GPU_IDS" != *,* ]]; then
    echo "GPU_IDS must contain at least two comma separated GPU IDs, for example 0,1." >&2
    exit 2
fi

if [[ -z "${VLLM_TEST_MINIMAX_H3_FL2VA_MODEL:-}" ]]; then
    echo "Set VLLM_TEST_MINIMAX_H3_FL2VA_MODEL to the MiniMax-H3 FL2VA checkpoint path." >&2
    exit 2
fi

mkdir -p \
    "$RESULT_DIR/pytest-cache" \
    "$RESULT_DIR/pytest-temp" \
    "$CACHE_DIR/huggingface" \
    "$CACHE_DIR/torch_extensions" \
    "$CACHE_DIR/torchinductor" \
    "$CACHE_DIR/triton" \
    "$CACHE_DIR/vllm" \
    "$CACHE_DIR/xdg" \
    "$RUNTIME_ROOT"

# vLLM creates Unix sockets under TMPDIR. Keep the runtime path short enough
# for the 107-character sockaddr_un limit while retaining it in this folder.
RUNTIME_DIR="$(mktemp -d "$RUNTIME_ROOT/run-XXXXXXXX")"

export ROCM_VERIFY_DIR="$RESULT_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TMPDIR="$RUNTIME_DIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

cd "$REPO_ROOT"

exec > >(tee -a "$RESULT_DIR/run-all.log") 2>&1

finish() {
    local status=$1
    if [[ $status -eq 0 ]]; then
        echo "ROCm INT8 verification passed."
    else
        echo "ROCm INT8 verification failed with status $status."
    fi
    echo "Results: $RESULT_DIR"
    return "$status"
}
trap 'finish "$?"' EXIT

run_logged() {
    local name=$1
    shift
    echo
    echo "Running $name"
    "$@" 2>&1 | tee "$RESULT_DIR/$name.log"
}

run_pytest() {
    local name=$1
    shift
    run_logged "$name" \
        "$@" \
        -o addopts='' \
        -o "cache_dir=$RESULT_DIR/pytest-cache/$name" \
        --basetemp="$RESULT_DIR/pytest-temp/$name" \
        --run-level advanced_model
}

check_environment() {
    python - <<'PY'
import os
import subprocess

import torch
import vllm
import vllm_omni

from vllm_omni.platforms import current_omni_platform

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("vLLM:", vllm.__version__)
print("vLLM-Omni source:", vllm_omni.__file__)
print("Git commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("Device type:", current_omni_platform.device_type)
print("Device count:", current_omni_platform.get_device_count())
print("ROCm platform:", current_omni_platform.is_rocm())
print("Verification directory:", os.environ["ROCM_VERIFY_DIR"])
print("Python temporary directory:", os.environ["TMPDIR"])
print("vLLM cache:", os.environ["VLLM_CACHE_ROOT"])

assert torch.version.hip is not None
assert current_omni_platform.is_rocm()
assert current_omni_platform.get_device_count() >= 2
assert vllm_omni.__file__.startswith(os.getcwd())
PY
}

aiter_is_installed() {
    python - <<'PY'
import importlib.util

raise SystemExit(0 if importlib.util.find_spec("aiter") is not None else 1)
PY
}

run_minimax() {
    python "$SCRIPT_DIR/minimax_int8_tp2.py" \
        --model "$VLLM_TEST_MINIMAX_H3_FL2VA_MODEL" \
        --output "$RESULT_DIR/minimax-int8-tp2-output.npz"
}

run_bagel() {
    local bagel_config="$RESULT_DIR/bagel-int8-tp2.yaml"
    cp vllm_omni/deploy/bagel_single_stage.yaml "$bagel_config"
    sed -i 's/devices: "0"/devices: "0,1"/' "$bagel_config"
    grep -q 'devices: "0,1"' "$bagel_config"

    python examples/offline_inference/text_to_image/text_to_image.py \
        --model "$BAGEL_MODEL" \
        --deploy-config "$bagel_config" \
        --quantization int8 \
        --tensor-parallel-size 2 \
        --enforce-eager \
        --prompt "A futuristic city skyline at twilight" \
        --height 512 \
        --width 512 \
        --num-inference-steps 2 \
        --seed 42 \
        --extra-body '{"timestep_shift":3.0,"cfg_text_scale":4.0,"cfg_img_scale":1.5,"cfg_interval":[0.4,1.0],"cfg_renorm_type":"global","cfg_renorm_min":0.0}' \
        --output "$RESULT_DIR/bagel-int8-tp2.png"

    test -s "$RESULT_DIR/bagel-int8-tp2.png"
    echo "BAGEL INT8 TP2 passed"
    echo "Saved: $RESULT_DIR/bagel-int8-tp2.png"
}

echo "Repository: $REPO_ROOT"
echo "Results: $RESULT_DIR"
echo "GPUs: $GPU_IDS"
echo "MiniMax model: $VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"
echo "BAGEL model: $BAGEL_MODEL"
echo "AITER check: $RUN_AITER"

run_logged environment check_environment
python -m pip freeze > "$RESULT_DIR/python-packages.txt"

run_pytest affected-suite \
    python -m pytest \
    tests/diffusion/quantization/test_int8_config.py \
    tests/diffusion/quantization/test_bitsandbytes_config.py \
    tests/diffusion/models/minimax_h3/test_minimax_h3_quantization.py \
    tests/diffusion/models/bagel/test_bagel_quantization.py \
    -m core_model \
    -vv -s -rs

run_pytest triton-kernel \
    env \
    HIP_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    VLLM_ROCM_USE_AITER=0 \
    python -m pytest \
    tests/diffusion/quantization/test_int8_config.py::TestGPUInt8Smoke \
    -m 'core_model and rocm' \
    -vv -s -rs
grep -q 'Selected TritonInt8ScaledMMLinearKernel' "$RESULT_DIR/triton-kernel.log"

case "$RUN_AITER" in
    1 | true | yes)
        run_aiter=1
        ;;
    0 | false | no)
        run_aiter=0
        ;;
    auto)
        if aiter_is_installed; then
            run_aiter=1
        else
            run_aiter=0
            echo "AITER is not installed, so the AITER kernel check is skipped."
        fi
        ;;
    *)
        echo "RUN_AITER must be auto, 1, or 0." >&2
        exit 2
        ;;
esac

if [[ $run_aiter -eq 1 ]]; then
    run_pytest aiter-kernel \
        env \
        HIP_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
        CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
        VLLM_ROCM_USE_AITER=1 \
        VLLM_ROCM_USE_AITER_LINEAR=1 \
        python -m pytest \
        tests/diffusion/quantization/test_int8_config.py::TestGPUInt8Smoke \
        -m 'core_model and rocm' \
        -vv -s -rs
    grep -q 'Selected AiterInt8ScaledMMLinearKernel' "$RESULT_DIR/aiter-kernel.log"
fi

run_pytest two-gpu-parity \
    env \
    HIP_VISIBLE_DEVICES="$GPU_IDS" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python -m pytest \
    tests/diffusion/quantization/test_int8_config.py::test_shared_quantizer_matches_native_kernel_on_two_gpus \
    -m 'core_model and rocm' \
    -vv -s -rs

run_logged minimax-int8-tp2 run_minimax
run_logged bagel-int8-tp2 run_bagel

printf '%s\n' \
    "ROCm INT8 verification passed." \
    "Git commit: $(git rev-parse HEAD)" \
    "Results: $RESULT_DIR" \
    > "$RESULT_DIR/SUMMARY.txt"

echo
echo "Verification artifacts"
ls -lh "$RESULT_DIR"
