#!/usr/bin/env bash
# Shared configuration only: source this file; it does not launch processes.
export VLLM_OMNI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VLLM_PYTHON="${VLLM_PYTHON:-$VLLM_OMNI_ROOT/.venv/bin/python}"
# Keep CANN's Python and library search paths. Always import this checkout first.
export PYTHONPATH="$VLLM_OMNI_ROOT:$VLLM_OMNI_ROOT/plugins/cosmos3_client${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$VLLM_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export COSMOS3_DEVICE="${COSMOS3_DEVICE:-cuda}"
COSMOS3_WORKER_ARGS=()
case "$COSMOS3_DEVICE" in
  cuda|gpu)
    export VLLM_TARGET_DEVICE=cuda
    export VLLM_PLUGINS=vllm_omni_register_models
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    ;;
  npu|ascend)
    export VLLM_TARGET_DEVICE=npu
    export VLLM_PLUGINS=ascend,vllm_omni_register_models,ascend_kv_connector,ascend_model,ascend_model_loader
    export ASCEND_RT_VISIBLE_DEVICES="${COSMOS3_NPU_DEVICES:-0}"
    export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    export COSMOS3_STRICT_PRECISION="${COSMOS3_STRICT_PRECISION:-1}"
    COSMOS3_WORKER_ARGS=(--dtype "${COSMOS3_DTYPE:-float32}" --enforce-eager --diffusion-attention-backend TORCH_SDPA)
    ;;
  *) echo "Unsupported COSMOS3_DEVICE=$COSMOS3_DEVICE; use cuda or npu" >&2; return 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
