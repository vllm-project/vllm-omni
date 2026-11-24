#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

usage() {
  echo "Usage: $0 /path/to/MiniMax-H3 [--num-gpus N]"
  echo "Set LORA_PATH to minimax_h3_turbo_v4_step600_ema.safetensors."
  echo "PORT defaults to 9098. The validated profile uses 4 Ascend A5 devices."
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 && $# -ne 3 ]] || [[ ! -f "$1/Ref2VA/model_index.json" ]]; then
  usage >&2
  exit 2
fi

model_root=$1
lora_path=${LORA_PATH:-}
if [[ ${lora_path##*/} != minimax_h3_turbo_v4_step600_ema.safetensors || ! -f "$lora_path" || ! -r "$lora_path" ]]; then
  echo "Set LORA_PATH to a readable minimax_h3_turbo_v4_step600_ema.safetensors file." >&2
  exit 2
fi

num_gpus=4
if [[ $# -eq 3 ]]; then
  if [[ $2 != --num-gpus || ! $3 =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected --num-gpus followed by a positive integer." >&2
    exit 2
  fi
  num_gpus=$3
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

exec vllm serve "$model_root" \
  --omni --task-type ref2va \
  --host 127.0.0.1 --port "${PORT:-9098}" --trust-remote-code \
  --num-gpus "$num_gpus" --usp "$num_gpus" --ring 1 --text-encoder-tp-size "$num_gpus" \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size "$num_gpus" \
  --enable-diffusion-pipeline-profiler \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
  --lora-path "$lora_path" --lora-backend distill --lora-scale 1 \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0,"end_step":0},"rainfusion":{"image_sparsity":0.8}}}' \
  --kv-offload-config '{"mode":"on","kv_storage":"npu","kv_refresh_interval":8,"kv_host_quantization":"none"}'
