#!/bin/bash

VLLM_NO_DEEP_GEMM=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 vllm serve "Tencent-Hunyuan/OmniWeaving" \
    --omni \
    --tensor-parallel-size 1 \
    --port 8000
