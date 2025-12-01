#!/bin/bash

# Detect if ROCm is available
if command -v rocminfo &> /dev/null || [ -d "/opt/rocm" ]; then
    echo "ROCm detected - Running with ROCm-specific environment variables..."
    VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_MHA=1 \
    VLLM_ROCM_USE_AITER_LINEAR=0 \
    VLLM_ROCM_USE_AITER_RMSNORM=0 \
    MIOPEN_FIND_MODE=FAST \
    python end2end.py --output-wav output_audio \
                    --query-type text \
                    --txt-prompts top10.txt
else
    python end2end.py --output-wav output_audio \
                    --query-type text \
                    --txt-prompts top10.txt
fi
