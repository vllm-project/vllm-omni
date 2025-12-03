#!/bin/bash

# Detect if ROCm is available
if command -v rocminfo &> /dev/null || [ -d "/opt/rocm" ]; then
    echo "ROCm detected - Running with ROCm-specific environment variables..."
    export MIOPEN_FIND_MODE=FAST
fi

python end2end.py --output-wav output_audio \
                  --query-type text \
                  --txt-prompts top10.txt
