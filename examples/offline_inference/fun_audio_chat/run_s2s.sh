#!/bin/bash
# Run Fun-Audio-Chat in S2S (Speech-to-Speech) mode on single GPU
# Requires ~35GB VRAM for all 3 stages

# Default model path - can be overridden with local path
MODEL_PATH="${MODEL_PATH:-FunAudioLLM/Fun-Audio-Chat-8B}"

# Check if local model exists
LOCAL_MODEL="$(dirname "$0")/../../../pretrained_models/Fun-Audio-Chat-8B"
if [ -d "$LOCAL_MODEL" ]; then
    echo "[Info] Using local model: $LOCAL_MODEL"
    MODEL_PATH="$LOCAL_MODEL"
fi

python end2end.py --mode s2s --output-dir output_s2s --model-path "$MODEL_PATH"
