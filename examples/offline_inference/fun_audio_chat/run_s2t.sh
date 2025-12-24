#!/bin/bash
# Run Fun-Audio-Chat in S2T (Speech-to-Text) mode
# This uses only Stage 0 (Main model) for audio understanding

# Default model path - can be overridden with local path
MODEL_PATH="${MODEL_PATH:-FunAudioLLM/Fun-Audio-Chat-8B}"

# Check if local model exists
LOCAL_MODEL="$(dirname "$0")/../../../pretrained_models/Fun-Audio-Chat-8B"
if [ -d "$LOCAL_MODEL" ]; then
    echo "[Info] Using local model: $LOCAL_MODEL"
    MODEL_PATH="$LOCAL_MODEL"
fi

python end2end.py --mode s2t --output-dir output_s2t --model-path "$MODEL_PATH"
