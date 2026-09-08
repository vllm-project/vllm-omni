#!/bin/bash

set -euo pipefail

MODEL="${MODEL:-robbyant/lingbot-video-dense-1.3b}"
PORT="${PORT:-8099}"

vllm serve "${MODEL}" --omni \
    --model-class-name LingBotVideoPipeline \
    --port "${PORT}"
