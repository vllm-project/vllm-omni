#!/bin/bash
# Launch vLLM-Omni server for Vevo2.
#
# Prerequisites:
#   1. Amphion on PYTHONPATH (Vevo2 is not on PyPI):
#        git clone https://github.com/open-mmlab/Amphion.git
#        export PYTHONPATH=$PWD/Amphion:$PYTHONPATH
#        pip install -r Amphion/models/svc/vevo2/requirements.txt
#
#   2. Vevo2 checkpoint downloaded (CC BY-NC-ND 4.0 -- non-commercial only):
#        hf download RMSnow/Vevo2 --local-dir ./ckpts/Vevo2
#
#   3. One-time checkpoint init. The published repo ships no root config.json,
#      no root weight file and no root tokenizer files, so the server cannot
#      load it as downloaded. This writes them (idempotent; safe to re-run):
#        python examples/offline_inference/text_to_speech/vevo2/init_vevo2_checkpoint.py ./ckpts/Vevo2
#
# Usage:
#   ./run_server.sh
#   MODEL=./ckpts/Vevo2 PORT=8092 CUDA_VISIBLE_DEVICES=0 ./run_server.sh

set -e

MODEL="${MODEL:-./ckpts/Vevo2}"
PORT="${PORT:-8092}"

echo "Starting Vevo2 server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni
