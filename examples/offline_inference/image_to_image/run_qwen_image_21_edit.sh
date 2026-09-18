#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

python "$(dirname "${BASH_SOURCE[0]}")/image_edit.py" \
    --model Qwen/Qwen-Image-2.1 \
    --color-format RGBA \
    --seed 42 \
    --image qwen_bear.png \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
    --negative-prompt "blurry, low quality, text, watermark" \
    --output qwen_image_21_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    "$@"
