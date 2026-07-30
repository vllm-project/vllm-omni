#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Reproducible end-to-end run for HunyuanImage-3.0-Instruct through the migrated
# (model_extras + shared task example) pathway. Covers all four modalities.
#
# Requirements: an 80GB-class multi-GPU box. On the default CUDA layout both the
# AR+DiT deploy (AR 2 + DiT 2) and the AR-only deploy use 4 GPUs. Adjust the
# deploy YAML / device count to your hardware.
set -euo pipefail

# ---- Environment ------------------------------------------------------------
# NOTE: Omni() has no `revision` passthrough today, so this always resolves
# to whatever HEAD currently is on the model repo -- there is no way to pin
# a verified snapshot from this script alone. prompt_utils.py's hardcoded
# HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS assumes a specific tokenizer.json; if the
# repo's tokenizer ever changes, validate_special_token_ids() (wired into
# the shared scripts via get_ar_tokenizer_validator) fails loudly instead of
# silently mis-tokenizing. Revision pinning would need `revision` support
# added to Omni's engine args first -- tracked as a follow-up, not in scope
# for this migration.
export MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# Match the AR+DiT deploy's expected device count (4 on the default CUDA layout).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUT_DIR="${OUT_DIR:-./hunyuan_image3_repro}"
mkdir -p "${OUT_DIR}"

AR_DIT_DEPLOY="vllm_omni/deploy/hunyuan_image_3_moe.yaml"
AR_DEPLOY="vllm_omni/deploy/hunyuan_image3_ar.yaml"

# ---- 1) Text-to-image (shared text_to_image.py) -----------------------------
python examples/offline_inference/text_to_image/text_to_image.py \
  --model "${MODEL}" \
  --deploy-config "${AR_DIT_DEPLOY}" \
  --trust-remote-code \
  --prompt "A brown and white dog is running on the grass" \
  --height 1024 --width 1024 \
  --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
  --extra-body '{"bot_task": "think", "use_system_prompt": "en_recaption"}' \
  --output "${OUT_DIR}/t2i.png"

# ---- 2) Image editing / IT2I (shared image_edit.py) -------------------------
# Provide a real reference image path via EDIT_IMAGE=...
if [[ -n "${EDIT_IMAGE:-}" ]]; then
  python examples/offline_inference/image_to_image/image_edit.py \
    --model "${MODEL}" \
    --deploy-config "${AR_DIT_DEPLOY}" \
    --trust-remote-code \
    --image "${EDIT_IMAGE}" \
    --prompt "Make the flowers neon pink" \
    --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
    --extra-args '{"bot_task": "think"}' \
    --output "${OUT_DIR}/it2i.png"
else
  echo "[skip] image-editing: set EDIT_IMAGE=/path/to/image.png to run it2i"
fi

# ---- 3) Image-to-text (this directory) --------------------------------------
if [[ -n "${UNDERSTAND_IMAGE:-}" ]]; then
  python examples/offline_inference/hunyuan_image3/run_hunyuan_image3_understanding.py \
    --model "${MODEL}" \
    --deploy-config "${AR_DEPLOY}" \
    --modality image2text \
    --image "${UNDERSTAND_IMAGE}" \
    --prompt "Describe the content of the picture." \
    | tee "${OUT_DIR}/i2t.txt"
else
  echo "[skip] image-to-text: set UNDERSTAND_IMAGE=/path/to/image.jpg to run i2t"
fi

# ---- 4) Text-to-text (this directory) ---------------------------------------
python examples/offline_inference/hunyuan_image3/run_hunyuan_image3_understanding.py \
  --model "${MODEL}" \
  --deploy-config "${AR_DEPLOY}" \
  --modality text2text \
  --prompt "What is the capital of France?" \
  | tee "${OUT_DIR}/t2t.txt"

echo "Done. Outputs under ${OUT_DIR}"
