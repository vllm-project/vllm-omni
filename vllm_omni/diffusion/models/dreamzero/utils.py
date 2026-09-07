# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero model constants shared by the pipeline."""

# --- Disaggregation wire contract -------------------------------------------
# The single transport key DreamZero uses on every disaggregated stage edge.
# Both edges (encode -> denoise, denoise -> decode) carry one entry under this
# key; the payload's own ``boundary`` field distinguishes the two. Keeping one
# transport key means a deploy config wires connectors per edge without the
# model having to know which transport is in use.
DREAMZERO_STAGE_PAYLOAD_KEY = "dreamzero_stage_payload"

# Payload boundaries, i.e. which stage edge a payload was produced for.
DREAMZERO_BOUNDARY_ENCODE_TO_DIT = "encode_to_dit"
DREAMZERO_BOUNDARY_DIT_TO_DECODE = "dit_to_decode"

# Wire-format version. Bump on any incompatible payload schema change; the
# consuming stage rejects a payload it cannot interpret instead of silently
# reading stale field names.
DREAMZERO_PAYLOAD_VERSION = 1

DEFAULT_NUM_INFERENCE_STEPS = 16
DEFAULT_CFG_SCALE = 5.0
DEFAULT_SIGMA_SHIFT = 5.0
DEFAULT_SEED = 1140

DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
    "style, artwork, painting, image, still, grayscale, dull, worst quality, "
    "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
    "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
    "image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)

DEFAULT_EMBODIMENT_NAME_TO_ID = {
    "oxe_droid": 17,
    "agibot": 26,
    "gr1_unified": 24,
    "xdof": 22,
    "yam": 32,
    "mecka_hands": 27,
    "lapa": 27,
    "dream": 31,
}
