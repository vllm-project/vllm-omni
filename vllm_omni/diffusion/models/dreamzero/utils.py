# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero model constants shared by the pipeline."""

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
