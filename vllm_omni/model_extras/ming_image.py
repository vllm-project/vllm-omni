# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.model_extras.ming_flash_omni import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
)

MING_IMAGE_EXTRA_BODY_PARAMS = frozenset(
    {
        "height",
        "width",
        "seed",
        "num_layers",
        "negative_prompt",
    }
)
MING_IMAGE_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()
MING_IMAGE_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES = True

__all__ = [
    "MING_IMAGE_EXTRA_BODY_PARAMS",
    "MING_IMAGE_EXTRA_OUTPUT_PARAMS",
    "MING_IMAGE_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES",
    "build_image_to_image_prompt",
    "build_text_to_image_prompt",
]
