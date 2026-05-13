# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    # max_multimodal_image_inputs:
    #   None = unknown, no restriction
    #   1   = single image only
    #   N>1 = multi-image, max N images
    max_multimodal_image_inputs: int | None = None


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
    ),
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to no restriction so new pipelines are not
    # accidentally blocked from multi-image input.
    if model_class_name is None:
        return DiffusionModelMetadata()
    return _DIFFUSION_MODEL_METADATA.get(model_class_name, DiffusionModelMetadata())
