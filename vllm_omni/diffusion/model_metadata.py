# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    supports_mixed_reference_inputs: bool = False
    attention_mask_free: bool = False


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4
# Upstream HunyuanImage-3.0 "Multi-Image Fusion" caps reference images at 3.
HUNYUAN_IMAGE3_MAX_INPUT_IMAGES = 3
# Boogu-Image editing (TI2I) supports a single reference image for now.
BOOGU_IMAGE_MAX_INPUT_IMAGES = 1


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
    ),
    "HunyuanImage3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
    ),
    # Shared by the Base (text-to-image) and Edit (TI2I) checkpoints, which use
    # the same ``BooguImagePipeline`` class. Text-to-image requests simply carry
    # no reference image.
    "BooguImagePipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=BOOGU_IMAGE_MAX_INPUT_IMAGES,
    ),
    "MiniMaxH3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
        # H3 represents alignment padding as a second packed sequence.  The
        # packed TRTLLM backend consumes cu_seqlens and isolates that padding.
        attention_mask_free=True,
    ),
    # The modular alias is served by MiniMaxH3Pipeline and has the same
    # Ref2VA request contract. Keep admission limits in sync with it.
    "MiniMaxH3ModularPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
    ),
    "WanPipeline": DiffusionModelMetadata(attention_mask_free=True),
    "WanImageToVideoPipeline": DiffusionModelMetadata(attention_mask_free=True),
    "WanVACEPipeline": DiffusionModelMetadata(attention_mask_free=True),
    "WanS2VPipeline": DiffusionModelMetadata(attention_mask_free=True),
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    metadata = _DIFFUSION_MODEL_METADATA.get(model_class_name)
    if metadata is not None:
        return metadata
    # Some checkpoints report the HF architecture name diff from internal pipeline class name
    # (e.g. HunyuanImage3ForCausalMM, WanVACEPipeline, OmniVoice ...).
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

    entry = _DIFFUSION_MODELS.get(model_class_name)
    if entry is not None:
        # Unpack instead of indexing so a future change to the registry tuple
        # shape fails loudly instead of silently reading the wrong element.
        _, _, pipeline_cls_name = entry
        # Note: the registry ``cls_name`` and the metadata keys are two separate
        # key spaces. Aliases whose pipeline class has no metadata entry (e.g.
        # Wan22VACEPipeline, OmniVoicePipeline) still fall back to the defaults;
        # that is not a regression, it just means no capability override.
        return _DIFFUSION_MODEL_METADATA.get(pipeline_cls_name, DiffusionModelMetadata())
    return DiffusionModelMetadata()
