# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    supports_mixed_reference_inputs: bool = False
    attention_mask_free: bool = False
    final_output_type: str | None = None


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
        final_output_type="video",
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
        final_output_type="video",
    ),
    "WanPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanImageToVideoPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanVACEPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanS2VPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanT2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "WanI2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2DistilledPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2T2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2I2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "HeliosPipeline": DiffusionModelMetadata(final_output_type="video"),
    "HeliosPyramidPipeline": DiffusionModelMetadata(final_output_type="video"),
    "HunyuanVideo15Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "HunyuanVideo15ImageToVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LingBotVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LongCatVideoAvatarPipeline": DiffusionModelMetadata(final_output_type="video"),
    "MagiHumanPipeline": DiffusionModelMetadata(final_output_type="video"),
    "DreamIDOmniPipeline": DiffusionModelMetadata(final_output_type="video"),
    "Cosmos3OmniDiffusersPipeline": DiffusionModelMetadata(final_output_type="video"),
    "Cosmos3OmniPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaImageToVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaWmPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
    ),
}

_DIFFUSION_MODEL_METADATA_ALIASES = {
    "WanDMDPipeline": "WanPipeline",
    "LTX2TwoStagePipeline": "LTX2Pipeline",
    "LTX2DistilledOneStagePipeline": "LTX2DistilledPipeline",
    "LTX2DistilledTwoStagePipeline": "LTX2DistilledPipeline",
    "LingBotWorldCausalDMDPipeline": "LingBotVideoPipeline",
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    metadata = _DIFFUSION_MODEL_METADATA.get(model_class_name)
    if metadata is not None:
        return metadata
    canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(model_class_name)
    if canonical_name is not None:
        return _DIFFUSION_MODEL_METADATA[canonical_name]
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
        canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(pipeline_cls_name, pipeline_cls_name)
        return _DIFFUSION_MODEL_METADATA.get(canonical_name, DiffusionModelMetadata())
    return DiffusionModelMetadata()
