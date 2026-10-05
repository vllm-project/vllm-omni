# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request parameters shared by official LTX guidance pipelines."""

LTX_EXTRA_BODY_PARAMS = frozenset(
    {
        "video_cfg_scale",
        "audio_cfg_scale",
        "video_cfg_guidance_scale",
        "audio_cfg_guidance_scale",
        "video_stg_scale",
        "audio_stg_scale",
        "video_stg_guidance_scale",
        "audio_stg_guidance_scale",
        "video_modality_scale",
        "audio_modality_scale",
        "a2v_guidance_scale",
        "v2a_guidance_scale",
        "video_rescale_scale",
        "audio_rescale_scale",
        "video_stg_blocks",
        "audio_stg_blocks",
        "sigmas",
        "stage_1_sigmas",
        "stage_2_sigmas",
        "image_crf",
    }
)

LTX_EXTRA_OUTPUT_PARAMS = frozenset()


def ltx_preserves_reference_image_size(*, model: str | None, revision: str | None = None) -> bool:
    """Keep source geometry until LTX-2.5 applies conditioning compression."""
    if model is None:
        return False
    from vllm_omni.diffusion.models.ltx2.ltx2_components import (
        preserves_reference_image_size as ltx_preserves_reference_image_size,
    )

    return ltx_preserves_reference_image_size(model=model, revision=revision)


def ltx_transformer_config_subfolder(
    *,
    model: str | None,
    revision: str | None = None,
) -> str:
    """Select the transformer config used by the requested LTX checkpoint."""
    if model is None:
        return "transformer"
    from vllm_omni.diffusion.models.ltx2.ltx2_components import detect_ltx_model_version

    return "transformer_full" if detect_ltx_model_version(model, revision=revision) == "2.5" else "transformer"
