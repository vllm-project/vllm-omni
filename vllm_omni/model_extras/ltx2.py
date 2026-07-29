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
    }
)

LTX_EXTRA_OUTPUT_PARAMS = frozenset()
