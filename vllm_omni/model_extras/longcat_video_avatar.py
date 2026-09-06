# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# LongCat-Video-Avatar-1.5 is an audio(+image) -> video model. Its model-specific
# knobs are read from sampling_params.extra_args in
# pipeline_longcat_video_avatar.py; declaring them here routes request
# `extra_body` fields into OmniDiffusionSamplingParams.extra_args so the model can
# be driven through the standard task example.
LONGCAT_VIDEO_AVATAR_EXTRA_BODY_PARAMS = frozenset(
    {
        "audio_guidance_scale",
        "audio_path",
        "audio_type",
        "bbox",
        "image_path",
        "input_json",
        "mask_frame_range",
        "max_sequence_length",
        "negative_prompt",
        "num_cond_frames",
        "num_segments",
        "ref_img_index",
        "offload_kv_cache",
        "resize_mode",
        "resolution",
        "save_fps",
        "stage",
        "text_guidance_scale",
        "use_distill",
        "use_kv_cache",
    }
)
LONGCAT_VIDEO_AVATAR_EXTRA_OUTPUT_PARAMS = frozenset()
