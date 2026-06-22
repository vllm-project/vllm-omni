# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# Helios is a text/image/video -> video model. Its model-specific knobs are read
# from sampling_params.extra_args in pipeline_helios.py; declaring them here
# routes request `extra_body` fields into OmniDiffusionSamplingParams.extra_args
# so the model can be driven through the standard task example.
HELIOS_EXTRA_BODY_PARAMS = frozenset(
    {
        # pyramid / staged sampling
        "is_enable_stage2",
        "pyramid_num_stages",
        "pyramid_num_inference_steps_list",
        "is_amplify_first_chunk",
        "is_skip_first_chunk",
        # CFG-zero* controls
        "use_cfg_zero_star",
        "use_zero_init",
        "zero_steps",
        # conditioning inputs
        "image",
        "video",
        "add_noise_to_image_latents",
        "image_noise_sigma_min",
        "image_noise_sigma_max",
        "add_noise_to_video_latents",
        "video_noise_sigma_min",
        "video_noise_sigma_max",
    }
)
HELIOS_EXTRA_OUTPUT_PARAMS = frozenset()
