# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

LINGBOT_VIDEO_EXTRA_BODY_PARAMS = frozenset(
    {
        "base_low_noise_threshold",
        "base_sigma_tail_steps",
        "batch_cfg",
        "duration",
        "flow_shift",
        "negative_prompt",
        "null_cond_clone_zero",
        "offload_vae_during_denoise",
        "output_type",
        "refiner_batch_cfg",
        "refiner_guidance_scale",
        "refiner_height",
        "refiner_max_video_frames",
        "refiner_null_cond_clone_zero",
        "refiner_output_fps",
        "refiner_sample_fps",
        "refiner_shift",
        "refiner_sigma_tail_steps",
        "refiner_steps",
        "refiner_t_thresh",
        "refiner_width",
        "resolution",
        "ratio",
        "run_refiner",
        "shift",
        "t_thresh",
    }
)
