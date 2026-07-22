# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

LINGBOT_VIDEO_EXTRA_BODY_PARAMS = frozenset(
    {
        "batch_cfg",
        "flow_shift",
        "negative_prompt",
        "null_cond_clone_zero",
        "offload_vae_during_denoise",
        "output_type",
        "refiner_sigma_tail_steps",
        "shift",
        "t_thresh",
    }
)
