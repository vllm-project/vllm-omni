# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

LINGBOT_VIDEO_EXTRA_BODY_PARAMS = frozenset(
    {
        "batch_cfg",
        "duration",
        "flow_shift",
        "negative_prompt",
        "null_cond_clone_zero",
        "offload_vae_during_denoise",
        "output_type",
        "refiner_sigma_tail_steps",
        "resolution",
        "ratio",
        "shift",
        "t_thresh",
    }
)
