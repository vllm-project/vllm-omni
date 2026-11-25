# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

# Request knobs read from ``extra_args`` by ``HiUniPipeline``.
HIUNI_EXTRA_BODY_PARAMS = frozenset(
    {
        "cfg_scale",
        "img_cfg_scale",
        "shift",
        "resize_to_edit_image",
        "min_pixels",
        "max_pixels",
        "think_mode",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "return_raw",
    }
)
