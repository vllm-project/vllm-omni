# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

SENSENOVA_U1_EXTRA_BODY_PARAMS = frozenset(
    {
        "think",
        "cfg_scale",
        "cfg_norm",
        "timestep_shift",
        "t_eps",
        "img_cfg_scale",
        "max_tokens",
    }
)
SENSENOVA_U1_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "think_text",
    }
)
