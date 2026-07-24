# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

DREAMX_WORLD_EXTRA_BODY_PARAMS = frozenset(
    {
        "action_seq",
        "action_speed_list",
    }
)

DREAMX_WORLD_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()
