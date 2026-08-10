# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# SkyReels V3 V2V consumes video conditioning through extra_args because the
# OpenAI diffusion chat path stores reference videos as `video_path` rather than
# decoding them into `multi_modal_data`.
SKYREELS_V3_V2V_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {
        "video_path",
        "input_video",
        "duration",
        "fps",
        "condition_frames",
        "sampling_steps",
        "cfg_text_scale",
        "shift",
        "block_offload",
        "include_input_video",
    }
)

SKYREELS_V3_V2V_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()
