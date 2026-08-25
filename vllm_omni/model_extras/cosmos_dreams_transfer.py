# SPDX-License-Identifier: Apache-2.0
"""Serving parameters exposed by Cosmos-Dreams-Transfer."""

COSMOS_DREAMS_TRANSFER_EXTRA_BODY_PARAMS = frozenset(
    {
        "blur",
        "close_session",
        "control_guidance",
        "control_hint",
        "control_video",
        "depth",
        "edge",
        "emphasize_control_in_prompt",
        "num_first_chunk_conditional_frames",
        "reset",
        "seg",
        "session_id",
        "share_vision_temporal_positions",
    }
)

COSMOS_DREAMS_TRANSFER_EXTRA_OUTPUT_PARAMS = frozenset()
