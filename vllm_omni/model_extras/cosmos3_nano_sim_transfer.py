# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving parameters exposed by Cosmos3-Nano-Sim-Transfer.

Every key read by Transfer geometry or admission is declared so online requests
reach the same contract as offline ``extra_args``.
"""

COSMOS3_NANO_SIM_TRANSFER_EXTRA_BODY_PARAMS = frozenset(
    {
        "aspect_ratio",
        "attention_sink_size",
        "blur",
        "close_session",
        "control_guidance",
        "control_hint",
        "control_video",
        "depth",
        "edge",
        "emphasize_control_in_prompt",
        "kv_cache_inference_size",
        "num_first_chunk_conditional_frames",
        "reset",
        "resolution",
        "seg",
        "session_id",
        "share_vision_temporal_positions",
    }
)

COSMOS3_NANO_SIM_TRANSFER_EXTRA_OUTPUT_PARAMS = frozenset()
