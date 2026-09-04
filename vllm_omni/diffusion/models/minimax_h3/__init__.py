# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 audio-video diffusion support."""

from vllm_omni.model_executor.models.minimax_h3 import reference_video

from .pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    get_minimax_h3_post_process_func,
)

__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
    "reference_video",
]
