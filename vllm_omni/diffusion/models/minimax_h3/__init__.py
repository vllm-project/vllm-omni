# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 audio-video diffusion support."""

__all__ = ["MiniMaxH3Pipeline", "get_minimax_h3_post_process_func", "reference_video"]


def __getattr__(name: str):
    if name == "reference_video":
        from vllm_omni.model_executor.models.minimax_h3 import reference_video

        return reference_video
    if name in ("MiniMaxH3Pipeline", "get_minimax_h3_post_process_func"):
        from . import pipeline_minimax_h3

        return getattr(pipeline_minimax_h3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
