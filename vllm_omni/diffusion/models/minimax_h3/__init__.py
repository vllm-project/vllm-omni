# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 audio-video diffusion support."""

from typing import TYPE_CHECKING

from vllm_omni.model_executor.models.minimax_h3 import reference_video

if TYPE_CHECKING:
    from .pipeline_minimax_h3 import MiniMaxH3Pipeline, get_minimax_h3_post_process_func

__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
    "reference_video",
]


def __getattr__(name: str):
    # Checkpoint detection also imports this package without loading the pipeline.
    if name in ("MiniMaxH3Pipeline", "get_minimax_h3_post_process_func"):
        from . import pipeline_minimax_h3

        return getattr(pipeline_minimax_h3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
