# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility re-export for MiniCPM-o CUDA graph wrappers.

Canonical implementation has moved to ``vllm_omni.model_executor.models.common.audio_graph_wrapper``.
"""

from vllm_omni.model_executor.models.common.audio_graph_wrapper import (
    CFMGraphWrapper,
    HiFTGraphWrapper,
    _format_memory_delta,
    _memory_snapshot,
)

__all__ = ["CFMGraphWrapper", "HiFTGraphWrapper", "_format_memory_delta", "_memory_snapshot"]
