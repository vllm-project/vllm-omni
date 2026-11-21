# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-duplex integration for Google Magenta RealTime 2."""

from vllm_omni.model_executor.models.magenta_rt2.duplex.capabilities import (
    CHANNELS,
    CHUNK_PERIOD_MS,
    FRAME_RATE,
    FRAME_SIZE,
    SAMPLE_RATE,
    magenta_rt2_capabilities,
)
from vllm_omni.model_executor.models.magenta_rt2.duplex.data_plane import MagentaRT2DataPlaneSession
from vllm_omni.model_executor.models.magenta_rt2.duplex.plugin import MagentaRT2DuplexPlugin

__all__ = [
    "CHANNELS",
    "CHUNK_PERIOD_MS",
    "FRAME_RATE",
    "FRAME_SIZE",
    "SAMPLE_RATE",
    "MagentaRT2DataPlaneSession",
    "MagentaRT2DuplexPlugin",
    "magenta_rt2_capabilities",
]
