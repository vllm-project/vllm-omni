# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility imports for streaming video decode."""

from vllm_omni.diffusion.distributed.autoencoders.streaming_decode import (
    StreamingDecodeState,
    SupportsStreamingDecode,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_streaming_decode import WanStreamingDecoder

__all__ = ["StreamingDecodeState", "SupportsStreamingDecode", "WanStreamingDecoder"]
