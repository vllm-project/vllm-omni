# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA diffusion MXFP8 using vLLM's existing online weight and kernel path."""

from vllm.platforms import current_platform

from vllm_omni.quantization.mxfp8_config import VllmMxfp8OnlineLinearMethod


class CUDAMxfp8OnlineLinearMethod(VllmMxfp8OnlineLinearMethod):
    """Enable native block-32 MXFP8 on Blackwell and Rubin.

    vLLM owns lazy checkpoint loading, quantization, scale layout and kernel
    selection. The shared Omni adapter flattens diffusion activations before
    dispatch and restores their leading dimensions afterwards.
    """

    def __init__(self) -> None:
        capability = current_platform.get_device_capability()
        if not current_platform.is_cuda() or capability is None or capability.to_int() not in (100, 103, 107):
            raise NotImplementedError("CUDA diffusion MXFP8 requires SM100, SM103 or SM107")
        super().__init__()
