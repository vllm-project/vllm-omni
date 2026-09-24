# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Numerical primitives shared by the exact LTX-2 Triton kernels.

The implementations live in ``vllm_omni.diffusion.layers.numerics`` because the
Qwen-Image 2.1 kernels need the same pinned-rounding helpers.
"""

from vllm_omni.diffusion.layers.numerics import (
    add_rn_f32,
    fma_rn_f32,
    mul_rn_f32,
    round_bf16_to_fp32,
    rsqrt_approx_f32,
    shfl_down_f32,
)

__all__ = [
    "add_rn_f32",
    "fma_rn_f32",
    "mul_rn_f32",
    "round_bf16_to_fp32",
    "rsqrt_approx_f32",
    "shfl_down_f32",
]
