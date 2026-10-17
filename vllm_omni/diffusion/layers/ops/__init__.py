# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared tensor operations for diffusion layers."""

from vllm_omni.diffusion.layers.ops.modulation import gated_residual

__all__ = ["gated_residual"]
