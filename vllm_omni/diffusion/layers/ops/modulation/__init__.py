# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared modulation operations for diffusion layers."""

from vllm_omni.diffusion.layers.ops.modulation.gated_residual import gated_residual

__all__ = ["gated_residual"]
