# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared fused operator surface for diffusion models."""

from .gated_residual import gated_residual

__all__ = ["gated_residual"]
