# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared diffusion tensor operations.

Importing an operation registers its torch schema in its canonical module.
Model parameters and execution policy remain with the calling layers/models.
"""

from .rope.qk_norm_rope import fused_qk_norm_rope

__all__ = ["fused_qk_norm_rope"]
