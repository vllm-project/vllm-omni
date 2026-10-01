# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 model-specific optimized operators."""

from .vae import install_h3_vae_optimizations

__all__ = ["install_h3_vae_optimizations"]
