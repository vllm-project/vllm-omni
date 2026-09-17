# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 model-specific optimized operators."""

from .vae import (
    H3VAEExactOpStatsSnapshot,
    install_h3_vae_optimizations,
    snapshot_h3_vae_exact_op_stats,
)

__all__ = [
    "H3VAEExactOpStatsSnapshot",
    "install_h3_vae_optimizations",
    "snapshot_h3_vae_exact_op_stats",
]
