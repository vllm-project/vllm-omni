# SPDX-License-Identifier: Apache-2.0
"""AR-Diffusion Engine (AR-Diffusion).

The AR-Diffusion engine: a ``DiffusionEngine`` subclass that adds engine-level KV
cache management for autoregressive / chunked diffusion models. Selected via
``OmniDiffusionConfig.engine_backend = "ar_diffusion"``.
"""

from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    SupportsARDiffusionPipeline,
    SupportsARDiffusionWarmup,
)
from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine

__all__ = [
    "ARDiffusionKVBranchSpec",
    "ARDiffusionCrossAttentionKVSpec",
    "ARDiffusionEngine",
    "ARDiffusionKVCacheSpec",
    "SupportsARDiffusionPipeline",
    "SupportsARDiffusionWarmup",
]
