# SPDX-License-Identifier: Apache-2.0
"""AR-Diffusion Engine (AR-Diffusion).

The AR-Diffusion engine: a ``DiffusionEngine`` subclass that adds engine-level KV
cache management for autoregressive / chunked "world-model" diffusion models
(DreamZero and the AR-DiT family). Selected via
``OmniDiffusionConfig.engine_backend = "ar_diffusion"``.
"""

from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine

__all__ = ["ARDiffusionEngine"]
