# SPDX-License-Identifier: Apache-2.0
"""Block Diffusion Engine (BDE).

The AR-Diffusion engine: a ``DiffusionEngine`` subclass that adds engine-level KV
cache management for autoregressive / chunked "world-model" diffusion models
(DreamZero and the AR-DiT family). Selected via
``OmniDiffusionConfig.engine_backend = "bde"``.
"""

from vllm_omni.bde.engine import BDEEngine

__all__ = ["BDEEngine"]
