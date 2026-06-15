# SPDX-License-Identifier: Apache-2.0
"""The Block Diffusion Engine (BDE)."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

logger = init_logger(__name__)


class BDEEngine(DiffusionEngine):
    """AR-Diffusion engine with engine-level KV cache management.

    BDE serves autoregressive / chunked blockwise-causal diffusion models
    (world models, AR-DiT) that materialize persistent attention KV. It reuses
    vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` / ``BlockTables``)
    as a library, driven from the engine rather than hand-rolled inside each
    model. See ``BDE_doc/diffusion_kv_cache_management.md`` for the design and
    ``BDE_doc/dreamzero_kv_phase1_plan.md`` for the rollout.

    It is selected per model via ``OmniDiffusionConfig.engine_backend = "bde"``
    (resolved by :meth:`DiffusionEngine.make_engine`), so models that do not opt
    in keep using the base ``DiffusionEngine`` unchanged.

    Phase 1 is a behavior-preserving subclass: it establishes the engine seam and
    isolation boundary. The KV-cache lifecycle (pool construction, per-chunk
    ``allocate_slots`` / ``free``) is added in later phases.
    """
