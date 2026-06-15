# SPDX-License-Identifier: Apache-2.0
"""BDEModelRunner — the diffusion model runner for the BDE engine.

Subclasses ``DiffusionModelRunner`` and owns the engine-level KV cache
(:class:`BDEKVCache`). It brackets a request's rollout with the KV lifecycle
(``begin_request`` / ``end_request``) and exposes the live ``BDEKVCache`` so the
model pipeline (DreamZero) can drive the per-chunk allocate / slot-mapping /
gather / commit operations during ``pipeline.forward``.

When KV management is disabled (the Phase-1 default), the runner is
behavior-preserving — it simply defers to the base ``DiffusionModelRunner``.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.bde.kv_cache.config import BDEKVConfig
from vllm_omni.bde.kv_cache.gather import BDEKVState
from vllm_omni.bde.kv_cache.manager import BDEKVCache
from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

logger = init_logger(__name__)


def resolve_bde_kv_config(od_config: OmniDiffusionConfig) -> BDEKVConfig:
    """Read the BDE KV config off the diffusion config (disabled if absent).

    The field name is not yet fixed on ``OmniDiffusionConfig`` (it must avoid the
    existing ``diffusion_kv_cache_*`` collision), so we look it up leniently and
    default to a disabled config — keeping the runner behavior-preserving until
    the field is wired.
    """
    raw = getattr(od_config, "bde_kv_config", None)
    if raw is None:
        raw = getattr(od_config, "kv_block_config", None)
    if isinstance(raw, BDEKVConfig):
        return raw
    if isinstance(raw, dict):
        return BDEKVConfig(**raw)
    return BDEKVConfig()


class BDEModelRunner(DiffusionModelRunner):
    def __init__(self, vllm_config, od_config: OmniDiffusionConfig, device) -> None:
        super().__init__(vllm_config, od_config, device)
        self.bde_kv_config = resolve_bde_kv_config(od_config)
        # Built after the model is loaded (dimensions known); stays None while KV
        # management is disabled.
        self.kv_cache: BDEKVCache | None = None

    def build_kv_cache(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype,
        block_size: int,
        max_model_len: int,
        available_bytes: int,
    ) -> None:
        """Construct the BDEKVCache once the model's dimensions are known.

        No-op when KV management is disabled.
        """
        if not self.bde_kv_config.enable:
            return
        self.kv_cache = BDEKVCache(
            self.bde_kv_config,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            block_size=block_size,
            max_model_len=max_model_len,
            available_bytes=available_bytes,
        )
        logger.info(
            "BDE KV cache enabled: %d blocks, chunk_size=%d, window_chunks=%s",
            self.kv_cache.num_blocks,
            self.bde_kv_config.chunk_size,
            self.bde_kv_config.window_chunks,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # KV disabled -> base behavior, unchanged.
        if self.kv_cache is None:
            return super().execute_model(req)

        kv = self.kv_cache
        pos = kv.begin_request(req.request_id)
        neg = kv.begin_request(req.request_id + "__neg")
        # Allocate the prefill chunk so the first gather has blocks to read.
        kv.allocate_chunk(pos)
        kv.allocate_chunk(neg)
        kv_state = BDEKVState(
            kv, pos, neg,
            num_layers=kv.num_layers,
        )
        self.pipeline._bde_kv_state = kv_state
        try:
            return super().execute_model(req)
        finally:
            self.pipeline._bde_kv_state = None
            kv.end_request(pos)
            kv.end_request(neg)
