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

import dataclasses
import os

import torch
from vllm.logger import init_logger

from vllm_omni.bde.kv_cache.config import BDEKVConfig
from vllm_omni.bde.kv_cache.gather import BDEKVState
from vllm_omni.bde.kv_cache.manager import BDEKVCache
from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

logger = init_logger(__name__)


def resolve_bde_kv_config(od_config: OmniDiffusionConfig) -> BDEKVConfig:
    """Resolve the BDE KV config (disabled unless explicitly enabled).

    Precedence: an ``OmniDiffusionConfig`` field (``bde_kv_config`` /
    ``kv_block_config``), then the ``BDE_KV_ENABLE=1`` env switch (with optional
    ``BDE_KV_WINDOW_CHUNKS``), else disabled. ``chunk_size`` / ``window_chunks``
    are finalized from the model geometry at load (see ``_preallocate_kv_cache``).
    """
    raw = getattr(od_config, "bde_kv_config", None)
    if raw is None:
        raw = getattr(od_config, "kv_block_config", None)
    if isinstance(raw, BDEKVConfig):
        return raw
    if isinstance(raw, dict):
        return BDEKVConfig(**raw)
    if os.environ.get("BDE_KV_ENABLE") == "1":
        window = os.environ.get("BDE_KV_WINDOW_CHUNKS")
        gpu_frac = os.environ.get("BDE_KV_GPU_FRACTION")
        return BDEKVConfig(
            enable=True,
            window_chunks=int(window) if window else None,
            gpu_memory_fraction=float(gpu_frac) if gpu_frac else 0.1,
        )
    return BDEKVConfig()


class BDEModelRunner(DiffusionModelRunner):
    def __init__(self, vllm_config, od_config: OmniDiffusionConfig, device) -> None:
        super().__init__(vllm_config, od_config, device)
        self.bde_kv_config = resolve_bde_kv_config(od_config)
        # Built after the model is loaded (dimensions known); stays None while KV
        # management is disabled.
        self.kv_cache: BDEKVCache | None = None
        # DreamZero KV is session-scoped (persists across forwards), so the BDE KV
        # state is held on the runner and reused, not created per request.
        self._bde_state: BDEKVState | None = None

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
        """Construct the BDEKVCache (preallocating GPU pools on ``self.device``).

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
            device=self.device,
        )
        logger.info(
            "BDE KV cache enabled: %d blocks, chunk_size=%d, window_chunks=%s, "
            "layers=%d kv_heads=%d head_dim=%d block_size=%d device=%s",
            self.kv_cache.num_blocks,
            self.bde_kv_config.chunk_size,
            self.bde_kv_config.window_chunks,
            num_layers,
            num_kv_heads,
            head_size,
            block_size,
            self.device,
        )

    # -- preallocation at load -------------------------------------------------

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        if self.bde_kv_config.enable and self.pipeline is not None:
            self._preallocate_kv_cache()

    def _infer_frame_seqlen(self) -> int:
        """frame_seqlen = (H//8)*(W//8)//4 from the configured image_resolution."""
        mc = getattr(self.od_config, "model_config", None) or {}
        psc = (mc.get("policy_server_config") if isinstance(mc, dict) else None) or {}
        res = psc.get("image_resolution", [180, 320])
        h, w = int(res[0]), int(res[1])
        return (h // 8) * (w // 8) // 4

    def _preallocate_kv_cache(self) -> None:
        """Build the KV pool once, from the loaded DreamZero transformer geometry."""
        t = self.pipeline.transformer
        num_layers = int(t.num_layers)
        num_kv_heads = int(getattr(t.blocks[0].self_attn, "tp_num_heads", t.num_heads))
        head_size = int(t.dim // t.num_heads)
        num_frame_per_block = int(t.num_frame_per_block)
        local_attn_size = int(t.local_attn_size)
        frame_seqlen = self._infer_frame_seqlen()
        # The model's own attention window, in tokens (= the [-max_attention_size:]
        # slice it applies). Read it directly rather than recomputing.
        max_attention_size = int(t.blocks[0].self_attn.max_attention_size)

        # Frame-granular paging: 1 block = 1 frame = frame_seqlen tokens, so the
        # resident window matches max_attention_size exactly (it need not be a whole
        # number of num_frame_per_block causal blocks).
        chunk_size = frame_seqlen
        window_chunks = self.bde_kv_config.window_chunks or (max_attention_size // frame_seqlen)

        self.bde_kv_config = dataclasses.replace(
            self.bde_kv_config, chunk_size=chunk_size, window_chunks=window_chunks
        )
        free_bytes = torch.cuda.mem_get_info(self.device)[0]
        logger.info(
            "BDE preallocating (paged): frame_seqlen=%d num_frame_per_block=%d "
            "local_attn_size=%d -> chunk_size=%d window_chunks=%d (window=%d tokens)",
            frame_seqlen, num_frame_per_block, local_attn_size,
            chunk_size, window_chunks, window_chunks * chunk_size,
        )
        self.build_kv_cache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=self.od_config.dtype,
            block_size=chunk_size,  # one pool block per chunk
            max_model_len=1 << 20,
            available_bytes=free_bytes,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # KV disabled -> base behavior, unchanged.
        if self.kv_cache is None:
            return super().execute_model(req)

        kv = self.kv_cache
        # Reuse the session-scoped BDE KV state so KV persists across the session's
        # forwards (Approach 1: the state holds the model's window per branch/layer).
        if self._bde_state is None:
            pos = kv.begin_request("bde")
            neg = kv.begin_request("bde__neg")
            self._bde_state = BDEKVState(kv, pos, neg, num_layers=kv.num_layers)
        logger.debug(
            "BDE execute_model: req=%s chunk_size=%d window_chunks=%s num_blocks=%d",
            req.request_id, kv.spec.chunk_size, kv.config.window_chunks, kv.num_blocks,
        )
        self.pipeline._bde_kv_state = self._bde_state
        try:
            return super().execute_model(req)
        finally:
            self.pipeline._bde_kv_state = None
