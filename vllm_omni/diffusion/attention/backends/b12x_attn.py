# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""B12X diffusion attention backend (SM120/SM121).

Runs packed attention through ``b12x.attention.varlen``, the CuTe DSL kernel shipped by
``local-inference-lab/b12x``. b12x is the SM120/SM121 CuTe DSL kernel library
and already provides the attention, MoE and linear backends the LLM profiles select;
this backend makes MiniMax-H3's DiT attention consistent with that choice instead of pulling in
a third-party kernel.

Two modes:

* dense -- the packed sequence, bidirectional, no KV cache (the H3 base model);
* block-list sparse -- an authoritative CSR list of 64-token K blocks per q tile, as produced
  by a learned compressor/gate (FastH3 VSA-style). Supplied per forward through
  ``attn_metadata.extra``::

      extra = {"block_indices": int32[T], "block_offsets": int32[num_q_tiles + 1]}

  The list is authoritative: no causal/window constraint is re-applied, and dense regions
  (text/audio) are expressed as contiguous runs in the list. Absent keys mean dense.

Only one GPU (no ring) is handled here; the packed cu_seqlens describe the whole sequence.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:  # b12x is optional: only SM120/SM121 hosts carry it.
    from b12x.attention import varlen as _b12x_varlen

    _B12X_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - import-time environment probe
    _b12x_varlen = None
    _B12X_IMPORT_ERROR = str(_exc)

# Capacity buckets keep the prepared program static: b12x plans are capacity-bounded and the
# live row count is a launch scalar, so a handful of buckets serves every request length.
_CAPACITY_STEP = 4096
_PLAN_CACHE: dict[tuple, Any] = {}
_SCRATCH_CACHE: dict[tuple, torch.Tensor] = {}


def _capacity(n: int) -> int:
    return max(_CAPACITY_STEP, ((int(n) + _CAPACITY_STEP - 1) // _CAPACITY_STEP) * _CAPACITY_STEP)


class B12xAttentionBackend(AttentionBackend):
    """SM120/SM121 attention through local-inference-lab/b12x."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 192, 256]

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # varlen takes the caller's packed cu_seqlens, so a padded packed sequence needs no
        # boolean attn_mask.
        return True

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[B12xAttentionImpl]:
        return B12xAttentionImpl

    @classmethod
    def validate_available(cls) -> None:
        if _b12x_varlen is None:
            raise ImportError(
                "B12X attention requires the b12x package on an SM120/SM121 device. "
                f"Import failed with: {_B12X_IMPORT_ERROR}"
            )
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
            raise RuntimeError("B12X attention requires a consumer-Blackwell (SM120/SM121) GPU")


class B12xAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("B12X diffusion attention is bidirectional; causal is not supported")
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError("B12X attention does not shard KV heads; GQA is unsupported here")
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("B12xAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def _plan_for(
        self,
        rows: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        sparse: bool,
        num_tiles: int,
        total_blocks: int,
    ):
        key = (rows, heads, head_dim, dtype, device.index, sparse, num_tiles, total_blocks)
        plan = _PLAN_CACHE.get(key)
        if plan is None:
            q = torch.empty((rows, heads, head_dim), dtype=dtype, device=device)
            cu = torch.zeros(2, dtype=torch.int32, device=device)
            plan = _b12x_varlen.plan(
                q,
                q,
                q,
                cu,
                cu,
                max_seqlen_q=rows,
                max_seqlen_k=rows,
                causal=False,
                block_sparse=sparse,
                num_q_tiles=(num_tiles if sparse else 0),
                total_blocks_cap=(max(1, total_blocks) if sparse else 0),
            )
            _PLAN_CACHE[key] = plan
        return key, plan

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if _b12x_varlen is None:
            self.__class__  # noqa: B018 - keep the failure message at the call site
            raise ImportError(
                "B12X attention requires the b12x package on an SM120/SM121 device. "
                f"Import failed with: {_B12X_IMPORT_ERROR}"
            )

        q3 = query.flatten(0, 1).contiguous()
        k3 = key.flatten(0, 1).contiguous()
        v3 = value.flatten(0, 1).contiguous()
        rows, heads, head_dim = q3.shape

        packed = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        if packed is not None:
            live = int(packed.q_length)
            cu_q = packed.cu_seqlens_q
            cu_k = packed.cu_seqlens_k
        else:
            live = rows
            cu_q = torch.tensor([0, live], dtype=torch.int32, device=q3.device)
            cu_k = cu_q

        extra = getattr(attn_metadata, "extra", None) or {}
        block_indices = extra.get("block_indices")
        block_offsets = extra.get("block_offsets")
        sparse = block_indices is not None and block_offsets is not None
        num_tiles = int(block_offsets.numel() - 1) if sparse else 0
        total_blocks = int(block_offsets[-1].item()) if sparse else 0

        plane = _capacity(live)
        cache_key, plan = self._plan_for(
            plane, heads, head_dim, q3.dtype, q3.device, sparse, num_tiles, max(1, total_blocks)
        )
        scratch = _SCRATCH_CACHE.get(cache_key)
        if scratch is None:
            (spec,) = _require_prepared(plan, q3.device).scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q3.device)
            _SCRATCH_CACHE[cache_key] = scratch

        state = _require_prepared(plan, q3.device)
        binding = _b12x_varlen.bind(
            plan,
            scratch=scratch,
            q=q3[:live],
            k=k3[:live],
            v=v3[:live],
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=live,
            max_seqlen_k=live,
            block_indices=block_indices,
            block_offsets=block_offsets,
        )
        result = state.run(binding)
        out = result[0] if isinstance(result, tuple) else result
        torch.accelerator.synchronize()
        if out.shape[0] != q3.shape[0]:
            full = q3.new_zeros((q3.shape[0],) + tuple(out.shape[1:]))
            full[: out.shape[0]] = out
            out = full
        return out.reshape_as(query)


def _require_prepared(plan, device):
    from b12x.preparation import require_prepared

    return require_prepared(plan, "attention.varlen", device)
