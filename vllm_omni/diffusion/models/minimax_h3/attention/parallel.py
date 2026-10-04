# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 gate ownership and producer scheduling over the shared Ulysses transport."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.forward_context import get_ulysses_mode

from . import overlap, qkv_overlap
from .overlap import H3_VSA_O_BUNDLE_ACTIVE_KEY, H3_VSA_O_BUNDLE_STATE_KEY, h3_vsa_o_bundle_enabled
from .vsa import _get_h3_layout, get_h3_vsa_owner_route_plan


@dataclass(frozen=True)
class H3OverlapContext(ParallelAttentionContext):
    ulysses_pg: dist.ProcessGroup
    o_bundle_gate_local: torch.Tensor
    o_bundle_metadata: AttentionMetadata
    o_bundle_state: dict
    strict_a2a_backend: str = "flashinfer-pcie"
    joint_len: int = 0
    use_uaa: bool = False


class H3UlyssesAttention(UlyssesParallelAttention):
    """Reuse ordinary resharding; specialize only the active H3 overlap scope."""

    def _exchange_qkv(self, query, key, value, attn_metadata):
        if overlap.ACTIVE.get() is None:
            return super()._exchange_qkv(query, key, value, attn_metadata)
        query = self._scatter_heads(query, "q")
        overlap.after_q(query, attn_metadata)
        qkv_overlap.after_q()
        key = self._scatter_heads(key, "k")
        value = overlap.before_v(query, key, value, attn_metadata, self._ulysses_pg)
        value = self._scatter_heads(value, "v")
        return query, key, value

    def pre_attention(self, query, key, value, attn_metadata):
        if overlap.ACTIVE.get() is None:
            return super().pre_attention(query, key, value, attn_metadata)
        if (
            get_ulysses_mode(default="strict") != "strict"
            or self._sp_group.ring_world_size != 1
            or self._sp_group.ulysses_world_size != 8
            or self._ulysses_a2a_backend != "flashinfer-pcie"
            or not self._ulysses_a2a_permute
            or (self._scatter_idx, self._gather_idx) != (2, 1)
        ):
            raise RuntimeError("H3 overlap requires strict TP1/SP8 FlashInfer Ulysses without Ring")
        if not h3_vsa_o_bundle_enabled() or attn_metadata is None:
            raise RuntimeError("H3 overlap requires reverse-O bundling and model layout metadata")
        if any(getattr(attn_metadata, key) is not None for key in ("joint_query", "joint_key", "joint_value")):
            raise RuntimeError("H3 overlap does not support joint attention rows")
        if any(key in attn_metadata.extra for key in (H3_VSA_O_BUNDLE_ACTIVE_KEY, H3_VSA_O_BUNDLE_STATE_KEY)):
            raise RuntimeError("H3 overlap received stale per-layer output state")
        gate = attn_metadata.extra.get("gate_compress")
        if not isinstance(gate, torch.Tensor) or gate.shape != query.shape or gate.dtype != torch.bfloat16:
            raise ValueError("H3 overlap requires a BF16 gate matching the local Q layout")
        if gate.device != query.device or not gate.is_contiguous() or gate.requires_grad:
            raise ValueError("H3 overlap gate must be contiguous, inference-only and on the Q device")
        layout = _get_h3_layout(attn_metadata)
        if layout is None:
            raise ValueError("H3 overlap requires prefix and target-video layout metadata")
        prefix, video_shape, target_start = layout
        if sum(prefix) != target_start:
            raise ValueError("H3 prefix/target boundary mismatch")
        import math

        valid_rows = sum(prefix) + math.prod(video_shape)
        aligned_rows = query.shape[1] * self._sp_group.ulysses_world_size
        packed = attn_metadata.packed_padding
        if packed is None:
            if valid_rows != aligned_rows:
                raise ValueError("H3 aligned rows require packed-padding metadata")
        elif packed.q_length != valid_rows or packed.kv_length != valid_rows:
            raise ValueError("H3 packed lengths must match the prefix/video geometry")
        plan = get_h3_vsa_owner_route_plan(prefix, video_shape, aligned_rows, self._sp_group.ulysses_world_size)
        state = {"plan": plan}
        attn_metadata.extra.pop("gate_compress")
        attn_metadata.extra[H3_VSA_O_BUNDLE_ACTIVE_KEY] = True
        attn_metadata.extra[H3_VSA_O_BUNDLE_STATE_KEY] = state
        try:
            query, key, value, attn_metadata, _ = super().pre_attention(query, key, value, attn_metadata)
        except BaseException:
            attn_metadata.extra.pop(H3_VSA_O_BUNDLE_ACTIVE_KEY, None)
            attn_metadata.extra.pop(H3_VSA_O_BUNDLE_STATE_KEY, None)
            raise
        return (
            query,
            key,
            value,
            attn_metadata,
            H3OverlapContext(
                name=self.name,
                ulysses_pg=self._ulysses_pg,
                o_bundle_gate_local=gate,
                o_bundle_metadata=attn_metadata,
                o_bundle_state=state,
            ),
        )

    def post_attention(self, attn_output, ctx):
        if not isinstance(ctx, H3OverlapContext):
            return super().post_attention(attn_output, ctx)
        try:
            result = overlap.finish_reverse(attn_output, ctx)
            if result is None:
                raise RuntimeError("H3 output producer scope ended before reverse exchange")
            return result
        finally:
            ctx.o_bundle_metadata.extra.pop(H3_VSA_O_BUNDLE_ACTIVE_KEY, None)
            ctx.o_bundle_metadata.extra.pop(H3_VSA_O_BUNDLE_STATE_KEY, None)


def configure_parallel_attention(attention) -> None:
    strategy = attention.parallel_strategy
    if isinstance(strategy, UlyssesParallelAttention) and strategy._ulysses_a2a_backend == "flashinfer-pcie":
        attention.parallel_strategy = H3UlyssesAttention(
            strategy._sp_group,
            strategy._scatter_idx,
            strategy._gather_idx,
            strategy._use_sync,
            strategy._ulysses_a2a_permute,
        )
