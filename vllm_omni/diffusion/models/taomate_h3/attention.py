# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming attention for TaoMate-H3.

Outside a streaming context the layer is the upstream ``MiniMaxH3Attention``
(packed varlen attention through the shared ``Attention`` layer, including its
Ulysses strategy); that path serves the audio teacher and the token refiner.

Inside a streaming context (installed by the pipeline around every student
forward) the layer implements TaoMate's causal conditioning for the live
document of the current chunk:

* condition (text) rows attend to the text rows only;
* media (audio + video) rows attend to ``[text K/V | persistent clean AV K/V | current chunk K/V]``;
* alignment-padding rows produce zeros;
* in clean-commit mode the layer additionally stages its post-all-to-all
  K/V rows of the current chunk into the session's ``CleanAVKVCache``.

Sequence parallelism is pure Ulysses: the layer performs the sequence-to-head
all-to-all itself (as LingBot-World does) so that the persistent cache holds
full-sequence rows for this rank's head shard, and gathers back before the
output projection. Attention runs as dense FlashAttention-3 (``fa3_fwd_interface``
or ``flash_attn_interface``), the same kernel family as the release; SDPA is the
CPU / no-FA fallback.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum

import torch

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D, all_to_all_5D
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

from .kv_cache import CleanAVKVCache


class StreamMode(str, Enum):
    NOISY = "noisy"
    CLEAN_COMMIT = "clean_commit"


@dataclass
class StreamContext:
    """Full-sequence metadata of the live chunk document, shared by all layers."""

    cache: CleanAVKVCache
    mode: StreamMode
    # Global (full packed sequence) row indices, on the compute device.
    condition_rows: torch.Tensor
    media_rows: torch.Tensor
    token_tags: torch.Tensor
    commit_mask: torch.Tensor
    seq_len: int
    # ``(start, length)`` when the rows are one contiguous range (the pinned
    # phase layout: text rows, then audio and video rows); slices then replace
    # the gathers.
    condition_span: tuple[int, int] | None = None
    media_span: tuple[int, int] | None = None
    kernel_calls: int = 0


_STREAM_CONTEXT: ContextVar[StreamContext | None] = ContextVar("taomate_h3_stream_context", default=None)


def current_stream_context() -> StreamContext | None:
    return _STREAM_CONTEXT.get()


@contextmanager
def stream_context(context: StreamContext) -> Iterator[StreamContext]:
    if _STREAM_CONTEXT.get() is not None:
        raise RuntimeError("TaoMate-H3 streaming attention context is nested")
    token = _STREAM_CONTEXT.set(context)
    try:
        yield context
    finally:
        _STREAM_CONTEXT.reset(token)


def _ulysses_state() -> tuple[int, int, torch.distributed.ProcessGroup | None]:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        coordinator = get_sp_group()
    except (AssertionError, ImportError, RuntimeError):
        return 1, 0, None
    return int(coordinator.ulysses_world_size), int(coordinator.ulysses_rank), coordinator.ulysses_group


def _resolve_flash_attention():
    try:
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_func

        return flash_attn_func
    except Exception:  # noqa: BLE001 - fall back to SDPA below
        return None


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    flash_attn_func,
) -> torch.Tensor:
    """Non-causal dense attention over ``[rows, heads, head_dim]`` tensors."""
    if flash_attn_func is not None and query.is_cuda and query.dtype in (torch.bfloat16, torch.float16):
        output = flash_attn_func(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            softmax_scale=scale,
            causal=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.squeeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


class TaoMateH3StreamingAttention(MiniMaxH3Attention):
    """``MiniMaxH3Attention`` with TaoMate's persistent-KV streaming path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Assigned by the DiT model; ``None`` keeps the layer on the base path
        # (token refiner blocks are never streamed).
        self.layer_name: str | None = None
        self.ulysses_world_size, self.ulysses_rank, self.ulysses_group = _ulysses_state()
        if self.num_heads % self.ulysses_world_size:
            raise ValueError(
                "TaoMate-H3 local attention heads must be divisible by the Ulysses degree: "
                f"heads={self.num_heads}, ulysses={self.ulysses_world_size}"
            )
        self.num_sp_heads = self.num_heads // self.ulysses_world_size
        self._flash_attn_func = _resolve_flash_attention()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_table: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int | None = None,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
    ) -> torch.Tensor:
        context = _STREAM_CONTEXT.get()
        if context is None or self.layer_name is None:
            return super().forward(
                x,
                rope_table=rope_table,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                packed_total=packed_total,
                num_requests=num_requests,
                sp_seq_lens=sp_seq_lens,
                video_layout=video_layout,
                vsa_prefix_segments=vsa_prefix_segments,
            )
        if num_requests != 1:
            raise ValueError("TaoMate-H3 streaming attention serves one live document per forward")
        if rope_table is None:
            raise ValueError("TaoMate-H3 streaming attention requires the phase RoPE table")
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        q, k = fused_qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            rope_table,
            self.q_norm.variance_epsilon,
        )
        if self.ulysses_world_size > 1:
            # One fused sequence->head all-to-all for q/k/v:
            # (1, S/N, 3, H, D) -> (1, S, 3, H/N, D).
            stacked = all_to_all_5D(
                torch.stack((q, k, v), dim=1).unsqueeze(0),
                scatter_idx=3,
                gather_idx=1,
                group=self.ulysses_group,
            )
            q, k, v = stacked[0].unbind(1)
        out = self._streaming_attention(q, k, v, context)
        if self.ulysses_world_size > 1:
            out = SeqAllToAll4D.apply(self.ulysses_group, out.unsqueeze(0), 1, 2, False)[0]
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out

    @torch.compiler.disable
    def _streaming_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: StreamContext,
    ) -> torch.Tensor:
        """Attention of the full live document with this rank's head shard."""
        seq_len = int(q.shape[0])
        if seq_len != context.seq_len:
            raise RuntimeError(
                f"{self.layer_name}: streaming attention saw {seq_len} rows, the phase document has {context.seq_len}"
            )
        scale = float(self.softmax_scale)
        condition_rows = context.condition_rows
        media_rows = context.media_rows
        c_span, m_span = context.condition_span, context.media_span
        if c_span is not None:
            q_condition = q[c_span[0] : c_span[0] + c_span[1]]
            k_condition = k[c_span[0] : c_span[0] + c_span[1]]
            v_condition = v[c_span[0] : c_span[0] + c_span[1]]
        else:
            q_condition = q.index_select(0, condition_rows)
            k_condition = k.index_select(0, condition_rows)
            v_condition = v.index_select(0, condition_rows)
        condition_out = _dense_attention(
            q_condition,
            k_condition,
            v_condition,
            scale=scale,
            flash_attn_func=self._flash_attn_func,
        )
        if m_span is not None:
            q_media = q[m_span[0] : m_span[0] + m_span[1]]
            k_media = k[m_span[0] : m_span[0] + m_span[1]]
            v_media = v[m_span[0] : m_span[0] + m_span[1]]
        else:
            q_media = q.index_select(0, media_rows)
            k_media = k.index_select(0, media_rows)
            v_media = v.index_select(0, media_rows)
        # One contiguous [history | media | condition] view per layer: the live
        # rows are copied into the cache's scratch rows; no per-layer
        # concatenation of the history.
        keys, values = context.cache.assemble(self.layer_name, k_condition, v_condition, k_media, v_media)
        media_out = _dense_attention(
            q_media,
            keys,
            values,
            scale=scale,
            flash_attn_func=self._flash_attn_func,
        )
        context.kernel_calls += 2
        out = torch.zeros_like(q)
        if c_span is not None:
            out[c_span[0] : c_span[0] + c_span[1]] = condition_out
        else:
            out.index_copy_(0, condition_rows, condition_out)
        if m_span is not None:
            out[m_span[0] : m_span[0] + m_span[1]] = media_out
        else:
            out.index_copy_(0, media_rows, media_out)
        if context.mode is StreamMode.CLEAN_COMMIT:
            # The clean media rows already sit right after the history.
            context.cache.stage_in_place(
                self.layer_name, int(k_media.shape[0]), context.token_tags, context.commit_mask
            )
        return out


__all__ = [
    "StreamContext",
    "StreamMode",
    "TaoMateH3StreamingAttention",
    "current_stream_context",
    "stream_context",
]
