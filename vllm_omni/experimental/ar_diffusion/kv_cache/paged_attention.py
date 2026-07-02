# SPDX-License-Identifier: Apache-2.0
"""Paged self-attention helpers for AR-Diffusion DreamZero KV reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _slot_mapping_for_blocks(
    block_ids: list[int],
    token_count: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    if token_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    positions = torch.arange(token_count, dtype=torch.long)
    return compute_slot_mapping(block_ids, positions, block_size).to(device=device)


@dataclass
class ARDiffusionPagedForwardContext:
    """Mutable branch-level state shared by all layer contexts in one forward."""

    kv_cache: Any
    adapter: Any
    is_negative: bool
    history_block_ids: list[int]
    seq_len: int
    commit_current: bool
    max_video_tokens: int
    current_video_block_ids: list[int] = field(default_factory=list)
    current_video_slot_mapping: torch.Tensor | None = None
    action_scratch_block_ids: list[int] = field(default_factory=list)
    action_slot_mapping: torch.Tensor | None = None
    query_len: int = 0
    kv_len: int = 0
    _allocated_video: bool = False
    _committed: bool = False
    _action_len: int = 0

    @property
    def block_size(self) -> int:
        return int(self.kv_cache.block_size)

    @property
    def num_current_video_blocks(self) -> int:
        if self.seq_len % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention expects frame-aligned seq_len "
                f"(multiple of block_size={self.block_size}), got {self.seq_len}"
            )
        return self.seq_len // self.block_size

    def ensure_video_slots(self, device: torch.device) -> None:
        """Allocate/write targets for the current video tokens, once per branch."""
        if self._allocated_video:
            return

        n_blocks = self.num_current_video_blocks
        if self.commit_current:
            start = int(self.adapter.num_computed_tokens)
            self.kv_cache.allocate_token_slots(self.adapter, self.seq_len)
            table = self.kv_cache.block_table(self.adapter)
            start_block = start // self.block_size
            self.current_video_block_ids = [int(b) for b in table[start_block : start_block + n_blocks]]
            positions = torch.arange(start, start + self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = compute_slot_mapping(table, positions, self.block_size).to(device=device)
        else:
            self.current_video_block_ids = self.kv_cache.scratch_block_ids(self.is_negative, 0, n_blocks)
            self.current_video_slot_mapping = _slot_mapping_for_blocks(
                self.current_video_block_ids,
                self.seq_len,
                self.block_size,
                device,
            )
        self._allocated_video = True

    def ensure_action_slots(self, action_len: int, device: torch.device) -> None:
        """Reserve scratch slots for action/state K/V, if present."""
        if action_len <= 0:
            self.action_scratch_block_ids = []
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
            self._action_len = 0
            return

        self.ensure_video_slots(device)
        if self.action_slot_mapping is not None and self._action_len == action_len:
            return

        action_blocks = _ceil_div(action_len, self.block_size)
        scratch_offset = 0 if self.commit_current else len(self.current_video_block_ids)
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.is_negative,
            scratch_offset,
            action_blocks,
        )
        self.action_slot_mapping = _slot_mapping_for_blocks(
            self.action_scratch_block_ids,
            action_len,
            self.block_size,
            device,
        )
        self._action_len = action_len

    def video_block_table(self, device: torch.device) -> tuple[list[int], int]:
        self.ensure_video_slots(device)
        if self.max_video_tokens % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention requires max_video_tokens to be block-aligned, "
                f"got max_video_tokens={self.max_video_tokens}, block_size={self.block_size}"
            )
        all_video_blocks = self.history_block_ids + self.current_video_block_ids
        max_video_blocks = self.max_video_tokens // self.block_size
        visible_video_blocks = all_video_blocks[-max_video_blocks:] if max_video_blocks else []
        video_len = min(len(all_video_blocks) * self.block_size, self.max_video_tokens)
        return visible_video_blocks, video_len

    def build_block_table(
        self,
        *,
        action_len: int,
        query_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build FlashAttention block-table metadata for one self-attn call."""
        video_blocks, video_len = self.video_block_table(device)
        self.ensure_action_slots(action_len, device)
        action_blocks = self.action_scratch_block_ids if action_len > 0 else []
        block_ids = video_blocks + action_blocks
        if not block_ids:
            raise RuntimeError("AR-Diffusion paged attention needs at least current video KV blocks")

        self.query_len = int(query_len)
        self.kv_len = int(video_len + action_len)
        block_table = torch.tensor([block_ids], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, self.query_len], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([self.kv_len], dtype=torch.int32, device=device)
        return block_table, query_start_loc, seq_lens, self.query_len, self.kv_len

    def mark_committed(self) -> None:
        self._committed = True


@dataclass
class ARDiffusionPagedLayerContext:
    """Layer-specific handle passed through DreamZero's existing ``kv_cache`` slot."""

    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    forward_ctx: ARDiffusionPagedForwardContext

    @property
    def kv_cache(self):
        return self.forward_ctx.kv_cache

    @property
    def adapter(self):
        return self.forward_ctx.adapter

    @property
    def is_negative(self) -> bool:
        return self.forward_ctx.is_negative

    @property
    def history_block_ids(self) -> list[int]:
        return self.forward_ctx.history_block_ids

    @property
    def current_video_block_ids(self) -> list[int]:
        return self.forward_ctx.current_video_block_ids

    @property
    def current_video_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.current_video_slot_mapping

    @property
    def action_scratch_block_ids(self) -> list[int]:
        return self.forward_ctx.action_scratch_block_ids

    @property
    def action_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.action_slot_mapping

    @property
    def seq_len(self) -> int:
        return self.forward_ctx.seq_len

    @property
    def query_len(self) -> int:
        return self.forward_ctx.query_len

    @property
    def kv_len(self) -> int:
        return self.forward_ctx.kv_len

    @property
    def commit_current(self) -> bool:
        return self.forward_ctx.commit_current


def is_ar_diffusion_paged_context(value: object) -> bool:
    return isinstance(value, ARDiffusionPagedLayerContext)


def _reference_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    if causal:
        raise NotImplementedError("AR-Diffusion paged self-attention uses causal=False")
    outs: list[torch.Tensor] = []
    block_size = key_cache.shape[1]
    for i in range(seq_lens.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        q = query[q_start:q_end]
        positions = torch.arange(kv_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        k = key_cache[physical_blocks, offsets]
        v = value_cache[physical_blocks, offsets]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(softmax_scale)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


def ar_diffusion_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Run non-causal paged attention over a vLLM block table.

    ``query`` may be ``(B, L, H, D)`` or already flattened as ``(T, H, D)``.
    ``key_cache`` / ``value_cache`` are ``(num_blocks, block_size, H, D)``.
    """
    batched = query.dim() == 4
    if batched:
        batch, q_len = query.shape[:2]
        query_flat = query.reshape(batch * q_len, *query.shape[2:])
    else:
        query_flat = query

    if not query_flat.is_cuda:
        out = _reference_paged_attention(
            query_flat,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            softmax_scale,
            causal=causal,
        )
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

            fa_version = get_flash_attn_version(requires_alibi=False, head_size=query_flat.shape[-1])
        except Exception:
            fa_version = 2

        out = torch.empty_like(query_flat)
        flash_attn_varlen_func(
            q=query_flat,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=int(max_query_len),
            seqused_k=seq_lens,
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
            block_table=block_table,
            fa_version=fa_version,
        )

    if batched:
        return out.reshape(query.shape)
    return out
