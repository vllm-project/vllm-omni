# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shard-layout arithmetic for BOOGU Image sequence parallelism.

BOOGU splits three independent sequences (noise image, reference images, and
instruction context) at one SP boundary. The framework records each sequence's
pre-padding global length under a ``shard_group`` key; everything else -- how
that global length maps onto a given rank's shard -- is arithmetic, so every
rank can derive the layout of *all* ranks without communicating.

That property is what lets the model build attention masks for the global,
rank-concatenated sequence that Ulysses produces after its all-to-all, instead
of all-gathering rank-local masks inside the attention layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.forward_context import get_sp_shard_original_seq_len

# (cos, sin) -- optionally (cos, sin, packed_table) where packed_table is the
# fp32 [tokens, head_dim] = [cos(theta) | sin(theta)] layout the fused
# qk-norm+RoPE op consumes; the eager path ignores the third element.
RotaryEmbedding = tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """How one globally padded sequence maps onto the SP ranks."""

    original_seq_len: int
    world_size: int
    rank: int

    @classmethod
    def resolve(cls, shard_group: str, *, local_seq_len: int) -> ShardLayout:
        """Read a boundary's layout from the ForwardContext.

        Falls back to an unsharded layout (world_size=1) when SP is off or the
        tensor was never split, in which case `local_seq_len` is the whole
        sequence.
        """
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        original_seq_len = get_sp_shard_original_seq_len(shard_group)
        if original_seq_len is None:
            return cls(original_seq_len=local_seq_len, world_size=1, rank=0)
        return cls(
            original_seq_len=original_seq_len,
            world_size=get_sequence_parallel_world_size(),
            rank=get_sequence_parallel_rank(),
        )

    @property
    def padded_seq_len(self) -> int:
        ws = self.world_size
        return ((self.original_seq_len + ws - 1) // ws) * ws

    @property
    def local_seq_len(self) -> int:
        return self.padded_seq_len // self.world_size

    @property
    def padding_size(self) -> int:
        return self.padded_seq_len - self.original_seq_len

    def bounds(self, rank: int) -> tuple[int, int]:
        """Half-open [start, end) span of the global sequence owned by `rank`."""
        start = rank * self.local_seq_len
        return start, start + self.local_seq_len

    def valid_lengths(self, global_lengths: list[int], *, rank: int) -> list[int]:
        """Per-sample valid prefix lengths clipped to `rank`'s shard."""
        start, end = self.bounds(rank)
        return [max(0, min(int(length), end) - start) for length in global_lengths]

    def segment_lengths(self, global_segments: list[list[int]], *, rank: int) -> list[list[int]]:
        """Per-sample contiguous segments intersected with `rank`'s shard."""
        start, end = self.bounds(rank)
        local: list[list[int]] = []
        for sample in global_segments:
            sample_local: list[int] = []
            offset = 0
            for length in sample:
                segment_end = offset + int(length)
                sample_local.append(max(0, min(segment_end, end) - max(offset, start)))
                offset = segment_end
            local.append(sample_local)
        return local


def rank_concat_mask_or_none(
    per_rank_lengths: list[list[int]],
    capacity: int,
    *,
    like: torch.Tensor,
    required: bool = False,
) -> torch.Tensor | None:
    """Mask over the rank-concatenated sequence Ulysses builds post all-to-all.

    ``per_rank_lengths[r][i]`` is sample ``i``'s valid length inside rank ``r``'s
    shard, and every shard has the same ``capacity`` (guaranteed by auto_pad),
    so the global sequence is ``world_size * capacity`` long. Returns ``None``
    for an all-valid mask unless a padding contract requires one.
    """
    if not required and all(int(length) == capacity for lengths in per_rank_lengths for length in lengths):
        return None

    batch_size = len(per_rank_lengths[0])
    mask = like.new_zeros(batch_size, len(per_rank_lengths) * capacity, dtype=torch.bool)
    for rank, lengths in enumerate(per_rank_lengths):
        base = rank * capacity
        for i, length in enumerate(lengths):
            mask[i, base : base + int(length)] = True
    return mask


def pack_local_rotary(
    context_rotary_emb: RotaryEmbedding,
    ref_img_rotary_emb: RotaryEmbedding,
    noise_rotary_emb: RotaryEmbedding,
    encoder_seq_lengths: list[int],
    ref_img_seq_lengths: list[list[int]],
    img_seq_lengths: list[int],
) -> tuple[RotaryEmbedding, RotaryEmbedding, list[int], list[int]]:
    """Pack local RoPE as ``[context, references, noise]``.

    Capacities stay equal across ranks; returned lengths exclude padding.
    Rotary embeddings arrive as ``(cos, sin)`` pairs -- the same packing is
    applied component-wise so both halves stay index-aligned.
    """
    context_cos, context_sin = context_rotary_emb
    ref_cos, ref_sin = ref_img_rotary_emb
    noise_cos, noise_sin = noise_rotary_emb

    combined_img_seq_lengths = [
        sum(ref_lengths) + img_length
        for ref_lengths, img_length in zip(
            ref_img_seq_lengths,
            img_seq_lengths,
        )
    ]
    seq_lengths = [
        context_length + combined_length
        for context_length, combined_length in zip(
            encoder_seq_lengths,
            combined_img_seq_lengths,
        )
    ]

    def _pack_component(
        context_t: torch.Tensor, ref_t: torch.Tensor, noise_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context_t.shape[0]
        rotary_dim = context_t.shape[-1]
        combined_img_capacity = ref_t.shape[1] + noise_t.shape[1]
        combined_img = context_t.new_zeros(batch_size, combined_img_capacity, rotary_dim)
        rotary = context_t.new_zeros(
            batch_size,
            context_t.shape[1] + combined_img_capacity,
            rotary_dim,
        )
        for i, (context_length, ref_lengths, img_length) in enumerate(
            zip(encoder_seq_lengths, ref_img_seq_lengths, img_seq_lengths)
        ):
            ref_length = sum(ref_lengths)
            combined_length = ref_length + img_length
            combined_img[i, :ref_length] = ref_t[i, :ref_length]
            combined_img[i, ref_length:combined_length] = noise_t[i, :img_length]
            rotary[i, :context_length] = context_t[i, :context_length]
            rotary[i, context_length : context_length + combined_length] = combined_img[i, :combined_length]
        return rotary, combined_img

    rotary_cos, combined_img_cos = _pack_component(context_cos, ref_cos, noise_cos)
    rotary_sin, combined_img_sin = _pack_component(context_sin, ref_sin, noise_sin)

    return (
        (rotary_cos, rotary_sin),
        (combined_img_cos, combined_img_sin),
        seq_lengths,
        combined_img_seq_lengths,
    )
