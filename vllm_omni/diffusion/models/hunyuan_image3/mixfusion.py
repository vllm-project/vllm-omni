# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MixFusion helpers for DiT-style HunyuanImage-3.0 mixed-resolution batches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd

import torch


@dataclass(frozen=True)
class MixFusionImageLayout:
    """Sequence layout for one original image request."""

    index: int
    token_height: int
    token_width: int
    seq_len: int
    chunk_start: int
    chunk_count: int


@dataclass(frozen=True)
class MixFusionSequencePlan:
    """Mapping between original image sequences and a flattened chunk batch."""

    token_shapes: tuple[tuple[int, int], ...]
    chunk_size: int
    layouts: tuple[MixFusionImageLayout, ...]

    @property
    def chunk_count(self) -> int:
        return sum(layout.chunk_count for layout in self.layouts)


def validate_mixfusion_sequence_plan(
    plan: MixFusionSequencePlan,
    *,
    min_chunk_tokens: int = 256,
    max_chunks: int = 128,
) -> tuple[bool, str]:
    """Return whether a MixFusion plan is worth executing.

    Very small GCDs create many tiny chunk rows. That preserves math, but tends
    to lose the expected padding savings to tokenizer/RoPE/mask/scatter overhead
    and can also violate downstream batch-size assumptions.
    """

    if plan.chunk_size < min_chunk_tokens:
        return False, f"chunk_size={plan.chunk_size} < min_chunk_tokens={min_chunk_tokens}"
    if plan.chunk_count > max_chunks:
        return False, f"chunk_count={plan.chunk_count} > max_chunks={max_chunks}"

    seq_lens = [layout.seq_len for layout in plan.layouts]
    max_seq_len = max(seq_lens)
    padded_token_work = len(seq_lens) * max_seq_len
    mixfusion_token_work = sum(seq_lens)
    padded_attention_work = len(seq_lens) * max_seq_len * max_seq_len
    mixfusion_attention_work = sum(seq_len * seq_len for seq_len in seq_lens)
    if mixfusion_token_work >= padded_token_work and mixfusion_attention_work >= padded_attention_work:
        return (
            False,
            "estimated padding savings are not positive "
            f"(token_work={mixfusion_token_work}/{padded_token_work}, "
            f"attention_work={mixfusion_attention_work}/{padded_attention_work})",
        )

    return True, "ok"


def build_mixfusion_sequence_plan(token_shapes: list[tuple[int, int]]) -> MixFusionSequencePlan:
    """Build a DiT MixFusion plan from per-request `(token_h, token_w)` shapes.

    The chunk size is the greatest common divisor of image sequence lengths and
    is computed once when the request batch enters the denoising pipeline.
    """

    shapes = tuple((int(h), int(w)) for h, w in token_shapes)
    if len(shapes) == 0:
        raise ValueError("MixFusion requires at least one token shape.")

    seq_lens = tuple(h * w for h, w in shapes)
    chunk_size = reduce(gcd, seq_lens)
    if chunk_size <= 0:
        raise ValueError(f"Invalid MixFusion chunk size {chunk_size} for token shapes {shapes}.")

    layouts: list[MixFusionImageLayout] = []
    chunk_start = 0
    for index, ((token_h, token_w), seq_len) in enumerate(zip(shapes, seq_lens, strict=True)):
        if seq_len % chunk_size != 0:
            raise ValueError(f"Sequence length {seq_len} is not divisible by chunk size {chunk_size}.")
        chunk_count = seq_len // chunk_size
        layouts.append(
            MixFusionImageLayout(
                index=index,
                token_height=token_h,
                token_width=token_w,
                seq_len=seq_len,
                chunk_start=chunk_start,
                chunk_count=chunk_count,
            )
        )
        chunk_start += chunk_count

    return MixFusionSequencePlan(token_shapes=shapes, chunk_size=chunk_size, layouts=tuple(layouts))


def split_sequences_to_mixfusion_chunks(
    sequences: list[torch.Tensor],
    plan: MixFusionSequencePlan,
) -> torch.Tensor:
    """Split `[1, S_i, D]` image token sequences into `[num_chunks, gcd(S), D]`."""

    if len(sequences) != len(plan.layouts):
        raise ValueError(f"Expected {len(plan.layouts)} sequences, got {len(sequences)}.")

    chunks: list[torch.Tensor] = []
    for layout in plan.layouts:
        sequence = sequences[layout.index]
        if sequence.shape[0] != 1 or sequence.shape[1] != layout.seq_len:
            raise ValueError(
                f"Sequence {layout.index} has shape {tuple(sequence.shape)}, expected "
                f"(1, {layout.seq_len}, hidden_size)."
            )
        chunks.append(sequence.reshape(layout.chunk_count, plan.chunk_size, sequence.shape[-1]))
    return torch.cat(chunks, dim=0)


def merge_mixfusion_chunks_to_sequences(
    chunks: torch.Tensor,
    plan: MixFusionSequencePlan,
) -> list[torch.Tensor]:
    """Merge flattened chunk outputs back to one `[1, S_i, D]` sequence per request."""

    sequences: list[torch.Tensor] = []
    for layout in plan.layouts:
        start = layout.chunk_start
        end = start + layout.chunk_count
        sequences.append(chunks[start:end].reshape(1, layout.seq_len, chunks.shape[-1]))
    return sequences
