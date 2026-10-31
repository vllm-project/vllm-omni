# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion sequence-packing layout primitives.

MixFusion treats chunked sequence packing the way Ulysses sequence
parallelism treats shards: every non-attention computation is token-local
and can run on sequence partitions (chunks stacked along the local batch
dimension), while attention needs the complete logical sequence and is
reassembled per request at the attention boundary. This module owns the
layout side of that split: sequence-dim chunking with aligned tensors
(hidden states and their RoPE frequencies) kept under the same partition,
the chunk-to-request mapping, per-chunk conditioning expansion, and the
packed varlen attention layout (cu_seqlens / max_seqlen). Models declare
their stream relationships; attention modules adapt the layout back into
their attention domain.

This module is layout-only: no model-specific semantics, no nn.Module
state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MixFusionPlan:
    """Per-step MixFusion packing layout, computed once and shared by all blocks.

    Holds the flat varlen sequence layout (cu_seqlens / max_seq_len) plus the
    index tensors that recover per-request text rows and per-chunk image rows
    from the packed attention output, so per-block work stays on-GPU.
    """

    cu_seqlens: list[int]
    cu_seqlens_tensor: torch.Tensor
    max_seq_len: int
    request_chunk_ranges: list[tuple[int, int]]
    real_txt_lens: list[int]
    seq_len_txt: int
    chunk_size: int
    chunk_to_request: torch.Tensor
    txt_out_idx: torch.Tensor
    img_out_idx: torch.Tensor
    # Target rows in the padded [num_requests * seq_len_txt] text-output space,
    # or None when every request uses the full padded text length (no mask).
    txt_pad_idx: torch.Tensor | None


def build_mixfusion_plan(
    *,
    seq_len_txt: int,
    chunk_size: int,
    request_chunk_ranges: list[tuple[int, int]],
    real_txt_lens: list[int],
    chunk_to_request: torch.Tensor,
    device: torch.device,
) -> MixFusionPlan:
    cu_seqlens = [0]
    txt_out_rows: list[int] = []
    img_out_rows: list[int] = []
    txt_pad_rows: list[int] = []
    max_seq_len = 0
    for req_idx, (chunk_start, chunk_end) in enumerate(request_chunk_ranges):
        txt_len = int(real_txt_lens[req_idx])
        img_len = (chunk_end - chunk_start) * chunk_size
        base = cu_seqlens[-1]
        txt_out_rows.extend(range(base, base + txt_len))
        img_out_rows.extend(range(base + txt_len, base + txt_len + img_len))
        cu_seqlens.append(base + txt_len + img_len)
        max_seq_len = max(max_seq_len, txt_len + img_len)
        txt_pad_rows.extend(range(req_idx * seq_len_txt, req_idx * seq_len_txt + txt_len))
    needs_padding = any(txt_len < seq_len_txt for txt_len in real_txt_lens)
    return MixFusionPlan(
        cu_seqlens=cu_seqlens,
        cu_seqlens_tensor=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        request_chunk_ranges=request_chunk_ranges,
        real_txt_lens=real_txt_lens,
        seq_len_txt=seq_len_txt,
        chunk_size=chunk_size,
        chunk_to_request=chunk_to_request,
        txt_out_idx=torch.tensor(txt_out_rows, dtype=torch.long, device=device),
        img_out_idx=torch.tensor(img_out_rows, dtype=torch.long, device=device),
        txt_pad_idx=torch.tensor(txt_pad_rows, dtype=torch.long, device=device) if needs_padding else None,
    )


def split_sequence_chunks(
    rows: Sequence[torch.Tensor],
    row_freqs: Sequence[torch.Tensor],
    chunk_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]], list[int]]:
    """Split per-row states and their aligned RoPE frequencies into chunks.

    ``rows[i]`` is a ``[1, S_i, C]`` latent row and ``row_freqs[i]`` the
    ``[S_i, D/2]`` RoPE frequency table for the same positions, so the two
    tensors always land under the same partition. Returns the per-chunk
    tensors for both (the caller stacks them along the local batch
    dimension), the per-row chunk ranges, and the chunk-to-request mapping.
    """
    chunks: list[torch.Tensor] = []
    freq_chunks: list[torch.Tensor] = []
    request_chunk_ranges: list[tuple[int, int]] = []
    chunk_to_request: list[int] = []
    for row_idx, sample in enumerate(rows):
        seq_len = int(sample.shape[1])
        if seq_len % chunk_size != 0:
            raise ValueError(f"MixFusion seq_len={seq_len} is not divisible by chunk_size={chunk_size}.")
        chunk_start = len(chunks)
        chunks.extend(sample.split(chunk_size, dim=1))
        freq_chunks.extend(row_freqs[row_idx].split(chunk_size, dim=0))
        chunk_end = len(chunks)
        request_chunk_ranges.append((chunk_start, chunk_end))
        chunk_to_request.extend([row_idx] * (chunk_end - chunk_start))
    return chunks, freq_chunks, request_chunk_ranges, chunk_to_request


def expand_per_chunk(temb: torch.Tensor, chunk_to_request: torch.Tensor) -> torch.Tensor:
    """Expand per-request conditioning rows to per-chunk rows.

    Chunk-level computations (AdaLN modulation on chunk-stacked image
    streams) need request-level conditioning duplicated per chunk.
    """
    return temb.index_select(0, chunk_to_request)


def reassemble_chunk_rows(
    chunk_rows: torch.Tensor,
    request_chunk_ranges: Sequence[tuple[int, int]],
    row_counts: Sequence[int],
) -> list[torch.Tensor]:
    """Rebuild one output tensor per request from its contiguous chunk rows.

    Request ``i`` owns ``row_counts[i]`` image rows whose chunks are
    contiguous in ``request_chunk_ranges``; the output preserves the
    external contract of one ``[num_images, S, C]`` tensor per request.
    """
    outputs: list[torch.Tensor] = []
    row_idx = 0
    last_dim = chunk_rows.shape[-1]
    for num_images in row_counts:
        rows = []
        for _ in range(num_images):
            chunk_start, chunk_end = request_chunk_ranges[row_idx]
            row_idx += 1
            rows.append(chunk_rows[chunk_start:chunk_end].reshape(1, -1, last_dim))
        outputs.append(torch.cat(rows, dim=0) if num_images > 1 else rows[0])
    return outputs
