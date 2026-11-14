# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion sequence-packing layout primitives.

Every non-attention computation is token-local and runs on sequence chunks
stacked along the local batch dimension; attention needs the complete logical
sequence and is reassembled per request at the attention boundary. This module
owns the layout side: the chunk-to-request mapping, per-chunk conditioning
expansion, chunk reassembly and the packed varlen layout (cu_seqlens /
max_seqlen). Layout only: no model-specific semantics, no ``nn.Module`` state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout


@dataclass
class PackedExecutionPlan:
    """Per-step MixFusion packing layout, computed once and shared by all blocks.

    Holds the flat varlen layout (cu_seqlens / max_seq_len) and the index
    tensors that recover each stream's rows, so per-block work stays on-GPU.
    The stream set and order come from the model's ``layout``; this plan
    records their lengths and indices.
    """

    layout: AttentionLayout
    chunk_size: int
    num_chunks: int
    request_chunk_ranges: list[tuple[int, int]]
    # Number of packed rows per caller-side state (a state may hold several
    # images/prompts); output reassembly concatenates each state's rows back
    # into one tensor.
    image_row_counts: tuple[int, ...]
    chunk_to_request: torch.Tensor
    # Stream name -> per-request token count, in request order.
    request_stream_lens: dict[str, tuple[int, ...]]
    # Padded width of each ``"request"`` stream's output buffer. The model
    # supplies this because only it knows the caller-side padded shape.
    stream_padded_lens: dict[str, int]
    cu_seqlens_tensor: torch.Tensor
    max_seq_len: int
    # Stream name -> (gather rows out of the packed result, scatter targets in
    # the padded per-request output space or None when no padding is needed).
    output_indices: dict[str, tuple[torch.Tensor, torch.Tensor | None]]


def build_execution_plan(
    *,
    layout: AttentionLayout,
    chunk_size: int,
    request_chunk_ranges: list[tuple[int, int]],
    image_row_counts: Sequence[int],
    request_stream_lens: dict[str, Sequence[int]],
    stream_padded_lens: dict[str, int],
    chunk_to_request: torch.Tensor,
    device: torch.device,
) -> PackedExecutionPlan:
    """Lay each request out as its streams in ``layout`` order.

    Inputs come straight from the adapter that just split the latents, so
    only the cross-field invariants are asserted.
    """
    lens_by_stream = {name: tuple(int(length) for length in request_stream_lens[name]) for name in layout.names}
    assert sum(image_row_counts) == len(request_chunk_ranges), "image_row_counts must cover every chunked row"

    cu_seqlens = [0]
    max_seq_len = 0
    gather_rows: dict[str, list[int]] = {name: [] for name in layout.names}
    for req_idx in range(len(request_chunk_ranges)):
        offset = cu_seqlens[-1]
        for spec in layout.streams:
            length = lens_by_stream[spec.name][req_idx]
            gather_rows[spec.name].extend(range(offset, offset + length))
            offset += length
        cu_seqlens.append(offset)
        max_seq_len = max(max_seq_len, offset - cu_seqlens[-2])

    output_indices: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for spec in layout.streams:
        gather = torch.tensor(gather_rows[spec.name], dtype=torch.long, device=device)
        scatter = None
        lens = lens_by_stream[spec.name]
        if spec.kind == "request" and any(length < stream_padded_lens[spec.name] for length in lens):
            padded = stream_padded_lens[spec.name]
            scatter_rows = [req_idx * padded + pos for req_idx, length in enumerate(lens) for pos in range(length)]
            scatter = torch.tensor(scatter_rows, dtype=torch.long, device=device)
        output_indices[spec.name] = (gather, scatter)

    return PackedExecutionPlan(
        layout=layout,
        chunk_size=chunk_size,
        num_chunks=request_chunk_ranges[-1][1] if request_chunk_ranges else 0,
        request_chunk_ranges=request_chunk_ranges,
        image_row_counts=tuple(image_row_counts),
        chunk_to_request=chunk_to_request,
        request_stream_lens=lens_by_stream,
        stream_padded_lens=dict(stream_padded_lens),
        cu_seqlens_tensor=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        output_indices=output_indices,
    )


def expand_per_chunk(temb: torch.Tensor, chunk_to_request: torch.Tensor) -> torch.Tensor:
    """Expand per-request conditioning rows to per-chunk rows.

    Chunk-level computations (AdaLN modulation) need request-level
    conditioning duplicated per chunk.
    """
    return temb.index_select(0, chunk_to_request)


def reassemble_chunk_rows(
    chunk_rows: torch.Tensor,
    request_chunk_ranges: Sequence[tuple[int, int]],
    row_counts: Sequence[int],
) -> list[torch.Tensor]:
    """Rebuild one output tensor per request from its contiguous chunk rows.

    Request ``i`` owns ``row_counts[i]`` image rows whose chunks are contiguous
    in ``request_chunk_ranges``; the result is one ``[num_images, S, C]``
    tensor per request.
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
