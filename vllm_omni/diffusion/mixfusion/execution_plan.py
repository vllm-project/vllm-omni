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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout


def gcd_chunk_size(latents: Sequence[torch.Tensor]) -> int:
    """Largest chunk length that divides every request's image sequence.

    Chunks are the packed unit, so each request's ``[num_images, S, C]``
    latents must split into equal chunks along ``S``; the GCD bounds the chunk
    below and keeps every request on the same chunk length.
    """
    chunk_size = int(latents[0].shape[1])
    for sample in latents[1:]:
        chunk_size = math.gcd(chunk_size, int(sample.shape[1]))
    return chunk_size


def plan_chunk_size(
    latents: Sequence[torch.Tensor],
    *,
    min_chunk_tokens: int,
    max_chunks: int,
) -> tuple[bool, str, int, int]:
    """Decide whether these per-request latents may be packed.

    Common admission policy: the GCD chunk length must be at least
    ``min_chunk_tokens`` (tokens per chunk below that make packing overhead
    dominate) and the total chunk count must stay within ``max_chunks`` (the
    stack length of the packed image stream). Returns
    ``(accepted, reason, chunk_size, chunk_count)``.
    """
    chunk_size = gcd_chunk_size(latents)
    chunk_count = sum(int(sample.shape[1]) // chunk_size * int(sample.shape[0]) for sample in latents)
    if chunk_size < min_chunk_tokens:
        return False, f"chunk_size={chunk_size} < min_chunk_tokens={min_chunk_tokens}", chunk_size, chunk_count
    if chunk_count > max_chunks:
        return False, f"chunk_count={chunk_count} > max_chunks={max_chunks}", chunk_size, chunk_count
    return True, "ok", chunk_size, chunk_count


@dataclass
class PackedExecutionPlan:
    """Per-step MixFusion packing layout, computed once and shared by all blocks.

    Holds the flat varlen sequence layout (cu_seqlens / max_seq_len) plus the
    index tensors that recover each attention stream's rows from the packed
    attention output, so per-block work stays on-GPU. Which streams exist and
    in what order they appear inside a request is the model's
    ``AttentionLayout``; this plan only records their lengths and indices.
    """

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
    cu_seqlens: list[int]
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
    stream_names = list(layout.names)
    if set(request_stream_lens) != set(stream_names):
        raise ValueError(
            f"request_stream_lens keys {sorted(request_stream_lens)} "
            f"do not match attention layout {sorted(stream_names)}."
        )

    num_requests = len(request_chunk_ranges)
    if sum(int(count) for count in image_row_counts) != num_requests:
        raise ValueError(f"image_row_counts {tuple(image_row_counts)} span {num_requests} chunked rows.")
    lens_by_stream: dict[str, tuple[int, ...]] = {}
    for name in stream_names:
        lens = tuple(int(length) for length in request_stream_lens[name])
        if len(lens) != num_requests:
            raise ValueError(f"request_stream_lens[{name!r}] has {len(lens)} entries, expected {num_requests}.")
        lens_by_stream[name] = lens

    num_chunks = request_chunk_ranges[-1][1] if request_chunk_ranges else 0
    cu_seqlens = [0]
    max_seq_len = 0
    gather_rows: dict[str, list[int]] = {name: [] for name in stream_names}
    for req_idx, (chunk_start, chunk_end) in enumerate(request_chunk_ranges):
        if chunk_end < chunk_start:
            raise ValueError(f"request {req_idx} has an inverted chunk range ({chunk_start}, {chunk_end}).")
        offset = cu_seqlens[-1]
        for spec in layout.streams:
            length = lens_by_stream[spec.name][req_idx]
            if spec.kind == "chunked":
                expected = (chunk_end - chunk_start) * chunk_size
                if length != expected:
                    raise ValueError(
                        f"Stream {spec.name!r} of request {req_idx} has {length} tokens, "
                        f"expected {expected} from {chunk_end - chunk_start} chunk(s)."
                    )
            gather_rows[spec.name].extend(range(offset, offset + length))
            offset += length
        cu_seqlens.append(offset)
        max_seq_len = max(max_seq_len, offset - cu_seqlens[-2])

    output_indices: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for spec in layout.streams:
        gather = torch.tensor(gather_rows[spec.name], dtype=torch.long, device=device)
        scatter = None
        if spec.kind == "request":
            if spec.name not in stream_padded_lens:
                raise ValueError(f"Stream {spec.name!r} needs an entry in stream_padded_lens.")
            padded = int(stream_padded_lens[spec.name])
            scatter_rows: list[int] = []
            needs_padding = False
            for req_idx, length in enumerate(lens_by_stream[spec.name]):
                if length > padded:
                    raise ValueError(
                        f"Stream {spec.name!r} of request {req_idx} has {length} tokens, padded to {padded}."
                    )
                scatter_rows.extend(range(req_idx * padded, req_idx * padded + length))
                needs_padding = needs_padding or length < padded
            if needs_padding:
                scatter = torch.tensor(scatter_rows, dtype=torch.long, device=device)
        output_indices[spec.name] = (gather, scatter)

    return PackedExecutionPlan(
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        request_chunk_ranges=request_chunk_ranges,
        image_row_counts=tuple(int(count) for count in image_row_counts),
        chunk_to_request=chunk_to_request,
        request_stream_lens=lens_by_stream,
        stream_padded_lens={name: int(stream_padded_lens[name]) for name in stream_names if name in stream_padded_lens},
        cu_seqlens=cu_seqlens,
        cu_seqlens_tensor=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        output_indices=output_indices,
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
